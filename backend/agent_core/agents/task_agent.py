"""Task agent reasoning loop — Text→Text only this phase (voice modes later).

Tool-calling tries the real provider function-calling API first
(`complete_with_tools_and_fallback`, returning structured `ToolCall`s — no
text parsing). If a provider/model doesn't return any native tool calls, the
loop falls back to the original prompted `TOOL_CALL: {"name": ..., "args":
{...}}` text convention — real defense-in-depth (a model that ignores the
`tools` field but was told the convention in the prompt manifest can still
get through), not just a compatibility shim. Verified live against Grok:
`complete_with_tools()` calling `calculate(17*23)` and getting the real
answer (391) back through this exact loop — see
tests/test_agentic_loop_real_tools.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from ..llm_adapter.base import LLMProviderError, LLMRouter, Message, ToolCall, ToolDefinition
from ..observability.tracing import start_span
from ..security.confirmation import ConfirmationGate, PendingConfirmation
from ..security.output_validation import sanitize_llm_output
from ..supervisor.state import Mode, SessionState
from .cancellation import CancellationToken, TurnCancelled
from .language_agent import LANGUAGE_NAMES
from .prompt_templates import load_template
from .untrusted import wrap_untrusted

logger = logging.getLogger("agent_core.turn_trace")

CLARIFYING_QUESTION = (
    "I want to make sure I get this right — could you clarify exactly what you'd like me to do?"
)

CONFIRMATION_REQUIRED_TEXT = (
    "This action can't be undone, so I need you to confirm it before I proceed — "
    "please tap or type to confirm."
)

LLM_UNAVAILABLE_APOLOGY = "I'm having trouble getting an answer right now — please try again in a moment."

_CORRECTION_REQUEST_TEMPLATE = (
    "Your previous draft violated a rule: {reason}. Revise your ENTIRE reply to fix "
    "this while keeping the same factual content and answering the same question. "
    "Reply with ONLY the corrected reply text — no preamble, no explanation of what "
    "you changed."
)

ToolFn = Callable[..., Awaitable[str]]


@dataclass
class TurnResult:
    text: str
    prompt_version: str
    tool_call_count: int
    self_check_ok: bool
    self_check_reason: str
    cancelled: bool = False
    pending_confirmation: PendingConfirmation | None = None


def _build_system_prompt(
    session: SessionState, *, version: str = "v1", tool_manifest: str = ""
) -> tuple[str, str]:
    template, version_id = load_template(session.mode, version=version)
    text = template.format(
        response_language=session.response_language or "unknown",
        language_confidence=(
            f"{session.language_confidence:.2f}" if session.language_confidence is not None else "unknown"
        ),
        is_code_mixed=session.is_code_mixed,
    )
    if tool_manifest:
        # Appended, not woven into the template — the manifest is generated
        # from whatever tools the caller actually registered (registry.py),
        # so it can't drift from the template file's static text.
        text = f"{text}\n\n{tool_manifest}"
    return text, version_id


def _is_latin_script(text: str) -> bool:
    """True when the user typed in the Latin alphabet — e.g. "Telugulish"
    (romanized Telugu-English, like "naku python gurinchi cheppu"), not actual
    Telugu Unicode script. Used to tell the model to reply in the SAME
    script, not just the same language — answering fluent Telugu is only half
    right if it's in Telugu script the user typed in Latin letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    return sum(1 for ch in letters if ch.isascii()) / len(letters) > 0.8


def _expected_script(session: SessionState, user_message: str) -> str | None:
    """"native", "latin", or None (no strong expectation — English, unknown
    language, or the user's own input already carries no Latin-script signal
    to preserve). Shared by `_language_directive` (what to ask for) and
    `_self_check` (what to verify the draft actually did) so the instruction
    and the check can never silently drift apart.

    Voice mode always wants native script regardless of input script — a
    native-script TTS voice maps phonemes off proper Unicode, and feeding it
    Latin-transliterated text produced exactly what was reported live:
    garbled/mispronounced characters and words the engine couldn't recognize
    and just skipped. Text-output modes mirror the user's own script back —
    "Telugulish" in, "Telugulish" out.
    """
    lang = session.response_language
    if not lang or lang == "en" or lang == "unknown":
        return None
    if not _is_latin_script(user_message):
        return None
    return "native" if session.mode.is_voice else "latin"


def _language_directive(session: SessionState, user_message: str) -> str:
    """A short, explicit language instruction placed right next to the user's
    actual message — not just buried in the system prompt. Fast/small models
    (the exact class this system routes to for cost/latency) follow an
    instruction adjacent to the query far more reliably than one several
    hundred words earlier in a system prompt; this is redundant with the
    system prompt's own LANGUAGE PRESERVATION section by design, not a
    replacement for it.
    """
    lang = session.response_language
    if not lang or lang == "unknown" or (lang == "en" and not session.is_code_mixed):
        return ""
    name = LANGUAGE_NAMES.get(lang, lang)
    # An EXPLICIT instruction either way, not just the absence of one —
    # verified live that merely omitting the "use romanized script"
    # instruction for voice mode still let the model default to matching the
    # user's own (romanized) input; it needs to be told outright.
    expected = _expected_script(session, user_message)
    script_note = ""
    if expected == "native":
        script_note = (
            f" The user typed in romanized Latin script, but this answer will be SPOKEN, "
            f"not read — write it in {name}'s native script so it's pronounced correctly, "
            f"never in Latin/romanized letters."
        )
    elif expected == "latin":
        script_note = (
            f" Write it in the same Latin/romanized script the user just typed in (a natural "
            f"'{name}lish' style, like they did) — do NOT switch to {name}'s native script."
        )
    if session.is_code_mixed:
        return (
            f"[Answer in {name}, naturally code-mixed with English the way the user just "
            f"spoke — do not answer in plain English.{script_note}]"
        )
    return f"[Answer in {name}. Do not answer in English or any other language.{script_note}]"


def _voice_brevity_directive(session: SessionState) -> str:
    """Reinforces the voice-mode system prompt's own LENGTH/FORMATTING
    sections right next to the query, same rationale as `_language_directive`
    — applies regardless of language, since a long, clause-heavy answer is
    hard for any TTS voice to render clearly, not just a non-English one."""
    if not session.mode.is_voice:
        return ""
    return "[Keep the answer short, simple, and clear — plain words, short sentences, easy to speak aloud naturally.]"


def _turn_directive(session: SessionState, user_message: str) -> str:
    return " ".join(d for d in (_language_directive(session, user_message), _voice_brevity_directive(session)) if d)


def _parse_tool_call(raw: str) -> ToolCall | None:
    """Legacy text-convention fallback — only consulted when the native path
    (`complete_with_tools_and_fallback`) returned no structured tool calls."""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("TOOL_CALL:"):
            try:
                payload = json.loads(line[len("TOOL_CALL:") :].strip())
                return ToolCall(id="legacy-text-call", name=payload["name"], args=payload.get("args", {}))
            except (json.JSONDecodeError, KeyError, TypeError):
                return None
    return None


_SELF_CHECK_SYSTEM = (
    "You are a strict reviewer, not the assistant. You will be shown a draft reply, "
    "the mode it must obey, and the language the user should be answered in. Reply "
    'with exactly one line: "OK" if the draft is compliant, or "VIOLATION: <short '
    'reason>" if not. VOICE_MODE rules: no markdown, no emoji, no parentheticals, at '
    "most ~6 sentences. TEXT_MODE rules: markdown is allowed, at most ~150 words. "
    "LANGUAGE rule (both modes): reject ONLY if the draft is answered in a clearly "
    "different language than the target (e.g. target is Telugu or Telugu-English "
    "code-mixed, but the draft is plain, unrelated English) — natural code-mixing "
    "that matches how the target language is actually spoken is NOT a violation."
)


async def _self_check(
    draft: str,
    mode: Mode,
    router: LLMRouter,
    *,
    response_language: str | None = None,
    is_code_mixed: bool = False,
    expected_script: str | None = None,
) -> tuple[bool, str]:
    """Bounded, separate critique call — never folded into the main generation
    call (agent_system_prompt.md §3: generate+critique in one pass is weaker
    than a second, narrowly-scoped critique).

    Cheap deterministic checks run first (word count, markdown markers) so an
    obvious violation doesn't need a second model call at all; only an
    ambiguous draft pays for the LLM critique.
    """
    word_count = len(draft.split())
    if mode.is_voice:
        if any(marker in draft for marker in ("**", "##", "- ", "* ", "```")):
            return False, "voice-mode formatting violation: markdown present"
        if word_count > 90:
            return False, "voice-mode length violation"
    else:
        if word_count > 160:
            return False, "text-mode length violation"

    # Deterministic, not left to the LLM reviewer below: verified live BOTH
    # directions are real — a model asked for native script sometimes
    # answers in romanized script anyway (garbled/mispronounced by a
    # native-script TTS voice), and a model asked to preserve the user's own
    # romanized "Telugulish" style sometimes switches to native script anyway
    # (wrong for a text reply meant to mirror how the user typed). Neither is
    # a detection gap — the instruction was there both times — so this is
    # checked and corrected rather than assumed.
    if expected_script == "native" and _is_latin_script(draft):
        name = LANGUAGE_NAMES.get(response_language, response_language) if response_language else "the target language"
        return False, (
            f"voice-mode script violation: answered in Latin/romanized script instead of "
            f"{name}'s native script — will be mispronounced by TTS"
        )
    if expected_script == "latin" and not _is_latin_script(draft):
        name = LANGUAGE_NAMES.get(response_language, response_language) if response_language else "the target language"
        return False, (
            f"script violation: answered in {name}'s native script instead of the romanized "
            "Latin script the user typed in and expects back"
        )

    target = response_language or "unspecified"
    if is_code_mixed:
        target = f"{target} (code-mixed with English)"
    reply = await router.complete_with_fallback(
        [
            {
                "role": "user",
                "content": f"MODE: {'VOICE' if mode.is_voice else 'TEXT'}\nTARGET_LANGUAGE: {target}\nDRAFT:\n{draft}",
            }
        ],
        system=_SELF_CHECK_SYSTEM,
        max_tokens=60,
    )
    reply = reply.strip()
    if reply.upper().startswith("OK"):
        return True, ""
    return False, reply


async def run_turn(
    session: SessionState,
    router: LLMRouter,
    user_message: str,
    *,
    tools: dict[str, ToolFn] | None = None,
    tool_definitions: Sequence[ToolDefinition] | None = None,
    max_tool_calls: int = 3,
    prompt_version: str = "v1",
    cancellation_token: CancellationToken | None = None,
    write_scope_tools: set[str] | None = None,
    confirmation_gate: ConfirmationGate | None = None,
    confirmation_token: str | None = None,
    tool_manifest: str = "",
    history: Sequence[Message] | None = None,
) -> TurnResult:
    """One turn of the reasoning loop. Tool-call budget is enforced here, in
    code — exceeding it stops the loop and returns a clarifying question
    instead of making another tool call, regardless of what the prompt says.

    `history` is the prior turns of THIS conversation (plain user/assistant
    text messages, no tool-call entries), prepended before the current
    message — without it every turn was stateless and the "agent" couldn't
    even answer "what did I just ask you?". The supervisor graph owns
    accumulating and capping it (per thread_id, via its checkpointer);
    run_turn just consumes it.

    `cancellation_token` (Phase 5 barge-in) is checked before every step —
    the next LLM call, and again right after any tool call completes — never
    mid-tool-call, so a barge-in can never leave a tool call half-executed:
    at the moment it fires, a dispatched tool has either already finished
    (its result is recorded) or hasn't started yet. The in-flight LLM call
    itself is raced against the token via `token.run()`, so cancellation
    actually aborts it rather than merely discarding its eventual result.

    `write_scope_tools` (Phase 6): tool names in this set are irreversible
    actions. In a voice-originated turn (`session.mode.is_voice`), such a
    tool is never executed on first request — a `PendingConfirmation` is
    returned instead, and only a matching `confirmation_token` (verified
    against that exact tool+args via `confirmation_gate.consume()`) lets a
    resubmitted turn actually run it. This is a code gate, not a prompt
    instruction the model or a fast talker could talk past.
    """
    tools = tools or {}
    write_scope_tools = write_scope_tools or set()
    system_prompt, version_id = _build_system_prompt(session, version=prompt_version, tool_manifest=tool_manifest)
    directive = _turn_directive(session, user_message)
    expected_script = _expected_script(session, user_message)

    initial_message = f"{directive}\n\n{user_message}" if directive else user_message
    messages: list[Message] = [*(history or []), {"role": "user", "content": initial_message}]
    tool_call_count = 0
    draft = ""

    with start_span("task_agent", "task_agent.run_turn", mode=session.mode.value, prompt_version=version_id):
        try:
            while True:
                try:
                    if cancellation_token is not None:
                        cancellation_token.check()
                        result = await cancellation_token.run(
                            router.complete_with_tools_and_fallback(
                                messages, system=system_prompt, tools=tool_definitions
                            )
                        )
                    else:
                        result = await router.complete_with_tools_and_fallback(
                            messages, system=system_prompt, tools=tool_definitions
                        )
                except LLMProviderError:
                    # Every configured provider (and its fallback chain) failed for
                    # this call — a fixed, safe apology beats an uncaught exception
                    # surfacing as a raw 500 to the client.
                    logger.info(
                        "turn_trace",
                        extra={"prompt_version": version_id, "tool_call_count": tool_call_count, "provider_error": True},
                    )
                    return TurnResult(
                        text=LLM_UNAVAILABLE_APOLOGY,
                        prompt_version=version_id,
                        tool_call_count=tool_call_count,
                        self_check_ok=True,
                        self_check_reason="",
                    )

                calls: list[ToolCall] = list(result.tool_calls)
                is_native = bool(calls)
                if not calls:
                    # No native tool calls came back — fall back to the
                    # legacy text convention before treating this as a final answer.
                    legacy_call = _parse_tool_call(result.text)
                    if legacy_call is not None:
                        calls = [legacy_call]

                if not calls:
                    draft = result.text
                    break

                if is_native:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": result.text or None,
                            "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in calls],
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": result.text})

                for call in calls:
                    if tool_call_count >= max_tool_calls:
                        logger.info(
                            "turn_trace",
                            extra={
                                "prompt_version": version_id,
                                "tool_call_count": tool_call_count,
                                "budget_exceeded": True,
                            },
                        )
                        return TurnResult(
                            text=CLARIFYING_QUESTION,
                            prompt_version=version_id,
                            tool_call_count=tool_call_count,
                            self_check_ok=True,
                            self_check_reason="",
                        )

                    if call.name in write_scope_tools and session.mode.is_voice:
                        confirmed = (
                            confirmation_token is not None
                            and confirmation_gate is not None
                            and confirmation_gate.consume(confirmation_token, call.name, call.args)
                        )
                        if not confirmed:
                            pending = (
                                confirmation_gate.request_confirmation(call.name, call.args)
                                if confirmation_gate is not None
                                else PendingConfirmation(token="", tool_name=call.name, args=call.args)
                            )
                            logger.info(
                                "turn_trace",
                                extra={
                                    "prompt_version": version_id,
                                    "tool_call_count": tool_call_count,
                                    "confirmation_required": call.name,
                                },
                            )
                            return TurnResult(
                                text=CONFIRMATION_REQUIRED_TEXT,
                                prompt_version=version_id,
                                tool_call_count=tool_call_count,
                                self_check_ok=True,
                                self_check_reason="",
                                pending_confirmation=pending,
                            )

                    tool_fn = tools.get(call.name)
                    tool_call_count += 1
                    if tool_fn is None:
                        result_text = f"Error: tool '{call.name}' is not available."
                    else:
                        try:
                            # A provider's native tool-calling sometimes returns
                            # `null`/None arguments for a zero-argument tool
                            # (json.loads("null") is a real None, not {}) — this
                            # is the one point every provider's tool calls funnel
                            # through, so it's fixed here rather than in each
                            # adapter. Any other tool-side failure (bad arg
                            # shape, an internal bug) is caught the same way —
                            # a reported error, never an uncaught 500.
                            result_text = await tool_fn(**(call.args or {}))
                        except Exception as e:  # noqa: BLE001 — any tool failure is reported, never crashes the turn
                            result_text = f"Error: tool '{call.name}' failed: {e}"
                    wrapped = wrap_untrusted(result_text, source=f"tool_result_{call.name}")
                    if is_native:
                        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": wrapped})
                    else:
                        messages.append({"role": "user", "content": wrapped})

                if cancellation_token is not None:
                    cancellation_token.check()

            async def check(text: str) -> tuple[bool, str]:
                coro = _self_check(
                    text,
                    session.mode,
                    router,
                    response_language=session.response_language,
                    is_code_mixed=session.is_code_mixed,
                    expected_script=expected_script,
                )
                return await (cancellation_token.run(coro) if cancellation_token is not None else coro)

            self_check_ok, self_check_reason = await check(draft)

            # Detected violations previously had zero effect on the shipped
            # answer — self_check_ok would just come back False while the
            # same non-compliant draft went out anyway. One bounded retry
            # (never more — same cost discipline as the check call itself)
            # gives the model a real chance to fix a length/language/
            # formatting violation before we give up and ship it regardless.
            if not self_check_ok:
                correction_text = _CORRECTION_REQUEST_TEMPLATE.format(reason=self_check_reason)
                if directive:
                    correction_text = f"{directive}\n\n{correction_text}"
                messages.append({"role": "assistant", "content": draft})
                messages.append({"role": "user", "content": correction_text})
                try:
                    correction_coro = router.complete_with_fallback(messages, system=system_prompt)
                    draft = await (
                        cancellation_token.run(correction_coro) if cancellation_token is not None else correction_coro
                    )
                    self_check_ok, self_check_reason = await check(draft)
                except LLMProviderError:
                    pass  # keep the original draft + its (failing) self_check result — better than crashing
        except TurnCancelled:
            logger.info(
                "turn_trace",
                extra={"prompt_version": version_id, "tool_call_count": tool_call_count, "cancelled": True},
            )
            return TurnResult(
                text="",
                prompt_version=version_id,
                tool_call_count=tool_call_count,
                self_check_ok=True,
                self_check_reason="",
                cancelled=True,
            )

        logger.info(
            "turn_trace",
            extra={
                "prompt_version": version_id,
                "tool_call_count": tool_call_count,
                "self_check_ok": self_check_ok,
            },
        )

        return TurnResult(
            text=sanitize_llm_output(draft),
            prompt_version=version_id,
            tool_call_count=tool_call_count,
            self_check_ok=self_check_ok,
            self_check_reason=self_check_reason,
        )
