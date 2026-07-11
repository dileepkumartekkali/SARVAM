import ReactMarkdown from "react-markdown";

/** TEXT_MODE allows markdown (lists/headers/bold) per agent_system_prompt.md
 * — rendered here for assistant messages only. User messages are shown as
 * plain text: it's an echo of what they typed, not an LLM output the
 * TEXT_MODE formatting rules apply to. */
export default function MessageBubble({ role, text, streaming }) {
  const isUser = role === "user";
  return (
    <div className={`message-row message-row--${role}`}>
      <div className={`message-bubble message-bubble--${role}`}>
        {isUser ? (
          <span>{text}</span>
        ) : (
          <div className="message-bubble__markdown">
            <ReactMarkdown>{text || " "}</ReactMarkdown>
          </div>
        )}
        {streaming && <span className="message-bubble__cursor" aria-hidden="true" />}
      </div>
    </div>
  );
}
