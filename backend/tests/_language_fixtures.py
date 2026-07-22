"""Shared test sentences for the 13 supported languages, reused across
test_language_agent.py and test_graph.py."""

# (text, expected_language_code) — one clean, native-script sentence per
# language. Devanagari (hi/mr) and Bengali-script (bn/as) pairs each carry a
# lexical/character marker that disambiguates them; the other 9 are
# unambiguous from Unicode script alone.
PURE_LANGUAGE_CASES = [
    ("Hello, how are you today?", "en"),
    ("మీరు ఎలా ఉన్నారు?", "te"),
    ("नमस्ते, आप कैसे हैं?", "hi"),
    ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "ta"),
    ("ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಿ?", "kn"),
    ("നമസ്കാരം, സുഖമാണോ?", "ml"),
    ("नमस्कार, तुम्ही कसे आहात?", "mr"),
    ("નમસ્તે, તમે કેમ છો?", "gu"),
    ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", "pa"),
    ("নমস্কার, আপনি কেমন আছেন?", "bn"),
    ("ନମସ୍କାର, ଆପଣ କେମିତି ଅଛନ୍ତି?", "or"),
    ("নমস্কাৰ, আপুনি কেনে আছে?", "as"),
    ("السلام علیکم، آپ کیسے ہیں؟", "ur"),
]

# (text, expected_language_code, needs_llm_fallback) — representative
# code-mixed examples. Only the Telugu one is confirmed from the task brief;
# the other four are illustrative Hindi/Tamil/Bengali/Punjabi-English mixes
# (the full "master brief" text wasn't available in this session) chosen to
# avoid any accidental overlap with the deterministic English-stopword list,
# so they genuinely exercise the LLM-fallback path.
CODE_MIXED_CASES = [
    ("Bro meeting ki vasthunnava?", "te", False),
    ("Kal ka plan kya hai yaar?", "hi", True),
    ("Enna panra ippo, meeting late aayiduma?", "ta", True),
    ("Tumi ki korcho ekhon, office ja bo na?", "bn", True),
    ("Tera plan ki hai aj, ghar aavega ki nahi?", "pa", True),
    # Real bug hit live: "mtouch labs ceo evaru?" (no marker word in the
    # original list) fell through to the LLM classifier, which
    # inconsistently misclassified the SAME exact text as Kannada at 0.95
    # confidence 2 times out of 3 across repeated real calls -- high enough
    # to bypass the low-confidence safety net entirely. "evaru"/"epudu"/
    # "ayindhi" added to the deterministic keyword list so this class of
    # query is now caught without ever touching the unreliable LLM path.
    ("mtouch labs ceo evaru?", "te", False),
    ("mtouch labs ceo evaru and epudu start ayindhi?", "te", False),
]
