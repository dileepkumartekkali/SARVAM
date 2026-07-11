const LANGUAGE_NAMES = {
  en: "English",
  hi: "Hindi",
  te: "Telugu",
  ta: "Tamil",
  kn: "Kannada",
  ml: "Malayalam",
  mr: "Marathi",
  gu: "Gujarati",
  pa: "Punjabi",
  bn: "Bengali",
  or: "Odia",
  as: "Assamese",
  ur: "Urdu",
  unknown: "Unknown",
};

/** Small, unobtrusive indicator — not a debug panel. Shows nothing until a
 * language has actually been detected. */
export default function LanguageBadge({ language, confidence, isCodeMixed }) {
  if (!language) return null;
  const name = LANGUAGE_NAMES[language] || language;
  const title =
    confidence != null ? `Detected: ${name} (${Math.round(confidence * 100)}% confidence)` : `Detected: ${name}`;

  return (
    <span className="language-badge" title={title}>
      <span className="language-badge__dot" aria-hidden="true" />
      {name}
      {isCodeMixed && <span className="language-badge__mixed">mixed</span>}
    </span>
  );
}
