import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";

export default function ChatView({ messages }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, messages[messages.length - 1]?.text]);

  return (
    <div className="chat-view">
      {messages.length === 0 && (
        <div className="chat-view__empty">
          <svg viewBox="0 0 24 24" width="36" height="36" aria-hidden="true">
            <path
              fill="currentColor"
              d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V21h2v-2.07A7 7 0 0 0 19 12h-2Z"
            />
          </svg>
          <p>Ask me anything — in any of 13 languages.</p>
        </div>
      )}
      {messages.map((m) => (
        <MessageBubble key={m.id} role={m.role} text={m.text} streaming={m.streaming} audioPath={m.audioPath} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
