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
        <div className="chat-view__empty">Ask me anything — in any of 13 languages.</div>
      )}
      {messages.map((m) => (
        <MessageBubble key={m.id} role={m.role} text={m.text} streaming={m.streaming} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
