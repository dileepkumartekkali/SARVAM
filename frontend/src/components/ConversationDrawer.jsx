import { useEffect, useRef, useState } from "react";

const TITLE_MAX_CHARS = 40;

function displayTitle(title) {
  if (!title) return "New chat";
  return title.length > TITLE_MAX_CHARS ? `${title.slice(0, TITLE_MAX_CHARS)}…` : title;
}

function ConversationRow({ conversation, active, onSwitch, onDelete }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!menuOpen) return;
    function onDocClick(e) {
      if (!rootRef.current?.contains(e.target)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [menuOpen]);

  return (
    <div className="conversation-drawer__row" ref={rootRef}>
      <button
        type="button"
        className={`conversation-drawer__item${active ? " conversation-drawer__item--active" : ""}`}
        onClick={onSwitch}
      >
        {displayTitle(conversation.title)}
      </button>
      <button
        type="button"
        className="conversation-drawer__row-menu-btn"
        onClick={() => setMenuOpen((o) => !o)}
        aria-label="Conversation options"
      >
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
          <path
            fill="currentColor"
            d="M12 8a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm0 6a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm0 6a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"
          />
        </svg>
      </button>
      {menuOpen && (
        <div className="conversation-drawer__row-menu">
          <button
            type="button"
            className="conversation-drawer__delete"
            onClick={() => {
              setMenuOpen(false);
              onDelete();
            }}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

/** Slide-in overlay — a ChatGPT/Claude-style conversation switcher, not a
 * permanent split-pane layout, so it works the same way on mobile and
 * desktop without a separate responsive variant. */
export default function ConversationDrawer({
  open,
  onClose,
  conversations,
  activeConversationId,
  onSwitchConversation,
  onNewConversation,
  onDeleteConversation,
}) {
  if (!open) return null;

  return (
    <div className="conversation-drawer-overlay" onClick={onClose}>
      <div className="conversation-drawer" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className="conversation-drawer__new"
          onClick={() => {
            onNewConversation();
            onClose();
          }}
        >
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
            <path fill="currentColor" d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6V5Z" />
          </svg>
          New chat
        </button>
        <div className="conversation-drawer__list">
          {conversations.map((c) => (
            <ConversationRow
              key={c.id}
              conversation={c}
              active={c.id === activeConversationId}
              onSwitch={() => {
                onSwitchConversation(c.id);
                onClose();
              }}
              onDelete={() => onDeleteConversation(c.id)}
            />
          ))}
          {conversations.length === 0 && <div className="conversation-drawer__empty">No conversations yet</div>}
        </div>
      </div>
    </div>
  );
}
