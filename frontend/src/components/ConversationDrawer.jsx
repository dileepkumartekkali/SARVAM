const TITLE_MAX_CHARS = 40;

function displayTitle(title) {
  if (!title) return "New chat";
  return title.length > TITLE_MAX_CHARS ? `${title.slice(0, TITLE_MAX_CHARS)}…` : title;
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
            <button
              key={c.id}
              type="button"
              className={`conversation-drawer__item${c.id === activeConversationId ? " conversation-drawer__item--active" : ""}`}
              onClick={() => {
                onSwitchConversation(c.id);
                onClose();
              }}
            >
              {displayTitle(c.title)}
            </button>
          ))}
          {conversations.length === 0 && <div className="conversation-drawer__empty">No conversations yet</div>}
        </div>
      </div>
    </div>
  );
}
