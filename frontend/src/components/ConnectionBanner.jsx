import { ConnectionState } from "../store/useAppStore";

/** A slim banner, not a full-screen loader — the conversation view stays
 * mounted and scrolled where it was underneath. Renders nothing when
 * connected, so it never occupies layout space in the common case. */
export default function ConnectionBanner({ connectionState }) {
  if (connectionState === ConnectionState.CONNECTED) return null;

  const isReconnecting = connectionState === ConnectionState.RECONNECTING;

  return (
    <div className={`connection-banner${isReconnecting ? " connection-banner--reconnecting" : ""}`} role="status">
      <span className="connection-banner__spinner" aria-hidden="true" />
      {isReconnecting ? "Reconnecting…" : "Disconnected — retrying"}
    </div>
  );
}
