import { useEffect, useRef, useState } from "react";

function initials(email) {
  if (!email) return "?";
  return email[0].toUpperCase();
}

/** Avatar button — user details + sign-out are behind a click, not shown by
 * default (replaces the old bare "Sign out" text button). */
export default function ProfileMenu({ user, onLogout }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e) {
      if (!rootRef.current?.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  return (
    <div className="profile-menu" ref={rootRef}>
      <button type="button" className="profile-menu__avatar" onClick={() => setOpen((o) => !o)} aria-label="Account">
        {user?.avatarUrl ? (
          <img src={user.avatarUrl} alt="" />
        ) : (
          <span className="profile-menu__initials">{initials(user?.email)}</span>
        )}
      </button>
      {open && (
        <div className="profile-menu__popover">
          <div className="profile-menu__email">{user?.email}</div>
          <button type="button" className="profile-menu__logout" onClick={onLogout}>
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
