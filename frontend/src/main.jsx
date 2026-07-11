import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { useAppStore } from "./store/useAppStore";
import "./styles/theme.css";
import "./styles/components.css";

// Dev-only debug hook so the store can be poked from the console without a
// live backend — never included in a production build.
if (import.meta.env.DEV) {
  window.__appStore = useAppStore;
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
