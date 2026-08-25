import { useUIStore } from "../state/uiStore";

export function ThemeToggle() {
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);
  const next = theme === "light" ? "dark" : "light";
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => setTheme(next)}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
    >
      {theme === "light" ? "🌙" : "☀"}
      <span>{theme === "light" ? "Dark" : "Light"}</span>
    </button>
  );
}
