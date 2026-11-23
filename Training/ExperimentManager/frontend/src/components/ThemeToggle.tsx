import { useUIStore } from "../state/uiStore";

export function ThemeToggle({ reload = true }: { reload?: boolean }) {
  const theme = useUIStore((s) => s.theme);
  const setTheme = useUIStore((s) => s.setTheme);
  const next = theme === "light" ? "dark" : "light";
  const switchTheme = async () => {
    await setTheme(next);
    if (reload) window.location.reload();
  };
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={() => { void switchTheme(); }}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
    >
      {theme === "light" ? "🌙" : "☀"}
      <span>{theme === "light" ? "Dark" : "Light"}</span>
    </button>
  );
}
