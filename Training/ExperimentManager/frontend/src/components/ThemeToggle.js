import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useUIStore } from "../state/uiStore";
export function ThemeToggle() {
    const theme = useUIStore((s) => s.theme);
    const setTheme = useUIStore((s) => s.setTheme);
    const next = theme === "light" ? "dark" : "light";
    return (_jsxs("button", { type: "button", className: "theme-toggle", onClick: () => setTheme(next), title: `Switch to ${next} theme`, "aria-label": `Switch to ${next} theme`, children: [theme === "light" ? "🌙" : "☀", _jsx("span", { children: theme === "light" ? "Dark" : "Light" })] }));
}
