import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useRef, useState } from "react";
import { HexColorPicker } from "react-colorful";
export function ColorCircle({ color, onChange, size = 14 }) {
    const [open, setOpen] = useState(false);
    const ref = useRef(null);
    const [draft, setDraft] = useState(color);
    useEffect(() => {
        if (!open)
            return;
        const onClickOutside = (e) => {
            if (ref.current && !ref.current.contains(e.target)) {
                setOpen(false);
                if (draft !== color)
                    onChange(draft);
            }
        };
        window.addEventListener("mousedown", onClickOutside);
        return () => window.removeEventListener("mousedown", onClickOutside);
    }, [open, draft, color, onChange]);
    useEffect(() => setDraft(color), [color]);
    return (_jsxs("div", { className: "color-circle-wrap", ref: ref, children: [_jsx("button", { type: "button", className: "color-circle-btn", style: { background: color, width: size, height: size }, onClick: (e) => {
                    e.stopPropagation();
                    setOpen((v) => !v);
                }, "aria-label": "Pick color" }), open && (_jsxs("div", { className: "color-picker-popover", onMouseDown: (e) => e.stopPropagation(), children: [_jsx(HexColorPicker, { color: draft, onChange: setDraft }), _jsxs("div", { className: "color-picker-footer", children: [_jsx("input", { type: "text", value: draft, onChange: (e) => setDraft(e.target.value) }), _jsx("button", { type: "button", className: "btn-primary", onClick: () => {
                                    onChange(draft);
                                    setOpen(false);
                                }, children: "Apply" })] })] }))] }));
}
