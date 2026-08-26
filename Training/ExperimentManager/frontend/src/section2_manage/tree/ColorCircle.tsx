import { useEffect, useRef, useState } from "react";
import { HexColorPicker } from "react-colorful";

interface Props {
  color: string;
  onChange: (c: string) => void;
  size?: number;
}

export function ColorCircle({ color, onChange, size = 14 }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState(color);

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        if (draft !== color) onChange(draft);
      }
    };
    window.addEventListener("mousedown", onClickOutside);
    return () => window.removeEventListener("mousedown", onClickOutside);
  }, [open, draft, color, onChange]);

  useEffect(() => setDraft(color), [color]);

  return (
    <div className="color-circle-wrap" ref={ref}>
      <button
        type="button"
        className="color-circle-btn"
        style={{ background: color, width: size, height: size }}
        onClick={(e) => {
          e.stopPropagation();
          if (open && draft !== color) onChange(draft);
          setOpen((v) => !v);
        }}
        aria-label="Pick color"
      />
      {open && (
        <div className="color-picker-popover" onMouseDown={(e) => e.stopPropagation()}>
          <HexColorPicker color={draft} onChange={setDraft} />
          <div className="color-picker-footer">
            <input
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
            />
          </div>
        </div>
      )}
    </div>
  );
}
