import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { HexColorPicker } from "react-colorful";
import { RUN_COLORS } from "../../state/uiStore";

interface Props {
  color: string;
  onChange: (c: string) => void;
  size?: number;
}

export function ColorCircle({ color, onChange, size = 14 }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const popoverRef = useRef<HTMLDivElement | null>(null);
  const [draft, setDraft] = useState(color);
  const [position, setPosition] = useState({ left: 0, top: 0 });

  const positionPopover = useCallback(() => {
    const button = buttonRef.current;
    const popover = popoverRef.current;
    if (!button || !popover) return;

    const buttonRect = button.getBoundingClientRect();
    const popoverRect = popover.getBoundingClientRect();
    const viewportMargin = 8;
    const gap = 4;
    let left = buttonRect.right - popoverRect.width;
    let top = buttonRect.bottom + gap;

    if (top + popoverRect.height > window.innerHeight - viewportMargin) {
      top = buttonRect.top - popoverRect.height - gap;
    }

    setPosition({
      left: Math.min(
        Math.max(viewportMargin, left),
        window.innerWidth - popoverRect.width - viewportMargin,
      ),
      top: Math.min(
        Math.max(viewportMargin, top),
        window.innerHeight - popoverRect.height - viewportMargin,
      ),
    });
  }, []);

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        ref.current &&
        !ref.current.contains(target) &&
        !popoverRef.current?.contains(target)
      ) {
        setOpen(false);
        if (draft !== color) onChange(draft);
      }
    };
    window.addEventListener("mousedown", onClickOutside);
    return () => window.removeEventListener("mousedown", onClickOutside);
  }, [open, draft, color, onChange]);

  useEffect(() => setDraft(color), [color]);

  useLayoutEffect(() => {
    if (!open) return;
    positionPopover();
    window.addEventListener("resize", positionPopover);
    window.addEventListener("scroll", positionPopover, true);
    return () => {
      window.removeEventListener("resize", positionPopover);
      window.removeEventListener("scroll", positionPopover, true);
    };
  }, [open, positionPopover]);

  return (
    <div className="color-circle-wrap" ref={ref}>
      <button
        ref={buttonRef}
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
      {open && createPortal(
        <div
          ref={popoverRef}
          className="color-picker-popover"
          style={position}
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <HexColorPicker color={draft} onChange={setDraft} />
          <div className="color-picker-presets" aria-label="Preset colors">
            {RUN_COLORS.map((preset) => (
              <button
                key={preset}
                type="button"
                className="color-picker-preset"
                style={{ background: preset }}
                title={`Use ${preset}`}
                aria-label={`Use ${preset}`}
                onClick={() => {
                  setDraft(preset);
                  onChange(preset);
                }}
              />
            ))}
          </div>
          <div className="color-picker-footer">
            <input
              type="text"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
            />
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
