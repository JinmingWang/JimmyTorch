import { useEffect, useRef, useState } from "react";
import { useUIStore } from "../../state/uiStore";

interface Props {
  tag: string;
}

/** Gear button that opens a popover with smoothing / xlim / ylim / log-scale controls. */
export function GearPopover({ tag }: Props) {
  const smoothing = useUIStore((s) => s.smoothing[tag] ?? 0);
  const setSmoothing = useUIStore((s) => s.setSmoothing);
  const xlim = useUIStore((s) => s.xlim[tag] ?? [null, null]);
  const setXlim = useUIStore((s) => s.setXlim);
  const ylim = useUIStore((s) => s.ylim[tag] ?? [null, null]);
  const setYlim = useUIStore((s) => s.setYlim);
  const logScale = useUIStore((s) => s.logScale[tag] ?? false);
  const setLogScale = useUIStore((s) => s.setLogScale);

  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onDocDown);
    return () => window.removeEventListener("mousedown", onDocDown);
  }, [open]);

  return (
    <span className="gear-wrap" ref={ref}>
      <button
        type="button"
        className={`curve-btn gear-btn${open ? " open" : ""}`}
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        title="Curve settings"
        aria-label="Curve settings"
        aria-expanded={open}
      >
        ⚙
      </button>
      {open && (
        <div className="gear-popover" onMouseDown={(e) => e.stopPropagation()}>
          <div className="gear-row">
            <label className="gear-label" htmlFor={`sm-${tag}`}>Smoothing</label>
            <input
              id={`sm-${tag}`}
              type="range"
              min={0}
              max={0.99}
              step={0.01}
              value={smoothing}
              onChange={(e) => setSmoothing(tag, Number(e.target.value))}
            />
            <span className="gear-value">{smoothing.toFixed(2)}</span>
          </div>

          <RangeRow
            label="X range"
            value={xlim}
            onChange={(a, b) => setXlim(tag, a, b)}
          />
          <RangeRow
            label="Y range"
            value={ylim}
            onChange={(a, b) => setYlim(tag, a, b)}
          />

          <div className="gear-row gear-row-toggle">
            <label className="gear-label" htmlFor={`log-${tag}`}>Log scale (y)</label>
            <input
              id={`log-${tag}`}
              type="checkbox"
              checked={logScale}
              onChange={(e) => setLogScale(tag, e.target.checked)}
            />
          </div>
        </div>
      )}
    </span>
  );
}

function RangeRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: [number | null, number | null];
  onChange: (min: number | null, max: number | null) => void;
}) {
  const [a, b] = value;
  const parse = (s: string): number | null => {
    const t = s.trim();
    if (t === "") return null;
    const n = Number(t);
    return isFinite(n) ? n : null;
  };
  return (
    <div className="gear-row">
      <label className="gear-label">{label}</label>
      <input
        className="gear-num"
        placeholder="auto"
        defaultValue={a == null ? "" : String(a)}
        onBlur={(e) => onChange(parse(e.target.value), b)}
        onKeyDown={(e) => { if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur(); }}
      />
      <input
        className="gear-num"
        placeholder="auto"
        defaultValue={b == null ? "" : String(b)}
        onBlur={(e) => onChange(a, parse(e.target.value))}
        onKeyDown={(e) => { if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur(); }}
      />
    </div>
  );
}
