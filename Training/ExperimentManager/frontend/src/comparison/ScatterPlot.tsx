import { useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from "react";

export type PlotPoint = { key: string; name: string; path: string; color: string; x: number; y: number };
type Props = {
  points: PlotPoint[]; selected: string | null; xTitle: string; yTitle: string;
  onPick: (key: string) => void; onClear: () => void;
};
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumSignificantDigits: 5 });
function format(value: number) {
  const v = Math.abs(value);
  if (v !== 0 && (v < 0.001 || v >= 1e15)) return Number(value.toPrecision(5)).toExponential();
  return compact.format(value);
}

// Work in normalized units so padding, tick spacing and interpolation also work
// for very large values and for small learning rates.
function scale(points: PlotPoint[], axis: "x" | "y", tickCount: number) {
  let min = points[0][axis], max = min;
  for (const point of points) {
    min = Math.min(min, point[axis]);
    max = Math.max(max, point[axis]);
  }
  const unit = Math.max(Math.abs(min), Math.abs(max)) || 1;
  const lo = min / unit, hi = max / unit;
  const padding = lo === hi ? 0.08 : (hi - lo) * 0.08;
  const lower = lo - padding, upper = hi + padding;
  const rawStep = (upper - lower) / tickCount;
  const exponent = Math.floor(Math.log10(rawStep) + Math.log10(unit));
  const stepUnit = Math.pow(10, Math.max(-323, exponent)) / unit || rawStep;
  const multiple = [1, 2, 2.5, 5, 10].find((n) => n * stepUnit >= rawStep) ?? 10;
  const step = multiple * stepUnit;
  const first = Math.ceil(lower / step);
  const stepValue = step * unit;
  const ticks: { at: number; label: string }[] = [];
  for (let i = 0; i <= tickCount + 1; i++) {
    const n = (first + i) * step;
    if (n > upper) break;
    const value = n * unit;
    if (Number.isFinite(value)) {
      // Closely spaced metrics can share the same compact label.
      const label = format(value) === format(value + stepValue) ? String(Number(value.toPrecision(14))) : format(value);
      ticks.push({ at: (n - lower) / (upper - lower), label });
    }
  }
  return {
    ticks,
    position: (v: number) => (v / unit - lower) / (upper - lower),
    value: (t: number) => (lower + t * (upper - lower)) * unit,
  };
}

export function ScatterPlot({ points, selected, xTitle, yTitle, onPick, onClear }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 640, height: 480 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  useLayoutEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
      setCursor(null);
    });
    if (box.current) observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  const left = size.width < 450 ? 64 : 84, top = 24;
  const width = Math.max(1, size.width - left - 28), height = Math.max(1, size.height - top - 68);
  const { xs, ys, dots } = useMemo(() => {
    const xs = scale(points, "x", Math.max(2, Math.min(10, Math.floor(width / 100))));
    const ys = scale(points, "y", Math.max(2, Math.min(10, Math.floor(height / 66))));
    const dots = points.map((p) => ({ ...p, cx: left + xs.position(p.x) * width,
      cy: top + (1 - ys.position(p.y)) * height }));
    return { xs, ys, dots };
  }, [points, width, height, left]);

  function locate(event: PointerEvent<SVGSVGElement> | React.MouseEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    return x >= left && x <= left + width && y >= top && y <= top + height ? { x, y } : null;
  }
  function hits(pos: typeof cursor) {
    if (!pos) return [];
    return dots.map((dot) => ({ dot, distance: Math.hypot(pos.x - dot.cx, pos.y - dot.cy) }))
      .filter(({ distance }) => distance <= 10).sort((a, b) => a.distance - b.distance);
  }
  const nearby = hits(cursor);
  const closest = nearby[0]?.dot;
  const stack = closest ? nearby.filter(({ dot }) => dot.cx === closest.cx && dot.cy === closest.cy) : [];
  const hover = stack.find(({ dot }) => dot.key === selected)?.dot ?? closest;
  const focus = dots.find((d) => d.key === selected);
  const cross = hover ? { x: hover.cx, y: hover.cy } : cursor;
  return <div className="scatter-frame" ref={box}>
    <svg className="scatter" role="group" aria-label={`Scatter plot of ${yTitle} by ${xTitle}`}
      onPointerMove={(e) => setCursor(locate(e))} onPointerLeave={() => setCursor(null)}
      onClick={(e) => {
        const nearest = hits(locate(e));
        const first = nearest[0]?.dot;
        if (!first) { onClear(); return; }
        const same = nearest.filter(({ dot }) => dot.cx === first.cx && dot.cy === first.cy);
        const index = same.findIndex(({ dot }) => dot.key === selected);
        onPick(same[(index + 1) % same.length].dot.key);
      }}>
      <rect className="scatter-area" x={left} y={top} width={width} height={height} rx="4" />
      {xs.ticks.map((t, i) => <g key={i} transform={`translate(${left + width * t.at},0)`}>
        <line className="scatter-grid" y1={top} y2={top + height} />
        <text className="scatter-tick" textAnchor="middle" y={top + height + 21}>{t.label}</text>
      </g>)}
      {ys.ticks.map((t, i) => <g key={i} transform={`translate(0,${top + height * (1 - t.at)})`}>
        <line className="scatter-grid" x1={left} x2={left + width} />
        <text className="scatter-tick" textAnchor="end" x={left - 10} dy=".34em">{t.label}</text>
      </g>)}
      <text className="scatter-title" x={left + width / 2} y={size.height - 14} textAnchor="middle">{xTitle}</text>
      <text className="scatter-title" transform={`translate(17 ${top + height / 2}) rotate(-90)`} textAnchor="middle">{yTitle}</text>
      {cross && <g className="scatter-cross">
        <line x1={cross.x} x2={cross.x} y1={top} y2={top + height} />
        <line y1={cross.y} y2={cross.y} x1={left} x2={left + width} />
      </g>}
      {dots.map((dot) => <circle key={dot.key} className="scatter-dot" cx={dot.cx} cy={dot.cy} r={6}
        fill={dot.color} tabIndex={0} role="button" aria-label={dot.path} aria-pressed={selected === dot.key}
        onFocus={() => setCursor({ x: dot.cx, y: dot.cy })} onBlur={() => setCursor(null)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(dot.key); }
          if (e.key === "Escape") onClear();
        }} />)}
      {focus && <circle className="scatter-selection" cx={focus.cx} cy={focus.cy} r={8} fill={focus.color}
        stroke={focus.color} strokeWidth={8} strokeOpacity={0.4} pointerEvents="none" />}
    </svg>
    {cross && <div role="tooltip" className="scatter-tip" style={{
      left: cross.x > size.width / 2 ? undefined : cross.x + 14,
      right: cross.x > size.width / 2 ? size.width - cross.x + 14 : undefined,
      top: cross.y > size.height - 115 ? undefined : cross.y + 12,
      bottom: cross.y > size.height - 115 ? size.height - cross.y + 12 : undefined,
    }}>
      {hover && <strong>{hover.name}</strong>}
      <div>X <b>{Number((hover?.x ?? xs.value((cross.x - left) / width)).toPrecision(10)).toString()}</b></div>
      <div>Y <b>{Number((hover?.y ?? ys.value(1 - (cross.y - top) / height)).toPrecision(10)).toString()}</b></div>
      {stack.length > 1 && <small>{stack.length} runs here · click to cycle</small>}
    </div>}
  </div>;
}
