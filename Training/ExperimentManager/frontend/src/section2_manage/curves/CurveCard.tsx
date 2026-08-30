import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import type { AlignedData, Options } from "uplot";
import "uplot/dist/uPlot.min.css";
import type { ScalarPoint } from "../../api/rest";
import { fetchScalars } from "../../api/rest";
import { effectiveColor, findRun, useUIStore } from "../../state/uiStore";
import { useLiveStore } from "../../state/liveStore";
import { computeRange, smoothSeries } from "./smooth";
import { bucketDownsample } from "./downsample";
import { GearPopover } from "./GearPopover";

interface Props {
  tag: string;
  runKeys: string[];
}

interface RunSeries {
  runKey: string;
  color: string;
  isLive: boolean;
  points: ScalarPoint[];
}

interface LineDrag { which: "start" | "end"; }
interface PanState { startClientX: number; startClientY: number; x0: number; x1: number; y0: number; y1: number; }
interface ContextMenuState { x: number; y: number; }

const HIT_DISTANCE_PX = 8;

export function CurveCard({ tag, runKeys }: Props) {
  const smoothing = useUIStore((s) => s.smoothing[tag] ?? 0);
  const xlim = useUIStore((s) => s.xlim[tag] ?? [null, null]);
  const ylim = useUIStore((s) => s.ylim[tag] ?? [null, null]);
  const logScale = useUIStore((s) => s.logScale[tag] ?? false);
  const rangeStartVal = useUIStore((s) => s.rangeStart[tag] ?? null);
  const rangeEndVal = useUIStore((s) => s.rangeEnd[tag] ?? null);
  const setRange = useUIStore((s) => s.setRange);
  const expandedTag = useUIStore((s) => s.expandedTag);
  const setExpandedTag = useUIStore((s) => s.setExpandedTag);
  const refreshTick = useUIStore((s) => s.refreshTick);
  const loadedTree = useUIStore((s) => s.loadedTree);

  const fullscreen = expandedTag === tag;

  const [rawSeries, setRawSeries] = useState<RunSeries[]>([]);

  const runKeysKey = runKeys.join("|");
  useEffect(() => {
    if (!loadedTree) return;
    let cancelled = false;
    async function loadAll() {
      const currentTree = useUIStore.getState().tree;
      const nodes = runKeys.map((k) => findRun(currentTree, k)).filter(Boolean);
      const results = await Promise.all(
        nodes.map(async (node) => {
          const resp = await fetchScalars(node!.dataset, node!.model, node!.run_name, tag, 20_000);
          const live = useLiveStore.getState().snapshot;
          const isLive =
            !!live?.run &&
            live.run.dataset === node!.dataset &&
            live.run.model === node!.model &&
            live.run.run_name === node!.run_name &&
            live.status !== "done" && live.status !== "error";
          return {
            runKey: `${node!.dataset}\u0000${node!.model}\u0000${node!.run_name}`,
            color: effectiveColor(node!),
            isLive,
            points: bucketDownsample(resp.points, 500),
          } as RunSeries;
        }),
      );
      if (!cancelled) setRawSeries(results.filter((x): x is RunSeries => x !== null));
    }
    void loadAll();
    return () => { cancelled = true; };
    // `tree` excluded from deps: live tree updates during training must not reset zoom/pan.
    // `loadedTree` included so the first fetch waits for the tree to actually load.
  }, [tag, runKeysKey, refreshTick, loadedTree]);

  const displayed = useMemo(() => {
    return rawSeries.map((s) => ({ ...s, points: smoothSeries(s.points, smoothing) }));
  }, [rawSeries, smoothing]);

  const hasLive = displayed.some((s) => s.isLive);

  const doReset = useCallback(() => {
    (window as unknown as { __curve_reset__?: Record<string, () => void> }).__curve_reset__?.[tag]?.();
  }, [tag]);

  return (
    <div className={["curve-card", fullscreen ? "expanded" : ""].filter(Boolean).join(" ")}>
      <div className="curve-header">
        <span className="curve-title">{tag}</span>
        {hasLive && <span className="curve-live-pill">● Training</span>}
        <span className="curve-header-actions">
          <button type="button" className="curve-btn" onClick={doReset} title="Reset zoom to fit data" aria-label="Reset zoom to fit data">⛶</button>
          <GearPopover tag={tag} />
          <button
            type="button"
            className="curve-btn"
            onClick={() => setExpandedTag(fullscreen ? null : tag)}
            title={fullscreen ? "Collapse to grid" : "Expand across row"}
            aria-label={fullscreen ? "Collapse" : "Expand"}
          >
            {fullscreen ? "⤡" : "⤢"}
          </button>
        </span>
      </div>
      <UPlotWrap
        tag={tag}
        series={displayed}
        xlim={xlim}
        ylim={ylim}
        logScale={logScale}
        fullscreen={fullscreen}
        rangeStartVal={rangeStartVal}
        rangeEndVal={rangeEndVal}
        setRange={setRange}
      />
    </div>
  );
}

function UPlotWrap({
  tag,
  series,
  xlim,
  ylim,
  logScale,
  fullscreen,
  rangeStartVal,
  rangeEndVal,
  setRange,
}: {
  tag: string;
  series: RunSeries[];
  xlim: [number | null, number | null];
  ylim: [number | null, number | null];
  logScale: boolean;
  fullscreen: boolean;
  rangeStartVal: number | null;
  rangeEndVal: number | null;
  setRange: (tag: string, s: number | null, e: number | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const rangeRef = useRef<{ start: number | null; end: number | null }>({ start: rangeStartVal, end: rangeEndVal });
  // While a line drag is in progress, suppress the useEffect sync so the in-progress drag isn't wiped by a re-render.
  const draggingLineRef = useRef(false);
  useEffect(() => {
    if (draggingLineRef.current) return;
    rangeRef.current = { start: rangeStartVal, end: rangeEndVal };
  }, [rangeStartVal, rangeEndVal]);

  // Interactive zoom/pan state kept in a ref so it survives data refreshes without any re-render.
  const scaleRef = useRef<{ x?: [number, number]; y?: [number, number] }>({});
  // Latest data-derived x/y ranges, used by Reset to snap back to the auto range.
  const dataRangeRef = useRef<{ x: [number, number]; y: [number, number] }>({ x: [0, 1], y: [0, 1] });
  // True while a mouse button is held down over the plot; blocks auto-refit during live refreshes.
  const mouseDownRef = useRef(false);
  // True when the user has zoomed/panned interactively; blocks auto-refit until Reset.
  const hasUserZoomedRef = useRef(false);
  // Positive counter while WE are calling setScale; the setScale hook uses this to distinguish
  // programmatic scale changes (ready hook, reset, auto-refit) from real user interactions.
  const programmaticSetScaleRef = useRef(0);
  // True immediately after a plot rebuild so the data-only effect can skip a redundant setData.
  const justRebuiltRef = useRef(false);

  const [size, setSize] = useState({ w: 400, h: fullscreen ? 480 : 260 });
  const [cursorIdx, setCursorIdx] = useState<number | null>(null);
  const [menu, setMenu] = useState<ContextMenuState | null>(null);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (!r) return;
      setSize({
        w: Math.max(80, Math.floor(r.width)),
        h: fullscreen ? Math.max(360, Math.floor(r.height)) : 260,
      });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fullscreen]);

  const { data, opts, xRange, yRange } = useMemo(() => {
    if (series.length === 0) {
      return {
        data: [[]] as unknown as AlignedData,
        opts: null as Options | null,
        xRange: [0, 1] as [number, number],
        yRange: [0, 1] as [number, number],
      };
    }
    const stepSet = new Set<number>();
    for (const s of series) for (const p of s.points) stepSet.add(p.step);
    const steps = Array.from(stepSet).sort((a, b) => a - b);
    const stepIndex = new Map<number, number>();
    steps.forEach((s, i) => stepIndex.set(s, i));

    const yArrays: (number | null)[][] = series.map(() => new Array(steps.length).fill(null));
    for (let i = 0; i < series.length; i++) {
      for (const p of series[i].points) {
        if (logScale && p.value <= 0) continue;
        yArrays[i][stepIndex.get(p.step)!] = p.value;
      }
    }
    const data: AlignedData = [steps, ...yArrays] as AlignedData;

    const [xMinSet, xMaxSet] = xlim;
    const [yMinSet, yMaxSet] = ylim;
    const xIsAuto = xMinSet == null && xMaxSet == null;
    const yIsAuto = yMinSet == null && yMaxSet == null;

    let seriesForY = series;
    if (yIsAuto && !xIsAuto) {
      seriesForY = series.map((s) => ({
        ...s,
        points: s.points.filter((p) => (xMinSet == null || p.step >= xMinSet) && (xMaxSet == null || p.step <= xMaxSet)),
      }));
    }
    if (logScale) seriesForY = seriesForY.map((s) => ({ ...s, points: s.points.filter((p) => p.value > 0) }));
    let seriesForX = series;
    if (xIsAuto && !yIsAuto) {
      seriesForX = series.map((s) => ({
        ...s,
        points: s.points.filter((p) => (yMinSet == null || p.value >= yMinSet) && (yMaxSet == null || p.value <= yMaxSet)),
      }));
    }

    const yRange = computeRange(seriesForY.map((s) => s.points), ylim, "y");
    const xRange = computeRange(seriesForX.map((s) => s.points), xlim, "x");

    const yScale: uPlot.Scale = logScale
      ? { distr: 3 }
      : {};

    const opts: Options = {
      width: size.w,
      height: size.h,
      padding: [24, 8, 0, 0],
      cursor: {
        drag: { x: true, y: true, uni: 0, setScale: true },
      },
      scales: {
        x: { time: false },
        y: yScale,
      },
      axes: [
        {
          stroke: cssVar("--text-dim", "#7f8ea0"),
          grid: { stroke: cssVar("--border", "#22303d") },
          values: (_u, ticks) => ticks.map((t) => formatXAxisStep(t)),
        },
        {
          stroke: cssVar("--text-dim", "#7f8ea0"),
          grid: { stroke: cssVar("--border", "#22303d") },
          size: 72,
          values: (_u, ticks) => ticks.map((t) => t == null ? "" : formatY(t)),
        },
      ],
      series: [
        { label: "step" },
        ...series.map((s) => ({
          label: labelForKey(s.runKey),
          stroke: s.color,
          width: s.isLive ? 2.5 : 1.75,
          spanGaps: true,
        })),
      ],
      legend: { show: false },
    };
    return { data, opts, yRange, xRange };
  }, [series, size.w, size.h, xlim[0], xlim[1], ylim[0], ylim[1], logScale]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Signature of everything that requires a full plot rebuild. Excludes point-values-only changes
  // so live data updates go through setData without destroying the plot.
  const optsStructKey = useMemo(
    () =>
      series.map((s) => `${s.runKey}|${s.color}|${s.isLive ? 1 : 0}`).join(";") +
      `#${size.w},${size.h}` +
      `#${xlim[0] ?? "a"},${xlim[1] ?? "a"}` +
      `#${ylim[0] ?? "a"},${ylim[1] ?? "a"}` +
      `#${logScale ? 1 : 0}`,
    [series, size.w, size.h, xlim[0], xlim[1], ylim[0], ylim[1], logScale],  // eslint-disable-line react-hooks/exhaustive-deps
  );

  // Ensure default range covers full data range on first data load OR after a reset that nulls it.
  const dataXMin = data && data[0] && (data[0] as number[]).length > 0 ? (data[0] as number[])[0] : null;
  const dataXMax = data && data[0] && (data[0] as number[]).length > 0 ? (data[0] as number[])[(data[0] as number[]).length - 1] : null;
  useEffect(() => {
    if (dataXMin === null || dataXMax === null) return;
    if (rangeStartVal === null && rangeEndVal === null) setRange(tag, dataXMin, dataXMax);
  }, [dataXMin, dataXMax, rangeStartVal, rangeEndVal]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Keep the data-range ref in sync so Reset knows what to snap back to.
  dataRangeRef.current = { x: xRange, y: yRange };

  useEffect(() => {
    scaleRef.current.y = undefined;
  }, [logScale]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !opts) return;

    // Refs to track active interaction that must suppress uPlot's default drag-select.
    let drag: LineDrag | null = null;
    let pan: PanState | null = null;

    // Compute range-line hit-test in data coordinates for correctness regardless of DPR.
    function nearLine(clientX: number, u: uPlot): "start" | "end" | null {
      const r = rangeRef.current;
      if (r.start == null && r.end == null) return null;
      const overRect = u.over.getBoundingClientRect();
      const relX = clientX - overRect.left;
      if (relX < 0 || relX > overRect.width) return null;
      const val = u.posToVal(relX, "x");
      const xMin = (u.scales.x.min as number | undefined) ?? val;
      const xMax = (u.scales.x.max as number | undefined) ?? val;
      const tolerance = (HIT_DISTANCE_PX / overRect.width) * (xMax - xMin);
      const ds = r.start != null ? Math.abs(val - r.start) : Infinity;
      const de = r.end != null ? Math.abs(val - r.end) : Infinity;
      if (Math.min(ds, de) > tolerance) return null;
      return ds <= de ? "start" : "end";
    }

    const optsWithHooks: Options = {
      ...opts,
      cursor: {
        ...opts.cursor,
        bind: {
          mousedown: ((u: uPlot, _t: HTMLElement, handler: (e: MouseEvent) => null) => (e: MouseEvent) => {
            // Suppress uPlot's built-in drag-select when we're starting our own alt-pan / line-drag.
            if (e.altKey) return null;
            if (nearLine(e.clientX, u) !== null) return null;
            handler(e);
            return null;
          }) as unknown as Options["cursor"] extends { bind?: infer B } ? (B extends { mousedown?: infer M } ? M : never) : never,
        },
      },
      hooks: {
        ready: [(u) => {
          // uPlot's `ready` fires inside _commit(); calling setScale directly is a no-op
          // because queuedCommit blocks the follow-up microtask. Queue our own microtask
          // so setScale is applied after _commit returns.
          const xHasExplicit = xlim[0] != null || xlim[1] != null;
          const yHasExplicit = ylim[0] != null || ylim[1] != null;
          const savedX = xHasExplicit ? undefined : scaleRef.current.x;
          const savedY = yHasExplicit ? undefined : scaleRef.current.y;
          if (xHasExplicit) scaleRef.current.x = undefined;
          if (yHasExplicit) scaleRef.current.y = undefined;
          queueMicrotask(() => {
            programmaticSetScaleRef.current++;
            if (savedX) {
              u.setScale("x", { min: savedX[0], max: savedX[1] });
            } else {
              u.setScale("x", { min: xRange[0], max: xRange[1] });
            }
            if (savedY) {
              u.setScale("y", { min: savedY[0], max: savedY[1] });
            } else if (!logScale) {
              u.setScale("y", { min: yRange[0], max: yRange[1] });
            }
            queueMicrotask(() => { programmaticSetScaleRef.current--; });
          });
        }],
        draw: [(u) => drawRangeOverlay(u, rangeRef.current)],
        setCursor: [(u) => {
          const idx = u.cursor.idx ?? null;
          setCursorIdx((prev) => prev === idx ? prev : idx);
        }],
        setScale: [(u, key) => {
          // Scale changes not caused by our own code count as user interaction (drag-zoom
          // via uPlot's built-in select). Alt-pan sets hasUserZoomedRef on mousedown separately.
          if (programmaticSetScaleRef.current === 0) hasUserZoomedRef.current = true;
          if (key === "x" && u.scales.x.min != null && u.scales.x.max != null) {
            scaleRef.current.x = [u.scales.x.min as number, u.scales.x.max as number];
          } else if (key === "y" && u.scales.y.min != null && u.scales.y.max != null) {
            scaleRef.current.y = [u.scales.y.min as number, u.scales.y.max as number];
          }
        }],
      },
    };
    // uPlot emits initial setScale hooks during construction. Treat those as setup,
    // not a user zoom, so untouched live charts continue to follow appended points.
    programmaticSetScaleRef.current++;
    const p = new uPlot(optsWithHooks, data, el);
    programmaticSetScaleRef.current--;
    plotRef.current = p;
    justRebuiltRef.current = true;

    const over = p.over;

    const onMouseMove = (e: MouseEvent) => {
      if (drag) {
        const overRect = over.getBoundingClientRect();
        const relX = e.clientX - overRect.left;
        const val = p.posToVal(relX, "x");
        const xMin = (p.scales.x.min as number | undefined) ?? val;
        const xMax = (p.scales.x.max as number | undefined) ?? val;
        const clamped = Math.max(xMin, Math.min(xMax, val));
        if (drag.which === "start") rangeRef.current = { ...rangeRef.current, start: clamped };
        else rangeRef.current = { ...rangeRef.current, end: clamped };
        p.redraw(false);
        return;
      }
      if (pan) {
        const dxPx = e.clientX - pan.startClientX;
        const dyPx = e.clientY - pan.startClientY;
        const overRect = over.getBoundingClientRect();
        const dataDx = (dxPx / overRect.width) * (pan.x1 - pan.x0);
        const dataDy = (dyPx / overRect.height) * (pan.y1 - pan.y0);
        p.setScale("x", { min: pan.x0 - dataDx, max: pan.x1 - dataDx });
        // Screen y is inverted from data y — dragging up should reveal higher values, so add dataDy.
        p.setScale("y", { min: pan.y0 + dataDy, max: pan.y1 + dataDy });
        return;
      }
      const kind = nearLine(e.clientX, p);
      over.style.cursor = kind ? "ew-resize" : "default";
    };

    const onMouseDown = (e: MouseEvent) => {
      if (e.button !== 0) return;
      mouseDownRef.current = true;
      if (e.altKey) {
        const x0 = p.scales.x.min as number | undefined;
        const x1 = p.scales.x.max as number | undefined;
        const y0 = p.scales.y.min as number | undefined;
        const y1 = p.scales.y.max as number | undefined;
        if (x0 == null || x1 == null || y0 == null || y1 == null) return;
        pan = { startClientX: e.clientX, startClientY: e.clientY, x0, x1, y0, y1 };
        hasUserZoomedRef.current = true;
        over.style.cursor = "grabbing";
        e.preventDefault();
        return;
      }
      const kind = nearLine(e.clientX, p);
      if (kind) {
        drag = { which: kind };
        draggingLineRef.current = true;
        over.style.cursor = "ew-resize";
        e.preventDefault();
      }
    };

    const onMouseUp = () => {
      mouseDownRef.current = false;
      if (drag) {
        setRange(tag, rangeRef.current.start, rangeRef.current.end);
        drag = null;
        draggingLineRef.current = false;
        over.style.cursor = "default";
      }
      if (pan) {
        pan = null;
        over.style.cursor = "default";
      }
    };

    const onContextMenu = (e: MouseEvent) => {
      e.preventDefault();
      setMenu({ x: e.clientX, y: e.clientY });
    };

    // Use capture: true so our handlers see mousedown before uPlot's own cursor bindings.
    over.addEventListener("mousemove", onMouseMove);
    over.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("mouseup", onMouseUp);
    over.addEventListener("contextmenu", onContextMenu);

    // Expose reset for the header's ⟲ button.
    const w = window as unknown as { __curve_reset__?: Record<string, () => void> };
    if (!w.__curve_reset__) w.__curve_reset__ = {};
    w.__curve_reset__[tag] = () => {
      scaleRef.current = {};
      hasUserZoomedRef.current = false;
      const dr = dataRangeRef.current;
      programmaticSetScaleRef.current++;
      p.setScale("x", { min: dr.x[0], max: dr.x[1] });
      p.setScale("y", { min: dr.y[0], max: dr.y[1] });
      queueMicrotask(() => { programmaticSetScaleRef.current--; });
    };

    return () => {
      over.removeEventListener("mousemove", onMouseMove);
      over.removeEventListener("mousedown", onMouseDown, true);
      window.removeEventListener("mouseup", onMouseUp);
      over.removeEventListener("contextmenu", onContextMenu);
      p.destroy();
      plotRef.current = null;
      const w2 = window as unknown as { __curve_reset__?: Record<string, () => void> };
      if (w2.__curve_reset__) delete w2.__curve_reset__[tag];
    };
  }, [optsStructKey]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Data-only refresh: update the existing plot in place so zoom/pan/mid-drag survive.
  // Auto-refits scales only when in default view AND the user is not touching the plot.
  useEffect(() => {
    const p = plotRef.current;
    if (!p) return;
    if (justRebuiltRef.current) { justRebuiltRef.current = false; return; }
    const xHasExplicit = xlim[0] != null || xlim[1] != null;
    const yHasExplicit = ylim[0] != null || ylim[1] != null;
    const canRefit = !xHasExplicit && !yHasExplicit && !hasUserZoomedRef.current && !mouseDownRef.current;
    programmaticSetScaleRef.current++;
    p.setData(data, canRefit);
    // uPlot skips its commit when setData receives false; rebuild the paths without
    // changing scales so a preserved viewport still renders newly appended points.
    if (!canRefit) p.redraw();
    queueMicrotask(() => { programmaticSetScaleRef.current--; });
  }, [data]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { plotRef.current?.redraw(false); }, [rangeStartVal, rangeEndVal]);

  const saveAsImage = useCallback(() => {
    const p = plotRef.current;
    if (!p) return;
    const link = document.createElement("a");
    link.download = `${tag.replace(/\//g, "_")}.png`;
    link.href = p.ctx.canvas.toDataURL("image/png");
    link.click();
    setMenu(null);
  }, [tag]);

  useEffect(() => {
    if (!menu) return;
    const onDoc = () => setMenu(null);
    window.addEventListener("mousedown", onDoc);
    return () => window.removeEventListener("mousedown", onDoc);
  }, [menu]);

  const cursorX = cursorIdx != null && data[0] ? ((data[0] as number[])[cursorIdx] as number | undefined) ?? null : null;

  return (
    <>
      <div className="curve-plot-wrap" style={{ height: size.h }}>
        <div ref={containerRef} className="curve-plot" />
        {series.length === 0 && <div className="curve-empty">No data for the selected runs.</div>}
      </div>
      {series.length > 0 && (
        <LegendTable
          series={series}
          cursorX={cursorX}
          rangeStart={rangeStartVal}
          rangeEnd={rangeEndVal}
        />
      )}
      {menu && (
        <div
          className="curve-context-menu"
          style={{ left: menu.x, top: menu.y }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <button type="button" onClick={saveAsImage}>💾 Save as Image</button>
        </div>
      )}
    </>
  );
}

function drawRangeOverlay(u: uPlot, r: { start: number | null; end: number | null }) {
  const bbox = u.bbox;
  const ctx = u.ctx;
  const strong = cssVar("--text-strong", "#0c1913");
  const border = cssVar("--border", "#2a3a4a");
  const panel = cssVar("--bg-1", "#141d29");
  const dim = cssVar("--text-dim", "#7f8ea0");
  const dpr = window.devicePixelRatio || 1;

  const drawLine = (val: number, label: string) => {
    const px = Math.round(u.valToPos(val, "x", true));
    if (px < bbox.left - 1 || px > bbox.left + bbox.width + 1) return;
    ctx.save();
    ctx.strokeStyle = strong;
    ctx.globalAlpha = 0.75;
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px + 0.5, bbox.top);
    ctx.lineTo(px + 0.5, bbox.top + bbox.height);
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.font = `${Math.round(11 * dpr)}px ui-monospace, "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const text = `${label} ${formatStep(val)}`;
    const metrics = ctx.measureText(text);
    const padX = 6 * dpr;
    const labelH = 14 * dpr;
    const labelW = metrics.width + padX * 2;
    let cx = px;
    const minCX = bbox.left + labelW / 2;
    const maxCX = bbox.left + bbox.width - labelW / 2;
    if (cx < minCX) cx = minCX;
    if (cx > maxCX) cx = maxCX;
    const cy = bbox.top / 2 + 1;
    const bx = cx - labelW / 2;
    const by = cy - labelH / 2;
    ctx.fillStyle = panel;
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    roundRect(ctx, bx, by, labelW, labelH, 4 * dpr);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = dim;
    ctx.fillText(text, cx, cy);
    ctx.restore();
  };

  if (r.start != null) drawLine(r.start, "S:");
  if (r.end != null) drawLine(r.end, "E:");
}

function LegendTable({
  series,
  cursorX,
  rangeStart,
  rangeEnd,
}: {
  series: RunSeries[];
  cursorX: number | null;
  rangeStart: number | null;
  rangeEnd: number | null;
}) {
  return (
    <div className="curve-legend">
      <div className="curve-legend-scroll">
        <table>
          <thead>
            <tr>
              <th className="col-swatch" />
              <th>Dataset</th>
              <th>Model</th>
              <th>Run</th>
              <th className="col-value" title={cursorX != null ? `Cursor at x=${formatStep(cursorX)}` : "Hover over the plot to inspect points"}>Cursor</th>
              <th className="col-value" title={rangeStart != null ? `Start at x=${formatStep(rangeStart)}` : ""}>Start</th>
              <th className="col-value" title={rangeEnd != null ? `End at x=${formatStep(rangeEnd)}` : ""}>End</th>
              <th className="col-value">Diff</th>
              <th className="col-value">Ratio</th>
            </tr>
          </thead>
          <tbody>
            {series.map((s) => {
              const [dataset, model, run] = s.runKey.split("\u0000");
              const cv = valueAtNearest(s.points, cursorX);
              const sv = valueAtNearest(s.points, rangeStart);
              const ev = valueAtNearest(s.points, rangeEnd);
              const diff = sv != null && ev != null ? ev - sv : null;
              const ratio = sv != null && ev != null && sv !== 0 ? diff! / sv : null;
              return (
                <tr key={s.runKey}>
                  <td className="col-swatch">
                    <span className="curve-legend-swatch" style={{ background: s.color }} />
                  </td>
                  <td title={dataset}>{dataset}</td>
                  <td title={model}>{model}</td>
                  <td title={run}>{run}</td>
                  <td className="col-value">{cv == null ? "—" : formatY(cv)}</td>
                  <td className="col-value">{sv == null ? "—" : formatY(sv)}</td>
                  <td className="col-value">{ev == null ? "—" : formatY(ev)}</td>
                  <td className="col-value">{diff == null ? "—" : formatY(diff)}</td>
                  <td className="col-value">{ratio == null ? "—" : formatRatio(ratio)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function valueAtNearest(points: ScalarPoint[], x: number | null): number | null {
  if (x === null || points.length === 0) return null;
  let bestIdx = 0;
  let bestDist = Infinity;
  for (let i = 0; i < points.length; i++) {
    const d = Math.abs(points[i].step - x);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  return points[bestIdx].value;
}

function labelForKey(runKey: string): string {
  const [, model, run] = runKey.split("\u0000");
  return `${model}/${run}`;
}

function formatStep(n: number): string {
  if (!isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return String(Math.round(n));
}

function formatXAxisStep(n: number): string {
  if (!isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs < 1000) return String(Math.round(n));
  return `${(n / 1000).toFixed(abs < 10_000 ? 1 : 0).replace(/\.0$/, "")}k`;
}

function formatY(n: number): string {
  if (!isFinite(n)) return "—";
  return n.toExponential(3);
}

function formatRatio(n: number): string {
  if (!isFinite(n)) return "—";
  return `${(n * 100).toFixed(4)}%`;
}

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}
