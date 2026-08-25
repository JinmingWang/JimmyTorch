import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { fetchScalars } from "../../api/rest";
import { useUIStore } from "../../state/uiStore";
import { effectiveColor, findRun } from "../../state/uiStore";
import { useLiveStore } from "../../state/liveStore";
import { computeRange, smoothSeries } from "./smooth";
const REFRESH_INTERVAL_MS = 4000;
export function CurveCard({ tag, runKeys, fullscreen = false, onFullscreen, onClose, }) {
    const tree = useUIStore((s) => s.tree);
    const smoothing = useUIStore((s) => s.smoothing[tag] ?? 0);
    const setSmoothing = useUIStore((s) => s.setSmoothing);
    const collapsed = useUIStore((s) => s.collapsed[tag] ?? false);
    const toggleCollapsed = useUIStore((s) => s.toggleCollapsed);
    const xlim = useUIStore((s) => s.xlim[tag] ?? [null, null]);
    const ylim = useUIStore((s) => s.ylim[tag] ?? [null, null]);
    const setXlim = useUIStore((s) => s.setXlim);
    const setYlim = useUIStore((s) => s.setYlim);
    const liveRun = useLiveStore((s) => s.snapshot?.run);
    const liveStatus = useLiveStore((s) => s.snapshot?.status);
    const [seriesByKey, setSeriesByKey] = useState({});
    useEffect(() => {
        let cancelled = false;
        async function loadAll() {
            const promises = runKeys.map(async (key) => {
                const node = findRun(tree, key);
                if (!node)
                    return null;
                const resp = await fetchScalars(node.dataset, node.model, node.run_name, tag, 2000);
                return {
                    runKey: key,
                    color: effectiveColor(node),
                    isLive: !!liveRun &&
                        liveRun.dataset === node.dataset &&
                        liveRun.model === node.model &&
                        liveRun.run_name === node.run_name &&
                        liveStatus !== "done" && liveStatus !== "error",
                    points: resp.points,
                };
            });
            const results = await Promise.all(promises);
            if (cancelled)
                return;
            const byKey = {};
            for (const r of results) {
                if (r)
                    byKey[r.runKey] = r;
            }
            setSeriesByKey(byKey);
        }
        void loadAll();
        // Refresh live runs periodically so curves update.
        const iv = window.setInterval(() => {
            const anyLive = runKeys.some((key) => {
                const node = findRun(tree, key);
                return (node && liveRun &&
                    liveRun.dataset === node.dataset &&
                    liveRun.model === node.model &&
                    liveRun.run_name === node.run_name &&
                    liveStatus !== "done" && liveStatus !== "error");
            });
            if (anyLive)
                void loadAll();
        }, REFRESH_INTERVAL_MS);
        return () => { cancelled = true; window.clearInterval(iv); };
    }, [tag, runKeys.join("|"), tree, liveRun, liveStatus]);
    const displayed = useMemo(() => {
        return runKeys
            .map((k) => seriesByKey[k])
            .filter(Boolean)
            .map((s) => ({ ...s, points: smoothSeries(s.points, smoothing) }));
    }, [runKeys, seriesByKey, smoothing]);
    return (_jsxs("div", { className: `curve-card${fullscreen ? " fullscreen" : ""}${collapsed ? " collapsed" : ""}`, children: [_jsxs("div", { className: "curve-header", children: [_jsx("button", { type: "button", className: "curve-toggle", onClick: () => toggleCollapsed(tag), "aria-label": collapsed ? "Expand" : "Collapse", children: collapsed ? "▶" : "▼" }), _jsx("span", { className: "curve-title", children: tag }), !collapsed && (_jsxs(_Fragment, { children: [_jsxs("span", { className: "curve-controls", children: [_jsxs("label", { children: ["smooth", _jsx("input", { type: "range", min: 0, max: 0.99, step: 0.01, value: smoothing, onChange: (e) => setSmoothing(tag, Number(e.target.value)) }), _jsx("span", { className: "curve-controls-value", children: smoothing.toFixed(2) })] }), _jsx(RangeInput, { label: "x", value: xlim, onChange: (a, b) => setXlim(tag, a, b) }), _jsx(RangeInput, { label: "y", value: ylim, onChange: (a, b) => setYlim(tag, a, b) })] }), fullscreen ? (_jsx("button", { type: "button", className: "curve-btn", onClick: onClose, children: "Close" })) : (_jsx("button", { type: "button", className: "curve-btn", onClick: onFullscreen, children: "\u2922" }))] }))] }), !collapsed && (_jsx(UPlotWrap, { series: displayed, xlim: xlim, ylim: ylim, fullscreen: fullscreen }))] }));
}
function RangeInput({ label, value, onChange, }) {
    const [a, b] = value;
    const parse = (s) => {
        const t = s.trim();
        if (t === "")
            return null;
        const n = Number(t);
        return isFinite(n) ? n : null;
    };
    return (_jsxs("span", { className: "range-input", children: [_jsx("span", { className: "range-input-label", children: label }), _jsx("input", { placeholder: "min", defaultValue: a == null ? "" : String(a), onBlur: (e) => onChange(parse(e.target.value), b) }), _jsx("input", { placeholder: "max", defaultValue: b == null ? "" : String(b), onBlur: (e) => onChange(a, parse(e.target.value)) })] }));
}
function UPlotWrap({ series, xlim, ylim, fullscreen, }) {
    const containerRef = useRef(null);
    const plotRef = useRef(null);
    const [size, setSize] = useState({ w: 400, h: fullscreen ? 500 : 220 });
    useLayoutEffect(() => {
        const el = containerRef.current;
        if (!el)
            return;
        const ro = new ResizeObserver((entries) => {
            const r = entries[0]?.contentRect;
            if (!r)
                return;
            setSize({ w: Math.max(80, Math.floor(r.width)), h: fullscreen ? Math.max(400, Math.floor(r.height)) : 220 });
        });
        ro.observe(el);
        return () => ro.disconnect();
    }, [fullscreen]);
    const { data, opts, liveMarkers } = useMemo(() => {
        if (series.length === 0) {
            return { data: [[]], opts: null, liveMarkers: [] };
        }
        // Build common x-axis by taking the union of all steps.
        const stepSet = new Set();
        for (const s of series)
            for (const p of s.points)
                stepSet.add(p.step);
        const steps = Array.from(stepSet).sort((a, b) => a - b);
        const stepIndex = new Map();
        steps.forEach((s, i) => stepIndex.set(s, i));
        const yArrays = series.map(() => new Array(steps.length).fill(null));
        for (let i = 0; i < series.length; i++) {
            for (const p of series[i].points)
                yArrays[i][stepIndex.get(p.step)] = p.value;
        }
        const data = [steps, ...yArrays];
        const yRange = computeRange(series.map((s) => s.points), ylim, "y");
        const xRange = computeRange(series.map((s) => s.points), xlim, "x");
        const opts = {
            width: size.w,
            height: size.h,
            cursor: { drag: { x: true, y: true, uni: 10 } },
            scales: {
                x: { time: false, range: [xRange[0], xRange[1]] },
                y: { range: [yRange[0], yRange[1]] },
            },
            axes: [
                {
                    stroke: cssVar("--text-dim", "#7f8ea0"),
                    grid: { stroke: cssVar("--border", "#22303d") },
                    values: (_u, ticks) => ticks.map((t) => formatStep(t)),
                },
                {
                    stroke: cssVar("--text-dim", "#7f8ea0"),
                    grid: { stroke: cssVar("--border", "#22303d") },
                    size: 60,
                    values: (_u, ticks) => ticks.map((t) => formatY(t)),
                },
            ],
            series: [
                { label: "step" },
                ...series.map((s) => ({
                    label: labelForKey(s.runKey),
                    stroke: s.color,
                    width: s.isLive ? 2 : 1.5,
                    points: { show: false },
                    value: (_u, v) => (v == null ? "—" : formatY(v)),
                })),
            ],
            legend: { show: false },
        };
        // Compute latest point per live series for star marker overlay.
        const liveMarkers = series
            .filter((s) => s.isLive && s.points.length > 0)
            .map((s) => {
            const last = s.points[s.points.length - 1];
            return { x: last.step, y: last.value, color: s.color };
        });
        return { data, opts, liveMarkers };
    }, [series, size.w, size.h, xlim, ylim]);
    useEffect(() => {
        const el = containerRef.current;
        if (!el || !opts)
            return;
        if (plotRef.current) {
            plotRef.current.destroy();
            plotRef.current = null;
        }
        plotRef.current = new uPlot(opts, data, el);
        return () => { plotRef.current?.destroy(); plotRef.current = null; };
    }, [opts, data]);
    return (_jsxs("div", { className: "curve-plot-wrap", style: { height: size.h }, children: [_jsx("div", { ref: containerRef, className: "curve-plot" }), _jsx(LiveMarkerLayer, { plotRef: plotRef, markers: liveMarkers }), series.length === 0 && _jsx("div", { className: "curve-empty", children: "No data for the selected runs." })] }));
}
function labelForKey(runKey) {
    const [, model, run] = runKey.split("\u0000");
    return `${model}/${run}`;
}
function formatStep(n) {
    if (!isFinite(n))
        return "—";
    if (Math.abs(n) >= 1000)
        return n.toLocaleString();
    return String(Math.round(n));
}
function formatY(n) {
    if (!isFinite(n))
        return "—";
    const abs = Math.abs(n);
    if (abs === 0)
        return "0";
    if (abs < 1e-3 || abs >= 1e4)
        return n.toExponential(2);
    return n.toPrecision(4).replace(/\.?0+$/, "");
}
function cssVar(name, fallback) {
    if (typeof window === "undefined")
        return fallback;
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
}
function LiveMarkerLayer({ plotRef, markers, }) {
    const [tick, setTick] = useState(0);
    useEffect(() => {
        if (markers.length === 0)
            return;
        const iv = window.setInterval(() => setTick((t) => t + 1), 400);
        return () => window.clearInterval(iv);
    }, [markers.length]);
    const plot = plotRef.current;
    if (!plot || markers.length === 0)
        return null;
    return (_jsx("div", { className: "live-marker-layer", children: markers.map((m, i) => {
            const px = plot.valToPos(m.x, "x", true);
            const py = plot.valToPos(m.y, "y", true);
            if (!isFinite(px) || !isFinite(py))
                return null;
            const scale = 1 + 0.2 * Math.sin((tick + i) * 0.9);
            return (_jsx("svg", { className: "live-marker", style: {
                    left: `${px}px`,
                    top: `${py}px`,
                    transform: `translate(-50%, -50%) scale(${scale})`,
                    color: m.color,
                }, width: "18", height: "18", viewBox: "-9 -9 18 18", children: _jsx("polygon", { points: "0,-7 2,-2 7,-2 3,1 5,6 0,3 -5,6 -3,1 -7,-2 -2,-2", fill: "currentColor", stroke: "rgba(0,0,0,0.35)", strokeWidth: "0.5" }) }, i));
        }) }));
}
export { RangeInput };
