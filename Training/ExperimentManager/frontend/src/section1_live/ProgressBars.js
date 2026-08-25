import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { formatInt, formatPercent } from "../utils/format";
export function ProgressBars({ progress }) {
    const overallPct = clampPct(progress.percent);
    const stepInEpoch = progress.step;
    const stepsPerEpoch = Math.max(1, progress.steps_per_epoch);
    const epochPct = clampPct((stepInEpoch / stepsPerEpoch) * 100);
    return (_jsxs("div", { className: "progress-bars", children: [_jsx(ProgressRow, { leftLabel: "Overall Progress", rightLabel: `Step ${formatInt(progress.overall)} / ${formatInt(progress.total)} (${formatPercent(progress.percent)})`, percent: overallPct, kind: "overall" }), _jsx(ProgressRow, { leftLabel: `Epoch ${formatInt(progress.epoch)}/${formatInt(progress.epochs)} Progress`, rightLabel: `Step ${formatInt(stepInEpoch)} / ${formatInt(stepsPerEpoch)} (${formatPercent((stepInEpoch / stepsPerEpoch) * 100)})`, percent: epochPct, kind: "epoch" })] }));
}
function ProgressRow({ leftLabel, rightLabel, percent, kind, }) {
    return (_jsxs("div", { className: `progress-row progress-${kind}`, children: [_jsxs("div", { className: "progress-labels", children: [_jsx("span", { className: "progress-left", children: leftLabel }), _jsx("span", { className: "progress-right", children: rightLabel })] }), _jsx("div", { className: "progress-track", children: _jsx("div", { className: "progress-fill", style: { width: `${percent}%` } }) })] }));
}
function clampPct(v) {
    if (!isFinite(v))
        return 0;
    return Math.max(0, Math.min(100, v));
}
