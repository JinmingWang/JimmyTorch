import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { formatBytes, formatDuration, formatPercent, formatRate } from "../utils/format";
export function MetricCards({ progress, system }) {
    const cards = [
        { title: "Elapsed", value: formatDuration(progress.elapsed), accent: "cyan" },
        { title: "Remaining", value: formatDuration(progress.remaining), accent: "yellow" },
        { title: "Throughput", value: formatRate(progress.rate), accent: "lime" },
        {
            title: "GPU Memory",
            value: system.gpu_mem_used != null && system.gpu_mem_total != null
                ? `${formatBytes(system.gpu_mem_used)} / ${formatBytes(system.gpu_mem_total)}`
                : "—",
            accent: "coral",
        },
        {
            title: "CPU Memory",
            value: system.cpu_mem_bytes != null && system.cpu_mem_total != null
                ? `${formatBytes(system.cpu_mem_bytes)} / ${formatBytes(system.cpu_mem_total)}`
                : system.cpu_mem_bytes != null
                    ? formatBytes(system.cpu_mem_bytes)
                    : "—",
            accent: "cyan",
        },
        {
            title: "GPU Utilization",
            value: system.gpu_util != null ? formatPercent(system.gpu_util, 0) : "—",
            accent: "lime",
        },
    ];
    return (_jsx("div", { className: "metric-grid", children: cards.map((c) => (_jsxs("div", { className: `metric-card accent-${c.accent}`, children: [_jsx("div", { className: "metric-title", children: c.title }), _jsx("div", { className: "metric-value", children: c.value })] }, c.title))) }));
}
