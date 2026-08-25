import type { Progress, SystemStats } from "../api/types";
import { formatBytes, formatDuration, formatPercent, formatRate } from "../utils/format";

interface Props {
  progress: Progress;
  system: SystemStats;
}

export function MetricCards({ progress, system }: Props) {
  const cards: CardData[] = [
    { title: "Elapsed", value: formatDuration(progress.elapsed), accent: "cyan" },
    { title: "Remaining", value: formatDuration(progress.remaining), accent: "yellow" },
    { title: "Throughput", value: formatRate(progress.rate), accent: "lime" },
    {
      title: "GPU Memory",
      value:
        system.gpu_mem_used != null && system.gpu_mem_total != null
          ? `${formatBytes(system.gpu_mem_used)} / ${formatBytes(system.gpu_mem_total)}`
          : "—",
      accent: "coral",
    },
    {
      title: "CPU Memory",
      value:
        system.cpu_mem_bytes != null && system.cpu_mem_total != null
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

  return (
    <div className="metric-grid">
      {cards.map((c) => (
        <div key={c.title} className={`metric-card accent-${c.accent}`}>
          <div className="metric-title">{c.title}</div>
          <div className="metric-value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}

interface CardData {
  title: string;
  value: string;
  accent: "cyan" | "lime" | "coral" | "yellow";
}
