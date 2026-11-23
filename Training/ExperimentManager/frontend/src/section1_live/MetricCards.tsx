import type { GpuStats, NodeStats, Progress, SystemStats } from "../api/types";
import { formatBytes, formatDuration, formatPercent, formatRate } from "../utils/format";

interface Props {
  progress: Progress;
  system: SystemStats;
  nodes?: NodeStats[];
}

export function MetricCards({ progress, system, nodes }: Props) {
  const cards = [
    { title: "Elapsed", value: formatDuration(progress.elapsed), accent: "cyan" },
    { title: "Remaining", value: formatDuration(progress.remaining), accent: "yellow" },
    { title: "Throughput", value: formatRate(progress.rate), accent: "lime" },
    {
      title: nodes?.length ? "Primary CPU Memory" : "CPU Memory",
      value: system.cpu_mem_bytes != null && system.cpu_mem_total != null
        ? `${formatBytes(system.cpu_mem_bytes)} / ${formatBytes(system.cpu_mem_total)}`
        : system.cpu_mem_bytes != null ? formatBytes(system.cpu_mem_bytes) : "—",
      accent: "cyan",
    },
  ];

  return (
    <div className="resource-metrics">
      <div className="metric-grid">
        {cards.map((card) => (
          <div key={card.title} className={`metric-card accent-${card.accent}`}>
            <div className="metric-title">{card.title}</div>
            <div className="metric-value">{card.value}</div>
          </div>
        ))}
      </div>
      {nodes?.length ? <div className="node-resources">
        {nodes.map((node) => <section className="node-card" key={node.node_id}>
          <div className="node-heading"><h3>Machine {node.node_id} · {node.hostname}</h3>
            <span className={node.connected ? "" : "node-stale"}>{node.connected ? "Reporting" : node.received_at == null ? "Telemetry unavailable" : "Telemetry inactive / stale"}</span></div>
          <p>Training ranks {node.ranks.join(", ")} · CPU memory {node.system.cpu_mem_bytes != null && node.system.cpu_mem_total != null
            ? `${formatBytes(node.system.cpu_mem_bytes)} / ${formatBytes(node.system.cpu_mem_total)}` : "—"}</p>
          <GpuCards system={node.system} />
        </section>)}
      </div> : <GpuCards system={system} />}

    </div>
  );
}

function GpuCards({ system }: { system: SystemStats }) {
  let gpus: GpuStats[] | null = system.gpus ?? null;
  if (system.gpus === undefined && [system.gpu_util, system.gpu_mem_used, system.gpu_mem_total].some((value) => value != null)) {
    gpus = [{ id: "legacy", index: 0, name: null, utilization: system.gpu_util ?? null,
      memory_used: system.gpu_mem_used ?? null, memory_total: system.gpu_mem_total ?? null,
      training_ranks: null }];
  }

  return (
      <section className="gpu-section" aria-label="GPU resources">
        <div className="gpu-section-heading"><h3>GPUs {gpus != null ? `(${gpus.length})` : ""}</h3><span>Device-wide usage</span></div>
        {gpus === null ? <p className="gpu-empty">GPU telemetry unavailable</p>
          : gpus.length === 0 ? <p className="gpu-empty">No GPUs detected</p>
          : <div className="gpu-grid">{[...gpus].sort((left, right) => left.index - right.index).map((gpu) => {
            const memory_percent = gpu.memory_used != null && gpu.memory_total != null && gpu.memory_total > 0
              ? gpu.memory_used / gpu.memory_total * 100 : null;
            const assignment = gpu.training_ranks == null ? "Assignment unknown"
              : gpu.training_ranks.length ? `Training · rank${gpu.training_ranks.length > 1 ? "s" : ""} ${gpu.training_ranks.join(", ")}` : "Not assigned to this run";
            return <article className="gpu-card" key={gpu.id}>
              <div className="gpu-card-heading"><strong>GPU {gpu.index}</strong><span className={`gpu-assignment${gpu.training_ranks?.length ? " is-training" : ""}`}>{assignment}</span></div>
              <div className="gpu-name">{gpu.name ?? "Device name unavailable"}</div>
              <div className="gpu-reading"><span>Utilization</span><strong>{gpu.utilization != null ? formatPercent(gpu.utilization, 0) : "—"}</strong></div>
              <progress aria-label={`GPU ${gpu.index} utilization`} max={100} value={Math.min(100, Math.max(0, gpu.utilization ?? 0))} className={gpu.utilization == null ? "is-unavailable" : ""} />
              <div className="gpu-reading"><span>Memory</span><strong>{gpu.memory_used != null && gpu.memory_total != null ? `${formatBytes(gpu.memory_used)} / ${formatBytes(gpu.memory_total)}` : "—"}</strong></div>
              <progress aria-label={`GPU ${gpu.index} memory`} max={100} value={Math.min(100, Math.max(0, memory_percent ?? 0))} className={memory_percent == null ? "is-unavailable" : ""} />
            </article>;
          })}</div>}
      </section>
  );
}
