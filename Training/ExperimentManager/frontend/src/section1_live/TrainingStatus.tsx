import type { RunStatus } from "../api/types";

interface Props {
  status: RunStatus;
  connected: boolean;
  wsState: "connecting" | "open" | "closed";
}

const LABEL: Record<RunStatus, string> = {
  idle: "Idle",
  training: "Training",
  evaluating: "Evaluating",
  done: "Done",
  error: "Error",
};

export function TrainingStatus({ status, connected, wsState }: Props) {
  const isTraining = status === "training" || status === "evaluating";
  const badgeClass = `status-badge status-${isTraining ? "training" : "idle"} status-${status}`;
  const label = connected || status === "done" || status === "error" ? LABEL[status] : "Not Training";
  const wsLabel = wsState === "open" ? "live" : wsState === "connecting" ? "connecting…" : "offline";
  return (
    <div className="status-row">
      <div className={badgeClass}>
        <span className="status-dot" />
        <span className="status-label">{label}</span>
      </div>
      <div className={`ws-pill ws-${wsState}`}>{wsLabel}</div>
    </div>
  );
}
