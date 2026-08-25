export type RunStatus =
  | "idle"
  | "training"
  | "evaluating"
  | "done"
  | "error";

export interface RunIdentity {
  dataset: string | null;
  model: string | null;
  run_name: string | null;
  run_dir: string | null;
}

export interface Progress {
  overall: number;
  total: number;
  percent: number;
  epoch: number;
  epochs: number;
  step: number;
  steps_per_epoch: number;
  elapsed: number;
  rate: number;
  remaining: number | null;
}

export interface LearningRateState {
  applied: number | null;
  pending: number | null;
}

export interface SystemStats {
  cpu_util?: number;
  cpu_mem_bytes?: number;
  cpu_mem_total?: number;
  gpu_util?: number;
  gpu_mem_used?: number;
  gpu_mem_total?: number;
}

export interface LiveSnapshot {
  status: RunStatus;
  connected: boolean;
  run: RunIdentity | null;
  progress: Progress;
  metrics: Record<string, number | null>;
  custom_fields: string[];
  learning_rate: LearningRateState;
  system: SystemStats;
  server_time: number;
}

export interface WsMessage {
  type: "live_snapshot" | "tree_updated";
  payload: any;
}
