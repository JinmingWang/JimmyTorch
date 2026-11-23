export type RunStatus =
  | "idle"
  | "training"
  | "evaluating"
  | "done"
  | "error";

export interface RunIdentity {
  id?: string | null;
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

export interface GpuStats {
  id: string;
  index: number;
  name: string | null;
  utilization: number | null;
  memory_used: number | null;
  memory_total: number | null;
  training_ranks: number[] | null;
}

export interface SystemStats {
  gpus?: GpuStats[] | null;
  cpu_util?: number;
  cpu_mem_bytes?: number;
  cpu_mem_total?: number;
  gpu_util?: number;
  gpu_mem_used?: number;
  gpu_mem_total?: number;
}

export interface NodeStats {
  node_id: string;
  hostname: string;
  ranks: number[];
  system: SystemStats;
  received_at: number | null;
  connected: boolean;
}

export interface LiveSnapshot {
  nodes?: NodeStats[];
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
