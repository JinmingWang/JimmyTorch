import type { LiveSnapshot } from "./types";

export interface RunNode {
  dataset: string;
  model: string;
  run_name: string;
  path: string;
  run_dir: string;
  has_db: boolean;
  has_best: boolean;
  has_last: boolean;
  has_arch: boolean;
  status: string;
  starred: boolean;
  color: string | null;
  created_at: number | null;
  updated_at: number | null;
  closed_at: number | null;
}

export interface TreeResponse {
  runs_root: string;
  datasets: Record<string, Record<string, Record<string, RunNode>>>;
}

export interface RunSummary {
  node: RunNode;
  meta: Record<string, string>;
  hparams: Record<string, unknown> | string | null;
  scalar_tags: string[];
  figure_tags: string[];
}

export interface ScalarPoint { step: number; wall_time: number; value: number; }
export interface FigureEntry { step: number; wall_time: number; mime: string; }

const jsonHeaders = { "Content-Type": "application/json" };

function encodePath(dataset: string, model: string, run: string): string {
  return `${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/${encodeURIComponent(run)}`;
}

export async function fetchTree(): Promise<TreeResponse> {
  const r = await fetch("/api/tree");
  return r.json();
}

export async function fetchLiveStatus(): Promise<LiveSnapshot> {
  const r = await fetch("/api/status");
  return r.json();
}

export async function fetchRunSummary(dataset: string, model: string, run: string): Promise<RunSummary> {
  const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/summary`);
  return r.json();
}

export async function fetchScalars(
  dataset: string, model: string, run: string,
  tag: string, maxPoints = 2000,
): Promise<{ tag: string; points: ScalarPoint[] }> {
  const params = new URLSearchParams({ tag, max_points: String(maxPoints) });
  const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/scalars?${params}`);
  return r.json();
}

export async function fetchFigureIndex(
  dataset: string, model: string, run: string, tag: string,
): Promise<{ tag: string; entries: FigureEntry[] }> {
  const params = new URLSearchParams({ tag });
  const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/figures?${params}`);
  return r.json();
}

export function figureBlobUrl(
  dataset: string, model: string, run: string, tag: string, step: number,
): string {
  const params = new URLSearchParams({ tag, step: String(step) });
  return `/api/runs/${encodePath(dataset, model, run)}/figure_blob?${params}`;
}

export async function saveComments(
  dataset: string, model: string, run: string, comments: string,
): Promise<void> {
  await fetch(`/api/runs/${encodePath(dataset, model, run)}/comments`, {
    method: "POST", headers: jsonHeaders, body: JSON.stringify({ comments }),
  });
}

export async function setStarred(
  dataset: string, model: string, run: string, starred: boolean,
): Promise<void> {
  await fetch(`/api/runs/${encodePath(dataset, model, run)}/star`, {
    method: "POST", headers: jsonHeaders, body: JSON.stringify({ starred }),
  });
}

export async function setColor(
  dataset: string, model: string, run: string, color: string | null,
): Promise<void> {
  await fetch(`/api/runs/${encodePath(dataset, model, run)}/color`, {
    method: "POST", headers: jsonHeaders, body: JSON.stringify({ color }),
  });
}

export async function openRunFolder(dataset: string, model: string, run: string): Promise<void> {
  await fetch(`/api/runs/${encodePath(dataset, model, run)}/open_folder`, { method: "POST" });
}

export async function getDeleteToken(
  dataset: string, model: string, run: string,
): Promise<string> {
  const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/delete_token`);
  const j = await r.json();
  return j.confirm_token as string;
}

export async function deleteRun(
  dataset: string, model: string, run: string, token: string,
): Promise<Response> {
  return fetch(`/api/runs/${encodePath(dataset, model, run)}`, {
    method: "DELETE", headers: jsonHeaders, body: JSON.stringify({ confirm_token: token }),
  });
}

export async function fetchGlobalSettings(): Promise<Record<string, unknown>> {
  const r = await fetch("/api/global_settings");
  return r.json();
}

export async function saveGlobalSettings(patch: Record<string, unknown>): Promise<void> {
  await fetch("/api/global_settings", {
    method: "POST", headers: jsonHeaders, body: JSON.stringify(patch),
  });
}
