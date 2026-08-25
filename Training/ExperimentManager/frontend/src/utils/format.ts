export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds) || seconds < 0) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const pad = (n: number) => n.toString().padStart(2, "0");
  if (h > 0) return `${h}:${pad(m)}:${pad(s)}`;
  return `${pad(m)}:${pad(s)}`;
}

export function formatInt(n: number | null | undefined): string {
  if (n == null || !isFinite(n)) return "—";
  return Math.round(n).toLocaleString();
}

export function formatFloat(n: number | null | undefined, digits = 3): string {
  if (n == null || !isFinite(n)) return "—";
  return n.toFixed(digits);
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`;
}

export function formatPercent(n: number | null | undefined, digits = 2): string {
  if (n == null || !isFinite(n)) return "—";
  return `${n.toFixed(digits)}%`;
}

export function formatRate(itemsPerSec: number | null | undefined): string {
  if (itemsPerSec == null || !isFinite(itemsPerSec) || itemsPerSec <= 0) return "—";
  if (itemsPerSec >= 1) return `${itemsPerSec.toFixed(1)} it/s`;
  return `${(1 / itemsPerSec).toFixed(1)} s/it`;
}
