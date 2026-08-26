import type { ScalarPoint } from "../../api/rest";

/**
 * TensorBoard-style bucketed downsampling: keep one representative every ~500 steps,
 * plus the very latest point (which may not fall on a bucket boundary yet).
 * Preserves x-value density (not index density), so points are evenly spaced in step space.
 */
export function bucketDownsample(points: ScalarPoint[], bucketSize = 500): ScalarPoint[] {
  if (points.length <= 2) return points.slice();
  const sorted = points.slice().sort((a, b) => a.step - b.step);
  const buckets = new Map<number, ScalarPoint>();
  for (const p of sorted) {
    const b = Math.floor(p.step / bucketSize);
    // Last write wins → we keep the latest point in each bucket.
    buckets.set(b, p);
  }
  const out = Array.from(buckets.values()).sort((a, b) => a.step - b.step);
  const last = sorted[sorted.length - 1];
  if (out.length === 0 || out[out.length - 1].step !== last.step) {
    out.push(last);
  }
  return out;
}
