/** Exponential moving average smoothing (TensorBoard-style). ``factor`` in [0, 1). */
export function smoothSeries(points, factor) {
    if (factor <= 0 || points.length === 0)
        return points;
    const f = Math.min(0.99, factor);
    const out = new Array(points.length);
    let ema = points[0].value;
    let debias = 0;
    for (let i = 0; i < points.length; i++) {
        ema = ema * f + points[i].value * (1 - f);
        debias = debias * f + (1 - f);
        out[i] = { ...points[i], value: ema / Math.max(debias, 1e-9) };
    }
    return out;
}
export function computeRange(seriesList, override, axis) {
    const [lo, hi] = override;
    let mn = Infinity;
    let mx = -Infinity;
    for (const s of seriesList) {
        for (const p of s) {
            const v = axis === "x" ? p.step : p.value;
            if (v < mn)
                mn = v;
            if (v > mx)
                mx = v;
        }
    }
    if (!isFinite(mn) || !isFinite(mx))
        return [0, 1];
    const pad = axis === "y" ? (mx - mn || 1) * 0.05 : 0;
    return [lo ?? mn - pad, hi ?? mx + pad];
}
