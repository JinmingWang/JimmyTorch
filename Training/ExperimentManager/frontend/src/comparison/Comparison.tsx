import { useEffect, useMemo, useState } from "react";
import { fetchComparison, type ComparisonRun, type Reduction } from "../api/rest";
import { effectiveColor, findRun, iterRuns, keyOf, useUIStore } from "../state/uiStore";
import { ScatterPlot, type PlotPoint } from "./ScatterPlot";
import "./comparison.css";

const measurementNames: Record<string, string> = {
  parameters: "Parameters",
  iterations_per_second: "Training throughput (it/s)",
  checkpoint_mib: "Model file size (MiB)",
  forward_macs: "Forward MACs",
  forward_flops: "Forward FLOPs",
};
const families = { measurements: "Measurement", metrics: "Metric", hparams: "Hyperparameter" };
type Family = keyof typeof families;
type Axis = { family: Family; field: string; reduction: Reduction };
type Fields = Record<Family, string[]>;

function axisTitle(axis: Axis) {
  if (axis.family === "measurements") return measurementNames[axis.field];
  if (axis.family === "metrics" && axis.field) return `${axis.field} · ${axis.reduction}`;
  return axis.field;
}

function axisValue(run: ComparisonRun, axis: Axis) {
  if (axis.family === "metrics") return run.metrics[axis.field]?.[axis.reduction];
  return run[axis.family][axis.field];
}

function AxisPicker({ name, axis, fields, onChange }: {
  name: string; axis: Axis; fields: Fields; onChange: (axis: Axis) => void;
}) {
  return <fieldset className="axis-picker">
    <legend>{name} axis</legend>
    <select aria-label={`${name} source`} value={axis.family} onChange={(event) => {
      const family = event.target.value as Family;
      onChange({ family, field: fields[family][0] ?? "", reduction: "last" });
    }}>
      {Object.entries(families).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
    </select>
    <select aria-label={`${name} field`} value={axis.field} disabled={!fields[axis.family].length}
      onChange={(event) => onChange({ ...axis, field: event.target.value })}>
      {!fields[axis.family].length && <option value="">No recorded fields</option>}
      {fields[axis.family].map((key) => <option key={key} value={key}>
        {axis.family === "measurements" ? measurementNames[key] : key}
      </option>)}
    </select>
    {axis.family === "metrics" && <select aria-label={`${name} statistic`} value={axis.reduction}
      onChange={(event) => onChange({ ...axis, reduction: event.target.value as Reduction })}>
      <option value="last">Last</option><option value="lowest">Lowest</option><option value="highest">Highest</option>
    </select>}
  </fieldset>;
}

function defaultAxis(family: Family, field: string): Axis {
  return { family, field, reduction: "last" };
}

export function Comparison({ highlighted, onPick, onClear }: {
  highlighted: string | null; onPick: (key: string) => void; onClear: () => void;
}) {
  const tree = useUIStore((s) => s.tree);
  const checked = useUIStore((s) => s.checkedKeys);
  const [checkedOnly, setCheckedOnly] = useState(false);
  const [rows, setRows] = useState<ComparisonRun[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [chosenX, setX] = useState<Axis | null>(null);
  const [chosenY, setY] = useState<Axis | null>(null);
  // Reload on run creation, completion, or profile arrival. Frequent status/color updates need no rescan.
  const membership = useMemo(() => tree && JSON.stringify(iterRuns(tree).map((r) => [keyOf(r), r.closed_at, r.has_profile])), [tree]);
  useEffect(() => {
    if (membership === null) return;
    const controller = new AbortController();
    setBusy(true);
    setError("");
    fetchComparison(controller.signal).then((data) => {
      if (!controller.signal.aborted) setRows(data);
    }).catch((reason) => {
      if (!controller.signal.aborted) setError(String(reason));
    }).finally(() => {
      if (!controller.signal.aborted) setBusy(false);
    });
    return () => controller.abort();
  }, [membership, refresh]);

  const fields = useMemo(() => {
    const keys = (family: Family) => Array.from(new Set(rows.flatMap((r) => Object.keys(r[family])))).sort();
    return { measurements: Object.keys(measurementNames), metrics: keys("metrics"), hparams: keys("hparams") };
  }, [rows]);
  const available = fields.measurements.map((field) => ({
    field, count: rows.filter((r) => Number.isFinite(r.measurements[field])).length,
  })).filter(({ count }) => count > 0).sort((a, b) => b.count - a.count).map(({ field }) => field);
  const metric = fields.metrics.includes("Eval/Main") ? "Eval/Main" : fields.metrics[0];
  const numericSetting = defaultAxis("hparams", fields.hparams[0] ?? "");
  const x = chosenX ?? (available.length ? defaultAxis("measurements", available[0]) :
    fields.hparams.length ? numericSetting : defaultAxis("metrics", fields.metrics[0] ?? ""));
  const y = chosenY ?? (metric ? defaultAxis("metrics", metric) :
    fields.hparams.length ? numericSetting : defaultAxis("measurements", available[1] ?? available[0] ?? "checkpoint_mib"));
  const runs = checkedOnly ? rows.filter((row) => checked.has(keyOf(row))) : rows;
  const points: PlotPoint[] = [];
  for (const row of runs) {
    const key = keyOf(row);
    const node = findRun(tree, key);
    const vx = axisValue(row, x), vy = axisValue(row, y);
    if (node && Number.isFinite(vx) && Number.isFinite(vy)) {
      points.push({ key, name: row.run_name, path: node.path, color: effectiveColor(node), x: vx, y: vy });
    }
  }
  const missing = runs.length - points.length;
  const empty = error || (busy ? "Loading comparison…" : !rows.length ? "No runs to compare." :
    !runs.length ? "Check runs in the tree to compare them." : "No runs have values for both axes.");

  return <main className="comparison-panel">
    <header className="comparison-toolbar">
      <div><h1>Run comparison</h1>
        <div className="comparison-count" role="status">{busy ? "Reading saved runs…" :
          `${points.length} of ${runs.length} runs plotted${missing ? ` · ${missing} missing axis values` : ""}`}</div>
      </div>
      <div className="comparison-actions">
        <select aria-label="Runs to compare" value={checkedOnly ? "checked" : "all"}
          onChange={(e) => setCheckedOnly(e.target.value === "checked")}>
          <option value="all">All runs</option><option value="checked">Checked runs</option>
        </select>
        <button className="curve-btn" onClick={() => setRefresh((n) => n + 1)} disabled={busy}>↻ Refresh</button>
      </div>
    </header>
    <div className="comparison-axes">
      <AxisPicker name="X" fields={fields} axis={x} onChange={setX} />
      <AxisPicker name="Y" fields={fields} axis={y} onChange={setY} />
    </div>
    {error || !points.length ? <div className="comparison-empty">{empty}</div> :
      <ScatterPlot points={points} xTitle={axisTitle(x)} yTitle={axisTitle(y)} selected={highlighted}
        onPick={onPick} onClear={onClear} />}
  </main>;
}
