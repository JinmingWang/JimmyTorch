import type { Progress } from "../api/types";
import { formatInt, formatPercent } from "../utils/format";

interface Props {
  progress: Progress;
}

export function ProgressBars({ progress }: Props) {
  const overallPct = clampPct(progress.percent);
  const stepInEpoch = progress.step;
  const stepsPerEpoch = Math.max(1, progress.steps_per_epoch);
  const epochPct = clampPct((stepInEpoch / stepsPerEpoch) * 100);

  return (
    <div className="progress-bars">
      <ProgressRow
        leftLabel="Overall Progress"
        rightLabel={`Step ${formatInt(progress.overall)} / ${formatInt(progress.total)} (${formatPercent(progress.percent)})`}
        percent={overallPct}
        kind="overall"
      />
      <ProgressRow
        leftLabel={`Epoch ${formatInt(progress.epoch)}/${formatInt(progress.epochs)} Progress`}
        rightLabel={`Step ${formatInt(stepInEpoch)} / ${formatInt(stepsPerEpoch)} (${formatPercent((stepInEpoch / stepsPerEpoch) * 100)})`}
        percent={epochPct}
        kind="epoch"
      />
    </div>
  );
}

function ProgressRow({
  leftLabel,
  rightLabel,
  percent,
  kind,
}: {
  leftLabel: string;
  rightLabel: string;
  percent: number;
  kind: "overall" | "epoch";
}) {
  return (
    <div className={`progress-row progress-${kind}`}>
      <div className="progress-labels">
        <span className="progress-left">{leftLabel}</span>
        <span className="progress-right">{rightLabel}</span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function clampPct(v: number): number {
  if (!isFinite(v)) return 0;
  return Math.max(0, Math.min(100, v));
}
