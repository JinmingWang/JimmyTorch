import { useState } from "react";
import type { LearningRateState } from "../api/types";
import { postLearningRate } from "../api/ws";
import { formatFloat } from "../utils/format";

interface Props {
  learningRate: LearningRateState;
}

export function LearningRateControl({ learningRate }: Props) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const lr = Number(text);
    if (!isFinite(lr) || lr <= 0) {
      setError("Enter a positive number.");
      return;
    }
    setBusy(true);
    try {
      await postLearningRate(lr);
      setText("");
    } catch {
      setError("Request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="lr-panel">
      <div className="lr-panel-header">Learning Rate</div>
      <div className="lr-values">
        <div>
          <div className="lr-values-label">Applied</div>
          <div className="lr-values-value">{formatLR(learningRate.applied)}</div>
        </div>
        <div>
          <div className="lr-values-label">Pending</div>
          <div className="lr-values-value">{formatLR(learningRate.pending)}</div>
        </div>
      </div>
      <form className="lr-form" onSubmit={submit}>
        <input
          type="text"
          inputMode="decimal"
          placeholder="e.g. 5e-5"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
        />
        <button type="submit" disabled={busy || text === ""}>
          Apply
        </button>
      </form>
      {error && <div className="lr-error">{error}</div>}
    </div>
  );
}

function formatLR(v: number | null): string {
  if (v == null) return "—";
  const abs = Math.abs(v);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return v.toExponential(3);
  return formatFloat(v, 6);
}
