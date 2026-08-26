import { useEffect, useState } from "react";
import type { RunNode } from "../../api/rest";
import { fetchFigureIndex, figureBlobUrl } from "../../api/rest";

interface Props {
  tag: string;
  node: RunNode;
  onClose: () => void;
}

export function FigurePopup({ tag, node, onClose }: Props) {
  const [step, setStep] = useState<number | null>(null);
  const [entries, setEntries] = useState<{ step: number }[]>([]);
  useEffect(() => {
    void (async () => {
      const idx = await fetchFigureIndex(node.dataset, node.model, node.run_name, tag);
      setEntries(idx.entries);
      if (idx.entries.length > 0) setStep(idx.entries[idx.entries.length - 1].step);
    })();
  }, [tag, node]);
  return (
    <div className="fullscreen-overlay" onClick={onClose}>
      <div className="fullscreen-inner figure-popup" onClick={(e) => e.stopPropagation()}>
        <div className="figure-popup-header">
          <span>{tag} — {node.dataset} / {node.model} / {node.run_name}</span>
          <button type="button" className="curve-btn" onClick={onClose}>Close</button>
        </div>
        <div className="figure-popup-body">
          {step !== null ? (
            <img
              src={figureBlobUrl(node.dataset, node.model, node.run_name, tag, step)}
              alt={`${tag} step ${step}`}
            />
          ) : (
            <div className="curve-empty">No figures.</div>
          )}
        </div>
        {entries.length > 1 && (
          <div className="figure-popup-footer">
            <input
              type="range"
              min={0}
              max={entries.length - 1}
              value={Math.max(0, entries.findIndex((e) => e.step === step))}
              onChange={(e) => setStep(entries[Number(e.target.value)].step)}
            />
            <span>step {step}</span>
          </div>
        )}
      </div>
    </div>
  );
}
