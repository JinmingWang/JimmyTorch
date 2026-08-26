import { useEffect, useState } from "react";
import type { RunNode } from "../../api/rest";
import { fetchFigureIndex, figureBlobUrl } from "../../api/rest";
import { useUIStore } from "../../state/uiStore";

interface Props {
  tag: string;
  node: RunNode;
  onFullscreen: () => void;
}

export function FigureCard({ tag, node, onFullscreen }: Props) {
  const [step, setStep] = useState<number | null>(null);
  const refreshTick = useUIStore((s) => s.refreshTick);
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const idx = await fetchFigureIndex(node.dataset, node.model, node.run_name, tag);
      if (cancelled) return;
      if (idx.entries.length > 0) setStep(idx.entries[idx.entries.length - 1].step);
    })();
    return () => { cancelled = true; };
  }, [tag, node, refreshTick]);
  return (
    <div className="figure-card" onClick={onFullscreen}>
      <div className="figure-card-header">
        <span className="figure-card-title">{tag}</span>
        <span className="figure-card-sub">{node.model}/{node.run_name}</span>
      </div>
      {step !== null ? (
        <img
          src={figureBlobUrl(node.dataset, node.model, node.run_name, tag, step)}
          alt={`${tag} step ${step}`}
        />
      ) : (
        <div className="figure-card-empty">No figures.</div>
      )}
    </div>
  );
}
