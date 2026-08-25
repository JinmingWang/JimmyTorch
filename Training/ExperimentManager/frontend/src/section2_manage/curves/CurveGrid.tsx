import { useEffect, useMemo, useState } from "react";
import { fetchRunSummary } from "../../api/rest";
import type { RunNode } from "../../api/rest";
import { findRun, useUIStore } from "../../state/uiStore";
import { CurveCard } from "./CurveCard";
import { FigureCard } from "./FigureCard";

/** Union of scalar+figure tags across all checked runs, cached until checked set changes. */
export function CurveGrid() {
  const tree = useUIStore((s) => s.tree);
  const checkedKeys = useUIStore((s) => s.checkedKeys);

  const [scalarTags, setScalarTags] = useState<string[]>([]);
  const [figureTags, setFigureTags] = useState<string[]>([]);
  const [fullscreenTag, setFullscreenTag] = useState<string | null>(null);
  const [fullscreenFigure, setFullscreenFigure] = useState<{ tag: string; node: RunNode } | null>(null);

  const runKeys = useMemo(() => Array.from(checkedKeys), [checkedKeys]);

  useEffect(() => {
    let cancelled = false;
    async function loadTags() {
      const nodes = runKeys.map((k) => findRun(tree, k)).filter(Boolean) as RunNode[];
      const results = await Promise.all(
        nodes.map((n) => fetchRunSummary(n.dataset, n.model, n.run_name).catch(() => null)),
      );
      if (cancelled) return;
      const sTags = new Set<string>();
      const fTags = new Set<string>();
      for (const s of results) {
        if (!s) continue;
        s.scalar_tags.forEach((t) => sTags.add(t));
        s.figure_tags.forEach((t) => fTags.add(t));
      }
      setScalarTags(Array.from(sTags).sort());
      setFigureTags(Array.from(fTags).sort());
    }
    void loadTags();
    return () => { cancelled = true; };
  }, [runKeys.join("|"), tree]);

  if (runKeys.length === 0) {
    return (
      <div className="curve-grid-empty">
        Check runs in the tree to display their curves and figures here.
      </div>
    );
  }

  return (
    <div className="curve-grid-wrap">
      {scalarTags.length > 0 && (
        <section className="curve-section">
          <div className="curve-section-header">Scalars</div>
          <div className="curve-grid">
            {scalarTags.map((tag) => (
              <CurveCard
                key={tag}
                tag={tag}
                runKeys={runKeys}
                onFullscreen={() => setFullscreenTag(tag)}
              />
            ))}
          </div>
        </section>
      )}
      {figureTags.length > 0 && (
        <section className="curve-section">
          <div className="curve-section-header">Figures</div>
          <div className="figure-grid">
            {figureTags.flatMap((tag) =>
              runKeys
                .map((k) => findRun(tree, k))
                .filter((n): n is RunNode => !!n)
                .map((node) => (
                  <FigureCard
                    key={`${tag}::${node.dataset}/${node.model}/${node.run_name}`}
                    tag={tag}
                    node={node}
                    onFullscreen={() => setFullscreenFigure({ tag, node })}
                  />
                )),
            )}
          </div>
        </section>
      )}

      {fullscreenTag && (
        <div className="fullscreen-overlay" onClick={() => setFullscreenTag(null)}>
          <div className="fullscreen-inner" onClick={(e) => e.stopPropagation()}>
            <CurveCard
              tag={fullscreenTag}
              runKeys={runKeys}
              fullscreen
              onClose={() => setFullscreenTag(null)}
            />
          </div>
        </div>
      )}

      {fullscreenFigure && (
        <FigurePopup
          tag={fullscreenFigure.tag}
          node={fullscreenFigure.node}
          onClose={() => setFullscreenFigure(null)}
        />
      )}
    </div>
  );
}

import { fetchFigureIndex, figureBlobUrl } from "../../api/rest";

function FigurePopup({
  tag,
  node,
  onClose,
}: {
  tag: string;
  node: RunNode;
  onClose: () => void;
}) {
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
