import { useEffect, useMemo, useState } from "react";
import { fetchRunSummary } from "../../api/rest";
import type { RunNode } from "../../api/rest";
import { findRun, useUIStore } from "../../state/uiStore";
import { CurveCard } from "./CurveCard";
import { FigureCard } from "./FigureCard";
import { FigurePopup } from "./FigurePopup";

/** Groups scalar tags by prefix into named sections. */
function classifyTag(tag: string): "train" | "eval" | "lr" | "other" {
  const t = tag.toLowerCase();
  if (t === "lr" || t.startsWith("lr/") || t.endsWith("/lr")) return "lr";
  if (t.startsWith("train/") || t.startsWith("training/")) return "train";
  if (t.startsWith("eval/") || t.startsWith("val/") || t.startsWith("validation/")) return "eval";
  return "other";
}

const SECTIONS: Array<{ key: "train" | "eval" | "lr" | "other"; label: string; id: string }> = [
  { key: "train", label: "Train losses", id: "curves-train" },
  { key: "eval", label: "Eval losses", id: "curves-eval" },
  { key: "lr", label: "Learning rate", id: "curves-lr" },
  { key: "other", label: "Other scalars", id: "curves-other" },
];

export function CurveGrid() {
  const tree = useUIStore((s) => s.tree);
  const checkedKeys = useUIStore((s) => s.checkedKeys);
  const refreshTick = useUIStore((s) => s.refreshTick);
  const bumpRefresh = useUIStore((s) => s.bumpRefresh);
  const runKeys = useMemo(() => Array.from(checkedKeys), [checkedKeys]);
  const runKeysKey = runKeys.join("|");

  const [scalarTags, setScalarTags] = useState<string[]>([]);
  const [figureTags, setFigureTags] = useState<string[]>([]);
  const [fullscreenFigure, setFullscreenFigure] = useState<{ tag: string; node: RunNode } | null>(null);

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
  }, [runKeysKey, tree, refreshTick]);

  const grouped = useMemo(() => {
    const map: Record<string, string[]> = { train: [], eval: [], lr: [], other: [] };
    for (const t of scalarTags) map[classifyTag(t)].push(t);
    return map;
  }, [scalarTags]);

  const sectionCollapsed = useUIStore((s) => s.sectionCollapsed);
  const toggleSection = useUIStore((s) => s.toggleSection);

  if (runKeys.length === 0) {
    return (
      <div className="curve-grid-empty">
        Check runs in the tree to display their curves and figures here.
      </div>
    );
  }

  return (
    <div className="curve-grid-wrap">
      <div className="curves-toolbar">
        <div className="curves-toolbar-summary">
          {runKeys.length} run{runKeys.length === 1 ? "" : "s"} · {scalarTags.length} scalar{scalarTags.length === 1 ? "" : "s"}
          {figureTags.length > 0 && ` · ${figureTags.length} figure tag${figureTags.length === 1 ? "" : "s"}`}
        </div>
        <button
          type="button"
          className="curve-btn"
          onClick={() => bumpRefresh()}
          title="Refresh scalars and figures"
          aria-label="Refresh"
        >
          ↻ Refresh
        </button>
      </div>

      {SECTIONS.map((sec) => {
        const tags = grouped[sec.key];
        if (tags.length === 0) return null;
        const collapsed = !!sectionCollapsed[sec.id];
        return (
          <section key={sec.key} id={sec.id} className={`curve-section curve-section-${sec.key}${collapsed ? " collapsed" : ""}`}>
            <button
              type="button"
              className="curve-section-header curve-section-toggle"
              aria-expanded={!collapsed}
              onClick={() => toggleSection(sec.id)}
            >
              <span className="curve-section-caret">{collapsed ? "▸" : "▾"}</span>
              <span className="curve-section-label">{sec.label}</span>
              <span className="curve-section-count">{tags.length}</span>
            </button>
            {!collapsed && (
              <div className="curve-grid">
                {tags.map((tag) => (
                  <CurveCard key={tag} tag={tag} runKeys={runKeys} />
                ))}
              </div>
            )}
          </section>
        );
      })}

      {(() => {
        const collapsed = !!sectionCollapsed["figures"];
        return (
          <section
            id="figures"
            className={`curve-section curve-section-figures${figureTags.length === 0 ? " is-empty" : ""}${collapsed ? " collapsed" : ""}`}
          >
            <button
              type="button"
              className="curve-section-header curve-section-toggle"
              aria-expanded={!collapsed}
              onClick={() => toggleSection("figures")}
            >
              <span className="curve-section-caret">{collapsed ? "▸" : "▾"}</span>
              <span className="curve-section-label">Figures</span>
              {figureTags.length > 0 && <span className="curve-section-count">{figureTags.length}</span>}
            </button>
            {!collapsed && (
              figureTags.length === 0 ? (
                <div className="curve-grid-empty">
                  No figures logged for the selected runs.
                </div>
              ) : (
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
              )
            )}
          </section>
        );
      })()}

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
