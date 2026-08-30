import { useEffect, useMemo, useRef, useState } from "react";
import { fetchRunSummary } from "../../api/rest";
import type { RunNode } from "../../api/rest";
import { findRun, type ExperimentNote, useUIStore } from "../../state/uiStore";
import { useLiveStore } from "../../state/liveStore";
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
  const loadedTree = useUIStore((s) => s.loadedTree);
  const runKeys = useMemo(() => Array.from(checkedKeys), [checkedKeys]);
  const runKeysKey = runKeys.join("|");

  const [scalarTags, setScalarTags] = useState<string[]>([]);
  const [figureTags, setFigureTags] = useState<string[]>([]);
  const [fullscreenFigure, setFullscreenFigure] = useState<{ tag: string; node: RunNode } | null>(null);

  useEffect(() => {
    if (!loadedTree) return;
    let cancelled = false;
    async function loadTags() {
      const currentTree = useUIStore.getState().tree;
      const nodes = runKeys.map((k) => findRun(currentTree, k)).filter(Boolean) as RunNode[];
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
    // `tree` excluded from deps: live tree updates during training must not reset curve zoom/pan.
    // `loadedTree` included so the first fetch waits for the tree to actually load.
  }, [runKeysKey, refreshTick, loadedTree]);

  // Auto-refresh curves + figures every 10s while a run is training. Safe now that CurveCard
  // uses setData for point updates, so zoom/pan/mid-drag are preserved.
  const isLiveTraining = useLiveStore((s) => s.snapshot?.status === "training");
  useEffect(() => {
    if (!isLiveTraining) return;
    const iv = window.setInterval(() => bumpRefresh(), 10_000);
    return () => window.clearInterval(iv);
  }, [isLiveTraining, bumpRefresh]);

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

      <NotesSection />

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

function NotesSection() {
  const notes = useUIStore((s) => s.notes);
  const setNotes = useUIStore((s) => s.setNotes);
  const sectionCollapsed = useUIStore((s) => s.sectionCollapsed);
  const toggleSection = useUIStore((s) => s.toggleSection);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<ExperimentNote | null>(null);
  const sectionRef = useRef<HTMLElement | null>(null);
  const collapsed = !!sectionCollapsed.notes;

  useEffect(() => {
    if (!expandedId) return;
    const collapseOnOutsideClick = (event: MouseEvent) => {
      if (sectionRef.current && !sectionRef.current.contains(event.target as Node)) setExpandedId(null);
    };
    window.addEventListener("mousedown", collapseOnOutsideClick);
    return () => window.removeEventListener("mousedown", collapseOnOutsideClick);
  }, [expandedId]);

  const updateNote = (id: string, patch: Partial<ExperimentNote>) => {
    setNotes(notes.map((note) => note.id === id ? { ...note, ...patch } : note));
  };
  const addNote = () => {
    const note = { id: crypto.randomUUID(), title: "Untitled note", content: "" };
    setNotes([...notes, note]);
    setExpandedId(note.id);
  };

  return (
    <section ref={sectionRef} id="notes" className={`curve-section curve-section-notes${collapsed ? " collapsed" : ""}`}>
      <button
        type="button"
        className="curve-section-header curve-section-toggle"
        aria-expanded={!collapsed}
        onClick={() => toggleSection("notes")}
      >
        <span className="curve-section-caret">{collapsed ? "▸" : "▾"}</span>
        <span className="curve-section-label">Notes</span>
      </button>
      {!collapsed && (
        <div className="notes-grid">
          {notes.map((note) => {
            const expanded = note.id === expandedId;
            return (
              <article
                key={note.id}
                className={`note-card${expanded ? " expanded" : ""}`}
                onClick={() => setExpandedId(note.id)}
              >
                {expanded ? (
                  <>
                    <div className="note-card-title-row">
                      <input
                        className="note-card-title-input"
                        value={note.title}
                        onChange={(event) => updateNote(note.id, { title: event.target.value })}
                        onClick={(event) => event.stopPropagation()}
                        aria-label="Note title"
                      />
                      <button type="button" className="note-delete-btn" onClick={(event) => { event.stopPropagation(); setDeleteTarget(note); }}>Delete</button>
                    </div>
                    <textarea
                      className="note-card-content"
                      value={note.content}
                      onChange={(event) => updateNote(note.id, { content: event.target.value })}
                      onClick={(event) => event.stopPropagation()}
                      placeholder="Write notes, decisions, and updates..."
                      aria-label="Note content"
                    />
                  </>
                ) : (
                  <h3 className="note-card-title">{note.title || "Untitled note"}</h3>
                )}
              </article>
            );
          })}
          <button type="button" className="note-card note-add-card" onClick={addNote}>
            <span className="note-add-symbol">+</span>
            <span>Add New</span>
          </button>
        </div>
      )}
      {deleteTarget && (
        <div className="note-dialog-backdrop" role="presentation" onMouseDown={() => setDeleteTarget(null)}>
          <div className="note-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-note-title" onMouseDown={(event) => event.stopPropagation()}>
            <h3 id="delete-note-title">Delete note?</h3>
            <p>Delete “{deleteTarget.title || "Untitled note"}”? This cannot be undone.</p>
            <div className="note-dialog-actions">
              <button type="button" className="curve-btn" onClick={() => setDeleteTarget(null)}>Cancel</button>
              <button type="button" className="note-delete-btn" onClick={() => {
                setNotes(notes.filter((note) => note.id !== deleteTarget.id));
                setExpandedId(null);
                setDeleteTarget(null);
              }}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
