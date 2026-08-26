import { useEffect, useState } from "react";
import {
  deleteRun,
  fetchRunSummary,
  getDeleteToken,
  openRunFolder,
  saveComments,
  setStarred,
} from "../../api/rest";
import type { RunSummary } from "../../api/rest";
import { parseKey, useUIStore } from "../../state/uiStore";

export function SummaryPanel() {
  const selectedKey = useUIStore((s) => s.selectedKey);
  const select = useUIStore((s) => s.select);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [comments, setComments] = useState("");
  const [initialComments, setInitialComments] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    if (!selectedKey) { setSummary(null); return; }
    const k = parseKey(selectedKey);
    let cancelled = false;
    void (async () => {
      try {
        const s = await fetchRunSummary(k.dataset, k.model, k.run_name);
        if (cancelled) return;
        setSummary(s);
        const c = s.meta.comments ?? "";
        setComments(c);
        setInitialComments(c);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [selectedKey]);

  if (!selectedKey) {
    return <div className="summary-empty">Select a run in the tree to see details.</div>;
  }
  if (error) return <div className="summary-empty">{error}</div>;
  if (!summary) return <div className="summary-empty">Loading…</div>;

  const dirty = comments !== initialComments;
  const k = parseKey(selectedKey);

  return (
    <div className="summary-panel">
      <div className="summary-run">
        <div className="summary-run-title">
          {k.run_name}
          {summary.node.starred && <span className="tree-star" title="Starred">★</span>}
        </div>
        <div className="summary-run-sub">
          <span>{k.dataset}</span>
          <span> / </span>
          <span>{k.model}</span>
        </div>
      </div>

      <div className="summary-actions">
        <button
          type="button"
          onClick={() => setStarred(k.dataset, k.model, k.run_name, !summary.node.starred)}
        >
          {summary.node.starred ? "★ Unstar" : "☆ Star"}
        </button>
        <button
          type="button"
          onClick={() => openRunFolder(k.dataset, k.model, k.run_name)}
        >
          📁 Open Folder
        </button>
        <button
          type="button"
          onClick={() => {
            const url = `/arch/${encodeURIComponent(k.dataset)}/${encodeURIComponent(k.model)}/${encodeURIComponent(k.run_name)}`;
            window.open(url, "_blank", "noopener");
          }}
          disabled={!summary.node.has_arch}
        >
          🏛 Architecture
        </button>
        <button
          type="button"
          className="btn-danger"
          onClick={() => setConfirming(true)}
        >
          🗑 Delete
        </button>
      </div>

      <div className="summary-section">
        <div className="summary-section-header">
          <span>Comments</span>
          {dirty && (
            <span className="summary-actions-inline">
              <button
                type="button"
                onClick={async () => {
                  await saveComments(k.dataset, k.model, k.run_name, comments);
                  setInitialComments(comments);
                }}
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setComments(initialComments)}
              >
                Cancel
              </button>
            </span>
          )}
        </div>
        <textarea
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          rows={8}
        />
      </div>

      <div className="summary-section">
        <div className="summary-section-header">Hyperparameters</div>
        <HparamsTable hparams={summary.hparams} />
      </div>

      <div className="summary-section summary-meta">
        <div className="summary-section-header">Meta</div>
        <dl>
          <dt>status</dt><dd>{summary.node.status}</dd>
          <dt>run_dir</dt><dd className="mono">{summary.node.run_dir}</dd>
          {summary.node.created_at && (
            <>
              <dt>created</dt>
              <dd>{new Date(summary.node.created_at * 1000).toLocaleString()}</dd>
            </>
          )}
          {summary.node.closed_at && (
            <>
              <dt>closed</dt>
              <dd>{new Date(summary.node.closed_at * 1000).toLocaleString()}</dd>
            </>
          )}
        </dl>
      </div>

      {confirming && (
        <DeleteConfirm
          runKey={k}
          onCancel={() => setConfirming(false)}
          onDeleted={() => {
            setConfirming(false);
            select(null);
          }}
        />
      )}
    </div>
  );
}

function DeleteConfirm({
  runKey,
  onCancel,
  onDeleted,
}: {
  runKey: { dataset: string; model: string; run_name: string };
  onCancel: () => void;
  onDeleted: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="fullscreen-overlay" onClick={onCancel}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="confirm-title">Delete run?</div>
        <div className="confirm-body">
          This permanently removes <br />
          <code className="mono">{runKey.dataset}/{runKey.model}/{runKey.run_name}</code>
          <br /> and all its files (checkpoints, logs, figures).
        </div>
        {err && <div className="confirm-error">{err}</div>}
        <div className="confirm-actions">
          <button type="button" onClick={onCancel} disabled={busy}>Cancel</button>
          <button
            type="button"
            className="btn-danger"
            disabled={busy}
            onClick={async () => {
              setBusy(true); setErr(null);
              try {
                const tok = await getDeleteToken(runKey.dataset, runKey.model, runKey.run_name);
                const r = await deleteRun(runKey.dataset, runKey.model, runKey.run_name, tok);
                if (!r.ok) throw new Error(`Delete failed (${r.status})`);
                onDeleted();
              } catch (e) {
                setErr(String(e));
                setBusy(false);
              }
            }}
          >
            {busy ? "Deleting…" : "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
}

function HparamsTable({ hparams }: { hparams: RunSummary["hparams"] }) {
  if (hparams == null) return <div className="summary-empty-inline">—</div>;
  if (typeof hparams === "string") return <pre className="mono summary-code">{hparams}</pre>;
  const entries = Object.entries(hparams);
  if (entries.length === 0) return <div className="summary-empty-inline">—</div>;
  return (
    <div className="hparams-cards">
      {entries.map(([k, v]) => (
        <div key={k} className="hparams-card">
          <div className="hparams-card-title">{k}</div>
          <div className="hparams-card-value mono">{formatValue(v)}</div>
        </div>
      ))}
    </div>
  );
}

function formatValue(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") return String(v);
  try { return JSON.stringify(v); } catch { return String(v); }
}
