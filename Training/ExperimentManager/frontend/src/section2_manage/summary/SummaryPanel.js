import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { deleteRun, fetchRunSummary, getDeleteToken, openRunFolder, saveComments, setStarred, } from "../../api/rest";
import { parseKey, useUIStore } from "../../state/uiStore";
export function SummaryPanel() {
    const selectedKey = useUIStore((s) => s.selectedKey);
    const select = useUIStore((s) => s.select);
    const [summary, setSummary] = useState(null);
    const [comments, setComments] = useState("");
    const [initialComments, setInitialComments] = useState("");
    const [confirming, setConfirming] = useState(false);
    const [error, setError] = useState(null);
    useEffect(() => {
        setError(null);
        if (!selectedKey) {
            setSummary(null);
            return;
        }
        const k = parseKey(selectedKey);
        let cancelled = false;
        void (async () => {
            try {
                const s = await fetchRunSummary(k.dataset, k.model, k.run_name);
                if (cancelled)
                    return;
                setSummary(s);
                const c = s.meta.comments ?? "";
                setComments(c);
                setInitialComments(c);
            }
            catch (e) {
                if (!cancelled)
                    setError(String(e));
            }
        })();
        return () => { cancelled = true; };
    }, [selectedKey]);
    if (!selectedKey) {
        return _jsx("div", { className: "summary-empty", children: "Select a run in the tree to see details." });
    }
    if (error)
        return _jsx("div", { className: "summary-empty", children: error });
    if (!summary)
        return _jsx("div", { className: "summary-empty", children: "Loading\u2026" });
    const dirty = comments !== initialComments;
    const k = parseKey(selectedKey);
    return (_jsxs("div", { className: "summary-panel", children: [_jsxs("div", { className: "summary-run", children: [_jsxs("div", { className: "summary-run-title", children: [k.run_name, summary.node.starred && _jsx("span", { className: "tree-star", title: "Starred", children: "\u2605" })] }), _jsxs("div", { className: "summary-run-sub", children: [_jsx("span", { children: k.dataset }), _jsx("span", { children: " / " }), _jsx("span", { children: k.model })] })] }), _jsxs("div", { className: "summary-actions", children: [_jsx("button", { type: "button", onClick: () => setStarred(k.dataset, k.model, k.run_name, !summary.node.starred), children: summary.node.starred ? "★ Unstar" : "☆ Star" }), _jsx("button", { type: "button", onClick: () => openRunFolder(k.dataset, k.model, k.run_name), children: "\uD83D\uDCC1 Open Folder" }), _jsx("button", { type: "button", onClick: () => {
                            const url = `/arch/${encodeURIComponent(k.dataset)}/${encodeURIComponent(k.model)}/${encodeURIComponent(k.run_name)}`;
                            window.open(url, "_blank", "noopener");
                        }, disabled: !summary.node.has_arch, children: "\uD83C\uDFDB Show Architecture" }), _jsx("button", { type: "button", className: "btn-danger", onClick: () => setConfirming(true), children: "\uD83D\uDDD1 Delete" })] }), _jsxs("div", { className: "summary-section", children: [_jsxs("div", { className: "summary-section-header", children: [_jsx("span", { children: "Comments" }), dirty && (_jsxs("span", { className: "summary-actions-inline", children: [_jsx("button", { type: "button", onClick: async () => {
                                            await saveComments(k.dataset, k.model, k.run_name, comments);
                                            setInitialComments(comments);
                                        }, children: "Save" }), _jsx("button", { type: "button", onClick: () => setComments(initialComments), children: "Cancel" })] }))] }), _jsx("textarea", { value: comments, onChange: (e) => setComments(e.target.value), rows: 8 })] }), _jsxs("div", { className: "summary-section", children: [_jsx("div", { className: "summary-section-header", children: "Hyperparameters" }), _jsx(HparamsTable, { hparams: summary.hparams })] }), _jsxs("div", { className: "summary-section summary-meta", children: [_jsx("div", { className: "summary-section-header", children: "Meta" }), _jsxs("dl", { children: [_jsx("dt", { children: "status" }), _jsx("dd", { children: summary.node.status }), _jsx("dt", { children: "run_dir" }), _jsx("dd", { className: "mono", children: summary.node.run_dir }), summary.node.created_at && (_jsxs(_Fragment, { children: [_jsx("dt", { children: "created" }), _jsx("dd", { children: new Date(summary.node.created_at * 1000).toLocaleString() })] })), summary.node.closed_at && (_jsxs(_Fragment, { children: [_jsx("dt", { children: "closed" }), _jsx("dd", { children: new Date(summary.node.closed_at * 1000).toLocaleString() })] }))] })] }), confirming && (_jsx(DeleteConfirm, { runKey: k, onCancel: () => setConfirming(false), onDeleted: () => {
                    setConfirming(false);
                    select(null);
                } }))] }));
}
function DeleteConfirm({ runKey, onCancel, onDeleted, }) {
    const [busy, setBusy] = useState(false);
    const [err, setErr] = useState(null);
    return (_jsx("div", { className: "fullscreen-overlay", onClick: onCancel, children: _jsxs("div", { className: "confirm-dialog", onClick: (e) => e.stopPropagation(), children: [_jsx("div", { className: "confirm-title", children: "Delete run?" }), _jsxs("div", { className: "confirm-body", children: ["This permanently removes ", _jsx("br", {}), _jsxs("code", { className: "mono", children: [runKey.dataset, "/", runKey.model, "/", runKey.run_name] }), _jsx("br", {}), " and all its files (checkpoints, logs, figures)."] }), err && _jsx("div", { className: "confirm-error", children: err }), _jsxs("div", { className: "confirm-actions", children: [_jsx("button", { type: "button", onClick: onCancel, disabled: busy, children: "Cancel" }), _jsx("button", { type: "button", className: "btn-danger", disabled: busy, onClick: async () => {
                                setBusy(true);
                                setErr(null);
                                try {
                                    const tok = await getDeleteToken(runKey.dataset, runKey.model, runKey.run_name);
                                    const r = await deleteRun(runKey.dataset, runKey.model, runKey.run_name, tok);
                                    if (!r.ok)
                                        throw new Error(`Delete failed (${r.status})`);
                                    onDeleted();
                                }
                                catch (e) {
                                    setErr(String(e));
                                    setBusy(false);
                                }
                            }, children: busy ? "Deleting…" : "Delete" })] })] }) }));
}
function HparamsTable({ hparams }) {
    if (hparams == null)
        return _jsx("div", { className: "summary-empty-inline", children: "\u2014" });
    if (typeof hparams === "string")
        return _jsx("pre", { className: "mono summary-code", children: hparams });
    const entries = Object.entries(hparams);
    if (entries.length === 0)
        return _jsx("div", { className: "summary-empty-inline", children: "\u2014" });
    return (_jsx("table", { className: "hparams-table", children: _jsx("tbody", { children: entries.map(([k, v]) => (_jsxs("tr", { children: [_jsx("th", { children: k }), _jsx("td", { className: "mono", children: formatValue(v) })] }, k))) }) }));
}
function formatValue(v) {
    if (v == null)
        return "—";
    if (typeof v === "string" || typeof v === "number" || typeof v === "boolean")
        return String(v);
    try {
        return JSON.stringify(v);
    }
    catch {
        return String(v);
    }
}
