import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from "react";
import { fetchRunSummary } from "../../api/rest";
import { findRun, useUIStore } from "../../state/uiStore";
import { CurveCard } from "./CurveCard";
import { FigureCard } from "./FigureCard";
/** Union of scalar+figure tags across all checked runs, cached until checked set changes. */
export function CurveGrid() {
    const tree = useUIStore((s) => s.tree);
    const checkedKeys = useUIStore((s) => s.checkedKeys);
    const [scalarTags, setScalarTags] = useState([]);
    const [figureTags, setFigureTags] = useState([]);
    const [fullscreenTag, setFullscreenTag] = useState(null);
    const [fullscreenFigure, setFullscreenFigure] = useState(null);
    const runKeys = useMemo(() => Array.from(checkedKeys), [checkedKeys]);
    useEffect(() => {
        let cancelled = false;
        async function loadTags() {
            const nodes = runKeys.map((k) => findRun(tree, k)).filter(Boolean);
            const results = await Promise.all(nodes.map((n) => fetchRunSummary(n.dataset, n.model, n.run_name).catch(() => null)));
            if (cancelled)
                return;
            const sTags = new Set();
            const fTags = new Set();
            for (const s of results) {
                if (!s)
                    continue;
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
        return (_jsx("div", { className: "curve-grid-empty", children: "Check runs in the tree to display their curves and figures here." }));
    }
    return (_jsxs("div", { className: "curve-grid-wrap", children: [scalarTags.length > 0 && (_jsxs("section", { className: "curve-section", children: [_jsx("div", { className: "curve-section-header", children: "Scalars" }), _jsx("div", { className: "curve-grid", children: scalarTags.map((tag) => (_jsx(CurveCard, { tag: tag, runKeys: runKeys, onFullscreen: () => setFullscreenTag(tag) }, tag))) })] })), figureTags.length > 0 && (_jsxs("section", { className: "curve-section", children: [_jsx("div", { className: "curve-section-header", children: "Figures" }), _jsx("div", { className: "figure-grid", children: figureTags.flatMap((tag) => runKeys
                            .map((k) => findRun(tree, k))
                            .filter((n) => !!n)
                            .map((node) => (_jsx(FigureCard, { tag: tag, node: node, onFullscreen: () => setFullscreenFigure({ tag, node }) }, `${tag}::${node.dataset}/${node.model}/${node.run_name}`)))) })] })), fullscreenTag && (_jsx("div", { className: "fullscreen-overlay", onClick: () => setFullscreenTag(null), children: _jsx("div", { className: "fullscreen-inner", onClick: (e) => e.stopPropagation(), children: _jsx(CurveCard, { tag: fullscreenTag, runKeys: runKeys, fullscreen: true, onClose: () => setFullscreenTag(null) }) }) })), fullscreenFigure && (_jsx(FigurePopup, { tag: fullscreenFigure.tag, node: fullscreenFigure.node, onClose: () => setFullscreenFigure(null) }))] }));
}
import { fetchFigureIndex, figureBlobUrl } from "../../api/rest";
function FigurePopup({ tag, node, onClose, }) {
    const [step, setStep] = useState(null);
    const [entries, setEntries] = useState([]);
    useEffect(() => {
        void (async () => {
            const idx = await fetchFigureIndex(node.dataset, node.model, node.run_name, tag);
            setEntries(idx.entries);
            if (idx.entries.length > 0)
                setStep(idx.entries[idx.entries.length - 1].step);
        })();
    }, [tag, node]);
    return (_jsx("div", { className: "fullscreen-overlay", onClick: onClose, children: _jsxs("div", { className: "fullscreen-inner figure-popup", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "figure-popup-header", children: [_jsxs("span", { children: [tag, " \u2014 ", node.dataset, " / ", node.model, " / ", node.run_name] }), _jsx("button", { type: "button", className: "curve-btn", onClick: onClose, children: "Close" })] }), _jsx("div", { className: "figure-popup-body", children: step !== null ? (_jsx("img", { src: figureBlobUrl(node.dataset, node.model, node.run_name, tag, step), alt: `${tag} step ${step}` })) : (_jsx("div", { className: "curve-empty", children: "No figures." })) }), entries.length > 1 && (_jsxs("div", { className: "figure-popup-footer", children: [_jsx("input", { type: "range", min: 0, max: entries.length - 1, value: Math.max(0, entries.findIndex((e) => e.step === step)), onChange: (e) => setStep(entries[Number(e.target.value)].step) }), _jsxs("span", { children: ["step ", step] })] }))] }) }));
}
