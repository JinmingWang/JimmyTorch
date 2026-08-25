import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useMemo, useState } from "react";
import { setColor, setStarred } from "../../api/rest";
import { effectiveColor, iterRuns, keyOf, useUIStore, } from "../../state/uiStore";
import { useLiveStore } from "../../state/liveStore";
import { ColorCircle } from "./ColorCircle";
function computeGroupState(memberKeys, checked) {
    let on = 0;
    for (const k of memberKeys)
        if (checked.has(k))
            on++;
    if (on === 0)
        return "unchecked";
    if (on === memberKeys.length)
        return "checked";
    return "partial";
}
export function TreeView({ tree }) {
    const selectedKey = useUIStore((s) => s.selectedKey);
    const checkedKeys = useUIStore((s) => s.checkedKeys);
    const select = useUIStore((s) => s.select);
    const setChecked = useUIStore((s) => s.setChecked);
    const toggleChecked = useUIStore((s) => s.toggleChecked);
    const liveRun = useLiveStore((s) => s.snapshot?.run);
    const liveStatus = useLiveStore((s) => s.snapshot?.status);
    const [query, setQuery] = useState("");
    const allRuns = useMemo(() => iterRuns(tree), [tree]);
    if (!tree || Object.keys(tree.datasets).length === 0) {
        return (_jsx("div", { className: "tree-empty", children: "No runs yet. Start a training and it will appear here." }));
    }
    const q = query.trim().toLowerCase();
    const matches = (s) => !q || s.toLowerCase().includes(q);
    return (_jsxs("div", { className: "tree-view", children: [_jsxs("div", { className: "tree-filter", children: [_jsx("input", { type: "text", value: query, placeholder: "Filter runs...", onChange: (e) => setQuery(e.target.value) }), _jsxs("span", { className: "tree-count", children: [allRuns.length, " run", allRuns.length === 1 ? "" : "s"] })] }), _jsx("ul", { className: "tree-root", children: Object.entries(tree.datasets).map(([dataset, models]) => {
                    const dsRuns = [];
                    for (const [m, runs] of Object.entries(models)) {
                        for (const r of Object.keys(runs)) {
                            dsRuns.push(keyOf({ dataset, model: m, run_name: r }));
                        }
                    }
                    const dsState = computeGroupState(dsRuns, checkedKeys);
                    const showDs = !q || matches(dataset) || Object.entries(models).some(([m, runs]) => matches(m) || Object.keys(runs).some(matches));
                    if (!showDs)
                        return null;
                    return (_jsxs("li", { className: "tree-node tree-dataset", children: [_jsx(NodeRow, { label: dataset, depth: 0, checkState: dsState, onCheck: () => setChecked(dsRuns, dsState !== "checked") }), _jsx("ul", { children: Object.entries(models).map(([model, runs]) => {
                                    const mRuns = Object.keys(runs).map((r) => keyOf({ dataset, model, run_name: r }));
                                    const mState = computeGroupState(mRuns, checkedKeys);
                                    const showM = !q || matches(dataset) || matches(model) || Object.keys(runs).some(matches);
                                    if (!showM)
                                        return null;
                                    return (_jsxs("li", { className: "tree-node tree-model", children: [_jsx(NodeRow, { label: model, depth: 1, checkState: mState, onCheck: () => setChecked(mRuns, mState !== "checked") }), _jsx("ul", { children: Object.entries(runs).map(([runName, node]) => {
                                                    const key = keyOf({ dataset, model, run_name: runName });
                                                    if (q && !matches(dataset) && !matches(model) && !matches(runName))
                                                        return null;
                                                    const isSelected = selectedKey === key;
                                                    const isLive = !!liveRun &&
                                                        liveRun.dataset === dataset &&
                                                        liveRun.model === model &&
                                                        liveRun.run_name === runName &&
                                                        liveStatus !== "done" &&
                                                        liveStatus !== "error";
                                                    const color = effectiveColor(node);
                                                    return (_jsx("li", { className: `tree-node tree-run${isSelected ? " selected" : ""}${isLive ? " live" : ""}`, children: _jsxs("div", { className: "tree-row tree-row-leaf", style: { paddingLeft: `${2 * 20}px` }, onClick: () => select(key), children: [_jsx("input", { type: "checkbox", checked: checkedKeys.has(key), onChange: (e) => {
                                                                        e.stopPropagation();
                                                                        toggleChecked(key);
                                                                    }, onClick: (e) => e.stopPropagation() }), _jsx("span", { className: `tree-status-dot status-${node.status}`, title: node.status }), _jsxs("span", { className: "tree-label", children: [runName, node.starred && _jsx("span", { className: "tree-star", title: "Starred", children: "\u2605" })] }), _jsx("span", { className: "tree-spacer" }), _jsx("button", { type: "button", className: `tree-star-btn${node.starred ? " on" : ""}`, title: node.starred ? "Unstar" : "Star", onClick: (e) => {
                                                                        e.stopPropagation();
                                                                        void setStarred(dataset, model, runName, !node.starred);
                                                                    }, children: node.starred ? "★" : "☆" }), _jsx(ColorCircle, { color: color, onChange: (c) => {
                                                                        void setColor(dataset, model, runName, c);
                                                                    } })] }) }, runName));
                                                }) })] }, model));
                                }) })] }, dataset));
                }) })] }));
}
function NodeRow({ label, depth, checkState, onCheck, }) {
    return (_jsxs("div", { className: "tree-row tree-row-group", style: { paddingLeft: `${depth * 20}px` }, children: [_jsx("input", { type: "checkbox", checked: checkState === "checked", ref: (el) => {
                    if (el)
                        el.indeterminate = checkState === "partial";
                }, onChange: onCheck }), _jsx("span", { className: "tree-label tree-label-group", children: label })] }));
}
