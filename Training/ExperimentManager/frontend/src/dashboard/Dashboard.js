import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect } from "react";
import { useLiveStore } from "../state/liveStore";
import { useUIStore } from "../state/uiStore";
import { TrainingStatus } from "../section1_live/TrainingStatus";
import { ProgressBars } from "../section1_live/ProgressBars";
import { MetricCards } from "../section1_live/MetricCards";
import { LearningRateControl } from "../section1_live/LearningRateControl";
import { TreeView } from "../section2_manage/tree/TreeView";
import { CurveGrid } from "../section2_manage/curves/CurveGrid";
import { SummaryPanel } from "../section2_manage/summary/SummaryPanel";
import { ThemeToggle } from "../components/ThemeToggle";
const EMPTY_PROGRESS = {
    overall: 0, total: 0, percent: 0, epoch: 0, epochs: 0,
    step: 0, steps_per_epoch: 0, elapsed: 0, rate: 0, remaining: null,
};
export function Dashboard() {
    const snapshot = useLiveStore((s) => s.snapshot);
    const wsState = useLiveStore((s) => s.wsState);
    const tree = useUIStore((s) => s.tree);
    const loadTree = useUIStore((s) => s.loadTree);
    useEffect(() => {
        void loadTree();
    }, [loadTree]);
    return (_jsxs("div", { className: "app", children: [_jsxs("header", { className: "app-header", children: [_jsxs("div", { className: "brand", children: [_jsx("span", { className: "brand-mark", children: "JT" }), _jsxs("div", { children: [_jsx("div", { className: "brand-eyebrow", children: "JimmyTorch" }), _jsx("div", { className: "brand-title", children: "Experiment Manager" })] })] }), _jsxs("div", { className: "app-header-right", children: [snapshot?.run && (_jsxs("div", { className: "run-tag", children: [snapshot.run.dataset ?? "?", " / ", snapshot.run.model ?? "?", " / ", snapshot.run.run_name ?? "?"] })), _jsx(ThemeToggle, {})] })] }), _jsxs("section", { className: "section section-live", children: [_jsx("h2", { className: "section-heading", children: "Current Training" }), _jsx(TrainingStatus, { status: snapshot?.status ?? "idle", connected: snapshot?.connected ?? false, wsState: wsState }), _jsx(ProgressBars, { progress: snapshot?.progress ?? EMPTY_PROGRESS }), _jsxs("div", { className: "section-live-grid", children: [_jsx(MetricCards, { progress: snapshot?.progress ?? EMPTY_PROGRESS, system: snapshot?.system ?? {} }), _jsx(LearningRateControl, { learningRate: snapshot?.learning_rate ?? { applied: null, pending: null } })] })] }), _jsxs("section", { className: "section section-manage", children: [_jsx("h2", { className: "section-heading", children: "Experiment Management" }), _jsxs("div", { className: "manage-layout", children: [_jsx("div", { className: "manage-tree", children: _jsx(TreeView, { tree: tree }) }), _jsx("div", { className: "manage-curves", children: _jsx(CurveGrid, {}) }), _jsx("div", { className: "manage-summary", children: _jsx(SummaryPanel, {}) })] })] })] }));
}
