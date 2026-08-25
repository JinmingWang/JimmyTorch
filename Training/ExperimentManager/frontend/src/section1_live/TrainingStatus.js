import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
const LABEL = {
    idle: "Idle",
    training: "Training",
    evaluating: "Evaluating",
    done: "Done",
    error: "Error",
};
export function TrainingStatus({ status, connected, wsState }) {
    const isTraining = status === "training" || status === "evaluating";
    const badgeClass = `status-badge status-${isTraining ? "training" : "idle"} status-${status}`;
    const label = connected || status === "done" || status === "error" ? LABEL[status] : "Not Training";
    const wsLabel = wsState === "open" ? "live" : wsState === "connecting" ? "connecting…" : "offline";
    return (_jsxs("div", { className: "status-row", children: [_jsxs("div", { className: badgeClass, children: [_jsx("span", { className: "status-dot" }), _jsx("span", { className: "status-label", children: label })] }), _jsx("div", { className: `ws-pill ws-${wsState}`, children: wsLabel })] }));
}
