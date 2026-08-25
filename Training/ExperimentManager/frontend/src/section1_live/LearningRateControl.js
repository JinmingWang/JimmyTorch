import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { postLearningRate } from "../api/ws";
import { formatFloat } from "../utils/format";
export function LearningRateControl({ learningRate }) {
    const [text, setText] = useState("");
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState(null);
    async function submit(e) {
        e.preventDefault();
        setError(null);
        const lr = Number(text);
        if (!isFinite(lr) || lr <= 0) {
            setError("Enter a positive number.");
            return;
        }
        setBusy(true);
        try {
            await postLearningRate(lr);
            setText("");
        }
        catch {
            setError("Request failed.");
        }
        finally {
            setBusy(false);
        }
    }
    return (_jsxs("div", { className: "lr-panel", children: [_jsx("div", { className: "lr-panel-header", children: "Learning Rate" }), _jsxs("div", { className: "lr-values", children: [_jsxs("div", { children: [_jsx("div", { className: "lr-values-label", children: "Applied" }), _jsx("div", { className: "lr-values-value", children: formatLR(learningRate.applied) })] }), _jsxs("div", { children: [_jsx("div", { className: "lr-values-label", children: "Pending" }), _jsx("div", { className: "lr-values-value", children: formatLR(learningRate.pending) })] })] }), _jsxs("form", { className: "lr-form", onSubmit: submit, children: [_jsx("input", { type: "text", inputMode: "decimal", placeholder: "e.g. 5e-5", value: text, onChange: (e) => setText(e.target.value), disabled: busy }), _jsx("button", { type: "submit", disabled: busy || text === "", children: "Apply" })] }), error && _jsx("div", { className: "lr-error", children: error })] }));
}
function formatLR(v) {
    if (v == null)
        return "—";
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4))
        return v.toExponential(3);
    return formatFloat(v, 6);
}
