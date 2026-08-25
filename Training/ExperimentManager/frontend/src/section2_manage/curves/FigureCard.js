import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { fetchFigureIndex, figureBlobUrl } from "../../api/rest";
export function FigureCard({ tag, node, onFullscreen }) {
    const [step, setStep] = useState(null);
    useEffect(() => {
        let cancelled = false;
        void (async () => {
            const idx = await fetchFigureIndex(node.dataset, node.model, node.run_name, tag);
            if (cancelled)
                return;
            if (idx.entries.length > 0)
                setStep(idx.entries[idx.entries.length - 1].step);
        })();
        return () => { cancelled = true; };
    }, [tag, node]);
    return (_jsxs("div", { className: "figure-card", onClick: onFullscreen, children: [_jsxs("div", { className: "figure-card-header", children: [_jsx("span", { className: "figure-card-title", children: tag }), _jsxs("span", { className: "figure-card-sub", children: [node.model, "/", node.run_name] })] }), step !== null ? (_jsx("img", { src: figureBlobUrl(node.dataset, node.model, node.run_name, tag, step), alt: `${tag} step ${step}` })) : (_jsx("div", { className: "figure-card-empty", children: "No figures." }))] }));
}
