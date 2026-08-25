import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "wouter";
import { ModelParseError, parseModelArchitecture } from "./model-parser";
import { ModelDiagram } from "./diagram";
import "./arch.css";
export function ArchViewer() {
    const params = useParams();
    const dataset = decodeURIComponent(params.dataset ?? "");
    const model = decodeURIComponent(params.model ?? "");
    const run = decodeURIComponent(params.run ?? "");
    const [source, setSource] = useState(null);
    const [error, setError] = useState(null);
    const [root, setRoot] = useState(null);
    const [selected, setSelected] = useState(null);
    const diagramContainer = useRef(null);
    const diagramInstance = useRef(null);
    useEffect(() => {
        document.title = `Architecture · ${dataset}/${model}/${run}`;
    }, [dataset, model, run]);
    useEffect(() => {
        let cancelled = false;
        setError(null);
        setSource(null);
        setRoot(null);
        void (async () => {
            const url = `/api/runs/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/${encodeURIComponent(run)}/arch_txt`;
            const r = await fetch(url);
            if (!r.ok) {
                if (!cancelled)
                    setError(`Could not load model_arch.txt (HTTP ${r.status}).`);
                return;
            }
            const txt = await r.text();
            if (cancelled)
                return;
            setSource(txt);
        })();
        return () => { cancelled = true; };
    }, [dataset, model, run]);
    useEffect(() => {
        if (source === null)
            return;
        try {
            const parsed = parseModelArchitecture(source);
            setRoot(parsed);
            setSelected(parsed);
            setError(null);
        }
        catch (e) {
            if (e instanceof ModelParseError)
                setError(e.message);
            else
                setError(String(e));
        }
    }, [source]);
    useEffect(() => {
        if (!diagramContainer.current || !root)
            return;
        const inst = new ModelDiagram(diagramContainer.current, { onSelect: setSelected });
        diagramInstance.current = inst;
        inst.setModel(root);
        return () => {
            diagramContainer.current?.replaceChildren();
            diagramInstance.current = null;
        };
    }, [root]);
    return (_jsxs("div", { className: "arch-viewer", children: [_jsx("div", { className: "arch-header", children: _jsxs("div", { children: [_jsxs("div", { className: "arch-header-path", children: [dataset, " / ", model, " / ", run] }), _jsx("div", { className: "arch-header-sub", children: "architecture \u00B7 click a card to expand \u00B7 deeper hierarchy appears to the right" })] }) }), error && _jsx("div", { className: "arch-error", children: error }), !error && (_jsxs("div", { className: "arch-workspace", children: [_jsx("div", { className: "arch-diagram-panel", children: _jsx("div", { className: "arch-diagram-scroll", children: _jsx("div", { className: "arch-diagram", ref: diagramContainer }) }) }), _jsxs("aside", { className: "arch-detail-panel", children: [_jsx("h2", { children: "Selected module" }), _jsx(ArchDetail, { node: selected })] })] })), !error && source === null && _jsx("div", { className: "arch-empty", children: "Loading\u2026" })] }));
}
function ArchDetail({ node }) {
    const items = useMemo(() => {
        if (!node)
            return [];
        return [
            { k: "variable", v: node.key ?? "(root)" },
            { k: "class", v: node.name },
            { k: "path", v: node.path.join(" ▸ ") || node.name },
            { k: "children", v: String(node.children.length) },
            { k: "parameters", v: node.parameters || "—" },
        ];
    }, [node]);
    if (!node)
        return _jsx("div", { className: "arch-empty", children: "Nothing selected." });
    return (_jsx("dl", { children: items.map(({ k, v }) => (_jsxs("div", { children: [_jsx("dt", { children: k }), _jsx("dd", { children: v })] }, k))) }));
}
