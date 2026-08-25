import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "wouter";
import type { ModuleNode } from "./model-parser";
import { ModelParseError, parseModelArchitecture } from "./model-parser";
import { ModelDiagram } from "./diagram";
import "./arch.css";

interface Params { dataset: string; model: string; run: string; }

export function ArchViewer() {
  const params = useParams<Params>();
  const dataset = decodeURIComponent(params.dataset ?? "");
  const model = decodeURIComponent(params.model ?? "");
  const run = decodeURIComponent(params.run ?? "");

  const [source, setSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [root, setRoot] = useState<ModuleNode | null>(null);
  const [selected, setSelected] = useState<ModuleNode | null>(null);

  const diagramContainer = useRef<HTMLDivElement | null>(null);
  const diagramInstance = useRef<ModelDiagram | null>(null);

  useEffect(() => {
    document.title = `Architecture · ${dataset}/${model}/${run}`;
  }, [dataset, model, run]);

  useEffect(() => {
    let cancelled = false;
    setError(null); setSource(null); setRoot(null);
    void (async () => {
      const url = `/api/runs/${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/${encodeURIComponent(run)}/arch_txt`;
      const r = await fetch(url);
      if (!r.ok) {
        if (!cancelled) setError(`Could not load model_arch.txt (HTTP ${r.status}).`);
        return;
      }
      const txt = await r.text();
      if (cancelled) return;
      setSource(txt);
    })();
    return () => { cancelled = true; };
  }, [dataset, model, run]);

  useEffect(() => {
    if (source === null) return;
    try {
      const parsed = parseModelArchitecture(source);
      setRoot(parsed);
      setSelected(parsed);
      setError(null);
    } catch (e) {
      if (e instanceof ModelParseError) setError(e.message);
      else setError(String(e));
    }
  }, [source]);

  useEffect(() => {
    if (!diagramContainer.current || !root) return;
    const inst = new ModelDiagram(diagramContainer.current, { onSelect: setSelected });
    diagramInstance.current = inst;
    inst.setModel(root);
    return () => {
      diagramContainer.current?.replaceChildren();
      diagramInstance.current = null;
    };
  }, [root]);

  return (
    <div className="arch-viewer">
      <div className="arch-header">
        <div>
          <div className="arch-header-path">{dataset} / {model} / {run}</div>
          <div className="arch-header-sub">architecture · click a card to expand · deeper hierarchy appears to the right</div>
        </div>
      </div>

      {error && <div className="arch-error">{error}</div>}

      {!error && (
        <div className="arch-workspace">
          <div className="arch-diagram-panel">
            <div className="arch-diagram-scroll">
              <div className="arch-diagram" ref={diagramContainer} />
            </div>
          </div>
          <aside className="arch-detail-panel">
            <h2>Selected module</h2>
            <ArchDetail node={selected} />
          </aside>
        </div>
      )}

      {!error && source === null && <div className="arch-empty">Loading…</div>}
    </div>
  );
}

function ArchDetail({ node }: { node: ModuleNode | null }) {
  const items = useMemo(() => {
    if (!node) return [] as { k: string; v: string }[];
    return [
      { k: "variable", v: node.key ?? "(root)" },
      { k: "class", v: node.name },
      { k: "path", v: node.path.join(" ▸ ") || node.name },
      { k: "children", v: String(node.children.length) },
      { k: "parameters", v: node.parameters || "—" },
    ];
  }, [node]);
  if (!node) return <div className="arch-empty">Nothing selected.</div>;
  return (
    <dl>
      {items.map(({ k, v }) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
