import { useEffect, useState } from "react";
import { Link } from "wouter";
import { Comparison } from "../comparison/Comparison";
import { useLiveStore } from "../state/liveStore";
import { keyOf, useUIStore } from "../state/uiStore";
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

export function Dashboard({ comparison = false }: { comparison?: boolean }) {
  const snapshot = useLiveStore((s) => s.snapshot);
  const wsState = useLiveStore((s) => s.wsState);
  const selectedKey = useUIStore((s) => s.selectedKey);
  const tree = useUIStore((s) => s.tree);
  const loadTree = useUIStore((s) => s.loadTree);
  const select = useUIStore((s) => s.select);
  const [highlighted, setHighlighted] = useState(selectedKey);
  const [reveal, setReveal] = useState<{ key: string } | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    void loadTree();
  }, [loadTree]);

  // Support deep-linking: ?dataset=…&model=…&run=… selects that run once on mount.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const dataset = params.get("dataset");
    const model = params.get("model");
    const run = params.get("run");
    if (dataset && model && run) {
      select(keyOf({ dataset, model, run_name: run }));
    }
  }, [select]);

  return (
    <div className={comparison ? "app comparison-page" : "app"}>
      <header className="app-header">
        <div className="brand">
          <img className="brand-mark" src="/jimmytorch-mark.svg" alt="" aria-hidden="true" />
          <div>
            <div className="brand-eyebrow">JimmyTorch</div>
            <div className="brand-title">{comparison ? "Model comparison" : "Experiment Manager"}</div>
          </div>
        </div>
        <div className="app-header-right">
          {!comparison && snapshot?.run && (
            <div className="run-tag">
              {snapshot.run.dataset ?? "?"} / {snapshot.run.model ?? "?"} / {snapshot.run.run_name ?? "?"}
            </div>
          )}
          <Link className="comparison-link" href={comparison ? "/" : "/compare"}>
            {comparison ? "← Dashboard" : "Compare models →"}
          </Link>
          <ThemeToggle reload={!comparison} />
        </div>
      </header>

      {!comparison && <section className="section section-live">
        <h2 className="section-heading">Current Training</h2>
        <TrainingStatus
          status={snapshot?.status ?? "idle"}
          connected={snapshot?.connected ?? false}
          wsState={wsState}
        />
        <ProgressBars progress={snapshot?.progress ?? EMPTY_PROGRESS} />
        <div className="section-live-grid">
          <MetricCards
            progress={snapshot?.progress ?? EMPTY_PROGRESS}
            system={snapshot?.system ?? {}}
            nodes={snapshot?.nodes}
          />
          <LearningRateControl
            learningRate={snapshot?.learning_rate ?? { applied: null, pending: null }}
          />
        </div>
      </section>}

      <section className="section section-manage">
        {!comparison && <h2 className="section-heading">Experiment Management</h2>}
        <div className={`manage-layout${sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
          <aside className={`manage-sidebar${sidebarCollapsed ? " is-collapsed" : ""}`}>
            <button
              type="button"
              className="manage-pane-toggle"
              onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
              title={sidebarCollapsed ? "Expand run panels" : "Collapse run panels"}
              aria-label={sidebarCollapsed ? "Expand run panels" : "Collapse run panels"}
            >
              <span className="manage-pane-toggle-label">Run tree &amp; summary</span>
              <span className="manage-pane-toggle-icon" aria-hidden="true">{sidebarCollapsed ? "›" : "‹"}</span>
            </button>
            {!sidebarCollapsed && (
              <>
                <div className="manage-tree">
                  <TreeView tree={tree} reveal={comparison ? reveal : null} onSelectRun={comparison ? setHighlighted : undefined} />
                </div>
                <div className="manage-summary">
                  <SummaryPanel key={selectedKey} />
                </div>
              </>
            )}
          </aside>
          {comparison ? (
            <Comparison highlighted={highlighted === selectedKey ? highlighted : null} onClear={() => setHighlighted(null)} onPick={(key) => {
              setHighlighted(key);
              select(key);
              setSidebarCollapsed(false);
              setReveal({ key });
            }} />
          ) : <div className="manage-curves"><CurveGrid /></div>}
        </div>
      </section>
    </div>
  );
}
