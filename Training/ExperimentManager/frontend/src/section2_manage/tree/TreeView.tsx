import { useMemo, useState } from "react";
import type { TreeResponse } from "../../api/rest";
import { setColor, setStarred } from "../../api/rest";
import {
  effectiveColor,
  iterRuns,
  keyOf,
  useUIStore,
} from "../../state/uiStore";
import { useLiveStore } from "../../state/liveStore";
import { ColorCircle } from "./ColorCircle";

interface Props {
  tree: TreeResponse | null;
}

type CheckState = "unchecked" | "partial" | "checked";

function computeGroupState(memberKeys: string[], checked: Set<string>): CheckState {
  let on = 0;
  for (const k of memberKeys) if (checked.has(k)) on++;
  if (on === 0) return "unchecked";
  if (on === memberKeys.length) return "checked";
  return "partial";
}

export function TreeView({ tree }: Props) {
  const selectedKey = useUIStore((s) => s.selectedKey);
  const checkedKeys = useUIStore((s) => s.checkedKeys);
  const select = useUIStore((s) => s.select);
  const setChecked = useUIStore((s) => s.setChecked);
  const toggleChecked = useUIStore((s) => s.toggleChecked);
  const liveRun = useLiveStore((s) => s.snapshot?.run);
  const liveStatus = useLiveStore((s) => s.snapshot?.status);
  const [query, setQuery] = useState("");
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());

  const allRuns = useMemo(() => iterRuns(tree), [tree]);

  if (!tree || Object.keys(tree.datasets).length === 0) {
    return (
      <div className="tree-empty">
        No runs yet. Start a training and it will appear here.
      </div>
    );
  }

  const q = query.trim().toLowerCase();
  const matches = (s: string) => !q || s.toLowerCase().includes(q);
  const toggleGroup = (groupKey: string) => {
    setCollapsedGroups((current) => {
      const next = new Set(current);
      if (next.has(groupKey)) next.delete(groupKey);
      else next.add(groupKey);
      return next;
    });
  };

  return (
    <div className="tree-view">
      <div className="tree-filter">
        <input
          type="text"
          value={query}
          placeholder="Filter runs..."
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="tree-count">{allRuns.length} run{allRuns.length === 1 ? "" : "s"}</span>
      </div>
      <ul className="tree-root">
        {Object.entries(tree.datasets).map(([dataset, models]) => {
          const dsRuns: string[] = [];
          for (const [m, runs] of Object.entries(models)) {
            for (const r of Object.keys(runs)) {
              dsRuns.push(keyOf({ dataset, model: m, run_name: r }));
            }
          }
          const dsState = computeGroupState(dsRuns, checkedKeys);
          const showDs = !q || matches(dataset) || Object.entries(models).some(
            ([m, runs]) => matches(m) || Object.keys(runs).some(matches),
          );
          if (!showDs) return null;
          const datasetGroupKey = `dataset:${dataset}`;
          const datasetCollapsed = !q && collapsedGroups.has(datasetGroupKey);
          return (
            <li key={dataset} className="tree-node tree-dataset">
              <NodeRow
                label={dataset}
                depth={0}
                checkState={dsState}
                onCheck={() => setChecked(dsRuns, dsState !== "checked")}
                onToggle={() => toggleGroup(datasetGroupKey)}
              />
              {!datasetCollapsed && <ul>
                {Object.entries(models).map(([model, runs]) => {
                  const mRuns = Object.keys(runs).map((r) =>
                    keyOf({ dataset, model, run_name: r }),
                  );
                  const mState = computeGroupState(mRuns, checkedKeys);
                  const showM = !q || matches(dataset) || matches(model) || Object.keys(runs).some(matches);
                  if (!showM) return null;
                  const modelGroupKey = `model:${dataset}\u0000${model}`;
                  const modelCollapsed = !q && collapsedGroups.has(modelGroupKey);
                  return (
                    <li key={model} className="tree-node tree-model">
                      <NodeRow
                        label={model}
                        depth={1}
                        checkState={mState}
                        onCheck={() => setChecked(mRuns, mState !== "checked")}
                        onToggle={() => toggleGroup(modelGroupKey)}
                      />
                      {!modelCollapsed && <ul>
                        {Object.entries(runs).map(([runName, node]) => {
                          const key = keyOf({ dataset, model, run_name: runName });
                          if (q && !matches(dataset) && !matches(model) && !matches(runName)) return null;
                          const isSelected = selectedKey === key;
                          const isLive =
                            !!liveRun &&
                            liveRun.dataset === dataset &&
                            liveRun.model === model &&
                            liveRun.run_name === runName &&
                            liveStatus !== "done" &&
                            liveStatus !== "error";
                          const color = effectiveColor(node);
                          return (
                            <li key={runName} className={`tree-node tree-run${isSelected ? " selected" : ""}${isLive ? " live" : ""}`}>
                              <div
                                className="tree-row tree-row-leaf"
                                style={{ paddingLeft: `${2 * 20}px` }}
                                onClick={() => select(key)}
                              >
                                <input
                                  type="checkbox"
                                  checked={checkedKeys.has(key)}
                                  onChange={(e) => {
                                    e.stopPropagation();
                                    toggleChecked(key);
                                  }}
                                  onClick={(e) => e.stopPropagation()}
                                />
                                <span className="tree-label">
                                  {runName}
                                </span>
                                <span className="tree-spacer" />
                                <button
                                  type="button"
                                  className={`tree-star-btn${node.starred ? " on" : ""}`}
                                  title={node.starred ? "Unstar" : "Star"}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    void setStarred(dataset, model, runName, !node.starred);
                                  }}
                                >
                                  {node.starred ? "★" : "☆"}
                                </button>
                                <ColorCircle
                                  color={color}
                                  onChange={(c) => {
                                    void setColor(dataset, model, runName, c);
                                  }}
                                />
                              </div>
                            </li>
                          );
                        })}
                      </ul>}
                    </li>
                  );
                })}
              </ul>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function NodeRow({
  label,
  depth,
  checkState,
  onCheck,
  onToggle,
}: {
  label: string;
  depth: number;
  checkState: CheckState;
  onCheck: () => void;
  onToggle: () => void;
}) {
  return (
    <div className="tree-row tree-row-group" style={{ paddingLeft: `${depth * 20}px` }} onClick={onToggle}>
      <input
        type="checkbox"
        checked={checkState === "checked"}
        ref={(el) => {
          if (el) el.indeterminate = checkState === "partial";
        }}
        onChange={onCheck}
        onClick={(event) => event.stopPropagation()}
      />
      <span className="tree-label tree-label-group">{label}</span>
    </div>
  );
}
