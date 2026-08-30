import { create } from "zustand";
import type { RunNode, TreeResponse } from "../api/rest";
import {
  fetchGlobalSettings,
  fetchTree,
  saveGlobalSettings,
} from "../api/rest";

export interface RunKey { dataset: string; model: string; run_name: string; }
export interface ExperimentNote { id: string; title: string; content: string; }

export function keyOf(k: RunKey): string {
  return `${k.dataset}\u0000${k.model}\u0000${k.run_name}`;
}

export function parseKey(s: string): RunKey {
  const [dataset, model, run_name] = s.split("\u0000");
  return { dataset, model, run_name };
}

interface UIState {
  tree: TreeResponse | null;
  loadedTree: boolean;

  selectedKey: string | null;
  checkedKeys: Set<string>;

  theme: "light" | "dark";

  // Curve card UI persisted globally, keyed by tag.
  smoothing: Record<string, number>;
  xlim: Record<string, [number | null, number | null]>;
  ylim: Record<string, [number | null, number | null]>;
  logScale: Record<string, boolean>;
  rangeStart: Record<string, number | null>;
  rangeEnd: Record<string, number | null>;
  notes: ExperimentNote[];

  // Section collapse state, keyed by section id (e.g. 'curves-train'). Persisted.
  sectionCollapsed: Record<string, boolean>;

  // Currently-expanded curve tag (only one at a time). Not persisted.
  expandedTag: string | null;

  // Bumped when the user clicks the manual refresh button. Not persisted.
  refreshTick: number;

  loadTree: () => Promise<void>;
  applyTreeUpdate: (t: TreeResponse) => void;

  loadGlobalSettings: () => Promise<void>;
  patchGlobalSettings: (patch: Record<string, unknown>) => Promise<void>;

  select: (key: string | null) => void;
  toggleChecked: (key: string) => void;
  setChecked: (keys: string[], value: boolean) => void;

  setTheme: (t: "light" | "dark") => Promise<void>;

  setSmoothing: (tag: string, v: number) => void;
  setXlim: (tag: string, min: number | null, max: number | null) => void;
  setYlim: (tag: string, min: number | null, max: number | null) => void;
  setLogScale: (tag: string, v: boolean) => void;
  setRange: (tag: string, start: number | null, end: number | null) => void;
  setNotes: (notes: ExperimentNote[]) => void;
  resetCurveSettings: (tag: string) => void;
  toggleSection: (id: string) => void;
  setExpandedTag: (tag: string | null) => void;
  bumpRefresh: () => void;
}

const SETTINGS_KEY_CHECKED = "tree.checked";
const SETTINGS_KEY_THEME = "theme";
const SETTINGS_KEY_SMOOTHING = "curves.smoothing";
const SETTINGS_KEY_XLIM = "curves.xlim";
const SETTINGS_KEY_YLIM = "curves.ylim";
const SETTINGS_KEY_LOGSCALE = "curves.logscale";
const SETTINGS_KEY_RANGE_START = "curves.rangeStart";
const SETTINGS_KEY_RANGE_END = "curves.rangeEnd";
const SETTINGS_KEY_SECTION_COLLAPSED = "curves.sectionCollapsed";
const SETTINGS_KEY_NOTES = "notes.cards";
const SETTINGS_KEY_NOTES_CONTENT = "notes.content";

function applyThemeToDocument(t: "light" | "dark") {
  document.documentElement.setAttribute("data-theme", t);
}

export const useUIStore = create<UIState>((set, get) => ({
  tree: null,
  loadedTree: false,
  selectedKey: null,
  checkedKeys: new Set(),
  theme: "light",
  smoothing: {},
  xlim: {},
  ylim: {},
  logScale: {},
  rangeStart: {},
  rangeEnd: {},
  notes: [],
  sectionCollapsed: {},
  expandedTag: null,
  refreshTick: 0,

  async loadTree() {
    const t = await fetchTree();
    set({ tree: t, loadedTree: true });
  },

  applyTreeUpdate(t) {
    // Prune checked keys that no longer exist.
    const currentKeys = new Set<string>();
    for (const [ds, models] of Object.entries(t.datasets)) {
      for (const [m, runs] of Object.entries(models)) {
        for (const r of Object.keys(runs)) {
          currentKeys.add(keyOf({ dataset: ds, model: m, run_name: r }));
        }
      }
    }
    const currentChecked = get().checkedKeys;
    const pruned = new Set<string>();
    for (const k of currentChecked) if (currentKeys.has(k)) pruned.add(k);
    const selected = get().selectedKey;
    set({
      tree: t,
      loadedTree: true,
      checkedKeys: pruned,
      selectedKey: selected && currentKeys.has(selected) ? selected : null,
    });
  },

  async loadGlobalSettings() {
    const s = await fetchGlobalSettings();
    const checked = new Set<string>(
      Array.isArray(s[SETTINGS_KEY_CHECKED]) ? (s[SETTINGS_KEY_CHECKED] as string[]) : [],
    );
    const theme = s[SETTINGS_KEY_THEME] === "dark" ? "dark" : "light";
    applyThemeToDocument(theme);
    set({
      checkedKeys: checked,
      theme,
      smoothing: (s[SETTINGS_KEY_SMOOTHING] as Record<string, number>) ?? {},
      xlim: (s[SETTINGS_KEY_XLIM] as Record<string, [number | null, number | null]>) ?? {},
      ylim: (s[SETTINGS_KEY_YLIM] as Record<string, [number | null, number | null]>) ?? {},
      logScale: (s[SETTINGS_KEY_LOGSCALE] as Record<string, boolean>) ?? {},
      rangeStart: (s[SETTINGS_KEY_RANGE_START] as Record<string, number | null>) ?? {},
      rangeEnd: (s[SETTINGS_KEY_RANGE_END] as Record<string, number | null>) ?? {},
      notes: Array.isArray(s[SETTINGS_KEY_NOTES])
        ? (s[SETTINGS_KEY_NOTES] as ExperimentNote[])
        : typeof s[SETTINGS_KEY_NOTES_CONTENT] === "string" && s[SETTINGS_KEY_NOTES_CONTENT].trim()
          ? [{ id: "legacy-notes", title: "Notes", content: s[SETTINGS_KEY_NOTES_CONTENT] }]
          : [],
      sectionCollapsed: (s[SETTINGS_KEY_SECTION_COLLAPSED] as Record<string, boolean>) ?? {},
    });
  },

  async patchGlobalSettings(patch) {
    await saveGlobalSettings(patch);
  },

  select(key) {
    set({ selectedKey: key });
  },

  toggleChecked(key) {
    const s = new Set(get().checkedKeys);
    if (s.has(key)) s.delete(key);
    else s.add(key);
    set({ checkedKeys: s });
    void saveGlobalSettings({ [SETTINGS_KEY_CHECKED]: Array.from(s) });
  },

  setChecked(keys, value) {
    const s = new Set(get().checkedKeys);
    for (const k of keys) {
      if (value) s.add(k);
      else s.delete(k);
    }
    set({ checkedKeys: s });
    void saveGlobalSettings({ [SETTINGS_KEY_CHECKED]: Array.from(s) });
  },

  async setTheme(t) {
    applyThemeToDocument(t);
    set({ theme: t });
    await saveGlobalSettings({ [SETTINGS_KEY_THEME]: t });
  },

  setSmoothing(tag, v) {
    const next = { ...get().smoothing, [tag]: v };
    set({ smoothing: next });
    void saveGlobalSettings({ [SETTINGS_KEY_SMOOTHING]: next });
  },

  setXlim(tag, min, max) {
    const next = { ...get().xlim, [tag]: [min, max] as [number | null, number | null] };
    set({ xlim: next });
    void saveGlobalSettings({ [SETTINGS_KEY_XLIM]: next });
  },

  setYlim(tag, min, max) {
    const next = { ...get().ylim, [tag]: [min, max] as [number | null, number | null] };
    set({ ylim: next });
    void saveGlobalSettings({ [SETTINGS_KEY_YLIM]: next });
  },

  setLogScale(tag, v) {
    const next = { ...get().logScale, [tag]: v };
    set({ logScale: next });
    void saveGlobalSettings({ [SETTINGS_KEY_LOGSCALE]: next });
  },

  setRange(tag, start, end) {
    const nextStart = { ...get().rangeStart, [tag]: start };
    const nextEnd = { ...get().rangeEnd, [tag]: end };
    set({ rangeStart: nextStart, rangeEnd: nextEnd });
    void saveGlobalSettings({
      [SETTINGS_KEY_RANGE_START]: nextStart,
      [SETTINGS_KEY_RANGE_END]: nextEnd,
    });
  },

  setNotes(notes) {
    set({ notes });
    void saveGlobalSettings({ [SETTINGS_KEY_NOTES]: notes });
  },

  resetCurveSettings(tag) {
    const nextSm = { ...get().smoothing }; delete nextSm[tag];
    const nextX = { ...get().xlim }; delete nextX[tag];
    const nextY = { ...get().ylim }; delete nextY[tag];
    const nextLog = { ...get().logScale }; delete nextLog[tag];
    const nextRS = { ...get().rangeStart }; delete nextRS[tag];
    const nextRE = { ...get().rangeEnd }; delete nextRE[tag];
    set({ smoothing: nextSm, xlim: nextX, ylim: nextY, logScale: nextLog, rangeStart: nextRS, rangeEnd: nextRE });
    void saveGlobalSettings({
      [SETTINGS_KEY_SMOOTHING]: nextSm,
      [SETTINGS_KEY_XLIM]: nextX,
      [SETTINGS_KEY_YLIM]: nextY,
      [SETTINGS_KEY_LOGSCALE]: nextLog,
      [SETTINGS_KEY_RANGE_START]: nextRS,
      [SETTINGS_KEY_RANGE_END]: nextRE,
    });
  },

  toggleSection(id) {
    const cur = get().sectionCollapsed;
    const next = { ...cur, [id]: !cur[id] };
    set({ sectionCollapsed: next });
    void saveGlobalSettings({ [SETTINGS_KEY_SECTION_COLLAPSED]: next });
  },

  bumpRefresh() {
    set({ refreshTick: get().refreshTick + 1 });
  },

  setExpandedTag(tag) {
    set({ expandedTag: tag });
  },
}));

export function iterRuns(tree: TreeResponse | null): RunNode[] {
  if (!tree) return [];
  const out: RunNode[] = [];
  for (const [_, models] of Object.entries(tree.datasets)) {
    for (const [__, runs] of Object.entries(models)) {
      for (const r of Object.values(runs)) out.push(r);
    }
  }
  return out;
}

export function findRun(tree: TreeResponse | null, key: string): RunNode | null {
  if (!tree) return null;
  const k = parseKey(key);
  return tree.datasets[k.dataset]?.[k.model]?.[k.run_name] ?? null;
}

export const RUN_COLORS = [
  "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#5e3c99",
  "#17becf", "#e7298a", "#66a61e", "#a6761d", "#1f78b4", "#e31a1c",
];

/** Deterministic default color for a run key when the user hasn't picked one. */
export function defaultColorForKey(key: string): string {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return RUN_COLORS[h % RUN_COLORS.length];
}

export function effectiveColor(node: RunNode): string {
  return node.color && node.color.length > 0
    ? node.color
    : defaultColorForKey(keyOf(node));
}
