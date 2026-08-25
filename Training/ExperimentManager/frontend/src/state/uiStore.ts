import { create } from "zustand";
import type { RunNode, TreeResponse } from "../api/rest";
import {
  fetchGlobalSettings,
  fetchTree,
  saveGlobalSettings,
} from "../api/rest";

export interface RunKey { dataset: string; model: string; run_name: string; }

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
  collapsed: Record<string, boolean>;
  xlim: Record<string, [number | null, number | null]>;
  ylim: Record<string, [number | null, number | null]>;

  loadTree: () => Promise<void>;
  applyTreeUpdate: (t: TreeResponse) => void;

  loadGlobalSettings: () => Promise<void>;
  patchGlobalSettings: (patch: Record<string, unknown>) => Promise<void>;

  select: (key: string | null) => void;
  toggleChecked: (key: string) => void;
  setChecked: (keys: string[], value: boolean) => void;

  setTheme: (t: "light" | "dark") => void;

  setSmoothing: (tag: string, v: number) => void;
  toggleCollapsed: (tag: string) => void;
  setXlim: (tag: string, min: number | null, max: number | null) => void;
  setYlim: (tag: string, min: number | null, max: number | null) => void;
}

const SETTINGS_KEY_CHECKED = "tree.checked";
const SETTINGS_KEY_THEME = "theme";
const SETTINGS_KEY_SMOOTHING = "curves.smoothing";
const SETTINGS_KEY_COLLAPSED = "curves.collapsed";
const SETTINGS_KEY_XLIM = "curves.xlim";
const SETTINGS_KEY_YLIM = "curves.ylim";

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
  collapsed: {},
  xlim: {},
  ylim: {},

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
      collapsed: (s[SETTINGS_KEY_COLLAPSED] as Record<string, boolean>) ?? {},
      xlim: (s[SETTINGS_KEY_XLIM] as Record<string, [number | null, number | null]>) ?? {},
      ylim: (s[SETTINGS_KEY_YLIM] as Record<string, [number | null, number | null]>) ?? {},
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

  setTheme(t) {
    applyThemeToDocument(t);
    set({ theme: t });
    void saveGlobalSettings({ [SETTINGS_KEY_THEME]: t });
  },

  setSmoothing(tag, v) {
    const next = { ...get().smoothing, [tag]: v };
    set({ smoothing: next });
    void saveGlobalSettings({ [SETTINGS_KEY_SMOOTHING]: next });
  },

  toggleCollapsed(tag) {
    const next = { ...get().collapsed, [tag]: !get().collapsed[tag] };
    set({ collapsed: next });
    void saveGlobalSettings({ [SETTINGS_KEY_COLLAPSED]: next });
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

const DEFAULT_COLORS = [
  "#55ded0", "#d9ef67", "#ff886c", "#ffd166", "#8db9df", "#d8a5d9",
  "#a6c98f", "#f3c96b", "#ea9a86", "#94d9c7", "#c78ce9", "#e57c62",
];

/** Deterministic default color for a run key when the user hasn't picked one. */
export function defaultColorForKey(key: string): string {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return DEFAULT_COLORS[h % DEFAULT_COLORS.length];
}

export function effectiveColor(node: RunNode): string {
  return node.color && node.color.length > 0
    ? node.color
    : defaultColorForKey(keyOf(node));
}
