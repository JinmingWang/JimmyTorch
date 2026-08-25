import { create } from "zustand";
import { fetchGlobalSettings, fetchTree, saveGlobalSettings, } from "../api/rest";
export function keyOf(k) {
    return `${k.dataset}\u0000${k.model}\u0000${k.run_name}`;
}
export function parseKey(s) {
    const [dataset, model, run_name] = s.split("\u0000");
    return { dataset, model, run_name };
}
const SETTINGS_KEY_CHECKED = "tree.checked";
const SETTINGS_KEY_THEME = "theme";
const SETTINGS_KEY_SMOOTHING = "curves.smoothing";
const SETTINGS_KEY_COLLAPSED = "curves.collapsed";
const SETTINGS_KEY_XLIM = "curves.xlim";
const SETTINGS_KEY_YLIM = "curves.ylim";
function applyThemeToDocument(t) {
    document.documentElement.setAttribute("data-theme", t);
}
export const useUIStore = create((set, get) => ({
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
        const currentKeys = new Set();
        for (const [ds, models] of Object.entries(t.datasets)) {
            for (const [m, runs] of Object.entries(models)) {
                for (const r of Object.keys(runs)) {
                    currentKeys.add(keyOf({ dataset: ds, model: m, run_name: r }));
                }
            }
        }
        const currentChecked = get().checkedKeys;
        const pruned = new Set();
        for (const k of currentChecked)
            if (currentKeys.has(k))
                pruned.add(k);
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
        const checked = new Set(Array.isArray(s[SETTINGS_KEY_CHECKED]) ? s[SETTINGS_KEY_CHECKED] : []);
        const theme = s[SETTINGS_KEY_THEME] === "dark" ? "dark" : "light";
        applyThemeToDocument(theme);
        set({
            checkedKeys: checked,
            theme,
            smoothing: s[SETTINGS_KEY_SMOOTHING] ?? {},
            collapsed: s[SETTINGS_KEY_COLLAPSED] ?? {},
            xlim: s[SETTINGS_KEY_XLIM] ?? {},
            ylim: s[SETTINGS_KEY_YLIM] ?? {},
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
        if (s.has(key))
            s.delete(key);
        else
            s.add(key);
        set({ checkedKeys: s });
        void saveGlobalSettings({ [SETTINGS_KEY_CHECKED]: Array.from(s) });
    },
    setChecked(keys, value) {
        const s = new Set(get().checkedKeys);
        for (const k of keys) {
            if (value)
                s.add(k);
            else
                s.delete(k);
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
        const next = { ...get().xlim, [tag]: [min, max] };
        set({ xlim: next });
        void saveGlobalSettings({ [SETTINGS_KEY_XLIM]: next });
    },
    setYlim(tag, min, max) {
        const next = { ...get().ylim, [tag]: [min, max] };
        set({ ylim: next });
        void saveGlobalSettings({ [SETTINGS_KEY_YLIM]: next });
    },
}));
export function iterRuns(tree) {
    if (!tree)
        return [];
    const out = [];
    for (const [_, models] of Object.entries(tree.datasets)) {
        for (const [__, runs] of Object.entries(models)) {
            for (const r of Object.values(runs))
                out.push(r);
        }
    }
    return out;
}
export function findRun(tree, key) {
    if (!tree)
        return null;
    const k = parseKey(key);
    return tree.datasets[k.dataset]?.[k.model]?.[k.run_name] ?? null;
}
const DEFAULT_COLORS = [
    "#55ded0", "#d9ef67", "#ff886c", "#ffd166", "#8db9df", "#d8a5d9",
    "#a6c98f", "#f3c96b", "#ea9a86", "#94d9c7", "#c78ce9", "#e57c62",
];
/** Deterministic default color for a run key when the user hasn't picked one. */
export function defaultColorForKey(key) {
    let h = 0;
    for (let i = 0; i < key.length; i++)
        h = (h * 31 + key.charCodeAt(i)) >>> 0;
    return DEFAULT_COLORS[h % DEFAULT_COLORS.length];
}
export function effectiveColor(node) {
    return node.color && node.color.length > 0
        ? node.color
        : defaultColorForKey(keyOf(node));
}
