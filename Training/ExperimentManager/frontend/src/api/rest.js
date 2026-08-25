const jsonHeaders = { "Content-Type": "application/json" };
function encodePath(dataset, model, run) {
    return `${encodeURIComponent(dataset)}/${encodeURIComponent(model)}/${encodeURIComponent(run)}`;
}
export async function fetchTree() {
    const r = await fetch("/api/tree");
    return r.json();
}
export async function fetchLiveStatus() {
    const r = await fetch("/api/status");
    return r.json();
}
export async function fetchRunSummary(dataset, model, run) {
    const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/summary`);
    return r.json();
}
export async function fetchScalars(dataset, model, run, tag, maxPoints = 2000) {
    const params = new URLSearchParams({ tag, max_points: String(maxPoints) });
    const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/scalars?${params}`);
    return r.json();
}
export async function fetchFigureIndex(dataset, model, run, tag) {
    const params = new URLSearchParams({ tag });
    const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/figures?${params}`);
    return r.json();
}
export function figureBlobUrl(dataset, model, run, tag, step) {
    const params = new URLSearchParams({ tag, step: String(step) });
    return `/api/runs/${encodePath(dataset, model, run)}/figure_blob?${params}`;
}
export async function saveComments(dataset, model, run, comments) {
    await fetch(`/api/runs/${encodePath(dataset, model, run)}/comments`, {
        method: "POST", headers: jsonHeaders, body: JSON.stringify({ comments }),
    });
}
export async function setStarred(dataset, model, run, starred) {
    await fetch(`/api/runs/${encodePath(dataset, model, run)}/star`, {
        method: "POST", headers: jsonHeaders, body: JSON.stringify({ starred }),
    });
}
export async function setColor(dataset, model, run, color) {
    await fetch(`/api/runs/${encodePath(dataset, model, run)}/color`, {
        method: "POST", headers: jsonHeaders, body: JSON.stringify({ color }),
    });
}
export async function openRunFolder(dataset, model, run) {
    await fetch(`/api/runs/${encodePath(dataset, model, run)}/open_folder`, { method: "POST" });
}
export async function getDeleteToken(dataset, model, run) {
    const r = await fetch(`/api/runs/${encodePath(dataset, model, run)}/delete_token`);
    const j = await r.json();
    return j.confirm_token;
}
export async function deleteRun(dataset, model, run, token) {
    return fetch(`/api/runs/${encodePath(dataset, model, run)}`, {
        method: "DELETE", headers: jsonHeaders, body: JSON.stringify({ confirm_token: token }),
    });
}
export async function fetchGlobalSettings() {
    const r = await fetch("/api/global_settings");
    return r.json();
}
export async function saveGlobalSettings(patch) {
    await fetch("/api/global_settings", {
        method: "POST", headers: jsonHeaders, body: JSON.stringify(patch),
    });
}
