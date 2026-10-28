// Examiner's Desk frontend. Untrusted text (student answers, model output) is only ever set via textContent.
export const $ = (sel) => document.querySelector(sel);
export const state = { result: null, edits: new Map(), activeLine: null };

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c);
  return el;
}

export const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
export const fmtMarks = (x) => (Math.round(x * 100) / 100).toString();
export const fileUrl = (name) => `/api/results/${state.result.result_id}/files/${encodeURIComponent(name)}`;

export async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const msg = body && body.error ? body.error.message : `Request failed (${res.status})`;
    throw new Error(msg);
  }
  return body;
}

export function toast(msg) {
  const t = h("div", { class: "toast", role: "status", text: msg });
  document.body.append(t);
  setTimeout(() => t.remove(), 2600);
}

export function show(view) {
  for (const v of ["setup", "loading", "result"]) $(`#view-${v}`).hidden = v !== view;
  $("#new-btn").hidden = view === "setup";
  window.scrollTo({ top: 0 });
}


// set by main.js so modules can trigger a full re-render without importing main (no cycles)
export const hooks = { renderResult: null };

// the one motion in the app: the award seal settles when marks change (restartable)
export function settleSeal() {
  const seal = document.querySelector("#seal");
  seal.classList.remove("settle");
  void seal.offsetWidth;
  seal.classList.add("settle");
}
