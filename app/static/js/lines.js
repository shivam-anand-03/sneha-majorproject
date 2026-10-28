// Examiner's Desk frontend. Untrusted text (student answers, model output) is only ever set via textContent.
import { $, api, fileUrl, fmt, h, hooks, settleSeal, state, toast } from "./util.js";

export function renderPage(r) {
  const frame = $("#page-frame");
  frame.replaceChildren();
  if (!r.page_images || !r.page_images.length) { frame.append(h("p", { class: "caption", text: "No page image." })); return; }
  const img = h("img", { src: fileUrl(r.page_images[0]), alt: "Preprocessed answer sheet page" });
  frame.append(img);
  img.addEventListener("load", () => placeRegions(frame, img, r), { once: true });
  $("#page-caption").textContent = `${r.lines.length} lines and ${(r.diagrams || []).length} diagram regions found. ` +
    `Preprocessing: ${((r.config_snapshot || {}).preprocess_steps || []).join(", ") || "n/a"}.`;
}

function placeRegions(frame, img, r) {
  const [pw, ph] = r.page_size || [img.naturalWidth, img.naturalHeight];
  const box = (b) => ({ left: `${(b[0] / pw) * 100}%`, top: `${(b[1] / ph) * 100}%`,
    width: `${((b[2] - b[0]) / pw) * 100}%`, height: `${((b[3] - b[1]) / ph) * 100}%` });
  const onPage = (x) => (x.page === undefined || x.page === 0);
  r.lines.filter(onPage).forEach((l) => {
    const el = h("button", { class: `region${l.needs_review ? " region-review" : ""}`, type: "button",
      "aria-label": `Line ${l.idx + 1}`, "data-line": l.idx, onclick: () => focusLine(l.idx) });
    Object.assign(el.style, box(l.bbox));
    frame.append(el);
  });
  (r.diagrams || []).filter(onPage).forEach((d) => {
    const el = h("button", { class: "region region-diagram", type: "button", "aria-label": `Diagram ${d.idx + 1}`,
      onclick: () => document.getElementById(`diagram-${d.idx}`)?.scrollIntoView({ behavior: "smooth", block: "center" }) });
    Object.assign(el.style, box(d.bbox));
    frame.append(el);
  });
}

function focusLine(idx) {
  document.querySelectorAll(".is-active").forEach((e) => e.classList.remove("is-active"));
  const li = document.getElementById(`line-${idx}`);
  li?.classList.add("is-active");
  document.querySelector(`.region[data-line="${idx}"]`)?.classList.add("is-active");
  li?.scrollIntoView({ behavior: "smooth", block: "center" });
}

function lineWords(line) {
  const text = state.edits.has(line.idx) ? state.edits.get(line.idx) : line.text;
  if (!text.trim()) return [h("span", { class: "empty", text: "Nothing read on this line" })];
  // per-word confidences describe the OCR reading; after a teacher edit they no longer match the text
  const wordsMatch = (line.words || []).map((w) => w.word).join(" ") === text.trim().split(/\s+/).join(" ");
  if (state.edits.has(line.idx) || !wordsMatch) return [text];
  const out = [];
  line.words.forEach((w, i) => {
    if (i) out.push(" ");
    const conf = w.conf === null || w.conf === undefined ? "" : ` (confidence ${fmt(w.conf)})`;
    out.push(w.low ? h("span", { class: "w-low", title: `Low confidence${conf}`, text: w.word }) : w.word);
  });
  return out;
}

function lineMeta(line) {
  const meta = [];
  if (line.confidence !== null && line.confidence !== undefined) meta.push(h("span", { class: "num", text: `OCR confidence ${fmt(line.confidence)}` }));
  if (state.edits.has(line.idx)) meta.push(h("span", { class: "edited", text: "Corrected, not yet re-marked" }));
  else if ((line.review_reasons || []).includes("edited by teacher")) meta.push(h("span", { class: "edited", text: "Corrected by teacher" }));
  (line.review_reasons || []).filter((x) => x !== "edited by teacher").forEach((x) => meta.push(h("span", { class: "flag", text: x })));
  return meta;
}

function lineCandidates(line) {
  const alts = Object.entries(line.candidates || {}).filter(([, t]) => t && t !== line.text);
  if (!alts.length) return null;
  return h("ul", { class: "candidates", "aria-label": "Other readings" }, alts.map(([eng, t]) =>
    h("li", {}, h("span", { class: "eng", text: eng }),
      h("button", { type: "button", title: "Use this reading", text: t, onclick: () => setEdit(line, t) }))));
}

function renderLine(line) {
  const textEl = h("div", { class: "line-text", tabindex: "0", role: "button", "aria-label": `Edit line ${line.idx + 1}` }, lineWords(line));
  const startEdit = () => editLine(line, textEl);
  textEl.addEventListener("click", startEdit);
  textEl.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); startEdit(); } });
  return h("li", { class: "line", id: `line-${line.idx}` },
    h("span", { class: "line-no", text: String(line.idx + 1) }),
    h("div", {},
      h("img", { class: "line-crop", src: fileUrl(line.crop), alt: `Handwritten line ${line.idx + 1}`, loading: "lazy" }),
      textEl, lineCandidates(line), h("div", { class: "line-meta" }, lineMeta(line))));
}

export function renderLines(lines) {
  const ol = $("#lines");
  ol.replaceChildren(...lines.map(renderLine));
  if (!lines.length) ol.append(h("li", { class: "caption", text: "No handwriting lines were found on this sheet." }));
}

function editLine(line, textEl) {
  const current = state.edits.has(line.idx) ? state.edits.get(line.idx) : line.text;
  const ta = h("textarea", { class: "line-edit", rows: "2", "aria-label": `Correct line ${line.idx + 1}` });
  ta.value = current;
  textEl.replaceWith(ta);
  ta.focus();
  const commit = () => setEdit(line, ta.value);
  ta.addEventListener("blur", commit);
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ta.blur(); }
    if (e.key === "Escape") { ta.value = current; ta.blur(); }
  });
}

function setEdit(line, text) {
  if (text.trim() === line.text.trim()) state.edits.delete(line.idx);
  else state.edits.set(line.idx, text.trim());
  document.getElementById(`line-${line.idx}`).replaceWith(renderLine(line));
  updateRegradeBar();
}

export function updateRegradeBar() {
  const n = state.edits.size;
  $("#regrade-bar").hidden = n === 0;
  $("#regrade-count").textContent = `${n} line${n === 1 ? "" : "s"} corrected`;
}

export async function regrade() {
  const btn = $("#regrade-btn");
  btn.disabled = true;
  btn.textContent = "Re-marking";
  try {
    const lines = [...state.edits].map(([idx, text]) => ({ idx, text }));
    const r = await api(`/api/results/${state.result.result_id}/regrade`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lines }) });
    hooks.renderResult(r);
    settleSeal();
    toast("Marks updated from your corrections");
  } catch (e) {
    toast(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Re-mark with corrections";
  }
}

