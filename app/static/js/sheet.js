// Examiner's Desk frontend. Untrusted text (student answers, model output) is only ever set via textContent.
import { $, api, fileUrl, fmt, fmtMarks, h, settleSeal, state, toast } from "./util.js";

/* ---------------- diagrams ---------------- */

function rubricTable(vlm) {
  const rows = ["components", "labels", "arrows", "structure"].filter((k) => vlm[k]).map((k) =>
    h("tr", {}, h("td", { text: k[0].toUpperCase() + k.slice(1) }),
      h("td", {}, h("span", { text: vlm[k].reason || "" }), h("div", { class: "meter" }, meterFill(vlm[k].score))),
      h("td", { class: "num", text: fmt(vlm[k].score) })));
  return h("table", { class: "rubric" }, h("tbody", {}, rows));
}

function meterFill(score) {
  const i = h("i");
  i.style.width = `${Math.max(0, Math.min(1, score || 0)) * 100}%`;
  return i;
}

function renderDiagram(d) {
  const figs = [h("figure", {}, h("img", { src: fileUrl(d.crop), alt: "Student diagram" }), h("figcaption", { text: "Student" }))];
  if (d.ref) figs.push(h("figure", {}, h("img", { src: fileUrl(d.ref), alt: "Reference diagram" }), h("figcaption", { text: "Reference" })));
  if (d.gradcam) figs.push(h("figure", {}, h("img", { src: fileUrl(d.gradcam), alt: "Regions that drove the similarity score" }),
    h("figcaption", { text: "Where similarity came from" })));
  const sim = d.sim ? Object.entries(d.sim).map(([k, v]) => `${k.toUpperCase()} ${fmt(v)}`).join(" · ") : null;
  const lines = [];
  if (sim) lines.push(h("p", { class: "caption num", text: `Image similarity ${sim} → blended ${fmt(d.sim_blend)} → transformed ${fmt(d.sim_transformed)}` }));
  if (d.vlm) {
    lines.push(rubricTable(d.vlm));
    if ((d.vlm.missing_elements || []).length) lines.push(h("p", { class: "caption", text: `Missing: ${d.vlm.missing_elements.join(", ")}` }));
  }
  (d.reasons || []).forEach((x) => lines.push(h("p", { class: "caption", text: x })));
  return h("section", { class: "diagram", id: `diagram-${d.idx}` },
    h("div", { class: "panel-head" }, h("h2", { class: "section-head", text: `Diagram ${d.idx + 1}` }),
      h("span", { class: "num", text: d.score === null || d.score === undefined ? "Not scored" : `Score ${fmt(d.score)}` })),
    h("div", { class: "diagram-figs" }, figs), lines);
}

export function renderDiagrams(diagrams) {
  $("#diagrams").replaceChildren(...diagrams.map(renderDiagram));
}

/* ---------------- marking sheet ---------------- */

export function renderSeal(r) {
  const s = r.score;
  const ov = r.override;
  const shown = ov ? ov.marks : s.marks;
  const sub = ov
    ? h("div", { class: "seal-sub" }, "Overridden by ", ov.teacher, ". Model awarded ", h("s", { text: fmtMarks(s.marks) }), ".")
    : h("div", { class: "seal-sub", text: r.review.needed ? "Provisional until reviewed" : "Model award" });
  $("#seal").replaceChildren(h("div", { class: "seal-label", text: "Marks awarded" }),
    h("div", { class: "seal-figure" }, fmtMarks(shown), h("small", { text: ` / ${fmtMarks(s.total_marks)}` })), sub);
  const tg = r.text_grade;
  const rows = [
    ["Text score", fmt(s.text_score)], [`  similarity`, fmt(tg.features.embed_sim)], [`  grader model`, fmt(tg.features.cross_encoder)],
    [`  key-point coverage`, fmt(tg.features.concept_coverage)],
    ["Diagram score", s.diagram_score === null || s.diagram_score === undefined ? "none" : fmt(s.diagram_score)],
    ["Weights text / diagram", `${fmt(s.weights.text)} / ${fmt(s.weights.diagram)}`],
    ["Unrounded marks", fmt(s.raw_marks)],
    ["History used in marks", s.history_used ? `yes (expected ${fmt(s.history_expected)})` : "no"],
  ];
  $("#breakdown").replaceChildren(...rows.flatMap(([k, v]) => [
    h("dt", { class: k.startsWith("  ") ? "sub" : "", text: k.trim() }), h("dd", { text: v })]));
}

const STATUS = { matched: ["chip-ok", "✓ Matched"], partial: ["chip-part", "½ Partly"], missing: ["chip-miss", "✗ Missing"] };

export function renderConcepts(rows) {
  const tb = $("#concepts");
  if (!rows.length) { tb.replaceChildren(h("tr", {}, h("td", { colspan: "3", class: "caption", text: "No key points extracted." }))); return; }
  tb.replaceChildren(...rows.map((r) => {
    const [cls, label] = STATUS[r.status] || ["", r.status];
    return h("tr", {}, h("td", { text: r.concept }),
      h("td", {}, h("span", { class: `chip ${cls}`, text: label }), r.method ? h("span", { class: "method", text: r.method }) : null),
      h("td", { text: r.match || "" }));
  }));
}

const ATTR_MIN = 0.15;
const ATTR_STRONG = 0.5;

export function renderAttribution(attrs) {
  const p = $("#attribution");
  if (!attrs.length) { p.replaceChildren(h("span", { class: "caption", text: "No attribution available." })); return; }
  const max = Math.max(...attrs.map((a) => Math.abs(a.attr))) || 1;
  p.replaceChildren(...attrs.flatMap((a, i) => {
    const s = h("span", { text: a.word, title: `Effect on text score ${a.attr >= 0 ? "+" : ""}${fmt(a.attr, 3)}` });
    const k = Math.abs(a.attr) / max;
    if (k >= ATTR_MIN) {  // tint scales with strength; only strong effects get the underline
      s.style.background = `color-mix(in srgb, var(${a.attr > 0 ? "--ok-wash" : "--miss-wash"}) ${Math.round(30 + 70 * k)}%, transparent)`;
      if (k >= ATTR_STRONG) s.style.boxShadow = `inset 0 -2px 0 var(${a.attr > 0 ? "--ok" : "--miss"})`;
    }
    return i ? [" ", s] : [s];
  }));
}

export function renderHistory(hist) {
  $("#history-panel").hidden = !hist;
  if (!hist) return;
  const rows = [["Expected marks from history", fmt(hist.expected)],
    ["Differs from expected", hist.anomaly ? `yes (z = ${fmt(hist.z)})` : "no"],
    ["Weak topics", (hist.weak_topics || []).join(", ") || "none"]];
  $("#history").replaceChildren(...rows.map(([k, v]) => h("div", { class: "history-row" }, h("span", { text: k }), h("span", { class: "num", text: v }))),
    (hist.feedback || []).length ? h("ul", { class: "history-feedback" }, hist.feedback.map((f) => h("li", { text: f }))) : null,
    h("p", { class: "caption", text: "History is shown for feedback only. It does not change the marks unless the score breakdown says so." }));
}

export function renderOverride(r) {
  const cur = $("#override-current");
  cur.replaceChildren();
  if (r.override) {
    cur.append(h("div", { class: "override-current" },
      h("strong", { text: `${fmtMarks(r.override.marks)} marks` }), ` recorded by ${r.override.teacher}: `, r.override.reason));
  }
  const old = r.superseded_override;
  if (!r.override && old) {
    cur.append(h("p", { class: "caption", text: `Earlier override of ${fmtMarks(old.marks)} by ${old.teacher} was set aside when the transcription was corrected. It stays in the audit log.` }));
  }
  const form = $("#override-form");
  form.marks.max = r.score.total_marks;
  form.marks.value = r.override ? r.override.marks : r.score.marks;
  $("#override-error").hidden = true;
}

export async function onOverride(ev) {
  ev.preventDefault();
  const f = ev.target;
  const err = $("#override-error");
  const marks = Number(f.marks.value);
  const total = state.result.score.total_marks;
  let msg = "";
  if (f.marks.value === "" || marks < 0 || marks > total) msg = `Marks must be between 0 and ${fmtMarks(total)}.`;
  else if (!f.teacher.value.trim()) msg = "Enter your name.";
  else if (f.reason.value.trim().length < 3) msg = "Give a short reason for the record.";
  err.hidden = !msg;
  err.textContent = msg;
  if (msg) return;
  const btn = f.querySelector("button[type=submit]");
  if (btn.disabled) return;  // one decision, one audit entry
  btn.disabled = true;
  try {
    const r = await api(`/api/results/${state.result.result_id}/override`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ marks, teacher: f.teacher.value.trim(), reason: f.reason.value.trim() }) });
    state.result = r;
    renderSeal(r);
    renderOverride(r);
    settleSeal();
    f.reason.value = "";
    toast("Override recorded in the audit log");
  } catch (e) {
    err.hidden = false;
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
}

