// Examiner's Desk frontend. Untrusted text (student answers, model output) is only ever set via textContent.
import { initSetup } from "./setup.js";
import { regrade, renderLines, renderPage, updateRegradeBar } from "./lines.js";
import { onOverride, renderAttribution, renderConcepts, renderDiagrams, renderHistory, renderOverride, renderSeal } from "./sheet.js";
import { $, api, fmt, h, hooks, show, state } from "./util.js";

/* ---------------- result ---------------- */

function renderResult(r) {
  state.result = r;
  state.edits.clear();
  show("result");
  $("#result-title").focus({ preventScroll: true });  // move focus out of the hidden setup form
  $("#result-title").textContent = `Sheet ${r.sheet_id}`;
  const who = r.student_id_anon ? `Student ${r.student_id_anon}` : "Student not given";
  const secs = r.seconds ? ` · marked in ${fmt(r.seconds, 1)} s` : "";
  $("#result-meta").textContent = `${who} · ${new Date(r.created_at).toLocaleString()}${secs}`;
  $("#export-csv").href = `/api/results/${r.result_id}/export?fmt=csv`;
  $("#export-json").href = `/api/results/${r.result_id}/export?fmt=json`;
  renderReview(r.review);
  renderPage(r);
  renderLines(r.lines);
  renderDiagrams(r.diagrams || []);
  renderSeal(r);
  renderConcepts(r.text_grade.concept_table || []);
  renderAttribution(r.text_grade.token_attributions || []);
  renderHistory(r.history);
  $("#justification").textContent = r.justification.text;
  $("#justification-engine").textContent = `Written by ${r.justification.engine} from the evidence on this page only.`;
  renderOverride(r);
  updateRegradeBar();
}

function renderReview(review) {
  const el = $("#review-banner");
  el.hidden = !review.needed;
  el.replaceChildren();
  if (!review.needed) return;
  el.append(h("strong", { text: "Needs your review before the marks are final" }),
    h("ul", {}, review.reasons.map((x) => h("li", { text: x }))));
}

hooks.renderResult = renderResult;

/* ---------------- boot ---------------- */

async function boot() {
  initSetup();
  $("#show-boxes").addEventListener("change", (e) => $("#page-frame").classList.toggle("no-boxes", !e.target.checked));
  $("#regrade-btn").addEventListener("click", regrade);
  $("#discard-edits").addEventListener("click", () => renderResult(state.result));
  $("#override-form").addEventListener("submit", onOverride);
  $("#new-btn").addEventListener("click", () => { history.pushState({}, "", "/"); show("setup"); });
  window.addEventListener("popstate", route);
  try {
    const hlt = await api("/api/health");
    const b = $("#pipeline-badge");
    b.hidden = false;
    b.textContent = hlt.pipeline === "demo" ? "Demo pipeline" : "Full pipeline";
    b.classList.toggle("badge-demo", hlt.pipeline === "demo");
    b.title = hlt.pipeline === "demo" ? "Segmentation is real; OCR and scores are placeholders" : "All models loaded";
  } catch (e) { /* health is informational only */ }
  route();
}

async function route() {
  const rid = new URLSearchParams(location.search).get("r");
  if (!rid) { show("setup"); return; }
  show("loading");
  $("#loading-text").textContent = "Opening saved result";
  try { renderResult(await api(`/api/results/${encodeURIComponent(rid)}`)); }
  catch (e) { show("setup"); $("#form-error").hidden = false; $("#form-error").textContent = e.message; }
}

boot();
