// Examiner's Desk frontend. Untrusted text (student answers, model output) is only ever set via textContent.
import { $, api, hooks, show } from "./util.js";

/* ---------------- setup ---------------- */

export function initSetup() {
  const input = $("#sheet-input");
  const dz = $("#dropzone");
  const showName = () => {
    const f = input.files[0];
    $("#sheet-name").hidden = !f;
    if (f) $("#sheet-name").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  };
  input.addEventListener("change", () => { showName(); dz.classList.remove("is-invalid"); });
  ["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((e) => dz.addEventListener(e, () => dz.classList.remove("is-over")));
  dz.addEventListener("drop", (ev) => {
    ev.preventDefault();
    if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; showName(); }
  });
  const ref = document.querySelector("input[name=reference_diagram]");
  ref.addEventListener("change", () => {
    const f = ref.files[0];
    $("#ref-pick").classList.toggle("has-file", !!f);
    $("#ref-pick .file-pick-text").textContent = f ? f.name : "Optional. PNG or JPG of the expected diagram.";
  });
  $("#setup-form").addEventListener("submit", onSubmit);
}

function validateSetup(form) {
  const errs = [];
  const mark = (el, bad) => el.setAttribute("aria-invalid", bad ? "true" : "false");
  const ref = form.reference_text;
  mark(ref, !ref.value.trim());
  if (!ref.value.trim()) errs.push("Enter the model answer.");
  const tm = Number(form.total_marks.value);
  mark(form.total_marks, !(tm > 0));
  if (!(tm > 0)) errs.push("Total marks must be more than 0.");
  const noSheet = !$("#sheet-input").files.length;
  $("#dropzone").classList.toggle("is-invalid", noSheet);
  if (noSheet) errs.push("Add the student's answer sheet.");
  return errs;
}

const LOADING_STEPS = [
  "Reading the sheet: cleaning the page and finding lines",
  "Transcribing handwriting line by line",
  "Comparing the answer with the model answer",
  "Checking the diagram and writing the justification",
];

async function onSubmit(ev) {
  ev.preventDefault();
  if ($("#view-setup").hidden) return;  // a stray Enter must never re-submit from another view
  const form = ev.target;
  const errs = validateSetup(form);
  const errEl = $("#form-error");
  errEl.hidden = !errs.length;
  errEl.textContent = errs.join(" ");
  if (errs.length) return;
  show("loading");
  let step = 0;
  $("#loading-text").textContent = LOADING_STEPS[0];
  const timer = setInterval(() => { step = Math.min(step + 1, LOADING_STEPS.length - 1); $("#loading-text").textContent = LOADING_STEPS[step]; }, 4000);
  try {
    const result = await api("/api/evaluate", { method: "POST", body: new FormData(form) });
    history.pushState({}, "", `?r=${result.result_id}`);
    hooks.renderResult(result);
  } catch (e) {
    show("setup");
    errEl.hidden = false;
    errEl.textContent = e.message;
  } finally {
    clearInterval(timer);
  }
}

