"use strict";

/* Rally web UI — a small vanilla-JS SPA over the FastAPI backend.
   Two views: a gallery of jobs and a per-job detail workspace (review +
   editable segments). No build step, no framework. */

// --------------------------------------------------------------------------- //
// tiny helpers                                                                //
// --------------------------------------------------------------------------- //
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmtTime = (s) => {
  if (s == null || !isFinite(s)) return "00:00";
  s = Math.max(0, s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const mm = `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
  return h ? `${h}:${mm}` : mm;
};
const fmtDur = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${(s || 0).toFixed(1)}s`);
const fmtBytes = (b) => {
  if (!b) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b < 10 && i ? 1 : 0)} ${u[i]}`;
};

let toastTimer = null;
function toast(msg, kind = "info") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = "toast"), 3200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// --------------------------------------------------------------------------- //
// app state                                                                   //
// --------------------------------------------------------------------------- //
const state = {
  jobs: [],
  current: null,        // full public job object
  pollTimer: null,
  waveform: null,       // {duration, strikes[], segments[]}
  editSegments: null,   // [[start,end], ...] working copy while editing
  dirty: false,
};

// --------------------------------------------------------------------------- //
// gallery                                                                     //
// --------------------------------------------------------------------------- //
async function refreshGallery() {
  try {
    const data = await api("/api/jobs");
    state.jobs = data.jobs || [];
  } catch (e) {
    toast(`Could not load videos: ${e.message}`, "error");
    return;
  }
  renderGallery();
}

function renderGallery() {
  const grid = $("#jobGrid");
  grid.innerHTML = "";
  const jobs = state.jobs;
  $("#gallerySummary").textContent = jobs.length
    ? `${jobs.length} video${jobs.length > 1 ? "s" : ""}`
    : "No videos yet";
  if (!jobs.length) {
    grid.innerHTML = `<div class="empty">Upload a match recording to get started.</div>`;
    return;
  }
  for (const job of jobs) {
    const r = job.result || {};
    const card = document.createElement("article");
    card.className = "job-card";
    const thumb = job.media?.thumbnail
      ? `<img src="${job.media.thumbnail}" alt="" loading="lazy" />`
      : `<div class="thumb-placeholder">▚</div>`;
    const stateLabel = job.processing?.label || job.status;
    card.innerHTML = `
      <div class="job-thumb">${thumb}<span class="job-state ${statusClass(job)}">${stateLabel}</span></div>
      <div class="job-body">
        <div class="job-name" title="${escapeHtml(job.filename)}">${escapeHtml(job.filename)}</div>
        <div class="job-stats muted small">
          ${r.n_rallies != null ? `${r.n_rallies} rallies · ${Math.round((r.compression_ratio || 0) * 100)}% kept` : "not processed yet"}
        </div>
      </div>`;
    card.addEventListener("click", () => openDetail(job.id));
    grid.appendChild(card);
  }
}

function statusClass(job) {
  const s = job.processing?.stage || job.status;
  if (["complete", "ready"].includes(s)) return "ok";
  if (["failed", "error"].includes(s)) return "err";
  if (["no_output"].includes(s)) return "warn";
  if (["running", "queued", "starting", "audio", "visual", "deciding", "rendering", "probing", "writing", "waveform", "refining"].includes(s)) return "busy";
  return "";
}

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// --------------------------------------------------------------------------- //
// upload                                                                      //
// --------------------------------------------------------------------------- //
function setupUpload() {
  const form = $("#uploadForm");
  const input = $("#videoFile");
  const dz = $("#dropzone");

  input.addEventListener("change", () => {
    $("#fileName").textContent = input.files[0]?.name || "Choose or drop a video";
  });
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) { input.files = e.dataTransfer.files; $("#fileName").textContent = f.name; }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const file = input.files[0];
    if (!file) { toast("Pick a video first", "error"); return; }
    uploadJob(file);
  });
}

function numOrEmpty(id) {
  const v = $(id).value.trim();
  return v === "" ? null : v;
}

function uploadJob(file) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("detect_players", $("#detectPlayers").checked);
  fd.append("static_camera", $("#staticCamera").checked);
  fd.append("fast", $("#fast").checked);
  fd.append("hysteresis", $("#hysteresis").checked);
  fd.append("no_labels", $("#noLabels").checked);
  fd.append("ball_arbiter", $("#ballArbiter").checked);
  fd.append("court_auto", $("#courtAuto").checked);
  fd.append("run_now", "true");
  const opt = {
    analysis_fps: "#analysisFps", min_rally: "#minRally", skip_intro: "#skipIntro",
    gap: "#gap", serve_preroll: "#servePreroll", tail: "#tail",
  };
  for (const [k, id] of Object.entries(opt)) {
    const v = numOrEmpty(id);
    if (v != null) fd.append(k, v);
  }

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  const bar = $("#uploadProgress");
  bar.classList.remove("hidden");
  $("#uploadButton").disabled = true;
  xhr.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    $("#uploadBar").value = pct;
    $("#uploadPct").textContent = `${pct}%`;
    $("#uploadText").textContent = pct < 100 ? "Uploading…" : "Processing on server…";
  });
  xhr.addEventListener("load", () => {
    $("#uploadButton").disabled = false;
    bar.classList.add("hidden");
    $("#uploadBar").value = 0;
    if (xhr.status >= 200 && xhr.status < 300) {
      const job = JSON.parse(xhr.responseText);
      $("#uploadForm").reset();
      $("#fileName").textContent = "Choose or drop a video";
      toast("Uploaded — processing started");
      refreshGallery();
      openDetail(job.id);
    } else {
      let msg = "upload failed";
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
      toast(msg, "error");
    }
  });
  xhr.addEventListener("error", () => {
    $("#uploadButton").disabled = false;
    bar.classList.add("hidden");
    toast("Upload failed (network)", "error");
  });
  xhr.send(fd);
}

// --------------------------------------------------------------------------- //
// detail view                                                                 //
// --------------------------------------------------------------------------- //
function showView(which) {
  $("#galleryView").classList.toggle("hidden", which !== "gallery");
  $("#detailView").classList.toggle("hidden", which !== "detail");
}

async function openDetail(id) {
  stopPolling();
  state.current = null;
  state.waveform = null;
  state.editSegments = null;
  state.dirty = false;
  resetLabelingState();
  showView("detail");
  window.scrollTo(0, 0);
  try {
    await loadJob(id);
  } catch (e) {
    toast(`Could not open video: ${e.message}`, "error");
    backToGallery();
    return;
  }
  startPolling(id);
}

function backToGallery() {
  stopPolling();
  showView("gallery");
  state.current = null;
  refreshGallery();
}

async function loadJob(id) {
  const job = await api(`/api/jobs/${id}`);
  const first = !state.current || state.current.id !== id;
  state.current = job;
  renderDetail(job, first);
  const running = ["queued", "running"].includes(job.status);
  if (!running && (job.result || job.status === "complete" || job.status === "no_output")) {
    await loadWaveform(id);
  }
  return job;
}

function renderDetail(job, first) {
  $("#detailName").textContent = job.filename;
  const r = job.result || {};
  const info = r.info || {};
  const metaBits = [];
  if (r.total_seconds) metaBits.push(fmtDur(r.total_seconds));
  if (info.width) metaBits.push(`${info.width}×${info.height}`);
  if (r.channels_used?.length) metaBits.push(`channels: ${r.channels_used.join(", ")}`);
  if (r.n_strikes != null) metaBits.push(`${r.n_strikes} strikes`);
  $("#detailMeta").textContent = metaBits.join("  ·  ");

  // progress
  const p = job.processing || {};
  $("#procStage").textContent = p.label || job.status || "—";
  $("#procPct").textContent = `${p.percent || 0}%`;
  $("#procBar").value = p.percent || 0;
  $("#procBar").className = job.status === "failed" ? "err" : "";
  $("#procDetail").textContent = job.error || p.detail || "";

  // metrics
  $("#mPoints").textContent = r.n_rallies ?? 0;
  $("#mKept").textContent = r.kept_seconds ? fmtDur(r.kept_seconds) : "0s";
  $("#mRatio").textContent = `${Math.round((r.compression_ratio || 0) * 100)}%`;

  // media
  const m = job.media || {};
  if (first) {
    const ov = $("#originalVideo");
    ov.src = m.original || "";
  }
  const outState = $("#outputState");
  const out = $("#outputVideo");
  if (m.output) {
    if (out.dataset.src !== m.output) { out.src = m.output; out.dataset.src = m.output; }
    outState.classList.add("hidden");
  } else {
    out.removeAttribute("src");
    out.dataset.src = "";
    outState.classList.remove("hidden");
    outState.textContent = job.status === "failed"
      ? "Processing failed — see log"
      : (job.status === "no_output" ? (p.detail || "No rally video") : "Output appears after processing");
  }

  // downloads
  setLink($("#dlOutput"), m.output_download);
  setLink($("#dlJson"), m.metadata_download);

  // labeling section reflects the job's labeling status
  renderLabelingSection(job);
}

function setLink(el, href) {
  if (href) { el.href = href; el.classList.remove("disabled"); }
  else { el.removeAttribute("href"); el.classList.add("disabled"); }
}

// --------------------------------------------------------------------------- //
// polling                                                                     //
// --------------------------------------------------------------------------- //
function startPolling(id) {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    if (!state.current || state.current.id !== id) return stopPolling();
    try {
      const job = await loadJob(id);
      if (!["queued", "running"].includes(job.status)) stopPolling();
    } catch (_) { stopPolling(); }
  }, 1500);
}
function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

// --------------------------------------------------------------------------- //
// waveform + timeline                                                         //
// --------------------------------------------------------------------------- //
async function loadWaveform(id) {
  try {
    state.waveform = await api(`/api/jobs/${id}/waveform`);
  } catch (_) {
    state.waveform = { duration: (state.current.result || {}).total_seconds || 0, strikes: [], segments: [] };
  }
  if (state.editSegments === null) {
    state.editSegments = (state.waveform.segments || []).map((s) => [s.start, s.end]);
  }
  renderSegments();
  drawTimeline();
}

function drawTimeline() {
  const cv = $("#timeline");
  const wf = state.waveform;
  if (!cv || !wf) return;
  const dur = wf.duration || 0;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || cv.parentElement.clientWidth || 800;
  const H = 80;
  cv.width = W * dpr;
  cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  // track background
  ctx.fillStyle = "#f1efe9";
  ctx.fillRect(0, H * 0.32, W, H * 0.4);

  if (dur > 0) {
    const x = (t) => (t / dur) * W;
    // rally segments (use working copy if editing)
    const segs = state.editSegments || (wf.segments || []).map((s) => [s.start, s.end]);
    ctx.fillStyle = "rgba(46,125,90,0.55)";
    for (const [s, e] of segs) {
      const x0 = x(s);
      const w = Math.max(1.5, x(e) - x0);
      ctx.fillRect(x0, H * 0.22, w, H * 0.6);
    }
    // strikes
    ctx.fillStyle = "rgba(184,84,42,0.9)";
    for (const t of wf.strikes || []) ctx.fillRect(x(t), H * 0.08, 1, H * 0.24);

    // playhead
    const ov = $("#originalVideo");
    if (ov && isFinite(ov.currentTime)) {
      ctx.fillStyle = "#1c1c1c";
      ctx.fillRect(x(ov.currentTime), 0, 1.5, H);
    }
  }
  const segN = (state.editSegments || wf.segments || []).length;
  $("#timelineMeta").textContent =
    `${fmtDur(dur)} total · ${(wf.strikes || []).length} strikes · ${segN} rallies` +
    (dur ? " · click to seek" : "");
  cv._seekDur = dur;
}

function setupTimelineInteractions() {
  const cv = $("#timeline");
  cv.addEventListener("click", (e) => {
    const dur = cv._seekDur || 0;
    if (!dur) return;
    const rect = cv.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * dur;
    const ov = $("#originalVideo");
    ov.currentTime = t;
    ov.play().catch(() => {});
  });
  $("#originalVideo").addEventListener("timeupdate", () => {
    $("#origTime").textContent = fmtTime($("#originalVideo").currentTime);
    drawTimeline();
  });
  $("#outputVideo").addEventListener("timeupdate", () => {
    $("#outTime").textContent = fmtTime($("#outputVideo").currentTime);
  });
  window.addEventListener("resize", () => { if (state.waveform) drawTimeline(); });
}

// --------------------------------------------------------------------------- //
// segment table (editable)                                                    //
// --------------------------------------------------------------------------- //
function renderSegments() {
  const body = $("#segBody");
  body.innerHTML = "";
  const segs = state.editSegments || [];
  $("#segCount").textContent = `(${segs.length})`;
  segs.forEach(([s, e], i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${i + 1}</td>
      <td><input class="seg-in" data-i="${i}" data-f="0" type="number" step="0.1" min="0" value="${s.toFixed(2)}" /></td>
      <td><input class="seg-in" data-i="${i}" data-f="1" type="number" step="0.1" min="0" value="${e.toFixed(2)}" /></td>
      <td class="mono muted">${fmtDur(Math.max(0, e - s))}</td>
      <td><button class="row-del" data-i="${i}" title="Delete">✕</button></td>`;
    tr.addEventListener("click", (ev) => {
      if (ev.target.matches("input, button")) return;
      const ov = $("#originalVideo");
      ov.currentTime = s;
      ov.play().catch(() => {});
    });
    body.appendChild(tr);
  });

  $$(".seg-in").forEach((inp) =>
    inp.addEventListener("change", () => {
      const i = +inp.dataset.i, f = +inp.dataset.f;
      const v = parseFloat(inp.value);
      if (isFinite(v)) { state.editSegments[i][f] = v; markDirty(); drawTimeline(); renderSegments(); }
    }));
  $$(".row-del").forEach((btn) =>
    btn.addEventListener("click", () => {
      state.editSegments.splice(+btn.dataset.i, 1);
      markDirty(); renderSegments(); drawTimeline();
    }));
}

function markDirty() {
  state.dirty = true;
  $("#applySegments").disabled = false;
}

function setupSegmentEditor() {
  $("#addSegment").addEventListener("click", () => {
    if (!state.editSegments) state.editSegments = [];
    const ov = $("#originalVideo");
    const t = isFinite(ov.currentTime) ? ov.currentTime : 0;
    state.editSegments.push([+t.toFixed(2), +(t + 4).toFixed(2)]);
    state.editSegments.sort((a, b) => a[0] - b[0]);
    markDirty(); renderSegments(); drawTimeline();
  });
  $("#revertSegments").addEventListener("click", () => {
    const wf = state.waveform;
    state.editSegments = (wf?.segments || []).map((s) => [s.start, s.end]);
    state.dirty = false;
    $("#applySegments").disabled = true;
    renderSegments(); drawTimeline();
    toast("Reverted to detected segments");
  });
  $("#applySegments").addEventListener("click", applySegments);
}

async function applySegments() {
  if (!state.current) return;
  const id = state.current.id;
  const segs = (state.editSegments || [])
    .map(([s, e]) => [Math.max(0, s), e])
    .filter(([s, e]) => e - s > 0.05)
    .sort((a, b) => a[0] - b[0]);
  $("#applySegments").disabled = true;
  toast("Re-cutting output…");
  try {
    const job = await api(`/api/jobs/${id}/segments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ segments: segs }),
    });
    state.current = job;
    state.dirty = false;
    state.editSegments = null;
    renderDetail(job, false);
    await loadWaveform(id);
    // force reload of output video (cache-busted URL from server)
    const out = $("#outputVideo");
    if (job.media?.output) { out.src = job.media.output; out.dataset.src = job.media.output; out.load(); }
    toast("Output re-cut from your edits");
  } catch (e) {
    toast(`Re-cut failed: ${e.message}`, "error");
    $("#applySegments").disabled = false;
  }
}

// --------------------------------------------------------------------------- //
// detail actions                                                              //
// --------------------------------------------------------------------------- //
function setupDetailActions() {
  $("#backButton").addEventListener("click", backToGallery);
  $("#galleryButton").addEventListener("click", backToGallery);
  $("#reprocessButton").addEventListener("click", async () => {
    if (!state.current) return;
    try {
      await api(`/api/jobs/${state.current.id}/process`, { method: "POST" });
      state.editSegments = null;
      toast("Re-processing…");
      startPolling(state.current.id);
      loadJob(state.current.id);
    } catch (e) { toast(e.message, "error"); }
  });
  $("#deleteButton").addEventListener("click", async () => {
    if (!state.current) return;
    if (!confirm(`Delete "${state.current.filename}"? This cannot be undone.`)) return;
    try {
      await api(`/api/jobs/${state.current.id}`, { method: "DELETE" });
      toast("Deleted");
      backToGallery();
    } catch (e) { toast(e.message, "error"); }
  });
}

// --------------------------------------------------------------------------- //
// global status pill (gallery activity)                                       //
// --------------------------------------------------------------------------- //
setInterval(async () => {
  if (!$("#galleryView").classList.contains("hidden")) {
    await refreshGallery();
  }
  const busy = state.jobs.filter((j) => ["queued", "running"].includes(j.status)).length;
  const pill = $("#globalStatus");
  pill.textContent = busy ? `${busy} processing` : "Idle";
  pill.className = busy ? "pill busy" : "pill";
}, 3000);

// --------------------------------------------------------------------------- //
// labeling                                                                    //
// --------------------------------------------------------------------------- //
const lab = {
  mode: "player_identity",
  tasks: [],          // all tasks (both kinds)
  roster: [],
  labels: {},         // task_id -> {values,...}
  index: 0,           // index within the current mode's tasks
  draft: {},          // in-progress values for the current task
  genTimer: null,
};

function labModeTasks() {
  return lab.tasks.filter((t) => t.kind === lab.mode);
}

function setupLabeling() {
  $("#labGenerate").addEventListener("click", generateSamples);
  $$(".lab-tab").forEach((btn) =>
    btn.addEventListener("click", () => setLabMode(btn.dataset.mode)));
  $("#labPrev").addEventListener("click", () => moveTask(-1));
  $("#labNext").addEventListener("click", () => moveTask(1));
  $("#labSave").addEventListener("click", () => saveCurrentLabel(false));
  $("#labSaveNext").addEventListener("click", () => saveCurrentLabel(true));
  $("#serveNotes").addEventListener("input", () => { lab.draft.notes = $("#serveNotes").value; });

  // choice buttons (event delegation for both forms)
  $("#formPlayer").addEventListener("click", onChoiceClick);
  $("#formServe").addEventListener("click", onChoiceClick);
}

function onChoiceClick(e) {
  const btn = e.target.closest(".choice");
  if (!btn) return;
  const k = btn.dataset.k, v = btn.dataset.v;
  lab.draft[k] = lab.draft[k] === v ? undefined : v;   // toggle off if re-clicked
  reflectDraftSelection();
}

function reflectDraftSelection() {
  $$("#formPlayer .choice, #formServe .choice").forEach((b) =>
    b.classList.toggle("on", lab.draft[b.dataset.k] === b.dataset.v));
}

function resetLabelingState() {
  lab.tasks = []; lab.roster = []; lab.labels = {}; lab.index = 0; lab.draft = {};
  if (lab.genTimer) { clearInterval(lab.genTimer); lab.genTimer = null; }
  $("#labWorkspace").classList.add("hidden");
  $("#labStatus").classList.remove("hidden");
}

function renderLabelingSection(job) {
  const l = job.labeling || {};
  const status = $("#labStatus");
  if (l.status === "generating") {
    status.classList.remove("hidden");
    status.textContent = `Generating samples… ${l.detail || ""}`;
    $("#labGenerate").disabled = true;
    if (!lab.genTimer) lab.genTimer = setInterval(() => pollLabeling(job.id), 1500);
  } else {
    $("#labGenerate").disabled = false;
    if (lab.genTimer) { clearInterval(lab.genTimer); lab.genTimer = null; }
    if (l.status === "failed") {
      status.classList.remove("hidden");
      status.textContent = `Sample generation failed: ${l.error || "unknown error"}`;
    } else if (l.status === "ready") {
      if (!lab.tasks.length) loadLabelTasks(job.id);   // load once; don't clobber progress
    } else {
      status.classList.remove("hidden");
    }
  }
  setLink($("#labDownload"), `/api/jobs/${job.id}/labels/download`);
}

async function pollLabeling(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    if (state.current && state.current.id === id) state.current.labeling = job.labeling;
    const l = job.labeling || {};
    if (l.status === "generating") {
      $("#labStatus").textContent = `Generating samples… ${l.detail || ""}`;
    } else {
      clearInterval(lab.genTimer); lab.genTimer = null;
      renderLabelingSection(job);
    }
  } catch (_) { clearInterval(lab.genTimer); lab.genTimer = null; }
}

async function generateSamples() {
  if (!state.current) return;
  const id = state.current.id;
  $("#labGenerate").disabled = true;
  lab.tasks = [];               // force a fresh load when generation completes
  $("#labWorkspace").classList.add("hidden");
  try {
    await api(`/api/jobs/${id}/label-tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kinds: ["player_identity", "serve_motion"],
        max_items: Math.max(2, parseInt($("#labCount").value, 10) || 10),
        match_type: $("#labMatchType").value,
        regenerate: true,
      }),
    });
    toast("Generating samples…");
    $("#labStatus").classList.remove("hidden");
    $("#labStatus").textContent = "Generating samples…";
    if (!lab.genTimer) lab.genTimer = setInterval(() => pollLabeling(id), 1500);
  } catch (e) {
    toast(`Could not start: ${e.message}`, "error");
    $("#labGenerate").disabled = false;
  }
}

async function loadLabelTasks(id) {
  let data;
  try { data = await api(`/api/jobs/${id}/label-tasks`); }
  catch (_) { return; }
  lab.tasks = data.tasks || [];
  lab.roster = data.roster || [];
  lab.labels = data.labels || {};
  if (!lab.tasks.length) {
    $("#labStatus").classList.remove("hidden");
    $("#labStatus").textContent = "No samples were produced (no players detected / no rallies). Try a different video or count.";
    $("#labWorkspace").classList.add("hidden");
    return;
  }
  $("#labStatus").classList.add("hidden");
  $("#labWorkspace").classList.remove("hidden");
  $("#labPlayerCount").textContent = lab.tasks.filter((t) => t.kind === "player_identity").length;
  $("#labServeCount").textContent = lab.tasks.filter((t) => t.kind === "serve_motion").length;
  renderRoster();
  setLabMode(lab.mode, true);
}

function renderRoster() {
  const box = $("#rosterBox");
  box.classList.toggle("hidden", lab.mode !== "player_identity" || !lab.roster.length);
  const list = $("#rosterList");
  list.innerHTML = "";
  lab.roster.forEach((r) => {
    const wrap = document.createElement("span");
    wrap.className = "roster-item";
    const input = document.createElement("input");
    input.value = r.name || r.id;
    input.dataset.id = r.id;
    input.title = `${r.id} (${r.side || ""} ${r.col || ""})`.trim();
    input.addEventListener("change", saveRoster);
    wrap.append(document.createTextNode(r.id + ": "), input);
    list.appendChild(wrap);
  });
  renderPlayerChoices();
}

async function saveRoster() {
  const roster = $$("#rosterList input").map((inp) => {
    const r = lab.roster.find((x) => x.id === inp.dataset.id) || { id: inp.dataset.id };
    return { ...r, name: inp.value.trim() || inp.dataset.id };
  });
  lab.roster = roster;
  renderPlayerChoices();
  try { await api(`/api/jobs/${state.current.id}/roster`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ roster }),
  }); } catch (e) { toast(`Roster save failed: ${e.message}`, "error"); }
}

function renderPlayerChoices() {
  const html = lab.roster.map((r) =>
    `<button type="button" class="choice" data-k="player" data-v="${r.id}">${escapeHtml(r.name || r.id)}</button>`).join("");
  $("#playerChoices").innerHTML = html;
  $("#serverChoices").innerHTML = html + `<button type="button" class="choice" data-k="server" data-v="unknown">Unknown</button>`;
  // server choices use key "server"
  $$("#serverChoices .choice").forEach((b) => { if (b.dataset.v !== "unknown") b.dataset.k = "server"; });
  reflectDraftSelection();
}

function setLabMode(mode, force) {
  if (mode === lab.mode && !force) return;
  lab.mode = mode;
  lab.index = 0;
  $$(".lab-tab").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("#formPlayer").classList.toggle("hidden", mode !== "player_identity");
  $("#formServe").classList.toggle("hidden", mode !== "serve_motion");
  renderRoster();
  showTask();
}

function moveTask(delta) {
  const tasks = labModeTasks();
  if (!tasks.length) return;
  lab.index = Math.max(0, Math.min(tasks.length - 1, lab.index + delta));
  showTask();
}

function showTask() {
  const tasks = labModeTasks();
  const stage = $("#labStageMedia");
  updateLabProgress();
  if (!tasks.length) {
    stage.innerHTML = `<div class="muted">No ${lab.mode === "player_identity" ? "player crops" : "serve clips"} yet.</div>`;
    $("#labWhich").textContent = "0 / 0";
    $("#labThumbs").innerHTML = "";
    lab.draft = {};
    reflectDraftSelection();
    $("#serveNotes").value = "";
    return;
  }
  const t = tasks[lab.index];
  $("#labWhich").textContent = `${lab.index + 1} / ${tasks.length}`;
  if (t.media_type === "image") {
    stage.innerHTML = `<img src="${t.asset_url}" alt="player crop" />`;
  } else {
    stage.innerHTML = `<video src="${t.asset_url}" controls autoplay loop muted playsinline></video>`;
  }
  // load existing label (or suggestion) into the draft
  const saved = lab.labels[t.id];
  lab.draft = saved ? { ...saved.values } : {};
  if (!saved && t.kind === "player_identity" && t.suggested_player && lab.draft.player === undefined) {
    lab.draft.player = t.suggested_player;    // pre-select the position-based guess
  }
  $("#serveNotes").value = lab.draft.notes || "";
  reflectDraftSelection();
  renderThumbs(tasks);
}

function renderThumbs(tasks) {
  const strip = $("#labThumbs");
  strip.innerHTML = "";
  tasks.forEach((t, i) => {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "lab-thumb" + (i === lab.index ? " active" : "") + (lab.labels[t.id] ? " done" : "");
    if (t.media_type === "image") {
      el.innerHTML = `<img src="${t.asset_url}" loading="lazy" alt="" />`;
    } else {
      el.innerHTML = `<span class="clip-badge">▶</span>`;
    }
    el.addEventListener("click", () => { lab.index = i; showTask(); });
    strip.appendChild(el);
  });
}

function updateLabProgress() {
  const tasks = labModeTasks();
  const done = tasks.filter((t) => lab.labels[t.id]).length;
  $("#labProgress").textContent = `${done} / ${tasks.length} labeled`;
}

async function saveCurrentLabel(advance) {
  const tasks = labModeTasks();
  if (!tasks.length) return;
  const t = tasks[lab.index];
  const values = { ...lab.draft };
  if (lab.mode === "serve_motion") values.notes = $("#serveNotes").value || undefined;
  try {
    const res = await api(`/api/jobs/${state.current.id}/labels`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: t.id, kind: t.kind, values }),
    });
    lab.labels = res.labels;
    updateLabProgress();
    renderThumbs(tasks);
    if (advance && lab.index < tasks.length - 1) moveTask(1);
    else toast("Saved");
  } catch (e) { toast(`Save failed: ${e.message}`, "error"); }
}

// keyboard shortcuts while labeling
document.addEventListener("keydown", (e) => {
  if ($("#detailView").classList.contains("hidden")) return;
  if ($("#labWorkspace").classList.contains("hidden")) return;
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (e.key === "ArrowLeft") { moveTask(-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { moveTask(1); e.preventDefault(); }
  else if (e.key === "Enter") { saveCurrentLabel(true); e.preventDefault(); }
});

// --------------------------------------------------------------------------- //
// boot                                                                        //
// --------------------------------------------------------------------------- //
// Reflect optional-feature availability from the backend: disable the ball-arbiter
// toggle (and note why) when TrackNet weights / PyTorch aren't installed, so the user
// isn't surprised by a silent fallback mid-job.
async function loadCapabilities() {
  const box = $("#ballArbiter");
  if (!box) return;
  try {
    const caps = await api("/api/capabilities");
    const ba = caps.ball_arbiter || {};
    const label = box.closest("label");
    if (!ba.available) {
      box.checked = false;
      box.disabled = true;
      if (label) {
        label.classList.add("disabled");
        if (ba.hint) label.title = ba.hint;
        if (!label.querySelector(".cap-note")) {
          const note = document.createElement("span");
          note.className = "cap-note";
          note.textContent = " (weights not installed)";
          label.appendChild(note);
        }
      }
    }
  } catch (_) { /* capabilities are best-effort; leave the toggle as-is */ }
}

function init() {
  setupUpload();
  setupDetailActions();
  setupSegmentEditor();
  setupTimelineInteractions();
  setupLabeling();
  loadCapabilities();
  refreshGallery();
}
document.addEventListener("DOMContentLoaded", init);
