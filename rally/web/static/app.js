"use strict";

/* Rally web UI — a small vanilla-JS SPA over the FastAPI backend. */

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
  golden: [],
  goldenSignature: "",
  current: null,        // full public job object
  pollTimer: null,
  signalPollTimer: null,
  signalGeneration: 0,
  signalRevision: "",
  goldenPollTimer: null,
  goldenGeneration: 0,
  galleryRequest: 0,
  processingClockTimer: null,
  waveform: null,       // {duration, serves[], segments[]}
  editSegments: null,   // [[start,end], ...] working copy while editing
  detailJobId: null,    // requested detail, even while its GET is in flight
  detailGeneration: 0, // invalidates stale job/waveform responses after navigation/mutation
  videoTab: null,
  hadOutput: false,
  outputLayout: [],
  signalData: null,
  fullscreenTelemetryTrack: null,
  fullscreenTelemetryCue: null,
  webkitVideoFullscreen: false,
  activeUploads: new Map(),
  uploadSequence: 0,
};

function beginDetailGeneration(id) {
  state.detailGeneration += 1;
  state.detailJobId = id;
  return state.detailGeneration;
}

function detailRequestIsCurrent(id, generation) {
  return state.detailJobId === id && state.detailGeneration === generation;
}

// --------------------------------------------------------------------------- //
// gallery                                                                     //
// --------------------------------------------------------------------------- //
async function refreshGallery() {
  const request = ++state.galleryRequest;
  try {
    const jobs = [];
    let offset = 0, total = 0;
    do {
      const data = await api(`/api/jobs?limit=100&offset=${offset}`);
      const page = data.jobs || [];
      jobs.push(...page);
      total = data.total ?? jobs.length;
      offset += page.length;
      if (!page.length) break;
    } while (offset < total);
    if (request !== state.galleryRequest) return;
    state.jobs = jobs;
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
  if (["no_output", "cancelled"].includes(s)) return "warn";
  if (["running", "queued", "cancelling", "starting", "audio", "visual", "racket_actions", "pose", "serve", "deciding", "rendering", "probing", "writing", "waveform", "refining"].includes(s)) return "busy";
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
    const files = Array.from(input.files || []);
    $("#fileName").textContent = files.length > 1
      ? `${files.length} videos selected`
      : (files[0]?.name || "Choose or drop videos");
  });
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) {
      input.files = files;
      $("#fileName").textContent = files.length > 1
        ? `${files.length} videos selected` : files[0].name;
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const files = Array.from(input.files || []);
    if (!files.length) {
      alert("Please select at least one local video before uploading.");
      toast("Select a video first", "error");
      return;
    }
    for (const file of files) uploadJob(file, files.length === 1);
    input.value = "";
    $("#fileName").textContent = "Choose or drop videos";
  });
}

function numOrEmpty(id) {
  const v = $(id).value.trim();
  return v === "" ? null : v;
}

function paintUploadProgress() {
  const uploads = Array.from(state.activeUploads.values());
  const bar = $("#uploadProgress");
  if (!uploads.length) {
    bar.classList.add("hidden");
    $("#uploadBar").value = 0;
    $("#uploadPct").textContent = "0%";
    return;
  }
  const pct = Math.round(uploads.reduce((sum, upload) => sum + upload.percent, 0) / uploads.length);
  bar.classList.remove("hidden");
  $("#uploadBar").value = pct;
  $("#uploadPct").textContent = `${pct}%`;
  $("#uploadText").textContent = uploads.length === 1
    ? `Uploading ${uploads[0].name}…`
    : `Uploading ${uploads.length} videos concurrently…`;
}

function uploadJob(file, openWhenBatchFinishes = false) {
  const uploadId = ++state.uploadSequence;
  state.activeUploads.set(uploadId, { name: file.name, percent: 0 });
  paintUploadProgress();
  const fd = new FormData();
  fd.append("file", file);
  fd.append("fast", $("#fast").checked);
  fd.append("no_labels", $("#noLabels").checked);
  fd.append("run_now", "true");
  const opt = {
    pose_fps: "#poseFps", min_rally: "#minRally", skip_intro: "#skipIntro",
    gap: "#gap", start_buffer: "#startBuffer", end_buffer: "#endBuffer",
  };
  for (const [k, id] of Object.entries(opt)) {
    const v = numOrEmpty(id);
    if (v != null) fd.append(k, v);
  }

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.addEventListener("progress", (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded / e.total) * 100);
    const upload = state.activeUploads.get(uploadId);
    if (upload) upload.percent = pct;
    paintUploadProgress();
  });
  xhr.addEventListener("load", () => {
    state.activeUploads.delete(uploadId);
    paintUploadProgress();
    if (xhr.status >= 200 && xhr.status < 300) {
      const job = JSON.parse(xhr.responseText);
      toast(`${file.name} uploaded — processing started`);
      refreshGallery();
      if (openWhenBatchFinishes && state.activeUploads.size === 0) openDetail(job.id);
    } else {
      let msg = "upload failed";
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
      toast(msg, "error");
    }
  });
  xhr.addEventListener("error", () => {
    state.activeUploads.delete(uploadId);
    paintUploadProgress();
    toast(`${file.name}: upload failed (network)`, "error");
  });
  xhr.addEventListener("abort", () => {
    state.activeUploads.delete(uploadId);
    paintUploadProgress();
  });
  xhr.send(fd);
}

// --------------------------------------------------------------------------- //
// detail view                                                                 //
// --------------------------------------------------------------------------- //
function showView(which) {
  if (which !== "signals" && which !== "player-signals") stopSignalPolling();
  $("#galleryView").classList.toggle("hidden", which !== "gallery");
  $("#goldenView").classList.toggle("hidden", which !== "golden");
  $("#detailView").classList.toggle("hidden", which !== "detail");
  $("#signalsView").classList.toggle("hidden", which !== "signals");
  $("#playerSignalsView").classList.toggle("hidden", which !== "player-signals");
  $("#galleryButton").classList.toggle("active", which === "gallery");
  $("#goldenButton").classList.toggle("active", which === "golden");
}

async function openGolden() {
  stopGoldenPolling();
  const generation = state.goldenGeneration;
  stopPolling();
  stopProcessingClock();
  beginDetailGeneration(null);
  state.current = null;
  $("#originalVideo").pause();
  $("#outputVideo").pause();
  showView("golden");
  window.scrollTo(0, 0);
  await refreshGolden(generation);
  if (generation !== state.goldenGeneration || $("#goldenView").classList.contains("hidden")) return;
  scheduleGoldenPoll(generation);
}

function scheduleGoldenPoll(generation) {
  if (state.goldenPollTimer || generation !== state.goldenGeneration) return;
  state.goldenPollTimer = setTimeout(async () => {
    state.goldenPollTimer = null;
    await refreshGolden(generation);
    if (generation === state.goldenGeneration && !$("#goldenView").classList.contains("hidden")) {
      scheduleGoldenPoll(generation);
    }
  }, 10000);
}

function stopGoldenPolling() {
  clearTimeout(state.goldenPollTimer);
  state.goldenPollTimer = null;
  state.goldenGeneration += 1;
}

async function refreshGolden(generation = state.goldenGeneration) {
  if (generation !== state.goldenGeneration || $("#goldenView").classList.contains("hidden")) return;
  try {
    const data = await api("/api/golden");
    if (generation !== state.goldenGeneration || $("#goldenView").classList.contains("hidden")) return;
    const datasets = data.datasets || [];
    const signature = JSON.stringify(datasets);
    if (signature !== state.goldenSignature) {
      state.golden = datasets;
      state.goldenSignature = signature;
      renderGolden();
    }
  } catch (e) {
    toast(`Could not load golden datasets: ${e.message}`, "error");
  }
}

function renderGolden() {
  const grid = $("#goldenGrid");
  grid.innerHTML = "";
  $("#goldenSummary").textContent = `${state.golden.length} labeled dataset${state.golden.length === 1 ? "" : "s"}`;
  if (!state.golden.length) {
    grid.innerHTML = `<div class="empty card">No labeled golden datasets found.</div>`;
    return;
  }
  for (const dataset of state.golden) {
    const ready = Boolean(dataset.media?.output);
    const article = document.createElement("article");
    article.className = "golden-card card";
    article.innerHTML = `
      <div class="golden-card-head">
        <div>
          <h3>${escapeHtml(dataset.name)}</h3>
          <p class="muted small">${dataset.expected_points} labeled points${dataset.predicted_points == null ? "" : ` · ${dataset.predicted_points} detected`}</p>
        </div>
        <span class="job-state golden-state ${ready ? "ok" : "warn"}">${ready ? "Evaluation ready" : "Not evaluated"}</span>
      </div>
      <div class="golden-video-grid">
        <div class="golden-video">
          <div class="video-head"><h3>Golden input</h3><a class="ghost small" href="${dataset.media.input}" download>Download</a></div>
          <video src="${dataset.media.input}" controls playsinline preload="metadata"></video>
        </div>
        <div class="golden-video">
          <div class="video-head"><h3>Processed evaluation</h3>${dataset.media.metadata ? `<a class="ghost small" href="${dataset.media.metadata}" download>JSON</a>` : ""}</div>
          ${ready
            ? `<video src="${dataset.media.output}" controls playsinline preload="metadata"></video>`
            : `<div class="golden-missing">No retained evaluation video yet.<br><span>Run the golden evaluator with artifact output enabled.</span></div>`}
        </div>
      </div>
      <div class="golden-links"><a href="${dataset.media.ground_truth}" download>Ground truth: ${escapeHtml(dataset.annotation)}</a></div>`;
    grid.appendChild(article);
  }
}

async function openDetail(id) {
  stopGoldenPolling();
  stopPolling();
  stopProcessingClock();
  const generation = beginDetailGeneration(id);
  state.current = null;
  state.waveform = null;
  state.editSegments = null;
  state.videoTab = null;
  state.hadOutput = false;
  state.outputLayout = [];
  for (const video of [$("#originalVideo"), $("#outputVideo")]) {
    video.pause();
    video.removeAttribute("src");
    video.dataset.src = "";
    video.load();
  }
  resetLabelingState();
  showView("detail");
  window.scrollTo(0, 0);
  try {
    const job = await loadJob(id, generation);
    if (!job) return;
  } catch (e) {
    if (!detailRequestIsCurrent(id, generation)) return;
    toast(`Could not open video: ${e.message}`, "error");
    backToGallery();
    return;
  }
  startPolling(id, generation);
}

function backToGallery() {
  stopGoldenPolling();
  stopPolling();
  stopProcessingClock();
  beginDetailGeneration(null);
  showView("gallery");
  state.current = null;
  $("#originalVideo").pause();
  $("#outputVideo").pause();
  window.history.replaceState({}, "", "/");
  refreshGallery();
}

function selectVideoTab(tab) {
  state.videoTab = tab === "original" ? "original" : "processed";
  const original = state.videoTab === "original";
  $("#processedPanel").classList.toggle("hidden", original);
  $("#originalPanel").classList.toggle("hidden", !original);
  $("#processedTab").classList.toggle("active", !original);
  $("#originalTab").classList.toggle("active", original);
  $("#processedTab").setAttribute("aria-selected", String(!original));
  $("#originalTab").setAttribute("aria-selected", String(original));
  if (original) {
    $("#outputVideo").pause();
    const video = $("#originalVideo");
    const pending = video.dataset.pendingSrc || "";
    if (pending && video.dataset.src !== pending) {
      video.src = pending;
      video.dataset.src = pending;
      video.load();
    }
  } else {
    $("#originalVideo").pause();
  }
}

async function loadJob(id, generation = state.detailGeneration) {
  const job = await api(`/api/jobs/${id}`);
  if (!detailRequestIsCurrent(id, generation)) return null;
  state.current = job;
  renderDetail(job);
  const running = ["queued", "running"].includes(job.status);
  if (!running && (job.result || job.status === "complete" || job.status === "no_output")) {
    await loadWaveform(id, generation);
  }
  return detailRequestIsCurrent(id, generation) ? job : null;
}

function renderDetail(job) {
  $("#detailName").textContent = job.filename;
  const r = job.result || {};
  const info = r.info || {};
  const metaBits = [];
  if (r.total_seconds) metaBits.push(fmtDur(r.total_seconds));
  if (info.width) metaBits.push(`${info.width}×${info.height}`);
  if (r.channels_used?.length) metaBits.push(`channels: ${r.channels_used.join(", ")}`);
  if (r.n_serves != null) metaBits.push(`${r.n_serves} visual serves`);
  $("#detailMeta").textContent = metaBits.join("  ·  ");

  // progress
  const p = job.processing || {};
  $("#procStage").textContent = p.label || job.status || "—";
  $("#procPct").textContent = `${p.percent || 0}%`;
  $("#procBar").value = p.percent || 0;
  $("#procBar").className = job.status === "failed" ? "err" : "";
  $("#procDetail").textContent = job.error || p.detail || "";
  const processing = ["queued", "running"].includes(job.status);
  const stopButton = $("#stopProcessingButton");
  stopButton.classList.toggle("hidden", !processing);
  stopButton.disabled = Boolean(job.cancel_requested || p.stage === "cancelling");
  stopButton.textContent = stopButton.disabled ? "Stopping…" : "Stop processing";
  $("#reprocessButton").disabled = processing;
  $("#addSegment").disabled = processing;
  $("#revertSegments").disabled = processing;
  if (processing) $("#applySegments").disabled = true;

  // metrics
  $("#mPoints").textContent = r.n_rallies ?? 0;
  $("#mKept").textContent = r.kept_seconds ? fmtDur(r.kept_seconds) : "0s";
  $("#mRatio").textContent = `${Math.round((r.compression_ratio || 0) * 100)}%`;

  // media
  const m = job.media || {};
  const ov = $("#originalVideo");
  const originalUrl = m.original || "";
  if (ov.dataset.pendingSrc !== originalUrl) {
    ov.dataset.pendingSrc = originalUrl;
    if (state.videoTab !== "original") {
      ov.pause();
      ov.removeAttribute("src");
      ov.dataset.src = "";
      ov.load();
    }
  }
  const outState = $("#outputState");
  const out = $("#outputVideo");
  const gainedOutput = !state.hadOutput && Boolean(m.output);
  state.hadOutput = Boolean(m.output);
  state.outputLayout = Array.isArray(r.output_layout) ? r.output_layout : [];
  if (m.output) {
    if (out.dataset.src !== m.output) { out.src = m.output; out.dataset.src = m.output; }
    outState.classList.add("hidden");
    $("#playerHitLayer").classList.remove("hidden");
    $("#pointTelemetry").classList.remove("hidden");
    $("#pointOutcome").classList.remove("hidden");
  } else {
    if (out.dataset.src || out.hasAttribute("src")) {
      out.pause();
      out.removeAttribute("src");
      out.load();
    }
    out.dataset.src = "";
    outState.classList.remove("hidden");
    const preview = $("#processingPreview");
    if (m.thumbnail) {
      if (preview.dataset.src !== m.thumbnail) {
        preview.src = m.thumbnail;
        preview.dataset.src = m.thumbnail;
      }
      preview.classList.remove("hidden");
    } else {
      preview.removeAttribute("src");
      preview.dataset.src = "";
      preview.classList.add("hidden");
    }
    const terminalState = p.stage || job.status;
    const isProcessing = ["queued", "running"].includes(job.status);
    $("#processingStatus").textContent = terminalState === "failed"
      ? "Processing failed — see log"
      : (terminalState === "cancelled" ? "Processing stopped"
        : (terminalState === "no_output" ? (p.detail || "No rally video")
          : (isProcessing ? (p.label || "Processing video") : "Output appears after processing")));
    const pct = Math.max(0, Math.min(100, Number(p.percent) || 0));
    $("#processingPercent").textContent = `${Math.round(pct)}%`;
    $("#processingBar").value = pct;
    $("#playerHitLayer").classList.add("hidden");
    $("#pointTelemetry").classList.add("hidden");
    $("#pointOutcome").classList.add("hidden");
  }
  updateProcessingClock(job);
  if (state.videoTab === null || gainedOutput) selectVideoTab("processed");
  else selectVideoTab(state.videoTab);
  updateOutputOverlay();

  // downloads
  setLink($("#dlOutput"), m.output_download);
  setLink($("#dlJson"), m.metadata_download);
  const signalsAvailable = Boolean(job.result || job.signals_live
    || ["queued", "running"].includes(job.status));
  setLink($("#inspectSignals"), signalsAvailable ? `/jobs/${job.id}/signals` : null);

  // labeling section reflects the job's labeling status
  renderMatchSetup(job);
  renderLabelingSection(job);
}

function processingStartMs(job) {
  const progress = Array.isArray(job?.progress) ? job.progress : [];
  // Start at submission so the clock includes queue time and never jumps backwards when
  // the worker changes the state from queued to running.
  for (let i = progress.length - 1; i >= 0; i -= 1) {
    if (progress[i]?.message === "queued for processing") {
      const value = Date.parse(progress[i].at);
      if (Number.isFinite(value)) return value;
    }
  }
  for (let i = progress.length - 1; i >= 0; i -= 1) {
    if (progress[i]?.message === "processing started") {
      const value = Date.parse(progress[i].at);
      if (Number.isFinite(value)) return value;
    }
  }
  return null;
}

function paintProcessingClock(job = state.current) {
  const running = ["queued", "running"].includes(job?.status);
  const started = processingStartMs(job);
  const elapsed = running && started != null ? Math.max(0, (Date.now() - started) / 1000) : 0;
  $("#processingElapsed").textContent = running ? `Elapsed ${fmtTime(elapsed)}` : "Processing stopped";
}

function updateProcessingClock(job) {
  paintProcessingClock(job);
  const running = ["queued", "running"].includes(job?.status);
  if (running && !state.processingClockTimer) {
    state.processingClockTimer = setInterval(() => paintProcessingClock(), 1000);
  } else if (!running) {
    stopProcessingClock();
  }
}

function stopProcessingClock() {
  if (state.processingClockTimer) {
    clearInterval(state.processingClockTimer);
    state.processingClockTimer = null;
  }
}

function currentOutputPoint(time = $("#outputVideo").currentTime) {
  const layout = state.outputLayout || [];
  if (!layout.length || !isFinite(time)) return null;
  for (let i = 0; i < layout.length; i += 1) {
    const point = layout[i];
    const nextStart = layout[i + 1]?.output_start ?? Infinity;
    if (time >= point.output_start && time < nextStart) return { point, index: i };
  }
  if (time < layout[0].output_start) return { point: layout[0], index: 0 };
  return { point: layout[layout.length - 1], index: layout.length - 1 };
}

function pointOutcomeText(current) {
  const participants = current?.point?.participants || {};
  const point = current?.point || {};
  const termination = point.termination || {};
  const server = matchPlayerName(participants.server_id);
  const endpoint = String(termination.source || "presentation estimate").replaceAll("_", " ");
  const bits = [String(point.classification || "pose-confirmed rally").replaceAll("_", " ")];
  bits.push(`Endpoint: ${endpoint}${termination.confidence ? ` (${termination.confidence})` : ""}`);
  if (server) bits.push(`Serve: ${server}`);
  bits.push("No bounce, line, let, or service outcome inferred");
  return bits.join(" · ");
}

function updateOutputOverlay() {
  const current = currentOutputPoint();
  $("#currentPoint").textContent = current ? `Point ${current.index + 1}` : "Point —";
  const outcome = $("#pointOutcome");
  outcome.textContent = pointOutcomeText(current);
  outcome.classList.remove("hidden");
  outcome.classList.remove("unknown");
  const actions = current?.point?.actions || [];
  const actionCounts = actions.reduce((counts, action) => {
    const name = action.action || "action";
    counts[name] = (counts[name] || 0) + 1;
    return counts;
  }, {});
  const summary = Object.entries(actionCounts).map(([name, count]) => `${count} ${name}`).join(" · ");
  $("#pointActions").textContent = `Pose actions: ${summary || "unavailable"}`;
  updateNativeFullscreenCue();
}

function matchPlayerName(playerId) {
  if (!playerId) return null;
  const roster = state.current?.match?.roster || state.current?.result?.match?.roster || [];
  const player = roster.find((record) => record.id === playerId);
  return player?.name || playerId;
}

function matchTeamName(teamId) {
  if (!teamId) return null;
  const match = state.current?.match || state.current?.result?.match || {};
  const team = (match.teams || []).find((record) => record.id === teamId);
  const names = (team?.player_ids || []).map(matchPlayerName).filter(Boolean);
  return names.length ? names.join(" / ") : teamId;
}

function renderMatchSetup(job) {
  const match = job.match || job.result?.match || {};
  const roster = Array.isArray(match.roster) ? match.roster : [];
  const format = match.format;
  const confidence = Number(match.format_confidence);
  const detection = $("#matchDetection");
  if (format === "singles" || format === "doubles") {
    const pct = Number.isFinite(confidence) ? ` — ${Math.round(confidence * 100)}% confidence` : "";
    detection.textContent = `Detected automatically: ${format[0].toUpperCase()}${format.slice(1)}${pct}`;
  } else {
    detection.textContent = ["queued", "running"].includes(job.status)
      ? "Player detection is running…" : "Match format could not be determined";
  }
  const box = $("#matchRoster");
  box.replaceChildren();
  roster.forEach((player) => {
    const wrap = document.createElement("div");
    wrap.className = "match-player";
    const visual = document.createElement("div");
    visual.className = "match-player-visual";
    const gallery = document.createElement("div");
    gallery.className = "match-player-gallery";
    const thumbnails = Array.isArray(player.thumbnails) ? player.thumbnails : [];
    thumbnails.forEach((thumbnail, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "match-player-thumb";
      button.title = Number.isFinite(Number(thumbnail.time_s))
        ? `Verify ${player.id} at ${fmtTime(Number(thumbnail.time_s))}`
        : `Verify ${player.id}`;
      const image = document.createElement("img");
      image.src = thumbnail.url;
      image.loading = "lazy";
      image.alt = `${player.name || player.id} reference ${index + 1}`;
      button.appendChild(image);
      if (Number.isFinite(Number(thumbnail.time_s))) {
        button.addEventListener("click", () => seekOriginal(Number(thumbnail.time_s)));
      }
      gallery.appendChild(button);
    });
    if (!thumbnails.length) {
      const missing = document.createElement("span");
      missing.className = "match-player-thumb-missing";
      missing.textContent = "No clear crop";
      gallery.appendChild(missing);
    }
    const badge = document.createElement("a");
    badge.className = "match-player-id";
    badge.href = player.signal_gallery_url || `/jobs/${job.id}/signals/players/${player.id}`;
    badge.textContent = `${player.id} · all crops →`;
    visual.append(gallery, badge);
    const label = document.createElement("label");
    const team = player.team_id ? ` · ${player.team_id}` : "";
    label.append(document.createTextNode(`Player name${team}`));
    const input = document.createElement("input");
    input.type = "text";
    input.maxLength = 100;
    input.value = player.name || player.id;
    input.dataset.playerId = player.id;
    input.addEventListener("input", () => { $("#saveMatchRoster").disabled = false; });
    label.appendChild(input);
    wrap.append(visual, label);
    box.appendChild(wrap);
  });
  $("#saveMatchRoster").disabled = !roster.length;
}

function signalChip(text, stateClass = "") {
  const chip = document.createElement("span");
  chip.className = `signal-chip ${stateClass}`.trim();
  chip.textContent = text;
  return chip;
}

function renderSignalDecision(container, title, detail, stateClass = "") {
  const item = document.createElement("div");
  item.className = `signal-decision ${stateClass}`.trim();
  const heading = document.createElement("strong");
  heading.textContent = title;
  const body = document.createElement("p");
  body.textContent = detail || "No additional evidence recorded";
  item.append(heading, body);
  container.appendChild(item);
}

function renderSignals(data) {
  state.signalData = data;
  const job = data.job || {};
  $("#signalsTitle").textContent = `Pipeline signals · ${job.filename || "video"}`;
  const processing = job.processing || {};
  const stage = data.current_stage || processing.stage || "waiting";
  const percent = Math.max(0, Math.min(100, Number(processing.percent) || 0));
  $("#signalsMeta").textContent = `${fmtDur(Number(data.duration) || 0)} · ${job.status || "unknown"} · ${stage.replaceAll("_", " ")} · ${Math.round(percent)}%${data.live ? " · live" : ""}`;
  $("#signalsBackJob").href = `/jobs/${job.id}`;

  const rail = $("#signalStageRail");
  rail.replaceChildren();
  (data.stage_summary || []).forEach((stage) => {
    const card = document.createElement("div");
    card.className = `signal-stage ${stage.status || "recorded"}`;
    const name = document.createElement("strong");
    name.textContent = String(stage.name || "stage").replaceAll("_", " ");
    const status = document.createElement("span");
    status.textContent = stage.status || "recorded";
    if (stage.reason) card.title = stage.reason;
    card.append(name, status);
    rail.appendChild(card);
  });
  if (!(data.stage_summary || []).length) {
    rail.innerHTML = '<p class="muted small">Waiting for the first completed analysis stage…</p>';
  }

  const timeline = data.pose_timeline || {};
  const candidates = data.candidate_generation || {};
  const timelineView = $("#signalTimeline");
  timelineView.replaceChildren();
  if (Object.keys(timeline).length) {
    timelineView.append(
      signalChip(`${timeline.coarse_frames ?? 0} coarse frames`),
      signalChip(`${timeline.refined_frames ?? 0} refined frames`),
      signalChip(`${timeline.pose_records ?? 0} player poses`),
      signalChip(`${(timeline.actors || []).length} actors`),
      signalChip(`${timeline.frames_with_both_ends ?? 0} two-sided frames`),
      signalChip(`${(timeline.between_like_intervals || []).length} between-point runs`),
      signalChip(`${(timeline.engaged_like_intervals || []).length} engaged-play runs`),
      signalChip(`${candidates.count ?? 0} point hypotheses`, candidates.count ? "ok" : "bad"),
      signalChip(timeline.reused_player_detections ? "tracker boxes reused" : "box source unknown"),
      signalChip(timeline.audio_used ? "audio used" : "audio disabled", timeline.audio_used ? "bad" : "ok"),
      signalChip(timeline.ball_tracking_used ? "ball used" : "ball disabled", timeline.ball_tracking_used ? "bad" : "ok"),
    );
  } else {
    timelineView.append(signalChip("waiting for pose timeline"));
  }

  const players = $("#signalPlayers");
  players.replaceChildren();
  (data.players || []).forEach((player) => {
    const link = document.createElement("a");
    link.className = "signal-player-card";
    link.href = player.signal_gallery_url || `/jobs/${job.id}/signals/players/${encodeURIComponent(player.id)}`;
    const image = document.createElement("img");
    const images = player.inspection_images || player.thumbnails || [];
    if (images[0]?.url) image.src = images[0].url;
    image.alt = `${player.name || player.id} crop`;
    image.loading = "lazy";
    const text = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = player.name || player.id;
    const detail = document.createElement("span");
    detail.textContent = `${player.id} · ${player.team_id || "team unknown"} · ${images.length} retained crops`;
    text.append(name, detail);
    link.append(image, text);
    players.appendChild(link);
  });
  if (!(data.players || []).length) {
    players.innerHTML = '<p class="muted">No persistent player identities were published.</p>';
  }

  const serveStage = data.serve_pose || {};
  const serves = serveStage.observations || [];
  $("#signalServeSummary").textContent = `${serves.length} proposals · ${serves.filter((item) => item.pose_accepted).length} pose/formation accepted`;
  const serveGrid = $("#signalServes");
  serveGrid.replaceChildren();
  serves.forEach((serve) => {
    const card = document.createElement("article");
    card.className = `signal-serve-card ${serve.pose_accepted ? "accept" : "reject"}`;
    if (serve.image) {
      const image = document.createElement("img");
      image.src = serve.image;
      image.loading = "lazy";
      image.alt = `Serve proposal ${serve.index + 1}`;
      card.appendChild(image);
    } else {
      const missing = document.createElement("div");
      missing.className = "signal-image-placeholder";
      missing.textContent = "Annotated frame unavailable for this older result";
      card.appendChild(missing);
    }
    const body = document.createElement("div");
    body.className = "signal-serve-body";
    const head = document.createElement("div");
    head.className = "signal-serve-head";
    const title = document.createElement("strong");
    title.textContent = `Proposal ${serve.index + 1} · ${fmtTime(Number(serve.time) || 0)}`;
    const decision = signalChip(serve.pose_accepted ? "accepted" : "rejected",
      serve.pose_accepted ? "ok" : "bad");
    head.append(title, decision);
    const chips = document.createElement("div");
    chips.className = "signal-chip-row";
    chips.append(
      signalChip(`sequence ${serve.serve_sequence ? "yes" : "no"}`, serve.serve_sequence ? "ok" : "bad"),
      signalChip(`wrist rise ${Number(serve.wrist_rise_span || 0).toFixed(2)} torso`, Number(serve.wrist_rise_span) > 0 ? "ok" : "bad"),
      signalChip(`server load ${serve.server_load_frames}`),
      signalChip(`bent knee ${serve.knee_bend_frames}`),
      signalChip(`leg drive ${serve.leg_drive_frames}`),
      signalChip(`baseline ${serve.server_baseline_frames}`),
      signalChip(`opposed formation ${serve.opposed_formation_frames}`, serve.opposed_formation_frames >= 2 ? "ok" : "bad"),
      signalChip(`racket ${serve.racket_wrist_associated ? "observed" : "pose proxy"}`, serve.racket_wrist_associated ? "ok" : ""),
      signalChip(`server court x ${serve.server_court_x_m == null ? "unknown" : Number(serve.server_court_x_m).toFixed(2) + "m"}`),
      signalChip(`pose frames ${serve.pose_frames}/${serve.sampled_frames}`),
    );
    body.append(head, chips);
    card.appendChild(body);
    serveGrid.appendChild(card);
  });
  if (!serves.length) {
    serveGrid.innerHTML = '<p class="muted">This result predates persisted serve-pose observations or produced no proposals.</p>';
  }

  const actionStage = data.racket_actions || {};
  const actions = actionStage.actions || [];
  const decisions = actionStage.decisions || [];
  const acceptedActions = actions.filter((item) => item.accepted !== false);
  const rejectedActions = actions.filter((item) => item.accepted === false);
  $("#signalActionSummary").textContent = `${actionStage.raw_action_proposals ?? "—"} raw proposals → ${actions.length} stroke episodes → ${acceptedActions.length} sequence-accepted · ${decisions.filter((item) => item.accepted).length}/${decisions.length} rallies kept`;
  const actionGrid = $("#signalActions");
  actionGrid.replaceChildren();
  actions.forEach((action) => {
    const card = document.createElement("article");
    const accepted = action.accepted !== false;
    card.className = `signal-serve-card ${accepted ? "accept" : "reject"}`;
    if (action.image) {
      const image = document.createElement("img");
      image.src = action.image;
      image.loading = "lazy";
      image.alt = `${action.action || "stroke"} by ${action.actor_id || "player"}`;
      card.appendChild(image);
    }
    const body = document.createElement("div");
    body.className = "signal-serve-body";
    const head = document.createElement("div");
    head.className = "signal-serve-head";
    const title = document.createElement("strong");
    title.textContent = `${action.action || "stroke"} · ${action.actor_id || "unassigned"} · ${fmtTime(Number(action.time) || 0)}`;
    head.append(title,
      signalChip(accepted ? "sequence accepted" : "rejected", accepted ? "ok" : "bad"),
      signalChip(`${Math.round(Number(action.confidence || 0) * 100)}%`, accepted ? "ok" : "bad"));
    const chips = document.createElement("div");
    chips.className = "signal-chip-row";
    chips.append(
      signalChip(`${action.hand || "?"} hand`),
      signalChip(`backswing ${Number(action.backswing_body_lengths || 0).toFixed(2)}`),
      signalChip(`forward ${Number(action.forward_span_body_lengths || 0).toFixed(2)}`),
      signalChip(`follow ${Number(action.follow_through_body_lengths || 0).toFixed(2)}`),
      signalChip(`speed ${Number(action.forward_speed_body_lengths_s || 0).toFixed(2)} torso/s`),
      signalChip(`actor ${action.actor_end || "unknown"} end`),
      signalChip(`racket ${action.racket_wrist_associated ? "observed" : "pose proxy"}`, action.racket_wrist_associated ? "ok" : ""),
      signalChip(`${action.proposal_count || 1} local peak${Number(action.proposal_count || 1) === 1 ? "" : "s"}`),
      ...(action.rejection_reason ? [signalChip(String(action.rejection_reason).replaceAll("_", " "), "bad")] : []),
    );
    body.append(head, chips);
    card.appendChild(body);
    actionGrid.appendChild(card);
  });
  decisions.forEach((decision, index) => renderSignalDecision(
    actionGrid,
    `Point window ${index + 1} · ${String(decision.classification || (decision.accepted ? "accepted" : "rejected")).replaceAll("_", " ")}`,
    `${decision.reason || "no reason"} · ${decision.raw_action_count || 0} raw proposals · ${decision.action_count || 0} episodes · ${decision.accepted_action_count || 0} sequence-accepted · endpoint ${decision.endpoint_source ? String(decision.endpoint_source).replaceAll("_", " ") : "none"}${decision.endpoint_confidence ? ` (${decision.endpoint_confidence})` : ""}`,
    decision.accepted ? "accept" : "reject",
  ));
  decisions.forEach((decision, index) => {
    (decision.service_attempts || []).forEach((attempt) => renderSignalDecision(
      actionGrid,
      `Window ${index + 1} · service attempt ${Number(attempt.attempt_index) + 1} · ${fmtTime(Number(attempt.time))}`,
      `${String(attempt.disposition || "unresolved").replaceAll("_", " ")} · ${attempt.server_end || "unknown"} end / ${attempt.server_court_half || "unknown"} half · ${attempt.rule_limit || "ball outcome unavailable"}`,
      attempt.disposition === "playable_service_candidate" ? "accept" : "recorded",
    ));
    (decision.state_transitions || []).forEach((transition) => renderSignalDecision(
      actionGrid,
      `State · ${transition.from || "?"} → ${transition.to || "?"} · ${fmtTime(Number(transition.time))}`,
      transition.reason || "No transition evidence recorded",
      transition.to === "LIVE_POINT" ? "accept" : "recorded",
    ));
  });
  if (!actions.length && !decisions.length) renderSignalDecision(actionGrid, "No racket actions", actionStage.reason || "No pose-confirmed point windows were published");

  const endpoints = data.endpoints || {};
  const endpointList = $("#signalEndpoints");
  endpointList.replaceChildren();
  const semantic = endpoints.semantic_rule_endpoints || endpoints.records || [];
  const presentation = endpoints.visual_presentation_endpoints || [];
  semantic.forEach((item) => renderSignalDecision(
    endpointList,
    `Point ${Number(item.point_index) + 1} · ${item.rule_event || "rule endpoint"}`,
    `event ${fmtTime(Number(item.event_time))} · previous end ${fmtTime(Number(item.previous_end))}`,
    "accept",
  ));
  presentation.forEach((item) => renderSignalDecision(
    endpointList,
    `Point ${Number(item.point_index) + 1} · presentation trim`,
    `${fmtTime(Number(item.presentation_time))} · ${String(item.visual_dead_signal || item.semantic_rule_event || "player transition cue").replaceAll("_", " ")}${item.endpoint_confidence ? ` · ${item.endpoint_confidence} confidence` : ""}`,
    "recorded",
  ));
  const quality = data.quality_control || {};
  if (Object.keys(quality).length) renderSignalDecision(
    endpointList,
    `Output quality guard · ${quality.status || "recorded"}`,
    `${Math.round(Number(quality.retention_fraction || 0) * 100)}% retained before guard · ${Math.round(Number(quality.zero_post_serve_stroke_fraction || 0) * 100)}% without observed post-serve strokes · ${Number(quality.boundary_invariant_violations || 0)} boundary violations${quality.reason ? ` · ${quality.reason}` : ""}${(quality.warnings || []).length ? ` · ${quality.warnings.join(" · ")}` : ""}`,
    quality.status === "rejected" ? "reject" : "accept",
  );
  if (!semantic.length && !presentation.length) {
    (data.points || []).forEach((point, index) => {
      const termination = point.termination || {};
      renderSignalDecision(endpointList, `Point ${index + 1} · ${termination.rule_event || "unknown"}`,
        termination.evidence?.join(" · ") || "No terminal rule evidence");
    });
  }
  if (!endpointList.children.length) renderSignalDecision(endpointList, "No endpoint evidence", "No validated points were published");
  $("#signalRaw").textContent = JSON.stringify(data, null, 2);
}

async function openSignals(jobId) {
  stopGoldenPolling();
  stopPolling();
  stopProcessingClock();
  stopSignalPolling();
  const generation = state.signalGeneration;
  showView("signals");
  window.scrollTo(0, 0);
  try {
    const data = await api(`/api/jobs/${jobId}/signals`, { cache: "no-store" });
    if (generation !== state.signalGeneration || $("#signalsView").classList.contains("hidden")) return;
    renderSignals(data);
    state.signalRevision = signalRevision(data);
    scheduleSignalPoll(jobId, null, generation, data);
  } catch (error) {
    toast(`Could not load signals: ${error.message}`, "error");
  }
}

function renderPlayerSignals(data, jobId) {
    const player = data.player || {};
    $("#playerSignalsBack").href = `/jobs/${jobId}/signals`;
    $("#playerSignalsTitle").textContent = player.name || player.id || "Player";
    const images = player.inspection_images || player.thumbnails || [];
    const stage = data.current_stage ? ` · ${String(data.current_stage).replaceAll("_", " ")}` : "";
    $("#playerSignalsMeta").textContent = `${player.id || ""} · ${player.team_id || "team unknown"} · ${images.length} retained identity observations${stage}${data.live ? " · live" : ""}`;
    const gallery = $("#playerSignalGallery");
    gallery.replaceChildren();
    images.forEach((image, index) => {
      const figure = document.createElement("figure");
      figure.className = "player-signal-item";
      const link = document.createElement("a");
      const time = Number(image.time_s);
      link.href = `/jobs/${jobId}${Number.isFinite(time) ? `?seek=${encodeURIComponent(time)}` : ""}`;
      const element = document.createElement("img");
      element.src = image.url;
      element.loading = "lazy";
      element.alt = `${player.name || player.id} observation ${index + 1}`;
      link.appendChild(element);
      const caption = document.createElement("figcaption");
      const count = document.createElement("span");
      count.textContent = `#${index + 1}`;
      const timestamp = document.createElement("span");
      timestamp.textContent = Number.isFinite(time) ? fmtTime(time) : "time unknown";
      caption.append(count, timestamp);
      figure.append(link, caption);
      gallery.appendChild(figure);
    });
    $("#playerSignalEmpty").classList.toggle("hidden", Boolean(images.length));
}

async function openPlayerSignals(jobId, playerId) {
  stopGoldenPolling();
  stopPolling();
  stopProcessingClock();
  stopSignalPolling();
  const generation = state.signalGeneration;
  showView("player-signals");
  window.scrollTo(0, 0);
  try {
    const data = await api(`/api/jobs/${jobId}/signals/players/${encodeURIComponent(playerId)}`, { cache: "no-store" });
    if (generation !== state.signalGeneration || $("#playerSignalsView").classList.contains("hidden")) return;
    renderPlayerSignals(data, jobId);
    state.signalRevision = signalRevision(data);
    scheduleSignalPoll(jobId, playerId, generation, data);
  } catch (error) {
    toast(`Could not load player signals: ${error.message}`, "error");
  }
}

function stopSignalPolling() {
  clearTimeout(state.signalPollTimer);
  state.signalPollTimer = null;
  state.signalGeneration += 1;
  state.signalRevision = "";
}

function signalRevision(data) {
  const processing = data?.job?.processing || {};
  return JSON.stringify([
    data?.updated_at, data?.job?.status, data?.current_stage,
    processing.stage, processing.percent, processing.detail,
  ]);
}

function scheduleSignalPoll(jobId, playerId, generation, data) {
  const processing = ["queued", "running"].includes(data?.job?.status);
  if (state.signalPollTimer || generation !== state.signalGeneration) return;
  state.signalPollTimer = setTimeout(async () => {
    state.signalPollTimer = null;
    if (generation !== state.signalGeneration) return;
    const path = playerId
      ? `/api/jobs/${jobId}/signals/players/${encodeURIComponent(playerId)}`
      : `/api/jobs/${jobId}/signals`;
    try {
      const fresh = await api(path, { cache: "no-store" });
      if (generation !== state.signalGeneration) return;
      const revision = signalRevision(fresh);
      if (revision !== state.signalRevision) {
        if (playerId) renderPlayerSignals(fresh, jobId);
        else renderSignals(fresh);
        state.signalRevision = revision;
      }
      scheduleSignalPoll(jobId, playerId, generation, fresh);
    } catch (error) {
      if (generation !== state.signalGeneration) return;
      scheduleSignalPoll(jobId, playerId, generation, data);
    }
  }, processing ? 1500 : 5000);
}

async function saveMatchRoster() {
  const jobId = state.current?.id;
  if (!jobId) return;
  const roster = $$("#matchRoster input").map((input) => ({
    id: input.dataset.playerId,
    name: input.value.trim(),
  }));
  if (roster.some((player) => !player.name)) {
    toast("Every player needs a name", "error");
    return;
  }
  $("#saveMatchRoster").disabled = true;
  try {
    const response = await api(`/api/jobs/${jobId}/match`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ roster }),
    });
    if (!state.current || state.current.id !== jobId) return;
    state.current.match = response.match;
    if (state.current.result) state.current.result.match = response.match;
    renderMatchSetup(state.current);
    updateOutputOverlay();
    toast("Player names saved");
  } catch (e) {
    $("#saveMatchRoster").disabled = false;
    toast(`Could not save player names: ${e.message}`, "error");
  }
}

function seekOutputPoint(delta) {
  const layout = state.outputLayout || [];
  if (!layout.length) return;
  const current = currentOutputPoint();
  const from = current ? current.index : 0;
  const target = Math.max(0, Math.min(layout.length - 1, from + delta));
  const video = $("#outputVideo");
  video.currentTime = Math.max(0, Number(layout[target].output_start) || 0);
  video.play().catch(() => {});
  updateOutputOverlay();
}

function seekOriginal(time) {
  selectVideoTab("original");
  const video = $("#originalVideo");
  const apply = () => {
    video.currentTime = Math.max(0, time);
    video.play().catch(() => {});
  };
  if (video.readyState >= 1) apply();
  else video.addEventListener("loadedmetadata", apply, { once: true });
}

function setLink(el, href) {
  if (href) { el.href = href; el.classList.remove("disabled"); }
  else { el.removeAttribute("href"); el.classList.add("disabled"); }
}

// --------------------------------------------------------------------------- //
// polling                                                                     //
// --------------------------------------------------------------------------- //
function startPolling(id, generation = state.detailGeneration) {
  stopPolling();
  const tick = async () => {
    if (!detailRequestIsCurrent(id, generation) || !state.current || state.current.id !== id) {
      return stopPolling();
    }
    try {
      const job = await loadJob(id, generation);
      if (!job) return;
      if (!["queued", "running"].includes(job.status)) return stopPolling();
    } catch (_) {
      // A transient request failure must not permanently freeze a running job view.
    }
    if (detailRequestIsCurrent(id, generation)) {
      state.pollTimer = setTimeout(tick, 1500);
    }
  };
  state.pollTimer = setTimeout(tick, 1500);
}
function stopPolling() {
  if (state.pollTimer) { clearTimeout(state.pollTimer); state.pollTimer = null; }
}

// --------------------------------------------------------------------------- //
// waveform + timeline                                                         //
// --------------------------------------------------------------------------- //
async function loadWaveform(id, generation = state.detailGeneration) {
  let waveform;
  try {
    waveform = await api(`/api/jobs/${id}/waveform`);
  } catch (_) {
    waveform = { duration: (state.current?.result || {}).total_seconds || 0, serves: [], segments: [] };
  }
  if (!detailRequestIsCurrent(id, generation)) return false;
  state.waveform = waveform;
  if (state.editSegments === null) {
    state.editSegments = (state.waveform.segments || []).map((s) => [s.start, s.end]);
  }
  renderSegments();
  drawTimeline();
  return true;
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
    // visually confirmed service actions
    ctx.fillStyle = "rgba(184,84,42,0.9)";
    for (const t of wf.serves || []) ctx.fillRect(x(t), H * 0.08, 1, H * 0.24);

    // playhead
    const ov = $("#originalVideo");
    if (ov && isFinite(ov.currentTime)) {
      ctx.fillStyle = "#1c1c1c";
      ctx.fillRect(x(ov.currentTime), 0, 1.5, H);
    }
  }
  const segN = (state.editSegments || wf.segments || []).length;
  $("#timelineMeta").textContent =
    `${fmtDur(dur)} total · ${(wf.serves || []).length} visual serves · ${segN} rallies` +
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
    seekOriginal(t);
  });
  $("#originalVideo").addEventListener("timeupdate", () => {
    $("#origTime").textContent = fmtTime($("#originalVideo").currentTime);
    drawTimeline();
  });
  $("#outputVideo").addEventListener("timeupdate", () => {
    $("#outTime").textContent = fmtTime($("#outputVideo").currentTime);
    updateOutputOverlay();
  });
  ["loadedmetadata", "seeked", "pause"].forEach((eventName) => {
    $("#outputVideo").addEventListener(eventName, updateOutputOverlay);
  });
  window.addEventListener("resize", () => {
    if (state.waveform) drawTimeline();
  });
}

function updateNativeFullscreenCue() {
  const cue = state.fullscreenTelemetryCue;
  if (!cue) return;
  cue.text = [
    $("#pointOutcome").textContent,
    $("#pointActions").textContent,
  ].filter(Boolean).join("\n");
}

function syncNativeFullscreenOverlay() {
  const output = $("#outputVideo");
  const activeElement = document.fullscreenElement || document.webkitFullscreenElement || null;
  const active = activeElement === output || state.webkitVideoFullscreen;
  if (state.fullscreenTelemetryTrack) {
    state.fullscreenTelemetryTrack.mode = active ? "showing" : "disabled";
  }
  updateNativeFullscreenCue();
}

function setupNativeFullscreenOverlay(output) {
  if (typeof window.VTTCue !== "function" || typeof output.addTextTrack !== "function") return;
  try {
    const track = output.addTextTrack("captions", "Rally point result", "en");
    const cue = new VTTCue(0, 24 * 60 * 60, "");
    cue.line = 1;
    cue.position = 98;
    cue.align = "end";
    cue.size = 58;
    track.addCue(cue);
    track.mode = "disabled";
    state.fullscreenTelemetryTrack = track;
    state.fullscreenTelemetryCue = cue;
    document.addEventListener("fullscreenchange", syncNativeFullscreenOverlay);
    document.addEventListener("webkitfullscreenchange", syncNativeFullscreenOverlay);
    output.addEventListener("webkitbeginfullscreen", () => {
      state.webkitVideoFullscreen = true;
      syncNativeFullscreenOverlay();
    });
    output.addEventListener("webkitendfullscreen", () => {
      state.webkitVideoFullscreen = false;
      syncNativeFullscreenOverlay();
    });
  } catch (_error) {
    state.fullscreenTelemetryTrack = null;
    state.fullscreenTelemetryCue = null;
  }
}

function setupVideoPlayer() {
  $("#processedTab").addEventListener("click", () => selectVideoTab("processed"));
  $("#originalTab").addEventListener("click", () => selectVideoTab("original"));

  const previous = $("#previousPointZone");
  const next = $("#nextPointZone");
  previous.addEventListener("dblclick", (e) => {
    e.preventDefault();
    seekOutputPoint(-1);
  });
  next.addEventListener("dblclick", (e) => {
    e.preventDefault();
    seekOutputPoint(1);
  });
  // Keyboard activation remains accessible even though pointer navigation deliberately
  // requires a double-click to avoid accidental point jumps.
  previous.addEventListener("click", (e) => { if (e.detail === 0) seekOutputPoint(-1); });
  next.addEventListener("click", (e) => { if (e.detail === 0) seekOutputPoint(1); });

  const output = $("#outputVideo");
  setupNativeFullscreenOverlay(output);

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
      seekOriginal(s);
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

function clearPublishedAnalysis() {
  state.waveform = null;
  state.editSegments = null;
  renderSegments();
  $("#applySegments").disabled = true;
  const canvas = $("#timeline");
  const ctx = canvas?.getContext("2d");
  if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("#timelineMeta").textContent = "Reprocessing — previous analysis removed";
}

function showReprocessingState(job) {
  const media = { ...(job.media || {}) };
  media.output = null;
  media.output_download = null;
  media.metadata_download = null;
  const queued = {
    ...job,
    status: "queued",
    result: null,
    error: null,
    cancel_requested: false,
    media,
    processing: {
      stage: "queued",
      label: "Starting re-processing",
      percent: 0,
      detail: "Removing the previous result and submitting a new analysis",
    },
  };
  state.current = queued;
  clearPublishedAnalysis();
  renderDetail(queued);
  selectVideoTab("processed");
}

function markDirty() {
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
    $("#applySegments").disabled = true;
    renderSegments(); drawTimeline();
    toast("Reverted to detected segments");
  });
  $("#applySegments").addEventListener("click", applySegments);
}

async function applySegments() {
  if (!state.current) return;
  const id = state.current.id;
  const generation = beginDetailGeneration(id);
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
    if (!detailRequestIsCurrent(id, generation)) return;
    state.current = job;
    state.editSegments = null;
    renderDetail(job);
    await loadWaveform(id, generation);
    // force reload of output video (cache-busted URL from server)
    const out = $("#outputVideo");
    if (job.media?.output) { out.src = job.media.output; out.dataset.src = job.media.output; out.load(); }
    toast("Output re-cut from your edits");
  } catch (e) {
    if (!detailRequestIsCurrent(id, generation)) return;
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
  $("#goldenButton").addEventListener("click", openGolden);
  $("#reprocessButton").addEventListener("click", async () => {
    if (!state.current) return;
    const id = state.current.id;
    const generation = beginDetailGeneration(id);
    showReprocessingState(state.current);
    try {
      const job = await api(`/api/jobs/${id}/process`, { method: "POST" });
      if (!detailRequestIsCurrent(id, generation)) return;
      state.current = job;
      state.editSegments = null;
      state.waveform = null;
      renderDetail(job);
      toast("Re-processing…");
      startPolling(id, generation);
      loadJob(id, generation);
    } catch (e) {
      if (detailRequestIsCurrent(id, generation)) {
        toast(e.message, "error");
        try {
          const job = await loadJob(id, generation);
          if (job && ["queued", "running"].includes(job.status)) {
            startPolling(id, generation);
          }
        } catch (_) { /* keep the current snapshot */ }
      }
    }
  });
  $("#stopProcessingButton").addEventListener("click", async () => {
    if (!state.current || !["queued", "running"].includes(state.current.status)) return;
    if (!confirm(`Stop processing "${state.current.filename}"?`)) return;
    const id = state.current.id;
    const generation = beginDetailGeneration(id);
    $("#stopProcessingButton").disabled = true;
    try {
      const job = await api(`/api/jobs/${id}/cancel`, { method: "POST" });
      if (!detailRequestIsCurrent(id, generation)) return;
      state.current = job;
      renderDetail(job);
      toast(job.status === "cancelled" ? "Processing stopped" : "Stopping processing…");
      if (["queued", "running"].includes(job.status)) startPolling(id, generation);
      else stopPolling();
    } catch (e) {
      if (detailRequestIsCurrent(id, generation)) {
        $("#stopProcessingButton").disabled = false;
        toast(e.message, "error");
        try {
          const job = await loadJob(id, generation);
          if (job && ["queued", "running"].includes(job.status)) {
            startPolling(id, generation);
          }
        } catch (_) { /* keep the current snapshot */ }
      }
    }
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
  revision: null,
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
  if (lab.genTimer) { clearTimeout(lab.genTimer); lab.genTimer = null; }
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
    scheduleLabelPoll(job.id);
  } else {
    $("#labGenerate").disabled = false;
    if (lab.genTimer) { clearTimeout(lab.genTimer); lab.genTimer = null; }
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

function scheduleLabelPoll(id) {
  if (lab.genTimer) return;
  lab.genTimer = setTimeout(async () => {
    lab.genTimer = null;
    await pollLabeling(id);
  }, 1500);
}

async function pollLabeling(id) {
  try {
    const job = await api(`/api/jobs/${id}`);
    if (!state.current || state.current.id !== id) return;
    state.current.labeling = job.labeling;
    const l = job.labeling || {};
    if (l.status === "generating") {
      $("#labStatus").textContent = `Generating samples… ${l.detail || ""}`;
      scheduleLabelPoll(id);
    } else {
      clearTimeout(lab.genTimer); lab.genTimer = null;
      renderLabelingSection(job);
    }
  } catch (_) { clearTimeout(lab.genTimer); lab.genTimer = null; }
}

async function generateSamples() {
  if (!state.current) return;
  const id = state.current.id;
  const kinds = [];
  if ($("#labKindPlayers").checked) kinds.push("player_identity");
  if ($("#labKindServe").checked) kinds.push("serve_motion");
  if (!kinds.length) {
    toast("Choose player classification, serve motion, or both", "error");
    return;
  }
  $("#labGenerate").disabled = true;
  lab.tasks = [];               // force a fresh load when generation completes
  $("#labWorkspace").classList.add("hidden");
  try {
    await api(`/api/jobs/${id}/label-tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        kinds,
        max_items: Math.max(2, parseInt($("#labCount").value, 10) || 10),
        match_type: "auto",
        regenerate: true,
      }),
    });
    if (!state.current || state.current.id !== id) return;
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
  if (!state.current || state.current.id !== id) return;
  lab.tasks = data.tasks || [];
  lab.revision = data.revision || null;
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
  if (!labModeTasks().length) {
    lab.mode = lab.tasks[0]?.kind || "player_identity";
  }
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
  const jobId = state.current?.id;
  if (!jobId) return;
  const roster = $$("#rosterList input").map((inp) => {
    const r = lab.roster.find((x) => x.id === inp.dataset.id) || { id: inp.dataset.id };
    return { ...r, name: inp.value.trim() || inp.dataset.id };
  });
  lab.roster = roster;
  renderPlayerChoices();
  try { await api(`/api/jobs/${jobId}/roster`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ revision: lab.revision, roster }),
  }); } catch (e) { toast(`Roster save failed: ${e.message}`, "error"); }
}

function renderPlayerChoices() {
  const playerBox = $("#playerChoices");
  const serverBox = $("#serverChoices");
  playerBox.replaceChildren();
  serverBox.replaceChildren();
  const choice = (key, value, label) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "choice";
    button.dataset.k = key;
    button.dataset.v = value;
    button.textContent = label;
    return button;
  };
  lab.roster.forEach((r) => {
    playerBox.appendChild(choice("player", r.id, r.name || r.id));
    serverBox.appendChild(choice("server", r.id, r.name || r.id));
  });
  serverBox.appendChild(choice("server", "unknown", "Unknown"));
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
    const image = document.createElement("img");
    image.src = t.asset_url;
    image.alt = "player crop";
    stage.replaceChildren(image);
  } else {
    const video = document.createElement("video");
    video.src = t.asset_url;
    video.controls = true; video.autoplay = true; video.loop = true;
    video.muted = true; video.playsInline = true;
    stage.replaceChildren(video);
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
      const image = document.createElement("img");
      image.src = t.asset_url; image.loading = "lazy"; image.alt = "";
      el.appendChild(image);
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
  const jobId = state.current?.id;
  if (!jobId) return;
  const values = { ...lab.draft };
  if (lab.mode === "serve_motion") values.notes = $("#serveNotes").value || undefined;
  try {
    const res = await api(`/api/jobs/${jobId}/labels`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision: lab.revision, task_id: t.id, kind: t.kind, values }),
    });
    if (!state.current || state.current.id !== jobId) return;
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
function initialRoute() {
  const path = window.location.pathname;
  let match = path.match(/^\/jobs\/([0-9a-f-]+)\/signals\/players\/([^/]+)$/i);
  if (match) return { view: "player-signals", jobId: match[1], playerId: decodeURIComponent(match[2]) };
  match = path.match(/^\/jobs\/([0-9a-f-]+)\/signals\/?$/i);
  if (match) return { view: "signals", jobId: match[1] };
  match = path.match(/^\/jobs\/([0-9a-f-]+)\/?$/i);
  if (match) return { view: "detail", jobId: match[1] };
  return { view: "gallery" };
}

async function init() {
  setupUpload();
  setupDetailActions();
  setupSegmentEditor();
  setupTimelineInteractions();
  setupVideoPlayer();
  $("#saveMatchRoster").addEventListener("click", saveMatchRoster);
  setupLabeling();
  const route = initialRoute();
  if (route.view === "signals") {
    await openSignals(route.jobId);
  } else if (route.view === "player-signals") {
    await openPlayerSignals(route.jobId, route.playerId);
  } else if (route.view === "detail") {
    await openDetail(route.jobId);
    const seek = Number(new URLSearchParams(window.location.search).get("seek"));
    if (Number.isFinite(seek)) seekOriginal(seek);
  } else {
    refreshGallery();
  }
}
document.addEventListener("DOMContentLoaded", init);
