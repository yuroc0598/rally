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
  goldenPollTimer: null,
  goldenGeneration: 0,
  galleryRequest: 0,
  processingClockTimer: null,
  waveform: null,       // {duration, strikes[], segments[]}
  editSegments: null,   // [[start,end], ...] working copy while editing
  detailJobId: null,    // requested detail, even while its GET is in flight
  detailGeneration: 0, // invalidates stale job/waveform responses after navigation/mutation
  videoTab: null,
  hadOutput: false,
  outputLayout: [],
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
  if (["running", "queued", "cancelling", "starting", "audio", "visual", "ball_tracking", "pose", "serve", "deciding", "rendering", "probing", "writing", "waveform", "refining"].includes(s)) return "busy";
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
  fd.append("play_mode", $("#playMode").value);
  fd.append("detect_players", "true");
  fd.append("static_camera", $("#staticCamera").checked);
  fd.append("fast", $("#fast").checked);
  fd.append("hysteresis", $("#hysteresis").checked);
  fd.append("no_labels", $("#noLabels").checked);
  fd.append("ball_arbiter", "true");
  fd.append("court_auto", "true");
  fd.append("run_now", "true");
  const opt = {
    analysis_fps: "#analysisFps", min_rally: "#minRally", skip_intro: "#skipIntro",
    gap: "#gap", start_buffer: "#startBuffer", end_buffer: "#endBuffer",
    serve_preroll: "#servePreroll", tail: "#tail",
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
  $("#galleryView").classList.toggle("hidden", which !== "gallery");
  $("#goldenView").classList.toggle("hidden", which !== "golden");
  $("#detailView").classList.toggle("hidden", which !== "detail");
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
  if (r.n_strikes != null) metaBits.push(`${r.n_strikes} strikes`);
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
    $("#ballSpeed").classList.remove("hidden");
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
    $("#ballSpeed").classList.add("hidden");
    $("#pointOutcome").classList.add("hidden");
  }
  updateProcessingClock(job);
  if (state.videoTab === null || gainedOutput) selectVideoTab("processed");
  else selectVideoTab(state.videoTab);
  updateOutputOverlay();

  // downloads
  setLink($("#dlOutput"), m.output_download);
  setLink($("#dlJson"), m.metadata_download);

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

function updateOutputOverlay() {
  const current = currentOutputPoint();
  $("#currentPoint").textContent = current ? `Point ${current.index + 1}` : "Point —";
  const estimate = current?.point?.ball_speed_estimate;
  const speed = estimate?.value_kmh ?? current?.point?.peak_ball_speed_kmh;
  const uncertainty = estimate?.uncertainty_kmh;
  const readout = $("#ballSpeed");
  if (Number.isFinite(speed)) {
    const spread = Number.isFinite(uncertainty) ? ` ± ${Math.round(uncertainty)}` : "";
    readout.textContent = `Ball ~${Math.round(speed)}${spread} km/h · uncertain ground-plane estimate`;
    const limitations = Array.isArray(estimate?.limitations)
      ? estimate.limitations.join("; ")
      : "Single-camera court-plane estimate; ball height and full 3-D speed are not recovered.";
    readout.title = limitations;
  } else {
    readout.textContent = "Ball speed unavailable";
    readout.title = "Requires reliable ball trajectory and court calibration.";
  }
  const outcome = $("#pointOutcome");
  const participants = current?.point?.participants || {};
  const termination = current?.point?.termination || {};
  const server = matchPlayerName(participants.server_id);
  const bits = [];
  if (server) bits.push(`Server: ${server}`);
  const winner = matchPlayerName(termination.winner_player_id)
    || matchTeamName(termination.winner_team_id);
  if (winner) bits.push(`Winner: ${winner}`);
  const eventNames = {
    double_fault: "Double fault", out: "Ball out", net_failure: "Hit into net",
    second_bounce: "Second bounce", body_or_net_touch: "Player/net touch",
  };
  const creditNames = {
    ace: "Ace", service_winner: "Service winner", clean_winner: "Winner",
    forced_error: "Forced error", unforced_error: "Unforced error",
    error_unknown: "Error",
  };
  const finish = creditNames[termination.credit] || eventNames[termination.rule_event];
  if (finish) bits.push(`Finish: ${finish}`);
  const errorPlayer = matchPlayerName(termination.error_player_id);
  if (errorPlayer) bits.push(`Error: ${errorPlayer}`);
  if (!bits.length) bits.push("Point result: insufficient evidence");
  outcome.textContent = bits.join(" · ");
  outcome.classList.toggle("unknown", !termination.rule_event || termination.rule_event === "unknown");
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
    const badge = document.createElement("span");
    badge.className = "match-player-id";
    badge.textContent = player.id;
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
    wrap.append(badge, label);
    box.appendChild(wrap);
  });
  $("#saveMatchRoster").disabled = !roster.length;
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
    waveform = { duration: (state.current?.result || {}).total_seconds || 0, strikes: [], segments: [] };
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
  window.addEventListener("resize", () => { if (state.waveform) drawTimeline(); });
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
    const players = caps.players || {};
    const playerBox = $("#detectPlayers");
    if (playerBox && !players.available) {
      playerBox.checked = false;
      playerBox.disabled = true;
      const playerLabel = playerBox.closest("label");
      if (playerLabel) {
        playerLabel.classList.add("disabled");
        playerLabel.title = players.hint || "Player detector unavailable";
      }
    }
  } catch (_) { /* capabilities are best-effort; leave the toggle as-is */ }
}

function init() {
  setupUpload();
  setupDetailActions();
  setupSegmentEditor();
  setupTimelineInteractions();
  setupVideoPlayer();
  $("#saveMatchRoster").addEventListener("click", saveMatchRoster);
  setupLabeling();
  loadCapabilities();
  refreshGallery();
}
document.addEventListener("DOMContentLoaded", init);
