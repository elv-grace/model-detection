/* model-detection search demo — UI.
 *
 * Three jobs: collect a query, POST it to the backend, and render the returned frames so
 * a person can judge in one pass whether the retrieval is reasonable. All scoring and
 * embedding happens server-side; nothing here interprets a vector.
 */

const $ = (id) => document.getElementById(id);

const state = {
  mode: "text",
  // The uploaded image, kept as an HTMLImageElement so the crop canvas can redraw it.
  image: null,
  file: null,
  // Fractional (x, y, w, h) of the drag-selected region, or null for the whole image.
  crop: null,
  indexInfo: null,
};

/* ------------------------------------------------------------------ persistence */

// The index qid and EVIE base are worth remembering between reloads. The token is too,
// deliberately: this is a local evaluation tool and re-pasting a 400-character token for
// every reload is the kind of friction that stops it being used.
const STORAGE_KEY = "model-detection-demo";

function saveSettings() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      indexQid: $("index-qid").value,
      token: $("token").value,
      contentTokens: $("content-tokens").value,
      evieBase: $("evie-base").value,
    }));
  } catch (e) { /* private browsing — not worth surfacing */ }
}

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    if (saved.indexQid) $("index-qid").value = saved.indexQid;
    if (saved.token) $("token").value = saved.token;
    if (saved.contentTokens) $("content-tokens").value = saved.contentTokens;
    if (saved.evieBase) $("evie-base").value = saved.evieBase;
  } catch (e) { /* ignore malformed storage */ }
}

/* ------------------------------------------------------------------ connection */

async function connect() {
  const indexQid = $("index-qid").value.trim();
  const token = $("token").value.trim();
  if (!indexQid || !token) {
    setStatus("index qid and token are both required", "bad");
    return;
  }
  saveSettings();
  setStatus("connecting…");
  $("index-details").innerHTML = "";

  try {
    const params = new URLSearchParams({ index_qid: indexQid, token });
    const response = await fetch(`/api/index_info?${params}`);
    const body = await response.json();
    if (!response.ok) throw new Error(formatError(body));

    state.indexInfo = body;
    const total = (body.tracks || []).reduce((sum, t) => sum + (t.count || 0), 0);
    setStatus(`${total.toLocaleString()} vectors · ${body.vector_size}-d`, "ok");
    renderIndexDetails(body);
  } catch (error) {
    state.indexInfo = null;
    setStatus("not connected", "bad");
    showMessage(error.message, true);
  }
}

function renderIndexDetails(info) {
  const provenance = info.index_provenance || {};
  const rows = [
    ["tracks", (info.tracks || []).map((t) => `${t.name} (${t.count.toLocaleString()})`).join(", ") || "—"],
    ["searching", info.selected_track || "—"],
    ["embedder", provenance.embedder || "unknown"],
    ["dim", provenance.dim ? `${provenance.dim} → ${info.vector_size}` : info.vector_size],
    ["patches", provenance.max_num_patches ?? "unknown"],
    ["crop padding", provenance.crop_padding ?? "unknown"],
  ];
  const warnings = (info.warnings || [])
    .map((w) => `<div class="warning">${escapeHtml(w)}</div>`).join("");

  $("index-details").innerHTML =
    `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("")}</dl>${warnings}`;
}

/* ------------------------------------------------------------------ search */

async function search() {
  const indexQid = $("index-qid").value.trim();
  const token = $("token").value.trim();
  if (!indexQid || !token) { showMessage("index qid and token are both required", true); return; }

  // One slider, two meanings. A text query's cosine tops out around 0.3, so a cosine
  // floor set for image search would silently return nothing; text is floored on the
  // calibrated probability instead.
  const floor = Number($("min-score").value) / 100;
  const shared = {
    index_qid: indexQid,
    token,
    content_tokens: $("content-tokens").value,
    limit: Number($("limit").value),
    track: state.indexInfo?.selected_track || "detection",
    min_similarity: state.mode === "image" && floor > 0 ? floor : null,
    min_probability: state.mode === "text" && floor > 0 ? floor : null,
    collapse: $("collapse").value,
    collapse_gap_ms: 1000,
    evie_base: $("evie-base").value.trim() || null,
  };

  $("search").disabled = true;
  showMessage("searching…");
  $("grid").innerHTML = "";
  $("query-summary").innerHTML = "";

  try {
    let response;
    if (state.mode === "text") {
      const query = $("query-text").value.trim();
      if (!query) throw new Error("enter a text query");
      response = await fetch("/api/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...shared,
          query,
          qids: splitQids($("qids").value),
        }),
      });
    } else {
      if (!state.file) throw new Error("choose an image to search with");
      const form = new FormData();
      form.append("file", state.file);
      Object.entries(shared).forEach(([key, value]) => {
        if (value !== null && value !== undefined) form.append(key, String(value));
      });
      form.append("qids", splitQids($("qids").value).join(","));
      form.append("match_index_padding", $("match-padding").checked ? "true" : "false");
      if (state.crop) form.append("crop", JSON.stringify(state.crop));
      response = await fetch("/api/search/image", { method: "POST", body: form });
    }

    const body = await response.json();
    if (!response.ok) throw new Error(formatError(body));
    render(body);
  } catch (error) {
    showMessage(error.message, true);
  } finally {
    $("search").disabled = false;
  }
}

function render(body) {
  const meta = body.meta || {};
  const results = body.results || [];

  const summary = [];
  if (meta.query_image) summary.push(`<img src="${meta.query_image}" alt="query">`);
  summary.push(
    `<div><strong>${results.length}</strong> shown of <strong>${meta.matched}</strong> matched ` +
    `(pool ${meta.pool}) · ${escapeHtml(meta.modality)} query` +
    (meta.query ? ` · “${escapeHtml(meta.query)}”` : "") +
    (meta.crop_applied ? " · cropped" : "") + "</div>"
  );
  $("query-summary").innerHTML = summary.join("");

  if (!results.length) {
    showMessage("no results — try lowering the minimum similarity, or check the track.");
    return;
  }
  showMessage("");

  reportDiagnostics(meta, results);

  $("grid").innerHTML = results.map(card).join("");
  $("grid").querySelectorAll(".card__frame[data-full]").forEach((node) => {
    node.addEventListener("click", () => openLightbox(node.dataset));
  });
  // Per-card switch between the whole frame and just this detection. Without it, several
  // results on one frame are visually identical and there is nothing to judge.
  $("grid").querySelectorAll("[data-toggle-crop]").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      const card = node.closest(".card");
      const img = card.querySelector(".card__frame img");
      const showingCrop = node.dataset.showing === "crop";
      img.src = showingCrop ? node.dataset.frame : node.dataset.crop;
      node.dataset.showing = showingCrop ? "frame" : "crop";
      node.textContent = showingCrop ? "Show detection" : "Show whole frame";
      card.querySelector(".card__frame").classList.toggle("card__frame--crop", !showingCrop);
    });
  });
}

// A missing frame or a missing box is indistinguishable from a bad query unless the app
// says which it was. The backend reports per content object what it tried.
function reportDiagnostics(meta, results) {
  const content = meta.content || [];
  const notes = [];

  const denied = content.filter((c) => !c.token);
  if (denied.length) {
    notes.push(
      `No token for ${denied.length} content object${denied.length > 1 ? "s" : ""} ` +
      `(${denied.reduce((n, c) => n + c.results, 0)} results): ` +
      denied.map((c) => c.qid).join(", ") +
      ". Add a token for them under Content tokens to see their frames."
    );
  }

  if (results.length && results.every((r) => !r.box)) {
    const reasons = new Set();
    content.forEach((c) => (c.boxes || []).forEach((b) => {
      if (b.detail) reasons.add(`${b.source} — ${b.detail}`);
    }));
    notes.push(
      "No bounding boxes for these results, so cards show whole frames and results on the " +
      "same frame look alike." + (reasons.size ? ` Sources tried: ${[...reasons].join("; ")}.` : "")
    );
  }

  showMessage(notes.join("\n\n"));
}

function card(result) {
  const info = result.additional_info || {};
  const score = result.probability !== null && result.probability !== undefined
    ? `p ${(result.probability * 100).toFixed(1)}%`
    : `cos ${result.similarity.toFixed(3)}`;

  const badges = [`<span class="badge badge--score">${score}</span>`];
  if (result.crops_in_frame > 1) {
    badges.push(`<span class="badge badge--multi">${result.crops_in_frame} crops</span>`);
  }
  if (info.prompt) badges.push(`<span class="badge">${escapeHtml(info.prompt)}</span>`);

  // The box, when there is one, is drawn into the image server-side rather than overlaid
  // in CSS — a percentage overlay only lines up while the frame's aspect ratio matches the
  // tile's, and it cannot follow the image once the card switches to the cropped view.
  const frame = result.thumbnail_url
    ? `<img src="${escapeAttr(result.thumbnail_url)}" alt="frame ${result.frame_idx}" loading="lazy">`
    : `<div class="placeholder">${escapeHtml(result.error || "no frame available")}</div>`;

  const rows = [
    ["time", result.seconds !== null ? formatTime(result.seconds) : "—"],
    ["frame", result.frame_idx ?? "—"],
    ["detector", info.detector || "—"],
    ["det. score", info.score ?? "—"],
    ["upscale", info.upscale ? `${info.upscale}×` : "—"],
    ["content", result.qid],
  ];

  const links = [];
  if (result.crop_url) {
    links.push(
      `<button class="linkish" data-toggle-crop data-showing="frame"
               data-frame="${escapeAttr(result.thumbnail_url)}"
               data-crop="${escapeAttr(result.crop_url)}">Show detection</button>`
    );
  }
  if (result.full_url) links.push(`<a href="${escapeAttr(result.full_url)}" target="_blank" rel="noopener">Full frame</a>`);
  if (result.evie_url) links.push(`<a href="${escapeAttr(result.evie_url)}" target="_blank" rel="noopener">Open in EVIE</a>`);

  return `
    <article class="card">
      <div class="card__frame" ${result.full_url ? `data-full="${escapeAttr(result.full_url)}"
           data-caption="${escapeAttr(captionFor(result, score))}"` : ""}>
        ${frame}
        <div class="card__scores">${badges.join("")}</div>
      </div>
      <div class="card__body">
        ${rows.map(([k, v]) => `<div class="card__row"><span>${k}</span><span>${escapeHtml(String(v))}</span></div>`).join("")}
        ${links.length ? `<div class="card__links">${links.join("")}</div>` : ""}
        ${result.error && result.thumbnail_url ? `<div class="card__error">${escapeHtml(result.error)}</div>` : ""}
      </div>
    </article>`;
}

function captionFor(result, score) {
  const info = result.additional_info || {};
  return `${score} · ${info.prompt || "detection"} · frame ${result.frame_idx} · ` +
         `${formatTime(result.seconds)} · ${result.qid}`;
}

/* ------------------------------------------------------------------ lightbox */

function openLightbox({ full, caption }) {
  // `full` already carries the box query parameter, so the backend draws the detection
  // into the full-resolution frame — nothing to overlay here.
  $("lightbox-image").src = full;
  $("lightbox-caption").innerHTML = `<strong>${escapeHtml(caption)}</strong>`;
  $("lightbox").classList.remove("lightbox--hidden");
}

function closeLightbox() {
  $("lightbox").classList.add("lightbox--hidden");
  $("lightbox-image").src = "";
}

/* ------------------------------------------------------------------ crop tool */

function initCropTool() {
  const canvas = $("crop-canvas");
  const area = $("crop-area");
  const rect = $("crop-rect");
  let dragging = false;
  let origin = null;

  const pointAt = (event) => {
    const bounds = canvas.getBoundingClientRect();
    return {
      x: clamp((event.clientX - bounds.left) / bounds.width, 0, 1),
      y: clamp((event.clientY - bounds.top) / bounds.height, 0, 1),
    };
  };

  const paint = (a, b) => {
    const bounds = canvas.getBoundingClientRect();
    const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
    const w = Math.abs(a.x - b.x), h = Math.abs(a.y - b.y);
    rect.style.display = "block";
    rect.style.left = `${x * bounds.width}px`;
    rect.style.top = `${y * bounds.height}px`;
    rect.style.width = `${w * bounds.width}px`;
    rect.style.height = `${h * bounds.height}px`;
    return [x, y, w, h];
  };

  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    origin = pointAt(event);
    canvas.setPointerCapture(event.pointerId);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    paint(origin, pointAt(event));
  });

  canvas.addEventListener("pointerup", (event) => {
    if (!dragging) return;
    dragging = false;
    const crop = paint(origin, pointAt(event));
    // A stray click is a zero-area box, not a request to search one pixel.
    if (crop[2] < 0.01 || crop[3] < 0.01) { clearCrop(); return; }
    state.crop = crop;
    $("crop-hint").textContent =
      `Searching with the selected region (${(crop[2] * 100).toFixed(0)}% × ${(crop[3] * 100).toFixed(0)}% of the image).`;
  });

  $("clear-crop").addEventListener("click", clearCrop);

  function clearCrop() {
    state.crop = null;
    rect.style.display = "none";
    $("crop-hint").textContent = "Drag on the image to search with just that region.";
  }
  window.clearCrop = clearCrop;
  return area;
}

function loadImage(file) {
  state.file = file;
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      state.image = image;
      const canvas = $("crop-canvas");
      // Cap the backing store; the full-resolution upload still goes to the server, this
      // is only what the crop tool draws on.
      const scale = Math.min(1, 640 / image.width);
      canvas.width = Math.round(image.width * scale);
      canvas.height = Math.round(image.height * scale);
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      $("crop-area").classList.remove("crop-area--empty");
      window.clearCrop();
    };
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
}

/* ------------------------------------------------------------------ helpers */

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function splitQids(raw) {
  return (raw || "").split(",").map((q) => q.trim()).filter(Boolean);
}

function formatTime(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${rest}`;
}

function formatError(body) {
  const detail = body && body.detail !== undefined ? body.detail : body;
  return typeof detail === "string" ? detail : JSON.stringify(detail, null, 2);
}

function setStatus(text, kind) {
  const node = $("index-status");
  node.textContent = text;
  node.className = "topbar__status" + (kind ? ` topbar__status--${kind}` : "");
}

function showMessage(text, isError = false) {
  const node = $("message");
  node.textContent = text || "";
  node.className = "message" + (isError && text ? " message--error" : "");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const escapeAttr = escapeHtml;

/* ------------------------------------------------------------------ wiring */

document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  initCropTool();

  $("connect").addEventListener("click", connect);
  $("search").addEventListener("click", search);
  $("query-text").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
  $("query-file").addEventListener("change", (e) => {
    if (e.target.files[0]) loadImage(e.target.files[0]);
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      state.mode = tab.dataset.mode;
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("tab--active", t === tab));
      $("pane-text").classList.toggle("pane--hidden", state.mode !== "text");
      $("pane-image").classList.toggle("pane--hidden", state.mode !== "image");
      $("min-score-label").textContent =
        state.mode === "text" ? "Min probability" : "Min similarity";
    });
  });

  $("limit").addEventListener("input", (e) => { $("limit-value").textContent = e.target.value; });
  $("min-score").addEventListener("input", (e) => {
    const value = Number(e.target.value);
    $("min-score-value").textContent = value === 0 ? "off" : (value / 100).toFixed(2);
  });
  ["index-qid", "token", "content-tokens", "evie-base"]
    .forEach((id) => $(id).addEventListener("change", saveSettings));

  $("lightbox-close").addEventListener("click", closeLightbox);
  $("lightbox").addEventListener("click", (e) => { if (e.target === $("lightbox")) closeLightbox(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });

  if ($("index-qid").value && $("token").value) connect();
});
