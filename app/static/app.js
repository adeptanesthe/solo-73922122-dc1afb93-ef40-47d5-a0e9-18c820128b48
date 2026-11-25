"use strict";

// ---- application state ----------------------------------------------------
const state = {
  jobId: null,
  n: 0,
  width: 0,
  height: 0,
  calib: null,          // committed-on-server annotation draft kept locally
  boxes: [],            // one rect or null per image
  current: 0,
  mode: "occ",
  offsets: null,
  enabled: [],
  edges: [],
  hasComposite: false,
  missingPixels: 0,
  totalPixels: 0,
  message: "",
  imageUrls: [],
};

const $ = (sel) => document.querySelector(sel);
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
let drag = null; // {kind, start:{x,y}, rect}

// ---- small helpers --------------------------------------------------------
function toast(msg, ms = 3200) {
  const el = $("#toast");
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-json */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || `请求失败 (${res.status})`);
    err.payload = data;
    err.status = res.status;
    throw err;
  }
  return data;
}

function applyServerState(s) {
  state.n = s.n;
  state.width = s.width;
  state.height = s.height;
  state.calib = s.calib;
  state.boxes = s.boxes.map((b) => (b ? [...b] : null));
  state.offsets = s.offsets;
  state.enabled = s.enabled;
  state.edges = s.edges || [];
  state.hasComposite = !!s.has_composite;
  state.missingPixels = s.missing_pixels || 0;
  state.totalPixels = s.total_pixels || 0;
  state.message = s.message || "";
  if (state.current >= state.n) state.current = 0;
  // Images are immutable within a job; build their URLs once and keep the
  // canvas image cache valid across repeated state syncs.
  if (state.imageUrls.length !== state.n) {
    state.imageUrls = Array.from({ length: state.n }, (_, i) =>
      `/api/jobs/${state.jobId}/images/${i}`);
    cachedImages.length = 0;
  }
}

function fmtOffset(v) {
  if (v === null || v === undefined) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

// ---- canvas drawing -------------------------------------------------------
const cachedImages = [];

function redraw() {
  if (!state.n) return;
  const img = cachedImages[state.current];
  if (!img || !img.complete) { loadAndDraw(); return; }
  canvas.width = state.width;
  canvas.height = state.height;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  drawRects();
}

function loadAndDraw() {
  if (!state.n) return;
  const img = new Image();
  img.onload = () => {
    cachedImages[state.current] = img;
    redraw();
  };
  img.src = state.imageUrls[state.current];
}

function strokeRect(rect, color, label) {
  if (!rect) return;
  const [x0, y0, x1, y1] = rect;
  ctx.save();
  ctx.lineWidth = Math.max(1, Math.round(state.width / 256));
  ctx.strokeStyle = color;
  ctx.fillStyle = color + "33";
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
  ctx.font = `${Math.max(10, Math.round(state.width / 22))}px sans-serif`;
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = color;
  ctx.fillRect(x0, Math.max(0, y0 - 14), tw + 8, 14);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x0 + 4, Math.max(11, y0 - 3));
  ctx.restore();
}

function drawRects() {
  if (state.calib) strokeRect(state.calib, "#1f6feb", "校准区");
  const b = state.boxes[state.current];
  if (b) strokeRect(b, "#d12f2f", `图${state.current + 1} 遮挡`);
  if (drag && drag.rect) {
    const color = drag.kind === "calib" ? "#1f6feb" : "#d12f2f";
    strokeRect(drag.rect, color, drag.kind === "calib" ? "校准区" : "遮挡框");
  }
}

function eventToImageCoord(ev) {
  const r = canvas.getBoundingClientRect();
  const x = Math.round((ev.clientX - r.left) * (canvas.width / r.width));
  const y = Math.round((ev.clientY - r.top) * (canvas.height / r.height));
  return {
    x: Math.max(0, Math.min(state.width - 1, x)),
    y: Math.max(0, Math.min(state.height - 1, y)),
  };
}

canvas.addEventListener("pointerdown", (ev) => {
  if (!state.n) return;
  canvas.setPointerCapture(ev.pointerId);
  const p = eventToImageCoord(ev);
  const kind = state.mode === "calib" ? "calib" : "occ";
  drag = { kind, start: p, rect: [p.x, p.y, p.x, p.y] };
});

canvas.addEventListener("pointermove", (ev) => {
  if (!drag) return;
  const p = eventToImageCoord(ev);
  drag.rect = [drag.start.x, drag.start.y, p.x, p.y];
  redraw();
});

canvas.addEventListener("pointerup", (ev) => {
  if (!drag) return;
  const p = eventToImageCoord(ev);
  const [sx, sy] = [drag.start.x, drag.start.y];
  const rect = [Math.min(sx, p.x), Math.min(sy, p.y), Math.max(sx, p.x), Math.max(sy, p.y)];
  const kind = drag.kind;
  drag = null;
  if (rect[2] - rect[0] < 2 || rect[3] - rect[1] < 2) {
    redraw();
    return; // treat tiny drags as no-op
  }
  if (kind === "calib") state.calib = rect;
  else state.boxes[state.current] = rect; // each image keeps at most one box
  redraw();
});

// ---- tabs & side panel ----------------------------------------------------
function renderTabs() {
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  for (let i = 0; i < state.n; i++) {
    const t = document.createElement("button");
    t.type = "button";
    t.className = "tab" + (i === state.current ? " active" : "");
    const on = state.enabled[i];
    t.innerHTML = `图 ${i + 1}<span class="dot ${on ? "on" : "off"}"></span>`;
    t.title = on ? "已连通基准，参与合成" : "未连通基准，已停用";
    t.addEventListener("click", () => { state.current = i; renderAll(); });
    tabs.appendChild(t);
  }
}

function renderOffsetTable() {
  const tb = $("#offset-table tbody");
  tb.innerHTML = "";
  for (let i = 0; i < state.n; i++) {
    const tr = document.createElement("tr");
    const off = state.offsets ? state.offsets[i] : null;
    const on = state.enabled[i];
    tr.innerHTML =
      `<td>图 ${i + 1}${i === 0 ? "（基准）" : ""}</td>` +
      `<td>${fmtOffset(off)}</td>` +
      `<td><span class="tag ${on ? "ok" : "off"}">${on ? "可用" : "停用"}</span></td>`;
    tb.appendChild(tr);
  }
  const edgeEl = $("#edge-list");
  if (state.edges.length) {
    edgeEl.innerHTML = state.edges
      .map((e) => `<div>图${e.a + 1} ↔ 图${e.b + 1}：Δ=${e.d >= 0 ? "+" : ""}${e.d.toFixed(2)}（${e.npix} 个校准像素）</div>`)
      .join("");
  } else {
    edgeEl.innerHTML = "<div>暂无满足 ≥16 个共用像素的约束</div>";
  }
}

function renderStatus() {
  const el = $("#status-msg");
  el.textContent = state.message || "请先圈定共用校准区，并为每张照片画遮挡框";
  el.className = "status";
  if (/冲突|失败|错误/.test(state.message)) el.classList.add("err");
  else if (/完成|已/.test(state.message)) el.classList.add("ok");
}

function renderComposite() {
  const img = $("#composite-img");
  const meta = $("#composite-meta");
  const dl = $("#download-png");
  if (!state.hasComposite) {
    img.hidden = true;
    img.removeAttribute("src");
    dl.disabled = true;
    meta.textContent = "尚未合成";
    return;
  }
  dl.disabled = false;
  const pct = state.totalPixels ? (100 * state.missingPixels / state.totalPixels).toFixed(2) : "0";
  meta.textContent = `待补拍像素 ${state.missingPixels} / ${state.totalPixels}（${pct}%），少于两张可用值的位置已标记`;
  const overlay = $("#toggle-overlay").checked;
  const url = `/api/jobs/${state.jobId}/${overlay ? "composite-overlay.png" : "composite.png"}?t=${Date.now()}`;
  img.src = url;
  img.hidden = false;
}

function renderAll() {
  renderTabs();
  renderOffsetTable();
  renderStatus();
  renderComposite();
  redraw();
}

// ---- upload / sample ------------------------------------------------------
async function createJobAndUpload(files) {
  if (files.length < 3 || files.length > 8) {
    toast("需要 3–8 张图片"); return;
  }
  const badName = Array.from(files).find((f) => !/\.png$/i.test(f.name));
  if (badName) { toast(`仅支持 PNG：${badName.name}`); return; }
  try {
    const job = await api("/api/jobs", { method: "POST" });
    state.jobId = job.job_id;
    const fd = new FormData();
    files.forEach((f) => fd.append("images", f));
    const s = await api(`/api/jobs/${state.jobId}/images`, { method: "POST", body: fd });
    applyServerState(s);
    $("#workspace").hidden = false;
    renderAll();
    toast("上传成功，请圈定校准区");
  } catch (e) { toast(e.message); }
}

$("#file-input").addEventListener("change", (ev) => {
  const files = ev.target.files;
  if (files && files.length) createJobAndUpload(Array.from(files));
  ev.target.value = "";
});

$("#load-samples").addEventListener("click", async () => {
  try {
    const job = await api("/api/jobs", { method: "POST" });
    state.jobId = job.job_id;
    const s = await api(`/api/jobs/${state.jobId}/samples`, { method: "POST" });
    applyServerState(s);
    $("#workspace").hidden = false;
    renderAll();
    toast("已载入 5 张示例照片及预置标注");
  } catch (e) { toast(e.message); }
});

// ---- toolbar actions ------------------------------------------------------
document.querySelectorAll('input[name="mode"]').forEach((r) => {
  r.addEventListener("change", () => {
    state.mode = document.querySelector('input[name="mode"]:checked').value;
    $("#mode-hint").textContent = state.mode === "calib"
      ? "模式：拖动圈定所有照片共用的校准区（替换原校准区）"
      : "模式：在当前照片上拖动画遮挡框（每张图至多一个）";
  });
});

$("#clear-box").addEventListener("click", () => {
  if (!state.n) return;
  state.boxes[state.current] = null;
  redraw();
  toast(`已清除图 ${state.current + 1} 的遮挡框（应用后生效）`);
});

$("#clear-calib").addEventListener("click", () => {
  state.calib = null;
  redraw();
  toast("已清除校准区（应用后生效）");
});

$("#apply").addEventListener("click", async () => {
  if (!state.jobId) return;
  try {
    const s = await api(`/api/jobs/${state.jobId}/annotation`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ calib: state.calib, boxes: state.boxes }),
    });
    applyServerState(s.state);
    renderAll();
    toast("标注已应用");
  } catch (e) {
    if (e.status === 409 && e.payload && e.payload.state) {
      // Contradiction: server kept the previous good annotation/base map.
      applyServerState(e.payload.state);
      renderAll();
      toast("约束冲突：已保留上次标注与底图，请调整后重试", 6000);
    } else {
      toast(e.message);
    }
  }
});

$("#toggle-overlay").addEventListener("change", renderComposite);

$("#download-png").addEventListener("click", () => {
  if (!state.jobId) return;
  const a = document.createElement("a");
  a.href = `/api/jobs/${state.jobId}/composite.png?download=1&t=${Date.now()}`;
  a.download = "base_map.png";
  a.click();
});
