"""Flask web service for the mural-restoration base-map builder."""

from __future__ import annotations

import io
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
from flask import (
    Flask,
    abort,
    jsonify,
    request,
    send_file,
    send_from_directory,
)
from PIL import Image

from . import core
from .core import (
    CalibrationConflictError,
    MAX_IMAGES,
    MIN_IMAGES,
    SUPPORTED_FORMAT,
    SUPPORTED_MODES,
    ValidationError,
)

DATA_ROOT = Path(os.environ.get("APP_DATA", Path(__file__).resolve().parent.parent / "data"))
SAMPLE_DIR = Path(os.environ.get("APP_SAMPLES", Path(__file__).resolve().parent.parent / "samples"))

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB total request


def _load_png(data: bytes) -> np.ndarray:
    """Decode an uploaded PNG, enforcing 8-bit grayscale and size limits."""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # Pillow raises a variety of error types
        raise ValidationError(f"无法解析图片：{exc}") from exc
    if im.format != SUPPORTED_FORMAT:
        raise ValidationError("仅接受 PNG 格式")
    if im.mode not in SUPPORTED_MODES:
        raise ValidationError(f"仅接受 8 位灰度图（模式 L），当前为 {im.mode}")
    arr = np.asarray(im, dtype=np.uint8)
    core.validate_image(arr)
    return arr


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="L").save(path, format="PNG")


class Job:
    """One browser session: images, annotation, calibration state, base map."""

    def __init__(self, job_id: str, root: Path):
        self.id = job_id
        self.root = root
        self.img_dir = root / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.images: list[np.ndarray] = []
        self.h = self.w = 0
        self.calib = None
        self.boxes: list = []
        # Last *committed, non-conflicting* calibration / composite.
        self.offsets: list | None = None
        self.enabled: list = []
        self.edges: list = []
        self.composite_arr: np.ndarray | None = None
        self.missing: np.ndarray | None = None
        self.counts: np.ndarray | None = None
        # Boxes behind the current offsets / composite, for incremental recompute.
        self.committed_boxes: list = []
        self.message = ""

    # ---------- persistence ----------
    def to_state(self) -> dict:
        return {
            "id": self.id,
            "n": len(self.images),
            "width": self.w,
            "height": self.h,
            "calib": self.calib,
            "boxes": self.boxes,
            "committed_boxes": self.committed_boxes,
            "offsets": self.offsets,
            "enabled": self.enabled,
            "edges": [
                {"a": e.a, "b": e.b, "d": e.offset_b_from_a, "npix": e.npix}
                for e in self.edges
            ],
            "has_composite": self.composite_arr is not None,
            "message": self.message,
        }


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._guard = threading.Lock()

    def create(self) -> Job:
        job_id = uuid.uuid4().hex
        with self._guard:
            job = Job(job_id, self.root / job_id)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job:
        with self._guard:
            job = self._jobs.get(job_id)
        if job is None:
            abort(404, description="任务不存在或已过期，请重新上传")
        return job


store = JobStore(DATA_ROOT)


# ---------------------------------------------------------------- helpers
def _box_from_payload(raw, w: int, h: int):
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValidationError("矩形必须是 [x0, y0, x1, y1] 或 null")
    return core.normalize_box(raw, w, h)


def _recompute(job: Job, *, boxes_before: list | None) -> dict:
    """Run calibration + composite for the job's current annotation.

    On conflict, nothing is committed: offsets, enabled flags, boxes and base
    map stay at their previous good values, and the conflict message is shown.
    When calibration stays consistent and only occlusion boxes changed, only
    affected pixels are recomputed.
    """
    n = len(job.images)
    h, w = job.h, job.w
    # Normalize into local copies first; job.boxes / job.calib are only
    # mutated after the calibration solver confirms there is no conflict.
    boxes = [core.normalize_box(b, w, h) for b in job.boxes]
    occ = np.stack([core.rect_mask(b, h, w) for b in boxes])

    calib = core.normalize_box(job.calib, w, h) if job.calib is not None else None

    edges = core.build_all_edges(np.stack(job.images), occ, calib)
    result = core.solve_offsets(n, edges)  # raises on contradiction

    job.boxes = boxes
    job.calib = calib

    incremental = (
        boxes_before is not None
        and job.offsets is not None
        and core.offsets_unchanged(
            job.offsets, job.enabled, result.offsets, result.enabled
        )
    )

    if incremental:
        changed = core.affected_pixel_mask(boxes_before, boxes, h, w)
        # Recompute only changed pixels, then merge into the existing base map.
        new_comp, new_missing, new_counts = core.composite(
            np.stack(job.images), occ, result.offsets, result.enabled,
            pixel_mask=changed,
        )
        job.composite_arr[changed] = new_comp[changed]
        job.missing[changed] = new_missing[changed]
        job.counts[changed] = new_counts[changed]
        mode = "incremental"
    else:
        job.composite_arr, job.missing, job.counts = core.composite(
            np.stack(job.images), occ, result.offsets, result.enabled
        )
        mode = "full"

    job.offsets = result.offsets
    job.enabled = result.enabled
    job.edges = result.edges
    job.committed_boxes = [b for b in boxes]
    n_reachable = sum(result.enabled)
    n_disabled = n - n_reachable
    job.message = (
        f"校准完成（{'增量重算' if mode == 'incremental' else '全量重算'}）："
        f"{len(edges)} 条约束，{n_reachable}/{n} 张图可用"
        + (f"，{n_disabled} 张无法连通基准已停用" if n_disabled else "")
    )
    return {"mode": mode, "edges": len(edges)}


def _public_state(job: Job) -> dict:
    state = job.to_state()
    if job.missing is not None:
        state["missing_pixels"] = int(job.missing.sum())
        state["total_pixels"] = int(job.missing.size)
    return state


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/jobs", methods=["POST"])
def create_job():
    job = store.create()
    return jsonify({"job_id": job.id})


@app.route("/api/jobs/<job_id>/images", methods=["POST"])
def upload_images(job_id):
    job = store.get(job_id)
    files = request.files.getlist("images")
    if not files:
        files = [request.files[k] for k in sorted(request.files)]
    if not (MIN_IMAGES <= len(files) <= MAX_IMAGES):
        raise ValidationError(f"需要上传 {MIN_IMAGES}–{MAX_IMAGES} 张图片，收到 {len(files)} 张")
    arrays = []
    shape = None
    with job.lock:
        for f in files:
            arr = _load_png(f.read())
            core.validate_image(arr, expected_shape=shape)
            shape = arr.shape
            arrays.append(arr)
        # Replace any previous upload wholesale.
        job.images = arrays
        job.h, job.w = shape
        job.boxes = [None] * len(arrays)
        job.committed_boxes = [None] * len(arrays)
        job.calib = None
        job.offsets = None
        job.enabled = []
        job.edges = []
        job.composite_arr = None
        job.missing = None
        job.counts = None
        job.message = "图片已上传，请先圈定共用校准区"
        for old in job.img_dir.glob("*.png"):
            old.unlink()
        for i, arr in enumerate(arrays):
            _save_png(job.img_dir / f"{i}.png", arr)
    return jsonify(_public_state(job))


@app.route("/api/jobs/<job_id>/samples", methods=["POST"])
def load_samples(job_id):
    job = store.get(job_id)
    paths = sorted(SAMPLE_DIR.glob("sample_*.png"))
    if not (MIN_IMAGES <= len(paths) <= MAX_IMAGES):
        abort(500, description="服务器示例照片缺失或数量不合法")
    arrays = []
    shape = None
    with job.lock:
        for p in paths:
            arr = _load_png(p.read_bytes())
            core.validate_image(arr, expected_shape=shape)
            shape = arr.shape
            arrays.append(arr)
        job.images = arrays
        job.h, job.w = shape
        # The sample ships with a suggested annotation for quick trial.
        meta_path = SAMPLE_DIR / "annotation.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        job.calib = core.normalize_box(meta.get("calib"), job.w, job.h) if meta.get("calib") else None
        job.boxes = [core.normalize_box(b, job.w, job.h) if b else None
                     for b in meta.get("boxes", [None] * len(arrays))]
        if len(job.boxes) != len(arrays):
            job.boxes = [None] * len(arrays)
        job.committed_boxes = [None] * len(arrays)
        job.offsets = None
        job.enabled = []
        job.edges = []
        job.composite_arr = None
        job.missing = None
        job.counts = None
        job.message = "已载入示例照片，可直接点“应用标注并合成”或自行调整框"
        for old in job.img_dir.glob("*.png"):
            old.unlink()
        for i, arr in enumerate(arrays):
            _save_png(job.img_dir / f"{i}.png", arr)
    return jsonify(_public_state(job))


@app.route("/api/jobs/<job_id>/state", methods=["GET"])
def get_state(job_id):
    job = store.get(job_id)
    with job.lock:
        return jsonify(_public_state(job))


@app.route("/api/jobs/<job_id>/annotation", methods=["POST"])
def update_annotation(job_id):
    """Commit a new annotation. On conflict the previous state is retained."""
    job = store.get(job_id)
    payload = request.get_json(force=True, silent=True) or {}
    with job.lock:
        if not job.images:
            raise ValidationError("请先上传图片")
        h, w = job.h, job.w
        prev_boxes = [b for b in job.boxes]
        prev_calib = job.calib
        prev_offsets = None if job.offsets is None else list(job.offsets)
        prev_enabled = list(job.enabled)
        prev_edges = list(job.edges)
        prev_committed = list(job.committed_boxes)
        prev_comp = None if job.composite_arr is None else job.composite_arr.copy()
        prev_missing = None if job.missing is None else job.missing.copy()
        prev_counts = None if job.counts is None else job.counts.copy()

        try:
            if "calib" in payload:
                job.calib = _box_from_payload(payload["calib"], w, h)
            if "boxes" in payload:
                raw_boxes = payload["boxes"]
                if not isinstance(raw_boxes, list) or len(raw_boxes) != len(job.images):
                    raise ValidationError("遮挡框数量必须与图片数一致")
                job.boxes = [_box_from_payload(b, w, h) for b in raw_boxes]
            if "box" in payload:  # single-image edit {index, box}
                idx = int(payload["box"].get("index", -1))
                if not (0 <= idx < len(job.images)):
                    raise ValidationError("图片序号越界")
                job.boxes[idx] = _box_from_payload(payload["box"].get("rect"), w, h)
            info = _recompute(job, boxes_before=prev_committed)
        except CalibrationConflictError as exc:
            # Roll back everything: keep last good annotation and base map.
            job.boxes = prev_boxes
            job.calib = prev_calib
            job.offsets = prev_offsets
            job.enabled = prev_enabled
            job.edges = prev_edges
            job.committed_boxes = prev_committed
            job.composite_arr = prev_comp
            job.missing = prev_missing
            job.counts = prev_counts
            job.message = f"约束冲突，已保留上次标注与底图：{exc}"
            return jsonify(
                {"ok": False, "conflict": True, "error": str(exc),
                 "state": _public_state(job)}
            ), 409
        except ValidationError:
            raise
    return jsonify({"ok": True, **info, "state": _public_state(job)})


@app.route("/api/jobs/<job_id>/recompute", methods=["POST"])
def recompute(job_id):
    """Force a full recompute from the current annotation."""
    job = store.get(job_id)
    with job.lock:
        # Snapshot for rollback in case the forced recompute reveals a conflict.
        prev_committed = list(job.committed_boxes)
        prev_offsets = None if job.offsets is None else list(job.offsets)
        prev_enabled = list(job.enabled)
        prev_edges = list(job.edges)
        prev_comp = None if job.composite_arr is None else job.composite_arr.copy()
        prev_missing = None if job.missing is None else job.missing.copy()
        prev_counts = None if job.counts is None else job.counts.copy()
        prev_calib = job.calib
        prev_draft_boxes = list(job.boxes)
        try:
            # boxes_before=None forces a full (non-incremental) recompute.
            info = _recompute(job, boxes_before=None)
        except CalibrationConflictError as exc:
            job.offsets = prev_offsets
            job.enabled = prev_enabled
            job.edges = prev_edges
            job.composite_arr = prev_comp
            job.missing = prev_missing
            job.counts = prev_counts
            job.committed_boxes = prev_committed
            job.calib = prev_calib
            job.boxes = prev_draft_boxes
            job.message = f"约束冲突，已保留上次标注与底图：{exc}"
            return jsonify(
                {"ok": False, "conflict": True, "error": str(exc),
                 "state": _public_state(job)}
            ), 409
    return jsonify({"ok": True, **info, "state": _public_state(job)})


@app.route("/api/jobs/<job_id>/images/<int:index>")
def get_image(job_id, index):
    job = store.get(job_id)
    if not (0 <= index < len(job.images)):
        abort(404)
    return send_file(job.img_dir / f"{index}.png", mimetype="image/png")


@app.route("/api/jobs/<job_id>/composite.png")
def get_composite(job_id):
    job = store.get(job_id)
    with job.lock:
        if job.composite_arr is None:
            abort(404, description="尚未合成底图")
        buf = io.BytesIO()
        Image.fromarray(job.composite_arr, mode="L").save(buf, format="PNG")
        buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/jobs/<job_id>/composite-overlay.png")
def get_composite_overlay(job_id):
    """Base map with 待补拍 pixels tinted red (RGB preview)."""
    job = store.get(job_id)
    with job.lock:
        if job.composite_arr is None:
            abort(404, description="尚未合成底图")
        base = job.composite_arr
        rgb = np.stack([base, base, base], axis=-1).copy()
        rgb[job.missing] = [220, 30, 30]
        buf = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
        buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.errorhandler(ValidationError)
def _on_validation(exc):
    return jsonify({"ok": False, "error": str(exc)}), 400


@app.errorhandler(404)
def _on_404(exc):
    return jsonify({"ok": False, "error": getattr(exc, "description", "未找到")}), 404


@app.errorhandler(413)
def _on_413(exc):
    return jsonify({"ok": False, "error": "上传内容超过 32MB 限制"}), 413
