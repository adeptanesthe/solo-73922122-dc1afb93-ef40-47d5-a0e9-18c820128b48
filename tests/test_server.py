"""End-to-end API tests using Flask's test client and generated sample set."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from werkzeug.datastructures import MultiDict

from app import server
from app.core import CalibrationConflictError


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(server, "store", server.JobStore(tmp_path / "data"))
    server.app.config.update(TESTING=True)
    return server.app.test_client()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def make_set(n=4, h=64, w=64, offsets=(0, 12, -9, 5), seed=1):
    rng = np.random.default_rng(seed)
    base = 120 + rng.normal(0, 6, (h, w))
    out = []
    for off in offsets[:n]:
        img = np.clip(np.rint(base + off + rng.normal(0, 1.2, (h, w))), 0, 255).astype(np.uint8)
        out.append(img)
    return out


def upload(client, imgs):
    r = client.post("/api/jobs")
    jid = r.get_json()["job_id"]
    data = MultiDict([("images", (io.BytesIO(png_bytes(a)), f"img_{i}.png"))
                      for i, a in enumerate(imgs)])
    resp = client.post(f"/api/jobs/{jid}/images", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return jid, resp.get_json()


def test_upload_validation_rejects_rgb_and_size_mismatch(client):
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    r = client.post("/api/jobs")
    jid = r.get_json()["job_id"]
    resp = client.post(
        f"/api/jobs/{jid}/images",
        data={"images": (io.BytesIO(png_bytes(rgb[:, :, 0])), "a.png")},
        content_type="multipart/form-data",
    )
    # only one image -> count error first; check count boundary explicitly
    assert resp.status_code == 400
    assert "3–8" in resp.get_json()["error"]

    imgs = make_set(n=3)
    imgs[1] = np.stack([imgs[1]] * 3, axis=-1)  # RGB
    data = MultiDict()
    for i, a in enumerate(imgs):
        payload = _rgb_png(a) if a.ndim == 3 else png_bytes(a)
        data.add("images", (io.BytesIO(payload), f"img_{i}.png"))
    resp = client.post(
        f"/api/jobs/{jid}/images",
        data=data,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "灰度" in resp.get_json()["error"]


def _rgb_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_full_workflow_calibration_composite_and_download(client):
    imgs = make_set()
    jid, state0 = upload(client, imgs)
    assert state0["n"] == 4 and state0["boxes"] == [None] * 4

    calib = [4, 4, 60, 30]
    resp = client.post(f"/api/jobs/{jid}/annotation",
                       json={"calib": calib, "boxes": [None] * 4})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] and body["mode"] == "full"
    st = body["state"]
    assert st["enabled"] == [True] * 4
    assert abs(st["offsets"][1] - 12.0) < 2.0
    assert st["has_composite"]
    assert st["missing_pixels"] == 0

    # PNG endpoints serve the base map and its overlay.
    assert client.get(f"/api/jobs/{jid}/composite.png").status_code == 200
    ov = client.get(f"/api/jobs/{jid}/composite-overlay.png")
    assert ov.status_code == 200
    assert Image.open(io.BytesIO(ov.data)).mode == "RGB"


def test_incremental_mode_when_only_boxes_change(client):
    imgs = make_set()
    jid, _ = upload(client, imgs)
    calib = [4, 4, 60, 30]
    r1 = client.post(f"/api/jobs/{jid}/annotation",
                     json={"calib": calib, "boxes": [None] * 4})
    assert r1.get_json()["mode"] == "full"

    boxes = [None, [10, 10, 30, 30], None, None]
    r2 = client.post(f"/api/jobs/{jid}/annotation",
                     json={"calib": calib, "boxes": boxes})
    body2 = r2.get_json()
    assert body2["mode"] == "incremental"
    assert body2["state"]["boxes"][1] == [10, 10, 30, 30]

    # Shrink the calibration region so edges can no longer form: image-level
    # enabled flags change, which forces a full recompute.
    r3 = client.post(f"/api/jobs/{jid}/annotation",
                     json={"calib": [4, 4, 7, 7], "boxes": boxes})
    body3 = r3.get_json()
    assert body3["mode"] == "full"
    assert all(flag is False for flag in body3["state"]["enabled"][1:])


def test_missing_region_when_too_few_contributors(client):
    # Four images; occlude the same patch in three of them -> only one
    # contributor there -> 待补拍.
    imgs = make_set()
    jid, _ = upload(client, imgs)
    patch = [20, 20, 45, 45]
    boxes = [None, patch, patch, patch]
    r = client.post(f"/api/jobs/{jid}/annotation",
                    json={"calib": [4, 4, 60, 30], "boxes": boxes})
    st = r.get_json()["state"]
    assert st["missing_pixels"] == 25 * 25


def test_conflict_keeps_previous_annotation_and_base_map(client):
    imgs = make_set()
    jid, _ = upload(client, imgs)
    calib = [4, 4, 60, 30]
    r1 = client.post(f"/api/jobs/{jid}/annotation",
                     json={"calib": calib, "boxes": [None] * 4})
    good = r1.get_json()["state"]
    good_comp = client.get(f"/api/jobs/{jid}/composite.png").data

    # Monkeypatch the solver to report a contradiction on the next call.
    original = server.core.solve_offsets

    def boom(n, edges):
        raise CalibrationConflictError("模拟冲突：偏移不一致")

    server.core.solve_offsets = boom
    try:
        boxes = [None, [1, 1, 8, 8], None, None]
        r2 = client.post(f"/api/jobs/{jid}/annotation",
                         json={"calib": calib, "boxes": boxes})
    finally:
        server.core.solve_offsets = original

    assert r2.status_code == 409
    body = r2.get_json()
    assert body["conflict"] is True
    kept = body["state"]
    # Previous annotation retained: draft boxes not committed.
    assert kept["boxes"] == good["boxes"]
    assert kept["offsets"] == good["offsets"]
    # Previous base map still served.
    assert client.get(f"/api/jobs/{jid}/composite.png").data == good_comp


def test_conflict_rollback_restores_unnormalized_draft(client, monkeypatch):
    imgs = make_set()
    jid, _ = upload(client, imgs)
    calib = [4, 4, 60, 30]
    client.post(f"/api/jobs/{jid}/annotation",
                json={"calib": calib, "boxes": [None] * 4})

    def boom(n, edges):
        raise CalibrationConflictError("模拟冲突")

    monkeypatch.setattr(server.core, "solve_offsets", boom)
    # Out-of-order / out-of-bounds draft box must not be normalized into
    # job.boxes when the solver rejects the annotation.
    draft = [[50, 50, 20, 20], None, None, None]
    r = client.post(f"/api/jobs/{jid}/annotation",
                    json={"calib": calib, "boxes": draft})
    monkeypatch.undo()
    assert r.status_code == 409
    kept = r.get_json()["state"]["boxes"]
    assert kept == [None] * 4


def test_unreachable_image_is_disabled(client):
    # Build images where image 3 shares too few good calibration pixels with
    # everyone else by occluding its calibration area.
    imgs = make_set()
    jid, _ = upload(client, imgs)
    calib = [4, 4, 60, 30]
    # Occlusion covering image 3's entire calibration region: zero shared
    # good pixels -> no edge -> unreachable from baseline -> disabled.
    boxes = [None, None, None, [4, 4, 60, 30]]
    r = client.post(f"/api/jobs/{jid}/annotation",
                    json={"calib": calib, "boxes": boxes})
    st = r.get_json()["state"]
    assert st["enabled"][3] is False
    assert st["offsets"][3] is None
    assert "停用" in st["message"]


def test_recompute_endpoint_forces_full_mode(client):
    imgs = make_set()
    jid, _ = upload(client, imgs)
    calib = [4, 4, 60, 30]
    client.post(f"/api/jobs/{jid}/annotation",
                json={"calib": calib, "boxes": [None] * 4})
    boxes = [None, [10, 10, 30, 30], None, None]
    inc = client.post(f"/api/jobs/{jid}/annotation",
                      json={"calib": calib, "boxes": boxes}).get_json()
    assert inc["mode"] == "incremental"
    forced = client.post(f"/api/jobs/{jid}/recompute").get_json()
    assert forced["mode"] == "full"


def test_sample_endpoint_serves_five_images(client, monkeypatch):
    sample_dir = Path(__file__).resolve().parent.parent / "samples"
    monkeypatch.setattr(server, "SAMPLE_DIR", sample_dir)
    r = client.post("/api/jobs")
    jid = r.get_json()["job_id"]
    resp = client.post(f"/api/jobs/{jid}/samples")
    assert resp.status_code == 200
    st = resp.get_json()
    assert st["n"] == 5 and st["calib"] is not None
    assert client.get(f"/api/jobs/{jid}/images/4").status_code == 200


def test_clear_box_and_oversized_input(client):
    big = np.zeros((600, 64), dtype=np.uint8)
    r = client.post("/api/jobs")
    jid = r.get_json()["job_id"]
    resp = client.post(
        f"/api/jobs/{jid}/images",
        data={"images": (io.BytesIO(png_bytes(big)), "big.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400

    imgs = make_set(n=3)
    jid, _ = upload(client, imgs)
    r = client.post(f"/api/jobs/{jid}/annotation",
                    json={"calib": [2, 2, 40, 20],
                          "boxes": [[5, 5, 20, 20], None, None]})
    assert r.status_code == 200
    r2 = client.post(f"/api/jobs/{jid}/annotation",
                     json={"box": {"index": 0, "rect": None}})
    assert r2.status_code == 200
    assert r2.get_json()["state"]["boxes"][0] is None
