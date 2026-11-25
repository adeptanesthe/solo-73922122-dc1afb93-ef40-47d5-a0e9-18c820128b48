"""Core image-processing logic for the mural-restoration base-map builder.

Glossary
--------
calib : (x0, y0, x1, y1) | None
    Shared calibration rectangle, image coordinates, half-open [x0, x1) x [y0, y1).
boxes : list of (x0, y0, x1, y1) | None
    One occlusion rectangle per image (scaffolding etc.). ``None`` means clear.
offset : float
    Exposure offset of an image relative to image 0.  A pixel of value ``v`` in
    image ``i`` is corrected to ``v - offset[i]`` so that every image matches the
    exposure of the baseline image 0.

Conflict policy
---------------
Pairwise constraints are propagated outwards from image 0 over a graph whose
edges are median gray differences on >= MIN_PAIR_PIXELS shared good pixels.
If an edge reaches an image whose offset is already known and the two values
disagree by more than CONFLICT_TOL gray levels, the whole calibration is
rejected as contradictory: the caller must keep the previous annotation /
base map instead of committing the new one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

MIN_SIDE = 1
MAX_SIDE = 512
MIN_IMAGES = 3
MAX_IMAGES = 8
MIN_PAIR_PIXELS = 16
SAT_LOW = 0
SAT_HIGH = 255
#: Two independently derived offsets of the same image are considered
#: contradictory when they differ by more than this many gray levels.
CONFLICT_TOL = 2.0
#: A corrected pixel needs at least this many contributing images.
MIN_CONTRIBUTORS = 2
SUPPORTED_MODES = ("L",)
SUPPORTED_FORMAT = "PNG"


class ValidationError(ValueError):
    """Raised when an uploaded image does not meet the input contract."""


class CalibrationConflictError(ValueError):
    """Raised when pairwise exposure constraints contradict each other."""


@dataclass
class Edge:
    a: int
    b: int
    offset_b_from_a: float
    npix: int


@dataclass
class CalibrationResult:
    offsets: list[Optional[float]]
    enabled: list[bool]
    edges: list[Edge] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


def validate_image(arr: np.ndarray, expected_shape: Optional[tuple[int, int]] = None) -> None:
    """Validate an 8-bit grayscale image array (H, W), uint8, side <= 512."""
    if not isinstance(arr, np.ndarray):
        raise ValidationError("图像必须是 NumPy 数组")
    if arr.dtype != np.uint8:
        raise ValidationError("仅支持 8 位灰度 PNG（uint8）")
    if arr.ndim != 2:
        raise ValidationError("仅支持单通道灰度图")
    h, w = arr.shape
    if h < MIN_SIDE or w < MIN_SIDE or h > MAX_SIDE or w > MAX_SIDE:
        raise ValidationError(f"边长需在 {MIN_SIDE}–{MAX_SIDE} 像素之间，当前为 {w}x{h}")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValidationError(
            f"尺寸不一致：期望 {expected_shape[1]}x{expected_shape[0]}，"
            f"实际 {w}x{h}"
        )


def normalize_box(box, w: int, h: int):
    """Normalize a rectangle to clipped half-open int coords, or None.

    Empty / fully outside rectangles collapse to None ("cleared").
    """
    if box is None:
        return None
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    except (TypeError, ValueError):
        raise ValidationError("矩形坐标必须是四个数值")
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return (x0, y0, x1, y1)


def rect_mask(box, h: int, w: int) -> np.ndarray:
    """Boolean (H, W) mask, True where the rectangle occludes."""
    m = np.zeros((h, w), dtype=bool)
    if box is not None:
        x0, y0, x1, y1 = box
        m[y0:y1, x0:x1] = True
    return m


def occlusion_union(boxes, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    for b in boxes:
        if b is not None:
            x0, y0, x1, y1 = b
            m[y0:y1, x0:x1] = True
    return m


def _calib_slice(calib, h: int, w: int):
    if calib is None:
        return None
    x0, y0, x1, y1 = calib
    return slice(y0, y1), slice(x0, x1)


def pair_offset(
    images: np.ndarray,
    occ_masks: np.ndarray,
    i: int,
    j: int,
    calib,
) -> Optional[Edge]:
    """Median gray difference (j - i) over shared good calibration pixels.

    Good = same position, unoccluded in both images, unsaturated in both.
    Returns None when fewer than MIN_PAIR_PIXELS such pixels exist.
    """
    if calib is None:
        return None
    sl = _calib_slice(calib, images.shape[1], images.shape[2])
    gi = images[i][sl]
    gj = images[j][sl]
    good = (~occ_masks[i][sl]) & (~occ_masks[j][sl])
    good &= (gi != SAT_LOW) & (gi != SAT_HIGH)
    good &= (gj != SAT_LOW) & (gj != SAT_HIGH)
    npix = int(good.sum())
    if npix < MIN_PAIR_PIXELS:
        return None
    diff = gj.astype(np.float64)[good] - gi.astype(np.float64)[good]
    return Edge(a=i, b=j, offset_b_from_a=float(np.median(diff)), npix=npix)


def build_all_edges(
    images: np.ndarray, occ_masks: np.ndarray, calib
) -> list[Edge]:
    """All usable pairwise constraints over the calibration region."""
    n = images.shape[0]
    edges: list[Edge] = []
    if calib is None:
        return edges
    for i in range(n):
        for j in range(i + 1, n):
            e = pair_offset(images, occ_masks, i, j, calib)
            if e is not None:
                edges.append(e)
    return edges


def solve_offsets(n: int, edges: list[Edge]) -> CalibrationResult:
    """BFS from image 0; detect contradictory constraints and unreachable images.

    Edge semantics: ``offset(b) = offset(a) + e.offset_b_from_a``.
    Image 0 is the baseline with offset 0.  Images unreachable from 0 are
    disabled.  A constraint that implies an offset differing by more than
    CONFLICT_TOL from an already known one raises CalibrationConflictError.
    """
    adj: list[list[Edge]] = [[] for _ in range(n)]
    for e in edges:
        adj[e.a].append(e)
        adj[e.b].append(e)

    offsets: list[Optional[float]] = [None] * n
    parent: list[Optional[int]] = [None] * n
    offsets[0] = 0.0
    conflicts: list[str] = []
    used: list[Edge] = []

    from collections import deque

    q = deque([0])
    seen = {0}
    while q:
        a = q.popleft()
        for e in adj[a]:
            if e.a == a:
                b, implied = e.b, offsets[a] + e.offset_b_from_a
            else:
                # Traversing edge backwards: offset(a) = offset(b) + d
                # => offset(b) = offset(a) - d
                b, implied = e.a, offsets[a] - e.offset_b_from_a
            if b not in seen:
                seen.add(b)
                offsets[b] = float(implied)
                parent[b] = a
                used.append(e)
                q.append(b)
            elif abs(offsets[b] - implied) > CONFLICT_TOL:
                msg = (
                    f"图 {a + 1} 与图 {b + 1} 的约束互相矛盾："
                    f"已有偏移 {offsets[b]:+.2f}，本边推算 {implied:+.2f}"
                    f"（相差 {abs(offsets[b] - implied):.2f} 灰度，"
                    f"基于 {e.npix} 个校准像素）"
                )
                conflicts.append(msg)

    if conflicts:
        raise CalibrationConflictError("；".join(conflicts))

    enabled = [o is not None for o in offsets]
    return CalibrationResult(offsets=offsets, enabled=enabled, edges=used)


def build_value_stack(
    images: np.ndarray,
    occ_masks: np.ndarray,
    offsets: list[Optional[float]],
    enabled: list[bool],
) -> np.ndarray:
    """(N, H, W) float64 stack: corrected gray level, NaN where unusable.

    A value is unusable when the image is disabled, the pixel is occluded,
    or the corrected gray level falls outside [0, 255].
    """
    n, h, w = images.shape
    stack = np.full((n, h, w), np.nan, dtype=np.float64)
    for i in range(n):
        if not enabled[i] or offsets[i] is None:
            continue
        corrected = images[i].astype(np.float64) - offsets[i]
        valid = (~occ_masks[i]) & (corrected >= 0.0) & (corrected <= 255.0)
        stack[i][valid] = corrected[valid]
    return stack


def _median_from_stack(stack: np.ndarray, pixel_mask: Optional[np.ndarray] = None):
    """Per-pixel median and contributor count of the value stack.

    ``pixel_mask`` optionally restricts the computed pixels (incremental mode).
    Pixels with zero contributors get NaN median.
    """
    with np.errstate(all="ignore"):
        if pixel_mask is None:
            med = np.nanmedian(stack, axis=0)
            cnt = np.sum(~np.isnan(stack), axis=0)
            return med, cnt
        h, w = stack.shape[1:]
        med = np.full((h, w), np.nan, dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.int16)
        import warnings

        idx = np.where(pixel_mask)
        sub = stack[:, idx[0], idx[1]]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice")
            med_sub = np.nanmedian(sub, axis=0)
        cnt_sub = np.sum(~np.isnan(sub), axis=0)
        med[idx] = med_sub
        cnt[idx] = cnt_sub
        return med, cnt


def composite(
    images: np.ndarray,
    occ_masks: np.ndarray,
    offsets: list[Optional[float]],
    enabled: list[bool],
    pixel_mask: Optional[np.ndarray] = None,
):
    """Median composite.

    Returns (result uint8 HxW, missing bool HxW, count int HxW).
    Pixels with fewer than MIN_CONTRIBUTORS usable values are marked missing
    ("待补拍") and rendered as 0 in the result.
    """
    stack = build_value_stack(images, occ_masks, offsets, enabled)
    med, cnt = _median_from_stack(stack, pixel_mask)
    missing = cnt < MIN_CONTRIBUTORS
    if pixel_mask is None:
        result = np.zeros(med.shape, dtype=np.uint8)
        good = ~missing
        # Corrected values that survive are guaranteed within [0, 255].
        result[good] = np.rint(med[good]).astype(np.uint8)
    else:
        # Partial (incremental) run: leave untouched pixels at 0 with count 0;
        # the caller merges only pixels flagged in pixel_mask.
        result = np.zeros(med.shape, dtype=np.uint8)
        good = pixel_mask & ~missing
        result[good] = np.rint(med[good]).astype(np.uint8)
    return result, missing, cnt.astype(np.int32)


def affected_pixel_mask(boxes_before, boxes_after, h: int, w: int) -> np.ndarray:
    """Pixels whose occlusion status changed between two box lists."""
    return occlusion_union(boxes_before, h, w) != occlusion_union(boxes_after, h, w)


def offsets_unchanged(
    off_before: list[Optional[float]],
    en_before: list[bool],
    off_after: list[Optional[float]],
    en_after: list[bool],
) -> bool:
    if en_before != en_after:
        return False
    for a, b in zip(off_before, off_after):
        if (a is None) != (b is None):
            return False
        if a is not None and not math.isclose(a, b, abs_tol=1e-12):
            return False
    return True
