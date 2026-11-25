"""Unit tests for the core calibration / composite algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from app import core
from app.core import (
    CalibrationConflictError,
    MIN_PAIR_PIXELS,
    ValidationError,
)


def make_images(n=4, h=64, w=64, seed=0, offsets=None, base_level=120):
    rng = np.random.default_rng(seed)
    base = base_level + rng.normal(0, 5, size=(h, w))
    if offsets is None:
        offsets = [0.0, 10.0, -8.0, 4.0][:n]
    imgs = []
    for off in offsets:
        imgs.append(np.clip(np.rint(base + off + rng.normal(0, 1.5, (h, w))), 0, 255).astype(np.uint8))
    return np.stack(imgs), offsets


def test_validate_image_rules():
    good = np.zeros((64, 64), dtype=np.uint8)
    core.validate_image(good)  # no raise
    with pytest.raises(ValidationError):
        core.validate_image(np.zeros((64, 64), dtype=np.float64))
    with pytest.raises(ValidationError):
        core.validate_image(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValidationError):
        core.validate_image(np.zeros((513, 64), dtype=np.uint8))
    with pytest.raises(ValidationError):
        core.validate_image(np.zeros((32, 32), dtype=np.uint8), expected_shape=(64, 64))


def test_normalize_box_clipping_and_empty():
    assert core.normalize_box(None, 10, 10) is None
    assert core.normalize_box([-5, -5, 20, 20], 10, 10) == (0, 0, 10, 10)
    assert core.normalize_box([5, 5, 5, 8], 10, 10) is None  # zero width
    assert core.normalize_box([8, 2, 3, 9], 10, 10) == (3, 2, 8, 9)  # swapped


def test_pair_offset_requires_16_good_pixels_and_median_diff():
    imgs, offsets = make_images(n=2, h=40, w=40)
    occ = np.zeros((2, 40, 40), dtype=bool)
    calib = (0, 0, 40, 40)
    edge = core.pair_offset(imgs, occ, 0, 1, calib)
    assert edge is not None
    assert edge.npix >= MIN_PAIR_PIXELS
    assert abs(edge.offset_b_from_a - (offsets[1] - offsets[0])) < 1.5

    # Occlude all but 10 columns of shared pixels (400 -> 400? no: 10*40=400... use 1 col -> 40)
    # Use a small patch so fewer than 16 good pixels remain.
    occ2 = np.zeros((2, 40, 40), dtype=bool)
    occ2[:, :, 3:] = True   # only 3 columns remain -> 3*40 = 120 still >=16
    occ2[:, 3:, :] = True   # intersect -> 3x3 = 9 good pixels < 16
    assert core.pair_offset(imgs, occ2, 0, 1, calib) is None


def test_saturated_pixels_excluded():
    h = w = 40
    a = np.full((h, w), 100, dtype=np.uint8)
    b = np.full((h, w), 255, dtype=np.uint8)  # saturated everywhere
    imgs = np.stack([a, b])
    occ = np.zeros((2, h, w), dtype=bool)
    assert core.pair_offset(imgs, occ, 0, 1, (0, 0, w, h)) is None

    # 20 unsaturated pixels with true difference +7.
    b2 = b.copy()
    b2[:, :20] = 107
    edge = core.pair_offset(np.stack([a, b2]), occ, 0, 1, (0, 0, w, h))
    assert edge is not None and edge.npix == 20 * h
    assert edge.offset_b_from_a == pytest.approx(7.0, abs=0.5)


def test_solve_offsets_chain_and_unreachable():
    # Edges: 0-1, 1-2 only. Image 3 isolated -> disabled.
    edges = [
        core.Edge(0, 1, 10.0, 100),
        core.Edge(1, 2, -5.0, 100),
    ]
    res = core.solve_offsets(4, edges)
    assert res.offsets[0] == 0.0
    assert res.offsets[1] == pytest.approx(10.0)
    assert res.offsets[2] == pytest.approx(5.0)
    assert res.offsets[3] is None
    assert res.enabled == [True, True, True, False]


def test_conflicting_constraints_raise_and_message():
    edges = [
        core.Edge(0, 1, 10.0, 100),
        core.Edge(0, 1, 30.0, 100),  # contradicts first, > CONFLICT_TOL
    ]
    with pytest.raises(CalibrationConflictError):
        core.solve_offsets(2, edges)


def test_consistent_transitive_constraints_pass():
    # Two paths 0->2: direct +12 and via 1 (5 + 7 = 12), within tolerance.
    edges = [
        core.Edge(0, 1, 5.0, 100),
        core.Edge(1, 2, 7.0, 100),
        core.Edge(0, 2, 12.5, 100),
    ]
    res = core.solve_offsets(3, edges)
    assert res.enabled == [True, True, True]
    assert abs(res.offsets[2] - 12.0) <= 2.0


def test_composite_median_and_missing_flag():
    # 3 identical images, one occluded region each in different places.
    imgs, _ = make_images(n=3, h=48, w=48, seed=3)
    occ = np.zeros((3, 48, 48), dtype=bool)
    occ[0, 0:10, 0:10] = True
    occ[1, 0:10, 0:10] = True  # only image 2 contributes -> 待补拍
    offs = [0.0, 0.0, 0.0]
    res, missing, cnt = core.composite(imgs, occ, offs, [True] * 3)
    assert res.dtype == np.uint8
    assert missing[0:10, 0:10].all()
    assert not missing[20:30, 20:30].any()
    assert cnt[5, 5] == 1 and cnt[25, 25] == 3


def test_corrected_values_out_of_range_excluded():
    # Image 1 shifted +200: most corrected pixels go negative -> excluded,
    # forcing 待补拍 where nothing else remains.
    a = np.full((32, 32), 100, dtype=np.uint8)
    b = np.full((32, 32), 250, dtype=np.uint8)
    imgs = np.stack([a, b])
    occ = np.zeros((2, 32, 32), dtype=bool)
    res, missing, cnt = core.composite(imgs, occ, [0.0, 200.0], [True, True])
    # b corrected = 50, actually in range -> 2 contributors. Shift harder:
    imgs2 = np.stack([a, np.full((32, 32), 255, dtype=np.uint8)])
    res2, missing2, cnt2 = core.composite(imgs2, occ, [0.0, 255.0], [True, True])
    # corrected second image = 0 everywhere -> in range inclusive; force -1:
    res3, missing3, cnt3 = core.composite(imgs2, occ, [0.0, 256.0], [True, True])
    assert missing3.all()
    assert not missing2.all()


def test_disabled_images_do_not_contribute():
    imgs, _ = make_images(n=3, h=32, w=32, seed=7)
    occ = np.zeros((3, 32, 32), dtype=bool)
    res, missing, cnt = core.composite(imgs, occ, [0.0, None, 0.0], [True, False, True])
    assert cnt.max() <= 2


def test_incremental_recompute_matches_full_and_touches_only_changed_pixels():
    imgs, offsets = make_images(n=4, h=64, w=64, seed=11)
    boxes_before = [None, (0, 0, 20, 20), None, None]
    boxes_after = [None, (0, 0, 20, 20), (40, 40, 60, 60), None]
    h = w = 64
    occ_b = np.stack([core.rect_mask(b, h, w) for b in boxes_before])
    occ_a = np.stack([core.rect_mask(b, h, w) for b in boxes_after])
    offs = [0.0, offsets[1], offsets[2], offsets[3]]
    enabled = [True] * 4

    full_b, miss_b, _ = core.composite(imgs, occ_b, offs, enabled)
    full_a, miss_a, _ = core.composite(imgs, occ_a, offs, enabled)

    changed = core.affected_pixel_mask(boxes_before, boxes_after, h, w)
    assert changed[50, 50] and not changed[5, 5]
    inc_res, inc_miss, _ = core.composite(imgs, occ_a, offs, enabled, pixel_mask=changed)
    merged = full_b.copy()
    merged[changed] = inc_res[changed]
    merged_miss = miss_b.copy()
    merged_miss[changed] = inc_miss[changed]
    assert np.array_equal(merged, full_a)
    assert np.array_equal(merged_miss, miss_a)
    # Outside the change mask the partial run reports no contributors,
    # i.e. the caller must not merge those positions.
    assert inc_miss[5, 5] and not inc_miss[50, 50]


def test_offsets_unchanged_helper():
    assert core.offsets_unchanged([0.0, 1.0], [True, True], [0.0, 1.0], [True, True])
    assert not core.offsets_unchanged([0.0, 1.0], [True, True], [0.0, 2.0], [True, True])
    assert not core.offsets_unchanged([0.0, None], [True, False], [0.0, 1.0], [True, True])


def test_end_to_end_offsets_recover_true_values():
    imgs, offsets = make_images(n=4, h=80, w=80, seed=5)
    occ = np.zeros((4, 80, 80), dtype=bool)
    calib = (5, 5, 75, 40)
    edges = core.build_all_edges(imgs, occ, calib)
    assert len(edges) == 6
    res = core.solve_offsets(4, edges)
    for got, want in zip(res.offsets, offsets):
        assert abs(got - want) < 2.0
