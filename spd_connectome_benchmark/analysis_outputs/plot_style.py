"""Shared plotting style helpers for result figures."""

from __future__ import annotations

import numpy as np


def rename_legend_labels(labels):
    """Map internal model labels to paper-facing legend labels."""
    renamed = []
    for label in labels:
        if not isinstance(label, str):
            renamed.append(label)
            continue
        is_harmonized = False
        for suffix_text in (" (original)", "(original)"):
            if label.endswith(suffix_text):
                label = label.removesuffix(suffix_text)
                break
        for suffix_text in (" (harmonization)", "(harmonization)"):
            if label.endswith(suffix_text):
                label = label.removesuffix(suffix_text)
                is_harmonized = True
                break
        if label == "VecCorrRidge":
            label = "CorrVec"
        elif label == "Ridge":
            label = "TS Ridge"
        if is_harmonized:
            label = f"{label}$^{{*}}$"
        renamed.append(label)
    return renamed


def compress_hue_offsets(ax, factor=0.85):
    """Pull hue-dodged artists closer to their category centers."""
    centers = np.array(ax.get_xticks(), dtype=float)
    if centers.size == 0:
        return

    def _closest_center(x):
        return centers[np.argmin(np.abs(centers - x))]

    for patch in ax.patches:
        if not hasattr(patch, "set_x"):
            continue
        x = patch.get_x()
        width = patch.get_width()
        center = _closest_center(x + 0.5 * width)
        patch.set_x(center + (x - center) * factor)

    for line in ax.lines:
        xdata = line.get_xdata()
        if xdata is None or len(xdata) == 0:
            continue
        center = _closest_center(np.mean(xdata))
        line.set_xdata(center + (np.array(xdata) - center) * factor)

    for coll in ax.collections:
        offsets = getattr(coll, "get_offsets", None)
        if offsets is None:
            continue
        offs = np.array(coll.get_offsets(), copy=True)
        if offs.size == 0:
            continue
        xs = offs[:, 0]
        centers_sel = np.array([_closest_center(x) for x in xs])
        offs[:, 0] = centers_sel + (xs - centers_sel) * factor
        coll.set_offsets(offs)


def align_strip_points_to_box_centers(ax):
    """Keep jittered strip points visually aligned with their box centers."""
    box_specs = []
    for patch in ax.patches:
        try:
            bbox = patch.get_extents()
            pts = ax.transData.inverted().transform(
                [[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]]
            )
        except Exception:
            continue
        x0, x1 = sorted(pts[:, 0])
        width = x1 - x0
        if not np.isfinite(width) or width <= 0:
            continue
        box_specs.append(((x0 + x1) * 0.5, width))
    if not box_specs:
        return

    point_specs = []
    for coll in ax.collections:
        offsets = getattr(coll, "get_offsets", None)
        if offsets is None:
            continue
        offs = np.array(coll.get_offsets(), copy=True)
        if offs.size == 0 or offs.shape[1] < 2:
            continue
        xs = np.asarray(offs[:, 0], dtype=float)
        if xs.size == 0 or not np.isfinite(xs).any():
            continue
        point_specs.append((float(np.nanmean(xs)), coll, offs))
    if not point_specs:
        return

    box_specs.sort(key=lambda item: item[0])
    point_specs.sort(key=lambda item: item[0])
    for (target_center, box_width), (_, coll, offs) in zip(box_specs, point_specs):
        xs = np.asarray(offs[:, 0], dtype=float)
        current_center = float(np.nanmean(xs))
        jitter = xs - current_center
        max_jitter = max(box_width * 0.35, 1e-4)
        offs[:, 0] = target_center + np.clip(jitter, -max_jitter, max_jitter)
        coll.set_offsets(offs)
