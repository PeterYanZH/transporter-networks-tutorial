#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
# Teaching supplement by PeterYanZH, derived from the ideas and public
# implementation of Transporter Networks (Zeng et al., 2020).
# Upstream: https://github.com/google-research/ravens (not an official release).
"""Replot saved rollout tensors with explicit, auditable local views.

No environment, checkpoint, or model inference is used. The main figure shows
different source/target ROIs; the companion figure retains every image pixel.
Run from anywhere: python plot_rollouts.py --root /absolute/path/to/teaching
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from figure_style import add_font_arguments, apply_style

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    add_font_arguments(parser)
    return parser.parse_args()

_parsed_args = parse_args() if __name__ == "__main__" else None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

NAVY = "#18344C"
TEAL = "#008B8B"
AMBER = "#E5A12A"
PLACE_COLOR = "#D97732"
PIXEL_MM = 3.125
LOG_MIN = -4.0
CMAP = LinearSegmentedColormap.from_list(
    "transparent_teal", ["#F5F8F8", "#C8E7E4", "#59B8AD", TEAL, NAVY]
)


def display_rgb(image: np.ndarray) -> np.ndarray:
    """Photometric display remapping only; all spatial pixels are retained.

    Neutral table tones become light gray, fixture edges become darker gray,
    and red pixels retain a muted red appearance. This is not model input.
    """
    rgb = np.clip(image[..., :3].astype(float) / 255.0, 0.0, 1.0)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    neutral = 0.94 - 0.60 * np.clip((luminance - 0.15) / 0.55, 0, 1)
    result = np.repeat(neutral[..., None], 3, axis=-1)
    red = red_mask(image)
    # The fixed red hue makes source and placed object immediately comparable.
    # A small brightness modulation preserves variation inside the object.
    result[red] = np.clip(
        np.array([0.89, 0.31, 0.29]) + 0.12 * (luminance[red, None] - 0.4), 0, 1
    )
    return result


def red_mask(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3].astype(float)
    return (rgb[..., 0] > 100) & (rgb[..., 0] > 1.6 * rgb[..., 1]) & (rgb[..., 0] > 1.6 * rgb[..., 2])


def fixture_mask(image: np.ndarray, target_rc: np.ndarray) -> np.ndarray:
    rgb = image[..., :3].astype(float)
    gray = (np.min(rgb, axis=-1) > 70) & (np.ptp(rgb, axis=-1) < 45)
    rr, cc = np.indices(gray.shape)
    # The target is the only neutral fixture in this task. A local bound avoids
    # selecting any unrelated bright pixels near the outer workspace boundary.
    near = (rr - target_rc[0]) ** 2 + (cc - target_rc[1]) ** 2 < 65**2
    return gray & near & ~red_mask(image)


def square_roi(mask: np.ndarray, anchor: np.ndarray, *, minimum: int = 64, pad: int = 10) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    rows = np.r_[rows, int(anchor[0])]
    cols = np.r_[cols, int(anchor[1])]
    height, width = mask.shape
    rlo, rhi, clo, chi = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    side = int(min(max(minimum, rhi - rlo + 2 * pad, chi - clo + 2 * pad), height, width))
    r0 = int(np.clip(round((rlo + rhi - side) / 2), 0, height - side))
    c0 = int(np.clip(round((clo + chi - side) / 2), 0, width - side))
    return r0, r0 + side, c0, c0 + side


def overview_roi(source: tuple, target: tuple, shape: tuple) -> tuple[int, int, int, int]:
    # A contains the complete B and C windows; displayed boxes locate the zooms.
    h, w = shape
    return max(0, min(source[0], target[0]) - 4), min(h, max(source[1], target[1]) + 4), max(0, min(source[2], target[2]) - 4), min(w, max(source[3], target[3]) + 4)


def relative_log(probability: np.ndarray) -> np.ndarray:
    maximum = float(np.max(probability))
    assert np.isfinite(maximum) and maximum > 0
    return np.log10(np.maximum(np.asarray(probability) / maximum, 10.0**LOG_MIN))


def heat_overlay(probability: np.ndarray) -> np.ndarray:
    logp = relative_log(probability)
    colors = CMAP(Normalize(LOG_MIN, 0, clip=True)(logp))
    # Transparency reveals the scene; low values do not become black fields.
    colors[..., 3] = 0.93 * np.clip((logp - LOG_MIN) / (-LOG_MIN), 0, 1) ** 0.7
    return colors


def base_panel(ax: plt.Axes, image: np.ndarray, roi: tuple, *, probability: np.ndarray | None = None) -> None:
    shown = display_rgb(image)
    if probability is not None:
        shown = 0.45 + 0.55 * shown
    ax.imshow(shown, origin="upper", interpolation="nearest")
    if probability is not None:
        ax.imshow(heat_overlay(probability), origin="upper", interpolation="nearest")
    r0, r1, c0, c1 = roi
    ax.set_xlim(c0 - 0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D5DDE1")
        spine.set_linewidth(0.65)


def marker(ax: plt.Axes, point: np.ndarray, kind: str, *, size: float = 34) -> None:
    if kind == "pick":
        ax.scatter(point[1], point[0], marker="x", color=AMBER, s=size, linewidths=1.4, zorder=8)
    else:
        ax.scatter(point[1], point[0], marker="o", edgecolors=PLACE_COLOR, facecolors="none", s=size, linewidths=1.3, zorder=8)


def scale_bar(ax: plt.Axes, roi: tuple, *, full: bool = False) -> None:
    r0, r1, c0, c1 = roi
    length_pixels = 50.0 / PIXEL_MM
    # Put every scale in the white footer, never on object pixels, predictions,
    # or ROI borders. Axes-relative bar length still equals exactly 16 pixels.
    x0, y0 = .045, -.056
    width = length_pixels / (c1 - c0)
    ax.plot([x0, x0 + width], [y0, y0], transform=ax.transAxes, clip_on=False,
            color=NAVY, lw=2.0, solid_capstyle="butt", zorder=10)
    ax.text(x0 + width + .035, y0, "50 mm", transform=ax.transAxes,
            clip_on=False, ha="left", va="center", fontsize=7.5, color=NAVY)


def roi_box(ax: plt.Axes, roi: tuple, label: str, color: str) -> None:
    r0, r1, c0, c1 = roi
    ax.add_patch(Rectangle((c0 - .5, r0 - .5), c1 - c0, r1 - r0, fill=False, ec=color, lw=0.9, ls=(0, (3, 2))))
    # The figure caption maps amber to (b) and teal to (c,d). No in-scene text
    # is used, so the fixture, action markers and borders cannot be obscured.


def audit_map(probability: np.ndarray, roi: tuple, *, interpretation: str) -> dict:
    inside = np.zeros(probability.shape, dtype=bool)
    r0, r1, c0, c1 = roi
    inside[r0:r1, c0:c1] = True
    maximum = float(probability.max())
    return {
        "roi_row_start_stop_col_start_stop": list(map(int, roi)),
        "retained_sum_fraction": float(probability[inside].sum(dtype=np.float64) / probability.sum(dtype=np.float64)),
        "largest_outside_score_relative_to_global_max": float(probability[~inside].max() / maximum) if np.any(~inside) else 0.0,
        "outside_pixels_above_1_percent_of_global_max": int(np.count_nonzero(probability[~inside] > maximum * .01)),
        "interpretation": interpretation,
    }


def make_figure(data: dict, rois: list, out: Path, *, full: bool) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 10.2 if full else 5.8))
    fig.subplots_adjust(left=.067, right=.985, top=.945, bottom=.098 if not full else .067, hspace=.17 if not full else .11, wspace=.075)
    headers = ["(a) Scene", "(b) Pick response", "(c) Place response", "(d) After action"]
    for ax, title in zip(axes[0], headers):
        ax.set_title(title, fontsize=10, fontweight="medium", color=NAVY, pad=7)
    for i in range(3):
        source, target, context = rois[i]
        if full:
            source = target = context = (0, data["images"].shape[1], 0, data["images"].shape[2])
        base_panel(axes[i, 0], data["images"][i], context)
        base_panel(axes[i, 1], data["images"][i], source, probability=data["pick_maps"][i])
        base_panel(axes[i, 2], data["images"][i], target, probability=data["place_maps"][i])
        base_panel(axes[i, 3], data["after"][i], target)
        marker(axes[i, 0], data["picks"][i], "pick")
        marker(axes[i, 0], data["places"][i], "place")
        marker(axes[i, 1], data["picks"][i], "pick")
        marker(axes[i, 2], data["places"][i], "place")
        if not full:
            roi_box(axes[i, 0], source, "b", AMBER)
            roi_box(axes[i, 0], target, "c / d", TEAL)
        for j, region in enumerate([context, source, target, target]):
            scale_bar(axes[i, j], region, full=full)
        # Row labels lie outside axes, aligned by the subplot cell rather than
        # image aspect, which can differ in the compact overview column.
        pos = axes[i, 0].get_subplotspec().get_position(fig)
        fig.text(.019, (pos.y0 + pos.y1) / 2, f"Seed {int(data['seeds'][i])}", rotation=90, ha="center", va="center", fontsize=9.5, fontweight="medium", color=NAVY)
    cax = fig.add_axes([.25, .039 if not full else .025, .35, .015 if not full else .009])
    colorbar = fig.colorbar(ScalarMappable(norm=Normalize(LOG_MIN, 0), cmap=CMAP), cax=cax, orientation="horizontal", ticks=[-4, -3, -2, -1, 0])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=1)
    colorbar.outline.set_visible(False)
    fig.text(.067, .046 if not full else .029, "Relative response", ha="left", va="center", fontsize=8.5, color=NAVY)
    fig.text(.625, .046 if not full else .029, r"$\log_{10}(P/P_{\max})$", ha="left", va="center", fontsize=9, color=NAVY)
    fig.text(.80, .046 if not full else .029, "Local views" if not full else "Full workspace", ha="left", va="center", fontsize=8.5, color=TEAL)
    stem = "model_rollouts_full" if full else "model_rollouts"
    fig.savefig(out / f"{stem}.pdf", facecolor="white", bbox_inches="tight", pad_inches=.04)
    fig.savefig(out / f"{stem}.png", facecolor="white", dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main() -> None:
    args = _parsed_args or parse_args()
    root = args.root.expanduser().resolve()
    source = root / "results" / "rollouts_arrays.npz"
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in ["images", "pick_maps", "place_maps", "picks", "places", "after", "seeds"]}
    assert data["images"].shape == (3, 320, 160, 6)
    assert data["after"].shape == data["images"].shape
    assert data["pick_maps"].shape == data["place_maps"].shape == (3, 320, 160)
    assert data["picks"].shape == (3, 2) and data["places"].shape == (3, 3)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "text.color": NAVY})
    from figure_style import apply_style
    apply_style(root, font_family=args.font_family, font_dir=args.font_dir)
    rois, records = [], []
    for i in range(3):
        image = data["images"][i]
        p, q = data["picks"][i], data["places"][i, :2]
        assert tuple(np.unravel_index(np.argmax(data["pick_maps"][i]), (320, 160))) == tuple(p)
        assert tuple(np.unravel_index(np.argmax(data["place_maps"][i]), (320, 160))) == tuple(q)
        source_roi = square_roi(red_mask(image), p)
        # The placed red object identifies the target footprint without the
        # bright table-edge noise that can contaminate a gray-fixture mask.
        # Ten pixels of padding + a minimum 64-pixel window retain its fixture.
        target_roi = square_roi(red_mask(data["after"][i]), q)
        context_roi = overview_roi(source_roi, target_roi, image.shape[:2])
        rois.append((source_roi, target_roi, context_roi))
        records.append({
            "seed": int(data["seeds"][i]), "pick_rc": p.astype(int).tolist(),
            "place_rc_rotation_bin": data["places"][i].astype(int).tolist(),
            "overview_roi": list(map(int, context_roi)),
            "pick_view": audit_map(data["pick_maps"][i], source_roi, interpretation="Fraction of full pick probability mass retained by the local source view."),
            "place_view": audit_map(data["place_maps"][i], target_roi, interpretation="Fraction of summed max-over-angle scores retained; this is not marginalized placement probability mass."),
        })
    make_figure(data, rois, out, full=False)
    make_figure(data, rois, out, full=True)
    metadata = {
        "source": "results/rollouts_arrays.npz", "font_family": args.font_family, "model_inferences": 0, "pixel_size_mm": PIXEL_MM,
        "display": "Neutral RGB tones remapped to light gray with darker fixture edges; red object retained as muted red. Heat layers use opacity over the scene. Photometric display changes are not model inputs.",
        "colorbar": {"quantity": "log10(P / global maximum P of each original full map)", "limits": [-4, 0], "note": "Place map is max over the 36 rotation bins, not a marginal sum. No local ROI renormalization."},
        "cropping": "Main figure has separately chosen source and target ROIs. Source uses the initial red-object bounds; target uses placed red-object bounds plus predicted placement, with at least 10 pixels padding where workspace boundaries permit and a minimum side of 64 pixels. After-action view uses the identical target ROI. Overview contains both ROIs; amber dashed box denotes pick view (b), teal denotes target views (c,d). All scale bars are outside images. The full-workspace companion preserves all pixels and candidate responses.",
        "full_workspace_companion": "model_rollouts_full.pdf",
        "records": records,
    }
    (out / "model_rollouts_plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": ["model_rollouts.pdf", "model_rollouts.png", "model_rollouts_full.pdf", "model_rollouts_full.png", "model_rollouts_plot_metadata.json"], "records": records}, indent=2))


if __name__ == "__main__":
    main()
