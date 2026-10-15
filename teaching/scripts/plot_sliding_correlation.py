#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
# Teaching supplement by PeterYanZH, derived from the ideas and public
# implementation of Transporter Networks (Zeng et al., 2020).
# Upstream: https://github.com/google-research/ravens (not an official release).
"""Channel dot products followed by spatial summation and sliding correlation.

Manually specified illustrative features, not actual CNN activations. Only
sliding_correlation.{pdf,png,json} is produced. No inference or training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from figure_style import add_font_arguments, apply_style

NAVY = "#18344C"
TEAL = "#008B8B"
AMBER = "#C98919"
GRID = "#BBC7CF"
PALE = "#F5F8FA"
TEAL_PALE = "#D7EDE9"
AMBER_PALE = "#FBECCC"
WIDTH, HEIGHT = 16.4 / 2.54, 13.4 / 2.54

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-plots", action="store_true", help="Verify all nine integer scores and write JSON only (NumPy only)")
    add_font_arguments(parser)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    query = np.stack([[[1, 0], [2, 1]], [[2, 1], [0, 1]], [[-1, 1], [1, 0]]], axis=-1).astype(np.int64)
    feature = np.stack([
        [[3, 2, 0, 1], [0, 1, 3, 2], [2, 1, 0, 1], [1, 0, 2, 3]],
        [[0, 2, 2, 0], [2, 3, 1, 1], [1, 2, 0, 1], [0, 1, 2, 0]],
        [[2, 0, 1, 1], [-1, 2, 0, 1], [0, 1, 2, 0], [1, 0, 1, 2]],
    ], axis=-1).astype(np.int64)
    expected = np.array([[6, 17, 13], [17, 9, 11], [11, 10, 7]], dtype=np.int64)
    scores = np.empty((3, 3), dtype=np.int64)
    windows = []
    for r in range(3):
        for c in range(3):
            patch = feature[r:r + 2, c:c + 2, :]
            products = query * patch
            position_dots = products.sum(axis=2, dtype=np.int64)
            score = int(position_dots.sum(dtype=np.int64))
            scores[r, c] = score
            windows.append({"output_rc": [r, c], "input_window_rows": [r, r + 2], "input_window_columns": [c, c + 2], "window_hwc": patch.tolist(), "channel_products_hwc": products.tolist(), "per_position_channel_dot_products": position_dots.tolist(), "spatial_sum_score": score})
    independent_windows = np.lib.stride_tricks.sliding_window_view(feature, (2, 2), axis=(0, 1))
    independent_scores = np.einsum("rcduv,uvd->rc", independent_windows, query)
    assert np.array_equal(scores, expected)
    assert np.array_equal(scores, independent_scores)
    assert windows[0]["per_position_channel_dot_products"] == [[1, 2], [-1, 4]]
    assert windows[1]["per_position_channel_dot_products"] == [[6, 3], [4, 4]]
    assert int(np.dot(query[0, 0], feature[0, 0])) == 1
    collisions = []
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Rectangle
        apply_style(root, font_family=args.font_family, font_dir=args.font_dir)
        plt.rcParams.update({"text.color": NAVY, "font.size": 10, "figure.facecolor": "white"})
        fig = plt.figure(figsize=(WIDTH, HEIGHT))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT), aspect="equal")
        ax.axis("off")
        texts = []

        def label(x, y, text, size=10, color=NAVY, **kwargs):
            item = ax.text(x, y, text, fontsize=size, color=color, ha="center", va="center", **kwargs)
            texts.append(item)
            return item

        def matrix(values, cx, top, cw, ch=None, *, fills=None, inks=None, indices=False, fontsize=12):
            values = np.asarray(values)
            h, w = values.shape
            ch = cw if ch is None else ch
            left = cx - w * cw / 2
            fills = {} if fills is None else fills
            inks = {} if inks is None else inks
            for r in range(h):
                for c in range(w):
                    ax.add_patch(Rectangle((left + c * cw, top - (r + 1) * ch), cw, ch, ec=GRID, fc=fills.get((r, c), PALE), lw=.65, zorder=1))
                    label(left + (c + .5) * cw, top - (r + .5) * ch, str(values[r, c]), size=fontsize, color=inks.get((r, c), NAVY), zorder=3)
            if indices:
                for c in range(w):
                    label(left + (c + .5) * cw, top + .13, str(c), size=9, color="#596F7E")
                for r in range(h):
                    label(left - .145, top - (r + .5) * ch, str(r), size=9, color="#596F7E")
            return left, top - h * ch, w * cw, h * ch

        def vector_strings(tensor):
            return np.array([["(" + ", ".join(str(int(v)).replace("-", "−") for v in tensor[r, c]) + ")" for c in range(tensor.shape[1])] for r in range(tensor.shape[0])])

        # These labels identify a CNN output representation, not a new CNN run.
        label(1.30, 5.075, "(a) Query CNN output", size=11)
        label(4.60, 5.075, "(b) Key CNN output", size=11)
        label(1.30, 4.81, "Q: 2 × 2 × 3", size=10)
        label(4.60, 4.81, "F: 4 × 4 × 3", size=10)
        query_fills = {(r, c): "#E1EAF0" for r in range(2) for c in range(2)}
        query_fills[(0, 0)] = NAVY
        matrix(vector_strings(query), 1.30, 4.48, .82, .38, fills=query_fills, inks={(0, 0): "white"}, indices=True, fontsize=10.8)
        label(1.30, 3.45, "Each cell is a 3D vector", size=10)
        label(1.30, 3.19, "Fixed Q; no kernel flip", size=10)
        feature_fills = {(r, c): TEAL_PALE for r in range(2) for c in range(2)}
        feature_fills[(0, 0)] = NAVY
        left, bottom, width, height = matrix(vector_strings(feature), 4.60, 4.48, .72, .35, fills=feature_fills, inks={(0, 0): "white"}, indices=True, fontsize=9.6)
        ax.add_patch(Rectangle((left, 4.48 - .70), 1.44, .70, fc="none", ec=TEAL, lw=1.6, zorder=4))
        ax.add_patch(Rectangle((left + .72, 4.48 - .70), 1.44, .70, fc="none", ec=AMBER, lw=1.2, ls=(0, (3, 2)), zorder=4))
        # Channel reduction is deliberately separated from spatial reduction.
        ax.add_patch(Rectangle((.26, 2.31), WIDTH - .52, .63, fc="#F0F4F7", ec="#D3DDE4", lw=.7))
        label(1.21, 2.76, "(c) Sum channels", size=10.8)
        label(1.21, 2.49, "q · f", size=12)
        label(4.17, 2.76, "(1, 2, −1) · (3, 0, 2)", size=11)
        label(4.17, 2.49, "1 × 3 + 2 × 0 + (−1) × 2 = 1", size=10.5)
        centers = [1.13, 3.03, 5.32]
        label(centers[0], 2.05, "(d) Window (0, 0)", size=10.5)
        label(centers[1], 2.05, "(e) Window (0, 1)", size=10.5)
        label(centers[2], 2.05, "(f) Score map S", size=10.5)
        for j, (cx, color, pale) in enumerate([(centers[0], TEAL, TEAL_PALE), (centers[1], AMBER, AMBER_PALE)]):
            dots = np.array(windows[j]["per_position_channel_dot_products"])
            fills = {(r, c): pale for r in range(2) for c in range(2)}
            inks = {}
            if j == 0:
                fills[(0, 0)] = NAVY
                inks[(0, 0)] = "white"
            printed_dots = np.array([[str(int(v)).replace("-", "−") for v in row] for row in dots])
            matrix(printed_dots, cx, 1.76, .43, .32, fills=fills, inks=inks, fontsize=13)
            ax.add_patch(FancyArrowPatch((cx, 1.04), (cx, .77), arrowstyle="-|>", mutation_scale=8, lw=1, color=color))
            label(cx + .49, .91, "Σ spatial", size=9.5, color=color)
            expression = "1 + 2 − 1 + 4 = 6" if j == 0 else "6 + 3 + 4 + 4 = 17"
            label(cx, .59, expression, size=10.5, color=color)
        ax.add_patch(FancyArrowPatch((1.77, 1.43), (2.38, 1.43), arrowstyle="-|>", mutation_scale=8, lw=.9, color=NAVY))
        label(2.075, 1.69, "+1 column", size=9)
        matrix(scores, centers[2], 1.74, .39, fills={(0, 0): TEAL_PALE, (0, 1): AMBER_PALE}, fontsize=13)
        label(centers[2], .36, "Unnormalized scores", size=9.5)
        label(WIDTH / 2, .105, "Illustrative features, not model activations  ·  Single orientation  ·  Stride 1, no padding", size=9)
        # No unique-maximum annotation: the value 17 is tied at two locations.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [item.get_window_extent(renderer) for item in texts]
        collisions = []
        for i, first in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                second = boxes[j]
                overlap_w = min(first.x1, second.x1) - max(first.x0, second.x0)
                overlap_h = min(first.y1, second.y1) - max(first.y0, second.y0)
                if overlap_w > .5 and overlap_h > .5:
                    collisions.append([texts[i].get_text(), texts[j].get_text()])
        assert not collisions, collisions
        assert all(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height for box in boxes)
        fig.savefig(out / "sliding_correlation.pdf", facecolor="white")
        fig.savefig(out / "sliding_correlation.png", facecolor="white", dpi=300)
        plt.close(fig)
    metadata = {
        "kind": "Three-channel illustrative CNN-output features; no CNN execution, inference or training",
        "tensor_layout": "height, width, channel", "query_hwc": query.tolist(), "feature_map_hwc": feature.tolist(), "scores": scores.tolist(),
        "stride": 1, "padding": 0, "channels": 3, "orientations": 1, "kernel_flipped": False, "softmax_applied": False,
        "highlighted_channel_dot": {"query_vector": query[0, 0].tolist(), "key_vector": feature[0, 0].tolist(), "channel_products": (query[0, 0] * feature[0, 0]).tolist(), "channel_sum": 1},
        "index_convention": "zero-based (row, column); window upper bounds are exclusive",
        "windows": windows, "exact_integer_result_verified": True, "independent_einsum_check_passed": True,
        "maximum_score": int(scores.max()), "maximum_locations_rc": np.argwhere(scores == scores.max()).tolist(),
        "figure_size_inches": [WIDTH, HEIGHT], "figure_size_cm": [16.4, 13.4], "png_dpi": 300,
        "rendered_figure": not args.no_plots,
        "font_family": args.font_family if not args.no_plots else None, "smallest_font_points": 9 if not args.no_plots else None,
        "text_overlap_check": {"performed": not args.no_plots, "renderer": "Matplotlib Agg" if not args.no_plots else None, "collisions": collisions},
        "interpretation": "First sum the three feature-channel products at each corresponding spatial position; then sum the four resulting spatial contributions. Slide the unchanged query to obtain one scalar per output location. These are signed, unnormalized correlation scores, not probabilities.",
    }
    (out / "sliding_correlation.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_directory": str(out), "scores": scores.tolist(), "first_two_spatial_dot_maps": [windows[0]["per_position_channel_dot_products"], windows[1]["per_position_channel_dot_products"]], "text_collisions": collisions, "figure_cm": [16.4, 13.4]}, indent=2))

if __name__ == "__main__":
    main()
