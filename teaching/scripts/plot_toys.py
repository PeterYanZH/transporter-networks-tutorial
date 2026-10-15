#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
# Teaching supplement by PeterYanZH, derived from the ideas and public
# implementation of Transporter Networks (Zeng et al., 2020).
# Upstream: https://github.com/google-research/ravens (not an official release).
"""Redraw the three saved teaching experiments without running an experiment.

Only toys/*_raw.npz and toys/toy_metrics.json are read. The raw arrays, metrics,
optimization trajectories and experimental definitions are never overwritten.
Outputs replace the three toy_*.pdf / toy_*.png figures under the same toys/.

Usage: python scripts/plot_toys.py --root teaching
Dependencies: numpy, matplotlib. No SciPy, Ravens, CNN or optimization is used.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle


NAVY = "#18344C"
TEAL = "#008B8B"
AMBER = "#E5A12A"
GRAY = "#AAB6BF"
INK = "#20313F"
LIGHT = "#E5EBEF"
SCORE = LinearSegmentedColormap.from_list(
    "saved_toy_score", ["#FFFFFF", "#BFE5E2", TEAL, NAVY]
)


def style(root: Path, font_family: str, font_dir) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "axes.titlelocation": "left", "axes.titlepad": 7,
        "axes.labelpad": 3, "axes.linewidth": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.axisbelow": True,
        "figure.constrained_layout.w_pad": 0.035,
        "figure.constrained_layout.h_pad": 0.045,
        "figure.constrained_layout.wspace": 0.05,
        "figure.constrained_layout.hspace": 0.045,
    })
    from figure_style import apply_style
    apply_style(root, font_family=font_family, font_dir=font_dir)


def panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(rf"$\mathbf{{({letter})}}$  {title}", loc="left")


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    # Fixed page size preserves the intended aspect ratio; layout stays inside
    # the canvas. Heatmaps retain discrete pixels, while text/curves are vector.
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def load(out: Path, stem: str) -> dict[str, np.ndarray]:
    with np.load(out / f"{stem}_raw.npz", allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def light_grid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=LIGHT, lw=0.6)


def image_ticks(ax: plt.Axes, xs, ys) -> None:
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


def correlation_roi(first: np.ndarray, second: np.ndarray) -> tuple[int, int, int, int]:
    """One shared square display window retaining all nonzero response pixels."""
    rows, cols = np.nonzero((first != 0) | (second != 0))
    extent = max(int(np.ptp(rows)), int(np.ptp(cols))) + 1 + 6
    extent += (extent + 1) % 2  # Keep a symmetric odd-sized display window.
    rmid, cmid = int((rows.min() + rows.max()) // 2), int((cols.min() + cols.max()) // 2)
    r0, c0 = rmid - extent // 2, cmid - extent // 2
    r0 = max(0, min(r0, first.shape[0] - extent))
    c0 = max(0, min(c0, first.shape[1] - extent))
    return r0, r0 + extent, c0, c0 + extent


def plot_correlation(out: Path, metrics: dict) -> None:
    d = load(out, "toy_correlation")
    m = metrics["correlation"]
    rotations = d["rotations_degrees"]
    angle_index = int(np.flatnonzero(rotations == m["target_angle_degrees"])[0])
    response, shifted = d["responses"][angle_index], d["shifted_responses"][angle_index]
    row, col = m["target_center_row_col"]
    dy, dx = m["shift_row_col"]
    r0, r1, c0, c1 = correlation_roi(response, shifted)
    extent = (c0 - .5, c1 - .5, r1 - .5, r0 - .5)
    scores = d["responses"].max(axis=(1, 2))
    np.testing.assert_allclose(scores, m["best_scores_by_rotation"], rtol=0, atol=1e-12)
    assert np.count_nonzero(response) == np.count_nonzero(response[r0:r1, c0:c1])
    assert np.count_nonzero(shifted) == np.count_nonzero(shifted[r0:r1, c0:c1])

    # Six panels remain in the original reading order, with a full-field
    # locator inside (b) and identical crop coordinates in (b), (c), and (d).
    fig = plt.figure(figsize=(6.8, 7.12), layout="constrained")
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, .06], height_ratios=[1, 1, .82])
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    c, e_shift = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    e, f = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    a.imshow(d["template"], cmap=SCORE, vmin=0, vmax=1, interpolation="nearest")
    panel(a, "a", "Query template")
    image_ticks(a, [0, 9, 18], [0, 9, 18])
    a.text(.98, .03, r"$19\!\times\!19$", transform=a.transAxes,
           ha="right", va="bottom", color=INK)

    b.imshow(d["target_canvas"][r0:r1, c0:c1], extent=extent,
             cmap=SCORE, vmin=0, vmax=1, interpolation="nearest")
    panel(b, "b", "Target and test shift")
    image_ticks(b, [c0, col, col + dx, c1 - 1], [r0, row, r1 - 1])
    b.plot(col, row, marker="+", color=TEAL, ms=7, mew=1.2)
    b.plot(col + dx, row + dy, marker="+", color=AMBER, ms=7, mew=1.5)
    # This arrow marks the second experiment's location, not another object.
    b.annotate("", xy=(col + dx, row - 9), xytext=(col, row - 9),
               arrowprops={"arrowstyle": "->", "color": AMBER, "lw": 1.5})
    b.text(col + dx / 2, row - 11, f"+{dx} px", color=AMBER, ha="center", va="bottom")
    locator = b.inset_axes([.66, .02, .33, .33])
    locator.imshow(d["target_canvas"], cmap=SCORE, vmin=0, vmax=1, interpolation="nearest")
    locator.add_patch(Rectangle((c0-.5, r0-.5), c1-c0, r1-r0,
                                fill=False, edgecolor=AMBER, lw=.8))
    locator.set_xticks([])
    locator.set_yticks([])
    for spine in locator.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.6)
        spine.set_color(GRAY)
    locator.text(.5, 1.03, r"$65\!\times\!65$", transform=locator.transAxes,
                 ha="center", va="bottom", fontsize=10)

    for ax, values, peak_col, letter, title in [
        (c, response, col, "c", "Response: original"),
        (e_shift, shifted, col + dx, "d", "Response: shifted"),
    ]:
        im = ax.imshow(values[r0:r1, c0:c1], extent=extent, cmap=SCORE,
                       vmin=0, vmax=1, interpolation="nearest")
        panel(ax, letter, title)
        image_ticks(ax, [c0, col, col + dx, c1 - 1], [r0, row, r1 - 1])
        ax.plot(peak_col, row, marker="+", color=AMBER, ms=7, mew=1.4)
        ax.axhline(row, color=GRAY, lw=.6, ls=(0, (3, 3)), alpha=.85)
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[1, 2]), ticks=[0, .5, 1])
    cb.set_label("Overlap", labelpad=4)
    cb.outline.set_linewidth(.6)

    x = np.arange(response.shape[1])
    e.plot(x, response[row], color=TEAL, lw=1.7, label="Original")
    e.plot(x, shifted[row], color=AMBER, lw=1.7, ls=(0, (4, 2)), label=f"Shift +{dx}")
    panel(e, "e", f"Row {row} cross-section")
    e.set(xlabel="Column", ylabel="Overlap", xlim=(0, len(x)-1), ylim=(-.02, 1.06))
    e.set_xticks([0, 16, 32, 48, 64])
    e.set_yticks([0, .5, 1])
    e.legend(loc="upper right", frameon=False, handlelength=1.5,
             handletextpad=.5, borderaxespad=.1, labelspacing=.25)
    light_grid(e)
    bars = f.bar(np.arange(len(rotations)), scores, width=.65,
                 color=[TEAL if i == angle_index else NAVY for i in range(len(rotations))])
    panel(f, "f", "Rotation search")
    f.set_xticks(np.arange(len(rotations)), [f"{int(v)}°" for v in rotations])
    f.set(xlabel="Query rotation", ylabel="Best overlap", ylim=(0, 1.17))
    f.set_yticks([0, .5, 1])
    for bar, value in zip(bars, scores):
        f.text(bar.get_x()+bar.get_width()/2, value+.025, f"{value:.3f}", ha="center", va="bottom")
    light_grid(f)
    save(fig, out, "toy_correlation")


def plot_multimodal(out: Path, metrics: dict) -> None:
    d = load(out, "toy_multimodal")
    m = metrics["multimodal"]
    centers = d["legal_island_centers_xy"]
    probability = d["learned_distribution"]
    n = probability.shape[0]
    radius = m["island_radius_pixels"]
    mean = np.asarray(m["regression_final_xy"])
    legal_mass = float(probability[d["legal_mask"]].sum())
    np.testing.assert_allclose(legal_mass, m["legal_probability_mass"], rtol=0, atol=1e-12)
    fig = plt.figure(figsize=(6.8, 5.72), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, .06], height_ratios=[1, .71])
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    c, e = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    panel(a, "a", "Same input, four labels")
    for x, y in centers:
        a.add_patch(Circle((x, y), radius, facecolor="#D6EEEB", edgecolor=TEAL, lw=1.0))
        a.plot(x, y, "o", color=TEAL, ms=3.5)
        a.text(x, y+radius+1.7, "1/4", ha="center", va="bottom", color=TEAL)
    a.plot(*mean, marker="X", ms=8, color=AMBER, zorder=5)
    a.text(mean[0], mean[1]-3.5, "MSE mean\n(illegal)", ha="center", va="top", color=NAVY)
    a.set(xlim=(-.5,n-.5), ylim=(-.5,n-.5), aspect="equal")

    im = b.imshow(probability, origin="lower", cmap=SCORE, vmin=0, vmax=.25,
                  interpolation="nearest")
    panel(b, "b", "Fitted softmax")
    for x, y in centers:
        # Rings mark legal islands; the heatmap cell itself remains exactly one
        # original grid pixel. The printed values make all four peaks readable.
        b.add_patch(Circle((x, y), radius, fill=False, edgecolor=GRAY, lw=.9))
        b.text(x, y+radius+1.7, f"{probability[y,x]:.4f}",
               ha="center", va="bottom", color=NAVY)
    b.text((n-1)/2, (n-1)/2, f"{legal_mass:.2%}\nlegal mass",
           ha="center", va="center", color=NAVY)
    for ax in (a,b):
        ax.set_xticks([0, 20, 40])
        ax.set_yticks([0, 20, 40])
        ax.set_xlabel("x (grid)")
        ax.set_ylabel("y (grid)")
    cb = fig.colorbar(im, cax=fig.add_subplot(grid[0, 2]), ticks=[0, .125, .25])
    cb.set_label("Probability", labelpad=4)
    cb.outline.set_linewidth(.6)

    mse, ce = d["mse_trace"], d["cross_entropy_trace"]
    steps = np.arange(1, len(mse))
    c.plot(steps, mse[1:], color=AMBER, lw=1.8)
    c.axhline(m["regression_loss_minimum"], color=NAVY, lw=1.1, ls=(0,(4,2)))
    c.text(30, m["regression_loss_minimum"]+.07, "Minimum = 0.5", color=NAVY)
    panel(c, "c", "Two-coordinate MSE fit")
    c.set(ylabel="Mean squared distance", ylim=(.43, 1.55))
    c.set_yticks([.5, 1, 1.5])
    e.plot(steps, ce[1:], color=TEAL, lw=1.8)
    e.axhline(m["classification_loss_infimum"], color=NAVY, lw=1.1, ls=(0,(4,2)))
    e.text(.52, .26, r"Infimum = $\log 4$", transform=e.transAxes,
           ha="left", va="bottom", color=NAVY)
    panel(e, "d", "1,681-logit CE fit")
    e.set(ylabel="Cross-entropy (nats)", ylim=(.9, 7.8))
    e.set_yticks([1, 3, 5, 7])
    for ax in (c, e):
        ax.set(xscale="log", xlim=(1, len(mse)-1), xlabel="Gradient step")
        ax.minorticks_off()
        light_grid(ax)
    save(fig, out, "toy_multimodal")


def plot_gradient(out: Path, metrics: dict) -> None:
    d = load(out, "toy_gradient")
    label = metrics["gradient"]["label_zero_based"]
    x = np.arange(len(d["probabilities"]))
    names = [chr(65+i) + ("*" if i == label else "") for i in x]
    colors = [AMBER if i == label else NAVY for i in x]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.72), layout="constrained")
    for ax, values, letter, title, ylabel in [
        (axes[0], d["probabilities"], "a", "Global softmax", "Probability"),
        (axes[1], d["analytic_gradient"], "b", r"Logit gradient: $p-y$", r"$\partial\mathcal{L}/\partial z_i$"),
    ]:
        bars = ax.bar(x, values, color=colors, width=.63)
        panel(ax, letter, title)
        ax.set_xticks(x, names)
        ax.set(xlabel="Candidate (* = label)", ylabel=ylabel)
        ax.axhline(0, color=GRAY, lw=.8)
        light_grid(ax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2,
                    value + (.016 if value >= 0 else -.075), f"{value:.3f}",
                    ha="center", va="bottom" if value >= 0 else "top")
    axes[0].set(ylim=(0, .66), yticks=[0, .2, .4, .6])
    axes[1].set(ylim=(-1.23, .73), yticks=[-1, -.5, 0, .5])
    save(fig, out, "toy_gradient")


def main() -> None:
    args = _parsed_args or parse_args()
    out = args.root / "toys"
    metrics = json.loads((out / "toy_metrics.json").read_text(encoding="utf-8"))
    style(args.root, args.font_family, args.font_dir)
    plot_correlation(out, metrics)
    plot_multimodal(out, metrics)
    plot_gradient(out, metrics)
    print(f"Redrawn 3 PDF/PNG pairs from saved arrays only: {out.resolve()}")


if __name__ == "__main__":
    main()
