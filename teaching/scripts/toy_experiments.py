#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PeterYanZH
# Teaching supplement by PeterYanZH, derived from the ideas and public
# implementation of Transporter Networks (Zeng et al., 2020).
# Upstream: https://github.com/google-research/ravens (not an official release).
"""Three small, auditable teaching experiments for Transporter Networks.

This is NOT a Ravens evaluation: there is no CNN, no learned visual descriptor,
and no test set. Figures explain operators and losses under explicit toy data.

Usage: python toy_experiments.py --out /path/to/output
Requires Python >=3.10 and NumPy. Matplotlib is needed only for figures.
Use --no-plots for pure numerical examples without plotting dependencies.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from figure_style import add_font_arguments, apply_style


def correlate2d(canvas, template, *, mode="same", boundary="fill"):
    """NumPy cross-correlation for odd-sized, zero-padded toy templates."""
    if mode != "same" or boundary != "fill":
        raise ValueError("Only zero-padded same-sized output is supported")
    rh, rw = (size // 2 for size in template.shape)
    if any(size % 2 != 1 for size in template.shape):
        raise ValueError("Toy templates must have odd height and width")
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(canvas, ((rh, rh), (rw, rw))), template.shape
    )
    return np.einsum("rcuv,uv->rc", windows, template, optimize=True)


def load_plotting():
    global matplotlib, plt, Circle, SCORE_CMAP
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle
    SCORE_CMAP = LinearSegmentedColormap.from_list(
        "teaching_score", ["#FFFFFF", "#BFE5E2", TEAL, NAVY]
    )


NAVY = "#18344C"
TEAL = "#008B8B"
AMBER = "#E5A12A"
GRAY = "#AAB6BF"
INK = "#20313F"



def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRAY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, out: Path, stem: str) -> None:
    # PDF retains vector labels, curves and bars; pixel-valued heatmaps are
    # rendered as embedded images rather than pretending to be vector data.
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def insert_patch(canvas: np.ndarray, patch: np.ndarray, row: int, col: int) -> None:
    radius = patch.shape[0] // 2
    assert patch.shape[0] == patch.shape[1] and patch.shape[0] % 2 == 1
    region = canvas[row - radius : row + radius + 1, col - radius : col + radius + 1]
    assert region.shape == patch.shape
    region[:] = patch


def shift_right(array: np.ndarray, pixels: int) -> np.ndarray:
    shifted = np.zeros_like(array)
    shifted[..., pixels:] = array[..., :-pixels]
    return shifted


def grid_axis(ax: plt.Axes, size: int, *, image: bool = True) -> None:
    ax.set_xticks([0, size // 2, size - 1])
    ax.set_yticks([0, size // 2, size - 1])
    ax.set_xlabel("column" if image else "x (grid)")
    ax.set_ylabel("row" if image else "y (grid)")


def correlation_experiment(out: Path, render: bool = True) -> dict:
    # One asymmetric L, with a nonzero margin on every side. The identity
    # descriptor is simply the binary pixel value, not a learned CNN feature.
    template = np.zeros((19, 19), dtype=np.float64)
    template[4:15, 4:7] = 1.0
    template[12:15, 4:14] = 1.0
    rotations = np.arange(0, 360, 90, dtype=int)
    rotated_templates = np.stack([np.rot90(template, k) for k in range(4)])
    canvas = np.zeros((65, 65), dtype=np.float64)
    target_row, target_col, target_angle = 32, 22, 90
    insert_patch(canvas, rotated_templates[1], target_row, target_col)
    dx = 12
    shifted_canvas = shift_right(canvas, dx)
    mass = float(template.sum())
    responses = np.stack(
        [correlate2d(canvas, t, mode="same", boundary="fill") / mass for t in rotated_templates]
    )
    shifted_responses = np.stack(
        [correlate2d(shifted_canvas, t, mode="same", boundary="fill") / mass for t in rotated_templates]
    )
    best = tuple(int(x) for x in np.unravel_index(np.argmax(responses), responses.shape))
    shifted_best = tuple(
        int(x) for x in np.unravel_index(np.argmax(shifted_responses), shifted_responses.shape)
    )
    assert best == (1, target_row, target_col), best
    assert shifted_best == (1, target_row, target_col + dx), shifted_best
    assert np.count_nonzero(np.isclose(responses, 1.0, atol=1e-12)) == 1
    residual = shifted_responses[..., dx:] - responses[..., :-dx]
    max_error = float(np.max(np.abs(residual)))
    assert max_error < 1e-12, max_error
    yy, xx = np.indices(canvas.shape)
    outside_radius3 = (yy - target_row) ** 2 + (xx - target_col) ** 2 > 3**2
    same_angle = responses[1]
    nonpeak = np.ones(canvas.shape, dtype=bool)
    nonpeak[target_row, target_col] = False
    neighbor_max = float(same_angle[nonpeak].max())
    sidelobe_max = float(same_angle[outside_radius3].max())
    assert 0.0 < sidelobe_max < neighbor_max < 1.0
    angle_scores = responses.max(axis=(1, 2))

    if render:
        fig, axes = plt.subplots(3, 2, figsize=(6.4, 9.3), layout="constrained")
        ax = axes[0, 0]
        ax.imshow(template, cmap=SCORE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("(a) Binary L query")
        grid_axis(ax, 19)
        ax = axes[0, 1]
        ax.imshow(canvas, cmap=SCORE_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("(b) Independent target")
        grid_axis(ax, 65)
        ax.annotate(
            "+12 px",
            xy=(target_col + dx, target_row),
            xytext=(45, 15),
            ha="center",
            color=AMBER,
            arrowprops={"arrowstyle": "->", "color": AMBER, "lw": 1.8},
        )
        ax.plot(target_col + dx, target_row, marker="+", color=AMBER, ms=8, mew=1.8)
        for ax, response, col, title in [
            (axes[1, 0], responses[1], target_col, "(c) Score at 90 degrees"),
            (axes[1, 1], shifted_responses[1], target_col + dx, "(d) Score after +12 px"),
        ]:
            im = ax.imshow(response, cmap=SCORE_CMAP, vmin=0, vmax=1, interpolation="nearest")
            ax.plot(col, target_row, marker="+", color=AMBER, ms=9, mew=2)
            ax.set_title(title)
            grid_axis(ax, 65)
            cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, ticks=[0, 0.5, 1])
            cb.ax.tick_params(labelsize=10)
        ax = axes[2, 0]
        ax.plot(np.arange(65), responses[1, target_row], color=TEAL, lw=2, label="Original")
        ax.plot(
            np.arange(65), shifted_responses[1, target_row], color=AMBER, lw=2, ls="--", label="Shift +12"
        )
        ax.set(title="(e) Peak has finite width", xlabel="column", ylabel="overlap / query area")
        ax.set(xlim=(0, 64), ylim=(-0.025, 1.06))
        ax.set_xticks([0, 16, 32, 48, 64])
        ax.legend(loc="upper right", frameon=False, handlelength=1.5, borderaxespad=0.1)
        ax.grid(axis="y", alpha=0.16)
        ax = axes[2, 1]
        bars = ax.bar(
            np.arange(4), angle_scores, width=0.62, color=[NAVY, TEAL, NAVY, NAVY], edgecolor="none"
        )
        ax.set_xticks(np.arange(4), [str(x) for x in rotations])
        ax.set(title="(f) Discrete rotation search", xlabel="query angle (degrees)", ylabel="best overlap")
        ax.set_ylim(0, 1.16)
        for bar, score in zip(bars, angle_scores):
            ax.text(bar.get_x() + bar.get_width() / 2, score + 0.03, f"{score:.2f}", ha="center", fontsize=10)
        ax.grid(axis="y", alpha=0.16)
        save_figure(fig, out, "toy_correlation")
    np.savez_compressed(
        out / "toy_correlation_raw.npz",
        template=template,
        rotated_templates=rotated_templates,
        target_canvas=canvas,
        shifted_target_canvas=shifted_canvas,
        rotations_degrees=rotations,
        responses=responses,
        shifted_responses=shifted_responses,
        overlap_residual=residual,
        side_lobe_mask=outside_radius3,
    )
    return {
        "description": "Binary identity descriptors; normalized raw cross-correlation, not softmax probability.",
        "canvas_shape": list(canvas.shape),
        "query_shape": list(template.shape),
        "query_nonzero_pixels": int(mass),
        "rotations_degrees": rotations.tolist(),
        "rotation_rule": "np.rot90; counter-clockwise when displayed with image rows increasing downward",
        "target_angle_degrees": target_angle,
        "target_center_row_col": [target_row, target_col],
        "shift_row_col": [0, dx],
        "best_angle_row_col": [int(rotations[best[0]]), best[1], best[2]],
        "shifted_best_angle_row_col": [int(rotations[shifted_best[0]]), shifted_best[1], shifted_best[2]],
        "peak_score": float(responses[best]),
        "best_scores_by_rotation": angle_scores.tolist(),
        "largest_other_position_same_rotation": neighbor_max,
        "sidelobe_exclusion_radius_pixels": 3,
        "sidelobe_max_same_rotation": sidelobe_max,
        "translation_overlap_max_absolute_error": max_error,
        "translation_comparison_domain": "all four orientations, all rows, translated columns 12:65 vs original 0:53",
        "interpretation_limit": "Exact integer translation for this unclipped identity-feature construction; no CNN or rotation-equivariance claim.",
    }


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def cross_entropy(logits: np.ndarray, target: np.ndarray) -> float:
    top = float(np.max(logits))
    logsumexp = top + float(np.log(np.exp(logits - top).sum()))
    return float(logsumexp - np.dot(target, logits))


def multimodal_experiment(out: Path, render: bool = True) -> dict:
    n = 41
    seed = 7
    steps = 5000
    lr_mse, lr_ce = 0.05, 1.0
    # All four labels share exactly the same input. Islands are defined as
    # legal by construction; the blank center is explicitly NOT legal.
    centers_xy = np.array([[10, 10], [30, 10], [10, 30], [30, 30]], dtype=int)
    radius = 3
    yy, xx = np.indices((n, n))
    valid = np.zeros((n, n), dtype=bool)
    for x, y in centers_xy:
        valid |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    target_2d = np.zeros((n, n), dtype=np.float64)
    target_2d[centers_xy[:, 1], centers_xy[:, 0]] = 0.25
    target = target_2d.ravel()
    labels_normalized = centers_xy.astype(np.float64) / 20.0 - 1.0
    mean_normalized = labels_normalized.mean(axis=0)
    mean_xy = (mean_normalized + 1.0) * 20.0
    assert not bool(valid[int(mean_xy[1]), int(mean_xy[0])])
    mse_floor = float(np.mean(np.sum((labels_normalized - mean_normalized) ** 2, axis=1)))
    ce_floor = float(np.log(4.0))

    prediction = np.array([-0.8, 0.6], dtype=np.float64)
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 0.02, n * n)
    initial_logits = logits.copy()
    regression_trace = np.empty((steps + 1, 2), dtype=np.float64)
    mse_trace = np.empty(steps + 1, dtype=np.float64)
    ce_trace = np.empty(steps + 1, dtype=np.float64)
    center_mass_trace = np.empty(steps + 1, dtype=np.float64)
    for step in range(steps + 1):
        regression_trace[step] = prediction
        mse_trace[step] = np.mean(np.sum((labels_normalized - prediction) ** 2, axis=1))
        ce_trace[step] = cross_entropy(logits, target)
        probs = stable_softmax(logits)
        center_mass_trace[step] = probs[target > 0].sum()
        if step < steps:
            # Derivatives of full-batch losses; all four labels occur once.
            prediction -= lr_mse * 2.0 * (prediction - mean_normalized)
            logits -= lr_ce * (probs - target)
    probability_2d = probs.reshape(n, n)
    learned_xy = (prediction + 1.0) * 20.0
    center_probabilities = probability_2d[centers_xy[:, 1], centers_xy[:, 0]]
    legal_mass = float(probability_2d[valid].sum())
    center_mass = float(center_probabilities.sum())
    top4_xy = [[int(i % n), int(i // n)] for i in np.argsort(probs)[-4:]]
    assert np.linalg.norm(prediction - mean_normalized) < 1e-12
    assert abs(float(mse_trace[-1]) - mse_floor) < 1e-12
    assert center_mass > 0.99 and legal_mass >= center_mass
    assert set(map(tuple, top4_xy)) == set(map(tuple, centers_xy.tolist()))
    assert np.ptp(center_probabilities) < 1e-5
    assert ce_trace[-1] > ce_floor and ce_trace[-1] - ce_floor < 0.01
    assert np.all(np.diff(ce_trace) <= 1e-12)
    assert np.all(np.diff(mse_trace) <= 1e-12)

    if render:
        fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.5), layout="constrained")
        ax = axes[0, 0]
        for x, y in centers_xy:
            ax.add_patch(Circle((x, y), radius, facecolor="#D6EEEB", edgecolor=TEAL, lw=1.4))
        ax.scatter(centers_xy[:, 0], centers_xy[:, 1], color=TEAL, s=25, zorder=4)
        ax.scatter([mean_xy[0]], [mean_xy[1]], marker="X", s=85, color=AMBER, zorder=5)
        ax.text(20, 27.0, "MSE mean\n(illegal)", ha="center", va="top", fontsize=10, color=NAVY)
        ax.text(20, 2, "Four legal islands", ha="center", va="bottom", fontsize=10, color=TEAL)
        ax.set_title("(a) Four labels, one input")
        ax.set(xlim=(-0.5, 40.5), ylim=(-0.5, 40.5), aspect="equal")
        grid_axis(ax, n, image=False)
        ax = axes[0, 1]
        im = ax.imshow(probability_2d, origin="lower", cmap=SCORE_CMAP, vmin=0, vmax=0.25, interpolation="nearest")
        for x, y in centers_xy:
            ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=GRAY, lw=1.0))
        ax.text(20, 2, f"Legal mass: {legal_mass:.2%}", ha="center", va="bottom", fontsize=10, color=NAVY)
        ax.set_title("(b) Learned softmax: four peaks")
        grid_axis(ax, n, image=False)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, ticks=[0, 0.125, 0.25])
        cb.ax.tick_params(labelsize=10)
        ax = axes[1, 0]
        iterations = np.arange(steps + 1)
        ax.plot(iterations[1:], mse_trace[1:], color=AMBER, lw=2)
        ax.axhline(mse_floor, color=NAVY, ls="--", lw=1.2)
        ax.text(30, mse_floor + 0.05, "Minimum = 0.5", fontsize=10, color=NAVY)
        ax.set(xscale="log", xlim=(1, steps), ylim=(0.43, 1.55))
        ax.set(title="(c) Two-parameter MSE fit", xlabel="gradient steps", ylabel="mean squared distance")
        ax.grid(alpha=0.16)
        ax = axes[1, 1]
        ax.plot(iterations[1:], ce_trace[1:], color=TEAL, lw=2)
        ax.axhline(ce_floor, color=NAVY, ls="--", lw=1.2)
        ax.text(14, ce_floor + 0.32, "Infimum = log 4", fontsize=10, color=NAVY)
        ax.set(xscale="log", xlim=(1, steps), ylim=(0.9, 7.8))
        ax.set(title="(d) 1,681-parameter CE fit", xlabel="gradient steps", ylabel="cross-entropy (nats)")
        ax.grid(alpha=0.16)
        save_figure(fig, out, "toy_multimodal")
    np.savez_compressed(
        out / "toy_multimodal_raw.npz",
        legal_mask=valid,
        legal_island_centers_xy=centers_xy,
        label_coordinates_normalized=labels_normalized,
        target_distribution=target_2d,
        learned_distribution=probability_2d,
        initial_logits=initial_logits.reshape(n, n),
        final_logits=logits.reshape(n, n),
        regression_trace_normalized=regression_trace,
        mse_trace=mse_trace,
        cross_entropy_trace=ce_trace,
        four_label_mass_trace=center_mass_trace,
    )
    return {
        "description": "Same observation, four equally frequent point labels inside four isolated legal contact disks.",
        "grid_shape": [n, n],
        "seed": seed,
        "steps": steps,
        "island_centers_xy": centers_xy.tolist(),
        "island_radius_pixels": radius,
        "legal_grid_point_count": int(valid.sum()),
        "labels_per_full_batch": 4,
        "coordinate_normalization": "x_normalized = x_grid / 20 - 1 (same for y)",
        "regression_parameters": 2,
        "regression_initial_normalized_xy": regression_trace[0].tolist(),
        "regression_learning_rate": lr_mse,
        "regression_loss_definition": "mean over labels of squared Euclidean distance, summed over x,y",
        "regression_analytic_mean_xy": mean_xy.tolist(),
        "regression_final_xy": learned_xy.tolist(),
        "regression_final_normalized_error_to_mean": float(np.linalg.norm(prediction - mean_normalized)),
        "regression_final_is_legal": bool(valid[int(round(learned_xy[1])), int(round(learned_xy[0]))]),
        "regression_loss_initial": float(mse_trace[0]),
        "regression_loss_final": float(mse_trace[-1]),
        "regression_loss_minimum": mse_floor,
        "classification_parameters": n * n,
        "classification_logit_initialization": "independent Gaussian, mean=0, standard deviation=0.02",
        "classification_learning_rate": lr_ce,
        "classification_loss_initial": float(ce_trace[0]),
        "classification_loss_final": float(ce_trace[-1]),
        "classification_loss_infimum": ce_floor,
        "classification_excess_loss": float(ce_trace[-1] - ce_floor),
        "four_label_probability_mass": center_mass,
        "legal_probability_mass": legal_mass,
        "illegal_probability_mass": float(probability_2d[~valid].sum()),
        "probability_at_four_labels": center_probabilities.tolist(),
        "top_four_xy": top4_xy,
        "interpretation_limit": "Direct fitted coordinates vs free grid logits; no CNN, no held-out inputs, unequal parameter counts, no generalization or ConvMLP ablation claim. Finite-logit softmax cannot reach exact zero mass elsewhere.",
    }


def gradient_experiment(out: Path, render: bool = True) -> dict:
    logits = np.array([2.0, 1.0, 0.0, -0.5, 0.5], dtype=np.float64)
    label = 3
    target = np.zeros(5)
    target[label] = 1
    probabilities = stable_softmax(logits)
    analytic = probabilities - target
    epsilon = 1e-6
    finite_diff = np.empty(5)
    for i in range(5):
        direction = np.zeros(5)
        direction[i] = epsilon
        finite_diff[i] = (
            cross_entropy(logits + direction, target) - cross_entropy(logits - direction, target)
        ) / (2 * epsilon)
    error = float(np.max(np.abs(analytic - finite_diff)))
    assert error < 1e-8, error
    assert abs(float(analytic.sum())) < 1e-12
    assert analytic[label] < 0 and np.all(np.delete(analytic, label) > 0)
    lr = 0.5
    updated_logits = logits - lr * analytic
    updated_probabilities = stable_softmax(updated_logits)
    assert updated_probabilities[label] > probabilities[label]
    assert cross_entropy(updated_logits, target) < cross_entropy(logits, target)
    labels = ["A", "B", "C", "D*", "E"]
    colors = [NAVY, NAVY, NAVY, AMBER, NAVY]
    if render:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), layout="constrained")
        for ax, values, title, ylabel in [
            (axes[0], probabilities, "(a) One global softmax", "probability"),
            (axes[1], analytic, "(b) Gradient = p - y", "loss gradient w.r.t. logit"),
        ]:
            bars = ax.bar(np.arange(5), values, color=colors, width=0.65)
            ax.set_xticks(np.arange(5), labels)
            ax.set(title=title, xlabel="candidate (* = label)", ylabel=ylabel)
            ax.axhline(0, color=GRAY, lw=0.9)
            ax.grid(axis="y", alpha=0.16)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + (0.025 if val >= 0 else -0.04),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=10,
                )
        axes[0].set_ylim(0, 0.7)
        axes[1].set_ylim(-1.19, 0.79)
        save_figure(fig, out, "toy_gradient")
    np.savez_compressed(
        out / "toy_gradient_raw.npz",
        initial_logits=logits,
        target=target,
        probabilities=probabilities,
        analytic_gradient=analytic,
        finite_difference_gradient=finite_diff,
        updated_logits=updated_logits,
        updated_probabilities=updated_probabilities,
    )
    return {
        "description": "Five mutually exclusive candidates in one global softmax; one positive label.",
        "logits": logits.tolist(),
        "label_zero_based": label,
        "probabilities": probabilities.tolist(),
        "analytic_gradient_p_minus_y": analytic.tolist(),
        "finite_difference_gradient": finite_diff.tolist(),
        "finite_difference_epsilon": epsilon,
        "gradient_check_max_absolute_error": error,
        "gradient_sum": float(analytic.sum()),
        "one_step_learning_rate": lr,
        "updated_probabilities": updated_probabilities.tolist(),
        "loss_before": cross_entropy(logits, target),
        "loss_after": cross_entropy(updated_logits, target),
        "interpretation_limit": "A derivative identity for one softmax over all candidates; not five independent binary classifiers.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "toys", help="directory for figures, metrics and raw arrays")
    parser.add_argument("--no-plots", action="store_true", help="Run numerical examples and save NPZ/JSON only; requires NumPy only")
    add_font_arguments(parser)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        load_plotting()
        set_style()
        apply_style(font_family=args.font_family, font_dir=args.font_dir)
    metrics = {
        "schema_version": 1,
        "scope": "Three teaching-only constructions; no CNN training and no claim of paper-level replication.",
        "rendered_figures": not args.no_plots,
        "font_family": args.font_family if not args.no_plots else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": None if args.no_plots else matplotlib.__version__,
        },
        "correlation": correlation_experiment(args.out, render=not args.no_plots),
        "multimodal": multimodal_experiment(args.out, render=not args.no_plots),
        "gradient": gradient_experiment(args.out, render=not args.no_plots),
    }
    (args.out / "toy_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_directory": str(args.out.resolve()),
                "correlation_translation_error": metrics["correlation"]["translation_overlap_max_absolute_error"],
                "correlation_sidelobe_max": metrics["correlation"]["sidelobe_max_same_rotation"],
                "multimodal_regression_is_legal": metrics["multimodal"]["regression_final_is_legal"],
                "multimodal_legal_mass": metrics["multimodal"]["legal_probability_mass"],
                "gradient_check_max_error": metrics["gradient"]["gradient_check_max_absolute_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
