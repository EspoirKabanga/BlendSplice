#!/usr/bin/env python3
"""Combined sequence-logo figures for direct evaluation revision.

Layout modes (--layout):
  all-models [default] GAN / VAE / Diffusion sections; each with Donor then Acceptor
               (3 species rows each) and 402 bp | 2002 bp condition columns.
  by-model     One figure per model (GAN / VAE / Diffusion separately).
  horizontal   Wide figure(s) with Real|Blend|No-Blend side-by-side in each cell.
  stacked      Legacy vertical stack (Real over Blend over No-Blend).

  demo         Mockup-style A/B panel (stacked logos + JS tracks) for one condition.

Examples:
    python direct_revision_sequence_logos_combined_figure.py
    python direct_revision_sequence_logos_combined_figure.py --a4-landscape
    python direct_revision_sequence_logos_combined_figure.py --font-scale 1.2 --fig-height 30

Styling (all layouts):
  - Exon/intron window: bp −20…−1 and +1…+10 (junction guide at 0)
  - logomaker information transform; 0–2 bit y-scale; low-IC fade
  - Mockup color scheme, vpad/width, left+bottom spines only
  - Grid spacing/fonts match the 6×6 mockup reference (scaled for 18 columns)
"""

import os
import argparse

import logomaker
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from direct_analysis import compute_pwm
from direct_revision_analysis import (
    CONTEXT_CFG,
    CONTEXTS,
    MODELS,
    OUTPUT_ROOT,
    SPECIES,
    SEQ_TYPES,
    load_real_positives,
    load_synthetic,
    species_display,
)

BASES = list("ACGT")
# Display labels: exon −20…−1, intron +1…+10 (no 0 — junction sits between them).
LOGO_POS_BP = list(range(-20, 0)) + list(range(1, 11))
N_LOGO = len(LOGO_POS_BP)
JUNCTION_IDX = 19.5  # between index 19 (−1) and index 20 (+1)

COLOR_SCHEME = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff9900", "T": "#d62728"}
X_TICK_IDX = [0, 5, 10, 15, JUNCTION_IDX, 24, N_LOGO - 1]
X_TICK_LABELS = ["-20", "-15", "-10", "-5", "0", "5", "10"]

LOGO_VPAD = 0.02
LOGO_WIDTH = 0.92

Y_BITS = (0.0, 2.0)
Y_TICKS = [0, 1, 2]
IC_THRESHOLD = 0.12
LOW_IC_ALPHA = 0.15

# 6×6 mockup reference layout (grid36_mockup.py).
MOCKUP_FIGSIZE = (16.0, 7.2)
MOCKUP_GRID = dict(
    left=0.075,
    right=0.994,
    top=0.895,
    bottom=0.105,
    hspace=0.30,
    wspace=0.09,
)
MOCKUP_FONTS = dict(
    col_fs=12.5,
    row_fs=12.0,
    block_fs=14.0,
    axis_fs=10.0,
    x_tick_fs=11.0,
    xlab_fs=13.5,
    title_pad=7,
)

# Dotted separators for Donor|Acceptor and 402|2002 dividers.
SEPARATOR_KW = dict(color="0.60", linewidth=1.0, linestyle=":", clip_on=False)
# Dashed separators between GAN / VAE / Diffusion blocks and 402 | 2002 bp columns.
MODEL_BLOCK_SEPARATOR_KW = dict(
    color="0.35", linewidth=1.2, linestyle="--", clip_on=False
)

FONT_KEYS = (
    "col_fs",
    "row_fs",
    "model_fs",
    "block_fs",
    "axis_fs",
    "x_tick_fs",
    "xlab_fs",
    "site_fs",
    "species_fs",
    "dataset_fs",
)

MODEL_LABEL = {"GAN": "GAN", "VAE": "VAE", "DIFFUSION": "Diffusion"}
SITE_LABEL = {"donor": "Donor", "acceptor": "Acceptor"}
DATASET_LABEL = {
    "real": "Real",
    "blend": "Blend",
    "no_blend": "No-Blend",
}
DATASET_KEYS = ["real", "blend", "no_blend"]

COL_PAIRS = [(model, context) for model in MODELS for context in CONTEXTS]
# by-model column order: 402 Real, 402 Blend, 402 No-Blend, 2002 Real, ...
BY_MODEL_COLS = [(context, ds) for context in CONTEXTS for ds in DATASET_KEYS]
ALL_MODEL_COLS = [
    (model, context, ds_key)
    for model in MODELS
    for context in CONTEXTS
    for ds_key in DATASET_KEYS
]
COLS_PER_MODEL = len(BY_MODEL_COLS)
ROW_KEYS = [(seq_type, species) for seq_type in SEQ_TYPES for species in SPECIES]
ALL_MODELS_ROW_KEYS = [
    (seq_type, species, model)
    for model in MODELS
    for seq_type, species in ROW_KEYS
]
ROWS_PER_MODEL = len(ROW_KEYS)
ROWS_PER_SITE = len(SPECIES)
MODEL_BLOCK_GAP_CM = 0.5
MODEL_BLOCK_GAP_IN = MODEL_BLOCK_GAP_CM / 2.54

STYLE = {
    "all_models": {
        "figsize": (
            MOCKUP_FIGSIZE[0],
            MOCKUP_FIGSIZE[1] * 3 + (len(MODELS) - 1) * MODEL_BLOCK_GAP_IN,
        ),
        **MOCKUP_GRID,
        "col_fs": 15.0,
        "row_fs": 14.0,
        "block_fs": 16.0,
        "axis_fs": 13.0,
        "x_tick_fs": 14.0,
        "xlab_fs": 16.0,
        "left": 0.13,
        "model_fs": 16.0,
        "model_label_x": 0.008,
        "site_label_x": 0.034,
        "model_block_gap_cm": MODEL_BLOCK_GAP_CM,
    },
    "by_model": {
        "figsize": MOCKUP_FIGSIZE,
        **MOCKUP_GRID,
        **MOCKUP_FONTS,
        "site_fs": MOCKUP_FONTS["block_fs"],
    },
    "horizontal": {
        "figsize": (24.0, 8.0),
        "left": 0.09,
        "right": 0.995,
        "top": 0.90,
        "bottom": 0.14,
        "hspace": 0.05,
        "wspace": 0.24,
        "inner_wspace": 0.10,
        "col_fs": 18.0,
        "species_fs": 19.0,
        "site_fs": 22.0,
        "dataset_fs": 16.0,
        "axis_fs": 18.0,
    },
    "stacked": {
        "figsize": (18.0, 20.0),
        "left": 0.11,
        "right": 0.995,
        "top": 0.965,
        "bottom": 0.035,
        "donor_bottom": 0.52,
        "acceptor_top": 0.48,
        "block_hspace": 0.04,
        "block_wspace": 0.30,
        "inner_hspace": 0.14,
        "col_fs": 17.0,
        "site_fs": 19.0,
        "species_fs": 18.0,
        "dataset_fs": 16.0,
        "axis_fs": 17.0,
    },
}


def scaled_style(style_key, font_scale=1.0, figsize=None):
    """Return a copy of a layout style with optional font scaling."""
    style = dict(STYLE[style_key])
    if figsize is not None:
        style["figsize"] = figsize
    if font_scale != 1.0:
        for key in FONT_KEYS:
            if key in style:
                style[key] *= font_scale
    return style


def pwm_relative(pwm_df, context):
    cfg = CONTEXT_CFG[context]
    lo, hi = cfg["logo_region"]
    sub = pwm_df.iloc[lo : hi + 1].copy()
    sub.index = np.arange(lo, hi + 1) - cfg["motif_pos"]
    return sub


def _mockup_bp_to_pwm_index(bp_label):
    """Map display bp (−20…−1, +1…+10) to PWM index (0 = first motif base)."""
    return bp_label if bp_label < 0 else bp_label - 1


def pwm_prob_window(pwm_df, context):
    """Slice PWM to the 30-position exon/intron mockup window."""
    rel = pwm_relative(pwm_df, context)
    rows = []
    for bp in LOGO_POS_BP:
        rel_pos = _mockup_bp_to_pwm_index(bp)
        if rel_pos in rel.index:
            row = rel.loc[rel_pos, BASES].values.astype(float)
            row = np.clip(row, 1e-12, None)
            row = row / row.sum()
        else:
            row = np.full(4, 0.25)
        rows.append(row)
    prob = pd.DataFrame(rows, columns=BASES, index=range(N_LOGO))
    return prob


def prob_to_info(prob_df):
    info = logomaker.transform_matrix(
        prob_df, from_type="probability", to_type="information"
    )
    ic = info.sum(axis=1)
    fade = np.where(ic < IC_THRESHOLD, LOW_IC_ALPHA, 1.0)
    return info.mul(fade, axis=0)


def draw_site_logo(ax, pwm_df, context, seq_type, style, panel):
    """Draw one sequence logo (mockup styling) from real PWM data."""
    prob = pwm_prob_window(pwm_df, context)
    info_df = prob_to_info(prob)

    logo = logomaker.Logo(
        info_df,
        ax=ax,
        color_scheme=COLOR_SCHEME,
        vpad=LOGO_VPAD,
        width=LOGO_WIDTH,
    )
    logo.style_spines(visible=False)
    logo.style_spines(spines=["left", "bottom"], visible=True)

    fs = style["axis_fs"]
    x_fs = style.get("x_tick_fs", fs)
    ax.set_xlim(-0.5, N_LOGO - 0.5)
    ax.set_ylim(*Y_BITS)
    ax.set_yticks(Y_TICKS)
    ax.axvline(
        JUNCTION_IDX,
        color="0.4",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        zorder=5,
        clip_on=True,
    )
    ax.grid(False)

    if panel.get("show_xticklabels", False):
        ax.set_xticks(X_TICK_IDX)
        ax.set_xticklabels(X_TICK_LABELS, fontsize=x_fs)
        ax.tick_params(axis="x", labelsize=x_fs, length=2.5, bottom=True, labelbottom=True)
    else:
        ax.set_xticks([])
        ax.tick_params(axis="x", bottom=True, labelbottom=False, length=2.5)

    if panel.get("show_yticklabels", False):
        ax.tick_params(axis="y", labelsize=fs, length=2.5, left=True, labelleft=True)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=True, labelleft=False, length=2.5)

    if panel.get("show_ylabel"):
        ax.set_ylabel(panel["ylabel"], fontsize=style["row_fs"])
    else:
        ax.set_ylabel("")

    if panel.get("show_xlabel", False):
        xlab_fs = style.get("xlab_fs", fs)
        ax.set_xlabel("Distance from splice site (bp)", fontsize=xlab_fs, labelpad=6)
    else:
        ax.set_xlabel("")

    ax.set_title("")


def _js_div(P, Q):
    """Jensen–Shannon divergence in bits, per position."""
    P = np.clip(P, 1e-12, None)
    Q = np.clip(Q, 1e-12, None)
    M = 0.5 * (P + Q)

    def kl(A, B):
        return np.sum(A * np.log2(A / B), axis=1)

    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def _draw_corner_title(ax, title, fs):
    ax.text(
        0.012,
        0.86,
        title,
        transform=ax.transAxes,
        fontsize=fs,
        fontweight="bold",
        va="top",
    )


def _draw_js_track(ax, vals, color, label, style, show_x=False):
    x = np.arange(N_LOGO)
    ax.fill_between(x, 0, vals, step="mid", color=color, alpha=0.85, linewidth=0)
    ymax = max(0.4, float(np.max(vals)) * 1.15)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, round(ymax * 0.75, 1)])
    ax.set_xlim(-0.5, N_LOGO - 0.5)
    ax.axvline(
        JUNCTION_IDX,
        color="0.35",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fs = style["axis_fs"]
    ax.set_ylabel("JS\n(bits)", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.text(
        0.012,
        0.92,
        label,
        transform=ax.transAxes,
        fontsize=fs + 0.5,
        fontweight="bold",
        va="top",
        color=color,
    )
    if show_x:
        ax.set_xticks(X_TICK_IDX)
        ax.set_xticklabels(X_TICK_LABELS)
        ax.set_xlabel("Distance from splice site (bp)", fontsize=fs)
    else:
        ax.set_xticks([])


def create_demo_figure(
    out_path,
    cache,
    dpi,
    model="DIFFUSION",
    context="402",
    species="human",
    seq_type="donor",
):
    """Mockup-style A/B panel (stacked logos + JS tracks) for one condition."""
    style = scaled_style("by_model", font_scale=1.0)
    style["axis_fs"] = 11.0
    data = cache[(context, model, species, seq_type)]["pwm"]
    real_prob = pwm_prob_window(data["real"], context).values
    blend_prob = pwm_prob_window(data["blend"], context).values
    nb_prob = pwm_prob_window(data["no_blend"], context).values
    js_blend = _js_div(real_prob, blend_prob)
    js_noblend = _js_div(real_prob, nb_prob)

    fig = plt.figure(figsize=(17.0, 9.0))
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.16,
        left=0.055,
        right=0.985,
        top=0.86,
        bottom=0.10,
    )

    gs_a = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0], hspace=0.18)
    for i, (ds_key, title) in enumerate(
        [("real", "Real"), ("blend", "Blend"), ("no_blend", "No-Blend")]
    ):
        ax = fig.add_subplot(gs_a[i])
        panel = {
            "show_xticklabels": i == 2,
            "show_yticklabels": True,
            "show_xlabel": i == 2,
        }
        draw_site_logo(ax, data[ds_key], context, seq_type, style, panel)
        ax.set_ylabel("bits", fontsize=style["axis_fs"])
        _draw_corner_title(ax, title, style["axis_fs"] + 1)

    fig.text(
        0.27,
        0.895,
        "A   Cleaned stacked logos",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )

    gs_b = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer[1], height_ratios=[2.0, 1.0, 1.0], hspace=0.22
    )
    ax_ref = fig.add_subplot(gs_b[0])
    panel = {"show_xticklabels": False, "show_yticklabels": True, "show_xlabel": False}
    draw_site_logo(ax_ref, data["real"], context, seq_type, style, panel)
    ax_ref.set_ylabel("bits", fontsize=style["axis_fs"])
    _draw_corner_title(ax_ref, "Real (reference)", style["axis_fs"] + 1)

    _draw_js_track(
        fig.add_subplot(gs_b[1]),
        js_blend,
        "#1b7837",
        "Blend  vs Real",
        style,
    )
    _draw_js_track(
        fig.add_subplot(gs_b[2]),
        js_noblend,
        "#b2182b",
        "No-Blend  vs Real",
        style,
        show_x=True,
    )

    fig.text(
        0.76,
        0.895,
        "B   Reference logo + divergence tracks",
        fontsize=12,
        fontweight="bold",
        ha="center",
    )
    fig.suptitle(
        f"{species_display(species)}  ·  {SITE_LABEL[seq_type]}  ·  "
        f"{MODEL_LABEL[model]} ({context} bp)",
        fontsize=11,
        y=0.97,
    )
    _save(fig, out_path, dpi)


def load_pwm_data(context, model, species, seq_type, sample_size):
    print(f"  {context} bp | {model} | {species} {seq_type}")
    real_seqs = load_real_positives(species, seq_type, context, sample_size)
    blend_seqs = load_synthetic(model, species, seq_type, context, 0.5, sample_size)
    no_blend_seqs = load_synthetic(model, species, seq_type, context, 0.0, sample_size)
    n = min(len(real_seqs), len(blend_seqs), len(no_blend_seqs))
    if sample_size:
        n = min(n, sample_size)
    real_seqs = real_seqs[:n]
    blend_seqs = blend_seqs[:n]
    no_blend_seqs = no_blend_seqs[:n]
    return {
        "pwm": {
            "real": compute_pwm(real_seqs),
            "blend": compute_pwm(blend_seqs),
            "no_blend": compute_pwm(no_blend_seqs),
        }
    }


def _load_cache(sample_size):
    cache = {}
    for context in CONTEXTS:
        for model in MODELS:
            for species in SPECIES:
                for seq_type in SEQ_TYPES:
                    key = (context, model, species, seq_type)
                    cache[key] = load_pwm_data(
                        context, model, species, seq_type, sample_size
                    )
    return cache


def _pwm_for(cache, context, model, species, seq_type, ds_key):
    return cache[(context, model, species, seq_type)]["pwm"][ds_key]


def _col_title(context, ds_key):
    return f"{context} bp\n{DATASET_LABEL[ds_key]}"


def _dataset_col_title(ds_key):
    return DATASET_LABEL[ds_key]


def _save(fig, out_path, dpi):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def _add_site_block_labels(fig, style, row_axes=None, row_keys=None):
    """DONOR / ACCEPTOR bracket labels on the far left (mockup style)."""
    fs = style.get("block_fs", style.get("model_fs", 14.0))
    x = 0.010
    if row_axes is not None and row_keys is not None:
        donor_ys, acceptor_ys = [], []
        for ri, row_key in enumerate(row_keys):
            seq_type = row_key[0]
            pos = row_axes[ri][0].get_position()
            y_mid = pos.y0 + pos.height / 2
            if seq_type == "donor":
                donor_ys.append(y_mid)
            else:
                acceptor_ys.append(y_mid)
        if donor_ys:
            fig.text(
                x,
                float(np.mean(donor_ys)),
                "DONOR",
                rotation=90,
                fontsize=fs,
                fontweight="bold",
                va="center",
                ha="center",
                color="#333",
                transform=fig.transFigure,
            )
        if acceptor_ys:
            fig.text(
                x,
                float(np.mean(acceptor_ys)),
                "ACCEPTOR",
                rotation=90,
                fontsize=fs,
                fontweight="bold",
                va="center",
                ha="center",
                color="#333",
                transform=fig.transFigure,
            )
        return
    fig.text(
        x,
        0.70,
        "DONOR",
        rotation=90,
        fontsize=fs,
        fontweight="bold",
        va="center",
        ha="center",
        color="#333",
    )
    fig.text(
        0.010,
        0.30,
        "ACCEPTOR",
        rotation=90,
        fontsize=fs,
        fontweight="bold",
        va="center",
        ha="center",
        color="#333",
    )


def _add_figure_xlabel(fig, style, row_axes=None):
    """Shared x-axis label centered under the grid (or below the last logo row)."""
    xlab_fs = style.get("xlab_fs", style["axis_fs"])
    if row_axes is not None:
        bot = row_axes[-1][0].get_position()
        right = row_axes[-1][-1].get_position()
        x_mid = (bot.x0 + right.x1) / 2
        y = max(bot.y0 - 0.022, 0.012)
        fig.text(
            x_mid,
            y,
            "Distance from splice site (bp)",
            ha="center",
            va="top",
            fontsize=xlab_fs,
            transform=fig.transFigure,
        )
        return
    fig.text(
        0.5,
        0.018,
        "Distance from splice site (bp)",
        ha="center",
        fontsize=xlab_fs,
        transform=fig.transFigure,
    )


def _add_model_group_headers(fig, row_axes, style):
    """Model names centered above each 6-column block."""
    for mi, model in enumerate(MODELS):
        ci0 = mi * COLS_PER_MODEL
        ci1 = ci0 + COLS_PER_MODEL - 1
        pos_l = row_axes[0][ci0].get_position()
        pos_r = row_axes[0][ci1].get_position()
        x_mid = (pos_l.x0 + pos_r.x1) / 2
        y_top = pos_l.y1 + 0.045
        fig.text(
            x_mid,
            y_top,
            MODEL_LABEL[model],
            ha="center",
            va="bottom",
            fontsize=style["model_fs"],
            fontweight="bold",
            transform=fig.transFigure,
        )


def _add_model_separators(fig, row_axes, style):
    """Vertical lines between model column groups."""
    y_bot = row_axes[-1][0].get_position().y0
    y_top = row_axes[0][0].get_position().y1
    for mi in range(1, len(MODELS)):
        ci = mi * COLS_PER_MODEL
        pos_prev = row_axes[0][ci - 1].get_position()
        pos_curr = row_axes[0][ci].get_position()
        x_sep = (pos_prev.x1 + pos_curr.x0) / 2
        fig.add_artist(
            Line2D(
                [x_sep, x_sep],
                [y_bot - 0.01, y_top + 0.06],
                transform=fig.transFigure,
                **SEPARATOR_KW,
            )
        )


def _draw_logo_grid(fig, block, cache, style, model_filter=None):
    """Shared 6-row logo grid; model_filter=None draws all models (18 cols)."""
    if model_filter is None:
        col_spec = ALL_MODEL_COLS
    else:
        col_spec = [(model_filter, ctx, ds) for ctx, ds in BY_MODEL_COLS]

    n_rows = len(ROW_KEYS)
    n_cols = len(col_spec)
    row_axes = []

    for ri, (seq_type, species) in enumerate(ROW_KEYS):
        col_axes = []
        for ci in range(n_cols):
            model, context, ds_key = col_spec[ci]
            ax = fig.add_subplot(block[ri, ci])
            panel = {
                "show_xticklabels": ri == n_rows - 1,
                "show_yticklabels": ci == 0,
                "show_xlabel": False,
                "show_ylabel": ci == 0,
                "ylabel": f"{species_display(species)}\nbits",
            }
            draw_site_logo(
                ax,
                _pwm_for(cache, context, model, species, seq_type, ds_key),
                context,
                seq_type,
                style,
                panel,
            )
            if ri == 0:
                ax.set_title(
                    _col_title(context, ds_key),
                    fontsize=style["col_fs"],
                    fontweight="bold",
                    pad=style.get("title_pad", 7),
                )
            col_axes.append(ax)
        row_axes.append(col_axes)

    return row_axes


def _all_models_row_ylabel(species):
    return f"{species_display(species)}\nbits"


def _outer_model_hspace(style, figsize):
    """GridSpec hspace between model blocks for a fixed physical gap (cm)."""
    gap_cm = style.get("model_block_gap_cm", MODEL_BLOCK_GAP_CM)
    gap_fig = (gap_cm / 2.54) / figsize[1]
    usable = style["top"] - style["bottom"]
    n_blocks = len(MODELS)
    block_h = (usable - (n_blocks - 1) * gap_fig) / n_blocks
    if block_h <= 0:
        return 0.05
    return gap_fig / block_h


def _draw_all_models_stacked_grid(fig, cache, style):
    """Nested grid: GAN / VAE / Diffusion blocks with fixed gaps; 6 cols each."""
    figsize = fig.get_size_inches()
    n_cols = len(BY_MODEL_COLS)
    n_blocks = len(MODELS)
    outer = gridspec.GridSpec(
        n_blocks,
        1,
        figure=fig,
        left=style["left"],
        right=style["right"],
        top=style["top"],
        bottom=style["bottom"],
        height_ratios=[1] * n_blocks,
        hspace=_outer_model_hspace(style, figsize),
    )
    row_axes = []

    for mi, model in enumerate(MODELS):
        inner = gridspec.GridSpecFromSubplotSpec(
            ROWS_PER_MODEL,
            n_cols,
            subplot_spec=outer[mi, 0],
            hspace=style["hspace"],
            wspace=style["wspace"],
        )
        for ri_local, (seq_type, species) in enumerate(ROW_KEYS):
            col_axes = []
            is_last_row = mi == n_blocks - 1 and ri_local == ROWS_PER_MODEL - 1
            for ci, (context, ds_key) in enumerate(BY_MODEL_COLS):
                ax = fig.add_subplot(inner[ri_local, ci])
                panel = {
                    "show_xticklabels": is_last_row,
                    "show_yticklabels": ci == 0,
                    "show_xlabel": False,
                    "show_ylabel": ci == 0,
                    "ylabel": _all_models_row_ylabel(species),
                }
                draw_site_logo(
                    ax,
                    _pwm_for(cache, context, model, species, seq_type, ds_key),
                    context,
                    seq_type,
                    style,
                    panel,
                )
                if mi == 0 and ri_local == 0:
                    ax.set_title(
                        _dataset_col_title(ds_key),
                        fontsize=style["col_fs"],
                        fontweight="bold",
                        pad=style.get("title_pad", 7),
                    )
                col_axes.append(ax)
            row_axes.append(col_axes)

    return row_axes


def _add_context_group_headers(fig, row_axes, header_row, style):
    """402 bp / 2002 bp headers above Real | Blend | No-Blend columns."""
    pos0 = row_axes[header_row][0].get_position()
    y_top = pos0.y1 + 0.030
    fs = style["col_fs"]
    for context, ci0, ci2 in (("402", 0, 2), ("2002", 3, 5)):
        x_mid = (
            row_axes[header_row][ci0].get_position().x0
            + row_axes[header_row][ci2].get_position().x1
        ) / 2
        fig.text(
            x_mid,
            y_top,
            f"{context} bp",
            ha="center",
            va="bottom",
            fontsize=fs,
            fontweight="bold",
            transform=fig.transFigure,
        )


def _add_all_models_hierarchy_labels(fig, row_axes, style):
    """Model section labels and Donor/Acceptor sub-labels on the left margin."""
    model_x = style.get("model_label_x", 0.008)
    site_x = style.get("site_label_x", 0.034)
    for mi, model in enumerate(MODELS):
        row_start = mi * ROWS_PER_MODEL
        row_end = row_start + ROWS_PER_MODEL - 1
        pos_top = row_axes[row_start][0].get_position()
        pos_bot = row_axes[row_end][0].get_position()
        y_model = (pos_bot.y0 + pos_top.y1) / 2
        fig.text(
            model_x,
            y_model,
            MODEL_LABEL[model],
            rotation=90,
            fontsize=style["model_fs"],
            fontweight="bold",
            va="center",
            ha="center",
            color="#333",
            transform=fig.transFigure,
        )
        for si, seq_type in enumerate(SEQ_TYPES):
            block_start = row_start + si * ROWS_PER_SITE
            block_end = block_start + ROWS_PER_SITE - 1
            pos_t = row_axes[block_start][0].get_position()
            pos_b = row_axes[block_end][0].get_position()
            y_site = (pos_b.y0 + pos_t.y1) / 2
            fig.text(
                site_x,
                y_site,
                SITE_LABEL[seq_type],
                rotation=90,
                fontsize=style["block_fs"],
                fontweight="bold",
                va="center",
                ha="center",
                color="#333",
                transform=fig.transFigure,
            )


def _add_all_models_separators(fig, row_axes, style):
    """Vertical 402|2002 divider; horizontal model and Donor|Acceptor separators."""
    right = style["right"]
    pos_402 = row_axes[0][2].get_position()
    pos_2002 = row_axes[0][3].get_position()
    x_ctx = (pos_402.x1 + pos_2002.x0) / 2
    y_bot = row_axes[-1][0].get_position().y0
    y_top = row_axes[0][0].get_position().y1 + 0.04
    fig.add_artist(
        Line2D(
            [x_ctx, x_ctx],
            [y_bot, y_top],
            transform=fig.transFigure,
            **MODEL_BLOCK_SEPARATOR_KW,
        )
    )
    x_left = style.get("site_label_x", 0.034) + 0.010
    for boundary in range(ROWS_PER_MODEL, len(ALL_MODELS_ROW_KEYS), ROWS_PER_MODEL):
        pos_above = row_axes[boundary - 1][0].get_position()
        pos_below = row_axes[boundary][0].get_position()
        y_sep = (pos_above.y0 + pos_below.y1) / 2
        fig.add_artist(
            Line2D(
                [x_left, right],
                [y_sep, y_sep],
                transform=fig.transFigure,
                **MODEL_BLOCK_SEPARATOR_KW,
            )
        )
    for mi in range(len(MODELS)):
        boundary = mi * ROWS_PER_MODEL + ROWS_PER_SITE
        pos_above = row_axes[boundary - 1][0].get_position()
        pos_below = row_axes[boundary][0].get_position()
        y_sep = (pos_above.y0 + pos_below.y1) / 2
        fig.add_artist(
            Line2D(
                [x_left, right],
                [y_sep, y_sep],
                transform=fig.transFigure,
                **SEPARATOR_KW,
            )
        )


def create_all_models_figure(out_path, cache, dpi, style):
    """Hierarchical all-models figure: GAN/VAE/Diffusion → Donor/Acceptor → species."""
    figsize = style["figsize"]
    fig = plt.figure(figsize=figsize)
    row_axes = _draw_all_models_stacked_grid(fig, cache, style)

    _add_context_group_headers(fig, row_axes, 0, style)
    _add_all_models_hierarchy_labels(fig, row_axes, style)
    _add_figure_xlabel(fig, style, row_axes)
    _add_all_models_separators(fig, row_axes, style)
    _save(fig, out_path, dpi)


def create_by_model_figures(out_dir, cache, dpi, style):
    """One 6x6 figure per model — one logo per panel."""
    figsize = style["figsize"]
    n_rows = len(ROW_KEYS)
    n_cols = len(BY_MODEL_COLS)

    for model in MODELS:
        fig = plt.figure(figsize=figsize)
        block = gridspec.GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            left=style["left"],
            right=style["right"],
            top=style["top"],
            bottom=style["bottom"],
            hspace=style["hspace"],
            wspace=style["wspace"],
        )
        row_axes = _draw_logo_grid(fig, block, cache, style, model_filter=model)

        _add_site_block_labels(fig, style)
        _add_figure_xlabel(fig, style)

        fig.add_artist(
            Line2D(
                [0.040, style["right"]],
                [0.500, 0.500],
                transform=fig.transFigure,
                **SEPARATOR_KW,
            )
        )
        fig.suptitle(
            MODEL_LABEL[model],
            fontsize=style["site_fs"],
            fontweight="bold",
            y=0.992,
        )
        out_path = os.path.join(
            out_dir, f"combined_sequence_logos_{model.lower()}.png"
        )
        _save(fig, out_path, dpi)


def _add_horizontal_labels(fig, row_axes, site_label, style, figsize):
    top_pos = row_axes[0][0].get_position()
    bot_pos = row_axes[-1][0].get_position()
    site_y = (top_pos.y1 + bot_pos.y0) / 2
    fig.text(
        0.02,
        site_y,
        site_label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=style["site_fs"],
        fontweight="bold",
    )
    line_extend = 2.3 / figsize[0]
    for si, species in enumerate(SPECIES):
        pos = row_axes[si][0].get_position()
        fig.text(
            0.065,
            pos.y0 + pos.height / 2,
            species_display(species),
            ha="right",
            va="center",
            fontsize=style["species_fs"],
            fontweight="bold",
        )
        if si < len(SPECIES) - 1:
            next_pos = row_axes[si + 1][0].get_position()
            y_line = (pos.y0 + next_pos.y1) / 2
            fig.add_artist(
                Line2D(
                    [0.07, 0.07 + line_extend],
                    [y_line, y_line],
                    transform=fig.transFigure,
                    color="0.35",
                    linewidth=0.8,
                    clip_on=False,
                )
            )


def create_horizontal_figure(out_path, cache, dpi, seq_types, style):
    """Real | Blend | No-Blend side-by-side; one species row per model-context col."""
    figsize = list(style["figsize"])
    if len(seq_types) == 1:
        figsize[1] *= 0.55
    figsize = tuple(figsize)

    fig = plt.figure(figsize=figsize)
    n_site_blocks = len(seq_types)
    last_row_axes = None

    for block_idx, seq_type in enumerate(seq_types):
        if n_site_blocks == 1:
            margins = (style["bottom"], style["top"])
        elif block_idx == 0:
            margins = (0.52, style["top"])
        else:
            margins = (style["bottom"], 0.48)

        block = gridspec.GridSpec(
            len(SPECIES),
            len(COL_PAIRS),
            figure=fig,
            left=style["left"],
            right=style["right"],
            top=margins[1],
            bottom=margins[0],
            hspace=style["hspace"],
            wspace=style["wspace"],
        )
        row_axes = []
        for si, species in enumerate(SPECIES):
            col_axes = []
            for ci, (model, context) in enumerate(COL_PAIRS):
                inner = block[si, ci].subgridspec(
                    1, len(DATASET_KEYS), wspace=style["inner_wspace"]
                )
                data = cache[(context, model, species, seq_type)]
                inner_axes = []
                for di, ds_key in enumerate(DATASET_KEYS):
                    ax = fig.add_subplot(inner[0, di])
                    panel = {
                        "show_xticklabels": si == len(SPECIES) - 1,
                        "show_yticklabels": ci == 0 and di == 0,
                        "show_xlabel": False,
                        "show_ylabel": ci == 0 and di == 0,
                        "ylabel": f"{species_display(species)}\nbits",
                    }
                    draw_site_logo(
                        ax,
                        data["pwm"][ds_key],
                        context,
                        seq_type,
                        style,
                        panel,
                    )
                    if si == 0 and di == 0:
                        ax.set_title(
                            f"{MODEL_LABEL[model]} {context} bp",
                            fontsize=style["col_fs"],
                            pad=6,
                        )
                    if ci == 0:
                        ax.text(
                            -0.35,
                            1.08 if si == 0 else 0.5,
                            DATASET_LABEL[ds_key] if si > 0 else "",
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=style["dataset_fs"],
                        )
                    inner_axes.append(ax)
                # Dataset labels below top row header area on first species row
                if si == 0:
                    for di, ds_key in enumerate(DATASET_KEYS):
                        inner_axes[di].text(
                            0.5,
                            1.18,
                            DATASET_LABEL[ds_key],
                            transform=inner_axes[di].transAxes,
                            ha="center",
                            va="bottom",
                            fontsize=style["dataset_fs"],
                        )
                col_axes.append(inner_axes[0])
            row_axes.append(col_axes)
        last_row_axes = row_axes

        if block_idx == 0 and n_site_blocks > 1:
            pos_donor = row_axes[-1][0].get_position()
        elif block_idx == 1 and n_site_blocks > 1:
            pos_acceptor = row_axes[0][0].get_position()
            y_sep = (pos_donor.y0 + pos_acceptor.y1) / 2
            fig.add_artist(
                Line2D(
                    [style["left"], style["right"]],
                    [y_sep, y_sep],
                    transform=fig.transFigure,
                    **SEPARATOR_KW,
                )
            )

    if n_site_blocks > 1:
        _add_site_block_labels(fig, style)
        fig.add_artist(
            Line2D(
                [0.040, style["right"]],
                [0.500, 0.500],
                transform=fig.transFigure,
                **SEPARATOR_KW,
            )
        )

    if last_row_axes is not None:
        _add_figure_xlabel(fig, style)

    _save(fig, out_path, dpi)


def create_stacked_figure(out_path, cache, dpi, style):
    """Legacy: Real over Blend over No-Blend (tall figure)."""
    figsize = style["figsize"]
    fig = plt.figure(figsize=figsize)
    last_row_axes = None

    for site_idx, seq_type in enumerate(SEQ_TYPES):
        if site_idx == 0:
            block_bottom, block_top = style["donor_bottom"], style["top"]
        else:
            block_bottom, block_top = style["bottom"], style["acceptor_top"]

        block = gridspec.GridSpec(
            len(SPECIES),
            len(COL_PAIRS),
            figure=fig,
            left=style["left"],
            right=style["right"],
            top=block_top,
            bottom=block_bottom,
            hspace=style["block_hspace"],
            wspace=style["block_wspace"],
        )
        row_axes = []
        for si, species in enumerate(SPECIES):
            col_axes = []
            for ci, (model, context) in enumerate(COL_PAIRS):
                inner = block[si, ci].subgridspec(
                    len(DATASET_KEYS), 1, hspace=style["inner_hspace"]
                )
                data = cache[(context, model, species, seq_type)]
                inner_axes = []
                for di, ds_key in enumerate(DATASET_KEYS):
                    ax = fig.add_subplot(inner[di, 0])
                    panel = {
                        "show_xticklabels": si == len(SPECIES) - 1,
                        "show_yticklabels": ci == 0 and di == 0,
                        "show_xlabel": False,
                        "show_ylabel": ci == 0 and di == 0,
                        "ylabel": f"{species_display(species)}\nbits",
                    }
                    draw_site_logo(
                        ax,
                        data["pwm"][ds_key],
                        context,
                        seq_type,
                        style,
                        panel,
                    )
                    if ci == 0:
                        ax.text(
                            -0.18,
                            0.5,
                            DATASET_LABEL[ds_key],
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=style["dataset_fs"],
                        )
                    inner_axes.append(ax)
                if si == 0:
                    inner_axes[0].set_title(
                        f"{MODEL_LABEL[model]} {context} bp",
                        fontsize=style["col_fs"],
                        pad=4,
                    )
                col_axes.append(inner_axes[0])
            row_axes.append(col_axes)
        last_row_axes = row_axes

        if site_idx == 0:
            pos_donor = row_axes[-1][0].get_position()
        else:
            pos_acceptor = row_axes[0][0].get_position()
            y_sep = (pos_donor.y0 + pos_acceptor.y1) / 2
            fig.add_artist(
                Line2D(
                    [style["left"], style["right"]],
                    [y_sep, y_sep],
                    transform=fig.transFigure,
                    **SEPARATOR_KW,
                )
            )

    _add_site_block_labels(fig, style)
    fig.add_artist(
        Line2D(
            [0.040, style["right"]],
            [0.500, 0.500],
            transform=fig.transFigure,
            **SEPARATOR_KW,
        )
    )

    if last_row_axes is not None:
        _add_figure_xlabel(fig, style)

    _save(fig, out_path, dpi)


def main():
    parser = argparse.ArgumentParser(
        description="Combined sequence-logo figures (all species, models, contexts)"
    )
    parser.add_argument(
        "--out-dir",
        default=OUTPUT_ROOT,
        help="Output directory (default: Revision_Results/direct_evaluation/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (all-models / horizontal / stacked; overrides --out-dir)",
    )
    parser.add_argument(
        "--layout",
        choices=["all-models", "by-model", "horizontal", "stacked", "demo"],
        default="all-models",
        help="Figure layout (default: all-models)",
    )
    parser.add_argument(
        "--demo-model",
        default="DIFFUSION",
        help="Model for --layout demo (default: DIFFUSION)",
    )
    parser.add_argument(
        "--demo-context",
        default="402",
        choices=CONTEXTS,
        help="Context length for --layout demo (default: 402)",
    )
    parser.add_argument(
        "--demo-species",
        default="human",
        choices=SPECIES,
        help="Species for --layout demo (default: human)",
    )
    parser.add_argument(
        "--demo-site",
        default="donor",
        choices=SEQ_TYPES,
        help="Splice-site type for --layout demo (default: donor)",
    )
    parser.add_argument(
        "--split-sites",
        action="store_true",
        default=True,
        help="For horizontal layout: separate Donor and Acceptor files (default)",
    )
    parser.add_argument(
        "--no-split-sites",
        action="store_false",
        dest="split_sites",
        help="For horizontal layout: keep Donor and Acceptor on one figure",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Max sequences per class (default 20000)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument(
        "--fig-width",
        type=float,
        default=None,
        help="Figure width in inches (height scales proportionally)",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=None,
        help="Figure height in inches (overrides proportional height from --fig-width)",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiply label/axis font sizes (default: 1.0; try 1.2 for A4)",
    )
    parser.add_argument(
        "--a4-landscape",
        action="store_true",
        help="Bump font sizes for A4 landscape print (font-scale=1.15; keeps mockup fig proportions)",
    )
    args = parser.parse_args()

    if args.a4_landscape:
        if args.font_scale == 1.0:
            args.font_scale = 1.15

    print("Loading sequence data for all conditions (this may take a few minutes)...")
    cache = _load_cache(args.sample_size)

    def _layout_style(layout_key):
        base = STYLE[layout_key]["figsize"]
        figsize = base
        if args.fig_width:
            scale = args.fig_width / base[0]
            figsize = (args.fig_width, base[1] * scale)
        if args.fig_height:
            figsize = (figsize[0], args.fig_height)
        return scaled_style(layout_key, font_scale=args.font_scale, figsize=figsize)

    if args.layout == "demo":
        out = args.out or os.path.join(
            args.out_dir, "combined_sequence_logos_demo.png"
        )
        create_demo_figure(
            out,
            cache,
            args.dpi,
            model=args.demo_model.upper(),
            context=args.demo_context,
            species=args.demo_species,
            seq_type=args.demo_site,
        )
    elif args.layout == "all-models":
        out = args.out or os.path.join(
            args.out_dir, "combined_sequence_logos_all_models.png"
        )
        create_all_models_figure(
            out, cache, args.dpi, _layout_style("all_models")
        )
    elif args.layout == "by-model":
        create_by_model_figures(
            args.out_dir, cache, args.dpi, _layout_style("by_model")
        )
    elif args.layout == "horizontal":
        hstyle = _layout_style("horizontal")
        if args.split_sites:
            for seq_type in SEQ_TYPES:
                out = args.out or os.path.join(
                    args.out_dir, f"combined_sequence_logos_{seq_type}.png"
                )
                if args.out and seq_type == "acceptor":
                    base, ext = os.path.splitext(args.out)
                    out = f"{base}_{seq_type}{ext}"
                create_horizontal_figure(
                    out, cache, args.dpi, [seq_type], hstyle
                )
        else:
            out = args.out or os.path.join(
                args.out_dir, "combined_sequence_logos.png"
            )
            create_horizontal_figure(
                out, cache, args.dpi, SEQ_TYPES, hstyle
            )
    else:
        out = args.out or os.path.join(
            args.out_dir, "combined_sequence_logos_stacked.png"
        )
        create_stacked_figure(
            out, cache, args.dpi, _layout_style("stacked")
        )


if __name__ == "__main__":
    main()
