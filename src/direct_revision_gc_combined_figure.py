#!/usr/bin/env python3
"""Grouped box-plot for GC content (402 + 2002 bp), donor over acceptor.

Two stacked rows in one figure: Donor (top) / Acceptor (bottom), same layout.

Output: Revision_Results/direct_evaluation/combined_gc_content_all_models.png
"""

import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = os.path.join(REPO, "results", "direct_evaluation")
DEFAULT_OUT = os.path.join(OUTPUT_ROOT, "combined_gc_content_all_models.png")

MODELS = ["GAN", "VAE", "DIFFUSION"]
MODEL_LABEL = {"GAN": "GAN", "VAE": "VAE", "DIFFUSION": "Diffusion"}
SPECIES = ["arabidopsis", "human", "danio"]
SEQ_TYPES = ["donor", "acceptor"]
CONTEXTS = ["402", "2002"]

SPECIES_LABEL = {
    "arabidopsis": "Arabidopsis",
    "human": "Human",
    "danio": "Danio",
}
SITE_LABEL = {
    "donor": "Donor",
    "acceptor": "Acceptor",
}

DATASET_ORDER = ["Real", "Blend", "No-Blend"]
DATASET_MAP = {
    "Real": "Real",
    "Blend": "Blend",
    "No Blend": "No-Blend",
}
PALETTE = {
    "Real": "#009E42",
    "Blend": "#0066CC",
    "No-Blend": "#D62728",
}

BOX_KW = dict(
    linewidth=1.0,
    showfliers=False,
    whiskerprops={"color": "0.15", "linewidth": 1.0},
    capprops={"color": "0.15", "linewidth": 1.0},
    medianprops={"linestyle": "none", "linewidth": 0, "marker": "None"},
    showmeans=True,
    meanline=True,
    meanprops={"color": "0.1", "linewidth": 1.4, "linestyle": "-"},
)


def load_gc_csv(context: str, model: str, species: str, seq_type: str) -> pd.DataFrame:
    path = os.path.join(
        OUTPUT_ROOT, context, model, f"{species}_{seq_type}_gc_content.csv"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_long_table(seq_type: str) -> pd.DataFrame:
    parts = []
    for context in CONTEXTS:
        for species in SPECIES:
            for model in MODELS:
                df = load_gc_csv(context, model, species, seq_type)
                chunk = df.copy()
                chunk["Dataset"] = chunk["Dataset"].map(DATASET_MAP)
                chunk["Context"] = context
                chunk["Model"] = model
                chunk["Species"] = SPECIES_LABEL[species]
                parts.append(
                    chunk[["Species", "Model", "Dataset", "Context", "GC_Content"]]
                )
    out = pd.concat(parts, ignore_index=True)
    out["Dataset"] = pd.Categorical(out["Dataset"], categories=DATASET_ORDER, ordered=True)
    return out


def _layout_slots():
    slots = []
    species_ranges = []
    x = 0

    for species in SPECIES:
        sp_start = x
        for model in MODELS:
            model_positions = []
            for context in CONTEXTS:
                slots.append(
                    {
                        "slot": x,
                        "Species": SPECIES_LABEL[species],
                        "Model": model,
                        "Context": context,
                    }
                )
                model_positions.append(x)
                x += 1
            center = sum(model_positions) / len(model_positions)
            for slot in slots:
                if slot["Species"] == SPECIES_LABEL[species] and slot["Model"] == model:
                    slot["model_center"] = center
        species_ranges.append((sp_start, x - 1, SPECIES_LABEL[species]))

    return slots, species_ranges


def _plot_row(ax, merged):
    sns.boxplot(
        data=merged,
        x="slot",
        y="GC_Content",
        hue="Dataset",
        hue_order=DATASET_ORDER,
        palette=PALETTE,
        order=sorted(merged["slot"].unique()),
        dodge=0.72,
        width=0.58,
        ax=ax,
        **BOX_KW,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    ax.set_xlabel("")
    ax.set_xticks([])
    ax.xaxis.set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)


def _add_context_labels(ax, species_ranges, y: float = -0.02, inside: bool = False):
    label_trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    va = "bottom" if inside else "top"
    for x0, _x1, _ in species_ranges:
        for mi, _model in enumerate(MODELS):
            x402 = x0 + mi * 2
            x2002 = x402 + 1
            ax.text(
                x402,
                y,
                "402bp",
                ha="center",
                va=va,
                fontsize=10,
                color="0.35",
                transform=label_trans,
                clip_on=False,
            )
            ax.text(
                x2002,
                y,
                "2002bp",
                ha="center",
                va=va,
                fontsize=10,
                color="0.35",
                transform=label_trans,
                clip_on=False,
            )


def _decorate_bottom_row(ax, slots, species_ranges):
    label_trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    seen_model_labels = set()
    for slot in slots:
        key = (slot["Species"], slot["Model"])
        if key not in seen_model_labels:
            seen_model_labels.add(key)
            ax.text(
                slot["model_center"],
                -0.14,
                MODEL_LABEL.get(slot["Model"], slot["Model"]),
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                transform=label_trans,
                clip_on=False,
            )

    _add_context_labels(ax, species_ranges, y=-0.02)

    for x0, x1, name in species_ranges:
        xm = (x0 + x1) / 2
        ax.plot(
            [x0 - 0.35, x0 - 0.35, x1 + 0.35, x1 + 0.35],
            [-0.20, -0.215, -0.215, -0.20],
            transform=label_trans,
            color="0.35",
            lw=1.0,
            clip_on=False,
        )
        ax.text(
            xm,
            -0.26,
            name,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            transform=label_trans,
            clip_on=False,
        )


def _add_species_dividers(ax, species_ranges, y_bottom: float = 0.08):
    label_trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(len(species_ranges) - 1):
        _, x1, _ = species_ranges[i]
        ax.plot(
            [x1 + 0.5, x1 + 0.5],
            [0.98, y_bottom],
            transform=label_trans,
            color="0.55",
            lw=1.0,
            ls="--",
            zorder=0,
            clip_on=True,
        )


def create_combined_figure(out_path: str, dpi: int = 300):
    slots, species_ranges = _layout_slots()
    slot_df = pd.DataFrame(slots)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15.0, 5.4),
        gridspec_kw={"hspace": 0.22},
    )

    y_min, y_max = None, None
    for row, seq_type in enumerate(SEQ_TYPES):
        merged = build_long_table(seq_type).merge(
            slot_df, on=["Species", "Model", "Context"], how="inner"
        )
        _plot_row(axes[row], merged)
        axes[row].set_ylabel(
            SITE_LABEL[seq_type],
            fontsize=13,
            fontweight="bold",
            labelpad=10,
        )
        axes[row].tick_params(axis="y", labelsize=11)
        lo, hi = axes[row].get_ylim()
        y_min = lo if y_min is None else min(y_min, lo)
        y_max = hi if y_max is None else max(y_max, hi)

    for ax in axes:
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(-0.5, len(slots) - 0.5)

    _add_context_labels(axes[0], species_ranges, y=-0.045, inside=False)
    _add_species_dividers(axes[0], species_ranges, y_bottom=0.08)
    _add_species_dividers(axes[1], species_ranges, y_bottom=0.08)
    _decorate_bottom_row(axes[1], slots, species_ranges)

    fig.text(
        0.02,
        0.5,
        "GC content (%)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )

    source_handles = [
        mpatches.Patch(facecolor=PALETTE[name], edgecolor="0.15", linewidth=0.8, label=name)
        for name in DATASET_ORDER
    ]
    axes[0].legend(
        handles=source_handles,
        title="Data source",
        loc="upper left",
        bbox_to_anchor=(1.0, 0.995),
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
        facecolor="white",
        framealpha=0.92,
        fontsize=14,
        title_fontsize=15,
        handlelength=1.4,
        handleheight=1.0,
        borderpad=0.6,
        labelspacing=0.5,
    )

    fig.subplots_adjust(left=0.09, right=0.88, top=0.97, bottom=0.22)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Combined GC content box-plot (donor + acceptor)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    sns.set_theme(style="white", context="paper", font_scale=1.0)
    create_combined_figure(args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
