#!/usr/bin/env python3
"""Revision supplementary comprehensive figures (402 + 2002 bp).

Adapted from create_comprehensive_comparisons_final.py:
  - Revision data paths (BlendSplice real + Revision_Generated_Sequences)
  - Lambda values 0.0, 0.25, 0.5, 0.75
  - Three species (Arabidopsis, Human, Danio) in 2×3 grids
  - GC content as box plots (not histograms)

Output: Revision_Supplemtary_Resuslts/direct_evaluation_comprehensive/{context}/{MODEL}/

Usage:
    python revision_supplementary_comprehensive_figures.py --context 402
    python revision_supplementary_comprehensive_figures.py --context 2002
    python revision_supplementary_comprehensive_figures.py --context all
"""

from typing import Dict, List, Optional, Tuple

import argparse
import os
import random
import sys
from collections import Counter
from math import ceil

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from direct_analysis import (  # noqa: E402
    calculate_position_conservation,
    compute_pwm,
    create_logo,
    gc_fraction,
)
from revision_supplementary_common import (  # noqa: E402
    COMPREHENSIVE_LAMBDAS,
    DATASETS,
    GRID_2X3,
    MODEL_LABEL,
    MODELS,
    TITLES_2X3,
    context_cfg,
    lambda_key,
    load_revision_lambda_data,
    output_dir,
)

SMALL_SIZE = 16
MEDIUM_SIZE = 18
LARGE_SIZE = 20

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=MEDIUM_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("lines", linewidth=1.5)

LAMBDA_ORDER = ["real", "lambda_0.0", "lambda_0.25", "lambda_0.5", "lambda_0.75"]
LABELS_MAP = {
    "real": "Real",
    "lambda_0.0": "λ=0.0 (No-Blend)",
    "lambda_0.25": "λ=0.25",
    "lambda_0.5": "λ=0.5 (Blend)",
    "lambda_0.75": "λ=0.75",
}
PALETTE = {
    "Real": "#1f77b4",
    "λ=0.0 (No-Blend)": "#2ca02c",
    "λ=0.25": "#ff7f0e",
    "λ=0.5 (Blend)": "#d62728",
    "λ=0.75": "#9467bd",
}
COLORS = {
    "real": "#1f77b4",
    "lambda_0.0": "#2ca02c",
    "lambda_0.25": "#ff7f0e",
    "lambda_0.5": "#d62728",
    "lambda_0.75": "#9467bd",
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

KMER_COLOR_MAP = {
    "Real": "#3274A1",
    "λ=0.0": "#3A923A",
    "λ=0.25": "#E1812C",
    "λ=0.5": "#C44E52",
    "λ=0.75": "#8172B3",
}
KMER_BAR_PAIRS = [
    ("real", "Real"),
    ("lambda_0.5", "λ=0.5"),
    ("lambda_0.0", "λ=0.0"),
    ("lambda_0.25", "λ=0.25"),
    ("lambda_0.75", "λ=0.75"),
]


def _panel_title(ax, key: str) -> None:
    ax.set_title(TITLES_2X3[key], fontsize=LARGE_SIZE, fontweight="bold", pad=10)


def analyze_3mers_for_dataset(sequences: List[str], motif_pos: int) -> Dict[str, List[Tuple[str, float]]]:
    window = 10
    regions = [
        seq[motif_pos - window : motif_pos + window]
        for seq in sequences
        if len(seq) > motif_pos + window
    ]
    positions = [7, 12]
    position_labels = [
        f"{motif_pos - 3} (Upstream)",
        f"{motif_pos + 2} (Downstream)",
    ]
    results = {}
    for pos, pos_label in zip(positions, position_labels):
        counter = Counter()
        for seq in regions:
            if len(seq) >= pos + 3:
                counter[seq[pos : pos + 3]] += 1
        total = sum(counter.values())
        results[pos_label] = [
            (kmer, count / total if total > 0 else 0) for kmer, count in counter.most_common(5)
        ]
    return results


def create_combined_conservation_grid(context: str, model: str, out: str, sample_size: int) -> None:
    print(f"  Conservation grid ({context} bp, {model})...")
    cfg = context_cfg(context)
    all_data = {
        f"{sp}_{st}": load_revision_lambda_data(context, sp, st, model, sample_size)
        for sp, st in DATASETS
    }

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    for key, (row, col) in GRID_2X3.items():
        ax = axes[row, col]
        data_dict = all_data[key]
        for lk in LAMBDA_ORDER:
            if lk in data_dict:
                cons = calculate_position_conservation(data_dict[lk])
                ax.plot(
                    np.arange(len(cons)),
                    cons,
                    label=LABELS_MAP[lk],
                    color=COLORS[lk],
                    linewidth=1.8,
                )
        ax.set_xlim(*cfg["cons_region"])
        ax.set_ylim(0, 1)
        _panel_title(ax, key)
        ax.set_xlabel("Position", fontsize=LARGE_SIZE)
        ax.set_ylabel("Conservation Score", fontsize=LARGE_SIZE)
        ax.legend(fontsize=SMALL_SIZE, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, f"{MODEL_LABEL[model]}_combined_conservation_all.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


def create_combined_gc_content_grid(context: str, model: str, out: str, sample_size: int) -> None:
    print(f"  GC box-plot grid ({context} bp, {model})...")
    all_data = {
        f"{sp}_{st}": load_revision_lambda_data(context, sp, st, model, sample_size)
        for sp, st in DATASETS
    }

    dataset_order = [LABELS_MAP[lk] for lk in LAMBDA_ORDER]
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    all_gc: List[float] = []
    for data_dict in all_data.values():
        for lk in LAMBDA_ORDER:
            if lk in data_dict:
                all_gc.extend(gc_fraction(s) * 100 for s in data_dict[lk])
    gc_min, gc_max = (min(all_gc), max(all_gc)) if all_gc else (0, 100)

    for key, (row, col) in GRID_2X3.items():
        ax = axes[row, col]
        data_dict = all_data[key]
        rows = []
        for lk in LAMBDA_ORDER:
            if lk not in data_dict:
                continue
            label = LABELS_MAP[lk]
            for val in (gc_fraction(s) * 100 for s in data_dict[lk]):
                rows.append({"GC_Content": val, "Dataset": label})
        if not rows:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            _panel_title(ax, key)
            continue
        gc_df = pd.DataFrame(rows)
        present = [d for d in dataset_order if d in gc_df["Dataset"].unique()]
        sns.boxplot(
            data=gc_df,
            x="Dataset",
            y="GC_Content",
            order=present,
            palette=PALETTE,
            ax=ax,
            **BOX_KW,
        )
        _panel_title(ax, key)
        ax.set_xlabel("")
        ax.set_ylabel("GC Content (%)", fontsize=MEDIUM_SIZE)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=SMALL_SIZE - 2)
        ax.set_ylim(gc_min, gc_max)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, f"{MODEL_LABEL[model]}_combined_gc_content_all.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


def create_combined_3mer_grid(context: str, model: str, out: str, sample_size: int) -> None:
    print(f"  3-mer grid ({context} bp, {model})...")
    cfg = context_cfg(context)
    motif_pos = cfg["motif_pos"]
    all_data = {
        f"{sp}_{st}": load_revision_lambda_data(context, sp, st, model, sample_size)
        for sp, st in DATASETS
    }

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    global_max = 0.0
    parsed = {}
    for key, data_dict in all_data.items():
        parsed[key] = {}
        for lk in LAMBDA_ORDER:
            if lk in data_dict:
                parsed[key][lk] = analyze_3mers_for_dataset(data_dict[lk], motif_pos)
                for pos_data in parsed[key][lk].values():
                    for _, freq in pos_data[:5]:
                        global_max = max(global_max, freq)
    y_max = max(0.2, ceil(global_max / 0.2) * 0.2)

    up_label = f"{motif_pos - 3} (Upstream)"
    down_label = f"{motif_pos + 2} (Downstream)"
    bar_width = 0.15

    for key, (row, col) in GRID_2X3.items():
        ax = axes[row, col]
        all_3mer = parsed.get(key, {})
        if "real" not in all_3mer:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            _panel_title(ax, key)
            continue

        upstream_top5 = [k for k, _ in all_3mer["real"].get(up_label, [])[:5]]
        downstream_top5 = [k for k, _ in all_3mer["real"].get(down_label, [])[:5]]
        all_kmers = upstream_top5 + downstream_top5
        x_offset = 0
        split_line_x = None

        for group, top5, pos_label in (
            ("up", upstream_top5, up_label),
            ("down", downstream_top5, down_label),
        ):
            if not top5:
                continue
            for i, kmer in enumerate(top5):
                x_base = x_offset + i
                for j, (lk, label) in enumerate(KMER_BAR_PAIRS):
                    if lk not in all_3mer:
                        continue
                    freq_dict = {k: f for k, f in all_3mer[lk].get(pos_label, [])}
                    ax.bar(
                        x_base + j * bar_width,
                        freq_dict.get(kmer, 0),
                        bar_width,
                        color=KMER_COLOR_MAP[label],
                        alpha=0.8,
                    )
            if group == "up":
                split_line_x = x_offset + len(top5) + 0.5
            x_offset += len(top5) + 1

        if split_line_x is not None:
            ax.axvline(split_line_x, linestyle=":", color="gray", linewidth=1.5, alpha=0.7)

        tick_positions = []
        label_offset = 0
        for n in (len(upstream_top5), len(downstream_top5)):
            for i in range(n):
                tick_positions.append(label_offset + i + bar_width * 2)
            label_offset += n + 1

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(all_kmers, rotation=45, ha="right", fontsize=SMALL_SIZE)
        _panel_title(ax, key)
        ax.set_xlabel("3-mer", fontsize=MEDIUM_SIZE)
        ax.set_ylabel("Frequency", fontsize=MEDIUM_SIZE)
        ax.set_ylim(0, y_max)
        ax.set_yticks(np.arange(0, y_max + 1e-6, 0.2))
        ax.tick_params(axis="both", which="major", labelsize=SMALL_SIZE)

        proxies = [
            mpatches.Patch(color=KMER_COLOR_MAP[label], label=label)
            for _, label in KMER_BAR_PAIRS
        ]
        ax.legend(handles=proxies, fontsize=SMALL_SIZE - 1, loc="best")

    plt.tight_layout()
    path = os.path.join(out, f"{MODEL_LABEL[model]}_combined_3mer_all.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")


def create_combined_sequence_logos_grid(context: str, model: str, out: str, sample_size: int) -> None:
    """One readable figure per splice-site type: 5 λ-rows × 3 species columns."""
    print(f"  Sequence logos ({context} bp, {model})...")
    cfg = context_cfg(context)
    logo_region = cfg["logo_region"]
    species_list = ["arabidopsis", "human", "danio"]
    species_titles = ["Arabidopsis", "Human", "Danio"]

    all_data = {
        f"{sp}_{st}": load_revision_lambda_data(context, sp, st, model, sample_size)
        for sp, st in DATASETS
    }

    for seq_type in ("donor", "acceptor"):
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(
            len(LAMBDA_ORDER),
            len(species_list),
            figure=fig,
            hspace=0.55,
            wspace=0.28,
            left=0.10,
            right=0.98,
            top=0.92,
            bottom=0.07,
        )

        for col, (species, sp_title) in enumerate(zip(species_list, species_titles)):
            key = f"{species}_{seq_type}"
            data_dict = all_data.get(key, {})

            for row_i, lk in enumerate(LAMBDA_ORDER):
                ax = fig.add_subplot(gs[row_i, col])
                if lk not in data_dict:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")
                    continue
                pwm = compute_pwm(data_dict[lk])
                create_logo(pwm, ax, region=logo_region)
                if row_i == 0:
                    ax.set_title(
                        f"{sp_title} {seq_type.title()}",
                        fontsize=LARGE_SIZE,
                        fontweight="bold",
                        pad=10,
                    )
                if col == 0:
                    ax.set_ylabel(
                        LABELS_MAP[lk],
                        fontsize=MEDIUM_SIZE,
                        fontweight="bold",
                        labelpad=8,
                    )
                else:
                    ax.set_ylabel("")
                if row_i < len(LAMBDA_ORDER) - 1:
                    ax.set_xlabel("")
                ax.tick_params(labelsize=SMALL_SIZE)

        site_label = seq_type.title()
        path = os.path.join(out, f"{MODEL_LABEL[model]}_combined_sequence_logos_{seq_type}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved {path}")


def run_context(context: str, sample_size: int, only: Optional[str] = None) -> None:
    print(f"\n{'=' * 80}\nComprehensive figures | {context} bp\n{'=' * 80}")
    base = output_dir("direct_evaluation_comprehensive", context)
    makers = {
        "conservation": create_combined_conservation_grid,
        "gc": create_combined_gc_content_grid,
        "3mer": create_combined_3mer_grid,
        "logos": create_combined_sequence_logos_grid,
    }
    selected = [makers[only]] if only else makers.values()

    for model in MODELS:
        model_out = os.path.join(base, MODEL_LABEL[model])
        os.makedirs(model_out, exist_ok=True)
        print(f"\n{MODEL_LABEL[model]} ({context} bp)")
        for fn in selected:
            try:
                fn(context, model, model_out, sample_size)
            except Exception as exc:
                print(f"  ERROR {model} ({fn.__name__}): {exc}")
                import traceback

                traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=["402", "2002", "all"], default="all")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument(
        "--only",
        choices=["conservation", "gc", "3mer", "logos"],
        default=None,
        help="Regenerate only one figure type.",
    )
    args = parser.parse_args()

    random.seed(42)
    sns.set_theme(style="white", context="paper", font_scale=1.0)

    contexts = ["402", "2002"] if args.context == "all" else [args.context]
    for ctx in contexts:
        run_context(ctx, args.sample_size, only=args.only)

    print(f"\nDone. Figures under {output_dir()}/direct_evaluation_comprehensive/")


if __name__ == "__main__":
    main()
