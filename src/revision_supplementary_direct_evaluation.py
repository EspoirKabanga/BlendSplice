#!/usr/bin/env python3
"""Revision supplementary direct evaluation (λ=0.25, 0.75) for 402 and 2002 bp.

Per-model/lambda CSV outputs and individual plots. GC content uses box plots.

Output: Revision_Supplemtary_Resuslts/direct_evaluation/
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    DATASETS,
    MODEL_LABEL,
    MODELS,
    SUPPLEMENTARY_LAMBDAS,
    context_cfg,
    lambda_key,
    load_revision_lambda_data,
    output_dir,
    species_display,
)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
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


def process_pair(context, species, seq_type, model, lam, sample_size, base_out):
    data = load_revision_lambda_data(
        context, species, seq_type, model, sample_size, lambdas=[lam]
    )
    if "real" not in data or lambda_key(lam) not in data:
        return None

    syn_key = lambda_key(lam)
    real_seqs, syn_seqs = data["real"], data[syn_key]
    prefix = os.path.join(
        base_out, MODEL_LABEL[model], f"lambda_{lam:g}",
        f"{species}_{seq_type}",
    )
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    cfg = context_cfg(context)

    real_gc = [gc_fraction(s) * 100 for s in real_seqs]
    syn_gc = [gc_fraction(s) * 100 for s in syn_seqs]
    pd.DataFrame(
        {
            "GC_Content": real_gc + syn_gc,
            "Dataset": ["Real"] * len(real_gc) + [f"λ={lam:g}"] * len(syn_gc),
        }
    ).to_csv(f"{prefix}_gc_content.csv", index=False)

    real_cons = calculate_position_conservation(real_seqs)
    syn_cons = calculate_position_conservation(syn_seqs)
    pd.DataFrame(
        {"Position": range(len(real_cons)), "Real": real_cons, "Synthetic": syn_cons}
    ).to_csv(f"{prefix}_conservation.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    gc_df = pd.DataFrame(
        {
            "GC_Content": real_gc + syn_gc,
            "Dataset": ["Real"] * len(real_gc) + [f"λ={lam:g}"] * len(syn_gc),
        }
    )
    sns.boxplot(
        data=gc_df,
        x="Dataset",
        y="GC_Content",
        palette={"Real": "#1f77b4", f"λ={lam:g}": "#ff7f0e"},
        ax=ax,
        **BOX_KW,
    )
    ax.set_ylabel("GC Content (%)")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(f"{prefix}_gc_content.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    create_logo(compute_pwm(real_seqs), axes[0], region=cfg["logo_region"])
    axes[0].set_title(f"Real — {species_display(species)} {seq_type.title()}")
    create_logo(compute_pwm(syn_seqs), axes[1], region=cfg["logo_region"])
    axes[1].set_title(f"{MODEL_LABEL[model]} λ={lam:g}")
    plt.tight_layout()
    plt.savefig(f"{prefix}_sequence_logos.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(real_cons, label="Real", color="#1f77b4")
    ax.plot(syn_cons, label=f"{MODEL_LABEL[model]} λ={lam:g}", color="#ff7f0e")
    ax.set_xlim(*cfg["cons_region"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Conservation Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_conservation.png", dpi=300, bbox_inches="tight")
    plt.close()
    return prefix


def run_context(context: str, sample_size: int) -> None:
    base_out = output_dir("direct_evaluation", context)
    for model in MODELS:
        for lam in SUPPLEMENTARY_LAMBDAS:
            for species, seq_type in DATASETS:
                try:
                    path = process_pair(
                        context, species, seq_type, model, lam, sample_size, base_out
                    )
                    if path:
                        print(f"OK {context} | {model} | λ={lam} | {species} {seq_type}")
                except Exception as exc:
                    print(f"FAIL {context} {model} λ={lam} {species} {seq_type}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", choices=["402", "2002", "all"], default="all")
    parser.add_argument("--sample-size", type=int, default=20000)
    args = parser.parse_args()
    random.seed(42)
    sns.set_theme(style="white")
    contexts = ["402", "2002"] if args.context == "all" else [args.context]
    for ctx in contexts:
        run_context(ctx, args.sample_size)


if __name__ == "__main__":
    main()
