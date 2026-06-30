#!/usr/bin/env python3
"""Polypyrimidine tract (PPT) quantification for acceptor splice sites.

Acceptor-only analysis in the PPT window (−40…−5 nt upstream of AG, intron side):
  - Pyrimidine fraction (C/T): severity of tract pyrimidine richness
  - Longest consecutive pyrimidine run: tract integrity / well-formedness
  - Malformation rate: fraction with pyrimidine fraction below the 5th
    percentile of Real (per species × context; data-driven threshold)

Compares Real vs synthetic No-Blend (λ=0.0) vs Blend (λ=0.5) for GAN, VAE,
and Diffusion at 402 bp and 2002 bp across all species.

Outputs (Revision_Results/direct_evaluation/):
  - ppt_summary.csv                 long format, all metrics
  - ppt_pyr_fraction_wide.csv       table-ready: mean pyrimidine fraction
  - ppt_malformation_rate_wide.csv  table-ready: malformation rates
  - ppt_longest_run_wide.csv        table-ready: mean longest pyr run

Usage:
    python direct_revision_ppt_analysis.py
    python direct_revision_ppt_analysis.py --sample-size 20000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from direct_revision_analysis import (
    CONTEXT_CFG,
    CONTEXTS,
    MODELS,
    OUTPUT_ROOT,
    SPECIES,
    load_real_positives,
    load_synthetic,
    species_display,
)

PYRIMIDINES = frozenset("CT")
PPT_REL_START = -40
PPT_REL_END = -5
SYNTHETIC_CONDITIONS = [
    ("Blend", 0.5),
    ("No-Blend", 0.0),
]
SOURCE_ORDER = ["Real"] + [
    f"{model}_{syn.replace('-', '_')}"
    for model in MODELS
    for syn, _ in SYNTHETIC_CONDITIONS
]


def ppt_window(seq: str, motif_pos: int) -> str:
    """Extract PPT region (−40…−5 relative to AG start at motif_pos)."""
    start = motif_pos + PPT_REL_START
    end = motif_pos + PPT_REL_END + 1
    if start < 0 or end > len(seq):
        return ""
    return seq[start:end]


def pyrimidine_fraction(window: str) -> float:
    if not window:
        return np.nan
    return sum(1 for b in window if b in PYRIMIDINES) / len(window)


def longest_pyrimidine_run(window: str) -> int:
    best = cur = 0
    for b in window:
        if b in PYRIMIDINES:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def per_sequence_metrics(
    seqs: List[str], motif_pos: int
) -> Tuple[np.ndarray, np.ndarray]:
    fracs: List[float] = []
    runs: List[int] = []
    for seq in seqs:
        window = ppt_window(seq, motif_pos)
        fracs.append(pyrimidine_fraction(window))
        runs.append(longest_pyrimidine_run(window))
    return np.asarray(fracs, dtype=float), np.asarray(runs, dtype=float)


def malformation_rate(fracs: np.ndarray, threshold: float) -> float:
    valid = fracs[~np.isnan(fracs)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid < threshold))


def summarize_source(
    fracs: np.ndarray,
    runs: np.ndarray,
    threshold: float,
    *,
    context: str,
    species: str,
    source: str,
    model: str,
    synthetic: str,
    lam: Optional[float],
    n: int,
) -> Dict:
    valid_frac = fracs[~np.isnan(fracs)]
    valid_run = runs[~np.isnan(runs)]
    return {
        "context_bp": int(context),
        "species": species,
        "species_label": species_display(species),
        "source": source,
        "model": model,
        "synthetic": synthetic,
        "lambda": lam,
        "n": n,
        "ppt_window_nt": PPT_REL_END - PPT_REL_START + 1,
        "pyr_fraction_threshold_p5": threshold,
        "pyr_fraction_mean": float(np.mean(valid_frac)) if valid_frac.size else np.nan,
        "pyr_fraction_std": float(np.std(valid_frac, ddof=1)) if valid_frac.size > 1 else np.nan,
        "pyr_fraction_median": float(np.median(valid_frac)) if valid_frac.size else np.nan,
        "malformation_rate": malformation_rate(fracs, threshold),
        "longest_pyr_run_mean": float(np.mean(valid_run)) if valid_run.size else np.nan,
        "longest_pyr_run_std": float(np.std(valid_run, ddof=1)) if valid_run.size > 1 else np.nan,
        "longest_pyr_run_median": float(np.median(valid_run)) if valid_run.size else np.nan,
    }


def collect_ppt_metrics(
    sample_size: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: List[Dict] = []
    thresholds: Dict[Tuple[str, str], float] = {}

    for context in CONTEXTS:
        motif_pos = CONTEXT_CFG[context]["motif_pos"]
        for species in SPECIES:
            real_seqs = load_real_positives(species, "acceptor", context, sample_size)
            if len(real_seqs) < 50:
                print(f"  Skip Real {context} {species}: n={len(real_seqs)}")
                continue

            real_fracs, real_runs = per_sequence_metrics(real_seqs, motif_pos)
            threshold = float(np.percentile(real_fracs[~np.isnan(real_fracs)], 5))
            thresholds[(context, species)] = threshold

            rows.append(
                summarize_source(
                    real_fracs,
                    real_runs,
                    threshold,
                    context=context,
                    species=species,
                    source="Real",
                    model="Real",
                    synthetic="Real",
                    lam=np.nan,
                    n=len(real_seqs),
                )
            )
            print(
                f"  Real {context} bp | {species} | n={len(real_seqs)} | "
                f"τ(p5)={threshold:.4f} | mean pyr={rows[-1]['pyr_fraction_mean']:.4f}"
            )

            for model in MODELS:
                for syn_label, lam in SYNTHETIC_CONDITIONS:
                    try:
                        syn_seqs = load_synthetic(
                            model, species, "acceptor", context, lam, sample_size
                        )
                    except FileNotFoundError as exc:
                        print(f"  Skip {model} {context} {species} {syn_label}: {exc}")
                        continue
                    if len(syn_seqs) < 50:
                        print(
                            f"  Skip {model} {context} {species} {syn_label}: "
                            f"n={len(syn_seqs)}"
                        )
                        continue

                    n = min(len(real_seqs), len(syn_seqs))
                    syn_use = (
                        rng.sample(syn_seqs, n) if len(syn_seqs) > n else list(syn_seqs)
                    )
                    syn_fracs, syn_runs = per_sequence_metrics(syn_use, motif_pos)
                    source = f"{model}_{syn_label.replace('-', '_')}"
                    rows.append(
                        summarize_source(
                            syn_fracs,
                            syn_runs,
                            threshold,
                            context=context,
                            species=species,
                            source=source,
                            model=model,
                            synthetic=syn_label,
                            lam=lam,
                            n=len(syn_use),
                        )
                    )
                    row = rows[-1]
                    print(
                        f"  OK {context} bp | {model} | {species} | {syn_label} | "
                        f"n={len(syn_use)} | pyr={row['pyr_fraction_mean']:.4f} | "
                        f"malform={row['malformation_rate']:.3f} | "
                        f"run={row['longest_pyr_run_mean']:.2f}"
                    )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["context_bp", "species", "source"]
    ).reset_index(drop=True)


def _wide_metric(df: pd.DataFrame, value_col: str, suffix: str) -> pd.DataFrame:
    sub = df.copy()
    sub["col"] = sub["source"]
    wide = sub.pivot_table(
        index=["context_bp", "species", "species_label", "pyr_fraction_threshold_p5"],
        columns="col",
        values=value_col,
        aggfunc="first",
    )
    ordered = [c for c in SOURCE_ORDER if c in wide.columns]
    extra = [c for c in wide.columns if c not in ordered]
    wide = wide[ordered + extra]
    wide.columns = [f"{c}_{suffix}" for c in wide.columns]
    return wide.reset_index().round(6)


def formatted_fraction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pyrimidine fraction with malformation rate in parentheses (display)."""
    rows = []
    for (context, species), grp in df.groupby(["context_bp", "species"], sort=False):
        base = grp.iloc[0]
        row = {
            "context_bp": int(context),
            "species": species,
            "species_label": base["species_label"],
            "pyr_fraction_threshold_p5": base["pyr_fraction_threshold_p5"],
        }
        for _, r in grp.sort_values("source").iterrows():
            key = r["source"]
            mean_val = r["pyr_fraction_mean"]
            malform = r["malformation_rate"]
            if pd.isna(mean_val):
                row[f"{key}_display"] = ""
            else:
                row[f"{key}_display"] = (
                    f"{mean_val:.3f} ({malform:.3f})" if not pd.isna(malform) else f"{mean_val:.3f}"
                )
            row[f"{key}_pyr_fraction_mean"] = mean_val
            row[f"{key}_malformation_rate"] = malform
        rows.append(row)
    out = pd.DataFrame(rows)
    display_cols = [c for c in out.columns if c.endswith("_display")]
    return out.sort_values(["context_bp", "species"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPT malformation analysis for acceptor splice sites."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Max sequences per class (default: 20000).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    sample_size = args.sample_size if args.sample_size > 0 else None

    print(
        f"PPT window: {PPT_REL_START}…{PPT_REL_END} nt upstream of AG "
        f"({PPT_REL_END - PPT_REL_START + 1} nt)"
    )
    print(f"Sample size: {sample_size or 'all'}")

    df = collect_ppt_metrics(sample_size=sample_size, seed=args.seed)
    if df.empty:
        print("No results.")
        return

    summary_path = os.path.join(OUTPUT_ROOT, "ppt_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(df)} rows)")

    frac_wide = _wide_metric(df, "pyr_fraction_mean", "pyr_fraction_mean")
    frac_wide_path = os.path.join(OUTPUT_ROOT, "ppt_pyr_fraction_wide.csv")
    frac_wide.to_csv(frac_wide_path, index=False)
    print(f"Wrote {frac_wide_path}")

    malform_wide = _wide_metric(df, "malformation_rate", "malformation_rate")
    malform_path = os.path.join(OUTPUT_ROOT, "ppt_malformation_rate_wide.csv")
    malform_wide.to_csv(malform_path, index=False)
    print(f"Wrote {malform_path}")

    run_wide = _wide_metric(df, "longest_pyr_run_mean", "longest_pyr_run_mean")
    run_path = os.path.join(OUTPUT_ROOT, "ppt_longest_run_wide.csv")
    run_wide.to_csv(run_path, index=False)
    print(f"Wrote {run_path}")

    formatted = formatted_fraction_table(df)
    formatted_path = os.path.join(OUTPUT_ROOT, "ppt_pyr_fraction_formatted.csv")
    formatted.to_csv(formatted_path, index=False)
    print(f"Wrote {formatted_path}")


if __name__ == "__main__":
    main()
