#!/usr/bin/env python3
"""Quantitative k-mer coverage analysis for direct evaluation (revision).

Compares real positive splice-site sequences to synthetic Blend (λ=0.5) and
No-Blend (λ=0.0) sets from GAN, VAE, and Diffusion at 402 bp and 2002 bp for
all species and splice-site types.

Metrics per comparison (k = 1…6 by default):
  - coverage_real_in_syn : fraction of real k-mers also observed in synthetic
  - coverage_syn_in_real : fraction of synthetic k-mers also observed in real
  - pearson, spearman, cosine similarity of normalized k-mer spectra
  - Jensen–Shannon divergence (base 2)

Outputs (Revision_Results/direct_evaluation/):
  - kmer_jsd_table.csv              (long format; primary table export)
  - kmer_jsd_wide_k3.csv / k6.csv   (wide format for copy-paste tables)
  - kmer_coverage_summary.csv       (full metrics, all k)
  - kmer_coverage_headline.csv

Usage:
    python direct_revision_kmer_coverage.py
    python direct_revision_kmer_coverage.py --sample-size 2000 --no-plots
"""

from __future__ import annotations

import argparse
import itertools
import os
import random
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr

from direct_revision_analysis import (
    CONTEXTS,
    MODELS,
    OUTPUT_ROOT,
    SEQ_TYPES,
    SPECIES,
    load_real_positives,
    load_synthetic,
    species_display,
)

BASES = ["A", "C", "G", "T"]
DEFAULT_KMER_SIZES = [1, 2, 3, 4, 5, 6]
SYNTHETIC_CONDITIONS = [
    ("Blend", 0.5),
    ("No-Blend", 0.0),
]


def all_kmers(k: int) -> List[str]:
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def kmer_spectrum(seqs: List[str], k: int) -> np.ndarray:
    """Global normalized k-mer frequency vector over all 4^k k-mers."""
    index = {km: i for i, km in enumerate(all_kmers(k))}
    counts = np.zeros(len(index), dtype=np.float64)
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            idx = index.get(seq[i : i + k])
            if idx is not None:
                counts[idx] += 1.0
    total = counts.sum()
    return counts / total if total > 0 else counts


def kmer_metrics(real_seqs: List[str], syn_seqs: List[str], k: int) -> Dict[str, float]:
    pr = kmer_spectrum(real_seqs, k)
    ps = kmer_spectrum(syn_seqs, k)

    real_present = pr > 0
    syn_present = ps > 0
    n_real = int(real_present.sum())
    n_syn = int(syn_present.sum())
    shared = int((real_present & syn_present).sum())

    cov_real_in_syn = shared / n_real if n_real else 0.0
    cov_syn_in_real = shared / n_syn if n_syn else 0.0

    cos = float(np.dot(pr, ps) / ((np.linalg.norm(pr) * np.linalg.norm(ps)) + 1e-12))
    jsd = float(jensenshannon(pr, ps, base=2))
    jsd = 0.0 if np.isnan(jsd) else jsd
    pear = float(pearsonr(pr, ps)[0]) if (pr.std() > 0 and ps.std() > 0) else np.nan
    spear = float(spearmanr(pr, ps)[0]) if (pr.std() > 0 and ps.std() > 0) else np.nan

    return {
        f"kmer{k}_coverage_real_in_syn": cov_real_in_syn,
        f"kmer{k}_coverage_syn_in_real": cov_syn_in_real,
        f"kmer{k}_pearson": pear,
        f"kmer{k}_spearman": spear,
        f"kmer{k}_cosine": cos,
        f"kmer{k}_jsd": jsd,
        f"kmer{k}_n_real_kmers": float(n_real),
        f"kmer{k}_n_syn_kmers": float(n_syn),
    }


def _real_cache_key(species: str, seq_type: str, context: str) -> Tuple[str, str, str]:
    return species, seq_type, context


def analyze_all(
    sample_size: Optional[int],
    kmer_sizes: List[int],
    min_seqs: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    real_cache: Dict[Tuple[str, str, str], List[str]] = {}
    rows: List[Dict] = []

    for context in CONTEXTS:
        for model in MODELS:
            for species in SPECIES:
                for seq_type in SEQ_TYPES:
                    key = _real_cache_key(species, seq_type, context)
                    if key not in real_cache:
                        real_cache[key] = load_real_positives(
                            species, seq_type, context, sample_size
                        )
                    real_seqs = real_cache[key]
                    if len(real_seqs) < min_seqs:
                        print(
                            f"  Skip real {context} {species} {seq_type}: "
                            f"n={len(real_seqs)}"
                        )
                        continue

                    for syn_label, lam in SYNTHETIC_CONDITIONS:
                        try:
                            syn_seqs = load_synthetic(
                                model, species, seq_type, context, lam, sample_size
                            )
                        except FileNotFoundError as exc:
                            print(f"  Missing: {exc}")
                            continue
                        if len(syn_seqs) < min_seqs:
                            print(
                                f"  Skip {model} {context} {species} {seq_type} "
                                f"{syn_label}: n={len(syn_seqs)}"
                            )
                            continue

                        n = min(len(real_seqs), len(syn_seqs))
                        real_use = (
                            rng.sample(real_seqs, n)
                            if len(real_seqs) > n
                            else list(real_seqs)
                        )
                        syn_use = (
                            rng.sample(syn_seqs, n)
                            if len(syn_seqs) > n
                            else list(syn_seqs)
                        )

                        row = {
                            "context_bp": int(context),
                            "model": model,
                            "species": species,
                            "species_label": species_display(species),
                            "seq_type": seq_type,
                            "synthetic": syn_label,
                            "lambda": lam,
                            "n_real": len(real_use),
                            "n_syn": len(syn_use),
                        }
                        for k in kmer_sizes:
                            row.update(kmer_metrics(real_use, syn_use, k))
                        rows.append(row)
                        print(
                            f"  OK {context} bp | {model} | {species} {seq_type} | "
                            f"{syn_label} | n={n}"
                        )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_cols = [
        "context_bp",
        "model",
        "species",
        "seq_type",
        "synthetic",
    ]
    return df.sort_values(sort_cols).reset_index(drop=True)


def table_export(df: pd.DataFrame, kmer_sizes: List[int]) -> pd.DataFrame:
    """Clean long-format table for manual LaTeX / spreadsheet layout."""
    site_label = {"donor": "Donor", "acceptor": "Acceptor"}
    out = df[
        [
            "context_bp",
            "species",
            "species_label",
            "seq_type",
            "model",
            "synthetic",
            "lambda",
            "n_real",
            "n_syn",
        ]
    ].copy()
    out["site"] = out["seq_type"].map(site_label)
    for k in sorted(kmer_sizes):
        jsd_col = f"kmer{k}_jsd"
        cov_col = f"kmer{k}_coverage_real_in_syn"
        if jsd_col in df.columns:
            out[f"k{k}_jsd"] = df[jsd_col].round(6)
        if cov_col in df.columns:
            out[f"k{k}_coverage_real_in_syn"] = df[cov_col].round(6)
    col_order = [
        "context_bp",
        "species",
        "species_label",
        "site",
        "seq_type",
        "model",
        "synthetic",
        "lambda",
        "n_real",
        "n_syn",
    ] + [
        c
        for k in sorted(kmer_sizes)
        for c in (f"k{k}_jsd", f"k{k}_coverage_real_in_syn")
        if c in out.columns
    ]
    return out[col_order].sort_values(
        ["context_bp", "species", "seq_type", "model", "synthetic"]
    )


def wide_jsd_export(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per context × species × site; columns = model × synthetic JSD."""
    col = f"kmer{k}_jsd"
    if col not in df.columns:
        raise KeyError(col)
    site_label = {"donor": "Donor", "acceptor": "Acceptor"}
    sub = df.copy()
    sub["site"] = sub["seq_type"].map(site_label)
    sub["synthetic_col"] = sub["synthetic"].str.replace("-", "_").str.lower()
    wide = sub.pivot_table(
        index=["context_bp", "species_label", "site", "seq_type"],
        columns=["model", "synthetic_col"],
        values=col,
        aggfunc="first",
    )
    wide.columns = [f"{model}_{syn}_jsd" for model, syn in wide.columns]
    return wide.reset_index().round(6)


def headline_table(df: pd.DataFrame, kmer_sizes: List[int]) -> pd.DataFrame:
    base_cols = [
        "context_bp",
        "model",
        "species",
        "species_label",
        "seq_type",
        "synthetic",
        "lambda",
        "n_real",
        "n_syn",
    ]
    metric_cols = []
    for k in sorted(kmer_sizes):
        for suffix in (
            "coverage_real_in_syn",
            "coverage_syn_in_real",
            "jsd",
            "cosine",
            "pearson",
        ):
            col = f"kmer{k}_{suffix}"
            if col in df.columns:
                metric_cols.append(col)
    cols = [c for c in base_cols if c in df.columns] + metric_cols
    return df[cols]


def _heatmap(
    df: pd.DataFrame,
    metric: str,
    synthetic: str,
    title: str,
    out_path: str,
    cmap: str = "viridis",
    fmt: str = ".2f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    sub = df[df["synthetic"] == synthetic].copy()
    if sub.empty or metric not in sub.columns:
        return
    sub["row"] = (
        sub["species_label"]
        + " "
        + sub["seq_type"].str.title()
        + " ("
        + sub["context_bp"].astype(str)
        + " bp)"
    )
    sub["col"] = sub["model"]
    table = sub.pivot_table(index="row", columns="col", values=metric, aggfunc="mean")
    if table.empty:
        return

    fig, ax = plt.subplots(
        figsize=(max(5.0, 0.9 * table.shape[1] + 2.5), max(4.0, 0.35 * table.shape[0] + 2.0))
    )
    vals = table.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=0, fontsize=10)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index, fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = vals[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    color="white" if val > np.nanmean(vals) else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def aggregate_figures(df: pd.DataFrame, fig_dir: str, kmer_sizes: List[int]) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    specs = []
    if 6 in kmer_sizes:
        specs.append(
            (
                "kmer6_coverage_real_in_syn",
                "6-mer coverage (real k-mers found in synthetic; higher = better)",
                "_aggregate_kmer6_coverage_real_in_syn_{}.png",
                "viridis",
                ".3f",
                0.0,
                1.0,
            )
        )
    if 4 in kmer_sizes:
        specs.append(
            (
                "kmer4_jsd",
                "4-mer Jensen–Shannon divergence (lower = closer to real)",
                "_aggregate_kmer4_jsd_{}.png",
                "magma_r",
                ".3f",
                None,
                None,
            )
        )
    if 3 in kmer_sizes:
        specs.append(
            (
                "kmer3_cosine",
                "3-mer cosine similarity of spectra (higher = closer to real)",
                "_aggregate_kmer3_cosine_{}.png",
                "viridis",
                ".3f",
                0.0,
                1.0,
            )
        )

    for syn in ("Blend", "No-Blend"):
        slug = syn.lower().replace("-", "_")
        for metric, title, fname, cmap, fmt, vmin, vmax in specs:
            _heatmap(
                df,
                metric,
                syn,
                f"{title}\n({syn})",
                os.path.join(fig_dir, fname.format(slug)),
                cmap=cmap,
                fmt=fmt,
                vmin=vmin,
                vmax=vmax,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-mer coverage analysis for all direct-evaluation datasets"
    )
    parser.add_argument(
        "--out-dir",
        default=OUTPUT_ROOT,
        help="Output directory (default: Revision_Results/direct_evaluation/)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Max sequences per class (default 20000)",
    )
    parser.add_argument(
        "--min-seqs",
        type=int,
        default=100,
        help="Skip comparisons with fewer than this many sequences",
    )
    parser.add_argument(
        "--kmer-sizes",
        default="1,2,3,4,5,6",
        help="Comma-separated k-mer sizes (default 1,2,3,4,5,6)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    kmer_sizes = [int(k.strip()) for k in args.kmer_sizes.split(",") if k.strip()]
    sample_size = args.sample_size if args.sample_size > 0 else None

    print("K-mer coverage analysis (all direct-evaluation datasets)...")
    df = analyze_all(sample_size, kmer_sizes, args.min_seqs, args.seed)
    if df.empty:
        print("No comparisons produced.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "kmer_coverage_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path} ({len(df)} rows)")

    headline = headline_table(df, kmer_sizes)
    headline_path = os.path.join(args.out_dir, "kmer_coverage_headline.csv")
    headline.to_csv(headline_path, index=False)
    print(f"Wrote {headline_path}")

    table = table_export(df, kmer_sizes)
    table_path = os.path.join(args.out_dir, "kmer_jsd_table.csv")
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")

    for k in kmer_sizes:
        if f"kmer{k}_jsd" not in df.columns:
            continue
        wide = wide_jsd_export(df, k)
        wide_path = os.path.join(args.out_dir, f"kmer_jsd_wide_k{k}.csv")
        wide.to_csv(wide_path, index=False)
        print(f"Wrote {wide_path}")

    if not args.no_plots:
        fig_dir = os.path.join(args.out_dir, "kmer_coverage_figures")
        aggregate_figures(df, fig_dir, kmer_sizes)

    print("\n=== Mean 6-mer real→syn coverage by model (Blend) ===")
    if "kmer6_coverage_real_in_syn" in df.columns:
        blend = df[df["synthetic"] == "Blend"]
        print(
            blend.groupby(["model", "context_bp"])["kmer6_coverage_real_in_syn"]
            .mean()
            .round(4)
            .to_string()
        )


if __name__ == "__main__":
    main()
