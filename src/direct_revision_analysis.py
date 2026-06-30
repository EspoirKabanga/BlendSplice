#!/usr/bin/env python3
"""Direct evaluation figures for revision data (GC, conservation, sequence logos).

Compares Real vs synthetic λ=0.0 (No-Blend) vs synthetic λ=0.5 (Blend) for
arabidopsis, human, and danio at 402 bp and 2002 bp.

Outputs PNG figures and CSV summaries under Revision_Results/direct_evaluation/.

Combined GC summary (donor and acceptor, 402+2002):
    python direct_revision_gc_combined_figure.py
"""

import os
import glob
import random
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from direct_analysis import (
    LARGE_SIZE,
    MEDIUM_SIZE,
    analyze_gc_content,
    analyze_nucleotide_conservation,
    create_logo,
    enforce_center_motif,
    load_sequences,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLEND_ROOT = os.path.join(REPO, "data")
GEN_ROOT = os.path.join(REPO, "generated_sequences")
OUTPUT_ROOT = os.path.join(REPO, "results", "direct_evaluation")

MODELS = ["GAN", "VAE", "DIFFUSION"]
SPECIES = ["arabidopsis", "human", "danio"]
SEQ_TYPES = ["donor", "acceptor"]
CONTEXTS = ["402", "2002"]

CONTEXT_CFG = {
    "402": {
        "seq_len": 402,
        "motif_pos": 200,
        "logo_region": (180, 220),
        "cons_region": (150, 250),
    },
    "2002": {
        "seq_len": 2002,
        "motif_pos": 1000,
        "logo_region": (980, 1020),
        "cons_region": (950, 1050),
    },
}

SPECIES_LABEL = {
    "arabidopsis": "Arabidopsis",
    "human": "Human",
    "danio": "Danio",
}


def species_display(species: str) -> str:
    return SPECIES_LABEL.get(species, species.title())


def trim_to_context(sequences: List[str], context: str) -> List[str]:
    cfg = CONTEXT_CFG[context]
    seq_len = cfg["seq_len"]
    if context == "402":
        trimmed = []
        for seq in sequences:
            if len(seq) == 2002:
                trimmed.append(seq[800:1202])
            elif len(seq) == seq_len:
                trimmed.append(seq)
        return list(dict.fromkeys(trimmed))
    return [s for s in sequences if len(s) == seq_len]


def filter_motif(sequences: List[str], seq_type: str, motif_pos: int) -> List[str]:
    motif = "GT" if seq_type == "donor" else "AG"
    return [
        s for s in sequences
        if len(s) >= motif_pos + 2 and s[motif_pos:motif_pos + 2] == motif
    ]


def load_real_positives(
    species: str,
    seq_type: str,
    context: str,
    sample_size: Optional[int],
) -> List[str]:
    folder = species
    path = os.path.join(
        BLEND_ROOT, folder, f"{folder}_{seq_type}_2002_positive_unique.txt"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = load_sequences(path, sample_size)
    cfg = CONTEXT_CFG[context]
    trimmed = trim_to_context(raw, context)
    return filter_motif(trimmed, seq_type, cfg["motif_pos"])


def lambda_tag(lam: float) -> str:
    text = f"{lam:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"lambda_{text}"


def discover_synthetic_file(
    generator: str,
    species: str,
    seq_type: str,
    context: str,
    lam: float,
) -> str:
    model_dir = generator if generator != "DIFFUSION" else "Diffusion"
    lam_s = lambda_tag(lam)
    prefixes = []
    if species == "danio" and context == "402":
        prefixes.append("danio_402")
    elif species == "danio" and context == "2002":
        prefixes.extend(["danio_2002", "danio"])
    elif species == "human" and context == "402":
        prefixes.extend(["homo", "human"])
    else:
        prefixes.append(species)

    gen_dirs = [
        os.path.join(GEN_ROOT, context, model_dir, lam_s),
        os.path.join(GEN_ROOT, model_dir),
    ]

    train_sizes = ("20k", "50k", "100k")

    for gen_dir in gen_dirs:
        patterns: List[str] = []
        for prefix in prefixes:
            if generator == "DIFFUSION":
                patterns.append(os.path.join(gen_dir, f"{prefix}_{seq_type}_{lam_s}_sequences.txt"))
            elif lam == 0.0:
                for n in train_sizes:
                    patterns.append(
                        os.path.join(gen_dir, f"{prefix}_{seq_type}_train_{n}_generated_sequences.txt")
                    )
                    patterns.append(
                        os.path.join(
                            gen_dir,
                            f"{prefix}_{seq_type}_train_{n}_{lam_s}_generated_sequences.txt",
                        )
                    )
            else:
                for n in train_sizes:
                    patterns.append(
                        os.path.join(
                            gen_dir,
                            f"{prefix}_{seq_type}_train_{n}_{lam_s}_generated_sequences.txt",
                        )
                    )
        for path in patterns:
            if os.path.isfile(path):
                return path

    gen_dir = gen_dirs[0]
    if context == "402" and species == "human":
        homo_pat = os.path.join(gen_dir, f"*homo*{seq_type}*")
        if lam == 0.0:
            homo_matches = sorted(
                p for p in glob.glob(homo_pat)
                if p.endswith("_generated_sequences.txt") or p.endswith("_sequences.txt")
            )
        else:
            homo_matches = sorted(glob.glob(f"{homo_pat}*{lam_s}*"))
        if homo_matches:
            return homo_matches[0]

    glob_pat = os.path.join(gen_dir, f"*{species}*{seq_type}*{lam_s}*")
    matches = sorted(glob.glob(glob_pat))
    if generator in ("GAN", "VAE"):
        matches = [p for p in matches if p.endswith("_generated_sequences.txt")]
    else:
        matches = [p for p in matches if p.endswith("_sequences.txt")]
    if context == "402" and species != "danio":
        matches = [p for p in matches if "402" not in os.path.basename(p)]
    elif context == "402" and species == "danio":
        matches = [p for p in matches if "danio_402" in os.path.basename(p)] or matches
    elif context == "2002":
        matches = [p for p in matches if "danio_402" not in os.path.basename(p)]
    return matches[0] if matches else ""


def load_synthetic(
    generator: str,
    species: str,
    seq_type: str,
    context: str,
    lam: float,
    sample_size: Optional[int],
) -> List[str]:
    path = discover_synthetic_file(generator, species, seq_type, context, lam)
    if not path:
        raise FileNotFoundError(
            f"No synthetic file for {generator} {species} {seq_type} "
            f"context={context} {lambda_tag(lam)}"
        )
    raw = load_sequences(path, sample_size)
    cfg = CONTEXT_CFG[context]
    trimmed = trim_to_context(raw, context)
    seqs = filter_motif(trimmed, seq_type, cfg["motif_pos"])
    if generator == "GAN":
        seqs = [enforce_center_motif(s, seq_type, cfg["seq_len"]) for s in seqs]
    return seqs


def output_prefix(context: str, model: str, species: str, seq_type: str) -> str:
    return os.path.join(OUTPUT_ROOT, context, model, f"{species}_{seq_type}")


def save_gc_plot(
    species: str,
    seq_type: str,
    context: str,
    model: str,
    data: Dict,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    real_gc = data["gc_content"]["real"]
    blend_gc = data["gc_content"]["blend"]
    no_blend_gc = data["gc_content"]["no_blend"]
    gc_df = pd.DataFrame({
        "GC_Content": real_gc + blend_gc + no_blend_gc,
        "Dataset": (
            ["Real"] * len(real_gc)
            + ["Blend (λ=0.5)"] * len(blend_gc)
            + ["No-Blend (λ=0.0)"] * len(no_blend_gc)
        ),
    })
    sns.histplot(data=gc_df, x="GC_Content", hue="Dataset", kde=True, bins=30, alpha=0.6, ax=ax)
    ax.axvline(data["means"]["real"], color="blue", linestyle="--", label="Real")
    ax.axvline(data["means"]["blend"], color="orange", linestyle="--", label="Blend")
    ax.axvline(data["means"]["no_blend"], color="green", linestyle="--", label="No-Blend")
    ax.set_title(
        f"{species_display(species)} {seq_type.title()} ({context} bp) - GC Content",
        fontsize=LARGE_SIZE,
    )
    ax.set_xlabel("GC Content (%)", fontsize=LARGE_SIZE)
    ax.set_ylabel("Count", fontsize=LARGE_SIZE)
    ax.legend(fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    path = f"{output_prefix(context, model, species, seq_type)}_gc_content.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_conservation_plot(
    species: str,
    seq_type: str,
    context: str,
    model: str,
    data: Dict,
) -> None:
    cfg = CONTEXT_CFG[context]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    positions = np.arange(len(data["conservation"]["real"]))
    ax.plot(positions, data["conservation"]["real"], label="Real", color="blue", linewidth=1.2)
    ax.plot(positions, data["conservation"]["blend"], label="Blend (λ=0.5)", color="orange", linewidth=1.2)
    ax.plot(positions, data["conservation"]["no_blend"], label="No-Blend (λ=0.0)", color="green", linewidth=1.2)
    ax.set_title(
        f"{species_display(species)} {seq_type.title()} ({context} bp) - Conservation",
        fontsize=LARGE_SIZE,
    )
    ax.set_xlabel("Position", fontsize=LARGE_SIZE)
    ax.set_ylabel("Conservation Score", fontsize=LARGE_SIZE)
    ax.set_ylim(0, 1)
    ax.set_xlim(*cfg["cons_region"])
    ax.legend(fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    path = f"{output_prefix(context, model, species, seq_type)}_conservation.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def save_sequence_logo_plot(
    species: str,
    seq_type: str,
    context: str,
    model: str,
    data: Dict,
) -> None:
    cfg = CONTEXT_CFG[context]
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    labels = ["Real", "Blend (λ=0.5)", "No-Blend (λ=0.0)"]
    pwms = [data["pwm"]["real"], data["pwm"]["blend"], data["pwm"]["no_blend"]]
    for i, (label, pwm) in enumerate(zip(labels, pwms)):
        create_logo(pwm, axes[i], region=cfg["logo_region"])
        axes[i].set_title(
            f"{label} - {species_display(species)} {seq_type.title()} ({context} bp)",
            fontsize=MEDIUM_SIZE,
        )
        if i < 2:
            axes[i].set_xlabel("")
    plt.tight_layout()
    path = f"{output_prefix(context, model, species, seq_type)}_sequence_logos.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def process_dataset(
    context: str,
    model: str,
    species: str,
    seq_type: str,
    sample_size: Optional[int],
) -> Dict:
    print(f"Processing {context} bp | {model} | {species} {seq_type}...")
    real_seqs = load_real_positives(species, seq_type, context, sample_size)
    blend_seqs = load_synthetic(model, species, seq_type, context, 0.5, sample_size)
    no_blend_seqs = load_synthetic(model, species, seq_type, context, 0.0, sample_size)
    n = min(len(real_seqs), len(blend_seqs), len(no_blend_seqs))
    if sample_size:
        n = min(n, sample_size)
    real_seqs = real_seqs[:n]
    blend_seqs = blend_seqs[:n]
    no_blend_seqs = no_blend_seqs[:n]
    print(f"  Using n={n} sequences per class")

    prefix = output_prefix(context, model, species, seq_type)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    cons = analyze_nucleotide_conservation(real_seqs, blend_seqs, no_blend_seqs, prefix)
    gc = analyze_gc_content(real_seqs, blend_seqs, no_blend_seqs, prefix)
    return {
        "pwm": cons["pwm"],
        "conservation": cons["conservation"],
        "gc_content": gc["gc_content"],
        "means": gc["means"],
    }


def main():
    parser = argparse.ArgumentParser(description="Revision direct evaluation figures")
    parser.add_argument(
        "--context",
        choices=CONTEXTS + ["all"],
        default="all",
        help="Sequence context: 402, 2002, or all (default)",
    )
    parser.add_argument(
        "--species",
        default=None,
        help="Comma-separated species (default: all)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Comma-separated generators: GAN,VAE,DIFFUSION (default: all)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20000,
        help="Max sequences per class (default 20000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    context_list = CONTEXTS if args.context == "all" else [args.context]
    species_list = (
        [s.strip() for s in args.species.split(",") if s.strip()]
        if args.species
        else SPECIES
    )
    model_list = (
        [m.strip().upper() for m in args.model.split(",") if m.strip()]
        if args.model
        else MODELS
    )

    for context in context_list:
        for model in model_list:
            for species in species_list:
                for seq_type in SEQ_TYPES:
                    try:
                        data = process_dataset(
                            context, model, species, seq_type, args.sample_size
                        )
                        save_gc_plot(species, seq_type, context, model, data)
                        save_conservation_plot(species, seq_type, context, model, data)
                        save_sequence_logo_plot(species, seq_type, context, model, data)
                    except Exception as exc:
                        print(f"ERROR {context}/{model}/{species}/{seq_type}: {exc}")

    print(f"\nDone. Results under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
