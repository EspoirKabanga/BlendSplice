"""Shared config and data loading for revision supplementary analyses."""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from direct_revision_analysis import (  # noqa: E402
    CONTEXT_CFG,
    CONTEXTS,
    MODELS,
    SPECIES,
    SEQ_TYPES,
    load_real_positives,
    load_synthetic,
    species_display,
)

REVISION_SUPP_ROOT = os.path.join(REPO, "results", "supplementary")
SUPPLEMENTARY_LAMBDAS = [0.25, 0.75]
COMPREHENSIVE_LAMBDAS = [0.0, 0.25, 0.5, 0.75]

MODEL_LABEL = {"GAN": "GAN", "VAE": "VAE", "DIFFUSION": "Diffusion"}

DATASETS = [
    ("arabidopsis", "donor"),
    ("arabidopsis", "acceptor"),
    ("human", "donor"),
    ("human", "acceptor"),
    ("danio", "donor"),
    ("danio", "acceptor"),
]

GRID_2X3 = {
    "arabidopsis_donor": (0, 0),
    "human_donor": (0, 1),
    "danio_donor": (0, 2),
    "arabidopsis_acceptor": (1, 0),
    "human_acceptor": (1, 1),
    "danio_acceptor": (1, 2),
}

TITLES_2X3 = {
    "arabidopsis_donor": "Arabidopsis Donor",
    "human_donor": "Human Donor",
    "danio_donor": "Danio Donor",
    "arabidopsis_acceptor": "Arabidopsis Acceptor",
    "human_acceptor": "Human Acceptor",
    "danio_acceptor": "Danio Acceptor",
}


def lambda_key(lam: float) -> str:
    text = f"{lam:.2f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return f"lambda_{text}"


def load_revision_lambda_data(
    context: str,
    species: str,
    seq_type: str,
    model: str,
    sample_size: Optional[int] = 20000,
    lambdas: Optional[List[float]] = None,
) -> Dict[str, List[str]]:
    """Load Real plus synthetic sequences for requested lambda values."""
    if lambdas is None:
        lambdas = COMPREHENSIVE_LAMBDAS

    real_seqs = load_real_positives(species, seq_type, context, sample_size)
    data: Dict[str, List[str]] = {"real": real_seqs}

    for lam in lambdas:
        try:
            syn = load_synthetic(model, species, seq_type, context, lam, sample_size)
        except FileNotFoundError as exc:
            print(f"  Warning: {exc}")
            continue
        if len(syn) < 50:
            print(f"  Warning: too few sequences for λ={lam}: n={len(syn)}")
            continue
        data[lambda_key(lam)] = syn
        print(f"  Loaded λ={lam}: {len(syn)} sequences")

    return data


def context_cfg(context: str) -> dict:
    return CONTEXT_CFG[context]


def output_dir(*parts: str) -> str:
    path = os.path.join(REVISION_SUPP_ROOT, *parts)
    os.makedirs(path, exist_ok=True)
    return path
