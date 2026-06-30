#!/usr/bin/env python3
"""Copy generated sequences into generated_sequences/{context}/{model}/{lambda}/."""

from __future__ import annotations

import os
import re
import shutil

BLENDSPLICE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.normpath(os.path.join(BLENDSPLICE, ".."))
DST = os.path.join(BLENDSPLICE, "generated_sequences")

REVISION_SRC = os.path.join(REPO, "Revision_Generated_Sequences")
SRC_402 = {
    "GAN": os.path.join(REPO, "src", "GAN_generated_sequences"),
    "VAE": os.path.join(REPO, "src", "VAE_generated_sequences"),
    "Diffusion": os.path.join(REPO, "src", "Lambda_sensitivity_analysis"),
}

MODELS = ("GAN", "VAE", "Diffusion")
LAM_RE = re.compile(r"lambda_(\d+(?:\.\d+)?)")


def infer_context_revision(filename: str) -> str:
    if "danio_402" in filename:
        return "402"
    return "2002"


def extract_lambda_dir(filename: str) -> str | None:
    match = LAM_RE.search(filename)
    if match:
        return f"lambda_{match.group(1)}"
    if filename.endswith("_generated_sequences.txt") and "lambda_" not in filename:
        return "lambda_0.0"
    return None


def should_copy_revision(fname: str) -> bool:
    return fname.endswith(".txt")


def should_copy_402_diffusion(fname: str) -> bool:
    if not fname.endswith("_sequences.txt"):
        return False
    if "danio_" in fname:
        return False
    if "lambda_1.0" in fname:
        return False
    return fname.startswith("arabidopsis_") or fname.startswith("homo_")


def should_copy_402_gan_vae(fname: str) -> bool:
    return fname.endswith("_generated_sequences.txt")


def copy_file(src_path: str, context: str, model: str, lam_dir: str, fname: str) -> None:
    dest_dir = os.path.join(DST, context, model, lam_dir)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(src_path, os.path.join(dest_dir, fname))
    print(f"  {context}/{model}/{lam_dir}/{fname}")


def copy_revision_sequences() -> int:
    copied = 0
    for model in MODELS:
        src_dir = os.path.join(REVISION_SRC, model)
        if not os.path.isdir(src_dir):
            print(f"Skip missing revision dir: {src_dir}")
            continue
        for fname in sorted(os.listdir(src_dir)):
            if not should_copy_revision(fname):
                continue
            lam_dir = extract_lambda_dir(fname)
            if not lam_dir:
                print(f"Skip (no lambda): {fname}")
                continue
            context = infer_context_revision(fname)
            copy_file(os.path.join(src_dir, fname), context, model, lam_dir, fname)
            copied += 1
    return copied


def copy_402_native_sequences() -> int:
    """Arabidopsis/human native 402 bp from src/GAN_generated_sequences etc."""
    copied = 0
    for model, src_dir in SRC_402.items():
        if not os.path.isdir(src_dir):
            print(f"Skip missing 402 dir: {src_dir}")
            continue
        for fname in sorted(os.listdir(src_dir)):
            if model == "Diffusion":
                if not should_copy_402_diffusion(fname):
                    continue
            elif not should_copy_402_gan_vae(fname):
                continue
            lam_dir = extract_lambda_dir(fname)
            if not lam_dir:
                print(f"Skip (no lambda): {fname}")
                continue
            copy_file(os.path.join(src_dir, fname), "402", model, lam_dir, fname)
            copied += 1
    return copied


def main() -> None:
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    n_rev = copy_revision_sequences()
    n_402 = copy_402_native_sequences()
    print(f"\nCopied {n_rev} revision + {n_402} native-402 files to {DST}")


if __name__ == "__main__":
    main()
