"""Score splice sites with the pretrained SpliceAI ensemble (fixed position).

Detection rate (per file):
    #{sequences with p_SpliceAI(center) >= tau} / #{sequences}

402 bp (--context 402):
  Real: DRANet positives (arabidopsis/human) + danio positive_unique trimmed
  Synthetic: src/GAN_generated_sequences, src/VAE_generated_sequences,
             src/Lambda_sensitivity_analysis, and (optionally)
             Revision_Generated_Sequences/danio_402_* (402 bp)

2002 bp (--context 2002):
  Real: BlendSplice_Dataset/*/*_2002_positive_unique.txt (all species)
  Synthetic: Revision_Generated_Sequences/{GAN,VAE,Diffusion} (danio_402 excluded)

Usage (conda BaP):
    python test_sequence_spliceai.py --context 402
    python test_sequence_spliceai.py --context 2002 --limit 500
    python test_sequence_spliceai.py --context 402 --danio-402-revision-only \\
        --limit 20000 --seed 42 --append
"""

import os
import re
import glob
import csv
import random
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLD = 0.5
CONTEXT_PAD = 10000
BATCH = 32
DEFAULT_SEED = 42
RESULTS_DIR = os.path.join(REPO, "results")

BLEND_ROOT = os.path.join(REPO, "data")
GEN_ROOT = os.path.join(REPO, "generated_sequences")

SITE_CHANNEL = {"acceptor": 1, "donor": 2}

# Per-context sequence geometry and SpliceAI output indices at the central site.
CONTEXT_CFG = {
    "402": {
        "seq_len": 402,
        "motif_pos": 200,
        "site_idx": {"acceptor": 202, "donor": 199},
        "summary_csv": "spliceai_detection_summary.csv",
        "detail_csv": "spliceai_detection_detail.csv",
    },
    "2002": {
        "seq_len": 2002,
        "motif_pos": 1000,
        "site_idx": {"acceptor": 1002, "donor": 999},
        "summary_csv": "spliceai_detection_summary_2002.csv",
        "detail_csv": "spliceai_detection_detail_2002.csv",
    },
}

# Legacy 402 bp synthetic roots (optional external paths).
GEN_ROOTS_402 = {
    "gan": os.path.join(REPO, "src", "GAN_generated_sequences"),
    "vae": os.path.join(REPO, "src", "VAE_generated_sequences"),
    "diffusion": os.path.join(REPO, "src", "Lambda_sensitivity_analysis"),
}
REVISION_GEN_DIRS = {"gan": "GAN", "vae": "VAE", "diffusion": "Diffusion"}

SPECIES_LIST = ["arabidopsis", "human", "danio"]
SEQ_TYPES = ["acceptor", "donor"]


def build_real_jobs_402() -> list:
    jobs = []
    for species in SPECIES_LIST:
        for seq_type in SEQ_TYPES:
            jobs.append({
                "species": species, "seq_type": seq_type, "data_type": "real",
                "generator": "real", "lambda": "",
                "path": os.path.join(
                    BLEND_ROOT, species,
                    f"{species}_{seq_type}_2002_positive_unique.txt",
                ),
                "trim": True, "enforce_gan": False,
            })
    return jobs


def build_real_jobs_2002() -> list:
    jobs = []
    for species in SPECIES_LIST:
        for seq_type in SEQ_TYPES:
            path = os.path.join(
                BLEND_ROOT, species,
                f"{species}_{seq_type}_2002_positive_unique.txt",
            )
            jobs.append({
                "species": species, "seq_type": seq_type, "data_type": "real",
                "generator": "real", "lambda": "",
                "path": path, "trim": False, "enforce_gan": False,
            })
    return jobs


def enforce_motif(seq: str, site_type: str, motif_pos: int) -> str:
    motif = "GT" if site_type == "donor" else "AG"
    if len(seq) >= motif_pos + 2:
        return seq[:motif_pos] + motif + seq[motif_pos + 2:]
    return seq


def load_sequences(path, site_type, cfg, trim=False, enforce_gan=False) -> list:
    seq_len = cfg["seq_len"]
    motif_pos = cfg["motif_pos"]
    motif = "GT" if site_type == "donor" else "AG"
    seqs = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper()
            if not s:
                continue
            if trim:
                if len(s) != 2002:
                    continue
                s = s[800:1202]
            if len(s) != seq_len:
                continue
            if enforce_gan:
                s = enforce_motif(s, site_type, motif_pos)
            if s[motif_pos:motif_pos + 2] != motif:
                continue
            seqs.append(s)
    return seqs


def sample_sequences(seqs, limit, seed):
    n_avail = len(seqs)
    if limit is None or limit >= n_avail:
        return seqs, list(range(n_avail))
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(n_avail), limit))
    return [seqs[i] for i in idxs], idxs


def skip_danio_402(name: str) -> bool:
    return "danio_402" in name


def parse_revision_gan_vae(name: str, generator: str):
    """Parse Revision_Generated_Sequences GAN/VAE filenames (2002 bp)."""
    if skip_danio_402(name):
        return None
    # danio_2002_acceptor_train_20k_lambda_0.5_generated_sequences.txt
    m = re.match(
        r"(arabidopsis|human|danio)(?:_2002)?_(acceptor|donor)_train_.*?(?:lambda_([\d.]+)_)?generated_sequences\.txt",
        name,
    )
    if not m:
        return None
    sp, st = m.group(1), m.group(2)
    lam = float(m.group(3)) if m.group(3) else 0.0
    return {"species": sp, "seq_type": st, "data_type": "synthetic",
            "generator": generator, "lambda": lam}


def parse_revision_diffusion(name: str):
    """Parse Revision_Generated_Sequences/Diffusion/*_lambda_*_sequences.txt."""
    if skip_danio_402(name):
        return None
    m = re.match(
        r"(arabidopsis|human|danio)(?:_2002)?_(acceptor|donor)_lambda_([\d.]+)_sequences\.txt",
        name,
    )
    if not m:
        return None
    return {"species": m.group(1), "seq_type": m.group(2), "data_type": "synthetic",
            "generator": "diffusion", "lambda": float(m.group(3))}


def parse_src_gan_vae_402(name: str, generator: str):
    """Parse src/GAN or src/VAE 402 bp filenames (arabidopsis/homo only)."""
    if not (name.startswith("arabidopsis_") or name.startswith("homo_")):
        return None
    species = "human" if name.startswith("homo_") else "arabidopsis"
    if "_acceptor_" in name:
        seq_type = "acceptor"
    elif "_donor_" in name:
        seq_type = "donor"
    else:
        return None
    m = re.search(r"lambda_([\d.]+)", name)
    lam = float(m.group(1)) if m else 0.0
    return {"species": species, "seq_type": seq_type, "data_type": "synthetic",
            "generator": generator, "lambda": lam}


def parse_src_diffusion_402(name: str):
    m = re.match(r"(arabidopsis|homo)_(acceptor|donor)_lambda_([\d.]+)_sequences\.txt", name)
    if not m:
        return None
    sp = "human" if m.group(1) == "homo" else "arabidopsis"
    return {"species": sp, "seq_type": m.group(2), "data_type": "synthetic",
            "generator": "diffusion", "lambda": float(m.group(3))}


def parse_revision_danio_402(name: str, generator: str):
    """Parse Revision_Generated_Sequences danio_402 402 bp filenames."""
    if generator in ("gan", "vae"):
        m = re.match(
            r"danio_402_(acceptor|donor)_train_.*?(?:lambda_([\d.]+)_)?generated_sequences\.txt",
            name,
        )
        if not m:
            return None
        lam = float(m.group(2)) if m.group(2) else 0.0
        return {"species": "danio_402", "seq_type": m.group(1), "data_type": "synthetic",
                "generator": generator, "lambda": lam}
    m = re.match(r"danio_402_(acceptor|donor)_lambda_([\d.]+)_sequences\.txt", name)
    if not m:
        return None
    return {"species": "danio_402", "seq_type": m.group(1), "data_type": "synthetic",
            "generator": "diffusion", "lambda": float(m.group(2))}


def discover_danio_402_revision_jobs() -> list:
    """Synthetic danio_402 jobs from generated_sequences/402/ (402 bp)."""
    jobs = []
    for gen in ("gan", "vae"):
        root = os.path.join(GEN_ROOT, "402", REVISION_GEN_DIRS[gen])
        if not os.path.isdir(root):
            continue
        for lam_dir in sorted(glob.glob(os.path.join(root, "lambda_*"))):
            for path in sorted(glob.glob(os.path.join(lam_dir, "danio_402_*.txt"))):
                meta = parse_revision_danio_402(os.path.basename(path), gen)
                if meta:
                    jobs.append({**meta, "path": path, "trim": False,
                                 "enforce_gan": gen == "gan"})
    diff_root = os.path.join(GEN_ROOT, "402", REVISION_GEN_DIRS["diffusion"])
    if os.path.isdir(diff_root):
        for lam_dir in sorted(glob.glob(os.path.join(diff_root, "lambda_*"))):
            for path in sorted(glob.glob(os.path.join(lam_dir, "danio_402_*_lambda_*_sequences.txt"))):
                meta = parse_revision_danio_402(os.path.basename(path), "diffusion")
                if meta:
                    jobs.append({**meta, "path": path, "trim": False, "enforce_gan": False})
    return jobs


def discover_jobs_from_generated_root(context: str) -> list:
    """Synthetic jobs from generated_sequences/{context}/{model}/lambda_*/*.txt."""
    jobs = []
    for gen in ("gan", "vae"):
        root = os.path.join(GEN_ROOT, context, REVISION_GEN_DIRS[gen])
        if not os.path.isdir(root):
            continue
        for lam_dir in sorted(glob.glob(os.path.join(root, "lambda_*"))):
            for path in sorted(glob.glob(os.path.join(lam_dir, "*.txt"))):
                name = os.path.basename(path)
                if context == "402" and name.startswith("danio_402_"):
                    meta = parse_revision_danio_402(name, gen)
                elif context == "402":
                    meta = parse_src_gan_vae_402(name, gen) or parse_revision_gan_vae(name, gen)
                elif context == "2002":
                    meta = parse_revision_gan_vae(name, gen)
                else:
                    meta = parse_src_gan_vae_402(name, gen) or parse_revision_gan_vae(name, gen)
                if meta:
                    jobs.append({**meta, "path": path, "trim": False,
                                 "enforce_gan": gen == "gan"})
    diff_root = os.path.join(GEN_ROOT, context, REVISION_GEN_DIRS["diffusion"])
    if os.path.isdir(diff_root):
        for lam_dir in sorted(glob.glob(os.path.join(diff_root, "lambda_*"))):
            for path in sorted(glob.glob(os.path.join(lam_dir, "*_lambda_*_sequences.txt"))):
                name = os.path.basename(path)
                if context == "402" and name.startswith("danio_402_"):
                    meta = parse_revision_danio_402(name, "diffusion")
                elif context == "402":
                    meta = parse_src_diffusion_402(name) or parse_revision_diffusion(name)
                elif context == "2002":
                    meta = parse_revision_diffusion(name)
                else:
                    meta = parse_src_diffusion_402(name) or parse_revision_diffusion(name)
                if meta:
                    jobs.append({**meta, "path": path, "trim": False, "enforce_gan": False})
    return jobs


def discover_synthetic_jobs(context: str, include_danio_402_revision: bool = False) -> list:
    jobs = discover_jobs_from_generated_root(context)
    if jobs:
        if include_danio_402_revision and context == "402":
            jobs.extend(discover_danio_402_revision_jobs())
        return jobs

    jobs = []
    if context == "402":
        for gen in ("gan", "vae"):
            root = GEN_ROOTS_402[gen]
            if not os.path.isdir(root):
                continue
            for path in sorted(glob.glob(os.path.join(root, "*.txt"))):
                meta = parse_src_gan_vae_402(os.path.basename(path), gen)
                if meta:
                    jobs.append({**meta, "path": path, "trim": False,
                                 "enforce_gan": gen == "gan"})
        diff_root = GEN_ROOTS_402["diffusion"]
        if os.path.isdir(diff_root):
            for path in sorted(glob.glob(os.path.join(diff_root, "*_lambda_*_sequences.txt"))):
                meta = parse_src_diffusion_402(os.path.basename(path))
                if meta:
                    jobs.append({**meta, "path": path, "trim": False, "enforce_gan": False})
    if include_danio_402_revision and context == "402":
        jobs.extend(discover_danio_402_revision_jobs())
    return jobs


def read_csv_rows(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_job(job, models, pad, limit, seed, cfg):
    site_type = job["seq_type"]
    channel = SITE_CHANNEL[site_type]
    site_idx = cfg["site_idx"][site_type]

    all_seqs = load_sequences(
        job["path"], site_type, cfg,
        trim=job.get("trim", False),
        enforce_gan=job.get("enforce_gan", False),
    )
    n_avail = len(all_seqs)
    seqs, orig_idxs = sample_sequences(all_seqs, limit, seed)

    label = (f"{job['species']} {site_type} | {job['data_type']} | "
             f"{job['generator']} lambda={job['lambda']}")

    print(f"\n{'=' * 60}")
    print(label)
    print(f"  file       : {job['path']}")
    print(f"  available  : {n_avail}  sampled: {len(seqs)}  (seed={seed})")
    print(f"  check idx  : {site_idx}  tau: {THRESHOLD}")

    summary = {
        **job,
        "context": cfg["seq_len"],
        "n_available": n_avail,
        "n_sampled": len(seqs),
        "n_detected": 0,
        "detection_rate": 0.0,
        "tau": THRESHOLD,
        "site_idx": site_idx,
        "seed": seed,
    }
    details = []

    if not seqs:
        print("  No sequences to check.")
        return summary, details

    progress = tqdm(total=len(seqs), desc=label[:40], unit="seq",
                    dynamic_ncols=True) if HAVE_TQDM else None
    hits = 0

    for start in range(0, len(seqs), BATCH):
        batch = seqs[start:start + BATCH]
        batch_orig = orig_idxs[start:start + BATCH]
        x = np.stack([one_hot_encode(pad + s + pad) for s in batch])
        y = np.mean([models[m](x, training=False).numpy() for m in range(5)], axis=0)

        for k, orig_i in enumerate(batch_orig):
            prob = float(y[k, site_idx, channel])
            detected = int(prob >= THRESHOLD)
            if detected:
                hits += 1
            details.append({
                "context": cfg["seq_len"],
                "species": job["species"],
                "seq_type": site_type,
                "data_type": job["data_type"],
                "generator": job["generator"],
                "lambda": job["lambda"],
                "file": job["path"],
                "seq_index": orig_i,
                "p_spliceai_center": prob,
                "detected": detected,
                "site_idx": site_idx,
                "tau": THRESHOLD,
            })
        if progress is not None:
            progress.update(len(batch))

    if progress is not None:
        progress.close()

    n_seqs = len(seqs)
    rate = hits / n_seqs
    summary["n_detected"] = hits
    summary["detection_rate"] = rate
    print(f"  Detection rate: {hits}/{n_seqs} = {rate:.4f} ({100 * rate:.2f}%)")
    return summary, details


def main():
    parser = argparse.ArgumentParser(description="SpliceAI fixed-position detection")
    parser.add_argument("--context", choices=["402", "2002"], default="402",
                        help='Sequence window: "402" (DRANet/src) or "2002" (BlendSplice/Revision)')
    parser.add_argument("--limit", type=int, default=None,
                        help="Random sample size per file (default: all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--out-dir", default=RESULTS_DIR,
                        help="Directory for output CSV files")
    parser.add_argument(
        "--danio-402-revision-only",
        action="store_true",
        help="Run only danio_402 synthetic files from Revision_Generated_Sequences (402 bp)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new rows to existing summary/detail CSVs instead of overwriting",
    )
    args = parser.parse_args()

    if args.danio_402_revision_only and args.context != "402":
        parser.error("--danio-402-revision-only requires --context 402")

    cfg = CONTEXT_CFG[args.context]
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, cfg["summary_csv"])
    detail_path = os.path.join(args.out_dir, cfg["detail_csv"])

    real_jobs = build_real_jobs_2002() if args.context == "2002" else build_real_jobs_402()
    if args.danio_402_revision_only:
        jobs = discover_danio_402_revision_jobs()
        print("Mode: danio_402 Revision_Generated_Sequences only (synthetic)")
    else:
        jobs = real_jobs + discover_synthetic_jobs(args.context)

    print(f"Context: {args.context} bp  (seq_len={cfg['seq_len']}, "
          f"acceptor idx={cfg['site_idx']['acceptor']}, "
          f"donor idx={cfg['site_idx']['donor']})")
    print(f"Jobs: {len(jobs)} total"
          + ("" if args.danio_402_revision_only
             else f" ({len(real_jobs)} real + {len(jobs) - len(real_jobs)} synthetic)"))
    if args.limit:
        print(f"Random sampling: {args.limit} sequences per file (seed={args.seed})")

    print("Loading SpliceAI ensemble (models 1-5)...")
    models = [load_model(resource_filename("spliceai", f"models/spliceai{x}.h5"))
              for x in range(1, 6)]
    pad = "N" * (CONTEXT_PAD // 2)

    summaries, all_details = [], []
    for job in jobs:
        if not os.path.isfile(job["path"]):
            print(f"\n[skip] missing: {job['path']}")
            continue
        summary, details = run_job(job, models, pad, args.limit, args.seed, cfg)
        summaries.append(summary)
        all_details.extend(details)

    sum_fields = [
        "context", "species", "seq_type", "data_type", "generator", "lambda",
        "n_available", "n_sampled", "n_detected", "detection_rate",
        "tau", "site_idx", "seed", "path",
    ]
    det_fields = [
        "context", "species", "seq_type", "data_type", "generator", "lambda",
        "file", "seq_index", "p_spliceai_center", "detected", "site_idx", "tau",
    ]

    if args.append:
        existing_sum = read_csv_rows(summary_path)
        existing_det = read_csv_rows(detail_path)
        new_paths = {s["path"] for s in summaries}
        existing_sum = [r for r in existing_sum if r.get("path") not in new_paths]
        existing_det = [
            r for r in existing_det
            if r.get("file") not in new_paths
        ]
        summaries = existing_sum + summaries
        all_details = existing_det + all_details

    write_csv_rows(summary_path, sum_fields, summaries)
    write_csv_rows(detail_path, det_fields, all_details)

    total_det = sum(s["n_detected"] for s in summaries)
    total_samp = sum(s["n_sampled"] for s in summaries)
    print(f"\n{'=' * 60}")
    print(f"Overall: {total_det}/{total_samp} detected "
          f"({100 * total_det / max(total_samp, 1):.2f}%)")
    print(f"Summary CSV : {summary_path}")
    print(f"Detail CSV  : {detail_path}")


if __name__ == "__main__":
    main()
