#!/usr/bin/env python3

import os
import glob
import json
import csv
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import re

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Reuse model factory and training/eval utilities where possible
from proxy import get_model_instance  # model factory
from proxy_test import create_loader as create_loader_eval  # evaluation loader only
from proxy_augmentation_test import load_ens_mask_trimmed  # ENS trimming loader


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(42)


def load_synthetic_sequences(file_path: str) -> List[str]:
    sequences: List[str] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip().upper()
            if s:
                sequences.append(s)
    return sequences


def discover_saved_models_real() -> List[str]:
    paths = glob.glob(os.path.join('src', 'saved_models', '*.pth'))
    out: List[str] = []
    for p in paths:
        try:
            meta = torch.load(p, map_location='cpu')
        except Exception:
            continue
        if meta.get('data_type') == 'real':
            out.append(p)
    return sorted(out)


def discover_saved_models_synthetic() -> List[str]:
    paths = glob.glob(os.path.join('src', 'saved_models', '*.pth'))
    out: List[str] = []
    for p in paths:
        try:
            meta = torch.load(p, map_location='cpu')
        except Exception:
            continue
        if meta.get('data_type') == 'synthetic':
            out.append(p)
    return sorted(out)


def discover_saved_models_augmented() -> List[str]:
    return sorted(glob.glob(os.path.join('src', 'augmented_saved_models', '*.pth')))


def discover_saved_models_baseline() -> List[str]:
    return sorted(glob.glob(os.path.join('src', 'baseline_saved_models', '*.pth')))


def parse_species_seqtype_from_generated_filename(name: str) -> Tuple[str, str]:
    # Expected patterns like: arabidopsis_donor_train_50k_generated_sequences.txt or arabidopsis_acceptor_train_50k_lambda_0.5_generated_sequences.txt
    parts = os.path.basename(name).split('_')
    species = parts[0]
    seq_type = parts[1]
    return species, seq_type


def discover_gan_generated_files() -> List[str]:
    base_dir = os.path.join('src', 'GAN_generated_sequences')
    return sorted(glob.glob(os.path.join(base_dir, '*.txt')))


def discover_vae_generated_files() -> List[str]:
    base_dir = os.path.join('src', 'VAE_generated_sequences')
    return sorted(glob.glob(os.path.join(base_dir, '*.txt')))


def infer_lambda_from_filename(path: str) -> str:
    name = os.path.basename(path)
    # Match patterns like lambda_0.5 or lambda0.5
    m = re.search(r"lambda[_-]?(\d+(?:\.\d+)?)", name)
    if m:
        return m.group(1)
    return '0.0'


def eval_on_ens(
    model_path: str,
    device: torch.device,
    ens_pos: List[str],
    ens_neg: List[str],
    test_size_per_class: int = None,
) -> Dict[str, Any]:
    payload = torch.load(model_path, map_location='cpu')
    model_name = payload.get('model_name')
    species = payload.get('species')
    seq_type = payload.get('seq_type')

    model = get_model_instance(model_name)
    state_dict = payload.get('model_state_dict') or payload.get('generator_state_dict') or payload.get('discriminator_state_dict')
    model.load_state_dict(state_dict, strict=False)

    # Keep a consistent test size across scenarios
    max_class_n = min(len(ens_pos), len(ens_neg))
    n = max_class_n if test_size_per_class is None else min(max_class_n, test_size_per_class)

    # Deterministic subsample
    rng = random.Random(42)
    pos_sample = list(ens_pos)
    neg_sample = list(ens_neg)
    rng.shuffle(pos_sample)
    rng.shuffle(neg_sample)
    pos_sample = pos_sample[:n]
    neg_sample = neg_sample[:n]

    test_sequences = pos_sample + neg_sample
    test_labels = [1] * len(pos_sample) + [0] * len(neg_sample)

    # Evaluate with probability outputs for AUC/AUPRC
    loader = create_loader_eval(test_sequences, test_labels, batch_size=64)
    import torch.nn.functional as F
    model.to(device)
    model.eval()
    all_scores: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_scores.extend(probs.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    total_samples = len(all_labels)
    # Metrics
    tp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 1)
    accuracy = (tp + tn) / max(total_samples, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_den = (precision + recall)
    f1 = (2 * precision * recall / f1_den) if f1_den > 0 else 0.0
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / (mcc_den ** 0.5) if mcc_den > 0 else 0.0
    if SKLEARN_AVAILABLE and len(set(all_labels)) > 1:
        try:
            auroc = roc_auc_score(all_labels, all_scores)
        except Exception:
            auroc = float('nan')
        try:
            auprc = average_precision_score(all_labels, all_scores)
        except Exception:
            auprc = float('nan')
    else:
        auroc = float('nan')
        auprc = float('nan')

    return {
        'file': os.path.basename(model_path),
        'model_name': model_name,
        'species': species,
        'seq_type': seq_type,
        'num_samples': total_samples,
        'accuracy': accuracy * 100.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc,
        'auprc': auprc,
        # For plotting/statistics
        '_y_true': all_labels,
        '_y_score': all_scores,
        '_y_pred': all_preds,
    }


def eval_real_train_test_synth(
    model_path: str,
    device: torch.device,
    synth_pos: List[str],
    ens_neg: List[str],
    test_size_per_class: int,
) -> Dict[str, Any]:
    payload = torch.load(model_path, map_location='cpu')
    model_name = payload.get('model_name')
    species = payload.get('species')
    seq_type = payload.get('seq_type')

    model = get_model_instance(model_name)
    state_dict = payload.get('model_state_dict') or payload.get('generator_state_dict') or payload.get('discriminator_state_dict')
    model.load_state_dict(state_dict, strict=False)

    max_class_n = min(len(synth_pos), len(ens_neg))
    n = min(max_class_n, test_size_per_class)

    rng = random.Random(42)
    pos_sample = list(synth_pos)
    neg_sample = list(ens_neg)
    rng.shuffle(pos_sample)
    rng.shuffle(neg_sample)
    pos_sample = pos_sample[:n]
    neg_sample = neg_sample[:n]

    test_sequences = pos_sample + neg_sample
    test_labels = [1] * len(pos_sample) + [0] * len(neg_sample)

    # Evaluate with probability outputs
    loader = create_loader_eval(test_sequences, test_labels, batch_size=64)
    import torch.nn.functional as F
    model.to(device)
    model.eval()
    all_scores: List[float] = []
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_scores.extend(probs.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    total_samples = len(all_labels)
    tp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(all_preds, all_labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(all_preds, all_labels) if p == 0 and y == 1)
    accuracy = (tp + tn) / max(total_samples, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_den = (precision + recall)
    f1 = (2 * precision * recall / f1_den) if f1_den > 0 else 0.0
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / (mcc_den ** 0.5) if mcc_den > 0 else 0.0
    if SKLEARN_AVAILABLE and len(set(all_labels)) > 1:
        try:
            auroc = roc_auc_score(all_labels, all_scores)
        except Exception:
            auroc = float('nan')
        try:
            auprc = average_precision_score(all_labels, all_scores)
        except Exception:
            auprc = float('nan')
    else:
        auroc = float('nan')
        auprc = float('nan')

    return {
        'file': os.path.basename(model_path),
        'model_name': model_name,
        'species': species,
        'seq_type': seq_type,
        'num_samples': total_samples,
        'accuracy': accuracy * 100.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'auroc': auroc,
        'auprc': auprc,
        '_y_true': all_labels,
        '_y_score': all_scores,
        '_y_pred': all_preds,
    }


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(os.path.join('src', 'final_results'), exist_ok=True)

    # Species/seq types in canonical order
    species_order = ["arabidopsis", "homo"]
    seq_type_order = ["acceptor", "donor"]
    lambda_values = ["0.0", "0.25", "0.5", "0.75"]  # skip 1.0 by request

    # 1) Build ENS test caches (to fix a common test size across scenarios)
    ens_cache: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}
    test_size_class: Dict[Tuple[str, str], int] = {}
    for sp in species_order:
        for st in seq_type_order:
            pos_ens, neg_ens = load_ens_mask_trimmed(sp, st)
            ens_cache[(sp, st)] = (pos_ens, neg_ens)
            test_size_class[(sp, st)] = min(len(pos_ens), len(neg_ens))
            print(f"ENS [{sp} {st}] pos={len(pos_ens)} neg={len(neg_ens)} -> test_n/class={test_size_class[(sp, st)]}")

    # 2) Baseline: Train-Real/Test-Real metrics on ENS (from saved real-trained models)
    baseline_results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    real_models = discover_saved_models_real()
    print(f"Found {len(real_models)} real-trained models.")
    for idx, mp in enumerate(real_models, start=1):
        try:
            payload = torch.load(mp, map_location='cpu')
        except Exception:
            continue
        sp = payload.get('species')
        st = payload.get('seq_type')
        if (sp, st) not in ens_cache:
            continue
        pos_ens, neg_ens = ens_cache[(sp, st)]
        n = test_size_class[(sp, st)]
        print(f"[Baseline {idx}/{len(real_models)}] Evaluating {os.path.basename(mp)} on ENS ({sp} {st})...")
        res = eval_on_ens(mp, device, pos_ens, neg_ens, test_size_per_class=n)
        key = (res['model_name'], res['species'], res['seq_type'])
        baseline_results[key] = res
        print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

    # 3) Scenario 2: Train-Synthetic/Test-Real (evaluate synthetic-trained models on ENS)
    s2_rows: List[Dict[str, Any]] = []
    synth_models = discover_saved_models_synthetic()
    print(f"Found {len(synth_models)} synthetic-trained models.")
    for idx, mp in enumerate(synth_models, start=1):
        try:
            payload = torch.load(mp, map_location='cpu')
        except Exception:
            continue
        sp = payload.get('species')
        st = payload.get('seq_type')
        lam = payload.get('lambda_val')
        # Skip lambda=1.0 synthetic-trained models by request
        if lam in (1.0, "1.0"):
            print(f"[S2 {idx}/{len(synth_models)}] Skipping {os.path.basename(mp)} (λ=1.0)")
            continue
        if (sp, st) not in ens_cache:
            continue
        pos_ens, neg_ens = ens_cache[(sp, st)]
        n = test_size_class[(sp, st)]
        print(f"[S2 {idx}/{len(synth_models)}] Evaluating {os.path.basename(mp)} on ENS ({sp} {st}) λ={lam}...")
        res = eval_on_ens(mp, device, pos_ens, neg_ens, test_size_per_class=n)
        res.update({'scenario': 'train_synth_test_real', 'lambda': lam, 'generator': 'diffusion'})
        # Compute delta vs baseline
        base = baseline_results.get((res['model_name'], res['species'], res['seq_type']))
        if base:
            res['delta_mcc'] = res['mcc'] - base['mcc']
            res['delta_f1'] = res['f1'] - base['f1']
            res['delta_acc'] = res['accuracy'] - base['accuracy']
        s2_rows.append(res)
        print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

    # Also evaluate GAN/VAE baseline-trained proxies (from src/baseline_saved_models) on ENS
    baseline_models = discover_saved_models_baseline()
    print(f"Found {len(baseline_models)} GAN/VAE baseline-trained models.")
    for idx, mp in enumerate(baseline_models, start=1):
        try:
            payload = torch.load(mp, map_location='cpu')
        except Exception:
            continue
        sp = payload.get('species')
        st = payload.get('seq_type')
        if (sp, st) not in ens_cache:
            continue
        pos_ens, neg_ens = ens_cache[(sp, st)]
        n = test_size_class[(sp, st)]
        # Generator and lambda from metadata/filename
        generator = (payload.get('baseline') or 'unknown').lower()
        lam = payload.get('lambda')
        # Some files encode lambda in name (e.g., lambda0.5); infer if missing
        if lam is None:
            fname = os.path.basename(mp)
            if 'lambda0.5' in fname or 'lambda_0.5' in fname:
                lam = '0.5'
            elif 'lambda0.25' in fname or 'lambda_0.25' in fname:
                lam = '0.25'
            elif 'lambda0.75' in fname or 'lambda_0.75' in fname:
                lam = '0.75'
            elif 'lambda1.0' in fname or 'lambda_1.0' in fname:
                lam = '1.0'
            else:
                lam = '0.0'
        if lam in (1.0, '1.0'):
            print(f"[S2-BL {idx}/{len(baseline_models)}] Skipping {os.path.basename(mp)} (λ=1.0)")
            continue
        print(f"[S2-BL {idx}/{len(baseline_models)}] Evaluating {os.path.basename(mp)} on ENS ({sp} {st}) gen={generator} λ={lam}...")
        res = eval_on_ens(mp, device, pos_ens, neg_ens, test_size_per_class=n)
        res.update({'scenario': 'train_synth_test_real', 'lambda': lam, 'generator': generator})
        base = baseline_results.get((res['model_name'], res['species'], res['seq_type']))
        if base:
            res['delta_mcc'] = res['mcc'] - base['mcc']
            res['delta_f1'] = res['f1'] - base['f1']
            res['delta_acc'] = res['accuracy'] - base['accuracy']
        s2_rows.append(res)
        print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

    # 4) Scenario 1: Train-Real/Test-Synthetic (evaluate real-trained models on synthetic test)
    s1_rows: List[Dict[str, Any]] = []
    print("Evaluating Train-Real/Test-Synthetic across λ values (Diffusion, GAN, VAE)...")
    for r_idx, mp in enumerate(real_models, start=1):
        try:
            payload = torch.load(mp, map_location='cpu')
        except Exception:
            continue
        sp = payload.get('species')
        st = payload.get('seq_type')
        if (sp, st) not in ens_cache:
            continue
        _, neg_ens = ens_cache[(sp, st)]
        n = test_size_class[(sp, st)]
        # Diffusion synthetic sets
        for lam in lambda_values:
            synth_file = os.path.join('src', 'Lambda_sensitivity_analysis', f"{sp}_{st}_lambda_{lam}_sequences.txt")
            if not os.path.exists(synth_file):
                continue
            synth_pos = load_synthetic_sequences(synth_file)
            print(f"[S1 {r_idx}/{len(real_models)}] {os.path.basename(mp)} -> ({sp} {st}) λ={lam} | synth={len(synth_pos)}")
            res = eval_real_train_test_synth(mp, device, synth_pos, neg_ens, test_size_per_class=n)
            res.update({'scenario': 'train_real_test_synth', 'lambda': lam, 'generator': 'diffusion'})
            base = baseline_results.get((res['model_name'], res['species'], res['seq_type']))
            if base:
                res['delta_mcc'] = res['mcc'] - base['mcc']
                res['delta_f1'] = res['f1'] - base['f1']
                res['delta_acc'] = res['accuracy'] - base['accuracy']
            s1_rows.append(res)
            print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

        # GAN synthetic sets
        gan_txts = [p for p in discover_gan_generated_files() if os.path.basename(p).startswith(f"{sp}_{st}_")]
        for gpath in gan_txts:
            lam = infer_lambda_from_filename(gpath)
            if lam in ('1.0', 1.0):
                continue
            synth_pos = load_synthetic_sequences(gpath)
            print(f"[S1 {r_idx}/{len(real_models)}] {os.path.basename(mp)} -> ({sp} {st}) GAN λ={lam} | synth={len(synth_pos)}")
            res = eval_real_train_test_synth(mp, device, synth_pos, neg_ens, test_size_per_class=n)
            res.update({'scenario': 'train_real_test_synth', 'lambda': lam, 'generator': 'gan'})
            base = baseline_results.get((res['model_name'], res['species'], res['seq_type']))
            if base:
                res['delta_mcc'] = res['mcc'] - base['mcc']
                res['delta_f1'] = res['f1'] - base['f1']
                res['delta_acc'] = res['accuracy'] - base['accuracy']
            s1_rows.append(res)
            print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

        # VAE synthetic sets
        vae_txts = [p for p in discover_vae_generated_files() if os.path.basename(p).startswith(f"{sp}_{st}_")]
        for vpath in vae_txts:
            lam = infer_lambda_from_filename(vpath)
            if lam in ('1.0', 1.0):
                continue
            synth_pos = load_synthetic_sequences(vpath)
            print(f"[S1 {r_idx}/{len(real_models)}] {os.path.basename(mp)} -> ({sp} {st}) VAE λ={lam} | synth={len(synth_pos)}")
            res = eval_real_train_test_synth(mp, device, synth_pos, neg_ens, test_size_per_class=n)
            res.update({'scenario': 'train_real_test_synth', 'lambda': lam, 'generator': 'vae'})
            base = baseline_results.get((res['model_name'], res['species'], res['seq_type']))
            if base:
                res['delta_mcc'] = res['mcc'] - base['mcc']
                res['delta_f1'] = res['f1'] - base['f1']
                res['delta_acc'] = res['accuracy'] - base['accuracy']
            s1_rows.append(res)
            print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

    # 5) Scenario 3: Data Augmentation (use saved augmented models, evaluate on ENS for consistency)
    s3_rows: List[Dict[str, Any]] = []
    aug_models = discover_saved_models_augmented()
    print(f"Found {len(aug_models)} augmentation models.")
    for idx, mp in enumerate(aug_models, start=1):
        try:
            payload = torch.load(mp, map_location='cpu')
        except Exception:
            continue
        model_name = payload.get('model_name')
        sp = payload.get('species')
        st = payload.get('seq_type')
        real_fraction = payload.get('real_fraction')
        if (sp, st) not in ens_cache:
            continue
        pos_ens, neg_ens = ens_cache[(sp, st)]
        n = test_size_class[(sp, st)]
        # Evaluate
        print(f"[S3 {idx}/{len(aug_models)}] Evaluating {os.path.basename(mp)} on ENS ({sp} {st}) rf={real_fraction}...")
        # Identify generator type for augmentation; default to diffusion if not provided
        generator = payload.get('generator', 'diffusion')
        res = eval_on_ens(mp, device, pos_ens, neg_ens, test_size_per_class=n)
        res.update({'scenario': 'augmentation_test_real', 'lambda': 0.5, 'real_fraction': real_fraction, 'generator': generator})
        base = baseline_results.get((model_name, sp, st))
        if base:
            res['delta_mcc'] = res['mcc'] - base['mcc']
            res['delta_f1'] = res['f1'] - base['f1']
            res['delta_acc'] = res['accuracy'] - base['accuracy']
        s3_rows.append(res)
        print(f"  -> done: acc={res['accuracy']:.2f}%, f1={res['f1']:.4f}, mcc={res['mcc']:.4f}, auroc={res['auroc']:.4f}, auprc={res['auprc']:.4f}")

    # 6) Add baseline rows to each scenario table for comparison
    baseline_rows_for_tables: List[Dict[str, Any]] = []
    for key, base in baseline_results.items():
        brow = dict(base)
        brow.update({'scenario': 'baseline_trainReal_testReal', 'lambda': '-', 'real_fraction': None, 'generator': 'baseline'})
        baseline_rows_for_tables.append(brow)

    s1_rows_with_baseline = baseline_rows_for_tables + s1_rows
    s2_rows_with_baseline = baseline_rows_for_tables + s2_rows
    s3_rows_with_baseline = baseline_rows_for_tables + s3_rows

    # 7) Save per-scenario CSVs and a combined summary
    out_dir = os.path.join('src', 'final_results')
    os.makedirs(out_dir, exist_ok=True)

    def write_csv(path: str, rows: List[Dict[str, Any]], extra_fields: List[str]) -> None:
        base_fields = ['file', 'model_name', 'species', 'seq_type', 'generator', 'num_samples', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'auroc', 'auprc']
        fieldnames = base_fields + extra_fields
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})

    write_csv(os.path.join(out_dir, 'scenario1_trainReal_testSynth.csv'), s1_rows_with_baseline, ['scenario', 'lambda', 'delta_mcc', 'delta_f1', 'delta_acc'])
    write_csv(os.path.join(out_dir, 'scenario2_trainSynth_testReal.csv'), s2_rows_with_baseline, ['scenario', 'lambda', 'delta_mcc', 'delta_f1', 'delta_acc'])
    write_csv(os.path.join(out_dir, 'scenario3_augmentation_testReal.csv'), s3_rows_with_baseline, ['scenario', 'lambda', 'real_fraction', 'delta_mcc', 'delta_f1', 'delta_acc'])

    combined = s1_rows_with_baseline + s2_rows_with_baseline + s3_rows_with_baseline
    write_csv(os.path.join(out_dir, 'final_eval_summary.csv'), combined, ['scenario', 'lambda', 'real_fraction', 'delta_mcc', 'delta_f1', 'delta_acc'])

    # 7) Figures: Bar charts (MCC, AUPRC) grouped per species/seq_type and scenario
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    def plot_bars(rows: List[Dict[str, Any]], title: str, x_labels: List[str], values: List[float], ylabel: str, path: str) -> None:
        plt.figure(figsize=(8, 4))
        plt.bar(x_labels, values, color='#4C78A8')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    # Group results per species/seq_type
    by_sp_st: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in s1_rows:
        by_sp_st[(r['species'], r['seq_type'])]['s1'].append(r)
    for r in s2_rows:
        by_sp_st[(r['species'], r['seq_type'])]['s2'].append(r)
    for r in s3_rows:
        by_sp_st[(r['species'], r['seq_type'])]['s3'].append(r)
    for key, base in baseline_results.items():
        sp, st = key[1], key[2]
        by_sp_st[(sp, st)]['baseline'] = [base]

    for (sp, st), groups in by_sp_st.items():
        # Scenario 1 bars (by lambda)
        if 's1' in groups:
            rows = sorted(groups['s1'], key=lambda r: float(r.get('lambda') if r.get('lambda') is not None else -1))
            x = ['baseline'] + [str(r.get('lambda')) for r in rows]
            mcc_vals = [baseline_results.get((rows[0]['model_name'], sp, st), {}).get('mcc', float('nan'))] if 'baseline' in groups else []
            auprc_vals = [baseline_results.get((rows[0]['model_name'], sp, st), {}).get('auprc', float('nan'))] if 'baseline' in groups else []
            # Choose the first model_name present for baseline lookup; different models produce separate files anyway
            mcc_vals += [r['mcc'] for r in rows]
            auprc_vals += [r['auprc'] for r in rows]
            plot_bars(rows, f"S1: Train-Real/Test-Synth MCC [{sp} {st}]", x, mcc_vals, 'MCC', os.path.join(fig_dir, f"S1_MCC_{sp}_{st}.png"))
            plot_bars(rows, f"S1: Train-Real/Test-Synth AUPRC [{sp} {st}]", x, auprc_vals, 'AUPRC', os.path.join(fig_dir, f"S1_AUPRC_{sp}_{st}.png"))

        # Scenario 2 bars (by lambda)
        if 's2' in groups:
            rows = sorted(groups['s2'], key=lambda r: float(r.get('lambda') if r.get('lambda') is not None else -1))
            x = ['baseline'] + [str(r.get('lambda')) for r in rows]
            mcc_vals = [baseline_results.get((rows[0]['model_name'], sp, st), {}).get('mcc', float('nan'))] if 'baseline' in groups else []
            auprc_vals = [baseline_results.get((rows[0]['model_name'], sp, st), {}).get('auprc', float('nan'))] if 'baseline' in groups else []
            mcc_vals += [r['mcc'] for r in rows]
            auprc_vals += [r['auprc'] for r in rows]
            plot_bars(rows, f"S2: Train-Synth/Test-Real MCC [{sp} {st}]", x, mcc_vals, 'MCC', os.path.join(fig_dir, f"S2_MCC_{sp}_{st}.png"))
            plot_bars(rows, f"S2: Train-Synth/Test-Real AUPRC [{sp} {st}]", x, auprc_vals, 'AUPRC', os.path.join(fig_dir, f"S2_AUPRC_{sp}_{st}.png"))

        # Scenario 3 bars (by augmentation fraction)
        if 's3' in groups:
            rows = sorted(groups['s3'], key=lambda r: float(r.get('real_fraction') if r.get('real_fraction') is not None else -1))
            x = ['baseline'] + [str(r.get('real_fraction')) for r in rows]
            # baseline metrics
            # Select baseline by matching model of first row
            mcc_vals = [baseline_results.get((rows[0]['model_name'], sp, st), {}).get('mcc', float('nan'))] if 'baseline' in groups else []
            auprc_vals = [baseline_results.get((rows[0]['model_name'], sp, st), {}).get('auprc', float('nan'))] if 'baseline' in groups else []
            mcc_vals += [r['mcc'] for r in rows]
            auprc_vals += [r['auprc'] for r in rows]
            plot_bars(rows, f"S3: Augmentation/Test-Real MCC [{sp} {st}]", x, mcc_vals, 'MCC', os.path.join(fig_dir, f"S3_MCC_{sp}_{st}.png"))
            plot_bars(rows, f"S3: Augmentation/Test-Real AUPRC [{sp} {st}]", x, auprc_vals, 'AUPRC', os.path.join(fig_dir, f"S3_AUPRC_{sp}_{st}.png"))

    # 8) PR and ROC curves (baseline vs scenarios) — per species/seq_type
    if SKLEARN_AVAILABLE:
        def plot_pr_roc(curves: Dict[str, Tuple[List[int], List[float]]], title_prefix: str, out_prefix: str) -> None:
            # PR
            plt.figure(figsize=(6, 5))
            for label, (y_true, y_score) in curves.items():
                try:
                    precision, recall, _ = precision_recall_curve(y_true, y_score)
                except Exception:
                    continue
                plt.plot(recall, precision, label=label)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{title_prefix} - PR')
            plt.legend(loc='lower left', fontsize=8)
            plt.tight_layout()
            plt.savefig(out_prefix + '_PR.png', dpi=200)
            plt.close()
            # ROC
            plt.figure(figsize=(6, 5))
            for label, (y_true, y_score) in curves.items():
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                except Exception:
                    continue
                plt.plot(fpr, tpr, label=label)
            plt.plot([0,1], [0,1], 'k--', linewidth=0.8)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title(f'{title_prefix} - ROC')
            plt.legend(loc='lower right', fontsize=8)
            plt.tight_layout()
            plt.savefig(out_prefix + '_ROC.png', dpi=200)
            plt.close()

        for (sp, st), groups in by_sp_st.items():
            # Build curves dicts for scenario 2 (Train-Synth/Test-Real) w.r.t. baseline
            curves_s2: Dict[str, Tuple[List[int], List[float]]] = {}
            base_key = None
            if 'baseline' in groups and groups['baseline']:
                b = groups['baseline'][0]
                curves_s2['baseline'] = (b.get('_y_true', []), b.get('_y_score', []))
                base_key = (b['model_name'], sp, st)
            if 's2' in groups:
                for r in sorted(groups['s2'], key=lambda r: float(r.get('lambda') if r.get('lambda') is not None else -1)):
                    label = f"lam={r.get('lambda')}"
                    curves_s2[label] = (r.get('_y_true', []), r.get('_y_score', []))
            if curves_s2:
                out_prefix = os.path.join(fig_dir, f"S2_curves_{sp}_{st}")
                plot_pr_roc(curves_s2, f"S2 {sp} {st}", out_prefix)

            # Scenario 3 curves (Augmentation/Test-Real)
            curves_s3: Dict[str, Tuple[List[int], List[float]]] = {}
            if 'baseline' in groups and groups['baseline']:
                b = groups['baseline'][0]
                curves_s3['baseline'] = (b.get('_y_true', []), b.get('_y_score', []))
            if 's3' in groups:
                for r in sorted(groups['s3'], key=lambda r: float(r.get('real_fraction') if r.get('real_fraction') is not None else -1)):
                    label = f"rf={r.get('real_fraction')}"
                    curves_s3[label] = (r.get('_y_true', []), r.get('_y_score', []))
            if curves_s3:
                out_prefix = os.path.join(fig_dir, f"S3_curves_{sp}_{st}")
                plot_pr_roc(curves_s3, f"S3 {sp} {st}", out_prefix)

    # 9) Basic McNemar statistics vs baseline (report b/c and chi2 with continuity correction)
    stats_path = os.path.join(out_dir, 'mcnemar_stats.json')
    mcnemar_stats: Dict[str, Any] = {}
    def mcnemar_bc(baseline_pred: List[int], other_pred: List[int], y_true: List[int]) -> Dict[str, float]:
        # Count b: baseline correct, other wrong; c: baseline wrong, other correct
        b = 0
        c = 0
        for bp, op, yt in zip(baseline_pred, other_pred, y_true):
            b_correct = (bp == yt)
            o_correct = (op == yt)
            if b_correct and not o_correct:
                b += 1
            elif (not b_correct) and o_correct:
                c += 1
        n = b + c
        if n == 0:
            return {'b': float(b), 'c': float(c), 'chi2_cc': 0.0}
        chi2_cc = ((abs(b - c) - 1) ** 2) / n
        return {'b': float(b), 'c': float(c), 'chi2_cc': float(chi2_cc)}

    for (sp, st), groups in by_sp_st.items():
        if 'baseline' not in groups or not groups['baseline']:
            continue
        base = groups['baseline'][0]
        yb = base.get('_y_pred', [])
        yt = base.get('_y_true', [])
        # Scenario 2
        if 's2' in groups:
            for r in groups['s2']:
                key = f"S2__{r['model_name']}__{sp}__{st}__lam_{r.get('lambda')}"
                stats = mcnemar_bc(yb, r.get('_y_pred', []), yt)
                mcnemar_stats[key] = stats
        # Scenario 3
        if 's3' in groups:
            for r in groups['s3']:
                key = f"S3__{r['model_name']}__{sp}__{st}__rf_{r.get('real_fraction')}"
                stats = mcnemar_bc(yb, r.get('_y_pred', []), yt)
                mcnemar_stats[key] = stats

    with open(stats_path, 'w') as f:
        json.dump(mcnemar_stats, f, indent=2)
    print(f"Saved figures to {fig_dir} and McNemar stats to {stats_path}")

    print(f"Saved scenario results and summary under {out_dir}")


if __name__ == "__main__":
    main()


