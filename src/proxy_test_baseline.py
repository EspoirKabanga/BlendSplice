#!/usr/bin/env python3

import os
import glob
import csv
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import classifier architectures
from models import SpliceRover, SpliceFinder, DeepSplicer, IntSplice, Spliceator


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(42)


class DNASequenceDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def one_hot_encode(sequences: List[str]) -> np.ndarray:
    base_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    encoded = np.zeros((len(sequences), len(sequences[0]), 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            encoded[i, j] = base_dict.get(base, [0, 0, 0, 0])
    return encoded


def trim_center_402(seq: str) -> str:
    s = seq.strip().upper()
    if len(s) < 402:
        return ""
    if len(s) >= 602:
        return s[100:len(s) - 100]
    pad = (len(s) - 402) // 2
    start = max(0, pad)
    end = start + 402
    return s[start:end] if len(s) >= 402 else ""


def load_ens_mask_trimmed(species: str, seq_type: str) -> Tuple[List[str], List[str]]:
    sp_dir = "Arab" if species == "arabidopsis" else "Homo"
    type_cap = "Donor" if seq_type == "donor" else "Acceptor"
    base = "/home/ekabanga/All_DataSet/Splice/ENSdata_for_mask"
    pos_path = os.path.join(base, sp_dir, f"{type_cap}_positive.txt")
    neg_path = os.path.join(base, sp_dir, f"{type_cap}_negative.txt")

    def read_and_trim(path: str) -> List[str]:
        out = []
        with open(path, 'r') as f:
            for line in f:
                seq = line.strip().upper()
                if not seq:
                    continue
                t = trim_center_402(seq)
                if len(t) == 402:
                    out.append(t)
        return out

    pos = read_and_trim(pos_path)
    neg = read_and_trim(neg_path)
    return pos, neg


def create_loader(seqs: List[str], labels: List[int], batch_size: int) -> DataLoader:
    encoded = one_hot_encode(seqs)
    dataset = DNASequenceDataset(torch.FloatTensor(encoded), torch.LongTensor(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    tp = tn = fp = fn = 0
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = (tp + tn) / max(total_samples, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_den = (precision + recall)
    f1 = (2 * precision * recall / f1_den) if f1_den > 0 else 0.0
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / (mcc_den ** 0.5) if mcc_den > 0 else 0.0
    return {
        "loss": avg_loss,
        "num_samples": total_samples,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def get_model_instance(model_name: str) -> nn.Module:
    model_classes = {
        "SpliceRover": SpliceRover,
        "SpliceFinder": SpliceFinder,
        "DeepSplicer": DeepSplicer,
        "IntSplice": IntSplice,
        "Spliceator": Spliceator,
    }
    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")
    return model_classes[model_name]()


def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    baseline_dir = os.path.join('src', 'baseline_saved_models')
    os.makedirs(baseline_dir, exist_ok=True)
    results_out = os.path.join(baseline_dir, 'baseline_eval_results.csv')

    saved_model_paths = glob.glob(os.path.join(baseline_dir, '*.pth'))
    print(f"Found {len(saved_model_paths)} baseline models.")

    # Build index: (model_name, species, seq_type, baseline)
    index: Dict[Tuple[str, str, str, str], str] = {}
    model_names_set = set()
    for model_path in sorted(saved_model_paths):
        try:
            payload = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(f"Skipping {os.path.basename(model_path)}: load error: {e}")
            continue
        model_name = payload.get('model_name')
        species = payload.get('species')
        seq_type = payload.get('seq_type')
        baseline = payload.get('baseline')  # 'gan' or 'vae'
        if model_name is None or species is None or seq_type is None or baseline is None:
            print(f"Skipping {os.path.basename(model_path)}: missing metadata")
            continue
        index[(model_name, species, seq_type, str(baseline))] = model_path
        model_names_set.add(model_name)

    species_order = ["arabidopsis", "homo"]
    seq_type_order = ["acceptor", "donor"]
    baseline_order = ["gan", "vae"]

    results = []
    ens_cache: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}

    for model_name in sorted(model_names_set):
        for species in species_order:
            for seq_type in seq_type_order:
                # Load/print ENS test set once per species/type
                key = (species, seq_type)
                if key not in ens_cache:
                    pos_test, neg_test = load_ens_mask_trimmed(species, seq_type)
                    ens_cache[key] = (pos_test, neg_test)
                    ex_len = len(pos_test[0]) if len(pos_test) > 0 else (len(neg_test[0]) if len(neg_test) > 0 else 0)
                    print(
                        f"ENS test ({species} {seq_type}) -> pos: {len(pos_test)}, neg: {len(neg_test)}, total: {len(pos_test)+len(neg_test)}, example_len: {ex_len}"
                    )
                else:
                    pos_test, neg_test = ens_cache[key]

                for baseline in baseline_order:
                    idx_key = (model_name, species, seq_type, baseline)
                    model_path = index.get(idx_key)
                    if not model_path:
                        continue

                    # Instantiate and load model
                    try:
                        payload = torch.load(model_path, map_location='cpu')
                        model = get_model_instance(model_name)
                    except Exception as e:
                        print(f"Skipping {os.path.basename(model_path)}: cannot instantiate/load: {e}")
                        continue

                    state_dict = payload.get('model_state_dict')
                    if state_dict is None:
                        print(f"Skipping {os.path.basename(model_path)}: no model_state_dict found")
                        continue
                    try:
                        model.load_state_dict(state_dict, strict=False)
                    except Exception as e:
                        print(f"Skipping {os.path.basename(model_path)}: load_state_dict error: {e}")
                        continue

                    test_sequences = pos_test + neg_test
                    test_labels = [1] * len(pos_test) + [0] * len(neg_test)
                    if len(test_sequences) == 0:
                        print(f"Skipping {os.path.basename(model_path)}: empty test set")
                        continue

                    test_loader = create_loader(test_sequences, test_labels, batch_size=64)
                    metrics = evaluate_model(model, test_loader, device)

                    result_row = {
                        'file': os.path.basename(model_path),
                        'model_name': model_name,
                        'baseline': baseline,
                        'species': species,
                        'seq_type': seq_type,
                        'num_samples': metrics['num_samples'],
                        'accuracy': metrics['accuracy'] * 100.0,
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'mcc': metrics['mcc'],
                        'loss': metrics['loss'],
                        'tp': metrics['tp'],
                        'tn': metrics['tn'],
                        'fp': metrics['fp'],
                        'fn': metrics['fn'],
                    }
                    results.append(result_row)

                    print(
                        f"Evaluated {result_row['file']} [{model_name} | {species} {seq_type} | {baseline}] acc={result_row['accuracy']:.2f}%, f1={result_row['f1']:.4f}, mcc={result_row['mcc']:.4f}, n={result_row['num_samples']}"
                    )

    # Save results in the printed order
    fieldnames = [
        'file', 'model_name', 'baseline', 'species', 'seq_type',
        'num_samples', 'accuracy', 'precision', 'recall', 'f1', 'mcc', 'loss', 'tp', 'tn', 'fp', 'fn'
    ]
    with open(results_out, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved baseline evaluation results to {results_out}")


if __name__ == "__main__":
    main()


