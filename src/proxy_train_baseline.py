import os
import glob
import random
from typing import List, Tuple

import torch
import numpy as np

# Reuse training utilities and model loader from proxy.py
from proxy import (
    set_seeds,
    one_hot_encode,
    DNASequenceDataset,
    create_data_loaders,
    train_model,
    get_model_instance,
)


def load_sequences_simple(file_path: str) -> List[str]:
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            seq = line.strip().upper()
            if not seq:
                continue
            sequences.append(seq)
    print(f"Loaded {len(sequences)} sequences from {os.path.basename(file_path)}")
    return sequences


def enforce_splice_motif(seq: str, seq_type: str) -> str:
    # Enforce motif at 0-based positions [200:202] (1-based 201-202)
    if len(seq) < 202:
        return seq
    motif = "GT" if seq_type == "donor" else "AG"
    return seq[:200] + motif + seq[202:]


def enforce_motif_for_gan_sequences(sequences: List[str], seq_type: str) -> List[str]:
    return [enforce_splice_motif(seq, seq_type) for seq in sequences]


def prepare_baseline_synthetic_data(
    species: str,
    seq_type: str,
    pos_sequences: List[str],
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """Prepare training (synthetic positives + equal negatives) and validation (real, balanced) data.
    Mirrors proxy.py's prepare_synthetic_data real-validation logic.
    """
    # Load negatives and real positives (for validation pool)
    neg_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_negative.txt"
    pos_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"

    # Note: do NOT filter real positives by motif here; proxy.py's prepare_synthetic_data uses filtering for pos
    # via load_sequences(pos_file, seq_type). We mirror that strictly by using the same ordering/shuffle behavior:
    from proxy import load_sequences as load_sequences_with_optional_filter
    real_pos_sequences = load_sequences_with_optional_filter(pos_file, seq_type)

    # Negatives (no motif filtering)
    real_neg_sequences = load_sequences_simple(neg_file)

    # Shuffle with seed 42 (order: positives to train, negatives, real positives for validation)
    random.seed(42)
    random.shuffle(pos_sequences)
    random.shuffle(real_neg_sequences)
    random.shuffle(real_pos_sequences)

    # Training data: synthetic positives + equal number of negatives
    pos_train = pos_sequences
    neg_train = real_neg_sequences[:len(pos_train)]

    # Validation data: use REAL sequences, skipping the 50k/100k used for generative model training
    train_size_k = (50 if species == "arabidopsis" else 100) * 1000
    remaining_real_pos = real_pos_sequences[train_size_k:]
    remaining_real_neg = real_neg_sequences[len(pos_train):]

    # Balanced validation: half of remaining min class
    max_val_size = min(len(remaining_real_pos), len(remaining_real_neg)) // 2
    pos_val = remaining_real_pos[:max_val_size]
    neg_val = remaining_real_neg[:max_val_size]

    # Combine and shuffle
    train_sequences = pos_train + neg_train
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    val_sequences = pos_val + neg_val
    val_labels = [1] * len(pos_val) + [0] * len(neg_val)

    train_data = list(zip(train_sequences, train_labels))
    val_data = list(zip(val_sequences, val_labels))
    random.shuffle(train_data)
    random.shuffle(val_data)

    train_sequences, train_labels = zip(*train_data) if train_data else ([], [])
    val_sequences, val_labels = zip(*val_data) if val_data else ([], [])

    print(
        f"Training set: {len(train_sequences)} sequences ("
        f"{len(pos_train)} synthetic pos, {len(neg_train)} neg)"
    )
    print(
        f"Validation set: {len(val_sequences)} sequences ("
        f"{len(pos_val)} real pos, {len(neg_val)} neg) - BALANCED"
    )

    return (list(train_sequences), list(train_labels)), (list(val_sequences), list(val_labels))


def discover_gan_generated_files() -> List[str]:
    base_dir = "src/GAN_generated_sequences"
    if not os.path.isdir(base_dir):
        return []
    return sorted(glob.glob(os.path.join(base_dir, "*.txt")))


def discover_vae_generated_files() -> List[str]:
    base_dir = "src/VAE_generated_sequences"
    if not os.path.isdir(base_dir):
        return []
    files = sorted(glob.glob(os.path.join(base_dir, "*.txt")))
    # Exclude any files containing lambda_
    return [f for f in files if "lambda_" not in os.path.basename(f)]


def discover_gan_lambda05_files() -> List[str]:
    base_dir = "src/GAN_generated_sequences"
    if not os.path.isdir(base_dir):
        return []
    return sorted([f for f in glob.glob(os.path.join(base_dir, "*.txt")) if "lambda_0.5" in os.path.basename(f)])


def discover_vae_lambda05_files() -> List[str]:
    base_dir = "src/VAE_generated_sequences"
    if not os.path.isdir(base_dir):
        return []
    return sorted([f for f in glob.glob(os.path.join(base_dir, "*.txt")) if "lambda_0.5" in os.path.basename(f)])


def ask_user_permission(message: str) -> bool:
    try:
        ans = input(f"{message} (Y/n): ").strip().lower()
    except EOFError:
        ans = "y"
    return ans in ("", "y", "yes")


def parse_species_seqtype_from_filename(path: str) -> Tuple[str, str]:
    name = os.path.basename(path)
    # Expected patterns like: arabidopsis_donor_train_50k_generated_sequences.txt
    # or homo_acceptor_train_100k_generated_sequences.txt
    parts = name.split("_")
    species = parts[0]
    seq_type = parts[1]
    return species, seq_type


def main():
    set_seeds(42)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("src/baseline_saved_models", exist_ok=True)

    model_names = ["SpliceRover", "SpliceFinder", "DeepSplicer", "IntSplice", "Spliceator"]

    # 1) Train on GAN-generated sequences (with enforced motif)
    gan_files = discover_gan_generated_files()
    print(f"\nDiscovered {len(gan_files)} GAN-generated sequence files (non-blended)")
    if gan_files and ask_user_permission("Proceed with training on GAN non-blended files?"):
        for file_path in gan_files:
            try:
                species, seq_type = parse_species_seqtype_from_filename(file_path)
                print(f"\n=== GAN: {species} {seq_type} (non-blended) ===")
                sequences = load_sequences_simple(file_path)
                sequences = enforce_motif_for_gan_sequences(sequences, seq_type)

                # Prepare data (synthetic pos + real neg; real validation as in proxy.py)
                train_data, val_data = prepare_baseline_synthetic_data(species, seq_type, sequences)

                # Create loaders
                train_loader, val_loader = create_data_loaders(train_data, val_data)

                # Train each model
                for model_name in model_names:
                    print(f"\nTraining {model_name} on GAN {species} {seq_type} (non-blended)")
                    model = get_model_instance(model_name)
                    trained_model = train_model(model, train_loader, val_loader, device)

                    save_name = f"{model_name}_baseline_gan_{species}_{seq_type}.pth"
                    save_path = os.path.join("src/baseline_saved_models", save_name)
                    torch.save({
                        'model_state_dict': trained_model.state_dict(),
                        'model_name': model_name,
                        'baseline': 'gan',
                        'species': species,
                        'seq_type': seq_type,
                        'lambda': 0.0,
                        'train_size': len(train_data[0]),
                        'val_size': len(val_data[0]),
                    }, save_path)
                    print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error training on GAN file {file_path}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Skipped training on GAN non-blended files.")

    # 1b) Train on GAN lambda=0.5 sequences
    gan_lambda_files = discover_gan_lambda05_files()
    print(f"\nDiscovered {len(gan_lambda_files)} GAN-generated sequence files (lambda=0.5)")
    if gan_lambda_files and ask_user_permission("Proceed with training on GAN lambda=0.5 files?"):
        for file_path in gan_lambda_files:
            try:
                species, seq_type = parse_species_seqtype_from_filename(file_path)
                print(f"\n=== GAN: {species} {seq_type} (lambda=0.5) ===")
                sequences = load_sequences_simple(file_path)
                sequences = enforce_motif_for_gan_sequences(sequences, seq_type)

                train_data, val_data = prepare_baseline_synthetic_data(species, seq_type, sequences)
                train_loader, val_loader = create_data_loaders(train_data, val_data)

                for model_name in model_names:
                    print(f"\nTraining {model_name} on GAN {species} {seq_type} (lambda=0.5)")
                    model = get_model_instance(model_name)
                    trained_model = train_model(model, train_loader, val_loader, device)

                    save_name = f"{model_name}_baseline_gan_lambda0.5_{species}_{seq_type}.pth"
                    save_path = os.path.join("src/baseline_saved_models", save_name)
                    torch.save({
                        'model_state_dict': trained_model.state_dict(),
                        'model_name': model_name,
                        'baseline': 'gan',
                        'species': species,
                        'seq_type': seq_type,
                        'lambda': 0.5,
                        'train_size': len(train_data[0]),
                        'val_size': len(val_data[0]),
                    }, save_path)
                    print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error training on GAN lambda file {file_path}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Skipped training on GAN lambda=0.5 files.")

    # 2) Train on VAE-generated sequences (non-blended; no motif enforcement)
    vae_files = discover_vae_generated_files()
    print(f"\nDiscovered {len(vae_files)} VAE-generated sequence files (non-blended)")
    if vae_files and ask_user_permission("Proceed with training on VAE non-blended files?"):
        for file_path in vae_files:
            try:
                species, seq_type = parse_species_seqtype_from_filename(file_path)
                print(f"\n=== VAE: {species} {seq_type} (non-blended) ===")
                sequences = load_sequences_simple(file_path)

                # Prepare data (synthetic pos + real neg; real validation as in proxy.py)
                train_data, val_data = prepare_baseline_synthetic_data(species, seq_type, sequences)

                # Create loaders
                train_loader, val_loader = create_data_loaders(train_data, val_data)

                # Train each model
                for model_name in model_names:
                    print(f"\nTraining {model_name} on VAE {species} {seq_type} (non-blended)")
                    model = get_model_instance(model_name)
                    trained_model = train_model(model, train_loader, val_loader, device)

                    save_name = f"{model_name}_baseline_vae_{species}_{seq_type}.pth"
                    save_path = os.path.join("src/baseline_saved_models", save_name)
                    torch.save({
                        'model_state_dict': trained_model.state_dict(),
                        'model_name': model_name,
                        'baseline': 'vae',
                        'species': species,
                        'seq_type': seq_type,
                        'lambda': 0.0,
                        'train_size': len(train_data[0]),
                        'val_size': len(val_data[0]),
                    }, save_path)
                    print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error training on VAE file {file_path}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Skipped training on VAE non-blended files.")

    # 2b) Train on VAE lambda=0.5 sequences (no motif enforcement)
    vae_lambda_files = discover_vae_lambda05_files()
    print(f"\nDiscovered {len(vae_lambda_files)} VAE-generated sequence files (lambda=0.5)")
    if vae_lambda_files and ask_user_permission("Proceed with training on VAE lambda=0.5 files?"):
        for file_path in vae_lambda_files:
            try:
                species, seq_type = parse_species_seqtype_from_filename(file_path)
                print(f"\n=== VAE: {species} {seq_type} (lambda=0.5) ===")
                sequences = load_sequences_simple(file_path)

                train_data, val_data = prepare_baseline_synthetic_data(species, seq_type, sequences)
                train_loader, val_loader = create_data_loaders(train_data, val_data)

                for model_name in model_names:
                    print(f"\nTraining {model_name} on VAE {species} {seq_type} (lambda=0.5)")
                    model = get_model_instance(model_name)
                    trained_model = train_model(model, train_loader, val_loader, device)

                    save_name = f"{model_name}_baseline_vae_lambda0.5_{species}_{seq_type}.pth"
                    save_path = os.path.join("src/baseline_saved_models", save_name)
                    torch.save({
                        'model_state_dict': trained_model.state_dict(),
                        'model_name': model_name,
                        'baseline': 'vae',
                        'species': species,
                        'seq_type': seq_type,
                        'lambda': 0.5,
                        'train_size': len(train_data[0]),
                        'val_size': len(val_data[0]),
                    }, save_path)
                    print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error training on VAE lambda file {file_path}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Skipped training on VAE lambda=0.5 files.")

    print("\nAll baseline trainings complete. Models saved to src/baseline_saved_models/")


if __name__ == "__main__":
    main()


