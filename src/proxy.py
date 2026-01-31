import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import glob

# Import models
from models import SpliceRover, SpliceFinder, DeepSplicer, IntSplice, Spliceator

# Set seeds for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# Create output directory
os.makedirs("src/saved_models", exist_ok=True)

class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def one_hot_encode(sequences):
    """One-hot encode DNA sequences"""
    base_dict = {'A': [1,0,0,0], 'T': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1]}
    encoded = np.zeros((len(sequences), len(sequences[0]), 4))
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            encoded[i, j] = base_dict.get(base, [0,0,0,0])
    return encoded

def load_sequences(file_path, seq_type=None):
    """Load sequences from file with optional filtering"""
    all_sequences = []
    total_count = 0
    filtered_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            seq = line.strip().upper()
            if not seq:
                continue
            
            total_count += 1
            
            # Check for correct splice site motif if seq_type is specified
            if seq_type:
                # Check at positions 200-201
                if seq_type == "donor" and seq[200:202] == "GT":
                    all_sequences.append(seq)
                elif seq_type == "acceptor" and seq[200:202] == "AG":
                    all_sequences.append(seq)
                else:
                    filtered_count += 1
            else:
                all_sequences.append(seq)
    
    if seq_type:
        print(f"  File: {os.path.basename(file_path)}")
        print(f"  Total sequences: {total_count}")
        print(f"  Filtered out: {filtered_count} ({filtered_count/total_count*100:.1f}% missing correct {seq_type} motif)")
        print(f"  Retained: {len(all_sequences)} ({len(all_sequences)/total_count*100:.1f}%)")
    else:
        print(f"Loaded {len(all_sequences)} sequences from {os.path.basename(file_path)}")
    
    return all_sequences

def prepare_real_data(species, seq_type, train_size):
    """Prepare real training and validation data"""
    print(f"\nPreparing real data for {species} {seq_type}...")
    
    # Load positive sequences
    pos_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"
    pos_sequences = load_sequences(pos_file, seq_type)
    
    # Load negative sequences
    neg_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_negative.txt"
    neg_sequences = load_sequences(neg_file)
    
    # Shuffle with seed 42
    random.seed(42)
    random.shuffle(pos_sequences)
    random.shuffle(neg_sequences)
    
    # Training data: specified amount (50k for arabidopsis, 100k for homo)
    train_size_k = train_size * 1000
    pos_train = pos_sequences[:train_size_k]
    neg_train = neg_sequences[:train_size_k]
    
    # Validation data: half of remaining (balanced)
    remaining_pos = pos_sequences[train_size_k:]
    remaining_neg = neg_sequences[train_size_k:]
    
    # Use the minimum of the two to ensure balanced validation set
    max_val_size = min(len(remaining_pos), len(remaining_neg)) // 2
    
    pos_val = remaining_pos[:max_val_size]
    neg_val = remaining_neg[:max_val_size]
    
    # Combine training data
    train_sequences = pos_train + neg_train
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    
    # Combine validation data
    val_sequences = pos_val + neg_val
    val_labels = [1] * len(pos_val) + [0] * len(neg_val)
    
    # Shuffle combined data
    train_data = list(zip(train_sequences, train_labels))
    val_data = list(zip(val_sequences, val_labels))
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    train_sequences, train_labels = zip(*train_data)
    val_sequences, val_labels = zip(*val_data)
    
    print(f"Training set: {len(train_sequences)} sequences ({len(pos_train)} pos, {len(neg_train)} neg)")
    print(f"Validation set: {len(val_sequences)} sequences ({len(pos_val)} pos, {len(neg_val)} neg) - BALANCED")
    
    return (list(train_sequences), list(train_labels)), (list(val_sequences), list(val_labels))

def prepare_synthetic_data(species, seq_type, lambda_val):
    """Prepare synthetic training data for a specific lambda value with real validation data"""
    print(f"\nPreparing synthetic data for {species} {seq_type} λ={lambda_val}...")
    
    # Load synthetic sequences for this specific lambda value
    synthetic_file = f"src/Lambda_sensitivity_analysis/{species}_{seq_type}_lambda_{lambda_val}_sequences.txt"
    if not os.path.exists(synthetic_file):
        raise FileNotFoundError(f"Synthetic file not found: {synthetic_file}")
    
    synthetic_sequences = load_sequences(synthetic_file)
    print(f"  Loaded {len(synthetic_sequences)} synthetic sequences for λ={lambda_val}")
    
    # Load negative sequences
    neg_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_negative.txt"
    neg_sequences = load_sequences(neg_file)
    
    # Load real positive sequences FOR VALIDATION
    pos_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"
    real_pos_sequences = load_sequences(pos_file, seq_type)
    
    # Shuffle with seed 42
    random.seed(42)
    random.shuffle(synthetic_sequences)
    random.shuffle(neg_sequences)
    random.shuffle(real_pos_sequences)
    
    # Training data: Synthetic positive + equal number of negatives
    pos_train = synthetic_sequences
    neg_train = neg_sequences[:len(pos_train)]
    
    # Validation data: Use REAL sequences (skip the amount used for real training)
    train_size = 50 if species == "arabidopsis" else 100
    train_size_k = train_size * 1000
    
    # Skip real sequences used for real training, then use remaining for validation
    remaining_real_pos = real_pos_sequences[train_size_k:]
    remaining_real_neg = neg_sequences[len(pos_train):]
    
    # Use half of remaining real sequences for balanced validation
    max_val_size = min(len(remaining_real_pos), len(remaining_real_neg)) // 2
    
    pos_val = remaining_real_pos[:max_val_size]
    neg_val = remaining_real_neg[:max_val_size]
    
    # Combine training data (synthetic positives + negatives)
    train_sequences = pos_train + neg_train
    train_labels = [1] * len(pos_train) + [0] * len(neg_train)
    
    # Combine validation data (real positives + negatives)
    val_sequences = pos_val + neg_val
    val_labels = [1] * len(pos_val) + [0] * len(neg_val)
    
    # Shuffle combined data
    train_data = list(zip(train_sequences, train_labels))
    val_data = list(zip(val_sequences, val_labels))
    
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    train_sequences, train_labels = zip(*train_data)
    val_sequences, val_labels = zip(*val_data)
    
    print(f"Training set: {len(train_sequences)} sequences ({len(pos_train)} synthetic pos, {len(neg_train)} neg)")
    print(f"Validation set: {len(val_sequences)} sequences ({len(pos_val)} real pos, {len(neg_val)} neg) - BALANCED")
    
    return (list(train_sequences), list(train_labels)), (list(val_sequences), list(val_labels))

def create_data_loaders(train_data, val_data, batch_size=64):
    """Create PyTorch data loaders"""
    train_sequences, train_labels = train_data
    val_sequences, val_labels = val_data
    
    # One-hot encode sequences
    train_encoded = one_hot_encode(train_sequences)
    val_encoded = one_hot_encode(val_sequences)
    
    # Create datasets
    train_dataset = DNASequenceDataset(torch.FloatTensor(train_encoded), torch.LongTensor(train_labels))
    val_dataset = DNASequenceDataset(torch.FloatTensor(val_encoded), torch.LongTensor(val_labels))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, patience=10):
    """Train a model with early stopping"""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"Training with early stopping (patience={patience})...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improvement_marker = "✅"
        else:
            patience_counter += 1
            improvement_marker = f"⏳ ({patience_counter}/{patience})"
        
        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% {improvement_marker}")
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered at epoch {epoch+1} (patience={patience})")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model

def get_model_instance(model_name):
    """Get instance of specified model"""
    model_classes = {
        "SpliceRover": SpliceRover,
        "SpliceFinder": SpliceFinder,
        "DeepSplicer": DeepSplicer,
        "IntSplice": IntSplice,
        "Spliceator": Spliceator
    }
    
    if model_name in model_classes:
        return model_classes[model_name]()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def ask_user_permission(prompt: str) -> bool:
    """Ask user if they want to proceed (Y/n). Returns True to proceed, False to skip."""
    while True:
        resp = input(f"{prompt} (Y/n): ").strip().lower()
        if resp in ["y", "yes", ""]:
            return True
        if resp in ["n", "no"]:
            return False
        print("Please answer with 'Y' or 'n'.")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_names = ["SpliceRover", "SpliceFinder", "DeepSplicer", "IntSplice", "Spliceator"]
    
    # Training sizes for different species
    species_train_sizes = {
        "arabidopsis": 50,  # 50k sequences
        "homo": 100         # 100k sequences
    }
    
    seq_types = ["donor", "acceptor"]
    
    # Main training loop
    for species in species_train_sizes.keys():
        for seq_type in seq_types:
            
            # Train on real data (ask before proceeding)
            print(f"\n{'='*80}")
            print(f"TRAINING ON REAL DATA: {species.upper()} {seq_type.upper()}")
            print('='*80)
            
            if not ask_user_permission(f"Proceed with REAL data for {species} {seq_type}?"):
                print(f"⏭️  Skipping REAL data for {species} {seq_type}")
            else:
                try:
                    train_data, val_data = prepare_real_data(species, seq_type, species_train_sizes[species])
                    
                    # Create data loaders
                    train_loader, val_loader = create_data_loaders(train_data, val_data)
                    
                    # Train each model on real data (ask per-model)
                    for model_name in model_names:
                        if not ask_user_permission(f"  Proceed with model {model_name} on REAL {species} {seq_type}?"):
                            print(f"  ⏭️  Skipping {model_name} on REAL {species} {seq_type}")
                            continue
                        
                        print(f"\n{'-'*60}")
                        print(f"Training {model_name} on real {species} {seq_type}")
                        print('-'*60)
                        
                        try:
                            # Get model instance
                            model = get_model_instance(model_name)
                            
                            # Train model
                            trained_model = train_model(model, train_loader, val_loader, device)
                            
                            # Save model
                            model_filename = f"{model_name}_real_{species}_{seq_type}.pth"
                            model_path = os.path.join("src/saved_models", model_filename)
                            
                            torch.save({
                                'model_state_dict': trained_model.state_dict(),
                                'model_name': model_name,
                                'data_type': 'real',
                                'species': species,
                                'seq_type': seq_type,
                                'train_size': len(train_data[0]),
                                'val_size': len(val_data[0])
                            }, model_path)
                            
                            print(f"✅ Model saved to {model_path}")
                            
                        except Exception as e:
                            print(f"❌ Error training {model_name} on real data: {e}")
                            import traceback
                            traceback.print_exc()
                            
                except Exception as e:
                    print(f"❌ Error preparing real data for {species} {seq_type}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Train on synthetic data (each lambda separately) with prompts
            lambda_values = ["0.0", "0.25", "0.5", "0.75", "1.0"]
            for lambda_val in lambda_values:
                print(f"\n{'='*80}")
                print(f"TRAINING ON SYNTHETIC DATA FOR λ={lambda_val}: {species.upper()} {seq_type.upper()}")
                print('='*80)
                
                if not ask_user_permission(f"Proceed with SYNTHETIC λ={lambda_val} for {species} {seq_type}?"):
                    print(f"⏭️  Skipping SYNTHETIC λ={lambda_val} for {species} {seq_type}")
                    continue
                
                try:
                    train_data, val_data = prepare_synthetic_data(species, seq_type, lambda_val)
                    
                    # Create data loaders
                    train_loader, val_loader = create_data_loaders(train_data, val_data)
                    
                    # Train each model on this synthetic dataset (ask per-model)
                    for model_name in model_names:
                        if not ask_user_permission(f"  Proceed with model {model_name} on SYNTHETIC {species} {seq_type} λ={lambda_val}?"):
                            print(f"  ⏭️  Skipping {model_name} on SYNTHETIC {species} {seq_type} λ={lambda_val}")
                            continue
                        
                        print(f"\n{'-'*60}")
                        print(f"Training {model_name} on synthetic {species} {seq_type} λ={lambda_val}")
                        print('-'*60)
                        
                        try:
                            # Get model instance
                            model = get_model_instance(model_name)
                            
                            # Train model
                            trained_model = train_model(model, train_loader, val_loader, device)
                            
                            # Save model
                            model_filename = f"{model_name}_synthetic_{species}_{seq_type}_lambda_{lambda_val}.pth"
                            model_path = os.path.join("src/saved_models", model_filename)
                            
                            torch.save({
                                'model_state_dict': trained_model.state_dict(),
                                'model_name': model_name,
                                'data_type': 'synthetic',
                                'species': species,
                                'seq_type': seq_type,
                                'lambda_val': lambda_val,
                                'train_size': len(train_data[0]),
                                'val_size': len(val_data[0])
                            }, model_path)
                            
                            print(f"✅ Model saved to {model_path}")
                            
                        except Exception as e:
                            print(f"❌ Error training {model_name} on synthetic λ={lambda_val}: {e}")
                            import traceback
                            traceback.print_exc()
                            
                except FileNotFoundError as e:
                    print(f"❌ Synthetic data file not found for λ={lambda_val}: {e}")
                except Exception as e:
                    print(f"❌ Error preparing synthetic data for λ={lambda_val}: {e}")
                    import traceback
                    traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print('='*80)
    print(f"All trained models saved to src/saved_models/")
    
    # List saved models
    saved_models = glob.glob("src/saved_models/*.pth")
    print(f"\nSaved {len(saved_models)} models:")
    for model_path in sorted(saved_models):
        print(f"  📁 {os.path.basename(model_path)}")

if __name__ == "__main__":
    main() 