#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import Counter
from collections import Counter
import argparse
import json

# Setting random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Helper function to filter and select sequences
def filter_sequences(file_path, train_size, label="", seq_type="donor", random_seed=42):
    all_sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            all_sequences.append(line.strip().upper())
    
    # Filter sequences
    valid_sequences = []
    discarded_count = 0
    
    for seq in all_sequences:
        # Check if sequence has length 402
        if len(seq) != 402:
            discarded_count += 1
            continue
            
        # Check if sequence has the correct splice site motif at positions 200-201
        if seq_type == "donor" and seq[200:202] != 'GT':
            discarded_count += 1
            continue
        elif seq_type == "acceptor" and seq[200:202] != 'AG':
            discarded_count += 1
            continue
            
        # Check if sequence contains 'N'
        if 'N' in seq:
            discarded_count += 1
            continue
            
        valid_sequences.append(seq)
    
    print(f"{label} Filtering results: {len(valid_sequences)} valid sequences, {discarded_count} discarded sequences")
    
    # Random selection with seed
    if train_size < len(valid_sequences):
        np.random.seed(random_seed)
        indices = np.random.choice(len(valid_sequences), size=train_size, replace=False)
        valid_sequences = [valid_sequences[i] for i in indices]
        print(f"{label} Randomly selected {train_size} sequences using seed {random_seed}")
    else:
        valid_sequences = valid_sequences[:train_size]
    
    print(f"{label} Using {len(valid_sequences)} sequences from the data")
    
    return valid_sequences

def load_and_encode_sequences(file_path, train_size, seq_type="donor", random_seed=42):
    valid_sequences = filter_sequences(file_path, train_size, "Training", seq_type, random_seed)
    
    # One-hot encode sequences
    encoded_sequences = []
    for seq in valid_sequences:
        encoded_seq = encode_sequence(seq)
        encoded_sequences.append(encoded_seq)
    
    return torch.stack(encoded_sequences)

def encode_sequence(seq):
    """Convert DNA sequence to one-hot encoding"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = torch.zeros(len(seq), 4)
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded

def decode_sequence(seq_tensor):
    """Convert one-hot encoding back to DNA sequence"""
    bases = ['A', 'C', 'G', 'T']
    return ''.join([bases[torch.argmax(pos)] for pos in seq_tensor])

def compute_conditional_frequency_tables_region(file_path, region_start, region_end, train_size, seq_type="donor"):
    """Compute simple neighbor-conditional nucleotide frequency tables over a region."""
    sequences = filter_sequences(file_path, train_size, label="Frequency table", seq_type=seq_type)
    prev_table = {}
    next_table = {}

    for s in sequences:
        region = s[region_start:region_end]
        L = len(region)
        for i in range(L):
            if i - 1 >= 0:
                cond_nt = region[i - 1]
                key = (i, cond_nt)
                prev_table.setdefault(key, Counter())[region[i]] += 1
            if i + 1 < L:
                cond_nt = region[i + 1]
                key = (i, cond_nt)
                next_table.setdefault(key, Counter())[region[i]] += 1

    prev_dist = {}
    for key, counter in prev_table.items():
        total = sum(counter.values())
        prev_dist[key] = {nt: count / total for nt, count in counter.items()}

    next_dist = {}
    for key, counter in next_table.items():
        total = sum(counter.values())
        next_dist[key] = {nt: count / total for nt, count in counter.items()}

    return prev_dist, next_dist

def apply_frequency_blending(seq_probs, full_prev_dist, full_next_dist, blend_weight, device):
    """Blend model probabilities with neighbor-conditional frequency priors.

    seq_probs: tensor (402, 4) with rows summing to 1
    returns: blended tensor (402, 4)
    """
    if blend_weight <= 0:
        return seq_probs

    bases = ['A','C','G','T']
    blended = seq_probs.clone()
    for i_pos in range(seq_probs.shape[0]):
        model_prob = seq_probs[i_pos]
        freq_prob = torch.zeros(4, device=device)
        count = 0
        # previous neighbor
        if i_pos - 1 >= 0:
            prev_idx = torch.argmax(seq_probs[i_pos - 1]).item()
            prev_nt = bases[prev_idx]
            key = (i_pos, prev_nt)
            if key in full_prev_dist:
                vec = torch.tensor([full_prev_dist[key].get(b, 0.0) for b in bases], device=device)
                freq_prob += vec
                count += 1
        # next neighbor
        if i_pos + 1 < seq_probs.shape[0]:
            next_idx = torch.argmax(seq_probs[i_pos + 1]).item()
            next_nt = bases[next_idx]
            key = (i_pos, next_nt)
            if key in full_next_dist:
                vec = torch.tensor([full_next_dist[key].get(b, 0.0) for b in bases], device=device)
                freq_prob += vec
                count += 1
        if count > 0:
            freq_prob = freq_prob / count
        else:
            freq_prob = torch.ones(4, device=device) / 4
        new_prob = (1 - blend_weight) * model_prob + blend_weight * freq_prob
        # ensure numerical stability
        new_prob = torch.clamp(new_prob, min=1e-8)
        new_prob = new_prob / new_prob.sum()
        blended[i_pos] = new_prob
    return blended

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        
        self.model = nn.Sequential(
            nn.Linear(config['z_dim'], 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(512, config['sequence_length'] * config['num_bases']),
            nn.LeakyReLU(0.2)
        )

    def forward(self, z):
        x = self.model(z)
        return x.view(-1, self.config['sequence_length'], self.config['num_bases'])

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config['sequence_length'] * config['num_bases'], 512),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def enforce_splice_site(sequences, seq_type="donor"):
    """Enforce correct splice site motifs on a batch tensor of shape (B, 402, 4)."""
    sequences = sequences.clone()
    device = sequences.device
    if seq_type == "donor":
        sequences[:, 200] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)  # G
        sequences[:, 201] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # T
    else:  # acceptor
        sequences[:, 200] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # A
        sequences[:, 201] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)  # G
    return sequences

def enforce_splice_site_single(sequence_probs, seq_type="donor"):
    """Enforce correct splice site motifs on a single sequence tensor of shape (402, 4)."""
    sequence_probs = sequence_probs.clone()
    device = sequence_probs.device
    if seq_type == "donor":
        sequence_probs[200] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
        sequence_probs[201] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    else:
        sequence_probs[200] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        sequence_probs[201] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)
    return sequence_probs

def train_gan(generator, discriminator, dataloader, device, seq_type, config):
    """Train GAN model with timing and loss tracking"""
    optimizer_G = optim.Adam(generator.parameters(), lr=config['generator_learning_rate'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['discriminator_learning_rate'], betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    
    # Timing and loss tracking
    training_start_time = time.time()
    epoch_times = []
    
    # Loss tracking
    loss_history = {
        'epoch': [],
        'discriminator_loss_real': [],
        'discriminator_loss_fake': [],
        'discriminator_loss_total': [],
        'generator_loss': [],
        'epoch_time': []
    }
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        epoch_d_loss_real = 0.0
        epoch_d_loss_fake = 0.0
        epoch_d_loss_total = 0.0
        epoch_g_loss = 0.0
        batch_count = 0
        
        for batch_idx, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real data
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data
            noise = torch.randn(batch_size, config['z_dim']).to(device)
            fake_data = generator(noise)
            fake_data = enforce_splice_site(fake_data, seq_type)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            noise = torch.randn(batch_size, config['z_dim']).to(device)
            fake_data = generator(noise)
            fake_data = enforce_splice_site(fake_data, seq_type)
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Accumulate losses
            epoch_d_loss_real += d_loss_real.item()
            epoch_d_loss_fake += d_loss_fake.item()
            epoch_d_loss_total += d_loss.item()
            epoch_g_loss += g_loss.item()
            batch_count += 1
        
        # Calculate average losses for the epoch
        avg_d_loss_real = epoch_d_loss_real / batch_count
        avg_d_loss_fake = epoch_d_loss_fake / batch_count
        avg_d_loss_total = epoch_d_loss_total / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Store loss data
        loss_history['epoch'].append(epoch)
        loss_history['discriminator_loss_real'].append(avg_d_loss_real)
        loss_history['discriminator_loss_fake'].append(avg_d_loss_fake)
        loss_history['discriminator_loss_total'].append(avg_d_loss_total)
        loss_history['generator_loss'].append(avg_g_loss)
        loss_history['epoch_time'].append(epoch_time)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{config["epochs"]}, D_Loss_Real: {avg_d_loss_real:.4f}, D_Loss_Fake: {avg_d_loss_fake:.4f}, G_Loss: {avg_g_loss:.4f}, Time: {epoch_time:.2f}s')
    
    training_time = time.time() - training_start_time
    
    return {
        'total_training_time': training_time,
        'avg_epoch_time': np.mean(epoch_times),
        'final_d_loss_real': loss_history['discriminator_loss_real'][-1],
        'final_d_loss_fake': loss_history['discriminator_loss_fake'][-1],
        'final_d_loss_total': loss_history['discriminator_loss_total'][-1],
        'final_g_loss': loss_history['generator_loss'][-1],
        'loss_history': loss_history
    }

def generate_sequences(generator, device, config, seq_type="donor", num_sequences=1000, blend_weight=0.0, full_prev_dist=None, full_next_dist=None):
    """Generate sequences with timing"""
    generator.eval()
    
    generation_start_time = time.time()
    generated_sequences = []
    
    with torch.no_grad():
        # Generate in batches
        batch_size = 100
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_sequences - i * batch_size)
            noise = torch.randn(current_batch_size, config['z_dim']).to(device)
            
            fake_data = generator(noise)
            fake_data = enforce_splice_site(fake_data, seq_type)
            
            # Convert to discrete sequences
            for j in range(current_batch_size):
                seq_tensor = fake_data[j]
                # Apply softmax to get proper probabilities
                seq_probs = torch.softmax(seq_tensor, dim=-1)
                
                # Optional frequency blending
                if blend_weight > 0 and full_prev_dist is not None and full_next_dist is not None:
                    seq_probs = apply_frequency_blending(seq_probs, full_prev_dist, full_next_dist, blend_weight, device)
                
                # Enforce splice site after blending for safety (single sequence)
                seq_probs = enforce_splice_site_single(seq_probs, seq_type)
                
                # Clamp to avoid numerical issues
                seq_probs = torch.clamp(seq_probs, min=1e-8, max=1.0)
                # Sample from the probability distribution
                sampled_indices = torch.multinomial(seq_probs, 1).squeeze()
                one_hot = torch.zeros_like(seq_tensor)
                one_hot.scatter_(1, sampled_indices.unsqueeze(1), 1)
                sequence = decode_sequence(one_hot)
                generated_sequences.append(sequence)
    
    generation_time = time.time() - generation_start_time
    
    return generated_sequences, {
        'generation_time': generation_time,
        'sequences_per_second': num_sequences / generation_time
    }

def train_and_generate_gan(species, seq_type, train_size):
    """Main function to train GAN and generate sequences"""
    print(f"Training GAN for {species} {seq_type}...")
    
    # Set device (force cuda:0 if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Species-specific training sizes
    actual_train_size = 50000 if species == "arabidopsis" else 100000
    print(f"Using {actual_train_size} sequences for {species}")
    
    # Configuration with user's specified parameters
    config = {
        'sequence_length': 402,
        'num_bases': 4,
        'z_dim': 100,
        'batch_size': 512,
        'epochs': 50,
        'data_sampling_ratio': 1.0,
        'num_examples': None,
        'generator_learning_rate': 0.00005,
        'discriminator_learning_rate': 0.00002,
        'dropout_rate': 0.3
    }
    
    # Load training data
    train_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"
    train_data = load_and_encode_sequences(train_file, actual_train_size, seq_type, random_seed=42)
    
    # Create data loader with config batch size
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize models with config
    generator = Generator(config).to(device)
    discriminator = Discriminator(config).to(device)
    
    # Train the model
    training_stats = train_gan(generator, discriminator, dataloader, device, seq_type, config)
    
    # Generate sequences (60,000 for each λ in {0.25, 0.5, 0.75})
    num_to_generate = 60000
    lambda_values = [0.25, 0.5, 0.75]
    # Compute frequency priors (once) for blending
    full_prev_dist, full_next_dist = compute_conditional_frequency_tables_region(
        train_file, 0, 402, actual_train_size, seq_type
    )
    
    os.makedirs('GAN_generated_sequences', exist_ok=True)
    generation_stats = {}
    for lam in lambda_values:
        print(f"Generating {num_to_generate} sequences with λ={lam}...")
        generated_sequences, gen_stats = generate_sequences(
            generator, device, config, seq_type, num_to_generate, blend_weight=lam,
            full_prev_dist=full_prev_dist, full_next_dist=full_next_dist
        )
        generation_stats[str(lam)] = gen_stats
        output_file = f"GAN_generated_sequences/{species}_{seq_type}_train_{actual_train_size//1000}k_lambda_{lam}_generated_sequences.txt"
        with open(output_file, 'w') as f:
            for seq in generated_sequences:
                f.write(seq + '\n')
        print(f"Generated {len(generated_sequences)} sequences and saved to {output_file}")
    
    # Save model and loss data
    model_dir = f"GAN_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save loss data as JSON
    loss_file = f"{model_dir}/{species}_{seq_type}_gan_losses.json"
    with open(loss_file, 'w') as f:
        json.dump(training_stats['loss_history'], f, indent=2)
    print(f"Loss data saved to {loss_file}")
    
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'config': config,
        'training_stats': training_stats,
        'generation_stats': generation_stats
    }, f"{model_dir}/{species}_{seq_type}_gan_model.pth")
    
    # Return combined statistics
    return {
        'model_type': 'GAN',
        'species': species,
        'seq_type': seq_type,
        'train_size': actual_train_size,
        'config': config,
        **training_stats,
        **generation_stats
    }

def ask_user_permission(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [Y/n]: ").strip().lower()
    except EOFError:
        ans = 'y'
    if ans == '' or ans.startswith('y'):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Train GAN baseline for splice site generation')
    parser.add_argument('--species', choices=['arabidopsis', 'homo'], default='arabidopsis')
    parser.add_argument('--seq_type', choices=['donor', 'acceptor'], default='donor')
    parser.add_argument('--train_size', type=int, default=50, help='Training size in thousands')
    parser.add_argument('--non_interactive', action='store_true', help='Run without interactive prompts')
    
    args = parser.parse_args()

    if args.non_interactive:
        stats = train_and_generate_gan(args.species, args.seq_type, args.train_size)
        print("\n" + "="*50)
        print("GAN TRAINING SUMMARY")
        print("="*50)
        print(f"Model: {stats['model_type']}")
        print(f"Species: {stats['species']}")
        print(f"Sequence Type: {stats['seq_type']}")
        print(f"Training Size: {stats['train_size']}k")
        print(f"Total Training Time: {stats['total_training_time']:.2f}s")
        print(f"Average Epoch Time: {stats['avg_epoch_time']:.2f}s")
        # Per-λ generation stats are saved; summarize keys available
        gen_lams = [k for k in ['0.25','0.5','0.75'] if k in stats]
        if gen_lams:
            k = gen_lams[0]
            print(f"Sample Generation (λ={k}): {stats[k]['generation_time']:.2f}s, {stats[k]['sequences_per_second']:.2f} seq/s")
        print("="*50)
        return

    # Interactive flow over all species/types
    combos = [
        ("arabidopsis", "donor"),
        ("arabidopsis", "acceptor"),
        ("homo", "donor"),
        ("homo", "acceptor"),
    ]

    for sp, st in combos:
        if ask_user_permission(f"Process {sp} {st} with GAN?"):
            train_and_generate_gan(sp, st, args.train_size)
        else:
            print(f"Skipping {sp} {st}.")

if __name__ == "__main__":
    main() 