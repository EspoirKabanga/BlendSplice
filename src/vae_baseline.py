#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
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
        prev_dist[key] = {nt: count/total for nt, count in counter.items()}
    next_dist = {}
    for key, counter in next_table.items():
        total = sum(counter.values())
        next_dist[key] = {nt: count/total for nt, count in counter.items()}
    return prev_dist, next_dist

def apply_frequency_blending(seq_probs, full_prev_dist, full_next_dist, blend_weight, device):
    if blend_weight <= 0:
        return seq_probs
    bases = ['A','C','G','T']
    blended = seq_probs.clone()
    for i_pos in range(seq_probs.shape[0]):
        model_prob = seq_probs[i_pos]
        freq_prob = torch.zeros(4, device=device)
        count = 0
        if i_pos - 1 >= 0:
            prev_idx = torch.argmax(seq_probs[i_pos - 1]).item()
            prev_nt = bases[prev_idx]
            key = (i_pos, prev_nt)
            if key in full_prev_dist:
                vec = torch.tensor([full_prev_dist[key].get(b, 0.0) for b in bases], device=device)
                freq_prob += vec
                count += 1
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
        new_prob = torch.clamp(new_prob, min=1e-8)
        new_prob = new_prob / new_prob.sum()
        blended[i_pos] = new_prob
    return blended

class DNA_VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(DNA_VAE_Encoder, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 50, latent_dim)
        self.fc_logvar = nn.Linear(128 * 50, latent_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 201)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 100)
        x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 50)
        x = self.flatten(x)                   # (B, 128*50)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class DNA_VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(DNA_VAE_Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 50)
        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)  # 50 -> 100
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)   # 100 -> 200
        self.deconv3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)   # 200 -> 400
        self.deconv4 = nn.ConvTranspose1d(16, 4, kernel_size=3, stride=1, padding=0)    # 400 -> 402

    def forward(self, z):
        x = self.fc(z)               # (B, 128*50)
        x = x.view(-1, 128, 50)      # (B, 128, 50)
        x = F.relu(self.deconv1(x))  # (B, 64, 100)
        x = F.relu(self.deconv2(x))  # (B, 32, 200)
        x = F.relu(self.deconv3(x))  # (B, 16, 400)
        x = self.deconv4(x)          # (B, 4, 402)
        return x

# class UNetBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNetBlock1D, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(4, out_channels),
#             nn.SiLU(),
#             nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.GroupNorm(4, out_channels),
#             nn.SiLU()
#         )

#     def forward(self, x):
#         return self.conv(x)

# class LatentDiffusionUNet(nn.Module):
#     def __init__(self, latent_dim=128, seq_len=402):
#         super().__init__()
#         self.input_proj = nn.Linear(latent_dim, 256)
#         self.down1 = UNetBlock1D(256, 512)
#         self.down2 = UNetBlock1D(512, 512)
#         self.up1 = UNetBlock1D(512, 256)
#         self.up2 = UNetBlock1D(256, 256)
#         self.output_proj = nn.Linear(256, latent_dim)

#     def forward(self, z_t, t_emb):
#         # z_t: (B, latent_dim)
#         x = self.input_proj(z_t).unsqueeze(-1)  # (B, 256, 1)
#         x = F.interpolate(x, size=25, mode='nearest')  # simulate temporal features

#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         u1 = self.up1(F.interpolate(d2, scale_factor=2))
#         u2 = self.up2(F.interpolate(u1, size=d1.shape[-1]))

#         x = self.output_proj(u2.mean(dim=-1))  # global average over time dimension
#         return x

class SpliceVAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512):
        super(SpliceVAE, self).__init__()
        self.encoder = DNA_VAE_Encoder(latent_dim=latent_dim)
        self.decoder = DNA_VAE_Decoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # x: (batch, seq_len, 4) -> (batch, 4, seq_len)
        x = x.permute(0, 2, 1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        # Convert back to (batch, seq_len, 4)
        recon_x = recon_x.permute(0, 2, 1)
        return recon_x, mu, logvar
    
    def generate(self, num_samples, device):
        """Generate new sequences"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            generated = self.decoder(z)
            # Convert from (batch, 4, seq_len) to (batch, seq_len, 4)
            generated = generated.permute(0, 2, 1)
        return generated

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def vae_loss_function(recon_x, x, mu, logvar):
    """VAE loss combining reconstruction loss and KL divergence"""
    # Convert input format for cross entropy: (batch, seq_len, 4) -> (batch, 4, seq_len)
    recon_logits = recon_x.permute(0, 2, 1)
    x_targets = x.argmax(dim=-1)  # Convert one-hot to class indices
    
    # Reconstruction loss
    recon_loss = F.cross_entropy(recon_logits, x_targets, reduction='mean')
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return recon_loss + kld, recon_loss, kld

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

def train_vae(model, dataloader, device, seq_type="donor", num_epochs=50):
    """Train VAE model with timing and loss tracking"""
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Timing and loss tracking
    training_start_time = time.time()
    epoch_times = []
    
    # Loss tracking
    loss_history = {
        'epoch': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'kl_divergence_loss': [],
        'epoch_time': []
    }
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        batch_count = 0
        
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            batch_count += 1
        
        # Calculate average losses
        avg_total_loss = total_loss / batch_count
        avg_recon_loss = total_recon_loss / batch_count
        avg_kl_loss = total_kl_loss / batch_count
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Store loss data
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_total_loss)
        loss_history['reconstruction_loss'].append(avg_recon_loss)
        loss_history['kl_divergence_loss'].append(avg_kl_loss)
        loss_history['epoch_time'].append(epoch_time)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, Time: {epoch_time:.2f}s')
    
    training_time = time.time() - training_start_time
    
    return {
        'total_training_time': training_time,
        'avg_epoch_time': np.mean(epoch_times),
        'final_total_loss': loss_history['total_loss'][-1],
        'final_recon_loss': loss_history['reconstruction_loss'][-1],
        'final_kl_loss': loss_history['kl_divergence_loss'][-1],
        'loss_history': loss_history
    }

def generate_sequences(model, device, seq_type="donor", num_sequences=1000, blend_weight=0.0, full_prev_dist=None, full_next_dist=None):
    """Generate sequences with timing"""
    model.eval()
    
    generation_start_time = time.time()
    generated_sequences = []
    
    with torch.no_grad():
        # Generate in batches
        batch_size = 100
        num_batches = (num_sequences + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_sequences - i * batch_size)
            
            # Generate from latent space
            z = torch.randn(current_batch_size, model.latent_dim).to(device)
            generated_batch = model.decoder(z)  # Returns (batch, 4, seq_len)
            
            # Convert to (batch, seq_len, 4) and apply softmax
            generated_batch = generated_batch.permute(0, 2, 1)
            generated_batch = torch.softmax(generated_batch, dim=-1)
            
            # Apply frequency blending if requested
            if blend_weight > 0 and full_prev_dist is not None and full_next_dist is not None:
                blended_list = []
                for b in range(generated_batch.size(0)):
                    seq_probs = generated_batch[b]
                    seq_probs = apply_frequency_blending(seq_probs, full_prev_dist, full_next_dist, blend_weight, device)
                    blended_list.append(seq_probs)
                generated_batch = torch.stack(blended_list, dim=0)
            
            # Enforce splice site constraints
            generated_batch = enforce_splice_site(generated_batch, seq_type)
            
            # Convert to discrete sequences
            for j in range(current_batch_size):
                seq_tensor = generated_batch[j]
                # Enforce per-sequence splice site safety
                seq_tensor = enforce_splice_site_single(seq_tensor, seq_type)
                # Sample from the probability distribution
                sampled_indices = torch.multinomial(seq_tensor, 1).squeeze()
                one_hot = torch.zeros_like(seq_tensor)
                one_hot.scatter_(1, sampled_indices.unsqueeze(1), 1)
                sequence = decode_sequence(one_hot)
                generated_sequences.append(sequence)
    
    generation_time = time.time() - generation_start_time
    
    return generated_sequences, {
        'generation_time': generation_time,
        'sequences_per_second': num_sequences / generation_time
    }

def train_and_generate_vae(species, seq_type, train_size):
    """Main function to train VAE and generate sequences"""
    print(f"Training VAE for {species} {seq_type}...")
    
    # Set device (force cuda:0 if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Species-specific training sizes
    actual_train_size = 50000 if species == "arabidopsis" else 100000
    print(f"Using {actual_train_size} sequences for {species}")
    
    # Load training data
    train_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"
    train_data = load_and_encode_sequences(train_file, actual_train_size, seq_type, random_seed=42)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SpliceVAE(latent_dim=128, hidden_dim=512).to(device)
    
    # Train the model
    training_stats = train_vae(model, dataloader, device, seq_type)
    
    # Generate sequences (60,000 for each λ in {0.25, 0.5, 0.75})
    num_to_generate = 60000
    lambda_values = [0.25, 0.5, 0.75]
    # Compute frequency priors for blending
    full_prev_dist, full_next_dist = compute_conditional_frequency_tables_region(
        train_file, 0, 402, actual_train_size, seq_type
    )
    
    os.makedirs('VAE_generated_sequences', exist_ok=True)
    generation_stats = {}
    for lam in lambda_values:
        print(f"Generating {num_to_generate} sequences with λ={lam}...")
        generated_sequences, gen_stats = generate_sequences(
            model, device, seq_type, num_to_generate, blend_weight=lam,
            full_prev_dist=full_prev_dist, full_next_dist=full_next_dist
        )
        generation_stats[str(lam)] = gen_stats
        output_file = f"VAE_generated_sequences/{species}_{seq_type}_train_{actual_train_size//1000}k_lambda_{lam}_generated_sequences.txt"
        with open(output_file, 'w') as f:
            for seq in generated_sequences:
                f.write(seq + '\n')
        print(f"Generated {len(generated_sequences)} sequences and saved to {output_file}")
    
    # Save model and loss data
    model_dir = f"VAE_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save loss data as JSON
    loss_file = f"{model_dir}/{species}_{seq_type}_vae_losses.json"
    with open(loss_file, 'w') as f:
        json.dump(training_stats['loss_history'], f, indent=2)
    print(f"Loss data saved to {loss_file}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_stats': training_stats,
        'generation_stats': generation_stats
    }, f"{model_dir}/{species}_{seq_type}_vae_model.pth")
    
    # Return combined statistics
    return {
        'model_type': 'VAE',
        'species': species,
        'seq_type': seq_type,
        'train_size': actual_train_size,
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
    parser = argparse.ArgumentParser(description='Train VAE baseline for splice site generation')
    parser.add_argument('--species', choices=['arabidopsis', 'homo'], default='arabidopsis')
    parser.add_argument('--seq_type', choices=['donor', 'acceptor'], default='donor')
    parser.add_argument('--train_size', type=int, default=50, help='Training size in thousands')
    parser.add_argument('--non_interactive', action='store_true', help='Run without interactive prompts')
    
    args = parser.parse_args()

    if args.non_interactive:
        stats = train_and_generate_vae(args.species, args.seq_type, args.train_size)
        print("\n" + "="*50)
        print("VAE TRAINING SUMMARY")
        print("="*50)
        print(f"Model: {stats['model_type']}")
        print(f"Species: {stats['species']}")
        print(f"Sequence Type: {stats['seq_type']}")
        print(f"Training Size: {stats['train_size']}k")
        print(f"Total Training Time: {stats['total_training_time']:.2f}s")
        print(f"Average Epoch Time: {stats['avg_epoch_time']:.2f}s")
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
        if ask_user_permission(f"Process {sp} {st} with VAE?"):
            train_and_generate_vae(sp, st, args.train_size)
        else:
            print(f"Skipping {sp} {st}.")

if __name__ == "__main__":
    main() 