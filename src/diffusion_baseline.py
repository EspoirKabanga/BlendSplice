#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import time
import argparse
import json

def gc_fraction(seq):
    """Calculate GC content without BioPython dependency"""
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

# Setting random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Specify device
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Ensure output directories exist
os.makedirs('Diffusion_generated_seq_UNET_Enhanced', exist_ok=True)
os.makedirs('Lambda_sensitivity_analysis', exist_ok=True)
os.makedirs('Computational_analysis', exist_ok=True)

# Lambda values for comprehensive sensitivity analysis
LAMBDA_VALUES = [0.5, 0.0, 0.25, 0.75, 1.0]

# Memory tracking utilities
def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
        }
    return {'allocated': 0, 'cached': 0}

# Helper function to filter and select sequences
def filter_sequences(file_path, train_size, label="", seq_type="donor"):
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
    
    valid_sequences = valid_sequences[:train_size]
    print(f"{label} Using {train_size} sequences from the data")
    
    return valid_sequences

# Data loading and preprocessing (full sequences of length 402)
def load_and_encode_sequences(file_path, train_size, seq_type="donor"):
    # Get filtered sequences
    valid_sequences = filter_sequences(file_path, train_size, label="Data loading", seq_type=seq_type)
    
    # One-hot encode the sequences
    seq_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    encoded_seqs = np.array([[seq_dict[nt] for nt in seq] for seq in valid_sequences])
    
    return torch.tensor(encoded_seqs).float().to(device)

# Compute Conditional Frequency Distributions for the entire sequence
def compute_conditional_frequency_tables_region(file_path, region_start, region_end, train_size, seq_type="donor"):
    # Get filtered sequences
    sequences = filter_sequences(file_path, train_size, label="Frequency table", seq_type=seq_type)
    prev_table = {}  # key: (position, conditioning_nt) using previous neighbor
    next_table = {}  # key: (position, conditioning_nt) using next neighbor

    # For each sequence, extract the region and re-index it from 0.
    for s in sequences:
        region = s[region_start:region_end]
        L = len(region)
        for i in range(L):
            # Condition on previous neighbor if available
            if i - 1 >= 0:
                cond_nt = region[i - 1]
                key = (i, cond_nt)
                prev_table.setdefault(key, Counter())[region[i]] += 1
            # Condition on next neighbor if available
            if i + 1 < L:
                cond_nt = region[i + 1]
                key = (i, cond_nt)
                next_table.setdefault(key, Counter())[region[i]] += 1

    # Convert counts to probability distributions.
    prev_dist = {}
    for key, counter in prev_table.items():
        total = sum(counter.values())
        prev_dist[key] = {nt: count/total for nt, count in counter.items()}

    next_dist = {}
    for key, counter in next_table.items():
        total = sum(counter.values())
        next_dist[key] = {nt: count/total for nt, count in counter.items()}

    return prev_dist, next_dist

# U-Net Architecture (same as original)
class DoubleConv(nn.Module):
    """(Conv -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 have the same size
        diff = x2.size()[2] - x1.size()[2]
        
        # Handle odd dimensions with padding if needed
        if diff % 2 != 0:
            x1 = torch.nn.functional.pad(x1, (0, 1))
            diff -= 1
            
        if diff > 0:
            x1 = torch.nn.functional.pad(x1, (diff // 2, diff // 2))
        elif diff < 0:
            x2 = torch.nn.functional.pad(x2, (-diff // 2, -diff // 2))
            
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, base_channels=32):
        super(UNet, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU()
        )
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        self.up1 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x, t):
        # x: [batch, seq_len, channels]
        # Convert to [batch, channels, seq_len] for 1D convolutions
        x = x.permute(0, 2, 1)
        
        # Time embedding for diffusion timestep
        temb = self.time_embed(t.unsqueeze(1).float())
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Add time embedding to bottleneck
        batch_size = x4.shape[0]
        channels = x4.shape[1]
        time_emb = temb.view(batch_size, -1, 1).repeat(1, 1, x4.shape[2])
        # Ensure time embedding has the right dimension to match bottleneck channels
        if time_emb.shape[1] != channels:
            # Use a projection if needed
            time_emb = time_emb[:, :channels]
        x4 = x4 + time_emb
        
        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        # Convert back to [batch, seq_len, channels]
        return logits.permute(0, 2, 1)

# Noise schedule and helper functions
def linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod

timesteps = 50
alphas_cumprod = linear_noise_schedule(timesteps)
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
betas = torch.clamp(betas, 0.0001, 0.9999)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    t = torch.clamp(t, min=0, max=a.size(0) - 1)
    out = a.gather(0, t.long())
    reshape = [batch_size] + [1] * (len(x_shape) - 1)
    return out.reshape(reshape)

# Helper function to decode a one-hot tensor into a DNA sequence string.
def decode_sequence(seq_tensor):
    bases = ['A', 'C', 'G', 'T']
    return ''.join([bases[torch.argmax(pos)] for pos in seq_tensor])

# Enhanced generation function with comprehensive timing and λ parameter
def generate_full_sequence_enhanced(full_prev_dist, full_next_dist, model, seq_type="donor", blend_weight=0.5):
    """Enhanced generation with detailed timing breakdown"""
    generation_start_time = time.time()
    memory_start = get_memory_usage()
    
    # Initialize a random noisy tensor of shape (1, 402, 4)
    x_t = torch.randn((1, 402, 4), device=device)
    
    # Reverse diffusion process timing
    diffusion_start_time = time.time()
    for i in reversed(range(timesteps)):
        t = torch.tensor([i], device=device).repeat(x_t.size(0))
        predicted_noise = model(x_t, t)
        alpha_t = extract(alphas, t, x_t.shape)
        alpha_cumprod_t = extract(alphas_cumprod, t, x_t.shape)
        beta_t = extract(betas, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)
        if i > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            x_t = mean + sigma_t * noise
        else:
            x_t = mean
    diffusion_time = time.time() - diffusion_start_time

    # Blending process timing
    blending_start_time = time.time()
    seq_tensor = x_t.clone()
    seq_tensor_soft = torch.softmax(seq_tensor, dim=-1)  # shape (1, 402, 4)
    seq_tensor_soft = seq_tensor_soft.squeeze(0)  # shape (402, 4)

    # Apply frequency blending based on λ parameter
    if blend_weight > 0:
        for i_pos in range(402):
            model_prob = seq_tensor_soft[i_pos]  # tensor (4,)
            # Get frequency distributions from previous and next neighbors, if available.
            freq_prob = torch.zeros(4, device=device)
            count = 0
            # Previous neighbor
            if i_pos - 1 >= 0:
                neighbor = decode_sequence(seq_tensor_soft[i_pos-1].unsqueeze(0))
                key = (i_pos, neighbor)
                if key in full_prev_dist:
                    vec = torch.tensor([full_prev_dist[key].get(b, 0.0) for b in ['A','C','G','T']], device=device)
                    freq_prob += vec
                    count += 1
            # Next neighbor
            if i_pos + 1 < 402:
                neighbor = decode_sequence(seq_tensor_soft[i_pos+1].unsqueeze(0))
                key = (i_pos, neighbor)
                if key in full_next_dist:
                    vec = torch.tensor([full_next_dist[key].get(b, 0.0) for b in ['A','C','G','T']], device=device)
                    freq_prob += vec
                    count += 1
            if count > 0:
                freq_prob = freq_prob / count
            else:
                freq_prob = torch.ones(4, device=device) / 4  # uniform if no info available
            new_prob = (1 - blend_weight) * model_prob + blend_weight * freq_prob
            seq_tensor_soft[i_pos] = new_prob
    blending_time = time.time() - blending_start_time

    # Splice site enforcement
    enforcement_start_time = time.time()
    if seq_type == "donor":
        # Force GT at positions 200-201
        seq_tensor_soft[200] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)  # G
        seq_tensor_soft[201] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # T
    else:  # acceptor
        # Force AG at positions 200-201
        seq_tensor_soft[200] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # A
        seq_tensor_soft[201] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device)  # G
    enforcement_time = time.time() - enforcement_start_time

    # Decode by taking argmax at each position.
    decoding_start_time = time.time()
    blended_tensor = torch.zeros_like(seq_tensor_soft)
    for i in range(402):
        max_idx = torch.argmax(seq_tensor_soft[i]).item()
        blended_tensor[i, max_idx] = 1.0
    # Add batch dimension back.
    blended_tensor = blended_tensor.unsqueeze(0)
    sequence = decode_sequence(blended_tensor.squeeze().cpu())
    decoding_time = time.time() - decoding_start_time
    
    total_generation_time = time.time() - generation_start_time
    memory_end = get_memory_usage()
    
    timing_stats = {
        'total_generation_time': total_generation_time,
        'diffusion_time': diffusion_time,
        'blending_time': blending_time,
        'enforcement_time': enforcement_time,
        'decoding_time': decoding_time,
        'memory_start': memory_start,
        'memory_end': memory_end,
        'memory_peak': memory_end['allocated'] - memory_start['allocated']
    }
    
    return sequence, timing_stats

def analyze_generated_sequences(sequences, seq_type):
    """Comprehensive analysis of generated sequences"""
    analysis_start_time = time.time()
    
    # Basic statistics
    total_sequences = len(sequences)
    valid_sequences = 0
    gc_contents = []
    
    # Motif analysis
    motif_counts = {'GT': 0, 'AG': 0}
    splice_site_correct = 0
    
    # Nucleotide frequency analysis
    nt_counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    
    for seq in sequences:
        if len(seq) == 402:
            valid_sequences += 1
            
            # GC content
            gc_contents.append(gc_fraction(seq) * 100)
            
            # Splice site accuracy
            if seq_type == "donor" and seq[200:202] == 'GT':
                splice_site_correct += 1
            elif seq_type == "acceptor" and seq[200:202] == 'AG':
                splice_site_correct += 1
            
            # Count motifs around splice site
            region = seq[190:210]  # 20bp around splice site
            motif_counts['GT'] += region.count('GT')
            motif_counts['AG'] += region.count('AG')
            
            # Nucleotide frequency
            for nt in seq:
                if nt in nt_counts:
                    nt_counts[nt] += 1
    
    # Calculate frequencies
    total_nucleotides = sum(nt_counts.values())
    nt_frequencies = {nt: count/total_nucleotides for nt, count in nt_counts.items()}
    
    analysis_time = time.time() - analysis_start_time
    
    return {
        'total_sequences': total_sequences,
        'valid_sequences': valid_sequences,
        'validity_rate': valid_sequences / total_sequences,
        'gc_content_mean': np.mean(gc_contents) if gc_contents else 0,
        'gc_content_std': np.std(gc_contents) if gc_contents else 0,
        'splice_site_accuracy': splice_site_correct / valid_sequences if valid_sequences > 0 else 0,
        'motif_conservation': motif_counts,
        'nucleotide_frequencies': nt_frequencies,
        'analysis_time': analysis_time
    }

def comprehensive_lambda_sensitivity_analysis(species, seq_type, training_size):
    """Comprehensive λ sensitivity analysis with detailed metrics"""
    print(f"Starting comprehensive λ sensitivity analysis for {species} {seq_type}...")
    
    results = {}
    
    # Set file paths
    if species == "arabidopsis":
        train_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/arabidopsis_{seq_type}_positive.txt"
        actual_train_size = 50000
    else:
        train_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/homo_{seq_type}_positive.txt"
        actual_train_size = 100000
    
    print(f"Using {actual_train_size} training sequences for {species}")
    
    # Load training data once
    print("Loading training data...")
    data_load_start = time.time()
    training_data = load_and_encode_sequences(train_file, actual_train_size, seq_type)
    data_load_time = time.time() - data_load_start
    
    # Train model once
    print("Training model...")
    training_start_time = time.time()
    model, full_prev_dist, full_next_dist = train_model_with_timing(training_data, seq_type, train_file)
    training_time = time.time() - training_start_time
    
    # Test each λ value
    for lambda_val in LAMBDA_VALUES:
        print(f"\nTesting λ = {lambda_val}...")
        
        # Generate sequences with this λ (60,000 for each λ)
        sequences = []
        generation_times = []
        timing_details = []
        
        num_sequences = 60000  # Generate 60,000 sequences for each λ value
        for i in range(num_sequences):
            seq, timing_stats = generate_full_sequence_enhanced(
                full_prev_dist, full_next_dist, model, seq_type, blend_weight=lambda_val
            )
            sequences.append(seq)
            generation_times.append(timing_stats['total_generation_time'])
            timing_details.append(timing_stats)
            
            if (i + 1) % 5000 == 0:
                print(f"Generated {i + 1}/{num_sequences} sequences")
        
        # Analyze sequences
        analysis_results = analyze_generated_sequences(sequences, seq_type)
        
        # Aggregate timing statistics
        timing_aggregate = {
            'avg_total_time': np.mean(generation_times),
            'std_total_time': np.std(generation_times),
            'avg_diffusion_time': np.mean([t['diffusion_time'] for t in timing_details]),
            'avg_blending_time': np.mean([t['blending_time'] for t in timing_details]),
            'avg_enforcement_time': np.mean([t['enforcement_time'] for t in timing_details]),
            'avg_decoding_time': np.mean([t['decoding_time'] for t in timing_details]),
            'avg_memory_peak': np.mean([t['memory_peak'] for t in timing_details])
        }
        
        # Store results
        results[lambda_val] = {
            'lambda': lambda_val,
            'sequences': sequences,
            'biological_analysis': analysis_results,
            'timing_analysis': timing_aggregate,
            'num_sequences': num_sequences
        }
        
        # Save sequences for this λ
        output_file = f"Lambda_sensitivity_analysis/{species}_{seq_type}_lambda_{lambda_val}_sequences.txt"
        with open(output_file, 'w') as f:
            for seq in sequences:
                f.write(seq + '\n')
        print(f"Saved {len(sequences)} sequences to {output_file}")
    
    # Compile comprehensive results
    comprehensive_results = {
        'species': species,
        'seq_type': seq_type,
        'training_size': actual_train_size,
        'data_load_time': data_load_time,
        'model_training_time': training_time,
        'lambda_analysis': {}
    }
    
    for lambda_val, data in results.items():
        comprehensive_results['lambda_analysis'][str(lambda_val)] = {
            'lambda': lambda_val,
            'biological_quality': {
                'gc_content_mean': data['biological_analysis']['gc_content_mean'],
                'gc_content_std': data['biological_analysis']['gc_content_std'],
                'splice_site_accuracy': data['biological_analysis']['splice_site_accuracy'],
                'validity_rate': data['biological_analysis']['validity_rate'],
                'nucleotide_frequencies': data['biological_analysis']['nucleotide_frequencies']
            },
            'computational_performance': data['timing_analysis'],
            'quality_vs_speed_ratio': data['biological_analysis']['splice_site_accuracy'] / data['timing_analysis']['avg_total_time']
        }
    
    # Save comprehensive results
    with open(f"Lambda_sensitivity_analysis/{species}_{seq_type}_comprehensive_lambda_analysis.json", 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"Comprehensive λ sensitivity analysis complete. Results saved to Lambda_sensitivity_analysis/")
    return comprehensive_results

def train_model_with_timing(training_data, seq_type, train_file):
    """Train model with detailed timing and loss tracking"""
    print("Training diffusion model with timing analysis...")
    
    # Model initialization timing
    init_start = time.time()
    model = UNet(n_channels=4, n_classes=4, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    init_time = time.time() - init_start
    
    # Frequency computation timing
    freq_start = time.time()
    full_prev_dist, full_next_dist = compute_conditional_frequency_tables_region(
        train_file, 0, 402, len(training_data), seq_type
    )
    freq_time = time.time() - freq_start
    
    # Training loop timing and loss tracking
    epochs = 50
    batch_size = 64
    num_samples = training_data.size(0)
    
    epoch_times = []
    memory_usage = []
    
    # Loss tracking
    loss_history = {
        'epoch': [],
        'loss': [],
        'epoch_time': []
    }
    
    training_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        batch_count = 0
        
        for i in range(0, num_samples, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_seqs = training_data[indices]
            t = torch.randint(0, timesteps, (batch_seqs.size(0),), device=device)
            sqrt_alpha_cumprod_t = extract(sqrt_alphas_cumprod, t, batch_seqs.shape)
            sqrt_one_minus_alpha_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, batch_seqs.shape)
            noise = torch.randn_like(batch_seqs)
            x_noisy = sqrt_alpha_cumprod_t * batch_seqs + sqrt_one_minus_alpha_cumprod_t * noise

            predicted_noise = model(x_noisy, t)
            loss = loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        memory_usage.append(get_memory_usage())
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / batch_count
        
        # Store loss data
        loss_history['epoch'].append(epoch)
        loss_history['loss'].append(avg_epoch_loss)
        loss_history['epoch_time'].append(epoch_time)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s')
    
    total_training_time = time.time() - training_start
    
    # Save timing and loss analysis
    timing_results = {
        'model_initialization_time': init_time,
        'frequency_computation_time': freq_time,
        'total_training_time': total_training_time,
        'average_epoch_time': np.mean(epoch_times),
        'epoch_times': epoch_times,
        'memory_usage_per_epoch': memory_usage,
        'loss_history': loss_history
    }
    
    # Save loss data separately
    os.makedirs('Enhanced_Diffusion_models', exist_ok=True)
    loss_file = f"Enhanced_Diffusion_models/enhanced_diffusion_{seq_type}_losses.json"
    with open(loss_file, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"Enhanced diffusion loss data saved to {loss_file}")
    
    with open(f"Computational_analysis/training_timing_{seq_type}.json", 'w') as f:
        json.dump(timing_results, f, indent=2, default=str)
    
    return model, full_prev_dist, full_next_dist

def main():
    parser = argparse.ArgumentParser(description='Enhanced diffusion model with comprehensive λ sensitivity analysis')
    parser.add_argument('--mode', choices=['sensitivity', 'benchmark', 'single'], required=True,
                       help='Mode: sensitivity (λ analysis), benchmark (comparison), single (single λ test)')
    parser.add_argument('--species', choices=['homo'], required=True) # ['arabidopsis', 'homo']
    parser.add_argument('--seq_type', choices=['acceptor'], required=True) # ['donor', 'acceptor']
    parser.add_argument('--train_size', type=int, default=50, help='Training size in thousands')
    parser.add_argument('--lambda_val', type=float, default=0.5, help='Lambda value for single mode')
    
    args = parser.parse_args()
    
    if args.mode == 'sensitivity':
        # Comprehensive λ sensitivity analysis
        results = comprehensive_lambda_sensitivity_analysis(args.species, args.seq_type, args.train_size)
        
        # Print summary
        print("\n" + "="*60)
        print("COMPREHENSIVE λ SENSITIVITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Species: {args.species}")
        print(f"Sequence Type: {args.seq_type}")
        print(f"Training Size: {args.train_size}k")
        print("\nλ Value Performance Summary:")
        print("-" * 40)
        
        for lambda_str, data in results['lambda_analysis'].items():
            lambda_val = float(lambda_str)
            bio_qual = data['biological_quality']
            comp_perf = data['computational_performance']
            
            print(f"λ = {lambda_val}:")
            print(f"  Splice Site Accuracy: {bio_qual['splice_site_accuracy']:.3f}")
            print(f"  GC Content Mean: {bio_qual['gc_content_mean']:.2f}%")
            print(f"  Avg Generation Time: {comp_perf['avg_total_time']:.4f}s")
            print(f"  Quality/Speed Ratio: {data['quality_vs_speed_ratio']:.2f}")
            print()
        
        # Find optimal λ
        best_lambda = max(results['lambda_analysis'].items(), 
                         key=lambda x: x[1]['quality_vs_speed_ratio'])
        print(f"Recommended λ: {best_lambda[0]} (Quality/Speed Ratio: {best_lambda[1]['quality_vs_speed_ratio']:.2f})")
        print("="*60)
    
    elif args.mode == 'single':
        print(f"Testing single λ value: {args.lambda_val}")
        # Implementation for single λ testing would go here
        
    elif args.mode == 'benchmark':
        print("Comprehensive benchmarking mode")
        # Implementation for benchmarking would go here

if __name__ == "__main__":
    main() 