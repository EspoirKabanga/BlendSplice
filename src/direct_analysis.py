#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import random
from collections import Counter
from math import log, ceil
import seaborn as sns
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

def gc_fraction(seq):
    """Calculate GC content without BioPython dependency"""
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) if len(seq) > 0 else 0

# Create output directory
os.makedirs("icml_analysis", exist_ok=True)

# Set consistent font sizes
SMALL_SIZE = 16
MEDIUM_SIZE = 18
LARGE_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize

# Set line styles for better visualization
plt.rc('lines', linewidth=1.2)            # thinner lines
plt.rc('lines', markersize=6)             # smaller markers

def load_sequences(file_path, sample_size=None):
    """Load DNA sequences from a file and optionally sample a subset
    
    Args:
        file_path: Path to the sequence file
        sample_size: If provided, randomly select this many sequences
    """
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            seq = line.strip().upper()
            if seq:
                sequences.append(seq)
    
    if sample_size and sample_size < len(sequences):
        return random.sample(sequences, sample_size)
    return sequences

def enforce_center_motif(seq, seq_type, seq_len):
    """Ensure canonical splice-site dinucleotide at the center of the sequence.

    - donor: enforce 'GT' at positions (mid-1, mid)
    - acceptor: enforce 'AG' at positions (mid-1, mid)
    """
    if not seq:
        return seq
    mid = seq_len // 2
    s = list(seq)
    if seq_type == 'donor':
        if mid - 1 >= 0 and mid < seq_len:
            s[mid - 1] = 'G'
            s[mid] = 'T'
    else:
        if mid - 1 >= 0 and mid < seq_len:
            s[mid - 1] = 'A'
            s[mid] = 'G'
    return ''.join(s)

def maybe_enforce_splice_site(sequences, seq_type, enforce=False):
    if not enforce or not sequences:
        return sequences
    seq_len = len(sequences[0])
    return [enforce_center_motif(s, seq_type, seq_len) for s in sequences]

def compute_pwm(sequences):
    """Compute position weight matrix from sequences"""
    # Assuming all sequences have the same length
    seq_len = len(sequences[0])
    counts_matrix = np.zeros((seq_len, 4))  # 4 for A, C, G, T
    
    # Define nucleotide to index mapping
    nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Count each nucleotide at each position
    for seq in sequences:
        for i, nt in enumerate(seq):
            if nt in nt_to_idx:
                counts_matrix[i, nt_to_idx[nt]] += 1
    
    # Convert counts to frequencies
    pwm = counts_matrix / counts_matrix.sum(axis=1, keepdims=True)
    
    # Create a DataFrame with the PWM
    pwm_df = pd.DataFrame(pwm, columns=['A', 'C', 'G', 'T'])
    
    return pwm_df

def create_logo(pwm_df, ax, region=None):
    """Create sequence logo on the given axis"""
    # Create Logo object
    logo = logomaker.Logo(pwm_df, ax=ax)
    
    # Style the logo
    logo.style_spines(visible=False)
    logo.style_xticks(anchor=0)
    
    # Set y-axis ticks to only show 0.0, 0.5, and 1.0
    ax.set_yticks([0.0, 0.5, 1.0])
    
    # Add axis labels with larger font size
    ax.set_xlabel('Position', fontsize=LARGE_SIZE)
    ax.set_ylabel('Bits', fontsize=LARGE_SIZE)
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=LARGE_SIZE)
    
    # Limit the region if specified
    if region:
        start, end = region
        ax.set_xlim(start, end)
        
        # Show only every 5th tick on x-axis
        x_ticks = list(range(start, end+1, 5))
        ax.set_xticks(x_ticks)
    
    # Remove grid
    ax.grid(False)
    
    return logo

def calculate_position_conservation(sequences):
    """Calculate conservation scores for each position in a set of sequences"""
    seq_len = len(sequences[0])
    conservation = np.zeros(seq_len)
    
    # For each position, calculate the frequency of the most common nucleotide
    for pos in range(seq_len):
        nucleotides = [seq[pos] for seq in sequences if len(seq) > pos]
        counter = Counter(nucleotides)
        most_common = counter.most_common(1)
        if most_common:
            total = len(nucleotides)
            if total > 0:
                conservation[pos] = most_common[0][1] / total
    
    return conservation

def analyze_nucleotide_conservation(real_seqs, blend_seqs, no_blend_seqs, output_prefix):
    """Analyze nucleotide conservation patterns in sequences"""
    print(f"Analyzing nucleotide conservation for {output_prefix}...")
    
    # Compute PWMs for each dataset
    real_pwm = compute_pwm(real_seqs)
    blend_pwm = compute_pwm(blend_seqs)
    no_blend_pwm = compute_pwm(no_blend_seqs)
    
    # Calculate conservation scores for each dataset
    real_cons = calculate_position_conservation(real_seqs)
    blend_cons = calculate_position_conservation(blend_seqs)
    no_blend_cons = calculate_position_conservation(no_blend_seqs)
    
    # Create DataFrame for conservation scores
    cons_df = pd.DataFrame({
        'Position': range(len(real_cons)),
        'Real': real_cons,
        'Blend': blend_cons,
        'No_Blend': no_blend_cons
    })
    
    # Save to CSV
    cons_df.to_csv(f"{output_prefix}_conservation.csv", index=False)
    
    # Return the computed data for combined plots
    return {
        'pwm': {
            'real': real_pwm,
            'blend': blend_pwm,
            'no_blend': no_blend_pwm
        },
        'conservation': {
            'real': real_cons,
            'blend': blend_cons,
            'no_blend': no_blend_cons
        }
    }

def analyze_gc_content(real_seqs, blend_seqs, no_blend_seqs, output_prefix):
    """Analyze GC content distribution in sequences"""
    print(f"Analyzing GC content for {output_prefix}...")
    
    # Calculate GC content for each sequence in each dataset
    real_gc = [gc_fraction(seq) * 100 for seq in real_seqs]
    blend_gc = [gc_fraction(seq) * 100 for seq in blend_seqs]
    no_blend_gc = [gc_fraction(seq) * 100 for seq in no_blend_seqs]
    
    # Calculate statistics
    real_mean = np.mean(real_gc)
    blend_mean = np.mean(blend_gc)
    no_blend_mean = np.mean(no_blend_gc)
    
    # Create DataFrame for results
    gc_df = pd.DataFrame({
        'GC_Content': real_gc + blend_gc + no_blend_gc,
        'Dataset': ['Real'] * len(real_gc) + ['Blend'] * len(blend_gc) + ['No Blend'] * len(no_blend_gc)
    })
    
    # Save to CSV
    gc_df.to_csv(f"{output_prefix}_gc_content.csv", index=False)
    
    # Return the computed data for combined plots
    return {
        'gc_content': {
            'real': real_gc,
            'blend': blend_gc,
            'no_blend': no_blend_gc
        },
        'means': {
            'real': real_mean,
            'blend': blend_mean,
            'no_blend': no_blend_mean
        }
    }

def analyze_3mers(real_seqs, blend_seqs, no_blend_seqs, output_prefix):
    """Analyze 3-mer frequencies around splice sites"""
    print(f"Analyzing 3-mers for {output_prefix}...")
    
    splice_pos = 200
    window = 10  # Analyze 10 positions before and after splice site
    
    # Extract regions around splice site
    real_regions = [seq[splice_pos-window:splice_pos+window] for seq in real_seqs]
    blend_regions = [seq[splice_pos-window:splice_pos+window] for seq in blend_seqs]
    no_blend_regions = [seq[splice_pos-window:splice_pos+window] for seq in no_blend_seqs]
    
    # Count 3-mers in each position
    positions = range(len(real_regions[0]) - 2)
    
    # Initialize dictionaries to store 3-mer counts for each position
    real_3mers = {pos: Counter() for pos in positions}
    blend_3mers = {pos: Counter() for pos in positions}
    no_blend_3mers = {pos: Counter() for pos in positions}
    
    # Count 3-mers at each position
    for seq in real_regions:
        for pos in positions:
            if len(seq) >= pos + 3:
                real_3mers[pos][seq[pos:pos+3]] += 1
    
    for seq in blend_regions:
        for pos in positions:
            if len(seq) >= pos + 3:
                blend_3mers[pos][seq[pos:pos+3]] += 1
    
    for seq in no_blend_regions:
        for pos in positions:
            if len(seq) >= pos + 3:
                no_blend_3mers[pos][seq[pos:pos+3]] += 1
    
    # Find the most common 3-mers at each position
    top_n = 5  # Number of top 3-mers to analyze
    
    # Create DataFrames to store results
    results = []
    
    for pos in positions:
        real_top = real_3mers[pos].most_common(top_n)
        blend_top = blend_3mers[pos].most_common(top_n)
        no_blend_top = no_blend_3mers[pos].most_common(top_n)
        
        # Calculate total counts
        real_total = sum(real_3mers[pos].values())
        blend_total = sum(blend_3mers[pos].values())
        no_blend_total = sum(no_blend_3mers[pos].values())
        
        # Add to results
        for i, (kmer, count) in enumerate(real_top):
            results.append({
                'Position': pos + splice_pos - window,
                'Dataset': 'Real',
                'Rank': i + 1,
                '3-mer': kmer,
                'Count': count,
                'Frequency': count / real_total if real_total > 0 else 0
            })
        
        for i, (kmer, count) in enumerate(blend_top):
            results.append({
                'Position': pos + splice_pos - window,
                'Dataset': 'Blend',
                'Rank': i + 1,
                '3-mer': kmer,
                'Count': count,
                'Frequency': count / blend_total if blend_total > 0 else 0
            })
        
        for i, (kmer, count) in enumerate(no_blend_top):
            results.append({
                'Position': pos + splice_pos - window,
                'Dataset': 'No-Blend',
                'Rank': i + 1,
                '3-mer': kmer,
                'Count': count,
                'Frequency': count / no_blend_total if no_blend_total > 0 else 0
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(f"{output_prefix}_3mer_analysis.csv", index=False)
    
    # Get species from output_prefix
    species = output_prefix.split('/')[-1].split('_')[0]
    seq_type = output_prefix.split('/')[-1].split('_')[1]
    
    # Return the results DataFrame for combined plots
    return {
        'results_df': results_df,
        'species': species,
        'seq_type': seq_type
    }

def create_combined_sequence_logos(species, donor_data, acceptor_data, model=None):
    """Create combined sequence logos for donor and acceptor"""
    print(f"Creating combined sequence logos for {species}{' (' + model + ')' if model else ''}...")

    try:
        # Create a figure with 2 columns (donor and acceptor) and 3 rows (real, blend, no-blend)
        fig, axes = plt.subplots(3, 2, figsize=(20, 6))

        # Define the splice site region
        splice_region = (180, 220)

        # Dataset labels
        dataset_labels = ['Real', 'Blend', 'No-Blend']
        species_label = "Human" if species.lower() == "homo" else species.title()
        site_labels = [f"{species_label} Donor", f"{species_label} Acceptor"]

        # Generate logos for donor (left column)
        for i in range(3):
            if i == 0:
                pwm = donor_data['pwm']['real']
            elif i == 1:
                pwm = donor_data['pwm']['blend']
            else:
                pwm = donor_data['pwm']['no_blend']

            create_logo(pwm, axes[i, 0], region=splice_region)

            # Keep axis labels visible on all subplots

        # Generate logos for acceptor (right column)
        for i in range(3):
            if i == 0:
                pwm = acceptor_data['pwm']['real']
            elif i == 1:
                pwm = acceptor_data['pwm']['blend']
            else:
                pwm = acceptor_data['pwm']['no_blend']

            create_logo(pwm, axes[i, 1], region=splice_region)

            # Keep axis labels visible on all subplots

        # Add dataset labels to the right side of each subplot
        for i in range(3):
            axes[i, 1].text(
                1.01,
                0.5,
                dataset_labels[i],
                transform=axes[i, 1].transAxes,
                fontsize=LARGE_SIZE,
                va='center',
                ha='left',
                fontstyle='italic',
            )

        # Add column titles
        for j in range(2):
            axes[0, j].set_title(site_labels[j], fontsize=LARGE_SIZE)

        plt.tight_layout()
        output_path = (
            f"icml_analysis/{model}/{species}_sequence_logos.png" if model else f"icml_analysis/{species}_sequence_logos.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved sequence logos to {output_path}")

    except Exception as e:
        print(f"❌ Error creating sequence logos for {species}: {e}")
        import traceback
        traceback.print_exc()

def create_combined_conservation_plot(arabidopsis_donor_data, arabidopsis_acceptor_data, human_donor_data, human_acceptor_data, model=None):
    """Create a single figure with conservation scores for both arabidopsis and human"""
    print("Creating combined conservation plot for arabidopsis and human...")

    try:
        # Create a figure with 2 columns (arabidopsis and human) and 2 rows (donor and acceptor)
        # Reduced height from 10 to 8
        fig, axes = plt.subplots(2, 2, figsize=(20, 8))

        # Species and site labels
        species_labels = ['Arabidopsis', 'Human']
        site_labels = ['Donor', 'Acceptor']

        # Define position range to display
        pos_start = 150
        pos_end = 250

        # Plot arabidopsis donor (top left)
        positions = np.arange(len(arabidopsis_donor_data['conservation']['real']))
        axes[0, 0].plot(positions, arabidopsis_donor_data['conservation']['real'], label='Real', color='blue', linewidth=1.2)
        axes[0, 0].plot(positions, arabidopsis_donor_data['conservation']['blend'], label='Blend', color='orange', linewidth=1.2)
        axes[0, 0].plot(positions, arabidopsis_donor_data['conservation']['no_blend'], label='No-Blend', color='green', linewidth=1.2)
        axes[0, 0].set_title(f"{species_labels[0]} {site_labels[0]}", fontsize=LARGE_SIZE)
        axes[0, 0].set_ylabel('Conservation Score', fontsize=LARGE_SIZE)
        axes[0, 0].set_xlabel('Position', fontsize=LARGE_SIZE)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_yticks([0.0, 0.5, 1.0])
        axes[0, 0].set_yticklabels(['0.0', '0.5', '1.0'])
        axes[0, 0].legend(fontsize=LARGE_SIZE)
        leg = axes[0, 0].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[0, 0].set_xlim(pos_start, pos_end)
        axes[0, 0].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        # Plot human donor (top right)
        positions = np.arange(len(human_donor_data['conservation']['real']))
        axes[0, 1].plot(positions, human_donor_data['conservation']['real'], label='Real', color='blue', linewidth=1.2)
        axes[0, 1].plot(positions, human_donor_data['conservation']['blend'], label='Blend', color='orange', linewidth=1.2)
        axes[0, 1].plot(positions, human_donor_data['conservation']['no_blend'], label='No-Blend', color='green', linewidth=1.2)
        axes[0, 1].set_title(f"{species_labels[1]} {site_labels[0]}", fontsize=LARGE_SIZE)
        axes[0, 1].set_ylabel('Conservation Score', fontsize=LARGE_SIZE)
        axes[0, 1].set_xlabel('Position', fontsize=LARGE_SIZE)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_yticks([0.0, 0.5, 1.0])
        axes[0, 1].set_yticklabels(['0.0', '0.5', '1.0'])
        axes[0, 1].legend(fontsize=LARGE_SIZE)
        leg = axes[0, 1].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[0, 1].set_xlim(pos_start, pos_end)
        axes[0, 1].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        # Plot arabidopsis acceptor (bottom left)
        positions = np.arange(len(arabidopsis_acceptor_data['conservation']['real']))
        axes[1, 0].plot(positions, arabidopsis_acceptor_data['conservation']['real'], label='Real', color='blue', linewidth=1.2)
        axes[1, 0].plot(positions, arabidopsis_acceptor_data['conservation']['blend'], label='Blend', color='orange', linewidth=1.2)
        axes[1, 0].plot(positions, arabidopsis_acceptor_data['conservation']['no_blend'], label='No-Blend', color='green', linewidth=1.2)
        axes[1, 0].set_title(f"{species_labels[0]} {site_labels[1]}", fontsize=LARGE_SIZE)
        axes[1, 0].set_xlabel('Position', fontsize=LARGE_SIZE)
        axes[1, 0].set_ylabel('Conservation Score', fontsize=LARGE_SIZE)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_yticks([0.0, 0.5, 1.0])
        axes[1, 0].set_yticklabels(['0.0', '0.5', '1.0'])
        axes[1, 0].legend(fontsize=LARGE_SIZE)
        leg = axes[1, 0].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[1, 0].set_xlim(pos_start, pos_end)
        axes[1, 0].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        # Plot human acceptor (bottom right)
        positions = np.arange(len(human_acceptor_data['conservation']['real']))
        axes[1, 1].plot(positions, human_acceptor_data['conservation']['real'], label='Real', color='blue', linewidth=1.2)
        axes[1, 1].plot(positions, human_acceptor_data['conservation']['blend'], label='Blend', color='orange', linewidth=1.2)
        axes[1, 1].plot(positions, human_acceptor_data['conservation']['no_blend'], label='No-Blend', color='green', linewidth=1.2)
        axes[1, 1].set_title(f"{species_labels[1]} {site_labels[1]}", fontsize=LARGE_SIZE)
        axes[1, 1].set_xlabel('Position', fontsize=LARGE_SIZE)
        axes[1, 1].set_ylabel('Conservation Score', fontsize=LARGE_SIZE)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_yticks([0.0, 0.5, 1.0])
        axes[1, 1].set_yticklabels(['0.0', '0.5', '1.0'])
        axes[1, 1].legend(fontsize=LARGE_SIZE)
        leg = axes[1, 1].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[1, 1].set_xlim(pos_start, pos_end)
        axes[1, 1].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        plt.tight_layout()
        output_path = (
            "icml_analysis/combined_conservation.png"
            if not model
            else f"icml_analysis/{model}/combined_conservation.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved conservation plot to {output_path}")

    except Exception as e:
        print(f"❌ Error creating conservation plot: {e}")
        import traceback
        traceback.print_exc()

def create_combined_gc_content_plot(arabidopsis_donor_data, arabidopsis_acceptor_data, human_donor_data, human_acceptor_data, model=None):
    """Create a single figure with GC content histograms for both arabidopsis and human"""
    print("Creating combined GC content plot for arabidopsis and human...")

    try:
        # Determine global GC-content range for consistent x-axis
        all_gc_values = []
        for dataset in [
            arabidopsis_donor_data,
            arabidopsis_acceptor_data,
            human_donor_data,
            human_acceptor_data,
        ]:
            for seq_set in dataset['gc_content'].values():
                all_gc_values.extend(seq_set)
        if all_gc_values:
            gc_min = min(all_gc_values)
            gc_max = max(all_gc_values)
        else:
            gc_min, gc_max = 0, 100

        # Create a figure with 2 columns (arabidopsis and human) and 2 rows (donor and acceptor)
        # Reduced height from 10 to 8
        fig, axes = plt.subplots(2, 2, figsize=(20, 8))

        # Species and site labels
        species_labels = ['Arabidopsis', 'Human']
        site_labels = ['Donor', 'Acceptor']

        # Plot arabidopsis donor (top left)
        real_gc = arabidopsis_donor_data['gc_content']['real']
        blend_gc = arabidopsis_donor_data['gc_content']['blend']
        no_blend_gc = arabidopsis_donor_data['gc_content']['no_blend']

        gc_df = pd.DataFrame({
            'GC_Content': real_gc + blend_gc + no_blend_gc,
            'Dataset': ['Real'] * len(real_gc) + ['Blend'] * len(blend_gc) + ['No-Blend'] * len(no_blend_gc)
        })

        sns.histplot(data=gc_df, x='GC_Content', hue='Dataset', kde=True, bins=30, alpha=0.6, ax=axes[0, 0], legend=False)
        real_mean = arabidopsis_donor_data['means']['real']
        blend_mean = arabidopsis_donor_data['means']['blend']
        no_blend_mean = arabidopsis_donor_data['means']['no_blend']
        axes[0, 0].axvline(real_mean, color='blue', linestyle='--', label=r'$\it{Real}$ mean')
        axes[0, 0].axvline(blend_mean, color='orange', linestyle='--', label=r'$\it{Blend}$ mean')
        axes[0, 0].axvline(no_blend_mean, color='green', linestyle='--', label=r'$\it{No-Blend}$ mean')
        axes[0, 0].set_title(f"{species_labels[0]} {site_labels[0]}", fontsize=LARGE_SIZE)
        axes[0, 0].set_xlabel('GC Content (%)', fontsize=LARGE_SIZE)
        axes[0, 0].set_ylabel('Count', fontsize=LARGE_SIZE)
        axes[0, 0].set_yticks([0, 10000, 20000])
        axes[0, 0].set_yticklabels(['0', '10,000', '20,000'])
        axes[0, 0].set_ylim(0, 20000)
        axes[0, 0].set_xlim(gc_min, gc_max)
        axes[0, 0].legend(fontsize=LARGE_SIZE)
        leg = axes[0, 0].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[0, 0].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        # Plot human donor (top right)
        real_gc = human_donor_data['gc_content']['real']
        blend_gc = human_donor_data['gc_content']['blend']
        no_blend_gc = human_donor_data['gc_content']['no_blend']

        gc_df = pd.DataFrame({
            'GC_Content': real_gc + blend_gc + no_blend_gc,
            'Dataset': ['Real'] * len(real_gc) + ['Blend'] * len(blend_gc) + ['No-Blend'] * len(no_blend_gc)
        })

        sns.histplot(data=gc_df, x='GC_Content', hue='Dataset', kde=True, bins=30, alpha=0.6, ax=axes[0, 1], legend=False)
        real_mean = human_donor_data['means']['real']
        blend_mean = human_donor_data['means']['blend']
        no_blend_mean = human_donor_data['means']['no_blend']
        axes[0, 1].axvline(real_mean, color='blue', linestyle='--', label=r'$\it{Real}$ mean')
        axes[0, 1].axvline(blend_mean, color='orange', linestyle='--', label=r'$\it{Blend}$ mean')
        axes[0, 1].axvline(no_blend_mean, color='green', linestyle='--', label=r'$\it{No-Blend}$ mean')
        axes[0, 1].set_title(f"{species_labels[1]} {site_labels[0]}", fontsize=LARGE_SIZE)
        axes[0, 1].set_xlabel('GC Content (%)', fontsize=LARGE_SIZE)
        axes[0, 1].set_ylabel('Count', fontsize=LARGE_SIZE)
        axes[0, 1].set_yticks([0, 10000, 20000])
        axes[0, 1].set_yticklabels(['0', '10,000', '20,000'])
        axes[0, 1].set_ylim(0, 20000)
        axes[0, 1].set_xlim(gc_min, gc_max)
        axes[0, 1].legend(fontsize=LARGE_SIZE)
        leg = axes[0, 1].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[0, 1].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        # Plot arabidopsis acceptor (bottom left)
        real_gc = arabidopsis_acceptor_data['gc_content']['real']
        blend_gc = arabidopsis_acceptor_data['gc_content']['blend']
        no_blend_gc = arabidopsis_acceptor_data['gc_content']['no_blend']

        gc_df = pd.DataFrame({
            'GC_Content': real_gc + blend_gc + no_blend_gc,
            'Dataset': ['Real'] * len(real_gc) + ['Blend'] * len(blend_gc) + ['No-Blend'] * len(no_blend_gc)
        })

        sns.histplot(data=gc_df, x='GC_Content', hue='Dataset', kde=True, bins=30, alpha=0.6, ax=axes[1, 0], legend=False)
        real_mean = arabidopsis_acceptor_data['means']['real']
        blend_mean = arabidopsis_acceptor_data['means']['blend']
        no_blend_mean = arabidopsis_acceptor_data['means']['no_blend']
        axes[1, 0].axvline(real_mean, color='blue', linestyle='--', label=r'$\it{Real}$ mean')
        axes[1, 0].axvline(blend_mean, color='orange', linestyle='--', label=r'$\it{Blend}$ mean')
        axes[1, 0].axvline(no_blend_mean, color='green', linestyle='--', label=r'$\it{No-Blend}$ mean')
        axes[1, 0].set_title(f"{species_labels[0]} {site_labels[1]}", fontsize=LARGE_SIZE)
        axes[1, 0].set_xlabel('GC Content (%)', fontsize=LARGE_SIZE)
        axes[1, 0].set_ylabel('Count', fontsize=LARGE_SIZE)
        axes[1, 0].set_yticks([0, 10000, 20000])
        axes[1, 0].set_yticklabels(['0', '10,000', '20,000'])
        axes[1, 0].set_ylim(0, 20000)
        axes[1, 0].set_xlim(gc_min, gc_max)
        axes[1, 0].legend(fontsize=LARGE_SIZE)
        leg = axes[1, 0].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[1, 0].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        # Plot human acceptor (bottom right)
        real_gc = human_acceptor_data['gc_content']['real']
        blend_gc = human_acceptor_data['gc_content']['blend']
        no_blend_gc = human_acceptor_data['gc_content']['no_blend']

        gc_df = pd.DataFrame({
            'GC_Content': real_gc + blend_gc + no_blend_gc,
            'Dataset': ['Real'] * len(real_gc) + ['Blend'] * len(blend_gc) + ['No-Blend'] * len(no_blend_gc)
        })

        sns.histplot(data=gc_df, x='GC_Content', hue='Dataset', kde=True, bins=30, alpha=0.6, ax=axes[1, 1], legend=False)
        real_mean = human_acceptor_data['means']['real']
        blend_mean = human_acceptor_data['means']['blend']
        no_blend_mean = human_acceptor_data['means']['no_blend']
        axes[1, 1].axvline(real_mean, color='blue', linestyle='--', label=r'$\it{Real}$ mean')
        axes[1, 1].axvline(blend_mean, color='orange', linestyle='--', label=r'$\it{Blend}$ mean')
        axes[1, 1].axvline(no_blend_mean, color='green', linestyle='--', label=r'$\it{No-Blend}$ mean')
        axes[1, 1].set_title(f"{species_labels[1]} {site_labels[1]}", fontsize=LARGE_SIZE)
        axes[1, 1].set_xlabel('GC Content (%)', fontsize=LARGE_SIZE)
        axes[1, 1].set_ylabel('Count', fontsize=LARGE_SIZE)
        axes[1, 1].set_yticks([0, 10000, 20000])
        axes[1, 1].set_yticklabels(['0', '10,000', '20,000'])
        axes[1, 1].set_ylim(0, 20000)
        axes[1, 1].set_xlim(gc_min, gc_max)
        axes[1, 1].legend(fontsize=LARGE_SIZE)
        leg = axes[1, 1].get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        axes[1, 1].tick_params(axis='both', which='major', labelsize=LARGE_SIZE)

        plt.tight_layout()
        output_path = (
            "icml_analysis/combined_gc_content.png"
            if not model
            else f"icml_analysis/{model}/combined_gc_content.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved GC content plot to {output_path}")

    except Exception as e:
        print(f"❌ Error creating combined GC content plot: {e}")
        import traceback
        traceback.print_exc()

def create_combined_3mer_plot(arabidopsis_donor_data, arabidopsis_acceptor_data, human_donor_data, human_acceptor_data, model=None):
    """Create a single elegant figure with 3-mer frequencies for both arabidopsis and human"""
    print("Creating combined 3-mer plot for arabidopsis and human...")

    try:
        # Create a figure with 2 columns (arabidopsis and human) and 2 rows (donor and acceptor)
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))

        # Define colors for better aesthetics
        colors = ['#3274A1', '#E1812C', '#3A923A']  # Blue, Orange, Green

        # Define datasets and their positions
        datasets = [
            (arabidopsis_donor_data, 'Arabidopsis Donor', 0, 0),
            (human_donor_data, 'Human Donor', 0, 1),
            (arabidopsis_acceptor_data, 'Arabidopsis Acceptor', 1, 0),
            (human_acceptor_data, 'Human Acceptor', 1, 1),
        ]

        # Determine global max frequency for consistent y-axis scaling
        global_freq_max = 0.0
        for data_tuple in datasets:
            df = data_tuple[0]['three_mer']['results_df']
            if not df.empty and 'Frequency' in df.columns:
                filtered_df = df[df['Position'].isin([197, 202])]
                if filtered_df.empty:
                    continue
                local_max = filtered_df['Frequency'].max()
                if local_max > global_freq_max:
                    global_freq_max = local_max
        if global_freq_max > 0:
            steps = int(ceil(global_freq_max / 0.2))
            y_limit = max(0.2, steps * 0.2)
        else:
            y_limit = 0.2

        def plot_3mer_data(data, title, row, col):
            """Helper function to plot 3-mer data for a single dataset"""
            ax = axes[row, col]

            # Get the results dataframe
            results_df = data['three_mer']['results_df']

            # Get position 197 data (upstream) and 202 data (downstream)
            up_df = results_df[results_df['Position'] == 197].copy()
            down_df = results_df[results_df['Position'] == 202].copy()

            if len(up_df) == 0 or len(down_df) == 0:
                print(f"Warning: No data found for positions 197/202 in {title}")
                ax.text(
                    0.5,
                    0.5,
                    f'No data available\nfor {title}',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=LARGE_SIZE,
                )
                ax.set_title(title, fontsize=LARGE_SIZE)
                return

            # Find top 5 3-mers at each position
            top_up = up_df.groupby('3-mer')['Frequency'].mean().nlargest(5).index.tolist()
            top_down = down_df.groupby('3-mer')['Frequency'].mean().nlargest(5).index.tolist()

            # Filter to keep only top 3-mers
            up_df = up_df[up_df['3-mer'].isin(top_up)]
            down_df = down_df[down_df['3-mer'].isin(top_down)]

            # Add position label column
            up_df['Position_Label'] = 'Upstream (197)'
            down_df['Position_Label'] = 'Downstream (202)'

            # Combine data
            plot_df = pd.concat([up_df, down_df])

            if len(plot_df) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f'No data to plot\nfor {title}',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=LARGE_SIZE,
                )
                ax.set_title(title, fontsize=LARGE_SIZE)
                return

            # Create a pivot table for easier plotting
            pivot_df = (
                plot_df.pivot_table(
                    index=['3-mer', 'Position_Label'],
                    columns='Dataset',
                    values='Frequency',
                    aggfunc='mean',
                )
                .reset_index()
            )

            # Plot bars for each position separately
            positions = ['Upstream (197)', 'Downstream (202)']
            x_offset = 0
            bar_width = 0.25
            split_line_x = None

            for i, pos in enumerate(positions):
                pos_data = plot_df[plot_df['Position_Label'] == pos]

                if len(pos_data) == 0:
                    continue

                # Get unique 3-mers for this position
                unique_3mers = sorted(pos_data['3-mer'].unique())

                for j, dataset in enumerate(['Real', 'Blend', 'No-Blend']):
                    dataset_data = pos_data[pos_data['Dataset'] == dataset]

                    if len(dataset_data) == 0:
                        # still advance positions for consistent spacing even if empty
                        continue

                    # Create x positions
                    x_positions = [x_offset + k + j * bar_width for k in range(len(unique_3mers))]

                    # Get frequencies in the same order as unique_3mers
                    frequencies = []
                    for kmer in unique_3mers:
                        kmer_data = dataset_data[dataset_data['3-mer'] == kmer]
                        if len(kmer_data) > 0:
                            frequencies.append(kmer_data['Frequency'].iloc[0])
                        else:
                            frequencies.append(0)

                    ax.bar(
                        x_positions,
                        frequencies,
                        bar_width,
                        label='',
                        color=colors[j],
                        alpha=0.8,
                    )

                # Update x_offset for next position
                if i == 0:
                    # store split position after upstream group for separator line
                    split_line_x = x_offset + len(unique_3mers) + 1 - 0.5
                x_offset += len(unique_3mers) + 1

            # Set labels and title
            ax.set_title(title, fontsize=LARGE_SIZE)
            ax.set_ylabel('Frequency', fontsize=MEDIUM_SIZE)
            ax.set_xlabel('3-mer', fontsize=MEDIUM_SIZE)

            # Set x-tick labels
            all_x_positions = []
            all_labels = []
            x_offset = 0

            for pos in positions:
                pos_data = plot_df[plot_df['Position_Label'] == pos]
                unique_3mers = sorted(pos_data['3-mer'].unique())

                if len(unique_3mers) > 0:
                    center_positions = [x_offset + k + bar_width for k in range(len(unique_3mers))]
                    all_x_positions.extend(center_positions)
                    all_labels.extend(unique_3mers)
                    x_offset += len(unique_3mers) + 1

            if all_x_positions:
                ax.set_xticks(all_x_positions)
                ax.set_xticklabels(all_labels, rotation=45, ha='right')

            # Add vertical dotted separator between upstream and downstream groups
            if split_line_x is not None:
                ax.axvline(split_line_x, linestyle=':', color='gray', linewidth=1)

            # Add legend on every subplot
            # Always include all three entries using proxy patches to ensure presence
            proxies = [
                mpl.patches.Patch(color=colors[0], label='Real'),
                mpl.patches.Patch(color=colors[1], label='Blend'),
                mpl.patches.Patch(color=colors[2], label='No-Blend'),
            ]
            ax.legend(handles=proxies, fontsize=MEDIUM_SIZE)
            leg = ax.get_legend()
            if leg is not None:
                for text in leg.get_texts():
                    if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                        text.set_fontstyle('italic')

            ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
            ax.set_ylim(0, y_limit)
            yticks = np.arange(0, y_limit + 1e-6, 0.2)
            ax.set_yticks(yticks)

        # Plot all four datasets
        for data, title, row, col in datasets:
            plot_3mer_data(data, title, row, col)

        # Main title for the entire figure
        #fig.suptitle(
        #    '3-mer Analysis at Positions 197 (Upstream) and 202 (Downstream)',
        #    fontsize=LARGE_SIZE,
        #    y=0.95,
        #)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        output_path = (
            "icml_analysis/combined_3mer_analysis.png"
            if not model
            else f"icml_analysis/{model}/combined_3mer_analysis.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved 3-mer plot to {output_path}")

    except Exception as e:
        print(f"❌ Error creating combined 3-mer plot: {e}")
        import traceback
        traceback.print_exc()

def create_individual_conservation_plot(species, seq_type, data, model=None):
    """Create conservation plot for a single dataset"""
    print(f"Creating conservation plot for {species} {seq_type}...")
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        positions = np.arange(len(data['conservation']['real']))
        ax.plot(positions, data['conservation']['real'], label='Real', color='blue', linewidth=1.2)
        ax.plot(positions, data['conservation']['blend'], label='Blend (λ=0.5)', color='orange', linewidth=1.2)
        ax.plot(positions, data['conservation']['no_blend'], label='No-Blend (λ=0.0)', color='green', linewidth=1.2)
        
        ax.set_title(f'{species.title()} {seq_type.title()} - Nucleotide Conservation', fontsize=LARGE_SIZE)
        ax.set_xlabel('Position', fontsize=LARGE_SIZE)
        ax.set_ylabel('Conservation Score', fontsize=LARGE_SIZE)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.legend(fontsize=MEDIUM_SIZE)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        ax.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
        
        # Focus on splice site region
        ax.set_xlim(150, 250)
        
        plt.tight_layout()
        output_path = (
            f"icml_analysis/{model}/{species}_{seq_type}_conservation.png"
            if model else f"icml_analysis/{species}_{seq_type}_conservation.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved conservation plot to {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating conservation plot for {species} {seq_type}: {e}")
        import traceback
        traceback.print_exc()

def create_individual_gc_plot(species, seq_type, data, model=None):
    """Create GC content plot for a single dataset"""
    print(f"Creating GC content plot for {species} {seq_type}...")
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        real_gc = data['gc_content']['real']
        blend_gc = data['gc_content']['blend']
        no_blend_gc = data['gc_content']['no_blend']
        real_mean = data['means']['real']
        blend_mean = data['means']['blend']
        no_blend_mean = data['means']['no_blend']
        
        gc_df = pd.DataFrame({
            'GC_Content': real_gc + blend_gc + no_blend_gc,
            'Dataset': ['Real'] * len(real_gc) + ['Blend (λ=0.5)'] * len(blend_gc) + ['No-Blend (λ=0.0)'] * len(no_blend_gc)
        })
        
        sns.histplot(data=gc_df, x='GC_Content', hue='Dataset', kde=True, bins=30, alpha=0.6, ax=ax, legend=False)
        ax.axvline(real_mean, color='blue', linestyle='--', label='Real')
        ax.axvline(blend_mean, color='orange', linestyle='--', label='Blend')
        ax.axvline(no_blend_mean, color='green', linestyle='--', label='No-Blend')
        
        ax.set_title(f'{species.title()} {seq_type.title()} - GC Content Distribution', fontsize=LARGE_SIZE)
        ax.set_xlabel('GC Content (%)', fontsize=LARGE_SIZE)
        ax.set_ylabel('Count', fontsize=LARGE_SIZE)
        ax.legend(fontsize=MEDIUM_SIZE)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                if text.get_text() in ('Real', 'Blend', 'No-Blend'):
                    text.set_fontstyle('italic')
        ax.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)
        
        plt.tight_layout()
        output_path = (
            f"icml_analysis/{model}/{species}_{seq_type}_gc_content.png"
            if model else f"icml_analysis/{species}_{seq_type}_gc_content.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved GC content plot to {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating GC content plot for {species} {seq_type}: {e}")
        import traceback
        traceback.print_exc()

def create_individual_sequence_logo(species, seq_type, data, model=None):
    """Create sequence logo for a single dataset"""
    print(f"Creating sequence logos for {species} {seq_type}...")
    
    try:
        fig, axes = plt.subplots(3, 1, figsize=(15, 8))
        
        # Define the splice site region
        splice_region = (180, 220)
        
        # Dataset labels
        dataset_labels = ['Real', 'Blend (λ=0.5)', 'No-Blend (λ=0.0)']
        pwm_data = [data['pwm']['real'], data['pwm']['blend'], data['pwm']['no_blend']]
        
        for i in range(3):
            create_logo(pwm_data[i], axes[i], region=splice_region)
            axes[i].set_title(f'{dataset_labels[i]} - {species.title()} {seq_type.title()}', fontsize=MEDIUM_SIZE)
            
            # Only show x-label for bottom row
            if i < 2:
                axes[i].set_xlabel('')
        
        plt.tight_layout()
        output_path = (
            f"icml_analysis/{model}/{species}_{seq_type}_sequence_logos.png"
            if model else f"icml_analysis/{species}_{seq_type}_sequence_logos.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved sequence logos to {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating sequence logos for {species} {seq_type}: {e}")
        import traceback
        traceback.print_exc()

def process_dataset(species, seq_type, model, sample_size=40000):
    """Process a single dataset (species + seq_type) for a given model with blend/no-blend variants"""
    print(f"Processing {model} - {species} {seq_type}...")

    real_file = f"/home/ekabanga/All_DataSet/Splice/DRANet/{species}_{seq_type}_positive.txt"

    if model == 'DIFFUSION':
        blend_file = f"src/Lambda_sensitivity_analysis/{species}_{seq_type}_lambda_0.5_sequences.txt"
        no_blend_file = f"src/Lambda_sensitivity_analysis/{species}_{seq_type}_lambda_0.0_sequences.txt"
        enforce = False  # diffusion files reflect final sequences as-is
    elif model == 'GAN':
        base_dir = "src/GAN_generated_sequences"
        expected_size = '50k' if species == 'arabidopsis' else '100k'
        alt_size = '100k' if expected_size == '50k' else '50k'
        blend_file = f"{base_dir}/{species}_{seq_type}_train_{expected_size}_lambda_0.5_generated_sequences.txt"
        no_blend_file = f"{base_dir}/{species}_{seq_type}_train_{expected_size}_generated_sequences.txt"
        # Fallback if expected file missing
        if not os.path.exists(blend_file):
            alt_blend = f"{base_dir}/{species}_{seq_type}_train_{alt_size}_lambda_0.5_generated_sequences.txt"
            if os.path.exists(alt_blend):
                print(f"Warning: {blend_file} not found. Falling back to {alt_blend}")
                blend_file = alt_blend
        if not os.path.exists(no_blend_file):
            alt_noblend = f"{base_dir}/{species}_{seq_type}_train_{alt_size}_generated_sequences.txt"
            if os.path.exists(alt_noblend):
                print(f"Warning: {no_blend_file} not found. Falling back to {alt_noblend}")
                no_blend_file = alt_noblend
        enforce = True
    elif model == 'VAE':
        base_dir = "src/VAE_generated_sequences"
        expected_size = '50k' if species == 'arabidopsis' else '100k'
        alt_size = '100k' if expected_size == '50k' else '50k'
        blend_file = f"{base_dir}/{species}_{seq_type}_train_{expected_size}_lambda_0.5_generated_sequences.txt"
        no_blend_file = f"{base_dir}/{species}_{seq_type}_train_{expected_size}_generated_sequences.txt"
        # Fallback if expected file missing
        if not os.path.exists(blend_file):
            alt_blend = f"{base_dir}/{species}_{seq_type}_train_{alt_size}_lambda_0.5_generated_sequences.txt"
            if os.path.exists(alt_blend):
                print(f"Warning: {blend_file} not found. Falling back to {alt_blend}")
                blend_file = alt_blend
        if not os.path.exists(no_blend_file):
            alt_noblend = f"{base_dir}/{species}_{seq_type}_train_{alt_size}_generated_sequences.txt"
            if os.path.exists(alt_noblend):
                print(f"Warning: {no_blend_file} not found. Falling back to {alt_noblend}")
                no_blend_file = alt_noblend
        enforce = True
    else:
        raise ValueError(f"Unknown model {model}")

    print(f"Loading sequences from {real_file}...")
    real_seqs = load_sequences(real_file, sample_size)

    print(f"Loading sequences from {blend_file}...")
    blend_seqs = load_sequences(blend_file, sample_size)

    print(f"Loading sequences from {no_blend_file}...")
    no_blend_seqs = load_sequences(no_blend_file, sample_size)

    # Enforce splice-site motif for non-blended sets for GAN/VAE
    if enforce:
        no_blend_seqs = maybe_enforce_splice_site(no_blend_seqs, seq_type, enforce=True)

    print(
        f"Loaded {len(real_seqs)} real sequences, {len(blend_seqs)} blend sequences, and {len(no_blend_seqs)} no-blend sequences"
    )

    output_prefix = f"icml_analysis/{model}/{species}_{seq_type}"
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    conservation_data = analyze_nucleotide_conservation(real_seqs, blend_seqs, no_blend_seqs, output_prefix)
    gc_data = analyze_gc_content(real_seqs, blend_seqs, no_blend_seqs, output_prefix)
    three_mer_data = analyze_3mers(real_seqs, blend_seqs, no_blend_seqs, output_prefix)

    return {
        'pwm': conservation_data['pwm'],
        'conservation': conservation_data['conservation'],
        'gc_content': gc_data['gc_content'],
        'means': gc_data['means'],
        'three_mer': three_mer_data,
    }

def ask_user_permission(species, seq_type):
    """Deprecated: kept for compatibility; returns True without prompting."""
    return True

def main():
    random.seed(42)

    sample_size = 40000
    datasets = [
        ('arabidopsis', 'donor'),
        ('arabidopsis', 'acceptor'),
        ('homo', 'donor'),
        ('homo', 'acceptor'),
    ]

    models = ['DIFFUSION', 'GAN', 'VAE']

    all_model_data = {model: {} for model in models}

    print("🔬 ICML Analysis (non-interactive)")
    print("=" * 50)

    for model in models:
        print(f"\n🧪 Model: {model}")
        processed_data = {}
        for species, seq_type in datasets:
            try:
                data = process_dataset(species, seq_type, model, sample_size)
                processed_data[f"{species}_{seq_type}"] = data
                print(f"✅ Completed {model} {species} {seq_type}")
            except Exception as e:
                print(f"❌ Error processing {model} {species} {seq_type}: {e}")

        # Save individual plots and combined per-model plots
        for dataset_key, data in processed_data.items():
            species, seq_type = dataset_key.split('_')
            create_individual_conservation_plot(species, seq_type, data, model=model)
            create_individual_gc_plot(species, seq_type, data, model=model)
            create_individual_sequence_logo(species, seq_type, data, model=model)

        if 'arabidopsis_donor' in processed_data and 'arabidopsis_acceptor' in processed_data:
            create_combined_sequence_logos('arabidopsis', processed_data['arabidopsis_donor'], processed_data['arabidopsis_acceptor'], model=model)

        if 'homo_donor' in processed_data and 'homo_acceptor' in processed_data:
            create_combined_sequence_logos('homo', processed_data['homo_donor'], processed_data['homo_acceptor'], model=model)

        if len(processed_data) == 4:
            create_combined_conservation_plot(
                processed_data['arabidopsis_donor'],
                processed_data['arabidopsis_acceptor'],
                processed_data['homo_donor'],
                processed_data['homo_acceptor'],
                model=model,
            )
            create_combined_gc_content_plot(
                processed_data['arabidopsis_donor'],
                processed_data['arabidopsis_acceptor'],
                processed_data['homo_donor'],
                processed_data['homo_acceptor'],
                model=model,
            )
            create_combined_3mer_plot(
                processed_data['arabidopsis_donor'],
                processed_data['arabidopsis_acceptor'],
                processed_data['homo_donor'],
                processed_data['homo_acceptor'],
                model=model,
            )
        else:
            print(
                f"⚠️  Only {len(processed_data)}/4 datasets processed for {model}. Skipping combined plots."
            )

        all_model_data[model] = processed_data

    print("\n✅ Analysis complete! Results saved to icml_analysis/<MODEL>/...")

if __name__ == "__main__":
    main() 