#!/bin/bash
################################################################################
# BlendSplice - VAE Training Example
################################################################################
# This script demonstrates how to train a VAE model for splice site generation
# 
# Usage: bash examples/train_vae_example.sh
################################################################################

echo "========================================================================"
echo "BlendSplice - VAE Training Example"
echo "========================================================================"
echo ""

# Configuration
SPECIES="arabidopsis"  # Options: arabidopsis, homo
SEQ_TYPE="donor"       # Options: donor, acceptor
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
LATENT_DIM=64
KL_WEIGHT=0.1
GPU_ID=0

echo "Training Configuration:"
echo "  Species: $SPECIES"
echo "  Sequence Type: $SEQ_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Latent Dimension: $LATENT_DIM"
echo "  KL Weight: $KL_WEIGHT"
echo "  GPU: $GPU_ID"
echo ""

# Train VAE model
echo "Starting VAE training..."
python src/vae_baseline.py \
    --species $SPECIES \
    --seq_type $SEQ_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --latent_dim $LATENT_DIM \
    --kl_weight $KL_WEIGHT \
    --gpu $GPU_ID

echo ""
echo "========================================================================"
echo "VAE training completed!"
echo "========================================================================"
echo ""
echo "Generated sequences saved to: VAE_generated_sequences/"
echo "Model saved to: VAE_models/"
echo ""
echo "Next steps:"
echo "  1. Evaluate generated sequences with: bash examples/evaluate_example.sh"
echo "  2. Train GAN with: bash examples/train_gan_example.sh"
echo "  3. Train Diffusion with: bash examples/train_diffusion_example.sh"
echo ""
