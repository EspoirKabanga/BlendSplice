#!/bin/bash
################################################################################
# BlendSplice - GAN Training Example
################################################################################
# This script demonstrates how to train a GAN model for splice site generation
# 
# Usage: bash examples/train_gan_example.sh
################################################################################

echo "========================================================================"
echo "BlendSplice - GAN Training Example"
echo "========================================================================"
echo ""

# Configuration
SPECIES="arabidopsis"  # Options: arabidopsis, homo
SEQ_TYPE="donor"       # Options: donor, acceptor
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.0002
LATENT_DIM=128
GPU_ID=0

echo "Training Configuration:"
echo "  Species: $SPECIES"
echo "  Sequence Type: $SEQ_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Latent Dimension: $LATENT_DIM"
echo "  GPU: $GPU_ID"
echo ""

# Train GAN model
echo "Starting GAN training..."
python src/gan_baseline.py \
    --species $SPECIES \
    --seq_type $SEQ_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --latent_dim $LATENT_DIM \
    --gpu $GPU_ID

echo ""
echo "========================================================================"
echo "GAN training completed!"
echo "========================================================================"
echo ""
echo "Generated sequences saved to: GAN_generated_sequences/"
echo "Model saved to: GAN_models/"
echo ""
echo "Next steps:"
echo "  1. Evaluate generated sequences with: bash examples/evaluate_example.sh"
echo "  2. Train VAE with: bash examples/train_vae_example.sh"
echo "  3. Train Diffusion with: bash examples/train_diffusion_example.sh"
echo ""
