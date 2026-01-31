#!/bin/bash
################################################################################
# BlendSplice - Diffusion Training Example
################################################################################
# This script demonstrates how to train a Diffusion model with frequency-guided
# blending for splice site generation
# 
# Usage: bash examples/train_diffusion_example.sh
################################################################################

echo "========================================================================"
echo "BlendSplice - Diffusion Training Example"
echo "========================================================================"
echo ""

# Configuration
SPECIES="arabidopsis"  # Options: arabidopsis, homo
SEQ_TYPE="donor"       # Options: donor, acceptor
EPOCHS=200
BATCH_SIZE=32
LEARNING_RATE=0.0001
TIMESTEPS=1000
LAMBDA_VAL=0.3         # Blending parameter (0.0-1.0)
GPU_ID=0

echo "Training Configuration:"
echo "  Species: $SPECIES"
echo "  Sequence Type: $SEQ_TYPE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Timesteps: $TIMESTEPS"
echo "  Lambda (Blending): $LAMBDA_VAL"
echo "  GPU: $GPU_ID"
echo ""
echo "Blending Strategy:"
echo "  λ = 0.0 → Pure frequency-based generation"
echo "  λ = 0.3 → Balanced (Recommended)"
echo "  λ = 1.0 → Pure model-based generation"
echo ""

# Train Diffusion model
echo "Starting Diffusion training..."
python src/diffusion_baseline.py \
    --species $SPECIES \
    --seq_type $SEQ_TYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --timesteps $TIMESTEPS \
    --lambda_val $LAMBDA_VAL \
    --gpu $GPU_ID

echo ""
echo "========================================================================"
echo "Diffusion training completed!"
echo "========================================================================"
echo ""
echo "Generated sequences saved to: Diffusion_generated_sequences/"
echo "Model saved to: Enhanced_Diffusion_models/"
echo ""
echo "Next steps:"
echo "  1. Evaluate generated sequences with: bash examples/evaluate_example.sh"
echo "  2. Train GAN with: bash examples/train_gan_example.sh"
echo "  3. Train VAE with: bash examples/train_vae_example.sh"
echo "  4. Try different λ values (0.0, 0.1, 0.3, 0.5, 1.0) to see the effect"
echo ""
