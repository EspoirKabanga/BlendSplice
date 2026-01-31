#!/bin/bash
################################################################################
# BlendSplice - Evaluation Example
################################################################################
# This script demonstrates how to evaluate generated sequences using both
# direct and indirect (proxy model) evaluation methods
# 
# Usage: bash examples/evaluate_example.sh
################################################################################

echo "========================================================================"
echo "BlendSplice - Evaluation Example"
echo "========================================================================"
echo ""

# Configuration
SPECIES="arabidopsis"  # Options: arabidopsis, homo
SEQ_TYPE="donor"       # Options: donor, acceptor
MODEL_TYPE="GAN"       # Options: GAN, VAE, Diffusion

echo "Evaluation Configuration:"
echo "  Species: $SPECIES"
echo "  Sequence Type: $SEQ_TYPE"
echo "  Model Type: $MODEL_TYPE"
echo ""

# ============================================================================
# DIRECT EVALUATION
# ============================================================================
echo "========================================================================"
echo "STEP 1: Direct Evaluation"
echo "========================================================================"
echo "Metrics: PWM similarity, GC content, 3-mer frequency, splice site accuracy"
echo ""

GENERATED_FILE="${MODEL_TYPE}_generated_sequences/${SPECIES}_${SEQ_TYPE}.txt"
REAL_FILE="data/${SPECIES^}/$(echo $SEQ_TYPE | sed 's/./\u&/')_positive.txt"
OUTPUT_DIR="results/direct_evaluation_${MODEL_TYPE}"

if [ ! -f "$GENERATED_FILE" ]; then
    echo "⚠️  Generated file not found: $GENERATED_FILE"
    echo "Please train the model first using:"
    echo "  bash examples/train_${MODEL_TYPE,,}_example.sh"
    exit 1
fi

if [ ! -f "$REAL_FILE" ]; then
    echo "⚠️  Real data file not found: $REAL_FILE"
    echo "Please ensure your data is organized in the correct directory structure."
    exit 1
fi

echo "Running direct evaluation..."
python src/direct_analysis.py \
    --generated_file $GENERATED_FILE \
    --real_file $REAL_FILE \
    --seq_type $SEQ_TYPE \
    --output_dir $OUTPUT_DIR

echo ""
echo "✅ Direct evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# ============================================================================
# INDIRECT EVALUATION (PROXY MODELS)
# ============================================================================
echo "========================================================================"
echo "STEP 2: Indirect Evaluation (Proxy Models)"
echo "========================================================================"
echo "Using proxy models: SpliceRover and Spliceator"
echo "Metrics: F1-score, AUROC, MCC"
echo ""

echo "Running indirect evaluation..."
python src/Final_proxy_evaluation.py \
    --species $SPECIES \
    --seq_type $SEQ_TYPE \
    --model_type $MODEL_TYPE

echo ""
echo "✅ Indirect evaluation completed!"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "========================================================================"
echo "EVALUATION SUMMARY"
echo "========================================================================"
echo ""
echo "Direct Evaluation Metrics:"
echo "  - PWM Similarity: Measures position-specific nucleotide patterns"
echo "  - GC Content: Distribution similarity using KS test"
echo "  - 3-mer Frequency: Cosine similarity of k-mer distributions"
echo "  - Splice Site Accuracy: Percentage with correct motif (GT/AG)"
echo ""
echo "Indirect Evaluation Metrics:"
echo "  - F1-Score: Harmonic mean of precision and recall"
echo "  - AUROC: Area under ROC curve"
echo "  - MCC: Matthews Correlation Coefficient"
echo ""
echo "Higher values indicate better sequence quality!"
echo ""
echo "========================================================================"
echo "Next Steps:"
echo "========================================================================"
echo "1. Compare results across different models (GAN, VAE, Diffusion)"
echo "2. Visualize results in the output directories"
echo "3. Analyze figures and tables generated"
echo "4. Experiment with different hyperparameters"
echo ""
