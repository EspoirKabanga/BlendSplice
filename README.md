# BlendSplice

Deep generative models for generating synthetic splice site sequences using GAN, VAE, and Diffusion approaches with frequency-guided blending.

## Overview

BlendSplice implements three generative models to create biologically realistic splice site sequences:
- **GAN** (Generative Adversarial Network)
- **VAE** (Variational Autoencoder)
- **Diffusion Model** with frequency-guided blending

Supports **Arabidopsis thaliana**, **Homo sapiens**, and **Danio rerio**, for donor (GT) and acceptor (AG) splice sites at **402 bp** and **2002 bp** context lengths.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

## Quick Start

### Training

**GAN:**
```bash
python src/gan_baseline.py --species arabidopsis --seq_type donor
```

**VAE:**
```bash
python src/vae_baseline.py --species arabidopsis --seq_type donor
```

**Diffusion:**
```bash
python src/diffusion_baseline.py --species arabidopsis --seq_type donor --mode comprehensive
```

### Evaluation

**Direct Evaluation** (PWM, GC content, 3-mer frequency):
```bash
python src/direct_analysis.py --generated_file <path> --real_file <path> --seq_type donor
```

**Indirect Evaluation** (Proxy models: SpliceRover, Spliceator):
```bash
python src/Final_proxy_evaluation.py --species arabidopsis --seq_type donor --model_type GAN
```

**Revision direct evaluation** (multi-species, multi-λ, comprehensive figures):

```bash
python src/revision_supplementary_comprehensive_figures.py --context all --sample-size 20000
python src/direct_revision_ppt_analysis.py
python src/test_sequence_spliceai.py --context 2002
```

## Repository Structure

```
BlendSplice/
├── src/
│   ├── gan_baseline.py              # GAN implementation
│   ├── vae_baseline.py              # VAE implementation
│   ├── diffusion_baseline.py        # Diffusion with blending
│   ├── indirect_models.py           # SpliceRover & Spliceator
│   ├── direct_analysis.py           # Direct evaluation
│   ├── direct_revision_*.py         # Revision direct evaluation
│   ├── revision_supplementary_*.py  # Supplementary figure pipelines
│   ├── test_sequence_spliceai.py    # SpliceAI detection-rate benchmark
│   └── proxy*.py                    # Proxy model utilities
├── scripts/
│   └── organize_generated_sequences.py
├── examples/                        # Example scripts
├── data/                            # Real sequences & annotations (see data/README.md)
├── generated_sequences/             # Synthetic outputs (see generated_sequences/README.md)
└── models/                          # Saved models directory
```

## Example Scripts

```bash
# Train all three models for Arabidopsis donor sites
bash examples/train_gan_example.sh
bash examples/train_vae_example.sh
bash examples/train_diffusion_example.sh

# Evaluate generated sequences
bash examples/evaluate_example.sh
```

## Citation

Will be updated soon

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.
