# BlendSplice

Deep generative models for generating synthetic splice site sequences using GAN, VAE, and Diffusion approaches with frequency-guided blending.

## Overview

BlendSplice implements three generative models to create biologically realistic splice site sequences:
- **GAN** (Generative Adversarial Network)
- **VAE** (Variational Autoencoder)
- **Diffusion Model** with frequency-guided blending

Supports both **Arabidopsis thaliana** and **Homo sapiens** genomes, for donor (GT) and acceptor (AG) splice sites.

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

## Repository Structure

```
BlendSplice/
├── src/
│   ├── gan_baseline.py              # GAN implementation
│   ├── vae_baseline.py              # VAE implementation
│   ├── diffusion_baseline.py        # Diffusion with blending
│   ├── indirect_models.py           # SpliceRover & Spliceator
│   ├── direct_analysis.py           # Direct evaluation
│   └── proxy*.py                    # Proxy model utilities
├── examples/                        # Example scripts
├── data/                            # Data directory (see data/README.md)
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
