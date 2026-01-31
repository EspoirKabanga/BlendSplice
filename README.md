# BlendSplice: Deep Generative Models for Splice Site Sequence Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**BlendSplice** is a comprehensive framework for generating synthetic splice site sequences using deep generative models. It implements and compares three state-of-the-art generative approaches: **Generative Adversarial Networks (GANs)**, **Variational Autoencoders (VAEs)**, and **Diffusion Models** with a novel frequency-guided blending strategy.

---

## 🎯 Key Features

- **Three Generative Models**: GAN, VAE, and Diffusion with frequency-guided generation
- **Multi-Species Support**: Arabidopsis thaliana and Homo sapiens
- **Dual Splice Site Types**: Donor (GT) and Acceptor (AG) sites
- **Comprehensive Evaluation**: 
  - **Direct metrics**: PWM similarity, GC content, 3-mer frequency, splice site accuracy
  - **Indirect metrics**: Proxy model evaluation (SpliceRover, Spliceator)
- **Sample Efficiency Analysis**: Investigate minimal training data requirements
- **Reproducible**: All experiments with fixed random seeds

---

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training Generative Models](#training-generative-models)
  - [Generating Sequences](#generating-sequences)
  - [Evaluation](#evaluation)
- [Data Format](#data-format)
- [Model Architectures](#model-architectures)
- [Citation](#citation)
- [License](#license)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM recommended

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/BlendSplice.git
cd BlendSplice
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
BlendSplice/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── LICENSE                        # License information
├── .gitignore                     # Git ignore rules
│
├── src/
│   ├── gan_baseline.py            # GAN training & generation
│   ├── vae_baseline.py            # VAE training & generation
│   ├── diffusion_baseline.py      # Diffusion training & generation
│   ├── indirect_models.py         # Proxy models (SpliceRover, Spliceator)
│   ├── direct_analysis.py         # Direct evaluation metrics
│   ├── proxy.py                   # Proxy model utilities
│   ├── proxy_train_baseline.py    # Train proxy models
│   ├── proxy_test_baseline.py     # Test proxy models
│   └── Final_proxy_evaluation.py  # Complete evaluation pipeline
│
├── data/
│   └── README.md                  # Data format & requirements
│
├── models/
│   └── README.md                  # Saved model information
│
└── examples/
    ├── train_gan_example.sh       # GAN training example
    ├── train_vae_example.sh       # VAE training example
    ├── train_diffusion_example.sh # Diffusion training example
    └── evaluate_example.sh        # Evaluation example
```

---

## ⚡ Quick Start

### 1. Prepare Your Data

Organize your splice site sequences in the following format:
```
data/
├── Arabidopsis/
│   ├── Donor_positive.txt
│   └── Acceptor_positive.txt
└── Homo/
    ├── Donor_positive.txt
    └── Acceptor_positive.txt
```

Each file should contain one sequence per line (402bp, centered on splice site).

### 2. Train a Model

**Example: Train GAN for Arabidopsis Donor sites**
```bash
python src/gan_baseline.py \
    --species arabidopsis \
    --seq_type donor \
    --epochs 100 \
    --batch_size 32 \
    --gpu 0
```

### 3. Generate Sequences

The models automatically generate sequences after training. Generated sequences are saved in:
```
<MODEL_TYPE>_generated_sequences/<species>_<seq_type>.txt
```

### 4. Evaluate Generated Sequences

**Direct Evaluation:**
```bash
python src/direct_analysis.py \
    --generated_file GAN_generated_sequences/arabidopsis_donor.txt \
    --real_file data/Arabidopsis/Donor_positive.txt \
    --output_dir results/
```

**Indirect Evaluation (Proxy Models):**
```bash
python src/Final_proxy_evaluation.py \
    --species arabidopsis \
    --seq_type donor \
    --model_type GAN
```

---

## 📚 Usage

### Training Generative Models

#### 1. **GAN Training**

```bash
python src/gan_baseline.py \
    --species <arabidopsis|homo> \
    --seq_type <donor|acceptor> \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0002 \
    --latent_dim 128 \
    --gpu 0
```

**Key Parameters:**
- `--species`: Species to train on (arabidopsis or homo)
- `--seq_type`: Splice site type (donor or acceptor)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--latent_dim`: Dimension of latent space (default: 128)
- `--gpu`: GPU device ID

#### 2. **VAE Training**

```bash
python src/vae_baseline.py \
    --species <arabidopsis|homo> \
    --seq_type <donor|acceptor> \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --latent_dim 64 \
    --kl_weight 0.1 \
    --gpu 0
```

**Key Parameters:**
- `--kl_weight`: Weight for KL divergence loss (default: 0.1)
- Other parameters same as GAN

#### 3. **Diffusion Training**

```bash
python src/diffusion_baseline.py \
    --species <arabidopsis|homo> \
    --seq_type <donor|acceptor> \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --timesteps 1000 \
    --lambda_val 0.3 \
    --gpu 0
```

**Key Parameters:**
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--lambda_val`: Blending weight for frequency guidance (0.0-1.0)
- Other parameters same as GAN/VAE

**Frequency-Guided Blending:** The diffusion model uses a novel blending strategy:
```
final_prediction = λ × model_prediction + (1-λ) × frequency_prior
```
- `λ = 0.0`: Pure frequency-based generation
- `λ = 1.0`: Pure model-based generation
- `λ = 0.3`: Recommended balance (default)

---

### Generating Sequences

All models automatically generate 10,000 sequences after training. To generate additional sequences from a saved model:

```bash
python src/<model>_baseline.py \
    --mode generate \
    --model_path models/<model>_arabidopsis_donor.pth \
    --num_sequences 10000 \
    --output_file custom_sequences.txt
```

---

### Evaluation

#### Direct Evaluation Metrics

Evaluate generated sequences using biologically-inspired metrics:

```bash
python src/direct_analysis.py \
    --generated_file <path_to_generated_sequences> \
    --real_file <path_to_real_sequences> \
    --seq_type <donor|acceptor> \
    --output_dir results/
```

**Metrics Computed:**
- **PWM Similarity**: Position Weight Matrix comparison (splice site region)
- **GC Content**: Distribution similarity using Kolmogorov-Smirnov test
- **3-mer Frequency**: Cosine similarity of k-mer distributions
- **Splice Site Accuracy**: Percentage with correct motif (GT for donor, AG for acceptor)

#### Indirect Evaluation (Proxy Models)

Train deep learning models to discriminate between real and synthetic sequences:

```bash
# Full evaluation pipeline
python src/Final_proxy_evaluation.py \
    --species arabidopsis \
    --seq_type donor \
    --model_type GAN \
    --proxy_model SpliceRover
```

**Proxy Models:**
- **SpliceRover**: CNN-based architecture (recommended)
- **Spliceator**: Alternative CNN architecture

**Metrics Computed:**
- F1-Score
- AUROC (Area Under ROC Curve)
- MCC (Matthews Correlation Coefficient)

---

## 📊 Data Format

### Input Sequences

- **Length**: 402 base pairs (bp)
- **Format**: Plain text, one sequence per line
- **Alphabet**: {A, C, G, T} (uppercase)
- **Splice Site Position**: Position 201 (center of sequence)
- **Donor Sites**: Must have GT at positions 201-202
- **Acceptor Sites**: Must have AG at positions 201-202

**Example (Donor sequence):**
```
ATCGATCGATCG...GT...ATCGATCGATCG
                ↑
          Position 201
```

### Directory Structure for Training Data

```
data/
├── Arabidopsis/
│   ├── Donor_positive.txt      # Donor splice sites
│   └── Acceptor_positive.txt   # Acceptor splice sites
└── Homo/
    ├── Donor_positive.txt
    └── Acceptor_positive.txt
```

---

## 🏗️ Model Architectures

### GAN Architecture

- **Generator**: 
  - Input: 128-dimensional noise vector
  - Hidden layers: [256, 512, 1024]
  - Output: 402 × 4 (one-hot encoded sequence)
  - Activation: ReLU + Tanh (output)

- **Discriminator**: 
  - Input: 402 × 4 (one-hot encoded sequence)
  - Hidden layers: [1024, 512, 256]
  - Output: 1 (real/fake probability)
  - Activation: LeakyReLU + Sigmoid (output)

### VAE Architecture

- **Encoder**: 
  - Input: 402 × 4
  - Hidden layers: [1024, 512, 256]
  - Output: 64-dimensional latent space (μ and σ)
  
- **Decoder**: 
  - Input: 64-dimensional latent vector
  - Hidden layers: [256, 512, 1024]
  - Output: 402 × 4
  - Loss: Reconstruction loss + KL divergence

### Diffusion Model

- **U-Net Architecture**:
  - Encoder: [64, 128, 256, 512]
  - Decoder: [512, 256, 128, 64]
  - Skip connections between encoder and decoder
  - Time embedding: Sinusoidal positional encoding
  
- **Frequency-Guided Blending**:
  - Previous nucleotide frequencies
  - Next nucleotide frequencies
  - Blending parameter λ (configurable)

---

## 🎓 Citation

If you use BlendSplice in your research, please cite:

```bibtex
@article{blendsplice2024,
  title={BlendSplice: Frequency-Guided Deep Generative Models for Splice Site Sequence Generation},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  doi={your-doi}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Scientific community for splice site datasets
- All contributors and users of BlendSplice

---

## 📌 Additional Resources

- [Detailed Documentation](docs/)
- [Sample Efficiency Study](docs/sample_efficiency.md)
- [Model Training Tips](docs/training_tips.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

---

**Last Updated**: January 2026
