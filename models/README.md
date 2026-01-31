# Models Directory

This directory will contain trained model checkpoints after running the training scripts.

## Model Storage Structure

After training, models will be saved in the following structure:

```
models/
├── GAN_models/
│   ├── gan_arabidopsis_donor.pth
│   ├── gan_arabidopsis_acceptor.pth
│   ├── gan_homo_donor.pth
│   ├── gan_homo_acceptor.pth
│   └── losses/
│       ├── gan_arabidopsis_donor_losses.json
│       └── ...
│
├── VAE_models/
│   ├── vae_arabidopsis_donor.pth
│   ├── vae_arabidopsis_acceptor.pth
│   ├── vae_homo_donor.pth
│   ├── vae_homo_acceptor.pth
│   └── losses/
│       ├── vae_arabidopsis_donor_losses.json
│       └── ...
│
└── Enhanced_Diffusion_models/
    ├── diffusion_arabidopsis_donor_lambda0.3.pth
    ├── diffusion_arabidopsis_acceptor_lambda0.3.pth
    ├── diffusion_homo_donor_lambda0.3.pth
    ├── diffusion_homo_acceptor_lambda0.3.pth
    └── losses/
        ├── diffusion_arabidopsis_donor_lambda0.3_losses.json
        └── ...
```

## Model File Format

All models are saved as PyTorch checkpoints (`.pth` files) containing:

### GAN Models
```python
{
    'generator_state_dict': OrderedDict(...),      # Generator weights
    'discriminator_state_dict': OrderedDict(...),  # Discriminator weights
    'g_optimizer_state_dict': OrderedDict(...),    # Generator optimizer
    'd_optimizer_state_dict': OrderedDict(...),    # Discriminator optimizer
    'epoch': int,                                   # Training epoch
    'config': dict,                                 # Model configuration
}
```

### VAE Models
```python
{
    'model_state_dict': OrderedDict(...),          # VAE weights (encoder + decoder)
    'optimizer_state_dict': OrderedDict(...),      # Optimizer state
    'epoch': int,                                   # Training epoch
    'config': dict,                                 # Model configuration
}
```

### Diffusion Models
```python
{
    'model_state_dict': OrderedDict(...),          # U-Net weights
    'optimizer_state_dict': OrderedDict(...),      # Optimizer state
    'epoch': int,                                   # Training epoch
    'config': dict,                                 # Model configuration
    'lambda_val': float,                            # Blending parameter
}
```

## Loss History Files

Training losses are saved as JSON files for analysis and visualization:

```json
{
    "epoch_losses": [0.5234, 0.4891, 0.4523, ...],
    "epoch_time": [45.2, 44.8, 45.1, ...],
    "total_training_time": 1234.56,
    "final_loss": 0.1234
}
```

## Loading Pre-trained Models

### GAN
```python
import torch
from src.gan_baseline import Generator, Discriminator

# Load checkpoint
checkpoint = torch.load('models/GAN_models/gan_arabidopsis_donor.pth')

# Initialize models
generator = Generator(latent_dim=128)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Generate sequences
with torch.no_grad():
    noise = torch.randn(100, 128)
    sequences = generator(noise)
```

### VAE
```python
import torch
from src.vae_baseline import VAE

# Load checkpoint
checkpoint = torch.load('models/VAE_models/vae_arabidopsis_donor.pth')

# Initialize model
vae = VAE(latent_dim=64)
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

# Generate sequences
with torch.no_grad():
    noise = torch.randn(100, 64)
    sequences = vae.decode(noise)
```

### Diffusion
```python
import torch
from src.diffusion_baseline import UNet

# Load checkpoint
checkpoint = torch.load('models/Enhanced_Diffusion_models/diffusion_arabidopsis_donor_lambda0.3.pth')

# Initialize model
model = UNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate sequences (requires full diffusion pipeline)
# See src/diffusion_baseline.py for complete generation code
```

## Model Size

Approximate file sizes for trained models:

| Model Type | Size per Checkpoint |
|------------|---------------------|
| GAN        | ~50-100 MB          |
| VAE        | ~100-150 MB         |
| Diffusion  | ~200-300 MB         |

**Total storage for all models**: ~5-10 GB (4 species × seq_type combinations per model)

## Important Notes

⚠️ **Model files are excluded from Git!**
- All `.pth` and `.pt` files are in `.gitignore`
- Model files are too large for GitHub
- Consider hosting models on:
  - Google Drive
  - Zenodo
  - Hugging Face Model Hub
  - Institutional repositories

⚠️ **Model Versioning:**
- Models are saved with descriptive names including species, seq_type, and hyperparameters
- Keep track of model versions in your experiments
- Consider using tools like MLflow or Weights & Biases for experiment tracking

⚠️ **Reproducibility:**
- Models use fixed random seeds (default: 42)
- Training is deterministic when using the same seed
- Save model configuration along with weights for reproducibility

## Model Performance

Expected performance metrics (approximate):

### Direct Evaluation
| Model     | PWM Similarity | GC Similarity | 3-mer Similarity | Splice Site Accuracy |
|-----------|----------------|---------------|------------------|----------------------|
| GAN       | 0.85-0.90      | 0.90-0.95     | 0.85-0.90        | 0.95-0.99            |
| VAE       | 0.80-0.85      | 0.85-0.90     | 0.80-0.85        | 0.90-0.95            |
| Diffusion | 0.90-0.95      | 0.95-0.98     | 0.90-0.95        | 0.98-0.99            |

### Indirect Evaluation (Proxy Models)
| Model     | F1-Score | AUROC | MCC   |
|-----------|----------|-------|-------|
| GAN       | 0.70-0.80| 0.75-0.85 | 0.65-0.75 |
| VAE       | 0.65-0.75| 0.70-0.80 | 0.60-0.70 |
| Diffusion | 0.80-0.90| 0.85-0.95 | 0.75-0.85 |

*Values may vary depending on dataset size and quality*

## Sharing Trained Models

If you want to share your trained models:

1. **Upload to cloud storage** (Google Drive, Zenodo, etc.)
2. **Include model card** with:
   - Training dataset description
   - Hyperparameters used
   - Performance metrics
   - Known limitations
3. **Provide download script** or instructions in main README
4. **Include checksums** (MD5/SHA256) for verification

## Questions?

For questions about model format or loading:
1. Check the training scripts in `src/`
2. Review the example scripts in `examples/`
3. Open an issue on GitHub
