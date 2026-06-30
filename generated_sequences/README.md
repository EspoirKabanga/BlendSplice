# Generated Sequences

Large sequence files are stored locally under this directory and are **not** committed to Git.

## Layout

```
generated_sequences/
├── 402/
│   ├── GAN/lambda_{0.0,0.25,0.5,0.75}/
│   ├── VAE/lambda_*/
│   └── Diffusion/lambda_*/
└── 2002/
    ├── GAN/lambda_*/
    ├── VAE/lambda_*/
    └── Diffusion/lambda_*/
```

## Populate locally

From the repository root:

```bash
python scripts/organize_generated_sequences.py
```

This copies:
- **402 bp** arabidopsis/human from `../src/GAN_generated_sequences`, `VAE_generated_sequences`, and `Lambda_sensitivity_analysis`
- **402 bp** danio and **2002 bp** all species from `../Revision_Generated_Sequences`
