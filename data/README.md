# Data Directory

This directory should contain your splice site sequence data for training the generative models.

## Required Directory Structure

```
data/
├── Arabidopsis/
│   ├── Donor_positive.txt
│   └── Acceptor_positive.txt
└── Homo/
    ├── Donor_positive.txt
    └── Acceptor_positive.txt
```

## Data Format Requirements

### Sequence Format

- **Length**: Exactly **402 base pairs (bp)**
- **Alphabet**: {A, C, G, T} (uppercase)
- **Format**: Plain text file, one sequence per line
- **Splice Site Position**: Position 201 (center of the 402bp sequence)

### Splice Site Requirements

**Donor Sites (GT splice sites):**
- Must have **GT** at positions 201-202
- Example: `...ATCGATCGATCG[GT]ATCGATCGATCG...`

**Acceptor Sites (AG splice sites):**
- Must have **AG** at positions 201-202
- Example: `...ATCGATCGATCG[AG]ATCGATCGATCG...`

### Example File Content

```
ATCGATCGATCGATCGATCGATCGATCGATCGATC...GTATCGATCGATCGATCGATCGATCGATCGATCG
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT...GTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT
TACGTACGTACGTACGTACGTACGTACGTACGTAC...GTTACGTACGTACGTACGTACGTACGTACGTACG
...
```

Each line represents one complete 402bp sequence.

## Data Sources

Splice site sequences can be obtained from:

1. **Ensembl Plants** (for Arabidopsis thaliana):
   - URL: https://plants.ensembl.org/
   - Download GTF/GFF files and extract splice site regions

2. **Ensembl** (for Homo sapiens):
   - URL: https://ensembl.org/
   - Download GTF/GFF files and extract splice site regions

3. **UCSC Genome Browser**:
   - URL: https://genome.ucsc.edu/
   - Download gene annotations

4. **Custom Datasets**:
   - You can use your own annotated splice sites
   - Ensure they meet the format requirements above

## Data Preprocessing

If you have raw genome sequences and annotations, you need to:

1. **Extract splice site regions**: 
   - 200bp upstream + 2bp splice site + 200bp downstream = 402bp total

2. **Filter for quality**:
   - Remove sequences with ambiguous bases (N)
   - Ensure correct splice site motifs (GT for donor, AG for acceptor)
   - Remove duplicates

3. **Convert to required format**:
   - Uppercase all sequences
   - One sequence per line
   - Save as plain text (.txt)

## Example Preprocessing Script

```python
def extract_splice_site(genome_seq, position, site_type='donor'):
    """
    Extract 402bp region centered on splice site
    
    Args:
        genome_seq: Full chromosome sequence
        position: 0-based position of splice site
        site_type: 'donor' or 'acceptor'
    
    Returns:
        402bp sequence or None if invalid
    """
    start = position - 200
    end = position + 202
    
    if start < 0 or end > len(genome_seq):
        return None
    
    seq = genome_seq[start:end].upper()
    
    # Check length
    if len(seq) != 402:
        return None
    
    # Check for ambiguous bases
    if 'N' in seq:
        return None
    
    # Check splice site motif
    motif = seq[200:202]
    if site_type == 'donor' and motif != 'GT':
        return None
    if site_type == 'acceptor' and motif != 'AG':
        return None
    
    return seq
```

## Dataset Statistics (Recommended)

For robust training, we recommend:

- **Arabidopsis**:
  - Donor sites: 40,000-50,000 sequences
  - Acceptor sites: 40,000-50,000 sequences

- **Homo sapiens**:
  - Donor sites: 80,000-100,000 sequences
  - Acceptor sites: 80,000-100,000 sequences

**Minimum requirements**: At least 10,000 sequences per category for reasonable results.

## Data Splitting

The training scripts will automatically split data:
- **Training**: 80% of sequences
- **Validation**: 10% of sequences
- **Testing**: 10% of sequences

All splits use fixed random seeds for reproducibility.

## Important Notes

⚠️ **Do not commit large data files to Git!**
- Data files are excluded in `.gitignore`
- Store data separately (e.g., cloud storage, institutional servers)
- Consider providing download scripts instead

⚠️ **Ensure proper train/test separation:**
- Never use test sequences for training
- Consider using separate datasets for final evaluation (e.g., ENSdata_for_mask)

## Revision dataset layout (2002 bp)

```
data/
├── arabidopsis/
│   ├── arabidopsis_{donor,acceptor}_2002_{positive,negative}_unique.txt
│   ├── Arabidopsis_thaliana.TAIR10.dna.toplevel.fa
│   └── Arabidopsis_thaliana.TAIR10.62.gff3
├── danio/
│   ├── danio_{donor,acceptor}_2002_{positive,negative}_unique.txt
│   ├── Danio_rerio.GRCz11.dna.primary_assembly.fa
│   └── Danio_rerio.GRCz11.115.gff3
└── human/
    ├── human_{donor,acceptor}_2002_{positive,negative}_unique.txt
    ├── GRCh38.primary_assembly.genome.fa
    └── gencode.v49.basic.annotation.gtf
```

Generated synthetic sequences live under `generated_sequences/{402|2002}/{GAN|VAE|Diffusion}/lambda_*/`.

**402 bp (native):** Arabidopsis and human from `src/GAN_generated_sequences`, `src/VAE_generated_sequences`, and `src/Lambda_sensitivity_analysis` (copied into `generated_sequences/402/`). Danio 402 bp from revision outputs (`danio_402_*`).

**2002 bp:** All species from revision outputs. Danio uses `danio_2002_*` prefixes; arabidopsis/human omit the length tag.

Files are gitignored due to size; copy locally from the project release bundle.

## Questions?

If you have questions about data format or preprocessing, please:
1. Check the main README.md
2. Review the example scripts in `examples/`
3. Open an issue on GitHub
