# GCMDiff (Sampling)

## Project Structure
```
GCMDiff_sampling/
├── denoising_diffusion/   # Model architecture and pre-trained weights
├── output/                # Generated compounds in .png (visualization) and .mol (structure) formats
├── utils/                 # Utility scripts for condition input, .mol file conversion, etc.
└── Sampling.py            # Main script for molecular generation
```
---

## Usage
### Define Generation Constraints
Open `Sampling.py`and modify the following parameters to define the target chemical properties:
> Note: For specific moiety codes (e.g., Rings in Drugs #1 for Benzene), please refer to the [519  structural and functional groups list](https://hisbim.life.nctu.edu.tw/Compound_moiety/functional_groups.php.).

```python
atomn         = [15]           # Targeted number of atoms
rule_5        = [1, 1, 1, 1]   # Lipinski's Rule of Five compliance
checkmol      = [201]          # Specific functional groups/moieties (Checkmol codes)
pubchem       = []             # PubChem-based features
ring          = [1]            # Targeted number of rings
batch_sample  = 3              # Number of molecules to generate per batch
```
---

### Run
```bash
CUDA_VISIBLE_DEVICES=1 python3 Sampling.py
```
