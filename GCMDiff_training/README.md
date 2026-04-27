# GCMDiff (Training)

## Project Structure
```
GCMDiff_training/
├── dataset/                             # Trainging data: img (processed SMILES) &label (SMILES features) 
├── denoising_diffusion/
│   ├── classifier_free_guidance_v4.py   # Model architecture
│   └── model_weight/                    # Directory for training weights
├── preprocess/                          # Data preprocessing scripts
└── GGCD.py                              # Main training script
```
---


## Model Configuration
### GaussianDiffusion
 
```python
diffusion = GaussianDiffusion(
    model,
    image_size = 40,    # Input image dimensions (40x40)
    timesteps  = 1000   # Number of diffusion timesteps
)
```
---

 ### Data Loader Settings
| Parameter | Description |
|---|---|
| `batch_size` | Recommended: `128` (Use 2 for local testing/debugging only) |
| `shuffle` | Set to `True` to randomize training order |
| `drop_last` | Set to `True` to drop the remaining data if it doesn't fit a full batch |
 
---

## Model Checkpoints
 Weights are automatically saved every 10 epochs to the following directory:
 ```
./denoising_diffusion/model_weight/{epoch}_{loss}.pt
```
 
Example：`./denoising_diffusion/model_weight/10_0.9652688503265381.pt`
 
---

### Run

```bash
CUDA_VISIBLE_DEVICES=1 python3 GGCD.py 
```
