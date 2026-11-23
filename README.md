# AI-Generated Image Detection (GenImage-20K)

Multi-branch detector for real vs. AI-generated images, built in PyTorch and run on a Kaggle GPU.
Team: CS24B1007 Maneesha, CS24B1014 Roshitha, CS24B1026 Sathwika, CS24B1070 Havila.

## Model
Three branches fused by a learned gate:
- **A** - frozen CLIP ViT-B/32, CLS tokens from layers 4, 8, 12 -> MLP
- **B** - CNN on the FFT log-magnitude spectrum
- **C** - CNN on a high-pass residual + 3 fixed SRM filters

Training uses branch dropout (0.15), an auxiliary loss per branch (weight 0.3), AdamW + OneCycle, AMP, gradient clipping,
early stopping on validation AUC. Confidence is calibrated with temperature scaling.

## Data
`leatwitchy/genimage-20k-preprocessed` (Kaggle): 19,802 images, 256x256 JPEG q85. Train/val/seen-test use ADM, GLIDE, SD v1.4, SD v1.5, VQDM, Wukong; Midjourney is the unseen test set. The dataset is not included here.

## Results (validation split only)
| Metric | Value |
|---|---|
| Accuracy / AUC | 0.9491 / 0.9885 |
| F1 / AP | 0.9495 / 0.9880 |
| TPR at 1% FPR | 0.7905 |
| Temperature T / ECE before -> after | 1.353 / 0.0272 -> 0.0134 |
| CLIP linear probe (val) acc / AUC | 0.9376 / 0.9858 |

Robustness, per-generator and leave-one-generator-out tables are in `results/`.

**Limitations (read before citing the numbers)**
- Only validation results are reported. `FINAL_EVAL` (seen test + Midjourney) and `RUN_ABLATIONS` were `False` in the run.
- Single seed (42); no confidence intervals; no ResNet-50 baseline.
- The gate puts ~99% of its weight on CLIP for most images, so FFT/noise add little in practice.
- Native image size alone gives ROC-AUC 1.000 on this dataset (real = JPEG, AI = PNG originals); per-size-bin results are in the report.

## Files
- `AI_Image_Detection.ipynb` - notebook with saved outputs from the Kaggle run
- `results/` - CSV tables and figures extracted from that run
- `models/` - put `detector_full.pt` here (downloaded from the Kaggle run output)
- `report/` - project report (PDF + Markdown)

## Run
1. Kaggle: add the dataset as input, enable GPU and Internet, open the notebook, Run All.
2. Locally: `pip install -r requirements.txt`, change `ROOT` in the setup cell to your dataset path.
3. `RUN_ABLATIONS` / `FINAL_EVAL` are flags in the config cell; set `FINAL_EVAL = True` only once, after all choices are frozen.
