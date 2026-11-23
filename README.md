# AI-Generated Image Detection

A PyTorch detector that classifies images as **real** or **AI-generated** by combining three kinds of evidence: semantic features, frequency-domain features and noise-residual features. It outputs a label, a calibrated confidence and the weight the model gave to each branch.

**Team:** CS24B1007 Maneesha · CS24B1014 Roshitha · CS24B1026 Sathwika · CS24B1070 Havila

---

## Overview

| Branch | Input | Encoder |
|---|---|---|
| **A: Semantic** | image resized to 224 | frozen CLIP ViT-B/32, CLS tokens from layers 4, 8, 12 → LayerNorm → MLP → 128-d |
| **B: Frequency** | grayscale FFT log-magnitude spectrum (256x256) | 4-block CNN → 128-d |
| **C: Noise residual** | Gaussian high-pass residual + 3 fixed SRM filters (6 channels, 256x256) | 4-block CNN → 128-d |

The three embeddings go to a **gated fusion** module: an MLP over the concatenated embeddings gives softmax weights, and the fused feature is the weighted sum. A small classifier head produces the final logit.

```
image ─┬─ CLIP view  ── frozen CLIP ViT-B/32 (layers 4/8/12) ──┐
       ├─ FFT view   ── CNN ───────────────────────────────────┼─ gate ─ weighted sum ─ classifier ─ real / AI
       └─ noise view ── CNN (residual + SRM) ──────────────────┘
```

**Training details**
- Branch dropout (p = 0.15) and an auxiliary loss on every branch (weight 0.3)
- AdamW (lr 3e-4, weight decay 1e-2), OneCycle schedule, gradient clipping 1.0, mixed precision
- 6 epochs, batch size 32, early stopping on validation AUC (patience 3), seed 42
- Training augmentations: flip, JPEG (q 60-95), Gaussian blur, down-up resize, additive noise
- 2,855,285 trainable parameters (CLIP is frozen)
- Confidence calibrated with temperature scaling (T fitted on validation)

## Dataset

[`leatwitchy/genimage-20k-preprocessed`](https://www.kaggle.com/datasets/leatwitchy/genimage-20k-preprocessed), a bias-controlled 20K subset of GenImage. All images are 256x256 JPEG, quality 85.

| Split | Real | AI | Generators |
|---|---|---|---|
| Train | 6,913 | 6,913 | ADM, GLIDE, SD v1.4, SD v1.5, VQDM, Wukong |
| Validation | 993 | 993 | same six |
| Seen test | 995 | 995 | same six |
| Unseen test | 1,000 | 1,000 | Midjourney only |

The dataset is not included in this repository. Add it as an input on Kaggle.

## Results

Validation split (1,986 images):

| Metric | Value |
|---|---|
| Accuracy | 0.9491 |
| ROC-AUC | 0.9885 |
| Average precision | 0.9880 |
| F1 | 0.9495 |
| TPR at 1% FPR | 0.7905 |
| Real / AI accuracy | 0.9426 / 0.9557 |
| Temperature T | 1.353 |
| ECE before → after temperature scaling | 0.0272 → 0.0134 |

**Comparison with a CLIP linear probe** (logistic regression on the same cached multi-layer CLIP features, validation):

| Model | Accuracy | AUC |
|---|---|---|
| CLIP linear probe | 0.9376 | 0.9858 |
| Three-branch gated model | 0.9491 | 0.9885 |

**Per-generator validation results**

| Generator | Accuracy | AUC |
|---|---|---|
| ADM | 0.9697 | 0.9978 |
| GLIDE | 0.9667 | 0.9927 |
| SD v1.4 | 0.9641 | 0.9907 |
| SD v1.5 | 0.9458 | 0.9917 |
| VQDM | 0.9091 | 0.9679 |
| Wukong | 0.9394 | 0.9872 |

**Robustness (validation AUC)**

| None | JPEG q95 | JPEG q75 | JPEG q50 | Blur σ1 | Blur σ2 | Down-up 0.5x | Noise σ5 |
|---|---|---|---|---|---|---|---|
| 0.9885 | 0.9841 | 0.9758 | 0.9647 | 0.9761 | 0.9237 | 0.9688 | 0.9784 |

**Leave-one-generator-out (CLIP probe, AUC on the held-out generator):** ADM 0.786 · GLIDE 0.951 · SD v1.4 0.982 · SD v1.5 0.985 · VQDM 0.809 · Wukong 0.960

Full tables are in [`results/`](results/) and the write-up is in [`report/`](report/).

## Repository structure

```
AI_Image_Detection.ipynb   # full pipeline with saved outputs
requirements.txt
results/                   # CSV tables and figures from the run
models/                    # place detector_full.pt here
report/                    # project report (PDF and Markdown)
```

### Notebook contents
1. Setup and configuration
2. Data loading and integrity checks (no Midjourney in training, no source image in more than one split)
3. Dataset audit: native-size analysis, radial spectrum of real vs. each generator
4. Degradations (used for augmentation and for robustness tests)
5. GPU-side views (CLIP, FFT, residual) and dataset class
6. Model definition
7. Training, with training curves
8. Validation analysis: per generator, per native-size bin, gate weights
9. Calibration (temperature scaling, reliability diagram)
10. Robustness under JPEG, blur, resize and noise
11. CLIP linear probe and leave-one-generator-out
12. Optional ablations and final test evaluation (controlled by flags)
13. Inference function and saved artifacts

## How to run

**On Kaggle (recommended)**
1. Create a new notebook and import `AI_Image_Detection.ipynb`.
2. Add the dataset `leatwitchy/genimage-20k-preprocessed` as an input.
3. Set Accelerator to GPU and turn Internet on (needed once to download CLIP).
4. Run all cells. A full run takes roughly 20 minutes on a Kaggle GPU.

**Locally**
```bash
pip install -r requirements.txt
jupyter notebook AI_Image_Detection.ipynb
```
Change `ROOT` in the setup cell to the path of your copy of the dataset. A GPU is strongly recommended.

**Flags in the config cell**

| Flag | Default | Effect |
|---|---|---|
| `RUN_ABLATIONS` | `False` | trains the CLIP-only, FFT-only, noise-only and FFT+noise variants |
| `RUN_LOGO` | `True` | leave-one-generator-out experiment with the CLIP probe |
| `FINAL_EVAL` | `False` | evaluates on seen test and Midjourney; set to `True` once, after all choices are fixed |

## Inference

```python
result = predict_image("path/to/image.jpg")
# {'verdict': 'REAL' | 'AI-GENERATED', 'p_ai': ..., 'confidence': ...,
#  'branch_weights': {'clip': ..., 'fft': ..., 'noise': ...},
#  'branch_p_ai': {...}}
```
The function applies the same preprocessing as the dataset (center crop, Lanczos resize to 256, JPEG q85), then returns the calibrated probability and the per-branch gate weights.

## References
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
- Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS D&B 2023
- Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models*, CVPR 2023
- Fridrich and Kodovsky, *Rich Models for Steganalysis of Digital Images*, IEEE TIFS 2012
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017
</Content>
