# Detecting AI-Generated Images with a Multi-Branch Semantic and Forensic Detector

**Team:** CS24B1007 Maneesha · CS24B1014 Roshitha · CS24B1026 Sathwika · CS24B1070 Havila
**Dataset:** GenImage-20K (preprocessed, Kaggle: `leatwitchy/genimage-20k-preprocessed`)
**Framework:** PyTorch, run on a Kaggle GPU notebook

---

## Abstract

Images from diffusion and other generative models now look realistic enough that people often cannot tell them from photographs. We built a detector that combines three kinds of evidence: (A) semantic features from several layers of a frozen CLIP ViT-B/32, (B) a CNN on the 2D FFT log-magnitude spectrum, and (C) a CNN on a noise residual made from a Gaussian high-pass residual and three fixed SRM filters. The three branch embeddings are fused with a learned gate. Training uses branch dropout and an auxiliary loss on every branch. We trained on 13,826 images from six generators (ADM, GLIDE, Stable Diffusion v1.4, Stable Diffusion v1.5, VQDM, Wukong). Midjourney is kept out of training entirely and is reserved as the unseen-generator test set. On the validation split the model reaches 94.91% accuracy and 0.9885 ROC-AUC. After temperature scaling (T = 1.353) the expected calibration error falls from 0.0272 to 0.0134. AUC stays above 0.96 under JPEG q50, noise and a 0.5x down-up resize, and drops to 0.924 under blur with sigma 2. A dataset audit shows that the original image size alone separates real from AI images perfectly (ROC-AUC 1.000), so we also report results per native-size bin. In our run the gate relied almost entirely on the CLIP branch. The held-out test sets (seen test and Midjourney) and the branch ablations were not evaluated in the run reported here, so all numbers below are validation numbers.

---

## 1. Introduction

Text-to-image models such as Stable Diffusion, GLIDE, ADM, VQDM, Wukong and Midjourney can produce images that look like real photographs. This causes problems for misinformation, journalism and image provenance, which is why automatic detection is an active research topic.

Detection is hard for two reasons. First, detectors tend to learn the quirks of the generators they were trained on and fail on new ones. Second, real-world images get compressed, resized and blurred, and this can wipe out the low-level traces many detectors rely on. A third problem is specific to benchmarks: if real and fake images differ in file format or resolution before they are processed, a model can score well by learning that difference and not the generation artifacts.

**Objectives of this project**

1. Audit the dataset for class balance, generator coverage and hidden shortcuts such as format and native resolution.
2. Build one detector that uses semantic, frequency-domain and noise-residual evidence together.
3. Stop a single branch from dominating by using branch dropout and per-branch auxiliary losses.
4. Measure robustness to JPEG compression, blur, resizing and noise.
5. Produce calibrated confidence values with temperature scaling and report ECE.
6. Estimate generalization to unseen generators with leave-one-generator-out (LOGO) experiments, and keep Midjourney as a final held-out test.

---

## 2. Survey

**Spatial-artifact CNN detectors.** Early work trained CNNs directly on images to separate real from generated content, first for GAN images (Wang et al., 2020). These models reach high accuracy on the generators they saw in training but often transfer poorly to different architectures.

**Frequency-domain methods.** Generated images often show unusual spectral behaviour, for example from upsampling layers. Durall et al. (2020) and Frank et al. (2020) showed that the Fourier spectrum can expose such patterns. This motivates our FFT branch and the radial spectrum plots we made for each generator.

**Noise residual and steganalysis features.** The Spatial Rich Model (SRM) of Fridrich and Kodovsky (2012) uses high-pass filters to suppress image content and keep the noise left by processing. Forensic networks commonly use these filters as a fixed first layer. We use three of the standard SRM kernels together with a plain Gaussian high-pass residual.

**Foundation-model features.** Ojha et al. (2023) showed that a linear classifier on frozen CLIP features (Radford et al., 2021) generalizes better to unseen generators than a CNN trained end to end. This is why we use a frozen CLIP backbone and include a CLIP linear probe as a baseline. We take features from several transformer layers because earlier layers keep more low-level information than the last layer.

**Benchmarks.** GenImage (Zhu et al., 2023) provides real ImageNet images and images from several generators, which makes cross-generator tests possible. The dataset we use is a 20,000-image subset of it that was prepared with a bias-controlled pipeline.

**Calibration.** Guo et al. (2017) showed that modern networks are often overconfident and that a single temperature parameter fitted on held-out data fixes most of this. We use this method.

**Where our work fits.** We combine the three families above (semantic, frequency, noise) in one gated model, and we spend as much effort on checking for shortcuts, robustness and calibration as on accuracy.

---

## 3. Dataset and Preprocessing

### 3.1 Dataset

We use the preprocessed GenImage-20K subset. The `training_manifest.csv` file gives 19,802 images after 99 matched pairs were removed because they were flagged as near-duplicates across splits. Every image is a 256x256 RGB JPEG at quality 85, made with a center crop and Lanczos resizing. Each real image has an AI partner with the same ImageNet content class.

| Split | Real | AI | Total | Generators |
|---|---|---|---|---|
| Train | 6,913 | 6,913 | 13,826 | ADM, GLIDE, SD v1.4, SD v1.5, VQDM, Wukong |
| Validation | 993 | 993 | 1,986 | same six |
| Seen test | 995 | 995 | 1,990 | same six |
| Unseen test | 1,000 | 1,000 | 2,000 | Midjourney only |
| **Total** | 9,901 | 9,901 | 19,802 | |

The data loading cell asserts that (a) no Midjourney image appears in train, validation or seen test, and (b) no original image hash appears in more than one split. We use the manifest file we were given and do not rebuild the splits ourselves, so the near-duplicate screening done by the dataset authors carries over. Because seen test and Midjourney are reserved for final reporting, all model choices were made on validation only.

### 3.2 Shortcut audit

All processed images have the same size, format and JPEG quality, so processed size or format cannot give away the label. The originals did differ, though:

- All real images were originally JPEG and all AI images were originally PNG.
- Real images are mostly 256-511 px on the native side. AI images from SD v1.4, SD v1.5 and Wukong are 512-1023 px, and Midjourney AI images are mostly 1024 px or larger. ADM, GLIDE and VQDM are almost entirely in the 256-511 bin for both classes.

To see how serious this is, we trained a gradient-boosting classifier on native width and height only, with no pixels. Its validation ROC-AUC was **1.000**. In other words, native size alone perfectly separates the classes on this dataset, and a pixel model could in principle exploit the resampling history that comes with it. Our response was to:

1. report validation metrics per native-size bin, and
2. report metrics on the subset of pairs whose real and AI native-size bins match (1,056 validation images).

The results of these checks are in Section 6.3. We treat this as a real limitation of the dataset and discuss it again in Section 7.

### 3.3 Preprocessing used in the model

All branches take the same 256x256 uint8 image from the data loader. The three input views are computed on the GPU:

| View | Computation |
|---|---|
| CLIP input | bicubic resize 256 -> 224 (antialiased), CLIP mean/std normalization |
| FFT input | grayscale (0.299R + 0.587G + 0.114B) -> `fft2` -> `fftshift` -> `log1p(abs)` -> per-image standardization, 1x256x256 |
| Noise input | residual `(x - GaussianBlur(x, 5x5, sigma=1)) * 5` (3 channels) concatenated with 3 SRM filter responses on the grayscale image (3 channels), 6x256x256 |

The noise and FFT branches use the 256x256 JPEG pixels directly and do not resize them, which follows the dataset README.

### 3.4 Training augmentation

Augmentation is applied only to the training set:

- horizontal flip with probability 0.5;
- with probability `aug_p = 0.5`, one or two (chosen at random) of the following: JPEG re-compression with quality uniform in 60-95; Gaussian blur with sigma uniform in 0.3-1.5; down-up resize with a factor uniform in 0.5-0.9 (bicubic); additive Gaussian noise with sigma uniform in 1-5 (on the 0-255 scale).

Validation and test images are clean except in the robustness study, where fixed degradations are applied (Section 6.4).

---

## 4. Methodology and Architecture

The model has three branches, a gate that fuses them, and a small classifier head.

```
256x256 image
   |-- CLIP view  -> frozen CLIP ViT-B/32 -> CLS tokens (layers 4, 8, 12) -> LayerNorm each -> MLP -> 128-d   (Branch A)
   |-- FFT view   -> ForensicCNN(1 channel)                                                  -> 128-d       (Branch B)
   '-- Noise view -> ForensicCNN(6 channels)                                                 -> 128-d       (Branch C)
                            |
        gate(concat of 3 embeddings) -> softmax weights -> weighted sum -> classifier -> logit (real / AI)
        each branch also has its own linear auxiliary head
```

### 4.1 Branch A: multi-layer CLIP

The CLIP vision encoder (`openai/clip-vit-base-patch32`, Hugging Face) is frozen. We take the CLS token from transformer layers 4, 8 and 12, apply a separate LayerNorm to each, concatenate them (3 x 768 = 2304 dimensions) and pass them through an MLP (Linear - GELU - Linear - LayerNorm) to a 128-d embedding. The encoder weights are not saved in our checkpoints and are downloaded again by name when needed.

### 4.2 Branch B: frequency branch

The FFT log-magnitude spectrum of the grayscale image goes into a small CNN ("ForensicCNN"): an input BatchNorm, then four blocks of two 3x3 convolutions each (32, 64, 128, 256 channels, with BatchNorm and GELU; the first three blocks end with 2x2 max-pooling). The output is global average pooling and global max pooling concatenated (512-d), followed by Linear and LayerNorm to 128-d.

### 4.3 Branch C: noise-residual branch

The input is six channels: the three-channel high-pass residual and three responses from fixed 5x5 SRM kernels applied to the grayscale image. The first two are the commonly used "KB" and "KV"-style kernels and the third is a second-order horizontal difference. The same ForensicCNN architecture is used, with six input channels.

### 4.4 Gated fusion

The three embeddings are concatenated and passed through a small MLP (Linear 384 -> 128, GELU, Linear 128 -> 3). A softmax over the three outputs gives per-image branch weights. The fused feature is the weighted sum of the three embeddings, and a classifier head (Linear 128 -> 64, GELU, Dropout 0.2, Linear 64 -> 1) outputs the logit. The gate weights are returned with every prediction so we can see which branch the model trusted.

### 4.5 Branch dropout and auxiliary losses

During training each branch is hidden from the gate with probability 0.15. A hidden branch is zeroed in the gate input and masked out of the softmax. If all three would be hidden for an image, one random branch is kept, so the gate always has at least one branch available. Each branch also has a linear head trained with binary cross-entropy. The total loss is

`L = BCE(main) + 0.3 * mean over branches of BCE(aux_branch)`.

The purpose of both mechanisms is to keep the gate from collapsing onto whichever branch is easiest to learn.

### 4.6 Calibration

After training we fit one temperature T on the validation logits by minimizing the binary cross-entropy of `sigmoid(z / T)` with L-BFGS. Calibrated probability is `p = sigmoid(z / T)`. The decision threshold is fixed at 0.5 because the classes are balanced. The inference function returns the label, the calibrated confidence and the branch weights.

---

## 5. Implementation Details

| Item | Setting |
|---|---|
| Hardware / framework | Kaggle GPU, PyTorch, mixed precision (AMP) |
| Seed | 42 (Python, NumPy, PyTorch) |
| Batch size | 32 (training), 64 (evaluation) |
| Epochs | 6, early stopping on validation AUC with patience 3 |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-2 |
| LR schedule | OneCycleLR, max LR 3e-4, 15% warm-up |
| Gradient clipping | max norm 1.0 |
| Embedding size | 128 for every branch |
| CLIP layers | 4, 8, 12 |
| Branch dropout / aux weight | 0.15 / 0.3 |
| Trainable parameters | 2,855,285 (CLIP is frozen) |
| Best-checkpoint rule | highest validation AUC, restored after training |
| Data loading | 2 workers, the loader only decodes JPEGs; CLIP, FFT and residual views are computed on the GPU |

Training took 6 epochs. Epoch 1 took 561 s (it included one-off start-up work) and each later epoch took about 160-175 s. The best validation AUC was reached at epoch 6, so early stopping did not trigger.

Other components in the same notebook:

- **CLIP linear probe.** Features from the three CLIP layers (2304-d) are cached once for all 17,802 non-Midjourney images, standardized, and fed to logistic regression (C = 0.05, 2,000 iterations). Midjourney was never passed through this cache.
- **Leave-one-generator-out (LOGO).** For each of the six seen generators, the probe is trained on the training images of the other five generators and tested on every image of the held-out generator.
- **Saved outputs.** `detector_full.pt`, `training_history_full.csv`, `validation_metrics.csv`, `robustness_validation.csv`, `logo_clip_probe.csv` and the cached CLIP features.

---

## 6. Results and Discussion

All numbers in this section come from the validation split (1,986 images) or, for LOGO, from the generator-wise features described above. The seen-test and Midjourney splits were not touched. The ablation and final-evaluation sections of the notebook are switched off by flags in this run (see Section 7).

### 6.1 Training curve and overall validation performance

| Epoch | Train loss (main+aux) | Val loss | Val acc | Val AUC | Val F1 |
|---|---|---|---|---|---|
| 1 | 0.6732 | 0.2672 | 0.8943 | 0.9646 | 0.8898 |
| 2 | 0.4947 | 0.2170 | 0.9129 | 0.9755 | 0.9101 |
| 3 | 0.4263 | 0.1740 | 0.9335 | 0.9833 | 0.9327 |
| 4 | 0.3808 | 0.1683 | 0.9441 | 0.9849 | 0.9435 |
| 5 | 0.3284 | 0.1491 | 0.9461 | 0.9881 | 0.9465 |
| 6 | 0.2980 | 0.1493 | 0.9491 | 0.9885 | 0.9495 |

Validation loss decreases steadily and the AUC rises each epoch, with no sign of overfitting. The train loss is higher than the validation loss because it includes the auxiliary losses and is measured under augmentation and branch dropout.

**Final validation metrics (full model)**

| Metric | Value |
|---|---|
| Accuracy | 0.9491 |
| ROC-AUC | 0.9885 |
| Average precision | 0.9880 |
| F1 | 0.9495 |
| TPR at 1% FPR | 0.7905 |
| Real-class accuracy | 0.9426 |
| AI-class accuracy | 0.9557 |

Real and AI accuracy are close, so the model is not biased toward one class at the 0.5 threshold. The TPR at 1% FPR (0.79) is much lower than the overall accuracy suggests, which means a strict false-positive budget would cost a noticeable fraction of detections.

### 6.2 Per-generator validation results

| Generator | n | Acc | AUC | Real acc | AI acc |
|---|---|---|---|---|---|
| ADM | 330 | 0.9697 | 0.9978 | 0.9394 | 1.0000 |
| GLIDE | 330 | 0.9667 | 0.9927 | 0.9636 | 0.9697 |
| SD v1.4 | 334 | 0.9641 | 0.9907 | 0.9641 | 0.9641 |
| SD v1.5 | 332 | 0.9458 | 0.9917 | 0.9337 | 0.9578 |
| VQDM | 330 | 0.9091 | 0.9679 | 0.9030 | 0.9152 |
| Wukong | 330 | 0.9394 | 0.9872 | 0.9515 | 0.9273 |

VQDM is the hardest of the seen generators, with the lowest accuracy and AUC. ADM has the highest AUC.

### 6.3 Checks for the size shortcut

| Subset | n | Acc | AUC |
|---|---|---|---|
| Native size 256-511 | 1,425 | 0.9495 | 0.9889 |
| Native size 512-1023 | 551 | 0.9474 | 0.9838 |
| Native size 1024+ | 10 | 1.0000 | undefined (only one class present) |
| Pairs with matching real/AI size bins | 1,056 | 0.9517 | 0.9885 |

Even though native size alone gives AUC 1.000 on this dataset, the detector performs about the same inside the 256-511 bin, where size carries no information about the label for most generators, and on the matched-pair subset. This suggests the model is not mainly reading image size. It does not prove that the model ignores all resampling or format history, because that history cannot be fully removed from this dataset. The 1024+ bin has only 10 validation images and is not informative.

### 6.4 Robustness (validation, fixed degradations)

| Perturbation | Acc | AUC | Real acc | AI acc |
|---|---|---|---|---|
| None | 0.9491 | 0.9885 | 0.9426 | 0.9557 |
| JPEG q95 | 0.9320 | 0.9841 | 0.9577 | 0.9063 |
| JPEG q75 | 0.9149 | 0.9758 | 0.9507 | 0.8792 |
| JPEG q50 | 0.8968 | 0.9647 | 0.9476 | 0.8459 |
| Blur sigma 1.0 | 0.9169 | 0.9761 | 0.9003 | 0.9335 |
| Blur sigma 2.0 | 0.7800 | 0.9237 | 0.6093 | 0.9507 |
| Down-up 0.5x | 0.9003 | 0.9688 | 0.8499 | 0.9507 |
| Gaussian noise sigma 5 | 0.9245 | 0.9784 | 0.9416 | 0.9074 |

AUC stays above 0.96 for every degradation except strong blur. The per-class columns show how the errors move:

- **JPEG** mainly lowers AI accuracy (0.956 to 0.846 at q50). Compression removes some of the traces of generation and the model calls more AI images real.
- **Blur and down-up resizing** push errors the other way: real accuracy falls (0.943 to 0.609 at sigma 2) while AI accuracy stays high. A blurred real photo starts to look "AI" to the model, which suggests smoothness itself is partly treated as a sign of AI generation.
- The fixed 0.5 threshold is part of the problem under blur: AUC is still 0.924 at sigma 2 while accuracy is 0.78, so the ranking holds up better than the decisions at that threshold.

### 6.5 Branch gate weights

Mean gate weights on the validation set, by true label and generator:

| Label | Generator | CLIP | FFT | Noise |
|---|---|---|---|---|
| AI | ADM | 0.098 | 0.051 | 0.851 |
| AI | GLIDE | 0.995 | 0.001 | 0.005 |
| AI | SD v1.4 | 0.996 | 0.001 | 0.004 |
| AI | SD v1.5 | 0.994 | 0.001 | 0.005 |
| AI | VQDM | 0.996 | 0.001 | 0.004 |
| AI | Wukong | 0.992 | 0.001 | 0.007 |
| Real | all six | 0.995-0.997 | 0.001-0.002 | 0.002-0.004 |

The gate trusts the CLIP branch for almost every image, and the FFT branch gets close to zero weight. The one exception is ADM AI images, where the noise branch takes 85% of the weight. So even with branch dropout and auxiliary losses, the fused model behaves mostly like a CLIP-based detector, with the noise branch used for one generator. The auxiliary heads of the FFT and noise branches give probabilities near 0.4-0.65 on the sample images we inspected, which means those branches carry only weak signal on their own. Because the branch ablations were not run, we cannot say how much FFT and noise add beyond CLIP. We do not claim an improvement from them.

### 6.6 Calibration

| | Value |
|---|---|
| Fitted temperature T | 1.353 |
| ECE before scaling (15 bins) | 0.0272 |
| ECE after scaling (15 bins) | 0.0134 |

Temperature scaling roughly halves the ECE, so confidence values are reasonable on validation. T was fitted on validation images from seen generators, so confidence on images from new generators may be less reliable, and we have not measured it.

### 6.7 CLIP linear-probe baseline and leave-one-generator-out

On validation, the logistic regression on cached multi-layer CLIP features gets:

| Model (validation) | Acc | AUC | AP | F1 | TPR at 1% FPR |
|---|---|---|---|---|---|
| CLIP linear probe (frozen features, logistic regression) | 0.9376 | 0.9858 | 0.9857 | 0.9374 | 0.7432 |
| Three-branch gated model | 0.9491 | 0.9885 | 0.9880 | 0.9495 | 0.7905 |

The full model is about 1.2 points better in accuracy and 0.003 in AUC. The two systems share the same CLIP features, and we used one run and one seed, so this small gap should not be read as a firm improvement.

LOGO results for the CLIP probe (trained on the other five generators, tested on all images of the held-out one):

| Held-out generator | n | AUC | Acc | Real acc | AI acc |
|---|---|---|---|---|---|
| ADM | 2,968 | 0.7855 | 0.6513 | 0.9468 | 0.3558 |
| GLIDE | 2,958 | 0.9512 | 0.8627 | 0.9466 | 0.7789 |
| SD v1.4 | 2,988 | 0.9815 | 0.9354 | 0.9364 | 0.9344 |
| SD v1.5 | 2,970 | 0.9853 | 0.9414 | 0.9428 | 0.9401 |
| VQDM | 2,938 | 0.8086 | 0.6113 | 0.9632 | 0.2594 |
| Wukong | 2,980 | 0.9603 | 0.8842 | 0.9470 | 0.8215 |

Generalization is uneven. SD v1.4 and v1.5 are almost the same model, so each is easy when the other is in training and these two rows are optimistic. Wukong and GLIDE transfer well. ADM and VQDM transfer badly: AUC drops to 0.79-0.81 and the AI accuracy falls to 0.36 and 0.26, so most of their images are called real. Both are also the generators whose images share the same native size range as the real images, so the CLIP probe has fewer extra cues to work with. We think this explains part of the gap but did not test it directly. These LOGO numbers are for the CLIP probe only, not for the full three-branch model.

### 6.8 Inference demo

The notebook includes a `predict_image` function that applies the same preprocessing as the dataset (center crop, Lanczos resize to 256, JPEG q85) and returns the verdict, calibrated confidence, per-branch gate weights and per-branch AI probabilities. On a sample of four validation images it gave three correct verdicts and one miss (an AI image from GLIDE called real with 98% confidence), which shows that confidence is not a guarantee of correctness.

---

## 7. Limitations

1. **Results are validation-only.** The seen-test and Midjourney evaluations are gated behind a flag so that they can be run once after all choices are frozen. The numbers in this report therefore do not include a held-out test of either kind, and we make no claim about performance on Midjourney or on any other unseen generator for the full model.
2. **No ablation numbers.** The CLIP-only, FFT-only, noise-only and FFT+noise variants were not trained in this run, and we did not train a ResNet-50 baseline. Our only baseline comparison is the CLIP linear probe.
3. **Single seed.** All results use seed 42, so we have no confidence intervals or standard deviations.
4. **Branch usage.** The gate puts almost all weight on CLIP, so the FFT and noise branches contribute little in practice (Section 6.5).
5. **Dataset shortcut.** Original format and native size separate the classes perfectly (metadata AUC 1.000). Our per-bin and matched-pair checks are reassuring but cannot remove the effect completely. Results are specific to this dataset and its 256x256 JPEG q85 processing.
6. **Unseen generators.** Detectors usually lose accuracy on generators they were not trained on. The LOGO results show this clearly for ADM and VQDM.
7. **Probabilistic output.** The detector gives a probability, not proof. Even a calibrated 98% confidence can be wrong, as the inference demo showed.

---

## 8. Conclusion and Future Work

We built and trained a three-branch AI-image detector (multi-layer CLIP, FFT spectrum, SRM/high-pass noise residual) with gated fusion, branch dropout, auxiliary losses and temperature-scaled confidence, using six generators for training. On validation it reaches 94.91% accuracy and 0.9885 AUC, and it keeps AUC above 0.96 under JPEG q50, noise and half-resolution resizing. Blur is the main weakness, because it makes real images look synthetic. Calibration works on validation (ECE 0.0272 to 0.0134). The shortcut audit shows that this dataset has a strong native-size confound, and the checks we ran suggest, but do not prove, that the model is not relying on it. In practice the gate relies on CLIP, so we cannot claim that the frequency and noise branches help. The cross-generator experiment with the CLIP probe shows that generalization is uneven.

**Future work**

- Run the held-out seen-test and Midjourney evaluations once, with the full model and the CLIP probe, and report them alongside this validation study.
- Run the branch ablations and a ResNet-50 baseline, and repeat the main model with several seeds to get confidence intervals.
- Try ways to make the FFT and noise branches useful, such as pre-training them alone, using a larger branch dropout, or giving them a minimum gate weight.
- Add blur to training at larger sigma values, and tune the decision threshold under blur.
- Test on data where real and AI images have matched formats and native sizes, and on images that have been through social-media style re-encoding.
- Check calibration on unseen generators.

---

## References

1. A. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021. (CLIP)
2. M. Zhu et al. "GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image." NeurIPS Datasets and Benchmarks, 2023.
3. U. Ojha, Y. Li, Y. J. Lee. "Towards Universal Fake Image Detectors that Generalize Across Generative Models." CVPR, 2023.
4. S.-Y. Wang, O. Wang, R. Zhang, A. Owens, A. A. Efros. "CNN-Generated Images Are Surprisingly Easy to Spot... for Now." CVPR, 2020.
5. R. Durall, M. Keuper, J. Keuper. "Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions." CVPR, 2020.
6. J. Frank et al. "Leveraging Frequency Analysis for Deep Fake Image Recognition." ICML, 2020.
7. J. Fridrich, J. Kodovsky. "Rich Models for Steganalysis of Digital Images." IEEE Transactions on Information Forensics and Security, 2012. (SRM)
8. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. "On Calibration of Modern Neural Networks." ICML, 2017.
9. I. Loshchilov, F. Hutter. "Decoupled Weight Decay Regularization." ICLR, 2019. (AdamW)
10. L. N. Smith, N. Topin. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." 2018. (One-cycle schedule)
11. A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS, 2019.

---

## Contribution of Team Members

> Draft split. Edit this table so it matches what each person actually did before submitting.

| Roll no. | Name | Contribution |
|---|---|---|
| CS24B1007 | Maneesha | Literature survey and report writing; dataset audit and sample/spectrum visualizations |
| CS24B1014 | Roshitha | Data pipeline, manifest checks, train-time augmentations and robustness evaluation |
| CS24B1026 | Sathwika | CLIP feature caching, linear-probe baseline and leave-one-generator-out experiment; calibration (temperature scaling, ECE) |
| CS24B1070 | Havila | Model architecture (three branches, gated fusion, branch dropout, auxiliary losses), training loop, Kaggle runs and notebook integration |
