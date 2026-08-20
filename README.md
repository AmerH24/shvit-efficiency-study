# SHViT Efficiency Study

A scaled reproduction and extension of **SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design** (CVPR 2024), focused on the accuracy–efficiency trade-off of partial-channel Single-Head Self-Attention (SHSA).

This project was completed as an individual final project for the **LebNet Tech Fellows Technical AI Bootcamp**.

> This repository builds on the official SHViT implementation by Seokju Yun and Youngmin Ro. The original implementation is preserved and credited; the experiment variants, scaled reproduction workflow, benchmarking utilities, result analysis, and progressive attention-allocation extension were added for this study.

## Project Question

Efficient Vision Transformers need global context, but attention can be expensive in practice because computation is only part of the cost: memory movement, reshaping, normalization, and large intermediate feature maps also contribute to latency.

SHViT addresses this through two ideas:

1. **Macro-level spatial efficiency**  
   Aggressively reduce redundant early spatial tokens using a 16×16 overlapping patchification stem and a three-stage architecture.

2. **Micro-level attention efficiency**  
   Replace potentially redundant multi-head attention with **Single-Head Self-Attention (SHSA)** and apply it to only a subset of the feature channels.

This project focuses on the second idea:

> How does changing the fraction of channels processed by SHSA affect classification accuracy and inference throughput?

A small extension also asks:

> Instead of using approximately the same partial-attention ratio across later stages, can attention capacity be shifted toward the deeper stage without increasing the overall attention-channel budget?

---

## SHSA in One Picture

For an input feature tensor with `C` channels, SHSA splits the channels into two groups:

```text
Input features
      |
      +------------------------+
      |                        |
      v                        v
Attention channels        Bypass channels
   Cp = rC                  C - Cp
      |                        |
Single-head attention       Identity
      |                        |
      +-----------+------------+
                  |
             Concatenate
                  |
          Output projection
                  |
                Output
```

Only `Cp = rC` channels receive global self-attention. The remaining channels bypass attention, after which all channels are mixed again through the output projection.

The original paper uses approximately:

```text
r = 1 / 4.67 ≈ 21%
```

as its default partial ratio.

---

## Reproduction

The original SHViT paper evaluates partial ratios of:

- `1/8`
- `1/4.67` — default SHViT setting
- `1/2`

and reports that increasing attention-channel capacity improves accuracy only up to a point while reducing inference throughput.

A full ImageNet-1K reproduction was outside the available compute/time budget, so this project performs a **scaled reproduction** using SHViT-S1 and CIFAR-100 while preserving the controlled comparison between partial ratios.

### Controlled Variants

| Variant | Stage 2 SHSA channels | Stage 3 SHSA channels |
|---|---:|---:|
| 1/8 | 28 / 224 | 40 / 320 |
| Default ≈1/4.67 | 48 / 224 | 68 / 320 |
| 1/2 | 112 / 224 | 160 / 320 |

All variants preserve the SHViT-S1 depth and embedding dimensions. Only the number of channels processed by SHSA changes.

---

## Extension: Progressive Attention Allocation

The paper uses approximately the same partial-attention ratio across the attention stages.

I tested a small modification:

```text
Original:
Stage 2 → 48 attention channels
Stage 3 → 68 attention channels

Progressive:
Stage 2 → 32 attention channels
Stage 3 → 80 attention channels
```

The motivation was to ask whether global-attention capacity is more useful after the representation becomes deeper and more semantically concentrated.

Importantly, this does **not simply add more attention capacity**.

Approximate attention-channel/block budget:

```text
Original:
4 × 48 + 5 × 68 = 532

Progressive:
4 × 32 + 5 × 80 = 528
```

The extension therefore redistributes approximately the same budget toward the later stage.

---

## Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 |
| Backbone | SHViT-S1 |
| Input resolution | 224 × 224 |
| Training epochs | 10 |
| Batch size | 64 |
| Optimizer | AdamW |
| Scheduler | Cosine |
| Evaluation | Top-1 and Top-5 accuracy |
| Efficiency metric | GPU inference throughput |
| Training environment | Google Colab GPU |
| Number of seeds | 1 |

The original paper trained on ImageNet-1K for 300 epochs with a much larger total batch size and benchmarked GPU throughput on an NVIDIA A100. Therefore, the absolute numbers here should **not** be compared directly with the paper's ImageNet results. The purpose is to test whether the relative behavior of the partial-attention variants persists in a constrained setting.

---

## Results

| Variant | Top-1 | Top-5 | Parameters | Throughput |
|---|---:|---:|---:|---:|
| **1/8** | **25.05%** | 55.62% | **6.013M** | **3361.52 img/s** |
| Default ≈1/4.67 | 24.47% | 54.85% | 6.042M | 3314.32 img/s |
| 1/2 | 24.89% | **55.68%** | 6.214M | 3212.21 img/s |
| Progressive extension | 24.53% | 54.79% | 6.046M | 3279.95 img/s |

![Accuracy-throughput trade-off](results/accuracy_throughput.png)

### What reproduced?

The **efficiency trend** reproduced clearly:

```text
1/8        3361.52 img/s
Default    3314.32 img/s
1/2        3212.21 img/s
```

Processing more channels with SHSA reduced throughput.

Relative to the default configuration:

- `1/8` was approximately **1.42% faster**
- `1/2` was approximately **3.08% slower**

This is directionally consistent with the original SHViT ablation.

### What did not reproduce?

The original paper selects the `1/4.67` configuration as the best overall accuracy–speed trade-off after full ImageNet training.

In this scaled experiment, the ordering was different: the `1/8` variant achieved both the highest Top-1 accuracy and highest throughput.

The accuracy differences are small, however, and these models were trained for only 10 epochs with one random seed. The result should therefore **not** be interpreted as evidence that `1/8` is universally superior.

Instead, it suggests that the optimal partial ratio may depend on the dataset, training budget, or convergence regime.

---

## Extension Result

The progressive variant achieved:

```text
Top-1:
24.53% vs 24.47% default

Throughput:
3279.95 vs 3314.32 images/s default
```

This corresponds to only a **+0.06 percentage-point** Top-1 change while reducing throughput by approximately **1.04%**.

Therefore, under this experimental setup, shifting attention capacity toward Stage 3 did **not** improve the accuracy–efficiency trade-off.

This is still informative: simply assuming that deeper semantic features deserve a larger share of global attention was not supported by this experiment.

---

## Key Takeaways

1. **Attention cost depends on how much of the representation participates in attention.**
2. Increasing the SHSA partial ratio produced a measurable throughput penalty in the scaled experiment.
3. More attention capacity did not produce a clear accuracy advantage within the limited 10-epoch training budget.
4. The progressive stage-wise allocation did not outperform the original allocation.
5. Hardware-aware architecture design involves more than FLOP counts: data movement and tensor organization can materially affect inference speed.
6. Reproducing a paper does not necessarily mean matching its absolute numbers; under constrained compute, carefully controlled relative experiments can still test an architectural claim.

---

## Repository Structure

```text
.
├── data/
│   ├── datasets.py
│   ├── samplers.py
│   └── threeaugment.py
│
├── model/
│   ├── build.py
│   ├── shvit.py
│   └── variants.py
│
├── experiments/
│   ├── benchmark_variants.py
│   ├── summarize_results.py
│   ├── trace_model.py
│   └── training scripts
│
├── results/
│   ├── accuracy.csv
│   ├── throughput.csv
│   ├── accuracy_throughput.png
│   ├── ratio_1_8/
│   ├── ratio_default/
│   ├── ratio_1_2/
│   └── progressive/
│
├── docs/
│   └── UPSTREAM_README.md
│
├── main.py
├── engine.py
├── losses.py
├── utils.py
├── requirements.txt
└── LICENSE
```

---

## Running the Variants

Example:

```bash
python main.py \
    --model shvit_s1_ratio_default \
    --data-set CIFAR \
    --data-path datasets/cifar100 \
    --input-size 224 \
    --epochs 10 \
    --batch-size 64 \
    --num_workers 2 \
    --no-repeated-aug \
    --no-model-ema \
    --warmup-epochs 1 \
    --cooldown-epochs 0 \
    --mixup 0 \
    --cutmix 0 \
    --output_dir results/ratio_default
```

Available experimental models:

```text
shvit_s1_ratio_1_8
shvit_s1_ratio_default
shvit_s1_ratio_1_2
shvit_s1_progressive
```

Summarize completed training runs:

```bash
python experiments/summarize_results.py
```

Benchmark all variants:

```bash
python experiments/benchmark_variants.py
```

---

## Attribution

This project is based on:

**Seokju Yun and Youngmin Ro,  
“SHViT: Single-Head Vision Transformer with Memory Efficient Macro Design,” CVPR 2024.**

Official implementation:  
https://github.com/ysj9909/SHViT

The original SHViT code is distributed under the MIT License. Original source files and licensing information are retained in this repository.

The experimental variants, CIFAR-100 reproduction workflow, benchmarking scripts, result analysis, and progressive stage-wise attention allocation were added as part of this project.

---

## Limitations and Future Work

The main limitation is the intentionally reduced experimental budget. The models were trained for 10 epochs on CIFAR-100 with one seed, whereas the original work uses a substantially larger ImageNet-1K training regime.

Future work could include:

- training to convergence;
- repeating experiments across multiple random seeds;
- reproducing the SHSA-vs-MHSA ablation directly;
- testing whether optimal partial ratios change with dataset complexity;
- profiling memory consumption in addition to throughput;
- testing dynamic or learned stage-wise attention allocation.

---

## Acknowledgment

Developed as an individual final project for the **LebNet Tech Fellows Technical AI Bootcamp**.