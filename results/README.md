# Experimental Results

This directory contains outputs from a scaled reproduction of the
partial-channel attention ablation in **SHViT: Single-Head Vision
Transformer with Memory Efficient Macro Design**.

## Experimental setup

- Dataset: CIFAR-100
- Input resolution: 128 × 128
- Backbone: SHViT-S1
- Training epochs: 15
- Optimizer: AdamW
- Evaluation metric: Top-1 classification accuracy
- Efficiency metric: GPU inference throughput (images/second)

The original paper trains on ImageNet-1K for 300 epochs. This project
uses a substantially smaller training regime because of compute and
time constraints. Therefore, absolute accuracy values are not intended
to reproduce the paper's ImageNet numbers. Instead, the experiment
tests whether the relative accuracy-efficiency behavior of different
partial-attention configurations persists at smaller scale.

## Reproduction variants

| Variant | Stage 2 attention channels | Stage 3 attention channels | Purpose |
|---|---:|---:|---|
| 1/8 | 28 / 224 | 40 / 320 | Small partial ratio |
| Default | 48 / 224 | 68 / 320 | Approx. 1/4.67 SHViT default |
| 1/2 | 112 / 224 | 160 / 320 | Large partial ratio |

## Extension

The progressive variant allocates attention channels differently across
the two attention stages:

| Stage | Original | Progressive |
|---|---:|---:|
| Stage 2 | 48 | 32 |
| Stage 3 | 68 | 80 |

This tests whether shifting global-attention capacity toward later,
more semantically dense feature representations can preserve or improve
the accuracy-efficiency trade-off.

Final measured values are stored in:

- `accuracy.csv`
- `throughput.csv`