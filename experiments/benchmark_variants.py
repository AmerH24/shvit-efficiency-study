"""
Benchmark inference throughput for all reproduction and extension variants.

Absolute throughput depends on hardware. The meaningful comparison in
this study is the relative throughput obtained under the same device,
resolution, batch size, and software environment.
"""

import csv
import os
import time

import torch
from timm.models import create_model

# Import model package so custom timm registrations execute.
import model  # noqa: F401


MODELS = [
    "shvit_s1_ratio_1_8",
    "shvit_s1_ratio_default",
    "shvit_s1_ratio_1_2",
    "shvit_s1_progressive",
]

RESOLUTION = 224
BATCH_SIZE = 64
WARMUP_RUNS = 20
TIMED_RUNS = 100


def benchmark(model_name, device):
    model_instance = create_model(
        model_name,
        num_classes=100,
        pretrained=False,
        distillation=False,
    ).to(device)

    model_instance.eval()

    inputs = torch.randn(
        BATCH_SIZE,
        3,
        RESOLUTION,
        RESOLUTION,
        device=device,
    )

    with torch.no_grad():
        # Warm-up lets CUDA kernels and caches stabilize before timing.
        for _ in range(WARMUP_RUNS):
            model_instance(inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(TIMED_RUNS):
            model_instance(inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    total_images = BATCH_SIZE * TIMED_RUNS
    throughput = total_images / elapsed
    latency_ms = (elapsed / TIMED_RUNS) * 1000

    parameters = sum(
        parameter.numel()
        for parameter in model_instance.parameters()
    )

    return {
        "model": model_name,
        "parameters": parameters,
        "throughput_images_per_second": throughput,
        "batch_latency_ms": latency_ms,
    }


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Benchmark device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []

    for model_name in MODELS:
        print(f"\nBenchmarking {model_name}...")

        result = benchmark(model_name, device)
        rows.append(result)

        print(
            f"Throughput: "
            f"{result['throughput_images_per_second']:.2f} images/s"
        )

        print(
            f"Batch latency: "
            f"{result['batch_latency_ms']:.2f} ms"
        )

        print(
            f"Parameters: "
            f"{result['parameters']:,}"
        )

    os.makedirs("results", exist_ok=True)

    output_path = "results/throughput.csv"

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark results to {output_path}")


if __name__ == "__main__":
    main()