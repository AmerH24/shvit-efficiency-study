import argparse
import time

import torch

from model.build import shvit_s1


def benchmark(model, device, batch_size=32, resolution=224, warmup=20, runs=100):
    model.eval()

    x = torch.randn(
        batch_size,
        3,
        resolution,
        resolution,
        device=device
    )

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(runs):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    total_images = batch_size * runs
    throughput = total_images / elapsed
    latency_ms = elapsed / runs * 1000

    return throughput, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = shvit_s1(num_classes=100).to(device)

    throughput, latency = benchmark(
        model,
        device,
        batch_size=args.batch_size,
        resolution=args.resolution
    )

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Throughput: {throughput:.2f} images/s")
    print(f"Latency per batch: {latency:.2f} ms")


if __name__ == "__main__":
    main()