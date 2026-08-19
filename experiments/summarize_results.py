"""
Extract the best validation accuracy from each SHViT experiment.

main.py writes one JSON object per epoch into log.txt.
This script converts those logs into one compact CSV table.
"""

import csv
import json
import os


EXPERIMENTS = {
    "ratio_1_8": "results/ratio_1_8/log.txt",
    "ratio_default": "results/ratio_default/log.txt",
    "ratio_1_2": "results/ratio_1_2/log.txt",
    "progressive": "results/progressive/log.txt",
}


def read_best_result(log_path):
    best = None

    with open(log_path, "r") as file:
        for line in file:
            row = json.loads(line)

            if best is None or row["test_acc1"] > best["test_acc1"]:
                best = row

    return best


def main():
    rows = []

    for experiment_name, log_path in EXPERIMENTS.items():

        if not os.path.exists(log_path):
            print(f"Missing: {log_path}")
            continue

        best = read_best_result(log_path)

        rows.append({
            "variant": experiment_name,
            "best_epoch": best["epoch"],
            "top1_accuracy": best["test_acc1"],
            "top5_accuracy": best["test_acc5"],
            "parameters": best["n_parameters"],
        })

    if not rows:
        print("No completed experiment logs found.")
        return

    output_path = "results/accuracy.csv"

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nBest validation results:")

    for row in rows:
        print(
            f"{row['variant']:<18} "
            f"Top-1={row['top1_accuracy']:.2f}% "
            f"epoch={row['best_epoch']}"
        )

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()