"""Plots for the interactive click benchmark: NoC bars, IoU-vs-clicks curves, per-class NoC."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_report(report: dict, out_dir: str, iou_targets=(0.85, 0.90),
                primary_target: float = 0.90) -> None:
    os.makedirs(out_dir, exist_ok=True)
    models = list(report.keys())
    if not models:
        print("[interactive] empty report; nothing to plot.")
        return

    target_keys = [str(int(round(t * 100))) for t in sorted(iou_targets)]
    pkey = str(int(round(primary_target * 100)))

    # --- 1. NoC@t per model (grouped bars) ---
    plt.figure(figsize=(max(8, len(models) * 1.2), 6))
    x = np.arange(len(models))
    width = 0.8 / max(1, len(target_keys))
    for i, tk in enumerate(target_keys):
        y = [report[m]["NoC"].get(tk, 0.0) for m in models]
        plt.bar(x + i * width, y, width, label=f"NoC@{tk}")
    plt.xticks(x + width * (len(target_keys) - 1) / 2, models, rotation=30, ha="right")
    plt.ylabel("Mean clicks to reach IoU target")
    plt.title("Click efficiency (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "NoC_comparison.png"), dpi=200)
    plt.close()

    # --- 2. IoU vs #clicks curve ---
    plt.figure(figsize=(9, 6))
    for m in models:
        curve = report[m]["iou_curve"]
        xs = np.arange(1, len(curve) + 1)
        plt.plot(xs, curve, marker="o", markersize=3, label=m)
    plt.axhline(primary_target, color="gray", linestyle="--", linewidth=1,
                label=f"target {pkey}%")
    plt.xlabel("Number of clicks")
    plt.ylabel("Mean IoU")
    plt.title("Mask quality vs. number of clicks")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "IoU_vs_clicks.png"), dpi=200)
    plt.close()

    # --- 3. Per-class NoC@primary (grouped bars) ---
    all_classes = sorted({c for m in models for c in report[m].get("per_class", {})})
    if all_classes:
        plt.figure(figsize=(max(9, len(all_classes) * 1.1), 6))
        x = np.arange(len(all_classes))
        width = 0.8 / max(1, len(models))
        metric = f"NoC@{pkey}"
        for i, m in enumerate(models):
            pc = report[m].get("per_class", {})
            y = [pc.get(c, {}).get(metric, 0.0) for c in all_classes]
            plt.bar(x + i * width, y, width, label=m)
        plt.xticks(x + width * (len(models) - 1) / 2, all_classes, rotation=30, ha="right")
        plt.ylabel(f"Mean clicks to IoU {pkey}%")
        plt.title(f"Per-class click efficiency (NoC@{pkey})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_class_NoC.png"), dpi=200)
        plt.close()

    print(f"Interactive plots saved to {out_dir}")
