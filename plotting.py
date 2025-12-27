from __future__ import annotations

"""
Matplotlib plotting helpers for multiverse simulations.
"""

from typing import Dict, List
import os

import matplotlib.pyplot as plt


def _ensure_plots_dir(outfile: str) -> None:
    outdir = os.path.dirname(outfile) or "."
    os.makedirs(outdir, exist_ok=True)


def plot_W(
    time: List[float],
    W_series: Dict[int, List[float]],
    labels: Dict[int, str],
    title: str,
    outfile: str,
) -> None:
    """
    Plot observer-weighted fractions W_i(T) vs time.
    - One line per vacuum_id
    - Include legend, xlabel 'Time', ylabel 'Observer-weighted share W_i'
    - Save to outfile (PNG)
    """
    _ensure_plots_dir(outfile)

    plt.figure(figsize=(8, 5))
    for vid in sorted(W_series.keys()):
        plt.plot(time, W_series[vid], label=labels.get(vid, f"Vacuum {vid}"))

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Observer-weighted share W_i")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_population(
    time: List[float],
    total_patches: List[int],
    title: str,
    outfile: str,
) -> None:
    """
    Plot total patch count vs time.
    - y-axis on log scale if possible
    - xlabel 'Time', ylabel 'Total patches (log scale)'
    - Save to outfile
    """
    _ensure_plots_dir(outfile)

    plt.figure(figsize=(8, 5))
    plt.plot(time, total_patches, label="Total patches")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Total patches (log scale)")

    # Log scale only if there are positive values
    if any(p > 0 for p in total_patches):
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_bb_fraction(
    time: List[float],
    bb_fraction: List[float],
    title: str,
    outfile: str,
) -> None:
    """
    Plot Boltzmann-brain fraction vs time.
    - xlabel 'Time', ylabel 'BB fraction'
    - Save to outfile
    """
    _ensure_plots_dir(outfile)

    plt.figure(figsize=(8, 5))
    plt.plot(time, bb_fraction, label="BB fraction")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("BB fraction")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

