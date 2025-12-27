from __future__ import annotations

"""
CLI entrypoint and experiment definitions for the reproducing multiverse toy model.

Usage example:
    python experiments.py --experiment baseline --topology smooth --init concentrated

Outputs:
    - results/ CSVs (time series + summary tables)
    - plots/ PNGs
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from multiverse_model import (
    Simulation,
    SimulationConfig,
    SimulationResult,
    make_cluster_topology,
    make_rugged_topology,
    make_smooth_topology,
    override_vacua_params,
)
from plotting import plot_W, plot_population, plot_bb_fraction


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _labels_from_vacua(vacua) -> Dict[int, str]:
    return {v.vacuum_id: (v.label or f"Vacuum {v.vacuum_id}") for v in vacua}


def _make_checkpoints(T_max: float) -> List[float]:
    base = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    return [float(t) for t in base if float(t) <= float(T_max)]


def _build_topology(topology: str):
    if topology == "smooth":
        return make_smooth_topology()
    if topology == "rugged":
        return make_rugged_topology()
    if topology == "cluster":
        return make_cluster_topology()
    raise ValueError(f"Unknown topology={topology}")


def _append_summary_statistics_row(row: Dict[str, object], outfile: str) -> None:
    _ensure_dir(os.path.dirname(outfile))

    df_row = pd.DataFrame([row])
    if os.path.exists(outfile):
        prev = pd.read_csv(outfile)
        out = pd.concat([prev, df_row], ignore_index=True)
    else:
        out = df_row
    out.to_csv(outfile, index=False)


def _result_to_dataframe(result: SimulationResult, include_bb: bool) -> pd.DataFrame:
    data: Dict[str, List[object]] = {
        "time": [float(t) for t in result.times],
        "total_patches": [int(x) for x in result.total_patches],
    }
    for vid, series in sorted(result.W_series.items()):
        data[f"W_{vid}"] = [float(x) for x in series]

    if include_bb:
        assert result.BB_weight is not None
        assert result.struct_weight is not None
        assert result.BB_fraction is not None
        data["BB_weight"] = [float(x) for x in result.BB_weight]
        data["struct_weight"] = [float(x) for x in result.struct_weight]
        data["BB_fraction"] = [float(x) for x in result.BB_fraction]

    return pd.DataFrame(data)


def _final_snapshot(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    out = {
        "final_time": float(last["time"]),
        "final_total_patches": float(last["total_patches"]),
        "final_W_0": float(last.get("W_0", 0.0)),
        "final_W_1": float(last.get("W_1", 0.0)),
        "final_W_2": float(last.get("W_2", 0.0)),
    }
    if "BB_fraction" in df.columns:
        out["final_BB_fraction"] = float(last["BB_fraction"])
    else:
        out["final_BB_fraction"] = float("nan")
    return out


def _run_single(
    *,
    experiment: str,
    topology: str,
    init_mode: str,
    seed: int,
    vacua_and_T: Tuple,
    T_max: float,
    max_events: int,
    N_max: int,
    log_interval_events: int,
    checkpoints: List[float],
    n_initial: int,
    initial_vacuum_id: Optional[int],
    include_bb_metrics: bool,
    results_csv: str,
    W_plot: str,
    pop_plot: str,
    bb_plot: Optional[str] = None,
) -> pd.DataFrame:
    vacua, T = vacua_and_T
    config = SimulationConfig(
        vacua=vacua,
        transition_matrix=T,
        n_initial=n_initial,
        initial_mode=init_mode,
        initial_vacuum_id=initial_vacuum_id,
        T_max=float(T_max),
        max_events=int(max_events),
        N_max=int(N_max),
        log_interval_events=int(log_interval_events),
        checkpoints=checkpoints,
        seed=int(seed),
    )
    sim = Simulation(config)
    result = sim.run(include_bb_metrics=include_bb_metrics)
    df = _result_to_dataframe(result, include_bb=include_bb_metrics)

    _ensure_dir(os.path.dirname(results_csv))
    df.to_csv(results_csv, index=False)

    labels = _labels_from_vacua(vacua)
    plot_W(
        time=df["time"].tolist(),
        W_series={0: df["W_0"].tolist(), 1: df["W_1"].tolist(), 2: df["W_2"].tolist()},
        labels=labels,
        title=f"{experiment}: W_i(T) ({topology}, {init_mode})",
        outfile=W_plot,
    )
    plot_population(
        time=df["time"].tolist(),
        total_patches=df["total_patches"].tolist(),
        title=f"{experiment}: total patches ({topology}, {init_mode})",
        outfile=pop_plot,
    )

    if include_bb_metrics:
        if bb_plot is None:
            raise ValueError("bb_plot must be provided when include_bb_metrics=True")
        plot_bb_fraction(
            time=df["time"].tolist(),
            bb_fraction=df["BB_fraction"].tolist(),
            title=f"{experiment}: BB fraction ({topology}, {init_mode})",
            outfile=bb_plot,
        )

    return df


def run_experiment(experiment: str, topology: str, init_mode: str, seed: int) -> None:
    """
    Run a single experiment scenario specified by (experiment, topology, init_mode).
    Writes per-run CSV + plots, and appends summary stats to results/summary_statistics.csv.
    """
    np.random.seed(seed)

    _ensure_dir("results")
    _ensure_dir("plots")

    # Shared defaults
    if experiment in ("baseline", "intelligence"):
        T_max = 200.0
        max_events = 50_000
        N_max = 100_000
    elif experiment == "bb":
        T_max = 1000.0
        max_events = 100_000
        N_max = 200_000
    else:
        raise ValueError(f"Unknown experiment={experiment}")

    log_interval_events = 100
    checkpoints = _make_checkpoints(T_max)

    # Initial conditions
    n_initial = 100
    if init_mode == "concentrated":
        initial_vacuum_id = 0
    elif init_mode == "uniform":
        initial_vacuum_id = None
    else:
        raise ValueError(f"Unknown init_mode={init_mode}")

    # Validate supported topology per experiment
    if experiment == "baseline" and topology not in ("smooth", "rugged"):
        raise ValueError("baseline experiment supports only topologies: smooth, rugged")
    if experiment in ("intelligence", "bb") and topology not in ("smooth", "cluster"):
        raise ValueError(f"{experiment} experiment supports only topologies: smooth, cluster")

    # Build topology and override vacuum params per experiment
    base_vacua, T = _build_topology(topology)

    if experiment == "baseline":
        vacua = override_vacua_params(
            base_vacua,
            alpha=[1.0, 1.0, 1.0],
            o_bb=[0.0, 0.0, 0.0],
            t_bb=[float(np.inf), float(np.inf), float(np.inf)],
        )
        include_bb = False
    elif experiment == "intelligence":
        vacua = override_vacua_params(
            base_vacua,
            alpha=[1.0, 2.0, 5.0],
            o_bb=[0.0, 0.0, 0.0],
            t_bb=[float(np.inf), float(np.inf), float(np.inf)],
        )
        include_bb = False
    else:  # bb
        vacua = override_vacua_params(
            base_vacua,
            alpha=[1.0, 2.0, 5.0],
            o_bb=[0.0, 0.0, 1e-4],
            t_bb=[float(np.inf), float(np.inf), 200.0],
        )
        include_bb = True

    vacua_and_T = (vacua, T)

    # Output paths
    if experiment == "baseline":
        results_csv = f"results/results_baseline_{topology}_{init_mode}.csv"
        W_plot = f"plots/W_baseline_{topology}_{init_mode}.png"
        pop_plot = f"plots/pop_baseline_{topology}_{init_mode}.png"
        df = _run_single(
            experiment="baseline",
            topology=topology,
            init_mode=init_mode,
            seed=seed,
            vacua_and_T=vacua_and_T,
            T_max=T_max,
            max_events=max_events,
            N_max=N_max,
            log_interval_events=log_interval_events,
            checkpoints=checkpoints,
            n_initial=n_initial,
            initial_vacuum_id=initial_vacuum_id,
            include_bb_metrics=False,
            results_csv=results_csv,
            W_plot=W_plot,
            pop_plot=pop_plot,
        )
    elif experiment == "intelligence":
        results_csv = f"results/results_intel_{topology}_{init_mode}.csv"
        W_plot = f"plots/W_intel_{topology}_{init_mode}.png"
        pop_plot = f"plots/pop_intel_{topology}_{init_mode}.png"
        df = _run_single(
            experiment="intelligence",
            topology=topology,
            init_mode=init_mode,
            seed=seed,
            vacua_and_T=vacua_and_T,
            T_max=T_max,
            max_events=max_events,
            N_max=N_max,
            log_interval_events=log_interval_events,
            checkpoints=checkpoints,
            n_initial=n_initial,
            initial_vacuum_id=initial_vacuum_id,
            include_bb_metrics=False,
            results_csv=results_csv,
            W_plot=W_plot,
            pop_plot=pop_plot,
        )

        # Also run baseline counterpart for comparison and write summary_intelligence_vs_baseline.csv
        base_vacua2, T2 = _build_topology(topology)
        vacua_baseline = override_vacua_params(
            base_vacua2,
            alpha=[1.0, 1.0, 1.0],
            o_bb=[0.0, 0.0, 0.0],
            t_bb=[float(np.inf), float(np.inf), float(np.inf)],
        )
        df_baseline = _run_single(
            experiment="baseline",
            topology=topology,
            init_mode=init_mode,
            seed=seed,
            vacua_and_T=(vacua_baseline, T2),
            T_max=200.0,
            max_events=50_000,
            N_max=100_000,
            log_interval_events=log_interval_events,
            checkpoints=_make_checkpoints(200.0),
            n_initial=n_initial,
            initial_vacuum_id=initial_vacuum_id,
            include_bb_metrics=False,
            results_csv=f"results/results_baseline_{topology}_{init_mode}.csv",
            W_plot=f"plots/W_baseline_{topology}_{init_mode}.png",
            pop_plot=f"plots/pop_baseline_{topology}_{init_mode}.png",
        )

        rows = []
        for exp_name, dff in [("baseline", df_baseline), ("intelligence", df)]:
            snap = _final_snapshot(dff)
            rows.append(
                {
                    "experiment": exp_name,
                    "topology": topology,
                    "init": init_mode,
                    "final_time": snap["final_time"],
                    "final_W_0": snap["final_W_0"],
                    "final_W_1": snap["final_W_1"],
                    "final_W_2": snap["final_W_2"],
                    "final_total_patches": snap["final_total_patches"],
                }
            )
        pd.DataFrame(rows).to_csv("summary_intelligence_vs_baseline.csv", index=False)
    else:
        results_csv = f"results/results_bb_{topology}_{init_mode}.csv"
        W_plot = f"plots/W_bb_{topology}_{init_mode}.png"
        pop_plot = f"plots/pop_bb_{topology}_{init_mode}.png"
        bb_plot = f"plots/bb_fraction_{topology}_{init_mode}.png"
        df = _run_single(
            experiment="bb",
            topology=topology,
            init_mode=init_mode,
            seed=seed,
            vacua_and_T=vacua_and_T,
            T_max=T_max,
            max_events=max_events,
            N_max=N_max,
            log_interval_events=log_interval_events,
            checkpoints=checkpoints,
            n_initial=n_initial,
            initial_vacuum_id=initial_vacuum_id,
            include_bb_metrics=True,
            results_csv=results_csv,
            W_plot=W_plot,
            pop_plot=pop_plot,
            bb_plot=bb_plot,
        )

    # Append summary statistics row for the requested run
    snap = _final_snapshot(df)
    _append_summary_statistics_row(
        {
            "experiment": experiment,
            "topology": topology,
            "init": init_mode,
            "final_time": snap["final_time"],
            "final_total_patches": int(round(snap["final_total_patches"])),
            "final_W_0": snap["final_W_0"],
            "final_W_1": snap["final_W_1"],
            "final_W_2": snap["final_W_2"],
            "final_BB_fraction": snap["final_BB_fraction"],
        },
        outfile="results/summary_statistics.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducing multiverse toy model experiments")
    parser.add_argument("--experiment", choices=["baseline", "intelligence", "bb"], required=True)
    parser.add_argument("--topology", choices=["smooth", "rugged", "cluster"], required=True)
    parser.add_argument("--init", dest="init_mode", choices=["concentrated", "uniform"], required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_experiment(args.experiment, args.topology, args.init_mode, args.seed)


if __name__ == "__main__":
    main()

