from __future__ import annotations

"""
Fire-and-forget paper pipeline runner for COSMOS.

Goals:
- One command runs the full paper data pack (simulations + tables + figures + report)
- No overwriting: each run writes to a unique outdir (default timestamp)
- Resume capable: --resume skips completed outputs
- Reproducible: deterministic seeds + stable plan file
- Minimal risk: does NOT modify or require new flags in experiments.py
"""

import argparse
import concurrent.futures as cf
import dataclasses
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


REQUIRED_JOBS_ORDERED: List[Tuple[str, str, str]] = [
    # A) Smooth topology
    ("baseline", "smooth", "uniform"),
    ("baseline", "smooth", "concentrated"),
    ("intelligence", "smooth", "uniform"),
    # B) Rugged topology
    ("baseline", "rugged", "uniform"),
    ("intelligence", "rugged", "uniform"),
    # C) Cluster topology
    ("baseline", "cluster", "uniform"),
    ("baseline", "cluster", "concentrated"),
    ("intelligence", "cluster", "uniform"),
]


def canonical_job_id(experiment: str, topology: str, init_mode: str) -> str:
    return f"{experiment}/{topology}/{init_mode}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_outdir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("paper_outputs") / stamp


def default_workers() -> int:
    if sys.platform == "darwin":
        return 4
    n = os.cpu_count() or 1
    return max(1, n // 2)


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


class Progress:
    def __init__(self, total: int, *, rolling: int = 50) -> None:
        self.total = int(total)
        self.start = time.time()
        self.completed = 0
        self.ok = 0
        self.fail = 0
        self._durations: Deque[float] = deque(maxlen=int(rolling))
        self._last_print = 0.0

    def update(self, *, ok: bool, duration_s: float) -> Dict[str, Any]:
        self.completed += 1
        if ok:
            self.ok += 1
        else:
            self.fail += 1
        if duration_s is not None and np.isfinite(duration_s) and duration_s > 0:
            self._durations.append(float(duration_s))
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start
        rate_per_min = (self.completed / elapsed * 60.0) if elapsed > 0 else 0.0
        if len(self._durations) > 0:
            avg = float(np.mean(self._durations))
            eta = (self.total - self.completed) * avg
        else:
            eta = float("nan")
        return {
            "completed": self.completed,
            "total": self.total,
            "ok": self.ok,
            "fail": self.fail,
            "elapsed_s": elapsed,
            "rate_per_min": rate_per_min,
            "eta_s": eta,
        }


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _setup_logger(outdir: Path) -> logging.Logger:
    _ensure_dir(outdir)
    logger = logging.getLogger("paper_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(outdir / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("paper_pipeline starting")
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("platform=%s", platform.platform())
    return logger


def load_plan(plan_path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load the plan file. Install dependencies: pip install -r requirements.txt"
        )
    with plan_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Plan file must parse to a mapping/object at top level.")
    return data


def validate_plan(plan: Dict[str, Any]) -> None:
    missing_fields = [k for k in ("seeds", "tmax", "max_events", "jobs") if k not in plan]
    if missing_fields:
        raise ValueError(f"Plan missing required fields: {missing_fields}")

    if not isinstance(plan["jobs"], list) or len(plan["jobs"]) == 0:
        raise ValueError("Plan 'jobs' must be a non-empty list.")

    if not isinstance(plan["seeds"], int) or plan["seeds"] <= 0:
        raise ValueError("Plan 'seeds' must be a positive integer.")
    if float(plan["tmax"]) <= 0:
        raise ValueError("Plan 'tmax' must be > 0.")
    if int(plan["max_events"]) <= 0:
        raise ValueError("Plan 'max_events' must be > 0.")

    # Validate jobs have required keys
    for i, job in enumerate(plan["jobs"]):
        if not isinstance(job, dict):
            raise ValueError(f"Job #{i} must be a mapping/object, got {type(job)}")
        for k in ("experiment", "topology", "init"):
            if k not in job:
                raise ValueError(f"Job #{i} missing required key: {k}")

    # Strict required jobs present
    jobs_set = {
        (str(j["experiment"]), str(j["topology"]), str(j["init"])) for j in plan["jobs"] if isinstance(j, dict)
    }
    missing_required = [j for j in REQUIRED_JOBS_ORDERED if j not in jobs_set]
    if missing_required:
        missing_str = [canonical_job_id(*j) for j in missing_required]
        raise ValueError(
            "Plan is missing REQUIRED jobs; refusing to run.\n"
            + "\n".join([f"  - {s}" for s in missing_str])
        )

    # Detect duplicates (same experiment/topology/init) in explicit job list
    seen: set[Tuple[str, str, str]] = set()
    dups: List[str] = []
    for job in plan["jobs"]:
        key = (str(job["experiment"]), str(job["topology"]), str(job["init"]))
        if key in seen:
            dups.append(canonical_job_id(*key))
        seen.add(key)
    if dups:
        raise ValueError(f"Plan contains duplicate jobs: {sorted(set(dups))}")

    # If enable_bb true, experiments.py in this repo does NOT support bb/rugged.
    enable_bb = bool(plan.get("enable_bb", False))
    if enable_bb:
        raise ValueError(
            "Plan enable_bb=true requested, but experiments.py currently does not support bb on topology 'rugged'. "
            "Refusing to run (set enable_bb: false, or extend experiments.py and then re-run)."
        )


@dataclasses.dataclass(frozen=True)
class JobSpec:
    experiment: str
    topology: str
    init: str
    mode: str  # "seeded" or "single"
    single_seed: Optional[int] = None

    @property
    def job_name(self) -> str:
        return canonical_job_id(self.experiment, self.topology, self.init)


@dataclasses.dataclass(frozen=True)
class RunSpec:
    job: JobSpec
    seed: Optional[int]  # None for "single" jobs
    tmax: float
    max_events: int

    @property
    def run_label(self) -> str:
        if self.seed is None:
            return f"{self.job.job_name} single"
        return f"{self.job.job_name} seed {self.seed}"


def jobs_from_plan(plan: Dict[str, Any]) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for j in plan["jobs"]:
        mode = str(j.get("mode", "seeded"))
        if mode not in ("seeded", "single"):
            raise ValueError(f"Unknown job mode={mode} for job {j}")
        single_seed = None
        if mode == "single":
            if "seed" in j:
                single_seed = int(j["seed"])
            else:
                single_seed = 42
        jobs.append(
            JobSpec(
                experiment=str(j["experiment"]),
                topology=str(j["topology"]),
                init=str(j["init"]),
                mode=mode,
                single_seed=single_seed,
            )
        )

    # Optional BB toggle (plan default is false). For v1, enable_bb is rejected by validate_plan
    # because experiments.py does not support bb/rugged.
    return jobs


def run_matrix(plan: Dict[str, Any]) -> List[RunSpec]:
    seeds_n = int(plan["seeds"])
    tmax = float(plan["tmax"])
    max_events = int(plan["max_events"])
    out: List[RunSpec] = []
    for job in jobs_from_plan(plan):
        if job.mode == "seeded":
            for seed in range(seeds_n):
                out.append(RunSpec(job=job, seed=int(seed), tmax=tmax, max_events=max_events))
        else:
            out.append(
                RunSpec(job=job, seed=None, tmax=tmax, max_events=max_events)
            )
    return out


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _expected_results_csv(experiment: str, topology: str, init_mode: str) -> str:
    if experiment == "baseline":
        return f"results/results_baseline_{topology}_{init_mode}.csv"
    if experiment == "intelligence":
        return f"results/results_intel_{topology}_{init_mode}.csv"
    if experiment == "bb":
        return f"results/results_bb_{topology}_{init_mode}.csv"
    # cluster_suite writes multiple CSVs; handled separately
    return ""


def _workdir_for_run(outdir: Path, run: RunSpec) -> Path:
    if run.seed is None:
        return outdir / "work" / run.job.job_name / "single"
    return outdir / "work" / run.job.job_name / f"seed_{run.seed}"


def _rawdir_for_run(outdir: Path, run: RunSpec) -> Path:
    if run.seed is None:
        return outdir / "raw" / run.job.job_name / "single"
    return outdir / "raw" / run.job.job_name / f"seed_{run.seed}"


def _completion_marker(outdir: Path, run: RunSpec) -> Path:
    # Completion defined as the presence of results.csv for seeded runs.
    if run.seed is None:
        return _rawdir_for_run(outdir, run) / "DONE"
    return _rawdir_for_run(outdir, run) / "results.csv"


def _run_one_worker(
    repo_root: str,
    outdir: str,
    run_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Worker entrypoint. Runs a single (job, seed) in an isolated working dir.
    Returns a dict of run metadata (including status, runtime, error).
    """
    repo = Path(repo_root).resolve()
    out = Path(outdir).resolve()

    job = JobSpec(**run_dict["job"])
    seed = run_dict["seed"]
    run = RunSpec(job=job, seed=seed, tmax=float(run_dict["tmax"]), max_events=int(run_dict["max_events"]))

    started_utc = utc_now_iso()
    t0 = time.time()
    status = "ok"
    err_msg = ""

    wdir = _workdir_for_run(out, run)
    _ensure_dir(wdir)
    _ensure_dir(wdir / "results")
    _ensure_dir(wdir / "plots")
    _ensure_dir(wdir / "logs")

    # Make repo files available in the isolated working directory.
    for fname in ("experiments.py", "multiverse_model.py", "plotting.py", "requirements.txt"):
        src = repo / fname
        if src.exists():
            _symlink_or_copy(src, wdir / fname)

    run_log_path = wdir / "logs" / "run.log"

    try:
        cmd = [
            sys.executable,
            "experiments.py",
            "--experiment",
            run.job.experiment,
            "--topology",
            run.job.topology,
            "--init",
            run.job.init,
            "--tmax",
            str(run.tmax),
            "--max-events",
            str(run.max_events),
        ]

        # Determine seed handling:
        # - seeded jobs: pass the run seed
        # - single jobs: pass job.single_seed if present
        if run.seed is not None:
            cmd += ["--seed", str(int(run.seed))]
        else:
            cmd += ["--seed", str(int(run.job.single_seed or 42))]

        proc = subprocess.run(
            cmd,
            cwd=str(wdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        run_log_path.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            status = "fail"
            err_msg = f"experiments.py exited with code {proc.returncode}"
    except Exception as e:  # pragma: no cover
        status = "fail"
        err_msg = f"Exception while running experiments.py: {e!r}"
        try:
            run_log_path.write_text(err_msg, encoding="utf-8")
        except Exception:
            pass

    runtime_s = time.time() - t0

    rawdir = _rawdir_for_run(out, run)
    _ensure_dir(rawdir)

    # Copy outputs
    copied_any = False
    if status == "ok":
        try:
            if run.job.experiment == "cluster_suite" and run.seed is None:
                # Copy all CSV artifacts from this invocation (results/ + cwd files).
                # Mark completion with DONE file.
                for p in sorted((wdir / "results").glob("*.csv")):
                    shutil.copy2(p, rawdir / p.name)
                    copied_any = True
                for p in sorted(wdir.glob("*.csv")):
                    shutil.copy2(p, rawdir / p.name)
                    copied_any = True
                (rawdir / "DONE").write_text("ok\n", encoding="utf-8")
            else:
                expected = _expected_results_csv(run.job.experiment, run.job.topology, run.job.init)
                expected_path = wdir / expected if expected else None

                src_csv: Optional[Path] = None
                if expected_path is not None and expected_path.exists():
                    src_csv = expected_path
                else:
                    # Fallback: pick the largest CSV in results/
                    candidates = list((wdir / "results").glob("*.csv"))
                    if len(candidates) == 1:
                        src_csv = candidates[0]
                    elif len(candidates) > 1:
                        src_csv = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]

                if src_csv is None or not src_csv.exists():
                    status = "fail"
                    err_msg = "No results CSV found after run."
                else:
                    shutil.copy2(src_csv, rawdir / "results.csv")
                    copied_any = True
        except Exception as e:
            status = "fail"
            err_msg = f"Failed copying outputs: {e!r}"

    # Always copy the run log for debugging
    try:
        if run_log_path.exists():
            shutil.copy2(run_log_path, rawdir / "run.log")
    except Exception:
        pass

    return {
        "timestamp": started_utc,
        "job_name": run.job.job_name,
        "experiment": run.job.experiment,
        "topology": run.job.topology,
        "init": run.job.init,
        "seed": (None if run.seed is None else int(run.seed)),
        "tmax": float(run.tmax),
        "max_events": int(run.max_events),
        "runtime_seconds": float(runtime_s),
        "status": status,
        "error_message": err_msg,
        "rawdir": str(rawdir),
        "workdir": str(wdir),
        "copied_any": bool(copied_any),
    }


def _read_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"results.csv missing 'time' column: {path}")
    if "total_patches" not in df.columns:
        raise ValueError(f"results.csv missing 'total_patches' column: {path}")
    return df


def _extract_final_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    W_cols = [c for c in df.columns if c.startswith("W_")]
    W_map: Dict[int, float] = {}
    for c in W_cols:
        try:
            idx = int(c.split("_", 1)[1])
        except Exception:
            continue
        W_map[idx] = float(last[c])
    # Standardize to W_0..W_2 (this model uses 3 vacua)
    return {
        "final_time": float(last["time"]),
        "final_total_patches": int(round(float(last["total_patches"]))),
        "final_W_0": float(W_map.get(0, 0.0)),
        "final_W_1": float(W_map.get(1, 0.0)),
        "final_W_2": float(W_map.get(2, 0.0)),
        "extinct_flag": int(round(float(last["total_patches"]))) == 0,
    }


def _tau_mapping_for_jobs(jobs: Sequence[JobSpec]) -> Dict[str, Dict[str, Any]]:
    """
    Determine job_name -> (tau values, long_index) where long_index is argmax(tau).
    Uses multiverse_model's topology builders (tau is topology-defined, not experiment-defined).
    """
    # Import only in parent process
    from multiverse_model import make_cluster_topology, make_rugged_topology, make_smooth_topology

    topo_to_builder = {
        "smooth": make_smooth_topology,
        "rugged": make_rugged_topology,
        "cluster": make_cluster_topology,
    }

    out: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        if job.topology not in topo_to_builder:
            continue
        vacua, _ = topo_to_builder[job.topology]()
        taus = [float(v.tau) for v in vacua]
        long_idx = int(np.argmax(np.array(taus)))
        out[job.job_name] = {"taus": taus, "long_index": long_idx}
    return out


def _ci95_mean(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    m = float(np.mean(x))
    if x.size == 1:
        return (m, m)
    s = float(np.std(x, ddof=1))
    se = s / np.sqrt(float(x.size))
    half = 1.96 * se
    return (m - half, m + half)


def write_tables_and_figures(
    *,
    outdir: Path,
    run_rows: List[Dict[str, Any]],
    plan: Dict[str, Any],
    jobs: List[JobSpec],
    logger: logging.Logger,
) -> None:
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)

    run_index = pd.DataFrame(run_rows)
    # Augment with final snapshots for ok seeded runs (single runs don't have results.csv)
    final_cols = ["final_time", "final_total_patches", "final_W_0", "final_W_1", "final_W_2", "extinct_flag"]
    for c in final_cols:
        if c not in run_index.columns:
            run_index[c] = np.nan

    for idx, row in run_index.iterrows():
        if row.get("status") != "ok":
            continue
        if row.get("seed") is None:
            continue
        rawdir = Path(str(row["rawdir"]))
        results_path = rawdir / "results.csv"
        if not results_path.exists():
            continue
        try:
            df = _read_results_csv(results_path)
            snap = _extract_final_snapshot(df)
            for k, v in snap.items():
                run_index.at[idx, k] = v
        except Exception as e:
            run_index.at[idx, "status"] = "fail"
            run_index.at[idx, "error_message"] = f"Postprocess failed reading results.csv: {e!r}"

    run_index_out = tables_dir / "run_index.csv"
    run_index.to_csv(run_index_out, index=False)
    logger.info("wrote %s", run_index_out)

    # Job summary (only seeded jobs contribute)
    tau_map = _tau_mapping_for_jobs(jobs)

    seeded = run_index[(run_index["seed"].notna()) & (run_index["status"] == "ok")].copy()
    if len(seeded) == 0:
        logger.info("no successful seeded runs; skipping job_summary and figures")
        return

    # W_long definition
    seeded["long_index"] = seeded["job_name"].map(lambda j: tau_map.get(str(j), {}).get("long_index", 2))
    seeded["final_W_long"] = np.where(seeded["long_index"] == 0, seeded["final_W_0"], seeded["final_W_2"])
    seeded.loc[seeded["long_index"] == 1, "final_W_long"] = seeded.loc[seeded["long_index"] == 1, "final_W_1"]

    summary_rows: List[Dict[str, Any]] = []
    for job_name, g in seeded.groupby("job_name", sort=False):
        x = g["final_W_long"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        n = int(len(g))
        n_ok = int(len(x))
        n_fail = int(run_index[(run_index["job_name"] == job_name) & (run_index["status"] != "ok")].shape[0])
        extinct_rate = float(np.mean(g["extinct_flag"].astype(float))) if n > 0 else float("nan")

        mean = float(np.mean(x)) if n_ok > 0 else float("nan")
        median = float(np.median(x)) if n_ok > 0 else float("nan")
        std = float(np.std(x, ddof=1)) if n_ok > 1 else float("nan")
        ci_lo, ci_hi = _ci95_mean(x)
        qs = {q: float(np.quantile(x, q)) if n_ok > 0 else float("nan") for q in (0.05, 0.25, 0.5, 0.75, 0.95)}

        summary_rows.append(
            {
                "job_name": str(job_name),
                "experiment": str(g["experiment"].iloc[0]),
                "topology": str(g["topology"].iloc[0]),
                "init": str(g["init"].iloc[0]),
                "n_runs": int(n),
                "n_ok": int(n_ok),
                "n_fail": int(n_fail),
                "extinction_rate": float(extinct_rate),
                "final_W_long_mean": mean,
                "final_W_long_median": median,
                "final_W_long_std": std,
                "final_W_long_ci95_lo": float(ci_lo),
                "final_W_long_ci95_hi": float(ci_hi),
                "final_W_long_q05": qs[0.05],
                "final_W_long_q25": qs[0.25],
                "final_W_long_q50": qs[0.5],
                "final_W_long_q75": qs[0.75],
                "final_W_long_q95": qs[0.95],
                "long_index": int(tau_map.get(str(job_name), {}).get("long_index", 2)),
                "taus": json.dumps(tau_map.get(str(job_name), {}).get("taus", [])),
            }
        )

    job_summary = pd.DataFrame(summary_rows)
    job_summary_out = tables_dir / "job_summary.csv"
    job_summary.to_csv(job_summary_out, index=False)
    logger.info("wrote %s", job_summary_out)

    # Figures (matplotlib only)
    import matplotlib.pyplot as plt

    def _ecdf(ax, xvals: np.ndarray, label: str) -> None:
        xvals = np.asarray(xvals, dtype=float)
        xvals = xvals[np.isfinite(xvals)]
        if xvals.size == 0:
            return
        xsorted = np.sort(xvals)
        y = np.arange(1, xsorted.size + 1) / float(xsorted.size)
        ax.step(xsorted, y, where="post", label=label)

    # Fig1: Rugged baseline vs intelligence distribution of final_W_long
    fig1 = plt.figure(figsize=(7.5, 5))
    ax1 = fig1.add_subplot(1, 1, 1)
    rugged = seeded[seeded["topology"] == "rugged"]
    for exp in ("baseline", "intelligence"):
        x = rugged[rugged["experiment"] == exp]["final_W_long"].to_numpy(dtype=float)
        _ecdf(ax1, x, label=exp)
    ax1.set_title("Fig1: Rugged topology — final W_long ECDF")
    ax1.set_xlabel("final W_long")
    ax1.set_ylabel("ECDF")
    ax1.legend()
    fig1.tight_layout()
    fig1_path = figures_dir / "Fig1_rugged_final_W_long_ecdf.png"
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    # Fig2: Across topologies — mean final_W_long with 95% CI (baseline vs intelligence)
    fig2 = plt.figure(figsize=(8.5, 5.5))
    ax2 = fig2.add_subplot(1, 1, 1)
    topologies = ["smooth", "rugged", "cluster"]
    xloc = np.arange(len(topologies))
    width = 0.35
    for i, exp in enumerate(("baseline", "intelligence")):
        means = []
        yerr_lo = []
        yerr_hi = []
        for topo in topologies:
            vals = seeded[(seeded["topology"] == topo) & (seeded["experiment"] == exp)]["final_W_long"].to_numpy(
                dtype=float
            )
            vals = vals[np.isfinite(vals)]
            m = float(np.mean(vals)) if vals.size > 0 else float("nan")
            lo, hi = _ci95_mean(vals)
            means.append(m)
            yerr_lo.append(m - lo if np.isfinite(lo) else 0.0)
            yerr_hi.append(hi - m if np.isfinite(hi) else 0.0)
        pos = xloc + (i - 0.5) * width
        ax2.bar(pos, means, width=width, label=exp, yerr=[yerr_lo, yerr_hi], capsize=4)
    ax2.set_xticks(xloc, topologies)
    ax2.set_title("Fig2: Mean final W_long with 95% CI")
    ax2.set_ylabel("mean final W_long")
    ax2.legend()
    fig2.tight_layout()
    fig2_path = figures_dir / "Fig2_topologies_mean_final_W_long_ci95.png"
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    # Fig3: W(t) mean trajectory with CI shading per topology (baseline vs intelligence)
    # Interpolate each run's W_long(t) onto a common grid.
    tmax = float(plan["tmax"])
    grid = np.linspace(0.0, tmax, int(tmax) + 1)

    def _trajectory_stats(job_rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        trajs: List[np.ndarray] = []
        for _, rr in job_rows.iterrows():
            rawdir = Path(str(rr["rawdir"]))
            p = rawdir / "results.csv"
            if not p.exists():
                continue
            df = _read_results_csv(p)
            long_idx = int(rr["long_index"])
            col = f"W_{long_idx}"
            if col not in df.columns:
                continue
            t = df["time"].to_numpy(dtype=float)
            y = df[col].to_numpy(dtype=float)
            # Ensure increasing t for interpolation
            order = np.argsort(t)
            t = t[order]
            y = y[order]
            # Clamp beyond range to endpoints
            yi = np.interp(grid, t, y)
            trajs.append(yi)
        if len(trajs) == 0:
            nan = np.full_like(grid, np.nan, dtype=float)
            return nan, nan, nan
        A = np.vstack(trajs)
        mean = np.nanmean(A, axis=0)
        if A.shape[0] > 1:
            sd = np.nanstd(A, axis=0, ddof=1)
            se = sd / np.sqrt(float(A.shape[0]))
            lo = mean - 1.96 * se
            hi = mean + 1.96 * se
        else:
            lo = mean
            hi = mean
        return mean, lo, hi

    fig3, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, topo in zip(axes, topologies):
        sub = seeded[seeded["topology"] == topo].copy()
        for exp in ("baseline", "intelligence"):
            exp_rows = sub[sub["experiment"] == exp]
            mean, lo, hi = _trajectory_stats(exp_rows)
            ax.plot(grid, mean, label=exp)
            ax.fill_between(grid, lo, hi, alpha=0.2)
        ax.set_title(f"{topo}")
        ax.set_xlabel("time")
    axes[0].set_ylabel("W_long(t)")
    axes[0].legend()
    fig3.suptitle("Fig3: Mean W_long(t) with 95% CI")
    fig3.tight_layout()
    fig3_path = figures_dir / "Fig3_W_long_trajectories_ci95.png"
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    # Fig4: Extinction probability by topology and experiment
    fig4 = plt.figure(figsize=(8.5, 5.5))
    ax4 = fig4.add_subplot(1, 1, 1)
    xloc = np.arange(len(topologies))
    width = 0.35
    for i, exp in enumerate(("baseline", "intelligence")):
        rates = []
        for topo in topologies:
            g = seeded[(seeded["topology"] == topo) & (seeded["experiment"] == exp)]
            rate = float(np.mean(g["extinct_flag"].astype(float))) if len(g) > 0 else float("nan")
            rates.append(rate)
        pos = xloc + (i - 0.5) * width
        ax4.bar(pos, rates, width=width, label=exp)
    ax4.set_xticks(xloc, topologies)
    ax4.set_ylim(0.0, 1.0)
    ax4.set_title("Fig4: Extinction probability by topology")
    ax4.set_ylabel("P(extinction)")
    ax4.legend()
    fig4.tight_layout()
    fig4_path = figures_dir / "Fig4_extinction_probability.png"
    fig4.savefig(fig4_path, dpi=200)
    plt.close(fig4)

    logger.info("wrote figures to %s", figures_dir)


def write_report(
    *,
    outdir: Path,
    plan_path: Path,
    plan: Dict[str, Any],
    jobs: List[JobSpec],
    total_wall_s: float,
    workers: int,
    run_rows: List[Dict[str, Any]],
) -> None:
    required_list = "\n".join([f"- `{canonical_job_id(*j)}`" for j in REQUIRED_JOBS_ORDERED])
    executed = [j.job_name for j in jobs]
    executed_list = "\n".join([f"- `{x}`" for x in executed])

    # Key numeric results from job_summary if available
    key_lines: List[str] = []
    job_summary_path = outdir / "tables" / "job_summary.csv"
    if job_summary_path.exists():
        js = pd.read_csv(job_summary_path)
        for topo in ("smooth", "rugged", "cluster"):
            for exp in ("baseline", "intelligence"):
                row = js[(js["topology"] == topo) & (js["experiment"] == exp) & (js["init"] == "uniform")]
                if len(row) == 0:
                    continue
                r = row.iloc[0]
                key_lines.append(
                    f"- **{topo} / {exp}**: mean final_W_long={r['final_W_long_mean']:.6f} "
                    f"(95% CI [{r['final_W_long_ci95_lo']:.6f}, {r['final_W_long_ci95_hi']:.6f}]), "
                    f"extinction_rate={float(r['extinction_rate']):.4f}"
                )

    failures = [r for r in run_rows if r.get("status") != "ok"]
    avg_run = float(np.mean([float(r["runtime_seconds"]) for r in run_rows])) if len(run_rows) else float("nan")
    all_required_present = "YES"

    tau_map = _tau_mapping_for_jobs(jobs)
    tau_lines = []
    for job_name in executed:
        m = tau_map.get(job_name, {})
        tau_lines.append(f"- `{job_name}` -> long_index={m.get('long_index')} taus={m.get('taus')}")

    cmd = " ".join([shlex_quote(x) for x in sys.argv])  # exact command used

    report = f"""# COSMOS paper pipeline report

## How to reproduce

Command run:

`{cmd}`

## Plan summary

- **plan**: `{plan_path}`
- **seeds**: {int(plan['seeds'])} (seed values: 0..{int(plan['seeds'])-1})
- **tmax**: {float(plan['tmax'])}
- **max_events**: {int(plan['max_events'])}
- **enable_bb**: {bool(plan.get('enable_bb', False))}

## EXACT JOBS EXECUTED

{executed_list}

### Required jobs list (from pipeline)

{required_list}

- **All required jobs present: {all_required_present}**

## Runtime summary

- **total wall time**: {fmt_seconds(total_wall_s)}
- **workers used**: {int(workers)}
- **avg per run (including failures)**: {fmt_seconds(avg_run)}

## Failures summary

- **n_fail**: {len(failures)}
- See `{(outdir / 'pipeline.log')}` and per-run `raw/<job>/seed_<seed>/run.log`.

## W_long definition (by job)

W_long is defined as the W component corresponding to the vacuum with the largest tau
in the topology configuration.

{os.linesep.join(tau_lines)}

## Key numeric results

{os.linesep.join(key_lines) if key_lines else "- (job_summary.csv not available)"}

## Outputs

- **run index**: `{(outdir / 'tables' / 'run_index.csv')}`
- **job summary**: `{(outdir / 'tables' / 'job_summary.csv')}`
- **figures**: `{(outdir / 'figures')}` (PNG)
- **log**: `{(outdir / 'pipeline.log')}`
"""
    (outdir / "REPORT.md").write_text(report, encoding="utf-8")


def shlex_quote(s: str) -> str:
    # Minimal quoting for report output (avoid importing shlex).
    if s == "":
        return "''"
    if all(c.isalnum() or c in ("-", "_", ".", "/", ":", "=") for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def print_dry_run(
    *,
    plan_path: Path,
    outdir: Path,
    plan: Dict[str, Any],
    jobs: List[JobSpec],
    runs: List[RunSpec],
) -> None:
    print("paper_pipeline.py --dry-run")
    print(f"plan: {plan_path}")
    print(f"outdir: {outdir}")
    print(f"seeds: {int(plan['seeds'])} (seed values: 0..{int(plan['seeds'])-1})")
    print(f"tmax: {float(plan['tmax'])}")
    print(f"max_events: {int(plan['max_events'])}")
    print(f"enable_bb: {bool(plan.get('enable_bb', False))}")
    print("")
    print("EXACT JOBS (canonical identifiers):")
    for j in jobs:
        print(f"  - {j.job_name}")
    print("")
    print(f"total_runs: {len(runs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=str, default="paper_plan.yaml")
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    plan_path = Path(args.plan).resolve() if args.plan else (repo_root / "paper_plan.yaml")
    outdir = Path(args.outdir).resolve() if args.outdir else (repo_root / default_outdir())
    workers = int(args.workers) if int(args.workers) > 0 else default_workers()

    plan = load_plan(plan_path)
    validate_plan(plan)

    jobs = jobs_from_plan(plan)
    runs = run_matrix(plan)

    if args.dry_run:
        print_dry_run(plan_path=plan_path, outdir=outdir, plan=plan, jobs=jobs, runs=runs)
        return

    # No overwriting: refuse to run into an existing non-empty outdir unless --resume.
    if outdir.exists() and any(outdir.iterdir()) and not args.resume:
        raise SystemExit(
            f"Refusing to run: outdir already exists and is non-empty: {outdir}\n"
            "Use --outdir <new_folder> for a fresh run, or --resume --outdir <same_folder> to continue."
        )

    logger = _setup_logger(outdir)
    logger.info("plan=%s", plan_path)
    logger.info("outdir=%s", outdir)
    logger.info("workers=%d", workers)
    logger.info("total_runs=%d", len(runs))

    _ensure_dir(outdir)
    _ensure_dir(outdir / "raw")
    _ensure_dir(outdir / "work")
    _ensure_dir(outdir / "tables")
    _ensure_dir(outdir / "figures")

    # Save a copy of the plan used for this run (stable artifact)
    shutil.copy2(plan_path, outdir / "plan_used.yaml")

    # Build list of pending runs (resume skips completed).
    pending: List[RunSpec] = []
    skipped = 0
    for run in runs:
        marker = _completion_marker(outdir, run)
        if args.resume and marker.exists():
            skipped += 1
            continue
        pending.append(run)
    logger.info("resume=%s | pending=%d | skipped=%d", args.resume, len(pending), skipped)

    # Run all pending runs in parallel
    t_wall0 = time.time()
    progress = Progress(total=len(runs))

    # If resuming, count skipped runs as completed for progress display
    for _ in range(skipped):
        progress.completed += 1

    bar = None
    if tqdm is not None:
        bar = tqdm(total=len(runs), unit="run", dynamic_ncols=True)
        if skipped:
            bar.update(skipped)

    run_rows: List[Dict[str, Any]] = []
    failures = 0

    def _log_progress(snap: Dict[str, Any]) -> None:
        msg = (
            f"completed {snap['completed']}/{snap['total']} "
            f"(ok={snap['ok']} fail={snap['fail']}) "
            f"rate={snap['rate_per_min']:.2f} runs/min "
            f"elapsed={fmt_seconds(snap['elapsed_s'])} "
            f"eta={fmt_seconds(snap['eta_s']) if np.isfinite(snap['eta_s']) else '??:??'}"
        )
        logger.info(msg)

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        fut_to_run: Dict[cf.Future, RunSpec] = {}
        for run in pending:
            rd = {
                "job": dataclasses.asdict(run.job),
                "seed": run.seed,
                "tmax": run.tmax,
                "max_events": run.max_events,
            }
            fut = ex.submit(_run_one_worker, str(repo_root), str(outdir), rd)
            fut_to_run[fut] = run

        for fut in cf.as_completed(list(fut_to_run.keys())):
            run = fut_to_run[fut]
            try:
                row = fut.result()
            except Exception as e:  # pragma: no cover
                row = {
                    "timestamp": utc_now_iso(),
                    "job_name": run.job.job_name,
                    "experiment": run.job.experiment,
                    "topology": run.job.topology,
                    "init": run.job.init,
                    "seed": (None if run.seed is None else int(run.seed)),
                    "tmax": float(run.tmax),
                    "max_events": int(run.max_events),
                    "runtime_seconds": float("nan"),
                    "status": "fail",
                    "error_message": f"Worker crashed: {e!r}",
                    "rawdir": str(_rawdir_for_run(outdir, run)),
                    "workdir": str(_workdir_for_run(outdir, run)),
                    "copied_any": False,
                }

            run_rows.append(row)

            ok = row.get("status") == "ok"
            if not ok:
                failures += 1
            snap = progress.update(ok=ok, duration_s=float(row.get("runtime_seconds") or 0.0))

            # Update progress bar/print ETA
            if bar is not None:
                bar.update(1)
                bar.set_postfix(
                    {
                        "ok": snap["ok"],
                        "fail": snap["fail"],
                        "rate/min": f"{snap['rate_per_min']:.2f}",
                        "eta": (fmt_seconds(snap["eta_s"]) if np.isfinite(snap["eta_s"]) else "??:??"),
                    }
                )
            else:
                # stdlib progress bar: single-line update
                pct = 100.0 * (snap["completed"] / snap["total"]) if snap["total"] else 100.0
                line = (
                    f"\r{pct:6.2f}% | {snap['completed']}/{snap['total']} | "
                    f"ok={snap['ok']} fail={snap['fail']} | "
                    f"{snap['rate_per_min']:.2f} runs/min | "
                    f"elapsed {fmt_seconds(snap['elapsed_s'])} | "
                    f"ETA {fmt_seconds(snap['eta_s']) if np.isfinite(snap['eta_s']) else '??:??'}"
                )
                sys.stdout.write(line)
                sys.stdout.flush()

            # Periodic progress snapshots into pipeline.log (and stdout)
            now = time.time()
            if (now - progress._last_print) >= 5.0:
                progress._last_print = now
                _log_progress(snap)

    if bar is not None:
        bar.close()
    else:
        sys.stdout.write("\n")
        sys.stdout.flush()

    total_wall = time.time() - t_wall0
    logger.info("runs complete | total_wall=%s | failures=%d", fmt_seconds(total_wall), failures)

    # Tables + figures
    write_tables_and_figures(outdir=outdir, run_rows=run_rows, plan=plan, jobs=jobs, logger=logger)

    # Report
    write_report(
        outdir=outdir,
        plan_path=plan_path,
        plan=plan,
        jobs=jobs,
        total_wall_s=total_wall,
        workers=workers,
        run_rows=run_rows,
    )
    logger.info("wrote %s", outdir / "REPORT.md")


if __name__ == "__main__":
    main()

