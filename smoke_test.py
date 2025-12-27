"""
Basic runtime import & construction test for Python 3.9 compatibility.
Ensures model imports correctly before running experiments.
"""

from multiverse_model import Patch, SimulationConfig, make_smooth_topology


def run_smoke_test() -> None:
    vacua, T = make_smooth_topology()
    cfg = SimulationConfig(
        vacua=vacua,
        transition_matrix=T,
        n_initial=1,
        initial_mode="concentrated",
        initial_vacuum_id=0,
        T_max=1.0,
        max_events=1,
        N_max=10,
        log_interval_events=1,
        checkpoints=[0.0, 1.0],
        seed=0,
    )
    _ = cfg  # avoid unused warning in some tooling

    p = Patch(
        id=0,
        vacuum_id=0,
        birth_time=0.0,
        death_time=1.0,
    )
    _ = p
    print("Smoke test passed: model imports & Patch/SimulationConfig construct OK")


if __name__ == "__main__":
    run_smoke_test()

