from __future__ import annotations

"""
Core simulation engine for a reproducing multiverse toy model.

Implements a continuous-time, multi-type, age-dependent branching process:
- Each patch lives for an exponential lifetime (rate set by tau_eff = tau * alpha)
- At decay, it fragments into a Poisson (or approximated) number of daughters, with
  mean kappa * age**beta, where age is the patch's lifetime.
- Each daughter transitions to a new vacuum based on a row-stochastic transition
  matrix with an implied sink probability (terminal state).

Observer-weighted measure:
At logged times T, compute observer weights for alive patches and derive:
W_i(T) = observer-weight in vacuum i / total observer-weight across all vacua.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq

import numpy as np


@dataclass(frozen=True)
class VacuumSpec:
    """
    Vacuum specification.

    Attributes:
        vacuum_id: Integer identifier.
        tau: Baseline mean decay time.
        alpha: Intelligence lifetime multiplier (>= 1). Effective lifetime tau_eff = tau * alpha.
        kappa: Fragmentation prefactor.
        beta: Fragmentation exponent.
        o_struct: Structured-observer scale (linear model by default).
        o_bb: Boltzmann-brain observer amplitude.
        t_bb: BB characteristic timescale. If o_bb == 0, set to np.inf.
        label: Optional display label.
    """

    vacuum_id: int
    tau: float
    alpha: float
    kappa: float
    beta: float
    o_struct: float
    o_bb: float
    t_bb: float
    label: str = ""

    @property
    def tau_eff(self) -> float:
        """Effective mean lifetime tau_eff = tau * alpha."""
        return float(self.tau * self.alpha)


def validate_transition_matrix(T: np.ndarray) -> None:
    """
    Ensure each row sums to <= 1.0 (remainder is sink probability)
    and all entries are non-negative.
    Raises ValueError if invalid.
    """
    if not isinstance(T, np.ndarray):
        raise ValueError("Transition matrix T must be a numpy array.")
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"Transition matrix must be square; got shape {T.shape}.")
    if np.any(T < -1e-15):
        raise ValueError("Transition matrix contains negative entries.")

    row_sums = T.sum(axis=1)
    if np.any(row_sums > 1.0 + 1e-12):
        bad_rows = np.where(row_sums > 1.0 + 1e-12)[0].tolist()
        raise ValueError(f"Transition matrix has rows summing to > 1.0: {bad_rows}")


@dataclass(slots=True)
class Patch:
    """
    A single "patch" (worldline) in a given vacuum.

    Attributes:
        id: Unique integer id.
        vacuum_id: Current vacuum type.
        birth_time: Global time of birth.
        death_time: Global time of decay.
        alive: Whether patch is alive (used with lazy deletion in the event heap).
    """

    id: int
    vacuum_id: int
    birth_time: float
    death_time: float
    alive: bool = True

    def age(self, current_time: float) -> float:
        """Return age at current_time, clipped at >= 0."""
        return float(max(current_time - self.birth_time, 0.0))

    def is_alive(self, current_time: float) -> bool:
        """Alive and within [birth_time, death_time)."""
        return bool(self.alive and (self.birth_time <= current_time < self.death_time))


def _safe_exp(x: float) -> float:
    """Exponent with overflow protection for large x."""
    if x <= 0.0:
        return float(np.exp(x))
    # exp(709) ~ 8e307 is near float max
    return float(np.exp(min(x, 709.0)))


def sample_lifetime(tau_eff: float) -> float:
    """Sample an exponential lifetime with mean tau_eff."""
    if not np.isfinite(tau_eff) or tau_eff <= 0:
        raise ValueError(f"tau_eff must be finite and > 0; got {tau_eff}")
    return float(np.random.exponential(scale=tau_eff))


def structured_observers_linear(vac: VacuumSpec, age: float) -> float:
    """Default linear structured observers: O_struct = o_struct * max(age, 0)."""
    return float(vac.o_struct * max(age, 0.0))


def boltzmann_brains(vac: VacuumSpec, age: float) -> float:
    """BB observers: O_BB = o_bb * exp(age / t_bb). If o_bb==0 -> 0."""
    if vac.o_bb == 0.0:
        return 0.0
    if not np.isfinite(vac.t_bb) or vac.t_bb <= 0.0:
        # If o_bb>0 but t_bb is invalid, treat as no BBs rather than blowing up.
        return 0.0
    return float(vac.o_bb * _safe_exp(age / vac.t_bb))


def total_observers(vac: VacuumSpec, age: float) -> Tuple[float, float, float]:
    """Return (O_total, O_struct, O_BB) for a vacuum and age."""
    o_s = structured_observers_linear(vac, age)
    o_b = boltzmann_brains(vac, age)
    return float(o_s + o_b), float(o_s), float(o_b)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """
    Configuration for a simulation run.
    """

    vacua: List[VacuumSpec]
    transition_matrix: np.ndarray
    n_initial: int
    initial_mode: str  # "concentrated" or "uniform"
    initial_vacuum_id: Optional[int]
    T_max: float
    max_events: int
    N_max: int
    log_interval_events: int
    checkpoints: List[float]
    seed: int


@dataclass(slots=True)
class SimulationResult:
    """
    Recorded simulation outputs.
    """

    times: List[float]
    total_patches: List[int]
    W_series: Dict[int, List[float]]  # vacuum_id -> list of W_i(T_k)
    BB_fraction: Optional[List[float]] = None
    BB_weight: Optional[List[float]] = None
    struct_weight: Optional[List[float]] = None
    meta: Optional[Dict[str, str]] = None


class Simulation:
    """
    Event-driven simulation of the multiverse branching process.

    Uses a min-heap of decay events (death_time, patch_id) with lazy deletion.
    """

    def __init__(self, config: SimulationConfig):
        validate_transition_matrix(config.transition_matrix)
        self.config = config

        self.vacua_by_id: Dict[int, VacuumSpec] = {v.vacuum_id: v for v in config.vacua}
        if len(self.vacua_by_id) != len(config.vacua):
            raise ValueError("Duplicate vacuum_id in vacua list.")

        self.n_vacua = len(config.vacua)
        if config.transition_matrix.shape != (self.n_vacua, self.n_vacua):
            raise ValueError(
                f"T shape {config.transition_matrix.shape} does not match n_vacua={self.n_vacua}"
            )

        if config.initial_mode not in ("concentrated", "uniform"):
            raise ValueError(f"Unknown initial_mode={config.initial_mode}")
        if config.n_initial <= 0:
            raise ValueError("n_initial must be > 0")
        if config.initial_mode == "concentrated" and config.initial_vacuum_id is None:
            raise ValueError("initial_vacuum_id required for concentrated initial_mode")

        self.current_time: float = 0.0
        self.events_processed: int = 0

        self._next_patch_id: int = 0
        self.patches: Dict[int, Patch] = {}
        self.alive_ids: set[int] = set()
        self.event_heap: List[Tuple[float, int]] = []

        # logging state
        self._checkpoints = sorted([c for c in config.checkpoints if c <= config.T_max])
        self._next_checkpoint_idx = 0

    def _new_patch(self, vacuum_id: int, birth_time: float) -> Patch:
        vac = self.vacua_by_id[vacuum_id]
        lifetime = sample_lifetime(vac.tau_eff)
        patch_id = self._next_patch_id
        self._next_patch_id += 1
        p = Patch(
            id=patch_id,
            vacuum_id=vacuum_id,
            birth_time=float(birth_time),
            death_time=float(birth_time + lifetime),
            alive=True,
        )
        self.patches[patch_id] = p
        self.alive_ids.add(patch_id)
        heapq.heappush(self.event_heap, (p.death_time, patch_id))
        return p

    def _sample_daughter_vacuum_or_sink(self, parent_vacuum_id: int) -> Optional[int]:
        probs_row = self.config.transition_matrix[parent_vacuum_id, :]
        sink_prob = float(max(1.0 - float(np.sum(probs_row)), 0.0))
        probs = np.append(probs_row, sink_prob)
        # Normalize defensively in case of tiny floating error
        s = float(np.sum(probs))
        if s <= 0.0:
            return None
        probs = probs / s
        outcome = int(np.random.choice(self.n_vacua + 1, p=probs))
        if outcome == self.n_vacua:
            return None
        return outcome

    def _sample_num_daughters(self, vac: VacuumSpec, age_at_death: float) -> int:
        mean = float(vac.kappa * (max(age_at_death, 0.0) ** vac.beta))
        if mean <= 0.0 or not np.isfinite(mean):
            return 0

        # Performance safeguards: approximate large means deterministically.
        if mean > 200.0:
            return int(max(0, round(mean)))
        return int(np.random.poisson(lam=mean))

    def _record_metrics(
        self,
        result: SimulationResult,
        include_bb: bool,
    ) -> None:
        T = float(self.current_time)
        total_alive = len(self.alive_ids)

        # Accumulate observer weights per vacuum over alive patches
        total_weight = 0.0
        per_vac_weight = {v.vacuum_id: 0.0 for v in self.config.vacua}
        bb_weight = 0.0
        struct_weight = 0.0

        for pid in list(self.alive_ids):
            p = self.patches[pid]
            if not p.is_alive(T):
                continue
            vac = self.vacua_by_id[p.vacuum_id]
            age = p.age(T)
            o_tot, o_s, o_b = total_observers(vac, age)
            per_vac_weight[p.vacuum_id] += o_tot
            total_weight += o_tot
            if include_bb:
                bb_weight += o_b
                struct_weight += o_s

        result.times.append(T)
        result.total_patches.append(int(total_alive))

        if total_weight > 0.0:
            for v in self.config.vacua:
                result.W_series[v.vacuum_id].append(float(per_vac_weight[v.vacuum_id] / total_weight))
        else:
            for v in self.config.vacua:
                result.W_series[v.vacuum_id].append(0.0)

        if include_bb:
            assert result.BB_fraction is not None
            assert result.BB_weight is not None
            assert result.struct_weight is not None

            denom = bb_weight + struct_weight
            frac = float(bb_weight / denom) if denom > 0.0 else 0.0
            result.BB_fraction.append(frac)
            result.BB_weight.append(float(bb_weight))
            result.struct_weight.append(float(struct_weight))

    def run(self, include_bb_metrics: bool = False) -> SimulationResult:
        np.random.seed(self.config.seed)

        result = SimulationResult(
            times=[],
            total_patches=[],
            W_series={v.vacuum_id: [] for v in self.config.vacua},
            BB_fraction=[] if include_bb_metrics else None,
            BB_weight=[] if include_bb_metrics else None,
            struct_weight=[] if include_bb_metrics else None,
            meta={},
        )

        # Initialize patches at time 0
        if self.config.initial_mode == "concentrated":
            v0 = int(self.config.initial_vacuum_id)  # type: ignore[arg-type]
            for _ in range(self.config.n_initial):
                self._new_patch(v0, birth_time=0.0)
        else:
            for _ in range(self.config.n_initial):
                v = int(np.random.randint(0, self.n_vacua))
                self._new_patch(v, birth_time=0.0)

        # Record initial state at T=0
        self.current_time = 0.0
        self._record_metrics(result, include_bb=include_bb_metrics)

        # Main event loop
        while True:
            if len(self.alive_ids) == 0:
                break
            if self.current_time >= self.config.T_max:
                break
            if self.events_processed >= self.config.max_events:
                break
            if len(self.alive_ids) >= self.config.N_max:
                break
            if len(self.event_heap) == 0:
                break

            t_next, pid = heapq.heappop(self.event_heap)
            p = self.patches.get(pid)
            if p is None or not p.alive:
                continue
            # Lazy deletion check: ensure this heap event matches current death_time
            if abs(p.death_time - t_next) > 1e-12:
                continue

            self.current_time = float(t_next)
            if self.current_time > self.config.T_max:
                break

            # Parent decays now
            vac = self.vacua_by_id[p.vacuum_id]
            age_at_death = float(self.current_time - p.birth_time)
            k = self._sample_num_daughters(vac, age_at_death)

            # Create daughters
            if k > 0:
                for _ in range(k):
                    dv = self._sample_daughter_vacuum_or_sink(p.vacuum_id)
                    if dv is None:
                        continue
                    self._new_patch(dv, birth_time=self.current_time)

            # Kill parent
            p.alive = False
            self.alive_ids.discard(p.id)

            self.events_processed += 1

            # Logging: event cadence and time checkpoints
            should_log = (self.events_processed % self.config.log_interval_events) == 0
            while self._next_checkpoint_idx < len(self._checkpoints) and self.current_time >= self._checkpoints[
                self._next_checkpoint_idx
            ]:
                should_log = True
                self._next_checkpoint_idx += 1

            if should_log:
                self._record_metrics(result, include_bb=include_bb_metrics)

        # Ensure final state is recorded at end time (if last record not already at current_time)
        if len(result.times) == 0 or result.times[-1] != self.current_time:
            self._record_metrics(result, include_bb=include_bb_metrics)

        return result


def make_smooth_topology() -> Tuple[List[VacuumSpec], np.ndarray]:
    """
    Smooth topology: linear vacua 0-1-2 with local transitions; 10% sink each.
    Uses baseline params (no intelligence amplification, no BBs).
    """
    tau = [1.0, 10.0, 100.0]
    alpha = [1.0, 1.0, 1.0]
    beta = [1.0, 1.2, 1.5]
    kappa = [0.8, 1.0, 1.2]
    o_struct = [1.0, 1.0, 1.0]
    o_bb = [0.0, 0.0, 0.0]
    t_bb = [float(np.inf), float(np.inf), float(np.inf)]

    vacua = [
        VacuumSpec(
            vacuum_id=i,
            tau=tau[i],
            alpha=alpha[i],
            kappa=kappa[i],
            beta=beta[i],
            o_struct=o_struct[i],
            o_bb=o_bb[i],
            t_bb=t_bb[i],
            label=f"Vacuum {i}",
        )
        for i in range(3)
    ]

    T = np.array(
        [
            [0.7, 0.2, 0.0],  # sink 0.1
            [0.2, 0.6, 0.1],  # sink 0.1
            [0.0, 0.3, 0.6],  # sink 0.1
        ],
        dtype=float,
    )
    validate_transition_matrix(T)
    return vacua, T


def make_rugged_topology() -> Tuple[List[VacuumSpec], np.ndarray]:
    """
    Rugged topology: weak locality / high randomness. Each row transitions equally
    to the three vacua, with 25% sink implied.
    Uses baseline vacuum params.
    """
    vacua, _ = make_smooth_topology()
    T = np.array(
        [
            [0.25, 0.25, 0.25],  # sink 0.25
            [0.25, 0.25, 0.25],  # sink 0.25
            [0.25, 0.25, 0.25],  # sink 0.25
        ],
        dtype=float,
    )
    validate_transition_matrix(T)
    return vacua, T


def make_cluster_topology() -> Tuple[List[VacuumSpec], np.ndarray]:
    """
    Cluster-basin topology:
    Cluster {0,1} with stronger intra-cluster transitions; vacuum 2 is a basin.
    Uses baseline vacuum params.
    """
    vacua, _ = make_smooth_topology()
    T = np.array(
        [
            [0.5, 0.3, 0.05],  # sum 0.85 -> sink 0.15
            [0.3, 0.5, 0.05],  # sum 0.85 -> sink 0.15
            [0.05, 0.05, 0.7],  # sum 0.8 -> sink 0.2
        ],
        dtype=float,
    )
    validate_transition_matrix(T)
    return vacua, T


def override_vacua_params(
    vacua: List[VacuumSpec],
    *,
    alpha: Optional[List[float]] = None,
    o_bb: Optional[List[float]] = None,
    t_bb: Optional[List[float]] = None,
) -> List[VacuumSpec]:
    """
    Return a new vacua list with selected parameters overridden by index.
    """
    n = len(vacua)
    alpha = alpha if alpha is not None else [v.alpha for v in vacua]
    o_bb = o_bb if o_bb is not None else [v.o_bb for v in vacua]
    t_bb = t_bb if t_bb is not None else [v.t_bb for v in vacua]
    if not (len(alpha) == len(o_bb) == len(t_bb) == n):
        raise ValueError("Override arrays must match number of vacua.")

    out: List[VacuumSpec] = []
    for i, v in enumerate(vacua):
        out.append(
            VacuumSpec(
                vacuum_id=v.vacuum_id,
                tau=v.tau,
                alpha=float(alpha[i]),
                kappa=v.kappa,
                beta=v.beta,
                o_struct=v.o_struct,
                o_bb=float(o_bb[i]),
                t_bb=float(t_bb[i]),
                label=v.label or f"Vacuum {v.vacuum_id}",
            )
        )
    return out

