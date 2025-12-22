from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import ks_2samp
import numpy as np


@dataclass
class SlicedKSMetrics:
    mean: float
    std: float
    sem: float
    ci_low: float
    ci_high: float
    per_direction: np.ndarray


def _ks_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Kolmogorov–Smirnov distance between two samples.

    If scipy is available, we use ks_2samp; otherwise we fall back to a simple implementation.
    """
    if a.size == 0 or b.size == 0:
        return 0.0

    if ks_2samp is not None:
        return ks_2samp(a, b, alternative="two-sided", mode="auto").statistic

    # Fallback: manual KS
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    data_all = np.concatenate([a_sorted, b_sorted])

    # Empirical CDFs
    cdf_a = np.searchsorted(a_sorted, data_all, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, data_all, side="right") / b_sorted.size

    return np.max(np.abs(cdf_a - cdf_b))


def sliced_ks_distance(
    emb_A: np.ndarray,
    emb_B: np.ndarray,
    n_directions: int = 64,
    random_state: Optional[int] = None,
) -> SlicedKSMetrics:
    """Compute Sliced Kolmogorov–Smirnov distance between two embedding sets.

    Parameters
    ----------
    emb_A : np.ndarray
        Embeddings of set A, shape (N_A, D).
    emb_B : np.ndarray
        Embeddings of set B, shape (N_B, D).
    n_directions : int
        Number of random projection directions.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    SlicedKSMetrics
        Container with mean, std, sem, CI, and per-direction KS values.
    """
    if emb_A.size == 0 or emb_B.size == 0:
        return SlicedKSMetrics(
            mean=0.0, std=0.0, sem=0.0, ci_low=0.0, ci_high=0.0,
            per_direction=np.zeros(n_directions)
        )

    assert emb_A.shape[1] == emb_B.shape[1], "Embedding dimensions must match."
    d = emb_A.shape[1]

    rng = np.random.default_rng(random_state)

    ks_values = []
    for _ in range(n_directions):
        # Sample a random direction on the unit sphere
        v = rng.normal(size=d)
        v /= np.linalg.norm(v) + 1e-12

        proj_A = emb_A @ v
        proj_B = emb_B @ v

        ks_val = _ks_1d(proj_A, proj_B)
        ks_values.append(ks_val)

    ks_values = np.array(ks_values)
    mean = float(ks_values.mean())
    std = float(ks_values.std(ddof=1)) if ks_values.size > 1 else 0.0
    sem = float(std / np.sqrt(ks_values.size)) if ks_values.size > 0 else 0.0
    # 95% CI (approx)
    ci_low = max(0.0, mean - 1.96 * sem)
    ci_high = min(1.0, mean + 1.96 * sem)

    return SlicedKSMetrics(
        mean=mean,
        std=std,
        sem=sem,
        ci_low=ci_low,
        ci_high=ci_high,
        per_direction=ks_values,
    )


def symmetry_from_sks(metrics: SlicedKSMetrics) -> Dict[str, float]:
    """Convert sliced KS metrics into a symmetry-oriented view.

    Symmetry is defined as 1 - mean KS, with the same uncertainty structure.
    """
    sym_mean = 1.0 - metrics.mean
    sym_ci_low = 1.0 - metrics.ci_high
    sym_ci_high = 1.0 - metrics.ci_low
    return {
        "sym_mean": sym_mean,
        "sym_std": metrics.std,
        "sym_sem": metrics.sem,
        "sym_ci_low": sym_ci_low,
        "sym_ci_high": sym_ci_high,
    }
