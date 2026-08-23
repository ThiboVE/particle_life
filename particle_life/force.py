from dataclasses import dataclass

import numpy as np


@dataclass
class ForceSettings:
    dt: float = 0.02
    rMax: float = 0.1
    frictionHalfLife: float = 0.040
    frictionFactor: float = pow(0.5, dt / frictionHalfLife)
    forceFactor: int = 40


# def force_func(r: float, a: float) -> float:
#     beta = 0.3
#     if r < beta:
#         return (r / beta) - 1
#     elif beta < r < 1:
#         return a * (1 - abs(2 * r - 1 - beta) / (1 - beta))
#     return 0


def force_func(
    dist: np.ndarray, interaction: np.ndarray, beta: float = 0.3
) -> np.ndarray:
    close_range = dist < beta
    mid_range = (dist >= beta) & (dist < 1)

    close_val = (dist / beta) - 1
    mid_val = interaction * (1 - np.abs(2 * dist - 1 - beta) / (1 - beta))

    return np.select([close_range, mid_range], [close_val, mid_val], default=0.0)


def compute_forces(
    positions: np.ndarray,
    type_pairs: np.ndarray,
    min_dist: float = 1e-3,
    beta: float = 0.3,
) -> np.ndarray:
    diff = positions[None, :, :] - positions[:, None, :]  # (N, N, 2)
    # diff -= np.round(diff)
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)
    dist_safe = np.where(dist < min_dist, min_dist, dist)

    r_max = 0.15
    r_scaled = dist_safe / r_max

    force_mag = force_func(r_scaled, type_pairs, beta)
    np.fill_diagonal(force_mag, 0.0)

    direction = diff / dist_safe[..., None]  # (N, N, 2)
    force = (direction * force_mag[..., None]).sum(axis=1)  # (N, 2)
    return force


def timer(function):
    from time import perf_counter

    def wrapper(*args, **kwargs):
        before = perf_counter()
        value = function(*args, **kwargs)
        after = perf_counter()
        print(f"'{function.__name__}' took {after - before} seconds to execute!")
        return value

    return wrapper
