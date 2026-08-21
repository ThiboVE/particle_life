from dataclasses import dataclass


@dataclass
class ForceSettings:
    dt: float = 0.02
    rMax: float = 0.1
    frictionHalfLife: float = 0.040
    frictionFactor: float = pow(0.5, dt / frictionHalfLife)
    forceFactor: int = 40


def force_func(r: float, a: float) -> float:
    beta = 0.3
    if r < beta:
        return (r / beta) - 1
    elif beta < r < 1:
        return a * (1 - abs(2 * r - 1 - beta) / (1 - beta))
    return 0


def timer(function):
    from time import perf_counter

    def wrapper(*args, **kwargs):
        before = perf_counter()
        value = function(*args, **kwargs)
        after = perf_counter()
        print(f"'{function.__name__}' took {after - before} seconds to execute!")
        return value

    return wrapper
