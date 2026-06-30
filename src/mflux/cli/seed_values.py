import random
import time
from collections.abc import Callable

AUTO_SEED_MAX = int(1e7)


def resolve_seed_values(
    *,
    seed_values: list[int] | None,
    auto_seeds: int,
    default_seed_factory: Callable[[], int] | None = None,
) -> list[int]:
    validate_auto_seed_count(auto_seeds)
    if seed_values is not None:
        seeds = list(seed_values)
    elif auto_seeds > 0:
        seeds = generate_auto_seeds(auto_seeds)
    else:
        factory = default_seed_factory or _default_seed_factory
        seeds = [factory()]
    validate_unique_seed_values(seeds)
    return seeds


def validate_auto_seed_count(auto_seeds: int) -> None:
    if auto_seeds != -1 and auto_seeds < 1:
        raise ValueError("--auto-seeds must be greater than zero.")


def validate_unique_seed_values(seeds: list[int]) -> None:
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "--seed values must be unique within one invocation because repeated seeds would "
            "overwrite the same output path."
        )


def generate_auto_seeds(auto_seeds: int) -> list[int]:
    if auto_seeds > AUTO_SEED_MAX + 1:
        return [random.randint(0, AUTO_SEED_MAX) for _ in range(auto_seeds)]
    return random.sample(range(AUTO_SEED_MAX + 1), auto_seeds)


def _default_seed_factory() -> int:
    return int(time.time())
