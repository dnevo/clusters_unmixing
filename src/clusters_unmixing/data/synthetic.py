from __future__ import annotations

import random

import torch


def generate_samples(num_samples: int, max_non_zero_endmembers: int, num_endmembers: int = 6) -> torch.Tensor:
    """Generate legacy-compatible synthetic abundance vectors.

    Behavior matches the legacy project:
    - include pure endmembers first
    - each sample has at most ``max_non_zero_endmembers`` active components
    - abundances sum to 1.0
    - non-zero abundances are multiples of 0.1

    Randomness is intentionally driven by Python's ``random`` module so callers can
    reproduce the legacy results by resetting ``random.seed(...)`` before calling.
    """
    generated_samples: list[torch.Tensor] = []

    for i in range(num_endmembers):
        pure_endmember = torch.zeros(num_endmembers, dtype=torch.float32)
        pure_endmember[i] = 1.0
        generated_samples.append(pure_endmember)

    while len(generated_samples) < num_samples:
        k = random.randint(1, min(max_non_zero_endmembers, num_endmembers))
        non_zero_indices = random.sample(range(num_endmembers), k)

        if k == 1:
            fractions = [1.0]
        else:
            split_points = sorted(random.sample(range(1, 10), k - 1))
            parts: list[int] = []
            prev_point = 0
            for point in split_points:
                parts.append(point - prev_point)
                prev_point = point
            parts.append(10 - prev_point)
            fractions = [p / 10.0 for p in parts]

        sample = torch.zeros(num_endmembers, dtype=torch.float32)
        for idx, val in zip(non_zero_indices, fractions):
            sample[idx] = float(val)
        generated_samples.append(sample)

    return torch.stack(generated_samples[:num_samples])
