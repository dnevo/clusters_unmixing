from __future__ import annotations

import random

import numpy as np


ABUNDANCE_DISTRIBUTIONS = {"grid", "dirichlet"}

def _generate_split_fractions(k: int, num_fractions_base: int) -> list[float]:
    """Generates k positive fractions that sum to 1.0, with each being a multiple of 1/num_fractions_base."""
    if k == 1:
        return [1.0]
    # Use 'stars and bars' method to generate k positive integer parts that sum to num_fractions_base
    # This involves choosing k-1 split points from num_fractions_base - 1 possible positions
    split_points = sorted(random.sample(range(1, num_fractions_base), k - 1))
    
    parts: list[int] = []
    prev_point = 0
    for point in split_points:
        parts.append(point - prev_point)
        prev_point = point
    parts.append(num_fractions_base - prev_point)
    
    return [p / float(num_fractions_base) for p in parts]


def generate_samples(
    num_samples: int,
    max_non_zero_endmembers: int,
    num_endmembers: int = 6,
    abundance_distribution: str = "grid",
    dirichlet_alpha: float = 0.3,
) -> np.ndarray:
    """Generate synthetic abundance vectors.

    ``abundance_distribution="grid"`` matches the legacy project:
    - include pure endmembers first
    - each sample has at most ``max_non_zero_endmembers`` active components
    - abundances sum to 1.0
    - non-zero abundances are multiples of 0.04 (1/25)

    ``abundance_distribution="dirichlet"`` draws every sample directly from
    a symmetric Dirichlet distribution over all endmembers. Values of
    ``dirichlet_alpha`` below 1 produce sparse-looking mixtures, 1 is uniform
    over the continuous simplex, and values above 1 produce more balanced
    mixtures. Dirichlet samples do not contain exact zeros and pure samples
    are not inserted automatically.

    Randomness is intentionally driven by Python's ``random`` module so callers can
    reproduce grid results by resetting ``random.seed(...)`` before calling.
    Dirichlet results use NumPy's RNG and are reproducible via ``np.random.seed``.
    """
    distribution = str(abundance_distribution).strip().lower()
    if distribution == "dirichlet":
        concentration = np.full(num_endmembers, float(dirichlet_alpha), dtype=np.float64)
        return np.random.dirichlet(concentration, size=num_samples).astype(np.float32)

    unique_generated_samples: set[tuple] = set()
    # The denominator for fraction granularity, currently set to 25 for 0.04 increments
    # total distinct combinations: math.comb(25-1, 6-1) = 42,504 
    # (should be > num_samples for 6 endmembers)
    num_fractions_base = 25

    # Add pure endmembers
    for i in range(num_endmembers):
        pure_endmember = np.zeros(num_endmembers, dtype=np.float32)
        pure_endmember[i] = 1.0
        unique_generated_samples.add(tuple(pure_endmember.tolist()))

    # If we already have enough (or more than enough) unique pure samples
    if len(unique_generated_samples) >= num_samples:
        final_samples = [np.array(s, dtype=np.float32) for s in list(unique_generated_samples)][:num_samples]
        return np.stack(final_samples, axis=0)

    # Generate random samples until we have num_samples unique entries
    # This loop might take a long time or run indefinitely if the sample space is exhausted or too small
    while len(unique_generated_samples) < num_samples:
        k = random.randint(1, min(max_non_zero_endmembers, num_endmembers))
        non_zero_indices = random.sample(range(num_endmembers), k)

        fractions = _generate_split_fractions(k, num_fractions_base)

        sample = np.zeros(num_endmembers, dtype=np.float32)
        for idx, val in zip(non_zero_indices, fractions):
            sample[idx] = float(val)
        unique_generated_samples.add(tuple(sample.tolist()))

    # Convert the set of unique tuples back to a list of numpy arrays
    final_samples = [np.array(s, dtype=np.float32) for s in list(unique_generated_samples)]

    # Take exactly num_samples, ensuring uniqueness
    return np.stack(final_samples[:num_samples], axis=0)
