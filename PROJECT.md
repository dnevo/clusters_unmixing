# clusters_unmixing — Project Document

A compact spectral unmixing experiment framework for comparing preprocessing choices and
unmixing models (both classical solvers and neural networks) on configurable spectral
cluster sets, with a focus on soil spectra from a Judean Desert climatic gradient.

This document is a comprehensive reference: domain background, architecture, data flow,
the models being compared, configuration, and open design considerations. For a quick
orientation, see `README.md`; for the historical refactor rationale, see `REFACTOR_NOTES.md`.

## Table of contents

1. [What the project does](#what-the-project-does)
2. [Domain background: the Judean Desert soil clusters](#domain-background-the-judean-desert-soil-clusters)
3. [Architecture](#architecture)
4. [End-to-end data flow](#end-to-end-data-flow)
5. [Synthetic data generation](#synthetic-data-generation)
6. [Mixing models: linear vs. nonlinear](#mixing-models-linear-vs-nonlinear)
7. [Unmixing models compared](#unmixing-models-compared)
8. [Configuration reference](#configuration-reference)
9. [Outputs](#outputs)
10. [Running the project](#running-the-project)
11. [Known limitations and open considerations](#known-limitations-and-open-considerations)
12. [References](#references)

## What the project does

The project runs configured *experiment batches*. Each batch:

- loads one or more cluster CSV files, each holding a small set of mean reflectance
  spectra ("endmembers") over a wavelength axis;
- generates synthetic abundance vectors and, from them, synthetic mixed pixel spectra
  (optionally with a tunable nonlinear mixing term and additive sensor noise);
- applies wavelength-range selection, optional normalization, and optional spectral
  transforms (first derivative, PCA) at each preprocessing "stage";
- computes endmember-separability correlation metrics (cosine similarity, spectral angle)
  at every stage, independent of any unmixing model;
- evaluates every registered unmixing model (classical solvers and neural networks) on
  the same synthetic pixels, scored on a shared held-out test split;
- writes summary CSVs for correlation, per-model metrics, and abundance previews;
- supports notebook-based review of the same outputs (spectra, metrics, abundance
  tables, synthetic pixel previews).

The default experiment configuration lives in `experiments/configs/configuration.yaml`.

## Domain background: the Judean Desert soil clusters

The primary cluster set, `data/6clusters_thomas.csv`, is derived from:

> Jarmer, T., & Shoshany, M. (2015). *Relationships between soil spectral and chemical
> properties along a climatic gradient in the Judean desert.* Arid Land Research and
> Management.

This context matters for how the rest of the project should be interpreted, so it's
worth being explicit about what the "6 clusters" actually are.

**What the clusters are not:** they are not 6 distinct surface materials (e.g. sand,
gravel, rock, vegetation). There is no vegetation or bare-rock spectrum anywhere in this
library.

**What the clusters actually are:** 6 clusters of the *same* limestone-derived soil,
distinguished only by the concentration of three chemical properties — organic carbon
(Corg), inorganic carbon/carbonate (Cinorg), and total iron (Fe_total) — that vary
continuously along a ~30 km climatic transect from the Judean Mountains (Mediterranean
climate, ~620 mm/year rainfall, 650 m elevation) down to the Dead Sea (hyper-arid,
~120 mm/year, 60 m below sea level). Fifty-five soil samples were collected along twelve
topographic transects, air-dried, and **sieved to <2 mm** — a step that explicitly
removes rock fragments and vegetation/organic litter before spectral measurement (ASD
FieldSpec II, 350–2500 nm, 6 resampled Landsat TM bands).

Hierarchical cluster analysis on (Corg, Cinorg, Fe_total) produced 6 clusters that are
**monotonically ordered along the rainfall/weathering gradient**:

- Clusters 1–2: wettest, best-developed soils (high Fe_total, high Corg, low Cinorg).
- Clusters 5–6: driest, poorly-developed soils (opposite pattern).
- Clusters 3–4: intermediate, distinguished from each other mainly by Corg.

Discriminant analysis on the Landsat-band reflectances classified samples into these 6
groups with 85.5% accuracy overall, but **clusters 3, 4, and 5 are only weakly separable
spectrally** — the paper's own analysis found their mean reflectance spectra close
together, differing mainly in near/mid-infrared brightness. This is the direct
motivation for the empirical band-selection comment in `configuration.yaml` (a
narrower, cosine-correlation-optimized band selection actually performed *worse* than
using nearly the full spectrum — endmember collinearity here is a real, hard problem,
not an artifact of a bad wavelength choice).

The CSV column order in `data/6clusters_thomas.csv` (`Cluster6, Cluster5, ..., Cluster1`)
walks this gradient monotonically, and `dataio.py`'s `load_wavelength_and_cluster_matrix`
preserves column order verbatim — so endmember index adjacency corresponds to gradient
adjacency for this cluster set specifically (not a general property of arbitrary cluster
sets).

**Practical implications for synthetic data design:**

- Real sub-pixel mixing between *distant* clusters (e.g. Cluster 1 with Cluster 6) is
  physically rare — they sit kilometers apart on the climatic transect. The paper does
  document one documented exception: a local micro-topographic runoff anomaly in the
  250–350 mm rainfall zone placed a well-developed Cluster 2 soil near a degraded
  Cluster 6 soil. So "distant" mixtures are rare but not impossible.
- Because rock outcrop (documented as covering roughly a third of these surfaces) and
  vegetation were both excluded by the <2 mm sieving, this endmember library cannot
  represent the most common real sub-pixel mixtures in this terrain (soil + rock,
  soil + vegetation) at all — only soil-cluster-with-soil-cluster mixtures. This is a
  bigger realism gap than any choice about the mixing model or abundance sampling
  strategy.
- Because the clusters are chemically graded variants of the *same* soil matrix rather
  than optically distinct materials, physical stories built around "two different
  materials touching at a sub-pixel boundary" (e.g. the bilinear/GBM nonlinear mixing
  model, see below) don't have a strong physical grounding for this specific dataset —
  they're still useful as numerical stress tests, just not as a claim about the actual
  optical mechanism at these field sites.

The secondary cluster set, `data/6clusters_digitized.csv`, is **deprecated** and will be
removed; it is already commented out of `cluster_sets` in `configuration.yaml`. Only
`6clusters_thomas` is in active use.

## Architecture

```
main.py                                  CLI entry point
notebooks/00_clusters_unmixing_experiments.ipynb   interactive review notebook
notebooks/_notebook_smoke.py             scriptable smoke test for the notebook flow

src/clusters_unmixing/
  config/schema.py                       pydantic experiment configuration + validation
  dataio.py                              cluster CSV loading (wavelength axis + endmember matrix)
  data/synthetic.py                      synthetic abundance vector generation
  core_math.py                           correlation metrics, SNR noise, GBM mixing (mix_pixels)
  transforms/spectral_views.py           wavelength-range selection, first-derivative, PCA
  transforms/normalization.py            quadratic-baseline normalization
  pipelines/experiment_pipeline.py       orchestrates a full experiment batch (run_experiments)
  models/
    runner_registry.py                   model name -> runner dispatch, tensor/device plumbing
    sunsal.py                            SUnSAL (ADMM constrained sparse regression)
    vpgdu.py                             VPGDU (vectorized projected gradient descent)
    mlm.py                               Multilinear Mixing Model (nonlinearity-aware comparison)
    base_nn.py                           shared supervised-NN training scaffolding
    small_mlp.py, convnext1d.py, kan.py, mamba.py   supervised NN unmixing models
  utils/notebook_diagnostics.py          notebook orchestration, tables, plotting helpers
```

Everything downstream of `run_experiments` operates on plain NumPy arrays at the
pipeline level; the model layer converts to PyTorch tensors (CUDA if available) inside
`runner_registry.run_registered_model` and converts back before returning.

## End-to-end data flow

For each configured run in `model_evaluation.runs`:

1. **Load the cluster set.** `dataio.load_wavelength_and_cluster_matrix` reads the CSV
   into a wavelength axis and an `(n_endmembers, n_bands)` matrix, preserving column
   order exactly as it appears in the file.
2. **Seed globally** (`random`, `numpy`, `torch` all seeded to `0`) so the same run
   config always regenerates the same synthetic pixels.
3. **Generate synthetic abundances.** `data.synthetic.generate_samples` produces
   `num_pixels` abundance vectors on the probability simplex (see below).
4. **Mix into pixel spectra.** `core_math.mix_pixels(abundances, endmembers,
   nonlinearity_gamma)` combines abundances and endmembers — linear mixing at
   `nonlinearity_gamma=0.0`, with an optional bilinear (GBM) nonlinear term otherwise.
5. **Add sensor noise.** `core_math.apply_snr_noise` adds Gaussian noise at the
   configured `snr_db`.
6. **Build preprocessing stage projections.** Starting from the raw (noisy) pixels and
   raw endmembers: select wavelength ranges (`transforms.spectral_views`), then apply
   normalization (`transforms.normalization`), then apply each configured transform step
   in order (`first_derivative`, `pca`). Each stage's `(endmembers, pixels)` pair is kept
   for later comparison.
7. **Assess endmember separability per stage.** For each configured metric (`cosine`,
   `sam`) and each stage, `core_math.compute_correlation_matrix` /
   `summarize_correlation_matrix` report how collinear the endmembers are — this is
   independent of any unmixing model and is exactly what surfaces the
   cluster-3/4/5 collinearity problem described above.
8. **Pick a shared held-out test split.** `_make_held_out_test_indices` reserves ~10% of
   pixels; the same indices are used to score every model in the run, so supervised NN
   models (trained on ~90% of the pixels) aren't compared unfairly against iterative
   solvers that never train on labels at all.
9. **Run every configured model** via `runner_registry.run_registered_model`, on the
   final preprocessing stage's `(endmembers, pixels)`.
10. **Score and log.** `abundance_rmse` and `reconstruction_rmse` are computed on the
    held-out test pixels for every model; a handful of preview pixels are logged with
    their true and per-model estimated abundances for notebook review.
11. **Write outputs**: `correlation_summary.csv`, `model_summary.csv`,
    `abundance_preview.csv` under `experiments/outputs/{experiment_name}/`.

## Synthetic data generation

`data/synthetic.py::generate_samples(num_samples, max_non_zero_endmembers,
num_endmembers=6)`:

- Always includes the `num_endmembers` pure endmember vectors first (one-hot).
- For every remaining sample: picks a sparsity level `k` uniformly in
  `[1, max_non_zero_endmembers]`, picks `k` active endmember indices uniformly at
  random from all `C(n, k)` combinations, then splits `1.0` among them using a
  stick-breaking construction (`k-1` random cut points among 50 discrete units) — this
  is the textbook-correct way to sample uniformly on a discrete simplex, not a naive
  "normalize k uniforms" approach that would bias toward the centroid.
- Non-zero abundances are therefore multiples of `1/50 = 0.02`.
- Randomness is driven by Python's `random` module (not `numpy.random`), so results are
  reproducible by seeding `random.seed(...)` before calling — which is exactly what
  `_set_global_seeds` does in the pipeline.

This generator samples **uniformly across all sparsity levels and all endmember
combinations**, including combinations that are physically implausible for this specific
domain (e.g. Cluster 1 mixed evenly with Cluster 6). That's a deliberate simplicity
trade-off: it's an even, unbiased stress test across mixing complexity, at the cost of
not reflecting the gradient-adjacency structure described in the domain section above. A
gradient-adjacency-biased variant was prototyped and explicitly rejected as too complex
for the value it added (see `REFACTOR_NOTES.md`-style history in git log); the generator
remains intentionally simple.

## Mixing models: linear vs. nonlinear

**Linear Mixing Model (LMM)** — the default (`nonlinearity_gamma=0.0`): pixel spectrum is
exactly `abundances @ endmembers`.

**Generalized Bilinear Model / Fan model (GBM)** — `core_math.mix_pixels`:

```
pixel = sum_i a_i * m_i  +  gamma * sum_{i<j} a_i * a_j * (m_i ⊙ m_j)
```

`gamma=0` recovers the LMM exactly; `gamma=1` is the Fan model (no free parameter per
endmember pair, full pairwise interaction). This models the classic bilinear/GBM story
of light bouncing once between two different materials before reaching the sensor. It's
wired into `model_evaluation.runs[*].nonlinearity_gamma` in `configuration.yaml`, so any
run can be swept from purely linear to strongly nonlinear.

As noted in the domain section, this project's specific cluster set (chemically graded
variants of one soil, not optically distinct materials) doesn't have strong physical
grounding for the "two distinct materials touching at a boundary" story — `gamma` is
best treated here as a numerical robustness knob (how well do solvers handle
mixing-model mismatch?) rather than a physically motivated simulation of the real scene.

**Multilinear Mixing Model (MLM)** — `models/mlm.py`, based on:

> Heylen, R., & Scheunders, P. (2016). *A Multilinear Mixing Model for Nonlinear
> Spectral Unmixing.* IEEE TGRS, 54(1), 240–251.

MLM is used only as a *comparison unmixing model* (see below), not as a pixel generator.
It derives a closed-form extension of the bilinear story to an infinite number of
photon bounces: modeling each interaction as having probability `P` of bouncing again
and `(1-P)` of escaping to the sensor (a discrete Markov chain), the infinite series of
increasingly higher-order interactions telescopes into:

```
x = (1 - P) * y / (1 - P * y),   y = sum_i a_i * w_i   (elementwise per band)
```

`P=0` recovers the LMM exactly. Unlike the paper's typical usage (`0 <= P < 1`, i.e.
reflectance always decreases relative to linear mixing), this project's `MLMConfig`
allows `P` to go as negative as `p_min` (default `-5.0`), because this project's own
GBM/Fan generator only ever *increases* reflectance relative to linear mixing — fitting
MLM to GBM-generated data requires negative `P`, exactly as the original paper found
when testing MLM against other bilinear models' synthetic data.

## Unmixing models compared

| Model | Type | Supervision | Key idea | Reference |
|---|---|---|---|---|
| `sunsal` | Classical, iterative | Unsupervised (no labels used) | ADMM solver for constrained sparse regression under ANC + ASC | Bioucas-Dias & Figueiredo (2010), IEEE JSTSP |
| `vpgdu` | Classical, iterative | Unsupervised | SAM-based initial estimate + vectorized projected gradient descent onto the simplex (Chen & Ye projection) | Kizel et al., "A Stepwise Analytical Projected Gradient Descent Search for Hyperspectral Unmixing..." |
| `mlm` | Classical, alternating fit | Unsupervised | Estimates abundances *and* a single global nonlinearity parameter `P` (Multilinear Mixing Model); internally alternates a grid search over `P` with re-fitting abundances via SunSAL on a transformed target | Heylen & Scheunders (2016), IEEE TGRS |
| `small_mlp` | Neural network | Supervised | 3-layer MLP, softmax output for simplex-constrained abundances | — |
| `convnext1d` | Neural network | Supervised | ConvNeXt-style 1D convolutional blocks over the spectrum, softmax abundances | — |
| `kan` | Neural network | Supervised | Kolmogorov-Arnold Network: per-edge base activation + learned B-spline, softmax abundances | — |
| `mamba` | Neural network | Supervised | Selective state-space model (Mamba) over patchified spectra; GPU-only, requires the optional `mamba-ssm` package (`pyproject.toml`'s `mamba` extra) | — |

All four neural models share `models/base_nn.py`'s training scaffolding:
70%/20%/held-out-test split (the held-out test set is supplied externally so every
model in a run is scored on identical pixels), joint loss
`abundance_MSE + lambda_recon * reconstruction_MSE` (predicted abundances reconstructed
against the endmembers), AdamW with gradient clipping, early stopping on validation loss,
and mixed-precision (AMP) training on CUDA.

`sunsal`, `vpgdu`, and `mlm` never see `true_abundances` at all — they only need
`endmembers` and `pixels`, which is why they can be scored on the *entire* pixel set
fairly (they never "trained" on any of it), unlike the supervised NN models.

## Configuration reference

`ExperimentConfig.from_config_file(project_root)` loads
`experiments/configs/configuration.yaml`. Top-level fields:

- `experiment_name` — output folder name under `experiments/outputs/`.
- `cluster_sets` — list of `{name, path}` cluster CSVs available to runs.
- `metrics` — required non-empty list of correlation metrics (`cosine`, `sam`) computed
  on endmembers at every preprocessing stage.
- `model_evaluation.models` — per-model-name hyperparameters (validated per-model where
  a stronger schema exists, e.g. `small_mlp`'s `SmallMLPParamsModel`; otherwise passed
  through as a dict to that model's own `dataclass` config).
- `model_evaluation.runs` — one or more concrete runs, each specifying:
  - `cluster_set` — which entry in `cluster_sets` to use.
  - `bands_ranges` — one or more `[x_min, x_max]` (µm) wavelength windows, each with
    `reduce: none` (keep all bands) or `reduce: mean` (collapse the window to one value).
  - `normalization` — `without` or `with_quadratic` (subtracts a fixed quadratic
    baseline in wavelength space).
  - `transform.steps` — ordered list of `first_derivative` and/or `pca` (PCA must come
    after first-derivative if both are used).
  - `num_pixels` — synthetic pixel count (must be `> 10`).
  - `snr_db` — signal-to-noise ratio for additive Gaussian noise (`inf` for no noise).
  - `nonlinearity_gamma` — GBM nonlinearity strength in `[0, 1]` (default `0.0`, i.e.
    pure linear mixing).
  - `models` — which registered model names to run for this configuration.

Relative paths (like `cluster_sets[*].path`) are resolved from the supplied project
root.

## Outputs

Each run batch writes results under `experiments/outputs/{experiment_name}/`:

- `correlation_summary.csv` — per-run, per-metric, per-stage endmember separability
  statistics (`mean_abs_offdiag`, `max_abs_offdiag`, `min_offdiag`, `max_offdiag`).
- `model_summary.csv` — one row per run and metric (`abundance_rmse`,
  `reconstruction_rmse`), one column per configured model, computed on the shared
  held-out test pixels.
- `abundance_preview.csv` — a handful of preview pixels per run, with `pixel_index`,
  `source` (`true` or a model name), error columns, and `endmember_*` abundance values —
  intended for the notebook's per-pixel comparison tables/plots.

## Running the project

```bash
python -m pip install -e .
python main.py
```

Notebook-only dependencies (for `notebooks/00_clusters_unmixing_experiments.ipynb` and
`notebooks/_notebook_smoke.py`):

```bash
python -m pip install ipython plotly
```

The `mamba` model additionally requires a CUDA-capable GPU and the optional
`mamba-ssm` package (`pip install -e .[mamba]`); its absence is handled gracefully —
`runner_registry.KNOWN_MODEL_NAMES` still recognizes `"mamba"` for config validation,
but `available_models()` (and hence actually running it) requires the package to import
successfully.

## Known limitations and open considerations

- **No rock or vegetation endmembers.** The <2 mm sieving that produced
  `6clusters_thomas.csv` removed both, even though bedrock outcrop reportedly covers
  about a third of these surfaces. Real desert-fringe pixels most plausibly mix soil
  with rock and/or vegetation — a mixture this endmember library cannot represent at
  all. This is a bigger realism gap than anything about the mixing model or abundance
  sampling strategy.
- **Uniform-arbitrary synthetic mixtures.** `generate_samples` treats all endmember
  combinations as equally likely, including physically implausible ones (e.g. Cluster 1
  with Cluster 6). A gradient-adjacency-aware alternative was prototyped and rejected as
  unnecessary complexity; the simplicity trade-off is intentional, not an oversight.
- **GBM (`nonlinearity_gamma`) is a numerical stress test, not a physical claim** for
  this specific cluster set, since the clusters aren't optically distinct materials.
- **MLM does not universally outperform the linear solvers** on GBM-generated data.
  Verified experimentally: on data generated by the exact MLM equation, MLM recovers
  abundances far more accurately than SunSAL; on `mix_pixels`-generated (GBM) data, MLM
  achieves a *better pixel reconstruction* than SunSAL (it has an extra free parameter)
  but a *worse abundance estimate* — consistent with the original paper's own finding
  that each nonlinear model tends to only outperform on data generated by its own
  assumptions. Its value in this project is as a differently-biased comparison point and
  a per-pixel nonlinearity-strength diagnostic (`P` estimates), not a drop-in upgrade.
- **`6clusters_digitized.csv` is deprecated** and slated for removal; it's already
  commented out of `configuration.yaml`'s `cluster_sets`.

## References

- Jarmer, T., & Shoshany, M. (2015). *Relationships between soil spectral and chemical
  properties along a climatic gradient in the Judean desert.* Arid Land Research and
  Management.
- Bioucas-Dias, J., & Figueiredo, M. (2010). *Alternating Direction Algorithms for
  Constrained Sparse Regression: Application to Hyperspectral Unmixing.* IEEE JSTSP.
- Kizel, F., et al. *A Stepwise Analytical Projected Gradient Descent Search for
  Hyperspectral Unmixing and Its Code Vectorization.*
- Fan, W., Hu, B., Miller, J., & Li, M. (2009). *Comparative study between a new
  nonlinear model and common linear model for analysing laboratory simulated-forest
  hyperspectral data.* International Journal of Remote Sensing.
- Nascimento, J. M. P., & Bioucas-Dias, J. M. (2009). *Nonlinear Mixture Model for
  Hyperspectral Unmixing.* Proc. SPIE.
- Halimi, A., Altmann, Y., Dobigeon, N., & Tourneret, J.-Y. (2011). *Nonlinear Unmixing
  of Hyperspectral Images Using a Generalized Bilinear Model.* IEEE TGRS.
- Heylen, R., & Scheunders, P. (2016). *A Multilinear Mixing Model for Nonlinear
  Spectral Unmixing.* IEEE TGRS, 54(1), 240–251.
