# Neighborhood sampling

Reproducible acquisition experiments and inference for **Preserving neighborhoods in tissue sampling**.

The method samples tissue cores with known positive probabilities, acquires a border wide enough to preserve local neighborhoods, and estimates a finite tissue-wide prevalence. The benchmark combines 24 public tissue maps containing 8,833,345 cells, with four development specimens and 20 held-out validation specimens. Compact core tables support reproduction without loading full-resolution images.

## Install

Use Python 3.12 and a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Six CPU workers and 16–18 GiB of available system memory are a practical configuration for full reproduction. Fewer workers reduce concurrent memory use. Synthetic experiments optionally use the PyTorch MPS backend; `--backend cpu` provides a CPU path. CPU and accelerator results are compared with explicit floating-point tolerances.

## Reproduce

Run all frozen validation settings, exact support variances and synthetic stress tests:

```sh
python src/run.py --mode full --workers 6 --backend auto
```

This executes five independent blocks of 2,000 acquisitions for each real-data setting. It verifies the resulting CSV values against `expected/`, and writes `results/reproduction_check.json`. The reference reproduction completed in 9.3 minutes with six workers and optional accelerator operations. Runtime depends on CPU libraries, concurrency and storage. Each specimen retains its own identifier and analysis record.

Run a small integrity check and two validation examples:

```sh
python src/run.py --mode smoke --workers 2
```

Rebuild from checksum-pinned original cell tables, then repeat the full experiment:

```sh
python src/run.py --mode from-source --workers 6 --backend auto
```

Original cell tables require approximately 0.9 GB of downloads. The pipeline checks SHA256 before use, applies the fixed gates and geometry, and rebuilds sufficient statistics. Independent support-inclusion Monte Carlo verification is also available:

```sh
python src/support_monte_carlo.py
```

## Contents

- `src/inference.py`: reusable acquisition planning and prevalence estimation.
- `src/neighborhood.py`: nearest distinct-neighbor geometry, including overlapping marks.
- `src/download.py`: verified downloads from the primary public releases.
- `src/prepare.py`: cell normalization, marker gates, core summaries and neighborhood totals.
- `src/evaluate.py`: randomized acquisition and estimator comparisons.
- `src/support_benchmark.py`: complete-neighborhood support construction from cell maps.
- `src/support_cached.py`: exact support variance from compact groups.
- `src/support_monte_carlo.py`: independent simulation check of product-inclusion variance.
- `src/simulate.py`: reproducible independent and clustered synthetic tissues.
- `src/summarize.py`: pooling of acquisition blocks into bias, RMSE, coverage and interval width.
- `src/test_inference.py`: exhaustive finite-population identities and geometry checks.
- `src/run.py`: coordinated execution and reference-output comparison.
- `data/processed/`: compact core counts and support groups.
- `metadata/`: input checksums, specimen metadata, fixed gates and protocol.
- `expected/`: numerical reference results for all reported settings.
- `DATA_NOTICE.txt`: source attribution and data terms.

## Use the method

`plan_acquisition` accepts a complete centroid census, core width, neighborhood radius and expected charged-area budget. It returns core membership, inclusion probabilities, selected cores and square acquisition bounds. A custom priority score can be specified for each occupied core.

`infer_prevalence` accepts complete neighborhood counts for the selected cores. Count anchor cells only in their assigned core and use the border cells solely to identify neighbors. The output includes the inverse-probability estimate, design standard error, effective contribution count, an approximate interval and a conservative bounded interval. Fixed external core predictions are optional.

The target is the supplied finite cell frame. Cell proximity means another centroid lies within the specified radius; it is not a direct measurement of contact or communication. The expected budget charges each acquisition square, including overlapping borders. Realized acquisition area is random. Approximate intervals may under-cover rare endpoints; bounded intervals provide the stated conditional guarantee under independent selection and fixed predictions.

## Numerical conventions

Seeds use the documented SHA256 mapping in `metadata/frozen_protocol.json`. Each per-setting summary combines five blocks by pooling first and second moments. Numeric CPU reference comparisons use absolute and relative tolerances of `1e-10`; accelerator stress-test comparisons allow `5e-5`. Every row in the primary benchmark uses 10,000 acquisitions; simulated tissues use 50,000.

Core NPZ fields include `N`, `counts`, `features`, `y`, `crop`, `guard`, `ij`, `radii`, and `endpoints`. The `guard` field records a geometric erosion diagnostic and is not an estimator in the primary comparison. Support NPZ files store grouped support indices, outcome totals and probabilities. Zero-outcome groups contribute zero and are omitted from these compact reference tables.

## Attribution

The software is MIT licensed. Cite the original CyCIF and STARmap PLUS studies and retain `DATA_NOTICE.txt` with redistributed benchmark tables. `CITATION.cff` supplies the software citation metadata.
