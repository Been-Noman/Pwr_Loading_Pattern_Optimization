# Computational setup

The calculations use two distinct software environments. Python/TensorFlow performs CNN training, candidate scoring and genetic search. OpenMC handles the neutronics calculations and requires its own Python API, executable and nuclear cross-section library. The source archives were deployed with OpenMC `0.15.3` on an HPC environment and a separately configured CNN installation.

The CNN dependencies are listed in `requirements-cnn.txt`, with the historical frozen environment in `requirements-cnn.lock.txt`. `hpc/environment/setup_cnn_environment.sh` contains the original environment setup procedure, while `hpc/environment/check_openmc_environment.sh` contains the OpenMC preflight check.

## Configuration

The site-specific environment is configured through `config.sh`, created from the checked-in `config.example.sh`. `PWR_PROJECT` denotes the project root, `PWR_RUN` the scratch-output tree, and `PWR_SCRIPTS` the organized `code/` directory. The configuration also supplies `PWR_ML_VENV`, `PWR_OPENMC_VENV`, `OPENMC_CROSS_SECTIONS`, input dataset paths and a Python module search path for the shared source modules.

For example, from the project directory:

```bash
cp config.example.sh config.sh
# Edit HPC and data paths in config.sh for the research installation.
source config.sh
python -m pytest -q
```

Source-level tests cover loading-pattern logic, JSON persistence and syntax. The transport and model-training calculations are scheduled separately with the corresponding HPC environments.

## Batch workflows

All scheduler files reside under `hpc/jobs/`. The `active_learning/` group handles initial data preparation, candidate generation, model fitting, new-case selection and OpenMC acquisition waves. `initialization_comparison/` contains the paired initial-GA jobs; `search_and_calibration/` contains the optimizer-tail, cross-fitting, safety and four-architecture benchmark jobs; `reference_simulation/` contains the original OpenMC generation and pilot jobs. `final_evaluation/` contains the final balanced calibration, prospective verification and associated cross-fitting jobs.

The submission-chain scripts are under `hpc/launch/`:

- `prepare_candidate_pool.sh` — creates the starting labeled/candidate pool sequence.
- `run_active_learning_wave.sh` — schedules a CNN–selection–OpenMC–merge acquisition wave.
- `train_final_surrogate.sh` — launches the final training sequence in the source snapshot.

The job numbering inside each directory records the order of the original computational pipeline. Runtime outputs, trained ensemble checkpoints and Monte Carlo statepoints are written under the configured run tree. The original source-to-new-path mapping is maintained in `checksums/original_source_map.csv` for tracing batch scripts back to the research backup.

## Physical model and evaluation

The reference transport setting in Section 2.4 uses BEAVRS-derived materials and ENDF/B-VII.1 cross sections. The physical screening limits of the static study are $1.0 \leq k_{eff} \leq 1.2$, $F_a \leq 2.0$ and $F_{q,r} \leq 2.5$. Results refer to the planar transport model and its defined 2D radial power metrics.

## Frozen scientific source

The historical paired-run manifests use SHA-256 checks on the original four GA engines. Their exact original bytes are retained in `provenance/hash_locked_search_sources.zip`, while the code intended for navigation and reuse has descriptive filenames in `code/optimization/`. The source-path correspondence is recorded in `checksums/original_source_map.csv`. For a historical hash-locked run, unpack the source snapshot into the repository root before using the archived execution manifests:

```bash
unzip provenance/hash_locked_search_sources.zip -d .
```

This restores the original checksummed module paths alongside the organized paper-facing `code/` tree. Each frozen manifest also expects its original input data and model artifacts.
