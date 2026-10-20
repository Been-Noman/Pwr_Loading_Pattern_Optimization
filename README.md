# PWR loading pattern optimization

**Research software accompanying** *PWR Loading Pattern Optimization Using a Convolutional Neural Network Surrogate with Conformal Safety Calibration and Annealing-Initialized Genetic Search*, by Noman Bin Morshed and Md. Ali Mahdi.

The study addresses a static, two-dimensional fuel loading problem for a 193-assembly pressurized water reactor. Candidate configurations satisfy fixed fuel inventory, radial zoning and assembly-neighborhood constraints. Simulated annealing (SA) and simulated quantum annealing (SQA) sample the constrained QUBO formulation and furnish initial configurations for genetic search. OpenMC transport calculations provide reference values of the effective multiplication factor, assembly radial peaking and radial pin-power peaking. A multi-output convolutional neural network (CNN) learns these quantities from the evaluated configurations, and active-learning rounds progressively enrich the training set with patterns encountered near the optimizer's operating region.

The surrogate is supplemented by a residual uncertainty model trained on out-of-fold predictions. Its local error scales enter one-sided conformal bounds, which are used to screen loading patterns during genetic optimization. The resulting study compares penalty-based and feasibility-first search, each with and without memetic local refinement, under random, SA and SQA initialization. Final candidates are evaluated against OpenMC rather than ranked solely by surrogate predictions.

## Repository structure

```text
code/
  core/           Core layout, loading constraints, I/O and pattern diversity
  annealing/      QUBO/SA/SQA formulation and feasible candidate sampling
  physics/        OpenMC evaluation, physical verification and result merging
  data/           Labeled datasets, candidate banks and active learning
  surrogate/      CNN architecture, ensemble training and evaluation
  uncertainty/    Out-of-fold residual estimation and conformal calibration
  optimization/   Genetic and memetic search, initialization comparisons
  evaluation/     Final calibrated 12-arm studies and OpenMC comparisons
hpc/
  environment/    CNN and OpenMC environment checks
  jobs/           SLURM jobs grouped by experiment and scientific role
  launch/         Batch submission workflows for training and acquisition
docs/            Methodology-to-code map and computational setup
results/         Summary of the findings reported in the manuscript
tests/           Geometry, file I/O and source-level regression checks
checksums/       Correspondence between the supplied source and this layout
provenance/      Untouched checksum-locked solver source snapshot
```

## Following the methodology

The code is arranged by scientific role, not by the historical order in which individual programs were written. The progression below corresponds to Sections 2.1–2.10 and Fig. 1 of the manuscript:

1. **Core and constraints** — the 15 × 15 representation, 193 fuel positions, fuel inventories, radial zones and adjacency rules.
2. **QUBO sampling and OpenMC labels** — SA/SQA candidate construction, D4-equivalent filtering and high-fidelity neutronics evaluation.
3. **Iterative surrogate development** — CNN ensemble training, candidate acquisition and new OpenMC labels from optimizer-relevant configurations.
4. **Residual uncertainty and safety calibration** — out-of-fold error scales and one-sided conformal bounds for the screened physics quantities.
5. **Genetic optimization and independent evaluation** — penalty-based and feasibility-first ranking, local memetic refinement, three initialization methods, and OpenMC evaluation of selected solutions.

[Methodology and code](docs/WORKFLOW.md) connects these steps to their implementations and equations. [Code index](docs/CODE_INDEX.md) identifies each Python file by purpose. [Computational setup](docs/COMPUTATIONAL_SETUP.md) describes environments, data paths, and SLURM orchestration. The [results overview](results/README.md) summarizes the findings reported in the paper.

## Source and citation

The scientific implementations originate from the supplied research backup. Python entrypoints have been given descriptive names and grouped by role; their computational methods have not been redesigned. Imports, file references and job launch paths have been adjusted to follow the new layout. The original paths and SHA-256 hashes are recorded in [`checksums/original_source_map.csv`](checksums/original_source_map.csv). The four original checksum-locked GA implementations are separately preserved in [`provenance/hash_locked_search_sources.zip`](provenance/hash_locked_search_sources.zip), alongside the clearly named implementations in `code/optimization/`.

The numerical results below are those reported in the paper and its tables. The two-dimensional radial pin peaking factor used here, \(F_{q,r}\), is distinct from the three-dimensional licensing quantity \(F_q(z)\).
