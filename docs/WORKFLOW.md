# Methodology and computational workflow

This guide follows the methodology of *PWR Loading Pattern Optimization Using a Convolutional Neural Network Surrogate with Conformal Safety Calibration and Annealing-Initialized Genetic Search*. Section and equation numbers refer to that manuscript.

## Core geometry and loading constraints · Section 2.1

The optimization space consists of 193 fuel positions embedded in a 15 × 15 grid. The fixed inventory contains 65 fresh, 64 once-burnt and 64 twice-burnt assemblies. Placement restrictions exclude twice-burnt fuel from peripheral positions, fresh fuel from inner positions, internal fresh–fresh adjacency and neighboring twice-burnt assemblies. Rotations and reflections are used to identify D4-equivalent configurations when forming datasets and GA populations.

`code/core/geometry.py` defines the grid, fuel counts, feasibility rules, transformations and loading-pattern encodings. `code/core/pattern_diversity.py` computes structural features and diversity representations; `code/core/data_io.py` provides consistent data persistence and hashes. These modules are shared by candidate generation, dataset selection and genetic search.

## QUBO formulation and annealing · Sections 2.2–2.3

The QUBO has terms for one fuel-type assignment per position, inventory counts, radial zoning, disallowed adjacency and (in symmetry-enabled cases) D4 consistency (Eqs. 4–9). SA and classical Suzuki–Trotter SQA sample this energy function. The warm-start experiments use constrained solutions after independent decoding and feasibility checks. SQA is the path-integral Monte Carlo sampler defined by Eq. 10, rather than a quantum-hardware experiment.

`code/annealing/qubo_and_reactor_model.py` contains the original sampling and reactor-model machinery; `code/annealing/sample_loading_patterns.py` generates sampling shards. `code/data/build_diverse_candidate_pool.py` collects feasible, distinct candidates for subsequent training and optimization.

## OpenMC neutronics · Section 2.4

The paper's reference model is a static 2D planar representation of a Westinghouse 17 × 17 assembly lattice based on BEAVRS, with reflective axial boundaries and a radial water reflector. Reference calculations use OpenMC to obtain $k_{eff}$, assembly peaking $F_a$ and two-dimensional radial pin peaking $F_{q,r}$ (Eq. 12). The paper specifies 500 batches (50 inactive), 500,000 particles per batch, and a 255 × 255 × 1 fission-rate mesh.

`code/physics/simulate_patterns.py` evaluates pattern batches. `code/physics/verify_optimized_patterns.py` performs direct neutronics checks of GA-selected patterns, and `code/physics/merge_simulation_results.py` consolidates active-learning wave results. The original reference-batch validator and pilot diagnostics are in the same directory. OpenMC, its nuclear-data library and the associated runtime environments are configured separately from TensorFlow.

## CNN ensemble and active learning · Sections 2.5–2.6

The initial broad dataset contains 10,000 OpenMC-evaluated patterns; the manuscript follows the progression through optimizer-tail refinements to 21,100 evaluated cases (Table 1). A five-member multi-output CNN ensemble predicts $k_{eff}$, $F_a$ and $F_{q,r}$. Its architecture uses three residual convolutional blocks, attention and three output heads. Predictions from the ensemble guide acquisition toward low-objective, boundary-sensitive, high-residual and structurally diverse designs.

`code/surrogate/cnn_architecture.py` defines the network. Training and evaluation are implemented by `train_cnn.py`, `train_balanced_cnn.py`, `train_balanced_cnn_streaming.py`, and the ensemble evaluation modules. `code/data/` prepares labeled sets, scores candidate pools and selects new designs for OpenMC. The final refinement uses `code/surrogate/train_final_refined_cnn.py` and the five-fold training routines `train_final_out_of_fold_cnn.py`, `prepare_final_crossvalidation.py`, and `assemble_final_crossvalidation.py`. This is an iterative physics–learning loop, not a one-pass surrogate trained only on broad random configurations.

## Residual uncertainty and conformal screening · Sections 2.7–2.8

Five-fold out-of-fold predictions provide the residual targets used to train an ExtraTrees error-scale model. The paper describes 202 inputs: 193 pattern entries, three ensemble means, three ensemble standard deviations, the predicted objective and two predicted peaking margins. Local residual scales $r_k$, $r_a$, and $r_q$ normalize the one-sided conformity scores (Eqs. 13–14). Four quantiles provide lower and upper reactivity bounds and upper peaking bounds (Eqs. 15–17), using the Bonferroni allocation of the family-wise error rate.

The code for splitting, merging and assessing out-of-fold predictions is in `code/uncertainty/`. `train_final_error_scales.py` fits the final optimizer-domain residual scale model from the cross-fitted predictions. In particular, `build_out_of_fold_splits.py`, `assemble_out_of_fold_predictions.py` and `calibrate_residual_safety.py` implement the route from held-out predictions to the safety model. The corresponding diagnostics examine how residual scales relate to actual OpenMC prediction error.

## Genetic search and memetic refinement · Sections 2.9–2.10

Each GA individual represents 193 assembly assignments satisfying the hard loading constraints. The paper compares penalty-based and feasibility-first global ranking, each with and without local refinement. Memetic search uses best-improvement single-swap descent at generations 75, 150, 225 and 300, with bounded two-swap escape from an unsafe region. All four architectures use population size 100, 40 selected parents and 300 generations (Tables 3–4).

The four primary implementations are `penalty_based_search_300.py`, `penalty_based_memetic_search.py`, `feasibility_first_search.py`, and `feasibility_first_memetic_search.py`; `local_refinement.py` contains the shared memetic procedures. Random, SA and SQA initializations are compared in the paired search routines, with OpenMC evaluation of selected patterns. The quality measure for designs satisfying the safety screen is $J = \tfrac12(F_a/2.0) + \tfrac12(F_{q,r}/2.5)$ (Eq. 18).

The final optimizer-domain calibration draws 960 winners over 12 search arms; `run_balanced_calibration_search.py` and the matching SLURM jobs generate the search cases. The subsequent 360-candidate prospective study is represented by `run_prospective_optimization.py` and the scripts in `code/evaluation/`: prospective cohort preparation, warm-start-bank construction, winner collection, OpenMC evaluation, and paired-search analysis. The manuscript evaluates the main effects of initialization, ranking strategy and local refinement using a replicate-blocked model (Eq. 19). See [Results](../results/README.md) for the reported comparisons and [Code index](CODE_INDEX.md) for the individual programs.
