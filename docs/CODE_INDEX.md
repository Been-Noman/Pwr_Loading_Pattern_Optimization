# Code index

The Python source files are grouped by their roles in the methodology. Each file below is the renamed original research source; the full correspondence and original SHA-256 hashes are in [`checksums/original_source_map.csv`](../checksums/original_source_map.csv).

## Core

| File | Role |
| --- | --- |
| `code/core/data_io.py` | Atomic JSON and pattern-data persistence |
| `code/core/geometry.py` | Geometry, fuel inventories, D4 transformations and hard feasibility |
| `code/core/pattern_diversity.py` | Structural descriptors and diverse candidate selection |

## Annealing

| File | Role |
| --- | --- |
| `code/annealing/qubo_and_reactor_model.py` | QUBO/SA/SQA routines and BEAVRS-derived OpenMC core construction |
| `code/annealing/sample_loading_patterns.py` | Feasible SA/SQA shard generation |

## Physics

| File | Role |
| --- | --- |
| `code/physics/estimate_runtime.py` | Estimate OpenMC execution cost |
| `code/physics/inspect_pilot_results.py` | Summarize pilot simulation outputs |
| `code/physics/merge_simulation_results.py` | Merge labeled acquisition-wave simulations |
| `code/physics/simulate_patterns.py` | Parallel OpenMC evaluation of candidate patterns |
| `code/physics/validate_reference_batch.py` | Validate and merge reference OpenMC batches |
| `code/physics/verify_optimized_patterns.py` | Independent OpenMC verification of GA-selected patterns |

## Data

| File | Role |
| --- | --- |
| `code/data/build_diverse_candidate_pool.py` | Build structurally balanced candidate pool |
| `code/data/check_evaluation_data.py` | Inspect evaluation labels and hard constraints |
| `code/data/prepare_initial_dataset.py` | Deduplicate and validate initial OpenMC labels |
| `code/data/prepare_second_refinement.py` | Assemble second optimizer-tail refinement dataset |
| `code/data/prepare_training_data.py` | Prepare model-ready data splits |
| `code/data/score_candidate_pool.py` | Use the CNN ensemble to score patterns |
| `code/data/score_optimizer_bank.py` | Score existing optimizer-domain candidates |
| `code/data/select_new_simulations.py` | Acquire additional patterns for OpenMC |
| `code/data/select_optimizer_tail.py` | Select diverse low-objective optimizer-tail patterns |
| `code/data/select_residual_learning_cases.py` | Choose optimization-tail residual learning samples |

## Surrogate

| File | Role |
| --- | --- |
| `code/surrogate/check_ml_environment.py` | CNN environment verification with a small synthetic tensor |
| `code/surrogate/cnn_architecture.py` | Residual CNN architecture and model configuration |
| `code/surrogate/evaluate_cnn_ensemble.py` | Ensemble prediction errors and evaluation |
| `code/surrogate/evaluate_first_refinement.py` | Assess first optimizer-tail CNN refinement |
| `code/surrogate/evaluate_second_refinement.py` | Assess second optimizer-tail CNN refinement |
| `code/surrogate/train_balanced_cnn.py` | Train CNN on balanced optimizer-tail data |
| `code/surrogate/train_balanced_cnn_streaming.py` | Balanced CNN training with tf.data |
| `code/surrogate/train_cnn.py` | Train a seeded CNN ensemble member |

## Uncertainty

| File | Role |
| --- | --- |
| `code/uncertainty/assemble_out_of_fold_predictions.py` | Combine fold-held-out predictions |
| `code/uncertainty/assess_earlier_residual_estimates.py` | Compare historical residual-scale performance |
| `code/uncertainty/assess_residual_estimates.py` | Diagnose error-scale accuracy in the current cohort |
| `code/uncertainty/build_out_of_fold_splits.py` | Prepare leakage-controlled out-of-fold datasets |
| `code/uncertainty/calibrate_residual_safety.py` | Fit error-scale model and evaluate conformal safety |
| `code/uncertainty/check_safety_on_candidate_bank.py` | Audit safety predictions over candidate bank |

## Optimization

| File | Role |
| --- | --- |
| `code/optimization/collect_optimization_tail.py` | Collect optimizer-visited patterns for active learning |
| `code/optimization/examine_local_descent.py` | Probe repeated single-swap improvement |
| `code/optimization/examine_swap_neighborhood.py` | Analyze local swap neighborhood |
| `code/optimization/examine_two_swap_escape.py` | Probe escape from locally trapped configurations |
| `code/optimization/feasibility_first_memetic_search.py` | Feasibility-first GA with local refinement |
| `code/optimization/feasibility_first_search.py` | Feasibility-first genetic search |
| `code/optimization/local_refinement.py` | Shared memetic single-swap descent and two-swap escape |
| `code/optimization/penalty_based_memetic_search.py` | Penalty-ranked GA with local refinement |
| `code/optimization/penalty_based_search.py` | Penalty-ranked genetic search |
| `code/optimization/penalty_based_search_300.py` | 300-generation penalty search used for matching comparisons |
| `code/optimization/run_initialization_comparison.py` | Early GA comparison across initializations |
| `code/optimization/run_paired_initialization_study.py` | Paired random/SA/SQA experiments |
| `code/optimization/run_random_initialization_variant.py` | Historical random-start GA variant |


## Final model and validation programs

The final refinement and independent 12-arm evaluation use additional original source recovered from the paper's final research archive. Their filenames are shortened here to describe the analysis performed.

| File | Role |
| --- | --- |
| `code/surrogate/train_final_refined_cnn.py` | Train the last CNN refinement on optimizer-domain samples |
| `code/surrogate/train_final_out_of_fold_cnn.py` | Produce cross-fitted CNN models for residual training |
| `code/uncertainty/prepare_final_crossvalidation.py` | Construct the final five-fold data partitions |
| `code/uncertainty/assemble_final_crossvalidation.py` | Combine final out-of-fold predictions |
| `code/uncertainty/train_final_error_scales.py` | Train the final ExtraTrees residual error-scale model |
| `code/optimization/run_balanced_calibration_search.py` | Search calibration winners across 12 arms |
| `code/optimization/run_prospective_optimization.py` | Execute independent 12-arm prospective searches |
| `code/optimization/run_paired_search_task.py` | Execute a paired seed/initializer manifest task |
| `code/optimization/run_frozen_search_adapter.py` | Adapt the original hash-pinned GA engines to paired seed inputs |
| `code/evaluation/prepare_prospective_cohort.py` | Construct the independent optimizer-selected cohort |
| `code/evaluation/plan_warmstart_repairs.py` | Identify required warm-start-bank repairs |
| `code/evaluation/prepare_repaired_warmstarts.py` | Assemble D4-checked repaired warm-start banks |
| `code/evaluation/define_prospective_search.py` | Freeze the prospective arm/replicate execution design |
| `code/evaluation/collect_prospective_winners.py` | Build prospective winning-pattern manifests |
| `code/evaluation/evaluate_prospective_openmc.py` | Merge OpenMC truth and evaluate prospective safety coverage |
| `code/evaluation/record_final_safety_calibration.py` | Record final calibrated safety multipliers and validation |
| `code/evaluation/analyze_search_comparison.py` | Paired 12-arm analysis and nonparametric comparisons |
| `code/evaluation/summarize_search_comparison.py` | Summarize the selected optimization method and statistics |
