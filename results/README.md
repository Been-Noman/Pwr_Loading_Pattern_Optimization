# Results

The numerical experiments in *PWR Loading Pattern Optimization Using a Convolutional Neural Network Surrogate with Conformal Safety Calibration and Annealing-Initialized Genetic Search* examine three connected questions: how well the CNN transfers to configurations found by the optimizer, whether the residual-calibrated bounds cover subsequent OpenMC calculations, and which elements of the hybrid genetic search influence the physical objective.

## Surrogate accuracy after active learning

Active learning improved prediction in the optimizer's region of the loading-pattern space. On the common independent balanced set of 600 patterns, the final CNN ensemble achieved an MAE of **0.000645** for \(k_{eff}\), **0.103476** for assembly peaking \(F_a\), and **0.114235** for radial pin peaking \(F_{q,r}\). The corresponding RMSE values were 0.000901, 0.131427 and 0.146435 (Table 7). Relative to the preceding cohort, the final refinement reduced MAE by 17.8%, 22.1% and 28.9%, respectively. The predicted optimization objective also improved by 25.9% in MAE (Fig. 6).

The earlier 2,500-case optimizer-tail evaluation revealed that the first surrogate could underestimate power peaking (Table 11). The additional acquisition rounds and out-of-fold analysis therefore addressed a distribution encountered during search, rather than treating accuracy on broad initial patterns as sufficient.

## Residual uncertainty and prospective safety coverage

The ExtraTrees model estimates the magnitude of surrogate residuals from held-out predictions, supplying the local scales used in the conformal bounds. The final calibration used 960 optimizer-generated patterns, equally distributed over the 12 search arms. With the quantile multipliers fixed, 360 new optimizer-selected patterns (30 per arm) were evaluated with OpenMC. The joint coverage of the four one-sided bounds was **347/360 (96.39%)**, compared with 923/960 (96.15%) in the calibration cohort (Table 14; Fig. 9). All 360 prospective winners satisfied the physical screening constraints.

## Genetic and memetic search

The 12-arm comparison combines four search architectures with random, SA and SQA initialization. In the independent 360-run OpenMC evaluation, **feasibility-first memetic search initialized with SA** had the lowest mean physical objective, \(J=0.722327\), and produced the best configuration in 11 of the 30 paired replicates (Table 15). Several memetic arms performed similarly under Holm-adjusted pairwise comparisons, and penalty-based memetic search with SA initialization was faster on average.

The larger **600-run replicate-blocked analysis** attributes the strongest improvement to local memetic refinement: its estimated change in OpenMC objective was \(\Delta J=-0.047479\), with 95% confidence interval [−0.050352, −0.044606] and Holm-adjusted \(p=2.56\times10^{-128}\) (Table 16; Figs. 10–11). In comparison, changing between feasibility-first and penalty-based global ranking produced a much smaller estimate, \(\Delta J=+0.000245\), with \(p=0.866819\). The study therefore identifies local refinement as the dominant contribution to final physical objective quality, while the two global ranking rules and SA/SQA warm starts have smaller effects.

These results are those reported in the manuscript's Tables 7–16 and associated figures. The implementation modules are organized in [`code/`](../code/), including the final prospective evaluation and paired-search analysis in [`code/evaluation/`](../code/evaluation/). The paper-to-code relationships are detailed in the [methodology guide](../docs/WORKFLOW.md).
