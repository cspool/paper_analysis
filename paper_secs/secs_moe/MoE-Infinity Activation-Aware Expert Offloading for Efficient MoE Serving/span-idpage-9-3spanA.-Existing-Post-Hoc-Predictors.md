# <span id="page-9-3"></span>A. Existing Post-Hoc Predictors

Current post-hoc predictor (without changing model architectures and finetuning) does not fully capture the sparse activation properties of the MoE models.

- (1) Prediction based on dependency. Predictors estimate expert activation based on memory dependency [\(Huang](#page-8-8) [et al.,](#page-8-8) [2020;](#page-8-8) [HuggingFace,](#page-8-18) [2024;](#page-8-18) [Aminabadi et al.,](#page-8-2) [2022\)](#page-8-2). As experts in one MoE layer all have memory dependency on the same router, such approaches fail to capture selective (S) and grouped (G) properties of sparse activation. Reuse (R) is not considered under the same scope.
- (2) Prediction based on counts. Predictors use aggregated frequency counters on each expert to estimate activation [\(Cui et al.,](#page-8-5) [2023;](#page-8-5) [Jung et al.,](#page-8-10) [2023\)](#page-8-10). As experts tend to show uniform activation in the long run, this fails to capture the sparsity (S). In addition, individual counters cannot instruct the grouped activation (G) within and across layers.
- (3) Prediction based on locality. Predictors estimate expert reuse based on heuristics such as LFU and LRU [\(Eliseev &](#page-8-3) [Mazur,](#page-8-3) [2023;](#page-8-3) [Jung et al.,](#page-8-10) [2023;](#page-8-10) [Cui et al.,](#page-8-5) [2023;](#page-8-5) [Aminabadi](#page-8-2) [et al.,](#page-8-2) [2022\)](#page-8-2). Although only activated experts are considered (S,R), the reuse prediction is not applied across iterations, failing in the decoding phase.

