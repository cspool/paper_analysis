# <span id="page-3-2"></span>4 Why Doesn't RLVR Improve Capability?

Prior work has shown across multiple models that RLVR yields substantial gains in accuracy but often fails to improve capability, as measured by pass@k with sufficiently large k (Yue et al., 2025). In this section, we extend this observation and analyze the phenomenon in greater depth. Specifically, we aim to answer: Why does RLVR raise accuracy while leaving capability unchanged or even degraded?

#### 4.1 Capability Analysis

We first performed pass@k experiments similar to those by Yue et al., confirming that RLVR improves accuracy but not capability (Fig. 7). Specifically, we evaluated two base models—Qwen2.5-1.5B-Math (Yang et al., 2024) and Qwen2.5-3B (Hui et al., 2024)—along with their corresponding RLVR-trained versions. The full training details and pass@k results are provided in Appendix A.9 and A.2, respectively. All models were trained on the MATH training set, which contains 7,500 questions, and evaluated on the MATH 500 test set with 500 problems. These same datasets are also used for all subsequent experiments in this section.

We report here results on the 1.5B model evaluated on the test set due to space constraints. However, the same pattern holds consistently across both the train and test sets, and across both model sizes (1.5B and 3B). Full results are provided in Appendix A.3. For clarity, we refer to the original 1.5B model as the base model and the RLVR-trained version as the RL model.

