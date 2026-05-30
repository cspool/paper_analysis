# A Detailed Methodology

This section provides the detailed mathematical formulation for the PreMoE framework, as referenced in Section 3.

## <span id="page-11-0"></span>A.1 Detailed Formulation of Predicted Expert Utility (PEU)

As described in the main text, the calculation of the Predicted Expert Utility (PEU) is a two-stage process designed to refine the raw router logit signal into a robust measure of an expert's utility.

Stage 1: Filtering for High-Confidence Activations. The first stage filters out low-confidence or irrelevant expert considerations on a per-token basis. This process is governed by two hyperparameters:  $K_a$ , the size of an initial candidate pool, and r, a confidence threshold. For each token  $\mathbf{x}$ , we first identify the set of  $K_a$  experts with the top raw router logits  $s_i(\mathbf{x})$ , denoted as  $\mathcal{E}_{K_a}(\mathbf{x})$ . Next, we compute locally normalized probabilities  $p_i(\mathbf{x})$  for this candidate set by applying a softmax function only over their logits. A confidence threshold r is then applied to these local probabilities. An expert is only considered for the next stage if it is in the candidate pool  $\mathcal{E}_{K_a}$  and its local probability  $p_i(\mathbf{x})$  exceeds the threshold r.

The adaptive threshold  $r_l$  for each MoE layer l is calculated as the average probability of the top-ranked expert within the top- $K_a$  pool across the calibration dataset:

$$r_l = \mathbb{E}_{\mathbf{x} \in \mathcal{X}_T} \left[ \max_{i \in \mathcal{E}_{K_q}(\mathbf{x})} p_i^l(\mathbf{x}) \right]$$
 (6)

This layer-wise adaptive rule makes our method robust across different models and domains with minimal tuning.

**Stage 2: Applying the Logit Transformation.** For the high-confidence expert logits that pass the filtering stage, we then apply the logit transformation f(s). Our default choice is  $f(s) = \max(s, \operatorname{sigmoid}(s))$ , which retains large positive evidence while rectifying negatives via  $\operatorname{sigmoid}(s)$ , avoiding compression of strong signals.

The final token-level score,  $\tilde{s}_i(\mathbf{x})$ , combines these two stages. The full calculation, along with the final PEU averaging, is as follows:

$$\mathcal{E}_{K_a} = \text{TopK}(\{s_j(\mathbf{x})\}_{j=1}^{N_r}, K_a)$$
(7)

$$p_i(\mathbf{x}) = \frac{\exp(s_i(\mathbf{x}))}{\sum_{k \in \mathcal{E}_{K_a}} \exp(s_k(\mathbf{x}))}, \quad i \in \mathcal{E}_{K_a}$$
(8)

$$\tilde{s}_i(\mathbf{x}) = \begin{cases} f(s_i(\mathbf{x})), & i \in \mathcal{E}_{K_a} \land p_i(\mathbf{x}) \ge r, \\ 0, & \text{otherwise.} \end{cases}$$
(9)

$$PEU_i^T = \frac{1}{|\mathcal{X}_T|} \sum_{\mathbf{x} \in \mathcal{X}_T} \tilde{s}_i(\mathbf{x}).$$
 (10)

This final PEU score represents the average, high-confidence, transformed logit for expert *i* on the calibration dataset, and is used to rank and select experts for the final computational pattern.

