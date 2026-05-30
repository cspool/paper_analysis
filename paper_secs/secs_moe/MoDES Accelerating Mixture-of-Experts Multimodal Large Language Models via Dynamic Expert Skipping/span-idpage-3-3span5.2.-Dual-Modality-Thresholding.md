# <span id="page-3-3"></span>5.2. Dual-Modality Thresholding

Building on *Insight (ii)* in Sec. 4.2, we introduce a dualmodality thresholding (DMT) method to adaptively determine modality-specific expert skipping thresholds for MLLMs. We define two thresholds:  $\tau_t$  for text tokens and  $\tau_v$ for visual tokens, which control the degree of expert skipping for each modality. This design considers the distinct behavior of tokens from different modalities, thereby allowing a tailored and effective skipping strategy.

To be specific, based on the importance scores (Eq. (3)) for the l-th layer, experts that should be skipped for the given token  $\mathbf{x}^{(l)}$  are:

<span id="page-3-5"></span>
$$\{ \text{Expert}_i^{(l)} \mid s_i^{(l)} < \tau_t \cdot \mathbb{I}_t + \tau_v \cdot \mathbb{I}_v \}, \tag{5}$$

where  $\mathbb{I}_t$  and  $\mathbb{I}_v$  are text and vision token indicator functions for  $\mathbf{x}^{(l)}$ , respectively.

To find the optimal  $\tau_t$  and  $\tau_v$  that balance computational efficiency with model performance, we propose a frontier search algorithm that effectively and efficiently determines these thresholds under an efficiency constraint. We first formulate the problem in the following.

Problem definition. For an MoE MLLM, the goal is to find the thresholds  $\tau_{\rm t}$  and  $\tau_{\rm v}$  that minimize the difference between the outputs of the original model and the expert-

## <span id="page-4-5"></span><span id="page-4-2"></span>Algorithm 1 Frontier search for optimal thresholds.

```
func FrontierSearch(\mathcal{B}, \rho)
Require:
     \mathcal{B} — Candidate set of thresholds \{\tau^{(1)}, \dots, \tau^{(D)}\}
     \rho — Target skipping ratio
 1: frontier \leftarrow \emptyset
 2: p \leftarrow D
 3: for q = 1 to D do
           while p \geq 1 and g(\tau^{(q)}, \tau^{(p)}) \geq \rho do
 4:
 5:
               p \leftarrow p - 1
           end while
          p_{(q)} \leftarrow p + 1
 7:
          if p_{(q)} \leq D then
 8:
               Compute and save f(\tau^{(q)}, \tau^{(p_{(q)})})
 9:
10:
                frontier \leftarrow frontier \cup \{(q, p_{(q)})\}
11:
12: end for
13: (q^*, p^*) \leftarrow \arg\min_{(q, p_{(q)}) \in \text{frontier}} f(\tau^{(q)}, \tau^{(p_{(q)})})
14: return (\tau^{(q^*)}, \tau^{(p^*)})
```

skipping one, while satisfying a pre-defined target skipping ratio  $\rho \in (0, 1)$ . Hence, the problem can be expressed as:

<span id="page-4-3"></span>
$$\min_{\tau_t \in \mathcal{B}, \tau_v \in \mathcal{B}} f(\tau_t, \tau_v) \quad \text{s.t.} \quad g(\tau_t, \tau_v) \ge \rho, \tag{6}$$

where  $\mathcal{B}=\{\tau^{(1)},\ldots,\tau^{(D)}\}$  is the search grid set with D candidates that satisfies  $\tau^{(1)}<\tau^{(2)}<\ldots<\tau^{(D)}$ .  $f(\tau_{\rm t},\tau_{\rm v})$  is the average KL divergence between the output distributions of the original model and the modified version, where experts are skipped according to Eq. (5).  $g(\tau_{\rm t},\tau_{\rm v})$  is the fraction of experts that are skipped for the modified model. **Frontier search.** We start with a monotonicity assumption:

**Assumption 1.** Holding other variables fixed, f is non-decreasing in its respective arguments: If  $q_1 \leq q_2$ , then  $f(\tau^{(q_1)}, \tau^{(p)}) \leq f(\tau^{(q_2)}, \tau^{(p)})$ ; and if  $p_1 \leq p_2$ , then  $f(\tau^{(q)}, \tau^{(p_1)}) \leq f(\tau^{(q)}, \tau^{(p_2)})$ .

Intuitively, higher thresholds will skip more experts and degrade accuracy; hence, the assumption is reasonable. Obviously, g is also non-decreasing in its respective arguments without any assumption. Given these monotonicity properties, we can search for a frontier set  $\{(q,p_{(q)})\}$  with a time complexity of  $\mathcal{O}(ND)^{-1}$  through Lines 1-12 in Alg. 1. Here,  $p_{(q)}$  for a given q is defined as:

$$p_{(q)} = \min \left\{ p \in \{1, \dots, D\} \mid g(\tau^{(p)}, \tau^{(q)}) \ge \rho \right\}.$$
 (7)

We provide detailed proofs for the correctness of the search algorithm and its time complexity in the Appendix. Finally, as demonstrated in Alg. 1, the optimal thresholds

 $(\tau^{(q^*)}, \tau^{(p^*)})$ , which lie in frontier (proofs can also be found in the Appendix), are obtained through Lines 13–14. Since all values of  $f(\tau^{(q)}, \tau^{(p_{(q)})})$  are already computed by Line 9, this step takes less than a second.

Overall, our *frontier search* algorithm achieves a time complexity of  $\mathcal{O}(ND)$ . In comparison, a naive solution involves an exhaustive search of all  $(\tau_t, \tau_v)$  pairs in  $\mathcal{B} \times \mathcal{B}$ , leading to a time complexity of  $\mathcal{O}(ND^2)$ . In practice, our method cuts the search time by a remarkable  $\sim$ 45× (as detailed in Sec. 6.3).

