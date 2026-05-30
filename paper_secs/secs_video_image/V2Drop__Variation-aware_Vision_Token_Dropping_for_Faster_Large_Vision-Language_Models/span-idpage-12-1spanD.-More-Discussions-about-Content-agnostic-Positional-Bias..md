# <span id="page-12-1"></span>D. More Discussions about Content-agnostic Positional Bias.

Figures 2 and 3 reveal the inherent content-agnostic positional bias of LLM attention-guided methods such as SparseVLM and FastV. Figure 2 illustrates how these methods, despite assigning higher scores to critical regions, disproportionately favor later-positioned tokens regardless of content relevance, leading to the discarding of informative earlier tokens and triggering multimodal hallucinations. In contrast, measuring token-wise variation (*e.g.*, L2 Norm)

<span id="page-12-3"></span>**Algorithm 1** V<sup>2</sup>Drop: Variation-aware Vision Token Dropping

```
Require: Vision tokens \mathbf{F}^v \in \mathbb{R}^{M \times D'}, Dropping lay-
         ers \mathcal{L} = \{l_1, l_2, \dots, l_K\}, Compression Targets
         \{K_{l_1}, K_{l_2}, \dots, K_{l_K}\}
Ensure: Compressed vision tokens
   1: Current token count M_{\text{curr}} \leftarrow M
   2: for l = 1, 2, \dots, L do
                if l \in \mathcal{L} then
   3:
   4:
                        Step 1: Variation Computation
                       \begin{aligned} & \textbf{for } i = 1 \text{ to } M_{\text{curr}} \textbf{ do} \\ & s_i^{(l)} \leftarrow \| \mathbf{f}_i^{(l)} - \mathbf{f}_i^{(l-1)} \|_2 \\ & \textbf{end for} \\ & \mathbf{S}^{(l)} = \{s_1^{(l)}, s_2^{(l)}, \dots, s_{M_{\text{curr}}}^{(l)} \} \end{aligned}
   5:
   6:
   7:
                        Step 2: Token Ranking and Selection
   9:
                        indices \leftarrow \operatorname{argsort}(\mathbf{S}^{(l)}, \operatorname{descending})
 10:
                        \hat{\mathbf{F}}_{l}^{v} \leftarrow \{\mathbf{f}_{\text{indices}[j]}^{(l)} : j = 1, \dots, K_{l}\}
 11:
                        \mathbf{F}_{\mathrm{curr}}^v \leftarrow \hat{\mathbf{F}}_l^v, M_{\mathrm{curr}} \leftarrow K_l
 12:
 13:
                        \mathbf{F}_{\text{curr}}^v \leftarrow \text{TransformerLayer}(\mathbf{F}_{\text{curr}}^v)
 14:
 15:
 16: end for
 17: return \mathbf{F}_{\text{curr}}^{v}
```

intuitively reflects token importance and selectively retains semantically critical tokens. To quantify this bias, Figure 3 analyzes LLaVA-1.5-7B and Qwen2-VL-7B across three datasets (TextVQA, POPE, and MME), partitioning tokens into 10 equal intervals and calculating retention probabilities after pruning 50% of tokens at the third layer. Results demonstrate that attention-guided methods exhibit strong end-of-sequence bias, while variation-aware evaluation produces naturally uniform spatial distributions. Below, we provide a detailed theoretical analysis to establish the relationship between token variation and model output.

