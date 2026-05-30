# 3 **ZSMerge** Methodology

Building upon the conceptual foundation of ZSMerge outlined earlier, this section delineates the technical framework underpinning its zero-shot compression mechanism. We propose a fourcomponent methodology consisting of adaptive budget allocation, context-sensitive contribution evaluation, residual token merging, and stabilized attention projection. The interplay of these components ensures robust generalization across compression ratios and task domains without auxiliary parameters or fine-tuning, addressing limitations of prior eviction and merging strategies.

## 3.1 Preliminaries

Consider an L-layer transformer with multi-head attention mechanisms. For a target attention head at decoding step T, the cached Key and Value matrices are defined as:

$$\mathbf{K}_T = [\mathbf{k}_1, \mathbf{k}_2, \dots, \mathbf{k}_T]^\top \in \mathbb{R}^{T \times d}, \ \mathbf{V}_T = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_T]^\top \in \mathbb{R}^{T \times d},$$
(1)

where kt, v<sup>t</sup> ∈ R d represent the Key/Value vectors for the t-th token. For the query vector q<sup>T</sup> ∈ R d at position T, the scaled dot-product attention computes output o (T) via:

<span id="page-3-0"></span>
$$a_t^{(T)} = \frac{\exp(\mathbf{q}_T^{\top} \mathbf{k}_t / \sqrt{d})}{\sum_{i=1}^T \exp(\mathbf{q}_T^{\top} \mathbf{k}_i / \sqrt{d})}, \ \mathbf{o}^{(T)} = \sum_{t=1}^T a_t^{(T)} \mathbf{v}_t.$$
 (2)

This formulation establishes the baseline for analyzing cache compression effects on attention distribution fidelity.

