# 2 Background

#### 2.1 Sparse Mixture of Experts

Inspired by conditional computation (Srivastava et al., 2013; Bengio et al., 2013) that activates only some relevant weights of a model on a per-token basis, the Sparse Mixture of Experts (SMoE) model (Shazeer et al., 2017), as an example of conditional computation, with each layer consists N experts and a trainable router which selects the most appropriate k experts to process each input sample. In this paper, we apply SMoE for Transformer-based architectures(Chi et al., 2022; Dai et al., 2022; Do et al., 2023) by replacing the feed-forward neural network layer in Transformers(Vaswani et al., 2023) with the Mixture-of-Experts layer, drawing inspiration from (Du et al., 2022; Zhou et al., 2024; Jiang et al., 2024). Each Mixture-of-Experts layer consists of a set of multi-layer perceptrons (MLPs), each with two layers and a ReLu non-linearity function(Agarap, 2019). Denoting the output of the multi-head attentions (MHA) as x, the output of SMoE with N experts is a weighted sum of each expert's computation  $E_i(x)$  by the router function G(x):

$$f_{\text{SMoE}}(\boldsymbol{x}) = \sum_{i=1}^{N} G(\boldsymbol{x})_i \cdot E_i(\boldsymbol{x})$$
 (1)

Where G(x) is computed by  $TOP_k$  function as equation (2) that determines the contribution of each expert to the SMoE output.

<span id="page-1-0"></span>
$$G(\mathbf{x}) = \text{TOP}_k(\text{softmax}(\mathbf{W}\mathbf{x} + b))$$
 (2)

In this research, we primarily focus on top-2 routing (K=2), as studies(Zhou et al., 2022b; Zoph et al., 2022; Sukhbaatar et al., 2024; Pham et al., 2024) have demonstrated its superior balance between training efficiency and testing performance.

