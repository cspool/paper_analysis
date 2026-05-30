# <span id="page-27-0"></span>**C Estimating FLOPs**

To analyze the efficiency of our models, we quantify the computational cost in terms of total training Floating Point Operations (FLOPs). Following standard practice [\(Kaplan et al.,](#page-24-3) [2020\)](#page-24-3), we estimate the total training FLOPs as approximately three times the cost of a single forward pass (*C*train ≈ 3 · *C*fwd). The forward pass FLOPs are the sum of computations from the attention and feed-forward network (FFN) layers, plus a final logit projection.

For a model with hidden size *d*model, batch size *B*, and sequence length *s*, the cost of the attention block per layer, *C*attn, which includes Grouped-Query Attention (GQA) [\(Ainslie et al.,](#page-22-2) [2023\)](#page-22-2) and all projections, is approximately:

$$C_{\text{attn}} \approx Bsd_{\text{model}}^2 \left(2 + \frac{2}{n_h/n_{kv}}\right) + 4Bs^2 d_{\text{model}}$$
 (14)

where *n<sup>h</sup>* and *nkv* are the number of attention and key-value heads, respectively. The FFN cost varies by layer type. A dense layer with intermediate size *d*ffn requires *C*dense\_ffn = 6*Bsd*model*d*ffn FLOPs. A MoE layer activating *E<sup>a</sup>* experts, each with size *d*expert, requires:

$$C_{\text{moe\_ffn}} \approx 6Bsd_{\text{model}}(E_a \cdot d_{\text{expert}})$$
 (15)

If a shared expert of size *d*shared is used, its cost, 6*Bsd*model*d*shared, is added. For a model with *L* layers (of which the first *L*dense are dense) and a vocabulary of size *V*, the total forward FLOPs are:

$$C_{\text{fwd}} = \sum_{i=1}^{L} (C_{\text{attn}} + C_{\text{ffn},i}) + 2Bsd_{\text{model}}V$$
(16)

<span id="page-28-0"></span>where *C*ffn,*<sup>i</sup>* is the FFN cost for the *i*-th layer, which can be either *C*dense\_ffn or *C*moe\_ffn.

