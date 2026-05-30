# 2 Background

Sparse Mixture of Experts architectures [\[Shazeer et al.,](#page-20-3) [2017;](#page-20-3) [Fedus et al.,](#page-17-3) [2022\]](#page-17-3) replace the feed-forward network (FFN) with an MoE layer in the Transformer block [\[Vaswani et al.,](#page-21-5) [2017\]](#page-21-5). An MoE layer consists of a router network R and a set of n experts, E1, ..., En, where each expert E<sup>i</sup> corresponds to an independent dense feed-forward network. The router network R is commonly parameterized by trainable weights W<sup>r</sup> ∈ R <sup>h</sup>×<sup>n</sup> where h is the model hidden dimension, and followed by a softmax function which takes an intermediate token representation x as input and combines the output of each expert based on the gating scores s1, ..., sn. Sparse MoEs only use the top-k experts E<sup>k</sup> based on experts gating scores s<sup>i</sup> .

$$s_i = R(x) = \operatorname{softmax}(W_r^T x)$$
 (Router) 
$$s_k = \operatorname{TopK}(s_i)$$
 (Top-K Routing) 
$$y = \sum_{i=1}^k s_k \cdot E_k(x)$$
 (MoE)

Recent work has also shown that using a shared expert E<sup>0</sup> that is always activated is beneficial to remove parameter redundancy among other experts [\[Rajbhandari et al.,](#page-20-4) [2022;](#page-20-4) [Dai et al.,](#page-16-3) [2024\]](#page-16-3):

$$y = E_0(x) + \sum_{i=1}^{k} s_k \cdot E_k(x)$$
 (MoE + shared expert)

Sparse Upcycling [\[Komatsuzaki et al.,](#page-18-3) [2023\]](#page-18-3) initializes an MoE model from a dense Transformer model. The dense model's FFN layers are copied n times to initialize each of the n experts, and the router layer is trained from scratch. BTX [\[Sukhbaatar et al.,](#page-21-2) [2024\]](#page-21-2) generalize this approach to initialize each expert from the FFN layer of a different expert model, and all other parameters as the average over all of these models. The experts models are finetuned versions of the original dense model, which allows weight merging without major losses.

Nexus leverages upcycling specialized expert models similar to BTX, however, it diverges in terms of MoE training, in particular with its novel MoE router, which enables to efficiently extend the MoE in multiple rounds after the sparse upcycling. We describe our method in the next section.

