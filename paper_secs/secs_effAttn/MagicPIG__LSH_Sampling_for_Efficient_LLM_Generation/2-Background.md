# 2 Background

<span id="page-2-1"></span>In this section, we formulate the targeted attention estimation problem and related works.

#### 2.1 Problem formulation

In LLM decoding phase, self-attention part calculates a weighted average of previous values by

$$o = \operatorname{Softmax}(\frac{qK^T}{\sqrt{d}})V = wV \quad q \in \mathbb{R}^{1 \times d} \quad K, V \in \mathbb{R}^{n \times d} \quad w \in \mathbb{R}^{1 \times n}$$
 (1)

where d is the head dimension and n is the context size.  $K = [k_1, k_2, ..., k_n], V = [v_1, v_2, ..., v_n], k_i, v_i \in \mathbb{R}^{1 \times d}$  is KV cache. Normalized attention weight  $w = \operatorname{Softmax}(\frac{qK^T}{\sqrt{d}}) \in \mathbb{R}^{1 \times n}$  is also called attention (score) distribution. Our target is to find sampling matrix  $\Pi \in \mathbb{R}^{n \times m}$  and diagonal matrix  $D \in \mathbb{R}^{m \times m}$  which minimize

$$\delta = ||wV - w\Pi D\Pi^T V|| \tag{2}$$

where  $m \ll n$  is computation budget. For TopK attention, suppose  $w_{r_1} > ... > w_{r_m} > ... > w_{r_n}$ , then

$$\Pi_{i,j} = \begin{cases} 1, & \text{if } i = r_j, \\ 0, & \text{otherwise.} \end{cases} D_{ii} = \frac{1}{\sum_{i=1}^m w_{r_i}}$$
(3)

### 2.2 Related works

Efficient Attention. Attention approximation has been long studied. Reformer [\(Kitaev et al.,](#page-14-3) [2020\)](#page-14-3), KDEformer [\(Zandieh et al.,](#page-15-7) [2023\)](#page-15-7) and ScatterBrain [\(Chen et al.,](#page-12-5) [2021\)](#page-12-5) tackle the problem via locality sensitive hashing. These methods work in training and encoder models like BigGAN [\(Brock et al.,](#page-12-7) [2019\)](#page-12-7). Theoretically, the error bounds and minimal workload required are continuously improved [\(Brand et al.,](#page-12-8) [2023;](#page-12-8) [Alman and](#page-12-9) [Song,](#page-12-9) [2023\)](#page-12-9) but have not proven to be practical for wall-clock acceleration in LLM decoding. Besides, flashattention [\(Dao et al.,](#page-13-5) [2022b;](#page-13-5) [Dao,](#page-13-6) [2023;](#page-13-6) [Dao et al.,](#page-13-7) [2022a\)](#page-13-7), flash-decoding [\(Ye et al.,](#page-15-8) [2024;](#page-15-8) [Hong et al.,](#page-13-8) [2024\)](#page-13-8) and SlimAttention [\(He et al.,](#page-13-9) [2024\)](#page-13-9) losslessly accelerate scaled product attention operator by maximizing the utilization of hardware, which is orthogonal to our approach.

Locality sensitive hashing. Locality sensitive hashing (LSH) [\(Backurs et al.,](#page-12-2) [2019,](#page-12-2) [2018\)](#page-12-4) is a family of hashing functions which assigns the same hash codes for similar inputs with higher probability than others [\(Chen](#page-12-10) [et al.,](#page-12-10) [2020b;](#page-12-10) [Jafari et al.,](#page-13-10) [2021\)](#page-13-10). LSH uses two hyper-parameters, (K, L). L hash tables are independently built. Each hash table has its own function H which projects a high-dimension vector to an integer by concatenating K random independent hash functions. In the sampling process, all vectors that share hash codes in at least one hash table with a query will be collected. SimHash [\(Charikar,](#page-12-11) [2002\)](#page-12-11) is the LSH family based on cosine similarity. For a vector x ∈ R d , SimHash generates a random hyperplane w and returns Sign(w <sup>T</sup> x). Vectors share the same sign if and only if the random projection is not in between them. For a random projection, all angles are equally likely, thus the probability that two vectors x, y share the same sign is p = 1 − θ π , where θ = arccos xy<sup>T</sup> ||x||·||y|| . If we have L hash tables each with K random hash functions, the probability of y to be retrieved by query x is 1 − (1 − p <sup>K</sup>) L.

KV Cache reduction. To get rid of memory bound introduced by KV cache thus enabling a larger batch size or serving a longer prompt, many methods are proposed to reduce the volume of KV cache. For example, H2O [\(Zhang et al.,](#page-16-0) [2023b\)](#page-16-0), SnapKV [\(Li et al.,](#page-14-4) [2024\)](#page-14-4) and Keyformer [\(Adnan et al.,](#page-12-12) [2024\)](#page-12-12) calculate heuristics during the prefilling phase to decide which tokens to preserve for decoding phase. Quest [\(Tang et al.,](#page-15-2) [2024\)](#page-15-2) and Loki [\(Singhania et al.,](#page-15-3) [2024\)](#page-15-3) do not evict KV cache but apply dynamic sparsity to reduce KV Cache loading at inference time. Besides the reduction along the dimension of sequence length, methods like KIVI [\(Liu et al.,](#page-14-5) [2024b\)](#page-14-5) and QServe [\(Lin et al.,](#page-14-6) [2024\)](#page-14-6) reduce the size of KV Cache by quantization.

