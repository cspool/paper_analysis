# Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

Junhan Kim, Chungman Lee, Eulrang Cho, Kyungphil Park,
Ho-young Kim, Joonyoung Kim, Yongkweon Jeon

Samsung Research

{jun\_one.kim, chungman.lee, dragwon.jeon}@samsung.com

#### **Abstract**

With the increasing complexity of generative AI models, post-training quantization (PTQ) has emerged as a promising solution for deploying hyper-scale models on edge devices such as mobile and TVs. Existing PTQ schemes, however, consume considerable time and resources, which could be a bottleneck in real situations where frequent model updates and multiple hyperparameter tunings are required. As a cost-effective alternative, learning-free PTQ schemes have been proposed. However, the performance is somewhat limited because they cannot consider the inter-layer dependency within the attention module, which is a significant feature of Transformers. In this paper, we thus propose a novel PTQ algorithm that balances accuracy and efficiency. The key idea of the proposed algorithm called *aespa* is to perform quantization layer-wise for efficiency while targeting attention-wise reconstruction to consider the cross-layer dependency. Through extensive experiments on various language models and complexity analysis, we demonstrate that *aespa* is accurate and efficient in quantizing Transformer models. The code will be available at https://github.com/SamsungLabs/aespa.

#### 1 Introduction

Model size has been gradually growing, resulting in deep generative models such as diffusion [25] and large-scale language models (LLMs) [29, 35] becoming more mainstream; the trend of AI is transitioning from discriminative models to generative models with numerous parameters in trillions. With the explosive growth in model complexity (parameters), the performance of AI models has been advancing and is now approaching or even exceeding human intelligence levels. However, this growth in scale has resulted in a corresponding increase in computational costs, which necessitates the efficient processing and compression of AI models. Interestingly, one attempts to expand the complexity of AI models to scale up performance, whereas the other aims to compress models to reduce cost.

Quantization is a promising solution and indispensable procedure facilitating the efficient deployment of AI models on devices that mainly support fixed-point arithmetic. By reducing the precision of weights, the memory bandwidth requirements can be relieved, and the embarrassing parallelism of quantized models can be SIMDified using highly efficient vector processing units such as NPU. To minimize the inevitable performance degradation caused by quantization, we can choose one of two approaches: quantization-aware training (QAT) [5, 14] and post-training quantization (PTQ) [23, 18]. Considering the model complexity and required resources (*e.g.*, training costs and available datasets), QAT is not practical for compressing models with billions of parameters. Consequently, recent quantization studies on hyper-scale Transformer [31] models have focused more on PTQ.

<sup>\*</sup>Equal Contribution, †Corresponding Author

Although existing PTQ schemes have successfully quantized relatively small-scale models (*e.g.,* ResNet) [\[23,](#page-11-2) [10,](#page-10-2) [18,](#page-11-3) [6,](#page-10-3) [11\]](#page-10-4), they have difficulty handling large-scale models because of their time and space complexity. As a cost-effective alternative, learning-free algorithms have been proposed recently [\[7,](#page-10-5) [13,](#page-10-6) [19\]](#page-11-5), but their performance is somewhat limited because they do not consider the inter-layer dependency and are reliant on the nearest rounding. There is an accuracy-efficiency trade-off; thus, we aim to bridge the gap toward next-level quantization of hyper-scale Transformer models.

In this paper, we propose a novel PTQ algorithm, called *aespa*, [2](#page-1-0) that pursues both accuracy and efficiency. The key idea of *aespa* is to perform quantization layer-wise for efficiency while targeting the attention-wise reconstruction to consider the cross-layer dependency.

Our contributions are summarized as follows:

- We propose a new quantization strategy that balances accuracy and efficiency. Our scheme aims to reconstruct the attention output to consider the cross-layer dependency while quantizing models layer-wise to pursue efficiency.
- To accelerate the quantization process, we propose refined quantization objectives for the attention module. Through a complexity analysis, we demonstrate that quantization that is approximately 10 times faster than existing block-wise approaches can be achieved by exploiting the proposed objectives.
- From extensive experiments on language models, we demonstrate that our approach outperforms conventional schemes by a significant margin, particularly for low-bit precision (INT2).

## 2 Background

#### 2.1 Classic PTQ methods

Recent studies on PTQ have mostly attempted to minimize the increase in the task loss incurred by quantization rather than the quantization error itself (∆W). Consider a pre-trained neural network parameterized by weights W. If we assume the well-convergence of the network, the problem of quantizing weights W to minimize the loss degradation can be formulated as [\[16,](#page-11-6) [23\]](#page-11-2)

<span id="page-1-2"></span><span id="page-1-1"></span>
$$\min_{\Delta \boldsymbol{w}} \mathbb{E} \left[ \Delta \boldsymbol{w}^T \cdot \mathbf{H}^{(\boldsymbol{w})} \cdot \Delta \boldsymbol{w} \right], \tag{1}$$

where H(w) is the Hessian related to the flattened weight w. Because computing and storing H(w) is infeasible, further assumptions have been made to simplify [\(1\)](#page-1-1). In [\[23\]](#page-11-2), for example, layer-wise independence has been assumed, relaxing [\(1\)](#page-1-1) into the layer-wise reconstruction problem:

<span id="page-1-4"></span>
$$\min_{\Delta \boldsymbol{W}^{(\ell)}} \mathbb{E} \left[ \left\| \mathcal{Q}(\boldsymbol{W}^{(\ell)}) \boldsymbol{X} - \boldsymbol{W}^{(\ell)} \boldsymbol{X} \right\|_F^2 \right], \tag{2}$$

where W(ℓ) denotes the weights of the ℓ-th layer, X is the input, and Q is a quantization function. For a uniform quantization, if the nearest-rounding is used to assign integer weights, Q is defined as

<span id="page-1-3"></span>
$$Q(x) = s\left(\text{clamp}\left(\left|\frac{x}{c}\right| + z, 0, 2^n - 1\right) - z\right),\tag{3}$$

where s, z, and n are the scale, zero-point, and bit-width, respectively, and ⌊·⌉ represents the round-off.

Early studies on PTQ focused on optimizing the weight-rounding policy [\[23,](#page-11-2) [10,](#page-10-2) [18,](#page-11-3) [11,](#page-10-4) [12\]](#page-10-7). These studies have attempted to assign each weight to a "proper" grid (instead of an adjacent grid), such that the loss degradation could be minimized. In [\[23\]](#page-11-2), a learning-based weight-rounding optimization algorithm, called AdaRound, has been proposed to solve the layer-wise reconstruction problem in [\(2\)](#page-1-2). In [\[18\]](#page-11-3), AdaRound has been extended to the following block-wise reconstruction problem:

$$\min_{\Delta \boldsymbol{W}^{(\ell)}} \mathbb{E}\left[ \left\| f^{(\ell)} \left( \mathcal{Q}(\boldsymbol{W}^{(\ell)}), \boldsymbol{X} \right) - f^{(\ell)} \left( \boldsymbol{W}^{(\ell)}, \boldsymbol{X} \right) \right\|_{F}^{2} \right], \tag{4}$$

where W(ℓ) denotes the weights of the ℓ-th block f (ℓ) (*e.g.,* ResNet or Transformer block). By considering the dependency between layers inside the block, this algorithm, termed BRECQ, not only performs better than AdaRound, but also exhibits robust performance for a low bit-width (*e.g.,* INT2).

<span id="page-1-0"></span>*aespa*: attention-centric efficient and scalable post-training quantization algorithm

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 1: Overview of *aespa*. Each weight is quantized separately to reconstruct the attention output.

#### 2.2 PTQ for LLMs

Although AdaRound and BRECQ have been successful in quantizing small-scale networks (*e.g.,* ResNet), scaling those learning-based schemes to LLMs with billions of parameters is challenging. In fact, BRECQ requires more than 20 GPU hours to quantize relatively small-sized language models (*e.g.,* OPT-2.7B; see Appendix [K\)](#page-25-0), which would not be suitable for the real-world deployment of LLMs where models to be deployed are frequently updated.

Owing to the excessive time and memory costs of classic PTQ schemes, recent studies have focused on developing cost-effective alternatives for quantizing LLMs. In OPTQ [\[7\]](#page-10-5), a one-shot PTQ scheme that optimizes a weight-rounding policy without relying on learning, has been proposed. In addition, PTQ schemes that enhance the performance of the nearest-rounding, rather than optimizing the weight-rounding policy, have been proposed. These schemes use additional "foldable" parameters[3](#page-2-0) to suppress activation outliers or quantize weights more precisely [\[33,](#page-12-1) [19,](#page-11-5) [13,](#page-10-6) [27,](#page-11-7) [20\]](#page-11-8).

Although previous studies have mitigated the computational overhead of classic PTQ methods, they often sacrifice the low-bit quantization performance or suffer from an unstable quantization process. The main reason for this unsatisfactory performance is that all the schemes mentioned above, except OPTQ, rely on nearest-rounding and do not optimize the weight-rounding policy. Moreover, most of them target layer-wise reconstruction in [\(2\)](#page-1-2), not block-wise reconstruction in [\(4\)](#page-1-3), thus ignoring the cross-layer dependency within the attention module. Although [\[27,](#page-11-7) [20\]](#page-11-8) target block-wise reconstruction via learning, they need to approximate gradients for a non-differentiable quantization function, which results in an unstable training process (see Table [1](#page-7-0) in Section [4\)](#page-6-0) [\[19\]](#page-11-5).

Thus, we propose a novel PTQ scheme that balances accuracy and efficiency. In contrast to conventional LLM quantization methods, our scheme optimizes a weight-rounding policy while targeting block-wise reconstruction to consider the cross-layer dependency. The key difference over classic block-wise weight-rounding optimization is that we quantize models layer-wise for scalability, whereas layers are jointly quantized in the existing methods. Furthermore, we present an efficient precomputation-based method for the computation of the block-wise objective in [\(4\)](#page-1-3), which significantly reduces the computational overhead caused by repeated attention operations.

## 3 Method

#### 3.1 Motivation

To gain insight into our approach, we first consider the objective of the layer-wise reconstruction in [\(2\)](#page-1-2). Let ∆W(ℓ) = Q(W(ℓ) ) − W(ℓ) , then the reconstruction error can be expressed as

<span id="page-2-2"></span>
$$\mathbb{E}\left[\left\|\Delta \mathbf{W} \mathbf{X}\right\|_{F}^{2}\right] = \mathbb{E}\left[\operatorname{tr}\left(\Delta \mathbf{W} \mathbf{X} \mathbf{X}^{T} \Delta \mathbf{W}^{T}\right)\right] = \operatorname{tr}\left(\Delta \mathbf{W} \cdot \mathbb{E}\left[\mathbf{X} \mathbf{X}^{T}\right] \cdot \Delta \mathbf{W}^{T}\right). \tag{5}$$

<span id="page-2-0"></span><sup>3</sup>By foldable parameters, we mean the parameters that can be merged into other layers within the Transformer block (*e.g.,* LayerNorm), thereby imposing no extra computational cost during the inference [\[13\]](#page-10-6).

<span id="page-3-4"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Quantization strategies (simplified)

Consequently, the layer-wise quantization problem can be recast as follows:

<span id="page-3-0"></span>
$$\min_{\Delta \mathbf{W}} \operatorname{tr} \left( \Delta \mathbf{W} \cdot \mathbb{E} \left[ \mathbf{X} \mathbf{X}^T \right] \cdot \Delta \mathbf{W}^T \right). \tag{6}$$

The new form of the quantization objective in [\(6\)](#page-3-0) implies that if E[XX<sup>T</sup> ] is pre-computed and stored before quantization, we can measure the reconstruction error over the entire calibration dataset with a single matrix multiplication and element-wise multiplication.[4](#page-3-1) This is in contrast to the original formulation in [\(2\)](#page-1-2) which requires the computation of Q(W)X or ∆W X for every input X.

A natural question that arises from this finding is *"Can we also measure the block reconstruction error efficiently based on such a pre-computation?"*. In the following subsections, we describe our main strategy to simplify block-wise quantization and then present a refined objective for the attention module, where the objective can be computed efficiently with certain pre-computed values.

## 3.2 Quantization strategy of *aespa*

When quantizing the attention module using conventional block-wise reconstruction methods (Figure [1\(](#page-2-1)a)), the query, key, and value projections have been jointly optimized such that

$$\min_{\Delta \boldsymbol{W}_{Q}, \Delta \boldsymbol{W}_{K}, \Delta \boldsymbol{W}_{V}} \mathbb{E} \left[ \left\| \operatorname{SA}(\widehat{\boldsymbol{Q}}, \widehat{\boldsymbol{K}}, \widehat{\boldsymbol{V}}) - \operatorname{SA}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) \right\|_{F}^{2} \right], \tag{7}$$

where the output of attention module SA(Q, K,V ) is defined as

<span id="page-3-3"></span>
$$SA(Q, K, V) = softmax \left(\frac{QK^T}{\sqrt{d}}\right) V = AV.$$
 (8)

In such a case, we need to compute SA(Qb, <sup>K</sup>c,V<sup>b</sup> ) for every batch sequence in each iteration, which is computationally heavy and time-consuming (see Section [3.5](#page-6-1) for details on complexity).

To overcome this computational overhead, we quantize each projection *separately* in a divide-andconquer manner. For example, when quantizing the query projection WQ, we fix W<sup>K</sup> and W<sup>V</sup> with full-precision (Figure [1\(](#page-2-1)b)), which facilitates the factoring out of common terms affected by W<sup>K</sup> and W<sup>V</sup> (see Section [3.3](#page-3-2) for details). We emphasize that this strategy differs from conventional layer-wise quantization schemes (*e.g.,* AdaRound and OPTQ) in that we aim to minimize the reconstruction error for the attention module, not the reconstruction error for each layer.

We conduct experiments to demonstrate the importance of targeting attention-wise reconstruction and validity of the proposed quantization strategy. In our experiments, we set the loss function for each projection as the attention reconstruction error in [\(7\)](#page-3-3) but quantize each projection separately (see Figure [2\(](#page-3-4)c)). Table [5](#page-14-0) in Appendix [B](#page-14-1) summarizes the performance of AdaRound, BRECQ, and our approach. As evident, our approach uniformly outperforms AdaRound for all bit-widths, although both methods quantize models layer-wise. This is because we can consider cross-layer dependency (*i.e.,* relationship between the query, key, and value) by targeting attention-wise reconstruction, which is different from AdaRound wherein layers are considered independent. Furthermore, once we target attention-wise reconstruction, separate layer-wise quantization does not incur severe performance degradation compared to the joint quantization method (BRECQ). In fact, our approach causes only a marginal performance degradation for 2-bit and exhibits comparable performance for 3-bit and 4-bit. For further discussion on the proposed strategy, see Appendix [B.](#page-14-1)

#### <span id="page-3-2"></span>3.3 Refined quantization objectives for *aespa*

One might ask whether our strategy incurs more computational cost than that required by the joint quantization because we update only one layer at a time (see Figure [1\(](#page-2-1)b)). This is in contrast

<span id="page-3-1"></span><sup>4</sup>We note that the computation of tr(ABC<sup>T</sup> ) can be implemented as torch.sum((AB) ⊙ C), where ⊙ denotes the element-wise product operation. They are mathematically equivalent.

to existing methods, in which the layers inside the attention module are updated simultaneously (Figure 1(a)). To reduce this additional cost, we refine the quantization objective in (7) for each projection.

Value projection When quantizing the value projection  $W_V$ , the query and key projections are fixed with full-precision. In this case, by factoring out the common term influenced by Q and K, we can simplify the attention reconstruction error  $\Delta SA_V$  as follows:

$$\Delta SA_{V} = \mathbb{E}\left[\left\|\boldsymbol{A}\widehat{\boldsymbol{V}} - \boldsymbol{A}\boldsymbol{V}\right\|_{F}^{2}\right] = \mathbb{E}\left[\left\|\boldsymbol{A}\Delta\boldsymbol{V}\right\|_{F}^{2}\right] = \mathbb{E}\left[\left\|\Delta\boldsymbol{W}_{V}\boldsymbol{X}\boldsymbol{A}^{T}\right\|_{F}^{2}\right].$$
 (9)

Thus, the problem to quantize  $W_V$  to minimize the attention reconstruction error can be recast as

<span id="page-4-4"></span>
$$\min_{\Delta \mathbf{W}_{V}} \mathbb{E}\left[\left\|\Delta \mathbf{W}_{V} \mathbf{X} \mathbf{A}^{T}\right\|_{F}^{2}\right]. \tag{10}$$

Query projection When the key and value projections are fixed with full-precision, the attention reconstruction error  $\Delta SA_Q$  caused by  $\Delta W_Q$  is expressed as

$$\Delta SA_{Q} = \mathbb{E}\left[\left\|SA(\widehat{Q}, K, V) - SA(Q, K, V)\right\|_{F}^{2}\right] = \mathbb{E}\left[\left\|\Delta AV\right\|_{F}^{2}\right],$$
(11)

where  $\Delta A = \operatorname{softmax}(\widehat{Q}K^T/\sqrt{d}) - \operatorname{softmax}(QK^T/\sqrt{d})$ . To avoid the computational overhead of repetitive softmax operations, we approximate  $\Delta A$  with its first-order Taylor series as

<span id="page-4-2"></span><span id="page-4-1"></span><span id="page-4-0"></span>
$$\Delta A \approx \frac{\Delta Q K^T}{\sqrt{d}} \cdot \mathbf{J}_{\text{softmax}}^T,$$
 (12)

where  $J_{\rm softmax}$  is the Jacobian of the softmax function. By combining (11) and (12), we obtain

$$\Delta SA_Q \approx \frac{1}{d} \mathbb{E} \left[ \left\| \Delta \mathbf{Q} \mathbf{K}^T \mathbf{J}_{\text{softmax}}^T \mathbf{V} \right\|_F^2 \right] = \frac{1}{d} \mathbb{E} \left[ \left\| \mathbf{V}^T \mathbf{J}_{\text{softmax}} \mathbf{K} \Delta \mathbf{W}_Q \mathbf{X} \right\|_F^2 \right]. \tag{13}$$

Although we can circumvent conducting attention operations using the modified form in (13), a large amount of memory is required to store the Jacobian  $J_{\rm softmax}$  (e.g., more than 100 GB of memory for OPT-125M).<sup>5</sup> As a cost-effective alternative, we build an upper bound of (13) and then employ it as a surrogate of  $\Delta SA_Q$  when quantizing  $W_Q$ . Specifically, by noting that

$$\|\boldsymbol{V}^{T}\mathbf{J}_{\text{softmax}}\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\|_{F}^{2} \leq \|\boldsymbol{V}^{T}\mathbf{J}_{\text{softmax}}\|_{F}^{2} \cdot \|\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\|_{F}^{2}$$
(14)

and the term  $\|V^T\mathbf{J}_{\mathrm{softmax}}\|_F^2$  is fixed in the quantization process, we minimize  $\|K\Delta W_QX\|_F^2$  with the hope that  $\Delta\mathrm{SA}_Q$  also decreases. In other words, our quantization objective for  $W_Q$  is

<span id="page-4-5"></span>
$$\min_{\Delta \mathbf{W}_{Q}} \mathbb{E}\left[\left\|\mathbf{K}\Delta \mathbf{W}_{Q}\mathbf{X}\right\|_{F}^{2}\right]. \tag{15}$$

**Key projection** By taking similar steps, the quantization objective for the key projection  $W_K$  can be formulated as (see Appendix C for the detailed derivation)

<span id="page-4-6"></span>
$$\min_{\Delta \boldsymbol{W}_{K}} \mathbb{E}\left[\left\|\boldsymbol{Q}\Delta \boldsymbol{W}_{K} \boldsymbol{X}\right\|_{F}^{2}\right]. \tag{16}$$

#### <span id="page-4-7"></span>3.4 Algorithm description

The proposed *aespa* consists of two main steps. Specifically, *aespa* first determines the quantization parameters (*i.e.*, scale and zero-point) and then optimizes an integer weight  $W_{int}$  for each weight.

Note that we only used the definition of the attention operation when developing the refined objectives in (10), (15), and (16). Thus, our objectives can be integrated into any layer-wise quantization scheme without effort. For example, we can compute the quantization parameters by combining existing parameter initialization algorithms (*e.g.*, AWQ [19] and Z-FOLD [13]) with the proposed objectives. We can also optimize a weight-rounding policy using conventional methods (*e.g.*, AdaRound [23])

<span id="page-4-3"></span><sup>&</sup>lt;sup>5</sup>Note that the shape of  $\mathbf{J}_{\mathrm{softmax}}$  is [L, L, L] (L is the input sequence length) for each attention head because  $\mathbf{J}_{\mathrm{softmax}}(\boldsymbol{a}_{\ell}) = \mathrm{diag}(\boldsymbol{a}_{\ell}) - \boldsymbol{a}_{\ell}^T \boldsymbol{a}_{\ell} \in \mathbb{R}^{L \times L}$  for each row  $\boldsymbol{a}_{\ell}$  of  $\boldsymbol{A}$ .

together with our objectives (see Appendix F for details). In the proposed *aespa*, we use Z-FOLD in computing the quantization parameters and employ AdaRound in optimizing a weight-rounding policy. In Algorithm 1 (see Appendix A), we summarize the proposed *aespa*.<sup>6</sup>

To accelerate the weight-rounding learning process, we further modify the objective functions such that the value can be computed efficiently via pre-computation, as in (5).

**Modified objective for (10)** The proposed objective for the value projection can be recast as

$$\mathbb{E}\left[\left\|\Delta \mathbf{W}_{V} \mathbf{X} \mathbf{A}^{T}\right\|_{F}^{2}\right] = \operatorname{tr}\left(\Delta \mathbf{W}_{V} \mathbb{E}\left[\mathbf{X} \mathbf{A}^{T} \mathbf{A} \mathbf{X}^{T}\right] \Delta \mathbf{W}_{V}^{T}\right). \tag{17}$$

The modified objective allows us to perform each iteration of the weight-rounding learning efficiently. Specifically, by computing  $\mathbb{E}[\boldsymbol{X}\boldsymbol{A}^T\boldsymbol{A}\boldsymbol{X}^T]$  before quantization and reusing it in the quantization process<sup>7</sup>, we can avoid the overhead of computing  $\|\Delta \boldsymbol{W}_V \boldsymbol{X} \boldsymbol{A}^T\|_F^2$  for every input  $\boldsymbol{X}$  and compute the loss with one simple matrix multiplication and a single element-wise multiplication (see Footnote 4).

Another intriguing feature of this modification is that it facilitates a more reliable update of  $\Delta W_V$  than the original objective in (10). Specifically, because  $\mathbb{E}[XA^TAX^T]$  is pre-computed using all calibration data, the loss computed with (17) considers the entire calibration dataset (*i.e.*, the batch size is the total number of data). Thus, a better estimate of the true gradient can be obtained without any memory issues, which could lead to more consistent updates of  $\Delta W_V$  and faster convergence [28].

The modified objective in (17) also implies that the Hessian  $\mathbf{H}_V$  for each row of  $\mathbf{W}_V$  is

<span id="page-5-7"></span><span id="page-5-3"></span><span id="page-5-2"></span>
$$\mathbf{H}_{V} = 2\mathbb{E}[\mathbf{X}\mathbf{A}^{T}\mathbf{A}\mathbf{X}^{T}]. \tag{18}$$

We note that the proposed Hessian  $\mathbf{H}_V$  differs from  $\mathbf{H} = 2\mathbb{E}[\boldsymbol{X}\boldsymbol{X}^T]$ , which has been commonly used as an approximated Hessian in conventional methods [6, 7, 13, 3]. The key reason for the difference is that we consider the dependency between  $\boldsymbol{W}_Q, \boldsymbol{W}_K$ , and  $\boldsymbol{W}_V$  by targeting attentionwise reconstruction, whereas the previous methods assumed independence. To observe the effect of considering the cross-layer dependency, we use different Hessians (i.e.,  $\mathbf{H}_V$  and  $\mathbf{H}$ ) when quantizing language models and then compare the performance of the quantized models (see Appendix D). Evidently, the quantization performance is much better when the proposed Hessian  $\mathbf{H}_V$  is used, which demonstrates the importance of considering the cross-layer dependency.

**Modified objectives for (15) and (16)** If we denote the vectorized representation of  $\Delta W_Q$  as  $\Delta w_Q$ , the proposed objective in (15) can be expressed as (see Appendix E for the derivation)

$$\mathbb{E}\left[\left\|\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\right\|_{F}^{2}\right] = \Delta\boldsymbol{w}_{Q}^{T} \cdot \mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\otimes\boldsymbol{K}^{T}\boldsymbol{K}\right] \cdot \Delta\boldsymbol{w}_{Q}.$$
(19)

where  $\otimes$  is the Kronecker product operation. To reduce the memory cost of storing the Kronecker product term  $\mathbb{E}[XX^T \otimes K^TK]$ , we approximate it as [2]

<span id="page-5-6"></span><span id="page-5-5"></span><span id="page-5-4"></span>
$$\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\otimes\boldsymbol{K}^{T}\boldsymbol{K}\right]\approx\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right]\otimes\mathbb{E}\left[\boldsymbol{K}^{T}\boldsymbol{K}\right].$$
(20)

By combining (19) and (20), we obtain

$$\mathbb{E}\left[\|\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\|_{F}^{2}\right] \approx \Delta\boldsymbol{w}_{Q}^{T} \cdot \left(\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right] \otimes \mathbb{E}\left[\boldsymbol{K}^{T}\boldsymbol{K}\right]\right) \cdot \Delta\boldsymbol{w}_{Q}$$

$$\stackrel{(a)}{=} \operatorname{tr}\left(\mathbb{E}\left[\boldsymbol{K}^{T}\boldsymbol{K}\right]\Delta\boldsymbol{W}_{Q}\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right]\Delta\boldsymbol{W}_{Q}^{T}\right), \tag{21}$$

where the proof of (a) is provided in Appendix E. By taking similar steps, the objective for the key projection can be recast as

$$\mathbb{E}\left[\|\boldsymbol{Q}\Delta\boldsymbol{W}_{K}\boldsymbol{X}\|_{F}^{2}\right] = \operatorname{tr}\left(\mathbb{E}\left[\boldsymbol{Q}^{T}\boldsymbol{Q}\right]\Delta\boldsymbol{W}_{K}\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right]\Delta\boldsymbol{W}_{K}^{T}\right). \tag{22}$$

The modified objectives in (21) and (22) imply that the loss over the total calibration dataset can be calculated efficiently by computing  $\mathbb{E}[K^TK]$ ,  $\mathbb{E}[Q^TQ]$ , and  $\mathbb{E}[XX^T]$  in advance.

<span id="page-5-0"></span><sup>&</sup>lt;sup>6</sup>We use the layer-wise objective in (6) for the weights other than the query, key, and value projections (*i.e.*, out-projection and weights inside the feed-forward network).

<span id="page-5-1"></span>The term  $\mathbb{E}[XA^TAX^T]$  is not affected by  $\Delta W_V$  and thus fixed in the quantization process.

#### <span id="page-6-1"></span>3.5 Complexity analysis for aespa

We discuss the computational complexity of *aespa*. Specifically, we analyze the number of floating-point operations (flops) required to perform one iteration for weight-rounding optimization (line 6 in Algorithm 1). For each projection, the required number of flops is summarized as follows.

- Value: By reusing the pre-computed  $\mathbb{E}[XA^TAX^T]$ , the loss value in (17) can be computed with one matrix multiplication and one element-wise multiplication/addition (see Footnote 4). The associated cost is  $2d_hd^2 + d_hd 1$  flops, where d is the hidden size and  $d_h$  is the input dimension for each attention head.
- Query/key: Once  $\mathbb{E}[K^TK]$ ,  $\mathbb{E}[Q^TQ]$ , and  $\mathbb{E}[XX^T]$  have been computed in advance, the loss values in (21) and (22) can be computed by performing two matrix multiplications and one element-wise multiplication/addition. This requires  $2d_hd^2 + 2d_h^2d 1$  flops for each projection.

To summarize, the total number of flops required in each iteration of the proposed aespa is

$$C_{aespa} = 6d_h d^2 + 4d_h^2 d + d_h d - 3 = \mathcal{O}(d_h d^2). \tag{23}$$

We emphasize that regardless of the amount of calibration data, the number of flops to compute the loss considering the entire dataset is fixed as  $C_{aespa}$ .

We now compare the complexities of aespa and conventional block-wise quantization methods. It can be easily verified that the existing methods require the following number of flops for handling B input sequences of length L (see Appendix G):

$$C_{exist} = B(6d_h dL + 4d_h L^2 + 2L^2 - L - 1) = \mathcal{O}(Bd_h L \cdot \max\{d, L\}). \tag{24}$$

Table 7 in Appendix G summarizes the computational costs for different sizes of OPT models. For the conventional methods, we report the cost of using four sequences in each iteration (B=4). We observe that the computational cost of aespa is considerably lower than that of conventional methods. In particular, for small-scale models, aespa performs ten times fewer number of flops. It can be observed that the gap between  $\mathcal{C}_{aespa}$  and  $\mathcal{C}_{exist}$  decreases as the model size increases. This is because the hidden size d exceeds the sequence length L (which is fixed for all models) for large models. Nevertheless, aespa still incurs a lower computational cost, and the gap increases if conventional methods use larger batch sizes.

## <span id="page-6-0"></span>4 Experimental results

#### 4.1 Experimental setup

We quantize publicly available LLMs (e.g., OPT [35], BLOOM [26], LLaMA [29], and LLaMA2 [30]) using the proposed aespa. When implementing aespa, we compute the quantization parameters with Z-FOLD [13] and optimize a weight-rounding policy via AdaRound [23], where the proposed rowwise Hessians and loss functions (see Table 4 in Appendix A) are utilized instead of the existing ones. When computing the quantization parameters, we follow the stopping criterion introduced by [13]. Before optimizing a weight-rounding policy, we update the full-precision weights via OPTQ [7], which empirically reduces the number of iterations required for weight-rounding optimization. When optimizing a weight-rounding policy, we set the number of iterations, learning rate, and weight of the rounding loss (see  $\lambda$  in (28)) to 2,000, 0.015, and 1.5, respectively.

When constructing the calibration dataset, we randomly sample 128 segments consisting of 2048 tokens from the C4 dataset [24] as in [7, 13, 3]. In our experiments, we quantize only weights and retain activations in full-precision because activations are not a significant bottleneck for LLMs [7] and the inference of LLMs can be accelerated sufficiently by reducing memory movement through weight quantization [15]. We evaluate the performance of the quantized models using benchmark datasets (e.g., WikiText-2 [22], C4 [24], and PTB [21]) and zero-shot tasks. Except for the experiments on the LLaMA2 models, which were performed using an NVIDIA H100 GPU, we conducted all experiments using a single NVIDIA A100 GPU (80 GB).

## 4.2 Comparison with prior arts

**Comparison with block-wise PTQ schemes** We compare the proposed *aespa* with conventional block-wise PTQ methods, among which BRECQ is a classic weight-rounding optimization method,

<span id="page-7-0"></span>Table 1: Performance (PPL ↓) of the proposed *aespa* and conventional block-wise PTQ methods.

#### (a) WikiText-2

|           |                  |       |       | OPT   |       |       | LLaMA |       |       | LLaMA2 |
|-----------|------------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| Precision | Method           | 125M  | 1.3B  | 2.7B  | 6.7B  | 7B    | 13B   | 30B   | 7B    | 13B    |
| FP16      | Baseline         | 27.65 | 14.63 | 12.47 | 10.86 | 5.677 | 5.091 | 4.101 | 5.472 | 4.884  |
|           | BRECQ [18]       | 33.25 | 16.09 | 13.37 | OOM   | OOM   | OOM   | OOM   | OOM   | OOM    |
|           | OmniQuant [27]   | 39.14 | 17.59 | 14.87 | 12.87 | 6.716 | 5.798 | 4.963 | 6.798 | 5.751  |
| INT3      | AffineQuant [20] | 36.15 | 17.26 | 14.25 | 12.30 | 6.712 | 5.820 | 4.951 | 6.795 | 5.757  |
|           | aespa            | 32.71 | 15.79 | 13.14 | 11.23 | 6.579 | 5.611 | 4.688 | 6.241 | 5.462  |
|           | BRECQ [18]       | 60.38 | 56.25 | 113.6 | OOM   | OOM   | OOM   | OOM   | OOM   | OOM    |
|           | OmniQuant [27]   | NaN   | 399.6 | 1.6e3 | 4.9e3 | 18.18 | NaN   | 10.15 | 35.40 | 20.19  |
| INT2      | AffineQuant [20] | 143.9 | 56.45 | 35.16 | 25.32 | 18.83 | 11.08 | NaN   | NaN   | 18.49  |
|           | aespa            | 71.18 | 24.26 | 22.22 | 15.71 | 11.94 | 10.30 | 7.845 | 13.99 | 12.14  |

#### (b) C4

|           |                  |       |       | OPT   |       |       | LLaMA |       |       | LLaMA2 |
|-----------|------------------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| Precision | Method           | 125M  | 1.3B  | 2.7B  | 6.7B  | 7B    | 13B   | 30B   | 7B    | 13B    |
| FP16      | Baseline         | 26.56 | 16.07 | 14.34 | 12.71 | 7.344 | 6.798 | 6.131 | 7.264 | 6.727  |
|           | BRECQ [18]       | 29.74 | 17.46 | 15.39 | OOM   | OOM   | OOM   | OOM   | OOM   | OOM    |
|           | OmniQuant [27]   | 34.92 | 18.83 | 16.80 | 14.21 | 8.605 | 7.604 | 6.822 | 9.085 | 7.821  |
| INT3      | AffineQuant [20] | 32.78 | 18.27 | 16.11 | 13.80 | 8.631 | 7.609 | 6.803 | 9.059 | 7.732  |
|           | aespa            | 29.51 | 17.10 | 15.27 | 13.15 | 8.465 | 7.399 | 6.634 | 8.225 | 7.392  |
|           | BRECQ [18]       | 47.85 | 41.05 | 83.32 | OOM   | OOM   | OOM   | OOM   | OOM   | OOM    |
|           | OmniQuant [27]   | NaN   | 239.1 | 1.1e3 | 4.4e3 | 18.59 | NaN   | 14.74 | 26.27 | 18.93  |
| INT2      | AffineQuant [20] | 95.86 | 43.66 | 29.75 | 24.04 | 16.87 | 12.79 | NaN   | NaN   | 15.20  |
|           | aespa            | 56.88 | 23.54 | 22.53 | 17.28 | 13.63 | 11.46 | 10.35 | 14.36 | 13.59  |

<sup>\*</sup> 'NaN' means that loss diverges in the quantization process.

and OmniQuant and AffineQuant are LLM quantization methods that mitigate the computational overhead of BRECQ by learning only a few quantization and foldable parameters [\[27,](#page-11-7) [20\]](#page-11-8). For OmniQuant and AffineQuant, we ran the official codes[8](#page-7-1) provided by the authors. For both methods, we activated the learnable equivalent transformation (LET) and learnable weight clipping (LWC) options and reported the obtained results. When implementing BRECQ, we employed the hyperparameter settings provided in [\[18\]](#page-11-3). In this comparison, the BLOOM models and OPT-350M were excluded because they are not supported by OmniQuant and AffineQuant.

As Table [1](#page-7-0) shows, *aespa* uniformly outperforms OmniQuant/AffineQuant.[9](#page-7-2) In particular, the performance gap is significant for 2-bit; while OmniQuant/AffineQuant suffer from instability (*i.e.,* loss diverges) or collapse (perplexity (PPL) > 10<sup>3</sup> ), *aespa* exhibits reasonable PPL. The outstanding performance is attributed to the fact that *aespa* optimizes a weight-rounding policy after determining the quantization parameters (lines 5-8 in Algorithm [1\)](#page-13-0), whereas OmniQuant/AffineQuant rely on the naive nearest rounding and approximate gradients for the non-differentiable quantization function.

Although BRECQ performs best for the 2-bit quantization of OPT-125M, it lacks scalability; BRECQ requires approximately 20 GPU hours for a relatively small-scale OPT-2.7B (see Table [14](#page-25-1) in Appendix [K\)](#page-25-0). Even for OPT-125M, BRECQ requires approximately 2 GPU hours, whereas the proposed *aespa* completes quantization in 5 minutes. One might wonder why the performance of BRECQ worsens as the model size increases. We assume that this is attributable to the choice of hyperparameters (*e.g.,* learning rate and weight of rounding loss). In fact, the hyperparameters presented in [\[18\]](#page-11-3) have been optimized for ImageNet, but not for LLMs. It is expected that we can obtain better performance for BRECQ via deliberate hyperparameter tuning; however, this would not be feasible for real-world deployment because it requires considerable time (see Table [14](#page-25-1) in Appendix [K\)](#page-25-0).

Comparison with layer-wise PTQ schemes We compare the proposed *aespa* with conventional layer-wise PTQ schemes, among which RTN is the method that naively assigns the nearest grid, OPTQ is a backpropagation-free weight-rounding optimization algorithm [\[7\]](#page-10-5), and Z-FOLD is the

<sup>\*</sup> 'OOM' means that out-of-memory issues occur when quantizing models with a single A100 GPU.

<sup>\*</sup> Results for high bit-widths are provided in Appendix [H](#page-20-0) due to the page limitation.

<span id="page-7-2"></span><span id="page-7-1"></span><sup>8</sup> https://github.com/OpenGVLab/OmniQuant, https://github.com/bytedance/AffineQuant

<sup>9</sup>We note that our results are different from those reported in [\[27,](#page-11-7) [20\]](#page-11-8) where a different calibration dataset (WikiText-2) was used; see Appendix [L](#page-26-0) for more discussion on this issue.

| Table 2: Performance (PPL ↓ | of aespa and existing layer-wise PTQ | methods on BLOOM models. |
|-----------------------------|--------------------------------------|--------------------------|
|                             |                                      |                          |

<span id="page-8-0"></span>

| Precision | Method   |       | 7     | WikiText-2 | 2     |       | C4    |       |       |       |       |  |  |
|-----------|----------|-------|-------|------------|-------|-------|-------|-------|-------|-------|-------|--|--|
| Fiecision | Method   | 560M  | 1.1B  | 1.7B       | 3B    | 7.1B  | 560M  | 1.1B  | 1.7B  | 3B    | 7.1B  |  |  |
| FP16      | Baseline | 22.42 | 17.69 | 15.39      | 13.48 | 11.37 | 26.60 | 22.05 | 19.49 | 17.49 | 15.20 |  |  |
|           | RTN      | 56.74 | 49.85 | 63.37      | 39.07 | 17.35 | 66.99 | 60.41 | 113.6 | 79.84 | 22.54 |  |  |
| INT3      | OPTQ     | 31.55 | 23.84 | 20.06      | 17.13 | 13.56 | 34.62 | 27.62 | 23.87 | 20.96 | 17.43 |  |  |
| 11113     | Z-Fold   | 26.52 | 20.99 | 17.39      | 15.11 | 12.26 | 29.97 | 24.43 | 21.52 | 19.01 | 16.12 |  |  |
|           | aespa    | 25.39 | 19.81 | 16.95      | 14.68 | 12.00 | 29.10 | 23.80 | 20.93 | 18.55 | 15.91 |  |  |
|           | RTN      | 7.8e5 | 9.8e5 | 3.5e5      | 1.4e5 | 2.1e5 | 1.4e6 | 2.1e6 | 2.7e5 | 9.2e4 | 1.3e5 |  |  |
| INT2      | OPTQ     | 1.7e3 | 1.9e3 | 1.4e3      | 796.5 | 194.2 | 533.4 | 538.0 | 562.9 | 351.6 | 112.8 |  |  |
| 11112     | Z-Fold   | 65.45 | 44.50 | 35.69      | 27.40 | 18.87 | 64.11 | 42.96 | 37.26 | 32.64 | 22.46 |  |  |
|           | aespa    | 44.91 | 34.12 | 27.67      | 21.65 | 16.31 | 45.04 | 35.12 | 29.95 | 25.04 | 20.00 |  |  |

<sup>\*</sup> Results for high bit-widths and other language models (e.g., OPT, LLaMA, and LLaMA2) are provided in Appendix I.

method exploiting additional foldable parameters to quantize weights more elaborately [13]. Table 2 and Tables 9-12 (see Appendix I) summarize the results for the OPT, BLOOM, LLaMA, and LLaMA2 models of various sizes. Evidently, *aespa* uniformly outperforms conventional schemes, regardless of the size and type of LLMs. In particular, for 2-bit, there exists a significant performance gap between *aespa* and existing methods; the PPL obtained by *aespa* is twice as low as those of conventional methods for small-scale models (*e.g.*, OPT-125M). The key factors leading to such an outstanding performance are: 1) the consideration of the cross-layer dependency achieved by targeting attentionwise reconstruction, and 2) efficient weight-rounding optimization based on pre-computations.

**Zero-shot task performance** We evaluate the reasoning performance of quantized models using zero-shot tasks (*e.g.*, ARC [4], HellaSwag [34], and MMLU [8]). We note that the zero-shot setting was ensured in our experiments because we used excerpts from randomly crawled websites (not task-specific data) as a calibration dataset. From the zero-shot results in Table 3 and Table 13 (see Appendix J), we observe that the proposed *aespa* performs the best in almost all cases, and the performance gap between *aespa* and the existing methods is large for 2-bit.

**Time cost** We summarize the processing times of the different quantization algorithms in Appendix K. We note that the processing time of *aespa* includes the time required for pre-computations (lines 2-4 in Algorithm 1). As expected, *aespa* completes quantization much faster than BRECQ. For example, while BRECQ requires more than 10 GPU hours for OPT-1.3B, *aespa* completes quantization in 1.24 hours, which demonstrates the effectiveness of the proposed pre-computation-based loss computation strategy. Although other block-wise methods (OmniQuant/AffineQuant) perform quantization faster than *aespa* for hyper-scale models, they suffer from unstable training processes or exhibit poor PPL performance (*e.g.*, PPL of OmniQuant is larger than 10<sup>3</sup> for OPT-6.7B; see Table 1). In addition, we observe that OPTQ performs quantization quickly, but its 2-bit performance collapses regardless of the model size (see Table 9 in Appendix I). Except for *aespa*, Z-FOLD is the only method that shows both reasonable performance and processing time.

**Discussion** In real situations, when one needs to preserve the performance of the original model as much as possible, the proposed *aespa* would be an intriguing solution. In particular, when deploying LLMs on resource-constrained platforms where up to 7B models are commonly employed (*e.g.*, mobile devices), *aespa* would be a good fit. Even when fast quantization of hyper-scale models is required, *aespa* can be used with a slight modification. Specifically, in time-limited cases, one can skip the weight-rounding optimization (lines 5-8 in Algorithm 1) and simply determine the quantization parameters using the proposed Hessian that considers the cross-layer dependency (line 4 in Algorithm 1). In doing so, we can not only save the time required to optimize a weight-rounding mechanism, but also save the memory required to store pre-computed values ( $\mathbb{E}[\mathbf{K}^T\mathbf{K}]$  and  $\mathbb{E}[\mathbf{Q}^T\mathbf{Q}]$ ). Indeed, when performing only quantization parameter computation, we achieved a significant reduction in the processing time (see Table 15 in Appendix K) while still exhibiting better performance than conventional methods (see Table 6 in Appendix D).

## 5 Conclusion

We proposed a next-level PTQ scheme for Transformers, called *aespa*. By targeting the attentionwise reconstruction while quantizing Transformers layer-wise, we could consider the cross-layer dependency within the attention module and complete the quantization much faster than the existing

<span id="page-9-0"></span>Table 3: INT2 zero-shot performance (accuracy ↑) of *aespa* and existing methods.

| Model      | Method           | ARC-c | ARC-e | HellaSwag | MMLU  | Average |
|------------|------------------|-------|-------|-----------|-------|---------|
|            | FP16             | 44.62 | 72.85 | 76.18     | 32.19 | 56.46   |
|            | RTN              | 28.67 | 25.00 | 26.43     | 25.72 | 26.46   |
|            | OPTQ [7]         | 29.18 | 26.14 | 26.18     | 24.04 | 26.39   |
| LLaMA-7B   | Z-FOLD [13]      | 30.63 | 52.44 | 53.55     | 23.27 | 39.97   |
|            | OmniQuant [27]   | 27.22 | 49.20 | 50.65     | 23.74 | 37.70   |
|            | AffineQuant [20] | 27.90 | 49.58 | 51.85     | 24.15 | 38.37   |
|            | aespa            | 33.36 | 55.64 | 58.31     | 23.12 | 42.61   |
|            | FP16             | 47.87 | 74.75 | 79.08     | 43.46 | 61.29   |
|            | RTN              | 28.16 | 27.15 | 26.09     | 25.53 | 26.73   |
|            | OPTQ [7]         | 27.22 | 25.76 | 25.67     | 25.05 | 25.93   |
| LLaMA-13B  | Z-FOLD [13]      | 32.68 | 58.08 | 57.89     | 26.44 | 43.77   |
|            | OmniQuant [27]   | NaN   | NaN   | NaN       | NaN   | NaN     |
|            | AffineQuant [20] | 32.17 | 56.36 | 60.29     | 25.22 | 43.51   |
|            | aespa            | 34.73 | 61.49 | 62.68     | 28.74 | 46.91   |
|            | FP16             | 52.90 | 78.96 | 82.63     | 54.66 | 67.29   |
|            | RTN              | 27.05 | 26.39 | 25.87     | 25.48 | 26.20   |
|            | OPTQ [7]         | 27.13 | 26.60 | 26.12     | 23.56 | 25.85   |
| LLaMA-30B  | Z-FOLD [13]      | 39.93 | 65.07 | 65.89     | 30.85 | 50.44   |
|            | OmniQuant [27]   | 34.22 | 58.50 | 64.83     | 25.91 | 45.87   |
|            | AffineQuant [20] | NaN   | NaN   | NaN       | NaN   | NaN     |
|            | aespa            | 41.13 | 67.00 | 67.90     | 35.67 | 52.93   |
|            | FP16             | 46.16 | 74.49 | 75.99     | 41.87 | 59.63   |
|            | RTN              | 28.33 | 26.01 | 25.88     | 23.02 | 25.81   |
|            | OPTQ [7]         | 26.37 | 26.09 | 25.11     | 25.10 | 25.67   |
| LLaMA2-7B  | Z-FOLD [13]      | 26.62 | 42.68 | 44.71     | 22.88 | 34.22   |
|            | OmniQuant [27]   | 25.00 | 38.80 | 42.97     | 23.03 | 32.45   |
|            | AffineQuant [20] | NaN   | NaN   | NaN       | NaN   | NaN     |
|            | aespa            | 30.29 | 51.47 | 56.75     | 25.59 | 41.03   |
|            | FP16             | 49.06 | 77.44 | 79.39     | 52.10 | 64.50   |
|            | RTN              | 27.22 | 25.04 | 25.58     | 24.69 | 25.63   |
|            | OPTQ [7]         | 26.71 | 27.19 | 25.42     | 23.74 | 25.77   |
| LLaMA2-13B | Z-FOLD [13]      | 28.41 | 48.32 | 51.59     | 23.98 | 38.08   |
|            | OmniQuant [27]   | 27.13 | 47.98 | 53.27     | 23.81 | 38.05   |
|            | AffineQuant [20] | 30.80 | 52.90 | 57.74     | 24.45 | 41.47   |
|            | aespa            | 31.91 | 55.18 | 55.49     | 29.97 | 43.14   |
|            |                  |       |       |           |       |         |

<sup>\*</sup> 'NaN' means that loss diverges in the quantization process.

approach for block-wise reconstruction (*i.e.,* BRECQ). Extensive experiments on language models have demonstrated the outstanding performance of *aespa*.

Limitations and future work While we focused on the attention output, the output of the entire Transformer block (containing fully connected layers) can be used to consider the dependencies between more layers. However, in this case, the objective functions would be more complicated than those in [\(13\)](#page-4-2) and [\(25\)](#page-15-1) due to nonlinear activation functions (*e.g.,* SiLU for LLaMA models), normalization layers, and weights of larger dimensions. Enhancing the quantization performance by developing an efficient form of the reconstruction error for the Transformer block would be an interesting future work. Furthermore, while we focused on weight-only quantization, activations may need to be quantized to deploy AI models on integer-only arithmetic hardware (*e.g.,* NPU). Extending the proposed *aespa* for weight-activation quantization by integrating existing techniques to suppress activation outliers [\[33,](#page-12-1) [1\]](#page-10-13) is also an interesting research direction. Finally, while we verified the performance of *aespa* with LLMs, we believe that *aespa* can also be used for the quantization of diffusion models. To that end, we may need to incorporate some diffusion-specific quantization strategies to overcome output distribution discrepancies over different time steps (*e.g.,* grouping of time-steps with similar distributions [\[32\]](#page-11-15), temporal feature preservation [\[9\]](#page-10-14), and separate quantization for shortcuts in U-Net [\[17\]](#page-11-16)), which will be considered in our future studies.

<sup>\*</sup> Results for high bit-widths are provided in Appendix [J](#page-24-1) due to the page limitation.

## References

- <span id="page-10-13"></span>[1] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4-bit inference in rotated LLMs. *arXiv:2404.00456*, 2024.
- <span id="page-10-9"></span>[2] Aleksandar Botev, Hippolyt Ritter, and David Barber. Practical Gauss-Newton optimisation for deep learning. In *International Conference on Machine Learning*, pages 557–565. PMLR, 2017.
- <span id="page-10-8"></span>[3] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. QuIP: 2-bit quantization of large language models with guarantees. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
- <span id="page-10-11"></span>[4] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. *arXiv:1803.05457v1*, 2018.
- <span id="page-10-0"></span>[5] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. In *International Conference on Learning Representations (ICLR)*, 2019.
- <span id="page-10-3"></span>[6] Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate posttraining quantization and pruning. *Advances in Neural Information Processing Systems*, 35:4475– 4488, 2022.
- <span id="page-10-5"></span>[7] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: Accurate quantization for generative pre-trained Transformers. In *The Eleventh International Conference on Learning Representations*, 2023.
- <span id="page-10-12"></span>[8] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In *International Conference on Learning Representations*, 2020.
- <span id="page-10-14"></span>[9] Yushi Huang, Ruihao Gong, Jing Liu, Tianlong Chen, and Xianglong Liu. TFMQ-DM: Temporal feature maintenance quantization for diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7362–7371, 2024.
- <span id="page-10-2"></span>[10] Itay Hubara, Yury Nahshan, Yair Hanani, Ron Banner, and Daniel Soudry. Accurate post training quantization with small calibration sets. In *International Conference on Machine Learning*, pages 4466–4475. PMLR, 2021.
- <span id="page-10-4"></span>[11] Yongkweon Jeon, Chungman Lee, Eulrang Cho, and Yeonju Ro. Mr. BiQ: Post-training nonuniform quantization based on minimizing the reconstruction error. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12329–12338, 2022.
- <span id="page-10-7"></span>[12] Yongkweon Jeon, Chungman Lee, and Ho-young Kim. GENIE: show me the data for quantization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12064–12073, 2023.
- <span id="page-10-6"></span>[13] Yongkweon Jeon, Chungman Lee, Kyungphil Park, and Ho-young Kim. A frustratingly easy post-training quantization scheme for llms. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 14446–14461, 2023.
- <span id="page-10-1"></span>[14] Sangil Jung, Changyong Son, Seohyung Lee, Jinwoo Son, Jae-Joon Han, Youngjun Kwak, Sung Ju Hwang, and Changkyu Choi. Learning to quantize deep networks by optimizing quantization intervals with task loss. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4350–4359, 2019.
- <span id="page-10-10"></span>[15] Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W Mahoney, and Kurt Keutzer. SqueezeLLM: Dense-and-sparse quantization. *arXiv:2306.07629*, 2023.

- <span id="page-11-6"></span>[16] Yann LeCun, John S Denker, Sara A Solla, Richard E Howard, and Lawrence D Jackel. Optimal brain damage. In *Advances in Neural Information Processing Systems (NIPS)*, volume 2, pages 598–605, 1989.
- <span id="page-11-16"></span>[17] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-Diffusion: Quantizing diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 17535–17545, 2023.
- <span id="page-11-3"></span>[18] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. BRECQ: Pushing the limit of post-training quantization by block reconstruction. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-11-5"></span>[19] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for llm compression and acceleration. In *MLSys*, 2024.
- <span id="page-11-8"></span>[20] Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, Xuefeng Xiao, Rui Wang, Shilei Wen, Fei Chao, and Rongrong Ji. AffineQuant: Affine transformation quantization for large language models. *arXiv:2403.12544*, 2024.
- <span id="page-11-14"></span>[21] Mitchell Marcus, Beatrice Santorini, and Mary Ann Marcinkiewicz. Building a large annotated corpus of english: The penn treebank. 1993.
- <span id="page-11-13"></span>[22] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. *arXiv:1609.07843*, 2016.
- <span id="page-11-2"></span>[23] Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? Adaptive rounding for post-training quantization. In *International Conference on Machine Learning (ICML)*, pages 7197–7206, 2020.
- <span id="page-11-12"></span>[24] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(1):5485–5551, 2020.
- <span id="page-11-0"></span>[25] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10684–10695, June 2022.
- <span id="page-11-10"></span>[26] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, ´ Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. BLOOM: A 176B-parameter open-access multilingual language model. *arXiv:2211.05100*, 2022.
- <span id="page-11-7"></span>[27] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. OmniQuant: Omnidirectionally calibrated quantization for large language models. *arXiv:2308.13137*, 2023.
- <span id="page-11-9"></span>[28] Samuel L Smith, Pieter-Jan Kindermans, Chris Ying, and Quoc V Le. Don't decay the learning rate, increase the batch size. In *International Conference on Learning Representations*, 2018.
- <span id="page-11-1"></span>[29] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*, 2023.
- <span id="page-11-11"></span>[30] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv:2307.09288*, 2023.
- <span id="page-11-4"></span>[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-11-15"></span>[32] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate post-training quantization for diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 16026–16035, 2024.

- <span id="page-12-1"></span>[33] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning*, pages 38087–38099. PMLR, 2023.
- <span id="page-12-2"></span>[34] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4791–4800, 2019.
- <span id="page-12-0"></span>[35] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open pre-trained transformer language models. *arXiv:2205.01068*, 2022.

## **Appendices**

## <span id="page-13-1"></span>A Pseudo-code for the proposed aespa

In this appendix, we provide the pseudo-code for the proposed *aespa* excluded in the main text due to the page limitation.

#### <span id="page-13-0"></span>Algorithm 1 Quantization 1: **def** QUANTIZATION(W,X) Approximate the Hessian **H** ⊳ See Table 4 Estimate $\mathbb{E}[K^TK], \mathbb{E}[Q^TQ]$ for $W_O, W_K$ ⊳ Table 4 3: 4: Set the step size S s.t. $\min_{S} \operatorname{tr} (\Delta W \mathbf{H} \Delta W^{T})$ 5: repeat 6: Compute the Loss $\mathcal L$ ⊳ Table 4 7: Optimize S or $W_{int}$ w.r.t $\mathcal{L}$ by certain algorithm until converged 8: return $\boldsymbol{S}$ and $\boldsymbol{W}_{int}$ 9: ⊳ step size and integer weight

<span id="page-13-2"></span>Table 4: Row-wise Hessian **H** and quantization loss  $\mathcal{L}$  for each layer

| Layer              | H                                                                                     | $\mathcal L$                                                                                                                                                |
|--------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $W_Q$              | $\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^T\right]$                               | $\operatorname{tr}\left(\mathbb{E}\left[\boldsymbol{K}^T\boldsymbol{K}\right]\cdot\Delta\boldsymbol{W}\mathbf{H}\Delta\boldsymbol{W}^T\right)$              |
| $\boldsymbol{W}_K$ | $\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^T\right]$                               | $\operatorname{tr} \left( \mathbb{E} \left[ \boldsymbol{Q}^T \boldsymbol{Q} \right] \cdot \Delta \boldsymbol{W} \mathbf{H} \Delta \boldsymbol{W}^T \right)$ |
| $\boldsymbol{W}_V$ | $\mathbb{E}\left[\boldsymbol{X}\boldsymbol{A}^T\boldsymbol{A}\boldsymbol{X}^T\right]$ | $\mathrm{tr}\left(\Delta\boldsymbol{W}\mathbf{H}\Delta\boldsymbol{W}^T\right)$                                                                              |
| Others             | $\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^T\right]$                               | $\mathrm{tr}\left(\Delta\boldsymbol{W}\mathbf{H}\Delta\boldsymbol{W}^T\right)$                                                                              |

As mentioned, the proposed *aespa* consists of two main steps; *aespa* first determines the quantization parameters (*i.e.*, scale s and zero-point z in (3)) together with foldable parameters, as in [19, 13, 27, 20] (see line 4 in Algorithm 1), and then optimizes an integer weight  $W_{int}$  for each weight (see lines 5-8 in Algorithm 1). We emphasize that each iteration for the integer weight optimization can be performed efficiently based on pre-computed values (*i.e.*,  $\mathbb{E}[XX^T]$ ,  $\mathbb{E}[XA^TAX^T]$ ,  $\mathbb{E}[K^TK]$ , and  $\mathbb{E}[Q^TQ]$  in Table 4). We also note that while we have used Z-FOLD in computing the quantization parameters and used AdaRound in optimizing integer weights, our refined objectives in (17), (21), and (22) can be integrated into any layer-wise quantization scheme without effort because we only used the definition of the attention operation in our derivation.

## <span id="page-14-1"></span>B Validity of the proposed separate quantization strategy

<span id="page-14-0"></span>

| Method   | Quantization | Reconstruction   | W2A16  |       | W3A16  |       | W4A16  |       |
|----------|--------------|------------------|--------|-------|--------|-------|--------|-------|
|          | Granularity  | Target           | Wiki-2 | C4    | Wiki-2 | C4    | Wiki-2 | C4    |
| AdaRound | Layer-wise   | Layer Output     | 160.7  | 95.63 | 35.44  | 31.86 | 29.51  | 27.78 |
| BRECQ    | Block-wise   | Attention Output | 60.38  | 47.85 | 33.25  | 29.74 | 28.86  | 27.43 |
| Proposed | Layer-wise   | Attention Output | 69.23  | 51.92 | 32.89  | 29.75 | 28.98  | 27.42 |

Table 5: Performance (PPL ↓) of OPT-125M quantized with different strategies.

We conduct experiments to demonstrate the importance of targeting attention-wise reconstruction and the validity of the proposed separate quantization strategy. In our experiments, we learn a weightrounding policy using conventional AdaRound, but we set the loss function for each projection as the attention reconstruction error in [\(7\)](#page-3-3) (not the layer reconstruction error; see Figure [2\(](#page-3-4)c)).

Table [5](#page-14-0) summarizes the quantization performance of AdaRound, BRECQ, and our approach on the OPT-125M model. As evident, our approach uniformly outperforms AdaRound for all bitwidths, although both methods quantize models layer-wise. This is because we can consider the cross-layer dependency (*i.e.,* the relationship between the query, key, and value) by targeting attentionwise reconstruction, which differs from AdaRound wherein layers are considered independent. Furthermore, once we target attention-wise reconstruction, separate layer-wise quantization does not incur severe performance degradation compared to the joint quantization method (BRECQ). Indeed, our approach causes only a marginal performance degradation for 2-bit and exhibits comparable performance for 3-bit and 4-bit.

One might wonder about the strategy of quantizing more than one layer jointly while maintaining remaining weights with full-precision, *e.g.,* simultaneous quantization of the query and key projections while fixing the value projection with full-precision. To say the conclusion first, in this case, we cannot use the proposed pre-computation-based loss computation strategy (see Section [3.4\)](#page-4-7), resulting in a much longer quantization processing time. Specifically, when quantizing W<sup>Q</sup> and W<sup>K</sup> simultaneously, the attention reconstruction error is expressed as

$$\Delta SA_{Q,K} = \mathbb{E}\left[\left\|SA(\widehat{Q},\widehat{K},V) - SA(Q,K,V)\right\|_F^2\right] = \mathbb{E}\left[\left\|\Delta AV\right\|_F^2\right],$$

where

$$\Delta \boldsymbol{A} = \operatorname{softmax} \left( \frac{\widehat{\boldsymbol{Q}} \widehat{\boldsymbol{K}}^T}{\sqrt{d}} \right) - \operatorname{softmax} \left( \frac{\boldsymbol{Q} \boldsymbol{K}^T}{\sqrt{d}} \right)$$

Then, by taking similar steps as in Section [3.3](#page-3-2) (*i.e.,* approximating ∆A with its first-order Taylor series and constructing an upper bound of ∆SAQ,K), we can obtain the following objective:

$$\begin{aligned} \min_{\Delta \boldsymbol{W}_Q, \Delta \boldsymbol{W}_K} \mathbb{E}\left[\left\|\widehat{\boldsymbol{Q}}\widehat{\boldsymbol{K}}^T - \boldsymbol{Q}\boldsymbol{K}^T \right|_F^2 \right] \ &= \min_{\Delta \boldsymbol{W}_Q, \Delta \boldsymbol{W}_K} \mathbb{E}\left[\left\|\boldsymbol{\Delta} \boldsymbol{Q} \boldsymbol{K}^T + \boldsymbol{Q} \Delta \boldsymbol{K}^T + \boldsymbol{\Delta} \boldsymbol{Q} \Delta \boldsymbol{K}^T \right|_F^2 \right] \ &= \min_{\Delta \boldsymbol{W}_Q, \Delta \boldsymbol{W}_K} \mathbb{E}\left[\left\|\boldsymbol{X}^T \Delta \boldsymbol{W}_Q^T \boldsymbol{K}^T + \boldsymbol{Q} \Delta \boldsymbol{W}_K \boldsymbol{X} + \boldsymbol{X}^T \Delta \boldsymbol{W}_Q^T \Delta \boldsymbol{W}_K \boldsymbol{X} \right|_F^2 \right] \end{aligned}$$

Obviously, the objective becomes much more complex than the proposed ones in [\(15\)](#page-4-5) and [\(16\)](#page-4-6), and it would be difficult to simplify and accelerate the loss computation by exploiting pre-computed values as in *aespa*. In fact, without the proposed pre-computation-based loss computation, the simultaneous quantization of W<sup>Q</sup> and W<sup>K</sup> requires 3.5 hours for the quantization of OPT-125M, which is about 44 times longer than the proposed *aespa* and even 1.9 times longer than BRECQ.

## <span id="page-15-0"></span>C Refined quantization objective (16) for the key projection

When quantizing the key projection  $W_K$ , we fix the query and value projections with full-precision. In this case, the attention reconstruction error  $\Delta SA_K$  can be expressed as

$$\Delta SA_K = \mathbb{E}\left[\left\|SA(Q, \widehat{K}, V) - SA(Q, K, V)\right\|_F^2\right] = \mathbb{E}\left[\left\|\Delta AV\right\|_F^2\right],$$

where

<span id="page-15-1"></span>
$$\Delta \boldsymbol{A} = \operatorname{softmax} \left( \frac{\boldsymbol{Q} \widehat{\boldsymbol{K}}^T}{\sqrt{d}} \right) - \operatorname{softmax} \left( \frac{\boldsymbol{Q} \boldsymbol{K}^T}{\sqrt{d}} \right).$$

To avoid the computational overhead of repetitive softmax operation, we approximate  $\Delta A$  with its first-order Taylor series, which leads to

$$\Delta SA_K \approx \frac{1}{d} \mathbb{E} \left[ \left\| \mathbf{Q} \Delta \mathbf{K}^T \mathbf{J}_{\text{softmax}}^T \mathbf{V} \right\|_F^2 \right] = \frac{1}{d} \mathbb{E} \left[ \left\| \mathbf{Q} \Delta \mathbf{W}_K \mathbf{X} \mathbf{J}_{\text{softmax}}^T \mathbf{V} \right\|_F^2 \right]. \tag{25}$$

Furthermore, to reduce the huge memory cost required to store the Jacobian  $\mathbf{J}_{\mathrm{softmax}}$  having  $L^3$  elements (see Footnote 5), we establish an upper bound of (25) and then use it as a surrogate of  $\Delta \mathrm{SA}_K$ . Specifically, we separate the term  $\|\boldsymbol{Q}\Delta\boldsymbol{W}_K\boldsymbol{X}\mathbf{J}_{\mathrm{softmax}}^T\boldsymbol{V}\|_F^2$  into two components as follows:

$$\left\| \boldsymbol{Q} \Delta \boldsymbol{W}_{K} \boldsymbol{X} \boldsymbol{\mathsf{J}}_{\mathrm{softmax}}^{T} \boldsymbol{V} \right\|_{F}^{2} \leq \left\| \boldsymbol{Q} \Delta \boldsymbol{W}_{K} \boldsymbol{X} \right\|_{F}^{2} \cdot \left\| \boldsymbol{\mathsf{J}}_{\mathrm{softmax}}^{T} \boldsymbol{V} \right\|_{F}^{2}.$$

Noting that the term  $\|\mathbf{J}_{\mathrm{softmax}}^T V\|_F^2$  is not affected by the quantization of  $W_K$  and thus fixed in the quantization process, we minimize  $\|Q\Delta W_K X\|_F^2$  to enforce  $\Delta \mathrm{SA}_K$  to be small, which leads to the proposed objective in (16).

## <span id="page-16-0"></span>D Effectiveness of the proposed Hessian in [\(18\)](#page-5-7)

We recall from Section [3.4](#page-4-7) that the proposed quantization objective for the value projection is

$$\operatorname{tr}\left(\Delta \boldsymbol{W}_{V}\mathbb{E}\left[\boldsymbol{X}\boldsymbol{A}^{T}\boldsymbol{A}\boldsymbol{X}^{T}\right]\Delta \boldsymbol{W}_{V}^{T}\right),$$

which implies that the Hessian H<sup>V</sup> for each row of W<sup>V</sup> is

$$\mathbf{H}_V = 2\mathbb{E}[\boldsymbol{X}\boldsymbol{A}^T\boldsymbol{A}\boldsymbol{X}^T].$$

We note that the proposed Hessian H<sup>V</sup> differs from

$$\mathbf{H} = 2\mathbb{E}[\boldsymbol{X}\boldsymbol{X}^T],$$

which has been commonly used as an approximated Hessian in existing methods [\[6,](#page-10-3) [7,](#page-10-5) [3,](#page-10-8) [13\]](#page-10-6). The key reason for the difference is that we consider the dependency between the query, key, and value projections by targeting attention-wise reconstruction, whereas the previous methods assumed independence.

To observe the effect of considering the cross-layer dependency, we use different Hessians (*i.e.,* H<sup>V</sup> and H) when quantizing language models via Z-FOLD and then compare the performance of the quantized models. As Table [6](#page-16-1) shows, the quantization performance is much better when the proposed Hessian H<sup>V</sup> is used, which demonstrates the importance of considering the cross-layer dependency.

Table 6: Quantization performance (PPL ↓) of Z-FOLD under different Hessians.

<span id="page-16-1"></span>

| (a) WikiText-2<br>OPT<br>LLaMA<br>Hessian<br>Precision<br>125M<br>350M<br>1.3B<br>2.7B<br>6.7B<br>7B<br>13B<br>30B<br>E[XXT ] [6, 7, 3, 13]<br>39.59<br>25.97<br>16.10<br>13.54<br>11.65<br>6.756<br>5.708<br>4.931<br>INT3<br>E[XAT AXT ] (ours)<br>35.05<br>24.81<br>16.25<br>13.40<br>11.43<br>6.529<br>5.669<br>4.693<br>E[XXT ] [6, 7, 3, 13]<br>190.1<br>102.5<br>33.97<br>27.10<br>18.07<br>14.93<br>13.03<br>9.250<br>INT2<br>E[XAT AXT ] (ours)<br>146.4<br>68.30<br>31.43<br>25.17<br>17.92<br>14.20<br>8.138<br>13.15<br>(b) PTB<br>OPT<br>LLaMA<br>Hessian<br>Precision<br>125M<br>350M<br>1.3B<br>2.7B<br>6.7B<br>7B<br>13B<br>30B<br>E[XXT ] [6, 7, 3, 13]<br>53.08<br>39.23<br>22.73<br>20.18<br>16.64<br>11.73<br>10.09<br>8.979<br>INT3<br>E[XAT AXT ] (ours)<br>49.88<br>37.62<br>22.66<br>19.78<br>16.55<br>11.39<br>10.48<br>8.657<br>E[XXT ] [6, 7, 3, 13]<br>331.6<br>130.7<br>53.80<br>46.08<br>26.79<br>26.87<br>19.37<br>15.23<br>INT2<br>E[XAT AXT ] (ours)<br>212.8<br>100.1<br>53.64<br>42.93<br>26.09<br>24.88<br>18.01<br>12.99<br>(c) C4<br>OPT<br>LLaMA<br>Hessian<br>Precision<br>125M<br>350M<br>1.3B<br>2.7B<br>6.7B<br>7B<br>13B<br>30B<br>E[XXT ] [6, 7, 3, 13]<br>33.67<br>26.45<br>17.33<br>15.50<br>13.28<br>8.719<br>7.554<br>6.912<br>INT3<br>E[XAT AXT ] (ours)<br>31.27<br>25.51<br>17.27<br>15.42<br>13.22<br>8.313<br>7.437<br>6.638<br>E[XXT ] [6, 7, 3, 13]<br>125.3<br>71.37<br>31.67<br>25.99<br>19.79<br>16.88<br>14.61<br>11.90<br>INT2<br>E[XAT AXT ] (ours)<br>112.6<br>56.48<br>30.06<br>25.34<br>19.32<br>16.87<br>13.46<br>10.32 |  |  |  |  |  |  |  |  |  |  |  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|--|--|--|--|--|--|--|--|
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |  |  |  |  |  |  |  |  |

## <span id="page-17-0"></span>E Proof of (19) and (21)

Note that  $\mathbb{E}\left[\left\|\mathbf{K}\Delta\mathbf{W}_{Q}\mathbf{X}\right\|_{F}^{2}\right] = \mathbb{E}\left[\left\|\operatorname{vec}\left(\mathbf{K}\Delta\mathbf{W}_{Q}\mathbf{X}\right)\right\|_{2}^{2}\right]$ , where  $\operatorname{vec}(\cdot)$  denotes the vectorization operation. Then, by exploiting the following properties of Kronecker product

$$\begin{aligned} \operatorname{vec}\left(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C}\right) &= \left(\boldsymbol{C}^T\otimes \boldsymbol{A}\right)\operatorname{vec}(\boldsymbol{B}), \ \left(\boldsymbol{A}\otimes \boldsymbol{B}\right)^T &= \boldsymbol{A}^T\otimes \boldsymbol{B}^T, \ \left(\boldsymbol{A}\otimes \boldsymbol{B}\right)\left(\boldsymbol{C}\otimes \boldsymbol{D}\right) &= \boldsymbol{A}\boldsymbol{C}\otimes \boldsymbol{B}\boldsymbol{D}, \end{aligned}$$

we have

$$\mathbb{E}\left[\left\|\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\right\|_{F}^{2}\right] = \mathbb{E}\left[\left\|\left(\boldsymbol{X}^{T}\otimes\boldsymbol{K}\right)\Delta\boldsymbol{w}_{Q}\right\|_{2}^{2}\right]$$

$$= \mathbb{E}\left[\Delta\boldsymbol{w}_{Q}^{T}\left(\boldsymbol{X}^{T}\otimes\boldsymbol{K}\right)^{T}\left(\boldsymbol{X}^{T}\otimes\boldsymbol{K}\right)\Delta\boldsymbol{w}_{Q}\right]$$

$$= \mathbb{E}\left[\Delta\boldsymbol{w}_{Q}^{T}\left(\boldsymbol{X}\otimes\boldsymbol{K}^{T}\right)\left(\boldsymbol{X}^{T}\otimes\boldsymbol{K}\right)\Delta\boldsymbol{w}_{Q}\right]$$

$$= \mathbb{E}\left[\Delta\boldsymbol{w}_{Q}^{T}\left(\boldsymbol{X}\boldsymbol{X}^{T}\otimes\boldsymbol{K}^{T}\boldsymbol{K}\right)\Delta\boldsymbol{w}_{Q}\right]$$

$$= \Delta\boldsymbol{w}_{Q}^{T}\cdot\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\otimes\boldsymbol{K}^{T}\boldsymbol{K}\right]\cdot\Delta\boldsymbol{w}_{Q},$$
(26)

which is the desired result in (19).

We now prove (21). By combining (19) and (20), we have

$$\mathbb{E}\left[\left\|\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\right\|_{F}^{2}\right]\approx\Delta\boldsymbol{w}_{Q}^{T}\cdot\left(\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right]\otimes\mathbb{E}\left[\boldsymbol{K}^{T}\boldsymbol{K}\right]\right)\cdot\Delta\boldsymbol{w}_{Q}.$$

Note that since  $\mathbb{E}[XX^T]$  and  $\mathbb{E}[K^TK]$  are symmetric, there exist  $G_X$  and  $G_K$  such that

<span id="page-17-1"></span>
$$\mathbb{E}[\boldsymbol{X}\boldsymbol{X}^T] = \boldsymbol{G}_{\boldsymbol{X}}\boldsymbol{G}_{\boldsymbol{X}}^T, \ \mathbb{E}[\boldsymbol{K}^T\boldsymbol{K}] = \boldsymbol{G}_{\boldsymbol{K}}^T\boldsymbol{G}_{\boldsymbol{K}}.$$

Then, by following the steps used to derive (26) in the reverse order, we have

$$\mathbb{E}\left[\left\|\boldsymbol{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{X}\right\|_{F}^{2}\right] = \Delta\boldsymbol{w}_{Q}^{T}\left(\boldsymbol{G}_{X}\boldsymbol{G}_{X}^{T}\otimes\boldsymbol{G}_{K}^{T}\boldsymbol{G}_{K}\right)\Delta\boldsymbol{w}_{Q}$$

$$= \left\|\boldsymbol{G}_{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{G}_{X}\right\|_{F}^{2}$$

$$= \operatorname{tr}\left(\boldsymbol{G}_{K}\Delta\boldsymbol{W}_{Q}\boldsymbol{G}_{X}\boldsymbol{G}_{X}^{T}\Delta\boldsymbol{W}_{Q}^{T}\boldsymbol{G}_{K}^{T}\right)$$

$$= \operatorname{tr}\left(\boldsymbol{G}_{K}^{T}\boldsymbol{G}_{K}\cdot\Delta\boldsymbol{W}_{Q}\cdot\boldsymbol{G}_{X}\boldsymbol{G}_{X}^{T}\cdot\Delta\boldsymbol{W}_{Q}^{T}\right)$$

$$= \operatorname{tr}\left(\mathbb{E}\left[\boldsymbol{K}^{T}\boldsymbol{K}\right]\Delta\boldsymbol{W}_{Q}\mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right]\Delta\boldsymbol{W}_{Q}^{T}\right),$$

which completes the proof of (21).

## <span id="page-18-0"></span>F Integration of proposed loss functions into existing PTQ schemes

We recall that we only utilized the definition of the attention operation when developing the proposed loss functions for the attention output reconstruction. Therefore, our loss functions can be integrated into any PTQ schemes based on layer-wise reconstruction and used to enhance their performance. In this section, we describe how to combine our loss functions with existing quantization schemes by taking AdaRound as an example.

In short, AdaRound learns a weight-rounding mechanism by solving the following optimization problem [\[23\]](#page-11-2):

$$\arg\min_{\boldsymbol{B}} \|\boldsymbol{W}\boldsymbol{X} - \widetilde{\boldsymbol{W}}\boldsymbol{X}\|_{F}^{2} + \lambda \sum_{i,j} \left(1 - |2h(\boldsymbol{B}_{i,j}) - 1|^{\beta}\right), \tag{27}$$

where <sup>B</sup> is the continuous variable to be learned, <sup>h</sup> is the rectified sigmoid function, and <sup>W</sup><sup>f</sup> is the soft-quantized weights defined as

<span id="page-18-2"></span><span id="page-18-1"></span>
$$\widetilde{\bm{W}} = s \cdot \mathrm{clamp}\left(\left\lfloor \frac{\bm{W}}{s} \right\rfloor + h(\bm{B}), n, p \right).$$

One can see that the loss function of AdaRound consists of two components, layer-wise reconstruction error and weight-rounding loss.

To consider the cross-layer dependency between WQ, WK, and W<sup>V</sup> in the learning process, we integrate the proposed loss functions developed for the attention output reconstruction into [\(27\)](#page-18-2). In other words, we replace the layer-wise reconstruction error in [\(27\)](#page-18-2) with our loss functions in [\(17\)](#page-5-2), [\(21\)](#page-5-5), and [\(22\)](#page-5-6). For example, when learning the rounding policy for the query projection matrix WQ, the objective of the proposed *aespa* is expressed as

$$\arg \min_{\boldsymbol{B}_{Q}} \operatorname{tr}\left(\mathbb{E}\left[\boldsymbol{K}^{T}\boldsymbol{K}\right] \Delta \boldsymbol{W}_{Q} \mathbb{E}\left[\boldsymbol{X}\boldsymbol{X}^{T}\right] \Delta \boldsymbol{W}_{Q}^{T}\right) + \lambda \sum_{i,j} \left(1 - \left|2h(\boldsymbol{B}_{Q,i,j}) - 1\right|^{\beta}\right), \quad (28)$$

where <sup>∆</sup>W<sup>Q</sup> <sup>=</sup> <sup>W</sup><sup>Q</sup> <sup>−</sup> <sup>W</sup>fQ.

## <span id="page-19-0"></span>G Complexity analysis for conventional block-wise quantization schemes

Recall from (7) that conventional block-wise quantization schemes require to compute  $SA(\widehat{Q}, \widehat{K}, \widehat{V})$  in each iteration. This means that for each input sequence, one needs to perform

- forward pass for  $\widehat{Q}$ ,  $\widehat{K}$ , and  $\widehat{V}$ :  $3d_hL(2d-1)$  flops
- matrix multiplications for computing  $\widehat{\bm{Q}}\widehat{\bm{K}}^T$  and  $\widehat{\bm{A}}\widehat{\bm{V}}$ :  $4d_hL^2-d_hL-L^2$  flops
- softmax operation with additional scaling (i.e., softmax( $\frac{\widehat{Q}\widehat{K}^{T}}{\sqrt{d_h}}$ )):  $3L^2 + d_hL L$  flops
- final computation of reconstruction error:  $3d_hL 1$  flops

If B input sequences are used in each quantization iteration, then the total number of flops required in conventional methods is

$$C_{exist} = B(6d_h dL + 4d_h L^2 + 2L^2 - L - 1) = \mathcal{O}(Bd_h L \cdot \max\{d, L\}).$$

Comparison of  $\mathcal{C}_{aespa}$  and  $\mathcal{C}_{exist}$  We now compare the complexities of aespa and conventional block-wise quantization methods in terms of the number of flops. Table 7 summarizes the computational costs required to quantize different sizes of OPT models. For conventional methods, we report the cost of using four sequences in each iteration (B=4). We observe that the computational cost of aespa is considerably lower than that of conventional methods. In particular, for small-scale models (e.g., OPT-125M, OPT-350M, and OPT-1.3B), aespa performs ten times fewer number of flops. One can notice that the gap between  $\mathcal{C}_{aespa}$  and  $\mathcal{C}_{exist}$  decreases as the model size increases. This is because the hidden size d exceeds the sequence length L (which is fixed for all models) as the model size increases. Nevertheless, aespa still incurs a lower computational cost, and the gap increases if conventional methods use larger batch sizes.

<span id="page-19-1"></span>Table 7: Cost of *aespa* and conventional methods (GFLOPS)

|                       | The state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the state of the s |      |      |      |      |     |  |  |  |  |  |  |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|------|------|------|-----|--|--|--|--|--|--|
|                       | 125M                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 350M | 1.3B | 2.7B | 6.7B | 13B |  |  |  |  |  |  |
| $\mathcal{C}_{exist}$ | 6.7                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | 7.5  | 11   | 15   | 34   | 41  |  |  |  |  |  |  |
| $C_{aespa}$           | 0.24                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 0.42 | 1.6  | 3.2  | 13   | 20  |  |  |  |  |  |  |

## <span id="page-20-0"></span>H Comparison with block-wise PTQ schemes

We provide experimental results excluded from the main text due to page limitations.

Table 8: Performance (PPL ↓) of the proposed *aespa* and conventional block-wise PTQ methods.

## (a) INT4 performance on WikiText-2 and C4

|            |                                                           |                                  |                                  | OPT                              |                                |                                | LLaMA                          |                                | LLaMA2                         |                                |
|------------|-----------------------------------------------------------|----------------------------------|----------------------------------|----------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| Dataset    | Method                                                    | 125M                             | 1.3B                             | 2.7B                             | 6.7B                           | 7B                             | 13B                            | 30B                            | 7B                             | 13B                            |
|            | FP16                                                      | 27.65                            | 14.63                            | 12.47                            | 10.86                          | 5.677                          | 5.091                          | 4.101                          | 5.472                          | 4.884                          |
| WikiText-2 | BRECQ [18]<br>OmniQuant [27]<br>AffineQuant [20]<br>aespa | 28.86<br>30.42<br>29.81<br>28.87 | 14.83<br>15.15<br>15.09<br>14.81 | 12.71<br>12.89<br>12.72<br>12.36 | OOM<br>11.20<br>11.12<br>10.95 | OOM<br>5.907<br>5.905<br>5.890 | OOM<br>5.256<br>5.256<br>5.226 | OOM<br>4.263<br>4.269<br>4.254 | OOM<br>5.850<br>5.782<br>5.684 | OOM<br>5.064<br>5.062<br>5.031 |
|            | FP16                                                      | 26.56                            | 16.07                            | 14.34                            | 12.71                          | 7.344                          | 6.798                          | 6.131                          | 7.264                          | 6.727                          |
| C4         | BRECQ [18]<br>OmniQuant [27]<br>AffineQuant [20]<br>aespa | 27.43<br>28.41<br>28.04<br>27.24 | 16.42<br>16.68<br>16.58<br>16.31 | 14.61<br>14.83<br>14.74<br>14.55 | OOM<br>12.99<br>12.92<br>12.82 | OOM<br>7.656<br>7.654<br>7.633 | OOM<br>6.976<br>6.974<br>6.945 | OOM<br>6.269<br>6.270<br>6.256 | OOM<br>7.686<br>7.644<br>7.508 | OOM<br>6.956<br>6.927<br>6.891 |

#### (b) Performance on PTB

| Precision | Method           |       |       | OPT   |       |       | LLaMA |       |  |  |
|-----------|------------------|-------|-------|-------|-------|-------|-------|-------|--|--|
|           |                  | 125M  | 1.3B  | 2.7B  | 6.7B  | 7B    | 13B   | 30B   |  |  |
| FP16      | Baseline         | 38.99 | 20.29 | 17.97 | 15.77 | 10.12 | 9.081 | 8.159 |  |  |
|           | BRECQ [18]       | 41.04 | 20.97 | 18.41 | OOM   | OOM   | OOM   | OOM   |  |  |
|           | OmniQuant [27]   | 42.34 | 21.32 | 18.70 | 16.04 | 10.57 | 9.330 | 8.354 |  |  |
| INT4      | AffineQuant [20] | 42.99 | 21.26 | 18.49 | 16.02 | 10.53 | 9.325 | 8.355 |  |  |
|           | aespa            | 40.50 | 20.78 | 18.30 | 15.84 | 10.43 | 9.277 | 8.283 |  |  |
|           | BRECQ [18]       | 46.93 | 23.41 | 19.82 | OOM   | OOM   | OOM   | OOM   |  |  |
|           | OmniQuant [27]   | 56.88 | 25.11 | 22.59 | 18.33 | 11.98 | 10.24 | 9.065 |  |  |
| INT3      | AffineQuant [20] | 51.47 | 24.38 | 21.03 | 17.40 | 11.92 | 10.24 | 8.998 |  |  |
|           | aespa            | 44.96 | 22.35 | 19.48 | 16.28 | 11.45 | 9.818 | 8.684 |  |  |
|           | BRECQ [18]       | 90.22 | 344.9 | 282.0 | OOM   | OOM   | OOM   | OOM   |  |  |
| INT2      | OmniQuant [27]   | NaN   | 377.9 | 2.0e3 | 7.7e3 | 33.51 | NaN   | 17.38 |  |  |
|           | AffineQuant [20] | 177.8 | 75.25 | 47.07 | 37.90 | 29.33 | 18.58 | NaN   |  |  |
|           | aespa            | 99.12 | 37.19 | 32.57 | 22.80 | 19.83 | 15.65 | 12.98 |  |  |

<sup>\*</sup> 'NaN' means that loss diverges in the quantization process.

<sup>\*</sup> 'OOM' means that out-of-memory issues occur when quantizing models with a single A100 GPU.

## <span id="page-21-0"></span>I Comparison with layer-wise PTQ schemes

We provide experimental results excluded from the main text due to page limitations.

## I.1 Results on OPT models

<span id="page-21-1"></span>Table 9: Performance (PPL ↓) of *aespa* and existing layer-wise PTQ methods on OPT models.

|           |                 |                |                | (a) WikiText-2 |                |                |                |                |
|-----------|-----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Precision | Method          | 125M           | 350M           | 1.3B           | 2.7B           | 6.7B           | 13B            | 30B            |
| FP16      | Baseline        | 27.65          | 22.00          | 14.63          | 12.47          | 10.86          | 10.13          | 9.56           |
|           | RTN             | 37.28          | 25.94          | 48.20          | 16.92          | 12.10          | 11.32          | 10.98          |
| INT4      | OPTQ            | 32.49          | 23.68          | 15.50          | 12.85          | 11.12          | 10.33          | 9.670          |
|           | Z-FOLD<br>aespa | 31.03<br>28.87 | 23.08<br>22.55 | 15.00<br>14.81 | 12.47<br>12.36 | 11.01<br>10.95 | 10.21<br>10.18 | 9.537<br>9.511 |
|           | RTN             | 1.3e3          | 64.57          | 1.3e4          | 1.6e4          | 5.8e3          | 3.4e3          | 1.6e3          |
|           | OPTQ            | 52.95          | 33.29          | 20.36          | 16.94          | 13.01          | 11.65          | 10.44          |
| INT3      | Z-FOLD          | 39.59          | 25.97          | 16.10          | 13.54          | 11.65          | 10.62          | 9.902          |
|           | aespa           | 32.71          | 24.45          | 15.79          | 13.14          | 11.23          | 10.52          | 9.760          |
|           | RTN             | 5.5e3          | 2.8e4          | 1.1e5          | 9.5e3          | 2.8e4          | 1.9e5          | 1.7e5          |
| INT2      | OPTQ            | 4.1e3          | 1.1e4          | 8.3e3          | 9.3e3          | 2.0e3          | 539.8          | 56.63          |
|           | Z-FOLD<br>aespa | 190.1<br>71.18 | 102.5<br>54.89 | 33.97<br>24.26 | 27.10<br>22.22 | 18.07<br>15.71 | 33.48<br>15.27 | 13.48<br>11.91 |
|           |                 |                |                |                |                |                |                |                |
|           |                 |                |                | (b) PTB        |                |                |                |                |
| Precision | Method          | 125M           | 350M           | 1.3B           | 2.7B           | 6.7B           | 13B            | 30B            |
| FP16      | Baseline        | 38.99          | 31.08          | 20.29          | 17.97          | 15.77          | 14.52          | 14.04          |
|           | RTN             | 53.88          | 36.79          | 75.37          | 32.41          | 18.86          | 16.41          | 15.44          |
| INT4      | OPTQ            | 46.54          | 33.27          | 21.74          | 19.04          | 16.42          | 14.88          | 14.21          |
|           | Z-FOLD<br>aespa | 44.17<br>40.50 | 33.51<br>32.17 | 20.96<br>20.78 | 18.45<br>18.30 | 15.98<br>15.84 | 14.65<br>14.65 | 14.11<br>14.09 |
|           |                 |                |                |                |                |                |                |                |
|           | RTN<br>OPTQ     | 1.4e3<br>74.07 | 87.21<br>46.10 | 1.5e4<br>29.76 | 1.4e4<br>25.06 | 5.3e3<br>19.22 | 2.2e3<br>16.42 | 1.5e3<br>15.08 |
| INT3      | Z-FOLD          | 53.08          | 39.23          | 22.73          | 20.18          | 16.64          | 15.23          | 14.60          |
|           | aespa           | 44.96          | 36.15          | 22.35          | 19.48          | 16.28          | 15.06          | 14.43          |
|           | RTN             | 4.3e3          | 2.8e4          | 1.1e4          | 6.8e3          | 1.8e4          | 1.2e5          | 1.7e5          |
| INT2      | OPTQ            | 3.5e3          | 1.2e4          | 6.6e3          | 8.0e3          | 2.5e3          | 458.4          | 83.81          |
|           | Z-FOLD          | 331.6          | 130.7          | 53.80          | 46.08          | 26.79          | 79.69          | 20.39          |
|           | aespa           | 99.12          | 79.86          | 37.19          | 32.57          | 22.80          | 23.93          | 17.51          |
|           |                 |                |                | (c) C4         |                |                |                |                |
| Precision | Method          | 125M           | 350M           | 1.3B           | 2.7B           | 6.7B           | 13B            | 30B            |
| FP16      | Baseline        | 26.56          | 22.59          | 16.07          | 14.34          | 12.71          | 12.06          | 11.44          |
|           | RTN             | 33.88          | 26.21          | 27.50          | 18.83          | 14.37          | 13.32          | 13.55          |
| INT4      | OPTQ            | 29.64          | 24.15          | 16.75          | 14.86          | 13.00          | 12.24          | 11.56          |
|           | Z-FOLD<br>aespa | 28.92<br>27.24 | 23.71<br>23.15 | 16.38<br>16.31 | 14.60<br>14.55 | 12.85<br>12.82 | 12.14<br>12.13 | 11.49<br>11.47 |
|           |                 |                |                |                |                |                |                |                |
|           | RTN<br>OPTQ     | 834.4<br>42.88 | 55.15<br>30.60 | 6.6e3<br>20.53 | 1.2e4<br>17.66 | 5.0e3<br>14.61 | 2.8e3<br>13.19 | 1.8e3<br>12.15 |
| INT3      | Z-FOLD          | 33.67          | 26.45          | 17.33          | 15.50          | 13.28          | 12.45          | 11.73          |
|           | aespa           | 29.51          | 24.96          | 17.10          | 15.27          | 13.15          | 12.39          | 11.68          |
|           | RTN             | 3.7e3          | 1.6e4          | 7.7e3          | 7.7e3          | 1.4e4          | 9.7e4          | 5.8e4          |
| INT2      | OPTQ            | 2.1e3          | 4.4e3          | 3.0e3          | 3.7e3          | 290.9          | 157.7          | 29.40          |
|           | Z-FOLD          | 125.3          | 71.37          | 31.67          | 25.98          | 19.79          | 47.10          | 14.51          |
|           | aespa           | 56.88          | 46.36          | 23.54          | 22.53          | 17.28          | 16.30          | 13.32          |

## I.2 Results on BLOOM models

Table 10: Performance (PPL ↓) of *aespa* and existing layer-wise PTQ methods on BLOOM models.

(a) INT4 performance on WikiText-2 and C4

|           |                                | WikiText-2                       |                                  |                                  |                                  | C4                               |  |                                  |                                  |                                  |                                  |                                  |
|-----------|--------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|--|----------------------------------|----------------------------------|----------------------------------|----------------------------------|----------------------------------|
| Precision | Method                         | 560M                             | 1.1B                             | 1.7B                             | 3B                               | 7.1B                             |  | 560M                             | 1.1B                             | 1.7B                             | 3B                               | 7.1B                             |
| FP16      | Baseline                       | 22.42                            | 17.69                            | 15.39                            | 13.48                            | 11.37                            |  | 26.60                            | 22.05                            | 19.49                            | 17.49                            | 15.20                            |
| INT4      | RTN<br>OPTQ<br>Z-FOLD<br>aespa | 25.82<br>23.83<br>23.60<br>23.21 | 19.98<br>18.74<br>18.44<br>18.28 | 16.96<br>16.16<br>15.87<br>15.76 | 14.75<br>14.01<br>13.90<br>13.81 | 12.09<br>11.72<br>11.59<br>11.56 |  | 29.80<br>27.74<br>27.36<br>27.20 | 24.42<br>23.05<br>22.66<br>22.49 | 21.24<br>20.26<br>20.00<br>19.86 | 18.75<br>18.00<br>17.87<br>17.76 | 16.05<br>15.54<br>15.42<br>15.38 |

|           | (b) Performance on PTB |       |       |       |       |       |  |  |  |  |  |
|-----------|------------------------|-------|-------|-------|-------|-------|--|--|--|--|--|
| Precision | Method                 | 560M  | 1.1B  | 1.7B  | 3B    | 7.1B  |  |  |  |  |  |
| FP16      | Baseline               | 43.69 | 57.96 | 30.00 | 25.34 | 20.83 |  |  |  |  |  |
| INT4      | RTN                    | 50.96 | 66.79 | 33.52 | 27.65 | 22.40 |  |  |  |  |  |
|           | OPTQ                   | 46.83 | 62.99 | 31.63 | 26.72 | 21.52 |  |  |  |  |  |
|           | Z-FOLD                 | 45.77 | 61.33 | 31.26 | 26.27 | 21.35 |  |  |  |  |  |
|           | aespa                  | 44.73 | 60.41 | 31.05 | 26.01 | 21.17 |  |  |  |  |  |
| INT3      | RTN                    | 124.8 | 184.0 | 105.5 | 66.24 | 34.94 |  |  |  |  |  |
|           | OPTQ                   | 64.43 | 82.91 | 40.27 | 33.13 | 25.94 |  |  |  |  |  |
|           | Z-FOLD                 | 53.01 | 69.93 | 35.12 | 28.41 | 22.83 |  |  |  |  |  |
|           | aespa                  | 48.87 | 67.01 | 33.06 | 27.61 | 22.03 |  |  |  |  |  |
| INT2      | RTN                    | 7.4e5 | 1.1e6 | 2.5e5 | 1.2e5 | 2.2e5 |  |  |  |  |  |
|           | OPTQ                   | 4.1e3 | 2.4e3 | 1.4e3 | 1.4e3 | 428.4 |  |  |  |  |  |
|           | Z-FOLD                 | 194.9 | 174.9 | 74.03 | 69.49 | 38.50 |  |  |  |  |  |
|           | aespa                  | 91.14 | 120.7 | 57.48 | 46.40 | 31.28 |  |  |  |  |  |

## I.3 Results on LLaMA models

Table 11: Performance (PPL ↓) of *aespa* and existing layer-wise PTQ methods on LLaMA models.

|           |          |       | WikiText-2 |       |       | PTB   |       |       | C4    |       |  |  |
|-----------|----------|-------|------------|-------|-------|-------|-------|-------|-------|-------|--|--|
| Precision | Method   | 7B    | 13B        | 30B   | 7B    | 13B   | 30B   | 7B    | 13B   | 30B   |  |  |
| FP16      | Baseline | 5.677 | 5.091      | 4.101 | 10.12 | 9.081 | 8.159 | 7.344 | 6.798 | 6.131 |  |  |
|           | RTN      | 6.291 | 5.525      | 4.536 | 11.25 | 9.775 | 8.653 | 8.121 | 7.232 | 6.537 |  |  |
|           | OPTQ     | 6.167 | 5.365      | 4.452 | 11.51 | 9.526 | 8.426 | 7.792 | 7.082 | 6.399 |  |  |
| INT4      | Z-FOLD   | 6.069 | 5.278      | 4.325 | 11.45 | 9.335 | 8.410 | 7.797 | 6.984 | 6.318 |  |  |
|           | aespa    | 5.890 | 5.226      | 4.254 | 10.43 | 9.277 | 8.283 | 7.633 | 6.945 | 6.256 |  |  |
|           | RTN      | 25.61 | 11.78      | 14.87 | 98.89 | 28.94 | 28.79 | 30.86 | 14.46 | 30.04 |  |  |
|           | OPTQ     | 8.290 | 6.729      | 5.705 | 16.11 | 11.91 | 9.964 | 10.51 | 8.832 | 7.977 |  |  |
| INT3      | Z-FOLD   | 6.756 | 5.708      | 4.931 | 11.73 | 10.09 | 8.979 | 8.719 | 7.554 | 6.912 |  |  |
|           | aespa    | 6.579 | 5.611      | 4.688 | 11.45 | 9.818 | 8.684 | 8.465 | 7.399 | 6.634 |  |  |
|           | RTN      | 1.1e5 | 5.7e4      | 2.7e4 | 9.9e4 | 8.1e4 | 3.3e4 | 1.1e5 | 5.9e4 | 2.8e4 |  |  |
|           | OPTQ     | 1.0e4 | 3.7e3      | 1.5e3 | 1.1e4 | 8.5e3 | 1.0e3 | 872.7 | 809.7 | 304.4 |  |  |
| INT2      | Z-FOLD   | 14.93 | 13.03      | 9.250 | 26.87 | 19.37 | 15.23 | 16.88 | 14.61 | 11.90 |  |  |
|           | aespa    | 11.94 | 10.30      | 7.845 | 19.83 | 15.65 | 12.98 | 13.63 | 11.46 | 10.35 |  |  |

## I.4 Results on LLaMA2 models

<span id="page-23-0"></span>Table 12: Performance (PPL ↓) of *aespa* and existing layer-wise PTQ methods on LLaMA2 models.

|           |          |       | WikiText-2 |       | C4    |
|-----------|----------|-------|------------|-------|-------|
| Precision | Method   | 7B    | 13B        | 7B    | 13B   |
| FP16      | Baseline | 5.472 | 4.884      | 7.264 | 6.727 |
| INT4      | RTN      | 6.116 | 5.205      | 8.165 | 7.142 |
|           | OPTQ     | 6.060 | 5.158      | 7.838 | 7.030 |
|           | Z-FOLD   | 5.815 | 5.099      | 7.602 | 6.996 |
|           | aespa    | 5.684 | 5.031      | 7.508 | 6.891 |
| INT3      | RTN      | 542.0 | 10.69      | 527.2 | 13.87 |
|           | OPTQ     | 8.664 | 6.554      | 11.24 | 8.761 |
|           | Z-FOLD   | 6.606 | 5.710      | 8.666 | 7.692 |
|           | aespa    | 6.241 | 5.462      | 8.225 | 7.392 |
| INT2      | RTN      | 1.8e4 | 5.1e4      | 2.8e4 | 5.3e4 |
|           | OPTQ     | 7.5e3 | 2.1e3      | 1.7e3 | 560.7 |
|           | Z-FOLD   | 20.79 | 15.56      | 21.98 | 16.90 |
|           | aespa    | 13.99 | 12.14      | 14.36 | 13.59 |

## <span id="page-24-1"></span>J Results for zero-shot tasks

<span id="page-24-0"></span>We provide INT3 zero-shot performance results that are excluded from the main text due to page limitations.

Table 13: INT3 zero-shot performance (accuracy ↑) of *aespa* and existing methods.

| Model      | Method           | ARC-c | ARC-e | HellaSwag | MMLU  | Average |
|------------|------------------|-------|-------|-----------|-------|---------|
|            | FP16             | 44.62 | 72.85 | 76.18     | 32.19 | 56.46   |
| LLaMA-7B   | RTN              | 27.47 | 45.45 | 45.46     | 24.94 | 35.83   |
|            | OPTQ [7]         | 36.95 | 62.63 | 68.33     | 25.51 | 48.36   |
|            | Z-FOLD [13]      | 41.21 | 66.92 | 72.50     | 28.90 | 52.38   |
|            | OmniQuant [27]   | 38.99 | 67.30 | 70.31     | 29.33 | 51.48   |
|            | AffineQuant [20] | 39.25 | 65.61 | 70.56     | 29.68 | 51.28   |
|            | aespa            | 40.87 | 69.15 | 71.54     | 30.57 | 53.03   |
|            | FP16             | 47.87 | 74.75 | 79.08     | 43.46 | 61.29   |
|            | RTN              | 36.09 | 56.23 | 62.03     | 26.20 | 45.14   |
|            | OPTQ [7]         | 43.00 | 67.89 | 72.45     | 28.62 | 52.99   |
| LLaMA-13B  | Z-FOLD [13]      | 44.88 | 71.00 | 75.66     | 36.88 | 57.11   |
|            | OmniQuant [27]   | 44.03 | 69.70 | 75.15     | 35.89 | 56.19   |
|            | AffineQuant [20] | 43.60 | 70.24 | 75.10     | 32.67 | 55.40   |
|            | aespa            | 45.82 | 71.80 | 75.87     | 38.63 | 58.03   |
|            | FP16             | 52.90 | 78.96 | 82.63     | 54.66 | 67.29   |
|            | RTN              | 27.90 | 43.64 | 31.42     | 23.34 | 31.58   |
|            | OPTQ [7]         | 45.31 | 71.55 | 77.17     | 42.01 | 59.01   |
| LLaMA-30B  | Z-FOLD [13]      | 50.34 | 75.84 | 79.69     | 51.00 | 64.22   |
|            | OmniQuant [27]   | 49.49 | 76.52 | 79.76     | 50.68 | 64.11   |
|            | AffineQuant [20] | 49.66 | 77.10 | 79.49     | 50.37 | 64.16   |
|            | aespa            | 50.34 | 77.53 | 79.79     | 50.55 | 64.55   |
|            | FP16             | 46.16 | 74.49 | 75.99     | 41.87 | 59.63   |
|            | RTN              | 25.94 | 35.48 | 35.39     | 23.14 | 29.99   |
|            | OPTQ [7]         | 37.46 | 63.01 | 64.85     | 28.79 | 48.53   |
| LLaMA2-7B  | Z-FOLD [13]      | 40.10 | 64.65 | 69.92     | 33.69 | 52.09   |
|            | OmniQuant [27]   | 40.36 | 67.30 | 71.00     | 31.26 | 52.48   |
|            | AffineQuant [20] | 40.78 | 67.21 | 70.75     | 30.93 | 52.42   |
|            | aespa            | 41.38 | 69.11 | 71.78     | 38.18 | 55.11   |
|            | FP16             | 49.06 | 77.44 | 79.39     | 52.10 | 64.50   |
|            | RTN              | 34.56 | 55.98 | 59.44     | 25.45 | 43.86   |
|            | OPTQ [7]         | 43.09 | 70.45 | 72.02     | 39.37 | 56.23   |
| LLaMA2-13B | Z-FOLD [13]      | 46.42 | 72.77 | 74.79     | 47.91 | 60.47   |
|            | OmniQuant [27]   | 45.65 | 74.33 | 74.77     | 43.92 | 59.67   |
|            | AffineQuant [20] | 47.18 | 75.42 | 75.28     | 45.61 | 60.87   |
|            | aespa            | 46.84 | 75.25 | 75.78     | 47.09 | 61.24   |

## <span id="page-25-1"></span><span id="page-25-0"></span>K Time and memory cost comparison

Table 14: Time and memory cost of *aespa* and existing methods

(a) INT2 quantization processing time

|                |                  |           | OPT      |          |          | LLaMA    |          |          |
|----------------|------------------|-----------|----------|----------|----------|----------|----------|----------|
| Target         | Method           | 125M      | 1.3B     | 2.7B     | 6.7B     | 7B       | 13B      | 30B      |
| layer-wise     | OPTQ [7]         | 0.66 min  | 0.08 hr  | 0.14 hr  | 0.29 hr  | 0.25 hr  | 0.45 hr  | 1.08 hr  |
| reconstruction | Z-FOLD [13]      | 1.09 min  | 0.27 hr  | 0.61 hr  | 2.58 hr  | 1.13 hr  | 2.48 hr  | 10.51 hr |
|                | BRECQ [18]       | 108.2 min | 10.71 hr | 19.15 hr | OOM      | OOM      | OOM      | OOM      |
| attention-wise | OmniQuant [27]   | 16.20 min | 1.02 hr  | 1.63 hr  | 2.93 hr  | 2.37 hr  | 4.20 hr  | 9.84 hr  |
| reconstruction | AffineQuant [20] | 28.33 min | 2.57 hr  | 4.60 hr  | 9.85 hr  | 10.09 hr | 18.76 hr | 47.84 hr |
|                | aespa            | 4.78 min  | 1.24 hr  | 2.83 hr  | 10.24 hr | 6.84 hr  | 15.89 hr | 53.69 hr |

(b) Memory cost (GB)

|                                  |                                                           |                              |                               | OPT                            |                                | LLaMA                          |                                |                                |
|----------------------------------|-----------------------------------------------------------|------------------------------|-------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| Target                           | Method                                                    | 125M                         | 1.3B                          | 2.7B                           | 6.7B                           | 7B                             | 13B                            | 30B                            |
| layer-wise<br>reconstruction     | OPTQ [7]<br>Z-FOLD [13]                                   | 1.39<br>1.39                 | 4.49<br>4.49                  | 6.43<br>6.43                   | 13.07<br>13.07                 | 8.76<br>8.76                   | 12.34<br>12.34                 | 18.59<br>18.59                 |
| attention-wise<br>reconstruction | BRECQ [18]<br>OmniQuant [27]<br>AffineQuant [20]<br>aespa | 3.39<br>1.94<br>3.47<br>1.68 | 16.60<br>5.87<br>9.96<br>5.47 | 27.79<br>7.09<br>12.25<br>6.84 | OOM<br>11.68<br>20.08<br>12.26 | OOM<br>12.61<br>24.28<br>21.69 | OOM<br>17.02<br>27.10<br>29.27 | OOM<br>24.53<br>38.59<br>43.00 |

<sup>\*</sup> 'OOM' means that out-of-memory issues occur when quantizing models with a single NVIDIA A100 GPU.

Table [14](#page-25-1) summarizes the processing time and memory cost of different quantization algorithms. We note that the processing time of the proposed *aespa* includes the time required for pre-computations (lines 2-4 in Algorithm [1\)](#page-13-0).

As expected, *aespa* completes quantization much faster than BRECQ. For example, while BRECQ requires more than 10 hours to quantize OPT-1.3B, *aespa* completes quantization in 1.24 hours, which demonstrates the effectiveness of the proposed objectives and pre-computation-based loss computation strategy. Although other block-wise PTQ methods (OmniQuant/AffineQuant) perform quantization faster than *aespa* for hyper-scale models, they suffer from unstable training process or exhibit poor PPL performance (*e.g.,* PPL of OmniQuant is larger than 10<sup>3</sup> for OPT-6.7B; see Table [1\)](#page-7-0). We also observe that OPTQ performs quantization very fast, but its PPL performance collapses completely regardless of the model size (see Table [9\)](#page-21-1). Except *aespa*, Z-FOLD is the only method that shows both reasonable performance and processing time.

In real situations, when one needs to preserve the performance of the original model as much as possible, the proposed *aespa* would be an intriguing solution. In particular, when deploying LLMs on resource-constrained platforms where up to 7B models are commonly employed (*e.g.,* mobile devices), *aespa* would be a good fit. Even when fast quantization of hyper-scale models is needed, *aespa* can be used with a slight modification. Specifically, in time-limited cases, one can skip weight-rounding optimization (lines 5-8 in Algorithm [1\)](#page-13-0) and simply perform the quantization parameter computation (line 4 in Algorithm [1\)](#page-13-0) using the proposed Hessian that considers the cross-layer dependency (see [\(18\)](#page-5-7)). In doing so, we can not only save the time required to perform weight-rounding learning, but also save the memory required to store pre-computed values (E[K<sup>T</sup> K] and E[Q<sup>T</sup> Q]). Indeed, when performing only quantization parameter computation, we achieved a significant reduction in the processing time (see Table [15](#page-25-2) below) while still exhibiting better performance than conventional methods (see Table [6](#page-16-1) in Appendix [D\)](#page-16-0).

<span id="page-25-2"></span>Table 15: INT2 quantization processing time of *aespa* without weight-rounding optimization

|          | OPT     |         |         | LLaMA   |         |          |
|----------|---------|---------|---------|---------|---------|----------|
| 125M     | 1.3B    | 2.7B    | 6.7B    | 7B      | 13B     | 30B      |
| 1.29 min | 0.35 hr | 0.74 hr | 2.92 hr | 1.47 hr | 3.26 hr | 12.50 hr |

## <span id="page-26-0"></span>L Experimental results for different calibration datasets

One might wonder why the PPL performances of OmniQuant summarized in Table 1 are much worse than those reported in the original paper [27]; INT2 PPL performances of quantized LLaMA models are 18.18, NaN, and 10.15 for WikiText-2 in Table 1, which are worse than the values (15.47, 13.21, and 8.71) reported in [27]. This is because we used a different calibration dataset for quantization. Specifically, we used C4 when constructing a calibration dataset, while [27] used WikiText-2.

Additionally, we evaluate the performance of the proposed *aespa* using WikiText-2 as a calibration dataset. From Table 16, we observe that when calibration data are sampled from WikiText-2, our results for OmniQuant are comparable with those reported in the original paper [27]. While it has been reported that the performance variance of OmniQuant across different calibration datasets is low for INT3 and INT4 (see [27, Table A10]), such low variance does not hold for INT2. Furthermore, we observe that the proposed *aespa* outperforms OmniQuant regardless of the type of the calibration dataset.

<span id="page-26-1"></span>Table 16: INT2 performances (PPL ↓) of aespa and OmniQuant for different calibration datasets

| Calibration<br>Dataset | Method    |       | LLaMA<br>13B | 30B   |
|------------------------|-----------|-------|--------------|-------|
| C4                     | OmniQuant | 18.18 | NaN          | 10.15 |
| C4                     | aespa     | 11.94 | 10.30        | 7.845 |
| WikiText-2             | OmniQuant | 15.59 | 13.76        | 9.230 |
| WINITEXU-2             | aespa     | 8.818 | 7.423        | 6.232 |

<sup>\*</sup> Test dataset: WikiText-2

## M Quantization performance of aespa for high bit-widths

While previous results demonstrate that the proposed *aespa* is very competitive for low-bit quantization (*e.g.*, INT2 and INT3), one might wonder whether *aespa* can preserve the performance of the original full-precision model at high bit-widths. We thus evaluate INT4 and INT6 quantization performances of *aespa* with LLaMA models. From Table 17, we observe that *aespa* almost preserves the performance of the original full-precision model for the INT6 quantization. Even for the INT4 quantization, the performance degradation is very marginal (*e.g.*, less than 1% degradation for 13B and 30B models).

<span id="page-26-2"></span>Table 17: INT4 and INT6 quantization performances of the proposed aespa (calibration data: C4)

| Model     | Precision  | Perplex | Perplexity (↓) |  | Zero-shot Accuracy (↑) |       |           |       |         |  |  |
|-----------|------------|---------|----------------|--|------------------------|-------|-----------|-------|---------|--|--|
| Wiodei    | 1 iccision | Wiki-2  | C4             |  | ARC-c                  | ARC-e | HellaSwag | MMLU  | Average |  |  |
|           | FP16       | 5.677   | 7.344          |  | 44.62                  | 72.85 | 76.18     | 32.19 | 56.46   |  |  |
| LLaMA-7B  | INT4       | 5.896   | 7.602          |  | 43.77                  | 71.51 | 74.90     | 31.33 | 55.38   |  |  |
|           | INT6       | 5.694   | 7.360          |  | 44.62                  | 72.35 | 75.96     | 32.27 | 56.30   |  |  |
|           | FP16       | 5.091   | 6.798          |  | 47.87                  | 74.75 | 79.08     | 43.46 | 61.29   |  |  |
| LLaMA-13B | INT4       | 5.232   | 6.938          |  | 47.53                  | 73.74 | 78.35     | 43.49 | 60.78   |  |  |
|           | INT6       | 5.096   | 6.809          |  | 48.04                  | 74.96 | 78.98     | 43.24 | 61.31   |  |  |
|           | FP16       | 4.101   | 6.131          |  | 52.90                  | 78.96 | 82.63     | 54.66 | 67.29   |  |  |
| LLaMA-30B | INT4       | 4.260   | 6.254          |  | 52.99                  | 78.16 | 82.28     | 53.62 | 66.76   |  |  |
|           | INT6       | 4.110   | 6.139          |  | 53.07                  | 78.96 | 82.60     | 54.61 | 67.31   |  |  |

## N Experimental results for different seeds

We recall that when constructing a calibration dataset, we randomly draw 128 sequences from the C4 dataset [\[24\]](#page-11-12). By changing the seed for the sampling, different calibration datasets can be constructed, which leads to different quantization results. In this appendix, we report the corresponding results and overall statistics.

Table 18: Quantization performance (PPL ↓) of *aespa* on OPT models for different seeds.

|           |      |       | (a) WikiText-2 |       |       |       |
|-----------|------|-------|----------------|-------|-------|-------|
| Precision | Seed | 125M  | 350M           | 1.3B  | 2.7B  | 6.7B  |
|           | 0    | 28.87 | 22.55          | 14.81 | 12.36 | 10.95 |
| INT4      | 10   | 28.60 | 22.55          | 14.91 | 12.31 | 10.83 |
|           | 100  | 28.75 | 22.85          | 14.94 | 12.35 | 10.90 |
|           | 0    | 32.71 | 24.45          | 15.79 | 13.14 | 11.23 |
| INT3      | 10   | 32.95 | 24.57          | 16.10 | 13.21 | 11.11 |
|           | 100  | 33.38 | 24.45          | 15.70 | 13.27 | 11.24 |
|           | 0    | 71.18 | 54.89          | 24.26 | 22.22 | 15.71 |
| INT2      | 10   | 74.41 | 50.84          | 24.38 | 22.36 | 15.06 |
|           | 100  | 77.03 | 53.12          | 25.93 | 22.39 | 15.66 |
|           |      |       | (b) PTB        |       |       |       |
| Precision | Seed | 125M  | 350M           | 1.3B  | 2.7B  | 6.7B  |
|           | 0    | 40.50 | 32.17          | 20.78 | 18.30 | 15.84 |
| INT4      | 10   | 40.62 | 32.33          | 20.56 | 18.21 | 15.91 |
|           | 100  | 40.11 | 32.60          | 20.55 | 18.20 | 15.86 |
|           | 0    | 44.96 | 36.15          | 22.35 | 19.48 | 16.28 |
| INT3      | 10   | 46.26 | 36.19          | 22.06 | 19.46 | 16.32 |
|           | 100  | 47.54 | 35.61          | 22.10 | 19.66 | 16.39 |
|           | 0    | 99.12 | 79.86          | 37.19 | 32.57 | 22.80 |
| INT2      | 10   | 110.0 | 73.98          | 35.94 | 32.25 | 21.51 |
|           | 100  | 106.0 | 79.09          | 37.33 | 31.90 | 21.86 |
|           |      |       | (c) C4         |       |       |       |
| Precision | Seed | 125M  | 350M           | 1.3B  | 2.7B  | 6.7B  |
|           | 0    | 27.24 | 23.15          | 16.31 | 14.55 | 12.82 |
| INT4      | 10   | 27.23 | 23.13          | 16.32 | 14.54 | 12.81 |
|           | 100  | 27.29 | 23.15          | 16.34 | 14.54 | 12.81 |
|           | 0    | 29.51 | 24.96          | 17.10 | 15.27 | 13.15 |
| INT3      | 10   | 29.59 | 24.98          | 17.06 | 15.29 | 13.15 |
|           | 100  | 29.58 | 25.00          | 17.09 | 15.37 | 13.15 |
|           | 0    | 56.88 | 46.36          | 23.54 | 22.53 | 17.28 |
| INT2      | 10   | 56.23 | 44.02          | 23.91 | 22.56 | 16.91 |
|           | 100  | 56.78 | 45.21          | 24.41 | 22.42 | 17.30 |

Table 19: Quantization performance statistics (PPL ↓) of *aespa* on OPT models.

| Precision | Dataset | 125M          | 350M          | 1.3B          | 2.7B          | 6.7B          |
|-----------|---------|---------------|---------------|---------------|---------------|---------------|
| INT4      | Wiki-2  | 28.74 ± 0.139 | 22.65 ± 0.172 | 14.89 ± 0.066 | 12.34 ± 0.023 | 10.89 ± 0.058 |
|           | PTB     | 40.41 ± 0.264 | 32.36 ± 0.217 | 20.63 ± 0.128 | 18.24 ± 0.057 | 15.87 ± 0.034 |
|           | C4      | 27.25 ± 0.036 | 23.14 ± 0.014 | 16.33 ± 0.016 | 14.55 ± 0.005 | 12.81 ± 0.002 |
| INT3      | Wiki-2  | 33.01 ± 0.340 | 24.49 ± 0.068 | 15.87 ± 0.209 | 13.21 ± 0.064 | 11.19 ± 0.068 |
|           | PTB     | 46.26 ± 1.287 | 35.98 ± 0.321 | 22.17 ± 0.159 | 19.54 ± 0.109 | 16.33 ± 0.058 |
|           | C4      | 29.56 ± 0.043 | 24.98 ± 0.024 | 17.08 ± 0.021 | 15.31 ± 0.050 | 13.15 ± 0.004 |
| INT2      | Wiki-2  | 74.20 ± 2.931 | 52.95 ± 2.029 | 24.86 ± 0.930 | 22.32 ± 0.088 | 15.48 ± 0.363 |
|           | PTB     | 105.0 ± 5.495 | 77.64 ± 3.195 | 36.82 ± 0.766 | 32.24 ± 0.335 | 22.06 ± 0.667 |
|           | C4      | 56.63 ± 0.350 | 45.20 ± 1.171 | 23.95 ± 0.438 | 22.50 ± 0.076 | 17.17 ± 0.219 |