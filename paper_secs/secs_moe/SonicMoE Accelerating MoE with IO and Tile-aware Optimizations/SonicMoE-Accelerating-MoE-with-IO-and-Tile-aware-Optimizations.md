# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

Wentao Guo<sup>1</sup> , Mayank Mishra<sup>2</sup> , Xinle Cheng<sup>1</sup> , Ion Stoica<sup>2</sup> , and Tri Dao<sup>1</sup>,<sup>3</sup>

> 1 Princeton University 2 University of California, Berkeley 3 Together AI

Correspondence to: wg0420@princeton.edu, tri@tridao.me

March 30, 2026

#### Abstract

Mixture of Experts (MoE) models have emerged as the de facto architecture for scaling up language models without significantly increasing the computational cost. Recent MoE models demonstrate a clear trend towards high expert granularity (smaller expert intermediate dimension) and higher sparsity (constant number of activated experts with a higher number of total experts), which improve model quality per FLOP. However, fine-grained MoEs suffer from increased activation memory footprint and reduced hardware efficiency due to higher IO costs, while sparser MoEs suffer from wasted computations due to padding in Grouped GEMM kernels. In response, we propose a memory-efficient algorithm to compute the forward and backward passes of MoEs with minimal activation caching for the backward pass. We also design GPU kernels that overlap memory IO with computation, benefiting all MoE architectures. Finally, we propose a novel "token rounding" method that minimizes the wasted compute due to padding in Grouped GEMM kernels. As a result, our method SonicMoE reduces activation memory by 45% and achieves a 1.86x compute throughput improvement on Hopper GPUs compared to ScatterMoE's BF16 MoE kernel for a fine-grained 7B MoE. Concretely, SonicMoE on 64 H100s achieves a training throughput of 213 billion tokens per day, comparable to ScatterMoE's 225 billion tokens per day on 96 H100s for a 7B MoE model training with FSDP-2 using the lm-engine codebase[1](#page-0-0) . On Blackwell GPUs, SonicMoE also achieves a 25% and 15% relative speedup on the forward and backward pass respectively compared to a highly optimized DeepGEMM baseline on OLMoE-sized 7B MoE models. Under high MoE sparsity settings, our tile-aware token rounding algorithm yields an additional 1.16x speedup on kernel execution time compared to vanilla top-K routing while maintaining similar downstream performance on Hopper GPUs. We open-source all our kernels[2](#page-0-1) to enable faster MoE model training.

## 1 Introduction

Mixture of Experts (MoE) (Shazeer et al. [2017\)](#page-18-0) models have emerged as a key technique for scaling up parameters (Kimi et al. [2025;](#page-17-0) Zhao et al. [2025a\)](#page-19-0) without increasing the training computational requirements. Modern transformers often have layers comprised of a sequence mixer block (e.g. Multi-head Attention (Vaswani et al. [2017\)](#page-18-1)), followed by a channel mixer block (e.g. dense MLPs) where MoEs are an excellent substitute for dense MLPs for FLOPs efficiency. An MoE block is typically composed of a token router and multiple smaller and often equal-sized subnetworks, called "experts". MoEs can reduce FLOPs consumption during training by only activating a subset of all experts per token. However, reducing FLOPs does not directly translate to better hardware utilization since MoE computation features more dynamic IO accesses when each expert needs to gather tokens from different positions, and also scatter the results back to the original positions. Moreover, such hardware-unfriendliness becomes worse as experts become more *granular* (experts have smaller intermediate sizes) and *sparser* (experts are increased while keeping the number of activated experts constant), as shown in Table [4.](#page-24-0)

MoE scaling laws (Clark et al. [2022;](#page-16-0) Krajewski et al. [2024;](#page-17-1) Tian et al. [2025\)](#page-18-2) predict better model quality per FLOP with increasing expert granularity (ratio between the model's embedding dimension and each expert's intermediate size) and sparsity. Recent MoE models like DeepSeek V3 (DeepSeek-AI et al. [2024\)](#page-16-1), Qwen3 MoE (QwenLM [2025\)](#page-17-2) and gpt-oss-120b (OpenAI [2025\)](#page-17-3), have demonstrated the superior performance of "fine-grained" MoEs over "coarse-grained" MoEs at scale. Besides granularity, the pursuit of MoEs with better model quality while keeping computational requirements constant has also led to modern MoEs becoming sparser. For example, Kimi K2 (Kimi et al. [2025\)](#page-17-0) has the same amount of activated parameters as DeepSeek V3 (DeepSeek-AI et al. [2024\)](#page-16-1) but much larger total parameter count. Overall, granularity and

<span id="page-0-0"></span><sup>1</sup><https://github.com/open-lm-engine/lm-engine>

<span id="page-0-1"></span><sup>2</sup><https://github.com/Dao-AILab/sonic-moe>

sparsity for MoEs have only increased over time as shown in Table 4. We note that the pursuit of granularity and sparsity is also adopted by recent alternative architectures to MoE such as PEER (He 2024), Memory Layers (Berges et al. 2024), and Ultra-Mem (Huang et al. 2025).

Though more granular and sparser MoEs increase model quality per FLOP, they suffer from hardware inefficiency due to: (1) larger activation memory footprint for granular MoE models as activation size typically scales linearly with the number of activated experts, (2) lower arithmetic intensity and increased IO cost due to granular experts and (3) wasted computations due to tile quantization effects of grouped GEMM for highly sparse MoEs. The high granularity and sparsity both push MoE training towards the memory-bound regime, requiring carefully designed MoE kernels to hide the increased IO costs. Existing state-of-the-art MoE kernels such as ScatterMoE (Tan et al. 2024) and MoMoE (Costin et al. 2025) are not designed to handle these high IO costs and they suffer significant training throughput degradation.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1: SonicMoE's per-layer activation memory footprint (left) stays constant even when expert granularity (d/n where d is the embedding dimension and n is the expert intermediate dimension) increases, and is 20%-159% more memory-efficient than other baselines. SonicMoE's forward computation throughput reaches an average of 88% of the upper bound (cuBLAS BMM + activation + cuBLAS BMM + aggregation) on both H100 (mid) and B300 (right) GPUs. Note that the cuBLAS upper bound baseline does *not* include the router computation. Here we use a 30B MoE configuration with microbatch size of 32768 tokens for both H100 and B300 GPUs, and we vary the activated experts / total number of experts as 2/32, 4/64, 8/128, and 16/256 from left to right.

We propose to co-design the MoE architecture with a GPU kernel tailored to NVIDIA Blackwell and Hopper generation GPUs and a novel routing method. (1) We derive an algorithm to compute the MoE backward pass more efficiently leading to a much smaller activation memory footprint that does not increase with increasing expert granularity. (2) We leverage new hardware features on Blackwell and Hopper GPUs to overlap memory IO with computation which can benefit all MoEs, and, in particular, fine-grained MoEs. (3) We propose a hardware-aware token rounding routing method where the routed number of tokens to an expert is always a multiple of the GEMM tile size. Using extensive experiments, we show that token rounding routing is 16% faster than the baseline token-choice routing when we scale up the number of experts 4 times from a 30B MoE. We also validate that TR preserves the MoE inference quality on 1.8B parameter scale. With (1) and (2), we can increase the end-to-end training throughput of a 7B MoE model by 42% (without changing the top-*K* token choice routing). Our token rounding routing method further improves training throughput by 16% when we scale up the number of experts without any accuracy loss.

**Summary of contributions.** We propose SonicMoE, a hardware and model architecture co-design solution to address MoE training efficiency problems, making the following contributions:

- MoE training with minimum possible activation memory footprint without increasing FLOPs: We analyze the impact of MoE granularity on the MoE layer's forward and backward passes and observe that increasing MoE granularity while maintaining constant FLOPs leads to a linear increase in activation memory required by the backward pass. Leveraging this observation, we carefully redesign the computation graph to avoid caching the activations for the router gradient computation while maintaining the mathematical equivalence to the original MoE formulation. As a result, for a fine-grained 7B MoE, SonicMoE reduces activation memory usage per layer by up to 45% on H100 GPUs.
- Efficient MoE kernel that overlaps IO with computation to yield SOTA training throughput: We show that increasing both granularity and sparsity leads to MoEs becoming increasingly memory bandwidth bound. To alleviate this bottleneck, we exploit the asynchrony of the GEMM and IO operations by overlapping them to maximize throughput. For the same fine-grained 7B MoE model on H100 GPUs, our approach increases the TFLOPS by 43% on the forward pass compared to a highly optimized DeepGEMM baseline, and by 83% and 115% on the backward pass compared to the state-of-the-art MoE baselines ScatterMoE and MoMoE, respectively. On B300 GPUs, our approach also achieves 25% more TFLOPS on the forward pass and 15% on the backward pass compared to a highly optimized DeepGEMM baseline on OLMoE-sized 7B MoE models. To evaluate the performance of these techniques, we conduct an extensive performance analysis through comprehensive kernel-level profiling and an IO-aware exploration of the MoE computational paths.
- Token rounding routing that eliminates wasted FLOPs from sparse MoEs: We introduce a drop-in routing

algorithm that rounds the per-expert token counts to multiples of the tile size (e.g., 128) used by grouped GEMM in MoE kernels. This rounding reduces compute wasted on padding while preserving the original token-to-expert assignment as much as possible. The algorithm ensures that, for each expert, the maximum deviation from the original top-K token-choice result is bounded by one tile. This method effectively eliminates padding waste in grouped GEMM while maintaining the same total number of tokens in expectation, and it delivers robust token-choice accuracy even under highly sparse MoE training regimes. We validate the performance of this token-rounding strategy in a 1.4B-parameter sparse training setting, demonstrating that its compute throughput consistently exceeds that of the vanilla top-K token-choice routing. In highly sparse regimes, the improvement reaches up to 16% higher TFLOPS for end-to-end MoE computation on H100 GPUs.

We release SonicMoE, mainly written in CuTe-DSL (NVIDIA 2025c) with a PyTorch interface and a permissive license to benefit researchers and practitioners. The GitHub link is https://github.com/Dao-AILab/sonic-moe. A preliminary version of this paper is accepted to ICLR'26: https://openreview.net/pdf?id=KzTJ1raEgB.

## 2 Background

We first provide an overview of the MoE architecture and a standard MoE kernel employing grouped GEMM in Section 2.1. In Section 2.2, we discuss how granularity and MoE sparsity will affect MoE's training efficiency. We then examine the impact of the MoE routing method on the MoE model quality and training efficiency in Section 2.3.

### <span id="page-2-0"></span>2.1 MoE using Grouped GEMM

Modern GPUs support Tensor Cores; specialized hardware units with high matrix multiplication throughput (NVIDIA 2022). A GEMM (general matrix multiply) (Lawson et al. 1979) kernel often has 3 stages: prologue (start input loading), mainloop (keep loading inputs and compute GEMM) and epilogue (miscellaneous IO/math operations on GEMM outputs). The kernel tiles computations (dividing large matrices into small tiles), and optionally pads dimensions so computation aligns with hardware-permissible tile sizes. In this paper, we follow standard GEMM notations in most BLAS (Lawson et al. 1979) libraries: we have  $A \in \mathbb{R}^{\mathbf{M} \times \mathbf{K}}, B \in \mathbb{R}^{\mathbf{K} \times \mathbf{N}}, C \in \mathbb{R}^{\mathbf{M} \times \mathbf{N}}$  for C = AB with problem shape ( $\mathbf{M}, \mathbf{N}, \mathbf{K}$ ). This notation is adopted by CUTLASS (NVIDIA 2025a) which implements efficient GEMM on CUDA.

```
Algorithm 1 MoE forward with Grouped GEMM
Input : X \in \mathbb{R}^{T \times d}, W_1 = \{W_{1,e}\}_{e \in [E]} \in \mathbb{R}^{d \times 2n}, W_2 =
           \{W_{2,e}\}_{e\in[E]}\in\mathbb{R}^{n\times d}, routing scores S\in\mathbb{R}^{T\times E},\,\pi\in
           \{0,1\}^{T\times E} as a binary-valued mask matrix where \pi_{t,e} repre-
           sents whether token t is routed to expert e.
Output : output activation O \in \mathbb{R}^{T \times d}
Parallel for e \in [E] do
     // up-proj
     X_e \leftarrow \text{Gather}(X, \pi_{:,e})
     // varlen-M Grouped GEMM
      H_e \leftarrow X_e W_{1,e}
     // apply activation function, e.g. SwiGLU
     A_e \leftarrow \operatorname{act\_func}(H_e)
     // down-proj, varlen-M Grouped GEMM
     Y_e \leftarrow A_e W_{2,e}
Parallel for t \in [T] do
     // expert aggregation
     O_t = \sum_{e \in [E]} \pi_{t,e} S_{t,e} Y_{e,t}
```

![](_page_2_Figure_7.jpeg)

Figure 2: MoE computation often requires a Grouped GEMM. Each expert gathers inputs from different positions on an input tensor (top) or reads a contiguous chunk on a grouped input array (bottom). This figure is adapted from Tan et al. (2024)'s Figure 2.

On both NVIDIA Hopper and Blackwell GPUs, GEMM is performed asynchronously with a producer-consumer paradigm (Shah et al. 2024) where producers are dedicated to loading a tile of data from High Bandwidth Memory (HBM), or global memory (GMEM) logically, to shared memory (SMEM) while consumer warpgroups are responsible for GEMM computation (Shah et al. 2024). In prologue and mainloop, producer warpgroups fetch a tile of data and cache it to a dedicated pipeline while the consumer warpgroups read from the cached tile in this pipeline, perform tiled matrix multiply (MMA) and accumulate over the  $\mathbf K$  dimension of GEMM. After the mainloop, we enter the epilogue stage where the consumer warpgroups apply post-processing (activation function and write results back to HBM) on the final MMA results.

An MoE block is typically composed of a token router and multiple smaller and often equal-sized subnetworks, called "experts". The router is responsible for dispatching tokens to the experts which are subsequently used by the specific expert for

computation. The outputs from all experts in the layer are then aggregated and passed onto the next layer. MoE computation[3](#page-3-2) can be performed using Grouped GEMM (a list of GEMMs with possibly different {M, N, K} dimensions). Algorithm [1](#page-2-1) illustrates running MoE forward with Grouped GEMM.

As shown in Algorithm [1,](#page-2-1) during the forward pass (and backward activation gradient computation), we have a variable number of tokens routed to every expert. A Grouped GEMM operation with fixed (N, K) dim (as the expert weight matrix) but variable M (token dim) is then performed. We refer to this Grouped GEMM as "varlen-M Grouped GEMM". During the backward weight gradient computation, the embedding dimension (M for backward) and intermediate hidden size (N for backward) are constant and instead, we reduce over the token dimension (K), which we refer to as "varlen-K Grouped GEMM". For each Grouped GEMM, we often have inputs gathered from different positions or contiguously-packed, as illustrated in Figure [2.](#page-2-1) For example in Algorithm [1,](#page-2-1) the inputs to up-proj are gathered while the inputs to down-proj are already contiguously-packed.

## <span id="page-3-0"></span>2.2 MoE computation

Arithmetic intensity, defined as the ratio of FLOPs over the number of transferred bytes (IO), is a metric to quantify whether a kernel is memory-bound (kernel runtime dominated by memory IO cost) or compute-bound (kernel runtime dominated by compute throughput).

The standard MoE computation for an expert e with SwiGLU activation can be broken down into the following components:

$$H_e = \text{up-projection}(X_e) = X_e W_{1,e} : \mathbb{R}^{T_e \times d} \to \mathbb{R}^{T_e \times 2n}$$
 (1)

<span id="page-3-4"></span>
$$A_e = \text{SwiGLU}(H_e) : \mathbb{R}^{T_e \times 2n} \to \mathbb{R}^{T_e \times n}$$
 (2)

$$Y_e = \text{down-projection}(A_e) = A_e W_{2,e} : \mathbb{R}^{T_e \times n} \to \mathbb{R}^{T_e \times d}$$
 (3)

where X<sup>e</sup> ∈ R <sup>T</sup>e×<sup>d</sup> denotes the input received by expert e.

Here, the up-projection uses 2T<sup>e</sup> · 2n · d FLOPs and 2Ted + 2 · 2n · d + 2Ten HBM memory transfer bytes (we ignore the writes for H<sup>e</sup> here). Similarly, down-projection uses 2Tend FLOPs with 2Ten + 2nd + 2Ted bytes. Defining ρ = K E as the MoE activation ratio, G = d n as the granularity and uniform routing i.e., T<sup>e</sup> = T ρ, the arithmetic intensity (ignoring the writes for He) for the forward pass of an expert is

$$\frac{2T_e \cdot 2n \cdot d + 2T_e nd}{4T_e n + 6nd + 4T_e d} = \frac{3}{\frac{2}{d} + \frac{2}{n} + \frac{3}{T_e}} = \frac{3}{\frac{2+2G}{d} + \frac{3}{T_\rho}}$$
(4)

For a specific model size (constant d), it can be seen that increasing granularity (increasing G) or increasing sparsity (decreasing ρ) leads to a decreasing arithmetic intensity. This is caused by the linear scaling of IO cost w.r.t. expert granularity. Therefore, for the case of fine-grained MoEs (high G) [4](#page-3-3) , it becomes increasingly important to address the increased IO cost by maximally reducing IO access and hiding IO latency. We examine a memory-efficient MoE kernel design in Section [3](#page-4-0) and discuss techniques to reduce IO access and latency in Section [4.](#page-6-0)

Existing MoE kernel designs. There are multiple MoE implementations available: ScatterMoE (Tan et al. [2024\)](#page-18-3), MoMoE (Costin et al. [2025\)](#page-16-3), MegaBlocks (Gale et al. [2023\)](#page-16-4), and Megatron (Shoeybi et al. [2019\)](#page-18-5). However, they do not specialize for the setting of fine-grained MoEs that have linearly-increasing IO cost w.r.t. increasing expert granularity. In contrast, our kernel design, SonicMoE, minimizes the impact of IO cost on the training throughput. In Figure [11a](#page-12-0) and [11b,](#page-12-0) we show that when expert granularity G increases, SonicMoE demonstrates a greater relative speedup over existing MoE kernel designs due to the IO-aware optimizations. We elaborate on the technical differences between SonicMoE and prior MoE kernels in Appendix [B](#page-21-0) and include an overview in Table [1.](#page-6-1)

## <span id="page-3-1"></span>2.3 MoE routing methods

In MoE, routing determines which experts to activate for each token. Token choice (TC) routing where each token independently selects the activated expert is often the default routing method for MoE models (Shazeer et al. [2017\)](#page-18-0). We often have top-K TC routing where the routing decision for token t is TopKe∈[E] (St,e, K) and St,e is the expert score for token t. Besides top-K, Huang et al. [\(2024\)](#page-17-10) introduce token-choice top-P routing to flexibly allocate compute during training.

<span id="page-3-2"></span><sup>3</sup>We refer to the computation that decides the activated expert for each token and relevant routing metadata as *MoE routing*, and how each expert processes the routed tokens and expert aggregation as *MoE computation*. Algorithm [2, 3,](#page-5-0) and [5](#page-23-0) are SonicMoE's MoE computation components that are compatible with arbitrary routing algorithms.

<span id="page-3-3"></span><sup>4</sup>Here we refer to "fine-grained MoE" as MoE with small intermediate size i.e., n is smaller than d. We assume the setting of both iso-FLOPs and iso-params.

However, it introduces nondeterminism in the number of activated experts and consumed FLOPs per token. Zeng et al. (2024b) also propose a similar idea that uses "null experts" to dynamically adjust the number of activated experts.

Besides TC routing, expert choice (EC) routing is developed to avoid load imbalance for expert parallelism (Zhou et al. 2022) by letting experts choose the tokens. However, EC routing is not directly usable for inference because it is incompatible with autoregressive decoding, and switching back to TC at inference time leads to a mismatch. In addition, EC breaks causality by future token information leakage (Wang et al. 2024). To address the inference issue of EC routing, Raposo et al. (2024) introduce an auxiliary loss to promote the agreement between TC and EC routing results, or train an auxiliary router to explicitly predict the routing result of EC router and use this auxiliary router during inference.

In this paper, we propose a novel Grouped GEMM tile-aware token rounding method that rounds the number of received tokens per expert ("expert frequency") to nearby multiples of Grouped GEMM tile sizes and alters at most one tile of tokens per expert. This approach effectively reduces wasted FLOPs caused by Grouped GEMM padding during sparse MoE training while preserving inference quality of trained MoE models. There are similar works that propose to drop and reroute tokens, including Rectify-Router (Zeng et al. 2024a), but they do not focus on the tile structure of Grouped GEMM. Other works such as TMA-adaptive FP8 Grouped GEMM (Fu et al. 2025) focus on reducing padding-related load traffic but the FLOPs wasted by non-aligned tile size in GEMM computation is not addressed.

## <span id="page-4-0"></span>3 Memory-efficient MoE algorithm

<span id="page-4-2"></span>![](_page_4_Figure_4.jpeg)

Figure 3: Computational workflow of SonicMoE's 8 launched kernels, grouped by yellow boxes. 3 and 5 kernels are launched during forward and backward computation respectively. The incoming arrows to a yellow circle indicate a variable loaded from HBM to SRAM, and an outgoing arrow represents a variable stored to HBM. We color the boxes of all variables on HBM, with purple boxes indicating the output of forward and backward while blue boxes indicate intermediate variables or weights  $(W_1, W_2)$ . We color all cached activations X, H,  $\pi$ , S in red. Algorithm 2 formally describes SonicMoE's forward pass, and Algorithm 3 and 5 describe the backward pass.

We first describe SonicMoE's high-level kernel design in Section 3.1 that illustrates SonicMoE's MoE computation<sup>3</sup> as shown in Algorithm 2, 3, and 5. We then focus on the activation memory usage of SonicMoE in Section 3.2.

#### <span id="page-4-1"></span>3.1 Overview of SonicMoE's MoE kernels

The MoE computation<sup>3</sup> in SonicMoE launches 8 kernels: during forward, we have up-proj (A), down-proj (Y), and expert aggregation (O) kernels; during backward, we have activation gradient kernels for dH (down-proj),  $d\tilde{X}$  (up-proj), dX (aggregating  $d\tilde{X}$  across experts), and weight gradient kernels  $dW_1$  and  $dW_2$ . Figure 3 illustrates the computational workflow of these 8 kernels. We provide an efficient TC top-K router, and an interface that accepts arbitrary routing input. However, it should be noted that SonicMoE's MoE computation is independent of the MoE router choice and is thus compatible with arbitrary router logic.

<span id="page-5-0"></span>**Algorithm 2** SonicMoE's MoE kernel forward pass. Variables stored in HBM are colored blue. load and store means load from / store into HBM respectively.

```
Input : X, S, \pi, W_1, W_2 same as Algorithm 1.
Output: MoE layer output O
Up-proj A kernel (X, W_1, \pi) \rightarrow (H, A):
// Gather + varlen-M Grouped GEMM + SwiGLU
      Parallel for e \in [E] do
            X_e, W_{1,e}, \pi_{:,e} \leftarrow \text{load}(X_e, W_{1,e}, \pi_{:,e})
            X_e \leftarrow \text{Gather}(X, \pi_{:,e})
            H_e \leftarrow X_e W_{1,e}
            // apply activation function, e.g. SwiGLU
            A_e \leftarrow \operatorname{act\_func}(H_e)
            H_e, A_e \leftarrow \text{store}(H_e, A_e)
Down-proj Y kernel (A, W_2) \rightarrow Y:
// varlen-M Grouped GEMM
      Parallel for e \in [E] do
            A_e, W_{2,e} \leftarrow \operatorname{load}(A_e, W_{2,e})
            Y_e \leftarrow A_e W_{2,e}
            Y_e \leftarrow \text{store}(Y_e)
Expert aggregation O Kernel (Y, S, \pi) \rightarrow O:
// Gather and sum
      Parallel for t \in [T] do
            Y_{e,t}, S_{t,e}, \pi_{t,e} \leftarrow \operatorname{load}(Y_{e,t}, S_{t,e}, \pi_{t,e})
            O_t \leftarrow \sum_{e \in [E]} \pi_{t,e} S_{t,e} Y_{e,t}
            O_t \leftarrow \text{store}(O_t)
```

**Algorithm 3** SonicMoE's MoE kernel backward pass of down projection.

```
Input :S, \pi, W_2, dO.
Output: dH, dW_2, dS.
Down-proj act dH kernel (dO, W_2, S, \pi) \rightarrow (dH, dS, A'):
// Gather + varlen-M Grouped GEMM + dSwiGLU + dS
// Appendix C elaborates this algorithm in more detail
      Parallel for e \in [E] do
            dO_e, W_{2,e}, S, \pi_{:,e}, H_e \leftarrow \text{load}(dO_e, W_{2,e}, S, \pi_{:,e}, H_e)
            dO_e \leftarrow \text{Gather}(dO, \pi_{:,e})
            // dA' is a temp variable for computing dA, dS, and A'
            dA_e' \leftarrow dO_e \ W_{2,e}^\top
                                                                          / / dA'_e \in \mathbb{R}^{T_e \times n}
            \mathbf{s}_e \leftarrow \text{Gather}(S, \pi_{:,e})
            dA_e \leftarrow \text{Broadcast}(\mathbf{s}_e) \ dA'_e
            // compute fwd act and bwd act grad simultaneously in dAct call
            A_e, dH_e \leftarrow dAct\_func(dA_e, H_e)
            A_e' \leftarrow \operatorname{Broadcast}(\mathbf{s}_e) A_e \ \ // \ A' \in \mathbb{R}^{T_e \times n}, input for dW_2
            dS_{e,t} \leftarrow \langle dA'_{e,t}, A_{e,t} \rangle
                                                                // reduce over n dim
            dH_e, dS, A'_e \leftarrow \text{store}(dH_e, dS, A'_e)
Down-proj weight dW_2 kernel (dO, A', \pi) \rightarrow dW_2:
// Gather + varlen-K Grouped GEMM
      Parallel for e \in [E] do
            dO_e, A'_e, \pi_{:,e} \leftarrow \operatorname{load}(dO_e, A'_e, \pi_{:,e})
            dO_e \leftarrow \text{Gather}(dO, \pi_{:,e})
            dW_{2,e} \leftarrow A_e^{\prime \top} dO_e
            dW_{2,e} \leftarrow \text{store}(dW_{2,e})
```

The implementation of SonicMoE's MoE computation is highly modularized: it only consists of (1) an optimized grouped GEMM kernel with modularized fusion and (2) an optimized expert aggregation kernel. The host dispatches to the best GEMM config and load/store strategies to launch the 8 kernels listed above. Besides such high modularity, SonicMoE still exhibits state-of-the-art training throughput and minimum activation memory usage which we describe below.

### <span id="page-5-1"></span>3.2 Activation memory efficiency

The FLOPs of MoE forward and backward computation is (6+12)TnKd. For a given  $T, d^5$ , we need to keep nK constant for constant FLOPs. Therefore, increasing granularity requires decreasing n and proportionally increasing K. Hence, any activations with memory O(TKd) should not be cached for backward computation to avoid activation memory scaling with granularity. For current MoE kernels like ScatterMoE, activations scale linearly with expert granularity. Activations Y (down-proj output) and  $X_e$  (gathered X) have size TKd and avoiding caching them eliminates activation memory dependency on granularity. We avoid writing dY (gradient for Y) and  $dO_e$  (gathered dO) to HBM as they increase the peak activation memory during the backward computation:

- For X and dO, fusing the gather operation with the HBM load eliminates the need for materialization and activation caching in HBM. We show in Figures 5a and 19a that this gather fusion significantly improves the throughput for fine-grained MoEs.
- A naive implementation to compute dS and dH would need Y and dY. Instead, we identify an alternative computation path to compute dS and dH without increasing FLOPs. This is achieved via expanding dS and dH into an equation that does not involve using Y and dY, as illustrated in Appendix C. SonicMoE's dH kernel is shown in Algorithm 3.

As a result, we only cache X and H along with routing metadata for a total size 2Td+4TKn bytes per layer. This activation memory usage is the same as a dense model with the same number of activated parameters, which is the minimum activation memory required for backward computation without doing activation recomputation with GEMM.<sup>6</sup> In Figure 10, we profile SonicMoE's activation memory for a 7B MoE training configuration and demonstrate that the activation memory of SonicMoE is independent of expert granularity.

<span id="page-5-3"></span><span id="page-5-2"></span><sup>&</sup>lt;sup>5</sup>Embedding dimension d is often picked independently of MoE layer so we always assume d is fixed.

<sup>&</sup>lt;sup>6</sup>Although SonicMoE still materializes a temporary *Y* variable, we can recycle *Y* after each layer. As long as the number of MoE layers (typically 32+ for 7B+ MoE) is larger than *K*, the transient memory usage of *Y* will be overshadowed. Removing such materialization requires an atomic add (Figure 17 right) to global memory which creates new issues with determinism (He and Machines 2025), numerical accuracy (for BF16 atomic add), and incompatibility with all2all or all-gather communication.

## <span id="page-6-0"></span>4 IO-aware kernel design

The expressivity of fine-grained MoE comes from the diversity of every token's expert selection, which in turn leads to linearly-scaled IO cost w.r.t. expert granularity. To sustain high throughput, we need to maximally (1) reduce IO access via fusion (2) overlap the IO latency with compute. We first examine the token gather fusion with computation, and math and IO fusion with epilogue in Section 4.1.1 and 4.1.2 respectively. We then describe the techniques to overlap MMA with IO in Section 4.2. In Appendix B, we compare SonicMoE with other MoE kernel designs with a summary in Table 1.

<span id="page-6-1"></span>

| Features \ Methods                                                           | SonicMoE | ScatterMoE   | МоМоЕ        | MegaBlocks | Megatron     | DeepGEMM |
|------------------------------------------------------------------------------|----------|--------------|--------------|------------|--------------|----------|
| Gather fused with GMEM-to-SMEM load (Sec. 4.1.1)                             | <b> </b> | fwd √, bwd 🗡 | fwd √, bwd X | Х          | Х            | Х        |
| SwiGLU and dSwiGLU fused with epilogue (Sec. 4.1.2)                          | ✓        | X            | ✓            | X          | $\checkmark$ | NA       |
| $dS$ computed as $\langle dA'_{e,t}, A_{e,t} \rangle$ (Sec. 4.1.2, App. C.1) | ✓        | X            | X            | X          | $\checkmark$ | NA       |
| Backward epilogue that computes $dH$ , $dS$ together (Sec. 4.1.2)            | <b>√</b> | X            | X            | X          | X            | NA       |
| Overlap MMA with epilogue/IO (Sec. 4.2)                                      | ✓        | X            | X            | X          | X            | ×        |
| Do not need a separate scatter kernel                                        | <b>√</b> | $\checkmark$ | ✓            | X          | NA           | NA       |
| Efficient top- $K$ sorting (App. D)                                          | ✓        | X            | X            | X          | X            | NA       |
| Do not need shape-alignment efforts outside GEMM kernels                     | <b>✓</b> | $\checkmark$ | $\checkmark$ | ×          | X            | ×        |

Table 1: Comparison between SonicMoE and prior MoE kernels. ✓ means that the kernel implements the feature or a functionality similar in semantics, and ✗ means the feature is missing from the kernel. "NA" means that the feature is out of the expected scope. We use the GroupedMLP for Megatron and ParallelDroplessMLP for MegaBlocks. More discussion is included in Appendix B.

### 4.1 SonicMoE's Grouped GEMM

SonicMoE is built on top of an efficient varlen-M and varlen-K Grouped GEMM. Inside the Grouped GEMM, we fuse the gather operations with the activation loading (4.1.1), and fuse SwiGLU/dSwiGLU/dS with epilogue (4.1.2). The gather fusion helps SonicMoE to be faster than MoE kernel designs that require a separate gather kernel such as MegaBlocks, Megatron, and DeepGEMM++, an optimized MoE forward pass implementation built on top of the DeepGEMM library (Zhao et al. 2025b). The epilogue fusion boosts SonicMoE to be faster than ScatterMoE in the backward pass. These fusions reduce unnecessary IO access and can be overlapped with compute MMA, as we discuss in Section 4.2.

#### <span id="page-6-2"></span>4.1.1 Gather fusion with GMEM-to-SMEM load

SonicMoE's Grouped GEMM accepts either contiguously-packed inputs or inputs gathered from different positions, illustrated in Figure 2. For the latter case, we fuse the input gather with the input loads from global memory (GMEM, often the HBM) to shared memory (SMEM) so we can batch them to perform GEMM on Tensor Core (Costin et al. 2025; Tan et al. 2024). This involves (1) fetching the routed token indices for each expert and then (2) using these indices to gather activations via the cp.async instruction.

As shown in Figure 5a and 5b, Gather fusion provides SonicMoE with a major advantage over existing MoE kernel designs on both H100 and B300 GPUs such as DeepGEMM. Although DeepGEMM's varlen-M Grouped GEMM kernel is highly optimized, DeepGEMM assumes the inputs are already contiguously packed and padded to multiples of 128, which requires a separate kernel launch for gather and pad before the Grouped GEMM.

In the backward pass, weight gradients for up-proj and down-proj ( $dW_1$  and  $dW_2$  respectively) need to gather X and dO, and the activation gradient for dH also needs to gather dO. Despite the backward having more kernels requiring the gather operation, existing approaches including ScatterMoE (Tan et al. 2024) and MoMoE (Costin et al. 2025) fuse the gather during forward but still launch a separate gather kernel during backward. Fusing this gather reduces the IO cost by 2TKd bytes and cuts down a major portion of fine-grained MoE training time.

**Blackwell GPUs.** On Blackwell GPUs, the gather fusion with cp.async encounters an architectural challenge when using 2-CTA clusters (Figure 4) for GEMM computation. The cp.async instruction, introduced in the Ampere generation, can only signal completion within the same CTA. However, Blackwell's 2-CTA GEMM requires the MMA instruction in the leader CTA (CTA 0) to wait for gather completion from both CTAs. To work around this limitation, CTA 1 requires a dedicated relay warp that receives the cp.async

<span id="page-6-3"></span>![](_page_6_Figure_11.jpeg)

Figure 4: Pipeline structure for gather fusion with cp.async on Blackwell GPUs using 2-CTA clusters.

completion signal and forwards it to CTA 0's MMA warp using

cluster-level synchronization primitives (e.g., mbarrier with cluster scope). This relay mechanism adds scheduling complexity but enables efficient gather fusion across the 2-CTA cluster, maintaining high throughput for Grouped GEMM.

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

(a) Runtime breakdown of 7B MoE training on H100 GPUs.

![](_page_7_Figure_4.jpeg)

(b) Runtime breakdown of OLMoE-sized 7B (Muennighoff et al. 2025) MoE training on B300 GPUs. Triton official example does not implement the backward pass.

Figure 5: Runtime breakdown of different MoE kernels (ms  $\downarrow$ ) on the 7B MoE training. We annotate the model memory bandwidth (TB/s  $\uparrow$ ) for memory-bound kernels (gather, SwiGLU/dSwiGLU, and expert aggregation kernel) and compute throughput (TFLOPS  $\uparrow$ , abbr as TF/s) for grouped GEMM kernels. Note that this profile is grouped by kernel runtime semantics and one block can contain multiple actual kernel timing results. For example, the "router related" on left subfigure includes both router GEMM and routing metadata computation time. In addition, we do not consider the CUDA stream bubble time across kernels in this figure. We use the GroupedMLP for Megatron, and ParallelDroplessMLP for MegaBlocks. DeepGEMM does *not* provide an efficient router implementation, gather and expert aggregation kernels during the forward pass, where we use a standard PyTorch implementation ("DeepGEMM-pt") or our highly optimized kernels ("DeepGEMM++") for them. During the backward pass, both "DeepGEMM++" and "DeepGEMM-pt" use the same computational path as SonicMoE, except we launch separate kernel(s) that compute dS, A', and dSwiGLU together. DeepGEMM++ is effectively the best possible MoE implementation built on top of DeepGEMM SM90/SM100 BF16 Grouped GEMM kernels without modifying DeepGEMM's source code. On B300 GPUs, we also compare with triton official example ("triton ex.") for MoE that is optimized for inference as it does not store pre-activation results H.

#### <span id="page-7-1"></span>4.1.2 Epilogue fusion

We exploit the epilogue computation to maximally reduce unnecessary IO accesses with the following design choices:

- SwiGLU and dSwiGLU fusion: We fuse the SwiGLU and backward of SwiGLU with the epilogue of forward up-proj and backward down-proj activation gradient kernel respectively (Costin et al. 2025).
- Computing dH and dS in backward down-proj activation gradient (dH) kernel's epilogue: This heavy epilogue fusion helps SonicMoE's dH kernel to produce the same output with far less total time than ScatterMoE's down-proj act, dS, and dSwiGLU combined together in Figure 5a and 5b.

In Appendix C.1, we show that SonicMoE's  $dS = \langle dA, A' \rangle = \langle dA, \operatorname{Broadcast}(\mathbf{s})A \rangle$  is the computationally and activation memory-efficient choice for fine-grained MoEs. However, both ScatterMoE and MoMoE choose to compute dS as  $\langle dO, Y \rangle$ , which requires an additional 2TKd HBM load cost and requires caching 2TKd bytes of activation memory. In Figure 5a (right subfigure), ScatterMoE launches a separate kernel for dS while MoMoE fuses dS with up-proj activation gradient which takes much longer time than SonicMoE's up-proj activation gradient.

The throughput of heavy epilogue fusion on backward down-proj activation gradient dH kernel is boosted by the overlap of asynchronous IO and MMA, which we will elaborate in Section 4.2. Such overlap helps SonicMoE to sustain a reasonable training throughput and memory bandwidth simultaneously even with the heavy epilogue fusion (load H and S, compute dH, dS, and A' as inputs to  $dW_2$ ) in dH kernel.

### <span id="page-8-1"></span><span id="page-8-0"></span>4.2 GEMM MMA Overlapping with Asynchronous IO

![](_page_8_Figure_2.jpeg)

(a) SonicMoE's Ping-Pong warpgroup scheduling on Hopper GPUs. The green arrows indicate that a consumer warpgroup signals the start of the epilogue and the other consumer warpgroup can proceed with the MMA. Once this step is complete, the roles of 2 consumer warpgroups are switched. SonicMoE mainly uses Ping-Pong for forward down-proj Y kernel and backward down-proj activation gradient dH kernel as they both have heavy epilogue. In dH kernel, SonicMoE has an asynchronous TMA load during epilogue. This figure is adapted from Wright and Hoque (2024a)'s blog on Ping-Pong scheduling.

![](_page_8_Figure_4.jpeg)

(b) SonicMoE's strategy to overlap GEMM MMA with epilogue IO on Blackwell GPUs. The green arrows indicate that the MMA warp signals the epilogue warps that the MMA accumulation result of a work tile is ready for epilogue. The yellow arrows indicate that the epilogue warps finish the epilogue of a work tile and signal the MMA warp that the epilogue warps' owned Tensor Memory (TMEM, an on-chip memory on Blackwell GPUs) stages now become available. Figure 7 illustrates this TMEM stage ownership transfer process.

Figure 6: SonicMoE's strategy to overlap GEMM MMA with asynchronous IO on Hopper and Blackwell GPUs.

**Hopper GPUs.** In NVIDIA Hopper GPUs, GEMM is performed asynchronously with a producer-consumer paradigm (Shah et al. 2024). Suppose we have 2 consumer warpgroups, we can overlap the IO of 1 warpgroup with the GEMM of another warpgroup. Once this is finished, we switch the roles of the warpgroups (effectively interleaving IO and GEMM). This is often referred to as *Ping-Pong scheduling* (Shah et al. 2024; Wright and Hoque 2024b) on Hopper GPUs in Figure 6a.

Ping-Pong scheduling is particularly useful to maintain high Tensor Core throughput with heavy epilogue. For example, the down-proj forward Y kernel's epilogue has heavy HBM store IO (2TKd) bytes) relative to the mainloop. In the down-proj activation gradient (dH) kernel's epilogue, we need to load H and execute multiple activation and reduction operations to compute and store dH, dS, and A' as inputs for  $dW_2$ . We note that the concept of overlapping MMA with IO and Ping-Pong scheduling is known in other places such as Flash Attention 3 (Shah et al. 2024), but the application of Ping-Pong scheduling to address the increasing IO costs of fine-grained MoE kernel design is novel.

Besides Ping-Pong scheduling, SonicMoE also relies on asynchronous TMA operations to perform GMEM-to-SMEM load and SMEM-to-GMEM store. We overlap the following asynchronous IO with the MMA operations:

- Asynchronous TMA load during dH kernel's epilogue: In the dH kernel's epilogue, we need to load H to compute dH from dA. We create a dedicated pipeline for asynchronous TMA load of H to overlap with other epilogue operations across epilogue stages.
- Asynchronous TMA store in forward down-proj Y and backward up-proj activation gradient dX kernel: in forward down-proj and backward up-proj activation gradient, SonicMoE does *not* fuse the scatter with HBM store where ScatterMoE and MoMoE both choose to fuse the HBM store with scatter. This is primarily because the scatter fusion requires a synchronous SMEM-to-GMEM store instruction on Hopper GPUs.<sup>7</sup> The synchronous GMEM store

<span id="page-8-2"></span> $<sup>^7</sup>$ Synchronous  $\operatorname{st.global}$  is the *only* available PTX instruction for scatter fusion with HBM store on Hopper GPUs if we do not use TMA 1D

blocks the execution of MMA of next tile and largely degrades the TFLOPS ( $\sim$ 20%) in the case of heavy HBM store, as illustrated by Figure 16.

Blackwell GPUs. In NVIDIA Blackwell GPUs, GEMM kernels use the same "Ping-Pong" scheduling in spirit, but the implementation differs from Hopper. Blackwell introduces Tensor Memory (TMEM), a dedicated 256KB on-chip memory per SM organized as 128 rows  $\times$  512 columns of 32-bit cells (NVIDIA 2025b; Research 2024). The accumulator results from matrix multiplication are stored directly in TMEM rather than in registers, with the 512-column structure naturally enabling a two-stage accumulator pipeline. Each stage uses 256 columns: while one stage performs MMA operations via the new UMMA (Unified Matrix Multiply-

<span id="page-9-0"></span>![](_page_9_Figure_2.jpeg)

Figure 7: Illustration that shows MMA warp would coordinate with epilogue warps to fully utilize TMEM resources while overlapping MMA with asynchronous IO.

Accumulate) instruction, the other stage executes the epilogue. This process is illustrated in Figure 7. Unlike Hopper's WGMMA which required warpgroup-level coordination and consumed significant register memory, Blackwell's UMMA is a single-threaded asynchronous operation that eliminates register pressure for accumulation. This architectural change allows epilogue warps to read and process results from one TMEM stage concurrently with MMA warps accumulating into the other stage, enabling better overlap of epilogue and MMA operations compared to Hopper's ping-pong scheduling.

#### 5 Token rounding routing

In this section, we analyze the hardware efficiency under sparse MoE training regime and identify that as MoEs become sparser, the wasted compute on padded GEMM tiles accumulates to a nontrivial amount, known as "tile quantization" effects. In response, we propose a novel routing method "token rounding" to eliminate tile quantization effects.

## <span id="page-9-1"></span>Training efficiency of sparse MoE

#### **Algorithm 4** Token rounding routing

 $X \in \mathbb{R}^{T \times d}$ ; number of experts E and expected activated number of experts K per token; tile size  $M_{\text{tile}}$ ; router scores  $S \in [0,1]^{T \times E}$ . round\_and\_sparsify that determines rounding up or down.

**Output**:  $M_{\text{tile}}$ -rounded routed token list and scores  $\{\pi'_e, s'_e\}_{e \in [E]}$ 

(1) Top-K token choice sorting

 $(S_{topK}, I_{topK}) \leftarrow TopK(S, K)$ 

(2) Calculate each expert's received token frequencies and its  $M_{\rm tile}$ rounded multiples

$$\begin{aligned} f_e \leftarrow & \sum_t \mathbf{1}_{\{e \in I_{\text{topK, t}}\}} \\ & [f_e]_{M_{\text{tile}}} \leftarrow [f_e/M_{\text{tile}}] \cdot M_{\text{tile}} \\ & [f_e]_{M_{\text{tile}}} \leftarrow \lfloor f_e/M_{\text{tile}} \rfloor \cdot M_{\text{tile}} \end{aligned}$$

(3) Build Top-K-preferred S' for expert-wise ranking

// ensure non-top-
$$K$$
 entries are smaller  $S'_e \leftarrow S_e - 1$  for  $t \in [T] \& k \in [K]$  in parallel do  $S'_{t,I_{\text{topK}}(t,k)} \leftarrow S_{\text{topK},t,k}$ 

(4) Token rounding per expert

$$\begin{aligned} & \textbf{for } e \in [E] \ \textbf{do} \\ & \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{ } \text{$$

![](_page_9_Figure_19.jpeg)

Figure 8: Wasted FLOPs by padding during MoE forward & backward pass with T = 16k, d = 4k, n = 1k, K = 4 as illustrated in the bottom right 2 subfigures of Figure 13.

![](_page_9_Figure_21.jpeg)

Figure 9: A demonstration of tile quantization effect for sparse MoE. The rounding subroutine in TR makes a binary decision for discarding or  $\pi'_e, s'_e \leftarrow \text{round\_and\_sparsify}(\pi_e, s_e, f_e, \lceil f_e \rceil_{M_{\text{tile}}}, \lfloor f_e \rfloor_{M_{\text{tile}}})$  padding tokens to guarantee that each expert receives an  $M_{\text{tile}}$ -multiple number of tokens.

Besides granularity, the arithmetic intensity of MoE also depends on the MoE activation ratio  $\rho$  as shown in Equation 4. When we scale down  $\rho$ , the expected number of received tokens per expert  $\mathbb{E}_{e \in [E]} T_e = \bar{T}_e = T \rho$  will also decrease linearly

store. This is different from the gather case as cp.async is asynchronous but it cannot be used for SMEM-to-GMEM store. Although asynchronous st.async.release.global instruction is available on Blackwell GPUs, the repeated index fetching would still make scatter a less favorable option.

and the GEMM computation shifts towards a memory-bound regime.

**Tile quantization effect.** GEMM on modern GPUs is often computed in tiles (NVIDIA 2022) and we always need to pad to the next tile-sized multiples if any dimensions of M, N, K are not fully divisible by tile sizes. Once the size of input (e.g. token dimension per expert) is small, the wasted TFLOPS by padding can be nontrivial.

Therefore, we propose to use token rounding to avoid launching such extra tiles, thereby leading to more efficient training. We also empirically show that our token rounding method does not affect model quality while achieving much higher training throughput.

### 5.2 Token rounding routing

As such, we propose to use the token rounding (TR) method as a 2-step sorting algorithm as shown in Algorithm 4. The token rounding algorithm first computes the vanilla token-choice (TC) routing results and applies a sorting of the router score over each expert's tokens, similar to the EC sorting step. We then choose to either discard tokens selected in the first step of TC top-K routing or pad additional tokens in the second step of sorting. Between these 2 steps, we process the routing weight matrix such that the TC tokens are always preferred over EC tokens. This is done so that both discarding or padding only affects the last input tile for each expert.

Token rounding requires a round\_and\_sparsify subroutine for making a binary decision about discarding or padding. Our default choice for such a subroutine is to round expert frequency to the nearest  $M_{\rm tile}$  multiples: we choose to pad EC selected tokens if  $\lceil f_e \rceil_{M_{\rm tile}} - f_e$  is smaller than  $f_e - \lfloor f_e \rfloor_{M_{\rm tile}}$ . <sup>8</sup> We further conduct an ablation in Table 6 and find that (1) our TR algorithm is quite robust w.r.t. the underlying rounding subroutine (2) this simple strategy of nearest rounding on expert frequency is often sufficient to yield excellent task performance. More detailed discussion on different rounding subroutines is included in Appendix G.2.

MoE training & inference quality. This simple algorithm guarantees that for each expert, the maximum deviation from token-choice routing is at most 1 tile. We find that this property has a surprisingly robust performance even under sparse MoE training regime and can serve as a substitute for token-choice under sparse MoE training settings, which is shown in Table 2. We also conduct an ablation study on the effect of microbatch size T and tile size T and tile quality of trained MoE model with TR in Table 7 and 8, and we find token rounding routing is generally robust when  $T_e/M_{\rm tile} \geq 2$ .

**MoE training throughput.** TR guarantees no tile quantization effects and in Section 6.3.3, we show that TR training throughput over vanilla TC top-K is consistently higher when in the highly sparse MoE training regime and can achieve 16% higher TFLOPS for the kernel runtime as we scale up E while keeping K constant.

## 6 Experiments

<span id="page-10-0"></span>![](_page_10_Figure_9.jpeg)

Figure 10: Peak activation memory usage per layer across different model scales (1.4B–120B) on H100 GPUs. MegaBlocks does not support small n. The benchmark configurations are listed in Table 9a. For "DeepGEMM++" and "DeepGEMM-pt", we only cache X, gathered  $X_e$ ,  $H_e$  for each expert e and routing metadata which is the minimum amount of activation memory required for backward computation without GEMM recomputation.

We evaluate SonicMoE's activation memory footprint (Section 6.1) and training throughput (Section 6.2) compared to other baseline MoE implementations. We also demonstrate the efficacy of the token rounding routing strategy and show that it is

<span id="page-10-1"></span> $<sup>^8</sup>$ For simplicity, we always use  $M_{\rm tile}$  as 128 in Table 2 and Figure 13.

possible to use token choice as a drop-in replacement after training with token rounding in Section [6.3.1.](#page-11-2) We also show that token rounding can maintain the training throughput under sparse MoE configuration in Section [6.3.3.](#page-14-1)

## <span id="page-11-0"></span>6.1 SonicMoE's activation memory

We demonstrate that SonicMoE achieves the lowest peak activation memory footprint for a single MoE layer as shown in Figure [10](#page-10-0) across all scales on H100 GPUs. For the 7B model with n = 256, our approach reduces memory usage by 45% compared to ScatterMoE, and more significantly compared to MoMoE. For 30B and 120B models, the gap becomes even wider: at 120B scale, our method saves more than 3GiB memory per layer compared to MoMoE. We also validate that SonicMoE's activation memory stays constant w.r.t. expert granularity as shown in Figure [1.](#page-1-0)

## <span id="page-11-1"></span>6.2 SonicMoE's training throughput

#### 6.2.1 Entire forward and backward throughput

Figure [11a](#page-12-0) reports the compute throughput of forward and backward pass of one MoE layer in various MoE training configurations on H100 GPUs. Across all model scales, our method consistently achieves the highest TFLOPS. For example, on a fine-grained 7B MoE model with n = 256, SonicMoE increases the TFLOPS by 43% on the forward pass compared to a highly optimized DeepGEMM baseline, and by 83% and 115% on the backward pass compared to ScatterMoE and MoMoE, respectively. SonicMoE also demonstrates speedup over DeepGEMM++ in the forward pass, which mainly arises from the gather X kernel and Ping-Pong scheduling. The effect of both features increases as the MoE becomes more fine-grained and thus SonicMoE's relative speedup over DeepGEMM++ becomes larger.

We further measure the real training throughput of a 7B MoE model with n = 256 with FSDP-2: SonicMoE on 64 H100s achieves 213 billion tokens per day, which achieves similar throughput to ScatterMoE on 96 H100s with 225 billion tokens per day. The throughput for this is measured using the lm-engine codebase[9](#page-11-3) (Mishra [2024\)](#page-17-15). We shard the model using ZeRO-3 within a single node (8x H100s) and replicate this sharded unit across nodes for this experiment.

Besides H100 GPUs, we also measure SonicMoE's compute throughput on B300 GPUs in Figure [11b.](#page-12-0) We mainly compare with DeepGEMM++ which is powered by DeepGEMM SM100 BF16 grouped GEMM kernels. SonicMoE still demonstrates an overall speedup over DeepGEMM++, and such speedup is more pronounced when we increase expert granularity similar to the trend on H100 GPUs. For example, when we increase the expert granularity d/n from 2 to 8 for 120B MoE training, SonicMoE achieves a greater relative speedup over DeepGEMM++ from 11.6% and 13.4% in forward and backward to 19.6% and 16.8% respectively. We also highlight that SonicMoE's forward pass has higher throughput than the triton official example, despite SonicMoE storing both pre- and post-activation to GMEM whereas triton official example is designed for inference and only stores post-activation A in the up-projection kernel.

In addition, we measure the training throughput of a single MoE layer with configurations from recent open-source MoEs in Figure [12a](#page-13-0) for H100 GPUs and in Figure [12b](#page-13-0) for B300 GPUs. On H100 GPUs, SonicMoE generally achieves more than 550 TFLOPS during both forward and backward pass, and consistently surpasses all baselines. We note that ScatterMoE, MoMoE, DeepGEMM-pt, and DeepGEMM++ all fail to run at the configuration for DeepSeek-V3.2-Exp, a 685B MoE model, while SonicMoE successfully runs on a single H100 GPU. We also note that SonicMoE's IO-aware kernel design can achieve a greater relative speedup (e.g. Qwen3-Next-80B-A3B-Thinking) for sparse and fine-grained MoEs.

On B300 GPUs, SonicMoE generally achieves more than 1100 TFLOPS during both forward and backward pass. For the OLMoE-sized 7B MoE models, SonicMoE demonstrates 25% and 15% higher TFLOPS during forward and backward pass respectively than DeepGEMM++. SonicMoE also reaches 11.8% higher TFLOPS for forward pass than the triton official example.

## 6.3 Token rounding

#### <span id="page-11-2"></span>6.3.1 Token rounding's general task evaluation

In this section, we assess the quality of trained MoEs using our token rounding ("TR") algorithm. We use TR for training and during evaluation we switch to token-choice top-K ("TC top-K") routing. This assesses the capability of replacement of TR with TC after training.[10](#page-11-4) We use the OLMoE codebase and construct MoE models with OLMoE base architecture

<span id="page-11-4"></span><span id="page-11-3"></span><sup>9</sup><https://github.com/open-lm-engine/lm-engine>

<sup>10</sup>Token rounding is not a token-choice routing method which creates difficulty for autoregressive generation. Here we do not apply any adaptation and switch to vanilla token choice top-K routing during evaluation/validation.

<span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

(a) Forward & backward TFLOPS for different MoE kernels on H100 GPUs.

![](_page_12_Figure_2.jpeg)

(b) Forward & backward TFLOPS for different MoE kernels on B300 GPUs.

Figure 11: Forward & backward TFLOPS for different MoE kernels on H100 and B300 GPUs. The definition of "DeepGEMM++", "DeepGEMM-pt", and "triton ex." are the same as in Figure 5.

(Muennighoff et al. 2025). We use a deduplicated version of FineWeb-Edu (Ben Allal et al. 2024) for training all our models. More details are included in Appendix I.

We consistently use  $M_{\rm tile}=128$  in Table 2, and the round\_and\_sparsify subroutine always rounds the expert frequency to the nearest multiple of  $M_{\rm tile}$  ("NR-f", see Appendix G.2). We also use softmax renormalization for TR. We compare TR to token-choice (TC) top-K routing and expert-choice (EC) routing (Zhou et al. 2022). However, EC routing results in future token leakage causing problems for autoregressive generation resulting in a performance drop during evaluation (Raposo et al. 2024; Wang et al. 2024). To address this issue, we consider MoD's approach (Raposo et al. 2024) that trains an auxiliary router to predict the EC router's selection during inference<sup>11</sup>. This baseline is referred to as "EC (aux router)" in each subtable in Table 2. We also adapt EC routing to TC routing by finetuning a learned TC top-K router and compare its task performance against TR's task performance without any adaptation. This is the "EC (ft TC router)" baseline in Table 2. Finally, we consider a token dropping baseline in which we set the capacity of each expert as the largest multiple of  $M_{\rm tile}$  not exceeding its token frequencies and we discard the tokens with the lowest scores. This is the "TC (token drop)" baseline, and we note that this is equivalent as we always round down in TR.

**TR's train-test gap.** We validate TR's performance on a 0.5B (subtable 2a) and 1.4B (subtable 2d) MoE model. We then increase the MoE sparsity by either decreasing K while keeping E constant (from 2a to 2b, from 2d to 2e) or increasing

<span id="page-12-1"></span> $<sup>^{11}</sup>$ However, for MoE we are solving a harder E-label prediction problem instead of MoD's(Raposo et al. 2024) binary prediction problem. This is because EC router can activate arbitrary number of experts for each token, and we have to independently predict the label for each expert. This approach is likely not scalable for MoE as the prediction problem size scales with E.

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

![](_page_13_Figure_1.jpeg)

(a) Forward & backward TFLOPS of a single MoE layer on H100 GPUs. ScatterMoE, MoMoE, DeepGEMM-pt, and DeepGEMM++ all fail to run (either due to index overflow or CUDA OOM errors) for the DeepSeek-V3.2-Exp configuration.

![](_page_13_Figure_3.jpeg)

![](_page_13_Figure_4.jpeg)

(b) Forward & backward TFLOPS of a single MoE layer on B300 GPUs. Triton official example for MoE does not support K=10 for the Qwen3-Next-80B-A3B-Thinking configuration.

Figure 12: Forward & backward TFLOPS of a single MoE layer for different MoE kernels for different configurations ranging from 7B to 685B parameters on H100 and B300 GPUs. The MoE configurations from left to right adopt the model size of *OLMoE-1B-7B-0125* (Muennighoff et al. 2025), *gpt-oss-20b* (OpenAI 2025), *Kimi-Linear-48B-A3B-Base* (Zhang et al. 2025), *Qwen3-Next-80B-A3B-Thinking* (Qwen 2025), *Qwen3-235B-A22B-Thinking-2507* (Qwen 2025), and *DeepSeek-V3.2-Exp* (DeepSeek-AI 2025). For fair comparison, we do not consider shared experts and expert biases, and we always use TC top-K router with softmax scores.

E while keeping K constant (from 2a to 2c). Across these sparse MoE configurations, we consistently observe similar model quality between TR and TC. In fact, TR achieves slightly lower validation perplexity and higher or the same average accuracy under the extremely sparse MoE ( $K/E \le 1/32$ ) settings for the 2c and 2e. There is a noticeable discrepancy between EC and TC as the train and val PPL for EC can have  $\gtrsim 3$  gap for 2c,2d and 2e compared to TC and TR's usual  $\lesssim 0.3$  gap. TC finetuning is more effective than the auxiliary router to close this gap, but TR's task evaluation is still always better. In addition, when we compare TR with the token dropping baseline, we also find TR consistently yields lower validation perplexity, and has higher average task accuracy for 2a, 2c, 2e. In this case, TR can serve as an in-place substitute for TC during training.

#### **6.3.2** Ablation studies on token rounding routing

There are 3 variables that can affect the trained MoE quality with token rounding routing: (1) rounding subroutine round\_and\_sparsify (2) microbatch size T, and (3) tile size  $M_{\rm tile}$  for rounding. We analyze their impacts:

- Choice of rounding subroutine: In Table 6, we assess the choice of different routing subroutines to train MoEs using TR. We find that our token rounding algorithm in general is robust to the specific rounding subroutines, and nearest rounding expert frequency to multiples of M<sub>tile</sub> ("NR-f" in Table 6) is often sufficient for providing an excellent downstream task performance despite its simplicity. Therefore, we choose NR-f as the default rounding subroutine.
- Effect of microbatch size T and tile size  $M_{\rm tile}$ : The token rounding is applied on the microbatch level so varying the microbatch size T will result in different qualitative results for TR. This also holds true for EC routing. For example, EC over sequence will result in different model quality as EC over a text segment. Nevertheless, in Table 7, we find that TR preserves its trained MoE quality when  $\bar{T}_e/M_{\rm tile} \geq 2$ , and even if  $\bar{T}_e/M_{\rm tile} = 1$  (the last row in both subtables), the trained MoE inference quality is still better than training with EC and finetuning with TC top-K routing. Similarly in Table 8, we can find that TR is generally robust w.r.t.  $M_{\rm tile}$  when  $\bar{T}_e/M_{\rm tile} \geq 2$ . However, when  $\bar{T}_e/M_{\rm tile} = 1$  there is a noticeable degradation compared to TC baseline but the model quality is still better than the EC baseline.

<span id="page-14-0"></span>Table 2: Comparison of different routing methods' task evaluation. "Train" and "Val" refer to the perplexity towards the end of training and on the validation set respectively. The next 11 columns are downstream tasks evaluated at the end of training and we report the accuracy for each. "Avg" is the mean accuracy across these 11 downstream tasks. We use TC top-K routing for TR, token dropping, and EC baselines when evaluating validation perplexity and task performance.  $\bar{T}_e$  represents the average number of received tokens in each microbatch per expert.

|                   |       | (a)   | 0.5B pa          | rams, 20 | )B toker | ıs, 8/64 a | ctivated  | $(\bar{T}_e = 4)$           | 4096, M          | $t_{\rm tile} = 12$ | 8)    |      |      |      |
|-------------------|-------|-------|------------------|----------|----------|------------|-----------|-----------------------------|------------------|---------------------|-------|------|------|------|
| Method            | Train | Val   | Wino             | SIQA     | SciQ     | PIQA       | OBQA      | HS                          | COPA             | CSQA                | BoolQ | ArcE | ArcC | Avg  |
| TR                | 15.91 | 15.94 | 51.9             | 41.3     | 80.8     | 65.5       | 35.0      | 38.7                        | 63.0             | 31.2                | 61.4  | 58.9 | 27.1 | 50.4 |
| TC top-K          | 16.04 | 16.01 | 51.0             | 41.4     | 79.2     | 65.5       | 31.6      | 38.4                        | 66.0             | 31.5                | 60.2  | 57.5 | 25.7 | 49.8 |
| TC (token drop)   | 16.52 | 16.46 | 51.1             | 41.1     | 79.5     | 64.6       | 30.2      | 37.3                        | 63.0             | 31.8                | 58.2  | 57.9 | 28.4 | 49.4 |
| EC                | 16.25 | 17.23 | 51.0             | 41.0     | 78.3     | 63.8       | 33.4      | 37.5                        | 69.0             | 31.4                | 54.4  | 56.1 | 29.4 | 49.7 |
| EC (aux router)   | 16.25 | 17.40 | 52.6             | 41.5     | 77.3     | 64.4       | 31.4      | 37.5                        | 65.0             | 30.9                | 55.4  | 55.8 | 30.4 | 49.3 |
| EC (ft TC router) | 16.34 | 16.40 | 49.3             | 41.4     | 78.0     | 64.4       | 33.4      | 37.5                        | 67.0             | 30.8                | 56.1  | 55.4 | 29.4 | 49.3 |
|                   |       | (b    | ) <b>0.5B</b> pa | arams, 4 | 0B toke  | ns, 2/64   | activated | $(\bar{T}_e =$              | 512, $M_{\rm t}$ | $_{\rm tile} = 128$ | 3)    |      |      |      |
| TR                | 16.22 | 15.92 | 51.4             | 41.6     | 78.4     | 65.4       | 31.6      | 38.1                        | 65.0             | 31.0                | 61.1  | 57.4 | 29.1 | 50.0 |
| TC top-K          | 16.34 | 15.94 | 51.0             | 41.9     | 78.5     | 64.8       | 33.0      | 38.1                        | 67.0             | 30.8                | 54.7  | 55.8 | 30.1 | 49.6 |
| TC (token drop)   | 16.44 | 16.10 | 51.1             | 41.4     | 78.7     | 64.9       | 31.6      | 38.0                        | 62.0             | 32.8                | 61.9  | 58.9 | 30.8 | 50.2 |
| EC                | 16.83 | 18.61 | 49.6             | 41.4     | 79.1     | 64.4       | 33.4      | 36.9                        | 62.0             | 32.8                | 60.2  | 55.8 | 29.1 | 49.5 |
| EC (aux router)   | 16.80 | 21.80 | 50.0             | 40.9     | 75.2     | 63.7       | 28.2      | 35.2                        | 61.0             | 31.5                | 57.2  | 53.3 | 24.7 | 47.4 |
| EC (ft TC router) | 16.81 | 16.98 | 50.0             | 41.7     | 79.7     | 64.9       | 31.6      | 36.8                        | 63.0             | 32.1                | 60.7  | 54.6 | 27.4 | 49.3 |
|                   |       | (c)   | 1.8B pa          | rams, 40 | )B toker | ns, 8/256  | activated | $\mathbf{I}$ $(\bar{T}_e =$ | 512, M           | $t_{\rm tile} = 12$ | 8)    |      |      |      |
| TR                | 13.34 | 13.10 | 53.4             | 42.1     | 81.7     | 69.6       | 35.2      | 45.3                        | 70.0             | 33.2                | 61.4  | 63.0 | 33.4 | 53.5 |
| TC top-K          | 13.51 | 13.12 | 50.1             | 42.9     | 81.3     | 69.8       | 33.8      | 45.2                        | 71.0             | 34.1                | 56.7  | 64.6 | 31.1 | 52.8 |
| TC (token drop)   | 13.62 | 13.19 | 55.4             | 41.6     | 82.2     | 68.6       | 34.8      | 45.0                        | 69.0             | 34.0                | 54.4  | 63.5 | 31.4 | 52.7 |
| EC                | 14.92 | 19.82 | 51.9             | 40.8     | 77.7     | 65.8       | 30.0      | 39.8                        | 67.0             | 30.9                | 60.7  | 56.0 | 28.4 | 49.9 |
| EC (aux router)   | 14.94 | 18.01 | 50.6             | 41.8     | 79.8     | 65.8       | 31.6      | 39.3                        | 62.0             | 31.8                | 59.7  | 55.8 | 29.8 | 49.8 |
| EC (ft TC router) | 14.81 | 15.01 | 52.7             | 41.1     | 79.6     | 66.9       | 30.6      | 40.2                        | 66.0             | 31.9                | 60.5  | 57.2 | 30.8 | 50.7 |
|                   |       | (d)   | 1.4B par         | rams, 50 | B token  | s, 8/128 a | activated | $(\bar{T}_e =$              | 2048, N          | $I_{\rm tile} = 12$ | 28)   |      |      |      |
| TR                | 13.51 | 13.28 | 52.6             | 42.6     | 81.5     | 69.6       | 33.6      | 45.4                        | 67.0             | 34.8                | 57.3  | 63.7 | 28.1 | 52.4 |
| TC top-K          | 13.50 | 13.32 | 51.8             | 41.7     | 81.5     | 69.3       | 32.4      | 45.3                        | 68.0             | 34.5                | 56.6  | 63.2 | 28.4 | 52.1 |
| TC (token drop)   | 13.52 | 13.30 | 51.8             | 42.2     | 84.1     | 69.2       | 34.4      | 45.2                        | 70.0             | 35.1                | 61.2  | 64.2 | 31.4 | 53.5 |
| EC                | 14.41 | 17.37 | 51.4             | 42.0     | 79.7     | 66.3       | 32.2      | 40.7                        | 64.0             | 31.8                | 59.0  | 57.4 | 27.4 | 50.2 |
| EC (aux router)   | 14.34 | 26.96 | 49.8             | 41.5     | 79.1     | 63.1       | 30.2      | 37.6                        | 61.0             | 31.0                | 60.9  | 46.7 | 25.1 | 47.8 |
| EC (ft TC router) | 14.67 | 14.90 | 51.9             | 41.8     | 80.1     | 66.4       | 32.6      | 41.1                        | 65.0             | 32.4                | 57.7  | 57.7 | 27.8 | 50.4 |
|                   |       | (e)   | 1.4B pai         | rams, 10 | 0B toke  | ns, 2/128  | activate  | $\mathbf{d}$ $(\bar{T}_e =$ | = 512, N         | $I_{\rm tile} = 12$ | 28)   |      |      |      |
| TR                | 13.31 | 13.22 | 52.8             | 41.8     | 80.8     | 68.7       | 33.0      | 43.4                        | 67.0             | 33.6                | 60.2  | 60.7 | 29.8 | 52.0 |
| TC top-K          | 13.50 | 13.32 | 51.3             | 42.0     | 83.2     | 68.2       | 34.0      | 43.4                        | 66.0             | 35.4                | 57.9  | 61.6 | 29.4 | 52.0 |
| TC (token drop)   | 13.35 | 13.29 | 50.0             | 42.2     | 81.7     | 68.3       | 31.2      | 43.3                        | 66.0             | 34.3                | 56.6  | 59.5 | 30.8 | 51.3 |
| EC                | 14.08 | 24.79 | 51.5             | 41.7     | 81.0     | 66.1       | 33.2      | 40.6                        | 64.0             | 34.0                | 56.3  | 56.5 | 27.4 | 50.2 |
| EC (aux router)   | 14.01 | 37.52 | 49.7             | 40.2     | 73.6     | 57.5       | 27.6      | 33.2                        | 61.0             | 27.8                | 58.8  | 45.2 | 24.2 | 45.3 |
| EC (ft TC router) | 14.24 | 14.75 | 52.2             | 42.6     | 79.4     | 65.7       | 32.8      | 40.8                        | 64.0             | 34.9                | 58.3  | 57.2 | 27.1 | 50.5 |

#### <span id="page-14-1"></span>6.3.3 Token rounding's training throughput

In Figure 13, we benchmark the token rounding's MoE main kernel runtime (without router) against top-K token choice routing. We focus on the iso-FLOPs setting by keeping T, n and K constant. We linearly increase the number of experts E while keeping K constant to increase MoE sparsity. As we linearly increase E, we observe a drop in TFLOPS for token-choice routing. This is due to the (1) tile quantization effect as the wasted FLOPs spent on padding roughly linearly increases with the MoE sparsity as shown in Figure 8 and (2) the linearly increased IO due to more expert weights. We observe a drop in FLOPs for both TC and TR as we increase E, but the drop is more pronounced for TC as shown in Figure 13.

For the  $3^{\rm rd}$  and  $4^{\rm th}$  column in the top row in Figure 13, an MoE model with 128 experts (K/E=1/64) and n=1k with token rounding routing achieves 16.5% model TFLOPS<sup>12</sup> improvement in forward and 6.1% in backward, resulting in an end-to-end improvement of 9.4%. For the  $3^{\rm rd}$  and  $4^{\rm th}$  column on the bottom row in Figure 13, when we have a MoE with 256 experts (K/E=1/64), token rounding routing achieves a 25.7% TFLOPS improvement in forward and 11.8% in backward resulting in an end-to-end improvement of 15.9%. In general, we observe that as we move to larger intermediate sizes

<span id="page-14-2"></span> $<sup>^{12}</sup>$ Note that the consumed FLOPs are calculated from  $(6+12)dn(\sum_e f_e)$  (note that  $\sum_e f_e = TK$  for TC top-K routing) as model FLOPs rather than hardware FLOPs. The speedup behind TR is to preserve the model FLOPs consumption on expectation but save the hardware FLOPs consumption by removing padding wastes, which in turn leads to model TFLOPS speedup.

<span id="page-15-0"></span>![](_page_15_Figure_0.jpeg)

Figure 13: Forward & backward model TFLOPS for SonicMoE MoE kernels with different routing methods. We compare TR equipped with "nearest rounding to  $M_{\rm tile}$ -multiples via expert frequency" subroutine against TC top-K routing. Configuration details are in Appendix H.

#### (more compute-bound) and higher MoE sparsity, the gap between TR and TC top-K becomes larger.

This trend also holds with configurations from recent open-source MoEs in Figure 14. When we equip SonicMoE's MoE kernel with TR router instead of TC top-K router, we observe a larger relative speedup for highly sparse MoEs such as Qwen3-Next-80B-A3B-Thinking (K/E=10/512), where TR achieves 19.6% and 7.9% speedup over TC top-K router during forward and backward pass respectively.

<span id="page-15-1"></span>![](_page_15_Figure_4.jpeg)

Figure 14: Forward & backward TFLOPS of a single MoE layer of SonicMoE equipped with different routing methods for different configurations ranging from 7B to 685B parameters on H100. The MoE configurations from left to right adopt the model size of OLMoE-1B-7B-0125, gpt-oss-20b, Kimi-Linear-48B-A3B-Base, Qwen3-Next-80B-A3B-Thinking, Qwen3-235B-A22B-Thinking-2507, and DeepSeek-V3.2-Exp (configurations identical to Figure 12a). We compare TR equipped with "nearest rounding to  $M_{\rm tile}$ -multiples via expert frequency" subroutine against TC top-K routing.

## 7 Conclusion

We present SonicMoE, a co-design solution that jointly optimizes MoE architecture and GPU kernels to address the training challenges of granular and sparse MoEs. Our contributions include: (1) a memory-efficient algorithm that minimizes activation size as MoEs become more fine-grained, (2) GPU kernels that overlap IO with computation for throughput improvement, and (3) tile-aware token rounding that yields additional speedup without quality loss. Future directions include extending to low-precision and microscaling formats (FP8, MXFP8, MXFP4) for further memory savings, and overlapping communication with computation in distributed settings like expert parallelism. We envision future model architecture designs that optimize for quality per compute hour rather than just quality per FLOP—jointly considering algorithmic and hardware efficiency.

## Acknowledgment

We gratefully acknowledge the support of Schmidt Sciences AI2050 fellowship, the Google ML and Systems Junior Faculty Awards, and the Google Research Scholar program. We also thank the Princeton Language Intelligence program for the computing resources support. We thank Shawn Tan for his generous support on our experiments. We also thank Songlin Yang, Yilong Zhao, Bharat Runwal, Xinyu Yang, Andrew Sheinberg, Lijie Yang, Yongye Zhu, Zhuoqing Song and numerous anonymous reviewers for providing valuable feedback.

## References

- <span id="page-16-9"></span>[1] K. E. Batcher. "Sorting networks and their applications". In: *Proceedings of the April 30–May 2, 1968, Spring Joint Computer Conference*. AFIPS '68 (Spring). Atlantic City, New Jersey: Association for Computing Machinery, 1968, 307–314. ISBN: 9781450378970. DOI: [10.1145/1468075.1468121](https://doi.org/10.1145/1468075.1468121). URL: [https://doi.org/10.1145/](https://doi.org/10.1145/1468075.1468121) [1468075.1468121](https://doi.org/10.1145/1468075.1468121).
- <span id="page-16-6"></span>[2] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. *SmolLM-Corpus*. 2024. URL: <https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus>.
- <span id="page-16-2"></span>[3] Vincent-Pierre Berges, Barlas Oguz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi Ghosh. "Memory ˘ layers at scale". In: *arXiv preprint arXiv:2412.09764* (2024).
- <span id="page-16-16"></span>[4] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. "PIQA: Reasoning about Physical Commonsense in Natural Language". In: *Thirty-Fourth AAAI Conference on Artificial Intelligence*. 2020.
- <span id="page-16-0"></span>[5] Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. "Unified scaling laws for routed language models". In: *International conference on machine learning*. PMLR. 2022, pp. 4057–4086.
- <span id="page-16-17"></span>[6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions". In: *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. 2019, pp. 2924–2936.
- <span id="page-16-18"></span>[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge". In: *arXiv:1803.05457v1* (2018).
- <span id="page-16-13"></span>[8] A Feder Cooper, Wentao Guo, Duc Khiem Pham, Tiancheng Yuan, Charlie Ruan, Yucheng Lu, and Christopher M De Sa. "Coordinating distributed example orders for provably accelerated training". In: *Advances in Neural Information Processing Systems* 36 (2023), pp. 56198–56210.
- <span id="page-16-8"></span>[9] NVIDIA Corporation. 2025. URL: [https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass\\_](https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass_3x_backwards_compatibility.html) [3x\\_backwards\\_compatibility.html](https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass_3x_backwards_compatibility.html).
- <span id="page-16-3"></span>[10] Bobby Costin, Timor Averbuch, Dhruv Pai, Nathan Chen, and Ben Keigwin. "MoMoE: Memory optimized Mixture of Experts". In: *Tilde Research Blog* (July 2025). Blog post. URL: [https://www.tilderesearch.com/blog/](https://www.tilderesearch.com/blog/momoe) [momoe](https://www.tilderesearch.com/blog/momoe).
- <span id="page-16-7"></span>[11] DeepSeek-AI. *DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention*. 2025.
- <span id="page-16-1"></span>[12] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, and et al. *DeepSeek-V3 Technical Report*. 2024. arXiv: [2412.19437](https://arxiv.org/abs/2412.19437) [\[cs.CL\]](https://arxiv.org/abs/2412.19437). URL: <https://arxiv.org/abs/2412.19437>.
- <span id="page-16-10"></span>[13] Bert Dobbelaere. 2025. URL: [https://bertdobbelaere.github.io/sorting\\_networks.html](https://bertdobbelaere.github.io/sorting_networks.html).
- <span id="page-16-14"></span>[14] Raaz Dwivedi and Lester Mackey. "Kernel thinning". In: *Journal of Machine Learning Research* 25.152 (2024), pp. 1–77.
- <span id="page-16-12"></span>[15] *Efficient GEMM in CUDA*. 2025. URL: [https://docs.nvidia.com/cutlass/media/docs/cpp/](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html#hopper-warp-specialization) [efficient\\_gemm.html#hopper-warp-specialization](https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html#hopper-warp-specialization).
- <span id="page-16-15"></span>[16] William Fedus, Barret Zoph, and Noam Shazeer. *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. 2022. arXiv: [2101.03961 \[cs.LG\]](https://arxiv.org/abs/2101.03961). URL: [https://arxiv.org/abs/2101.](https://arxiv.org/abs/2101.03961) [03961](https://arxiv.org/abs/2101.03961).
- <span id="page-16-5"></span>[17] Rong Fu, Weihan Cao, Jianfei Gao, Minxi Jin, Hui Wang, et al. "TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper". In: *ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models*. 2025.
- <span id="page-16-4"></span>[18] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. "Megablocks: Efficient sparse training with mixtureof-experts". In: *Proceedings of Machine Learning and Systems* 5 (2023), pp. 288–304.
- <span id="page-16-11"></span>[19] IBM Granite. *Granite 3.1 Language Models*. [https://github.com/ibm- granite/granite- 3.1](https://github.com/ibm-granite/granite-3.1-language-models) [language-models](https://github.com/ibm-granite/granite-3.1-language-models). GitHub repository. 2024.

- <span id="page-17-12"></span>[20] Horace He and Thinking Machines. *Defeating Nondeterminism in LLM Inference*. 2025. URL: [https://thinkingm](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl)achines. [ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl).
- <span id="page-17-4"></span>[21] Xu Owen He. "Mixture of a million experts". In: *arXiv preprint arXiv:2407.04153* (2024).
- <span id="page-17-10"></span>[22] Quzhe Huang, Zhenwei An, Nan Zhuang, Mingxu Tao, Chen Zhang, Yang Jin, Kun Xu, Liwei Chen, Songfang Huang, and Yansong Feng. "Harder Task Needs More Experts: Dynamic Routing in MoE Models". In: *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. 2024, pp. 12883–12895.
- <span id="page-17-5"></span>[23] Zihao Huang, Qiyang Min, Hongzhi Huang, Yutao Zeng, Defa Zhu, Ran Guo, et al. "Ultra-Sparse Memory Network". In: *The Thirteenth International Conference on Learning Representations*. 2025.
- <span id="page-17-22"></span>[24] Matt Gardner Johannes Welbl Nelson F. Liu. "Crowdsourcing Multiple Choice Science Questions". In: 2017.
- <span id="page-17-0"></span>[25] Team Kimi, Yifan Bai, Yiping Bao, Guanduo Chen, and et al. *Kimi K2: Open Agentic Intelligence*. 2025. arXiv: [2507.20534 \[cs.LG\]](https://arxiv.org/abs/2507.20534). URL: <https://arxiv.org/abs/2507.20534>.
- <span id="page-17-1"></span>[26] Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pioro, Michał Krutul, Szymon Antoniak, Kamil ´ Ciebiera, Krystian Krol, Tomasz Odrzyg ´ o´zd´ z, Piotr Sankowski, et al. "Scaling laws for fine-grained mixture of ´ experts". In: *arXiv preprint arXiv:2402.07871* (2024).
- <span id="page-17-8"></span>[27] Chuck L Lawson, Richard J. Hanson, David R Kincaid, and Fred T. Krogh. "Basic linear algebra subprograms for Fortran usage". In: *ACM Transactions on Mathematical Software (TOMS)* 5.3 (1979), pp. 308–323.
- <span id="page-17-18"></span>[28] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding". In: *International Conference on Learning Representations*. 2024.
- <span id="page-17-21"></span>[29] Yucheng Lu, Wentao Guo, and Christopher M De Sa. "Grab: Finding provably better data permutations than random reshuffling". In: *Advances in Neural Information Processing Systems* 35 (2022), pp. 8969–8981.
- <span id="page-17-20"></span>[30] Microsoft. *Announcing the Availability of PHI-3.5 MoE in Azure AI Studio and GitHub*. [https://techcommunity](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-the-availability-of-phi-3-5-moe-in-azure-ai-studio-and-github/4256278). [microsoft.com/blog/azure-ai-foundry-blog/announcing-the-availability-of-phi-](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-the-availability-of-phi-3-5-moe-in-azure-ai-studio-and-github/4256278)[3-5-moe-in-azure-ai-studio-and-github/4256278](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/announcing-the-availability-of-phi-3-5-moe-in-azure-ai-studio-and-github/4256278). Microsoft Tech Community. 2024.
- <span id="page-17-23"></span>[31] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering". In: *EMNLP*. 2018.
- <span id="page-17-15"></span>[32] Mayank Mishra. *LM Engine: A Hyper-Optimized Library for Pretraining and Finetuning*. June 2024. URL: [https:](https://github.com/ibm/lm-engine) [//github.com/ibm/lm-engine](https://github.com/ibm/lm-engine).
- <span id="page-17-19"></span>[33] Mistral. *Mixtral of Experts: A high quality Sparse Mixture-of-Experts.* [https://mistral.ai/news/mixtral](https://mistral.ai/news/mixtral-of-experts)[of-experts](https://mistral.ai/news/mixtral-of-experts). Mistral AI News. 2024.
- <span id="page-17-13"></span>[34] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. "OLMoE: Open Mixture-of-Experts Language Models". In: *The Thirteenth International Conference on Learning Representations*. 2025.
- <span id="page-17-7"></span>[35] NVIDIA. *NVIDIA H100 Tensor Core GPU Architecture: Exceptional Performance, Scalability, and Security for the Data Center*. Whitepaper V1.01. Grace Hopper "Hopper" Architecture. NVIDIA, 2022. URL: [https://www.](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) [advancedclustering . com / wp - content / uploads / 2022 / 03 / gtc22 - whitepaper - hopper .](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) [pdf](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf).
- <span id="page-17-9"></span>[36] NVIDIA. *CUTLASS: CUDA Templates for Linear Algebra Subroutines*. [https://github.com/NVIDIA/](https://github.com/NVIDIA/cutlass) [cutlass](https://github.com/NVIDIA/cutlass). Version 4.2.0, Accessed: 2025-09-19. 2025.
- <span id="page-17-14"></span>[37] NVIDIA. *NVIDIA Blackwell Architecture Technical Brief*. Whitepaper. Blackwell Architecture. NVIDIA, 2025. URL: <https://resources.nvidia.com/en-us-blackwell-architecture?ncid=no-ncid>.
- <span id="page-17-6"></span>[38] NVIDIA. *NVIDIA CUTLASS Documentation*. 2025. URL: [https://docs.nvidia.com/cutlass/media/](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) [docs/pythonDSL/cute\\_dsl\\_general/dsl\\_introduction.html](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html).
- <span id="page-17-3"></span>[39] OpenAI. *gpt-oss-120b & gpt-oss-20b Model Card*. 2025. arXiv: [2508.10925 \[cs.CL\]](https://arxiv.org/abs/2508.10925). URL: [https://arxiv.](https://arxiv.org/abs/2508.10925) [org/abs/2508.10925](https://arxiv.org/abs/2508.10925).
- <span id="page-17-17"></span>[40] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach DeVito, Martin Raison, ¨ Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. 2019. arXiv: [1912.01703 \[cs.LG\]](https://arxiv.org/abs/1912.01703). URL: [https:](https://arxiv.org/abs/1912.01703) [//arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703).
- <span id="page-17-16"></span>[41] Team Qwen. *Qwen3 Technical Report*. 2025. arXiv: [2505.09388 \[cs.CL\]](https://arxiv.org/abs/2505.09388). URL: [https://arxiv.org/](https://arxiv.org/abs/2505.09388) [abs/2505.09388](https://arxiv.org/abs/2505.09388).
- <span id="page-17-2"></span>[42] QwenLM. *Qwen3: Think Deeper, Act Faster*. <https://qwenlm.github.io/blog/qwen3/>. Official Blog. 2025.
- <span id="page-17-11"></span>[43] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. "Mixture-of-depths: Dynamically allocating compute in transformer-based language models". In: *arXiv preprint arXiv:2404.02258* (2024).

- <span id="page-18-9"></span>[44] Colfax Research. *CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs*. Accessed: 2025-09-21. 2024. URL: [https://research.colfax- intl.com/cutlass- tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)[writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/).
- <span id="page-18-18"></span>[45] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. "Choice of plausible alternatives: An evaluation of commonsense causal reasoning". In: *2011 AAAI Spring Symposium Series*. 2011.
- <span id="page-18-15"></span>[46] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. "WinoGrande: An Adversarial Winograd Schema Challenge at Scale". In: *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 34. 2020, pp. 8732–8740.
- <span id="page-18-16"></span>[47] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. "SocialIQA: Commonsense Reasoning about Social Interactions". In: *Conference on Empirical Methods in Natural Language Processing*. 2019.
- <span id="page-18-4"></span>[48] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*. 2024. arXiv: [2407.08608 \[cs.LG\]](https://arxiv.org/abs/2407.08608). URL: [https:](https://arxiv.org/abs/2407.08608) [//arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608).
- <span id="page-18-0"></span>[49] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. 2017. arXiv: [1701.06538](https://arxiv.org/abs/1701.06538) [\[cs.LG\]](https://arxiv.org/abs/1701.06538). URL: <https://arxiv.org/abs/1701.06538>.
- <span id="page-18-5"></span>[50] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. "Megatronlm: Training multi-billion parameter language models using model parallelism". In: *arXiv preprint arXiv:1909.08053* (2019).
- <span id="page-18-19"></span>[51] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge". In: *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*. Minneapolis, Minnesota: Association for Computational Linguistics, June 2019, pp. 4149–4158. DOI: [10.18653/v1/N19-1421](https://doi.org/10.18653/v1/N19-1421). arXiv: [1811.00937 \[cs\]](https://arxiv.org/abs/1811.00937). URL: <https://aclanthology.org/N19-1421>.
- <span id="page-18-3"></span>[52] Shawn Tan, Yikang Shen, Rameswar Panda, and Aaron Courville. "Scattered mixture-of-experts implementation". In: *arXiv preprint arXiv:2403.08245* (2024).
- <span id="page-18-13"></span>[53] Databricks The Mosaic Research Team. *Introducing DBRX: A New State-of-the-Art Open LLM*. [https://www.](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) [databricks.com/blog/introducing-dbrx-new-state-art-open-llm](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm). Databricks Blog, March 27, 2024. 2024.
- <span id="page-18-2"></span>[54] Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. "Towards greater leverage: Scaling laws for efficient mixture-of-experts language models". In: *arXiv preprint arXiv:2507.17702* (2025).
- <span id="page-18-10"></span>[55] Philippe Tillet, H. T. Kung, and David Cox. "Triton: an intermediate language and compiler for tiled neural network computations". In: *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*. MAPL 2019. Phoenix, AZ, USA: Association for Computing Machinery, 2019, 10–19. ISBN: 9781450367196. DOI: [10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973). URL: [https://doi.org/10.1145/3315508.](https://doi.org/10.1145/3315508.3329973) [3329973](https://doi.org/10.1145/3315508.3329973).
- <span id="page-18-1"></span>[56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. *Attention Is All You Need*. 2017. arXiv: [1706.03762 \[cs.CL\]](https://arxiv.org/abs/1706.03762). URL: [https://arxiv.org/](https://arxiv.org/abs/1706.03762) [abs/1706.03762](https://arxiv.org/abs/1706.03762).
- <span id="page-18-6"></span>[57] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. "Auxiliary-loss-free load balancing strategy for mixture-of-experts". In: *arXiv preprint arXiv:2408.15664* (2024).
- <span id="page-18-11"></span>[58] Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, et al. "TileLang: A Composable Tiled Programming Model for AI Systems". In: *arXiv preprint arXiv:2504.17577* (2025).
- <span id="page-18-7"></span>[59] Less Wright and Adnan Hoque. *Deep dive on Cutlass Ping-Pong Gemm Kernel*. 2024. URL: [https://pytorch.](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) [org/blog/cutlass-ping-pong-gemm-kernel/](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/).
- <span id="page-18-8"></span>[60] Less Wright and Adnan Hoque. *Deep Dive on CUTLASS Ping-Pong GEMM Kernel*. Accessed: 2025-09-21. 2024. URL: <https://docs.pytorch.org/blog/cutlass-ping-pong-gemm-kernel/>.
- <span id="page-18-12"></span>[61] Xi Xie, Yuebo Luo, Hongwu Peng, and Caiwen Ding. "RTop-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on GPUs". In: *The Thirteenth International Conference on Learning Representations*. 2025. URL: <https://openreview.net/forum?id=PHg4rAXFVH>.
- <span id="page-18-17"></span>[62] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. "HellaSwag: Can a Machine Really Finish Your Sentence?" In: *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. 2019.
- <span id="page-18-14"></span>[63] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. "Glm-4.5: Agentic, reasoning, and coding (arc) foundation models". In: *arXiv preprint arXiv:2508.06471* (2025).

- <span id="page-19-3"></span>[64] Zhiyuan Zeng, Qipeng Guo, Zhaoye Fei, Zhangyue Yin, Yunhua Zhou, Linyang Li, Tianxiang Sun, Hang Yan, Dahua Lin, and Xipeng Qiu. "Turn Waste into Worth: Rectifying Top-k Router of MoE". In: *EMNLP*. 2024.
- <span id="page-19-1"></span>[65] Zihao Zeng, Yibo Miao, Hongcheng Gao, Hao Zhang, and Zhijie Deng. "AdaMoE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models". In: *Findings of the Association for Computational Linguistics: EMNLP 2024*. Ed. by Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 6223–6235. DOI: [10.18653/v1/2024.findings-emnlp.361](https://doi.org/10.18653/v1/2024.findings-emnlp.361). URL: <https://aclanthology.org/2024.findings-emnlp.361/>.
- <span id="page-19-5"></span>[66] Yu Zhang, Zongyu Lin, Xingcheng Yao, and et al. *Kimi Linear: An Expressive, Efficient Attention Architecture*. 2025. arXiv: [2510.26692 \[cs.CL\]](https://arxiv.org/abs/2510.26692).
- <span id="page-19-0"></span>[67] Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, and Y.X. Wei. "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures". In: *Proceedings of the 52nd Annual International Symposium on Computer Architecture*. ISCA '25. New York, NY, USA: Association for Computing Machinery, 2025, 1731–1745. ISBN: 9798400712616. DOI: [10.1145/3695053.3731412](https://doi.org/10.1145/3695053.3731412). URL: <https://doi.org/10.1145/3695053.3731412>.
- <span id="page-19-4"></span>[68] Chenggang Zhao, Liang Zhao, Jiashi Li, and Zhean Xu. *DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling*. 2025. URL: <https://github.com/deepseek-ai/DeepGEMM>.
- <span id="page-19-6"></span>[69] Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. *DeepEP: an efficient expert-parallel communication library*. [https://github.com/deepseek](https://github.com/deepseek-ai/DeepEP)[ai/DeepEP](https://github.com/deepseek-ai/DeepEP). 2025.
- <span id="page-19-2"></span>[70] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. "Mixture-of-experts with expert choice routing". In: *Advances in Neural Information Processing Systems* 35 (2022), pp. 7103–7114.
- <span id="page-19-7"></span>[71] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. "St-moe: Designing stable and transferable sparse expert models". In: *arXiv preprint arXiv:2202.08906* (2022).

## Appendix

We provide a table listing all notations and their explanations in Table [3.](#page-20-0) In Section [B,](#page-21-0) we compare SonicMoE's kernel design with other open-source MoE kernel designs. In Section [C,](#page-22-0) we elaborate on SonicMoE's computational path for dS and dH that does not use Y and dY . In Section [C.1,](#page-22-1) we justify SonicMoE's computational path for dS is both activation memory and computationally-efficient. In Section [D,](#page-23-1) We examine SonicMoE's top-K sorting kernel. In Table [4,](#page-24-0) we provide a trending overview for open-source frontier MoE models. We present SonicMoE's expert aggregation strategy in Figure [17.](#page-24-1) Figure [16](#page-24-2) illustrates that on Hopper GPUs, asynchronous TMA store (top) has higher memory bandwidth and can naturally overlap with TensorCore MMA. In addition, SonicMoE's up-projection backward is included in Algorithm [5.](#page-23-0) In Section [F,](#page-25-0) we present ablation studies of training throughput for SonicMoE's MoE computation kernels to examine the impact of each design choice made for SonicMoE. In Section [G,](#page-29-1) we assess the quality improvements of MoE models trained by varying expert granularity. We then focus on various ablation studies on our token rounding routing algorithm to assess the quality difference of the trained MoE models from the choice of rounding subroutine. We also study the effect of microbatch size T and tile size Mtile on token rounding. In Section [H,](#page-33-3) we describe the configurations for benchmarking the memory and training throughput. In Section [I,](#page-33-2) we include the details of model training and the evaluation setup.

## A Notations

<span id="page-20-0"></span>In Table [3,](#page-20-0) we describe the notations used in this paper.

Table 3: Notations and their explanations

| Notations                | Explanation                                                                                                                                                                   |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| T                        | number of tokens in a microbatch                                                                                                                                              |
| d                        | model embedding dimension (hidden size)                                                                                                                                       |
| n                        | each expert's intermediate dimension                                                                                                                                          |
| E                        | total number of experts                                                                                                                                                       |
| K                        | number of activated experts                                                                                                                                                   |
| ρ                        | ρ = K/E represents MoE activation ratio                                                                                                                                       |
| T¯e                      | T¯e<br>= Ee∈[E]<br>[Te] = T ρ represents the expected number of received tokens in each microbatch by expert e                                                                |
| G                        | G = d/n represents the MoE expert granularity. Greater G means a MoE is more fine-grained                                                                                     |
| M, N, K                  | Dimensions for GEMM in CUTLASS. We define A ∈ RM×K,<br>B ∈ RK×N, and<br>C ∈ RM×N for<br>AB = C                                                                                |
| Mtile, Ntile, Ktile      | tile size of M, N, K dimension for a single GEMM tile                                                                                                                         |
| Re                       | tile quantization residue Re := Te mod Mtile                                                                                                                                  |
| X                        | X ∈ RT ×d, input token embeddings for an MoE layer                                                                                                                            |
| W1                       | W1 ∈ RE×d×2n, weight of up projection                                                                                                                                         |
| W2                       | W2 ∈ RE×n×d, weight of down projection                                                                                                                                        |
| π                        | T ×E, a binary-valued matrix where πt,e<br>π ∈ {0, 1}<br>represents if token t is routed to expert e                                                                          |
| S                        | S ∈ RT ×E, router scores. In practice, we only materialize the sparsified<br>S instead of the full S                                                                          |
| H                        | H ∈ RTK×2n, output of up projection                                                                                                                                           |
| A                        | A ∈ RTK×n, output of SwiGLU                                                                                                                                                   |
| Y                        | Y ∈ RTK×d, output of down projection                                                                                                                                          |
| O                        | O ∈ RT ×d, output of expert aggregation, also output of the entire MoE layer                                                                                                  |
| dO                       | dO ∈ RT ×d, activation gradient for<br>O                                                                                                                                      |
| dA′                      | ∈ RT ×n, GEMM output of<br>dA′ = dO W⊤<br>dO and W2. Intermediate result for computing dA and dS<br>2                                                                         |
| dA                       | dA = Broadcast(s) dA′ ∈ RT ×n, activation gradient for<br>A                                                                                                                   |
| dY                       | dY = Broadcast(s) dO ∈ RTK×d, activation gradient for<br>Y . dY is not used in SonicMoE.                                                                                      |
| dS                       | dS ∈ RT ×E, activation gradient for<br>S                                                                                                                                      |
| A′                       | A′ = Broadcast(s) A ∈ RT ×n, intermediate result and input for computing<br>dW2                                                                                               |
| dH                       | dH ∈ RT ×2n, activation gradient for<br>H                                                                                                                                     |
| ˜<br>dX                  | X˜ ∈<br>RTK×d, activation gradient for<br>d<br>X before aggregation                                                                                                           |
| dX                       | dX ∈ RT ×d, activation gradient for<br>X after aggregation                                                                                                                    |
| dW1                      | dW1 ∈ RE×d×2n, weight gradient for<br>W1                                                                                                                                      |
| dW2                      | dW2 ∈ RE×n×d, weight gradient for<br>W2                                                                                                                                       |
| A kernel                 | forward up-proj kernel                                                                                                                                                        |
| Y kernel                 | forward down-proj kernel                                                                                                                                                      |
| O kernel                 | forward expert aggregation kernel where each token aggregates all routed expert's result as the final forward output                                                          |
| dH kernel                | backward down-proj activation gradient kernel                                                                                                                                 |
| dW2 kernel               | backward down-proj weight gradient kernel                                                                                                                                     |
| dX˜ kernel               | backward up-proj activation gradient kernel                                                                                                                                   |
| dW1 kernel               | backward up-proj weight gradient kernel                                                                                                                                       |
| dX kernel                | backward expert aggregation kernel where each token aggregates the routed expert's dX˜                                                                                        |
| ⌈fe⌉Mtile<br>, ⌊fe⌋Mtile | Mtile-rounded multiples of expert frequency fe. ⌈fe⌉Mtile<br>= ⌈fe/Mtile⌉ · Mtile<br>⌊fe⌋Mtile<br>= ⌊fe/Mtile⌋ · Mtile                                                        |
| ⌊S⌉Mtile<br>, ⌊fe⌉Mtile  | ⌊fe⌉Mtile<br>is Mtile-rounded multiples of expert frequency fe. ⌊fe⌉Mtile<br>∈ {⌈fe⌉Mtile<br>, ⌊fe⌋Mtile<br>}, and ⌊S⌉Mtile<br>is the<br>score after rounding in Algorithm 4. |

## <span id="page-21-0"></span>B SonicMoE's comparison with existing MoE kernel design

Existing efficient MoE kernels also frame MoE computation as a Grouped GEMM, but their ingredients are different from SonicMoE. Here we provide an overview (but not a complete list) of key differences:

- ScatterMoE (Tan et al. [2024\)](#page-18-3)[13](#page-21-1) implements gather fusion for varlen-M Grouped GEMM but not for varlen-K Grouped GEMM. ScatterMoE also does not overlap MMA computation with memory IO. Moreover, ScatterMoE is also built on older versions of Triton where TMA is not supported. ScatterMoE computes dS as dS = ⟨dO, Y ⟩ which requires caching Y . This results in large IO cost and activation memory requirement. Both ScatterMoE's forward and backward pass have limited fusion and hence ScatterMoE is much slower than SonicMoE, especially for the backward pass.
- MoMoE (Costin et al. [2025\)](#page-16-3)[14](#page-21-2) also implements the gather fusion for varlen-M but not varlen-K Grouped GEMM similarly to ScatterMoE. Although fused with up-proj activation gradient, the dS computation still utilizes dS = ⟨dO, Y ⟩. Similar to ScatterMoE, MoMoE does not use TMA for IO. The scatter operation in MoMoE is (much) slower than SonicMoE, as shown in Figure [21.](#page-29-2)
- MegaBlocks (Gale et al. [2023\)](#page-16-4) has multiple MoE implementations and we focus on ParallelDroplessMLP[15](#page-21-3) which is built on top of block-sparse matrix multiplication[16](#page-21-4) . ParallelDroplessMLP first gathers and pads the tokens and then launches block-sparse GEMM for up and down-proj. Then, it launches a scatter kernel before reducing across the expert results. These sparse matrix multiplications usually take a longer time than the highly-optimized Grouped GEMM, as shown in Figure [5a,](#page-7-0) and the gather and scatter kernel have a total IO cost of 8TKd bytes which can be a bottleneck for fine-grained MoEs. We consider MegaBlocks's ParallelDroplessMLP as a block-sparse GEMM baseline in our benchmark and find that MoE implemented via Grouped GEMMs often have a higher training throughput than MoEs implemented via block-sparse GEMM.
- Megatron-LM (Shoeybi et al. [2019\)](#page-18-5) also has multiple MoE implementations and we focus on GroupedMLP[17](#page-21-5) , which uses Grouped GEMM[18](#page-21-6) from the CUTLASS library (Corporation [2025\)](#page-16-8) with JIT epilogue fusion as the GEMM backend. Similar to DeepGEMM, GroupedMLP does not fuse gather with the prologue (it assumes contiguouslypacked inputs). A recent memory-efficient patch[19](#page-21-7) fuses S weighting with SwiGLU computation during forward, and during backward which allows the PyTorch autograd engine (Paszke et al. [2019\)](#page-17-17) to follow a similar computational path as SonicMoE.

Megatron-LM also implements TEGroupedMLP[20](#page-21-8) which launches 4 CUDA streams to execute a list of GEMM (without contiguously-packed inputs, and without a persistent tile scheduler). In this case, each expert independently launches a new GEMM kernel leading to "bubbles" on the CUDA streams. This leads to underutilization of the GPU resources. We empirically find that TEGroupedMLP runs slower than GroupedMLP and so we use GroupedMLP across all benchmarks.

• DeepGEMM (Zhao et al. [2025b\)](#page-19-4) designs a Grouped GEMM kernel for contiguously-packed inputs. They also don't implement any other fusion for both SM90 (Hopper) and SM100 (Blackwell) BF16 Grouped GEMM. DeepGEMM specializes more on distributed training with expert parallelism (Zhao et al. [2025c\)](#page-19-6), and it is common to launch a separate all2all kernel (Lepikhin et al. [2024\)](#page-17-18) which is then followed by a contiguous Grouped GEMM. DeepGEMM SM90 BF16 kernel also assumes that each expert receives a multiple of Mtile tokens as it does not implement the TMA tensor descriptor online update during the Grouped GEMM computation. DeepGEMM's BF16 GEMM on SM90 also does not employ Ping-Pong scheduling. DeepGEMM's varlen-M grouped GEMM usually uses tile shape (128, 256, 64) for both Hopper and Blackwell GPUs, while SonicMoE often picks (256, 256, 64) with 2-CTA MMA for Blackwell GPUs. DeepGEMM does not use CLC persistent tile scheduler for grouped GEMM on Blackwell GPUs while SonicMoE often adopts CLC persistent tile scheduler on Blackwell.

Additionally, ScatterMoE and MoMoE are both implemented in Triton (Tillet, Kung, and Cox [2019\)](#page-18-10) for the ease of development at the expense of losing full programmability of the asynchronous compute and memory IO of Hopper and

<span id="page-21-2"></span><span id="page-21-1"></span><sup>13</sup><https://github.com/shawntan/scattermoe/blob/47b5e1502e5a10e82c8e5945d761b877849871e7/scattermoe/mlp.py#L51>

<span id="page-21-3"></span><sup>14</sup><https://github.com/tilde-research/MoMoE-impl/blob/d6e2d683185bfe4030265c3ca062564356faa61e/momoe/momoe.py#L914>

<sup>15</sup>[https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/layers/dmoe.](https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/layers/dmoe.py#L18) [py#L18](https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/layers/dmoe.py#L18)

<span id="page-21-4"></span><sup>16</sup>[https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/layers/mlp.py#](https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/layers/mlp.py#L308) [L308](https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/layers/mlp.py#L308)

<span id="page-21-5"></span><sup>17</sup>[https://github.com/NVIDIA/Megatron-LM/blob/610a75ef3a4a80c2ce2da436c19244e5362978d4/megatron/core/transformer/](https://github.com/NVIDIA/Megatron-LM/blob/610a75ef3a4a80c2ce2da436c19244e5362978d4/megatron/core/transformer/moe/experts.py#L100) [moe/experts.py#L100](https://github.com/NVIDIA/Megatron-LM/blob/610a75ef3a4a80c2ce2da436c19244e5362978d4/megatron/core/transformer/moe/experts.py#L100)

<span id="page-21-6"></span><sup>18</sup>[https://github.com/fanshiqing/grouped\\_gemm/blob/main/csrc/grouped\\_gemm.cu](https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu)

<span id="page-21-8"></span><span id="page-21-7"></span><sup>19</sup><https://github.com/NVIDIA/Megatron-LM/commit/2659630721ac87237c8cb772b1c2f1b34176f443>

<sup>20</sup>[https://github.com/NVIDIA/Megatron-LM/blob/610a75ef3a4a80c2ce2da436c19244e5362978d4/megatron/core/transformer/](https://github.com/NVIDIA/Megatron-LM/blob/610a75ef3a4a80c2ce2da436c19244e5362978d4/megatron/core/transformer/moe/experts.py#L746) [moe/experts.py#L746](https://github.com/NVIDIA/Megatron-LM/blob/610a75ef3a4a80c2ce2da436c19244e5362978d4/megatron/core/transformer/moe/experts.py#L746)

Blackwell GPUs (NVIDIA 2022, 2025b). For example, they cannot implement fine-grained control of asynchronous load and store during the GEMM's epilogue. They also cannot overlap MMA with heavy epilogue operations using Ping-Pong scheduling. It becomes increasingly important to overlap IO operations in epilogue when the GEMM computations are small in size (as in the case of fine-grained MoEs) to achieve high GPU utilization.

## <span id="page-22-0"></span>**C** Gradient computation

For an expert e, let

$$X_e \in \mathbb{R}^{T_e \times d}, \quad W_{1,e} \in \mathbb{R}^{d \times 2n}, \quad W_{2,e} \in \mathbb{R}^{n \times d}$$
 (5)

The forward activation computation is given by:

$$H_e = X_e W_{1,e} \in \mathbb{R}^{T_e \times 2n}, \qquad A_e = \text{SwiGLU}(H_e) \in \mathbb{R}^{T_e \times n}, \qquad Y_e = A_e W_{2,e} \in \mathbb{R}^{T_e \times d}.$$
 (6)

The token aggregation with scores  $S = \{s_{t,e}\}$  is given by

$$O_t = \sum_e s_{t,e} Y_{e,t}, \qquad dO_t \in \mathbb{R}^{1 \times d} \text{ as the gathered results from } dO.$$
 (7)

We know

$$dY_{e,t} = s_{t,e} dO_t \implies dY_e = \text{Broadcast}(\mathbf{s}_e) dO_e.$$
 (8)

Define the Grouped GEMM output as  $dA'_e := dO_e \ W_{2.e}^{\top} \in \mathbb{R}^{T_e \times n}$ .

Then from Equation 8

<span id="page-22-2"></span>
$$dA_e = dY_e W_{2,e}^{\top} = \text{Broadcast}(\mathbf{s}_e) dA_e'. \tag{9}$$

The activation gradient for score dS is<sup>21</sup>

$$dS_{t,e} = \langle dO_t, Y_{e,t} \rangle = \langle dO_t W_{2,e}^{\top}, A_{e,t} \rangle = \langle dA'_{e,t}, A_{e,t} \rangle.$$
(10)

In addition, we can derive  $dH_e$  from  $dA_e$  and  $A_e$  (recomputed from  $H_e$ ) as:

$$dH_e = dSwiGLU(dA_e, H_e).$$
 (11)

Using Equation 8,

$$dW_{2,e} = A_e^{\top} dY_e = A_e^{\top} \left( \text{Broadcast}(\mathbf{s}_e) dO_e \right) = \left( \underbrace{\text{Broadcast}(\mathbf{s}_e) A_e}_{A'_e} \right)^{\top} dO_e.$$
 (12)

### <span id="page-22-1"></span>C.1 Computational choices for dS

If we do not implement custom kernels and rely solely on PyTorch's autograd (AD) engine, we can add the expert weighting (S) either (1) before down-proj forward or (2) after down-proj forward. Both yield identical results for forward and backward, but the computation for dS is different. For (1), we need to compute  $\langle dA'_{e,t}, A_{e,t} \rangle$  which is used by SonicMoE and Megatron<sup>22</sup>. MoMoE<sup>23</sup>, ScatterMoE<sup>24</sup>, and MegaBlocks<sup>2526</sup> compute  $\langle dO_t, Y_{e,t} \rangle$  as required in (2).

Note that dS can be computed as any of  $dS_{t,e} = \langle dA'_{e,t}, A_{e,t} \rangle = \langle dO_t, Y_{e,t} \rangle$ , however computing it as  $dS_{t,e} = \langle dA'_{e,t}, A_{e,t} \rangle$  is a computationally and activation memory-efficient choice due to the following reasons:

<span id="page-22-3"></span> $<sup>^{21}</sup>$ If we also consider expert biases, we will have  $dS_{e,t} = \langle dA'_{e,t}, A_{e,t} \rangle + \langle dO_{e,t}, \operatorname{Broadcast}(S_{e,t}) \rangle$ . The activation memory efficiency can still be preserved but we need to perform a separate computation for  $\langle dO_{e,t}, \operatorname{Broadcast}(S_{e,t}) \rangle$  either via a separate kernel (SonicMoE's current choice) or fuse it with dH Grouped GEMM mainloop. Please refer to the released code of SonicMoE.

<span id="page-22-4"></span><sup>&</sup>lt;sup>22</sup>https://github.com/NVIDIA/Megatron-LM/blob/44af130cc4568d324860646996b6a5bfd6c5e3e6/megatron/core/transformer/moe/experts.py#L284

<span id="page-22-6"></span><span id="page-22-5"></span><sup>&</sup>lt;sup>23</sup>https://github.com/tilde-research/MoMoE-imp1/blob/d6e2d683185bfe4030265c3ca062564356faa6le/momoe/momoe.py#L702

<sup>&</sup>lt;sup>24</sup>https://github.com/shawntan/scattermoe/blob/47b5e1502e5a10e82c8e5945d761b877849871e7/scattermoe/parallel\_experts.ov#L79

<span id="page-22-7"></span><sup>25</sup> https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/ops/binned\_scatter.py#L12

<span id="page-22-8"></span> $<sup>^{26} \</sup>texttt{https://github.com/databricks/megablocks/blob/78eea65fda01e638af36ae38853bc51efb04a4b4/megablocks/ops/binned\_scatter.py \#L49$ 

- Additional HBM traffic (0 vs. 2TKd bytes):  $\langle dA'_{e,t}, A_{e,t} \rangle$  requires  $dA'_{e,t}$  and  $A_{e,t}$  are already computed during the dH kernel, so we can avoid extra unnecessary loads.
- Extra cached activation memory (0 vs. 2TKd bytes): One of the reasons why the cached activation memory for ScatterMoE, MoMoE and MegaBlocks fails to stay constant w.r.t. expert granularity is the required caching of Y for computing dS.
- Parallel reduction rounds ( $\log_2(n)$  vs.  $\log_2(d)$ ):  $\langle dA'_{e,t}, A_{e,t} \rangle$  reduces over n while  $\langle dO_t, Y_{e,t} \rangle$  reduces over d. This difference saves at least  $\log_2(d/n)$  rounds of reduction.

## <span id="page-23-1"></span>**D** Efficient top-K sorting kernel for MoE

official example, and RTop-K (Xie et al. 2025) in Figure 22a.

Existing MoE approaches such as ScatterMoE, MoMoE, and MegaBlocks use the PyTorch top-K (torch.topk) to compute the expert assignments for each token. We find that the PyTorch top-K kernel can take approximately 40% of the router's computation time. We implement an efficient top-K kernel in SonicMoE to reduce the overhead due to PyTorch top-K. Our top-K kernel supports E and K when  $E \le 4096$  and  $K \le 16$  and is optimized for the case of a large number of tokens T. We also offer an optional softmax fusion on top-K values within the top-K kernel.

The top-K kernel accepts the router output with shape (T,E) and parallelizes over T. The kernel uses bitonic sort (Batcher 1968) over every row (sorts E values) and selects the first K columns as the sort output. After loading the input, we pack the column indices of the first K columns (for  $\operatorname{argtop} K$ ) into the lower  $\log_2(E)$  mantissa bits of FP32 values in registers, except that we specialize the base sorting cases (number of values  $\leq 64$ ) to follow the comparison strategies obtained from optimal low-latency sorting networks (Dobbelaere 2025), which provide the minimum number of parallel operation steps and required compare\_and\_swap calls.

The bitonic compare and merging occurs within the same thread or the same warp via warp-shuffle. Therefore, every swap and merge operation only uses intra-thread or intra-warp registers. This achieves a higher memory bandwidth for the kernel over alternative kernel designs such as PyTorch TopK (Paszke et al. 2019), the Triton (Tillet, Kung, and Cox 2019) and Tilelang (Wang et al. 2025)

new FP32 val after packing column idx

sign exponent higher mantissa column idx

Figure 15: The sorting is conducted over values after we pack the column index bits into lower mantissa bits. This value format ensures a stable sorting result. Triton's official top-K kernel follows a similar format.

Since the assigned column indices for values on each row are always unique, there will not be any equal numbers after we pack the column index to the lower mantissa bits. Therefore, SonicMoE's top-K kernel is always stable as there will not be any tie-breaking scenarios during bitonic compare and merge.

## E Referenced tables, figures, and SonicMoE algorithms

In Table 4, we provide a trending overview for open-source frontier MoE models. We present SonicMoE's expert aggregation strategy in Figure 17. Figure 16 illustrates that on Hopper GPUs, asynchronous TMA store (top) has higher memory bandwidth and can naturally overlap with TensorCore MMA whereas synchronous st.global (bottom) PTX instruction, necessary for scatter fusion on Hopper GPUs, blocks the execution of next Tensor Core MMA tile and leads to longer kernel runtime.

<span id="page-23-0"></span>**Algorithm 5** SonicMoE's MoE kernel backward pass of up-proj.

```
Input :X, \pi, W_1, dH
Output: dX, dW_1.
Up-proj act d\tilde{X} kernel (dH, W_1) \rightarrow d\tilde{X}:
// varlen-M Grouped GEMM
      Parallel for e \in [E] do
             dH_e, W_{1,e} \leftarrow \operatorname{load}(dH_e, W_{1,e})
             d\tilde{X}_e \leftarrow dH_e \, W_{1.e}^\top
             d\tilde{X}_e \leftarrow \text{store}(d\tilde{X}_e)
Up-proj weight dW_1 kernel (X, dH, \pi) \rightarrow dW_1:
// Gather + varlen-K Grouped GEMM
      Parallel for e \in [E] do
             X, \pi_{:,e}, d\dot{H}_e \leftarrow \operatorname{load}(X, \pi_{:,e}, dH_e)
             X_e \leftarrow \text{Gather}(X, \pi_{:,e})
             dW_{1,e} \leftarrow X_e^{\top} dH_e
             dW_{1,e} \leftarrow \text{store}(dW_{1,e})
Expert aggregation dX kernel (d\tilde{X}, \pi) \rightarrow dX:
// Gather and sum
       Parallel for t \in [T] do
             d\tilde{X}, \ \pi_{t,e} \leftarrow \operatorname{load}(d\tilde{X}, \ \pi_{t,e})
             dX_t \leftarrow \sum_{e \in [E]} \pi_{t,e} d\tilde{X}_{e,t}
             dX_t \leftarrow \text{store}(dX_t)
```

<span id="page-24-0"></span>Table 4: MoE Scaling Trends: Here, we show the activation ratio as experts activated per token K / total experts E and expert granularity is shown as model embedding dimension (d) / expert intermediate size (n) for frontier open source models. We do not include the shared experts for the MoE sparsity calculation. The trend indicates new open-source MoE models tend to be more granular and sparser.

| Model                                   | Release date | Parameters | Expert activation ratio (K/E) | Expert granularity (d/n) |
|-----------------------------------------|--------------|------------|-------------------------------|--------------------------|
| Mixtral 8x22B (Mistral 2024)            | 11/23        | 131B       | 25.0% (2/8)                   | 6144/16384 = 0.38        |
| DBRX (The Mosaic Research Team 2024)    | 03/24        | 132B       | 25.0% (4/16)                  | 6144/10752 = 0.57        |
| Phi-3.5-MoE (Microsoft 2024)            | 09/24        | 42B        | 12.5% (2/16)                  | 4096/6400 = 0.64         |
| OLMoE (Muennighoff et al. 2025)         | 09/24        | 7B         | 12.5% (8/64)                  | 2048/1024 = 2.00         |
| Granite 3.1-MoE (Granite 2024)          | 12/24        | 3B         | 20.0% (8/40)                  | 1536/512 = 3.00          |
| DeepSeek-V3 (DeepSeek-AI et al. 2024)   | 12/24        | 671B       | 3.13% (8/256)                 | 7168/2048 = 3.50         |
| Qwen3 MoE (QwenLM 2025)                 | 04/25        | 235B       | 6.25% (8/128)                 | 4096/1536 = 2.67         |
| QWen3-30B-A3B (Qwen 2025)               | 05/25        | 30.5B      | 6.25% (8/128)                 | 2048/768 = 2.67          |
| Kimi K2 (Kimi et al. 2025)              | 07/25        | 1.04T      | 2.08% (8/384)                 | 7168/2048 = 3.50         |
| gpt-oss-120b (OpenAI 2025)              | 08/25        | 120B       | 3.13% (4/128)                 | 2880/2880 = 1.00         |
| GLM-4.5-Air (Zeng et al. 2025)          | 08/25        | 106B       | 6.25% (8/128)                 | 4096/1408 = 2.91         |
| Qwen3-Next-80B-A3B-Instruct (Qwen 2025) | 09/25        | 81B        | 1.95% (10/512)                | 2048/512 = 4.00          |
| DeepSeek-V3.2-Exp (DeepSeek-AI 2025)    | 10/25        | 685B       | 3.13% (8/256)                 | 7168/2048 = 3.50         |

<span id="page-24-2"></span>![](_page_24_Figure_2.jpeg)

Figure 16: Illustration to show asynchronous TMA store (top) has higher memory bandwidth and can naturally overlap with TensorCore MMA while synchronous st.global (bottom) PTX instruction, necessary for scatter fusion on Hopper GPUs, blocks the execution of next Tensor Core MMA tile and leads to longer kernel runtime. This figure is supported by the 20.1% speedup on average of "SonicMoE (gemm + gth w. sum)" (TMA store) over "SonicMoE (gemm w. sct + sum)" (st.global) in Figure [21'](#page-29-2)s transparent bars. As a result, SonicMoE does not fuse scatter with HBM store and instead, lets each token gather the expert results in the expert aggregation kernel. Both ScatterMoE and MoMoE do not adopt such design and SonicMoE can achieve 1.75x and 3.11x speedup respectively on average during forward down-proj kernel in Figure [5a.](#page-7-0)

<span id="page-24-1"></span>![](_page_24_Figure_4.jpeg)

Figure 17: Possible strategies for storing the results and aggregating the results for each token. SonicMoE chooses the first strategy (left) in which each expert directly stores contiguously-packed outputs via TMA in the GEMM epilogue. In the expert aggregation kernel, each token gathers and sums over activated expert outputs. ScatterMoE and MoMoE (middle) choose to fuse HBM store with scatter in epilogue and launch a summation kernel afterwards. We note that each token gathering (left) the Grouped GEMM result is equivalent to each expert scattering (middle) the Grouped GEMM outputs. In Figure [21,](#page-29-2) we implement both strategies on SonicMoE and observe the left strategy can have 17% speedup over the middle strategy. It is also possible to fuse atomic add in the epilogue to circumvent the requirement of an expert aggregation kernel as the right subfigure illustrated. However, this atomic add operation creates new issues like non-determinism (He and Machines [2025\)](#page-17-12) and numerical accuracy (for BF16 atomic add). This figure is adapted from Tan et al. [\(2024\)](#page-18-3)'s Figure 2.

## <span id="page-25-0"></span>F SonicMoE's ablation study on kernel-level throughput

In this section, we present kernel-level ablation studies on training throughput to examine the impact of each of the implemented features on SonicMoE. In Section F.1, we investigate the Grouped GEMM throughput with and without the gather fusion on Hopper and Blackwell GPUs. In Section F.2, we profile the memory bandwidth of expert aggregation kernels. In section F.3, we compare the left and middle expert aggregation strategy when both implemented on SonicMoE on H100 GPUs in Figure 17. In Section F.4, we compare SonicMoE's top-K kernels with other efficient top-K implementations.

### <span id="page-25-1"></span>F.1 Grouped GEMM

<span id="page-25-6"></span>![](_page_25_Figure_3.jpeg)

(a) Varlen-M Grouped GEMM with contiguously-packed inputs to up and down-proj during forward pass on H100 GPUs.

![](_page_25_Figure_5.jpeg)

(b) Varlen-M Grouped GEMM with contiguously-packed inputs to up and down-proj during forward pass on B300 GPUs. Figure 18: Varlen-M Grouped GEMM with contiguously-packed inputs on H100 and B300 GPUs. We use the same configurations as in Figure 10. "cuBLAS BMM" is a *dense* BMM baseline equivalent to all expert receiving equal number of tokens (perfectly load balanced), whose TFLOPS can be considered as an *upper bound* for any Grouped GEMM kernel.

We also benchmark SonicMoE's base Grouped GEMM kernel with both contiguously-packed inputs and gathered inputs without any epilogue fusion on H100 and B300 GPUs. For contiguously-packed inputs, we mainly compare with DeepGEMM kernels (sm90\_m\_grouped\_bf16\_gemm\_contiguous<sup>27</sup> and sm90\_bf16\_k\_grouped\_gemm<sup>28</sup> on H100 GPUs, sm100\_m\_grouped\_bf16\_gemm\_contiguous<sup>29</sup> and sm100\_bf16\_k\_grouped\_gemm<sup>30</sup> on B300 GPUs). We note that at the

<span id="page-25-2"></span> $<sup>^{27} \</sup>texttt{https://github.com/deepseek-ai/DeepGEMM/blob/9b680f428484625f4f35dc3617f134187c6bcd4a/csrc/jit\_kernels/impls/sm90\_bf16\_gemm.hpp\#L127$ 

<span id="page-25-3"></span> $<sup>{}^{28} \</sup>text{https:}//\text{github.com/deepseek-ai/DeepGEMM/blob/9b680f428484625f4f35dc3617f134187c6bcd4a/csrc/jit\_kernels/impls/sm90\_bf16\_gemm.hpp\#L234$ 

<span id="page-25-4"></span> $<sup>^{29} \</sup>texttt{https://github.com/deepseek-ai/DeepGEMM/blob/35c4bc87713726d048f65275f6f1b551a4e7a6dc/csrc/jit\_kernels/impls/sm100\_bf16\_gemm.hpp\#L124$ 

<span id="page-25-5"></span> $<sup>^{30} \</sup>texttt{https://github.com/deepseek-ai/DeepGEMM/blob/35c4bc87713726d048f65275f6f1b551a4e7a6dc/csrc/jit\_kernels/impls/sm100\_bf16\_gemm.hpp\#L233$ 

time of writing, DeepGEMM's k Grouped GEMM kernel only accepts the form of D = AB + C for GEMM[31](#page-26-0) where for correctness, we use a zero-filled FP32 C weight gradient buffer as accumulator input, but we use C as uninitialized weight gradient buffer during benchmarking. For inputs requiring a gather operation, we mainly compare with ScatterMoE and MoMoE. We also benchmark cuBLAS dense BMM (CUDA toolkit v12.9 for H100 GPUs and v13.0 for B300 GPUs) assuming each expert receives the same amount of tokens. On Blackwell GPUs, we also benchmark the GEMM example in triton official example[32](#page-26-1) .

Grouped GEMM with contiguously-packed inputs. We compare SonicMoE with DeepGEMM on varlen-M Grouped GEMM without any other fusion. We also benchmark cuBLAS dense BMM (perfect load balance and no tensormap update needed) as a reference for the TFLOPS upper bound for Grouped GEMM.

- H100 GPUs: In Figure [18a,](#page-25-6) we find that SonicMoE's up-proj has 2.7% higher TFLOPS while down-proj has 10.0% higher TFLOPS than DeepGEMM on average. We use Ping-Pong scheduling for down-proj with n < 1024 while DeepGEMM (Zhao et al. [2025b\)](#page-19-4) uses cooperative scheduling (*[Efficient GEMM in CUDA](#page-16-12)* [2025;](#page-16-12) Zhao et al. [2025b\)](#page-19-4), and as a result, we observe SonicMoE has a greater speedup over DeepGEMM when intermediate size is small. For example, SonicMoE has 57.4%, 14.0%, and 5.3% more TFLOPS than DeepGEMM for 30B down-proj config.
- B300 GPUs: In Figure [18b,](#page-25-6) we find that SonicMoE's up-proj has 8.1% higher TFLOPS while down-proj has 12.7% higher TFLOPS than DeepGEMM on average. Compared to the triton official GEMM example, SonicMoE's upproj has 13.3% higher TFLOPS while down-proj has 15.6% higher TFLOPS on average. SonicMoE's speedup for fine-grained MoEs are still maintained: for example, SonicMoE has 20.2%, 8.5%, and 8.1% higher TFLOPS than DeepGEMM, and 22.2%, 14.9%, and 7.5% higher TFLOPS than triton official example for 30B down-proj config.

Grouped GEMM with gather fusion. We report SonicMoE, ScatterMoE, MoMoE, and DeepGEMM with and without gather fusion (as opaque and transparent bars for each method) on Hopper GPU. ScatterMoE and MoMoE both have gather fusion for varlen-M but not varlen-K Grouped GEMM, so we benchmark their gather with varlen-K Grouped GEMM time (opaque bar) by adding up the time of their contiguously-packed weight gradient kernel (transparent bar) with the time of their own gather kernel. We also adopt the similar benchmark approach for DeepGEMM and cuBLAS dense BMM. We note that the cuBLAS dense BMM results can be considered as the upper bound of the TFLOPS for any Grouped GEMM kernel without gather fusion.

### • Gather on M dimension.

- H100 GPUs: in Figure [19a,](#page-27-0) the average relative TFLOPS difference of SonicMoE with and without gather fusion on M dim is 6.3%. SonicMoE consistently achieves higher TFLOPS than ScatterMoE (avg 9.7%), MoMoE (avg 30.9%), and DeepGEMM (avg 38.3%) with gather fusion.
- B300 GPUs: in Figure [19b,](#page-27-0) the average relative TFLOPS difference of SonicMoE with and without gather fusion on M dim is 3.4%. Triton official example also implements gather fusion with TMA on Blackwell GPUs, but the difference of TFLOPS with and without gather fusion is 6.3%, higher than SonicMoE. Similar to the case on Hopper GPUs, SonicMoE consistently achieves higher TFLOPS than ScatterMoE, MoMoE, and DeepGEMM with gather fusion.

#### • Gather on K dimension.

- H100 GPUs: In Figure [19a,](#page-27-0) the average relative TFLOPS difference of SonicMoE with and without gather fusion on K dim is 8.5%. When we compare SonicMoE with gather fusion (opaque bars) against ScatterMoE and MoMoE with their gather kernel together, we observe a wider gap as expert granularity increases (from right to left on each 3 bar groups).
- B300 GPUs: In Figure [19b,](#page-27-0) the average relative TFLOPS difference of SonicMoE with and without gather fusion on K dim is -0.1%. This means gather fusion on K dim virtually does not induce any throughput degradation on TFLOPS. We still observe a wider gap of TFLOPS between SonicMoE with gather fusion and ScatterMoE and MoMoE with a separate gather kernel as expert granularity increases.

<span id="page-26-1"></span><span id="page-26-0"></span><sup>31</sup>This formula naturally fits with the gradient accumulation.

<sup>32</sup>[https://github.com/triton-lang/triton/blob/c833d995f93d2eadb5627afae129496a852bc9fb/python/triton\\_kernels/bench/](https://github.com/triton-lang/triton/blob/c833d995f93d2eadb5627afae129496a852bc9fb/python/triton_kernels/bench/bench_mlp.py#L53) [bench\\_mlp.py#L53](https://github.com/triton-lang/triton/blob/c833d995f93d2eadb5627afae129496a852bc9fb/python/triton_kernels/bench/bench_mlp.py#L53)

<span id="page-27-0"></span>![](_page_27_Figure_0.jpeg)

(a) Forward pass up-proj (gather on M dim) and backward up-proj weight gradient  $dW_1$  (gather on K dim) kernel on H100 GPUs.

![](_page_27_Figure_2.jpeg)

(b) Forward pass up-proj (gather on M dim) and backward up-proj weight gradient  $dW_1$  (gather on K dim) kernel on B300 GPUs.

Figure 19: Forward pass up-proj (gather on  $\mathbf{M}$  dim) and backward up-proj weight gradient  $dW_1$  (gather on  $\mathbf{K}$  dim) kernel on H100 and B300 GPUs. SonicMoE supports both inputs gathered from different positions (opaque bars) and contiguously-packed inputs (transparent bars). ScatterMoE and MoMoE both have gather fusion for varlen- $\mathbf{M}$  but not varlen- $\mathbf{K}$ , so we benchmark their gather with varlen- $\mathbf{K}$  Grouped GEMM time (opaque bar) by adding up the time of their contiguously-packed weight gradient kernel (transparent bar) with the time of their gather kernel. DeepGEMM does not have gather fusion for both varlen- $\mathbf{M}$  and varlen- $\mathbf{K}$  Grouped GEMM, so we provide an optimized gather kernel in both cases. We also provide a "cuBLAS dense BMM" (transparent bar) baseline and the gather with GEMM time (opaque bar) by adding up the time with a heavily-tuned gather kernel's time with the same input shape, which can be considered as the upper bound TFLOPS for *any* Grouped GEMM kernel without gather fusion.

### <span id="page-27-1"></span>F.2 Expert aggregation

In Figure 20a and 20b, we compare the bandwidth of SonicMoE's gather-and-sum aggregation (Figure 17 left) with ScatterMoE's torch.bmm and MoMoE torch.sum aggregation on contiguous Y inputs (Figure 17 middle). In addition, we implement 2 optimized triton kernels that either use vanilla triton tl.load or TMA load and they both sum over contiguous Y inputs. We report the max bandwidth of both results as "max(tl.load, TMA)". This result achieves 85%+ peak for most MoE configurations so we consider it as an upper bound for any aggregation kernel on both H100 and B300 GPUs. On B300 GPUs, we also use Gluon to implement gather-and-sum aggregation with TMA gather<sup>33</sup>.

• **H100 GPUs**: although SonicMoE's aggregation kernel requires a gather fusion during HBM load, the memory bandwidth of SonicMoE still surpasses ScatterMoE (2.92x on average) and MoMoE (1.05x on average), and is only slightly slower (0.98x on average) than the triton upper bound of summing over contiguous *Y*.

<span id="page-27-2"></span> $<sup>^{33}</sup>$ adopted from https://github.com/triton-lang/triton/blob/8ecbfb066a28e231e9fd54ca0ce6de8ace4950aa/python/tutorials/gluon/09-tma-gather-scatter.py#L298

<span id="page-28-2"></span>![](_page_28_Figure_0.jpeg)

![](_page_28_Figure_1.jpeg)

(b) Expert aggregation kernels (O kernel) during forward pass of MoE on B300 GPUs.

Figure 20: Expert aggregation kernel (O kernel) on H100 and B300 GPUs. Same configurations as in Figure [10.](#page-10-0) ScatterMoE uses torch.bmm call to reduce over K for a contiguous Y input, MoMoE uses a torch.sum call. We take the maximum bandwidth between PyTorch eager and PyTorch compile on default mode with PyTorch 2.9.0. We also implement 2 optimized triton kernels that either use vanilla triton tl.load or TMA load and they both sum over contiguous Y inputs. We report the max bandwidth of both results as "max(tl.load, TMA)". On B300 GPUs, we also use Gluon to implement gather-and-sum aggregation with TMA gather. On H100 GPUs, we use Triton 3.3.1, and on B300 GPUs, we use Triton 3.6.0 for benchmarking.

• B300 GPUs: the results are similar to that of H100 GPUs. The memory bandwidth of SonicMoE aggregation kernel still largely surpasses ScatterMoE (6.72x on average) and MoMoE (3.32x on average), and is only slightly slower (0.98x on average) than the triton upper bound of summing over contiguous Y . We also note that SonicMoE gather-and-sum is faster than Gluon TMA gather-and-sum (1.05x on average).

## <span id="page-28-0"></span>F.3 Strategies of combining grouped gemm and expert aggregation

We compare the left and middle expert aggregation strategy when both are implemented on SonicMoE on H100 GPUs in Figure [17.](#page-24-1) We observe the left strategy (gemm + gth w. sum) achieves a 20% higher TFLOPS than the middle strategy (gemm w. sct + sum) and we therefore choose the left one for forward down-proj and backward up-proj activation gradient kernel.

## <span id="page-28-1"></span>F.4 Top-K sorting

We benchmark the bandwidth of SonicMoE's top-K kernel in Figure [22a](#page-30-0) and [22b.](#page-30-0) For each GPU, we compare SonicMoE with PyTorch[34](#page-28-3), triton official example[35](#page-28-4), tilelang official example[36](#page-28-5), and RTop-K[37](#page-28-6) (Xie et al. [2025\)](#page-18-12) on BF16 and FP32 inputs.

• PyTorch single block Top-K: PyTorch (Paszke et al. [2019\)](#page-17-17) implements a radix-select followed by a gather algorithm for top-K, and it dispatches to a single or multiple block version depending on T, E, K. For large T with modest E and K, PyTorch uses the single-block version that performs 2 SMEM scans. In this case, SonicMoE's sorting networks with pure register-based communication are much faster.

<span id="page-28-3"></span><sup>34</sup>[https://github.com/pytorch/pytorch/blob/b54e466fd04e5e736662a6206d81ab0d5fe85d91/aten/src/ATen/native/cuda/](https://github.com/pytorch/pytorch/blob/b54e466fd04e5e736662a6206d81ab0d5fe85d91/aten/src/ATen/native/cuda/TensorTopK.cu#L40) [TensorTopK.cu#L40](https://github.com/pytorch/pytorch/blob/b54e466fd04e5e736662a6206d81ab0d5fe85d91/aten/src/ATen/native/cuda/TensorTopK.cu#L40)

<span id="page-28-4"></span><sup>35</sup>[https://github.com/triton-lang/triton/blob/de8e71503fea971dfb65308147798657e18f8568/python/triton\\_kernels/](https://github.com/triton-lang/triton/blob/de8e71503fea971dfb65308147798657e18f8568/python/triton_kernels/triton_kernels/topk_details/_topk_forward.py#L90) [triton\\_kernels/topk\\_details/\\_topk\\_forward.py#L90](https://github.com/triton-lang/triton/blob/de8e71503fea971dfb65308147798657e18f8568/python/triton_kernels/triton_kernels/topk_details/_topk_forward.py#L90)

<span id="page-28-5"></span><sup>36</sup>[https://github.com/tile-ai/tilelang/blob/5c62d00a64f2f52cf6b2536a2492a29fc5323723/examples/topk/example\\_topk.py#](https://github.com/tile-ai/tilelang/blob/5c62d00a64f2f52cf6b2536a2492a29fc5323723/examples/topk/example_topk.py#L18) [L18](https://github.com/tile-ai/tilelang/blob/5c62d00a64f2f52cf6b2536a2492a29fc5323723/examples/topk/example_topk.py#L18)

<span id="page-28-6"></span><sup>37</sup>[https://github.com/xiexi51/RTopK/blob/952f515321a4bdc5c4a57944bca0d32641052460/rtopk\\_kernel.cu#L7](https://github.com/xiexi51/RTopK/blob/952f515321a4bdc5c4a57944bca0d32641052460/rtopk_kernel.cu#L7)

<span id="page-29-2"></span>![](_page_29_Figure_0.jpeg)

Figure 21: Throughput of Grouped GEMM and expert aggregation kernel on H100 GPUs. "SonicMoE (gemm + gth w. sum)" is the final design choice for SonicMoE as illustrated in Figure [17](#page-24-1) left strategy. We compare this design against "SonicMoE (gemm w. sct + sum)" that implements the Figure [17](#page-24-1) middle strategy on SonicMoE. We use identical tile sizes and other GEMM configs for both "SonicMoE (gemm + gth w. sum)" and "SonicMoE (gemm w. sct + sum)". We also compare with ScatterMoE's design (fused scatter with GEMM + torch.bmm, labeled as "ScatterMoE (gemm w. sct + BMM)") and MoMoE's design (fused scatter with GEMM + torch.sum, labeled as "MoMoE (gemm w. sct + sum)"). For each method, we report the GEMM TFLOPS in transparent bars and TFLOPS of total runtime of GEMM and expert aggregation in the opaque bars.

- Triton official example: triton (Tillet, Kung, and Cox [2019\)](#page-18-10) provides a top-K example kernel that is also based on bit packing and bitonic merge. The main algorithmic difference is that SonicMoE relies on optimal sorting networks on the base cases while the Triton implementation directly calls triton.language.topk. During the top-K benchmark in Figure [22a,](#page-30-0) we observe that Triton is much faster than PyTorch torch.topk but it is still consistently slower than SonicMoE's top-K across all configurations.
- Tilelang official example: tilelang (Wang et al. [2025\)](#page-18-11) provides a top-K example kernel that performs K-pass maximum reduction. This design is more targeted for small K and we observe that as both E and K become larger, the Tilelang top-K kernel's throughput decreases compared to other baselines (and SonicMoE)'s increasing trend. Such a trend makes tilelang's example top-K kernel (K-pass top-K kernel) unsuitable for fine-grained MoEs.
- RTop-K: RTop-K (Xie et al. [2025\)](#page-18-12) follows a threshold-based binary search. Each bisection step utilizes warp-level primitives and follows a *selection-by-count* method instead of SonicMoE's sorting network. RTop-K is also an iterative algorithm with iterations dependent on the value range and vector size. In addition, RTop-K heavily utilizes SMEM for scanning while SonicMoE solely relies on registers for its compare and swap subroutine. We find that SonicMoE's top-K generally achieves higher throughput on both H100 and B300 GPUs.

## <span id="page-29-1"></span>G More experiments

In this section, we investigate the qualitative improvement from fine-grained MoE in Section [G.1.](#page-29-3) We also investigate the effect of rounding subroutines round and sparsify in Algorithm [4](#page-9-1) and the effects from microbatch size T and tile size Mtile for token rounding in Section [G.3.](#page-32-1)

## <span id="page-29-3"></span>G.1 Effect of expert granularity

Here we validate the effectiveness of adopting fine-grained MoE. We fix the MoE activation ratio ρ = K/E for the 0.5B and 1.4B model and we proportionally scale up K and E while linearly decreasing n from row 1 to row 3 in Table [5a](#page-31-0) and [5b.](#page-31-0)

In general, we observe a better performance for n = 256 than n = 1024 which is also consistent with the MoE scaling trends mentioned in Table [4.](#page-24-0) In Figure [1](#page-1-0) right subfigure, we find both SonicMoE and cuBLAS can still sustain the throughput from n = 1024 to n = 256 under iso-FLOPs, but starting from n = 256 FLOPs will drop linearly w.r.t. granularity. Therefore, we choose n = 256 for all experiments in Table [2.](#page-14-0)

## <span id="page-29-0"></span>G.2 Ablation study on different rounding subroutines for token rounding

We conduct ablation studies to study the effect of the different rounding subroutines on the trained MoE by TR. We compare token rounding with nearest rounding ("NR") on per-expert token counts alongside other rounding methods. Specifically, we compare against stochastic rounding with per-expert token count ("SR"), always round up ("UP"), and always round

<span id="page-30-0"></span>![](_page_30_Figure_0.jpeg)

(a) Top-K kernels with BF16 inputs (1st row) and FP32 inputs (2nd row) during forward pass of MoE on H100 GPUs.

![](_page_30_Figure_2.jpeg)

(b) Top-K kernels with BF16 inputs ( $1^{st}$  row) and FP32 inputs ( $2^{nd}$  row) during forward pass of MoE on B300 GPUs.

Figure 22: Top-K kernels during forward pass of MoE. Same configurations as in Figure 10. "torch" is a direct torch.topk call. "triton" and "tilelang" are taken from their official examples with slight modifications to support BF16 inputs. For the triton official kernel, we remove the unnecessary bit matrix store and disable the softmax fusion in this example for a fair comparison. "RTop-K"(Xie et al. 2025) only supports FP32 inputs. We set  $\epsilon = 0$  and maximum iteration as 8 for RTop-K.

down ("DOWN"). The results are shown in Table 6 and we find that our token rounding algorithm in general is robust to the specific rounding subroutines.

Following Algorithm 4, for expert e, we denote the expert frequency from the TC sorting as  $f_e$ , and the last  $M_{\rm tile}$ -divisible expert frequency as  $\lfloor f_e \rfloor_{M_{\rm tile}}$ , and the next  $M_{\rm tile}$ -divisible expert frequency as  $\lceil f_e \rceil_{M_{\rm tile}}$ . We also denote the expert scores from TC sorting as  $s_e$ , the expert scores from the selected tokens in  $\pi_e[: \lfloor f_e \rfloor_{M_{\rm tile}}]$  as  $\lfloor s_e \rfloor_{M_{\rm tile}}$  and the scores for  $\pi_e[: \lceil f_e \rceil_{M_{\rm tile}}]$  as  $\lceil s_e \rceil_{M_{\rm tile}}$ . We note that all rounding algorithms only make a binary decision between discarding TC tokens and padding EC tokens for each expert. The following are simple heuristics to perform rounding:

- NR-f: nearest rounding to  $M_{\rm tile}$ -multiples via expert frequency: We pad EC tokens if  $\lceil f_e \rceil_{M_{\rm tile}} f_e < f_e \lfloor f_e \rfloor_{M_{\rm tile}}$ . "NR-f" is our default choice of token rounding and we use it for Table 2, 7, 8, and Figures 8 and 13.
- SR-f: stochastic rounding to  $M_{\text{tile}}$ -multiples via expert frequency: We sample from Bernoulli  $\left(\frac{f_e \lfloor f_e \rfloor_{M_{\text{tile}}}}{M_{\text{tile}}}\right)$  distribution for deciding whether to pad EC tokens for expert e.
- NR-s: nearest rounding to  $M_{\rm tile}$ -multiples via expert scores: We sample from the following distribution for deciding whether to pad EC tokens for expert e:

<span id="page-31-0"></span>Table 5: Evaluation of MoE w.r.t. granularity with iso-FLOPs (nK is constant) and iso-params (nE is constant) settings. "PPL" refers to the validation perplexity at the end of training. "Avg" is the mean accuracy across the 11 downstream tasks. The "dense, iso-FLOPs" refers to a dense model with nK as the intermediate size, while the "dense, iso-params" refers to a dense model with nE as the intermediate size.

| (a) <b>0.5B</b> | params, | <b>20B</b> | tokens, | 8/64 | activated |
|-----------------|---------|------------|---------|------|-----------|
|                 |         |            |         |      |           |

| $\overline{(E, K, n)}$ | PPL   | Wino                                    | SIQA  | SciQ             | PIQA      | OBQA      | HS       | COPA      | CSQA  | BoolQ | ArcE | ArcC | Avg  |
|------------------------|-------|-----------------------------------------|-------|------------------|-----------|-----------|----------|-----------|-------|-------|------|------|------|
| (2, 11, 10)            |       | *************************************** | 514.1 | 56.2             | 114.1     | 024.1     |          |           | 004.1 | 2001  |      |      |      |
| 16, 2, 1024            | 16.23 | 53.0                                    | 41.3  | <b>79.8</b>      | 65.0      | 32.6      | 37.8     | 66.0      | 32.2  | 55.8  | 53.9 | 29.1 | 49.7 |
| 64, 8, 256             | 16.01 | 51.0                                    | 41.4  | 79.2             | 65.5      | 31.6      | 38.4     | 66.0      | 31.5  | 60.2  | 57.5 | 25.7 | 49.8 |
| 256, 32, 64            | 16.13 | 51.2                                    | 41.5  | 78.9             | 65.3      | 34.2      | 38.4     | 63.0      | 32.4  | 60.6  | 59.5 | 28.1 | 50.3 |
| Dense, iso-FLOPs       | 19.90 | 48.9                                    | 41.4  | 74.9             | 62.2      | 30.2      | 32.6     | 62.0      | 31.6  | 61.7  | 53.2 | 27.1 | 47.8 |
| Dense, iso-params      | 15.46 | 52.1                                    | 41.5  | 78.9             | 65.3      | 34.0      | 39.2     | 69.0      | 32.2  | 58.5  | 59.3 | 28.8 | 50.8 |
|                        |       |                                         | (1    | b) <b>1.4B</b> p | params, 5 | 0B tokens | s, 8/128 | activated |       |       |      |      |      |
|                        |       |                                         | `     |                  |           |           |          |           |       |       |      |      |      |
| 32, 2, 1024            | 13.38 | 52.2                                    | 41.7  | 81.7             | 69.2      | 33.6      | 44.3     | 64.0      | 33.5  | 61.1  | 60.9 | 29.8 | 52.0 |
| 128, 8, 256            | 13.32 | 51.8                                    | 41.7  | 81.5             | 69.3      | 32.4      | 45.3     | 68.0      | 34.5  | 56.6  | 63.2 | 28.4 | 52.1 |
| 512, 32, 64            | 13.50 | 52.5                                    | 41.2  | 82.9             | 68.9      | 34.4      | 44.7     | 69.0      | 33.6  | 58.7  | 62.6 | 30.1 | 52.6 |
| Dense, iso-FLOPs       | 17.90 | 52.2                                    | 41.0  | 79.2             | 63.4      | 31.0      | 34.7     | 61.0      | 30.5  | 60.3  | 51.8 | 25.1 | 48.2 |
| Dense, iso-params      | 12.74 | 52.2                                    | 42.6  | 83.3             | 70.1      | 34.8      | 46.8     | 67.0      | 35.1  | 61.7  | 63.5 | 31.8 | 53.5 |

Bernoulli 
$$\left(\frac{\sum_{t} s_{e,t} - \sum_{t} \lfloor s_{e,t} \rfloor_{M_{\text{tile}}}}{\sum_{t} \lceil s_{e,t} \rceil_{M_{\text{tile}}} - \sum_{t} \lfloor s_{e,t} \rfloor_{M_{\text{tile}}}}\right)$$
 (13)

• Balance-f: balanced rounding to  $M_{\rm tile}$ -multiples via expert frequency: The Balance algorithm (Cooper et al. 2023; Dwivedi and Mackey 2024; Lu, Guo, and De Sa 2022) can be adapted to ensure the total number of routed tokens to all experts after tile-rounding is preserved regardless of the number of experts E. Algorithm 6 is such an example that ensures

$$\max_{e \in [E]} |\lceil f_e \rfloor_{M_{\text{tile}}} - f_e| \le M_{\text{tile}}/2, \qquad \left| \sum_{e=1}^{E} \lceil f_e \rfloor_{M_{\text{tile}}} - \sum_{e=1}^{E} f_e \right| \le M_{\text{tile}}/2$$
(14)

where the other rounding subroutine will have an expected deviation of  $O\left(M_{\rm tile}\sqrt{E}\right)$  for  $\sum_{e=1}^{E}\lceil f_e \rfloor_{M_{\rm tile}}$ 

- UP: always round up expert frequency as  $\lceil f_e \rceil_{M_{\text{tile}}}$ : We always pad EC tokens chosen in the second step of sorting in Algorithm 4. This gives a model TFLOPS lower-bound for Figure 13.
- DOWN: always round down expert frequency as  $\lfloor f_e \rfloor_{M_{\text{tile}}}$ : We always discard TC top-K tokens chosen in the first step of sorting in Algorithm 4. This gives a model TFLOPS upper-bound for Figure 13.

Always discarding TC tokens ("DOWN"). "DOWN" is a baseline in which we always drop the last TC tile if the expert frequency is not  $M_{\rm tile}$ -divisible. This idea is similar to the idea of *token dropping* in expert parallelism where the expert will sort and drop the token with the lowest scores when it receives too many tokens (Fedus, Zoph, and Shazeer 2022). We note that "DOWN" produces the shortest MoE kernel runtime for any rounding algorithm. However, in Table 6, we observe that "DOWN" yields a much higher validation perplexity than "NR-f", "SR-f" and "NR-s". Although we can expect a shorter MoE kernel runtime by always discarding TC tokens, such quality degradation might not be acceptable in practice.

Always padding EC tokens ("UP"). "UP" is a baseline in which we always pad extra EC tokens to the last TC tile if the expert frequency is not  $M_{\rm tile}$ -divisible. Contrary to "DOWN", "UP" produces the longest MoE kernel runtime for any rounding algorithm. In Table 6, we find that "UP" often produces lower validation perplexity, but the average downstream task accuracy is not necessarily higher than other rounding algorithms. Given the longer MoE kernel runtime but not necessarily better trained MoE quality, we do not recommend the usage of always rounding up. We speculate this is due to the train-test gap between TC and EC routing and "UP" reinforces the bias towards EC more strongly than other TR algorithms.

For a balance between training efficiency and trained MoE quality, neither always discarding TC tokens nor padding EC tokens is the right solution. In Table 2, we pick "NR-f" as the round\_and\_sparsify subroutine for TR's main experiments.

<span id="page-32-2"></span>Algorithm 6 Balanced rounding to  $M_{\text{tile}}$ -multiples via expert frequency ("Balance-f" in Table 6).

This algorithm satisfies 
$$\max_{e \in [E]} \left| \lceil f_e \rfloor_{M_{\text{tile}}} - f_e \right| \le M_{\text{tile}}/2$$
 and  $\left| \sum_{e=1}^{E} \lceil f_e \rfloor_{M_{\text{tile}}} - \sum_{e=1}^{E} f_e \right| \le M_{\text{tile}}/2$ 

Input  $:f^{\text{TC}} = \{f_e\}_{e \in [E]}$  as a list of expert frequency with TC top-K routing,  $\{\lceil f_e \rceil_{M_{\text{tile}}}\}_{e \in [E]}$  as a list of expert frequency with TC top-K routing and with potential EC padding to ensure each expert receives a multiple of  $M_{\text{tile}}$  tokens,  $\{\lfloor f_e \rfloor_{M_{\text{tile}}}\}_{e \in [E]}$  as a list of expert frequency with TC top-K routing and with potential token dropping to ensure each expert receives a multiple of  $M_{\text{tile}}$  tokens; We should ensure  $\max_{e \in [E]} \left(\lceil f_e \rceil_{M_{\text{tile}}} - \lfloor f_e \rfloor_{M_{\text{tile}}} \right) \leq M_{\text{tile}}$ .

 $\textbf{Output:} f^{\text{TR}} = \{\lceil f_e \rceil_{M_{\text{tile}}} \}_{e \in [E]} \text{ as a list of expert frequency that ensures each expert receives a multiple of } M_{\text{tile}} \text{ tokens}$ 

```
// an accumulator that ensures the preservation of total expert frequency z \leftarrow 0;  
for e \in [E] do  
// calculate the residual error of both rounding choice r_e^{\text{up}} \leftarrow \lceil f_e \rceil_{M_{\text{tile}}} - f_e;  
r_e^{\text{down}} \leftarrow \lfloor f_e \rfloor_{M_{\text{tile}}} - f_e;  \nif |r_e^{\text{up}} + z| < |r_e^{\text{down}} + z| then  
// choose to pad with EC tokens  
\lceil f_e \rfloor_{M_{\text{tile}}} \leftarrow \lceil f_e \rceil_{M_{\text{tile}}};  
z \leftarrow z + r_e^{\text{up}};
```

else

// choose to discard TC tokens  $\lceil f_e \rfloor_{M_{\text{tile}}} \leftarrow \lfloor f_e \rfloor_{M_{\text{tile}}};$ 

13.12

50.1

42.9

TC top-K

<span id="page-32-0"></span>Table 6: Evaluation of token rounding algorithms equipped with different round\_and\_sparsify subroutines in Algorithm 4. "PPL" refers to the validation perplexity at the end of training. "Avg" is the mean accuracy across the 11 downstream tasks.

|                |       | (-   | a) <b>0.3D p</b> a | 11 ams, 40 | D tokens  | , 2/04 acu  | vateu (1 | e = 512,           | $M_{\rm tile} = 1$ | 120)  |      |      |      |
|----------------|-------|------|--------------------|------------|-----------|-------------|----------|--------------------|--------------------|-------|------|------|------|
| Method         | PPL   | Wino | SIQA               | SciQ       | PIQA      | OBQA        | HS       | COPA               | CSQA               | BoolQ | ArcE | ArcC | Avg  |
| TR (NR-f)      | 15.92 | 51.4 | 41.6               | 78.4       | 65.4      | 31.6        | 38.1     | 65.0               | 31.0               | 61.1  | 57.4 | 29.1 | 50.0 |
| TR (SR-f)      | 15.93 | 50.8 | 40.9               | 77.4       | 66.9      | 33.0        | 38.4     | 64.0               | 31.1               | 60.7  | 55.8 | 28.1 | 49.7 |
| TR (NR-s)      | 15.91 | 51.3 | 40.9               | 80.3       | 65.4      | 30.8        | 37.7     | 67.0               | 31.0               | 61.6  | 55.4 | 28.4 | 50.0 |
| TR (Balance-f) | 15.93 | 51.9 | 41.8               | 78.8       | 65.9      | 32.6        | 38.4     | 66.0               | 31.6               | 60.3  | 56.8 | 27.1 | 50.1 |
| TR (UP)        | 15.89 | 50.5 | 40.9               | 78.6       | 64.5      | 32.2        | 38.2     | 68.0               | 29.9               | 55.2  | 54.2 | 30.1 | 49.3 |
| TR (DOWN)      | 16.10 | 51.1 | 41.4               | 78.7       | 64.9      | 31.6        | 38.0     | 62.0               | 32.8               | 61.9  | 58.9 | 30.8 | 50.2 |
| TC top-K       | 15.94 | 51.0 | 41.9               | 78.5       | 64.8      | 33.0        | 38.1     | 67.0               | 30.8               | 54.7  | 55.8 | 30.1 | 49.6 |
|                |       | (b   | ) <b>1.8B pa</b>   | rams, 40   | B tokens, | , 8/256 act | ivated ( | $\bar{T}_e = 512,$ | $M_{\rm tile} =$   | 128)  |      |      |      |
| TR (NR-f)      | 13.10 | 53.4 | 42.1               | 81.7       | 69.6      | 35.2        | 45.3     | 70.0               | 33.2               | 61.4  | 63.0 | 33.4 | 53.5 |
| TR (SR-f)      | 13.08 | 52.7 | 41.6               | 82.6       | 69.4      | 34.4        | 45.6     | 70.0               | 33.0               | 59.1  | 62.5 | 34.8 | 53.2 |
| TR (NR-s)      | 13.09 | 54.1 | 42.3               | 82.8       | 69.3      | 33.8        | 45.7     | 70.0               | 34.1               | 59.0  | 64.6 | 32.4 | 53.5 |
| TR (Balance-f) | 13.08 | 52.5 | 42.0               | 82.7       | 70.0      | 33.2        | 45.3     | 68.0               | 34.6               | 59.4  | 63.3 | 33.4 | 53.1 |
| TR (UP)        | 13.07 | 50.4 | 41.7               | 81.4       | 68.4      | 37.2        | 45.4     | 69.0               | 31.9               | 51.7  | 62.2 | 33.4 | 52.1 |
| TR (DOWN)      | 13 19 | 55.4 | 41.6               | 82.2       | 68.6      | 34.8        | 45.0     | 69.0               | 34.0               | 54.4  | 63.5 | 31.4 | 52.7 |

(a) **0.5B** params, **40B** tokens, **2/64** activated ( $\bar{T}_e = 512$ ,  $M_{\rm tile} = 128$ )

### <span id="page-32-1"></span>G.3 Ablation study on the effects of microbatch size T and tile size $M_{\rm tile}$

69.8

81.3

Effect of microbatch size T. Since the token rounding is applied on the microbatch level, the choice of microbatch size T will result in different qualitative results for TR. Note that this also holds true for EC routing. For example, EC over sequence will result in different model quality as EC over a text segment. In Table 7, we vary the microbatch size while keeping the minibatch size (consumed tokens per optimization step) constant.

33.8

45.2

71.0

34.1

56.7

64.6

31.1

52.8

We find that TR will preserve its trained MoE quality when  $T_e/M_{\rm tile} \geq 2$ , but if  $T_e/M_{\rm tile} = 1$  (the last row in both subtables), there is a noticeable quality degradation for both validation perplexity and downstream task performance. However, the trained MoE quality with  $\bar{T}_e/M_{\rm tile} = 1$  is still better than training with EC and finetuning with TC top-K routing.

Effect of the tile quantization size  $M_{\rm tile}$ . Similarly in Table 8, we can find that TR is generally robust w.r.t.  $M_{\rm tile}$  when  $\bar{T}_e/M_{\rm tile} \geq 2$ , and when  $\bar{T}_e/M_{\rm tile} = 1$  there is a noticeable degradation but the overall result is still better than EC baseline.

<span id="page-33-0"></span>Table 7: Evaluation of token rounding algorithms when we vary microbatch size T to change average number of tokens per expert  $(\bar{T}_e)$ . For each trial, we vary the microbatch size from 4  $(\bar{T}_e = 512)$  to 1  $(\bar{T}_e = 128)$  and keep minibatch size constant. The  $M_{\rm tile}$  is always kept as 128. "PPL" refers to the validation perplexity at the end of training. "Avg" is the mean accuracy across the 11 downstream tasks.

| (a) 0.5B params, 40B tokens, 2/64 activated $(M_{\rm tile} = 128$ | (a) 0.5R na | arams, 40B | tokens, 2/6 | 4 activated | (M <sub>+:1</sub> = | = 128) |
|-------------------------------------------------------------------|-------------|------------|-------------|-------------|---------------------|--------|
|-------------------------------------------------------------------|-------------|------------|-------------|-------------|---------------------|--------|

| Method                                                    | PPL   | Wino | SIQA             | SciQ     | PIQA       | OBQA       | HS        | COPA                | CSQA   | BoolQ | ArcE | ArcC | Avg  |
|-----------------------------------------------------------|-------|------|------------------|----------|------------|------------|-----------|---------------------|--------|-------|------|------|------|
| $\overline{\text{TR}\left(\bar{T}_e = 1024\right)}$       | 15.91 | 52.9 | 41.0             | 80.1     | 65.1       | 31.0       | 37.9      | 63.0                | 32.3   | 59.3  | 54.9 | 28.1 | 49.6 |
| $TR  (\bar{T}_e = 512)$                                   | 15.92 | 51.4 | 41.6             | 78.4     | 65.4       | 31.6       | 38.1      | 65.0                | 31.0   | 61.1  | 57.4 | 29.1 | 50.0 |
| $TR (\bar{T}_e = 256)$                                    | 15.98 | 52.2 | 41.4             | 77.7     | 66.1       | 32.2       | 37.9      | 66.0                | 31.0   | 59.6  | 57.2 | 30.1 | 50.1 |
| $TR (\bar{T}_e = 128)$                                    | 16.11 | 51.7 | 41.7             | 77.9     | 66.1       | 30.8       | 37.7      | 67.0                | 31.9   | 61.2  | 54.7 | 29.1 | 50.0 |
| TC top-K                                                  | 15.94 | 51.0 | 41.9             | 78.5     | 64.8       | 33.0       | 38.1      | 67.0                | 30.8   | 54.7  | 55.8 | 30.1 | 49.6 |
| EC (ft TC router)                                         | 16.98 | 50.0 | 41.7             | 79.7     | 64.9       | 31.6       | 36.8      | 63.0                | 32.1   | 60.7  | 54.6 | 27.4 | 49.3 |
|                                                           |       |      | (b) <b>1.8</b> F | 3 params | s, 40B tok | ens, 8/250 | 6 activat | ed ( $M_{\rm tile}$ | = 128) |       |      |      |      |
| $\overline{\text{TR}\left(\bar{T}_e = 1024\right)} \Big $ | 13.08 | 51.5 | 42.0             | 81.7     | 68.9       | 34.8       | 45.7      | 72.0                | 32.6   | 59.5  | 61.4 | 32.1 | 52.9 |
| $TR  (\bar{T}_e = 512)$                                   | 13.10 | 53.4 | 42.1             | 81.7     | 69.6       | 35.2       | 45.3      | 70.0                | 33.2   | 61.4  | 63.0 | 33.4 | 53.5 |
| $TR  (\bar{T}_e = 256)$                                   | 13.12 | 51.9 | 41.2             | 81.8     | 69.7       | 33.6       | 45.2      | 73.0                | 34.2   | 56.9  | 63.2 | 34.1 | 53.2 |
| $TR (\bar{T}_e = 128)$                                    | 13.55 | 51.9 | 41.5             | 82.0     | 69.2       | 32.8       | 44.4      | 69.0                | 34.4   | 59.8  | 64.0 | 30.4 | 52.7 |
| TC top-K                                                  | 13.12 | 50.1 | 42.9             | 81.3     | 69.8       | 33.8       | 45.2      | 71.0                | 34.1   | 56.7  | 64.6 | 31.1 | 52.8 |
| EC (ft TC router)                                         | 15.01 | 52.7 | 41.1             | 79.6     | 66.9       | 30.6       | 40.2      | 66.0                | 31.9   | 60.5  | 57.2 | 30.8 | 50.7 |

<span id="page-33-1"></span>Table 8: Evaluation of token rounding algorithms when we vary the size of tile  $M_{\rm tile}$  for token rounding. "PPL" refers to the validation perplexity at the end of training. "Avg" is the mean accuracy across the 11 downstream tasks.

(a) 0.5B params, 40B tokens, 2/64 activated ( $\bar{T}_e = 512$ )

| Method                                                  | PPL   | Wino | SIQA             | SciQ    | PIQA      | OBQA       | HS       | COPA                | CSQA   | BoolQ | ArcE | ArcC | Avg  |
|---------------------------------------------------------|-------|------|------------------|---------|-----------|------------|----------|---------------------|--------|-------|------|------|------|
| $TR (M_{\text{tile}} = 64)$                             | 15.90 | 51.3 | 41.7             | 78.1    | 65.6      | 31.4       | 37.9     | 67.0                | 32.4   | 59.8  | 57.2 | 28.8 | 50.1 |
| $TR (M_{\rm tile} = 128)$                               | 15.92 | 51.4 | 41.6             | 78.4    | 65.4      | 31.6       | 38.1     | 65.0                | 31.0   | 61.1  | 57.4 | 29.1 | 50.0 |
| $TR (M_{tile} = 256)$                                   | 16.00 | 51.7 | 41.4             | 78.7    | 66.3      | 32.4       | 37.7     | 67.0                | 31.3   | 60.1  | 58.2 | 29.1 | 50.4 |
| $TR (M_{\text{tile}} = 512)$                            | 16.17 | 52.5 | 41.2             | 80.2    | 65.2      | 32.0       | 37.9     | 62.0                | 31.0   | 59.4  | 57.2 | 30.4 | 49.9 |
| TC top-K                                                | 15.94 | 51.0 | 41.9             | 78.5    | 64.8      | 33.0       | 38.1     | 67.0                | 30.8   | 54.7  | 55.8 | 30.1 | 49.6 |
| EC (ft TC router)                                       | 16.98 | 50.0 | 41.7             | 79.7    | 64.9      | 31.6       | 36.8     | 63.0                | 32.1   | 60.7  | 54.6 | 27.4 | 49.3 |
|                                                         |       |      | (b) <b>1.8</b> l | B param | s, 40B to | kens, 8/25 | 6 activa | ted ( $\bar{T}_e$ = | = 512) |       |      |      |      |
| $\overline{\text{TR}\left(M_{\text{tile}} = 64\right)}$ | 13.07 | 52.3 | 42.9             | 82.7    | 69.4      | 35.4       | 45.6     | 70.0                | 32.4   | 56.6  | 64.4 | 31.4 | 53.0 |
| $TR (M_{\rm tile} = 128)$                               | 13.10 | 53.4 | 42.1             | 81.7    | 69.6      | 35.2       | 45.3     | 70.0                | 33.2   | 61.4  | 63.0 | 33.4 | 53.5 |
| $TR (M_{tile} = 256)$                                   | 13.13 | 52.0 | 41.6             | 82.1    | 69.2      | 35.4       | 45.3     | 69.0                | 34.2   | 58.0  | 65.6 | 32.1 | 53.1 |
| $TR (M_{\rm tile} = 512)$                               | 13.56 | 53.0 | 41.8             | 81.2    | 68.4      | 34.0       | 44.2     | 68.0                | 33.3   | 58.1  | 59.5 | 30.1 | 52.0 |
| TC top-K                                                | 13.12 | 50.1 | 42.9             | 81.3    | 69.8      | 33.8       | 45.2     | 71.0                | 34.1   | 56.7  | 64.6 | 31.1 | 52.8 |
| EC (ft TC router)                                       | 15.01 | 52.7 | 41.1             | 79.6    | 66.9      | 30.6       | 40.2     | 66.0                | 31.9   | 60.5  | 57.2 | 30.8 | 50.7 |

## <span id="page-33-3"></span>H Activation memory and training throughput benchmark configurations

The configurations of Figure 10 and 11a are included in Table 9a.

The configurations for the 4 subfigures in Figure 13 are listed below. Notice that we consistently use  $M_{\rm tile}$  as 128 when we benchmark the TR's speed.

- Top-left 2 subfigures: We use (T, d, n, K) = (16384, 1536, 256, 8) and we vary E from 64 to 512.
- **Top-right 2 subfigures**: We use (T, d, n, K) = (16384, 1536, 1024, 2) and we vary E from 16 to 128.
- **Bottom-left 2 subfigures**: We use (T, d, n, K) = (16384, 4096, 512, 8) and we vary E from 64 to 512.
- **Bottom-right 2 subfigures:** We use (T, d, n, K) = (16384, 4096, 1024, 4) and we vary E from 32 to 256.

