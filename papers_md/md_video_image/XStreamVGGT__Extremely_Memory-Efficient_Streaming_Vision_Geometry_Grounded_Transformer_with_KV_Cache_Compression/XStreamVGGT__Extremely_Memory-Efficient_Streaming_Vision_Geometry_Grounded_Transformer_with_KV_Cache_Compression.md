# XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression

Zunhai Su<sup>1\*</sup> Weihao Ye<sup>2\*</sup> Hansen Feng<sup>1</sup> Keyu Fan<sup>1</sup> Jing Zhang<sup>3</sup> Dahai Yu<sup>4</sup> Zhengwu Liu<sup>5</sup> Ngai Wong<sup>5</sup>

<sup>1</sup>Shenzhen International Graduate School, Tsinghua University
 <sup>2</sup>Institute of Artificial Intelligence, Xiamen University
 <sup>3</sup>China Star Optoelectronics Technology
 <sup>4</sup>TCL Corporate Research (HK) Co., Ltd.
 <sup>5</sup>Department of Electrical and Electronic Engineering, The University of Hong Kong

#### **Abstract**

Learning-based 3D visual geometry models have significantly advanced with the advent of large-scale transformers. Among these, StreamVGGT leverages framewise causal attention to deliver robust and efficient streaming 3D reconstruction. However, it suffers from unbounded growth in the Key-Value (KV) cache due to the massive influx of vision tokens from multi-image and long-video inputs, leading to increased memory consumption and inference latency as input frames accumulate. This ultimately limits its scalability for long-horizon applications. To address this gap, we propose XStreamVGGT, a tuning-free approach that seamlessly integrates pruning and quantization to systematically compress the KV cache, enabling extremely memory-efficient streaming inference. Specifically, redundant KVs generated from multi-frame inputs are initially pruned to conform to a fixed KV memory budget using an efficient token-importance identification mechanism that maintains full compatibility with high-performance attention kernels (e.g., FlashAttention). Additionally, leveraging the inherent distribution patterns of KV tensors, we apply dimension-adaptive KV quantization within the pruning pipeline to further minimize memory overhead while preserving numerical accuracy. Extensive evaluations show that XStreamVGGT achieves mostly negligible performance degradation while substantially reducing memory usage by 4.42× and accelerating inference by 5.48×, enabling practical and scalable streaming 3D applications. The code is available at https://github.com/ywh187/XStreamVGGT/.

# 1 Introduction

Recovering 3D geometric structures from image sequences has long been a fundamental challenge in 3D computer vision [21]. This task is crucial for numerous real-world applications, such as robotics, augmented reality, and autonomous driving [20, 10, 6]. For years, 3D vision has been dominated by classical methods, primarily structure-from-motion (SfM) [1, 11, 32] and multi-view stereo (MVS) [14, 12]. While these methods demonstrate strong performance in high-fidelity geometric optimization, they rely on fragmented multi-stage pipelines that incur substantial processing delays. Such pipelines are also susceptible to cascading errors, potentially compromising the accuracy and integrity of the final reconstruction. In recent years, learning-based feed-forward models have revolutionized this field.

<sup>\*</sup>Equal Contribution.

Models such as DUSt3R, CUT3R, and VGGT [\[29,](#page-12-2) [31,](#page-12-3) [30\]](#page-12-4) have shifted the paradigm from conventional approaches to end-to-end deep learning frameworks, showcasing not only remarkable performance but also impressive generalization across diverse datasets. As a significant milestone in this evolution, the Visual Geometry-Grounded Transformer (VGGT) consolidates multiple 3D vision tasks within a unified framework, consistently surpassing task-specific methods across a broad spectrum of applications, including dense depth estimation, point map regression, and camera pose prediction [\[29\]](#page-12-2). To support robust streaming applications,

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1: Efficiency analysis on a single 80GB A100 GPU. As the number of input frames increases, StreamVGGT and VGGT exhibit significant FPS degradation and rapidly encounter out-of-memory (OOM) errors. In contrast, XStreamVGGT consistently delivers significantly higher frames per second (FPS) without encountering OOM issues.

StreamVGGT [\[40\]](#page-13-1) replaces the global attention mechanism in Alternative-Attention of VGGT with frame-wise causal attention, following a design philosophy similar to autoregressive large language models (LLMs) [\[5,](#page-11-6) [4,](#page-11-7) [28\]](#page-12-5). This change shifts the model from an offline to an online streaming paradigm, substantially improving practical usability.

At the core of StreamVGGT is its reliance on the Key-Value (KV) cache of previous input frames, which functions as an explicit and persistent memory mechanism. However, as streaming inference progresses, the model is exposed to a significant influx of vision tokens from multi-image and long-video inputs. This influx drives the KV cache to expand linearly with the number of input frames, eventually resulting in unbounded growth [\[17,](#page-12-6) [15\]](#page-11-8). As illustrated in Figure [1,](#page-1-0) memory consumption and inference latency escalate rapidly as input frames accumulate in StreamVGGT. This issue significantly restricts the system's scalability for long-horizon applications, creating a critical bottleneck for real-world deployment.

To address this challenge, we propose XStreamVGGT, a tuning-free approach that seamlessly integrates pruning and quantization to systematically compress the KV cache, enabling highly memoryefficient streaming inference. Specifically, XStreamVGGT first eliminates redundant KV cache from multi-frame inputs through an efficient token importance identification mechanism, pruning the cache to a bounded budget while preserving the first-frame KVs as geometric references [\[29\]](#page-12-2). Furthermore, our analysis reveals pronounced channel-wise outliers in the Key tensors, while the Value tensors exhibit much weaker outlier behavior in StreamVGGT. Leveraging the inherent distribution patterns of KV tensors, we develop a dimension-adaptive KV quantization scheme that incorporates per-channel Key and per-token Value quantization. This quantization scheme is seamlessly integrated into the pruning pipeline, further minimizing memory overhead while maintaining numerical accuracy. Our key contributions can be summarized as follows:

- We introduce XStreamVGGT, the first method to seamlessly integrate pruning and quantization for systematically compressing the KV cache in StreamVGGT. XStreamVGGT effectively addresses the issue of unbounded growth in KV memory, enabling extremely memory-efficient streaming inference.
- We conducted a comprehensive analysis of the KV distribution in StreamVGGT, unveiling, for the first time, the distinctive distribution patterns of Key and Value tensors in 3D reconstruction transformer models. By leveraging dimension-adaptive quantization, we effectively mitigate its impact on quantization accuracy.
- Extensive evaluations on 3D reconstruction, camera pose estimation and depth estimation demonstrate that XStreamVGGT achieves mostly negligible performance degradation, while reducing memory usage by 4.42× and accelerating inference by 5.48×, offering a powerful and scalable solution for efficient streaming 3D applications.

# 2 Related Work

### 2.1 Learning-Based 3D Reconstruction

By implicitly encoding scene priors within their learned parameters, recent neural network-based approaches have substantially advanced the robustness and generalization of 3D vision models, establishing new performance standards across a variety of tasks [29]. A significant milestone in this progression was DUSt3R [31], which demonstrated the direct regression of view-consistent 3D pointmaps from a pair of RGB images, thereby circumventing the traditional requirement for explicit camera calibration. Building on this foundation, CUT3R [30] introduced a stateful recurrent framework capable of incrementally updating a unified scene representation from a sequential stream of images. Extending this paradigm, TTT3R [7] incorporates a Test-Time Training (TTT) strategy that dynamically refines memory updates by assessing the alignment confidence between previously encoded states and incoming observations, thereby enhancing generalization over extended sequences. A pivotal advancement was realized with VGGT [29], which scales this philosophy within a 1.2 billion-parameter Alternative-Attention transformer architecture. By jointly predicting multiple 3D attributes, VGGT achieves state-of-the-art performance across a comprehensive range of 3D vision tasks. Most recently, StreamVGGT [40] adapts this powerful model to the demands of online streaming reconstruction. However, its reliance on an unbounded KV cache presents a critical bottleneck, substantially impeding its practical viability and scalability in real-world streaming applications.

## 2.2 KV Cache Compression

The KV cache facilitates efficient autoregressive inference by storing previously computed Key and Value representations, thereby circumventing the need for redundant recomputation. However, as sequence lengths grow, the KV cache rapidly becomes a critical bottleneck in terms of memory footprint, motivating a substantial body of research on cache compression techniques [16, 27]. Existing approaches to KV cache compression can be broadly classified into two main categories: pruning-based and quantization-based methods. KV pruning seeks to reduce memory and computational overhead by selectively discarding past KV entries deemed to have low importance. Importance is typically estimated using criteria such as attention scores, token saliency, or other heuristics [36, 33]. In parallel, KV quantization compresses the cache by representing high-precision Key and Value tensors in low-bit formats, thereby reducing storage requirements and memory access costs [17, 25, 26, 15]. Despite the significant progress made in these areas, existing KV cache compression techniques have primarily been developed for LLMs that operate on textual data. In contrast, KV caches in 3D vision models exhibit spatial and temporal redundancies, presenting greater potential for compression. However, effectively compressing KV caches in the context of 3D vision tasks remains an open and underexplored research challenge.

## 3 Methodology

In this section, we present XStreamVGGT, with an overview illustrated in Figure 2. We begin by reviewing the preliminaries of StreamVGGT in Section 3.1, then present our approach to eliminating multi-frame redundancy through KV cache pruning in Section 3.2. In Section 3.3, we analyze the distributional properties of KV tensors in StreamVGGT, which motivate the development of an effective dimension-adaptive KV quantization scheme.

#### <span id="page-2-0"></span>3.1 Preliminaries

StreamVGGT is a streaming 4D visual geometry transformer that processes video frames in an online manner, producing per-frame geometry outputs without the need to reprocess the entire history. Given an incoming RGB frame  $I_t \in \mathbb{R}^{3 \times H \times W}$  at time step t, the model first converts it into a sequence of visual tokens  $F_t \in \mathbb{R}^{N \times C}$  using a patch embedding network, where N is the number of image patches and C is the embedding dimension. In addition to the patch tokens, StreamVGGT prepends a camera token  $g_t \in \mathbb{R}^{1 \times C}$  to encode global camera-related information, which is later decoded by task-specific heads to predict camera parameters. The model also includes R register tokens  $r_t \in \mathbb{R}^{R \times C}$ , which act as auxiliary latent slots to absorb redundant attention responses during

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2: Overview of XStreamVGGT. Upon receiving a new input frame (Step 1), Queries from the global attention layer are aggregated via average pooling to form a compact representation, which is then matched against the Key to estimate token importance (Step 2). Guided by these Key-derived importance scores, low-importance historical KV pairs are selectively pruned, while KVs from the first frame are explicitly retained to preserve geometric consistency. The remaining high-importance KVs are concatenated with the first-frame KVs and the newly generated KVs from the current frame (Step 3). Following pruning, the KV cache is further compressed using dimension-adaptive quantization, employing per-channel Key quantization and per-token Value quantization to reduce the impact of outlier channels on quantization accuracy (Step 4). This results in a compact KV cache for efficient subsequent updates (Step 5).

transformer processing. Thus, the input token sequence for frame t consists of the camera token, register tokens, and patch tokens, with a total length of 1+R+N. This token sequence is fed into a spatio-temporal transformer encoder with L layers, adopting an Alternating-Attention design [29]. Each layer first applies frame-wise spatial self-attention to model intra-frame structure, followed by temporal causal attention to aggregate information from past frames under a strict causal constraint.

In each transformer layer  $\ell$ , the temporal attention module maintains a KV cache that stores token-level representations from all previous frames:

$$C_{t-1}^{(\ell)} = \{ K_{1:t-1}^{(\ell)}, V_{1:t-1}^{(\ell)} \}, \tag{1}$$

where  $K_{\tau}^{(\ell)}, V_{\tau}^{(\ell)} \in \mathbb{R}^{(1+R+N) \times C}$  denote the Key and Value tensors corresponding to all tokens of frame  $\tau$ . For the current frame, only the Query, Key, and Value tensors  $Q_t^{(\ell)}, K_t^{(\ell)}$ , and  $V_t^{(\ell)}$  are newly computed. Temporal attention at layer  $\ell$  is computed as:

$$\operatorname{Attn}(Q_t^{(\ell)}, \ [K_{1:t-1}^{(\ell)}, K_t^{(\ell)}], \ [V_{1:t-1}^{(\ell)}, V_t^{(\ell)}]), \tag{2}$$

with a causal mask to prevent access to future frames. After the attention computation,  $K_t^{(\ell)}$  and  $V_t^{(\ell)}$  are appended to the cache for use in subsequent time steps. The outputs of the final transformer layer are then passed to lightweight, task-specific heads to predict per-frame geometry signals, such as camera parameters, dense point maps, and others.

During inference, the size of the Query tensor  $Q_t^{(\ell)}$  remains constant over time, as it depends solely on the tokens of the current frame. In contrast, the Key and Value tensors grow linearly with the number of processed frames due to the accumulation of cached representations. Consequently, the memory cost of temporal attention at each layer increases linearly with the sequence length, making long-duration streaming inference progressively more expensive.

#### <span id="page-3-1"></span>3.2 Eliminating Multi-Frame Redundancy through KV Cache Pruning

Despite the shared use of KV caching, StreamVGGT differs fundamentally from autoregressive LLMs. In LLMs, KV caches are formed from text tokens carrying rich semantic context, whereas in StreamVGGT they originate from visual tokens extracted from video frames. Unlike textual tokens, vision tokens exhibit substantial redundancy due to both intra-frame spatial correlations and interframe temporal consistency. As illustrated in Figure 3, this redundancy manifests as sparse attention patterns across multi-frame inputs, suggesting significant opportunities for cache compression.

We propose a query-guided KV cache pruning mechanism to eliminate multi-frame redundancy while retaining the most informative historical tokens within a fixed cache length  $\mathcal{L}_{max}$ . The pruning operation is applied independently at each transformer layer. Notably, since the spatio-temporal

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

(a) The input query frame.

(b) Attention heatmaps of previous input frames.

Figure 3: Attention sparsity analysis. The visualization shows attention heatmaps from Layer 14 of StreamVGGT. The visualization of attention heatmaps reveals that attention weights are predominantly concentrated on Query-relevant regions. In contrast, other areas exhibit significantly lower attention, indicating substantial redundancy in the feature representation.

encoder employs an Alternating-Attention design, only the temporal global attention module maintains a KV cache and is thus subject to pruning. We begin by computing the similarity between the pooled Query tokens of the current frame and each Key token to assess token importance. This approach strikes a balance between using attention scores as a metric for token importance and efficiently computing the attention kernel. Attention scores have been widely used in previous studies to identify token importance [36, 33]. However, the attention calculation in StreamVGGT is designed to be compatible with highly optimized, efficient attention kernels (such as FlashAttention [9, 38]), which do not provide intermediate computation results, including attention scores. While repeatedly computing the QK attention scores yields accurate attention scores, it incurs excessive computational cost. By leveraging pooled Query groups for importance identification, we efficiently compute token importance while ensuring compatibility with optimized attention kernels.

Specifically, given the Query  $Q_t^{(\ell)} \in \mathbb{R}^{(1+R+N) \times C}$  of the current frame at layer  $\ell$ , we first separate the special tokens (camera token and register tokens), denoted as  $Q_{t,\mathrm{special}}^{(\ell)}$ , from the ordinary patch tokens  $Q_{t,\text{normal}}^{(\ell)}$ . The patch tokens are grouped into fixed-size groups of size g. The pooled Query is then formed as:

$$Q_{t, \mathrm{pooled}}^{(\ell)} = \mathrm{concat}\Big(Q_{t, \mathrm{special}}^{(\ell)}, \ \mathrm{GroupAvg}(Q_{t, \mathrm{normal}}^{(\ell)}, g)\Big). \tag{3}$$
 We further average the pooled Query across all attention heads to obtain:

$$\bar{Q}_t^{(\ell)} = \frac{1}{H} \sum_{h=1}^{H} Q_{t,\text{pooled}}^{(\ell)}[h] \in \mathbb{R}^{N_{\text{pooled}} \times C}, \tag{4}$$

where  $N_{\text{pooled}}$  denotes the number of tokens after grouping. The grouping size g is a fixed hyperpa-

Pruning is triggered immediately after the temporal global attention at layer  $\ell$  completes, once the total cache length T exceeds the budget  $\mathcal{L}_{max}$ . Tokens from the first and current frames are always preserved, serving as stable geometric references and up-to-date visual evidence, respectively. Let  $T_{\rm first}$  and  $T_{\rm current}$  denote the number of tokens corresponding to the first and current frames. The remaining middle segment is subject to pruning:  $T_{\rm prunable} = T - T_{\rm first} - T_{\rm current}$ . For the prunable middle tokens, we compute a head-averaged key summary:

$$\bar{K}_{\text{prunable}}^{(\ell)} = \frac{1}{H} \sum_{h=1}^{H} K_{1:t-1}^{(\ell)} \left[ h, \ T_{\text{first}} : T - T_{\text{current}} \right] \in \mathbb{R}^{T_{\text{prunable}} \times C}. \tag{5}$$

All token types in the middle frames, including camera, register, and patch tokens, are treated uniformly during pruning. Token importance scores are computed via the inner product between the pooled queries and the prunable keys:

$$S_{\text{matrix}}^{(\ell)} = \bar{Q}_t^{(\ell)} \left( \bar{K}_{\text{prunable}}^{(\ell)} \right)^{\top} \in \mathbb{R}^{N_{\text{pooled}} \times T_{\text{prunable}}}, \tag{6}$$

followed by averaging along the Query dimensi

$$S^{(\ell)} = \frac{1}{N_{\text{pooled}}} \sum_{i=1}^{N_{\text{pooled}}} S_{\text{matrix}}^{(\ell)}[i,:] \in \mathbb{R}^{T_{\text{prunable}}}. \tag{7}$$

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Magnitude distributions of the Key and Value. The Key demonstrates significant channelwise outliers, with a small subset of channels exhibiting magnitudes substantially larger than the others. In contrast, the distribution of the Value is more uniform, with no prominent outlier behavior.

<span id="page-5-2"></span>Table 1: Analysis of quantization errors for per-token and per-channel schemes. The errors are computed based on the Mean Squared Error (MSE) metric with a group size of 64.

|        | Bits | Per-Token Quantization | Per-Channel Quantization |
|--------|------|------------------------|--------------------------|
| Vove   | INT4 | $2.007 \times 10^{-3}$ | $3.635 \times 10^{-4}$   |
| Keys   | INT2 | $5.183 \times 10^{-2}$ | $9.181 \times 10^{-3}$   |
| Values | INT4 | $5.035 \times 10^{-4}$ | $4.704 \times 10^{-4}$   |
| values | INT2 | $2.641 \times 10^{-2}$ | $2.190 \times 10^{-2}$   |

Based on the resulting importance scores, the top-k tokens are selected from the prunable middle region. Let  $\mathcal{I}_{\text{middle}}$  denote the indices of the selected tokens. The final set of retained tokens is then given by

$$\mathcal{I}_{\text{keep}} = \{1, \dots, T_{\text{first}}\} \cup \mathcal{I}_{\text{middle}} \cup \{T - T_{\text{current}} + 1, \dots, T\}.$$
(8)

The same indices are synchronously applied to both the Key and Value tensors to maintain consistency. With the proposed pruning strategy, the cache size increases linearly only until reaching the budget  $\mathcal{L}_{max}$ , after which it remains fixed. Consequently, the per-frame temporal attention cost transitions from linear growth to a constant upper bound.

#### <span id="page-5-0"></span>3.3 Dimension-Adaptive KV Quantization Based on KV Distribution Characteristics

To achieve extreme compression of the KV cache, we leverage the inherent distribution patterns of KV tensors and propose a dimension-adaptive KV quantization scheme. In this work, we adopt the widely used asymmetric uniform quantization scheme [17], which is parameterized by a scale factor s, a zero-point z, and a bit-width b. Given a floating-point tensor  $x \in \mathbb{R}^d$ , its quantized representation  $\hat{x}$  is computed as

$$\hat{x} = \operatorname{clamp}\left(\left\lfloor \frac{x}{s} \right\rfloor + z, \ 0, \ 2^b - 1\right),\tag{9}$$

with the scale factor and zero-point defined as

$$s = \frac{x_{\text{max}} - x_{\text{min}}}{2^b - 1}, \qquad z = \left\lfloor -\frac{x_{\text{min}}}{s} \right\rfloor. \tag{10}$$

Here,  $\lfloor \cdot \rfloor$  denotes rounding to the nearest integer, and  $\operatorname{clamp}(\cdot)$  restricts values to the valid range. The scale factor s determines the quantization step size, while the zero-point z ensures that the zero is mapped to the quantized range.

After establishing the basic quantization strategy, we next determine the appropriate quantization granularity (i.e., per-tensor, per-token, or per-channel). A principled understanding of the distributional properties of KV tensors in StreamVGGT is crucial for achieving robust and accurate quantization. We conducted an extensive analysis of the KV distribution in StreamVGGT, revealing the distinct distribution patterns of Keys and Values within 3D reconstruction transformer models. As shown in Figure 4, the Key tensors exhibit significant channel-wise outliers, a behavior that is much less pronounced in the Value tensors. When applying standard per-tensor or per-token quantization, these outliers dominate the dynamic range, leading to inflated quantization scales and a marked reduction in effective precision, which, in turn, causes substantial performance degradation [18].

Table 2: 3D reconstruction comparison on NRGBD dataset.

<span id="page-6-0"></span>

|                    | NRGBD          |                  |       |       |       |       |       |         |
|--------------------|----------------|------------------|-------|-------|-------|-------|-------|---------|
|                    |                |                  | Ac    | c ↓   | Con   | np↓   | NC    | <u></u> |
| Method             | Inference Type |                  | Mean  | Med.  | Mean  | Med.  | Mean  | Med.    |
| VGGT               | Offline        | Unbounded Memory | 0.079 | 0.020 | 0.083 | 0.023 | 0.909 | 0.989   |
| StreamVGGT         | On line        | Unbounded Memory | 0.085 | 0.044 | 0.079 | 0.038 | 0.862 | 0.986   |
| XStreamVGGT (Ours) | Online         | Bounded Memory   | 0.085 | 0.049 | 0.075 | 0.038 | 0.850 | 0.986   |

Table 3: 3D reconstruction comparison on 7-Scenes dataset.

<span id="page-6-1"></span>

|                    |                |                       | 7 Scenes |       |       |       |       |       |
|--------------------|----------------|-----------------------|----------|-------|-------|-------|-------|-------|
|                    |                |                       | Ac       | c ↓   | Con   | np↓   | NC    | !↑    |
| Method             | Inference Type |                       | Mean     | Med.  | Mean  | Med.  | Mean  | Med.  |
| VGGT               | Offline        | Unbounded Memory      | 0.085    | 0.038 | 0.089 | 0.039 | 0.786 | 0.889 |
| StreamVGGT         | Online         | Unbounded Memory      | 0.132    | 0.058 | 0.116 | 0.042 | 0.749 | 0.863 |
| XStreamVGGT (Ours) | Online         | <b>Bounded Memory</b> | 0.142    | 0.068 | 0.125 | 0.048 | 0.734 | 0.848 |

Guided by these observations, we propose a per-channel Key and per-token Value quantization scheme that explicitly addresses channel-wise outliers. As presented in Table 1, our analysis confirms that this scheme significantly mitigates quantization error. Unlike LLMs, where the KV cache typically expands by a single token at each decoding step, StreamVGGT processes inputs in a frame-wise manner. This frame-wise processing results in a rapid expansion of the KV cache, adding a large number of patch tokens per time step. As a result, StreamVGGT is inherently well-suited for perchannel Key quantization schemes that span multiple tokens. As illustrated in Figure 2, we tightly couple quantization with pruning. Specifically, quantization is applied to the final KV cache,  $\tilde{K}_{1:t-1}^{(\ell)}$  and  $\tilde{V}_{1:t-1}^{(\ell)}$ , which comprises both the pruned historical KVs and the preserved KVs from the first and current frames. The quantization is defined as

$$\hat{K}_c = \mathcal{Q}_c \left( \tilde{K}_c; s_c^K, z_c^K \right), \qquad \hat{V}_t = \mathcal{Q}_t \left( \tilde{V}_t; s_t^V, z_t^V \right), \tag{11}$$

where c and t denote channel and token indices, respectively, and  $s_c^K, z_c^K$  (for Keys) and  $s_t^V, z_t^V$  (for Values) are the corresponding scale and zero-point parameters. Here,  $\mathcal{Q}_c(\cdot)$  and  $\mathcal{Q}_t(\cdot)$  denote asymmetric uniform quantization along the channel and token dimensions, respectively. During attention computation, the quantized tensors are dequantized as

$$\tilde{K}_c = \mathcal{D}\mathcal{Q}_c\Big(\hat{K}_c; s_c^K, z_c^K\Big), \qquad \tilde{V}_t = \mathcal{D}\mathcal{Q}_t\Big(\hat{V}_t; s_t^V, z_t^V\Big), \tag{12}$$

where  $\mathcal{DQ}_c(\cdot)$  and  $\mathcal{DQ}_t(\cdot)$  denote the dequantization operators.

#### 4 Experiments

## 4.1 Experimental Details

We perform a comprehensive evaluation of XStreamVGGT, encompassing benchmark performance, efficiency analysis, ablation studies, and qualitative results. XStreamVGGT is first assessed on various 3D tasks, including 3D reconstruction, camera pose estimation and depth estimation. Following the Point3R protocol [34], input images are processed with variable aspect ratios and resized to ensure the maximum edge length does not exceed 518 pixels. For pruning, the pooling size is set to 16, and the cache length is set to 2K, ensuring that at least the first and current frames are preserved. KV quantization is performed using KIVI with INT4 and a group size of 64 [17]. Remarkably, even with a cache length of only 2K and additional quantization, XStreamVGGT achieves exceptional efficiency while maintaining robust performance, with negligible loss across most tasks compared to the more expensive full-KV-length StreamVGGT [40].

## 4.2 3D Reconstruction

The 3D reconstruction experiments were conducted on the 7-Scenes [22] and NRGBD [2] datasets by measuring the discrepancies between predictions and the corresponding ground-truth point clouds.

Table 4: Camera pose estimation.

<span id="page-7-0"></span>

|                    |                |                  |       | TUM                    |                      | ScanNet         |             |           |  |
|--------------------|----------------|------------------|-------|------------------------|----------------------|-----------------|-------------|-----------|--|
| Method             | Inference Type |                  | ATE ↓ | RPE trans $\downarrow$ | RPE rot $\downarrow$ | $ATE\downarrow$ | RPE trans ↓ | RPE rot ↓ |  |
| VGGT               | Offline        | Unbounded Memory | 0.053 | 0.032                  | 3.209                | 0.157           | 0.056       | 3.635     |  |
| StreamVGGT         | Online         | Unbounded Memory | 0.062 | 0.033                  | 3.208                | 0.160           | 0.057       | 3.688     |  |
| XStreamVGGT (Ours) | Online         | Bounded Memory   | 0.068 | 0.035                  | 3.184                | 0.171           | 0.061       | 3.837     |  |

Table 5: Monocular depth estimation.

<span id="page-7-1"></span>

|             | Sintel         |                       |          | Box      | nn       | KIT      | ΓI       |         |
|-------------|----------------|-----------------------|----------|----------|----------|----------|----------|---------|
| Method      | Inference Type |                       | Abs Rel↓ | δ<1.25 ↑ | Abs Rel↓ | δ<1.25 ↑ | Abs Rel↓ | δ<1.25↑ |
| VGGT        | Offline        | Unbounded Memory      | 0.274    | 67.4     | 0.054    | 97.2     | 0.071    | 94.1    |
| StreamVGGT  | Online         | Unbounded Memory      | 0.254    | 68.5     | 0.052    | 97.1     | 0.072    | 94.7    |
| XStreamVGGT | Online         | <b>Bounded Memory</b> | 0.254    | 68.5     | 0.052    | 97.1     | 0.072    | 94.7    |

We adopt Accuracy (Acc), Completion (Comp), and Normal Consistency (NC) as evaluation metrics. Although XStreamVGGT uses bounded memory, it effectively maintains performance across various metrics. As shown in Table 2, on the NRGBD dataset, XStreamVGGT maintains stable performance across several metrics, with only a slight reduction in the others. NC sees only a slight drop of 1.4% (Mean) and 0.3% (Med) compared to StreamVGGT, demonstrating efficient memory management with minimal trade-offs. As shown in Table 3, XStreamVGGT demonstrates strong geometric fidelity on the 7-Scenes dataset, achieving a mean NC score of 0.734, which represents only a 2% drop compared to StreamVGGT (0.749).

#### 4.3 Camera Pose Estimation

We evaluate camera pose estimation on the TUM Dynamics [24] and ScanNet [8] datasets, truncating all sequences to 90 frames. We report Absolute Translation Error (ATE), Relative Translation Error (RPE $_{trans}$ ), and Relative Rotation Error (RPE $_{rot}$ ) as evaluation metrics. As shown in Table 4, XStreamVGGT achieves nearly lossless performance on the Sintel dataset, exhibiting only a minimal increase in ATE (0.006) and translational RPE (0.002) compared with StreamVGGT. The change in rotational RPE is even smaller, with an increase of only 0.025 (approximately 0.8%). These minor increases highlight XStreamVGGT's ability to efficiently manage memory while maintaining nearly identical performance across key metrics.

#### 4.4 Depth Estimation

Following Monst3R [37], we evaluate monocular and video depth estimation performance on the Sintel [3], Bonn [19], and KITTI [13] datasets, which encompass a diverse range of dynamic and static, indoor and outdoor scenes. The evaluation metrics include Absolute Relative Error (Abs Rel) and  $\delta < 1.25$  (the percentage of predicted depths within a 1.25 factor of the ground-truth depth). For monocular depth estimation, the results in Table 5 demonstrate that XStreamVGGT fully preserves the performance of StreamVGGT, yielding no observable degradation in any evaluation metric. For video depth estimation, as shown in Table 5 and 6, XStreamVGGT experiences mostly negligible performance degradation with a 2K cache length. Although the relative drop on Bonn is more noticeable due to its exceptionally strong baseline, XStreamVGGT still achieves very high absolute performance (0.073 Abs Rel). In contrast to StreamVGGT, which faces significant GPU memory surges and ultimately runs into Out of Memory (OOM) issues as the number of input frames increases, XStreamVGGT maintains a consistent GPU memory usage and achieves nearly 6x speedup with lossless results.

#### 4.5 Ablation Study

Figure 5a presents an ablation study on cache length, with lengths varying across 2K, 4K, 6K, and 8K. As shown, performance changes minimally as cache length increases, suggesting significant redundancy among multi-frame patch tokens. A cache length of 2K provides strong performance while maintaining high efficiency, making it the optimal choice. Additionally, we conduct an ablation study on both the pruning and quantization processes of XStreamVGGT to assess their individual

Table 6: Video depth evaluation.

<span id="page-8-0"></span>

|             | Sintel         |                  |          | Box      | nn       | KITTI    |          |          |
|-------------|----------------|------------------|----------|----------|----------|----------|----------|----------|
| Method      | Inference Type |                  | Abs Rel↓ | δ<1.25 ↑ | Abs Rel↓ | δ<1.25 ↑ | Abs Rel↓ | δ<1.25 ↑ |
| VGGT        | Offline        | Unbounded Memory | 0.301    | 68.3     | 0.057    | 96.8     | 0.061    | 97.0     |
| StreamVGGT  | On line        | Unbounded Memory | 0.328    | 65.8     | 0.058    | 97.2     | 0.094    | 94.4     |
| XStreamVGGT | Online         | Bounded Memory   | 0.341    | 61.9     | 0.073    | 94.3     | 0.098    | 94.3     |

<span id="page-8-2"></span>Table 7: Ablation study of the pruning and quantization processes in video depth estimation.

|             |              |         |              | Sintel   |         | Bo       | nn       | KIT      | ГІ       |
|-------------|--------------|---------|--------------|----------|---------|----------|----------|----------|----------|
| Method      | Cache Length | Pruning | Quantization | Abs Rel↓ | δ<1.25↑ | Abs Rel↓ | δ<1.25 ↑ | Abs Rel↓ | δ<1.25 ↑ |
| StreamVGGT  | Unbounded    | w/o     | w/o          | 0.358    | 61.1    | 0.080    | 96.5     | 0.198    | 69.0     |
| XStreamVGGT | 2K           | w/      | w/o          | 0.343    | 61.4    | 0.067    | 94.9     | 0.224    | 58.0     |
| XStreamVGGT | 2K           | w/      | w/           | 0.341    | 61.9    | 0.073    | 94.3     | 0.206    | 63.9     |

effects. As indicated in Table 7, although pruning causes a slight performance decline, quantization introduces no additional degradation.

<span id="page-8-1"></span>![](_page_8_Figure_5.jpeg)

Figure 5: Ablation study of cache length and analysis of memory with increasing frame length.

#### 4.6 Efficiency Analysis

We evaluate the efficiency of XStreamVGGT using input sequences ranging from 50 to 1000 frames, measuring GPU memory consumption and inference speed. All experiments are conducted on a single 80GB A100 GPU. As shown in Figures 1 and 5b, both StreamVGGT and VGGT exhibit significant FPS degradation as the number of input frames increases and quickly encounter OOM errors. By contrast, XStreamVGGT consistently maintains substantially higher FPS without OOM, yielding significant efficiency gains with 4.42X lower memory usage and 5.48X faster inference compared with StreamVGGT.

#### 4.7 Qualitative Results

Figures 6 and 7 present visual comparisons of 3D reconstruction and depth estimation results between StreamVGGT and XStreamVGGT. As shown, XStreamVGGT effectively preserves visual quality, producing results that closely match the original while delivering a notable improvement in computational efficiency.

# 5 Conclusion

We present XStreamVGGT, a tuning-free method for memory-efficient streaming inference of StreamVGGT. By combining KV cache pruning with quantization, it bounds memory growth while preserving model fidelity. Extensive experiments demonstrate minimal performance loss alongside substantial reductions in memory footprint and inference latency, enabling scalable 3D streaming applications. Future work will explore adaptive cache budgets that dynamically adjust based on scene complexity and motion characteristics.

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 6: Qualitative reconstruction results comparing StreamVGGT and XStreamVGGT.

<span id="page-10-0"></span>![](_page_10_Figure_0.jpeg)

Figure 7: Qualitative depth estimation results comparing StreamVGGT and XStreamVGGT.

# 6 Acknowledgments

This work was supported in part by the Research Grants Council of the Hong Kong Special Administrative Region Government through the TRS project T45-701/22-R and GRF project 17203224, and the TCL Corporate Research (Hong Kong) Co., Limited.

# References

- <span id="page-11-2"></span>[1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. *Communications of the ACM*, 54(10):105–112, 2011.
- <span id="page-11-11"></span>[2] Dejan Azinovic, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. ´ Neural rgb-d surface reconstruction. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6290–6301. IEEE, 2022.
- <span id="page-11-13"></span>[3] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In *European conference on computer vision*, pages 611–625. Springer, 2012.
- <span id="page-11-7"></span>[4] Team M L C, Gui A, Li B, et al. Introducing longcat-flash-thinking: A technical report. *arXiv preprint arXiv:2509.18883*, 2025.
- <span id="page-11-6"></span>[5] Team M L C, Li B, Lei B, et al. Longcat-flash technical report. *arXiv preprint arXiv:2509.01322*, 2025.
- <span id="page-11-1"></span>[6] Long Chen, Yuchen Li, Chao Huang, Bai Li, Yang Xing, Daxin Tian, Li Li, Zhongxu Hu, Xiaoxiang Na, Zixuan Li, et al. Milestones in autonomous driving and intelligent vehicles: Survey of surveys. *IEEE Transactions on Intelligent Vehicles*, 8(2):1046–1056, 2022.
- <span id="page-11-9"></span>[7] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. *arXiv preprint arXiv:2509.26645*, 2025.
- <span id="page-11-12"></span>[8] Angela Dai et al. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 5828–5839. IEEE, 2017.
- <span id="page-11-10"></span>[9] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*, 2023.
- <span id="page-11-0"></span>[10] Shaveta Dargan, Shally Bansal, Munish Kumar, Ajay Mittal, and Krishan Kumar. Augmented reality: A comprehensive review. *Archives of Computational Methods in Engineering*, 30(2):1057–1080, 2023.
- <span id="page-11-3"></span>[11] Jan-Michael Frahm, Pierre Fite-Georgel, David Gallup, Tim Johnson, Rahul Raguram, Changchang Wu, Yi-Hung Jen, Enrique Dunn, Brian Clipp, Svetlana Lazebnik, et al. Building rome on a cloudless day. In *European conference on computer vision*, pages 368–381. Springer, 2010.
- <span id="page-11-5"></span>[12] Yasutaka Furukawa and Jean Ponce. Accurate, dense, and robust multiview stereopsis. *IEEE transactions on pattern analysis and machine intelligence*, 32(8):1362–1376, 2009.
- <span id="page-11-14"></span>[13] Andreas Geiger, Philip Lenz, et al. Vision meets robotics: The kitti dataset. *The international journal of robotics research*, 32(11):1231–1237, 2013.
- <span id="page-11-4"></span>[14] Xiaodong Gu, Zhiwen Fan, Siyu Zhu, Zuozhuo Dai, Feitong Tan, and Ping Tan. Cascade cost volume for high-resolution multi-view stereo and stereo matching. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 2495–2504. IEEE, 2020.
- <span id="page-11-8"></span>[15] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun S Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. *Advances in Neural Information Processing Systems*, 37:1270–1303, 2024.

- <span id="page-12-7"></span>[16] Haoyang Li, Yiming Li, Anxin Tian, Tianhao Tang, Zhanchao Xu, Xuejia Chen, Nicole Hu, Wei Dong, Qing Li, and Lei Chen. A survey on large language model acceleration based on kv cache management. *arXiv preprint arXiv:2412.19442*, 2024.
- <span id="page-12-6"></span>[17] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. *arXiv preprint arXiv:2402.02750*, 2024.
- <span id="page-12-11"></span>[18] Markus Nagel et al. A white paper on neural network quantization. *arXiv preprint arXiv:2106.08295*, 2021.
- <span id="page-12-14"></span>[19] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguere, and Cyrill Stachniss. Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In *2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 7855–7862. IEEE, 2019.
- <span id="page-12-1"></span>[20] Nicole Robinson, Brendan Tidd, Dylan Campbell, Dana Kulic, and Peter Corke. Robotic ´ vision for human-robot interaction and collaboration: A survey and systematic review. *ACM Transactions on Human-Robot Interaction*, 12(1):1–66, 2023.
- <span id="page-12-0"></span>[21] Taha Samavati and Mohsen Soryani. Deep learning-based 3d reconstruction: a survey. *Artificial Intelligence Review*, 56(9):9175–9219, 2023.
- <span id="page-12-12"></span>[22] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. Scene coordinate regression forests for camera relocalization in rgb-d images. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2930–2937. IEEE, 2013.
- [23] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In *European conference on computer vision*, pages 746–760. Springer, 2012.
- <span id="page-12-13"></span>[24] Jürgen Sturm, Nikolas Engelhard, et al. A benchmark for the evaluation of rgb-d slam systems. In *2012 IEEE/RSJ international conference on intelligent robots and systems*, pages 573–580. IEEE, 2012.
- <span id="page-12-9"></span>[25] Zunhai Su, Zhe Chen, Wang Shen, Hanyu Wei, Linge Li, Huangqi Yu, and Kehong Yuan. Rotatekv: Accurate and robust 2-bit kv cache quantization for llms via outlier-aware adaptive rotations. *arXiv preprint arXiv:2501.16383*, 2025.
- <span id="page-12-10"></span>[26] Zunhai Su, Wang Shen, Linge Li, Zhe Chen, Hanyu Wei, Huangqi Yu, and Kehong Yuan. Akvq-vl: Attention-aware kv cache adaptive 2-bit quantization for vision-language models. *arXiv preprint arXiv:2501.15021*, 2025.
- <span id="page-12-8"></span>[27] Zunhai Su and Kehong Yuan. Kvsink: Understanding and enhancing the preservation of attention sinks in kv cache quantization for llms. *arXiv preprint arXiv:2508.04257*, 2025.
- <span id="page-12-5"></span>[28] Meituan LongCat Team, Anchun Gui, Bei Li, Bingyang Tao, Bole Zhou, Borun Chen, Chao Zhang, Chen Gao, Chen Zhang, Chengcheng Han, et al. Longcat-flash-thinking-2601 technical report. *arXiv preprint arXiv:2601.16725*, 2026.
- <span id="page-12-2"></span>[29] Jianyuan Wang, Minghao Chen, Nikita Karaev, et al. Vggt: Visual geometry grounded transformer. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 5294–5306. IEEE, 2025.
- <span id="page-12-4"></span>[30] Qianqian Wang, Yifei Zhang, Aleksander Holynski, et al. Continuous 3d perception model with persistent state. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 10510–10522. IEEE, 2025.
- <span id="page-12-3"></span>[31] Shuzhe Wang, Vincent Leroy, Yohann Cabon, et al. Dust3r: Geometric 3d vision made easy. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20697–20709. IEEE, 2024.

- <span id="page-13-0"></span>[32] Changchang Wu. Towards linear-time incremental structure from motion. In *2013 International Conference on 3D Vision-3DV 2013*, pages 127–134. IEEE, 2013.
- <span id="page-13-3"></span>[33] Qiong Wu, Wenhao Lin, Yiyi Zhou, Weihao Ye, Zhanpeng Zen, Xiaoshuai Sun, and Rongrong Ji. Accelerating multimodal large language models via dynamic visual-token exit and the empirical findings. *arXiv preprint arXiv:2411.19628*, 2024.
- <span id="page-13-5"></span>[34] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. *arXiv preprint arXiv:2507.02863*, 2025.
- [35] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*, 2023.
- <span id="page-13-2"></span>[36] Weihao Ye et al. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pages 22128–22136. AAAI, 2025.
- <span id="page-13-6"></span>[37] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. *arXiv preprint arXiv:2410.03825*, 2024.
- <span id="page-13-4"></span>[38] Yifan Zhang et al. Snapmla: Efficient long-context mla decoding via hardware-aware fp8 quantized pipelining. *arXiv preprint arXiv:2602.10718*, 2026. v2.
- [39] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36:34661–34710, 2023.
- <span id="page-13-1"></span>[40] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. *arXiv preprint arXiv:2507.11539*, 2025.