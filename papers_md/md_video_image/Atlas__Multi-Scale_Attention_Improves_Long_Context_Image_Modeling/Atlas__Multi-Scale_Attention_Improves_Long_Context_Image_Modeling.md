## Atlas: Multi-Scale Attention Improves Long Context Image Modeling

Kumar Krishna Agrawal \* 1 † Long Lian \* 1 Longchao Liu 1 Natalia Harguindeguy 1 2 Boyi Li 1 Alexander Bick 3 Maggie Chung 2 Trevor Darrell 1 Adam Yala 1 2

#### **Abstract**

Efficiently modeling massive images is a longstanding challenge in machine learning. To this end, we introduce Multi-Scale Attention (MSA). MSA relies on two key ideas, (i) multi-scale representations (ii) bi-directional cross-scale communication. MSA creates O(log N) scales to represent the image across progressively coarser features and leverages cross-attention to propagate information across scales. We then introduce Atlas, a novel neural network architecture based on MSA. We demonstrate that Atlas significantly improves the compute-performance tradeoff of long-context image modeling in a high-resolution variant of ImageNet 100. At 1024px resolution, Atlas-B achieves 91.04% accuracy, comparable to ConvNext-B (91.92%) while being 4.3x faster. Atlas is 2.95x faster and 7.38% better than FasterViT, 2.25x faster and 4.96% better than LongViT. In comparisons against MambaVision-S, we find Atlas-S achieves 5%, 16% and 32% higher accuracy at 1024px, 2048px and 4096px respectively, while obtaining similar runtimes. Code for reproducing our experiments and pretrained models is available at https://github.com/yalalab/atlas.

## 1. Introduction

Long-context image modeling remains a fundamental challenge in computer vision with broad applications to biomedicine (Xu et al., 2024), satellite imagery (Rad, 2024), and vision-language modeling (Gemini-Team et al., 2023; Wang et al., 2024; Qwen-Team, 2025; Chen et al., 2024). At the core of this challenge is a compute expressivity trade-off; we aim to develop models that efficiently scale to massive input sequences while capturing arbitrary pair-wise depen-

![](_page_0_Figure_8.jpeg)

<span id="page-0-0"></span>Figure 1. (a) Training efficiency comparison of different vision architectures on HR-IN100 across increasing input resolutions (1024-4096px). (b) *Atlas* exhibits similar runtime scaling as MambaVision while obtaining significantly better accuracy.

dencies between input tokens. As shown in Figure 1(a), self-attention, as used in Vision Transformers, is highly expressive, but its computational cost scales poorly (i.e., quadratically) with sequence length. It remains infeasible to train end-to-end Vision Transformers on massive imaging modalities such as mammograms or whole-slide pathology images. At another end of the spectrum, state space models (SSMs) and recurrent architectures are highly efficient, achieving linear computational complexity; however, SSM-based models perform poorly in long-context imaging modeling (Figure 1b).

<sup>\*</sup>Equal contribution, †Project lead <sup>1</sup>University of California, Berkeley <sup>2</sup>University of California San Francisco <sup>3</sup>Vanderbilt University . Correspondence to: Kumar Krishna Agrawal <kagrawal@berkeley.edu>, Adam Yala <yala@berkeley.edu>.

Long-context image modeling requires novel neural primitives and new benchmarks to guide their development. Recent work in efficient architecture design, such as FasterViT [\(Hatamizadeh et al.,](#page-8-2) [2023\)](#page-8-2) and MambaVision [\(Hatamizadeh](#page-8-3) [& Kautz,](#page-8-3) [2024\)](#page-8-3), has primarily focused on improving the throughput vs accuracy trade-offs in the context of standard resolution ImageNet experiments (224 × 224). While valuable, this setting yields little insight into how methods scale to larger input resolutions. To this end, we propose a new high-resolution benchmark based on ImageNet-100 (HR-IN 100). We evaluate the speed vs accuracy trade-off of different neural networks at progressively larger resolutions, ranging from 1024 × 1024 to 4096 × 4096 images. As input resolution increases, long-range communication across distant parts of the image becomes more essential for image classification, and asymptotic computational complexity begins to dominate model runtime.

In designing a novel neural primitive, we aim to enable arbitrary cross-token interaction with minimal intermediate steps (i.e., communication complexity) while minimizing computational complexity as a function of input sequence length. To this end, we propose Multiscale Attention (MSA), a novel primitive for high-resolution image modeling. MSA is built on two key ideas: multiscale representations and cross-scale communication. In each MSA block, we leverage a simple S-token max-pooling kernel to summarize small spatial regions (e.g., 4x4 input region), into progressively coarser summary representations across O(log<sup>S</sup> N) spatial scales, where N is the total sequence length. We then leverage a windowed cross-attention mechanism to enable information-sharing between tokens at different scales. At each scale, tokens attend to nearby tokens of the same scale and tokens from all coarser scales. This "top-down" communication enables MSA to integrate information across the entire sequence. Each scale's tokens also cross-attend to its "parent" finer-grain scale tokens, allowing each coarse token to refine its representation through "bottom-up" communication. Altogether, this bi-directional communication pattern enables information mixing between all input tokens through O(log N) intermediate tokens (i.e. coarse scale representations) and within O(N log N) runtime. In controlled block-level experiments (see Table [3\)](#page-7-0), we find that MSA outperforms alternative neural network primitives in long-context modeling, including LongNet's dilated attention [\(Ding et al.,](#page-8-4) [2023\)](#page-8-4), MambaVision Mixer [\(Hatamizadeh](#page-8-3) [& Kautz,](#page-8-3) [2024\)](#page-8-3), and FasterViT's Hierarchical Attention [\(Hatamizadeh et al.,](#page-8-2) [2023\)](#page-8-2).

We propose *Atlas*, a novel architecture designed around the unique advantages of MSA. Given a sequence length N, which defines log N scales within MSA, Atlas leverages log N macro-stages to progressively down-sample the input until MSA recovers only a single scale. We leverage the rich scale-2 representations of our MSA block as a

down-sampling mechanism, enabling both faster and more performant down-sampling than traditional approaches. We demonstrate that *Atlas* significantly improves the Pareto frontier in long-context modeling. In 1024 × 1024 experiments, as shown in Table [1,](#page-6-0) *Atlas* obtains comparable runtime to MambaVision (23.1hr vs 22.6hr) on the same hardware, while obtaining 6.1% higher accuracy (91.04 vs 84.86). Compared to FasterViT and LongViT, *Atlas* is 2.95× and 2.25× faster, obtaining 7.38% (91.04 vs 83.66) and 4.96% (91.04 vs 86.08) higher accuracy, respectively. Moreover, the performance advantage of *Atlas* is especially pronounced as we scale input resolution to 4096px, achieving a 34% accuracy improvement over MambaVision at similar runtime.

We summarize our contributions as follows:

- We propose a High-Res ImageNet-100 (HR-IN 100), an efficient benchmark with input resolutions ranging from 1024 × 1024 to 4096 × 4096 for evaluating the frontier of long-context image modeling.
- We introduce Multi-Scale Attention (MSA), a novel neural network primitive that maintains representations across O(log N) spatial scales and enables bidirectional information mixing across all scales within O(N log N) runtime. Building on MSA, we introduce *Atlas*, a novel neural network architecture.
- With extensive experiments on High-Res ImageNet-100, we demonstrate that *Atlas* improves the Pareto frontier in long-context image modeling. *Atlas* outperforms representative efficient architectures in longcontext image modeling, including FasterViT, MambaVision, and LongViT.

## 2. Related Work

Vision Transformers (ViTs). ViTs [\(Dosovitskiy,](#page-8-5) [2020\)](#page-8-5) directly apply Transformers [\(Vaswani,](#page-9-4) [2017\)](#page-9-4) architecture to image patches, demonstrating the effectiveness of selfattention in visual data processing. Building on this, DeiT [\(Touvron et al.,](#page-9-5) [2021\)](#page-9-5) improves training data efficiency. However, the self-attention primitive in ViT, which scales quadratically with input sequence length, limits its application toward high-resolution imaging. Our study focuses on developing efficient alternatives to standard self-attention to enable expressive and computationally efficient longcontext image modeling.

Efficient Long Sequence Modeling in Language. To address the challenges of long-sequence modeling, LongNet [\(Ding et al.,](#page-8-4) [2023\)](#page-8-4) introduces a dilated attention mechanism, allowing transformers to process sequences with up to one million tokens. LongNet was later adapted into a vision model as LongViT [\(Wang et al.,](#page-9-6) [2023\)](#page-9-6) to process whole-slide pathology images. State Space Models (SSMs), such as Mamba [\(Gu & Dao,](#page-8-6) [2023\)](#page-8-6), provide a linear-time

![](_page_2_Figure_1.jpeg)

Figure 2. The Atlas architecture consists of a convolutional stem for initial feature extraction, followed by a series of Multi-Scale Attention (MSA) blocks that progressively downsample the feature maps while preserving global context. This hierarchical design facilitates the effective processing of high-resolution images with efficient communication between features.

alternative to full attention for efficient long sequence modeling. RetNet [\(Sun et al.,](#page-9-7) [2023\)](#page-9-7) combines the strengths of recurrence and attention, enabling linear-time sequence modeling. Longformer [\(Beltagy et al.,](#page-8-7) [2020\)](#page-8-7) integrated local and global attention for effective long-document processing. Our work is most similar to LongNet, which also achieves a communication complexity of O(log N), where N is the length of the input sequence.

Instead of using dilated attention, we propose multiscale attention (MSA), which captures distant dependencies by attending to a subset of the input through intermediate "coarser scale" tokens. Unlike dilated attention, MSA effectively leverages locality in the input, resulting in significantly improved long-context vision modeling.

Efficient Visual Modeling. Vim and VMamba [\(Zhu et al.,](#page-9-8) [2024;](#page-9-8) [Liu et al.,](#page-8-8) [2024\)](#page-8-8) adapted State-space models (SSMs) to vision-specific tasks and demonstrated the effectiveness of SSMs for visual representation learning. MambaVision [\(Hatamizadeh & Kautz,](#page-8-3) [2024\)](#page-8-3) proposed a hybrid SSM and self-attention-based architecture, and demonstrated improved performance over other SSM-based architectures. Swin [\(Liu et al.,](#page-8-9) [2021\)](#page-8-9) proposes leveraging window-shifting for cross-window communication and a hierarchical design to aggregate context. CSwin [\(Dong et al.,](#page-8-10) [2022\)](#page-8-10) proposes cross-shaped window attention to capture global and local dependencies. CrossViT [\(Chen et al.,](#page-8-11) [2021a\)](#page-8-11) uses a dualbranch architecture to process image patches of varying sizes. EdgeViT [\(Pan et al.,](#page-9-9) [2022\)](#page-9-9) and EfficientFormer [\(Li](#page-8-12) [et al.,](#page-8-12) [2022\)](#page-8-12) designed lightweight transformers that are specially optimized for edge devices. VisFormer [\(Chen et al.,](#page-8-13) [2021b\)](#page-8-13) combined convolutions and transformers for vision tasks. Twins [\(Chu et al.,](#page-8-14) [2021\)](#page-8-14) improved the spatial attention mechanisms for improved performance. The long-short transformer [\(Zhu et al.,](#page-9-10) [2021\)](#page-9-10) introduced hybrid attention for efficient modeling in vision and language. FasterViT [\(Hatamizadeh et al.,](#page-8-2) [2023\)](#page-8-2) introduced hierarchical attention for fast visual information processing, and demonstrated improved performance over Swin, Twins, CrossViT, and EfficientFormer. Focal Transformer [\(Yang et al.,](#page-9-11) [2021\)](#page-9-11) explores

<span id="page-2-0"></span>a new form of attention and Pyramid Vision Transformer (PVT) [\(Wang et al.,](#page-9-12) [2021\)](#page-9-12) explores hierarchical attention for efficient modeling. Unlike these works, which focus on improving compute-accuracy trade-offs in modest resolution regimes (i.e. 224 x 224 pixels), our work focuses on modeling high-resolution images. In this context, we find that representative efficient architectures, including MambaVision and FasterViT, fail to effectively process high-resolution images.

#### Multi-resolution representations in Neural Networks.

Dense cross-scale communication has been explored in the context of CNNs, as in DenseNets [\(Huang et al.,](#page-8-15) [2017\)](#page-8-15) and Feature Pyramid Networks (FPNs) [\(Lin et al.,](#page-8-16) [2017\)](#page-8-16). In these works, feature maps across multiple resolutions are integrated using fixed operations, including concatenation or summation. In contrast, we propose to fuse representations across scales in our MSA block through cross-attention. This data-dependent multi-scale integration strategy allows our model to learn complex interactions between features at different resolutions and optimize the fusion process jointly with feature extraction.

## 3. Method

We propose Multi-Scale attention (MSA), a novel neural primitive for long-context image modeling. MSA builds representations across multiple spatial scales and leverages dense cross-attention operations to share information across scales. Building on this primitive, we build *Atlas*, a hierarchical macro-architecture that uses the intermediate scales in MSA as a novel down-sampling mechanism.

## 3.1. Preliminaries

Windowed self-attention (WA) adapts the standard Multi-Head self-attention (MHSA) to operate efficiently on local regions of an input feature map. To lay the groundwork for multi-scale attention (MSA), we first describe the WA operation and analyze its computational benefits and limitations.

Windowed Self-attention. Consider a feature map X ∈

![](_page_3_Figure_1.jpeg)

Figure 3. Illustration of top-down and bottom-up hierarchical communication in Multi-Scale Attention (MSA). The top-down Global Context Aggregation enables coarse-to-fine feature propagation. The bottom-up fine-to-coarse pathway propagates high resolution features into coarser scale representations.

 $\mathbb{R}^{H \times H \times C}$ , where H is the spatial and C is the channel dimension<sup>1</sup>. The WA mechanism operates in two key steps:

- 1. Window Partitioning: Divide the feature map into non-overlapping windows of size  $k \times k$ , with the number of windows per dimension: H' = H/k, total number of windows  $M = H' \times H' = (H/k)^2$ , and each window  $W_{ij}$   $(i,j \in \{1,\ldots,H'\})$  containing  $k^2$  tokens. Further, each window  $W_{ij}$  is reshaped into a sequence, where  $W_{ij} \in \mathbb{R}^{k \times k \times C}$  is viewed as  $W_{ij} \in \mathbb{R}^{k^2 \times C}$  after reshape.
- 2. **Local Self-Attention**: Apply standard Multi-Head Self-Attention (MHSA) within each window:

$$Attention(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
 where  $Q, K, V = \operatorname{Linear Projections}(W_{ij})$ 

The computational complexity of WA within a single window is  $O(k^2 \cdot k^2) = O(k^4)$  due to the attention operation within  $k^2$  tokens. Since there are  $M = (H/k) \times (H/k) = H^2/k^2$  windows, the total complexity of WA across the entire feature map becomes  $O(M \cdot k^4) = O(\frac{H^2}{k^2} \cdot k^4) = O(H^2k^2) = O(Nk^2)$ , where  $N = H^2$  is the total number of tokens in the feature map. This is a significant reduction from the  $O(N^2)$  complexity of global attention, especially when  $k \ll \sqrt{N}$ .

While computationally efficient, WA suffers from two key limitations: 1) **Limited Receptive Field**: Each window processes information independently, preventing direct communication between different image regions, and 2) **Boundary Effects**: Objects or features spanning multiple windows can-

<span id="page-3-3"></span>not be directly modeled within a single attention operation. For example, an object split across two windows must be processed independently in each window, relationships between parts can only be learned indirectly when all features merged at the final readout.

#### 3.2. Multi-Scale Attention

MSA's core design centers on two key components: 1) a **hierarchical representation** that creates intermediate feature scales using fixed-size summarization kernels to preserve information density and 2) **bi-directional communication** that enables effective information exchange across multiple windows and scales, through dense cross-attention.

#### <span id="page-3-4"></span>3.2.1. HIERARCHICAL REPRESENTATION

Multi-Scale Attention (MSA) builds hierarchical representations through iterative summarization with a fixed-size kernel of S-tokens. Starting with the input feature map  $F^{(1)}$  at scale-1, we create coarser representations through a summarization operation S:

<span id="page-3-1"></span>
$$F^{(l)} = \mathcal{S}(F^{(l-1)}, S), \quad \text{for } l = 2, \dots, L$$
 (1)

![](_page_3_Figure_16.jpeg)

<span id="page-3-2"></span>Figure 4. Multi-Scale features with iterative summarization.

<span id="page-3-0"></span><sup>&</sup>lt;sup>1</sup>For simplicity, we focus on square 2D feature maps, but the concept is generalizable to 1D sequences or 3D volumes.

#### Algorithm 1 Multi-Scale Attention (MSA) Block

```
\overline{\textbf{Input:} \ \mathcal{X} = [X^{(1)}, ..., X^{(L)}], \text{ where } X^{(l)} : (\mathtt{B}, \mathtt{N}_1, \mathtt{C}_{\mathtt{in}})
     k \times k \leftarrow \text{Window Size}, S \leftarrow \text{Downsampling Rate}
Output: \overline{X} = [X^{(1)}, ..., X^{(L)}], \text{ where } X^{(l)} : (B, N_1, C_{in})
     ⊳ Iterative summarization
 1: for l = 2, ..., L do

    ► Iterate from fine to coarse

          X^{(l)} += Summarize(X^{(l-1)}, S)
 2:
                                                                    ⊳ Equation (1)
 3: end for
     ▷ Top-Down Communication: Global Context Aggregation
 4: for l = L, L - 1, ..., 1 do

    ▶ Iterate from coarse to fine

          X^{(l)} \leftarrow \mathsf{CrossAttention}(X^{(l)}, [X^{(l)}, X^{(l+1)}, ..., X^{(L)}])
                                                             ⊳ as in Equation (2)
 6: end for
     ▷ Bottom-Up Communication: Fine-to-Coarse Refinement
 7: for l = 2, 3, ..., L do

    ▶ Iterate from fine to coarse

          X^{(l)} \leftarrow \mathsf{CrossAttention}(X^{(l)}, X^{(l-1)})

    b as in Equation (3)

 9: end for
10: return \overline{X} = [X^{(1)}, ..., X^{(L)}]
```

where S is implemented as strided max-pooling with a fixed stride s (i.e. downsampling rate  $S = s \times s$  tokens). This process continues until the feature map size at scale L is no larger than the window size  $k \times k$ . With input sequence length N and downsampling rate S, the number of scales L grows logarithmically as  $O(\log_S N)$ , where  $S = s^2$ .

At each scale l, we operate on windows, by partitioning the feature map  $F^{(l)}$  into non-overlapping regions of size  $k \times k$  (i.e.  $K = k^2$  tokens), yielding windows  $\{W_{ij}^{(l)}\}$ , for l = 1, ..., L scales. As shown in Figure 4, this scheme creates a directed acyclic graph (DAG) between windows at different spatial scales. With every summarization operation, we merge "parent" windows into new coarser "child" windows.

## 3.2.2. CROSS-SCALE COMMUNICATION: ATTENTION-BASED FUSION

The expressive power of MSA comes from its ability to efficiently propagate information across scales through two complementary mechanisms:

#### I. Top-Down Communication

In our top-down communication scheme, we propagate information from coarse "child" windows to their "parent" windows through a dense set of cross-attention operations.

Let  $W^{(l)}$  be a window at scale l, and  $\{W^{l+1}, \ldots, W^L\}$  denote the corresponding coarse "child" windows as illustrated in Figure 4. The cross-attention operation using standard Multi-Head Attention (MHA), as visualized in Figure 3, is

#### Algorithm 2 Atlas Architecture Pseudocode

```
Input: Img : (B, H_{in}, W_{in}, C_{in}), k \times k \leftarrow \text{window size}
     P \leftarrow \text{Patch Size}, S \leftarrow \text{Downsampling Rate}
     D \leftarrow \{d_1, d_2, ..., d_L\}

Output: predictions : (B, D<sub>out</sub>)

 1: X^{(1)} \leftarrow \mathsf{ConvPatchify}(\mathsf{Img}, P)
                                                           ⊳ scale 1 feature map
     ▶ Initialize Multi-Scale features
 2: \  \, {\bf for} \ l=2,...,L \  \, {\bf do}
       X^{(l)} \leftarrow \mathsf{Summarize}(X^{(l-1)}, S)

    ▷ Strided MaxPool

 4: end for
     ▷ Progressive Downsampling
 5: for s = 1, 2, ..., L do

    ▶ Iterate through stages

          for blk = 1, 2, ..., d_s do
 6:
               [X^{(s)}, ..., X^{(L)}] \leftarrow \mathsf{MSABlock}([X^{(s)}, ..., X^{(L)}], k, S)
 7:
                                                            ▶ Apply MSA Block
 8:
 9:
          end for
10: end for
11: predictions \leftarrow readout(X^{(L)})
12: return predictions
```

then given by:

<span id="page-4-0"></span>
$$W^{(l)} = MHA(Q_l, [K_{l:L}], [V_{l:L}])$$
(2)

where  $Q_l$ ,  $K_l$ ,  $V_l$  are query/key/value projections of  $W^{(l)}$ , and  $K_{l+1:L}$ ,  $V_{l+1:L}$  are concatenated key/value projections from coarser scales. This operation enables MSA to model relationships between tokens within the window, while also allowing each window to read from long-context information from all coarser scale "child" windows. This dense cross-attention design allows each scale to directly observe global context through the coarsest scale "child" window. At the coarsest scale l=L, this operation recovers standard self-attention.

#### **II. Bottom-Up Communication**

The bottom-up communication in MSA complements the top-down aggregation by refining coarser-scale "child" representations with detailed information from finer-grain "parent" tokens. This is a localized operation, in the sense that the fine grain refinement for each token is guided only by its *direct* parent window.

Specifically, let  $Q_l$  be the query projection of  $W^l$ , and let  $K_{l-1}$  and  $V_{l-1}$  be the key and value projections from the parent window  $W^{(l-1)}$ . The updated window representation  $W^{(l)}$  after bottom-up communication is obtained through cross-attention as:

<span id="page-4-1"></span>
$$W^{(l)} = MHA(Q_l, K_{l-1}, V_{l-1})$$
(3)

This targeted cross-attention allows for the recovery and integration of crucial local information potentially lost in the initial summarization.

The pseudocode for the full MSA block is shown in Algorithm [1.](#page-4-2)

Asymptotic Complexity. With a feature map X ∈ R N×C and window of K tokens (typically K = k × k), downsampling rate S, MSA creates L = log<sup>S</sup> N feature scales. The most expensive operation is the dense *top-down* crossattention. In particular, for scale-1, each token cross attends to LK tokens (one window per scale), which scales to NLK complexity across a N-length sequence. The runtime for all subsequent scales 2, .., L is upper-bounded by NLK, giving an effective runtime complexity of O(NLK).

Plugging in L = log<sup>S</sup> N, we recover O(NK log<sup>S</sup> N) as the net runtime complexity of *Atlas*. Note that K and S are typically small constants depending on the hardware; in our experiments we find K = 256 (i.e. 16 × 16 window) and S = 16 (i.e. 4 × 4) to be most performant on an 8×H100 node. Our dense cross-scale communication strategy guarantees that each token must propagate information across at most O(log<sup>S</sup> N) intermediate tokens to interact with any another token in the sequence, where standard self-attention would obtain O(1) communication complexity but O(N<sup>2</sup> ) runtime complexity.

#### <span id="page-5-0"></span>3.3. *Atlas*

The MSA block can be used as drop-in replacement for the standard MHA block in existing architectures like ViT [\(Dosovitskiy,](#page-8-5) [2020\)](#page-8-5) or Swin Transformer [\(Liu et al.,](#page-8-9) [2021\)](#page-8-9). To fully leverage the benefits of MSA, we co-design the network structure for *Atlas* to optimize performance and efficiency. Our full architecture is illustrated in Figure [2,](#page-2-0) with the pseudocode in Algorithm [2.](#page-4-3)

Atlas is a multi-stage architecture, with a convolutional stem [\(Hatamizadeh et al.,](#page-8-2) [2023;](#page-8-2) [Hatamizadeh & Kautz,](#page-8-3) [2024;](#page-8-3) [Xiao et al.,](#page-9-13) [2021\)](#page-9-13), followed by multiple stages of MSA blocks. We leverage the same convolutional stem as FasterViT [\(Hatamizadeh et al.,](#page-8-2) [2023\)](#page-8-2) to obtain localized patch-level representations. In particular, the stem utilizes two stages of residual convolutional blocks, yielding in feature map of R H/16×W/16×<sup>C</sup> . Given this feature map, fixed window size K and downsampling rate S, MSA builds a multi-scale layout with L = log<sup>S</sup> N scales, as outlined in Section [3.2.1.](#page-3-4)

As part of the co-design of Atlas, we fix the number of stages of MSA blocks to be identical to the number of scales, i.e. L = log<sup>S</sup> N. The key insight behind Atlas is to progressively reduce the number of tokens at each scale, focusing computational resources on high-level features. Given the multiscale structure of the MSA block, we propose a progressive scale-dropping strategy in Atlas. In other words, for a multi-scale input X = [X(1), X(2), ..., X(L) ], stage l of the *Atlas* only processes [X(l) , X(l+1), ..., X(L) ] actively.

As a concrete instance, for MSA with L scales, let us define an Atlas config D = {d1, d2, ..., dL} with L corresponding stages. Here, d<sup>l</sup> is the number of blocks at stage l. For example, for a 4-scale MSA block, an Atlas config would have 4 stages, e.g. D = {2, 2, 2, 6}. This config indicates that the first scale is the finest resolution for the first two blocks, after which it becomes inactive and is dropped. Subsequently, the second scale becomes the finest active resolution for the next two blocks, with X(4) being the only active features for the last block.

This strategy is quite flexible, in that for a single scale MSA block, and K = N, it recovers the standard ViT with MHSA block. For the readout, there are multiple strategies to aggregate the final representations across scales. We find that simply using the last scale as the final representation works well in practice.

## 4. Experiments

#### 4.1. Image Classification

Setup. We propose using a novel high-resolution benchmark based on Imagenet-100 [\(Tian et al.,](#page-9-14) [2020\)](#page-9-14), High-Resolution ImageNet-100. The dataset extends the original Imagenet-1k dataset with ∼126K unique training samples, 5000 validation samples, and 100 classes with high-resolution images (up to 4096px), where the images are upsampled to the desired resolution. We first focus on a system's level comparison against representative architectures, including ViT, Swin, FasterViT, MambaVision, ConvNext, and LongViT. Together, these architectures encompass sparse attention, SSMs, convolutional, and dilated attention approaches. For each baseline , we utilize the provided code as is, without modifications to gradient accumulation, employing a linearly decaying learning rate proportional to the batch size, following [\(Goyal,](#page-8-17) [2017\)](#page-8-17). This ensures consistency with prior work and facilitates a fair comparison.

Comparing Architectures: We benchmark all architectures on the same hardware, 1 server with 8×H100 Nvidia GPUs using 1024px input resolution (equivalent to 4K tokens with patch-size=16). To understand the runtime-performance tradeoff of *Atlas* design against existing architectures, we train Base-scale models (i.e. 12 head, 768 embed-dim following prior work [\(Dosovitskiy,](#page-8-5) [2020\)](#page-8-5)) for 320 epochs.

Long-Context Image Modeling: To understand the efficacy of *Atlas* in long-context image modeling tasks, we seek to scale the evaluation to higher resolutions. Due to extreme cost of running our baselines for full convergence runs (320 epochs) at Base models, we focus our scaling experiments on only our two fastest models, namely Atlas and MambaVision in Small regime. As shown in Figure [1,](#page-0-0) all other architectures are significantly slower at higher resolutions. Prior work in architecture design for vision models [\(Xiao](#page-9-13)

[et al.,](#page-9-13) [2021\)](#page-9-13) demonstrates meaningful comparisons with shorter training schedules. We adopt a similar approach and train *Atlas*-S and MambaVision-S models for 100 epochs for 1024px, 2048px and 4096px, scaling upto 64K tokens.

#### 4.2. Ablations

Attention Mechanism. To understand the efficacy of different token-mixing (e.g. attention or SSM-based) blocks in long-context image modeling, we conduct controlled experiments, using the same optimizer, learning rate schedules. We consider 384 × 384 inputs, with 4 × 4 patches, giving a sequence length N = 9216. We use 4-block architectures, with Base-equivalent blocks (i.e. 12 head, 768 embed-dim following prior work [\(Dosovitskiy,](#page-8-5) [2020\)](#page-8-5). We compare our MSA block with Hierarchical Attention block [\(Hatamizadeh](#page-8-2) [et al.,](#page-8-2) [2023\)](#page-8-2), MambaVision Mixer [\(Hatamizadeh & Kautz,](#page-8-3) [2024\)](#page-8-3), dilated attention with the LongViT block [\(Ding et al.,](#page-8-4) [2023\)](#page-8-4) and standard ViT, Window-ViT blocks.

Communication Mechanism. Our proposed Multi-Scale Attention (MSA) block relies on bi-directional communication to effectively model long-context. To understand the contribution of each mechanism, we conduct controlled ablations with 256 × 256 inputs, using 4 × 4 patches, giving a sequence length N = 4096, K=256 (i.e. 16 × 16 windows), S=16 (i.e. 4 × 4 strided max-pool). In this setting we have features at two scales, providing a sandbox to test the contribution of different communication mechanisms. We use a Small-scale 4-block architecture (i.e. with 6 heads, 384 dim following [\(Dosovitskiy,](#page-8-5) [2020\)](#page-8-5)). The predictions from both scales are merged via average pool, before readout. In this setting, we compare the following variants of the block

- no-multiscale : equivalent to vanilla single-scale WA
- no communication: equivalent to WA at both scales.
- top-down only: propagates from coarse to fine only
- bottom-up only: propagates from fine to coarse only
- top-down + bottom-up: both mechanisms as in MSA.

Composition Strategies. To identify the best MSA block composition strategy, we compare three different strategies of incorporating MSA

- stack: vanilla stacking of blocks as in [\(Dosovitskiy,](#page-8-5) [2020\)](#page-8-5), with averaging tokens across scales for readout.
- convolutional downsampling : similar to prior work as in [\(Liu et al.,](#page-8-9) [2021;](#page-8-9) [Hatamizadeh et al.,](#page-8-2) [2023\)](#page-8-2) we use separate downsampling layer to reduce spatial resolution by 2 × 2 per stage. For this variant, we use a uniform 4-stage architecture, i.e. {3, 3, 3, 3}
- *Atlas*: a {d1=2, d2=10} config outlined in Section [3.3](#page-5-0)

We run each ablations with 512 × 512 inputs, using 8 × 8 patches, giving a sequence length N = 4096, with 12 Smallscale MSA blocks.

## 5. Results

#### 5.1. Image Classification

Comparing Architectures at 1024px resolution: The experimental results in Table [1](#page-6-0) demonstrate that Atlas-B/16 is competitive with/outperforms existing vision backbones in accuracy, while being computationally efficient. In particular, Atlas achieves 91.04% accuracy while delivering substantial speed advantages: 4.3× faster than ConvNext-B (91.92%), 1.15× faster than ViT (90.66%), and 1.6× faster than Swin (90.89%) with competitive accuracy. Compared to other sparse-transformer backbones, *Atlas* is 2.95x faster and 7.3% better than FasterViT, 2.25x faster and 4.96% better than LongViT. Notably, while the runtimes are comparable, *Atlas* is 6.05% better than MambaVision. Additional experimental results from our 50-epoch runs are available in the supplementary material (Table [6\)](#page-11-0).

<span id="page-6-0"></span>Table 1. Comparison of vision backbones on 1024x1024 image resolution on the HR-IN100 benchmark. Each model is evaluated on runtime (in hours), relative speed compared to Atlas, and Top-1 accuracy (in %). All models are base scale and were trained for 320 epochs until convergence on single 8 × H100 GPU node.

|                          | Architecture  | Runtime<br>(hr) ↓ | Relative<br>speedup ↓ | Top-1 Acc.<br>(%) ↑ |  |
|--------------------------|---------------|-------------------|-----------------------|---------------------|--|
|                          | ViT-B         | 26.77             | 1.15x                 | 90.66               |  |
| Transformer              | Swin-B        | 37.25             | 1.6x                  | 90.89               |  |
|                          | FasterViT-4   | 68.31             | 2.9×                  | 83.66               |  |
|                          | LongViT-B     | 52.23             | 2.2×                  | 86.08               |  |
| Convolutional ConvNext-B |               | 100.11            | 4.3×                  | 91.92               |  |
| Mamba                    | MambaVision-B | 22.69             | 0.98×                 | 84.86               |  |
| Multi-Scale              | Atlas-B       | 23.12             | 1.00×                 | 91.04               |  |

Long-Context Image Modeling: The results in Table [2](#page-7-1) demonstrate the superior scaling capabilities of *Atlas* over MambaVision on high-resolution images. While both architectures show comparable runtime efficiency on a single 8×H100 node (MambaVision requiring 4.56, 14.73, and 55.5 hours for 1024px, 2048px, and 4096px respectively), Atlas-S/16 outperforms MambaVision-S/16 by 3.62% at 1024px resolution (81.82% vs. 78.82%), with this gap widening to 16.50% at 2048px and 32.84% at 4096px. These results highlight *Atlas*'s capability to effectively capture long-range dependencies at extreme context lengths up to 64K tokens where state-space based models struggle.

## 5.2. Ablations

Attention Mechanism. To understand the efficacy of the MSA block, we run controlled ablations against existing primitives for long-context modelling. The results in Table [3](#page-7-0) highlight the effectiveness of MSA for classification. While faster in runtime, the window-attention blocks of WViT and

<span id="page-7-1"></span>Table 2. Comparison of Mamba-based (MambaVision-S/16) and Multi-Scale Attention (Atlas-S/16) models across three image resolutions: 1024px, 2048px, and 4096px. The table presents both computational efficiency (runtime in hours on single 8xH100 node) and performance (Top-1 accuracy in %) metrics. All models were trained for 100 epochs per resolution. Atlas-S/16 demonstrates superior accuracy across all resolutions, with particularly significant advantages at higher resolutions (2048px and 4096px), while maintaining comparable computational demands. The substantial increase in runtime as resolution scales highlights the computational challenges inherent in high-resolution image processing.

| Model                            |                  | Runtime (hr) ↓ |        |        | Top-1 Accuracy (%) ↑ |        |        |  |
|----------------------------------|------------------|----------------|--------|--------|----------------------|--------|--------|--|
|                                  |                  | 1024px         | 2048px | 4096px | 1024px               | 2048px | 4096px |  |
| Mamba-Based                      | MambaVision-S/16 | 4.56           | 14.73  | 55.5   | 78.2                 | 51.42  | 23.36  |  |
| Multi-Scale Attention Atlas-S/16 |                  | 3.64           | 14.23  | 54.72  | 81.82                | 67.92  | 55.84  |  |

Swin perform ∼29% worse than MSA. MambaVisionMixer [\(Hatamizadeh & Kautz,](#page-8-3) [2024\)](#page-8-3) performs ∼12% worse than MSA while requiring 0.88x the runtime. MSA outperforms the standard attention-block of ViT and the Hierarchichal Attention block from [\(Hatamizadeh et al.,](#page-8-2) [2023\)](#page-8-2), both in runtime and accuracy. MSA is 1.76× faster and ∼10% better than ViT block, while being 1.15× faster and 27% better than Hierarchical Attention.

<span id="page-7-0"></span>Table 3. Comparing different attention mechanisms at a block-level in controlled setting (100epoch runs).

| Block                          | Runtime<br>(min) ↓ | Relative<br>speedup ↓ | Top-1 acc.<br>(in %)↑ |
|--------------------------------|--------------------|-----------------------|-----------------------|
| Window ViT                     | 55                 | 0.60×                 | 41.65                 |
| ShiftedWindow ViT (Swin)       | 68                 | 0.75×                 | 41.48                 |
| ViT                            | 160                | 1.76×                 | 60.57                 |
| Hierarchical Attn. (FasterViT) | 105                | 1.15×                 | 43.19                 |
| Dilated Attn. (LongViT)        | 218                | 2.39×                 | 49.88                 |
| MambaVisionMixer               | 80                 | 0.88×                 | 58.79                 |
| Multi-Scale Attn. (Atlas)      | 91                 | 1.00×                 | 70.81                 |

Finally, the MSA block is 2.39x faster and 20.9% better than Dilated Attention block from LongViT [\(Ding et al.,](#page-8-4) [2023\)](#page-8-4). Our results suggest that the MSA block can be used as dropin replacement to existing primitives, offering significant improvements for long-context modeling.

Communication Mechanism. The MSA block develops a bi-directional communication to efficiently model longcontext modeling. The results in Table [4](#page-7-2) demonstrate that MSA significantly improves on vanilla Window-Self Attention (WA), improving classification accuracy by ∼12.5% (72.02 vs 59.39). Furthermore, we show that relying only on multi-scale features with WA is suboptimal, resulting in a 6.9% performance drop. While the *top-down* and *bottom-up* communication mechanisms, independently boost the accuracy by ∼3.5% each, they are complementary to each other. Using the bi-directional communication strategy (i.e. MSA) improves ∼3% over relying on only one of the mechanisms.

Composition Strategies. Next, we studied how to best com-

Table 4. Ablations on the communication strategies.

<span id="page-7-2"></span>

| Communication Strategy                   | Top-1 (%) ↑ |
|------------------------------------------|-------------|
| single-scale only (WindowViT)            | 59.39       |
| multi-scale only                         | 65.14       |
| multi-scale + bottom-up                  | 69.92       |
| multi-scale + top-down                   | 69.04       |
| multi-scale + bottom-up + top-down (MSA) | 72.02       |

pose MSA blocks into an efficient macro-architecture. As shown in Table [5,](#page-7-3) stacking MSA blocks without progressive downsampling resulted in an accuracy of 69.88% at a runtime of 75 minutes. Convolutional downsampling between MSA blocks accelerated training, with a runtime of 40 minutes; however, this led to a significant performance drop, with accuracy decreasing to 56.14%. The Atlas-specific D2D10 configuration, which progressively processes lowerresolution scales, emerged as the most effective strategy, achieving the highest accuracy of 70.09% at a runtime of 38 minutes. Our novel composition strategy is both led to faster runtimes than traditional convolutional downsampling while yielding comparable performance to no downsampling.

<span id="page-7-3"></span>Table 5. Comparison of different composition strategies.

| Composition          | Runtime (m) ↓ | Top-1 (%) ↑ |  |  |
|----------------------|---------------|-------------|--|--|
| Stack                | 75m           | 69.88       |  |  |
| Downsample (Conv)    | 40m           | 56.14       |  |  |
| Atlas (D2D10) (ours) | 38m           | 70.09       |  |  |

## 6. Conclusion

We propose Multiscale Attention (MSA), a novel primitive for long-context image modeling. In a controlled blocklevel experiment, we demonstrated that MSA significantly outperformed alternative cross-token communication strategies, including FasterVIT's Hierarchical Attention block, and MambaVision Mixer. MSA achieves this performance through two key insights: multi-scale representations and bidirectional cross-scale communication. Building on rich multi-scale representations introduced by MSA, we propose

*Atlas*, a novel neural network architecture for long context modeling. In system-level experiments, we find that *Atlas* significantly improves accuracy-runtime trade-offs in efficient long-context modeling, achieving massive gains over FasterViT, MambaVision, ConvNext, Swin and LongViT. Overall, these results demonstrate that multi-scale attention significantly improves long-context image modeling.

## 7. Acknowledgements

We thank the UCSF Facility of Advanced Computing team, including Hunter McCallum, Sandeep Giri, Rhett Hillary, Marissa Jules, Sean Locke, and John Gallias, for their work in supporting our computational environment. This was supported by a grant from EvansMDS, a funding initiative of the Edward P. Evans Foundation. Research reported in this publication was also supported by the National Cancer Institute of the National Institutes of Health under Award Number R37CA289821. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

## References

- <span id="page-8-7"></span>Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.
- <span id="page-8-11"></span>Chen, C.-F. R., Fan, Q., and Panda, R. Crossvit: Crossattention multi-scale vision transformer for image classification. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 357–366, 2021a.
- <span id="page-8-13"></span>Chen, Z., Xie, L., Niu, J., Liu, X., Wei, L., and Tian, Q. Visformer: The vision-friendly transformer. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 589–598, 2021b.
- <span id="page-8-1"></span>Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. *arXiv preprint arXiv:2412.05271*, 2024.
- <span id="page-8-14"></span>Chu, X., Tian, Z., Wang, Y., Zhang, B., Ren, H., Wei, X., Xia, H., and Shen, C. Twins: Revisiting the design of spatial attention in vision transformers. *Advances in neural information processing systems*, 34:9355–9366, 2021.
- <span id="page-8-4"></span>Ding, J., Ma, S., Dong, L., Zhang, X., Huang, S., Wang, W., Zheng, N., and Wei, F. Longnet: Scaling transformers to 1,000,000,000 tokens. *arXiv preprint arXiv:2307.02486*, 2023.
- <span id="page-8-10"></span>Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., and Guo, B. Cswin transformer: A general

- vision transformer backbone with cross-shaped windows. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 12124–12134, 2022.
- <span id="page-8-5"></span>Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.
- <span id="page-8-0"></span>Gemini-Team, Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.
- <span id="page-8-17"></span>Goyal, P. Accurate, large minibatch sg d: training imagenet in 1 hour. *arXiv preprint arXiv:1706.02677*, 2017.
- <span id="page-8-6"></span>Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.
- <span id="page-8-3"></span>Hatamizadeh, A. and Kautz, J. Mambavision: A hybrid mamba-transformer vision backbone. *arXiv preprint arXiv:2407.08083*, 2024.
- <span id="page-8-2"></span>Hatamizadeh, A., Heinrich, G., Yin, H., Tao, A., Alvarez, J. M., Kautz, J., and Molchanov, P. Fastervit: Fast vision transformers with hierarchical attention. *arXiv preprint arXiv:2306.06189*, 2023.
- <span id="page-8-15"></span>Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 4700–4708, 2017.
- <span id="page-8-12"></span>Li, Y., Yuan, G., Wen, Y., Hu, J., Evangelidis, G., Tulyakov, S., Wang, Y., and Ren, J. Efficientformer: Vision transformers at mobilenet speed. *Advances in Neural Information Processing Systems*, 35:12934–12949, 2022.
- <span id="page-8-16"></span>Lin, T.-Y., Dollar, P., Girshick, R., He, K., Hariharan, B., ´ and Belongie, S. Feature pyramid networks for object detection. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 2117–2125, 2017.
- <span id="page-8-8"></span>Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., and Liu, Y. Vmamba: Visual state space model, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2401.10166) [abs/2401.10166](https://arxiv.org/abs/2401.10166).
- <span id="page-8-9"></span>Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 10012–10022, 2021.

- <span id="page-9-9"></span>Pan, J., Bulat, A., Tan, F., Zhu, X., Dudziak, L., Li, H., Tzimiropoulos, G., and Martinez, B. Edgevits: Competing light-weight cnns on mobile devices with vision transformers. In *European Conference on Computer Vision*, pp. 294–311. Springer, 2022.
- <span id="page-9-3"></span>Qwen-Team. Qwen2.5-vl, January 2025. URL [https:](https://qwenlm.github.io/blog/qwen2.5-vl/) [//qwenlm.github.io/blog/qwen2.5-vl/](https://qwenlm.github.io/blog/qwen2.5-vl/).
- <span id="page-9-1"></span>Rad, R. Vision transformer for multispectral satellite imagery: Advancing landcover classification. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pp. 8176–8183, January 2024.
- <span id="page-9-7"></span>Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models. *arXiv preprint arXiv:2307.08621*, 2023.
- <span id="page-9-14"></span>Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XI 16*, pp. 776–794. Springer, 2020.
- <span id="page-9-5"></span>Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., and Jegou, H. Training data-efficient image transform- ´ ers & distillation through attention. In *International conference on machine learning*, pp. 10347–10357. PMLR, 2021.
- <span id="page-9-4"></span>Vaswani, A. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017.
- <span id="page-9-2"></span>Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., Fan, Y., Dang, K., Du, M., Ren, X., Men, R., Liu, D., Zhou, C., Zhou, J., and Lin, J. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024.
- <span id="page-9-12"></span>Wang, W., Xie, E., Li, X., Fan, D.-P., Song, K., Liang, D., Lu, T., Luo, P., and Shao, L. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 568–578, 2021.
- <span id="page-9-6"></span>Wang, W., Ma, S., Xu, H., Usuyama, N., Ding, J., Poon, H., and Wei, F. When an image is worth 1,024 x 1,024 words: A case study in computational pathology. *arXiv preprint arXiv:2312.03558*, 2023.
- <span id="page-9-13"></span>Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollar, P., ´ and Girshick, R. Early convolutions help transformers see better. *Advances in neural information processing systems*, 34:30392–30400, 2021.

- <span id="page-9-0"></span>Xu, H., Xu, Q., Cong, F., Kang, J., Han, C., Liu, Z., Madabhushi, A., and Lu, C. Vision transformers for computational histopathology. *IEEE Reviews in Biomedical Engineering*, 17:63–79, 2024.
- <span id="page-9-11"></span>Yang, J., Li, C., Zhang, P., Dai, X., Xiao, B., Yuan, L., and Gao, J. Focal self-attention for local-global interactions in vision transformers. *arXiv preprint arXiv:2107.00641*, 2021.
- <span id="page-9-10"></span>Zhu, C., Ping, W., Xiao, C., Shoeybi, M., Goldstein, T., Anandkumar, A., and Catanzaro, B. Long-short transformer: Efficient transformers for language and vision. *Advances in neural information processing systems*, 34: 17723–17736, 2021.
- <span id="page-9-8"></span>Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., and Wang, X. Vision mamba: Efficient visual representation learning with bidirectional state space model. *arXiv preprint arXiv:2401.09417*, 2024.

#### A. Implementation Details

For the baselines that we compared with in Table 1, we utilize the provided code as is, without modifications to gradient accumulation, employing a linearly decaying learning rate proportional to the batch size, following (Goyal, 2017). This ensures consistency with prior work and facilitates a fair comparison of performance. For the various model hyperparamters, we use the configs as provided by authors for Imagenet-1K where available.

#### **B.** Additional Experiments

To validate our findings, we conducted 50-epoch training runs following prior work showing that shorter training schedules still provide reliable signals about architectural performance (Xiao et al., 2021). These abbreviated runs maintain the same relative performance trends across architectures while requiring significantly less computational resources. As shown in Table 1, Atlas-B/16 maintains its superior accuracy-runtime trade-off across resolutions, achieving high accuracy while maintaining reasonable training times even at 2048px resolution, where several competing architectures exceed our 24-hour runtime limit.

# C. Additional Optimizations: QKV Caching for Multi-Scale Attention

A naive implementation of Multi-Scale Attention (MSA) would require recomputing Query, Key, and Value (QKV) projections for each window involved in cross-attention operations across different scales. Consider a window  $W^{(l)}$  at scale l performing cross-attention with windows at coarser scales  $\{W^{(l+1)},\ldots,W^{(L)}\}$  in the top-down pathway. In a naive implementation, the QKV for each window  $W^{(l)}$  would be recalculated for every cross-attention instance, even if the underlying feature representation of  $W^{(l)}$  remains unchanged. This repeated computation becomes increasingly inefficient as the number of scales and windows grows.

To overcome this challenge, we introduce a QKV cache mechanism within MSA. During both the top-down and bottom-up pathways, we maintain a cache at each scale l to store the QKV projections for all windows  $\{W_{ij}^{(l)}\}$ . When a window at scale l needs to perform cross-attention, it first queries this cache. If a valid QKV set for the current version of  $W^{(l)}$  is available, it is directly retrieved from the cache. The cache is updated only when the feature representation of a window at a given scale is modified. This occurs after self-attention at the coarsest scale L, and after each dense cross-attention operation in the top-down and parent cross-attention in the bottom-up pathways. By reusing QKV projections, our cache significantly accelerates MSA

in long sequences where cross-scale attention operations are frequent.

<span id="page-11-0"></span>Table 6. Comparison of vision models across different image resolutions. Each model has two rows: one for runtime (in minutes) and one for Top-1 accuracy (in %). We trained all models for 50 epochs for each resolution. We limited each experiment to a maximum runtime of 24hrs on an 8 × H100 GPU node and report "–" for experiments that could not be complete within our runtime limit.

| Model                            |                  | Runtime (min) ↓ |       |        | Top-1 Accuracy (%) ↑ |       |       |        |        |
|----------------------------------|------------------|-----------------|-------|--------|----------------------|-------|-------|--------|--------|
|                                  |                  | 256px           | 512px | 1024px | 2048px               | 256px | 512px | 1024px | 2048px |
| Transformer-Based                | ViT-B/16         | 18              | 51    | 247    | 3480                 | 63.68 | 72.60 | 69.42  | –      |
|                                  | WViT-B/16        | 18              | 44    | 137    | 638                  | 64.21 | 68.95 | 63.61  | 53.93  |
| Convolutional                    | ConvNext-B/16    | 66              | 237   | 955    | 3825                 | 78.84 | 75.94 | 67.50  | –      |
| Sparse-Transformer               | FasterViT-4      | 49              | 168   | 675    | 2400                 | 77.64 | 74.40 | 53.62  | –      |
|                                  | LongViT-B/16     | 39              | 116   | 442    | 2000                 | 55.20 | 51.88 | 45.32  | –      |
| Mamba-Based                      | MambaVision-B/16 | 21              | 56    | 197    | 750                  | 73.10 | 69.94 | 51.68  | 24.64  |
| Multi-Scale Attention Atlas-B/16 |                  | 25              | 54    | 198    | 786                  | 80.05 | 83.75 | 82.73  | 74.74  |