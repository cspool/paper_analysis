# 2 RELATED WORK

#### 2.1 KV CACHE COMPRESSION

Most of the existing KV cache compression work is carried out within a single transformer layer, namely the intra-layer compression. For example, StreamingLLM [\(Xiao et al., 2023\)](#page-12-8) only retains the attention sink in the KV cache, avoiding a significant increase in memory demand when generating long texts. H2O [\(Zhang et al., 2024b\)](#page-12-3) reduces memory usage by removing the keys and values stored by unimportant tokens from the full KV cache. Compared to H2O, Scissorhands [\(Liu et al.,](#page-11-4) [2024b\)](#page-11-4) discards as many tokens as possible from the KV cache in each round, rather than just one token. PyramidInfer [\(Yang et al., 2024b\)](#page-12-2) considers calculating the key-values only for important tokens during generation. FastGen [\(Ge et al., 2023\)](#page-10-7) also discards the attention values of certain nonspecial tokens in the KV cache but sets a maximum approximation error for the attention matrix to ensure model performance. SnapKV [\(Li et al., 2024\)](#page-11-5) builds on the observation that attention heads tend to consistently focus on certain prompt features, especially those toward the end, to compress KV caches by selecting key positions for each head. While these methods have shown effective compression ability, they achieve KV cache sparsification by discarding tokens within a single layer. However, they do not address layer-wise KV cache compression.

![](_page_2_Figure_1.jpeg)

<span id="page-2-0"></span>Figure 2: An illustration of the strategy searching process of the *KVSharer*. For a given LLM, process (a) performs inference on the calibration dataset and computes the euclidean distance between flattened KV cache vectors from any two layers, sorting pairs in descending order. (b) KV cache pairs are sequentially replaced, ensuring the final hidden-state similarity with the original model exceeds threshold  $\mathcal{T}$  until the KV cache compression ratio reaches  $\mathcal{R}$ .

Recently, only a few works have focused on layer-wise compression strategies for the KV cache. MiniCache (Liu et al., 2024a) merges the KV caches from different layers to enhance throughput. LCKV (Wu & Tu, 2024) proposes a novel method that computes and caches the KVs for only a small number of layers, thereby significantly reducing memory consumption and improving inference throughput. CLA (Brandon et al., 2024) design an inter-layer attention mechanism to share the KV cache across different layers. YOCO (Sun et al., 2024) designs a decoder-decoder architecture that enforces the reuse of the lower layer's KV cache in the higher layers' KV cache. However, all of them require further training of the model rather than being plug-and-play on well-trained LLMs. In contrast, we are the first to propose a layer-wise KV cache compression method for well-trained LLMs without further training. Moreover, our method is directly compatible with the current intralayer KV cache compression techniques.

#### 2.2 ATTENTION MAP & PARAMETER SHARING

Since the introduction of Transformer-based pre-trained language models (PLMs) like BERT (Devlin et al., 2018), some research has focused on attention map sharing and parameter sharing. Lazyformer (Ying et al., 2021) reuses attention maps from lower layers in higher layers of the Transformer, thereby enhancing the throughput of PLMs. Xiao et al. (2019) directly share the attention weights across layers, improving inference speed in machine translation tasks. Takase & Kiyono (2021) design three parameter sharing strategies based on rules within the Transformer architecture, improving model efficiency in machine translation tasks. Shim et al. (2023) conduct a comprehensive evaluation of various attention map sharing strategies. Since the advent of the era of LLMs, various works utilizing parameter sharing or attention map sharing have been widely adopted. Multi-Query attention (MQA) (Shazeer, 2019) and Grouped-Query attention (GQA) (Ainslie et al., 2023) have become standard strategies in modern LLMs, improving model efficiency by sharing attention queries and keys within a layer. Cao et al. (2024) investigate the similarity of attention maps and attention parameters in LLMs and propose various attention map sharing strategies to reduce inference memory consumption. However, none of these works have extended to the KV cache. They all rely on replacing layers with higher parameter similarity or activation values, which aligns with intuition, whereas we replace dissimilar KV cache.

#### 3 KVSHARER

The main steps of *KVSharer* are divided into two parts. First, for a given LLM, it searches a sharing strategy, a list that specifies which layers' KV caches should be replaced by those of other specific layers. Then, during the subsequent prefill and generation processes on all the tasks, the KV caches of the relevant layers are directly replaced according to this list, enabling efficient inference.

### 3.1 STRATEGY SEARCHING

To heuristically search for a sharing strategy, our approach is to first perform inference on a calibration dataset and calculate the euclidean distance between the KV caches of any two layers. Then, we sort these KV cache pairs in descending order of euclidean distance. Subsequently, we attempt to replace the corresponding KV caches in sequence, while ensuring that the model's output remains as consistent as possible with the original model during the replacement process. The search process can be referenced in Algorithm [1](#page-3-0) and Figure [2.](#page-2-0)

```
Algorithm 1 Workflow of Strategy Searching
```

```
Require: LLM M, Target Shared KV Cache Layers C, Calibration Dataset D, Threshold for rep-
   resentation similarity T
Ensure: Sharing Strategy Z
 1: S ← Euclidean KV Dis(M, D) ▷ Perform inference on the calibration dataset D, compute
   the euclidean distance between the KV caches of any two layers, and record the corresponding
   layer pairs and their distance values as S
 2: R ← Descend Rank(S) ▷ Sort the KV cache layer pairs in descending order based on their
   euclidean distance
 3: Z ← ∅ ▷ Initialize candidate sharing strategy as Z
 4: P ← 0 ▷ Initialize current number of shared layers as P
 5: for each r in R do
 6: Z ← Z ∪ r ▷ Add the current pair r to the candidate set
 7: Mtmp ← Sharing KV(M, Z) ▷ Apply layer-wise KV cache sharing to M according to
   the current candidate strategy and get candidate model Mtmp
 8: s ← Avg Cos Sim(Mtmp, M, D) ▷ Compute the similarity of the final layer hidden-state
   between the two models on the calibration dataset as s
 9: if s <= T then
10: Z ← Z \ r ▷ If the output similarity between the current model and the original model
   falls below the threshold, the current pair r is discarded
11: else
12: P ← P + 1 ▷ Find a replacement and increase the shared layers P by 1
13: if P == C then
14: return Z ▷ Return the currently found optimal strategy when the number of
   compressed layers reaches the preset value C
15: end if
16: end if
17: end for
18: return None
```

#### 3.1.1 PREPARATION

For a given LLM M, we set the target number of shared KV cache layers C. We specify a calibration dataset D, which typically consists of several plain sentences. We conduct forward computations on D using both the model with shared KV cache and the original model to obtain output representations, ensuring that the cosine similarity of these representations exceeds the threshold T .

#### 3.1.2 SEARCHING

KV Cache Similarity Calculation & Initialization (1-4) First, we perform a forward pass using the original model M on the calibration dataset D, saving the KV cache for each layer during the forward pass of each sentence. Then, we average the KV cache for each layer across all samples to obtain the average KV cache for each layer. Finally, we flatten the keys and values of the KV cache for each layer into a one-dimensional vector, and then average the keys and values separately to represent the KV cache for that layer. We then calculate the euclidean distance between the KV cache representations of any two layers to obtain S. We then sort S in descending order to get R, as a larger euclidean distance indicates lower similarity. Consequently, dissimilar layer pairs are prioritized. We then set two variables, Z and P, to record the candidate KV cache sharing strategy and the current number of shared layers.

Sharing Strategy Searching (5-18) Based on the values in R, we sequentially select a pair of layers r to add to Z for sharing. When sharing, we replace the layer closer to the output with the one closer to the input, as the layers near the input end in LLMs are more sensitive, and modifying them could result in significant performance degradation [\(Cao et al., 2024;](#page-10-6) [Yang et al., 2024c\)](#page-12-11).

We then apply the candidate strategy Z by directly replacing the KV cache of one layer with another during the forward pass. Using the model with KV cache sharing and the original model, we perform inference on the calibration dataset to obtain the output representation from the last layer. We then average these representations across different sentences. If the cosine similarity between the averaged output representations of the two models exceeds the threshold T , we retain the current pair replacement r; otherwise, we discard it. This iteration continues until the predefined number of compressed layers C is reached. At the end of the iteration, we obtain an optimal KV cache sharing strategy Z through the heuristic search.

#### 3.2 INFERENCE WITH KV CACHE SHARING

After obtaining the KV cache sharing strategy Z, we apply it to all subsequent inference tasks, including both prefill and generation processes. As illustrated in Figure [3,](#page-4-0) during forward computations, when a layer's KV cache needs to be replaced based on Z, we directly copy the KV cache from the previously computed layer. The subsequent computations then follow the original model's process.

