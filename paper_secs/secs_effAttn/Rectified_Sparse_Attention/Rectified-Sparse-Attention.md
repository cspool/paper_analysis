# **Rectified Sparse Attention**

Yutao Sun\* 12 Tianzhu Ye\* 12 Li Dong\* 1 Yuqing Xia\* 1

Jian Chen 1 Yizhao Gao 13 Shijie Cao 1 Jianyong Wang 2 Furu Wei 1 of Microsoft Research 2 Tsinghua University

3 The University of Hong Kong
https://aka.ms/GeneralAI

#### **Abstract**

Efficient long-sequence generation is a critical challenge for Large Language Models. While recent sparse decoding methods improve efficiency, they suffer from KV cache misalignment, where approximation errors accumulate and degrade generation quality. In this work, we propose Rectified Sparse Attention (ReSA), a simple yet effective method that combines block-sparse attention with periodic dense rectification. By refreshing the KV cache at fixed intervals using a dense forward pass, ReSA bounds error accumulation and preserves alignment with the pretraining distribution. Experiments across math reasoning, language modeling, and retrieval tasks demonstrate that ReSA achieves near-lossless generation quality with significantly improved efficiency. Notably, ReSA delivers up to 2.42× end-to-end speedup under decoding at 256K sequence length, making it a practical solution for scalable long-context inference. Code is available at https://aka.ms/ReSA-LM.

#### 1 Introduction

The ability to process long contexts has become a core requirement for Large Language Models, with context lengths up to millions of tokens [20, 26]. In particular, long sequence generation has received growing attention, especially due to the demand for test-time scaling [9, 13].

Despite this progress, efficient long-sequence generation remains a significant challenge. In standard autoregressive decoding, each token must attend to the full KV cache, leading to frequent memory access and increased IO pressure. This bottleneck severely limits throughput, especially in long-context scenarios where memory access dominates latency.

Recent works [18, 23] used sparse decoding to alleviate this issue. These methods selectively attend to a subset of the context, achieving accuracy comparable to dense attention on long inputs while reducing computational cost. However, as shown in Figure 1, they often suffer from worse performance with increasing length. Since **computation errors accumulate in the KV cache during sparse decoding**, the attention computation will suffer from the misalignment between training and inference, contributing to performance degradation.

In this work, we propose Rectified Sparse Attention (ReSA), a simple yet effective approach that achieves near-lossless long-sequence generation quality while maintaining high inference efficiency. ReSA leverages block-sparse attention [23] for fast

<span id="page-0-0"></span>Figure 1: Sparse decoding performance becomes worse with increasing decoding length due to error accumulation of KV cache.

<sup>0.79
0.79
0.79
0.79
0.77
0.76
0 1000 2000 3000 4000</sup>Decoding Length

<sup>\*</sup> Equal contribution. \$\phi\$ Corresponding author.

![](_page_1_Figure_0.jpeg)

Figure 2: Overview of ReSA. After completing the prefill stage, the model enters sparse decoding. Once the number of generated tokens reaches the rectification frequency, a rectification step is performed to construct a lossless compact KV cache, after which sparse decoding resumes.

retrieval and further improves memory efficiency by applying

shared grouping [28], allowing query heads to reuse attention patterns. To address the error accumulation issue, we introduce dense rectification, where the sparse KV cache is periodically refreshed with a parallel dense forward pass. This ensures that approximation errors are bounded within a constant range, preventing long-term degradation.

We conduct comprehensive experiments to demonstrate the effectiveness of ReSA. On math reasoning benchmarks, ReSA achieves strong test-time scaling and matches dense attention in long-sequence settings. In language modeling, ReSA significantly closes the quality gap between sparse and dense decoding. On the efficiency side, our approach yields up to  $2.42 \times$  end-to-end speedup under INT4 decoding at 256K context length, showing strong practical utility for real-world deployment.

#### 2 Rectified Sparse Attention

ReSA primarily involves two alternating phases, sparse decoding and periodic rectification. During the decoding phase, we employ the group block sparse attention mechanism, which significantly reduces computational and memory overhead, enabling fast autoregressive inference. During the rectification stage, the decoding tokens are forwarded in parallel to correct approximation errors in KV cache introduced by sparse decoding. By alternating between sparse generation and dense rectification, ReSA enables scalable long-context inference while ensuring the generation quality.

#### <span id="page-1-0"></span>2.1 Group Block Sparse Attention

Self-attention mechanisms are the core component of Transformer architectures, enabling each token to attend to all previous tokens. Formally, in Group-Query Attention (GQA) [2], given a sequence of n tokens, we compute the query  $Q \in \mathbb{R}^{h \times g \times n \times d}$ , key  $K \in \mathbb{R}^{h \times n \times d}$ , and value  $V \in \mathbb{R}^{h \times n \times d}$  matrices through learned projections. The attention output is computed as:

Attention
$$(Q, K, V)_{ij} = \operatorname{softmax}\left(\frac{Q_{ij}K_i^{\top}}{\sqrt{d}}\right)V_i$$
 (1)

where  $\operatorname{softmax}(\cdot)$  is applied along each query row. The pairwise computation requires  $\mathcal{O}(n^2d)$  operations, making standard attention prohibitively expensive for long-context inference.

We adopt a block-sparse attention design that selectively attends to a small number of relevant memory blocks rather than the entire context. Given the block size b and block sparse mask  $M \in \{0,1\}^{h \times n \times n/b}$ , the block-sparse attention is computed as:

$$GBSA(Q, K, V, M)_{ij} = \operatorname{softmax}\left(\frac{Q_{ij}K_i^{\top} \cdot \overline{M}_i}{\sqrt{d}}\right) V_i, \ \overline{M}_{ijk} = M_{ij \lfloor k/b \rfloor}$$
 (2)

![](_page_2_Picture_0.jpeg)

Figure 3: Overview of Group Block Sparse Attention. For each group of query heads, we perform average pooling and enforce the selection of the same KV blocks across all heads within the group.

GBSA adopts a query-dependent sparsity pattern, where each query attends to a limited set of key blocks determined by M. Since each selected key block corresponds to a contiguous memory region in the KV cache, this design ensures both high performance and memory efficiency during inference. Note that we further accelerate decoding by maintaining a shared sparse pattern within each GQA group [28].

**Block Representation** Following the Quest algorithm [23], we represent the key-value memory using blocks to enable efficient retrieval. Specifically, given a key matrix  $k \in \mathbb{R}^{n \times d}$ , we partition it into non-overlapping blocks of size b, where each block contains b consecutive tokens. For the i-th block, we compute two block descriptors:

$$k_{\text{block\_min},i} = \min(k_{ib:(i+1)b}), \ k_{\text{block\_max},i} = \max(k_{ib:(i+1)b})$$
 (3)

where  $\min(\cdot)$  and  $\max(\cdot)$  are applied element-wise across the block dimension.

Each block is thus summarized by a pair of vectors  $(k_{\mathrm{block} \, \mathrm{min},i}, k_{\mathrm{block} \, \mathrm{max},i})$ , which compactly describe the distribution range of keys within the block. This representation allows efficient approximate matching without exhaustively scanning all individual tokens. During decoding, newly generated keys can be incrementally incorporated by updating the block key statistics, enabling an online update mechanism without recomputing from scratch.

Notably, the block representation is entirely training-free, relying solely on statistical descriptors. Our method remains compatible with more advanced block representation strategies, such as SeerAttention [8], where block keys are fine-tuned jointly with the model to achieve higher retrieval precision if needed.

**Block Selection** During decoding, given a pooling query  $q \in \mathbb{R}^d$  for each GQA group and a set of block descriptors  $\{(k_{\text{block\_min},i}, k_{\text{block\_max},i})\}_{i=1}^M$ , we compute similarity scores following the Quest algorithm [23]. Specifically, the score between the pooling query and block i is calculated as:

$$score_{i} = \sum_{j=1}^{d} \max(q_{j} \times (k_{block\_max,i})_{j}, q_{j} \times (k_{block\_min,i})_{j})$$

$$(4)$$

where  $q_j$  denotes the j-th dimension of the pooling query, and  $(k_{\text{block min},i})_j$ ,  $(k_{\text{block max},i})_j$  are the j-th dimensions of the minimum and maximum vectors of block i, respectively.

To select the attended blocks, we adopt a dynamic top-n strategy. First, a fixed number of recent blocks, denoted as  $n_{\rm local}$ , are always preserved by setting their scores to  $+\infty$ , ensuring that the latest context is available for local coherence. Second, we enforce a minimal block number  $n_{\rm min}$  to avoid significant performance degradation on short sequences. Finally, the value of n is dynamically determined based on a active ratio p, following:

$$n = \max\left(n_{\min}, \lceil M \times p \rceil\right),\tag{5}$$

where M is the total number of available memory blocks. Attention computation is restricted to the selected blocks, significantly reducing memory accesses while maintaining retrieval quality.

### 2.2 Dense Rectification

Transformer inference implicitly consists of two distinct phases: **context encoding**, realized through the construction of the KV cache, and **next-token prediction**, realized through the forward pass of

#### Algorithm 1 Rectified Sparse Decoding

```
Require: Initial prompts P, model M, rectification frequency f, maximum generation steps T
Ensure: Generated tokens G
  Initialize KV cache K by Prefill(P, K)
  Initialize block key cache B
  Initialize output sequence G ← empty
  for i = 1 to T do
    t ← SparseForward(G[i − 1], K, B)
    Append t to G
    Update KV cache K with t
    Update block key cache B with t
    if i mod f = 0 then
       K, B ← DenseForward(G[i − f : i], K, B)
       Update block key cache B
    end if
  end for
```

the current token. While sparse attention effectively approximates the next-token prediction phase, it inevitably introduces errors. Crucially, these prediction errors accumulate in the KV cache during decoding, leading to compounding inaccuracies over long sequences. To mitigate this issue, we propose Dense Rectification, a lightweight mechanism that periodically refreshes the KV cache to maintain its quality. This design constrains error accumulation within a constant window size and enables efficient sparse decoding without compromising generation consistency.

Rectification Algorithm Given a rectification frequency f, we perform standard sparse decoding for up to f tokens, appending newly generated tokens into the KV cache. After every f token, we batch these recent tokens and re-encode them using dense attention to reconstruct an updated KV cache. This two-phase approach — serial sparse decoding followed by parallel rectification — ensures that errors introduced by approximate attention are corrected at regular intervals, keeping the memory quality close to that of dense decoding. Importantly, the rectification step amortizes efficiently over large batches, maintaining high throughput even when dense recomputation is involved. To maintain consistency, we also refresh the associated block keys during rectification. otherwise, the misalignment between the block keys and the updated KV cache would degrade subsequent sparse retrieval accuracy.

Compatibility with LLM Serving Systems Dense Rectification is naturally compatible with modern LLM serving optimizations such as continuous batching [\[27\]](#page-10-5) and chunked prefill [\[1,](#page-9-4) [12\]](#page-9-5). Since rectification only requires periodic batched re-encoding, it seamlessly fits into systems that dynamically group decoding and prefill workloads to maximize GPU utilization. By maintaining a fixed rectification frequency per request, our method can operate within the batching and scheduling pipelines without introducing special synchronization barriers or inefficiencies.

