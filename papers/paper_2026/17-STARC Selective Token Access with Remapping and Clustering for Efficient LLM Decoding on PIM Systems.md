**==> picture [93 x 45] intentionally omitted <==**

## **STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems** 

Zehao Fan 

Yunzhen Liu 

## Garrett Gagnon 

Rensselaer Polytechnic Institute Troy, NY, USA fanz2@rpi.edu 

University of Massachusetts, Amherst Amherst, MA, USA yunzhenliu@umass.edu 

Rensselaer Polytechnic Institute Troy, NY, USA gagnog@rpi.edu 

## Zhenyu Liu 

Hadjer Benmeziane IBM Research – Ruschlikon Ruschlikon, Switzerland hadjer.benmeziane@ibm.com 

Yayue Hou 

Rensselaer Polytechnic Institute Troy, NY, USA liuz32@rpi.edu 

Rensselaer Polytechnic Institute Troy, NY, USA houy4@rpi.edu 

Kaoutar El Maghraoui 

Liu Liu 

IBM T. J. Watson Research Center Yorktown Heights, NY, USA kelmaghr@us.ibm.com 

Rensselaer Polytechnic Institute Troy, NY, USA liu.liu@rpi.edu 

## **Abstract** 

## **ACM Reference Format:** 

Zehao Fan, Yunzhen Liu, Garrett Gagnon, Zhenyu Liu, Yayue Hou, Hadjer Benmeziane, Kaoutar El Maghraoui, and Liu Liu. 2026. STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems. In _Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’26), March 22–26, 2026, Pittsburgh, PA, USA._ ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3779212.3790226 

Serving large language models (LLMs) places significant pressure on memory systems due to frequent accesses and growing key–value (KV) caches as context lengths increase. Processing-in-memory (PIM) architectures offer high internal bandwidth and near-data compute parallelism, but current designs target dense attention and perform poorly under the irregular access patterns of dynamic KV cache sparsity. To mitigate this limitation, we propose STARC, a sparsityoptimized data mapping scheme for efficient LLM decoding on PIM. STARC clusters semantically similar KV pairs and co-locates them contiguously within PIM banks, enabling retrieval at cluster granularity by matching queries against precomputed centroids. This bridges the gap between finegrained sparse attention and row-level PIM operations, improving utilization while minimizing overhead. On a simulated HBM-PIM system, under constrained KV budgets, STARC achieves up to 78% and 65% reductions in attentionlayer latency and energy over token-wise sparsity methods, and up to 93% and 92% reductions relative to full attention, while preserving model accuracy. 

## **1 Introduction** 

Large language models (LLMs) have demonstrated exceptional capabilities across a wide range of natural language processing tasks and are increasingly deployed in real-world applications such as interactive chat systems [1, 57, 61], code generation tools [38, 45, 48], and decision support [29, 47, 55]. During decoding, however, LLMs operate auto-regressively, requiring repeated attention over a growing key-value (KV) cache [41]. As context lengths scale, the KV cache expands proportionally, leading to frequent and large memory accesses. Despite high computational throughput, modern GPUs are constrained by limited memory bandwidth, making attention layers predominantly memory-bound [25]. Processingin-memory (PIM) architectures [8, 13, 19, 20, 39] offer a promising solution by alleviating bandwidth bottlenecks and enabling efficient in-memory computation. Recent work has explored heterogeneous designs (e.g., GPU-PIM, NPUPIM) that offload memory-bound attention layers to PIM while leveraging traditional accelerators (xPUs) for computeintensive feed-forward networks (FFNs) and Query-KeyValue (QKV) generation [15, 40]. 

## _**CCS Concepts:**_ • **Computer systems organization** → _Architectures_ ; • **Computing methodologies** → **Machine learning** . 

_**Keywords:**_ Processing-in-memory (PIM); Large language model (LLM); Sparse attention; KV clustering; KV cache 

This work is licensed under a Creative Commons Attribution 4.0 International License. _ASPLOS ’26, Pittsburgh, PA, USA_ 

However, the trend toward longer contexts continues to impose substantial computation and memory costs, driven by the quadratic complexity of attention. Recent methods alleviate this by introducing **KV cache sparsity** through 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 https://doi.org/10.1145/3779212.3790226 

1863 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

**==> picture [500 x 186] intentionally omitted <==**

**----- Start of picture text -----**<br>
Model<br>A sequence of 16 tokens Sparsity 50%  Attention score<br>0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0.3 0.4 0.6 0 0 0.8 0 0.7 0 0 0.7 0 0.4 0.3 0<br>Hardware<br>Store for PIM Compute attention score<br>Row 0 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 Execution 1 Limit: Inability to<br>skip execution<br>Challenge Row 1 8 9 10 11 12 13 14 15 8 9 10 11 12 13 14 15 Execution 2<br>Solution Algorithm  Clusters Store for PIM Compute attention score<br>(Ours) Cluster 1 0 4 5 7 9 10 12 15 Row 0 0 4 5 7 9 10 12 15 0 4 5 7 9 10 12 15 Row skipping via<br>clusters<br>Cluster 2 1 2 3 6 8 11 13 14 Row 1 1 2 3 6 8 11 13 14 1 2 3 6 8 11 13 14 Execution 1<br>**----- End of picture text -----**<br>


**Figure 1.** Enhanced PIM execution efficiency through STARC. Due to the coarse row-level access granularity of PIM, directly applying sparsity to KV caches often fails to skip computation. STARC addresses this by clustering keys and values such that selected tokens are physically co-located, enabling effective computation skipping and realizing the speedup benefits of sparsity on PIM. 

selective retrieval or compression, retrieving only a subset of tokens to approximate full attention. While such methods can reduce retrieval by over 90% with minimal accuracy loss, they introduce irregular and dynamic access patterns that traditional PIM designs—optimized for dense, row-level accesses—struggle to support. Most existing PIM-enabled systems largely assume full KV cache attention, leading to underutilization when sparsity is applied. Techniques such as Quest [49] address this by retrieving at page granularity, aligning with memory row organization and improving bandwidth efficiency. Yet, page-based layouts remain coarsegrained, often fetching semantically irrelevant tokens, which wastes compute and undermines accuracy. This mismatch between **dynamic sparsity** and **rigid PIM data layouts** remains a fundamental barrier to efficient LLM decoding. To address this challenge, we propose STARC, a sparsityoptimized data mapping scheme designed specifically for PIM architectures. The key idea, illustrated in Figure 1, is to cluster semantically similar tokens and physically co-locate their KV pairs in memory. Our design aligns sparse attention with PIM’s row-level organization. 

To overcome the mismatch between dynamic sparsity and rigid PIM layouts, we propose STARC, a sparsity-optimized data mapping scheme for LLM decoding on PIM. STARC clusters semantically similar tokens and co-locates their KV entries contiguously in memory, enabling sparse attention to align with row-level PIM operations. Queries retrieve clusters by matching against precomputed centroids, ensuring most fetched vectors are relevant and improving hardware utilization. By performing lightweight clustering directly within PIM and fixing clusters across decoding steps, STARC 

achieves efficient support for sparse attention in LLM serving. This paper makes the following contributions: 

- We analyze the challenges of applying KV cache sparsity to PIM-enabled LLM inference and identify the mismatch between dynamic sparse retrieval and rigid row-level PIM data layouts. 

- We propose STARC, a novel clustering-based data mapping scheme that co-locates semantically similar KV entries to align sparse attention with PIM bank organization. 

- We introduce efficient in-memory designs that directly leverage existing PIM primitives and hardware to implement cosine-based K-means clustering for KV clustering, avoiding additional area overhead while minimizing GPU involvement and exploiting near-data compute. 

- We demonstrate that STARC significantly improves throughput, utilization, and energy efficiency over state-of-the-art PIM system baselines while preserving model accuracy under sparse attention. It reduces attention-layer latency and energy by up to 78% and 65% compared to token-wise sparsity, and under a KVcache budget of 1024, achieves up to 93% latency and 92% energy reduction relative to full KV retrieval. 

## **2 Background** 

In this section, we first introduce PIM architectures as a solution to the memory bandwidth bottlenecks that arise during decoding. We then describe sparse attention techniques, which alleviate the growing computational and memory access costs associated with long-context sequences, providing basic understandings of our proposed design. 

1864 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

## **2.1 PIM for LLM Attention** 

Transformer-based LLMs [1, 6, 21, 31, 50] perform inference in two stages: **prefill** and **decoding** . In prefill, the entire input sequence (i.e., the user prompt) is processed in parallel to produce the first output token. In decoding, tokens are generated autoregressively, with each new token appended to the sequence and used as input for the next step. This iterative process requires the model to repeatedly read from the KV cache, which stores the key and value vectors projected from all previously generated tokens. 

Although modern GPUs offer high FLOPs, attention computation during decoding is typically memory-bound, given its low arithmetic intensity and frequent memory accesses to tokens stored in the KV cache. As context lengths grow to hundreds of thousands of tokens, data transfers between GPU and external memory become a bottleneck. This results in suboptimal resource utilization, as much of computational capacity remains idle while waiting for memory transactions to complete. 

**==> picture [227 x 81] intentionally omitted <==**

**Figure 2.** Typical xPU–PIM hybrid system: QKV Generation and Feed-Forward Networks are executed on xPU such as GPU and NPU, while Attention is executed on PIM. 

A more detailed examination of the decoder block architecture provides insight into the source of this memory bottleneck. Each decoder block consists of three fundamental components: **(1) Query-Key-Value (QKV) generation** , which projects the input hidden states into separate query, key, and value vectors; **(2) Multi-Head Attention (MHA)** , where attention weights are computed and applied across multiple heads in parallel; and **(3) Feed-Forward Networks (FFNs)** , which apply independent linear transformations and nonlinear activations to each token embedding. Among these components, the MHA module, particularly during the decoding phase, incurs the highest memory bandwidth demand due to its frequent token access to KV cache. 

PIM architectures have emerged as a promising solution to mitigate such memory bandwidth bottlenecks by integrating computation directly within memory systems. Figure 2 illustrates the typical execution partitioning adopted by recent PIM-enabled heterogeneous systems, where memory-bound MHA is offloaded to PIM units, while QKV generation and FFNs remain on xPU compute cores. Attention layers are especially well-suited for PIM acceleration, primarily for two reasons. First, once the KV matrices for a decoding step are 

written to the memory arrays, they can be reused repeatedly for subsequent query vectors in the same or following decoding iterations. This reuse pattern allows PIM systems to take full advantage of the high internal bandwidth. Second, MHA operations rely heavily on general matrix-vector multiplication (GEMV) to compute attention scores and outputs. Distributing these computations across parallel memory banks allows the PIM architecture to exploit abundant internal bandwidth while offloading repeated GEMV operations. 

## **2.2 Selective Token Access with Attention Sparsity** 

Recent studies on attention distributions in LLMs have revealed that attention scores during inference are often highly sparse. In many cases, only a small subset of tokens significantly contributes to the output, while the majority of tokens receive negligible weights. This observation has motivated a range of **sparse attention** techniques that aim to reduce the number of KV pairs accessed during decoding by performing selective retrieval or compression of the KV cache. 

To be more specific, in the Transformer architecture, each attention head operates on projected query, key, and value vectors. Let _𝑞_ ∈ R[1][×] _[𝑑][ℎ]_ denote the query vector corresponding to the most recent token in a single head, and let _𝐾,𝑉_ ∈ R _[𝐿]_[×] _[𝑑][ℎ]_ represent the cached key and value matrices for the _𝐿_ previous tokens. Here, _𝑑ℎ_ denotes the hidden dimension of a single attention head. Sparse attention methods select a subset of _𝐵_ ≪ _𝐿_ KV pairs, typically based on similarity metrics such as dot-product or cosine similarity, yielding the reduced matrices _𝐾𝑆,𝑉𝑆_ ∈ R _[𝐵]_[×] _[𝑑][ℎ]_ . The attention output is then computed as softmax( _𝑞𝐾𝑆_[⊤][/√] _𝑑ℎ_ ) _𝑉𝑆_ , where _𝑞_ ∈ R[1][×] _[𝑑][ℎ]_ is the query vector. This selective computation significantly reduces both the memory access and the per-step computational cost, while maintaining model quality in many scenarios. Crucially, it also decouples the per-token decoding complexity from the total context length. 

**==> picture [215 x 16] intentionally omitted <==**

**----- Start of picture text -----**<br>
Dynamic Sparsity Dynamic Sparsity<br>Static Sparsity<br>(Token-wise) (Page-wise)<br>**----- End of picture text -----**<br>


**==> picture [239 x 69] intentionally omitted <==**

**Figure 3.** Common attention sparsity patterns. 

In practice, sparse attention mechanisms can be broadly categorized into three representative classes, illustrated in Figure 3: Firstly, **static sparsity** restricts each query to attend only to fixed historical token positions or to a fixed-size window of past tokens (e.g., the most recent _𝐵_ tokens), independent of content. This type of sparsity typically evicts other tokens and is hardware-friendly, but fails to capture 

1865 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

long-range dependencies. Secondly, **token-wise dynamic sparsity** selects the top- _𝐵_ most relevant tokens for each query dynamically based on similarity scores. It provides finer control over which tokens are attended but introduces irregular access patterns. Lastly, **page-wise dynamic sparsity** groups the context into fixed-size pages and selects relevant pages rather than individual tokens. Compared with token-wise, this method maintains hardware-friendly access patterns but compromises the effectiveness of per-iteration token access due to the retrieval of irrelevant tokens within a page. In this work, we mainly discuss the latter two sparsity methods. Both are KV cache retrieval methods that preserve the full KV cache without eviction. This focus aligns with our emphasis on model accuracy rather than reducing the KV cache storage footprint, since in our system the KV cache resides in HBM-PIM and does not pressure GPU memory capacity. 

## **3 Motivation** 

This section motivates our proposed design by analyzing the limitations of existing attention mechanisms on PIM architectures. We first highlight the inefficiencies of dense and token-wise sparsity under PIM’s row-level access granularity. We then consider page-wise sparsity, which aligns better with hardware constraints but suffers from low relevance density and reduced attention quality. Finally, we motivate a clustering-based remapping strategy that groups semantically similar tokens into contiguous memory rows, aiming to improve execution efficiency without sacrificing the accuracy of token retrieval. 

## **3.1 Challenges of Attention on Existing PIM Architectures** 

Prior PIM architectures for attention are designed to work with a fully dense KV cache [14, 15, 24, 40, 59], where all past tokens are retained throughout decoding. However, with the long contexts used by modern LLMs, dense attention places heavy demands not only on internal bandwidth but also on the limited computational capacity of PIM architectures. Specifically, the near-memory logic embedded close to the memory arrays is typically lightweight and optimized for simple row-wise operations. These resources lack the deep pipelining and wide parallelism of traditional GPU compute units, and are constrained by area and energy budgets within the memory die. Moreover, each attention query must access a large number of stored key-value vectors, which are laid out across many memory rows. In PIM architectures, processing even a single token requires activating entire memory rows, since the logic operates at row granularity. When dense attention forces many such activations per query, the system suffers from frequent row switching and high energy costs due to repeated bitline toggling and row precharging. 

**==> picture [239 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
Dynamic Sparsity<br>Static Sparsity<br>(Token-wise)<br>act act<br>act act<br>act<br>act<br>act<br>PIM rows<br>act<br>act<br>act<br>act<br>act act<br>Dynamic Sparsity<br>STARC<br>(Page-wise)<br>act<br>act<br>act act<br>act<br>act<br>act<br>**----- End of picture text -----**<br>


**Figure 4.** Comparison of row activation patterns under different sparse attention methods. 

This behavior severely reduces the efficiency of row-parallel execution across memory banks. 

Applying sparse attention can potentially alleviate this overhead by accessing only a subset of past tokens. However, applying these methods in current PIM architectures introduces new challenges. A main issue is that token importance changes dynamically during decoding. A token that is unimportant at one step may become crucial later, limiting the effectiveness of static data placement or scheduling strategies in PIM. 

These limitations become especially severe under tokenwise sparsity, which requires fine-grained retrieval of tokens. As illustrated in Figure 4, such fine-grained access patterns are poorly aligned with the row-level access granularity of PIM architectures. Each PIM array operates at row granularity: the near-memory logic must activate an entire row, bring all entries onto the bit-lines, and perform computation there. When relevant tokens are scattered across multiple rows, the memory controller is forced to read and process every row individually, leading to substantial over-fetching of irrelevant data and redundant computation. 

**==> picture [228 x 115] intentionally omitted <==**

**----- Start of picture text -----**<br>
Page 246 (3936 to 3951) Page 247 (3952 to 3967)<br>Page 248 (3968 to 3983) Page 249 (3984 to 3999) 0.05<br>0.04<br>Page 250 (4000 to 4015) Page 251 (4016 to 4031)<br>0.03<br>Page 252 (4032 to 4047) Page 253 (4048 to 4063) 0.02<br>0.01<br>Page 254 (4064 to 4079) Page 255 (4080 to 4095)<br>**----- End of picture text -----**<br>


**Figure 5.** Page-wise retrieval with less important tokens. 

1866 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

## **3.2 Hardware Efficiency vs. Attention Quality** 

To reduce the overhead of selecting important tokens during decoding, several recent attention sparsity methods, such as Quest [49], adopt a page-wise retrieval strategy. In this approach, tokens are grouped into fixed-size pages, and attention is computed over selected pages rather than individual tokens. Quest estimates the importance of each page by comparing the query vector with the minimal and maximal key vectors of that page, and retrieves only the most relevant ones. This simplifies the sparsity decision process and reduces the complexity of token scoring. 

As shown in Figure 4, page-wise token access also aligns well with the memory organization of a PIM accelerator. When the page size is a multiple of the physical memory row size, PIM can fetch and process entire rows efficiently. This allows the accelerator to fully utilize internal memory bandwidth and avoid partial-row access overhead. In HBMPIM architectures, where computation occurs near DRAM banks, this alignment improves data locality and reduces unnecessary data movement. 

However, this hardware compatibility comes at the cost of attention quality and model accuracy. Page boundaries are defined purely based on token position, not on token relevance. As a result, selected pages often include many irrelevant tokens. These tokens are still accessed and processed, wasting bank-level bandwidth and compute resources. 

We illustrate this issue using attention heatmaps with a context length of 4K in Figure 5, where the model used is LLaMA3.1-8B. The example uses Quest’s page-wise method with a page size of 16 tokens. In the heatmaps, lighter cells represent tokens with higher attention weights. As shown, most pages contain only one or two important tokens. This inefficiency limits the usefulness of page-wise sparsity, despite its compatibility with PIM architectures. 

## **3.3 Motivation for Remapping and Clustering** 

To address the limitations of sparse attention on PIM, we propose a remapping strategy that clusters semantically similar key–value vectors and places each cluster in contiguous memory rows. By aligning row-level layout with attention relevance, activating a single row retrieves multiple relevant tokens, reducing redundant row activations and unnecessary computation, as illustrated in Figure 4. Compared with conventional sequential mapping, this design increases the usefulness of each row access and enables coarse-grained execution skipping. The approach thus balances hardware and algorithmic needs: PIM architectures benefit from regular row-granular access patterns, while clustering ensures that each accessed row contains semantically important tokens. 

## **4 STARC System Architecture** 

This section details the hardware-algorithm co-design principles of STARC. We begin by introducing the underlying PIM 

architecture, which provides massive near-bank parallelism but imposes rigid row-level access constraints. We then describe how STARC leverages this architecture to perform efficient KV clustering directly inside HBM-PIM, thereby eliminating costly GPU offloading and reusing existing PIM primitives and hardware without introducing additional area overhead. 

## **4.1 PIM Architecture Overview** 

To enable high-throughput execution of attention mechanisms in Transformer-based models, we adopt AttAcc [40] as our PIM architecture—a PIM system specifically designed to accelerate the attention layer. As illustrated in Figure 6, AttAcc places compute units near each bank within an HBM stack. Specifically, a single HBM channel contains 2 pseudochannels (pCHs), each pCH is divided into 2 ranks, and every rank further breaks down into 4 bank groups, with 4 banks in each group. This results in a total of 64 banks per channel, which can be activated simultaneously to collectively utilize the full channel bandwidth and drive the near-bank compute fabric efficiently. 

**==> picture [251 x 198] intentionally omitted <==**

**----- Start of picture text -----**<br>
1KB KV Cache<br>2 [13] rows<br>2 Channels<br>2 [13] rows<br>pCH0 pCH0 Rank0 Rank1 Bank0<br>pCH1 pCH1<br>BG0   BG1     BG0   BG1<br>BG2   BG3     BG2   BG3<br>Bank1 16 Keys<br>Bank0<br>2 GEMV Units 16 Values<br>HBM Stack Bank1Bank2<br>2 GEMV Units<br>Bank3 Bank2<br>PHY Accumulator<br>TSVs<br>Accumulators Softmax Units<br>Buffer Die<br>Bank3<br>…<br>…<br>…<br>…<br>…<br>…<br>…<br>…<br>**----- End of picture text -----**<br>


**Figure 6.** HBM-PIM architecture and KV cache organization. 

A key principle of STARC is an architecture–algorithm codesign strategy: we select the number of clusters in K-means such that the arithmetic intensity of clustering matches the hardware-defined tipping point between memory-bound and compute-bound execution. This balance is determined by the architecture of our simulated HBM-PIM system. Each bank hosts a dedicated GEMV compute unit, and each pCH integrates 32 GEMV units. Each GEMV contains 16 FP16 fusedmultiply-add (FMA) pairs operating at 666 MHz. The system includes 40 HBM stacks, each consisting of 16 channels, yielding a total of 40 × 16 = 640 channels and 640 × 2 = 1280 pCHs. With 32 GEMV units per pCH and 2 FMA operations per unit per cycle, the peak compute throughput is: 

1867 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

Peak FLOPs = 32 × 2 × 16 × 1280 × 666 MHz 

**==> picture [160 x 11] intentionally omitted <==**

**==> picture [251 x 22] intentionally omitted <==**

This arithmetic intensity value serves as a hardware-defined tipping point: workloads with intensity below _𝐼_[∗] are memorybound, while those above are compute-bound. In our algorithm design (Section 5), we exploit this principle by selecting the number of clusters _𝐾_ in K-means such that the arithmetic intensity of the clustering workload matches _𝐼_[∗] . 

Additionally, despite HBM-PIM’s high throughput, its execution model offers limited flexibility. As illustrated in Figure 6, under our configuration, each DRAM bank row stores 1KB of data. Assuming FP16 precision (2B per element) and an attention head dimension of 128 (as in typical LLaMAstyle models), a single key or value vector occupies 256B. To fully utilize the parallelism across banks, each vector is dimension-partitioned across the four banks within a bank group, such that each bank stores a contiguous 64B slice of the vector. Consequently, one row across a bank group can accommodate 16 complete key or value vectors, yielding a row-level block size of blkrow = 16, meaning that a single row activation accesses 16 complete key or value vectors at once. 

## **4.2 Efficient KV Clustering Implementation on PIM** 

Although clustering-based remapping can mitigate row-level inefficiencies, performing clustering efficiently on hardware presents additional challenges. During decoding, the QKV generation stage already writes the key and value vectors into HBM. Offloading these vectors to GPUs for clustering would incur substantial transfer overhead across the memory interface, negating the benefits of in-memory data layout optimization. To avoid this bottleneck, we perform KV clustering directly inside HBM-PIM, leveraging AttAcc’s nearbank compute fabric to execute the three phases of K-means: **normalization** , **assignment** , and **update** . 

Table 1 details the command-level breakdown of cosinebased K-means clustering implemented on PIM. We denote _𝐷_ as the number of vector dimensions and _𝑆_ as the byte size of an FP16 value (two bytes). Each GEMV unit supports 64-way SIMD MACs, so computing a dot product between two _𝐷_ -dimensional vectors requires _𝑇𝐷_ = _𝐷_ /64 **MAC_AB** operations. Following Section 4.1, we use blkrow to denote the number of _𝐷_ -dimensional vectors accommodated in one DRAM row across a bank group. To compare against _𝐾_ centroids, the system requires _𝑇𝐾_ = _𝐾_ /blkrow such operations. Finally, we denote _𝑁_ , _𝐾_ , and _𝐼_ as the number of samples, clusters, and clustering iterations, respectively. Following 

**Table 1.** Command-level breakdown of cosine-based K- means clustering on PIM. Read/write bytes include only PIM-side memory traffic; host-side scalar operations are excluded. 

|**Operation**|**Command Count**|**MAC**|**Read Bytes**|**Write Bytes**|
|---|---|---|---|---|
|**Normalization (per vector)**|||||
|MAC_AB(self-dot)<br>MVSB(norm)<br>VNORM(vector/~~√~~<br>·)<br>**Total / vector**|_𝑇𝐷_<br>1<br>_𝑇𝐷_<br>—|_𝐷_<br>0<br>_𝐷_<br>2_𝐷_|_𝐷𝑆_<br>—<br>_𝐷𝑆_<br>2_𝐷𝑆_|—<br>_𝑆_<br>—<br>_𝑆_|
|**Assignment (per iteration)**|||||
|WRGB(samples)<br>MAC_AB<br>MVSB(scores)<br>Host(argmax)<br>**Total / iteration**|_𝑁_<br>_𝑁𝑇𝐷_×_𝑇𝐾_<br>_𝑁𝑇𝐾_<br>—<br>—|0<br>_𝑁𝐾𝐷_<br>0<br>—<br>_𝑁𝐾𝐷_|—<br>samples:_𝑁𝐷𝑆_,<br>centroids:_𝐾𝐷𝑆_<br>—<br>_𝑁𝐾𝑆_<br>(_𝑁𝐷_+_𝐾𝐷_+_𝑁𝐾_)_𝑆_|_𝑁𝐷𝑆_<br>—<br>_𝑁𝐾𝑆_<br>only labels<br>_𝑁𝐷𝑆_+_𝑁𝐾𝑆_|
|**Update (per iteration)**|||||
|MVGB(broadcast_𝑣𝑖_)<br>MAC_AB<br>(accumulation & averaging)<br>WRGB(new_𝜇𝑘_)<br>**Total / iteration**|_𝑁_<br>_𝑁𝑇𝐷_<br>1<br>—|0<br>(_𝑁𝐷_+_𝐾𝐷_)/2<br>0<br>(_𝑁𝐷_+_𝐾𝐷_)/2|_𝑁𝐷𝑆_<br>—<br>—<br>_𝑁𝐷𝑆_|—<br>—<br>_𝐾𝐷𝑆_<br>_𝐾𝐷𝑆_|



prior modeling practice, we approximate one addition, multiplication, or division as half a MAC, since each corresponds to a single FLOP. 

**Normalization.** To enable cosine similarity computation, each vector must first be normalized into the form _𝑣_ /∥ _𝑣_ ∥. As shown in Table 1, this process begins with a self dotproduct via **MAC_AB** , requiring _𝑇𝐷_ commands, _𝐷_ multiplyaccumulate operations, and reading _𝐷𝑆_ bytes from memory. The resulting scalar norm is then transferred into the softmax buffer using **MVSB** . To avoid host involvement and reduce data transfers across the memory interface, we introduce a fused command **VNORM** , implemented via a small lookuptable (LUT)–based reciprocal square-root approximation and the scaling datapath, since the ~~√~~ ∥1 _𝑣_ ∥[2][term used in clustering] does not require high precision and can be approximated using a piecewise-defined LUT. Both the LUT lookup and the ensuing multiply–accumulate and scaling operations are native to AttAcc’s PIM primitives, and thus neither **VNORM** nor the clustering control logic introduces new hardware structures or additional area overhead. This step requires another _𝑇𝐷_ commands and _𝐷_ operations, reading the vector once more ( _𝐷𝑆_ bytes). In total, per-vector normalization entails 2 _𝐷_ MACs, 2 _𝐷𝑆_ bytes of reads, and _𝑆_ bytes of writes. 

**Assignment.** After normalization, each sample must be assigned to its closest centroid. For each of the _𝑁_ samples, we first write the sample vector into the GEMV buffer with a **WRGB** command, incurring _𝑁𝐷𝑆_ bytes of writes. The sample is then compared against all _𝐾_ centroids using _𝑁𝑇𝐷_ × _𝑇𝐾_ **MAC_AB** operations, corresponding to _𝑁𝐾𝐷_ MACs. Here, the read volume includes both the sample ( _𝑁𝐷𝑆_ bytes) and the centroids ( _𝐾𝐷𝑆_ bytes). The resulting similarity scores are dispersed across different row blocks, so they must be gathered into the softmax buffer before the host can perform argmax. This gathering is carried out with **MVSB** commands: 

1868 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

each sample requires _𝑇𝐾_ such transfers to collect all _𝐾_ scores, leading to _𝑁𝑇𝐾_ commands and _𝑁𝐾𝑆_ bytes of writes in total. Finally, the host performs the argmax across _𝐾_ scores per sample, which involves reading _𝑁𝐾𝑆_ bytes and returning only cluster labels. Overall, the assignment phase per iteration requires _𝑁𝐾𝐷_ MACs, ( _𝑁𝐷_ + _𝐾𝐷_ + _𝑁𝐾_ ) _𝑆_ bytes of reads, and _𝑁𝐷𝑆_ + _𝑁𝐾𝑆_ bytes of writes. 

**Update.** Once assignments are made, cluster centroids must be updated by averaging the vectors assigned to each cluster. To enable accumulation across all centroids, each of the _𝑁_ sample vectors is broadcast to the GEMV buffer across all banks using _𝑁_ **MVGB** commands, corresponding to _𝑁𝐷𝑆_ bytes of reads. Accumulation is then carried out via _𝑁𝑇𝐷_ **MAC_AB** operations. We approximate the operation count as ( _𝑁𝐷_ + _𝐾𝐷_ )/2 equivalent MACs, accounting for vector additions and the final scalar divisions when averaging. Because samples are already broadcast into GEMV buffers, no additional read traffic is incurred. Finally, the new centroids _𝜇𝑘_ are written back to memory with a single **WRGB** command, writing _𝐾𝐷𝑆_ bytes. In total, the update phase per iteration requires ( _𝑁𝐷_ + _𝐾𝐷_ )/2 equivalent MACs, _𝑁𝐷𝑆_ bytes of reads, and _𝐾𝐷𝑆_ bytes of writes. 

Through this breakdown, Table 1 demonstrates that all three phases of cosine-based K-means can be expressed as compositions of existing PIM commands ( **MAC_AB** , **WRGB** , **MVSB** , **MVGB** ) augmented with one lightweight fused command ( **VNORM** ). By carefully mapping normalization, assignment, and update into these command sequences, STARC leverages existing PIM primitives and hardware to achieve inmemory clustering of KV vectors directly within HBM-PIM, eliminating costly GPU offloading and enabling hardwareaware clustering aligned with AttAcc’s memory architecture. 

**==> picture [245 x 331] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prefill tokens<br>Values Values Values Values<br>Keys Keys Keys Keys<br>0 0 1 1<br>1 1 2 2<br>2 2 3 3<br>3 3 0 0<br>4 4 4 4<br>5 5 5 5<br>67 1 Block partitioning 6 7 2 local clusteringBlock-wise  6 7 3 cluster centroids [Select KV using ] 6 7<br>8 8 8 Query 8<br>9 9 9 9<br>10 10 11 11<br>11 11 14 14<br>12 12 10 10<br>13 13 12 12<br>14 14 13 13<br>15 15 15 15<br>Values Values<br>Keys Keys<br>1 1<br>2 2<br>3 3<br>0 Values 0 Values<br>Keys Keys<br>4 16 4 16<br>5 17 5 19<br>6 18 4 Incremental clustering  6 20<br>every  N  decoding steps<br>7 19 7 22<br>8 20 8 17<br>9 21 9 18<br>11 22 11 21<br>14 23 14 23<br>10 Newly generated  10<br>12 tokens during  12<br>13 decoding 13<br>15 15<br>**----- End of picture text -----**<br>


**Figure 7.** Flowchart of the clustering algorithm. We perform incremental clustering on the KV pairs using K-means, meaning that only the newly generated segment of KV pairs is clustered during decoding. 

## **5 Algorithm Design** 

Building upon the STARC framework, we propose an online clustering strategy that incrementally reorganizes the KV cache during decoding. The aim is to balance model accuracy with HBM-PIM’s row-level access granularity by grouping semantically similar KV pairs into hardware-aware clusters. While these clusters may not always align exactly with HBM rows, the resulting regularized access pattern effectively reduces row over-fetch and improves bandwidth utilization. The overall procedure is outlined in Algorithm 1. 

We begin by quantifying the arithmetic intensity (AI) of cosine K-means, defined as the ratio between floating-point operations (FLOPs) and main-memory traffic in bytes, using the notation ( _𝑁, 𝐾, 𝐷, 𝐼,𝑆_ ) in Section 4.2. 

**One-off normalization cost.** Each vector undergoes an _ℓ_ 2 normalization prior to clustering. Computing the squared norm requires _𝐷_ multiply-add pairs (2 _𝐷_ FLOPs), followed by one square root and one reciprocal (host-side scalar operations, excluded from FLOPs). The normalized vector is reconstructed by _𝐷_ scalar multiplications ( _𝐷_ FLOPs). Thus, 

each vector incurs 3 _𝐷_ FLOPs and 3 _𝐷𝑆_ bytes of traffic. For _𝑁_ + _𝐾_ vectors, this yields 

**==> picture [217 x 11] intentionally omitted <==**

Given _𝐼_ ≫ 1 and _𝑁_ ≫ _𝐾_ , this one-off cost is amortized and omitted from the per-iteration AI. 

**Per-iteration cost.** Each Lloyd iteration consists of: 

**(1) Assignment:** Each sample is compared with all _𝐾_ centroids via _𝐷_ -dimensional dot products, each requiring 2 _𝐷_ FLOPs. Across all _𝑁_ samples and _𝐾_ centroids: 

**==> picture [199 x 12] intentionally omitted <==**

where the byte count accounts for reading both _𝑁_ samples and _𝐾_ centroids from main memory. 

**(2) Update:** Updating centroids involves adding _𝑁_ samples into _𝐾_ cluster sums ( _𝑁𝐷_ additions) and scaling each centroid by 1/ _𝑛𝑘_ ( _𝐾𝐷_ scalar multiplications/divisions): 

**==> picture [188 x 12] intentionally omitted <==**

1869 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

**Algorithm 1** Clustering-Based Retrieval during Decoding 

- **Require:** Prefill KV pairs Kpre _,_ Vpre; Decoding stream { _𝑥𝑡_ }; Block size _𝑁_ ; KV cache budget _𝐵_ 

- 1: // **Initial clustering after prefill** 

- 2: Partition (Kpre _,_ Vpre) into non-overlapping blocks of size _𝑁_ 

- 3: **for** each block (K _𝑏,_ V _𝑏_ ) **do** 4: C _𝑏_ ← KMeans(K _𝑏_ ) _⊲_ cosine similarity 5: Assign each ( _𝑘𝑖, 𝑣𝑖_ ) ∈(K _𝑏,_ V _𝑏_ ) to its cluster in C _𝑏_ 6: C ←C ∪C _𝑏_ 7: **end for** 

- 8: Initialize: Knew ←∅, Vnew ←∅ 

- 9: **for** each decoding step _𝑡_ **do** 

**==> picture [239 x 103] intentionally omitted <==**

- 18: // **KV retrieval for current step** 19: Compute scores _𝑠 𝑗_ = _𝑞𝑡_[⊤] _[𝜇][𝑗]_[for all centroids] _[ 𝜇][𝑗]_[∈C] 20: Sort clusters by _𝑠 𝑗_ in descending order 21: Select top clusters until total token count reaches _𝐵_ 22: Truncate final cluster if needed to fit budget _𝐵_ 23: Include all non-clustered tokens in Knew _,_ Vnew 24: **end for** 

where the byte count corresponds to writing updated centroids back to memory. 

**Total per-iteration AI.** The per-iteration arithmetic intensity is therefore 

**==> picture [220 x 27] intentionally omitted <==**

For _𝑁_ ≫ _𝐾_ , this simplifies to 

**==> picture [140 x 21] intentionally omitted <==**

Thus, under ideal centroid reuse and negligible host overhead, the algorithm-level AI scales linearly with _𝐾_ for FP16 data. On the hardware side, Section 4.1 established the peak throughput and compute-to-memory tipping point _𝐼_[∗] , yielding Peak FLOPs ≈ 873 TFLOPs/s and _𝐼_[∗] ≈ 4 FLOPs/Byte. Comparing the two results gives a clear co-design rule: choose _𝐾_ so that AI ≈ _𝐼_[∗] . Since AI ≈ _𝐾_ under FP16, we set _𝐾_ =4 to ensure the clustering workload operates near the hardwaredefined balance point. 

Based on this principle, we design a hardware-aware online clustering method that reorganizes the KV cache into contiguous, row-aligned clusters and keeps the clusters fixed after their initial formation, so that each vector is clustered only once. As shown in Figure 7, at the start of decoding, the prefill tokens are divided into non-overlapping blocks of size _𝑁_ ➊. We apply cosine K-means with _𝐾_ =4 and random initialization to each block, limiting the number of iterations _𝐼_ to 16 to control runtime ➋. Clustering is applied to keys only, and the corresponding values inherit the same labels. The resulting clusters are stored in contiguous physical locations that match the PIM bank layout. With a PIM row size of blkrow = 16 and _𝐾_ =4, we set _𝑁_ = _𝐾_ × blkrow = 64 so that each cluster contains about 16 tokens, aligning the access granularity with the row size and reducing row overfetch and internal data movement. Once these prefill clusters are formed, they remain unchanged to avoid costly reshuffling under row-level access. 

During decoding, newly generated tokens are kept in full for attention computation until their number reaches the size _𝑁_ , as they strongly influence the immediate attention distribution. The same as the processing of tokens generated in the prefill stage, every _𝑁_ = 64 decoding steps, we cluster only the most recent 64-token block using the same configuration ( _𝐾_ =4, up to 16 iterations), append the resulting clusters, and store them contiguously ➍. Once formed, clusters remain fixed and are never updated. As a result, STARC does not require re-clustering throughout inference, thereby avoiding the costly remapping of clustered KV vectors already stored in memory. This incremental, append-only design not only reduces the clustering overhead but also draws on two observations. First, the distribution of decoding keys gradually diverges from that of the prefill keys (Figure 8), which justifies clustering the two stages separately. Second, key vectors exhibit locality, meaning that adjacent tokens tend to have high cosine similarity. Clustering only the most recent contiguous segment takes advantage of this property, improving clustering quality while keeping the approach suitable for online inference. 

**==> picture [212 x 128] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prefill Tokens<br>Decoding Tokens 350<br>300<br>250<br>200<br>150<br>100<br>50<br>0<br>Decoding Steps<br>**----- End of picture text -----**<br>


**Figure 8.** The distributions of key vectors differ significantly between the prefill and decoding stages. 

1870 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

**Table 2.** LongBench results for STARC and baseline sparsity methods (KV cache budget: 1024 tokens). 

|KV Budget: 1024|Single-Document QA<br>NrtvQA<br>Qasper<br>MF-en|Multi-Document QA<br>HotpotQA<br>2WikiMQA<br>Musique|Summarization<br>GovReport<br>QMSum<br>MultiNews|Few-Shot Learning<br>TREC<br>TriviaQA<br>SAMSum|Synthetic<br>PCount<br>PRe|Code<br>Lcc<br>RB-P<br>Avg.|
|---|---|---|---|---|---|---|
|**LongChat**|||||||
|Full KV<br>STARC<br>SparQ<br>InfniGen<br>Quest|19.51<br>25.98<br>**43.80**<br>17.55<br>29.44<br>40.92<br>**19.56**<br>**29.90**<br>40.90<br>15.41<br>29.56<br>41.92<br>14.58<br>29.23<br>43.67|31.94<br>**23.20**<br>11.38<br>32.32<br>19.29<br>9.73<br>31.05<br>22.84<br>**12.92**<br>**36.20**<br>20.35<br>8.89<br>28.37<br>18.62<br>10.51|**31.77**<br>21.66<br>26.06<br>31.22<br>22.08<br>25.01<br>30.98<br>**23.19**<br>**26.49**<br>29.36<br>22.22<br>24.73<br>29.12<br>22.29<br>24.91|**66.00**<br>82.00<br>20.79<br>64.00<br>80.80<br>21.82<br>64.00<br>**84.53**<br>25.89<br>64.00<br>84.38<br>**29.75**<br>**66.00**<br>79.31<br>20.88|**2.00**<br>30.00<br>**2.00**<br>32.00<br>0.00<br>30.50<br>**2.00**<br>32.00<br>**2.00**<br>**34.00**|53.86<br>48.68<br>33.66<br>**57.16**<br>48.82<br>33.38<br>54.34<br>**55.72**<br>34.55<br>51.84<br>51.06<br>33.98<br>52.60<br>49.00<br>32.82|
|**Mistral**|||||||
|Full KV<br>STARC<br>SparQ<br>InfniGen<br>Quest|23.94<br>40.07<br>57.58<br>19.97<br>34.93<br>57.70<br>**29.36**<br>**40.93**<br>53.68<br>23.34<br>37.73<br>**57.90**<br>22.79<br>30.88<br>52.39|49.10<br>36.71<br>22.27<br>**51.49**<br>35.48<br>23.39<br>51.33<br>37.36<br>**27.22**<br>51.41<br>**39.45**<br>19.69<br>47.12<br>38.63<br>18.73|35.66<br>**25.77**<br>26.80<br>**35.67**<br>24.72<br>26.72<br>34.49<br>25.67<br>**27.66**<br>35.06<br>24.89<br>26.29<br>33.45<br>24.23<br>27.26|**80.00**<br>87.67<br>47.35<br>76.00<br>**88.87**<br>**48.16**<br>74.00<br>88.86<br>47.17<br>76.00<br>85.67<br>47.60<br>66.00<br>88.42<br>44.73|4.00<br>98.00<br>2.00<br>98.00<br>5.00<br>**99.00**<br>2.00<br>98.00<br>**8.18**<br>92.00|58.98<br>56.36<br>46.89<br>**61.74**<br>55.76<br>46.29<br>60.43<br>**62.14**<br>47.77<br>59.82<br>59.58<br>46.53<br>60.86<br>57.52<br>44.57|
|**Llama-3.1**|||||||
|Full KV<br>STARC<br>SparQ<br>InfniGen<br>Quest|27.02<br>13.98<br>28.04<br>**31.73**<br>13.57<br>**28.14**<br>29.53<br>13.83<br>26.97<br>28.80<br>**14.15**<br>27.88<br>18.66<br>11.75<br>22.96|18.30<br>17.45<br>**13.01**<br>20.40<br>**18.08**<br>11.54<br>17.64<br>16.85<br>10.27<br>**24.27**<br>17.79<br>9.75<br>16.90<br>13.52<br>5.46|**35.83**<br>23.66<br>25.91<br>35.26<br>23.53<br>25.62<br>33.95<br>**23.79**<br>**26.73**<br>34.15<br>23.31<br>26.59<br>34.22<br>22.12<br>25.87|**74.00**<br>89.77<br>**44.56**<br>72.00<br>88.57<br>44.25<br>71.00<br>**91.47**<br>44.20<br>70.00<br>89.81<br>44.05<br>70.00<br>85.60<br>42.94|3.92<br>97.50<br>5.67<br>**98.33**<br>**7.12**<br>98.21<br>4.67<br>96.00<br>0.80<br>96.27|63.30<br>55.06<br>39.46<br>**64.30**<br>54.42<br>39.71<br>64.19<br>**60.44**<br>39.76<br>61.98<br>59.02<br>39.51<br>58.90<br>56.08<br>36.38|



At inference time, KV retrieval operates at the cluster level. At each decoding step, the current query is compared against all cluster centroids using dot products ➌. Clusters are ranked by the resulting scores, and the top-ranked clusters are retrieved until the KV budget is reached. Because clusters may contain different numbers of KV entries, the last retrieved cluster may be partially truncated to stay within the budget. 

## **6 Evaluation** 

## **6.1 Evaluation Methodology** 

_**Accuracy Evaluation.**_ To evaluate the effectiveness of STARC under long-context scenarios, we consider three representative LLMs: LongChat-7B-v1.5-32K (MHA) [28], LLaMA-3.1-8B-Instruct (GQA) [6], and Mistral-7B-Instructv0.3 (GQA) [21]. These models cover both multi-head and grouped-query attention mechanisms, enabling a comprehensive study of STARC across different attention designs. For benchmarking, we use the LongBench benchmark [2], consisting of 16 datasets across diverse tasks: multi-document QA (HotpotQA [54], 2WikiMQA [53], Musique [51]), singledocument QA (QASPER [5], MultiFieldQA-en, NarrativeQA [23]), summarization (GovReport [18], QMSum [58], MultiNews [7]), few-shot learning (TriviaQA [22], TREC [30], SAMSum [11]), synthetic reasoning (PCount, PRe [43]), and code completion (Lcc [12], RB-P [34]). We also present an evaluation on the RULER benchmark [17], which is designed to stress-test model robustness under extreme long-context scenarios. In addition, we evaluate on PG-19 [42] for language modeling using perplexity as the evaluation metric. 

We compare STARC against three recent sparsity methods: Quest [49], InfiniGen [26], and SparQ [44]. Each baseline follows the configurations in its original paper (e.g., page size for Quest, partial weights and threshold for InfiniGen, and largest retained components _𝑟_ for SparQ). For a fair comparison, we reproduce all methods under the same framework and adopt the Quest setting of using full KV cache in the first two layers, which typically exhibit low sparsity [49]. Unless otherwise specified, results are reported at a KV cache budget of 1024 tokens, matching the budget used in our performance experiments. Results under other budgets (256, 512, 2048) are provided in the appendix. For STARC, we perform clustering over every 64 consecutive tokens using cosine-based K-means, with the number of clusters fixed at _𝐾_ = 4. 

_**Performance on PIM Systems.**_ To investigate how attention sparsity impacts PIM architectures and evaluate the effectiveness of STARC, we adopt the AttAcc simulator [40], which extends Ramulator [36] to model heterogeneous GPU–PIM systems, and evaluate on a DGX+AttAcc platform where attention kernels are offloaded to PIM units while FC layers remain on GPU. The DGX consists of 8 NVIDIA H100 cores and 40 HBM3 stacks (5.2 Gbps per pin), with a total memory capacity of 1.28 TB. The AttAcc side contains an additional 40 HBM3 stacks, also totaling 1.28 TB. Each DRAM bank integrates one GEMV unit (1P1B configuration), and all arithmetic and buffer components follow the microarchitectural assumptions in AttAcc [40]. 

We configure inference workloads to emphasize longcontext, memory-bound decoding scenarios, with prefill/decoding sequence pairs of (2K, 16K), (2K, 24K), and (2K, 32K). Batch size is fixed at 16. The evaluated models include 

1871 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

LLaMA-7B, Mistral-7B, and GPT-13B, all at FP16 precision. To highlight trade-offs between accuracy and efficiency, we incorporate STARC’s KV clustering overhead into the simulation. Page-wise sparsity is represented by Quest, while token-wise sparsity is represented by SparQ, which achieves the highest accuracy in our experiments. We adopt AttAcc’s optimal configuration by enabling both head-level pipelining and feedforward co-processing. All other simulator configurations follow AttAcc defaults. 

## **6.2 Accuracy Evaluation** 

_**Results on LongBench.**_ Table 2 presents the results on LongBench datasets under a KV cache budget of 1024. Several consistent trends emerge across models. First, STARC outperforms the page-wise sparsity method Quest in terms of average accuracy across all models. Second, STARC achieves accuracy comparable to token-wise sparsity methods (SparQ and InfiniGen), and on grouped-query attention models (LLaMA3.1 and Mistral) it achieves the best results among all sparsity methods on many datasets. These results indicate that STARC provides robust accuracy across both MHA and GQA models, while aligning better with PIM hardware. 

**==> picture [242 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
STARC SparQ InfiniGen Quest Full KV<br>13<br>Zoomed-In View<br>7.0<br>12<br>11 6.8<br>10<br>6.6<br>30000 31000 32000<br>9<br>8<br>7<br>6<br>0 5000 10000 15000 20000 25000 30000<br>Input Length<br>Perplexity<br>**----- End of picture text -----**<br>


**Figure 9.** Language modeling on PG-19 dataset. 

**==> picture [242 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
STARC SparQ InfiniGen Quest<br>1.0 HotpotQA 1.0 NarrativeQA<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>KV Cache Budget KV Cache Budget<br>128256 512 1024 2048 128256 512 1024 2048<br>Recall Rate Recall Rate<br>**----- End of picture text -----**<br>


**Figure 10.** Recall rate of important tokens. 

_**Results on RULER.**_ Table 3 reports the results on the RULER benchmark for LLaMA-3.1-8B-Instruct at a context length of 32K. RULER consists of 13 tasks grouped into four categories: Retrieval, Multi-Hop Tracing, Aggregation, and Question Answering. All methods are evaluated under the same KV budget of 1024. Overall, STARC achieves average accuracy close to the full-KV and SparQ baseline, while outperforming InfiniGen. Moreover, STARC outperforms the page-wise sparsity baseline Quest across most tasks. These results further support the robustness of STARC under longcontext scenarios. 

_**Results on Language Modeling.**_ Figure 9 shows the perplexity of generated tokens on the PG-19 test set across varying input lengths, ranging from 1 to 32,000 tokens, under a fixed KV budget of 1024. STARC outperforms both Quest and InfiniGen, particularly at longer input lengths. Although SparQ slightly outperforms STARC, the gap remains narrow, and STARC consistently tracks closely with the Full-KV baseline. 

_**Recall Rate of Important Tokens.**_ Figure 10 reports the recall rate of important tokens on HotpotQA and NarrativeQA. Although STARC does not surpass SparQ, it achieves higher recall than both Quest and InfiniGen across all budgets. This demonstrates that STARC’s clustering strategy improves the selection of semantically important tokens, which explains its strong downstream task performance. 

## **6.3 Performance on PIM Systems** 

We evaluate attention sparsity on PIM systems using three models (LLaMA-7B, GPT-13B, and Mistral-7B) under longcontext decoding scenarios with sequence pairs (2k, 16k), (2k, 24k), and (2k, 32k). All methods use a KV cache budget of 1024 tokens. 

To assess hardware efficiency, we analyze the attention masks produced by each method at each decoding step and map them to the row-level granularity of the PIM architecture, where each DRAM row activation fetches blkrow = 16 key/value vectors in parallel. The efficiency thus depends on how well the retrieved tokens align with row boundaries. Page-wise sparsity naturally avoids over-fetching, since each page matches the row size exactly. In contrast, token-wise sparsity often scatters tokens across many rows, leading to additional memory accesses and the processing of irrelevant data. STARC retrieves tokens at the cluster level, so semantically similar tokens are stored in the same or adjacent rows during cluster construction, significantly reducing redundant memory activations. 

Figure 11 presents the normalized end-to-end decoding latency (top) and energy (bottom) per token. Each bar is broken down into attention, feed-forward, communication, and miscellaneous costs. The yellow markers show the additional KV clustering overhead of STARC, plotted against the right _𝑦_ -axis. 

1872 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

**Table 3.** RULER results on LLaMA-3.1-8B-Instruct with 32K context length. 

|Niah1<br>Niah2<br>Niah3<br>MKey1<br>MKey2<br>MKey3<br>MValue<br>MQuery<br>VT<br>CWE<br>FWE<br>QA1<br>QA2|Avg.|
|---|---|
|||
|Full KV<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>0.9844<br>1.0000<br>0.9938<br>0.1479<br>0.9444<br>0.8542<br>0.5312<br>STARC<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>0.9688<br>0.9479<br>0.9688<br>0.9948<br>0.9896<br>0.1729<br>0.9167<br>0.8542<br>0.5312<br>SparQ<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>1.0000<br>0.9844<br>1.0000<br>0.9854<br>0.2396<br>0.8854<br>0.8542<br>0.5312<br>InfniGen<br>1.0000<br>1.0000<br>0.9896<br>1.0000<br>0.9583<br>0.7812<br>0.9193<br>0.9974<br>0.9542<br>0.1917<br>0.7882<br>0.8542<br>0.5104<br>Quest<br>0.9792<br>1.0000<br>0.8854<br>1.0000<br>1.0000<br>0.2500<br>0.9609<br>0.9870<br>0.8688<br>0.1115<br>0.8472<br>0.8333<br>0.4792|0.8812<br>0.8727<br>0.8831<br>0.8419<br>0.7848|



**==> picture [199 x 6] intentionally omitted <==**

**----- Start of picture text -----**<br>
ATTN FC COMM ETC CLUSTER<br>**----- End of picture text -----**<br>


**==> picture [500 x 212] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.001<br>0.84 0.86 0.86 0.0008<br>0.80.60.4 0.35 0.70 0.70 0.43 0.74 0.61 0.61 0.50 0.66 0.54 0.54 0.75 0.75 0.37 0.77 0.67 0.67 0.43 0.70 0.61 0.60 0.37 0.68 0.68 0.46 0.75 0.59 0.59 0.53 0.66 0.52 0.51 0.00060.0004<br>0.29<br>0.2 0.19 0.17 0.15 0.15 0.14 0.13 0.23 0.21 0.18 0.0002<br>0.05 0.05 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.03 0.03 0.03 0.06 0.05 0.05 0.04 0.04 0.04<br>0.0 0<br>1.0 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 0.001<br>0.0008<br>0.8 0.74<br>0.60.4 0.48 0.68 0.59 0.57 0.55 0.59 0.51 0.50 0.60 0.53 0.46 0.44 0.40 0.66 0.64 0.48 0.65 0.58 0.57 0.53 0.59 0.52 0.51 0.50 0.69 0.57 0.55 0.57 0.60 0.49 0.48 0.62 0.53 0.44 0.42 0.00060.0004<br>0.2 0.16 0.14 0.13 0.14 0.13 0.12 0.19 0.17 0.15 0.0002<br>0.07 0.05 0.06 0.04 0.05 0.04 0.06 0.04 0.05 0.04 0.05 0.04 0.07 0.05 0.06 0.05 0.06 0.04<br>0.0 0<br>Decoding Length: 16K 24K 32K 16K 24K 32K 16K 24K 32K<br>Model: LLAMA-7B GPT-13B MISTRAL-7B<br>Full KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wiseFull KVToken-wiseSTARCPage-wise<br>Normalized Execution Time Normalized Cluster Latency<br>Normalized Energy<br>Normalized Cluster Energy<br>**----- End of picture text -----**<br>


**Figure 11.** Normalized end-to-end decoding latency and energy on PIM systems across different models and sequence lengths. 

Several consistent trends can be observed across all three models. As the decoding length increases, the attention layer rapidly becomes the dominant contributor to both latency and energy, and the benefits of sparsity grow accordingly. At the level of overall decoding, even token-wise sparsity achieves up to 34% speedup and 47% energy reduction compared to full KV retrieval. STARC further improves efficiency, providing 25%–48% speedup and 34%–56% energy reduction, corresponding to 13%–21% faster execution and 11%–18% lower energy consumption than token-wise methods. 

When isolating the attention layer, the improvements are even more pronounced. Relative to full KV retrieval, STARC reduces attention latency by up to 93% and energy by up to 92%. Compared to token-wise sparsity, STARC still achieves up to 78% latency reduction and 65% energy reduction. Importantly, in both latency and energy, STARC approaches the ideal efficiency of page-wise sparsity, while preserving much higher model accuracy. 

Notably, these improvements come at virtually no additional cost: the clustering overhead of STARC is negligible. Unlike full or sparse attention where each decoding step 

requires past tokens (on the order of ( _𝐿_ in + _𝐿_ out) _𝐿_ out/2 or _𝐵_ · _𝐿_ out tokens, respectively), STARC only clusters each token once, resulting in _𝐿_ in+ _𝐿_ out clustering operations in total. This incremental design makes the overhead scale linearly with context length rather than quadratically, which explains why it remains around 0 _._ 02% of total decoding latency and energy in long-context settings, as shown by the yellow markers. 

Overall, STARC achieves significant reductions in attentionlayer latency and energy relative to token-wise sparsity methods, while providing substantially higher accuracy than page-wise sparsity. These results demonstrate STARC’s effectiveness as a hardware-aware sparse attention mechanism tailored for long-context inference on PIM architectures. 

## **7 Related Work** 

## **7.1 PIM-enabled LLM Accelerators** 

PIM has emerged as an effective architectural paradigm to overcome the bandwidth bottlenecks in LLMs, particularly during autoregressive decoding. By placing compute units near memory arrays, PIM boosts bandwidth utilization and 

1873 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

parallelism for memory-intensive workloads. This has motivated many recent efforts to integrate PIM into LLM acceleration pipelines [4, 14, 15, 24, 40, 59]. 

**Hybrid Strategy.** To better balance the compute and memory workloads in LLMs, hybrid xPU–PIM designs have been proposed. AttAcc [40] maps attention layers to HBM-based PIM while keeping feed-forward computation on GPUs. NeuPIMs [15] combines NPUs (for GEMM) and PIMs (for GEMV) with dual-row buffers and sub-batch interleaving to reduce contention. PAPI [14] extends this model by dynamically scheduling workloads between GPUs and PIM units based on runtime profiling. IANUS [46] further unifies the NPU and PIM memory space, with a dedicated scheduling logic to interleave PIM execution and NPU memory accesses. However, none of these designs account for the irregular memory access patterns introduced by sparse attention. 

**Optimization for LLM with PIM.** Several works optimize LLM inference on PIM architectures [24, 27, 33, 37, 59]. TransPIM [59] improves Transformer inference via tokenbased dataflows and lightweight hardware extensions to HBM, yet is still tuned for dense computation. LoL-PIM [24] supports long-context LLMs with a distributed PIM design and dynamic memory management, but ignores token relevance. PIM-LLM [37] accelerates 1-bit LLMs by using analog PIM crossbars to perform binary projection matrix multiplications and digital systolic arrays to execute 8-bit attention matrix multiplications, yet it still assumes dense, fixed access patterns. Hermes [33] leverages near-data processing DIMMs to offload cold neurons in activation-heavy workloads, focusing on activation sparsity rather than attention - sparsity and lacking support for fine grained token selection. 

In summary, existing PIM-enabled LLM accelerators largely assume dense attention patterns and fail to address the challenges of sparse attention, such as irregular access and dynamic KV reuse, and fine-grained selection. This results in workload imbalance and poor memory efficiency. In contrast, our work introduces a sparsity-aware co-design of both memory layout and access strategy, enabling efficient execution of sparse attention under PIM architectures. 

## **7.2 Efficient LLM Inference** 

Sparsity-based methods have been widely explored to reduce the inference cost of LLMs, particularly under long-context scenarios where the KV cache becomes a memory and latency bottleneck. 

**KV Cache Eviction.** Several works propose permanently discarding less important tokens from KV cache to reduce memory footprint. H2O [56] and Scissorhands [35] rely on ranking tokens by cumulative attention scores or recency, while StreamingLLM [52] follows a similar recency-oriented design by retaining a small set of initial tokens as attention sinks together with a fixed sliding window. FastGen [9] introduces head-specific strategies for token selection. MorphKV [10] improves this by maintaining a fixed-size cache 

with correlation-aware updates, mitigating early-token bias. However, this kind of method results in the loss of crucial information, as previously evicted tokens may become relevant again during decoding. 

**Dynamic Token Access.** To avoid permanent loss, another line of work keeps the full KV cache but uses dynamic sparse attention to load only the relevant tokens at runtime. SparQ [44] approximates the relevant tokens using querykey projections to reduce memory transfers. InfiniGen [26] uses partial attention simulation to predict which tokens to prefetch. RocketKV [3] bridges permanent eviction and dynamic selection by first filtering the KV cache through coarse-grained token eviction and then applying fine-grained dynamic fetching. These approaches improve bandwidth efficiency, but ignore the architectural constraints of emerging memory systems like PIM. 

**Block-Based Optimization.** To bridge dynamic token access and hardware efficiency, several works adopt block-level optimization. Quest [49] partitions the KV cache into fixedsize pages and selects relevant blocks using query-aware scoring, which aligns better with PIM memory layouts. However, coarse page-level division may fetch irrelevant tokens. To address this, ClusterKV [32] and Squeezed Attention [16] introduce clustering-based KV retrieval for finer granularity and semantic relevance. SentenceKV [60] focuses on semantic clustering during the prefill stage but does not cluster or compress newly generated tokens during decoding. More broadly, these clustering-based methods do not target GPUPIM systems, as well as the deployment considerations such as data mapping and clustering in PIM. 

Our method, STARC, builds on this line of work by jointly designing clustering-based sparsity and a memory-aware layout for PIM systems. This co-design provides a balanced solution that improves both model accuracy and hardware efficiency for long-context inference. 

## **8 Conclusion** 

In this work, we propose STARC, a clustering-based data mapping strategy that enables efficient sparse attention execution on PIM architectures. By co-locating semantically similar KV pairs and remapping them to contiguous memory regions, STARC bridges the gap between dynamic tokenwise sparsity and the rigid row-level access granularity of PIM. This co-design improves both throughput and energy efficiency without compromising model accuracy. Experiments show that STARC achieves up to 78% latency reduction and 65% energy savings on the attention layer compared to token-wise sparsity baselines. We hope that our work inspires further integration of PIM architectures with emerging LLM optimization techniques, ultimately enabling scalable and efficient LLM inference in real-world deployments. 

1874 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

## **Acknowledgments** 

This work was supported in part by RPI-IBM Future of Computing Research Collaboration and the National Science Foundation under Award Number 2442271. We thank the anonymous reviewers for their constructive feedback and our shepherd Jongse Park for guidance throughout the revision process. We also thank Yinan Wang for insightful discussions. 

## **Appendix** 

## **A Additional Results** 

To complement the main evaluation, we present additional LongBench results that examine STARC’s effectiveness across a range of KV cache budgets (256, 512, and 2048), beyond the budget of 1024 used in the main results (Table 4, 5). These experiments illustrate how varying the KV cache budget affects model quality when serving long-context LLMs, and how STARC adapts its clustering-based mapping to improve efficiency while preserving model quality. 

## **B Artifact Appendix** 

## **B.1 Abstract** 

This artifact provides a complete workflow to reproduce the key results of STARC, including (1) the implementation of STARC’s selective token access with KV remapping and online clustering, (2) evaluation scripts to reproduce accuracy results on LongBench and RULER, and perplexity results on 

PG-19, and (3) the simulator setup to reproduce the systemlevel performance/energy results on GPU–PIM platforms based on the AttAcc simulator (Ramulator-based). 

## **B.2 Artifact check-list (meta-information)** 

- **Algorithm:** The STARC algorithm, which enables efficient long-context LLM inference by selectively accessing and remapping KV cache entries via online clustering under a fixed KV-cache budget. 

- **Program:** The STARC artifact running public long-context benchmarks: LongBench (16 datasets) and RULER (13 datasets). 

- **Model:** LongChat-7B-v1.5-32K; LLaMA-3.1-8B-Instruct; Mistral7B-Instruct-v0.3, all of which are publicly available and can be downloaded from Huggingface. 

- **Dataset:** LongBench (16 datasets; e.g., HotpotQA, QASPER, GovReport, _etc._ ); PG-19; RULER (13 datasets; e.g., NIAH Single, Multi-key NIAH, Multi-value NIAH, _etc._ ), all of which are publicly available and can be downloaded from Huggingface. 

- **Run-time environment:** Linux; Python 3.10; CUDA 12.8. 

- **Hardware:** See B.3.2. 

- **Metrics:** LongBench task scores; PG-19 perplexity; RULER task scores; System metrics such as latency and energy. 

- **Output:** Key results of our paper, including LongBench/RULER scores, PG-19 perplexity, and system-level performance and energy metrics with breakdowns. 

- **Experiments:** See B.5 

- **How much disk space required (approximately)?:** Approximately 80 GB in total. 

- **How much time is needed to prepare workflow (approximately)?:** 20 minutes. 

- **How much time is needed to complete experiments (approximately)?:** Excluding the additional results reported in 

**Table 4.** LongBench results for STARC and baseline sparsity methods (KV cache budget: 256 tokens). 

||Single-Document QA<br>NrtvQA<br>Qasper<br>MF-en|Multi-Document QA<br>HotpotQA<br>2WikiMQA<br>Musique|Summarization<br>GovReport<br>QMSum<br>MultiNews|Few-Shot Learning<br>TREC<br>TriviaQA<br>SAMSum|Synthetic<br>PCount<br>PRe|Code<br>Lcc<br>RB-P<br>Avg.|
|---|---|---|---|---|---|---|
||||**KV Budget: 256**||||
|**LongChat**|||||||
|Full KV<br>19.51<br>25.98<br>**43.80**<br>STARC<br>18.82<br>28.35<br>34.79<br>SparQ<br>**19.87**<br>**30.77**<br>40.71<br>InfniGen<br>13.68<br>27.47<br>36.05<br>Quest<br>10.49<br>26.47<br>34.90||31.94<br>23.20<br>11.38<br>**34.41**<br>18.64<br>8.10<br>31.70<br>20.93<br>**12.89**<br>27.86<br>20.41<br>7.75<br>20.04<br>**24.23**<br>12.53|**31.77**<br>21.66<br>26.06<br>30.50<br>21.74<br>24.64<br>30.93<br>**22.80**<br>**26.38**<br>26.27<br>20.49<br>24.97<br>21.59<br>20.48<br>25.29|**66.00**<br>82.00<br>20.79<br>62.00<br>81.01<br>24.17<br>64.00<br>**85.17**<br>31.37<br>62.00<br>77.22<br>**32.47**<br>56.00<br>63.80<br>22.62|**2.00**<br>30.00<br>**2.00**<br>**32.00**<br>0.50<br>31.50<br>**2.00**<br>18.00<br>**2.00**<br>28.00|53.86<br>48.68<br>33.66<br>55.56<br>45.00<br>32.61<br>**55.63**<br>**55.58**<br>35.05<br>52.70<br>50.28<br>31.23<br>47.86<br>38.58<br>28.43|
|**Mistral**|||||||
|Full KV<br>23.94<br>40.07<br>**57.58**<br>STARC<br>20.19<br>35.71<br>56.50<br>SparQ<br>**27.12**<br>**40.87**<br>53.94<br>InfniGen<br>19.52<br>37.95<br>54.54<br>Quest<br>16.81<br>30.88<br>36.99||49.10<br>36.71<br>22.27<br>44.43<br>**45.85**<br>20.32<br>**49.32**<br>39.51<br>**23.97**<br>42.28<br>38.98<br>10.34<br>35.62<br>27.66<br>10.12|**35.66**<br>**25.77**<br>26.80<br>34.06<br>24.06<br>26.54<br>35.31<br>25.14<br>**27.48**<br>31.14<br>22.82<br>27.21<br>29.18<br>21.11<br>26.04|**80.00**<br>87.67<br>47.35<br>68.00<br>87.11<br>**48.97**<br>73.00<br>**88.78**<br>47.28<br>74.00<br>83.45<br>47.70<br>66.00<br>78.39<br>37.84|4.00<br>98.00<br>**6.00**<br>88.00<br>4.50<br>**99.50**<br>4.00<br>90.00<br>4.89<br>83.50|58.98<br>56.36<br>46.89<br>**61.94**<br>57.60<br>45.33<br>61.56<br>**63.15**<br>47.53<br>61.70<br>52.34<br>43.62<br>57.22<br>43.98<br>37.89|
|**Llama-3.1**|||||||
|Full KV<br>27.02<br>13.98<br>28.04<br>STARC<br>**30.84**<br>12.91<br>26.42<br>SparQ<br>29.70<br>12.35<br>26.97<br>InfniGen<br>21.86<br>**16.53**<br>**29.63**<br>Quest<br>8.68<br>9.90<br>18.18||18.30<br>17.45<br>13.01<br>**21.88**<br>**18.34**<br>**13.48**<br>17.69<br>15.31<br>11.40<br>21.47<br>17.76<br>5.36<br>12.19<br>9.48<br>3.02|**35.83**<br>**23.66**<br>25.91<br>34.96<br>22.18<br>25.46<br>33.89<br>23.38<br>**27.00**<br>32.38<br>22.70<br>25.50<br>25.33<br>18.36<br>23.50|**74.00**<br>89.77<br>44.56<br>66.00<br>86.71<br>**44.94**<br>70.00<br>**92.19**<br>44.58<br>68.00<br>86.40<br>44.58<br>44.00<br>73.23<br>31.53|3.92<br>**97.50**<br>**12.00**<br>94.33<br>6.90<br>**97.50**<br>7.25<br>96.00<br>3.55<br>83.00|63.30<br>55.06<br>39.46<br>65.52<br>57.32<br>39.58<br>64.55<br>**60.79**<br>39.64<br>**67.36**<br>55.38<br>38.64<br>51.90<br>46.52<br>28.90|



1875 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

**Table 5.** LongBench results for STARC and baseline sparsity methods (KV cache budget: 512 and 2048 tokens). 

||Single-Document QA<br>NrtvQA<br>Qasper<br>MF-en|Multi-Document QA<br>HotpotQA<br>2WikiMQA<br>Musique|Summarization<br>GovReport<br>QMSum<br>MultiNews|Few-Shot Learning<br>TREC<br>TriviaQA<br>SAMSum|Synthetic<br>PCount<br>PRe|Code<br>Lcc<br>RB-P<br>Avg.|
|---|---|---|---|---|---|---|
||||**KV Budget: 512**||||
|**LongChat**|||||||
|Full KV<br>**19.51**<br>25.98<br>**43.80**<br>STARC<br>17.59<br>**29.86**<br>39.44<br>SparQ<br>19.20<br>29.18<br>40.81<br>InfniGen<br>16.37<br>25.37<br>38.10<br>Quest<br>13.39<br>28.08<br>40.90||31.94<br>23.20<br>11.38<br>**33.92**<br>18.70<br>10.21<br>32.27<br>22.30<br>13.43<br>28.48<br>18.15<br>**13.52**<br>25.34<br>**24.59**<br>7.59|**31.77**<br>21.66<br>26.06<br>30.46<br>20.49<br>25.11<br>30.81<br>**22.81**<br>**26.29**<br>28.71<br>21.10<br>25.06<br>27.82<br>21.48<br>25.39|**66.00**<br>82.00<br>20.79<br>64.00<br>79.81<br>22.48<br>64.50<br>**84.70**<br>29.05<br>64.00<br>79.03<br>**31.93**<br>**66.00**<br>76.67<br>21.94|**2.00**<br>30.00<br>**2.00**<br>30.00<br>0.00<br>30.00<br>0.00<br>28.00<br>0.00<br>**36.00**|53.86<br>48.68<br>33.66<br>**57.60**<br>48.38<br>33.13<br>55.11<br>**55.70**<br>34.76<br>52.60<br>53.42<br>32.74<br>53.36<br>45.08<br>32.10|
|**Mistral**|||||||
|Full KV<br>23.94<br>40.07<br>57.58<br>STARC<br>21.49<br>37.26<br>**58.73**<br>SparQ<br>**29.00**<br>**40.09**<br>53.70<br>InfniGen<br>22.76<br>36.82<br>58.67<br>Quest<br>18.39<br>33.14<br>45.93||49.10<br>36.71<br>22.27<br>47.18<br>**40.15**<br>23.68<br>**50.43**<br>37.75<br>**26.49**<br>49.17<br>31.64<br>15.34<br>41.79<br>33.64<br>18.21|**35.66**<br>**25.77**<br>26.80<br>34.32<br>23.66<br>26.64<br>34.23<br>25.68<br>**27.50**<br>33.80<br>23.87<br>26.51<br>32.57<br>22.77<br>26.45|**80.00**<br>87.67<br>47.35<br>74.00<br>87.67<br>48.38<br>74.00<br>**89.07**<br>47.35<br>78.00<br>83.67<br>**49.67**<br>64.00<br>84.50<br>41.63|4.00<br>98.00<br>4.00<br>94.00<br>5.00<br>**99.50**<br>2.00<br>96.00<br>**6.50**<br>92.67|58.98<br>56.36<br>46.89<br>62.18<br>58.90<br>46.39<br>60.72<br>**62.07**<br>47.66<br>**62.88**<br>58.04<br>45.55<br>59.92<br>49.84<br>42.00|
|**Llama-3.1**|||||||
|Full KV<br>27.02<br>13.98<br>28.04<br>STARC<br>**31.78**<br>13.06<br>**28.77**<br>SparQ<br>30.30<br>13.30<br>26.19<br>InfniGen<br>23.36<br>**16.90**<br>27.18<br>Quest<br>15.57<br>10.77<br>21.82||18.30<br>17.45<br>13.01<br>18.49<br>**18.58**<br>**14.24**<br>17.90<br>16.12<br>10.43<br>**22.17**<br>18.30<br>8.76<br>12.42<br>13.25<br>5.93|**35.83**<br>23.66<br>25.91<br>34.33<br>22.65<br>25.80<br>34.11<br>**23.83**<br>**27.17**<br>33.69<br>22.79<br>25.85<br>29.48<br>22.05<br>26.65|**74.00**<br>89.77<br>44.56<br>70.00<br>88.57<br>44.26<br>70.50<br>**91.97**<br>43.80<br>66.00<br>89.90<br>**45.10**<br>60.00<br>78.87<br>37.44|3.92<br>97.50<br>4.67<br>95.83<br>**8.29**<br>**98.08**<br>6.67<br>98.07<br>2.54<br>90.52|63.30<br>55.06<br>39.46<br>63.56<br>60.08<br>39.67<br>64.43<br>**61.34**<br>39.86<br>**66.46**<br>50.72<br>38.87<br>61.18<br>53.94<br>33.90|
||||**KV Budget: 2048**||||
|**LongChat**|||||||
|Full KV<br>19.51<br>25.98<br>43.80<br>STARC<br>17.70<br>28.27<br>41.15<br>SparQ<br>**20.01**<br>28.48<br>42.21<br>InfniGen<br>15.76<br>30.35<br>40.52<br>Quest<br>14.93<br>**31.48**<br>**45.33**||31.94<br>23.20<br>11.38<br>**33.75**<br>23.28<br>11.21<br>31.02<br>**23.53**<br>12.68<br>31.81<br>20.05<br>9.00<br>31.60<br>19.70<br>**12.93**|**31.77**<br>21.66<br>26.06<br>30.97<br>**23.13**<br>26.50<br>31.06<br>23.07<br>**26.69**<br>29.82<br>22.13<br>26.00<br>30.83<br>22.07<br>25.61|**66.00**<br>82.00<br>20.79<br>63.00<br>81.53<br>20.80<br>65.00<br>**84.41**<br>24.61<br>62.00<br>81.78<br>**25.94**<br>62.00<br>81.33<br>20.35|**2.00**<br>30.00<br>1.00<br>31.00<br>0.00<br>30.00<br>0.00<br>**36.00**<br>**2.00**<br>30.00|53.86<br>48.68<br>33.66<br>54.15<br>50.92<br>33.65<br>52.86<br>**55.69**<br>34.46<br>**57.62**<br>50.38<br>33.70<br>55.68<br>49.92<br>33.49|
|**Mistral**|||||||
|Full KV<br>23.94<br>40.07<br>57.58<br>STARC<br>28.71<br>**43.73**<br>54.06<br>SparQ<br>**29.58**<br>40.25<br>53.37<br>InfniGen<br>25.34<br>39.30<br>**59.51**<br>Quest<br>23.48<br>40.55<br>58.73||49.10<br>36.71<br>22.27<br>48.62<br>37.87<br>23.36<br>**51.01**<br>37.94<br>**27.22**<br>50.20<br>**41.79**<br>18.54<br>48.94<br>37.63<br>25.41|**35.66**<br>**25.77**<br>26.80<br>34.82<br>25.75<br>**27.87**<br>34.45<br>25.68<br>27.76<br>34.83<br>24.68<br>26.85<br>32.79<br>24.07<br>27.28|**80.00**<br>87.67<br>47.35<br>72.00<br>85.76<br>**47.87**<br>74.50<br>**89.06**<br>47.01<br>78.00<br>87.67<br>47.38<br>70.00<br>88.33<br>47.07|4.00<br>98.00<br>**9.00**<br>**100.00**<br>5.00<br>99.00<br>2.00<br>96.00<br>6.00<br>98.00|58.98<br>56.36<br>46.89<br>59.64<br>57.91<br>47.31<br>**59.76**<br>**62.04**<br>47.73<br>58.96<br>59.46<br>46.91<br>57.86<br>60.54<br>46.67|
|**Llama-3.1**|||||||
|Full KV<br>27.02<br>**13.98**<br>28.04<br>STARC<br>**30.61**<br>13.88<br>27.94<br>SparQ<br>29.76<br>13.06<br>26.61<br>InfniGen<br>27.98<br>13.33<br>**32.01**<br>Quest<br>24.41<br>13.34<br>23.39||18.30<br>17.45<br>**13.01**<br>**20.85**<br>**19.62**<br>11.53<br>17.30<br>16.85<br>11.26<br>19.49<br>18.79<br>12.86<br>15.97<br>15.59<br>10.59|**35.83**<br>**23.66**<br>25.91<br>34.56<br>22.75<br>26.30<br>34.02<br>23.50<br>**26.69**<br>35.45<br>23.10<br>26.66<br>35.03<br>23.33<br>25.58|**74.00**<br>89.77<br>44.56<br>72.00<br>88.57<br>**45.54**<br>71.00<br>91.48<br>43.82<br>72.00<br>89.81<br>44.63<br>**74.00**<br>**92.60**<br>45.23|3.92<br>97.50<br>2.92<br>**99.00**<br>6.37<br>98.01<br>**7.00**<br>96.67<br>5.18<br>97.50|63.30<br>55.06<br>39.46<br>62.66<br>55.54<br>39.64<br>**63.38**<br>**59.78**<br>39.56<br>61.12<br>55.74<br>39.79<br>59.44<br>56.52<br>38.61|



the appendix, the model accuracy experiments take approximately 12 hours. In addition, the system-level performance experiments take approximately 24 hours. 

- **Publicly available?:** https://doi.org/10.5281/zenodo.18050293 

- **Code licenses (if publicly available)?:** MIT license. 

## **B.3 Description** 

**B.3.1 How to access.** The STARC algorithm, benchmarks, and scripts are available at GitHub: EPIC-RPI/STARC 

## **B.3.2 Hardware dependencies.** 

- LLM accuracy evaluation (LongBench / PG-19 / RULER): Compatible with commonly used NVIDIA GPUs. We recommend NVIDIA H100 or L40 with sufficient GPU memory (e.g., at least 48 GB per GPU). 

- System-level simulation: CPU-only execution is sufficient. Experiments in the paper were conducted on a dual-socket AMD EPYC 9334 system with 64 CPU cores in total (2×32 cores). 

**B.3.3 Software dependencies.** The software is performed using Python 3.10, and CUDA version 12.8. The dependent Python packages can be found in the pyproject.toml file. 

## **B.4 Installation** 

- **Code access.** First, please access the code by: 

git clone --recurse -submodules https :// github.com/EPIC -RPI/STARC cd STARC 

1876 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

**E3: RULER (32K context) on LLaMA-3.1-8B-Instruct.** The RULER testing data are already included in the STARC/ruler directory. To reproduce the RULER results under a 32K context length, please run: 

- **Environment setup.** To better reproduce the results and avoid potential conflicts, we recommend using Python 3.10 and CUDA 12.8. We provide scripts for the recommended environment setup. Please follow the instructions to create the conda environment and install the STARC packages: 

cd <Your Path >/ STARC/scripts/ sh RULER.sh 

conda create -yn STARC python =3.10 conda activate STARC 

**E4: GPU–PIM system simulation.** The system-level simulation experiments are conducted using the AttAcc-based simulator. To reproduce the results for _full attention_ , please run the following command: 

pip install ninja ==1.11.1.1 packaging pip install -e . pip install flash -attn ==2.3.0 --no -build -isolation conda install -c conda -forge cupy conda install numpy scikit -learn conda install cmake 

   - python main.py --system dgx -attacc --gpu H100 --ngpu 8 --model Mistral -7B \ 

      - --lin 2048 --lout 32000 --batch 16 --pim bank \ 

      - --powerlimit --ffopt --pipeopt 

- **PIM system simulator setup.** Next is the setup for the PIM system simulator. In this artifact, we mainly build on the AttAcc simulator: 

To reproduce the results for configurations with _sparse attention methods_ , please run: 

cd simulator_starc 

- python main.py --system dgx -attacc --gpu H100 --ngpu 8 --model Mistral -7B \ 

git submodule update --init --recursive 

   - --lin 2048 --lout 32000 --batch 16 --pim bank \ 

- **Build Ramulator2.** 

bash set_pim_ramulator.sh cd ramulator2 

- --powerlimit --ffopt --pipeopt \ 

- --sparsity --kv_budget_table kv_budget_Mistral_STARC.txt 

mkdir build cd build cmake .. 

Different sparse attention methods and models use different .txt files specified by the –kv_budget_table option. These files are derived from the attention masks produced by each method at each decoding step in real inference tasks (e.g., LongBench), and map them to the row-level granularity of the PIM architecture, where each DRAM row activation fetches 16 key/value vectors in parallel. They define how many memory rows are activated at each decoding step and are used to guide the simulator accordingly. Detailed explanations are provided in the script comments and the GitHub repository README.md. 

-DCMAKE_POLICY_VERSION_MINIMUM =3.5 make -j cp ramulator2 ../ ramulator2 cd ../../ 

## **B.5 Experiment workflow** 

This section describes how to reproduce the key results reported in the paper. 

**E1: LongBench accuracy.** To reproduce the LongBench accuracy results, please run: 

## **B.6 Evaluation and expected results** 

cd <Your Path >/STARC/scripts/ sh longbench.sh 

- **Model accuracy experiments:** For **LongBench** , the evaluation generates a corresponding .jsonl file for each model and each task. These files contain the ground-truth answers and model predictions. The final results are summarized in result.json. For **RULER** , evaluation results are printed directly to the terminal. 

If you want to evaluate more models, first you can find the corresponding model paths in: 

STARC/evaluation/LongBench/config/model2path.json 

- **PG-19 perplexity:** A log_PG19.txt file is generated to record the evolution of perplexity during evaluation. 

By replacing the model name in longbench.sh, you can evaluate STARC under different models reported in the paper. 

- **Simulation experiments:** The simulator produces an output.csv file that records the breakdown of endto-end latency and energy consumption. 

**E2: PG-19 perplexity.** To reproduce the perplexity results on PG-19, please run: 

cd <Your Path >/STARC/scripts/ sh ppl_eval.sh 

## **B.7 Methodology** 

Submission, reviewing and badging methodology: 

1877 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Zehao Fan et al. 

- https://www.acm.org/publications/policies/artifact-reviewand-badging-current 

- https://cTuning.org/ae 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ (2023). 

- [2] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics_ 

   - _(Volume 1: Long Papers)_ . Association for Computational Linguistics, Bangkok, Thailand, 3119–3137. doi:10.18653/v1/2024.acl-long.172 

- [3] Payman Behnam, Yaosheng Fu, Ritchie Zhao, Po-An Tsai, Zhiding Yu, and Alexey Tumanov. 2025. RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression. _arXiv preprint arXiv:2502.14051_ (2025). 

- [4] Benjamin Y Cho, Jeageun Jung, and Mattan Erez. 2021. Accelerating bandwidth-bound deep learning inference with main-memory accelerators. In _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis_ . 1–14. 

- [5] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. _arXiv preprint arXiv:2105.03011_ (2021). 

- [6] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. _arXiv e-prints_ (2024), arXiv–2407. 

- [7] Alexander R Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir R Radev. 2019. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. _arXiv preprint arXiv:1906.01749_ (2019). 

- [8] Fei Gao, Georgios Tziantzioulis, and David Wentzlaff. 2019. Computedram: In-memory compute using off-the-shelf drams. In _Proceedings of the 52nd annual IEEE/ACM international symposium on microarchitecture_ . 100–113. 

- [9] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2023. Model tells you what to discard: Adaptive kv cache compression for llms. _arXiv preprint arXiv:2310.01801_ (2023). 

- [10] Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, and Poulami Das. 2025. Dialogue without limits: Constant-sized kv caches for extended responses in llms. _arXiv preprint arXiv:2503.00979_ (2025). 

- [11] Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. 2019. SAMSum corpus: A human-annotated dialogue dataset for abstractive summarization. _arXiv preprint arXiv:1911.12237_ (2019). 

- [12] Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Julian McAuley. 2023. Longcoder: A long-range pre-trained language model for code completion. In _International Conference on Machine Learning_ . PMLR, 12098–12107. 

- [13] Mingxuan He, Choungki Song, Ilkon Kim, Chunseok Jeong, Seho Kim, Il Park, Mithuna Thottethodi, and TN Vijaykumar. 2020. Newton: A DRAM-maker’s accelerator-in-memory (AiM) architecture for machine learning. In _2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 372–385. 

- [14] Yintao He, Haiyu Mao, Christina Giannoula, Mohammad Sadrosadati, Juan Gómez-Luna, Huawei Li, Xiaowei Li, Ying Wang, and Onur Mutlu. 2025. PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System. _arXiv preprint arXiv:2502.15470_ (2025). 

- [15] Guseul Heo, Sangyeop Lee, Jaehong Cho, Hyunmin Choi, Sanghyeon Lee, Hyungkyu Ham, Gwangsun Kim, Divya Mahajan, and Jongse Park. 2024. Neupims: Npu-pim heterogeneous acceleration for batched llm inferencing. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ . 722–737. 

- [16] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Monishwaran Maheswaran, June Paik, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. 2024. Squeezed attention: Accelerating long context length llm inference. _arXiv preprint arXiv:2411.09688_ (2024). 

- [17] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the Real Context Size of Your Long-Context Language Models? _arXiv preprint arXiv:2404.06654_ (2024). 

- [18] Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. 2021. Efficient attentions for long document summarization. _arXiv preprint arXiv:2104.02112_ (2021). 

- [19] Bongjoon Hyun, Taehun Kim, Dongjae Lee, and Minsoo Rhu. 2024. Pathfinding future pim architectures by demystifying a commercial pim technology. In _2024 IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ . IEEE, 263–279. 

- [20] Mohsen Imani, Saransh Gupta, Yeseong Kim, and Tajana Rosing. 2019. Floatpim: In-memory acceleration of deep neural network training with high precision. In _Proceedings of the 46th International Symposium on Computer Architecture_ . 802–815. 

- [21] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https://arxiv.org/abs/2310.06825 

- [22] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. _arXiv preprint arXiv:1705.03551_ (2017). 

- [23] Tomáš Kočisky,` Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor Melis, and Edward Grefenstette. 2018. The narrativeqa reading comprehension challenge. _Transactions of the Association for Computational Linguistics_ 6 (2018), 317–328. 

- [24] Hyucksung Kwon, Kyungmo Koo, Janghyeon Kim, Woongkyu Lee, Minjae Lee, Hyungdeok Lee, Yousub Jung, Jaehan Park, Yosub Song, Byeongsu Yang, et al. 2024. LoL-PIM: Long-Context LLM Decoding with Scalable DRAM-PIM System. _arXiv preprint arXiv:2412.20166_ (2024). 

- [25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles_ . 611–626. 

- [26] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. {InfiniGen}: Efficient generative inference of large language models with dynamic {KV} cache management. In _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)_ . 155–172. 

- [27] Cong Li, Yihan Yin, Xintong Wu, Jingchen Zhu, Zhutianya Gao, Dimin Niu, Qiang Wu, Xin Si, Yuan Xie, Chen Zhang, et al. 2025. H2LLM: Hardware-Dataflow Co-Exploration for Heterogeneous HybridBonding-based Low-Batch LLM Inference. In _Proceedings of the 52nd Annual International Symposium on Computer Architecture_ . 194–210. 

- [28] Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. 2023. How Long Can Open-Source LLMs Truly Promise on Context Length? https://lmsys.org/blog/2023-06-29-longchat 

- [29] Shuang Li, Xavier Puig, Chris Paxton, Yilun Du, Clinton Wang, Linxi Fan, Tao Chen, De-An Huang, Ekin Akyürek, Anima Anandkumar, et al. 2022. Pre-trained language models for interactive decisionmaking. _Advances in Neural Information Processing Systems_ 35 (2022), 

1878 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

STARC 

31199–31212. 

- [30] Xin Li and Dan Roth. 2002. Learning question classifiers. In _COLING 2002: The 19th International Conference on Computational Linguistics_ . 

- [31] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. _arXiv preprint arXiv:2412.19437_ (2024). 

- [32] Guangda Liu, Chengwei Li, Jieru Zhao, Chenqi Zhang, and Minyi Guo. 2024. Clusterkv: Manipulating llm kv cache in semantic space for recallable compression. _arXiv preprint arXiv:2412.03213_ (2024). 

- [33] Lian Liu, Shixin Zhao, Bing Li, Haimeng Ren, Zhaohui Xu, Mengdi Wang, Xiaowei Li, Yinhe Han, and Ying Wang. 2025. Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM. In _2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, 1751–1765. 

- [34] Tianyang Liu, Canwen Xu, and Julian McAuley. 2023. Repobench: Benchmarking repository-level code auto-completion systems. _arXiv preprint arXiv:2306.03091_ (2023). 

- [35] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. _Advances in Neural Information Processing Systems_ 36 (2023), 52342–52364. 

- [36] Haocong Luo, Yahya Can Tuğrul, F Nisa Bostancı, Ataberk Olgun, A Giray Yağlıkçı, and Onur Mutlu. 2023. Ramulator 2.0: A modern, modular, and extensible dram simulator. _IEEE Computer Architecture Letters_ 23, 1 (2023), 112–116. 

- [37] Jinendra Malekar, Peyton Chandarana, Md Hasibul Amin, Mohammed E Elbtity, and Ramtin Zand. 2025. PIM-LLM: A HighThroughput Hybrid PIM Architecture for 1-bit LLMs. _arXiv preprint arXiv:2504.01994_ (2025). 

- [38] Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. 2024. Using an llm to help with code understanding. In _Proceedings of the IEEE/ACM 46th International Conference on Software Engineering_ . 1–13. 

- [39] Geraldo F Oliveira, Juan Gómez-Luna, Saugata Ghose, Amirali Boroumand, and Onur Mutlu. 2022. Accelerating neural network inference with processing-in-DRAM: from the edge to the cloud. _IEEE Micro_ 42, 6 (2022), 25–38. 

- [40] Jaehyun Park, Jaewan Choi, Kwanhee Kyung, Michael Jaemin Kim, Yongsuk Kwon, Nam Sung Kim, and Jung Ho Ahn. 2024. AttAcc! Unleashing the power of PIM for batched transformer-based generative model inference. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ . 103–119. 

- [41] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently scaling transformer inference. _Proceedings of Machine Learning and Systems_ 5 (2023), 606–624. 

- [42] Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, and Timothy P Lillicrap. 2019. Compressive transformers for long-range sequence modelling. _arXiv preprint arXiv:1911.05507_ (2019). 

- [43] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of machine learning research_ 21, 140 (2020), 1–67. 

- [44] Luka Ribar, Ivan Chelombiev, Luke Hudlass-Galley, Charlie Blake, Carlo Luschi, and Douglas Orr. 2023. Sparq attention: Bandwidthefficient llm inference. _arXiv preprint arXiv:2312.04985_ (2023). 

- [45] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. 2023. Code llama: Open foundation models for code. _arXiv preprint arXiv:2308.12950_ (2023). 

- [46] Minseok Seo, Xuan Truong Nguyen, Seok Joong Hwang, Yongkee Kwon, Guhyun Kim, Chanwook Park, Ilkon Kim, Jaehan Park, Jeongbin Kim, Woojae Shin, et al. 2024. Ianus: Integrated accelerator based on npu-pim unified memory system. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ . 545–560. 

- [47] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. _Advances in Neural Information Processing Systems_ 36 (2023), 8634–8652. 

- [48] Alexey Svyatkovskiy, Ying Zhao, Shengyu Fu, and Neel Sundaresan. 2019. Pythia: Ai-assisted code completion system. In _Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining_ . 2727–2735. 

- [49] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. 2024. Quest: Query-aware sparsity for efficient longcontext llm inference. _arXiv preprint arXiv:2406.10774_ (2024). 

- [50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_ (2023). 

- [51] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. _Transactions of the Association for Computational Linguistics_ 10 (2022), 539–554. 

- [52] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. _arXiv preprint arXiv:2309.17453_ (2023). 

- [53] Yi Yang, Wen-tau Yih, and Christopher Meek. 2015. Wikiqa: A challenge dataset for open-domain question answering. In _Proceedings of the 2015 conference on empirical methods in natural language processing_ . 2013–2018. 

- [54] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. _arXiv preprint arXiv:1809.09600_ (2018). 

- [55] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. _Advances in neural information processing systems_ 36 (2023), 11809–11822. 

- [56] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. 2023. H2o: Heavy-hitter oracle for efficient generative inference of large language models. _Advances in Neural Information Processing Systems_ 36 (2023), 34661–34710. 

- [57] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric P Xing, et al. 2023. Lmsys-chat-1m: A large-scale real-world llm conversation dataset. _arXiv preprint arXiv:2309.11998_ (2023). 

- [58] Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, et al. 2021. QMSum: A new benchmark for query-based multi-domain meeting summarization. _arXiv preprint arXiv:2104.05938_ (2021). 

- [59] Minxuan Zhou, Weihong Xu, Jaeyoung Kang, and Tajana Rosing. 2022. TransPIM: A memory-based acceleration via software-hardware codesign for transformer. In _2022 IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ . IEEE, 1071–1085. 

- [60] Yuxuan Zhu, Ali Falahati, David H Yang, and Mohammad Mohammadi Amiri. 2025. SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching. _arXiv preprint arXiv:2504.00970_ (2025). 

- [61] Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2024. Toolqa: A dataset for llm question answering with external tools. _Advances in Neural Information Processing Systems_ 36 (2024). 

1879 

