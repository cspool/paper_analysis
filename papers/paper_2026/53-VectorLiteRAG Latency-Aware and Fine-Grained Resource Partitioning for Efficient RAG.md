2026 IEEE International Symposium on High-Performance Computer Architecture (HPCA) 

# VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG 

Junkyum Kim and Divya Mahajan 

Georgia Institute of Technology _{_ jun-kyum.kim, divya.mahajan _}_ @gatech.edu 

_**Abstract**_ **—Retrieval-Augmented Generation leverages vector similarity search to enhance large language models with upto-date, external knowledge, enabling accurate and reliable responses. While CPU-only vector search incurs high latency on large, high-dimensional indices, co-locating the retriever and the LLM on the GPU leads to resource sharing that can create resource contention. Specifically, vector search is memory and I/O intensive, placing it in direct conflict with LLM inference, which demands memory for KV cache and compute for higher throughput. We present VECTORLITERAG, a latency-aware RAG serving system that explicitly orchestrates data placement and execution across retrieval and inference to meet strict endto-end SLOs. VECTORLITERAG is driven by access-pattern analysis and performance estimation to regulate how retrieval variability can be mitigated and managed in the system with LLM inference, enabling SLO-compliant execution under skewed and dynamic workloads. By jointly modeling search latency and query hit-rate distributions, VECTORLITERAG identifies an optimal index partitioning point across CPU and GPU that minimizes contention and stabilizes batching behavior, thereby maximizing sustained throughput under skewed access patterns. A low-overhead online index update mechanism allows VECTORLITERAG to continuously adapt to evolving request distributions, preserving batching efficiency and throughput as access patterns evolve. Our evaluations demonstrate that VECTORLITERAG consistently expands the range of SLO-compliant request rate across all tested configurations. Without increasing the generation latency or requiring additional hardware, VECTORLITERAG outperforms both naive and existing alternative frameworks, improving attainable SLO-bound throughput by up to 1.5** _×_ **.** 

## I. INTRODUCTION 

Retrieval-Augmented Generation (RAG) is a powerful system in natural language processing, particularly for domainspecific question answering and information retrieval tasks [7], [8], [10], [22], [33]. Its key strength lies in combining parametric memory, encoded in the weights of a large language model, with non-parametric memory retrieved from an external knowledge corpus. Although parametric memory provides strong generalization, it is expensive to train and difficult to update. To mitigate this, RAG pipelines first perform similarity search using approximate nearest neighbor search (ANNS) algorithms to retrieve relevant documents from a large database. The retrieved documents are then fed into the LLM’s context to generate up-to-date and reliable responses. 

RAG frameworks [20], [24], [38] typically adopt heterogeneous hardware configurations, where vector retrieval is executed on CPUs and LLM generation is served by GPUs. This is driven by the system characteristics: LLM inference requires 

**==> picture [253 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
CPU MEM GPU MEM<br>Vector<br>Index<br>KV Cache<br>𝓦𝓱𝓪𝓽 𝓲𝓼 𝓽𝓱𝓮<br>𝓟𝓪𝓻𝓲𝓼 𝓲𝓼 …<br>𝓬𝓪𝓹𝓲𝓽𝓪𝓵 …<br>User Embedding Similarity Large Language Output<br>Query Model Search Models<br>𝓦𝓱𝓪𝓽 𝓲𝓼 𝓽𝓱𝓮  User<br>Knowledge Base Relevant 𝓬𝓪𝓹𝓲𝓽𝓪𝓵 … Query<br>Information<br>**----- End of picture text -----**<br>


Fig. 1. End-to-end pipeline of a RAG system, where the input query is indexed into the vector database stored in memory, while the knowledge corpus resides in storage. The LLM prefill and decode execute on the GPU. 

massive matrix multiplications and benefits significantly from GPU acceleration, whereas retrieval has traditionally been seen as a lighter task suited for CPUs. Offloading retrieval to CPUs allows GPUs to be dedicated to the more compute-intensive generation phase. CPU-based vector search may be sufficient for small vector databases, however, as the dimensionality of the embeddings and the size of the dataset grow, retrieval becomes increasingly compute- and memory- bound. CPUs, with limited parallelism, narrower vector units, and lower memory bandwidth, struggle to handle high-throughput similarity search at scale. 

This latency imbalance creates a bottlenecked pipeline where the relatively slow CPU-based retrieval delays the GPUaccelerated generation phase, reducing the benefits of fast LLM inference and degrading overall system responsiveness. In our observations, CPU-based retrieval can take up to twice as long as the LLM prefill phase, increasing the total Time-toFirst-Token (TTFT) from 197ms to 606ms when using a large database with 128M vectors, compared to a language model (Llama3-8B) operating without retrieval. 

Although the retrieval operation is computationally lighter than the generation phase, it can still benefit significantly from GPU acceleration for two reasons: (1) GPUs feature wide and powerful vector units that enable highly parallelized distance computations, offering superior performance for similarity calculations on long embedding vectors. (2) The retrieval process involves scanning intermediate distance tensors to 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

identify the closest data points in the vector space. These operations are typically implemented as memory lookups a task where GPUs outperform CPUs due to their vectorized memory access and higher I/O. 

In addition to compute and bandwidth demands, vector retrieval introduces significant memory pressure. To reduce memory footprint and speed up the search process, vector databases are commonly compressed into vector indexes using quantization techniques such as product quantization (PQ) [13]. Nevertheless, even after compression, vector indexes still occupy significant memory space, often exceeding the memory capacity of a GPU. Furthermore, intermediate data structures such as distances between cluster centroids and queries consume additional memory. 

These compute and memory pressures create a resource tension between the retrieval and generation stages, especially as the vector database grows and CPU-based search fails to meet strict latency requirements. GPU memory is already constrained, with most of it reserved for model weights and KV cache for the LLM. Naively sharding the vector index across all GPUs can lead to memory contention and reduced overall throughput. Alternatively, assigning a disaggregated GPU for retrieval can prevent direct interference between stages, but degrades overall system throughput by reducing the number of available LLM instances, in particular when models require multiple GPUs, enforcing rigid allocation schemes. 

Motivated by these challenges, this work explores a holistic approach to optimizing distributed RAG pipelines through joint resource allocation between vector search and LLM generation. We present VECTORLITERAG, a system that partitions the vector index between GPU and CPU-based on query access patterns and LLM deployment configurations, aiming to maximize throughput while meeting latency targets by exploiting the compute power of GPUs across both stages of the RAG pipeline. By analytically modeling similarity search latency, we determine the smallest index portion that needs to be placed on the GPU to satisfy the latency requirement under a given system configuration. Accordingly, VECTORLITERAG offers a latency-aware, throughput-optimized solution that requires no additional hardware resources. This approach is grounded in two key insights: 

**Access-Skew-Aware Data Layout.** VECTORLITERAG leverages a key characteristic of Inverted File (IVF) based retrieval systems [46], that query accesses exhibit skew across clusters. To take advantage of this, VECTORLITERAG incorporates an analytical model that determines the optimal partitioning point and corresponding layout for a multi-GPU system. While the coarse quantizer and cold clusters remain on the CPU, a small subset of hot clusters are cached and distributed across GPUs. The system allocates just enough hot clusters to the GPUs, avoiding both oversubscription of GPU resources during retrieval. 

**Inter/Intra-Query Variance-Aware Routing.** When hot clusters are distributed across GPUs, hit rates vary both across queries (inter-query variance) and across device shards within a query (intra-query variance). Existing systems that enforce 

**==> picture [227 x 178] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 1 2 3 4<br>1<br>1 2.1 1.9 1.0 -4.4 0.0 1.9 4.1 -6.6<br>2 -7.2 4.5 -1.1 2.9 -1.9 8.5 7.0 1.4<br>C4(2) 3 5.3 -0.2 -2.1 5.6 5.4 2.3 2.1 2.8<br>X2 4 -8.7 1.3 0.1 7.7 6.0 -1.8 5.2 1.9<br>C1(2) X<br>X1<br>C1(2) Q[1] Q[2] Q[3] Q[4] Q[5] Q[6] Q[7] Q[8]<br>X3 C2(2) 1 2 3 4<br>1 Q[1:2] C[1,1] Q[3:4] C[1,2] Q[5:6] C[1,3] Q[7:8] C[1,4]<br>2 Q[1:2] C[2,1] Q[3:4] C[2,2] Q[5:6] C[2,3] Q[7:8] C[2,4]<br>3 Q[1:2] C[3,1] Q[3:4] C[3,2] Q[5:6] C[3,3] Q[7:8] C[3,4]<br>4 Q[1:2] C[4,1] Q[3:4] C[4,2] Q[5:6] C[4,3] Q[7:8] C[4,4]<br>3<br>X1 4 1 2 4<br>1 2 3 4<br>X2 2 3 1 4 1 Q[1:2] C[1,1] Q[3:4] C[1,2] Q[5:6] C[1,3] Q[7:8] C[1,4]<br>X3 1 1 1 3 2 Q[1:2] C[2,1] Q[3:4] C[2,2] Q[5:6] C[2,3] Q[7:8] C[2,4]<br>X…n 1 3 3 2 34 Q[1:2] C[3,1]Q[1:2] C[4,1] Q[3:4] C[3,2]Q[3:4] C[4,2] Q[5:6] C[3,3]Q[5:6] C[4,3] Q[7:8] C[3,4]Q[7:8] C[4,4]<br>Q . X1 = Q[1:2] C[4,1] + Q[3:4] C[1,2] + Q[5:6] C[2,3] + Q[7:8] C[4,4]<br>**----- End of picture text -----**<br>


Fig. 2. Three stages of vector search in IVF–based index: (1) coarse quantization to identify clusters most semantically similar to the query, (2) construction of a LUT containing partial distances between the query and codewords, and (3) scanning the LUT and re-ranking candidates from the selected clusters based on aggregated distances. 

fixed retrieval configurations across devices fail to account for this variability and often over-allocate GPU threads. VECTORLITERAG introduces query- and shard- aware routing to avoid such inefficiencies. After determining the most relevant clusters, it dispatches work to CPU or GPU based on their actual expected contribution. It also monitors per-query progress, forwarding early-finishing queries to reduce straggler-induced delays and improve batching efficiency. 

Our contributions are summarized as follows: 

- **Access-skew modeling and hit-rate estimation.** We characterize access skew in IVF-based retrieval systems and develop a hit-rate estimation method based on observed cluster access patterns. 

- **Analytical latency model and SLO-aware partitioning.** We construct a latency model that accounts for inter-query variance and use it to determine the optimal CPU-GPU index partitioning point that meets latency targets. 

- **Distributed retrieval pipeline.** We design a distributed retrieval pipeline that adaptively allocates search tasks across CPUs and GPUs by exploiting inter-device hit rate variance, improving efficiency and avoiding unnecessary GPU resource usage. 

## II. RETRIEVAL AUGMENTED GENERATION 

In a RAG system, user queries are first transformed into vector embeddings using embedding models [30], [34], [36], [42]. These embeddings capture the semantics of the input and enable similarity search by comparing query vectors to a vector database constructed from the knowledge corpus, typically encoded using the same embedding model. State-ofthe-art embedding models produce vectors of several thousand dimensions for higher quality, but this increased dimensionality raises the cost of distance computations. 

Since exhaustive pairwise search is computationally infeasible at scale, large vector retrieval relies on approximate nearest 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
IVF IVF-FS CQ LUT Cmp LUT Scan<br>0.3<br>1.0<br>0.2<br>0.5<br>0.1<br>0.0 0.0<br>4 16 2 8<br>Batch Size Batch Size<br>Latency (s)<br>Normalized Latency<br>**----- End of picture text -----**<br>


Fig. 3. **Left** : Search latency comparison between standard IVF and IVF with fast scan (IVF-FS). Except for the fast scan optimization, both indexes share identical configurations. IVF-FS achieves significantly faster search speed. **Right** : Latency breakdown of IVF-FS on a 128M vector index. Lookup table operations dominate the overall search time. 

neighbor search to efficiently identify relevant documents. The retrieved vectors are mapped back to their corresponding documents, which are provided as additional context to the LLM alongside the original query. 

## _A. Inverted List Index IVF_ 

There are several approaches for structuring a vector database into a searchable index. Among them, HNSW and IVF are the most widely used. 

HNSW [27] (Hierarchical Navigable Small World) is a graph-based structure where each vector forms a node connected to its nearest neighbors. It enables rapid search via hierarchical traversal and offers fast index construction. However, the additional edge information significantly increases memory usage as the dataset grows. 

In contrast, the Inverted File (IVF) index [46] organizes the index as a hierarchical clustering structure. A subset of vectors is first clustered via K-Means to obtain centroids. Then, each database vector is assigned to the closest centroid, forming an inverted list. This structure narrows the search space using only centroid metadata, resulting in low memory overhead and high scalability. As such, IVF is widely adopted and studied in retrieval systems for large knowledge corpora [6], [11], [14], [21], [35], [43]. To further reduce memory usage, quantization techniques are applied on top of IVF. Scalar quantization (SQ) reduces each vector element to a smaller numerical type (e.g., float32 to int8), offering simplicity but limited compression. For higher compression ratios, product quantization (PQ) [13] is commonly used. 

## _B. Search Operation in IVF Index_ 

Figure 2 illustrates the search process in an IVF-PQ index, where an inverted list structure is combined with product quantization. When a query is received, the retriever first identifies the closest clusters, narrowing the search space. The number of clusters searched is controlled by the parameter nprobe, which trades off speed and accuracy. 

Next, a distance lookup table is constructed. Since each vector is quantized into discrete sub-vector codes, each code maps to a representative value, trained and stored in the codebook. By pre-computing distances between the query vector and these representative values, the system avoids computing full distances to every vector. During the scan stage, these 

**==> picture [253 x 78] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.00<br>200<br>0.75<br>100 0.50<br>0.25<br>0<br>CPU IVF GPU IVF 0.4 0.2 0.0<br>Fast Scan Search Relative KV Space<br>Normalized<br>Search Time (ms) LLM Throughput<br>**----- End of picture text -----**<br>


Fig. 4. **Left** : While fast scanning accelerates IVF-based vector search on CPU(64 core Xeon 8462Y+), GPU(H100)-based IVF search offers superior performance. **Right** : Relationship between KV cache size and LLM throughput for the Qwen3-30B model on two H100 GPUs. Reducing KV cache space leads to a significant drop in throughput. 

LUTs are used to accumulate approximate distances and retrieve the top-k nearest vectors. 

A deeper analysis of IVF search, shown in Figure 3, reveals that the large portion of the search time is spent on constructing and scanning the distance lookup table. This highlights the LUT stage as a key bottleneck in retrieval latency. To mitigate this overhead, fast scanning techniques [4] have been proposed and implemented in libraries such as Faiss [6] and ScaNN [43]. These methods leverage SIMD instructions and CPU vector registers to accelerate distance lookup operations. By carefully organizing lookup tables and quantization codes into memory-aligned layouts, they significantly outperform conventional IVF scan routines, particularly in CPU-based environments. 

Motivated by their superior latency-performance trade-off, we adopt fast scanning in our system to enable efficient and low-latency vector retrieval. However, despite the SIMD capabilities of modern CPUs, CPU-based search can still become a bottleneck, ultimately degrading the responsiveness of the end-to-end RAG system. 

## III. CHALLENGES AND OPPORTUNITIES IN RAG SERVING 

## _A. GPU search vs. CPU search_ 

While fast scan indexes significantly improve the latency of vector similarity search on CPUs, GPU-based retrieval can offer even greater speedups, due to their wider vector processing units and higher memory bandwidth. As shown in Figure 4 (left), GPU-accelerated IVF search can outperform fast scan methods by nearly an order of magnitude. 

Thus, offloading retrieval to the GPU can offer higher speedups for large-scale vector databases where CPU-based search remains a bottleneck. However, this comes with a fundamental trade-off: GPU memory is already heavily utilized by LLMs, particularly for storing KV cache and model weights. Allocating additional memory for the vector index can reduce available cache space, ultimately degrading LLM throughput, as illustrated in Figure 4 (right). 

Beyond memory capacity, GPU retrieval additionally incurs scheduling overheads due to increased contention for compute resources. Shared memory is used to stage partial distance lookup tables, and each query–cluster pair typically maps to a thread block. As the number of probed clusters increases, so does the occupancy and scheduling pressure on the GPU, further impacting performance. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 78] intentionally omitted <==**

**----- Start of picture text -----**<br>
Wiki-All ORCAS<br>1.0 1.0<br>0.93<br>0.8 0.8<br>0.6 0.59 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0.00 0.25 0.50 0.75 1.00 0.00 0.25 0.50 0.75 1.00<br>Percentile of Clusters Percentile of Clusters<br>CDF CDF<br>**----- End of picture text -----**<br>


Fig. 5. CDF of cluster access frequency for queries from the WikiAll [37] and ORCAS [5] datasets. While the two distributions exhibit different levels of skewness, in both cases, the top 20% of clusters account for over 50% of the total distance computations. 

**Takeaway 1.** _**GPU-based retrieval can substantially outperform even the fastest CPU-based methods, but due to contention with LLM inference workloads, careful memory and compute allocation is essential.**_ 

## _B. Opportunity of Tiered Search Structure_ 

The distribution of query access patterns in IVF indexes reveals the presence of hot clusters, a small subset that dominates retrieval 

As shown on the left of Figure 5, the cumulative distribution of coarse quantization results exhibits a strong skew: the top 20% of clusters account for nearly 60% of accesses in Wiki-All [37] and over 93% in ORCAS [5]. This skew is especially pronounced in ORCAS, which reflects real-world query behavior through unfiltered click-through logs, capturing both popularity bias and the imbalance introduced by k-means quantization. 

This imbalance results in inefficient memory usage, as significant resources are allocated to rarely accessed clusters with limited contribution to retrieval quality. 

**Takeaway 2.** _**IVF index access patterns are highly skewed: a small number of clusters account for the vast majority of retrievals. This motivates a tiered index design, where frequently accessed clusters are prioritized for acceleration (e.g., GPU caching), and cold clusters are offloaded to lowertier compute and storage.**_ 

Embedding access patterns in recommendation systems are also known to exhibit significant skew, where a small subset of items or users dominates embedding lookup frequency. This observation has motivated several tiered architecture designs that prioritize popular embeddings for faster access [1]–[3], [19], [25], [25], [29]. Inspired by this insight, our work offers tiered acceleration to vector similarity search. However, a key distinction lies in the granularity of memory accesses. In recommendation systems, embedding look-ups are performed via embedding IDs. In contrast, vector similarity search systems conduct fully content-based retrieval, where relevant vectors must be located by computing distances to hundreds or thousands of candidates per query. To identify the nearest vector, the search must access not only the target vector but also neighboring vectors within the cluster. 

Moreover, even if each embedding is uniformly accessed, clusters can contain varying numbers of vectors, exacerbating the access skew. This imbalance causes certain clusters to dominate query traffic, creating hot regions in memory access. 

**==> picture [253 x 76] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0<br>0.5<br>Wiki-All<br>ORCAS<br>0.0<br>5% 10% 20%<br>Cache Coverage<br>Hit Rate<br>**----- End of picture text -----**<br>


Fig. 6. Violin plot of hit rate distribution at different cache-coverages. The width of the violin indicates the density of queries with similar hit rates, while the white dot and black bar denote the median and inter-quartile range, respectively. This highlights that increasing cache coverage improves overall hit rates but does not eliminate tail queries with poor hit rates. 

As a result, skew in our setting emerges more prominently at the cluster level rather than the vector level. 

Consequently, although both domains benefit from tiered designs, the unit of optimization and the manifestation of skew differ substantially. Our approach explicitly targets clusterlevel skew in large-scale retrieval workloads, enabling effective tiered placement and latency-aware resource allocation that are not directly addressed by prior embedding-centric designs. 

## _C. Variance of Hit Rate across Queries_ 

While tiered resource allocation strategies can accelerate vector search by caching frequently accessed clusters, their effectiveness in deployment is often hindered by query-level variance in hit rates. Long-tail queries with less cache hits can significantly limit the overall performance gains. 

Figure 6 presents a violin plot of hit rate distributions across queries, measured by counting the number of clusters (among the total nprobe) that fall within the cached hot cluster set. As cache coverage increases from 5% to 20% of total clusters, the average hit rate improves accordingly. However, the variance remains substantial, especially in highly skewed datasets such as ORCAS, where a long tail of queries exhibits minimal cache 

This variance introduces a deployment challenge. Since vector search throughput scales with batch size, retrievers are typically deployed with batching enabled. However, in the presence of low-hit queries within a batch, the entire batch’s processing time is effectively bounded by the slowest query. As a result, even if the average per-query latency is reduced by GPU acceleration, end-to-end latency improvements are constrained. Therefore, to fully realize the benefits of tiered or cached retrieval in real-world deployments, it is essential to account for such hit rate variance and long-tail behavior during system design. 

**Takeaway 3.** _**Variance in hit rate across queries poses a challenge in latency-critical deployments, due to long-tail queries as batching amplifies the impact of long-tail queries, limiting the effectiveness of caching.**_ 

In summary, while GPU-based retrieval can vastly outperform CPU methods, it introduces a resource contention problem when co-located with LLMs, due to limited GPU memory and compute capacity. Meanwhile, query access patterns exhibit strong skew: a small fraction of clusters account for most retrieval traffic, making selective caching and tiered 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [516 x 224] intentionally omitted <==**

**----- Start of picture text -----**<br>
Hybrid Index Construction Distributed VectorLiteRAG Pipeline<br>Profiling Latency Bounded Partitioning CPU GPU 0<br>Coarse Quantizer Queries<br>LLM<br>LLM CapacityServing SLO Router<br>Performance<br>Training Model Access Counter<br>Vector Queries LatencySearch Throughput SLO Compl. RateAvg Hit Rate Cid Cid Mapping TableNew Cid Shard id LLM GPU 1<br>Index<br>Access  Hit Rate Cache Size CPU nprobe    GPU 0   GPU 1   GPU 2   GPU 3<br>Pattern<br>Refresh Access Statistics CPU Distance Scanning GPU 0<br>SIMD  LLM<br>Index Splitter Processor<br>Dispatcher<br>GPU 3<br>LLM<br>Knowledge<br>Base<br>Cold Clusters Hot Clusters / Mapping Table<br>Hot Clusters Migration<br>Update Mapping<br>**----- End of picture text -----**<br>


Fig. 7. System architecture of VECTORLITERAG. The system has two stages, **Left** : offline hybrid index construction and **Right** : runtime distributed pipeline. Profiling guides latency-bounded partitioning to determine cache size and split point, producing sharded indices and mapping tables. At runtime, queries are routed via coarse quantizer and mapping tables, hot clusters run on GPUs, cold clusters on CPUs. A dynamic dispatcher forwards early-finished queries to LLM workers in a timely manner. Blue trails and boxes indicate runtime index refresh and update procedures. 

search strategies effective. However, significant variance in hit rates across queries, especially long-tail queries, poses a major challenge in latency-sensitive deployments, as batching magnifies the bottleneck introduced by slow queries. These insights motivate the design of VECTORLITERAG, which adaptively partitions the index across GPU and CPU tiers, accounting for workload skew, hit rate variance, and end-to-end latency constraints to optimize throughput and responsiveness. 

## IV. VECTORLITERAG 

VECTORLITERAG is an optimized RAG system that determines the optimal configuration for a CPU–GPU hybrid vector index. It is organized around tightly integrated components: (1) performance modeling and latency-aware hybrid index construction, and (2) a distributed runtime pipeline for inference serving. Given the latency constraint, LLM, index, and system configuration, VECTORLITERAG computes a partitioning point for tiered search, constructs the hybrid index, and serves inference requests through a tailored pipeline. 

**Hybrid Index Construction.** The first component of VECTORLITERAG focuses on understanding the performance characteristics of the underlying system. This stage profiles CPU-based search latency, query-to-cluster access patterns, and standalone LLM throughput to characterize contention between retrieval and generation. These measurements drive a performance model and cache-coverage estimator, enabling a latency-bounded partitioning algorithm to select hot clusters. The hot clusters are then sharded into GPU sub-indexes. 

**Distributed VECTORLITERAG Pipeline.** The second component is the runtime pipeline that operationalizes the hybrid index. At runtime, batched queries are routed to CPU 

or GPU shards using mapping tables generated during index construction, allowing each shard to operate with a flexible nprobe budget and reducing contention with LLM. A dynamic dispatcher further improves batching efficiency by advancing early-completing queries to mitigate tail latency. 

The partitioning scheme and runtime pipeline are independent of the distance metric or compression method. As long as the index exhibits clustered structure and benefits from GPU acceleration, VECTORLITERAG can identify an effective hybrid configuration and deliver SLO-compliant RAG service. 

## _A. Hybrid Index Construction_ 

_1) Profiling-based Performance Modeling:_ Since GPU resources are limited, accurately modeling performance is critical for determining the optimal index partitioning point. To construct these models, VECTORLITERAG profiles latency and access statistics using calibration queries from a training set. Specifically, it collects: (1) latency breakdown of CPUbased vector search and (2) cluster access frequency distributions. Additionally, throughput of the bare LLM is measured to guide partitioning decisions under joint CPU-GPU execution. 

As described in Section II-B, IVF index search latency is dominated by two components: coarse quantization (CQ) and LUT operations. We profile both stages across varying batch sizes and construct independent models for each. However, in our design, only the LUT stage, which corresponds to the individual distance computation and scanning step, is considered for GPU offloading for two main reasons: 

First, CQ is a similarity search over the quantizer (centroid) vectors, which is often implemented using memory-intensive graph-based structures such as HNSW. Offloading CQ to GPU 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

would require additional memory for the graph and complicate memory management. Second, if CQ were distributed across GPU shards, the resulting search path would involve repeated device transitions: CPU _→_ GPU (quantization) _→_ CPU (merge and routing) _→_ GPU (search) _→_ CPU (final merge). This induces costly inter-device communication and synchronization overheads. Moreover, our objective is to ensure stable performance within the latency budgets rather than to minimize absolute latency. Thus, for our purpose, we retain CQ on the CPU and use GPUs for distance computations, as this offers performance benefits while simplifying the optimization space. 

Empirically, as shown in Figure 8 (left), CPU search latency exhibits a piecewise linear relationship with batch size. Initial steps appear as the system transitions from single-threaded (single query) to multi-threaded execution (batched queries). Accordingly, we model _T_ CQ[CPU] and _T_ LUT[CPU][as][piecewise][linear] functions of batch size. 

When hot clusters are cached, the overall search time reduces accordingly. LUT operations offloaded to GPUs are fully hidden under CPU’s execution, and the CPU processing time decreases in proportion to the number of hits. As a result, we model the latency of the hybrid partitioned index as: 

**==> picture [202 x 14] intentionally omitted <==**

where _η_ denotes the hit rate, in particular the minimum hit rate among all queries in the batch. 

_2) Tail Query Hit Rate Estimation:_ As discussed in Section III-C, caching hot clusters leads to varying hit rates across queries. Because, CPU side LUT workload is proportional to the miss rate (1 _− η_ ), this variance directly translates into differences in search latency. Moreover, since vector search is typically executed in batches to maximize throughput, the completion time of the entire batch is dictated by the slowest query, one with the fewest hits. Therefore, modeling the minimum hit rate within a batch is critical for accurate performance estimation. 

We model the distribution of per-query hit rates using a Beta distribution _f_ ( _x_ ), which is widely used in Bayesian statistics for variables constrained to the [0 _,_ 1] range. For a batch of size _b_ , the expected minimum hit rate _η_ min, i.e., the first-order statistic, is computed as: 

**==> picture [224 x 26] intentionally omitted <==**

where _F_ ( _x_ ) is the cumulative distribution function of _f_ ( _x_ ). The mean hit rate _η_ ¯ can be obtained directly from the query–cluster access profile, which reflects the cumulative fraction of accesses covered by the cached clusters. Estimating the variance is more challenging, as it would require rerunning queries through the quantizer and counting individual hits after masking hot clusters, a process that is both computationally expensive and incompatible with iterative partitioning algorithm. 

Instead, we approximate the hit rate variance as a function of the mean. We observe that hit/miss variance peaks when 

**==> picture [253 x 83] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.3 CQ LUT Search 0.03<br>0.2 0.02<br>0.01<br>0.1<br>0 10 20 30 0.2 0.4 0.6 0.8<br>Batch Size Mean Hit Rate<br>Variance<br>Latency (s)<br>**----- End of picture text -----**<br>


Fig. 8. **Left:** Search latency of ORCAS queries on a 64-core Intel Xeon 8426Y CPU. **Right:** Empirical variance of hit rates across queries in the Wiki-All dataset as a function of mean hit rate. The observed parabolic shape supports our variance approximation model. 

_η_ ¯ = 0 _._ 5, and becomes more uniform as _η_ ¯ _→_ 0 or _η_ ¯ _→_ 1. This mirrors the variance behavior of the Beta distribution; ¯ ¯ Var( _X_ ) _∝ η_ (1 _−η_ ). Thus, by empirically profiling the variance at _η_ ¯ = 0 _._ 5, denoted _σ_ max[2][,][we][can][approximate][the][variance] at arbitrary _η_ ¯ as: 

**==> picture [101 x 12] intentionally omitted <==**

Figure 8 (right) validates the approximation. This allows instantiating a Beta distribution _f_ ( _x_ ) with inferred mean and variance for any cache coverage configuration. 

Finally, using Eq. 2, we compute the minimum hit rate within a batch for a given cache coverage. Inverting this relation numerically yields the function: 

**==> picture [143 x 11] intentionally omitted <==**

which is used in the main partitioning algorithm to identify the optimal cache coverage that satisfies latency constraints. _3) Latency-Bounded Partitioning Algorithm:_ In the hybrid RAG pipeline, LLM throughput decreases as more GPU memory is allocated to the vector index, due to contention between KV cache and index storage. To balance these competing demands, we introduce an iterative algorithm that determines an index partitioning point satisfying the latency constraint. 

Algorithm 1 outlines the proposed latency-bounded partitioning algorithm. It takes the following inputs: the latency target, the baseline KV cache memory footprint when no vector index is loaded, and the peak bare LLM throughput. The goal is to find the largest feasible cache coverage for the GPU index (partitioning point _ρ_ ) that satisfies SLO constraint. 

We first compute the latency bound for the hybrid vector search stage. To account for queuing delay, the analysis considers a worst-case scenario in which a request arrives immediately after the previous batch begins processing. Under steady-state load with uniformly arriving requests, this tail query experiences full batch latency _W_ ( _b_ ) as queuing delay. To maintain the total response time within the latency budget, the search latency must satisfy _τs ≤_ SLOsearch _− W_ ( _b_ ). 

To avoid circular dependency (as _W_ ( _b_ ) depends on _τs_ ), we approximate this term using a queuing factor _ϵ_ , leading to: 

**==> picture [158 x 22] intentionally omitted <==**

In our setting, we set _ϵ_ = 1, as it represents the worst case where the queuing delay equals one batch latency. This choice 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

is empirically supported from the CPU-only baseline, where _ϵ_ ranged between 0.9 and 1.0. 

**Search iteration.** The algorithm then performs a binary search over possible values of _ρ_ using the modeled latency and hit rate behavior. For each candidate _ρ_ , the reduced LLM throughput is estimated based on the corresponding decrease in KV cache capacity. Although this interpolation is coarse, it provides a conservative lower bound because the throughput–cache curve is generally convex. The INFERPARTITION function is subsequently invoked to compute the expected batch size, given by _B_ = _µ · τs_ , where _µ_ is the current throughput bound. Since batch size _B_ must be an integer, two rounding strategies are considered: 

- **Rounding up.** This implies longer latency and thus requires more cache coverage to meet _τs_ . From the hybrid latency model (Eq. 1), we solve for _η_ 1 and convert it to coverage _ρ_ 1 via the HITRATE2COVERAGE function. 

- **Rounding down.** This yields a smaller batch size (shorter latency), but may not meet the required throughput. To ensure throughput _µ_ is met, we solve for _η_ 2 using the adjusted latency bound _B/µ_ from the throughput constraint. 

At the end of the iteration, the smaller of _ρ_ 1 and _ρ_ 2 is selected, as it requires less GPU memory. This value is used to update the binary search interval. 

**Convergence.** If the newly computed partitioning point _ρ_ increases, the resulting drop in throughput leads to a smaller batch size in the next iteration, which in turn drives _ρ_ back down. Conversely, if _ρ_ shrinks, the throughput bound increases, allowing for more cache coverage. This feedback loop ensures convergence of the algorithm within a limited number of iterations. In practice, convergence takes less than one minute as shown in Figure 9. 

_4) Index Splitter:_ Once the partitioning point _ρ_ is determined, it is passed to the final stage of index construction, which is the index splitter. The splitter first identifies the hot clusters based on the access profile and the target cache coverage _ρ_ . These hot clusters are then sorted by size and distributed to GPU shards in a round-robin fashion to balance memory usage across sub-indexes. 

Alongside the construction of each sub-index, the splitter generates a set of mapping tables. These tables encode the correspondence between original cluster IDs and their assigned shard as well as the remapped local cluster IDs, enabling efficient routing during query execution. 

## _B. Distributed VectorLiteRAG Pipeline_ 

The right side of Figure 7 illustrates the runtime architecture of VECTORLITERAG. At initialization, memory is allocated sequentially for the index and then for the LLM to prevent memory interference between the vector search and LLM engines. The two components operate through different processes and thus use separate GPU streams for concurrency. 

Similar to other IVF-based indexes, the pipeline begins with coarse quantization to identify candidate clusters. However, from this point on, VECTORLITERAG introduces a 

## **Algorithm 1** Latency Bounded Partitioning 

**Input:** SLOsearch, _MEMKV cache_ , _µLLM_ 0 **Output:** _ρ_ 

|1: <br>2: <br>3: <br>4:<br>5:|_τs ←_SLOsearch<br>1+_ε_<br> _ρ_low _←_0, _ρ_high _←_1<br> **while** _ρ_high_−ρ_low _> δ_ **do**<br>_ρm ←ρ_low+_ρ_high<br>2<br>_µ_LLM _←MEMKV cache−MEMIndex_(_ρ_)<br>_MEMKV cache_|_µLLM_0|
|---|---|---|
|6:|_ρ ←_INFERPARTITION(_ts, µ_LLM)||
|7:|**if** _ρ > ρm_ **then**||
|8:|_ρ_low _←ρ_||
|9:|**else**||
|10:|_ρ_high _←ρm_||
|11:|**end if**||
|12:|**end while**||
|13:|**return** _ρ_||
|14:|||
|15:|**function** INFERPARTITION(_τs, µ_)||
|16:|_B ←⌈τs · µ⌉_||
|17:<br>18:<br>19:|_T_ CPU<br>search(_B_)_, T_CPU<br>LUT (_B_)_←_PERFMODEL(_B_)<br>_η_1 _←T_CPU<br>search(_B_)_−τs_<br>_T_ CPU<br>LUT (_B_)<br>_ρ_1 _←_HITRATE2COVERAGE(_η_1_, B_)||
|20:<br>21:<br>22:<br>23:<br>24:<br>25:|_B ←⌊τs · µ⌋_<br>_T_ CPU<br>search(_B_)_, T_CPU<br>LUT (_B_)_←_PERFMODEL(_B_)<br>_η_2 _←T_CPU<br>search(_B_)_−B/µ_<br>_T_ CPU<br>LUT (_B_)<br>_ρ_2 _←_HITRATE2COVERAGE(_η_2_, B_)<br>**return** min(_ρ_1_, ρ_2)<br> **end function**||



customized retrieval pipeline tailored for hybrid CPU-GPU execution. We now describe each component in detail. 

_1) Router:_ To support efficient vector retrieval on a distributed multi-GPU system, VECTORLITERAG implements a custom routing mechanism rather than relying on Faiss’s builtin IndexIVFShards. The default implementation in Faiss is suboptimal in constrained environments for two main reasons. (1) IndexIVFShards partitions the index uniformly by vector or cluster ID, ignoring access frequency. While, convenient for implementation, it retains centroid metadata even for clusters that are not locally resident, causing unnecessary memory overhead, especially problematic when the number of clusters is large. (2) During search, each sub-index is instructed to probe the same number of clusters, even if many of them are not resident on that shard. Although certain probes are ultimately skipped at runtime, the batched execution of cluster scanning kernels still launches GPU thread blocks for them. These launches consume scheduling bandwidth and shared memory resources, regardless of whether the actual computation is needed. Since shared memory usage increases with nprobe, this results in inefficient kernel launches and exacerbates resource contention, especially in large-scale vector databases. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

To address these issues, VECTORLITERAG uses the mapping tables generated during index splitting to route each query to the appropriate GPU shards and prune irrelevant probes, thereby accounting for the device-level variance. This substantially reduces the effective nprobe per shard, lowering both memory pressure and kernel scheduling overhead. At runtime, only GPU workers holding relevant clusters receive and execute the search request, while the remaining portion of the search is handled by the CPU. This hybrid execution minimizes contention and enables more efficient use of GPU memory and compute resources. 

_2) Dynamic Dispatcher:_ Because hit rates vary across queries, the effective nprobe differs even within a batch. As batch size increases, the minimum hit rate tends to decrease, increasing the search latency for the entire batch. To mitigate this issue, VECTORLITERAG employs a dynamic dispatcher that accelerates early query completion. 

When search is initiated, a separate dispatcher thread is launched. Each GPU worker sets a completion flag once its assigned clusters are scanned. After all GPU flags are set, the dispatcher begins polling for queries that have completed their full search. To facilitate timely query promotion, a callback mechanism connects the CPU search loop and the dispatcher, as CPU processes clusters one-by-one, grouped by related queries. At the end of each iteration, the current scan count is compared with the expected nprobe for each query. When all assigned clusters for a query are scanned, the callback is invoked, and the query and its results are inserted into a threadsafe queue. 

The dispatcher polls this queue at short intervals. Once a completed query is available, it merges the CPU and GPU results, re-ranks them to obtain the final top- _k_ vectors, and forwards the result to the downstream document retriever. This proactive execution reduces head-of-line blocking within batches and improves end-to-end latency, particularly for highhit-rate queries. It also enhances batching continuity by enabling smoother transitions between retrieval and generation stages, which already employs continuous batching schemes. 

_3) Adaptive Runtime Index Update:_ Our model is built upon the distributional characteristics of queries aggregated across batches. While correlations among queries may temporarily shift access patterns, they primarily reduce the number of statistically independent samples rather than altering the overall distributional trend. Nevertheless, temporal bias can arise in practice, and to mitigate potential performance degradation caused by such drift, VECTORLITERAG employs an adaptive re-profiling and update process. 

VECTORLITERAG can swiftly react to shifts in query distribution without interrupting service. During runtime, the router monitors (1) average hit rates and (2) per-cluster access frequencies. For every few minutes or after a few thousand requests, it periodically resets the counters to detect distributional drift. When the average SLO attainment falls below a threshold and observed hit rates diverge from their expected values, an update cycle is triggered: re-profiling query access patterns, rerunning the latency-bounded partitioning algorithm, 

**==> picture [253 x 75] intentionally omitted <==**

**----- Start of picture text -----**<br>
60 Profiling Algorithm Splitting Loading 200ms<br>40 300ms<br>100ms 150ms 200ms<br>150ms<br>20<br>0<br>Wiki-All ORCAS 1K ORCAS 2K<br>Latency (s)<br>**----- End of picture text -----**<br>


Fig. 9. Time consumed for re-building the GPU index shards using updated query access data. Numbers above the bars denote the search time SLO constraints applied for the system. 

generating shards, and loading the updated indices onto GPUs. 

All stages, from profiling to loading, complete in under a minute, allowing updates to run in the background. At the per-shard level, index generation and loading take less than ten seconds. The detailed timing breakdown for each stage is shown in Figure 9. While a GPU shard is being refreshed, the router temporarily redirects queries for those clusters to CPU paths, preserving the service continuity. Once the updated shard is loaded, routing automatically returns to the GPU. 

Per-cluster updates are avoided because clusters are stored contiguously to enable high-bandwidth access. Since clusters vary in size, updating clusters individually would lead to memory fragmentation and inefficient data placement. Instead, VECTORLITERAG performs full-shard updates, as migration of each shard takes only a few seconds, providing robustness and simplicity. 

According to our observations, profiling with only 0.5% of the queries from a separate training set successfully captured the distribution of 10M ORCAS queries. We therefore assume that a single index update can sustain stable service for roughly one hour under steady traffic, given the system throughput measured in our experiments. 

## V. METHODOLOGY 

## _A. Experiment Setup_ 

To evaluate VECTORLITERAG, we conduct experiments across various datasets, models, and hardware configurations. This section describes the datasets, models, evaluation metrics, and system setup. 

**Datasets and Models.** We use two datasets: Wiki-All and ORCAS. We construct the IVF index following the configuration guidelines provided by the Faiss library. The Wiki-All [37] vector database contains 88M 768-dimensional vectors derived from Wikitext [28] and Cohere Wikipedia embeddings, yielding a compressed IVF index with a footprint of 18GB. We also construct two additional indexes from chunked Wikipedia documents [40] using the Stella [42] embedding model of dimensions 1024 and 2048, and queries from the Microsoft ORCAS dataset [5]. ORCAS consists of real Bing queries and preserves duplicates to reflect realistic query distributions. The ORCAS 1K and ORCAS 2K indexes occupy 40GB and 80GB of memory, respectively. 

Our retrieval pipeline builds on Faiss v1.9.0 [6], [17], with internal extensions for flexible nprobe settings and dispatcher callbacks. The overall system, including the profiler and latency-aware scheduler, is implemented in Python. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

For generation, we evaluate three models—Llama3-8B, Qwen3-32B, and Llama3-70B [9], [41]—served using vLLM v0.9.1 [18]. The retriever and LLM run as separate subprocesses, with the main process coordinating request generation and document fetching to integrate the full RAG pipeline. 

To evaluate system performance, we sample queries from a dedicated test set that is disjoint from the profiling set. The request arrival process follows a Poisson distribution, a commonly adopted modeling choice in prior work [18], [31], [45]. For each query, the top-25 documents are retrieved, and a 1024-token input is constructed and passed to the LLM, which then generates a 256-token output, following the setup in [35]. The initial nprobe is set to 2048, which is sufficient to achieve an average retrieval quality of 0.91 Normalized Discounted Cumulative Gain (NDCG) [39] at 50. 

**SLO Settings.** The SLOs for retrieval and generation stages were defined separately and then combined. For retrieval, since no standard criteria exist, we set the SLOs heuristically, relaxing them for larger databases (see Table I). For generation, the SLO was defined as the latency measured at the model’s throughput limit. These capacity values were also used in building our performance model. 

TABLE I 

SLO TARGET VALUES USED IN THE MAIN EVALUATION 

|**Vector Index**<br>_SLOsearch_|**LLM**<br>_SLOLLM_|
|---|---|
|Wiki-All<br>150ms<br>ORCAS 1K<br>200ms<br>ORCAS 2K<br>300ms|Llama3-8B<br>217ms<br>Qwen3-32B<br>191ms<br>Llama3-70B<br>311ms|



**System Configuration.** We conduct our experiments on two types of nodes, each equipped with eight NVIDIA GPUs. The L40S node includes L40S GPUs with 48GB GDDR memory and dual Xeon 6426Y CPUs. The H100 node uses H100 GPUs with 80GB HBM and Xeon Platinum 8462Y CPUs. We use the L40S node for smaller models (Llama3-8B), while larger models requiring model parallelism (Qwen3-32B, Llama370B) are run on the H100 node for maximum throughput. **Baseline Configurations.** We compare VECTORLITERAG against several key baselines. Since VECTORLITERAG builds on FAISS, we use vanilla FAISS-CPU IVF FastScan (CPUOnly), FAISS-GPU IVF on a dedicated GPU (DED-GPU), and a sharded FAISS-GPU IVF index distributed across all GPUs (ALL-GPU). To further demonstrate the strength of our approach, we also compare against HedraRAG [11] in section VI-D, which also uses a skew-aware caching strategy. 

## VI. EVALUATIONS 

## _A. Performance Model and Hit Rate Estimator_ 

Figure 10 evaluates the accuracy of VECTORLITERAG’s performance model. The right panel compares the predicted and actual minimum hit rates within each batch. As expected from order statistics, the minimum hit rate declines rapidly as batch size increases, and the rate of decline gradually flattens in the large-batch regime. Close alignment of two curves confirms that our Beta-distribution-based approximation reliably captures caching effectiveness. 

**==> picture [253 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.8 Wiki-All<br>150 0.7 ORCAS 1K<br>0.6 ORCAS 2K<br>0.5<br>100<br>0.4<br>0.3<br>50 0.2<br>1 4 7 10 13 1 4 7 10 13<br>Batch Size Batch Size<br>Search Time (ms) Tail Query Hit Rate<br>**----- End of picture text -----**<br>


Fig. 10. Comparison of measured (solid line) vs estimated (dotted line) values from VECTORLITERAG’s performance model. **Left:** Search latency across batch sizes. **Right:** Tail hit rates within a batch. 

The left panel compares the predicted latency of the hybrid index search with the measured latency. While the predictions generally follow the same trend, a noticeable offset exists between the two. This deviation mainly results from the dispatcher’s early-query handling, as discussed in Section VI-E1. 

Precisely capturing the dispatcher’s impact would require evaluating full order statistics to model per-request completion times, which greatly increases complexity while providing only marginal benefit. Despite these approximations, the resulting configurations perform robustly in practice, as shown in the following sections. 

## _B. SLO Attainment_ 

Figure 11 presents SLO attainment curves across all nine combinations of vector databases and LLMs. In each subplot, the horizontal dashed line marks the 90th percentile latency target, and the vertical dashed line indicates the standalone LLM throughput. All experiments use on-demand dynamic batching, where retrieval requests are served immediately after the previous search completes, allowing throughput to scale with arrival rate through adaptive batch sizing. 

Across all configurations, VECTORLITERAG sustains the extended SLO budget ( _SLO_ LLM + _SLO_ Search, defined in Table I) over the widest input rate ranges among evaluated baselines. CPU-based fast scan can support relatively high request per second (RPS) rates, its limited per-request performance leads to consistent SLO violations even under light traffic. As arrival rate increases, batch sizes grow (up to 9–10 under _>_ 40 RPS), incurring high latency and poor tail response. 

Dedicated GPU retrieval performs poorly with large models due to rigid model parallelism constraints. For instance, Llama3-70B requires a tensor parallelism degree of 4 for efficient execution. While it fits within 2 H100 GPUs, the achievable LLM throughput drops from 8 RPS to less than 2 RPS. In such settings, dedicating GPU(s) to retrieval results in resource oversubscription, harming overall system throughput. 

For small vector databases and under light loads, ALLGPU configurations can satisfy SLOs over wide traffic ranges. However, as the arrival rate approaches its reduced throughput, latency increases sharply. Although VECTORLITERAG is subject to this limitation as well, its optimized partitioning algorithm extends the SLO-attainable region nearly up to the standalone LLM throughput limit. 

To better illustrate the dynamics of RAG systems, we present a detailed TTFT breakdown in Figure 12 for the 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [516 x 195] intentionally omitted <==**

**----- Start of picture text -----**<br>
Llama3-8B Qwen3-32B Llama3-70B Llama3-8B Qwen3-32B Llama3-70B<br>1.0<br>0.8 50<br>0.6<br>0.4<br>0.2<br>0.0 0<br>20 30 40 20 30 40 10 15 20 20 30 40 20 30 40 10 14 18<br>Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s)<br>1.0<br>0.8 50<br>0.6<br>0.4<br>0.2<br>0.0 0<br>20 30 40 20 30 40 10 15 20 20 30 40 20 30 40 10 14 18<br>Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s)<br>1.0<br>0.8 50<br>0.6<br>0.4<br>0.2<br>0.0 0<br>20 30 40 20 30 40 10 15 20 20 30 40 20 30 40 10 14 18<br>Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s)<br>CPU Only DED-GPU ALL-GPU vLiteRAG<br>Wiki-All Wiki-All<br>ORCAS 1K ORCAS 1K<br>SLO Attainment<br>End-to-End Latency (s)<br>ORCAS 2K ORCAS 2K<br>**----- End of picture text -----**<br>


Fig. 11. **Left** : TTFT SLO attainment and **Right** : end-to-end latency of RAG pipeline under increasing arrival rates across different LLMs (columns) and datasets (rows). Our work (vLiteRAG) achieves higher SLO attainment across all regimes compared to baselines. 

**==> picture [253 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
Queuing Delay DED-GPU ALL-GPU vLiteRAG CPU Only Prefill<br>0.4<br>0.2<br>0.0<br>19 32 38 19 32 38<br>Wiki-All - Arrival Rate (req/s) ORCAS 1K - Arrival Rate (req/s)<br>Latency (s)<br>**----- End of picture text -----**<br>


Fig. 12. TTFT breakdown for Wiki-All and ORCAS 1K indexes with Qwen3-32B. Each group shows results from four configurations. Bars are stacked to show the contribution of queuing delay, vector search latency (colored segments), and LLM prefill latency (grey) 

Qwen3-32B model with Wiki-All and ORCAS 1K indices under varying input rates. As search latency increases, especially with CPU-based retrieval, queuing delays compound, further inflating TTFT. While both dedicated and ALL-GPU shared baselines perform well under low traffic, they exhibit latency spikes at higher rates due to resource contention. In contrast, VECTORLITERAG sustains stable latency by balancing throughput and latency, enabling finer control over resource allocation across the RAG stages. 

## _C. End-to-End Latency_ 

Since GPU resources are shared between retrieval and generation, interference with the decoding phase is inevitable. To assess the impact of such interference, we present the endto-end latency results from the nine configurations discussed earlier, shown in Figure 11. 

Retrieval contention is most severe for smaller models that can sustain higher loads, whereas large models saturate compute resources before retrieval pressure dominates. In the lowtraffic regime, contention is minimal, except in DED-GPU, which reduces the number of GPUs available to the LLM. However, under high traffic and with large vector databases, contention becomes significant. This is evident in the more than 2 _×_ increase in end-to-end latency observed in ALL-GPU baselines for ORCAS 2K with Llama3-8B and Qwen3-32B. Although Llama3-70B involves more intensive computation, 

**==> picture [253 x 75] intentionally omitted <==**

**----- Start of picture text -----**<br>
6<br>HedraRAG<br>30<br>vLiteRAG<br>4<br>20<br>2<br>0 10<br>20 30 40 20 30 40<br>Arrival Rate (req/s) Arrival Rate (req/s)<br>TTFT (s)<br>E2E Latency (s)<br>**----- End of picture text -----**<br>


Fig. 13. Comparison with HedraRAG. HedraRAG exhibits lower TTFT at low request rates, but latency increases sharply once the system exceeds its throughput limit. VECTORLITERAG is configured with _SLOsearch_ = 400ms. 

its low throughput ceiling causes TTFT to diverge before retrieval-induced interference becomes the dominant factor. 

In contrast, VECTORLITERAG matches CPU-based retrieval in end-to-end latency. This demonstrates that its partitioning strategy and distributed execution pipeline effectively minimizes interference by carefully limiting GPU memory and usage of GPU threads for retrieval, thereby preserving LLM generation performance, while maintaining latency lower than SLO requirements. 

## _D. Comparison with HedraRAG_ 

We compare VECTORLITERAG with HedraRAG [11], which also exploits skewed cluster access patterns in RAG pipelines. While both systems adopt tiered caching strategies for vector indices, their partitioning principles and target objectives differ fundamentally. 

HedraRAG selects GPU-resident clusters by identifying the maximum KV cache size that can sustain the throughput of the slower stage, either the LLM or the retriever. Although this approach is simple and throughput-aware, it does not account for latency constraints that are critical for real-time serving. In configurations where the LLM stage exhibits lower peak throughput than retrieval, as in 11, HedraRAG allocates the entire GPU memory to LLMs and performs vector search on the CPU. As noted in their paper, HedraRAG is most effective when retrieval becomes extremely heavy. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 75] intentionally omitted <==**

**----- Start of picture text -----**<br>
Disp. On 5.7<br>140 Disp. Off 5.4 5<br>3.9 [4.0] 150<br>4<br>120<br>2.6 [2.8] 3<br>100 100<br>24 32 41 24 32 41<br>Arrival Rate (req/s) Arrival Rate (req/s)<br>Batch Size<br>Avg Time (ms) P90 Time (ms)<br>**----- End of picture text -----**<br>


Fig. 14. **Left:** Average search latency and batch sizes. **Right:** P90 tail latency on ORCAS 2K index with dispatcher enabled and disabled. 

To enable a fair comparison, we replicate the HedraRAG setting by building an IVF index with _[√] N_ vector clusters and measuring retrieval throughput using batch sizes below 64. At nprobe = 256, CPU-only retrieval achieves 35 RPS at 0.94 NDCG@50; we increase nprobe to 6144 in our system to match this accuracy. Since HedraRAG does not support distributed retrieval, we apply their GPU caching scheme using IndexIVFShard without our optimized pipeline. 

Figure 13 summarizes the results. HedraRAG places 73% of index clusters in GPU memory, whereas VECTORLITERAG identifies a partitioning point of 31.5% under a 400,ms SLO. While HedraRAG achieves lower retrieval latency under low traffic, its operable range narrows as input rates increase. In contrast, VECTORLITERAG maintains latency near the target constraint across a wider traffic range and achieves lower overall end-to-end latency through its distributed pipeline. 

The key distinction lies in how partitioning decisions are made. VECTORLITERAG allows operators to specify a target SLO and computes the largest GPU-resident index region that satisfies this constraint, whereas HedraRAG balances throughput between stages without explicit latency objectives, which can lead to suboptimal GPU allocation. 

## _E. Ablation Studies_ 

**==> picture [253 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Length Ablation Output Length Ablation<br>500<br>0<br>11 25 39 53 67 12 23 34 45 56<br>500<br>0<br>5 10 15 20 25 6 10 14 18 22<br>Arrival Rate (req/s) Arrival Rate (req/s)<br>CPU Only vLiteRAG ALL-GPU CPU Only vLiteRAG ALL-GPU<br>2048/256 1024/256 512/256 1024/512 1024/256 1024/128<br>Llama-3-8B<br>P90 TTFT (ms)<br>Llama-3-70B P90 TTFT (ms)<br>**----- End of picture text -----**<br>


Fig. 15. **Left** : P90 TTFT across different input and **Right** : output lengths. Darker curves represent longer input/output sequences, while brighter curves correspond to shorter ones. Experiments were conducted using the ORCAS-2K index. 

Longer inputs increase prefill cost, raising TTFT and shifting SLO violations to lower arrival rates as compute resources saturate. Similarly, longer outputs reduce the SLO-compliant range due to extended generation time and higher KV cache usage. Across both dimensions, VECTORLITERAG maintains serviceability over a wider range than the baselines, highlighting the robustness of its partitioning scheme. 

_3) Sensitivity study on SLOsearch:_ To evaluate the robustness of our system under varying service constraints, we test VECTORLITERAG across multiple SLO _search_ targets. All plots in Figure16 use P95 TTFT as the primary metric, with P90 results additionally shown as dashed lines for VECTORLITERAG. Changing the quantile slightly expands or shrinks the SLO-compliant range; in our evaluation, the difference between P90 and P95 was at most 1 RPS. 

## TABLE II 

SLO TARGETS AND CORRESPONDING INDEX SHARD SIZES. 

_1) Dynamic Dispatcher:_ Figure 14 illustrates the effectiveness of the dynamic dispatcher in the distributed VECTORLITERAG pipeline. By polling the scanning loop and dispatching queries immediately upon completion, the dispatcher reduces search latency by up to 16%, improving both average and tail latency. This gain is achieved by overlapping the merging and re-ranking of early-completed queries with the ongoing scanning of slower queries, avoiding bulk merging at the end. 

Figure 14 also reports average batch sizes under varying arrival rates. With adaptive batching, requests are grouped dynamically based on current pipeline load. Since vector search has higher throughput capacity than the LLM, it absorbs higher arrival rates by increasing batch size while maintaining stable service time. In contrast, fixed or capped batch sizes lead to request backlogs and performance degradation. 

_2) Impact of LLM Input and Output Lengths:_ Figure 15 illustrates latency sensitivity to varying input and output lengths for Llama3-8B and Llama3-70B. The red dashed line denotes the combined SLO target of vector search and LLM stages, corresponding to the 1024/256 setting in Table I. For consistency, SLO _LLM_ is fixed across configurations. 

|**SLO (ms)**|**Index (GB)**|**Param (GB)**<br>**KV **|**Cache (GB)**|
|---|---|---|---|
|100<br>150<br>200<br>250|3.80<br>2.95<br>2.47<br>2.21|30.59|33.24<br>34.09<br>34.57<br>34.83|



Table II summarizes the target SLOs and their associated memory allocations. Under relaxed SLO constraints, the latency-bounded partitioning algorithm assigns a smaller fraction of the index to GPU shards, yielding latency behavior closer to the CPU-only baseline. As the SLO becomes stricter, the latency curve moves toward the all-GPU configuration. While tighter SLOs reduce available KV-cache space and modestly shrink the operable region, VECTORLITERAG still delivers a wider SLO-compliant throughput range than the baselines, highlighting the adaptability of its partitioning strategy and the effectiveness of its execution pipeline. 

_4) Robustness to Hardware Capacity:_ Finally, we evaluate how VECTORLITERAG adapts to different hardware capacities of the system. Following the provisioning policy commonly adopted by cloud providers, which allocates additional CPU cores as more GPUs are added, we test three 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
Search SLO: 100 ms Search SLO: 150 ms<br>1000<br>CPU Only vLiteRAG<br>ALL-GPU vLiteRAG P90<br>500<br>0<br>Search SLO: 200 ms Search SLO: 250 ms<br>1000<br>500<br>0<br>20 25 30 35 40 20 25 30 35 40<br>Arrival Rate (req/s) Arrival Rate (req/s)<br>P95 TTFT (ms)<br>P95 TTFT (ms)<br>**----- End of picture text -----**<br>


Fig. 16. P95 tail latency (and P90 for VECTORLITERAG) under different search-stage SLO constraints. Results are obtained using the Qwen3-32B model and the ORCAS 1K index. 

configurations: 4 GPUs + 32 cores, 6 GPUs + 48 cores, and 8 GPUs + 64 cores. For each configuration, we re-profile the CPU-only search latency and apply the same latency-bounded partitioning algorithm. Aside from the number of compute devices, all experiments use identical model and index setups. 

The results in Figure 17 show that VECTORLITERAG sustains the target SLO across all configurations while extending the SLO-compliant throughput roughly in proportion to the number of GPUs. While the reduced memory capacity in the GPU baseline causes decoding latency to grow rapidly with scale, VECTORLITERAG effectively contains this growth, keeping decoding latency comparable to CPU-only search cases. This demonstrates that VECTORLITERAG can be readily deployed across clusters of different sizes with minimal setup effort while maintaining consistent latency behavior. 

## VII. RELATED WORKS 

RAG applications with iterative retrieval or multi-stage generation often exhibit semantic similarity across successive queries. Motivated by this observation, several optimization techniques have been proposed, including prefetching [23], speculative retrieval [44], and pipelined execution [15]. In contrast, our work builds upon application-agnostic, generic retrieval–generation pipelines without relying on semantic priors or intermediate signals. RagCache [16] improves throughput by managing KV cache reuse between tenants, focusing on scheduling and reuse optimizations on the LLM side. Hermes [35], on the other hand, scales via disaggregation by adding CPU nodes to offload vector search. 

Efforts such as [12], [14], [21], [26], [32] propose specialized hardware or memory-centric architectures to accelerate RAG pipelines. While these approaches offer significant performance gains, they often rely on custom infrastructure, which may limit deployability in general-purpose environments. Among prior works, HedraRAG [11] also co-locates retrieval and generation on GPUs. Our work builds on this direction with an analytical model for latency and hit rate, enabling principled GPU memory partitioning under explicit SLOs. To our knowledge, VECTORLITERAG is the first 

**==> picture [253 x 84] intentionally omitted <==**

**----- Start of picture text -----**<br>
CPU Only ALL-GPU vLiteRAG 4GPUs 6GPUs 8GPUs<br>1.0<br>30<br>0.5 20<br>10<br>0.0<br>10 20 30 40 10 20 30 40<br>Arrival Rate (req/s) Arrival Rate (req/s)<br>4GPUs Cap 6GPUs Cap 8GPUs Cap<br>SLO Attainment E2E Latency (s)<br>**----- End of picture text -----**<br>


Fig. 17. **Left:** SLO attainment (the vertical dashed line denotes bare LLM capacity) and **Right:** end-to-end latency measured on 4-, 6-, and 8-GPU systems. Evaluated using the Qwen3-32B model and the ORCAS 2K index. 

solution to provide fine-grained resource control for co-located RAG pipelines. 

Future work may extend our approach to prefill–decode disaggregation frameworks [31], [45], where bandwidth-bound retrieval may run alongside compute-intensive prefill. This would require jointly modeling vector search and the throughput of both stages, but our framework offers a natural basis for such integration. 

## VIII. CONCLUSION 

This paper presents VECTORLITERAG, a latency-aware orchestration framework for Retrieval-Augmented Generation (RAG) systems that explicitly manages the tight coupling between vector retrieval and LLM inference. We show that under skewed access patterns, variability in retrieval latency interacts with inference batching, causing tail effects amplification that cannot be mitigated by optimizing either stage in isolation. 

VECTORLITERAG is driven by the insight that meeting strict RAG SLOs requires balancing batching behavior rather than maximizing instantaneous GPU utilization. By coordinating retrieval progress with inference scheduling, VECTORLITERAG suppresses tail cascades and sustains predictable endto-end latency under bursty workloads. This enables SLO compliance across a substantially wider operating regime, supporting up to 1.5× higher request rates than baseline RAG systems. Across extensive evaluation, we demonstrate that these benefits generalize across latency targets, hardware configurations, and LLM input/output lengths. VECTORLITERAG further exposes explicit control knobs that allow RAG operators to trade throughput for tail latency under constrained GPU memory budgets, making it practical for real-world deployment. 

## IX. ACKNOWLEDGMENT 

This research was supported in part through cyberinfrastructure research resources and services provided by the Partnership for an Advanced Computing Environment (PACE) at the Georgia Institute of Technology, Atlanta, Georgia, USA. This work was partially supported by gifts from Google and AMD. The views and conclusions contained herein are those of the authors and should not be interpreted as representing the official policies or endorsements, either expressed or implied, of Georgia Tech. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

## ARTIFACT APPENDIX 

## _A. Abstract_ 

The artifact includes the complete source code of the coreVECTORLITERAG system, together with our modified FAISS library used for hybrid CPU–GPU vector search. To ensure reproducibility of both preprocessing and evaluation, we also provide a collection of shell scripts and Python utilities that automate the full experimental workflow, including dataset preparation, index construction, performance profiling, and end-to-end RAG pipeline evaluation. These scripts are designed to reproduce all major results reported in the paper. 

All code and supporting materials are publicly available on GitHub https://github.com/sitar-lab/VectorLiteRAG-AE and Zenodo https://zenodo.org/records/18195323 

## _B. Artifact Check-list_ 

- **Program:** Modified FAISS library, vLLM 

- **Compilation:** gcc-11.3, nvcc-12.1, cmake. 

- **Models:** Llama-3 8B, Llama-3 70B, and Qwen-3 32B. 

- **Datasets:** MS ORCAS and NVIDIA Wiki-All. 

- **Run-time Environment:** RHEL 9 with Anaconda3. 

- **Hardware:** Single node equipped with 8 NVIDIA L40S GPUs and 8 NVIDIA H100 GPUs. 

- **Metrics:** SLO attainment, end-to-end latency, vector search hit rate estimation. 

- **Output:** CSV logs and visualization plots. 

- **Disk Space Required:** _∼_ 256 GB for evaluation; _∼_ 1.5 TB for preprocessing and index construction. 

- **Workflow Preparation Time:** 40–50 hours. 

- **Experiment Completion Time:** 10 hours. 

- **Publicly Available?:** Yes. 

- **Code Licenses?:** CC BY 4.0 

- **Archived(DOI)?:** https://zenodo.org/records/18195323 

## _C. Description_ 

_1) How to access:_ All source cod and scripts are accessible via github repository. 

_2) Hardware dependencies:_ All experiments were conducted on a single L40S node or a single H100 node, each equipped with 8 GPUs. Because larger language models rely on tensor model parallelism, the H100 system is expected to provide NVLink connectivity to ensure reproducible performance. The L40S node was configured with a 32-core Intel CPU, and the H100 node with a 64-core Intel CPU. CPU core count is an important factor, as a substantial portion of the workload executes on the host processor. 

_3) Software dependencies:_ The evaluation environment was run on RHEL 9 (or a compatible Linux distribution) using Anaconda3. Successful compilation of the FAISS library depends on specific toolchain versions, including Python 3.10, GCC 11.3, and NVCC 12.1. Intel MKL is also required to support vectorized CPU operations. 

_4) Datasets:_ The Wikiall benchmark is directly downloadable. The ORCAS 1K and ORCAS 2K benchmarks require both the MS ORCAS dataset and the English Wikipedia dump, which are publicly accessible but require long preprocessing. 

## _D. Installation and Testing_ 

## _1) Installation:_ 

# Create conda environment 

cd VectorLiteRAG 

conda create -n vlite -f ./scripts/env.yml conda activate vlite 

# Build faiss library 

git submodule update --init --recursive ./scripts/build.sh 

## _2) Preprocessing:_ 

# Download a samll dataset for testing 

./database/download.sh test 

# Chunk documents and run embedding model 

- ./database/encode.sh test 

# Train index and construct base IVF 

- ./scripts/train.sh test 

## _3) Testing:_ 

# Run a round of short cpu search based RAG # Output csv files will appear under results/test ./scripts/test.sh 

## _E. Experiment Workflow_ 

We provide scripts for evaluation along with corresponding plotting utilities. The workflow is straightforward. After completing the preprocessing steps, execute the following commands from the project’s root directory: 

- (1) Download the datasets: ./database/download.sh 

- (2) Perform preprocessing and train the base indexes: ./database/encode.sh <dataset> ./scripts/train.sh <dataset> 

- (3) Run all experiments sequentially: 

   - ./scripts/runall.sh 

For running experiments individually, please refer to the repository README. 

- (4) After all experiments are completed, generate the figures: ./scripts/plotall.sh 

   - Individual plotting options are also documented in the README. 

## _F. Evaluation and Expected Results_ 

This artifact reproduces the primary experimental results presented in Figures 10–17 of the paper. During evaluation, latency logs are saved under results/<datasets>, and all vector indexes and their associated metadata are stored in database/<dataset>. After the evaluation completes, the provided plotting scripts generate the corresponding figures and place them in figures directory. 

The reproduced results are expected to closely align with those reported in the paper, though minor variations may occur due to system-level factors. Individual evaluation runs for each data point are also supported pyand documented in the repository, enabling users to verify and assess any deviation. Von 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] M. Adnan, Y. E. Maboud, D. Mahajan, and P. J. Nair, “Accelerating recommendation system training by leveraging popular choices,” _Proc. VLDB Endow._ , vol. 15, no. 1, p. 127–140, sep 2021. 

- [2] M. Adnan, Y. E. Maboud, D. Mahajan, and P. J. Nair, “Ad-rec: Advanced feature interactions to address covariate-shifts in recommendation networks,” 2023. [Online]. Available: https://arxiv.org/abs/2308.14902 

- [3] M. Adnan, Y. E. Maboud, D. Mahajan, and P. J. Nair, “Heterogeneous acceleration pipeline for recommendation system training,” in _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 2024, pp. 1063–1079. 

- [4] F. Andr´e, A.-M. Kermarrec, and N. Le Scouarnec, “Cache locality is not enough: High-performance nearest neighbor search with product quantization fast scan,” in _42nd International Conference on Very Large Data Bases_ , vol. 9, no. 4, 2016, p. 12. 

- [5] N. Craswell, D. Campos, B. Mitra, E. Yilmaz, and B. Billerbeck, “Orcas: 18 million clicked query-document pairs for analyzing search,” _arXiv preprint arXiv:2006.05324_ , 2020. 

- [6] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazar´e, M. Lomeli, L. Hosseini, and H. J´egou, “The faiss library,” _arXiv preprint arXiv:2401.08281_ , 2024. 

- [7] W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li, “A survey on rag meeting llms: Towards retrieval-augmented large language models,” in _Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_ , 2024, pp. 6491–6501. 

- [8] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang, “Retrieval-augmented generation for large language models: A survey,” _arXiv preprint arXiv:2312.10997_ , 2023. 

- [9] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. AlDahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Guzm´an, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, G. Thattai, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra, I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. van der Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li, K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. Rantala-Yeary, L. van der Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. de Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova, M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. Chatterji, N. Zhang, O. Duchenne, O. C¸ elebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu, R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly, R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie, S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman, S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti, V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang, X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain, A. Kelsey, A. Shajnfeld, A. Gangidi, A. Victoria, A. Goldstand, A. Menon, A. Sharma, A. Boesenberg, A. Baevski, A. Feinstein, A. Kallet, A. Sangani, A. Teo, A. Yunus, A. Lupu, A. Alvarado, A. Caples, A. Gu, A. Ho, A. Poulton, A. Ryan, A. Ramchandani, A. Dong, A. Franco, A. Goyal, A. Saraf, 

   - A. Chowdhury, A. Gabriel, A. Bharambe, A. Eisenman, A. Yazdan, B. James, B. Maurer, B. Leonhardi, B. Huang, B. Loyd, B. D. Paola, B. Paranjape, B. Liu, B. Wu, B. Ni, B. Hancock, B. Wasti, B. Spence, B. Stojkovic, B. Gamido, B. Montalvo, C. Parker, C. Burton, C. Mejia, C. Liu, C. Wang, C. Kim, C. Zhou, C. Hu, C.-H. Chu, C. Cai, C. Tindal, C. Feichtenhofer, C. Gao, D. Civin, D. Beaty, D. Kreymer, D. Li, D. Adkins, D. Xu, D. Testuggine, D. David, D. Parikh, D. Liskovich, D. Foss, D. Wang, D. Le, D. Holland, E. Dowling, E. Jamil, E. Montgomery, E. Presani, E. Hahn, E. Wood, E.-T. Le, E. Brinkman, E. Arcaute, E. Dunbar, E. Smothers, F. Sun, F. Kreuk, F. Tian, F. Kokkinos, F. Ozgenel, F. Caggioni, F. Kanayet, F. Seide, G. M. Florez, G. Schwarz, G. Badeer, G. Swee, G. Halpern, G. Herman, G. Sizov, Guangyi, Zhang, G. Lakshminarayanan, H. Inan, H. Shojanazeri, H. Zou, H. Wang, H. Zha, H. Habeeb, H. Rudolph, H. Suk, H. Aspegren, H. Goldman, H. Zhan, I. Damlaj, I. Molybog, I. Tufanov, I. Leontiadis, I.-E. Veliche, I. Gat, J. Weissman, J. Geboski, J. Kohli, J. Lam, J. Asher, J.-B. Gaya, J. Marcus, J. Tang, J. Chan, J. Zhen, J. Reizenstein, J. Teboul, J. Zhong, J. Jin, J. Yang, J. Cummings, J. Carvill, J. Shepard, J. McPhie, J. Torres, J. Ginsburg, J. Wang, K. Wu, K. H. U, K. Saxena, K. Khandelwal, K. Zand, K. Matosich, K. Veeraraghavan, K. Michelena, K. Li, K. Jagadeesh, K. Huang, K. Chawla, K. Huang, L. Chen, L. Garg, L. A, L. Silva, L. Bell, L. Zhang, L. Guo, L. Yu, L. Moshkovich, L. Wehrstedt, M. Khabsa, M. Avalani, M. Bhatt, M. Mankus, M. Hasson, M. Lennie, M. Reso, M. Groshev, M. Naumov, M. Lathi, M. Keneally, M. Liu, M. L. Seltzer, M. Valko, M. Restrepo, M. Patel, M. Vyatskov, M. Samvelyan, M. Clark, M. Macey, M. Wang, M. J. Hermoso, M. Metanat, M. Rastegari, M. Bansal, N. Santhanam, N. Parks, N. White, N. Bawa, N. Singhal, N. Egebo, N. Usunier, N. Mehta, N. P. Laptev, N. Dong, N. Cheng, O. Chernoguz, O. Hart, O. Salpekar, O. Kalinli, P. Kent, P. Parekh, P. Saab, P. Balaji, P. Rittner, P. Bontrager, P. Roux, P. Dollar, P. Zvyagina, P. Ratanchandani, P. Yuvraj, Q. Liang, R. Alao, R. Rodriguez, R. Ayub, R. Murthy, R. Nayani, R. Mitra, R. Parthasarathy, R. Li, R. Hogan, R. Battey, R. Wang, R. Howes, R. Rinott, S. Mehta, S. Siby, S. J. Bondu, S. Datta, S. Chugh, S. Hunt, S. Dhillon, S. Sidorov, S. Pan, S. Mahajan, S. Verma, S. Yamamoto, S. Ramaswamy, S. Lindsay, S. Lindsay, S. Feng, S. Lin, S. C. Zha, S. Patil, S. Shankar, S. Zhang, S. Zhang, S. Wang, S. Agarwal, S. Sajuyigbe, S. Chintala, S. Max, S. Chen, S. Kehoe, S. Satterfield, S. Govindaprasad, S. Gupta, S. Deng, S. Cho, S. Virk, S. Subramanian, S. Choudhury, S. Goldman, T. Remez, T. Glaser, T. Best, T. Koehler, T. Robinson, T. Li, T. Zhang, T. Matthews, T. Chou, T. Shaked, V. Vontimitta, V. Ajayi, V. Montanez, V. Mohan, V. S. Kumar, V. Mangla, V. Ionescu, V. Poenaru, V. T. Mihailescu, V. Ivanov, W. Li, W. Wang, W. Jiang, W. Bouaziz, W. Constable, X. Tang, X. Wu, X. Wang, X. Wu, X. Gao, Y. Kleinman, Y. Chen, Y. Hu, Y. Jia, Y. Qi, Y. Li, Y. Zhang, Y. Zhang, Y. Adi, Y. Nam, Yu, Wang, Y. Zhao, Y. Hao, Y. Qian, Y. Li, Y. He, Z. Rait, Z. DeVito, Z. Rosnbrick, Z. Wen, Z. Yang, Z. Zhao, and Z. Ma, “The llama 3 herd of models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21783 

- [10] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented language model pre-training,” in _International conference on machine learning_ . PMLR, 2020, pp. 3929–3938. 

- [11] Z. Hu, V. Murthy, Z. Pan, W. Li, X. Fang, Y. Ding, and Y. Wang, “Hedrarag: Coordinating llm generation and database retrieval in heterogeneous rag serving,” _arXiv preprint arXiv:2507.09138_ , 2025. 

- [12] J. Jang, H. Choi, H. Bae, S. Lee, M. Kwon, and M. Jung, “Cxlanns:software-hardware collaborative memory disaggregation and computation for billion-scale approximate nearest neighbor search,” in _2023 USENIX Annual Technical Conference (USENIX ATC 23)_ , 2023, pp. 585–600. 

- [13] H. Jegou, M. Douze, and C. Schmid, “Product quantization for nearest neighbor search,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 33, no. 1, pp. 117–128, 2010. 

- [14] W. Jiang, M. Zeller, R. Waleffe, T. Hoefler, and G. Alonso, “Chameleon: a heterogeneous and disaggregated accelerator system for retrievalaugmented language models,” _arXiv preprint arXiv:2310.09949_ , 2023. 

- [15] W. Jiang, S. Zhang, B. Han, J. Wang, B. Wang, and T. Kraska, “Piperag: Fast retrieval-augmented generation via algorithm-system co-design,” _arXiv preprint arXiv:2403.05676_ , 2024. 

- [16] C. Jin, Z. Zhang, X. Jiang, F. Liu, X. Liu, X. Liu, and X. Jin, “Ragcache: Efficient knowledge caching for retrieval-augmented generation,” _arXiv preprint arXiv:2404.12457_ , 2024. 

- [17] J. Johnson, M. Douze, and H. J´egou, “Billion-scale similarity search 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

with GPUs,” _IEEE Transactions on Big Data_ , vol. 7, no. 3, pp. 535– 547, 2019. 

- [18] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in _Proceedings of the 29th Symposium on Operating Systems Principles_ , 2023, pp. 611–626. 

- [19] Y. Kwon and M. Rhu, “Training personalized recommendation systems from (gpu) scratch: Look forward not backwards,” in _Proceedings of the 49th Annual International Symposium on Computer Architecture_ , 2022, pp. 860–873. 

- [20] LangChain-Team, “Langchain: Context-aware reasoning framework,” https://github.com/langchain-ai/langchain, 2025, accessed: 2025-07-24. 

- [21] Y. Lee, H. Choi, S. Min, H. Lee, S. Beak, D. Jeong, J. W. Lee, and T. J. Ham, “Anna: Specialized architecture for approximate nearest neighbor search,” in _2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 2022, pp. 169–183. 

- [22] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K¨uttler, M. Lewis, W.-t. Yih, T. Rockt¨aschel, S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledge-intensive nlp tasks,” in _Proceedings of the 34th International Conference on Neural Information Processing Systems_ , ser. NIPS ’20. Red Hook, NY, USA: Curran Associates Inc., 2020. 

- [23] C.-Y. Lin, K. Kamahori, Y. Liu, X. Shi, M. Kashyap, Y. Gu, R. Shao, Z. Ye, K. Zhu, S. Wang, A. Krishnamurthy, R. Kadekodi, L. Ceze, and B. Kasikci, “Telerag: Efficient retrieval-augmented generation inference with lookahead retrieval,” 2025. [Online]. Available: https://arxiv.org/abs/2502.20969 

- [24] J. Liu, “Llamaindex,” https://github.com/jerryjliu/llama index, Nov. 2022, released on November 1, 2022. [Online]. Available: https: //github.com/jerryjliu/llama index 

- [25] Y. E. Maboud, M. Adnan, D. Mahajan, and P. J. Nair, “Slipstream: Semantic-based training acceleration for recommendation models,” in _2025 Design, Automation & Test in Europe Conference (DATE)_ , 2025, pp. 1–7. 

- [26] R. Mahapatra, H. Santhanam, C. Priebe, H. Xu, and H. Esmaeilzadeh, “In-storage acceleration of retrieval augmented generation as a service,” in _Proceedings of the 52nd Annual International Symposium on Computer Architecture_ , 2025, pp. 450–466. 

- [27] Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” _IEEE transactions on pattern analysis and machine intelligence_ , vol. 42, no. 4, pp. 824–836, 2018. 

- [28] S. Merity, C. Xiong, and R. Socher, “Pointer sentinel mixture models,” _arXiv preprint arXiv:1609.07843_ , 2016. 

- [29] D. Mudigere, Y. Hao, J. Huang, Z. Jia, A. Tulloch, S. Sridharan, X. Liu, M. Ozdal, J. Nie, J. Park, L. Luo, J. A. Yang, L. Gao, D. Ivchenko, A. Basant, Y. Hu, J. Yang, E. K. Ardestani, X. Wang, R. Komuravelli, C.-H. Chu, S. Yilmaz, H. Li, J. Qian, Z. Feng, Y. Ma, J. Yang, E. Wen, H. Li, L. Yang, C. Sun, W. Zhao, D. Melts, K. Dhulipala, K. Kishore, T. Graf, A. Eisenman, K. K. Matam, A. Gangidi, G. J. Chen, M. Krishnan, A. Nayak, K. Nair, B. Muthiah, M. khorashadi, P. Bhattacharya, P. Lapukhov, M. Naumov, A. Mathews, L. Qiao, M. Smelyanskiy, B. Jia, and V. Rao, “Software-hardware co-design for fast and scalable training of deep learning recommendation models,” in _Proceedings of the 49th Annual International Symposium on Computer Architecture_ , ser. ISCA ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 993–1011. [Online]. Available: https://doi.org/10.1145/3470496.3533727 

   - [33] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. LeytonBrown, and Y. Shoham, “In-context retrieval-augmented language models,” _Transactions of the Association for Computational Linguistics_ , vol. 11, pp. 1316–1331, 2023. 

   - [34] N. Reimers, “Sentence-bert: Sentence embeddings using siamese bertnetworks,” _arXiv preprint arXiv:1908.10084_ , 2019. 

   - [35] M. Shen, M. Umar, K. Maeng, G. E. Suh, and U. Gupta, “Hermes: Algorithm-system co-design for efficient retrieval-augmented generation at-scale,” in _Proceedings of the 52nd Annual International Symposium on Computer Architecture_ , 2025, pp. 958–973. 

   - [36] K. Song, X. Tan, T. Qin, J. Lu, and T.-Y. Liu, “Mpnet: Masked and permuted pre-training for language understanding,” _Advances in neural information processing systems_ , vol. 33, pp. 16 857–16 867, 2020. 

   - [37] R. A. Team, “Wiki-all dataset,” https://docs.rapids.ai/api/cuvs/stable/ cuvs bench/wiki all dataset/, 2024, accessed: 2025-07-31. 

   - [38] J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li, X. Xu, K. Yu, Y. Yuan, Y. Zou, J. Long, Y. Cai, Z. Li, Z. Zhang, Y. Mo, J. Gu, R. Jiang, Y. Wei, and C. Xie, “Milvus: A purpose-built vector data management system,” in _Proceedings of the 2021 International Conference on Management of Data_ , ser. SIGMOD ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 2614–2627. [Online]. Available: https://doi.org/10.1145/3448016.3457550 

   - [39] Y. Wang, L. Wang, Y. Li, D. He, and T.-Y. Liu, “A theoretical analysis of ndcg type ranking measures,” in _Conference on learning theory_ . PMLR, 2013, pp. 25–54. 

   - [40] Wikimedia Foundation, “Wikipedia dumps,” https://dumps.wikimedia. org/enwiki/latest/, accessed: 2025-12-01. 

   - [41] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388 

   - [42] D. Zhang, J. Li, Z. Zeng, and F. Wang, “Jasper and stella: distillation of sota embedding models,” _arXiv preprint arXiv:2412.19048_ , 2024. 

   - [43] J. Zhang, Q. Liu, D. Lian, Z. Liu, L. Wu, and E. Chen, “Anisotropic additive quantization for fast inner product search,” in _Proceedings of the AAAI conference on Artificial Intelligence_ , vol. 36, no. 4, 2022, pp. 4354–4362. 

   - [44] Z. Zhang, A. Zhu, L. Yang, Y. Xu, L. Li, P. M. Phothilimthana, and Z. Jia, “Accelerating retrieval-augmented language model serving with speculation,” _arXiv preprint arXiv:2401.14021_ , 2024. 

   - [45] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, “Distserve: Disaggregating prefill and decoding for goodputoptimized large language model serving,” in _18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)_ , 2024, pp. 193–210. 

   - [46] J. Zobel and A. Moffat, “Inverted files for text search engines,” _ACM computing surveys (CSUR)_ , vol. 38, no. 2, pp. 6–es, 2006. 

- [30] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, J. Heidecke, P. Shyam, B. Power, T. E. Nekoul, G. Sastry, G. Krueger, D. Schnurr, F. P. Such, K. Hsu, M. Thompson, T. Khan, T. Sherbakov, J. Jang, P. Welinder, and L. Weng, “Text and code embeddings by contrastive pre-training,” 2022. [Online]. Available: https://arxiv.org/abs/2201.10005 

- [31] P. Patel, E. Choukse, C. Zhang, A. Shah,[´] I. Goiri, S. Maleki, and R. Bianchini, “Splitwise: Efficient generative llm inference using phase splitting,” in _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 2024, pp. 118–132. 

- [32] D. Quinn, M. Nouri, N. Patel, J. Salihu, A. Salemi, S. Lee, H. Zamani, and M. Alian, “Accelerating retrieval-augmented generation,” in _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1_ , 2025, pp. 15–32. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:35:53 UTC from IEEE Xplore.  Restrictions apply. 

