# NasZip: Software and Hardware Co-Design to Accelerate Approximate Nearest Neighbor Search with DIMM-Based Near-Data Processing

Cheng Zou\* $\|$ , Shuo Yang $^{\$*}\|$ , Chen Nie\* $^{\dagger}$ , Yu Zou $^{\ddagger}$ , Yu He $^{\P}$ , Chao Jiang $^{\P}$ , Limin Xiao $^{\P}$ , Weifeng Zhang $^{\P}$ , Zhezhi He\* $^{\dagger**}$ 

\*Intelligent Computing Research Group, School of Computer Science, Shanghai Jiao Tong University, Shanghai, CN

†Shanghai AI Laboratory, Shanghai, CN; ‡Institute of Information Engineering, Chinese Academy of Sciences, Beijing, CN

§School of Integrated Circuits, Shanghai Jiao Tong University, Shanghai, CN; ¶Lenovo Research, Beijing, CN

Email: chenchen\_zou@sjtu.edu.cn, yangshuo1230@sjtu.edu.cn, zhezhi.he@sjtu.edu.cn

| Equal contribution, \*\*Corresponding author

Abstract—As large language models (LLMs) continue to advance, retrieval-augmented generation (RAG) has become the key mechanism for expanding model knowledge and reducing hallucinations. Central to RAG is approximate nearest neighbor search (ANNS), which retrieves database vectors most similar to a given query. However, distance calculation over high-dimensional vectors is inherently memory-bound, causing retrieval performance to be constrained by I/O bandwidth on mainstream platforms such as CPUs and GPUs. Although many prior early exiting (EE) techniques attempt to reduce memory accesses by only computing partial dimensions, the partial distance converges too slowly to the EE threshold, which ultimately limits their performance gains. To address these challenges, we propose NASZIP, a hardware-software co-designed framework that integrates neardata processing (NDP) with a novel feature-level early exiting guided by statistics-based principal component analysis (PCA). Instead of relying solely on partial distances, NASZIP incorporates estimation and correction parameters to approximate fulldimensional distances accurately, enabling earlier exiting without compromising accuracy. We further introduce a bit-level NDPaware dynamic-float scheme that significantly reduces memory access for vector data. On the hardware side, we develop a dataaware neighbor list mapping strategy that reduces neighborretrieval latency and inter-channel communication overhead, complemented by a dedicated cache that exploits data locality and enhances prefetch efficiency. With these co-optimized techniques, NASZIP delivers speedups of up to  $8.4 \times /1.4 \times$  over CPU baseline and state-of-the-art GPU implementation at equal accuracy. Relative to the state-of-the-art NDP ANNS accelerator ANSMET, NASZIP achieves 1.69× performance improvement.

#### I. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks [1]. To enhance their factual grounding and adaptability, retrieval-augmented generation (RAG) has emerged by enabling LLMs to query and integrate external knowledge during inference [2]. At the core of RAG lies approximate nearest neighbor search (ANNS), which retrieves semantically relevant content from large-scale vector databases [3]. ANNS typically consists of two stages: *index construction*, where the corpus vectors are organized into a searchable structure, and *query search*, where relevant vectors are identified based on embedding proximity.

There are various kinds of index construction methods, including tree-based [4]–[6], hash-based [7]–[9], cluster-based [10], and graph-based [11]–[15] approaches. Among them, graph-based ANNS (GANNS) offers superior performance with high accuracy and low latency [16], and thus forms the focus of this work.

In contrast to the one-time index construction, the query search is executed repeatedly during LLM inference and therefore determines overall system performance. The distance computations between the query vector and candidate vectors exhibit low arithmetic intensity, making performance fundamentally limited by the data access bandwidth. Prior studies [17], [18] introduce early exiting (EE) techniques that compute distances over partial vector dimensions and terminate once the accumulated partial distance exceeds a predefined threshold. However, partial distances still require a relatively large number of dimensions to converge to the threshold, leading to conservative performance gains (~20%).

As a countermeasure, we propose a *feature-level EE with statistics-based PCA* (FEE-sPCA), which estimates full vector distances from partial computing results to enable earlier exiting. We propose a bit-level compression scheme using *dynamic floating-point* (Dfloat) to further reduce data accesses.

Meanwhile, due to the memory-bound nature of the ANNS, NDP devices are increasingly adopted to improve performance by utilizing the high internal memory bandwidth [17], [19], [20]. ANSMET [17] and CXL-ANNS [19] place customized acceleration logic near DRAM chips, while DReX [20] deploys PIM (processing-in-memory) units to accelerate the filter stage in the retrieval. However, their effectiveness remains limited. We break down the latency for a naive NDP, and observe that existing designs [17] cannot fully exploit NDP's internal bandwidth. For example, the CPU-side ANNS index processing incurs a heavy overhead and causes heavy cross-channel data communications (occupying over 50% ANNS latency), because the CPU is not aware of low-level vector data mapping on NDPs. This becomes a new performance bottleneck even though the distance calculation is optimized

via early exiting.

To improve upon this, we introduce a *data-aware index mapping* (DaM) and offload neighbor lookup operations to NDP. DaM ensures that each node's index and vector data reside within the same channel, minimizing cross-channel communications and fully exploiting NDP's internal parallelism. Additionally, we find that similar or repeated queries show locality when accessing graph nodes, *i.e*., frequent accesses to neighbor list entries and vector data entries. Therefore, we further design a local neighbor cache, consisting of a *local neighbor cache for table* (LNC-T) and a *local neighbor cache for data* (LNC-D), to exploit the locality of ANNS queries.

In summary, we propose NASZIP, a software-hardware cooptimization to efficiently accelerate ANNS on DIMM-based NDP devices. More specifically, our contributions are:

- ▷ On the algorithm level, we propose a novel two-fold optimization (*VD-Zip*) consisting of a feature-level early exiting algorithm (FEE-sPCA) and a bit-level dynamic floatingpoint representation (Dfloat) to reduce data accesses and computations while maintaining the recall rate.
- ▷ On the hardware level, we propose several dedicated architectural components to accelerate GANNS on NDPs. Data-aware neighbor list mapping (DaM) is proposed to offload neighbor list lookup from CPUs to NDPs and reduce communication between DRAM channels. A local neighbor cache (LNC) is proposed to exploit locality across ANNS queries and enable prefetching.

Evaluated on six datasets at the same recall, NASZIP achieves up to 8.4× and 1.4× speedup over the CPU baseline [\[21\]](#page-14-12) and the GPU implementation CAGRA [\[15\]](#page-14-6) on NVIDIA A100, respectively. NASZIP also achieves a 1.69× speedup over the SOTA NDP accelerator ANSMET [\[17\]](#page-14-8).

