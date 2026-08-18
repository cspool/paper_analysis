# I. INTRODUCTION

Sparse high-order tensor operations are pervasive in modern machine learning and scientific applications ranging from large language models (LLMs) [5], [6] to recommendation systems [7]. For example, transformer-based LLMs [5] operate on a 4-dimensional activation tensor, including batch, head, sequence length, and channel dimensions, and therefore, the attention computation is abstracted as a high-order tensor contraction rather than a simple two-dimensional matrix multiplication. However, sparse high-order tensor computations pose fundamental challenges because sparsity varies across dimensions, which limits both spatial and temporal data locality. The challenge is compounded by complex sparsity dependencies that span multiple tensor modes.

Prior work [8]–[11] attempted to accelerate sparse tensor contraction by unfolding tensors into two-dimensional sparse–dense matrix multiplications (SpMM). This reformulation enables the direct use of well-established matrix multiplication optimization techniques, including dataflow scheduling, tiling, and matrix reordering, to enhance data reuse and parallelism. For instance, inner-product [10], [12], [13], outer-product [14], [15], and row-wise dataflows [16]–[18] have been extensively adopted to improve locality in SpMM kernels. Sparse matrix reordering techniques such as matrix reordering [19], column permutation [20], [21], and graph islandization [22] have been explored to regularize sparsity patterns. In addition, adaptive matrix tiling [18], [23], [24] has proven effective at balancing the distribution of nonzero elements across partitions. However, tensor unfolding dismantles high-dimensional data reuse, erasing the correlations among nonzero elements across multiple tensor modes [25].

In addition, recent efforts have explored tensor-native approaches that optimize contraction directly in the multidimensional tensor domain rather than unfolding it to SpMMs. For example, Gund ¨ uz et al. [26] represent sparse tensors as ¨ hypergraphs and leverage KaHyPar [27] to obtain partitions with minimal edge cuts while maintaining load balance. TCP [4] directly optimizes reuse at the tensor level rather than relying on unfolded matrix forms. GSpTC [3] further proposes fine-grained partitioning and scheduling techniques to improve parallelism in sparse tensor contraction. Despite significant efforts, existing approaches primarily depend on high-dimensional loop transformations or hypergraph formulations to extract data reuse in sparse tensors. However, their complexity increases rapidly with tensor order, making it difficult to extract reuse across dimensions.

In this paper, we argue that the core limitation of existing sparse tensor contraction methods is the absence of crossdimensional correlations. Conventional tensor and hypergraph abstractions are ineffective at exposing high-dimensional data reuse across tensor modes and parallelism opportunities. For example, in hypergraphs, coordinate overlap is not directly provided to capture cross-dimensional nonzero relationships. Our key insight is to record such crucial information through a co-occurrence graph that exposes cross-dimensional relationships, enabling a new transformation framework for designing efficient tensor contraction accelerators.

Specifically, this paper makes the following contributions:

• We propose a new graph formulation that transforms sparse high-order tensors into a co-occurrence graph whose weighted edges encode cross-mode nonzero correlations, enabling quantification of both input and output

TABLE I
TENSOR CONTRACTION APPLICATIONS AND CHARACTERISTICS.

| Application                             | Order  | Density Range           | Representative Designs                 |  |  |  |  |  |
|-----------------------------------------|--------|-------------------------|----------------------------------------|--|--|--|--|--|
| Large Language & Deep Learning Models   |        |                         |                                        |  |  |  |  |  |
| LLM Attention Activation                | 4      | Highly-Mildly (0%-50%)  | BigBird [28], MInference [29]          |  |  |  |  |  |
| DNN Convolutional Feature Map           | 4      | Highly-Mildly (0%-40%)  | NullHop [30], Cnvlutin [31]            |  |  |  |  |  |
| Computer Vision                         |        |                         |                                        |  |  |  |  |  |
| Video Clip Temporal Tensor              | 5      | Highly-Mildly (0%-50%)  | DeltaCNN [32], SparseTem [33]          |  |  |  |  |  |
| Volumetric Medical Image (CT/MRI)       | 3-5    | Highly Sparse (0%-10%)  | Submanifold [34], MinkowskiEngine [35] |  |  |  |  |  |
| Scientific & High-Performance Computing |        |                         |                                        |  |  |  |  |  |
| LiDAR Point Cloud Voxel                 | 3      | Highly Sparse (0%-10%)  | VoxelNet [36], PointPillars [37]       |  |  |  |  |  |
| Numerical Multilinear Algebra           | 3-6+   | Diverse Sparse (0%-99%) | Tucker [38], TensorLy [39]             |  |  |  |  |  |
| Atmospheric Spatial Event               | 3-5    | Highly Sparse (0%-10%)  | ESS'20 [40],ESA'26 [41]                |  |  |  |  |  |
| Recommendation Systems                  |        |                         |                                        |  |  |  |  |  |
| User-Item-Context Interaction           | 3-4    | Highly Sparse (0%-5%)   | PITF [42], RecSys'10 [43]              |  |  |  |  |  |
| Quantum Computing Simulation            |        |                         |                                        |  |  |  |  |  |
| Quantum State Tensor                    | 20-400 | Highly Mildly (0% 50%)  | GraFeyn [44], ablaze [45]              |  |  |  |  |  |

reuse and providing a theoretical basis for dataflow and tiling optimization.

- Leveraging the co-occurrence graph abstraction, we formulate high-order tensor tiling as a modified Prize-Collecting Steiner Tree (PCST) problem to exploit data reuse across high-order tensors while ensuring workload balance. This reduces the complexity of tiling from traditional matrix or hypergraph approaches to linear time. Moreover, we introduce a graph-based dataflow that systematically captures high-dimensional data reuse, in contrast to traditional techniques constrained to two-dimensional representations.
- We propose a hardware accelerator based on the graph dataflow, including a co-occurrence-graph scheduler and unified PE array specialized for clique-based tensor contraction. The architecture dynamically constructs partitions, exploits discovered reuse, and handles irregular tensor shapes.

We conduct a detailed performance and energy evaluation through simulation and show that TensorPrism delivers performance speedups of  $2.22\times$ ,  $2.40\times$ ,  $1.71\times$ , and  $1.76\times$  over state-of-the-art designs SPADE [1], HotTiles [2], GSpTC [3], and TCP [4], respectively.

#### II. BACKGROUND

