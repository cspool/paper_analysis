# C. Insights and Positioning of Our Work–FEnc<sup>2</sup>

FEnc<sup>2</sup> addresses these gaps by providing a unified, automated, and layer-aware framework for HE-CNN packing. Its parameterized block-size model partitions feature maps across ciphertexts, decoupling adjacent-pixel dependencies within each block to minimize inner rotations, and balances inner- and outer-rotation costs through a principled convex optimization. To maintain high throughput across layers, FEnc<sup>2</sup> dynamically applies ciphertext compression and merging, maximizing slot utilization for subsequent layer computations without altering

<span id="page-2-2"></span>TABLE II: SOTA HE packing methods comparison.

| Model             | 1          | Densely-Packe        | Rot Optimization     |               |                |
|-------------------|------------|----------------------|----------------------|---------------|----------------|
| WIOGCI            | at initial | after channel-reduce | after feature-reduce | multi-channel | single-channel |
| CryptoNets [23]   | ×          | Х                    | ×                    | ×             | X              |
| CHET [15]         | ×          | ×                    | ×                    | ×             | ×              |
| HELayers [4]      | ×          | ×                    | ✓                    | ×             | ✓              |
| Multiplxed [49]   | /          | ×                    | ✓                    | ✓             | ×              |
| Fhelipe [45]      | /          | X                    | ✓                    | ✓             | ×              |
| Hyena [62]        | /          | ×                    | ×                    | ✓             | ×              |
| Orion [18]        | ✓          | ×                    | ✓                    | ✓             | ×              |
| FFnc <sup>2</sup> |            | _/                   | ./                   | _/            | ./             |

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Fig. 3: An overview of *FEnc*<sup>2</sup>, which includes two components: 1) Conv-aware HE fragment encoding for outputting an optimal block size selection; 2) Arch-ware Ciphertext Compression for adapting to channel dimension change.

the packing format. This ensures that each layer can process as many output channels per ciphertext as possible, reducing the total number of ciphertexts while preserving SIMD efficiency. Unlike prior static or single-layer heuristics,  $FEnc^2$  automatically adapts to layer-wise computation dependencies, producing ciphertext layouts with provable guarantees on both rotation complexity and slot utilization. By delivering efficient, architecture-aware data layouts to the runtime,  $FEnc^2$  reduces the overall HE workload exposed to hardware and complements low-level optimizations (e.g., NTT, key-switching), enabling end-to-end HE-CNN acceleration on any hardware platform.

