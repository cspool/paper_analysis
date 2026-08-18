# III. TOWARDS HIGH-PERFORMANCE LLM INFERENCE WITH NEAR-SHANNON LIMIT COMPRESSED WEIGHTS

Incorporating compressed weights into LLM inference naturally consists of two stages: (a) **Offline compression**, where weights are compressed during preprocessing using algorithms

<span id="page-3-1"></span>TABLE I: Comparison of widely used compression algorithms. Entropy efficiency measures proximity to the Shannon limit, while streaming capability indicates the smallest granularity at which data can be decoded without global synchronization.

| Algorithm / Family         | Core Principle          | <b>Entropy Efficiency</b> | Access Granularity                    |  |  |
|----------------------------|-------------------------|---------------------------|---------------------------------------|--|--|
| gzip (DEFLATE) [11]        | LZ77 + Huffman          | 80–90%                    | Sequential (per block, ∼64 KB)        |  |  |
| LZ4 [48]                   | LZ77 (no entropy stage) | ≈80%                      | Byte-level (continuous)               |  |  |
| Zstandard (Zstd) [7]       | LZ77 + FSE (rANS)       | 90–95%                    | Chunk-level (64 KB-4 MB configurable) |  |  |
| Brotli [2]                 | Context + Huffman       | 90–95%                    | Block-level (windowed)                |  |  |
| <b>Huffman Coding [32]</b> | Static symbol code      | 90–95%                    | Symbol-level (per byte or token)      |  |  |
| Arithmetic Coding [43]     | Range interval          | 98–99%                    | Bit-level (serial)                    |  |  |
| BWT / PPM [14]             | Transform + context     | 95–98%                    | File-level (global transform)         |  |  |
| rANS / tANS / FSE [13]     | Finite-state entropy    | >99% (near Shannon Limit) | Byte-level (fully streaming)          |  |  |

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Fig. 3: Existing coarse-grain pipelining between decompression and transformer layer computation.

aggressive enough to approach the Shannon entropy limit; and **(b) On-demand decompression at inference**, where compressed weights must be decoded just before the transformer kernels consume them. As shown in Figure 3, existing systems such as ZipNN [19] decompress an entire layer before launching its GEMM. This places decoding on the critical path, introduces a layer-level synchronization barrier, and produces idle compute, redundant global-memory traffic, and memory stalls.

# III. TOWARDS HIGH-PERFORMANCE LLM INFERENCE WITH NEAR-SHANNON LIMIT COMPRESSED WEIGHTS

Incorporating compressed weights into LLM inference naturally consists of two stages: (a) **Offline compression**, where weights are compressed during preprocessing using algorithms

<span id="page-3-1"></span>TABLE I: Comparison of widely used compression algorithms. Entropy efficiency measures proximity to the Shannon limit, while streaming capability indicates the smallest granularity at which data can be decoded without global synchronization.

| Algorithm / Family         | Core Principle          | <b>Entropy Efficiency</b> | Access Granularity                    |  |  |
|----------------------------|-------------------------|---------------------------|---------------------------------------|--|--|
| gzip (DEFLATE) [11]        | LZ77 + Huffman          | 80–90%                    | Sequential (per block, ∼64 KB)        |  |  |
| LZ4 [48]                   | LZ77 (no entropy stage) | ≈80%                      | Byte-level (continuous)               |  |  |
| Zstandard (Zstd) [7]       | LZ77 + FSE (rANS)       | 90–95%                    | Chunk-level (64 KB-4 MB configurable) |  |  |
| Brotli [2]                 | Context + Huffman       | 90–95%                    | Block-level (windowed)                |  |  |
| <b>Huffman Coding [32]</b> | Static symbol code      | 90–95%                    | Symbol-level (per byte or token)      |  |  |
| Arithmetic Coding [43]     | Range interval          | 98–99%                    | Bit-level (serial)                    |  |  |
| BWT / PPM [14]             | Transform + context     | 95–98%                    | File-level (global transform)         |  |  |
| rANS / tANS / FSE [13]     | Finite-state entropy    | >99% (near Shannon Limit) | Byte-level (fully streaming)          |  |  |

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Fig. 3: Existing coarse-grain pipelining between decompression and transformer layer computation.

aggressive enough to approach the Shannon entropy limit; and **(b) On-demand decompression at inference**, where compressed weights must be decoded just before the transformer kernels consume them. As shown in Figure 3, existing systems such as ZipNN [19] decompress an entire layer before launching its GEMM. This places decoding on the critical path, introduces a layer-level synchronization barrier, and produces idle compute, redundant global-memory traffic, and memory stalls.

