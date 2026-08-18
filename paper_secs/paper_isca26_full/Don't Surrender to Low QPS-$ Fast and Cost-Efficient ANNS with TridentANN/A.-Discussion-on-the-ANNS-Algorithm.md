# *A. Discussion on the ANNS Algorithm*

Quantization (e.g., PQ in [40, 66, 69] and RaBitQ [38]) is orthogonal to TRIDENTANN. Quantization reduces pervector payload (i.e., memory usage and I/O traffic), while TRIDENTANN targets bandwidth-centric scaling for performance. TRIDENTANN can store compressed codes in cluster lists on SSDs for approximate scoring, then rerank a small candidate set with raw vectors from SSDs as well, which is expected to be beneficial for saving I/O traffic of ultra-highdimensional embeddings (e.g., 4096). We reserve integrating quantization for future work. Additionally, unlike quantization used in [69], TRIDENTANN can scale to higher dimensions and larger datasets by only adding inexpensive SSDs and host memory, without increasing costly GPU memory.

Dynamic updating, including vector insertions [44] and deletions, is also important in ANNS. Although TRIDENTANN now focuses on search, updates can be supported by integrating existing mechanisms such as SPFresh and Quake [54, 75]. Because TRIDENTANN also follows the balanced clustering structure, it can split oversized clusters and merge undersized ones for future dynamic workloads as [54, 75].

# *A. Discussion on the ANNS Algorithm*

Quantization (e.g., PQ in [40, 66, 69] and RaBitQ [38]) is orthogonal to TRIDENTANN. Quantization reduces pervector payload (i.e., memory usage and I/O traffic), while TRIDENTANN targets bandwidth-centric scaling for performance. TRIDENTANN can store compressed codes in cluster lists on SSDs for approximate scoring, then rerank a small candidate set with raw vectors from SSDs as well, which is expected to be beneficial for saving I/O traffic of ultra-highdimensional embeddings (e.g., 4096). We reserve integrating quantization for future work. Additionally, unlike quantization used in [69], TRIDENTANN can scale to higher dimensions and larger datasets by only adding inexpensive SSDs and host memory, without increasing costly GPU memory.

Dynamic updating, including vector insertions [44] and deletions, is also important in ANNS. Although TRIDENTANN now focuses on search, updates can be supported by integrating existing mechanisms such as SPFresh and Quake [54, 75]. Because TRIDENTANN also follows the balanced clustering structure, it can split oversized clusters and merge undersized ones for future dynamic workloads as [54, 75].

