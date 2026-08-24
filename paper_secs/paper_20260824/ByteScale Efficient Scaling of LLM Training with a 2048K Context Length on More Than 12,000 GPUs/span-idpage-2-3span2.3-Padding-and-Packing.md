# <span id="page-2-3"></span>2.3 Padding and Packing

To support variable-length sequences in current static parallelism strategies, techniques such as padding and packing are necessary. As illustrated in Figure 2, padding pads the sequences in the same batch to be of the same length, but causes wasted computation. Packing [22] concatenates multiple sequences into a single one without padded tokens. It employs a special segmented attention mask to ensure that each sequence is processed independently by self-attention.

#### <span id="page-2-4"></span>2.4 Long Context Training

As self-attention exhibits both time and memory complexity of  $O(S^2)$ , when the context length scales, this quadratic complexity becomes a bottleneck. Flash Attention [7, 8] optimizes memory I/O and employs the tiling technique

to reduce memory complexity from  $O(S^2)$  to O(S), while still maintaining  $O(S^2)$  time complexity. Context Parallelism (CP) [4, 23, 25, 31] further partitions the sequence across N devices, reducing the memory from O(S) to  $O(\frac{S}{N})$ . Following Figure 1, CP shards QKV along the sequence dimension, and cross-tokens operations require KV slices to be exchanged across devices using a ring-style P2P communication, which overlaps with computation. This technique is also applicable to packed sequences, and we will detail its implementation in §7. Notably, each subsequence must also be sharded across all CP ranks, as illustrated in Figure 2(c) and 3(a).

