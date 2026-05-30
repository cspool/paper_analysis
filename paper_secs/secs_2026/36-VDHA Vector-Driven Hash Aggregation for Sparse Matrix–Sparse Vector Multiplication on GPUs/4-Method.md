# 4 Method

We target a vector-driven SpMSpV with the matrix stored in CSC format and a sparse input vector. Figure 3 illustrates our overall workflow: (①) long columns are split into smaller segments, while short columns are directly mapped; (②) large segments are reordered to enhance locality; and (③) all segments are block-mapped to GPU SMs and aggregated in shared memory before flushing.

Based on this workflow, our design consists of two coordinated stages, with an additional pipelined execution scheme to reduce hash computation costs:

- (1) Vector processing. The input vector is scanned to identify active columns, which are classified as short or long. Long columns are further split and reordered, producing segments that are then assigned to GPU blocks.
- (2) Block-level aggregation. Each block processes its assigned segment using a private hash table in shared memory. Partial products are accumulated locally and then flushed in partial order, reducing global write conflicts and improving coalescing efficiency.
- (3) Fetch-compute-writeback pipeline. To exploit the memory stalls observed in atomic write-back, we restructure the execution into a three-stage pipeline. During hash insertions and aggregations on the current tile, asynchronous copy instructions (e.g., cp. async) are issued to fetch the next tile from global memory.

The following subsections provide detailed descriptions of these three stages.

