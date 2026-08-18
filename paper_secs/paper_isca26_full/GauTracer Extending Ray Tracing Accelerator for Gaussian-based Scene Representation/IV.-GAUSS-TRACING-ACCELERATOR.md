# IV. GAUSS TRACING ACCELERATOR

This section introduces our proposed GauTracer architecture, which is designed to accelerate both shader execution and BVH traversal. We first describe the modifications made to the baseline RTA, highlighting an incremental design strategy for lightweight extension. We then detail the design of hardware shaders, including a reconfigurable Ray-Gauss Intersection Unit (RGIU) that supports both 3DGS and 2DGS, and a max-heap-based Any-Gauss-Hit Unit (AGHU) that efficiently handles the insertion and sorting of the hit Gaussians. Finally, we present a memory-efficient, treelet-based BVH traversal scheme that incorporates a far-node pruning strategy to reduce redundant node visits and enhance traversal efficiency.

