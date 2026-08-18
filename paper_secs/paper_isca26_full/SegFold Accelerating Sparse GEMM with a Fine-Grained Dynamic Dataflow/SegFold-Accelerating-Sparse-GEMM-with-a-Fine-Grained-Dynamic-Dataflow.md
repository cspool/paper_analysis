# SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow

Xinrui Wu, Hanyu Wang, Jason Cong, and Tony Nowatzki University of California, Los Angeles, USA {alicewu,hanyuwang,cong,tjn}@cs.ucla.edu

*Abstract*—Generalized sparse matrix-matrix multiplication (SpGEMM) is critical in many domains. Existing CPUs and GPUs, as well as specialized accelerators, rely on static dataflows (e.g., inner product, outer product, Gustavson, etc.). Each static dataflow sacrifices some data reuse opportunities and imposes constraints on load balance.

To address this inefficiency, we extend the typical SpGEMM dataflows by considering dynamism. Specifically, we add finegrained dynamic scheduling to optimize reuse and reduce resource contention. We also develop dynamic remapping of partially completed work to improve load balance and parallelism. These ideas are formalized into a specific dataflow called Segment. To demonstrate Segment, we codesign a SpGEMM accelerator called SegFold. SegFold includes a memory controller that identifies fine-grained reuse opportunities in a local window of the stationary input array and exploits them through dynamic work assignment. It also incorporates a merge network that routes inputs to appropriate processing elements (PEs) for computation while dynamically remapping the work assigned to each PE to balance load. Across diverse densities and matrix sizes, SegFold achieves a geometric-mean 1.95× speedup over stateof-the-art SpGEMM accelerators and 5.3× over the best static dataflow configuration, demonstrating that adding dynamism to the dataflow design space unlocks reuse and load-balance gains that no static scheduling choice can achieve in isolation.

*Index Terms*—SpGEMM, accelerator, dataflow, sparsity

