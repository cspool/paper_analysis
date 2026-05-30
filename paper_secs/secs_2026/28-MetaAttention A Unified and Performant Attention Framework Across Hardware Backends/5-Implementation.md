# 5 Implementation

In this section, we detail the implementation of the MetaAttention frontend and backend, focusing on the end-to-end lowering process that transforms user-defined attention variants into optimized hardware-specific kernels.

Lowering customizable functions. To lower user-defined customizable functions into executable code, MetaAttention first traces the computation into a directed acyclic graph (DAG) of tensors. Each node in the DAG represents a specific computing primitive, which is categorized as either an

elementwise operation (e.g., add, tanh) or a row-reduce operations (e.g., reduceSum, reduceMax). These nodes encapsulate metadata such as tensor shapes and dependencies, and include a grad field to facilitate automatic differentiation. During the lowering process, MetaAttention maps these nodes to optimized hardware-specific implementations: elementwise operations are executed in a SIMT style with register-level or on-chip memory fusion to minimize data movement overhead, while row-reduce operations utilize intra-warp parallel reduction to maximize locality and reduce synchronization costs.

Implementing attention runtime. The attention runtime acts as the orchestration layer that translates an optimized scheduling plan into a complete kernel. We implement a suite of kernel templates for both parallel and recurrent patterns. These templates include operations for moving intermediate tensors between different memory hierarchies—such as from global memory to shared memory, global memory to registers, and shared memory to registers—as well as matrix multiplication with inputs residing in shared memory or registers. Based on the scheduling plan, the attention runtime selects the appropriate template and performs code inlining, where the hardware-mapped customizable functions (traced in the previous stage) are directly fused into the highperformance attention execution loop. This ensures that custom logic incurs zero additional kernel launch overhead and benefits from the same memory-efficient pipelining and hardware-native optimizations as the core attention mechanism.

Mapping to hardware backend. We map our kernel templates to diverse hardware backends by leveraging specialized architectural features to ensure peak performance.

On NVIDIA GPUs, MetaAttention utilizes the Tensor Memory Accelerator (TMA) for asynchronous data loading and Tensor Cores for accelerated matrix-multiplyaccumulate (MMA) operations. These are implemented using two distinct backend frameworks: TileLang [\[29\]](#page-12-14) and CUTE [\[7\]](#page-11-15).

For AMD GPUs, MetaAttention targets modern highperformance accelerators including the MI250 and MI300X. We utilize AMD Matrix Cores for matrix operations and leverage asynchronous copy units to optimize memory transfers. This support is implemented using the TileLang backend.

