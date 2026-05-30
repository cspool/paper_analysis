# <span id="page-7-1"></span>5.3 Runtime Optimizations

This subsection introduces a variety of runtime optimizations that minimize the MPK runtime overhead.

Paged shared-memory abstraction. In conventional GPU programming models such as CUDA and Triton, shared memory is a fast on-chip memory that is private to each thread block. Shared memory exists only for the lifetime of a kernel once the kernel finishes, its shared memory is automatically released. In the current kernel-per-operator approach, each kernel assumes *exclusive* access to the entire shared memory to comply with this programming model. However, this design prevents cross-task software pipelining (§ [2\)](#page-1-2), which overlaps data loading for a subsequent task with the computation of the current task, since both tasks need to access shared memory.

To enable such pipelining, MPK introduces a *paged sharedmemory* abstraction. Shared memory is partitioned into multiple fixed-size pages, and task implementations are modified to operate on these pages instead of assuming monolithic

<span id="page-8-1"></span>allocation. A task may *acquire* one or multiple pages based on its shared-memory footprint and must *release* the pages when they are no longer needed. Once a task releases any of its pages, it is no longer permitted to acquire more shared memory, ensuring monotonic usage patterns that simplify scheduling. When the current task signals its release, MPK can immediately preallocate available pages for the next task and begin data prefetching. This design enables fine-grained, on-demand allocation of shared memory resources within the mega-kernel execution model.

Cross-task software pipelining. To enable software pipelining across tasks executed on the same worker (§ [2.1\)](#page-1-1), MPK decomposes each task into a *pre-loading* phase and a *compute* phase. The pre-loading phase issues data transfer instructions to fetch a chunk of the required tensor from device memory into shared memory, initializing the intra-task software pipeline but performing no computation.

MPK opportunistically overlaps the compute phase of the current task *T*<sup>1</sup> with the pre-loading phase of the subsequent task *T*<sup>2</sup> when two conditions hold: (1) *T*<sup>1</sup> has already issued all of its own data-transfer instructions, and (2) sufficient sharedmemory pages are available for *T*2's pre-loading phase. Note that such a pipeline does not interfere with *T*1's execution, as MPK inserts appropriate intra-SM synchronization barriers to ensure that *T*2's memory transfers do not conflict with ongoing data transfers for *T*1.

Pre-fetching task descriptions. Each worker maintains both JIT and AOT task queues in device memory. Every task is associated with a task description that encodes its input tensors, output tensors, and configuration parameters; in our current implementation, each description occupies 352 bytes (§ [6.1\)](#page-8-0). To reduce enqueue/dequeue latency and hide devicememory access costs, MPK employs a lightweight prefetching mechanism that retrieves upcoming task descriptions into shared memory before they are needed.

