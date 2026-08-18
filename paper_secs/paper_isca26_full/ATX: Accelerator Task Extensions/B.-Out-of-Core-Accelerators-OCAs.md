# B. Out-of-Core Accelerators (OCAs)

The second approach to integrate accelerators in CPUs is to place them outside of the core pipeline [6], [21], [24], [27], [29], [33], [52], [53], [87]. Examples of commercial offerings following this approach are the Intel Data Streaming Accelerator (DSA) and In-memory Analytics Accelerator (IAA) [53], [87], the Intel NPU [24], and the GPU in the AMD APU [21], [52]. These accelerators are placed on the chip far away from the cores (sometimes in the corner of the die [82], [87]), and can only access the LLC and main memory. Other accelerator proposals [6], [29] can also access private cache levels.

Such accelerators interact with the system's state by following what we call the *Out-of-Core Accelerator* (OCA) abstraction. OCAs can read their inputs from the memory system and write their outputs to the memory system. The execution of a task may leave state in the accelerator after the task completes [39].

An advantage of OCAs is that they can access the memory system with high performance, as they can utilize specialized out-of-core memory access hardware [29], [33], [72]. A limitation of OCAs is that they cannot be invoked out-of-order or speculatively by CPU cores, as there is no way of undoing the effects of their execution in case of misspeculation—e.g., if an exception or branch misprediction happens. In many cases, the communication between a core and an OCA is performed with polling [8], [53], where the accelerator is treated as an MMIO device and is invoked using non-speculative CPU stores. The CPU first checks the accelerator status by reading accelerator registers and, when the accelerator is ready, starts a new task by writing to other accelerator registers. Fences are required between accelerator invocations (which are memory stores) and subsequent polling on the accelerator (which are memory loads), to prevent store→load reordering [27].

To understand the inefficiencies of the current approach for CPU cores to communicate with OCAs, consider Figure 2, which shows three examples of the contents of a core's reorder buffer (ROB). In the figure, the ROB head is at the top. In Figure 2(a), we show that an OCA invocation instruction (which is a regular memory store instruction) is blocked from

<span id="page-2-1"></span>![](_page_2_Figure_8.jpeg)

Fig. 2: Examples of core ROB state when invoking an OCA.

being issued until it reaches the head of the ROB—and hence is not speculative. Figure 2(b) shows that the invocations of two OCAs cannot overlap, since only one invocation can be at the ROB head at a time. Note that this does not mean that OCA tasks cannot overlap, but rather that the operations of starting tasks cannot overlap. Finally, Figure 2(c) shows that, for correctness, a fence is needed between an OCA invocation (i.e., a write) and a subsequent load to check the status of that OCA. Unfortunately, the fence also blocks normal (non-OCA) loads (e.g., the one at ROB index 3).

These communication inefficiencies do not matter for long tasks. However, they are detrimental to performance for small tasks that are finely interleaved with core execution. In an ideal scenario, OCA invocations would behave like normal instructions: they would execute speculatively and out-of-order when their inputs are available, and be squashed and reexecuted on branch mispredictions or exceptions.

# B. Out-of-Core Accelerators (OCAs)

The second approach to integrate accelerators in CPUs is to place them outside of the core pipeline [6], [21], [24], [27], [29], [33], [52], [53], [87]. Examples of commercial offerings following this approach are the Intel Data Streaming Accelerator (DSA) and In-memory Analytics Accelerator (IAA) [53], [87], the Intel NPU [24], and the GPU in the AMD APU [21], [52]. These accelerators are placed on the chip far away from the cores (sometimes in the corner of the die [82], [87]), and can only access the LLC and main memory. Other accelerator proposals [6], [29] can also access private cache levels.

Such accelerators interact with the system's state by following what we call the *Out-of-Core Accelerator* (OCA) abstraction. OCAs can read their inputs from the memory system and write their outputs to the memory system. The execution of a task may leave state in the accelerator after the task completes [39].

An advantage of OCAs is that they can access the memory system with high performance, as they can utilize specialized out-of-core memory access hardware [29], [33], [72]. A limitation of OCAs is that they cannot be invoked out-of-order or speculatively by CPU cores, as there is no way of undoing the effects of their execution in case of misspeculation—e.g., if an exception or branch misprediction happens. In many cases, the communication between a core and an OCA is performed with polling [8], [53], where the accelerator is treated as an MMIO device and is invoked using non-speculative CPU stores. The CPU first checks the accelerator status by reading accelerator registers and, when the accelerator is ready, starts a new task by writing to other accelerator registers. Fences are required between accelerator invocations (which are memory stores) and subsequent polling on the accelerator (which are memory loads), to prevent store→load reordering [27].

To understand the inefficiencies of the current approach for CPU cores to communicate with OCAs, consider Figure 2, which shows three examples of the contents of a core's reorder buffer (ROB). In the figure, the ROB head is at the top. In Figure 2(a), we show that an OCA invocation instruction (which is a regular memory store instruction) is blocked from

<span id="page-2-1"></span>![](_page_2_Figure_8.jpeg)

Fig. 2: Examples of core ROB state when invoking an OCA.

being issued until it reaches the head of the ROB—and hence is not speculative. Figure 2(b) shows that the invocations of two OCAs cannot overlap, since only one invocation can be at the ROB head at a time. Note that this does not mean that OCA tasks cannot overlap, but rather that the operations of starting tasks cannot overlap. Finally, Figure 2(c) shows that, for correctness, a fence is needed between an OCA invocation (i.e., a write) and a subsequent load to check the status of that OCA. Unfortunately, the fence also blocks normal (non-OCA) loads (e.g., the one at ROB index 3).

These communication inefficiencies do not matter for long tasks. However, they are detrimental to performance for small tasks that are finely interleaved with core execution. In an ideal scenario, OCA invocations would behave like normal instructions: they would execute speculatively and out-of-order when their inputs are available, and be squashed and reexecuted on branch mispredictions or exceptions.

