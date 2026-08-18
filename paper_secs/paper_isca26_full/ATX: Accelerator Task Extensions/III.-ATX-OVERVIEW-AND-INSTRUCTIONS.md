# III. ATX OVERVIEW AND INSTRUCTIONS

To support NCAs, we propose a general framework that we call *Accelerator Task Extensions* (ATX). ATX consists of a set of instructions and hardware extensions to support the interaction between a CPU core, multiple NCAs, and the cache subsystem. ATX unlocks the full potential of NCAs by supporting speculative and out-of-order NCA invocation, and accelerated data provision from the memory system. In this section, we first describe the high-level interaction between cores, NCAs, and the memory system. We then detail the ATX instructions, which cores use to invoke and control NCAs.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Fig. 3: ATX instructions and an ATX NCA task.

## A. High-Level Overview

A single ATX instruction executed by a core invokes an NCA *task* and returns the task's output in core registers. Figure 3(a) shows ATX instructions (called ATX Ins). As an NCA task executes, it reads data from the memory system without core involvement, performs computation, and then returns a result. As represented in Figure 3(b), the reads from memory use configurable and potentially inter-dependent memory streams [79]. An ATX instruction includes input core register operands that carry metadata encoding all the necessary information to configure those streams. We discuss how this is done in Section IV. An ATX instruction also defines output core register operands to which the NCA will write output data when the task completes. An ATX instruction retires from the ROB when three conditions occur: the task has completed, the task output has been written to core registers, and the instruction is at the head of the ROB. An NCA never writes to memory.

From a core's viewpoint, an ATX instruction behaves like a load instruction that loads data from memory into registers. The computations performed by the task are invisible to the core. The core treats ATX instructions like normal loads; they can be issued speculatively and out-of-order, as soon as the instructions that produce their register inputs have finished. If an ATX instruction is squashed due to a wrong speculation, it is interrupted in the NCA without affecting architectural state.

Figure 4 shows a high-level overview of the interaction between a CPU core, potentially several NCAs that it controls, and the memory system. The figure includes the ATX Unified Transfer Engine (UTE), a hardware module that interfaces the three subcomponents. There is one UTE per core. The UTE schedules the tasks invoked by the CPU cores on the appropriate NCAs. Further, it reads task inputs from the memory system on behalf of the NCAs, forwards the fetched data to the NCA input buffers (i.e., scratchpads), and routes task outputs from NCAs back to the CPU. We describe the UTE in detail in Section IV.

While the NCA abstraction does not specify the location of the accelerator in the cache hierarchy of a core, the ATX design that we propose places the UTE and NCAs next to the L2 cache. This is a "goldilocks zone" that enables fast core-accelerator communication, while limiting intrusion into the core pipeline. Further, such proximity enables the UTE to share the L2 TLB with the core and utilize the L2 for buffering and reusing data—minimizing the need for large scratchpads.

<span id="page-3-1"></span>![](_page_3_Figure_7.jpeg)

Fig. 4: Interaction between a core, multiple NCAs, and the memory system.

The interaction between core, NCAs, and memory system is as follows. At step ①, the UTE receives an ATX instruction from the core's ATX Port for an NCA task. The UTE generates the addresses of the input data of the task, utilizing metadata in the ATX instruction, and generates read requests directed to the core's L2 at step ②. We describe the exact mechanism of address generation later. Requests use the existing memory access flow when they encounter an L2 cache miss.

The UTE operates on virtual addresses and uses the core's L2 TLB and MMU for memory address translation similar to [27], [29], [33]. Also, when the UTE reads the L2, it always gets the latest value of the data. If the requested line is dirty in the core's L1, existing coherence hardware mechanisms will provide the latest version of the line from L1. The UTE never writes to the L2 or memory.

At step ③, the input data received from the L2 is written to the input buffers of the appropriate NCA. Once all the input data is collected, the UTE signals the NCA to start execution. When the NCA execution completes, the UTE is notified and, at step ④, the UTE reads the output data. At that point, the NCA is freed and can accept a new task. Then, at step ⑤, the NCA output is returned to the core through the ATX port and written to a core register (specified in the ATX instruction) in the core's physical register file (PRF). The ATX instruction then completes. The core may use the output data for other computations or write it to the memory system (step ⑥).

# III. ATX OVERVIEW AND INSTRUCTIONS

To support NCAs, we propose a general framework that we call *Accelerator Task Extensions* (ATX). ATX consists of a set of instructions and hardware extensions to support the interaction between a CPU core, multiple NCAs, and the cache subsystem. ATX unlocks the full potential of NCAs by supporting speculative and out-of-order NCA invocation, and accelerated data provision from the memory system. In this section, we first describe the high-level interaction between cores, NCAs, and the memory system. We then detail the ATX instructions, which cores use to invoke and control NCAs.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Fig. 3: ATX instructions and an ATX NCA task.

## A. High-Level Overview

A single ATX instruction executed by a core invokes an NCA *task* and returns the task's output in core registers. Figure 3(a) shows ATX instructions (called ATX Ins). As an NCA task executes, it reads data from the memory system without core involvement, performs computation, and then returns a result. As represented in Figure 3(b), the reads from memory use configurable and potentially inter-dependent memory streams [79]. An ATX instruction includes input core register operands that carry metadata encoding all the necessary information to configure those streams. We discuss how this is done in Section IV. An ATX instruction also defines output core register operands to which the NCA will write output data when the task completes. An ATX instruction retires from the ROB when three conditions occur: the task has completed, the task output has been written to core registers, and the instruction is at the head of the ROB. An NCA never writes to memory.

From a core's viewpoint, an ATX instruction behaves like a load instruction that loads data from memory into registers. The computations performed by the task are invisible to the core. The core treats ATX instructions like normal loads; they can be issued speculatively and out-of-order, as soon as the instructions that produce their register inputs have finished. If an ATX instruction is squashed due to a wrong speculation, it is interrupted in the NCA without affecting architectural state.

Figure 4 shows a high-level overview of the interaction between a CPU core, potentially several NCAs that it controls, and the memory system. The figure includes the ATX Unified Transfer Engine (UTE), a hardware module that interfaces the three subcomponents. There is one UTE per core. The UTE schedules the tasks invoked by the CPU cores on the appropriate NCAs. Further, it reads task inputs from the memory system on behalf of the NCAs, forwards the fetched data to the NCA input buffers (i.e., scratchpads), and routes task outputs from NCAs back to the CPU. We describe the UTE in detail in Section IV.

While the NCA abstraction does not specify the location of the accelerator in the cache hierarchy of a core, the ATX design that we propose places the UTE and NCAs next to the L2 cache. This is a "goldilocks zone" that enables fast core-accelerator communication, while limiting intrusion into the core pipeline. Further, such proximity enables the UTE to share the L2 TLB with the core and utilize the L2 for buffering and reusing data—minimizing the need for large scratchpads.

<span id="page-3-1"></span>![](_page_3_Figure_7.jpeg)

Fig. 4: Interaction between a core, multiple NCAs, and the memory system.

The interaction between core, NCAs, and memory system is as follows. At step ①, the UTE receives an ATX instruction from the core's ATX Port for an NCA task. The UTE generates the addresses of the input data of the task, utilizing metadata in the ATX instruction, and generates read requests directed to the core's L2 at step ②. We describe the exact mechanism of address generation later. Requests use the existing memory access flow when they encounter an L2 cache miss.

The UTE operates on virtual addresses and uses the core's L2 TLB and MMU for memory address translation similar to [27], [29], [33]. Also, when the UTE reads the L2, it always gets the latest value of the data. If the requested line is dirty in the core's L1, existing coherence hardware mechanisms will provide the latest version of the line from L1. The UTE never writes to the L2 or memory.

At step ③, the input data received from the L2 is written to the input buffers of the appropriate NCA. Once all the input data is collected, the UTE signals the NCA to start execution. When the NCA execution completes, the UTE is notified and, at step ④, the UTE reads the output data. At that point, the NCA is freed and can accept a new task. Then, at step ⑤, the NCA output is returned to the core through the ATX port and written to a core register (specified in the ATX instruction) in the core's physical register file (PRF). The ATX instruction then completes. The core may use the output data for other computations or write it to the memory system (step ⑥).

