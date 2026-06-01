# dataflow Sim

buffer+ALU组成的PE就是dataflow架构，因为没有使用细粒度指令的SIMD pipeline。

[GitHub - scalesim-project/SCALE-Sim: Repository to host and maintain SCALE-Sim code · GitHub](https://github.com/scalesim-project/SCALE-Sim)

ReDas: A Lightweight Architecture for Supporting Fine-Grained Reshaping and Multiple Dataflows on Systolic Array

## 21：Gemmini（chiplet-tile Acc+host）

[https://github.com/ucb-bar/gemmini](https://github.com/ucb-bar/gemmini)

虚拟内存提供编程易用性，更容易端到端测试负载。

> **[图片提取文字 (image.png)]:**
> ciency, and extensibility. Unlike existing DNN accelerator generators that focus on standalone accelerators, Gemmini also provides a complete solution spanning both the hardware and software stack, and a complete SoC integration that is compatible with the RISC-V ecosystem. In addition, Gemmini implements a multi-level software stack with an easy-to-use programming interface to support different programming requirements, as well as tight integration with Linuxcapable SoCs which enable the execution of any arbitrary software. Gemmini-generated accelerators have been successfully fabricated
> 
> in both TSMC 16nm FinFET and Intel 22nm FinFET Low Power
![image.png](dataflow%20Sim/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: Gemmini hardware architectural template overview.
![image.png](dataflow%20Sim/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Systolic spatial array (TPU-like)
> 
> Parallel vector engines (NVDLA-like)
![image.png](dataflow%20Sim/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2: Microarchitecture of Gemmini's two-level spatial array.
![image.png](dataflow%20Sim/image%203.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5: Example dual-core SoC with a Gemmini accelerator attached to each CPU, as well as a shared L2 cache and standard peripherals.
![image.png](dataflow%20Sim/image%204.png)

> **[图片提取文字 (image.png)]:**
> | Component size        | Area $(\mu m^2)$ | % of<br>System<br>Area |
> |-----------------------|------------------|------------------------|
> | Spatial Array (16x16) | 116K             | 11.3%                  |
> | Scratchpad (256 KB)   | 544K             | 52.9%                  |
> | Accumulator (64 KB)   | 146K             | 14.2%                  |
> | CPU (Rocket, 1 core)  | 171K             | 16.6%                  |
> | Total                 | 1,029K           | 100.0%                 |
> |                       |                  |                        |
> 
> ![](_page_0_Picture_1.jpeg)
> 
> (a) Area breakdown.
> 
> (b) Layout.
> 
> Fig. 6: Area breakdown and layout of accelerator with host CPU.
![image.png](dataflow%20Sim/image%205.png)

## 25：PyTorchSim

[https://github.com/PSAL-POSTECH/PyTorchSim](https://github.com/PSAL-POSTECH/PyTorchSim)

Pytorch编译。TOG建模延迟（片上访存和计算延迟可以线下确定）。TPU指令。VCIX接口。多租户。

编译可能替代trace生成（event生成）或指令生成。

> **[图片提取文字 (image.png)]:**
> requirements that make it difficult to accomplish this goal while maintaining cycle accuracy. Notably, the demand for high-speed simulation directly contradicts the need for detailed, cycle-accurate modeling of the core, interconnect, and DRAM. In particular, as data-dependent timing models are typically implemented with Instruction-Level Simulation (ILS) [37, 65] approach, they do not achieve high speed. Furthermore, unlike CPU and GPU architectures, NPUs lack a widely adopted ISA, which has also led to the lack of open-source deep learning compilers for NPUs that are mature enough to support various DNNs for both inference and training. Consequently, NPU simulators often rely on custom DNN description formats rather than ML frameworks such as Py-Torch [57, 89, 94].
> 
> However, there are several challenges and conflicts among the
> 
> NPU simulation framework, to facilitate effective design space exploration for future NPUs. It consists of a high-level request scheduler, a compiler backend for PyTorch 2 [31], and functional/timing NPU models that can target various dataflows. Since PyTorch is used in the majority of recent deep learning projects [88], in most cases, we do not require the user to convert existing DNN implementations into a special format, unlike existing simulators [57, 85, 94].
> 
> To address these challenges, we propose *PyTorchSim*, a novel
> 
> We build PyTorchSim upon the compilation infrastructure of PyTorch 2, which captures a computational graph in FX graph format for forward/backward passes and lowers it into Aten IR in the frontend. Its Inductor backend then lowers it into loop-level IR for various device-specific backends. Our codegen backend lowers it
![image.png](dataflow%20Sim/image%206.png)

> **[图片提取文字 (image.png)]:**
> PyTorchSim can also flexibly model different NPU architectures. As a generic and extensible ISA targeted by our compiler backend, our NPU model adopts an extended RISC-V ISA. It supports custom instructions for DMA and dataflow unit (e.g., systolic array)
> 
> tom instructions for DMA and dataflow unit (e.g., systolic array) operations while exploiting existing scalar/vector instructions to express various operations in modern DNNs. As a generic interface between the dataflow unit and the rest of the NPU core, we adopt the approach of SiFive Vector Coprocessor Interface (VCIX) [14]. Thus, while we currently support the systolic array (SA) dataflow
> 
> unit, users can replace it with other units with different microarchi-
> 
> tectures. It is even possible to replace the NPU core model entirely, as we demonstrate with a heterogeneous NPU case study (§5.1). After compilation for this target NPU, the machine code can be executed one instruction at a time with ILS, by Gem5 [37, 76] (for NPU core's timing model) and Spike [1] (for NPU core's functional model) simulators that we have extended and integrated to accurately model NPU cores. The compiler-generated binary is also used for functional correctness and DNN model accuracy validation. However, such a detailed simulation approach inevitably
> 
> runs slowly, conflicting with the goal of achieving high simulation
> 
> pose an alternative approach called *Tile-Level Simulation (TLS)*.
> 
> To realize fast simulation without losing accuracy, we also pro-
> 
> speed.
> 
> The TLS method is proposed based on the observation that the scalar, vector, and matrix instructions between load/store DMAs can be executed with *deterministic* latencies, because the operands are accessed from core-internal SRAM after DMA transfers from DRAM [54, 63]. Since the operations are performed in tensor tile-granularity [89], the deterministic latency of a tile-operation can be obtained *offline* and repeatedly used during different simulation
> 
> runs to achieve high speed. In PyTorchSim, the tile-operation laten-
> 
> cies are obtained through our extended Gem5 and Spike simulators.
> 
> However, in contrast to the tile's compute latency, the latencies
> 
> of the DMAs are *non-deterministic* due to contention. Thus, they
> 
> are modeled online using cycle-accurate NoC and DRAM simu-
> 
> lators [59, 78]. To express the tile compute operations and their
> 
> dependencies with DMAs in a DNN, we propose a *Tensor Operation Graph (TOG)*. The TOG can be automatically generated by our Py-Torch 2 compiler backend, and then be executed on our NPU timing simulator called *TOGSim* with high speed and high accuracy. Furthermore, we provide a novel observation that, even if the tile operation is data-dependent (e.g., sparse tensors), its compute
> 
> latency is deterministic for *each particular* tile, while it can vary *across* tiles. To enable TLS for this case, our extended Spike simulator is used with Gem5 to obtain the latencies for each tile and generate an auxiliary tile-latency file associated with the TOG. The TOGSim can then use them to achieve both high simulation speed and accuracy. It also extends to different core models (§3.3.2), even including sparse cores (§5.1). After the TOGs are obtained for different datasets, they can be reused over simulations with different NPU
> 
> configurations as in typical exploration use cases. For multi-model
![image.png](dataflow%20Sim/image%207.png)

> **[图片提取文字 (image.png)]:**
> Table 1: Comparison of the features of different NPU simulators (ICNT: interconnect).
> 
> |                    | High     | Multi-     | Multi-DNN | Cycle-accurate | General    | Compiler | Training | Base ISA | Data-dependent | Model        |
> |--------------------|----------|------------|-----------|----------------|------------|----------|----------|----------|----------------|--------------|
> |                    | speed    | core       | tenancy   | DRAM&ICNT      | vector ops | support  | support  |          | timing model   | input format |
> | Accel-Sim [65]     | X        | ✓          | X         | Both           | ✓          | ✓        | ✓        | PTX/SASS | ✓              | Instr. trace |
> | mNPUsim [57]       | X        | <b>√</b> ‡ | <b>✓</b>  | DRAM           | X          | X        | X        | X        | X              | Custom       |
> | SCALE-Sim v3 [91]  | ✓        | ✓          | X         | DRAM           | Х          | Х        | X        | X        | Partial**      | Custom       |
> | SMAUG [104]        | Х        | ✓          | X         | Both           | ✓          | Х        | Х        | x86      | Х              | Custom       |
> | SST-STONNE [5, 85] | X        | X          | X         | Both           | X          | X        | X        | Custom   | ✓              | PyTorch***   |
> | MAESTRO [69]       | ✓        | Х          | X         | <b>X</b> *     | X          | X        | X        | X        | ✓              | Custom       |
> | Timeloop [89]      | <b>✓</b> | ✓          | X         | <b>X</b> *     | X          | X        | X        | X        | X              | Custom       |
> | Sparseloop [103]   | ✓        | ✓          | X         | <b>X</b> *     | X          | X        | X        | X        | Partial**      | Custom       |
> | GeneSys [55]       | ✓        | Х          | X         | DRAM           | ✓          | ✓        | X        | Custom   | X              | ONNX         |
> | PyTorchSim         | <b>✓</b> | ✓          | ✓         | Both           | ✓          | ✓        | ✓        | RISC-V   | ✓              | PyTorch      |
> 
> While Accel-Sim is a GPGPU simulator, it is included because GPUs are widely used to accelerate DNNs. We exclude simulators, such as LLMServingSim [45] and vTrain [34], that do not have an NPU microarchitecture model and rely on an external NPU simulator or real GPU profiling results.
> 
> <sup>\*</sup>mNPUsim's core model is limited to supporting only a batch size of one.
> 
> <sup>\*</sup>Their DRAM model does not model the microarchitectural details (e.g., row buffer hits/misses) while the network model ignores contention.
> 
> <sup>\*\*</sup>Sparseloop estimates compute cycles based on tensor sparsity statistics, rather than actual data values. SCALE-Sim v3 lacks unstructured sparsity support.
> \*\*\*STONNE's PyTorch interface only supports GEMM and CONV, while other operations are ignored.
![image.png](dataflow%20Sim/image%208.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Overview of PyTorchSim framework.
> 
> Table 2: Simulator usages and corresponding components.
> 
> | Usage                                            | Simulator(s)   |
> |--------------------------------------------------|----------------|
> | DNN accuracy validation                          | Spike          |
> | TOG (§3.7) generation (no data dependence)       | Gem5           |
> | TOG (§3.7) generation (data-dependent)           | Gem5 + Spike   |
> | Inference performance evaluation                 | TOGSim         |
> | Single-iteration training performance evaluation | TOGSim         |
> | Full training performance evaluation             | TOGSim + Spike |
> 
> ## 3.2 Example Use Cases and Limitations
> 
> PyTorchSim is a versatile simulator that can support various NPU HW/SW evaluation scenarios beyond the cases supported by prior NPU simulators (§5). We describe some key example use cases below, while this is not an exhaustive list:
> 
> **HW** microarchitecture evaluation. Any HW components of Py-TorchSim, including NPU core, scalar/vector units, DMA engines, on-/off-chip memory, and interconnect can be configured or modified to study different architectures. For instance, we present a case
![image.png](dataflow%20Sim/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: NPU core microarchitecture model.
![image.png](dataflow%20Sim/image%2010.png)

> **[图片提取文字 (image.png)]:**
> |                                                  | 7 bits | 5 bits    | 3 bits | 5 bits          | 5 bits         | 7 bits  |
> |--------------------------------------------------|--------|-----------|--------|-----------------|----------------|---------|
> | (a) mvin/mvout                                   | opcode | Χ         | func3  | mm addr reg     | sram addr reg  | funct7  |
> | (b) config                                       | opcode | Х         | func3  | stride info reg | shape info reg | funct7  |
> | (c) push                                         | opcode | Х         | func3  | Х               | src vreg       | vfunct7 |
> | (d) pop                                          | opcode | dest vreg | func3  | Х               | Х              | vfunct7 |
> | (e) SFU Inst.                                    | opcode | dest vreg | func3  | Х               | src vreg       | vfunct7 |
> | Figure 3: Custom instructions for our NPU model. |        |           |        |                 |                |         |
![image.png](dataflow%20Sim/image%2011.png)

> **[图片提取文字 (image.png)]:**
> ## 3.2 Example Use Cases and Limitations
> 
> PyTorchSim is a versatile simulator that can support various NPU HW/SW evaluation scenarios beyond the cases supported by prior NPU simulators (§5). We describe some key example use cases below, while this is not an exhaustive list:
> 
> **HW microarchitecture evaluation.** Any HW components of Py-TorchSim, including NPU core, scalar/vector units, DMA engines, on-/off-chip memory, and interconnect can be configured or modified to study different architectures. For instance, we present a case study on a heterogeneous NPU with dense-sparse matrix units and the impact of the memory system (§5.1).
> 
> **SW optimization.** SW optimizations can have a significant performance impact. We show studies on compute-DMA overlap and tensor layout modification (§5.3).
> 
> **HW/SW co-design.** New HW features often require SW support. For example, to support a multi-dimensional DMA feature [62], our compiler generates the necessary DMA commands (§3.6.3).
> 
> **DNN training study.** Through the compiler support, we enable studies on the training performance (§5.5).
> 
> Multi-model tenancy study. In contrast to other simulators, we support different scenarios with concurrent multi-DNN execution (e.g., performance interference between different DNNs (§5.2)).
![image.png](dataflow%20Sim/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 4: (a) Compilation workflow with simplified example IR pseudo-codes. (b) Example TOG.
![image.png](dataflow%20Sim/image%2013.png)

## 24：ONNXim

[https://github.com/PSAL-POSTECH/ONNXim?tab=readme-ov-file](https://github.com/PSAL-POSTECH/ONNXim?tab=readme-ov-file)

分时、分空的多应用。

指令：im2col、gemm、conv、vector、dma。PSAL-

tile=npu=chiplet。

输入ONNX计算图，模拟性能。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1. Overview of ONNXim simulator (a–c) and SA operation timing model (d–f). (d) An SA instruction (with green input) begins execution. (e) Another SA instruction (with shaded blue input) can begin execution with overlap. (f) the first SA instruction (green) finishes.
![image.png](dataflow%20Sim/image%2014.png)

> **[图片提取文字 (image.png)]:**
> model description format. We leverage the ONNX runtime to optimize the graph and Timeloop [3] to optimize the mapping of operations to NPU cores. Custom fuse kernels (e.g., FlashAttention [13]) are modeled by adding them to our template kernel library. ONNXim also supports extensible global scheduler policies, which are necessary for studying multi-tenant workloads.
> 
> ONNXim achieves high simulation speed based on the observation that typical NPU cores with systolic arrays (SAs) process tensor tiles from on-chip scratchpad memory with deterministic latency [2]. Since its compute latency can be determined by the sizes of tensor tiles and the SA, we avoid modeling all individual arithmetic operations in a fine-grained manner, unlike conventional CPU/GPU simulators, while still preserving simulation accuracy. Meanwhile, shared resources with contention are modeled in detail using cycle-accurate simulators as they introduce non-determinism. While prior work [8] adopts a similar approach, ONNXim improves upon it in three significant ways. First, ONNXim enables DNN inferences with dynamic input shapes, which is important for modeling the growing key-value cache in large language models (LLMs). Second, we enable a more diverse set of operations beyond GEMM and CONV, including layer normalization and skip connections, which can collectively take up a significant portion of runtime [14]. Third, in ONNXim, the generation and execution of the dynamic instruction sequence for cores are optimized for fast simulation speed.
![image.png](dataflow%20Sim/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ## TABLE I NPU CONFIGURATIONS AND DNNs USED FOR EVALUATION
> 
> | NPU Configuration        |                                                     |                |                    |
> |--------------------------|-----------------------------------------------------|----------------|--------------------|
> | Parameter                |                                                     | Mobile NPU     | Server NPU         |
> | SA size, # of cores      |                                                     | 8×8, 4 @ 1 GHz | 128×128, 4 @ 1 GHz |
> | Vector unit (# of ALUs)  |                                                     | 128            | 2048               |
> | Scratchpad (acc.) / core |                                                     | 64 KB (16 KB)  | 32 MB (4 MB)       |
> | DRAM                     |                                                     | DDR4 (12 GB/s) | 2 HBM2 (614 GB/s)  |
> | Evaluated DNNs           |                                                     |                |                    |
> | DNN Model                | Description                                         |                |                    |
> | ResNet-18/50             | (C:3, H:224, W:224) input image                     |                |                    |
> | GPT-3 Small              | 512-token prefill (P), 100-token generation (G)     |                |                    |
> | BERT-large               | Process a 512-token input prompt                    |                |                    |
> | Llama-3 (8B)             | Generating a single token with a 1024-token context |                |                    |
![image.png](dataflow%20Sim/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ## A. Front End
> 
> Model optimization and lowering: The input DNN can be first optimized using the ONNX runtime's offline graph optimizer to reduce computation and memory traffic. Its optimization techniques include constant folding, redundancy elimination, and fusion, such as fusing CONV/MHA (Multi-Head Attention) with batch/layer normalization layers and/or skip connection. Then, ONNXim lowers each node (or layer) in the ONNX graph to multiple tile operations using the tiled mapping scheme chosen by a mapping optimizer (e.g., Timeloop [3]) and the kernel templates in our template library. The library provides templates for basic layers, such as GEMM and CONV, as well as custom fused kernels (e.g., FlashAttention [13]) that users can choose from. Users can also add their own custom kernels written in our core ISA.
> 
> ISA: The kernel templates for tile operations use our NPU core's ISA to specify operations using the following instructions: 1) MVIN/MVOUT for DMA load/store between scratchpad memory and DRAM, 2) GEMM Preload for loading weights into the SA, 3) GEMM for performing matrix multiplication on the SA, 4) IM2COL for performing image-to-column transformations, and 5) vector operations (e.g., GELU). The ISA is an extension of the Gemmini ISA [9], with additional instructions for vector operations and activation functions.
> 
> Scheduler: To enable multi-tenant scenarios, ONNXim takes a request trace in JSON format that describes the inference request stream with various DNNs, batch sizes, and timestamps. The requests can be scheduled with different scheduling policies. A time-sharing policy schedules a layer from one request at a time before switching to a layer from another request in a round-robin fashion. Alternatively, a spatial-sharing policy partitions the NPU cores among multiple models, enabling Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded
![image.png](dataflow%20Sim/image%2017.png)

> **[图片提取文字 (image.png)]:**
> them to execute concurrently. Users can also easily add new, sophisticated policies using our modular implementation.
> 
> In the scheduler (Fig. 1(a)), a *layer-level queue* exists for each inference request and tracks layer-wise dependency information
> 
> from the ONNX graph. A layer without unresolved dependencies can then be popped from the queue and lowered into multiple tile-level operations that are pushed into a tile-level queue. Multiple independent layers can be pushed into the tile-level queue, allowing for concurrent execution. The tile-level queue exists per user-specified NPU core partition, such that an idle NPU core context can pop a tile-level operation and begin execution,
> 
> based on the scheduling policy.
![image.png](dataflow%20Sim/image%2018.png)

SA使用event-level而非cycle-level，因为计算延迟通过尺寸可预测，似乎可行。

> **[图片提取文字 (image.png)]:**
> organization with an SA, weight buffer, scratchpad memory, accumulator, and vector units (Fig. 1(c)). The accumulator includes its own SRAM as well as arithmetic units for accumulation operations. The core receives a tile operation from the global scheduler and executes its instructions from the instruction buffer. To support double-buffering, there are two instruction buffers and each PE in the SA also has two weights
> 
> for the two contexts, adopting the Gemmini SA model [9]. The
> 
> scratchpad and accumulator memory also have two banks and
> 
> their interfaces match the width (w) and height (h) of the SA to
> 
> avoid bottlenecks. An instruction scheduler issues instructions
> 
> Core Organization. ONNXim models a typical NPU core's
> 
> to the DMA engine or execution pipeline when there are no structural or data hazards, by tracking the unit's availability and the dependencies between compute and DMA instructions.
> 
> Core implementation: ONNXim improves the simulation speed for NPU core computation by adopting an event-driven approach and avoiding the cycle-by-cycle simulation of the SA and vector unit as in conventional CPU/GPU simulators. The SA can be implemented with different dataflows: weight-stationary, input-stationary, and output-stationary. Their relative
> 
> performance depends on the dimensions of the SA and input ten-
> 
> sor sizes [2]. We assume the widely-adopted weight-stationary
> 
> SA [15] with dimensions  $w \times h$ . For an SA operation, the
> 
> weights are first preloaded into the SA over h cycles. Then the SA is fed with an input matrix tile of size  $l \times h$  in a skewed manner (Fig. 1(d)), taking (l-1)+w+h cycles for the computation (Fig. 1(f)). During the SA operation, the SA operation for the next tile can be partially overlapped to minimize pipeline bubbles if the next input tile is ready in the scratchpad memory and the preloaded weight for the current SA operation can be reused. Specifically, the next tile can start the SA operation after the
> 
> preloaded weight for the current SA operation can be reused. Specifically, the next tile can start the SA operation after the top row of the current tile is pushed to the SA (Fig. 1(e)). The SA events are generated deterministically based on this timing model in ONNXim. Thus, high SA utilization can be achieved for large matrix multiplications with a long steady-state operation. Vector instructions are also modeled to use
![image.png](dataflow%20Sim/image%2019.png)

> **[图片提取文字 (image.png)]:**
> operands from on-chip SRAM and, therefore, have deterministic latency calculated from the operand's size, vector unit width, and instruction latency configurations. In addition, ONNXim supports heterogeneous NPU cores with different SA and SRAM configurations to model NPUs designed to execute DNNs with different scales efficiently [8].
> 
> Shared resources: We model the shared off-chip DRAM using Ramulator 2.0 [16], a fast cycle-level simulator that can model various DRAM devices (e.g., DDR, HBM, and LPDDR), to accurately model memory traffic contention. For tensor data movement to/from the cores, the per-core DMA engine generates memory requests with the DRAM-access-granularity and sends them to memory controllers.
> 
> For NoCs, ONNXim supports a versatile cycle-accurate simulator, Booksim [17], as well as a simple latency and bandwidth model. While a detailed NoC model may not be necessary for NPUs with abundant on-chip bandwidth, multi-die NPUs [18] with limited die-to-die interconnect bandwidth require an accurate interconnect model. By integrating these cycle-level models, ONNXim can accurately model contention for shared resources.
![image.png](dataflow%20Sim/image%2020.png)

yolo、fcnn，周三开会。