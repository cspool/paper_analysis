# C. Processing Element (PE)

Figure 3 shows the internal architecture of a PE. Each PE includes two RISC-V cores, fixed-function units for accelerating compute and data movement, and internal memory with a memory bridge connecting all components.

**Memory Bridge (MB):** The MB provides data and configuration connectivity between all components in the PE through an internal NoC. It also contains peripherals such as an interrupt controller, machine timer, and debug/trace modules.

**Local Memory:** Each PE includes 512 KB of fast local memory (LS). The LS is software-managed and split into Circular Buffers (CBs) of controllable sizes.

RISC-V Cores: The RISC-V cores execute the application code and issue commands to the Command Processor for the fixed-function units. MTIA 300 has two 64 B-wide vector cores that provide additional SIMD throughput and a symmetric programming model, allowing the same code sequence to run on either core. MTIA uses an asynchronous

TABLE I: Comparing the specifications of MTIA 300 and MTIA-2i.

|                   |                     | MTIA 300                             | MTIA-2i                        |  |
|-------------------|---------------------|--------------------------------------|--------------------------------|--|
| Frequency         |                     | 1.9 GHz                              | 1.35 GHz                       |  |
| Instances         |                     | 7.2B gates, 511M FLOPS               | 2.35B gates, 103M FLOPS        |  |
| Area              |                     | Compute chiplet: 25.6mm x 31.4mm     |                                |  |
|                   |                     | Network chiplet (2x): 25.6mm x 9.3mm | 25.6mm x 16.4mm                |  |
| Package           |                     | 77.5mm x 77.5mm                      | 50mm x 40mm                    |  |
|                   |                     | (50.3mm x 51.9mm 3.2x interposer)    |                                |  |
| Voltage           |                     | 0.85 V                               | 0.85 V                         |  |
| TDP               |                     | 912W (667W typical)                  | 85W (65W typical)              |  |
| Host connection   |                     | 16x PCIe Gen5 (64 GB/s)              | 8x PCIe Gen5 (32 GB/s)         |  |
|                   | Domain size         | 16 nodes                             | N/A                            |  |
| Scale-up network  | Bandwidth           | 800 GB/s (up to 1000 GB/s)           |                                |  |
| Scale-out network | Domain size         | 4096 L1, L2 unlimited                | N/A                            |  |
|                   | Bandwidth           | 200 GB/s                             |                                |  |
| GEMM TOPS         |                     | 1120 TFLOPS/s (FP8)                  | 354 TOPS/s (INT8)              |  |
|                   |                     | 560 TFLOPS/s (FP16/BF16)             | 177 TFLOPS/s (FP16/BF16)       |  |
|                   | RISC-V vector core  | 42.5 (INT8/FP16)                     | 5.5 (INT8), 2.8 (FP16)         |  |
| SIMD TOPS         |                     | 21.3 (BF16/FP32)                     | 1.4 (BF16/FP32)                |  |
|                   | SIMD engine         | 42.5 (FP8/FP16/BF16/FP32)            | 5.5 (INT8/FP16/BF16/FP32)      |  |
|                   | Per-PE local memory | 512 KB                               | 384 KB                         |  |
| Memory capacity   | On-chip SRAM        | 192 MB                               | 256 MB                         |  |
|                   | Off-chip memory     | 216 GB (6 stacks HBM3E)              | 64-128 GB (16 channels LPDDR5) |  |
|                   | Per-PE local memory | 1.9 TB/s (R+W)                       | 1.0 TB/s (R+W)                 |  |
| Memory bandwidth  | On-chip SRAM        | 11.4 TB/s (R+W)                      | 2.7 TB/s (R+W)                 |  |
|                   | Off-chip memory     | 6.1 TB/s (R or W)                    | 204.8 GB/s (R or W)            |  |

![](_page_3_Figure_2.jpeg)

Fig. 3: Processing Element (PE) architecture.

dataflow execution model. The programmer writes a kernel that generates a sequence of custom instructions for the fixedfunction units, where data movement and computation occur as dependencies are resolved.

Memory Layout Unit (MLU): The MLU performs memory layout transformations, including transpose, reshape, slice, and concatenation.

Dot Product Engine (DPE): The DPE performs General Matrix Multiplication (GEMM) operations and is used in both the forward and backward passes of training. It operates on two input tensors: the first is read and cached in the DPE, while the second streams from LS to compute a dot product with all rows of the first tensor. The DPE includes two 32×64B×32 Multiply-Accumulate (MAC) tiles, delivering a total throughput of 7.82 TFLOPS per PE with FP16/BF16 inputs and FP32 output. It also supports FP8 inputs (in S1E4M3 or S1E5M2 formats) and TF32 inputs, which are useful for certain ranking and recommendation use cases where higher precision is required.

Reduction Engine (RE): The RE stores intermediate matrix multiplication results from the DPE and performs inter-PE reductions via a dedicated reduction network. It can receive and accumulate results before forwarding them to the next PE or to the SIMD engine for further processing.

SIMD Engine (SFU): The SFU supports quantization, elementwise operations, and non-linear functions. It consists of an execution pipeline with floating-point ALUs and lookup tables (LUTs) to approximate non-linear functions. The SFU can receive input from the RE or read directly from the LS. For training, we removed INT8 and added FP8 alongside FP16, BF16, and FP32. The SIMD width was increased from 32 to 128 elements per cycle to achieve a GEMM:SIMD ratio of 16:1 (half of MTIA-2i), reflecting the high portion

TABLE II: Differentiating features of MTIA 300 versus H100 GPUs and how they enable efficient DLRM training.

| Differentiator                                                           | MTIA 300 Advantage                                                     | Rationale and Benefit                                                                                                                                                                                             |
|--------------------------------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| HBM bytes-to-FLOPS ra<br>tio                                             | > 2x higher                                                            | DLRM's large sparse features typically prevent effective utilization of high peak<br>FLOPS. Instead, a higher HBM bytes-to-FLOPS ratio enables balanced resource use and<br>higher model FLOPS utilization (MFU). |
| Network bytes-to-FLOPS<br>ratio                                          | > 5x higher                                                            | DLRM training tends to have low compute complexity and high exposed<br>communication, so higher network bandwidth improves MFU.                                                                                   |
| Large global SRAM                                                        | > 3x capacity                                                          | Improves utilization of GEMM and SIMD compute by capturing more locality in both<br>dense and sparse operators.                                                                                                   |
| Flexible IO to meet net<br>work demands                                  | NICs can be partitioned between<br>scale-up and scale-out networks.    | Supports a flat network of up to 1.2 TB/s and allows adjusting scale-up to scale-out<br>ratios based on model needs.                                                                                              |
| Hardware<br>support<br>for<br>collectives and on-device<br>data movement | High utilization due to compute<br>communication overlap.              | Message Engines (MEs) and Near Memory Compute (NMCs) work in coordination with<br>the control cores and the PE grid to maximize concurrency.                                                                      |
| Optimizations for DLRMs                                                  | Accelerates sparse feature process<br>ing and backward pass operators. | Table-batched embedding (TBE) and sparse optimizers are memory- and<br>instruction-bound. MTIA 300 provides dedicated embedding caches and indexed DMA<br>support for TBE, as well as radix-sort acceleration.    |
| Hardware support for ea<br>ger mode                                      | Fast and flexible job dispatch with<br>Work Queue Engines (WQEs)       | Eager mode is important for development and debugging and allows for more flexible<br>model deployment; speeding it up improves developer velocity.                                                               |
| Reliability                                                              | Additional compute and memory<br>improves yield and tolerates errors.  | Employs a redundant row of PEs in the 12x6 PE grid as well as provides ECC for<br>SRAMs and HBM to mitigate failures.                                                                                             |

of computation spent on non-GEMM operations. Additional SIMD throughput is available via the two RISC-V cores.

Based on the requirements of DLRM training, we implement several SFU improvements, including higher throughput for non-linear operations on high-precision data types and added support for min/max, clamping, and stochastic rounding. MTIA 300 also supports hardware-accelerated radix sort to accelerate the embedding backward operation. In the forward pass, sparse offsets and indices are packed so that a single output index maps to a contiguous subset of inputs. In the backward pass, sparse indices must be sorted so that a contiguous subset maps to a single embedding table index. Radix sort fetches elements from LS, sorts them via bucketization, creates histograms, and stores the bucketized elements in memory, speeding up the backward embedding operation.

Command Processor (CP): This processor handles the execution of custom instructions from the RISC-V cores across the fixed-function units, including scheduling and dependency checking. The CP arbitrates LS access between the RISC-V cores and the fixed-function units. It also provides the programmer with a circular buffer (CB) abstraction and manages dependency tracking to ensure correct producer-consumer usage of the CBs.

Fabric Interface (FI): The FI is a DMA engine for transferring data between PE local memory and on-chip or off-chip memory via the NoC. It also enforces packet fragmentation and leaky-bucket traffic shaping to smooth traffic and limit congestion.

Two enhancements to the FI and Command Processor provide more powerful data-movement abstractions. First, MTIA 300 supports byte-aligned DMA for tensor slicing, eliminating the software overhead of layout transformations. Second, it adds hardware-accelerated indexed DMA transfers for scatters and gathers. The Command Processor generates sequences of reads or writes using a list of indices in LS, which is particularly useful for embedding table lookups.

