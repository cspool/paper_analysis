# *F. Instruction Set*

Table V summarizes the Uni-STC instruction set (UWMMA), which follows WMMA semantics and includes the cycle ranges for FP64 operations. Data types are categorized by suffixes: 'i' for 8-bit indexes, 'b' for 16-bit bitmaps, and 'v' for 64-bit values.

TABLE V: Uni-STC@FP64 instruction set (UWMMA).

| Operation |      | Registers per threads         | Cycles |  |
|-----------|------|-------------------------------|--------|--|
|           | Meta | A16b1, A16b2, X16b,           | 1      |  |
| Load      | (MV) | A4b1/A4i1, A4b2/A4i2          |        |  |
|           | Meta | A16b, B16b, C16b,             | 1      |  |
|           | (MM) | A4b/A4i, B4b/B4i, C4b/C4i     |        |  |
|           | A    | Av(0 ∼ 7)                     | 2      |  |
| T3 Task   | MV   | Use meta data saved in buffer | 1∼4    |  |
| Generate  | MM   | Use meta data saved in buffer | 1∼8    |  |
| Calculate | MV   | Av2(0 ∼ 7), Xv, Yv            | 1∼8    |  |
| and Store | MM   | Bv(0 ∼ 7), Cv(0 ∼ 7)          | 1∼64   |  |

To comply with the operand limits of PTX instructions (e.g., the 'mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64' variant allows a maximum of 20 FP64 register operands per thread) and better aligns with the properties of sparse kernels, we choose to store both the block values of matrix A and the corresponding block structures within Uni-STC's internal buffers. Integrating the UWMMA instruction set and this data handling approach necessitates compiler modifications.

## *G. Execution Lifecycle and Control Interaction*

Uni-STC executes sparse kernels through a coordinated UWMMA instruction sequence. This lifecycle relies on interaction with the SM and internal state registers to achieve asynchronous task generation and synchronous computation:

- (1) Operand collection. The cycle begins with stc.load instructions. The SM uses the operand collector to fetch numerical data or metadata from the register files and stores them into Uni-STC's internal buffers (Matrix A Buffer or Meta Buffer). This phase is synchronous and memory-bound.
- (2) Asynchronous task generation. Upon issuing a stc.task instruction, the Uni-STC transitions its state register from IDLE to BUSY. This triggers the TMS and DPGs to begin processing metadata and filling the two task queues. This asynchronous process allows the SM to immediately retire the stc.task\_gen instruction and proceed with other work, effectively hiding the task generation latency.
- (3) Synchronized computation. The stc.numeric instruction initiates computation on the SDPU by first checking the flag register:
  - Stall (BUSY): If the flag is BUSY, indicating insufficiently populated task queues, the pipeline stalls.
  - Execute (READY): Once the DPGs populate the queues, the flag transitions to READY. The SDPU then begins execution, consuming T4 tasks, performing segmented dot-products, and accumulating the results.
- (4) Completion. When the batch of T4 tasks is fully processed, the flag returns to IDLE, enabling the results to be written back to the register files.

