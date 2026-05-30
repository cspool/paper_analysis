# 6 PIM Access Scheduling

The integrated NPU-PIM architecture with a unified main memory presents challenges as the main memory is used by both the NPU and the PIM compute logic. In this section, we propose *PIM Access Scheduling* (PAS) that enables efficient sharing of the physical memory between NPU and PIM. In particular, PAS includes workload mapping between the NPU and PIM, and scheduling normal DRAM and PIM operations.

#### 6.1 Workload Mapping

We present the execution flow and workload mapping in Figure 6. To maximize parallelism across all computing units, we exploit attention head parallelism for multiple attention heads, which can be parallelized. The weights of the FC for Q, K, and V are distributed into separate memory modules with a head-wise partitioning. This ensures that each core

concurrently loads the weights or the results of the assigned head computed in the PIM unit. For other FC operations, we leverage intra-layer parallelism to minimize weight data movement, which is considerably larger than input or activation data in LLMs. To reduce synchronization overhead between each core in NPU, we divide the weights of FC column-wise. Synchronization occur at four distinct points: one point after multi-head attention, two points after residual addition, and the other point after GELU.

As in Figure 6, layer normalization and residual addition are mapped to the vector unit (VU) for efficient processing. Meanwhile, FCs can be performed on either the matrix unit (MU) or PIM. The *summarization* stage often takes a large input token size, making it advisable to execute FC on an MU with high computing power. However, when the input token size is small, the weight loading time becomes the bottleneck, necessitating an appropriate choice between PIM and MU.

To determine the suitable unit for FC, we first develop a simple analytical model that estimates the execution time of command execution units based on the number of input tokens at compile time. We then propose Algorithm 1, which utilizes this analytical model. Algorithm 1 first takes ordered commands, initially generated with mapping FC to MU. When estimating the time of FC on MU, we consider a pipelined scheme for both weight loading and computation, as well as column-tiling with the tile size of MU (lines 9-13). We also account for weight prefetching time if an operation of VU precedes the FC operation (lines 5-6). Lastly, we compare the estimated time of FC on MU with that of PIM and assign the FC to the command execution unit requiring less time (lines 15-17). If the first FC of FFN is mapped to the PIM, the GELU will also be allocated to the PIM since our PIM is designed to support GELU right after FC.

## 6.2 Mapping-aware Scheduling for Multi-head Attention

As in Figure 1b, multi-head attention comprises a series of operations characterized by varying computational requirements, which lead to considerable latency without careful scheduling. To mitigate this issue, we investigate mapping-aware scheduling taking into account unified constraint. We consider both the *summarization* and *generation* stages.

**Summarization stage**: As in Figure 7a, during this stage, FC layers for Q, K, and V typically operate as matrix-matrix multiplications with multiple input tokens (x) and thus are computed in the matrix unit, while weight matrices  $(W^{Q,K,V})$  are loaded from PIM via DMA. To efficiently process self-attention with FC layers, we utilize both intra-attention head parallelism and inter-attention head pipelining. We prioritize key generation to execute key transposition in parallel with value generation. As DMAs are utilized for on-chip transposition, they are not used for PIM access during transposition (1). Given that the matrix unit supports scaling, as in Section 4.1, the key scaling operation is omitted. We also ensure that

![](_page_7_Figure_2.jpeg)

**Figure 7.** Mapping-aware scheduling of IANUS for (a) *sum-marization* stage where FCs are mapped to the matrix unit and for *generation* stage where FCs are mapped to the PIM:  $QK^T$  and SV mapping to (b) PIM or (c) matrix unit. Figures (b) and (c) are drwan on the same time scale to show the latency difference.

key and value are stored during computations (2). To hasten the start of the SV operation, values are moved to the weight scratch-pad via on-chip data transfer during the softmax (3). Moreover, we utilize inter-attention head pipelining by prefetching the weight of the next head (4).

Generation stage: FC layers mainly perform matrix-vector multiplications with one input token (x), making them well-suited for PIM computation. Similarly, since  $QK^T$  and SV operations involve matrix-vector multiplications and require loading previously generated keys and values, their executions seem more suitable in PIM. As in Figure 7b, mapping  $QK^T$  and SV to PIM can omit such loading operations. However, the opportunity for performance gain through scheduling is restricted because PIM executes most operations in the attention head. Moreover, computing  $QK^T$  and SV in PIM is inefficient in exploiting the parallelism of PIM. For instance, with a head dimension of 64, the computational efficiency of  $QK^T$  is a mere 6.25%. This results from only 64 BF16 elements out of the 1024 elements available in one DRAM row being utilized for computation.

As a result, we explore scheduling for mapping  $QK^T$  and SV to the matrix unit. To exploit inter-attention head parallelism, as shown in Figure 7c, we execute key concatenation in the vector unit instead of storing the key ( $\P$ ), enabling its simultaneous execution with query generation in PIM. Loading the previously generated keys ( $K_{pre}$ ) of ith head is omitted in Figure 7c, as its small size compared to the FC weight

**Table 1.** Simulation parameters for IANUS.

| NPU    | Composition          | 4 cores, 8 PIM memory controllers                                     |  |  |  |  |
|--------|----------------------|-----------------------------------------------------------------------|--|--|--|--|
|        | Host interface       | PCIe 5.0 ×16                                                          |  |  |  |  |
|        | Frequency            | 700 MHz                                                               |  |  |  |  |
|        | Matrix unit          | 128x64 processing elements (PEs), 4 MACs per PE, 46 TFLOPS            |  |  |  |  |
|        | Vector unit          | Sixteen 4-wide VLIW processors                                        |  |  |  |  |
| Core   | Scheduler            | 4 command slots per issue queue of units,                             |  |  |  |  |
|        |                      | 256 command slots in pending queue                                    |  |  |  |  |
|        | Scratch-pad          | Activation 12 MB, Weight 4 MB                                         |  |  |  |  |
|        | Memory               | GDDR6 16 Gb/s; ×16 organization; 8 channels; 256 GB/s;                |  |  |  |  |
|        | configuration        | 2 channels per chip, 16 banks per channel, row (page) size 2 KB       |  |  |  |  |
| PIM    | Timing parameters    | $t_{CK} = 0.5ns$ , $t_{CCD_S} = t_{CCD_L} = 1ns$ , $t_{RAS} = 21ns$ , |  |  |  |  |
| I IIVI |                      | $t_{WR} = 36ns, t_{RP} = 30ns, t_{RCDRD} = 36ns, t_{RCDWR} = 24ns$    |  |  |  |  |
|        | Processing unit (PU) | 1 GHz; 1 PU per bank; 32 GFLOPS per PU                                |  |  |  |  |
|        | Global buffer        | One 2 KB global buffer per channel                                    |  |  |  |  |

Table 2. Specifications of A100 GPU, DFX, and IANUS.

|          |             | A100 [38]         | DFX [19]    | IANUS                         |
|----------|-------------|-------------------|-------------|-------------------------------|
| Compute  | Frequency   | 1155 MHz          | 200 MHz     | 700 MHz                       |
| Compute  | Throughput  | 255 TFLOPS        | 1.64 TFLOPS | 184 TFLOPS                    |
| On-chip  | Capacity    | RF, L1, L2: 84 MB | ~40 MB      | Activation Scratch-pad: 48 MB |
| Memory   | Сарасну     | Kr, L1, L2. 04 MD | ~40 MD      | Weight Scratch-pad: 16 MB     |
|          | Type        | HBM2e             | HBM2        | GDDR6                         |
| Off-chip | Capacity    | 80 GB             | 32 GB       | 8 GB                          |
| Memory   | Bandwidth   | 2039 GB/s         | 1840 GB/s   | 256 GB/s                      |
|          | Internal BW | N/A               | N/A         | 4096 GB/s                     |

allows for prefetching. We then transpose concatenated keys within on-chip while performing query generation in PIM. Furthermore, we execute  $QK^T$  and softmax respectively in parallel with value generation by mapping  $QK^T$  to matrix unit (2). After value generation, storing generated keys and values and loading concatenated values ( $V_{cat}$ ) are performed during softmax (3). We also employ inter-attention head pipelining by prefetching  $K_{pre}$  of the next head during SV (4). If the prefetching ends before the completion of SV, the key generation of the next head is performed in conjunction with SV. Consequently, our scheduling enhances performance by maximizing both intra-parallelism and interpipelining of attention head.

