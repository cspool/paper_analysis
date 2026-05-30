# I. INTRODUCTION

The field of Machine Learning (ML), specifically Deep Neural Networks (DNNs) is pervasive today across image classification [12], [40], object detection [3], [37], text summarization [20] and sentiment analysis [25]. Such a plethora of ML models introduces great diversity in structure (serial or parallel layers connectivity), layer types (depth-width, pointwidth, dilation convolutions, or even a fusion of them), and sizes (number of channels, kernels, height, and width) [7], [49].

The mechanism for orchestrating a DNN layer over the accelerator's on-chip compute and memory resources is called dataflow. It can be precisely defined by transformations of the loop nest, as shown in Fig. 1. Several prior works [33], [41] have demonstrated that dataflows can lead to significant differences in compute utilization and up to two orders of magnitude variance in latency and energy, and thereby motivated the need to support per-layer dataflow flexibility.

![](_page_0_Figure_12.jpeg)

Fig. 1: Terminology of convolution workload and dataflow

Changing dataflows on accelerators requires (a) reconfiguring datapaths in computation, distribution, and reduction networks, and (b) modifying data layout in on-chip buffers. Almost all prior works have focused on the first aspect, and several clever interconnect topologies for data distribution and reduction have been proposed that activate subset of paths at runtime through reconfiguration depending on the dataflow being run [42], [44]. However, data layout in the on-chip buffer is a critical and often overlooked in past work.

In this work, we demonstrate that the high performance of dataflows is unachievable in practice without layout reordering capability. This is because, without a suitable data layout, the required data may be located in the same SRAM banks and compete at the same SRAM reading ports. Such bank conflict slows down the delivery of data to computation engines, leading to stalling and computation underutilization. Overlooking layout reordering thus introduces a significant 128× performance gap between theory and practice as quantified in Fig. 2. We discuss this with more depth in §II.

Unfortunately, layout reordering comes with severe latency and energy overheads. Off-chip layout reordering requires back-and-forth data movement between off-chip DRAM/HBM and computation, while on-chip layout reordering requires additional intermediate storage and extra latency in the critical path. In fact, these costs can outweigh the benefits of switching dataflows, leading existing ML accelerators to compromise settling on a single dataflow (e.g., Xilinx DPU, Gemmini, NVDLA, Eyeriss in Table I) that provides good average utilization across all layers, but sub-optimal performance.

To unleash optimal performance, we propose a novel accelerator *FEATHER*, <u>Flexible Engine</u> for <u>Acceleration of Tensors</u> with Hardware Element for Reordering, which includes a novel reconfigurable reduction network called Butterfly Interconnect

TABLE I: Feature comparison: how FEATHER resolves challenges of prior works without on-chip layout reordering.

| Work                          | Dataflow Switching | Layout Reorder | Challenge                               | FEATHER solution (key component)               |
|-------------------------------|--------------------|----------------|-----------------------------------------|------------------------------------------------|
| NVDLA [39]                    | Х                  | no reorder     | underutilization from fixed parallelism | flexible dataflows (NEST)                      |
| Xilinx DPU [51], Gemmini [21] | Х                  | no reorder     | linear reduction                        | parallel logarithmic reduction (BIRRD)         |
| SIMBA [47], Eyeriss [13]      | Х                  | no reorder     | load imbalance across PE                | pick load-balance dataflows (NEST)             |
| Eyeriss_v2 [15], SARA [44]    | ✓                  | off-chip       | high latency of moving data off-chip    | on-chip reordering with latency hidden (BIRRD) |
| MAERI [35], SIGMA [42]        | ✓                  | off-chip       | long wires of reduction network         | small standalone reduction network (BIRRD)     |

![](_page_1_Figure_2.jpeg)

Fig. 2: Latency evaluation of dataflows on  $16 \times 16$  PE array with various layouts (error bar shows layout impacts, less latency is better). The best flexible dataflow (green bar) *theoretically* reduces overall latency of fixed dataflow-layout (blue bar) by 63.3%. However, ignoring the impact of layout considerations in theoretical dataflows results in up to a  $128 \times$  latency gap in *practice* (yellow bar). FEATHER eliminates the gap by coswitching dataflow-layout (red bar).

for <u>Reduction</u> and <u>Reordering</u> in <u>Dataflows</u> (*BIRRD*). With *BIRRD*, the latency of layout reordering is completely hidden in data reduction, allowing data layout in on-chip storage to be manipulated for the demand of optimal dataflow without any latency costs. We call this approach as reordering in data reduction (RIR). Thus, *FEATHER* fully achieves the theoretical performance of optimal dataflows without incurring bank conflicts. Furthermore, *FEATHER* also pioneers a new paradigm to co-switch both dataflows and data layouts at layer granularity, with minimal switching overheads. This ability to accommodate low-cost layout-dataflow co-switching is, as far as we know, unsupported by any existing accelerator.

To fully explore the potential of *FEATHER*, we also developed a tool that facilitates: (a) dataflow evaluation factoring in data layout, and (b) (layout, dataflow) co-exploration.

Our key contributions can be summarized as follows:

- •We demonstrate the interaction between dataflows and data layouts, motivating the need for data reordering support within reconfigurable dataflow accelerators. We further categorize existing reordering patterns and implementations (§II).
- •We present a novel accelerator *FEATHER* with several novel features (§III). First, a neural engine with temporal local reduction and spatial forwarding, *NEST*, for dataflow flexibility. Second, a multi-stage network called *BIRRD* enabling flexible reductions from arbitrary groups of multiple inputs to multiple results, at lower area overhead compared to prior works with similar capabilities. Further, BIRRD supports Arbitrary reorder via a novel technique RIR, that completely conceals data layout reordering latency behind reduction (§IV).
- We extend a state-of-the-art accelerator modeling framework Timeloop [41] with support for physical on-chip storage, layout

TABLE II: On-chip memory terminology

| Term     | Meaning                                                              |
|----------|----------------------------------------------------------------------|
| Buffer   | A logical 2D on-chip memory (num_line × line_size) stacking multiple |
|          | SRAM banks both vertically (num_line) and horizontally (line_size).  |
| Bank     | A physical 2D SRAM (entries $\times$ io) with address/data ports.    |
| Line/Row | A buffer line (line_size = accumulated IO of horizontal SRAM banks). |
| Port     | An input/output port, each bank has at most two ports in TSMC 28nm.  |

representation, and dataflow-layout co-search. We call this new framework *Layoutloop* (§V) and use it for our evaluations.

•We implement and deploy *FEATHER*, end-to-end, on an edge ZCU 104 FPGA device and also model it using *Layoutloop*. *FEATHER* achieves  $1.27 \sim 2.89 \times$  inference latency speedup and  $1.3 \sim 6.43 \times$  energy efficiency improvement compared to various SoTAs across multiple DNN models, and  $2.65 \times /3.91 \times$  more throughput than Xilinx DPU/Gemmini on real FPGAs. On average, efficient pairs of (dataflow, layout) results in an energy savings of 27% to 33% across workloads despite the energy costs of layout reordering. Remarkably, all enhancements come at only 6% area over a fixed-dataflow Eyeriss-like accelerator.

## II. BACKGROUND AND MOTIVATION

#### A. Dataflow Space in Convolution

Fig. 1 depicts a convolution operation with seven dimensions with various shapes. Dataflows can be represented as a nested loop with four types of optimizations [24], [34].

- $\bullet$ (**T**)iling breaks down dimensions of iActs N, C, H, W into smaller chunks, and enables executing workloads in tile granularity as on-chip storage is limited.
- (O)rdering allows arbitrary loop reordering (aka "stationarity" [13]) to reuse more data since dimensions N, M, C, P, Q, R, S do not come with loop-carried dependencies except reduction-dependencies over C, R, and S.
- (P)arallelism allows for arbitrary parallelism over any dimensions as all dependencies are loop-independent, leading to different spatial reuse opportunities.
- (S)hape defines the virtual grouping of the physical PE array.

These dataflow flexibility (TOPS) [34] create an extremely large dataflow design space with a complexity of  $O(10^{36})$  for a single convolution layer [27]. The choice of the dataflow affects both runtime performance (as it affects overall compute utilization) and energy efficiency (as it affects the number of accesses across the memory hierarchy). Not surprisingly, no single dataflow is generally optimal for all types of layers given their diverse sizes and shapes [33], [41]. This can be seen by comparing the first two bars (blue and green bars) in Fig. 2.

