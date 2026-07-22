# Spatial Arch和dataflow Acc

## FEATHER: 2024

GEMM和Conv的非对称量化的多dataflow加速器，没有细粒度指令，而编译出每层的dataflow配置指令（**dataflow指令**）和每层的输入输出地址。

PE是执行**非对称量化**的MAC，包含AH深度的Weight Ping-Pong RF，缓存flow中AH个fliter的权重，避免循环读取Stream（缓存换访存的优化）。

SA计算不同并行方式的Conv（WS、OS）、GEMM，将数据重排模块融入pipeline中的reduction。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> C=2048 H=W=7 R=S=3 Stride=1 padding=1
> 
> | 4             | Line Size = 8 i |               |               |  |  |  |  |  |
> |---------------|-----------------|---------------|---------------|--|--|--|--|--|
> | <u> </u>      | Line#           |               | )W0:7         |  |  |  |  |  |
> | ayout<br>W8)  | 1               | H0C0W8        | Empty         |  |  |  |  |  |
> | _< a          | l . l           | ***           |               |  |  |  |  |  |
> | _ ≥           | r4              | H0C1W0:7      |               |  |  |  |  |  |
> | Aajor<br>(HCW | r4+1            | H0C1W8        | Empty         |  |  |  |  |  |
> | ೯೬            |                 |               |               |  |  |  |  |  |
> | Ī             | r5              | H0C2W0:7      |               |  |  |  |  |  |
> | 30V           | r6              | H0C3W0:7      |               |  |  |  |  |  |
> |               | ,               | $\overline{}$ | $\overline{}$ |  |  |  |  |  |
> 
> (M5) FEATHER's pick
> 
> | i | Cycle | iActs      | Line #     | Slowdown        | Theoretical | Practical   |
> |---|-------|------------|------------|-----------------|-------------|-------------|
> | ì | "#    | Required   | being      | (total ports/   | Compute     |             |
> | i |       | by Mapping | Accessed   | accessed lines) | Utilization | Utilization |
> | i | 0     | H0W0C0:3   | 0,r4,r5,r6 | 2/4=0.5         | 100%        | 50%         |
> | ŀ | 1     | H0W1C0:3   | 0.r4.r5.r6 | 2/4=0.5         | 100%        | 50%         |
> | 1 | 2     | H0W2C0:3   | 0.r4.r5.r6 | 2/4=0.5         | 100%        | 50%         |
> | i | 3     | H0W3C0:3   | 0,r4,r5,r6 | 2/4=0.5         | 100%        | 50%         |
> |   |       |            |            |                 |             |             |
> 
> Memory Efficiency: read 4 line per cycle; Compute Utilization: 50% (M7)
> 
> (M6)
> 
> | CI-        | iActs      | Line #   | Slowdown        | Theoretical | Practical   |
> |------------|------------|----------|-----------------|-------------|-------------|
> | Cycle<br># | Hequirea   | being    | (total ports/   | Compute     |             |
> |            | by Mapping | Accessed | accessed lines) | Utilization | Utilization |
> | 0          | H0C0W0:3   | 0        | no slowdown     | 100%        | 100%        |
> | 1          | H0C0W4:7   | 0        | no slowdown     | 100%        | 100%        |
> | 2          | H1C0W0:3   | one line | no slowdown     | 100%        | 100%        |
> | 3          | H1C0W4:7   | one line | no slowdown     | 100%        | 100%        |
> |            |            |          |                 | ***         |             |
> 
> Memory Efficiency: read 1 or 2 lines per cycle; Compute Utilization: 100% (M8)
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2075.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 10: Comparison between per-layer flexible dataflows in *FEATHER* and fixed-dataflow in the systolic array under GEMM. *FEATHER* dynamically alters layout by redirecting oActs to various banks with distinct writing addresses, exemplified by rerouting a blue result from bank 0 (Workload A) to bank 2 (Workload A Change oAct Layout). *FEATHER* consistently outperforms SA in irregular-sized GEMM (Workload B, C, D), achieving near full utilization. Enhanced utilization arises from (1) enabling cross-column spatial reduction using *BIRRD* in *FEATHER*, e.g. *FEATHER* maps K dimension across the entire 2D array instead of a single PE in SA under workload D. (2) Eliminating SA's horizontal rigid reuse links, thereby enabling independent mappings across columns, e.g. (Workload C) adopting iAct stationary in first three columns and weights stationary in the last column. *BIRRD* could perform pure reordering to change the layout when no spatial reduction is required (e.g. *BIRRD* reordering all incoming results to target banks directly under workload B). **Takeaway:** BIRRD's flexible reduction enhances compute utilization across diverse skewed shapes, expanding the range of dataflows that NEST can efficiently support.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2076.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8: Micro-architecture of *FEATHER*'s datapath for convolution/GEMM. For convolution, the NEST reads iActs from StaB and weights from StrB, streaming both in a top-to-bottom pipeline. PEs in a column time-multiplex a common output bus. *BIRRD* conducts global spatial reduction and reorders results for targeted StaB banks during reduction, altering data layout in StaB. NEST facilitates inter-layer pipelining by reading iActs from StaB Ping (or Pong) and writes oActs (next-layer iActs) back to StaB Pong (or Ping). Note: *FEATHER* is scalable architecture and we show 8-input *BIRRD* as an example.
![image.png](FEATHER%20A%20Reconfigurable%20Accelerator%20with%20Data%20Reo/image%2032.png)

## Eyeriss v1：2017

row-stationary Conv dataflow。

> **[图片提取文字 (Screenshot 2025-11-03 at 14.41.51.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2. Eyeriss system architecture.
![Screenshot 2025-11-03 at 14.41.51.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/Screenshot_2025-11-03_at_14.41.51.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 12. PE architecture. The datapaths in red show the data gating logic to skip the processing of zero ifmap data.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%201.png)

## Eyeriss v2：2019

NoC for **dataflow** of DPConv and Conv、GEMM

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3. Two common DNN accelerator designs: (a) Spatial accumulation array [29-32]: iacts are reused vertically and psums are accumulated horizontally. (b) Temporal accumulation array [15-17]: iacts are reused vertically and weights are reused horizontally.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%207.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4. Array utilization of different architectures for depth-wise (DW) layers in MobileNet. The colored blocks are the utilized part of the PE array. For Eyeriss [33], the different colors denote the parts that run different channel groups (G). Please refer to Table I for the meaning of the variables.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%208.png)

> **[图片提取文字 (image.png)]:**
> such as MobileNet. This is achieved through the design of a highly flexible on-chip network (NoC), which is currently the bottleneck for dealing with a more diverse set of DNNs. In addition, Eyeriss v2 also supports sparse DNNs by exploiting the sparsity in the weights and activations across a variety of DNN layers and translates them into improvements in both energy efficiency and processing speed. Finally, similar to the original Eyeriss, Eyeriss v2 does not make any assumption about whether the total storage capacity required by a DNN layer can fit on-chip or not; instead, it optimizes the way to tile data of different types to achieve high on-chip reuse and energy efficiency. In summary, the contributions of this paper include:
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2018.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 9. Examples of weight and iact hierarchical mesh networks configured in different modes for different types of DNN layers: (a) CONV layers; (b) depthwise (DW) CONV layers; (c) fully-connected (FC) layers. Green arrows and blue arrows show the routing paths in the weight and iact NoC, respectively.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5. Comparison of the architecture of original Eyeriss and Eyeriss v2.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Fig. 6. Eyeriss v2 top-level architecture.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 8. (a) High-level structure of the hierarchical mesh network (HM-NoC), and its different operating modes: (b) High bandwidth mode, (c) High reuse mode, (d) grouped-multicast mode, and (e) interleaved-multicast mode. In each mode, the colored arrows show the routing path; different colors denote the path for unique data.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2021.png)

## Simba：2019

chiplet Acc是tile阵列（tile=chiplet），每个Tile是PE阵列，每个PE包含Buffer和计算单元。

每个PE使用**dataflow执行小张量运算**（buffer+ALUs），等效SM（TensorCores、CUDA Cores）。

PE之间路由组成Spatial架构，执行**大张量dataflow或计算图pipeline**，等效SM组成的GPC。

tile之间路由组成更大Spatial架构，等效GPU或多卡Node。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Simba architecture from package to processing element (PE).
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: Simba silicon prototype.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2029.png)

> **[图片提取文字 (image.png)]:**
> Table 1: Simba system communication capability.
> 
> | Packet<br>Source | Unicast<br>Destination            | Multicast<br>Destination |
> |------------------|-----------------------------------|--------------------------|
> | PE               | Local PEs, Global PE, Controller  | -                        |
> |                  | Remote PEs, Global PE, Controller | -                        |
> | Global PE        | Local PEs, Controller             | Local PEs                |
> |                  | Remote PEs, Controller            | Remote PEs               |
> | Controller       | Local PEs, Global PE              |                          |
> |                  | Remote PEs, Global PE, Controller |                          |
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2027.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 12: Illustration of communication-aware, non-uniform work partitioning. The top green tensors represent weights (W), the left blue tensors represent input activations (IA), and the bottom red tensors represent the output activation (OA). In this example, IA is stored in Chiplet0 and Chiplet2.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2032.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 13: Non-uniform work partition for ResNet-50 with speedup normalized to the best-performing tiling.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2033.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 14: Data placement on the Simba system. (a) Assessment of the relative latency to different chiplets that receive data from Src. (b) Default input activation (IA) and output activation (OA) placement where data is sequentially placed from the Global PE of the first chiplet. (c) An improved IA placement at the center of the package so that data can be multicast to all chiplets. (d) OA placement with even distribution along the periphery of the package to minimize OA communication latency.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2034.png)

算子融合Mapping到chiplets上，作pipeline并行。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 16: Pipelining a residual block of ResNet-50 in the Simba system.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2035.png)

> **[图片提取文字 (image.png)]:**
> is achieving high utilization when the layer computation has a limited amount of parallelism [14, 39, 71]. To address this challenge, recent DNN accelerators support pipelined execution to improve overall system utilization [26, 71]. ScaleDeep supports column-wise pipelining in the PE array, where different columns can be assigned to different pipelined layers [71]. This low-overhead implementation still results in low utilization when layers cannot be easily mapped across columns. Tangram supports flexible partitioning across the PE array for different layer shapes but does not consider the non-uniformity of communication latency and bandwidth [26]. Figure 16 illustrates how a residual block of ResNet-50 can be pipelined across the Simba package. Because the Simba hierarchical interconnect supports flexible communication patterns, we can assign different-sized clusters of chiplets to different layers. In the example shown, res2a\_branch2b uses four chiplets,
> 
> while res2a\_branch2a uses only two chiplets. Figure 17 shows
> 
> One key challenge of mapping DNN layers to large-scale systems
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2036.png)

## TPUv4：2021

> **[图片提取文字 (image.png)]:**
> don't fit in Vector Memory (§5). Figures 5 and 6 show the new 128 MB *Common Memory* (*CMEM*) of TPUv4i. This expanded memory hierarchy reduces the number of accesses to the slowest and least energy efficient memory (see §5).
> 
> We picked 128MB as the knee of the curve between good performance and a reasonable chip size, as the amortized chip cost is a significant fraction of TCO ③. Figure 6 shows that the resulting CMEM is 28% of the die area. Since TPUv4i is aimed at inference, its die size is closer to TPUv1's die size than to TPUv3's size (Table 1).
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 5. TPUv4i chip block diagram. Architectural memories are HBM, Common Memory (CMEM), Vector Memory (VMEM), Scalar Memory (SMEM), and Instruction Memory (IMEM). The data path is the Matrix Multiply Unit (MXU), Vector Processing Unit (VPU), Cross-Lane Unit (XLU), and TensorCore Sequencer (TCS). The uncore (everything not in blue) includes the On-Chip Interconnect (OCI), ICI Router (ICR), ICI Link Stack (LST), HBM Controller (HBMC), Unified Host Interface (UHI), and Chip Manager (MGR).
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2062.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6. TPUv4i chip floorplan. The die is <400 mm² (see Table 1). CMEM is 28% of the area. OCI blocks are stretched to fill space in the abutted floorplan because the die dimensions and overall layout are dominated by the TensorCore, CMEM, and SerDes locations. The TensorCore and CMEM block arrangements are derived from the TPUv4 floorplan.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2063.png)

## ButterFly Acc

特殊算法Acc，等效PE（SM）或tile（GPC）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6. Hardware overview of the adaptable butterfly accelerator.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2015.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 11. Hardware design of *Index Coalescing* module.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 13. Different overlapping strategies.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 12. Different address mapping strategies.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 15. Flow of the algorithm-hardware co-design process.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2022.png)

## YOLO v3-tiny Acc：2023

dataflow Acc。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5. Convolution process within the GEMM fashion. Both input FMAPs and filter weights were getting flattened to form a GEMM-suitable 2-D tensor format. (a) Convolution process for  $3 \times 3$  filter weights. (b) Convolution process for  $1 \times 1$  filter weights.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2077.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 6. High-level accelerator architecture diagram.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2078.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 7. Proposed multicore network schemes to split the computational workload between identical systolic cores. The division process is getting handled entirely by the matrix generators of the neuron inputs (input FMAP, weight, and bias). (a) FMAP-MC method. (b) W-MC method.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2079.png)

## FPGA Overlay：2023

line Buffer实现类似im2col的功能。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4. The line buffer pipeline inside our overlay that generates stencil windows over a streaming input feature map.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2080.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5. The line buffer is operating in two configurations generating  $3 \times 3$  windows with different stride values. The connection of the input to the FIFOs and connections between the FIFOs change with the stride factor.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2081.png)

将2D数据展开成1D，方便通过**Tree**分块卸载到PE Cores，Tree可在每个layer之间重配置，因此支持不同数据流；

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 6. The micro-architecture of the Processing engine. An example is shown on the right of how the distribution tree distributes a window vector over 8 multipliers.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2082.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 7. (a) The distribution tree configuration for processing 3 convolutions with SP = 3,  $K^2 = 9$ , and FP = 1. (b) The reduction tree configuration for processing 3 convolutions with SP = 3,  $K^2 = 9$ , and FP = 1. The copy index is 1 throughout, since FP = 1.
![image.png](meeting-25%2011%206%EF%BC%88%E5%BC%A0%E9%87%8F%E5%8A%A0%E9%80%9F%E5%99%A8%EF%BC%89/image%2083.png)

## **ISCA24**:Soter

> **[图片提取文字 (image.png)]:**
> ```
> // DRAM [Step ①]
>                                             I: 1024x1024 W: 1024x1024
>     for n4 = [0 : N4=32):
>                                            ① Load to DRAM,
>      for k4 = [0 : K4=2):
> 3
>                                               decompose into 64 smaller GEMMs,
>     // Global Buffer level [Step 2]
>                                              deliver to global buffer
>     for c3 = [0 : C4=4) :
>                                             I_1: 32 \times 1024 \quad W_1: 1024 \times 512
>      for n3 = [0 : N3=2) :
>                                            2 Load to global buffer,
>       for k3 = [0 : K3=16) :
>                                               decompose into 2048 smaller GEMMs,
>        parallel_for n2 = [0 : N2=4) :
>                                              deliver to PEs
>          parallel_for k2 = [0 : K2=4) :
>                                                I_2:4x256 \ W_2:256x8
>  10 // PE level [Step ③]
>                                            3 Load I_2 to input buffer and W_2 to
>  11 for c1 = [0 : C1=64) :
>                                               weight buffer,
>  12 for k1 = [0 : K1=2) :
>                                              decompose into 128 smaller GEMMs
>  13 // MAC level [Step 4]
>                                                   I_3: 4x4 \quad W_3: 4x4
>  14 parallel_for c0 = [0 : C0=4) :
>     parallel_for n0 = [0 : N0=4):
>                                           4 Use 64 MAC units for the GEMM,
>     parallel _for k0 = [0 : K0=4) :
>                                              write the results to accumulation buffer
>  16
>  17 n = ..., k = ..., c = ...
>                                                        O_2:4x8
>  18 O[n,k] += I[n,c] * W[c,k]
>                                               Stores the results from all PEs to
>  19 ... // Write back [Step ⑤]
>                                               global buffer.
>                                                           (b)
>                   (a)
> Fig. 2. (a) Tensor program tuned by our Soter for the example GEMM on
> Simba [12]. The loops in DRAM are the outermost while the loops in PE are
> ```
> 
> the innermost. (b) Workflow of using the tensor program for Simba.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2077.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1. Abstract architecture of spatial accelerators.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2067.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5. Program tuning flow. This example generates a program for a GEMM on Simba. (a) Tensor computation description. (b) Spatial architecture description. (c) Tunable parameter definition. (d) Program design space description. The program space is determined in the analytical tensor-architecture model and updated in the coordination. (e) Sequence description of a tensor program. The sequence is generated through Transformer-based exploration and transformed into a tensor program.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2086.png)

## 24:GEMINI（Simba架构）

chiplet Processor，每个tile是一个chiplet，Buffer+PE阵列是dataflow Acc。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Picture_0.jpeg)
> 
> Fig. 1. Trade-offs Introduced by Chiplet
![image.png](Spatial%20Arch%E5%92%8Cdataflow%20Acc/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Scalable Hardware Template
> 
> Fig. 2. Architecture of Scalable Hardware Template
![image.png](Spatial%20Arch%E5%92%8Cdataflow%20Acc/image%201.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3. Parsing an Encoded LMS in the LP SPM Optimization Space into an Actual SPM Scheme
![image.png](Spatial%20Arch%E5%92%8Cdataflow%20Acc/image%202.png)

## **25：GCC**

GCC pipeline按照GS为任务粒度

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: An illustration of the standard and our proposed dataflow for 3DGS inference. GCC exploits a cross-stage conditional processing scheme to dynamically eliminate redundant Gaussian preprocessing, and a Gaussian-wise approach to mitigate duplicated Gaussian accesses during rendering.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%207.png)

GS preprocess & **render**数据流

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 3: Overview of GCC dataflow.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%208.png)

**加速器设计**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 5: GCC Architecture.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%209.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 9: Alpha Unit and alpha-based controller.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 7: Grouping by shared resources.
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 8: Key logic in Projection Unit.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2011.png)

## **TACO25：RL Scheduler**

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1. (a) A ResNet50 [18] layer of dimension N=4, K=C=256, P=Q=14, R=S=3. (b) Abstract spatial accelerator architecture. The arrows represent the data movement from DRAM to PEs. (c) An example of scheduling the ResNet50 layer on the Simba [40] accelerator. The left part is the nested loop representation of the schedule configuration. The right part represents three schedule keys. This representation is also used in Figure 3. The dimensions from left to right represent loop ordering at each level. The blue and red digits represent temporal and spatial tiling in the form of prime factors, respectively.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ## **ALGORITHM 1:** Calculating the available hardware resources
> 
> ```
> 1: Input: Dim, cur_level, PFs, APFs, MC, SC
>    /* Dim: A dimension to be tiled */
>    /* cur_level: The current level of memory */
>    /* PFs: All prime factors of the dimension */
>    /* MC, SC: The whole memory capacity and spatial capacity at the current level */
> 2: Output: AMC, ASC
>    /* AMC, ASC: The available memory capacity and spatial capacity at the current level */
> 3: AMC = GET_AMC(Dim, cur_level, PFs, APFs, MC)
> 4: ASC = GET_ASC(cur_level, APFs, SC)
>  5: function GET_AMC(Dim, cur_level, PFs, APFs, MC)
>        Tiles = [1, 1, 1, 1, 1, 1, 1]
>  6:
>        for level = [0:cur_level) do
>  7:
>           for pf in prime factors do
>  8:
>               Tiles *= pow(pf, APFs[level, pf])
>            end for
> 10:
>        end for
> 11:
>        N_T, K_T, C_T, P_T, Q_T, R_T, S_T = \text{Tiles * Matrix C[cur level]}
> 12:
>        /*Matrix C is calculated in Figure 4 */
> 13:
>        Input_tiles = N_T * (P_T - 1 + R_T) * (Q_T - 1 + S_T) * C_T
> 14:
>        /* Assume Stride and Dilation are 1 for simplicity */
> 15:
>        Weights_tiles = K_T * R_T * S_T * C_T
> 16:
>        Outputs_tiles = P_T * Q_T * K_T * N_T
> 17:
>        MC ≥ Input_tiles + Weight_tiles + Output_tiles
> 18:
>        Calculate a subtract and a divisor in Table 1
> 19:
>        AMC = (MC - subtract) / divisor
> 20:
>        return AMC
> 21:
> 22: end function
> 23: function GeT_ASC(cur_level, APFs, SC)
>        used SC = 1
>        for dim in ['N', 'K', 'C', 'P', 'Q', 'R', 'S'] do
> 25:
>            used_SC *= pow(2, APFs[cur_level, dim, 2, spatial])
> 26:
>        end for
> 27:
>        ASC = SC / used SC
> 28:
>        return ASC
> 29:
> 30: end function
> ```
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2036.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4. Matrix A: The dependency between memory levels and tensors. Matrix B: The dependency between tensors and layer dimensions. The input tensors are related to P, Q, R, and S dimensions since the input tensors' indices are computed by these dimensions. Matrix  $C = AB^T$ : The dependency between memory levels and layer dimensions.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2037.png)

> **[图片提取文字 (image.png)]:**
> Table 1. The Subtract and Divisor for Each Dimension
> 
> |                                                                                        | Subtract                                                | Divisor                                                             |
> |----------------------------------------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------------------|
> | N                                                                                      | Weight_tiles                                            | $((P_T-1+R_T)*(Q_T-1+S_T)*C_T+P_T*Q_T*K_T)*N_T$                     |
> | K                                                                                      | Input_tiles                                             | $(R_T * S_T * C_T + P_T * Q_T * N_T) * K_T$                         |
> | С                                                                                      | Output_tiles                                            | $(N_T * (P_T - 1 + R_T) * (Q_T - 1 + S_T) + K_T * R_T * S_T) * C_T$ |
> | P                                                                                      | Weight_tiles+ $N_T * (R_T - 1) * (Q_T - 1 + S_T) * C_T$ | $(N_T * (Q_T - 1 + S_T) * C_T + Q_T * K_T * N_T) * P_T$             |
> | Q                                                                                      | Weight_tiles+ $N_T * (S_T - 1) * (P_T - 1 + R_T) * C_T$ | $(N_T * (P_T - 1 + R_T) * C_T + P_T * K_T * N_T) * Q_T$             |
> | R                                                                                      | Output_tiles+ $N_T * (P_T - 1) * (Q_T - 1 + S_T) * C_T$ | $(N_T * (Q_T - 1 + S_T) * C_T + K_T * S_T * C_T) * R_T$             |
> | S                                                                                      | Output_tiles+ $N_T*(Q_T-1)*(P_T-1+R_T)*C_T$             | $(N_T * (P_T - 1 + R_T) * C_T + K_T * R_T * C_T) * S_T$             |
> | Innut tiles Weight tiles and Output tiles are calculated an lines 10.00 of Algorithm 1 |                                                         |                                                                     |
> 
> Input\_tiles, Weight\_tiles, and Output\_tiles are calculated on lines 19-22 of Algorithm 1.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2038.png)

> **[图片提取文字 (image.png)]:**
> Table 2. The Architecture Parameters of Simba and Eyeriss
> 
> | Accelerators |                          | Simba   | Eyeriss |
> |--------------|--------------------------|---------|---------|
> | Chip         | Global buffer size       | 64 KB   | 128 KB  |
> |              | Number of PEs            | 16      | 256     |
> | PE           | Input buffer size        | 8 KB    | 12 B    |
> |              | Weight buffer size       | 32 KB   | 192 B   |
> |              | Accumulation buffer size | 3 KB    | 16 B    |
> |              | Number of MACs           | 64      | 1       |
> |              | Input/Weight Precision   | 8 bits  | 16 bits |
> |              | Partial-Sum Precision    | 24 bits | 16 bits |
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%2041.png)

## ISCA21：Cosa

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2: DNN scheduling problem formulation with CoSA. CoSA takes 1) DNN layer dimensions and 2) DNN accelerator parameters and expresses the scheduling problem into a constrained optimization problem to produce a performant schedule in one shot.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20121.png)

> **[图片提取文字 (image.png)]:**
> ```
> //DRAM level
> for q2 = [0 : 2):
> // Global Buffer level
> for p2 = [0 : 7) :
>  for q1 = [0 : 7) :
>   for n0 = [0 : 3):
>    spatial\_for r0 = [0 : 3) :
>    spatial\_for k1 = [0 : 2) :
>     // Input Buffer level
>     spatial\_for k0 = [0 : 2) :
>      // Weight Buffer level
>      for c1 = [0 : 2) :
>       for p1 = [0 : 2):
>        // Accumulation Buffer level
>        for s0 = [0 : 3) :
>        for p0 = [0 : 2):
>         spatial\_for c0 = [0 : 8) :
>          // Register
>          for q0 = [0 : 2):
> ```
> 
> 3
> 
> 4
> 
> 5
> 
> 6
> 
> 7
> 
> 8
> 
> 10
> 
> 11
> 
> 12
> 
> 13
> 
> 14
> 
> 15
> 
> 16
> 
> 17
> 
> 18
> 
> 19
> 
> Listing 1: An example schedule using the loop nest representation for a DNN layer of dimension R = S = 3, P = Q = 28, C = 8, K = 4, N = 3. Same variable prefix indicates tiles from the same problem dimension.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20122.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 5: Different traffic patterns based on the constant matrix **A**. The two figures (top) show how the constant **A** encodes the traffic types (multicast, unicast, reduction) for different data tensors from the global buffer to PEs. The figures on the bottom show its implication on output tensor reduction traffics.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20150.png)

> **[图片提取文字 (image.png)]:**
> |               | Related W   IA   OA |          |          | Idx v |                   | Related W   IA   OA |          |          | Idx v |
> |---------------|---------------------|----------|----------|-------|-------------------|---------------------|----------|----------|-------|
> | R             | ✓<br>✓              | -        |          | j     | Register AccBuf   | <b>✓</b>            | <b>✓</b> | <i>J</i> | i     |
> | P<br>Q        |                     | <i>√</i> | ✓<br>✓   |       | WBuf<br>InputBuf  | <b>✓</b>            | <b>✓</b> |          |       |
> | $\frac{C}{K}$ | ✓<br>✓              | <i>\</i> | <i>J</i> |       | GlobalBuf<br>DRAM | ✓<br>✓              | ✓<br>✓   | <b>✓</b> |       |
> 
> TABLE IV: Constant binary matrices A (left) and B (right). A encodes how different layer dimensions associate with data tensors. B encodes which data tensor can be stored in which memory hierarchy.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20138.png)

## TC24:TensorMap

> **[图片提取文字 (image.png)]:**
> ## II. BACKGROUND
> 
> ## A. Mapping for Tensor Computation
> 
> We provide an example of mapping a 1D convolution onto spatial accelerators. Given weight tensor W[2], input tensor I[6], and output tensor O[5], the 1D convolution can be represented as a loop nest:
> 
> ```
> for (x=0; x<6-2+1; x++) {
>   for (k=0; k<2; k++) {
>     O[x] += I[x+k] * W[k];
>   }
> }</pre>
> ```
> 
> In this loop, there exist no unrolling loops, the tiling size of each loop is 1, and the ordering is from the input tensor to the weight tensor (denoted as  $I \to W$ ).
> 
> To improve parallelism and data reuse, the loop nest can be rewritten. First, we can interchange the loops from  $I \to W$  to  $W \to I$  as follows:
> 
> ```
> for (k=0; k<2; k++) {
>   for (x=0; x<6-2+1; x++) {
>     O[x] += I[x+k] * W[k];
>   }
> }</pre>
> ```
> 
> Then, we can unroll the weight loop using multiple PEs of spatial accelerators as shown here:
> 
> ```
> parallel_for (k=0; k<2; k++) {
>   for (x=0; x<6-2+1; x++) {
>     O[x] += I[x+k] * W[k];
>   }
> }</pre>
> ```
> 
> Finally, we can add loops by changing the tiling size to exploit data reuse. This loop nest is rewritten as follows:
> 
> ```
> parallel_for (k=0; k<2; k++) {
>     for (x1=0; x1<6/2; x1++) {
>         for (x0=0; x0<2; x0++) {
>             x = x1*2 + x0;
>             O[x] += I[x+k] * W[k];
>         }
>     }
> }</pre>
> ```
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20162.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3. An example of mapping the 1D convolution layer on the spatial accelerator with 2 PEs.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20163.png)

> **[图片提取文字 (image.png)]:**
> an outer loop over the tiled loop. Fig. 3 shows the process of mapping 1D convolution onto a spatial accelerator. The weight tensor is spatially distributed across two PEs. Each weight of W[0] and W[1] is multicast across time, and each input of I[1], I[3] and I[5] is multicast across PEs. This mapping configuration exploits parallelism of weight loop, spatial reuse of input data, and temporal reuse of weight data. Similarly, the mapping of a 2D convolution onto accelerators can be achieved with the NVDLA mapping configuration shown in Fig. 1(c).
> 
> In this case, we set the tiling size of the input loop to 2, and add
> 
> ## The sub-space of each mapping primitive is described as: • $M_0 = \mathbb{Z}$ : the level of unrolling.
> 
> B. Mapping Space
> 
> - M<sub>1</sub> = Z: the unrolling loop at each level.
>   M<sub>2</sub> = Z<sup>d</sup>: the tiling size of each loop at each level, where
> - d is the number of loops. •  $M_3 = \mathbb{Z}$ : the loop ordering at each level.
> 
> Given a spatial accelerator A and a targe tensor computation T, the mapping space is defined as:
> 
>  $\mathbf{M_{A,T}} = \{ map \in (M_1 \times M_2 \times M_3)^{M_0} | 0 < H(map) \le H^* \}$  where H(map) represents the required hardware resources when using the mapping configuration map, and  $H^*$  represents the available hardware resources of the accelerator.
> 
> pends on both the target tensor computation and the available hardware resources of the spatial accelerator. The tensor computation impacts on the maximum of  $M_0$ ,  $M_1$ ,  $M_2$ , and  $M_3$ , while the accelerator determines the number of valid mapping configurations in the search space. For example, the dimension
> 
> In this manner, the size of the mapping space  $M_{A,T}$  de-
> 
> configurations in the search space. For example, the dimension of the UNet layer in Fig. 1(c) is represented as (K=128, C=128, Y=198, X=198, R=3, S=3). Given this layer, the complexity of the mapping space is expressed as  $O(10^{13*M_0}) = (6*6!*128*128*198*198*3*3)^{M_0}$ .
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20164.png)

## ICCD24：TileMap

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1. (a) Tensor operators in multi-head attention. (b) The operator-fusion mapping flow for attention.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20183.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 4. Map BMM-Softmax-BMM on an abstracted accelerator with a three-level memory hierarchy.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20192.png)

> **[图片提取文字 (image.png)]:**
> memory. In terms of NVDLA, L2, L1, and L0 represent global memory, shared memory, and register respectively. Input tiles flow towards lower-level memory, while output tiles flow towards successor operators and higher-level memory.
> 
> Based on tile alignment, the mapping space is expressed by
> 
> three primitives: inter-operator tiling, intra-operator tiling, and intra-operator parallelization. The inter-operator tiling involves choosing efficient tile factors in off-chip memory, including  $B1, H1, L_11$ . The intra-operator tiling is independent between operators, aiming to find optimized tile factors across the hierarchy. Specifically, we further split loops B0, H0,  $L_10$ ,  $L_2$ , and D in shared memory and transfer data to registers. The intra-operator parallelization is to parallelize loops using compute cores and select proper parallel factors. We observe that most mappings in the space are invalid to meet resource constraints of spatial accelerators. TileMap pre-calculates the maximal tile/parallel factors to ensure that each mapping
> 
> candidate is valid.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20193.png)

> **[图片提取文字 (image.png)]:**
> ```
> 1 // Off-chip memory
>   for b1, h1, l_11 in grid(B1, H1, L_11):
>     // On-chip memory
> 3
>     for b0, h0, l_10, l_2, d in grid(B0, H0, L_10, L_2, D):
>      b = b1*B0+b0, h = h1*H0+h0, ...
>      A[b, h, l_1, l_2] +=
>        Q[b, h, l_1, d] * K[b, h, d, l_2]
>     for b0, h0, l_10, l_2 in grid(B0, H0, L_10, L_2):
>      b = b1*B0+b0, h = h1*H0+h0, ...
>      A'[b, h, l_1, l_2] = Softmax(A[b, h, l_1, l_2])
> 10
>     for b0, h0, l_10, l_2, d in grid(B0, H0, L_10, L_2, D):
>       b = b1*B0+b0, h = h1*H0+h0, ...
> 12
>       O[b, h, l_1, d] +=
> 13
>        A'[b, h, l_1, l_2] * V[b, h, l_2, d]
> 14
> ```
> 
> Fig. 2. Pseudo-code for BMM-Softmax-BMM mapping.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20188.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1. (a) Tensor operators in multi-head attention. (b) The operator-fusion mapping flow for attention.
![image.png](meeting-25%2010%2017%EF%BC%88%E5%A4%9A%E6%A0%B8%E4%BB%BF%E7%9C%9F%E3%80%813DGS%E3%80%81RL%20schedule%EF%BC%89/image%20183.png)

## 25：BSMM

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: Tradeoffs among different implementations of transformer blocks.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2049.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 7: Our approach: hybridizing structured sparsity and FFT (Decompression is symmetric and omitted)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2064.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Improve transformer blocks using structured sparsity.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Improve transformer blocks using structured sparsity.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Continuous BPMM Applied on a Vector (Lower half omitted)
> 
> matrix multiplications (BSMMs).
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2056.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 10: Allocating computing resources for BSMMs. (For clarity, batch-based SIMD and vertical hops for stride =4,8 are omitted.)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2090.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> is vectorized
> 
> (a) Map a Single Layer to PE Mesh
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Fig. 11: Mapping a dense MM to MLX in multi-layer dataflow
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2091.png)

4个边输入4个块，中间十字输入barrier，对比其他PE。

激活稀疏最好在局部SA？

recfg SA可能用于隐藏稀疏解码延迟，dense+sp