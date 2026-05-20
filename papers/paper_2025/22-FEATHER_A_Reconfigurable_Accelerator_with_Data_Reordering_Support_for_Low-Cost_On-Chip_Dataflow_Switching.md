2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA) 

## FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching 

Jianming Tong Anirudh Itagi _Georgia Institute of Technology Georgia Institute of Technology_ Atlanta, Georgia, USA Atlanta, Georgia, USA jianming.tong@gatech.edu aitagi7@gatech.edu 

## Prasanth Chatarasi Tushar Krishna 

_IBM Research Georgia Institute of Technology_ Yorktown Heights, USA Atlanta, Georgia, USA prasanth@ibm.com tushar@ece.gatech.edu 

_**Abstract**_ **—The inference of ML models composed of diverse structures, types, and sizes boils down to the execution of different dataflows (i.e. different tiling, ordering, parallelism, and shapes). Using the optimal dataflow for every layer of workload can reduce latency by up to two orders of magnitude over a suboptimal dataflow. Unfortunately, reconfiguring hardware for different dataflows involves on-chip data layout reordering and datapath reconfigurations, leading to non-trivial overhead that hinders ML accelerators from exploiting different dataflows, resulting in suboptimal performance. To address this challenge, we propose** _**FEATHER**_ **, an innovative accelerator that leverages a novel spatial array termed** _**NEST**_ **and a novel multi-stage reduction network called** _**BIRRD**_ **for performing flexible data reduction with layout reordering under the hood, enabling seamless switching between optimal dataflows with negligible latency and resources overhead. For systematically evaluating the performance interaction between dataflows and layouts, we enhance Timeloop, a state-of-theart dataflow cost modeling and search framework, with layout assessment capabilities, and term it as Layoutloop. We model** _**FEATHER**_ **into Layoutloop and also deploy** _**FEATHER**_ **end-to-end on the edge ZCU104 FPGA.** _**FEATHER**_ **delivers** 1 _._ 27 _∼_ 2 _._ 89 _×_ **inference latency speedup and** 1 _._ 3 _∼_ 6 _._ 43 _×_ **energy efficiency improvement compared to various SoTAs like NVDLA, SIGMA and Eyeriss under ResNet-50 and MobiletNet-V3 in Layoutloop. On practical FPGA devices,** _**FEATHER**_ **achieves** 2 _._ 65 **/** 3 _._ 91 _×_ **higher throughput than Xilinx DPU/Gemmini. Remarkably, such performance and energy efficiency enhancements come at only** 6 **% area over a fixed-dataflow Eyeriss-like accelerator. Our code is released at https://github.com/maeri-project/FEATHER.** 

_**Index Terms**_ **—Reconfigurable Accelerator, Dataflow, Layout** 

## I. INTRODUCTION 

The field of Machine Learning (ML), specifically Deep Neural Networks (DNNs) is pervasive today across image classification [12], [40], object detection [3], [37], text summarization [20] and sentiment analysis [25]. Such a plethora of ML models introduces great diversity in structure (serial or parallel layers connectivity), layer types (depth-width, pointwidth, dilation convolutions, or even a fusion of them), and sizes (number of channels, kernels, height, and width) [7], [49]. 

The mechanism for orchestrating a DNN layer over the accelerator’s on-chip compute and memory resources is called _dataflow_ . It can be precisely defined by transformations of the loop nest, as shown in Fig. 1. Several prior works [33], [41] have demonstrated that dataflows can lead to significant differences in compute utilization and up to two orders of magnitude variance in latency and energy, and thereby motivated the need to support per-layer dataflow flexibility. 

**==> picture [236 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
Input Activations (iActs) Sliding Window Result Output Activations (oActs)<br>Sliding Window<br>Level-0 Tilefor nt in [0,N,1): Tile Level-1 Tilefor n in [0,Kernel 11Kernel 2,1): Kernel MVirtual Grouping of PEsShape Data OrganizationLine SizeLayout<br> for mt in [0,M,16):  for x in [0,3,1): PE PE PE PE C0H0W0:7<br>  for ct in [0,C,   for xt in [0,R,    for yt in [0,Q,     for rt in [0,R,643):):3):3):   for y in [0,   for r in [0,    for s in [0,     for m in [0,3,13,13,1):16):):,16): PEPEPE PEPEPE PEPEPE PEPEPE Bank C0H1W0:7C0H2W0:7C0H3W0:70 1 2<br>      for st in [0,Q,3):       for c in [0,64,16): Memory Hierarchy<br>Order Order Parallelism #Line /HBM DRAM<br>**----- End of picture text -----**<br>


Fig. 1: Terminology of convolution workload and dataflow 

Changing dataflows on accelerators requires (a) reconfiguring datapaths in computation, distribution, and reduction networks, and (b) modifying data layout in on-chip buffers. Almost all prior works have focused on the first aspect, and several clever interconnect topologies for data distribution and reduction have been proposed that activate subset of paths at runtime through reconfiguration depending on the dataflow being run [42], [44]. However, data layout in the on-chip buffer is a critical and often overlooked in past work. 

In this work, we demonstrate that the high performance of dataflows is unachievable in practice without layout reordering capability. This is because, without a suitable data layout, the required data may be located in the same SRAM banks and compete at the same SRAM reading ports. Such bank conflict slows down the delivery of data to computation engines, leading to stalling and computation underutilization. Overlooking layout reordering thus introduces a significant 128 _×_ performance gap between theory and practice as quantified in Fig. 2. We discuss this with more depth in §II. 

Unfortunately, layout reordering comes with severe latency and energy overheads. Off-chip layout reordering requires back-and-forth data movement between off-chip DRAM/HBM and computation, while on-chip layout reordering requires additional intermediate storage and extra latency in the critical path. In fact, these costs can outweigh the benefits of switching dataflows, leading existing ML accelerators to compromise settling on a single dataflow (e.g., Xilinx DPU, Gemmini, NVDLA, Eyeriss in Table I) that provides good average utilization across all layers, but sub-optimal performance. 

To unleash optimal performance, we propose a novel accelerator _FEATHER_ , Flexible Engine for Acceleration of Tensors with Hardware Element for Reordering, which includes a novel reconfigurable reduction network called Butterfly Interconnect 

979-8-3503-2658-1/24/$31.00 ©2024 IEEE 198 DOI 10.1109/ISCA59077.2024.00024 

TABLE I: Feature comparison: how FEATHER resolves challenges of prior works without on-chip layout reordering. 

||Work|Work|||||||Datafow Switching|Datafow Switching|Datafow Switching|Datafow Switching|Datafow Switching|Datafow Switching|Datafow Switching||Layout Reorder|Layout Reorder|Layout Reorder|Layout Reorder|Layout Reorder|Layout Reorder|Challenge|**_FEATHER_** solution (key component)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||NVDLA [39]||||||||||||||||no reorder||||||underutilization from fxed parallelism|fexible datafows (_NEST_)|
||Xilinx DPU [51], Gemmini|||||[21]|||||||||||no reorder||||||linear reduction|parallel logarithmic reduction (_BIRRD_)|
||SIMBA [47],|Eyeriss|||[13]||||||||||||no reorder||||||load imbalance across PE|pick load-balance datafows (_NEST_)|
||Eyeriss<br>v2 [15], SARA [44]|||||||||||||||||off-chip|||||high latency of moving data off-chip|on-chip reordering with latency hidden (_BIRRD_)|
||MAERI [35],|SIGMA|||[42]|||||||||||||off-chip|||||long wires of reduction network|small standalone reduction network (_BIRRD_)|
||||||||||||||||||||||||TABLE II:|On-chip memory terminology|
||||||||||||||||||||||||Term<br>|Meaning<br><br><br>|



||TABLE II: On-chip memory terminology|
|---|---|
|Term|Meaning|
|Buffer|A logical 2D on-chip memory (num<br>line _×_ line<br>size) stacking multiple<br>SRAM banks both vertically (num<br>line) and horizontally (line<br>size).|
|Bank|A physical 2D SRAM (entries _×_ io) with address/data ports.|
|Line/Row|A buffer line (line<br>size = accumulated IO of horizontal SRAM banks).|
|Port|An input/output port, each bank has at most two ports in TSMC 28nm.|



**==> picture [200 x 45] intentionally omitted <==**

representation, and dataflow-layout co-search. We call this new framework _Layoutloop_ (§V) and use it for our evaluations. 

_•_ We implement and deploy _FEATHER_ , end-to-end, on an edge ZCU 104 FPGA device and also model it using _Layoutloop_ . _FEATHER_ achieves 1 _._ 27 _∼_ 2 _._ 89 _×_ inference latency speedup and 1 _._ 3 _∼_ 6 _._ 43 _×_ energy efficiency improvement compared to various SoTAs across multiple DNN models, and 2 _._ 65 _×_ /3 _._ 91 _×_ more throughput than Xilinx DPU/Gemmini on real FPGAs. On average, efficient pairs of (dataflow, layout) results in an energy savings of 27% to 33% across workloads despite the energy costs of layout reordering. Remarkably, all enhancements come at only 6% area over a fixed-dataflow Eyeriss-like accelerator. 

Fig. 2: Latency evaluation of dataflows on 16 _×_ 16 PE array with various layouts (error bar shows layout impacts, less latency is better). The best flexible dataflow (green bar) _theoretically_ reduces overall latency of fixed dataflow-layout (blue bar) by 63 _._ 3%. However, ignoring the impact of layout considerations in theoretical dataflows results in up to a 128 _×_ latency gap in _practice_ (yellow bar). FEATHER eliminates the gap by coswitching dataflow-layout (red bar). 

for Reduction and Reordering in Dataflows ( _BIRRD_ ). With _BIRRD_ , the latency of layout reordering is completely hidden in data reduction, allowing data layout in on-chip storage to be manipulated for the demand of optimal dataflow without any latency costs. We call this approach as reordering in data reduction (RIR). Thus, _FEATHER_ fully achieves the theoretical performance of optimal dataflows without incurring bank conflicts. Furthermore, _FEATHER_ also pioneers a new paradigm to co-switch both dataflows and data layouts at layer granularity, with minimal switching overheads. This ability to accommodate low-cost layout-dataflow co-switching is, as far as we know, unsupported by any existing accelerator. 

## II. BACKGROUND AND MOTIVATION 

## _A. Dataflow Space in Convolution_ 

Fig. 1 depicts a convolution operation with seven dimensions with various shapes. Dataflows can be represented as a nested loop with four types of optimizations [24], [34]. 

_•_ **(T)iling** breaks down dimensions of iActs _N,C, H,W_ into smaller chunks, and enables executing workloads in tile granularity as on-chip storage is limited. 

_•_ **(O)rdering** allows arbitrary loop reordering (aka “stationarity” [13]) to reuse more data since dimensions _N, M,C, P, Q, R, S_ do not come with loop-carried dependencies except reductiondependencies over C, R, and S. 

To fully explore the potential of _FEATHER_ , we also developed a tool that facilitates: (a) dataflow evaluation factoring in data layout, and (b) (layout, dataflow) co-exploration. Our key contributions can be summarized as follows: 

_•_ **(P)arallelism** allows for arbitrary parallelism over any dimensions as all dependencies are loop-independent, leading to different spatial reuse opportunities. 

> _•_ We demonstrate the interaction between dataflows and data layouts, motivating the need for data reordering support within reconfigurable dataflow accelerators. We further categorize existing reordering patterns and implementations (§II). 

_•_ **(S)hape** defines the virtual grouping of the physical PE array. 

These _dataflow flexibility (TOPS)_ [34] create an extremely large dataflow design space with a complexity of _O_ (10[36] ) for a single convolution layer [27]. The choice of the dataflow affects both runtime performance (as it affects overall compute utilization) and energy efficiency (as it affects the number of accesses across the memory hierarchy). Not surprisingly, no single dataflow is generally optimal for all types of layers given their diverse sizes and shapes [33], [41]. This can be seen by comparing the first two bars (blue and green bars) in Fig. 2. 

_•_ We present a novel accelerator _FEATHER_ with several novel features (§III). First, a neural engine with temporal local reduction and spatial forwarding, _NEST_ , for dataflow flexibility. Second, a multi-stage network called _BIRRD_ enabling flexible reductions from arbitrary groups of multiple inputs to multiple results, at lower area overhead compared to prior works with similar capabilities. Further, BIRRD supports Arbitrary reorder via a novel technique RIR, that completely conceals data layout reordering latency behind reduction (§IV). 

## _B. Data Layout in on-chip Storage_ 

Various organizations of on-chip storage are logically a 2D buffer (Tab. II), where the width of each logical buffer row, termed “line size”, represents bandwidth (max number of data 

_•_ We extend a state-of-the-art accelerator modeling framework Timeloop [41] with support for physical on-chip storage, layout 

199 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [237 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
Layer Size: C56H8W8 - Layout:�# Inter-line dimension orderCHW_W4H2C2 H0:1 W0:3W4:7 W0:3W4:7H0:1H0:1C0:1C0:1<br>C      [Start, End, Step) for ct in [0,56,2): C0:1 H2:3 W0:3 W0:3H2:3C0:1<br>H  for ht in [0,8,2): H6:7 W4:7 W4:7H6:7C0:1<br>W  for wt in [0,8,4): C2:3 H0:1 W0:3 W0:3H0:1C2:3<br># Intra-line dimension order WH<br>W4H2  for w in [wt,wt+for h in [ht,ht+42,1):,1): W0H0W0H0 W0H1 W0H1 W1H0 W1H0 W1H1 W1H1 W2H0 W2H0 W2H1 W2H1 W3H0 W3H0 W3H1 W3H1 C<br>C2  for c in [ct,ct+2,1): C0 C1 C0 C1 C0 C1 C0 C1 C0 C1 C0 C1 C0 C1 C0 C1<br>Detailed View<br>**----- End of picture text -----**<br>


Fig. 3: Layout terminology example: ‘CHW W4H2C2’. ‘CHW’ signifies the inter-line dimension order as C _→_ H _→_ W across lines. ‘W4H2C2’ indicates the intra-line dimension order: (4,2,2) elements from the (W,H,C) dimensions are flattened into a single row in the order of W _→_ H _→_ C. 

words a buffer could supply per cycle) and the depth represents the total number of buffer row entries as shown in Fig. 1. 

Physically, on-chip storage is implemented by BRAM/URAM in FPGA and SRAM in ASICs, which come with a _fixed number (often two) read or write ports_ . Therefore, once arranged into the logical 2D buffer, the number of lines being concurrently accessed is limited by the number of ports. A request that accesses more lines than the available ports will lead to bank conflicts, resulting in a slowdown from the reading/writing delay (resource hazard). 

**Data Layout Terminology.** In this paper, data layout is represented as “(Inter-line dimension order) (Intra-line dimension order interleaved with sizes)” with one example shown in Fig. 3. For instance, two commonly used PyTorch data layouts, channel-last [18] and row-major [38], can be interpreted as Channel (C) or Width (W) being the innermost dimension in both inter and intra-line orders, separately. 

## _C. Interaction of Dataflow and Data Layout_ 

In the rest of the paper, we refer to a (dataflow, layout) pair with bank conflicts as _**discordant**_ , whereas its non-conflicting counterpart is termed _**concordant**_ , i.e. a layout is concordant to a dataflow if there are no bank conflicts. And we use _**concordant dataflow space of a layout**_ to refer to all concordant dataflows choices under a layout. Switching optimal dataflows for different layers is not trivial given that it necessitates a costly reordering to convert the data layout into a concordant form to prevent bank conflicts. 

In this subsection, we discuss some crucial insights, underscoring the necessity of co-switching dataflows-layouts for different layers by evaluating the performance of various combinations of dataflows and data layouts as shown in Fig. 4. 

_Insight 1: Discordance between dataflow and data layout leads to bank conflicts and results in performance degradation._ 

A discordance between dataflow and data layout leads to slowdown because compute units have to stall and wait for data to arrive, as illustrated by the slowdown from green bar to yellow bar in Fig. 2. Taking ResNet-50 layer 47 as an example (Fig. 4-M7), the channel-parallel dataflow requires concurrent access to iActs (H0W0C0:3), which are distributed across four separate lines, including line 0, r4, r5 and r6, in the row-major 

layout (Fig. 4-L4). Therefore, a 0 _._ 5 slowdown is encountered, resulting in 50% practical computation utilization. Such bank conflicts cannot be resolved by line rotation, since moving one conflicted line to another bank leaves the remaining three lines still in conflict. This slowdown analysis also applies to Fig. 4-M1,2,3,6. 

_Insight 2: Co-switching (dataflow, layout) for different layers is necessary for high performance with optimal efficiency._ 

For certain workloads, picking a fixed layout might not suffer a slowdown from bank conflicts, like choosing rowmajor layout for both two layers of ResNet-50 (M4 and M8 in Fig. 4). However, the mapping M5 (“ _FEATHER_ ’s pick”) delivers better energy efficiency than M8 as it supplies data with reading less number of lines. Therefore, even under a small parallelism of four, co-switching dataflows and layouts is essential to maximize performance and energy efficiency. Practical designs (e.g. 128 _×_ 128 systolic array in Google TPU) will further amplify such a need as it brings higher parallelism in more dimensions and requires more concurrent data. 

_Insight 3: Systematic layout modeling should be factored into dataflow exploration for bridging the theory-practice gap._ 

Dataflow has a huge space, which requires systematic modeling and searching algorithms to identify the optimum. However, many dataflow exploration frameworks [33], [41] and algorithms [9], [27], [28], [30] purely model on-chip storage as bandwidth, often assuming ideal data layouts, which could lead to significant theory-practice performance gap. For instance, all layouts in Fig. 4 possess identical bandwidth, but they result in markedly different compute utilization and energy efficiency for two workloads, which is not the case in the existing frameworks as they do not model layout. In Fig. 2, we find that the best dataflow reported by a mapper from an existing framework [41] (green bar), can in practice perform 2 orders of magnitude worse (yellow bar) than the fixed dataflow case (blue bar) due to the discordant accesses to the on-chip memory. Thus, taking layout into consideration _during_ search (red bar) is necessary and crucial. 

## _D. Data Reordering Patterns_ 

_1) Reorder Target (iActs):_ As established above, both weights and input activations (iActs) necessitate layout reordering within the on-chip memory when switching dataflows. For ML inference, the structure and weights of ML models are established prior to deployment, enabling the offline optimal dataflow-layout determination for each layer and offline reordering of all weights. Consequently, an optimal layout for weights within the on-chip scratchpad is assured. However, iActs are generated in real-time, so that iActs reordering happens online. Therefore, this work focuses on layout reordering of iActs. 

_2) Reorder Patterns vs. Implementations:_ Layout transformations require certain reorder capabilities, referred to as reorder patterns. A reorder pattern has different hardware implementations _with different critical-path latency_ . To decouple the concept of reorder patterns from their physical implementations, we analyze reordering in two steps: (1) categorize reordering 

200 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [432 x 362] intentionally omitted <==**

**----- Start of picture text -----**<br>
Channel-parallel� Dataflow (D1) iActs sliding-windows parallel  Dataflow (D2)<br>C Parallel=4 W Parallel=4<br>000111222333444<br>0 1 2 3 4 0 0 0 0 1 1 1 1<br>5 ... 2 2 2 2 3 3 3 3<br>Weights iActs Weights iActs<br>Line#0 Line Size = 8 iActsH0W0:1C0:2 Empty Cycle# byRequired MaiActspping AccessedLine #being accessed lines(total ports/Slowdown ) TheoreticalCompute�Utilization UtilizationComputePractical Cycle# iActs Required byMapping AccessedLine #being accessed lines(total ports/Slowdown ) TheoreticalCompute�Utilization UtilizationComputePractical<br>1 H0W2:3C0:2 Empty 0 H0W0 C0:2 0 no slowdown 75% 75% 0 H0C0 W0,2,4,6 0,1,2,3 2/4=0.5 100% 50%<br>ResNet-50 342 H0W4:5H0W6:7H0W8:9C0:2C0:2C0:2 EmptyEmptyEmpty ...123 H0W1H0W2H0W3... C0:2C0:2C0:2 ...011 no slowdownno slowdownno slowdown... 75%75%75%... 75%75%75%... ...12 H0C0H0C0Stride=2 in workload W16W8 ... ,10,18,12,20,14,22 four lines4,5,6,7... 2/4=0.52/4=0.5 ... 100%100%... 50%50% ...<br>layer 1 5 H0W10:11C0:2 Empty C=3 in workload<br>6 H0W12:13C0:2 Empty Memory Efficiency: read 1 line per cycle; Memory Efficiency: read 4 lines per cycle;<br>C=3 Compute Utilization: 75% Compute Utilization: 50%<br>H=224 (M1) (M2)<br>padding=3Stride=2W=224R=S=7 Line#01 Line Size = 8 iActsH0C0W8:15H0C0W0:7 Cycle01# bH0W0H0W1yRequired MaiActspp C0:2C0:2 ing Accessed0, r1, r20, r1, r2Line #being accessed lines(total ports/Slowdown2/3=0.6672/3=0.667 ) TheoreticalCompute�Utilization75%75% UtilizationComputePractical 50%50% Cycle01# H0C0iActs Required byH0C0Mapping W8W0,10,2,12,4,,614 AccessedLine #being01 accessed lines no slowdownno slowdown (total ports/Slowdown ) TheoreticalCompute�Utilization100%100% Compute�UtilizationPractical100%100%<br>r1 H0C1W0:7 2 H0W2 C0:2 0, r1, r2 2/3=0.667 75% 50% ...2 H0C0 W16 ... ,18,20,22 one line... no slowdown... 100%... 100%...<br>r1+1 H0C1W8:15 ...... H0W3C=3 in workload... C0:2 0, r1, r2... 2/3=0.667 ... 75%... 50% ... Stride=2 in workload FEATHER's pick<br>r2 H0C2W0:7 Memory Efficiency: read 3 line per cycle; Memory Efficiency: read 1 lines per cycle;<br>r2+1 H0C2W8:15 Compute Utilization: 50% Compute Utilization: 100%<br>(M3) (M4)<br>Line#01 Line Size = 8 iActsH0W0C0:7H0W1C0:7 Cycle#01 bH0W0H0W1yRequired MaiActspp C0:3C0:3 ing AccessedLine #being01 accessed lines no slowdownno slowdown (total ports/Slowdown ) TheoreticalCompute�Utilization100%100% ComputeUtilizationPractical100%100% Cycle01# bH0C0H0C0yRequired MaiActs W0:3W4:7 pping Accessed0,1,2,31,2,3,4Line #being accessed lines(total ports/Slowdown2/4=0.52/4=0.5 ) TheoreticalCompute�Utilization100%100% Compute�UtilizationPractical 50%50%<br>2 H0W2C0:7 2 H0W2 C0:3 2 no slowdown 100% 100% 2 H1C0 W0:3 four lines 2/4=0.5 100% 50%<br>3 H0W3C0:7 3 H0W3 C0:3 3 no slowdown 100% 100% 3 H1C0 W4:7 four lines 2/4=0.5 100% 50%<br>ResNet-50 ... ... ... ... ... ... ... ... ... ... ... ...<br>r3 H2W1C0:7 Efficiency: read 1 line/cycle, better than M8; Memory Efficiency: read 4 lines per cycle;<br>layer 47� r3+1 H2W2C0:7 Compute Utilization: 100% Compute Utilization: 50%<br>C=2048 (M5)� FEATHER's pick (M6)<br>padding=1Stride=1H=W=7R=S=3 Line#01 Line Size = 8 iActsH0C0W8H0C0W0:7Empty Cycle01# bH0W0H0W1yRequired MaiActspp C0:3C0:3 ing Accessed0,r4,r5,r60,r4,r5,r6Line #being accessed lines(total ports/Slowdown2/4=0.52/4=0.5 ) TheoreticalCompute�Utilization100%100% UtilizationComputePractical 50%50% Cycle01# bH0C0H0C0yRequired MaiActs W0:3W4:7 pping AccessedLine #being00 accessed lines no slowdownno slowdown (total ports/Slowdown ) TheoreticalCompute�Utilization100%100% Compute�UtilizationPractical100%100%<br>r4 H0C1W0:7 2 H0W2 C0:3 0,r4,r5,r6 2/4=0.5 100% 50% 2 H1C0 W0:3 one line no slowdown 100% 100%<br>r4+1 H0C1W8 Empty ...3 H0W3... C0:3 0,r4,r5,r6... 2/4=0.5 ... 100%... 50% ... ... 3 H1C0 ... W4:7 one line ... no slowdown ... 100% ... 100% ...<br>r5 H0C2W0:7 Memory Efficiency: read 4 line per cycle; Memory Efficiency: read 1 or 2 lines per cycle;<br>r6 H0C3W0:7 Compute Utilization: 50% Compute Utilization: 100%<br>(M7) (M8)<br>)L1 M Parallel=4 M Parallel=4<br> Layout (<br>(HWC_W2C3)<br>Channel-Last<br>)L2<br> Layout (<br>(HCW_W8)<br>Row-Major<br>)L3<br> Layout (<br>(HWC_W2C3)<br>Channel-Last<br>)L4<br> Layout (<br>(HCW_W8)<br>Row-Major<br>**----- End of picture text -----**<br>


Fig. 4: Memory efficiency and computation utilization of various **(workload, dataflow, data layout)** combinations on weightstationary 4 _×_ 4 Systolic Array (SA). **Dataflows:** input channel-parallel (D1) and sliding-window parallel (D2). Dataflow D1/D2 reads at most four iActs from C/W dimension concurrently from the on-chip buffer every cycle, separately. The digit in iActs indicates the cycle index such iActs get read. **Workloads:** (1) ResNet-50 layer 1 with a large height and width, and (2) ResNet-50 layer 47 with a large channel number. **Layouts:** channel last-layout (L1, L3) and row-major layout (L2, L4). In the channel-last layout, data from different input channels (dimension _C_ ) are spread across an individual line, while in the row-major layout, multiple data from different input width (dimension _W_ ) are flattened. The performance of mappings (M1 _∼_ M8) for different (workload, dataflow, layout) combinations are analyzed in the tables. In each table, “iActs Required by Mapping” lists all iActs that need to be concurrently read from on-chip buffer every cycle, and the corresponding index (#) of lines being accessed are listed in “Line # being Accessed”. We assume dual read ports (because TSMC offers SRAM with at most two ports), such that a concurrent read for more than two lines leads to slowdown, which reduces “Theoretical Computation Utilization” (estimated as mapping efficiency over the array) into “Practical Compute Utilization” (computed as multiplication of theoretical utilization with slow down). **Takeaway:** For optimal performance, co-switching (dataflow, layout) is crucial, because _dataflow_ matters (comparing M1 vs. M4), and _layout_ also matters (comparing M2 vs. M4). 

into distinct functional patterns, as illustrated in Fig. 5, and analyze its impact on dataflow flexibility in §II-D3. (2) pinpoint specific hardware implementations to these patterns in §II-E. 

_3) Impact of Reorder Patterns on Dataflow Flexibility:_ A fixed layout has limited concordant dataflow space, restricting fully-flexible accelerators to less-performant dataflow choices. To improve performance, reordering is required to enlarge 

concordant dataflow space with more flexibility in TOPS. 

_•_ **Fixed layout** (Fig. 5a) is only concordant to dataflows which concurrently access up-to two rows within a single bank, such as (0 _,_ 1 _,_ 2 _, ··· ,_ 7). This restricts concordant dataflow space to limited T,O,P,S flexibility (see purple quadrilateral in Fig. 5f). _•_ **Line Rotation** (Fig. 5b) arguments concordant dataflow space 

201 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [449 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
Typical 2D buffer (C,D,E,F) moves from Bank 0 -> 1 Transpose in-row Reorder Arbitrary Reorder O rder<br>0 1 2 3 0 1 2 3 C D E F 0 4 8 C 0 2 3 1 0 4 3 1 Fix Layout<br>4 5 6 7 4 5 6 7 1 5 9 A 6 4 7 5 6 2 7 C Line Rotation<br>8 9 A B 8 9 A B 2 6 A E B 8 9 A B 8 E A Row ReorderTranspose<br>C D E F C D E F 3 7 B F D F E C D F 9 5 Arbitrary Reorder<br>bank 0 bank 0 bank 1 bank 0 bank 0 bank 0 S hape<br>(a) Initial layout (b) Line Rotation (c) Transpose (d) Row-Reorder (e) Arbitrary Reorder (f) Concordant Dataflow Space<br>Move<br>arallel<br>P<br>2 ports<br>iles<br>T<br>**----- End of picture text -----**<br>


Fig. 5: Overview of reordering _patterns_ . The 2D layout without any reordering is shown in 5a, which only allows reading two rows concurrently, assuming true dual-port SRAM. Line Rotation (5b, e.g., Medusa [48]) moves a row from bank 0 to bank 1 prior to reading, enabling simultaneous access to at most three rows from bank 0 through dual-bank ports. This technique, however, utilizes additional port from bank 1, potentially limiting access to other data in bank 1. Transpose (5c, e.g., MTIA [19] and TPUv4i [26]) could swap rows with columns. Row Reorder (5d, e.g., TPUv4i [26]) permutes data within each row. Arbitrary reorder (5e, proposed in this work) enables arbitrary permutation for data within the entire 2D buffer. Line Rotation, Transpose and Row-Reorder are done by prior works by reading at most two rows per bank, leverage Transpose/Permute unit to reorder and then write data back in concordant order (On-chip RAR in 6b). In contrast, _FEATHER_ ’s _BIRRD_ network (§III-B) performs the Arbitrary-Reorder during the reduction phase of the matrix multiplication or convolution computation (RIR in Fig. 6c). The concordant dataflow space supported by each layout reorder pattern is shown in 5f. _Reordering enables a given layout to alter the order of data it could provide_ _**per cycle** and_ _**across cycles** ._ Among four dimensions (T,O,P,S) of concordant dataflow space, reordering enlarges O,P,S by supporting dataflows to read from or write to layout in different order. Note that reordering by itself cannot enlarge T dimension flexibility because higher Tiles flexibility requires accessing more data per cycle. 

|**iA**|DRAM<br>On-Chip<br>Storage(OCS)<br>Compute<br>Distributio~~n~~<br>Reduction<br>**Extralatency/energy overhead**<br>**from redundant data movements**<br>**cts (row-major)**<br>**cts (channel-last)**<br>**Off-chip data reorder criticalpath**<br>pute<br>CPU<br>Reorder<br>Write Data<br>to DRAM<br>Read Data<br>to OCS<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**8 write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**8 Read (4 cycles)**<br>**iActs (Channel-last)**<br>**iActs (row-major)**<br>**Multi-cycle latency from bank conficts**<br>**On-chip data reorder critical path**<br>Compute<br>Reorder<br>Compute<br>Reduction<br>**Reorder After Reduction**|DRAM<br>On-Chip<br>Storage(OCS)<br>Compute<br>Distributio~~n~~<br>Reduction<br>**Extralatency/energy overhead**<br>**from redundant data movements**<br>**cts (row-major)**<br>**cts (channel-last)**<br>**Off-chip data reorder criticalpath**<br>pute<br>CPU<br>Reorder<br>Write Data<br>to DRAM<br>Read Data<br>to OCS<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**8 write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**8 Read (4 cycles)**<br>**iActs (Channel-last)**<br>**iActs (row-major)**<br>**Multi-cycle latency from bank conficts**<br>**On-chip data reorder critical path**<br>Compute<br>Reorder<br>Compute<br>Reduction<br>**Reorder After Reduction**|DRAM<br>On-Chip<br>Storage(OCS)<br>Compute<br>Distributio~~n~~<br>Reduction<br>**Extralatency/energy overhead**<br>**from redundant data movements**<br>**cts (row-major)**<br>**cts (channel-last)**<br>**Off-chip data reorder criticalpath**<br>pute<br>CPU<br>Reorder<br>Write Data<br>to DRAM<br>Read Data<br>to OCS<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**8 write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**8 Read (4 cycles)**<br>**iActs (Channel-last)**<br>**iActs (row-major)**<br>**Multi-cycle latency from bank conficts**<br>**On-chip data reorder critical path**<br>Compute<br>Reorder<br>Compute<br>Reduction<br>**Reorder After Reduction**|DRAM<br>On-Chip<br>Storage(OCS)<br>Compute<br>Distributio~~n~~<br>Reduction<br>**Extralatency/energy overhead**<br>**from redundant data movements**<br>**cts (row-major)**<br>**cts (channel-last)**<br>**Off-chip data reorder criticalpath**<br>pute<br>CPU<br>Reorder<br>Write Data<br>to DRAM<br>Read Data<br>to OCS<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**8 write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**8 Read (4 cycles)**<br>**iActs (Channel-last)**<br>**iActs (row-major)**<br>**Multi-cycle latency from bank conficts**<br>**On-chip data reorder critical path**<br>Compute<br>Reorder<br>Compute<br>Reduction<br>**Reorder After Reduction**|DRAM<br>On-Chip<br>Storage(OCS)<br>Compute<br>Distributio~~n~~<br>Reduction<br>**Extralatency/energy overhead**<br>**from redundant data movements**<br>**cts (row-major)**<br>**cts (channel-last)**<br>**Off-chip data reorder criticalpath**<br>pute<br>CPU<br>Reorder<br>Write Data<br>to DRAM<br>Read Data<br>to OCS<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**8 write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**8 Read (4 cycles)**<br>**iActs (Channel-last)**<br>**iActs (row-major)**<br>**Multi-cycle latency from bank conficts**<br>**On-chip data reorder critical path**<br>Compute<br>Reorder<br>Compute<br>Reduction<br>**Reorder After Reduction**|DRAM<br>On-Chip<br>Storage(OCS)<br>Compute<br>Distributio~~n~~<br>Reduction<br>**Extralatency/energy overhead**<br>**from redundant data movements**<br>**cts (row-major)**<br>**cts (channel-last)**<br>**Off-chip data reorder criticalpath**<br>pute<br>CPU<br>Reorder<br>Write Data<br>to DRAM<br>Read Data<br>to OCS<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**8 write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**8 Read (4 cycles)**<br>**iActs (Channel-last)**<br>**iActs (row-major)**<br>**Multi-cycle latency from bank conficts**<br>**On-chip data reorder critical path**<br>Compute<br>Reorder<br>Compute<br>Reduction<br>**Reorder After Reduction**|**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**ts (Channel-last)**<br>**iActs (row-major)**<br>**On-chip data reorder critical path**|**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**ts (Channel-last)**<br>**iActs (row-major)**<br>**On-chip data reorder critical path**|**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**ts (Channel-last)**<br>**iActs (row-major)**<br>**On-chip data reorder critical path**|**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**ts (Channel-last)**<br>**iActs (row-major)**<br>**On-chip data reorder critical path**|**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**ts (Channel-last)**<br>**iActs (row-major)**<br>**On-chip data reorder critical path**|**H0W0C0:7**<br>**H0W1C0:7**<br>**H0C0W0:7**<br>**H0C1W0:7**<br>**H0C7W0:7**<br>**write (4 cycles)**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**ts (Channel-last)**<br>**iActs (row-major)**<br>**On-chip data reorder critical path**|Compute<br>Compute<br>**Reorder**in<br>Reduction<br>**RIR critical path**<br>**Multi-cycle reorder be hidden behind reduction**<br>**iActs -> Compute (RIR)-> oActs w/ new layout**<br>**Reorder**in<br>Reduction<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**P0M0Q0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>Reduction<br>**M0P0Q0**<br>**oActs**<br>**iActs (row-major) for next layer**<br>**iActs (Channel-last)**<br>**oActs (row-major)**|Compute<br>Compute<br>**Reorder**in<br>Reduction<br>**RIR critical path**<br>**Multi-cycle reorder be hidden behind reduction**<br>**iActs -> Compute (RIR)-> oActs w/ new layout**<br>**Reorder**in<br>Reduction<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**P0M0Q0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>Reduction<br>**M0P0Q0**<br>**oActs**<br>**iActs (row-major) for next layer**<br>**iActs (Channel-last)**<br>**oActs (row-major)**|Compute<br>Compute<br>**Reorder**in<br>Reduction<br>**RIR critical path**<br>**Multi-cycle reorder be hidden behind reduction**<br>**iActs -> Compute (RIR)-> oActs w/ new layout**<br>**Reorder**in<br>Reduction<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**P0M0Q0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>Reduction<br>**M0P0Q0**<br>**oActs**<br>**iActs (row-major) for next layer**<br>**iActs (Channel-last)**<br>**oActs (row-major)**|Compute<br>Compute<br>**Reorder**in<br>Reduction<br>**RIR critical path**<br>**Multi-cycle reorder be hidden behind reduction**<br>**iActs -> Compute (RIR)-> oActs w/ new layout**<br>**Reorder**in<br>Reduction<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**P0M0Q0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>Reduction<br>**M0P0Q0**<br>**oActs**<br>**iActs (row-major) for next layer**<br>**iActs (Channel-last)**<br>**oActs (row-major)**|Compute<br>Compute<br>**Reorder**in<br>Reduction<br>**RIR critical path**<br>**Multi-cycle reorder be hidden behind reduction**<br>**iActs -> Compute (RIR)-> oActs w/ new layout**<br>**Reorder**in<br>Reduction<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**P0M0Q0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>Reduction<br>**M0P0Q0**<br>**oActs**<br>**iActs (row-major) for next layer**<br>**iActs (Channel-last)**<br>**oActs (row-major)**|Compute<br>Compute<br>**Reorder**in<br>Reduction<br>**RIR critical path**<br>**Multi-cycle reorder be hidden behind reduction**<br>**iActs -> Compute (RIR)-> oActs w/ new layout**<br>**Reorder**in<br>Reduction<br>Compute<br>**H0W0C0:7**<br>**H0W1C0:7**<br>**P0M0Q0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>Reduction<br>**M0P0Q0**<br>**oActs**<br>**iActs (row-major) for next layer**<br>**iActs (Channel-last)**<br>**oActs (row-major)**|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|CPU<br>**iA**|||||||||||||**RIR critical path**<br>**H0W1C0:7**<br>**P0M1Q0:7**<br>**P0M2Q0:7**<br>**H0W2C0:7**<br>**H0W7C0:7**<br>**M0P0Q0**<br>**oActs**||||||
|Com|pute|Write Data<br>to DRAM|CPU<br>Reorder|Read Data<br>to OCS<br>C|ompute||Compute|Reduction|Reorde|r<br>Compute||**8**|Compute|**Reorder**i<br>Reductio|Compute<br>n<br>n<br>|**Reorder**in<br>Reduction|Compute||
||||||||||||||||||||



(a) Off-chip Data Reorder. (b) Reorder after Reduction (prior works). (c) Reorder in Reduction (RIR, this work). 

Fig. 6: Comparison of data reordering _implementations_ . This work proposes RIR that eliminates reorder latency and bank conflicts. We discuss on-chip reorder patterns, including transpose, line rotation, row-reorder and arbitrary reorder, in Fig. 5. 

to concurrently access up-to **three** rows within a single bank by storing a copy of a row in other banks. For example, to access three rows including data (0 _,_ 1 _, ··· ,_ 7 _,C, D, E, F_ ) from bank 0 in Fig. 5b, row ( _C, D, E, F_ ) is moved to bank such that it provides (0 _,_ 1 _, ··· ,_ 7) from bank 0 and ( _C, D, E, F_ ) from bank 1 to avoid bank conflicts. However, line rotation comes at the price of (1) extra bandwidth: it employs three ports for reading data that could be accessed with up-to two ports under concordant layout, (2) storage: it stores a copy of ( _C, D, E, F_ ). Such price could have been used for supporting more parallelism under arbitrary reordering to improve performance. 

_•_ **Transpose** (Fig. 5c) enables concurrently access to up-to two rows _or columns_ within a bank, hence augmenting concordant dataflow choice with higher P flexibility than fixed layout. But pure transpose falls short of supporting tiled layout transformation, such as changing layout from HWC W2C3 (Fig. 4, L1) to HWC W8 (Fig. 4, L2) 

> _•_ **Row Reorder** (Fig. 5d) does not support more concurrent access within a single bank, but enables arbitrary order within each row, hence supporting dataflows with higher O flexibility. Further, row reorder also supports im2col [11], which does not reduce bank conflicts because it still accesses the same number of rows from on-chip buffers. 

_•_ **Arbitrary Reorder** (Fig. 5e) enables arbitrary layout trans- 

TABLE III: SoTA on-chip reordering vs. _FEATHER_ . 

||Work|Datafow|On-chip Reorder Patterns|Implement|
|---|---|---|---|---|
||im2col [11]<br>Medusa [48]<br>MTIA [19]<br>TPUv4 [26]|N/A<br>N/A<br>TOP<br>TO|Row-Reorder (Fig. 5b)<br>Line Rotation (Fig. 5b)<br>Transpose (Fig. 5c)<br>Trans.+Row-Reorder (Fig. 5d)|RAR<br>RAR<br>RAR<br>RAR|
||**This Work**|TOPS|Arbitrary Reorder (Fig. 5e)|**RIR**|



formations, hence making all dataflows concordant with fullfledged O,P,S flexibility, as shown by red diamond in Fig. 5f. 

## _E. Data Reordering Implementations_ 

The layout reorder patterns described in Fig. 5 could have different implementations with _different critical-path latency_ . 

_1)_ _**Existing Implementations** :_ We classify existing reordering implementations into three categories. 

_a) No Reordering:_ If there is no reordering, either the accelerator needs to run a fixed dataflow or a subset of dataflows that are concordant to the fixed layout, or pay the cost of bank conflicts due to discordant accesses. This can lead to suboptimal performance (as shown by blue bar in Fig. 2). 

_b) Off-chip Reordering:_ SoTA that support dataflow switching (Tab. I) require iActs to move to off-chip DRAM, get reordered there by CPU, and then move back to the accelerator. This naturally incurs extra latency and energy costs (Fig. 6a). 

202 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

_c) On-chip Reorder After Reduction (_ _**RAR** ):_ Existing onchip reordering techniques essentially perform reordering after reduction. The post-reduction oActs are first written to the on-chip buffer, then read and sent to a separate unit to perform a layout transformation, and then fed back to compute unit as iActs of the next layer. This puts reordering in the critical path, as shown in Fig. 6b. Previous arts all fall into this bucket with _explicit reordering latency_ , as listed in Tab. III. For example, Medusa [48] proposes dedicated hardware between on-chip buffer to compute unit to implement line rotation (Fig. 5b); Meta’s MTIA [19] proposes a Memory Layout Unit (MLU) to implement transpose; Google’s TPUv4 [26] also supports row-reordering (Fig. 5d) to facilitate im2col. 

_2)_ _**Proposed Implementation** - On-Chip Reorder In Reduction (RIR):_ This work proposes to perform reordering on output during reduction phase of computation, such that oActs are written in the layout concordant with the dataflow of the next layer. We call this Reorder in Reduction ( **RIR** ). RIR _implicitly_ modifies the layout during the reduction process when _**generating oActs**_ instead of transforming iActs from one layout to another, as depicted inFig. 6c. This approach (i) removes reordering from critical path, (ii) reduces the total number of partial sums into fewer final sums, reducing buffer access and effectively minimizing potential bank conflicts. §IV provides more details. 

## _F. Inefficiency of SoTA Reconfigurable Dataflow Accelerators_ 

**Data Reordering Support.** Driven by the observation that on-chip dataflow plays a crucial role (§II-A), there has been a suite of past work on accelerators with hardware support for running diverse dataflows [32]. Their key observation is that different dataflows trade-off spatial and temporal reuse, and thereby flexible dataflow requires support for different operand stationarity within buffers and variable-sized spatial and temporal reductions through the interconnect. Unfortunately, these accelerators have **two** limitations as elaborated in §II-D and §II-E: (i) either they do not support any onchip reordering (Tab. I) or support limited transformations including transpose, line rotation or row-reorder (Tab. III). This work extends support to arbitrary reordering. (ii) prior on-chip reordering support can cause bank conflicts, increasing reordering time. This work removes reordering from critical path by doing it during the reduction phase of the computation. 

**Dataflow-Layout Co-Search.** There has also been a suite of dataflow/mapping search tools [23], [27], [41] that can recommend the optimal dataflow given a layer and hardware resources. _However, none of these tools explore on-chip data layouts as part of the search process._ 

**Contributions of this work.** This work addresses the aforementioned gaps via three key contributions: (i) a reconfigurable accelerator _FEATHER_ with a novel on-chip fabric called _BIRRD_ that provides support for _both_ dataflow flexibility and layout flexibility through arbitrary reorder, (ii) a new on-chip data reordering mechanism called RIR (implemented by _BIRRD_ ) whose key goal is to _generate_ data in the layout required by the next layer instead of explicitly requiring layout conversion 

**==> picture [190 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
Stationary Buffer (StaB) Streaming�Buffer (StrB)<br>NEST Functional Engine<br>zp Conv/GEMM ReLU/BN/MaxPooling<br>BIRRD<br>Instruction Buffer<br>Reorder in Reduction (RIR)<br>Output Buffer (OB) Controller<br>Quantize Module (QM) On-Chip Data Path<br>DRAM<br>New Data Layout<br>Scale<br>Zero Point (ZP)/ Scale Buffer<br>**----- End of picture text -----**<br>


Fig. 7: Overview of _FEATHER_ architecture. The compute pipeline (NEST _→_ BIRRD _→_ OB _→_ QM) reads iActs from StaB Ping (or Pong) and writes oActs to StaB Pong (or Ping) **with a new data layout** . 

(§IV), (iii) a tool called LayoutLoop for dataflow and layout co-exploration (§V). _FEATHER_ provides two specific benefits over prior work in data reordering: (i) supporting arbitrary reorder, and (ii) proposing RIR to hide reordering latency behind computation, and minimize bank conflicts. 

## III. _FEATHER_ OVERVIEW 

In this section, we provide an overview of _FEATHER_ architecture in Fig. 7 and its micro-architectures in Fig. 8. 

## _A. FEATHER’s Neural Engine – NEST_ 

Accelerators typically use tens of thousands of PEs organized in 1D arrays like MAERI [35] and SIGMA [42], or 2D arrays like Google TPUv4 [26] and Meta’s MTIA [19]. 2D PE arrays have better scalability but are limited in their dataflow options due to their rigid structure, leading to suboptimal utilization due to mismatch of layer shapes and array aspect ratios, as prior works have shown [35], [45]. 1D arrays with flexible distribution and reduction NoCs [32] have been shown to support arbitrary dataflows with full-range of TOPS (§II-A), specifically flexible parallelism and shape. However, they suffer from scalability issues due to their all-to-all NoCs. 

This work tries to marry the best of both styles. We find that the all-to-all reduction networks in prior works [35], [42] come with prohibitive resource overheads because of redundant reduction paths. This is to accommodate _arbitrary sized reductions_ . In contrast, _FEATHER_ ’s Neural Engine enables all rows of the 2D PE array to share the same reduction network in a time-multiplexing manner (thereby reducing its cost), without compromising flexibility, throughput, or utilization. 

Specifically, _FEATHER_ ’s **N** eural **E** ngine with **S** patial forwarding and **T** emporal reduction (NEST) works in two phases. One walk-through example for convolution is shown in Fig. 9. 

**Phase 1: Local Temporal Reduction.** _NEST_ involves local registers in each PE for temporal (local) reduction of partial sums. This is then followed by a phase of global reduction via the reduction network (described in §III-B). 

**Phase 2: Interleaved Spatial Forwarding and Reduction.** However, unlike prior works where all PEs participate simultaneously in the spatial reduction, the PE rows in _FEATHER_ perform spatial reduction one after another, temporally multiplexing on the reduction network. Further, while each PE 

203 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [214 x 335] intentionally omitted <==**

**----- Start of picture text -----**<br>
Stationary Buffer (StaB) Streaming Buffer (StrB)<br>D [entry] 1  Byte<br>Pong Bank Bank Bank 1  Bank Pong<br>Ping 0 1 AW-1 AW  Byte Ping<br>iActs Weights<br>register<br>PE PE PE 8 8<br>iActs AH<br>Ping/Pong<br>PE PE PE - 9 9 -<br>8 8<br>iActs Zp Weight Zp<br>32 18<br>PE PE PE Result + Weight<br>32<br>AW<br>temporal<br>BIRRD Config.(column-wise bus) reduced result�<br>Swap (x) Pass (=)<br>Add-Left (∓)Add-Right (±)<br>Reorder-Reduction Switch<br>�2 bit control<br>Write Addr OB:�reduce partial sum<br>� � �arriving over time<br>QM: Scale Buffer (1 Bank )<br>INT32 -> INT8 AW  Byte<br>oActs (iAct for next layer)<br>Read Ping => Write Pong�<br>StaB (vice versa)<br>+<br>) Bank<br>1<br>Byte<br>AW  (row-wise bus) NESTAH<br>Zero Point Buffer (<br>)<br>) Bits<br>Bank AW=8<br>1<br>-input BIRRD (E.g.  Reorder in Reduction (RIR)<br>Instruction Buffer (AW*(2*log(AW)-1) + logD� AW<br>OB0 OB1 OB2 OB3 OB4 OB5 OB6 OB7<br>QM0 QM1 QM2 QM3 QM4 QM5 QM6 QM7<br>Bank 0 Bank 1 Bank 2 Bank 3 Bank 4 Bank 5 Bank 6 Bank 7<br>+ +<br>+ +<br>**----- End of picture text -----**<br>


Fig. 8: Micro-architecture of _FEATHER_ ’s datapath for convolution/GEMM. For convolution, the NEST reads iActs from StaB and weights from StrB, streaming both in a top-to-bottom pipeline. PEs in a column time-multiplex a common output bus. _BIRRD_ conducts global spatial reduction and reorders results for targeted StaB banks during reduction, altering data layout in StaB. NEST facilitates inter-layer pipelining by reading iActs from StaB Ping (or Pong) and writes oActs (next-layer iActs) back to StaB Pong (or Ping). Note: _FEATHER_ is scalable architecture and we show 8-input _BIRRD_ as an example. 

row sends its locally reduced results to the reduction network, PEs in other rows continue computation and reduction locally. This is ensured via a pipelining mechanism that guarantees that each row performs _AH_ number of local reductions, before participating in the global reduction. 

**Flexible Dataflow:** _FEATHER_ retains the ability to support arbitrary dataflow parallelism strategies and shapes (§II-A). This is because Phase 2 can be configured to create arbitrarysized reduction groups (i.e., all outputs can be unique or any combinations can be reduced) enhancing mapping flexibility. 

_FEATHER_ supports inter-layer pipelining. We deploy distinct computation engines for ReLU, BatchNorm, and MaxPooling. For AvgPooling layers, they are transformed into convolution operations and executed within the _NEST_ . When there is a sole 

requirement for reorder and reduction, the PE Array can be bypassed, directing inputs from _NEST_ directly to the _BIRRD_ . To optimize storage utilization and reduce data movement costs, all computation engines utilize the same on-chip storage. 

## _B. FEATHER’s Reordering/Reduction Network – BIRRD_ 

The Butterfly Interconnect for Reduction and Reordering in Dataflows ( _BIRRD_ ) is a multi-stage network designed to reorganize data during the reduction phase. It receives computation results from the previous stage and directs them to new positions in the output buffer while concurrently reducing the data. This process aligns the data in the format needed for the subsequent dataflow, enabling _FEATHER_ to seamlessly co-switch (dataflow, layout) for each layer. 

||Algorithm 1: Inter-stage Connectivity for _AW_-input _BIRRD_|
|---|---|
|1:|output[_i_][_id_]/input[_i_][_id_] (_id ∈_[0_,AW_)) refers to _id_-th out-|
||put/input port of _BIRRD_ switches at the stage _i_.|
|2:|**FUNCTION** reverse<br>bits(data, bit<br>range)|
|3:|mask = (1_≪_bit<br>range) - 1|
|4:|reversed<br>bits = 0|
|5:|**for** i FROM 0 TO bit<br>range - 1|
|6:|**if** (data (1_≪_i))|
|7:|reversed<br>bits _|_= (1_≪_(bit<br>range - 1 - i))|
|8:|**return** (data & _∼_mask) _|_ reversed<br>bits|



|9:|**for** _i_ in [0, 2_×log_2(_AW_)) // i is|stage|id|
|---|---|---|---|
|10:|**for** _j_ in [0, _AW_) // j is port<br>id|||
|11:|output[_i_][_j_]-input[_i_+1][reverse|bits(_j_,|min(_log_2(_AW_)_,_2+|
||_i,_2_×log_2(_AW_)_−i_))] (- indicates output connects to input)|||



_1) BIRRD Topology:_ The _BIRRD_ topology is interfaced with _NEST_ engine one side and output buffer on the other side, and is composed of two butterfly networks back-to-back with _log_ ( _AW_ )-bit bit reverse connections [16]. This topology grants symmetry with respect to the middle, enabling the construction of each half separately. Each input of _BIRRD_ receives data from one column-wise bus of the _NEST_ while each output of _BIRRD_ forwards the result to one output buffer and eventually back to one bank of stationary buffer (StaB, refer to Fig. 7) . For _NEST_ with _AW_ columns in total ( _AW_ must be a power of 2), the _BIRRD_ encompasses 2 _× log_ ( _AW_ ) stages[1] with _AW /_ 2 switches located at every stage. The inter-stage connections of _BIRRD_ are outlined in Alg. 1. 

The topology of _BIRRD_ has been proven to be strictly nonblocking for unicast (any single data point among concurrent inputs sent to a single output) [5] and rearrangeably nonblocking for multicasting (at least one data point among all concurrent inputs sent to multiple output ports) [8], [16], [36]. We found no multicasting case that it cannot accommodate. 

_2) BIRRD Reorder-Reduction Switch:_ The _BIRRD_ is built on 2-input _×_ 2-output switch (which we call _Egg_ ) with adder as shown in Fig. 8. Each _Egg_ is governed by a 2-bit 

> 14-input _BIRRD_ is a special case with only 2 _× log_ ( _AW_ ) _−_ 1 = 3 stages, i.e. the last stages of two half butterfly networks get merged into a single stage. 

204 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [485 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
Stride=1 Warm Up: Local temporal reduction (Phase 1) Steady: All PEs are working in the state of either Phase 1 or 2<br>iActs Cycle 0 Cycle 1 Cycle 2 Cycle 3 Cycle 4 Cycle 5 Cycle 6<br>00  [1] w0w1 w4w5 w0w1 w4w5 w0w1 w0w1 w0w1 w0w1 w0w1 w0w1<br>w2 w6 w2 w6 w2 w2 w2 w2 w2 w2<br>w3 w7 w3 w7 w3 w3 w3 w3 w3 w3<br>M0 M0 M1 M1 M0 M0 M1 M1 M0 M0 M1 M1 M0 M0 M1 M1 M0 M0 M1 M1 M0 M0 M1 M1 M0 M0 M1 M1<br>3<br>0 3 w0w1 w4w5 w0w1 w4w5 w0w1 w0w1 w0w1 w0w1 w0w1 w0w1<br>0 w0w4w6w1w5w7 M2 w2w3 M2 w6w7 M3 w2w3 M3 w6w7 M2 w2w3 M2 M3 M3 M2 w2w3 M2 M3 M3 w2w3 M2 M2 M3 M3 M2 w2w3 M2 M3 M3 M2 w2w3 M2 M3 M3 M2 w2w3 M2 M3 M3<br>w2 w3<br>... w0 w4 w0 w4 w0 w0 w0 w0 w0 w0<br>w4 w5 w1 w5 w1 w5 w1 w1 w1 w1 w1 w1<br>15 w0w2w6w1w3w7 M4 w2w3 M4 w6w7 M5 w2w3 M5 w6w7 M4 w2w3 M4 M5 M5 M4 w2w3 M4 M5 M5 w2w3 M4 M4 M5 M5 M4 w2w3 M4 M5 M5 M4 w2w3 M4 M5 M5 M4 w2w3 M4 M5 M5<br>Weights loaded w0 w4 w0 w4 w0 w0 w0 w0 w0 w0<br>w1 w5 w1 w5 w1 w1 w1 w1 w1 w1<br>Each PE holds w2 w6 w2 w6 w2 w2 w2 w2 w2 w2<br>w3 w7 w3 w7 w3 w3 w3 w3 w3 w3<br>one channel of M6 M6 M7 M7 M6 M6 M7 M7 M6 M6 M7 M7 M6 M6 M7 M7 M6 M6 M7 M7 M6 M6 M7 M7 M6 M6 M7 M7<br>one kernel AW=4 4-input BIRRD  [4:2�][Spatial] BIRRD BIRRD BIRRD<br>(4 weights)� Phase  1 : Local  Temporal  Reduction Phase  2 :�Interleaved Global� Spatial  Reduction Reduction<br>Phase 1 Phase 2 Phase 1<br>Phase 1 Phase 1<br>Phase 1 Phase 2 Phase 1<br>AH=4<br>Phase 1 Phase 2<br>Phase 1<br>Phase 1 Phase 2<br>**----- End of picture text -----**<br>


Fig. 9: Illustration of the _FEATHER_ with _NEST_ and _BIRRD_ employing a convolutional operation with a 2 _×_ 2 weights featuring 2 input channels ( _C_ = 2) and generating 16 output channels ( _M_ = 16) across a 4 _×_ 4 iAct with 2 input channels. The depicted dataflow utilizes a weight-stationary approach, where each PE has a local register file containing a channel of weights (2 _×_ 2). The dataflow is parallelized for two input channel and two output channel across four PE columns, and for four kernels across four PE rows. In each row, four PEs generate 4 partial sums, contributing to 2 final sums, which thus necessitates a 4:2 spatial reduction in the _BIRRD_ to produce two outputs. We assume the weights are already preloaded into _NEST_ before the first cycle in this illustration. The iActs are streamed from the top, undergo multiplication with corresponding weight values (e.g., _w_ 0 in the top-left PE at cycle-0), and are locally accumulated for the next set of inputs (e.g., until cycle-3 in the top-left PE). Following this initial phase of local temporal reduction, the top row transmits the locally reduced result to the _BIRRD_ for the second phase of spatial reduction. In the steady state, _BIRRD_ reduces data from one _NEST_ row per cycle (cycles 4-6). _In steady state, all PEs are working and there is no output bus conflict for PEs of the same column. This is because, during phase-2 of spatial reduction in one PE, remaining PEs of the same column perform local reduction._ In general, _AW × AH NEST_ takes _AH_[2] cycles to load weights, and ping-pong local registers are instantiated to hide such latency behind computation. _BIRRD_ could reduce results from PEs at different rows as long as only one PE per column uses the output bus. **Takeaway:** NEST utilizes local temporal and global spatial reduction to (i) ensure all PEs of the same column share the same output bus without competition while achieving full utilization, and (ii) hide weight loading latency in steady phase. 

configuration word, allowing for control of four reorder-inreduction functionalities (shown in Fig. 8) as follows. 

_•_ **Pass (** = **) / Swap (** _×_ **)** : directly pass left (right) input data to left (right) output port, or swap them. 

_•_ **Add-Left (** _∓_ **) / Add-Right (** _±_ **)** : Accumulates data from input ports and transmits results to the left/right output port, with the secondary output inheriting the input from the same direction. 

Extra broadcast functions could be added in the Eggs to duplicate accumulated results in multiple banks of StaB. 

## _3) BIRRD Capability and Routing: BIRRD_ supports 

_• Arbitrary Reduction_ : We define “reduction group” as a group of inputs that get reduced into one output. _AW_ -input _BIRRD_ supports arbitrary number of reduction groups (up to _AW_ ). 

> _•Arbitrary Reordering_ : The rearrangeably multicasting capability enables _BIRRD_ to route results from many reduction groups to many arbitrary output ports concurrently. 

The examples of _BIRRD_ supporting various reordering and reduction patterns are shown in Fig. 10. 

From a routing perspective, _reduction can be viewed as a reverse multicasting operation_ , where multiple input data points target the same output port and are reduced upon encountering each other at _BIRRD_ Eggs. Thus, we adopt the multicasting routing algorithm [4] to establish paths and configurations for _BIRRD_ Eggs, enabling reordering during reduction. If a 

certain input-output connection cannot be established by the algorithm [4], we will brute force all possible configurations. Fig. 10 showcases how _BIRRD_ supports arbitrary dataflows and layout switching requirements. 

_4) Microarchitectural Benefits of BIRRD:_ Generally, distribution networks like Benes in SIGMA [42] or fat-tree in MAERI [35] necessitate unicast or multicast capabilities to direct data from relevant on-chip buffer banks to specific processing elements (PEs). This necessity becomes obsolete with _BIRRD_ (via _RIR_ ), as it harmonizes data layouts to coincide with dataflows. This enables _FEATHER_ to utilize a straightforward point-to-point connection to the input ports of _NEST_ without sacrificing flexibility. Consequently, _BIRRD_ simplifies the requirements for distribution networks in accelerators, thereby minimizing control, resource, and latency expenses. 

## _C. On-chip Storage and Post-processing_ 

On-chip storage is physically divided into separate buffers with different organizations for concordance with dataflows. 

_1) Stationary (StaB) and Streaming Buffer (StrB):_ The typical paradigm of processing convolution or GEMM will keep one type of data stationary, termed a stationary tensor, and stream the other type of data, termed a streaming tensor. _FEATHER_ fetches and processes the streaming tensor in the tile granularity. Both StaB and StrB implement a ping-pong 

205 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [484 x 322] intentionally omitted <==**

**----- Start of picture text -----**<br>
Workload A Workload A Workload B Workload C n0n1n2 Workload D n0<br>iActs Weights Change oAct Layout<br>k7k6 K=8 n0n1n2n3 k7k6 K=8 n0n1n2n3<br>m0m1 m0m1 m0k: 5 4 3 2 1 0 n: 0 1 2 3 4 5 6 7 k11k10 K=12 k15k14 K=16<br>m1 m0 m0<br>m2 m1 m1<br>M=8 M=8 m3 m2 m2<br>m4 m3 m3<br>m5 m4<br>k k Time k k Time k k k k k<br>m3m2m1 m7m6m5 m3m2m1 m7m6m5 m3m2m1 m7m6m5 m3m2m1 m7m6m5 m2m1 m2m1 m5m4 m5m4 n2n1 n2n1 n2n1 m4m4 k k k k<br>m0 m4 m0 m4 m0 m4 m0 m4 m0 m0 m3 m3 n0 n0 n0 m4 n0 n0 n0 n0<br>n0 n0 n0 n0 n0 n0 n0 n0 n0 n4 n0 n4 m0 m0 m0 n0 n0 n0 m0 m0 m0 m0<br>n1 n1 n1 n1 n1 n1 n1 n1 n1 n5 n1 n5 m1 m1 m1 n1 n1 n1 m1 m1 m1 m1<br>n2 n2 n2 n2 n2 n2 n2 n2 n2 n6 n2 n6 m2 m2 m2 n2 n2 n2 m2 m2 m2 m2<br>Ping-Pong<br>n3 n3 n3 n3 n3 n3 n3 n3 n3 n7 n3 n7 m3 m3 m3 Switch m3 m3 m3 m3<br>= = = = x x =<br>± ± x x ± x ±<br>= = = = = = x = =<br>Layout<br>Requirement:<br>2:1 2:1 2:1 2:1 1:1 3:1 4:1<br>Dataflow Global Spatial Global Spatial No Global Pass Global Spatial Global Spatial<br>Requirement: Reduction Reduction Spatial Reduction Through Reduction Reduction<br>Output Stationary 18.75%<br>m4 m5 50% m4<br>Time m4m0 m5m1 m6m2 m7m3 Time m4m0 m5m1 m6m2 m7m3 Time m0 m1 m2 m3 Time m0 m1 m2 m3 75% Time m0 m1 m2 m3<br>n0 n0 n0 n0 n0 2 5%<br>n1 100% n1 100%� n1 n1 7 5%<br>n2 Utilization n2 Utilization n2 50%� n2<br>n3 n3 n3<br>+ + + + +<br>4� � 7 0� � 3 4� � 7 0� � 3 0� � 3 0� � 3 4� � 7 8� � 11 8� � 11<br>Streaming Streaming Streaming Streaming 4� � �70� � 3 Streaming 0� � 3 4� � 7 8� � 11 12� � 15<br>&oActs Stationary<br>&oActs Stationary<br>�Weights & oActs Stationary �Weights & oActs Stationary �Weights & oActs Stationary �iActs �Weights �Weights & oActs Stationary<br>FEATHER<br>Bank 0 Bank 3 Bank 0 Bank 2 Bank 0 Bank 1 Bank 2 Bank 3 Bank 0 Bank 2 Bank 2<br>%<br>5<br>8.7<br>1<br>Systolic Array<br>Time<br>**----- End of picture text -----**<br>


Fig. 10: Comparison between per-layer flexible dataflows in _FEATHER_ and fixed-dataflow in the systolic array under GEMM. _FEATHER_ dynamically alters layout by redirecting oActs to various banks with distinct writing addresses, exemplified by rerouting a blue result from bank 0 (Workload A) to bank 2 (Workload A Change oAct Layout). _FEATHER_ consistently outperforms SA in irregular-sized GEMM (Workload B, C, D), achieving near full utilization. Enhanced utilization arises from (1) enabling cross-column spatial reduction using _BIRRD_ in _FEATHER_ , e.g. _FEATHER_ maps K dimension across the entire 2D array instead of a single PE in SA under workload D. (2) Eliminating SA’s horizontal rigid reuse links, thereby enabling independent mappings across columns, e.g. (Workload C) adopting iAct stationary in first three columns and weights stationary in the last column. _BIRRD_ could perform pure reordering to change the layout when no spatial reduction is required (e.g. _BIRRD_ reordering all incoming results to target banks directly under workload B). **Takeaway:** BIRRD’s flexible reduction enhances compute utilization across diverse skewed shapes, expanding the range of dataflows that NEST can efficiently support. 

buffer to enable (1) the latency hiding of fetching the next tile from off-chip DRAM, and (2) on-chip inter-layer pipelining. 

As for convolution/GEMM (Fig. 8), iActs are kept stationary within StaB Ping (or Pong), and the resulting oActs are written back into StaB Pong (or Ping) with a new layout. Meanwhile, weights are streamed via StrB (Ping/Pong). StaB requires a multi-bank organization ( _AW_ banks), with each bank storing a single data piece, to accommodate the varied write addresses in different banks necessitated by layout changes in _FEATHER_ . Conversely, StrB adopts a simplified single-bank structure with an _AW_ -data bandwidth to conserve area, because weights do not need layout reordering. 

_2) Instruction Buffer (IB):_ The configurations for _BIRRD_ are generated offline and get fetched into IB to configure the reduction networks at run-time. 

_3) Output Buffer (OB):_ enables in-situ temporal reduction of partial sums when the reduction size of workloads exceeds the overall reduction capacity of both _NEST_ and _BIRRD_ . OB has _AW_ banks, and each equipped with a 32-bit adder. 

_4) ZP/Scale Buffer and Quantization Module (QM):_ employing quantization schemes from PyTorch FBGEMM [31] and QNNPACK [17], with 8-bit zero points and 32-bit scales (housed in ZP/Scale Buffer). The quantization module rescaled down 32-bit oActs and then quantized to 8-bit oActs. 

## IV. _FEATHER_ IN ACTION 

In this section, we first showcase one example (Fig. 11) of how _FEATHER_ leverages _RIR_ to resolve bank conflicts mentioned in Fig. 6b when co-switching dataflow-layout. Then we deep dive into how _FEATHER_ enables general layout transformations without bank conflicts through two insights. 

206 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [207 x 247] intentionally omitted <==**

**----- Start of picture text -----**<br>
2048 iActs Dataflow Channel-last (HWC _ C4)<br>0 Line# StaB-Ping (iActs)<br>0 0 H0W0C0Bank0 H0W0C1Bank1 H0W0C2Bank2 H0W0C3Bank3<br>1 H0W1C0 H0W1C1 H0W1C2 H0W1C3<br>2 H0W2C0 H0W2C1 H0W2C2 H0W2C3<br>7<br>r0 H1W0C0 H1W0C1 H1W0C2 H1W0C3<br>0 7 r0+1 H1W1C0 H1W1C1 H1W1C2 H1W1C3<br>r0+2 H1W2C0 H1W2C1 H1W2C2 H1W2C3<br>r0+3 H1W3C0 H1W3C1 H1W3C2 H1W3C3<br>Weight C Parallel<br>Cycle# Required byMaiActspping (Line#Read TraceStaB , Bank#Ping ) M0C0 M0C1 M0C2 M0C3<br>0 H0W0 C0:3 (0, 0:3) M1C0 M1C1 M1C2 M1C3<br>1 H0W1 C0:3 (1, 0:3)<br>23 H1W0H1W1 C0:3C0:3 (r0+1(r0, 0:3, 0:3) ) M2C0 M2C1 M2C2 M2C3<br>45 H0W1H0W2 C0:3C0:3 ((12,, 0:3 0:3)) M3C0 M3C1 M3C2 M3C3<br>6 H1W1 C0:3 (r0+1, 0:3)<br>... 7 H1W2 ... C0:3 (r0+2 ... , 0:3) ∓ ∓ ∓ ∓<br>�BIRRD BIRRD<br>Cycle# GeneratedoActs (Line#Write TraceStaB�, Bank #Pong� ) cycle 6~9Config. ∓ = == ∓ x == cycle 10~13Config.<br>0~2 Temporal Reduction in NEST<br>3~675 M0P0Q0M1P0Q0Spatial RIR in BIRRD((r10,, 0 0)) Line#0 M0P0Q0 M0P0Q1 M0P0Q2 M0P0Q3<br>8 M2P0Q0 (r2, 0)<br>9 M3P0Q0 (r3, 0) r1 M1P0Q0 M1P0Q1 M1P0Q2 M1P0Q3<br>10 M0P0Q1 (0, 1)<br>11 M1P0Q1 (r1, 1) r2 M2P0Q0 M2P0Q1 M2P0Q2 M2P0Q3<br>12 M2P0Q1 (r2, 1)<br>Row-major MPQ_Q4 (oActs) 13 ... M3P0Q1 ... (r3 ... , 1) r3 M3P0Q0Bank0 M3P0Q1Bank1 M3P0Q2Bank2 M3P0Q3Bank3<br>(CHW _ W4, iActs for next layer) StaB-Pong (oActs)<br>M Parallel<br>**----- End of picture text -----**<br>


Fig. 11: Example of _FEATHER_ switching from channel-last layout ( _HWC C_ 4) to a row-major format ( _MPQ Q_ 4( _CHW W_ 4)) during reduction without incurring bank conflicts. This is because multiple iActs are reduced into fewer oActs, thereby reducing accesses within each bank. In this example, _NEST_ leverages parallelism along the kernel _M_ and channel _C_ dimensions, reading and vertically streaming four iActs of four input channels from top to bottom. Specifically, at cycle 0, _NEST_ fetches _H_ 0 _W_ 0 _C_ 0 : 3 from (line 0, banks 0 : 3), as recorded in the StaB Ping read trace. Subsequent cycles involve a two-stage reduction: temporal reduction within the PE for cycles 0 to 2, and spatial reduction within _BIRRD_ for cycles 3 to 5, culminating in a single oAct _M_ 0 _P_ 0 _Q_ 0. This oAct is reordered to bank 0 during reduction and written to line 0 in the StaB Pong during cycle 6. _FEATHER_ ’s pipelined processing of following iActs is further exemplified in the read/write trace. _M_ 0 : 3 _P_ 0 _Q_ 0 target bank 0 and use connectivity of _BIRRD_ as shown in the left while _M_ 0 : 3 _P_ 0 _Q_ 1 use the right. For brevity, the notation of _R_ 0 : 1 _S_ 0 : 1 is omitted, which indicates that each PE in _NEST_ holds four weights of one channel. **Takeaway:** FEATHER reorders oActs into next layer’s desirable layout during reduction, enabling dataflow/layout co-switching. 

## _A. RIR for Bank Conflicts Mitigation and Layout Transform_ 

In the example shown in Fig. 11, the layout conversion from iActs to oActs is realized via _RIR_ , thereby avoiding the explicit latency in reorder after reduction. This efficiency stems from the key insight that _RIR reorders post-reduction oActs into a new layout, rather than directly transforming iActs from one layout to another_ . 

Specifically, in the reduction phase, numerous iActs naturally get accumulated into fewer oActs and consequently target fewer banks. For example, four iActs get accumulated to one 

oAct that targets a single line in Fig. 11. Conversely, if we directly transform the layout of iActs from channel-last to row-major, four iActs (H0W0C0:3) would target four different lines within the same bank under row-major layout, leading to bank 

## _B. (Dataflow, Layout) Flexibility for Bank Conflicts Eradication_ 

While the strategy of ‘reordering post-reduction oActs’ aids in reducing bank conflicts, conflicts may still arise when the number of partial sums to write into memory exceeds the number of writing ports of the memory. This scenario is particularly common in scaled-up 128 _×_ 128 compute array (Google TPUv4 [26]), as it generates more oActs concurrently. 

_FEATHER_ fully eliminates conflicts with the second key insight that _FEATHER picks the dataflow with the number of oActs (partial sums) matching with the number of memory write ports._ In essence, _FEATHER_ employs dataflows free from bank conflicts, and the flexible reduction of the _BIRRD_ consistently allows _FEATHER_ to identify such dataflows with high performance and efficiency. 

In summary, _RIR_ together with flexible dataflows selection enable _FEATHER_ to switch among arbitrary layouts without incurring bank conflicts. 

## V. LAYOUTLOOP 

_FEATHER_ enables (dataflow, layout) co-switching at the layer granularity to achieve optimal latency and energy efficiency. However, deciding which (dataflow, layout) to use for _FEATHER_ is not trivial because both dataflow and layout have huge space, e.g. 10[36] _×_ 10[8] for a single convolution layer (ResNet-50 layer 1) [27][2] , necessitating systematic exploration. For this aim, we enhance Timeloop [41], a state-of-the-art dataflow search framework with (1) physical storage modeling and (2) systematic layout assessment capabilities, and term it as Layoutloop to distinguish it from native Timeloop. We employ Layoutloop to explore dataflows under various layouts for _FEATHER_ , selecting the dataflow-layout pair that minimizes energy delay product for each layer. 

## _A. Physical Storage Modeling_ 

Layoutloop models physical storage as ( _num line × line size_ ) 2D array with “ _conflict depth_ ” specifying number of lines in each bank with the following reasoning. 

**Bank Organizations:** Current storage uses diverse organizations, including 2D/3D with various groupings. Managing these disparate physical organizations can be complex. However, as storage is usually accessed line by line (or block), we can abstract different organizations into a logical _num line×line size_ 2D array. This abstraction allows layout modeling to handle these 2D abstract arrays directly, retaining generality without dealing with specific physical organizations. 

**Bank Port Constraints:** Storage comes with an inherent limitation of the total number of ports in each bank. Concurrent 

> 2A flattening of 4 iActs dimensions ( _N_ = 1 _,C_ = 3 _, H_ = 224 _,W_ = 224) into two nested loop (Fig. 3) introduces 8! = 40320 order possibilities and (1 _,_ 2 _,_ 16 _,_ 16) factorization possibility. The product leads to 10[8] layout choices. 

207 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

read/write operations exceeding available read/write ports lead to bank conflicts. Thus, _conflict depth_ is utilized to denote the total number of lines within a single bank. 

## _B. Bank Conflicts Assessment_ 

Layoutloop models slowdown by judging whether bank conflicts occur when analyzing data access to the on-chip buffer with a specific layout. A max( _NP/NL,_ 1) slowdown is introduced if _NL_ lines are accessed from a bank with _NP_ ports. Finally, we also modify Timeloop’s mapper to consider data layout during dataflow search. 

## VI. EVALUATION 

**==> picture [33 x 42] intentionally omitted <==**

Fig. 12: FEATHER vs. SoTAs on real devices. We run each layer for 100 times to obtain average layer latency, and then normalize throughput by number of PE and clock frequency. 

## _A. Methodology_ 

We implement _FEATHER_ in Verilog and Xilinx HLS. Verilog-based implementation delivers precise microarchitecture design while HLS-based implementation enables the native usage of Xilinx IPs for buffer, control, and peripherals for better end-to-end performance on Xilinx FPGAs. We evaluate its resources on TSMC 28 nm high performance technology node using the Verilog-based implementation. We compare its end-to-end wall-clock latency against SoTAs with open-sourced end-to-end implementations on real FPGA devices. We also model _FEATHER_ in Layoutloop (§V), including energy overheads, to compare it against SoTA accelerators that do not have open-sourced end-to-end deployable codes. Tab. IV summarizes our evaluation setup. 

_1) Baselines and Workloads:_ 

**Baselines for real-device** evaluations. We compare _FEATHER_ against Xilinx DPU [2] , Gemmini [21], and Edge TPU [46], as they can be deployed in an end-to-end fashion. _FEATHER_ and Xilinx DPU [2] are deployed on the same Xilinx ZCU 104 FPGA board. While Gemmini is deployed on AWS-F1 FPGA server [21] using FireSim to emulate its per-layer processing latency. Edge TPU [46] runs on a USB accelerator [1] attached to a Raspberry Pi 4B. As for all four designs, we normalize throughput by the number of PEs (i.e., MAC units) and clock frequency[3] for a fair comparison. 

**Baselines for Layoutloop.** _FEATHER_ is further compared against NVDLA [39], Eyeriss [14] and SIGMA [42] in Layoutloop. Detailed modifications/specs are listed in Tab. IV 

**Workload.** BERT (representative of cloud workloads); ResNet-50 and MobiletNet-V3 (Mob-V3) as edge workloads. 

## _2) FEATHER Dataflow/Layout Setup:_ 

_•_ **Search Space.** Dataflow design space is constructed by arbitrary nested loops as shown in Fig. 1. We use layout patterns used by prior accelerators [43] as layout space[4] . 

_•_ **Searching Algorithm.** We exhaustively search layout space for global optimal. To find optimized dataflows, we use Timeloop’s internal hybrid search algorithm (exhaustive + 

> 3Both GEMMINI and FEATHER could run at 1 GHz under TSMC 28 nm ASIC flow. However, the parallel simulation synthesis toolchain of firesim limits GEMMINI’s clock frequency to 50 MHz on AWS’s f1.2xlarge FPGA. 

> 4Conv: HWC C32, HWC W32, HWC H32, HWC C4W8, HWC C4H8, HWC W4H8, HWC C4W4H2; GEMM: we note input/weights/output as _M × K_ / _N × K_ / _M × N_ with inputs layout as MK K32, MK M32, MK M4K8. 

search-space pruning). Recent works [29] show that its results are comparable to sophisticated search methods [22], [23], [27] but is slower in wall clock time. We ran the search with multiple threads constrained on search size and victory conditions. 

> _•_ **Performance Metric.** We use Energy-Delay-Product (EDP) as the performance metric for a dataflow/layout pair. 

_•_ **Overall Search Flow.** Dataflow/layout cosearch is conducted for each layer independently. The optimal dataflow-layout pair with the best EDP is chosen for each layer of ResNet-50 and Mob-V3 in the Layoutloop evaluation. For end-to-end FPGA deployment of ResNet-50, we simplify engineering efforts by selecting the two layouts with the best latency and energy efficiency on DepthWise Conv. and typical Conv., and enable _FEATHER_ to switch between them per layer. 

## _B. End-to-end Real-device Latency Evaluation_ 

_1) FEATHER vs. Gemmini: FEATHER_ achieves a 3 _._ 91 _×_ geomean normalized throughput improvement than Gemmini as shown in Fig. 12, as Gemmini adopts a fixed dataflow (weights stationary with degree of parallelism being 16 in both C and M), leading to under-utilization when C of workload is not divisible by 16. The flexibility of _FEATHER_ in the parallelism of M,C,H,W delivers its performance improvement. 

_2) FEATHER vs. Xilinx DPU:_ The 2 _._ 65 _×_ more throughput of _FEATHER_ over Xilinx DPU stems from the low steadystate utilization of Xilinx DPU under convolution 3 _×_ 3 (75% utilization), 7 _×_ 7 (21.8 _∼_ 87.5% utilization), and _FEATHER_ pushes both utilization to 100% and 90.4% in the steady state. This is because Xilinx’s DPU with 1152 PEs only supports a single dataflow with parallelism (12 _,_ 12 _,_ 8) in ( _M,C, H/W_ ). In deep layers with a large number of input channels (C) and kernels (M), both _FEATHER_ and Xilinx DPU achieve a steady utilization of 100%. However, Xilinx DPU outperforms _FEATHER_ for these layers as our controller is not as optimized as DPU’s (an engineering optimization part of our future work). 

_3) FEATHER vs. Edge TPU:_ 4 _._ 91 _×_ speedup comes from flexibility of FEATHER in dataflow and layout. 

## _C. Layoutloop-based Latency/Energy Evaluation_ 

Latency and energy efficiency are affected by (1) compute utilization determined by dataflows, and (2) effective memory bandwidth considering bank conflicts. 

208 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

TABLE IV: Evaluation setup for SoTAs and _FEATHER_ ( _→_ indicates the modifications from the original design). 

||(Layout,|Run-time <br>Datafow|Flexibility in<br>2 ,<br>Reorder)|Flexibility in<br>2 ,<br>Reorder)|(#PE,|on-chip BW<br>bit/cycle,|DataType)|Area (_μm_2)/Clock Frequency|Evaluation Method<br>Real-device/Layoutloop|
|---|---|---|---|---|---|---|---|---|---|
|Edge TPU|(fx ,|T,||none)|(1024,|unknown,|int8)|500 MHz (ASIC)|Coral USB accelerator [1]|
|Xilinx DPU|(fx ,|T,||none)|(1152,|unknown,|int8)|100 MHz (FPGA)|end-to-end on ZCU104|
|Gemmini|(fx,|T,||none)|(1024,|512,|int8)|50 MHz<br>1 (FPGA)|FireSim on AWS EC2 F1|
|**_FEATHER_**|(fexible,|TOPS,|Reorder In Reduction)||(1296,|720,|int8)|100 MHz (FPGA)|end-to-end on ZCU104|
|NVDLA-like|(fx,|T,||none)|(16_×_16,|25_→_256,|int8)|808K (TSMC-28, 1 GHz)<br>5|Layoutloop|
|Eyeriss-like|(fx,|TS,||none)|(14_×_12_→_16_×_16,|192_→_256,|int16_→_int8)|1394K (TSMC-65, 200 MHz)|Layoutloop|
|SIGMA-like|(fx<br>3 ,|TOPS,||none)||||||
|SIGMA-like|(fexible,|TOPS,|off-chip reordering<br>4 )||(65536_→_256,|256,|bf16_→_int8)|990K (TSMC-28, 500 MHz)|Layoutloop|
|Medusa-like|(fexible,|TOPS,||on-chip line rotation)||||||
|TPU-like|(fexible,|TO,|transpose/row reorder)||(8_×_“128_×_128”,|256,|int8)|600K (7nm, 1050 MHz)|Layoutloop|
|MTIA-like|(fexible,|TOP,||transpose)|(64_×_“32_×_32”,|1024,|int8)|373K (TSMC-7, 800 MHz)|Layoutloop|
|**_FEATHER_**|(fexible,|TOPS,||_RIR_)|(16_×_16,|8000,|int8)|338K (TSMC-28, 500 MHz)|Layoutloop|



1 : Latency scaled to 100 MHz. We standardized the frequency at 100 MHz just for a fair comparison purpose, which is not indicative of the maximum clock frequency achievable on ASIC implementations.; 2 : Terminology defined in §II-A; 3 : HWC C4W8 or HWC C32; 4 : off-chip bandwidth=128 GB/s. 5 : compute area only. 

||Layout Setup|
|---|---|
|Label|compile<br>runtime|
|NVDLA-like|fxed<br>fxed|
|Eyeriss-like|fxed<br>fxed|
|SIGMA-like|fex<br>fxed|
|Medusa-like|line rotation (Fig. 5b)|
|MTIA-like|transpose (Fig. 5c)|
|TPU-like|transpose + row<br>reorder (Fig. 5d)|
|FEATHER|arbitrary reorder (Fig. 5e)|



**==> picture [339 x 81] intentionally omitted <==**

Fig. 13: _FEATHER_ vs. SoTA using Layoutloop (Percentage inside each blue bar indicates average steady-state PE utilization. Red bar indicates bank conflict slowdown, while yellow bar indicates off-chip reordering costs. Lower is better. (The red text in the x-axis of the right chart mentions the fixed layout or the layout reordering mechanism for each design.) 

With per-layer dataflow-layout switching, _FEATHER_ achieves the peak steady-state utilization of 100%, 100%, and 98.3% with zero bank conflict slowdown under BERT, ResNet50, and Mob-V3, separately. This indicates that FEATHER consistently provides desirable layout for all three workloads. 

_1) FEATHER vs. NVDLA: FEATHER_ achieves 2 _×_ /2 _×_ /2.89 _×_ speedup and 6.43 _×_ /1.3 _×_ /1.35 _×_ higher efficiency over NVDLA under BERT/ResNet-50/Mob-V3. Leveraging fixed weights/output stationary dataflows under a fixed HWC C32 layout, NVDLA will not encounter any bank conflicts, which delivers good energy efficiency. However, NVDLA only allows flexible tiling sizes _T_ (definition in §II-A), and such dataflows suffer from low utilization (50% and 39%), explaining higher normalized latency in Fig. 13. 

_2) FEATHER vs. Eyeriss:_ The 1.43 _×_ /1.27 _×_ /1.87 _×_ speedup and 5.98 _×_ /3.09 _×_ /1.92 _×_ higher efficiency of _FEATHER_ over Eyeriss comes from both bank conflicts elimination and higher-performance dataflows. Specifically, Eyeriss adopts rowstationary dataflow and enables flexible tiling and shape but such sub-flexibility comes with the price of bank conflicts. Compared against Eyeriss-like, _FEATHER_ incurs 6% more area by introducing _BIRRD_ and controller, as shown in Fig. 14b. 

_3) FEATHER vs. SIGMA:_ SIGMA supports flexibility in all 4 dimensions of dataflows (§II-A), but does not have reordering capability (§II-E2). We thus evaluate static layout, off-chip and three on-chip reordering scenarios, as illustrated in Fig. 6. 

_•_ **Fixed Layout + No Reordering:** SIGMA adopts a fixed layout at runtime and keeps iActs and oActs always on-chip. And we leverage Layoutloop to search dataflows with minimal bank conflicts. Results under two layouts (HWC C32 and 

HWC C4W8) out of seven layouts delivering relatively better latency and energy efficiency are depicted in Fig. 13. Proposed Layoutloop could fully utilize SIGMA’s flexibility in TOPS to identify dataflows with fewer bank conflicts, resulting in a small speedup of _FEATHER_ over SIGMA (the same for BERT, 1.01 _×_ /1.03 _×_ for ResNet-50, and 1.17 _×_ /1.07 _×_ for Mob-V3). Although Layoutloop could identify dataflows for SIGMA with good latency, such dataflows always need to read more lines than dataflows adopted by _FEATHER_ , resulting in the high energy efficiency of _FEATHER_ (1.44 _×_ for BERT, 1.09 _×_ /1.46 _×_ for ResNet-50, and 1.29 _×_ /1.54 _×_ for Mob-V3). 

_•_ **Concordant Layout + Off-chip Reordering:** SIGMA pays the latency and energy costs to send oActs back off-chip and changes layout per layer. Such reordering cost is explicitly shown in Fig. 13. In the case of high compute-intensive ResNet50, the latency of off-chip reordering could be almost hidden behind computation latency when adopting HBM with 128 GB/s, leading to the pure energy costs of moving data back and forth between HBM and compute. By contrast, in low compute-intensive MobV3, off-chip reordering exposes 24% critical latency, which further restricts the performance of dataflows as SIGMA has to use some dataflows with the least off-chip accesses. This explains the 1.7 _×_ /1.7 _×_ speedup and 1.99 _×_ /1.66 _×_ efficiency improvement of _FEATHER_ . 

_•_ **Flexible Layout + On-chip line Rotation:** SIGMA is equipped with line rotation, proposed in Medusa [48], to mitigate bank conflicts when reading three lines from the same bank. But typical workloads often access more than 4 lines per cycle. Further, all seven on-chip layouts utilized in the paper require word-granularity data reordering to switch 

209 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

from one to the other, a capability supported by _FEATHER_ but not line rotation, which explains the 1.01 _×_ /1.18 _×_ speedup and 1.90 _×_ /1.85 _×_ efficiency improvement of _FEATHER_ . 

_•_ **Flexible Layout + On-chip Transpose (MTIA-like):** We enhance SIGMA with on-chip data transpose (Fig. 5c), the reordering capability provided by MTIA and TPUv4. Transpose is effective for reducing bank conflicts caused by singledimensional parallelism. However, multi-dimensional parallelism based bank conflicts require finer-grain data reordering, a function supported by _RIR_ but not by transpose. Therefore, FEATHER demonstrates speedups of 1.15 _×_ /1.36 _×_ and achieves 2.2 _×_ /2.06 _×_ greater efficiency, highlighting its superior handling of complex data layout transformations. 

_•_ **Flexible Layout + On-chip Transpose and Row-reorder (TPU-like):** On top of MTIA-like, we further add row-reorder (Fig. 5d). Yet, this enhancement does not reduce on-chip buffer accesses nor modify data locations compared to transpose alone, resulting in no further latency saving or efficiency gains. 

## _D. Resources and Timing Evaluation under TSMC 28nm_ 

_1) BIRRD vs. FAN [35]/ART [42]:_ We have implemented _BIRRD_ in Verilog, and obtained its post-layout resources under different scales. Fig. 14a provides a comparative evaluation of _BIRRD_ with other reduction networks like SIGMA’s FAN and MAERI’s ART, considering int32 adders. Here, the area is represented by lines while the bars demonstrate power consumption. An _AW_ -input _BIRRD_ has more stages (2 _log_ 2( _AW_ )) than FAN/ART ( _log_ 2( _AW_ ) _−_ 1). Consequently, _BIRRD_ consumes about 1.43 _×_ /2.21 _×_ more area and 1.17 _×_ /2.07 _×_ more power than FAN/ART. Despite these overheads, the adoption of _BIRRD_ is justifiable for two primary reasons: 

_•_ Unlike FAN/ART which requires one _AW × AH_ -input instance for all 1D PEs, a single _AW_ -input _BIRRD_ instance can fulfill the reordering and reduction needs for all 2D PEs. As a result, when integrated into _FEATHER_ , _BIRRD_ achieves a resourcesaving of 94% over the FAN in SIGMA, as indicated by the Reduction NoC (Redn. NoC in Fig. 14b). 

> _•_ While both MAERI and SIGMA necessitate a complex distribution NoC such as fat-tree, crossbar, or Benes, _BIRRD_ eliminates such requirements as data always come in a perfect layout without further redistribution demands (§III-B4). Thus _FEATHER_ replaces distribution NoC with pt-to-pt connections. 

_2) FEATHER vs. SIGMA/NVDLA:_ The combined simplification of the distribution NoC and implementation of a singular _BIRRD_ instance results in a substantial 2 _._ 93 _×_ resource reduction of SIGMA - for _FEATHER_ with an equal number of 256 PEs, as illustrated in Fig. 14b. _FEATHER_ has large local memory as each PE in _AW × AH FEATHER_ needs to keep sufficient data inside local memory to perform local reduction when other PE rows are using oAct buses. Further, _FEATHER_ adopts 2D PE array with better scalability compared with 1D PE design in SIGMA. We further implement a NVDLA-like 1D PE array serving as a fix-dataflow baseline. 

**==> picture [111 x 90] intentionally omitted <==**

**==> picture [98 x 77] intentionally omitted <==**

**==> picture [205 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Reduction Network. (b) Resources Breakdown.<br>**----- End of picture text -----**<br>


Fig. 14: ASIC resource comparison ( _FEATHER_ vs. SoTA). 16 _×_ 16 _FEATHER_ place-and-route at TSMC 28nm. 

## _E. Timing Analysis_ 

We layout _FEATHER_ with 64, 256, and 1024 PEs, requiring _BIRRD_ with 8, 16, and 32 inputs. The die photo of _FEATHER_ with 16 _×_ 16 PEs is shown in Fig. 14b revealing that _BIRRD_ consumes only 4% of the overall post-layout area in the TSMC 28nm process. _BIRRD_ does not have long wires because it is placed outside the PE array as a standalone module instead of spreading among PEs like FAN in SIGMA or ART in MAERI. _BIRRD_ attains a peak clock frequency of 1.5 GHz across all scales. The timing critical path of _FEATHER_ is the wire connecting local weights registers to 9-bit multiplier in PE, maxing at 1 GHz, similar to SoTA accelerators [26]. 

## VII. RELATED WORK 

**DNN Accelerators.** Most DNN accelerators today (especially those with end-to-end deployment support [6], [7], [10], [21], [26], [39], [51]) either rely on fixed dataflows - thus fixed layouts or support flexible dataflows [15], [35], [42], [50] but do not consider effects of data layout when switching dataflows, both of which hurt performance. Tab. I,III contrast these. 

**Layout Reordering Support.** To mitigate bank conflicts, Medusa [48] introduces line rotation, which rotates one line inside the conflicted bank into different banks when moving data from off-chip DRAM to on-chip compute. However, typical accelerators with higher parallelism require data in finer word granularity, which leads bank conflicts to line rotation. 

To enable word-level layout reordering, MTIA [19] introduces a Memory Layout Unit (MLU) that enables transposing, concatenating, and reshaping with 4/8/16/32-bit data types. Besides these three layout transforms, _FEATHER_ further supports arbitrary data reordering layout transformations through _BIRRD_ . Moreover, extra reordering latency from MLU is completely hidden behind reduction in _FEATHER_ through RIR. 

## VIII. CONCLUSION 

This work motivates the need for data layout reordering support for switching between dataflows in DNN accelerators. We introduce _FEATHER_ , an innovative accelerator incorporating a novel multi-stage reduction networks called _BIRRD_ to implement a unique on-chip reordering strategy for reordering data in reduction. This facilitates simultaneous dataflow and layout switching in _FEATHER_ without explicit latency costs, resulting in speedup over SoTAs by using less area. 

210 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

## IX. ACKNOWLEDGEMENT 

We thank Hyoukjun Kwon, Geonhwa Jeong and Raveesh Garg for feedbacks, Angshuman Parashar for concordant layout terminology, Sixu Li for place-and-routing, Yongan Zhang for ZCU104 maintenance. This work was supported in part by ACE, one of seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA. 

**==> picture [43 x 7] intentionally omitted <==**

## _A. Abstract_ 

This artifact contains three different flows to evaluate FEATHER with different fidelity against different baselines: (1) end-to-end evaluation on the realistic ZCU 104 FPGA evaluation board, (2) analytic analysis using the mapping search and evaluation framework, LayoutLoop (details in §V) and (3) Verilog implementation with synthesis and place-and-routing evaluation 

## _B. Artifact check-list (meta-information)_ 

Inside the repo, we provide three different experiments in three separate folders. Each folder consists of (1) pre-run results for each experiments, and (2) detailed step-by-step operation to reproduce the pre-run results. 

_1) Experiment Set 1 - Fig. 12:_ deploys _FEATHER_ on ZCU 104 FPGA board and evaluate the end-to-end layer-wise latency of processing ResNet-50. 

_•_ Pre-run per-layer results for inference convolution 3 _×_ 3, 1 _×_ 1 layers of ResNet50 are stored in the output of provided jupyter notebook “feather.ipynb”. 

- Pre-built FPGA bitstream for running on ZCU104 FPGA. 

_2) Experiment Set 2 - Fig. 13:_ automated dataflow-layout coexploration, leveraging Layoutloop, to investigate performance of various baselines and _FEATHER_ . 

_•_ LayoutLoop Framework in path “LayoutLoop/layoutloop”. 

> _•_ Configurations for FEATHER (arbitrary layout choice), SIGMA (arbitrary layout choice), SIGMA (off-chip reordering), MTIA-like (Transpose), TPU-like (Transpose + Shift), SIGMAlike (HWC C4W8), SIGMA-like (HWC C32), Medusa-like (Line Rotation), Eyeriss-like (HWC C32), NVDLA-like (HWC C32), at path “LayoutLoop/configurations”. 

_•_ Pre-run results at path “LayoutLoop/pre run results”. 

- Dataflow-Layout Co-searching scripts ( _>_ 24 hours runtime). 

_3) Experiment Set 3 - Fig. 14:_ ASIC synthesis and placeand-route on Verilog implementation of _FEATHER_ . 

- Pre-run results. 

- automated scripts for synthesis and PnR ( _>_ 96 hours). An overview of the dependency is listed as follows. 

   - **Algorithm** : We adopt pruned random searching algorithm to explore large dataflow design space under various layouts. 

   - **Program** : We provide three experiment sets with click-and-run program entries. 

   - **Compilation** : We provide step-by-step compilation guideline in the repo and a pre-built docker with all pre-compiled programs. 

   - **Binary** : The pre-built hardware binary for realistic FPGA deployment is provided in the repo. 

- **Model:** We use ResNet-50 for end-to-end FPGA evaluation, and meta data from ResNet-50, MobileNet-v3 and BERT for LayoutLoop analytic analysis. 

- **Run-time environment:** Ubuntu, version does not matter. 

- **Hardware:** We provide the access for ZCU 104 evaluation board. Users need to have multi-core CPU with 32 GB memory. 

- **Metrics:** Latency, Latency-Energy-Product 

- **How much time is needed to prepare workflow (approximately)?:** Estimated (1) 1 minutes (2) 10 mins (3) 10 mins. 

- **How much time is needed to complete experiments (approximately)?:** Estimated (1) 10 minutes (2 & 3) _>_ 96 hours. 

- **Publicly available?:** Yes 

- **Code licenses (if publicly available)?:** MIT 

- **Workflow framework used?:** Our proposed Layoutloop is built upon Timeloop [41] 

## _C. Description_ 

- _1) How to access: The artifact is available at 10.5281/_ 

- _zenodo.10999154._ 

_2) Hardware dependencies:_ ZCU104 FPGA evaluation board is required to reproduce the end-to-end evaluation results of FEATHER under ResNet-50. 

_3) Software dependencies:_ 

_•_ (Experiment-1) Prebuilt PYNQ 3.0.1 image for ZCU104 FPGA board with Python 3.10.2, which is available at http: //www.pynq.io/boards.html. 

_•_ (Experiment-2) scons, libconfig++-dev, libboost-dev, libboostiostreams-dev, libboost-serialization-dev, libyaml-cpp-dev, libncurses-dev, libtinfo-dev, libgpm-dev, cmake; Python 3.8 with matplotlib, numpy, pandas. 

_•_ (Experiment-3) Synopsys 2022.12-SP5 and Cadence innovus v21.14-s109 1, both could be other versions. TSMC 28nm technology standard cell libraries. 

## _D. Installation_ 

We provide detailed installation guide and step-by-step instructions to reproduce the results in the repository. 

_1) Results Visualization:_ Our visualization requires the following python packages. 

- $ git clone _<_ repo url _>_ 

- $ conda create −n _<_ favoriate name _>_ python =3.8 

- $ conda activate _<_ favoriate name _>_ 

- $ pip3 install matplotlib numpy pandas 

- $ python results generation.py 

_2) LayoutLoop Setup:_ The compilation and code base requires following libraries on Ubuntu system. 

$ sudo apt install scons libconfig++−dev _\_ libboost−dev libboost−iostreams−dev _\_ libtinfo−dev libboost−serialization −dev cmake _\_ libyaml−cpp−dev libncurses −dev libgpm−dev 

We also provide pre-built docker at https://tinyurl.com/ layoutloop. Install the following packages to run the docker. 

$ sudo apt−get install docker−ce containerd.io _\_ docker−ce−cli docker−buildx−plugin _\_ docker−compose−plugin $docker load −i feather layoutloop docker.tar.gz 

211 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

## _E. Experiment workflow_ 

_1) Experiment Set 1, End-to-end Inference on FPGA:_ 

- We require ZCU104 with pre-built PYNQ image, which 

- provides a jupyter notebook portal. 

- Copy the provided “feather/feather.ipynb” into the jupyter 

- notebook and then run all blocks. 

_2) Experiment Set 2, LayoutLoop Analytic Analysis:_ run the following commands within the pre-built docker. The DSE might take 1 day to finish, depending on the machine. 

$ docker run −it feather layoutloop 

$ git clone _<_ provided url _>_ 

$ cd FEATHER/LayoutLoop/configurations 

$ git pull 

$ make clean 

$ make conv dse _# DSE for ResNet−50, Mob−v3_ 

- feather top power.rpt 

- feather top timing.rpt 

The final reports of PnR contain 

- area.rpt, which contains Post-PnR area value. 

- power.rpt, which contains Post-PnR power value. 

- time, timingReports. # Both are timing reports. 

TABLE V: Post-PnR _FEATHER_ Area/Power at various shapes. 

|Shape|Area (_μ_m2)|Power(mW)|Frequency (GHz)|
|---|---|---|---|
|64_×_128|36920519.69|26400.00|1.00|
|64_×_64|18389176.19|13200.00|1.00|
|32_×_32|2727906.70|961.70|1.00|
|16_×_32|965665.10|655.55|1.00|
|16_×_16|475897.19|323.48|1.00|
|8_×_8|97976.46|65.25|1.00|
|4_×_4|24693.98|16.28|1.00|



$ make gemm dse _# DSE for Bert_ 

## _G. Experiment customization_ 

_3) Experiment Set 3, Synthesis and PNR under TSMC 28nm:_ For synthesis, we use design compiler “dc shell” 

_<_ setup environment for synopsys _>_ $ cd FEATHER/FEATHER RTL/scripts/ 

$ source :run syn 

For place and routing, we use innovus. 

_<_ setup environment for innovus _>_ 

_<_ Finish Synthesis First _>_ $ cd FEATHER/FEATHER RTL/ $ innovus _>_ source PnR.tcl 

## _F. Evaluation and expected results_ 

_1) Exp. Set 1: End-to-end latency on FPGA:_ The layerwise latency of running various models will be shown in the end of the jupyter notebook. We normalize results from different designs using “normalized throughput per PE”, where throughput is measured by inverse of latency under single batch. The visualized result is shown in Fig. 12. 

_2) Exp. Set 2: LayoutLoop Analytic Analysis:_ Per-layer results from Layoutloop could be found at “LayoutLoop/configurations/results” with following naming pattern. 

_•_ design name layout policy slowdown.csv 

- design name layout policy utilization.csv 

- design name layout policy pj commpute.csv 

- design name layout policy cycle.csv 

We calculate GeoMean of “pJ/compute” and “cycle”, and then normalize all results by FEATHER’s performance with the visualized results shown as Fig. 13. 

_3) Exp. Set 3: Synthesis and PNR under TSMC 28nm:_ The final reports of synthesizing _FEATHER_ at a specific scale will be listed in the “reports” folder, including 

- feather top area.rpt 

## _1) Exp. Set 2: LayoutLoop Analytic Analysis:_ 

**Different Configurations:** LayoutLoop adopts the same architecture, dataflow constraint and mapper configurations format as TimeLoop with detailed documentations listed at https://timeloop.csail.mit.edu/v4/input-formats/design. Further, we argument LayoutLoop to support the analysis of layouts. The layout definition is shown in §3. The locations of these configurations are listed below. 

_•_ architecture design: “FEATHER/LayoutLoop/configurations/arch designs/” 

> _•_ dataflow constraints: “FEATHER/LayoutLoop/configurations/arch designs/systolic constraint/mapspace.yaml”, the dataflow constraint needs to match hierarchies of components in the architecture design. 

_•_ mapper: “FEATHER/LayoutLoop/configurations/mapper/” 

_•_ Layout: “FEATHER/LayoutLoop/configurations/layout/” 

**Different on-chip reordering modeling methods** are activated by enabling different global macro 

## _•Transpose:_ ENABLE TRANSPOSE 

## _• Line Rotation:_ MEDUSA 

By default, Layoutloop assumes no on-chip reordering. 

_2) Exp. Set 3: LayoutLoop Analytic Analysis:_ The provided Verilog implementation of _FEATHER_ is a parameterized scalable template, which allows users to change the shape of _FEATHER_ by modifying the input parameters at the top module “FEATHER/FEATHER RTL/RTL/feather top.v”. Users could modify the following parameters into value from (4 _,_ 8 _,_ 16 _,_ 32 _,_ 64) to investigate the area and power of _FEATHER_ at different scsales. 

## module feather top _#(_ 

parameter DPE COL NUM = 64, parameter DPE ROW NUM = 64, 

... 

- feather top dw area.rpt 

212 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] “Coral usb accelerator,” https://coral.ai/products/accelerator/, accessed: 2024-02-23. 

- [2] “Xilinx deep learning processing unit,” https://docs.xilinx.com/r/1.2English/ug1414-vitis-ai/Deep-Learning-Processor-Unit-DPU, accessed: 2022-12-10. 

- [3] W. Ali, S. Abdelkarim, M. Zidan, M. Zahran, and A. El Sallab, “Yolo3d: End-to-end real-time 3d oriented object bounding box detection from lidar point cloud,” in _Proceedings of the European Conference on Computer Vision (ECCV) Workshops_ , 2018, pp. 0–0. 

- [4] S. Arora, T. Leighton, and B. Maggs, “On-line algorithms for path selection in a nonblocking network,” in _Proceedings of the TwentySecond Annual ACM Symposium on Theory of Computing_ , ser. STOC ’90. New York, NY, USA: Association for Computing Machinery, 1990, p. 149–158. [Online]. Available: https://doi.org/10.1145/100216.100232 

- [5] ——, “On-line algorithms for path selection in a nonblocking network,” in _Proceedings of the twenty-second annual ACM symposium on Theory of computing_ , 1990, pp. 149–158. 

- [6] P. Behnam, J. Tong, A. Khare, Y. Chen, Y. Pan, P. Gadikar, A. Bambhaniya, T. Krishna, and A. Tumanov, “Hardware–software co-design for real-time latency–accuracy navigation in tiny machine learning applications,” _IEEE Micro_ , vol. 43, no. 06, pp. 93–101, nov 2023. 

- [7] P. Behnam, J. Tong, A. Khare, Y. Chen, Y. Pan, P. Gadikar, A. R. Bambhaniya, T. Krishna, and A. Tumanov, “Subgraph stationary hardwaresoftware inference co-design,” 2023. 

- [8] B. Bogdanski, “Optimized routing for fat-tree topologies,” _Department of Informatics, Faculty of Mathematics and Natural Sciences, University of Oslo, Norway_ , 2014. 

- [9] P. Chatarasi, H. Kwon, A. Parashar, M. Pellauer, T. Krishna, and V. Sarkar, “Marvel: A data-centric approach for mapping deep learning operators on spatial accelerators,” _ACM Trans. Archit. Code Optim._ , vol. 19, no. 1, dec 2021. [Online]. Available: https://doi.org/10.1145/3485137 

- [10] P. Chatarasi, S. Neuendorffer, S. Bayliss, K. Vissers, and V. Sarkar, “Vyasa: A high-performance vectorizing compiler for tensor convolutions on the xilinx ai engine,” in _2020 IEEE High Performance Extreme Computing Conference (HPEC)_ , 2020, pp. 1–10. 

- [11] K. Chellapilla, S. Puri, and P. Simard, “High performance convolutional neural networks for document processing,” in _Tenth international workshop on frontiers in handwriting recognition_ . Suvisoft, 2006. 

- [12] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, “Rethinking atrous convolution for semantic image segmentation,” _arXiv preprint arXiv:1706.05587_ , 2017. 

- [13] Y.-H. Chen, T. Krishna, J. S. Emer, and V. Sze, “Eyeriss: An EnergyEfficient Reconfigurable Accelerator for Deep Convolutional Neural Networks,” _IEEE Journal of Solid-State Circuits_ , vol. 52, no. 1, pp. 127–138, 2016. 

- [14] ——, “Eyeriss: An energy-efficient reconfigurable accelerator for deep convolutional neural networks,” _IEEE Journal of Solid-State Circuits_ , vol. 52, no. 1, pp. 127–138, 2017. 

- [15] Y.-H. Chen, T.-J. Yang, J. Emer, and V. Sze, “Eyeriss v2: A Flexible Accelerator for Emerging Deep Neural Networks on Mobile Devices,” _arXiv preprint arXiv:1807.07928_ , 2018. 

- [16] W. J. Dally and B. P. Towles, _Principles and practices of interconnection networks_ . Elsevier, 2004. 

- [17] M. Dukhan, Y. Wu, and H. Lu, “Qnnpack: Open source library for optimized mobile deep learning,” 2018. 

- [18] V. Fedyunin, “(beta) channels last memory format in pytorch¶.” [Online]. Available: https://pytorch.org/tutorials/intermediate/memory format tutorial.html 

- [19] A. Firoozshahian, J. Coburn, R. Levenstein, R. Nattoji, A. Kamath, O. Wu, G. Grewal, H. Aepala, B. Jakka, B. Dreyer, A. Hutchin, U. Diril, K. Nair, E. K. Aredestani, M. Schatz, Y. Hao, R. Komuravelli, K. Ho, S. Abu Asal, J. Shajrawi, K. Quinn, N. Sreedhara, P. Kansal, W. Wei, D. Jayaraman, L. Cheng, P. Chopda, E. Wang, A. Bikumandla, A. Karthik Sengottuvel, K. Thottempudi, A. Narasimha, B. Dodds, C. Gao, J. Zhang, M. Al-Sanabani, A. Zehtabioskuie, J. Fix, H. Yu, R. Li, K. Gondkar, J. Montgomery, M. Tsai, S. Dwarakapuram, S. Desai, N. Avidan, P. Ramani, K. Narayanan, A. Mathews, S. Gopal, M. Naumov, V. Rao, K. Noru, H. Reddy, P. Venkatapuram, and A. Bjorlin, “Mtia: First generation silicon targeting meta’s recommendation systems,” in _Proceedings of the 50th Annual International Symposium on Computer Architecture_ , ser. ISCA ’23. New York, NY, USA: 

Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3579371.3589348 

- [20] S. Gao, X. Chen, P. Li, Z. Ren, L. Bing, D. Zhao, and R. Yan, “Abstractive text summarization by incorporating reader comments,” in _Proceedings of the AAAI Conference on Artificial Intelligence_ , vol. 33, no. 01, 2019, pp. 6399–6406. 

- [21] H. Genc, A. Haj-Ali, V. Iyer, A. Amid, H. Mao, J. C. Wright, C. Schmidt, J. Zhao, A. J. Ou, M. Banister, Y. S. Shao, B. Nikolic, I. Stoica, and K. Asanovic, “Gemmini: An agile systolic array generator enabling systematic evaluations of deep-learning architectures,” _CoRR_ , vol. abs/1911.09925, 2019. [Online]. Available: http://arxiv.org/abs/1911.09925 

- [22] K. Hegde, P.-A. Tsai, S. Huang, V. Chandra, A. Parashar, and C. W. Fletcher, “Mind mappings: enabling efficient algorithm-accelerator mapping space search,” in _Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , ser. ASPLOS ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 943–958. [Online]. Available: https://doi.org/10.1145/3445814.3446762 

- [23] Q. Huang, M. Kang, G. Dinh, T. Norell, A. Kalaiah, J. Demmel, J. Wawrzynek, and Y. S. Shao, “Cosa: Scheduling by constrained optimization for spatial accelerators,” in _Proceedings of the 48th Annual International Symposium on Computer Architecture_ , ser. ISCA ’21. IEEE Press, 2021, p. 554–566. [Online]. Available: https://doi.org/10.1109/ISCA52012.2021.00050 

- [24] G. Jeong, G. Kestor, P. Chatarasi, A. Parashar, P.-A. Tsai, S. Rajamanickam, R. Gioiosa, and T. Krishna, “Union: A unified hw-sw co-design ecosystem in mlir for evaluating tensor operations on spatial accelerators,” in _2021 30th International Conference on Parallel Architectures and Compilation Techniques (PACT)_ , 2021, pp. 30–44. 

- [25] H. Jiang, P. He, W. Chen, X. Liu, J. Gao, and T. Zhao, “SMART: Robust and efficient fine-tuning for pre-trained natural language models through principled regularized optimization,” in _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_ . Online: Association for Computational Linguistics, Jul. 2020, pp. 2177–2190. [Online]. Available: https://aclanthology.org/2020.acl-main.197 

- [26] N. P. Jouppi, D. Hyun Yoon, M. Ashcraft, M. Gottscho, T. B. Jablin, G. Kurian, J. Laudon, S. Li, P. Ma, X. Ma, T. Norrie, N. Patil, S. Prasad, C. Young, Z. Zhou, and D. Patterson, “Ten lessons from three generations shaped google’s tpuv4i : Industrial product,” in _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ , 2021, pp. 1–14. 

- [27] S.-C. Kao and T. Krishna, “Gamma: Automating the hw mapping of dnn models on accelerators via genetic algorithm,” in _Proceedings of the 39th International Conference on Computer-Aided Design_ , ser. ICCAD ’20. New York, NY, USA: Association for Computing Machinery, 2020. [Online]. Available: https://doi.org/10.1145/3400302.3415639 

- [28] ——, “Magma: An optimization framework for mapping multiple dnns on multiple accelerator cores,” in _2022 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ , 2022, pp. 814–830. 

- [29] S.-C. Kao, A. Parashar, P.-A. Tsai, and T. Krishna, “Demystifying map space exploration for npus,” 2022. 

- [30] S.-C. Kao, M. Pellauer, A. Parashar, and T. Krishna, “Digamma: Domain-aware genetic algorithm for hw-mapping co-optimization for dnn accelerators,” in _2022 Design, Automation Test in Europe Conference Exhibition (DATE)_ , 2022, pp. 232–237. 

- [31] D. Khudia, J. Huang, P. Basu, S. Deng, H. Liu, J. Park, and M. Smelyanskiy, “Fbgemm: Enabling high-performance low-precision deep learning inference,” _arXiv preprint arXiv:2101.05615_ , 2021. 

- [32] T. Krishna, H. Kwon, A. Parashar, M. Pellauer, and A. Samajdar, “Data orchestration in deep learning accelerators,” 2020. 

- [33] H. Kwon, P. Chatarasi, V. Sarkar, T. Krishna, M. Pellauer, and A. Parashar, “Maestro: A data-centric approach to understand reuse, performance, and hardware cost of dnn mappings,” _IEEE Micro_ , vol. 40, no. 3, pp. 20–29, 2020. 

- [34] H. Kwon, M. Pellauer, A. Parashar, and T. Krishna, “Flexion: A quantitative metric for flexibility in dnn accelerators,” _IEEE Computer Architecture Letters_ , vol. 20, no. 1, pp. 1–4, 2021. 

- [35] H. Kwon, A. Samajdar, and T. Krishna, “MAERI: Enabling Flexible Dataflow Mapping over DNN Accelerators via Reconfigurable Interconnects,” in _Proceedings of the 23rd International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)_ , 2018. 

213 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

- [36] C. E. Leiserson, “Fat-trees: universal networks for hardware-efficient supercomputing,” _IEEE transactions on Computers_ , vol. 100, no. 10, pp. 892–901, 1985. 

- [37] W. Liu, S. Liao, W. Ren, W. Hu, and Y. Yu, “High-level semantic feature detection: A new perspective for pedestrian detection,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2019, pp. 5187–5196. 

- [38] D. Matani, “Efficient pytorch: Tensor memory format matters.” [Online]. Available: https://pytorch.org/blog/tensor-memory-format-matters/ 

- [39] NVIDIA. (2016) NVIDIA Deep Learning Accelerator (NVDLA). [Online]. Available: http://nvdla.org/primer.html 

- [40] K. Ovtcharov, O. Ruwase, J.-Y. Kim, J. Fowers, K. Strauss, and E. S. Chung, “Accelerating deep convolutional neural networks using specialized hardware,” _Microsoft Research Whitepaper_ , vol. 2, no. 11, pp. 1–4, 2015. 

- [41] A. Parashar, P. Raina, Y. S. Shao, Y.-H. Chen, V. A. Ying, A. Mukkara, R. Venkatesan, B. Khailany, S. W. Keckler, and J. Emer, “Timeloop: A Systematic Approach to DNN Accelerator Evaluation,” in _Proceedings of the International Symposium on Performance Analysis of Systems and Software (ISPASS)_ , 2019. 

- [42] E. Qin, A. Samajdar, H. Kwon, V. Nadella, S. Srinivasan, D. Das, B. Kaul, and T. Krishna, “Sigma: A sparse and irregular gemm accelerator with flexible interconnects for dnn training,” in _2020 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ , 2020, pp. 58–70. 

- [43] A. Reuther, P. Michaleas, M. Jones, V. Gadepally, S. Samsi, and J. Kepner, “Ai and ml accelerator survey and trends,” 2022. [Online]. Available: https://arxiv.org/abs/2210.04055 

- [44] A. Samajdar, M. Pellauer, and T. Krishna, “Self-adaptive reconfigurable arrays (sara): Using ml to assist scaling gemm acceleration,” _ArXiv_ , vol. abs/2101.04799, 2021. 

- [45] A. Samajdar, Y. Zhu, P. Whatmough, M. Mattina, and T. Krishna, 

   - “SCALE-Sim: Systolic CNN Accelerator Simulator,” _arXiv preprint arXiv:1811.02883_ , 2018. 

- [46] K. Seshadri, B. Akin, J. Laudon, R. Narayanaswami, and A. Yazdanbakhsh, “An evaluation of edge tpu accelerators for convolutional neural networks,” 2022. 

- [47] Y. S. Shao, J. Clemons, R. Venkatesan, B. Zimmer, M. Fojtik, N. Jiang, B. Keller, A. Klinefelter, N. Pinckney, P. Raina, S. G. Tell, Y. Zhang, W. J. Dally, J. Emer, C. T. Gray, B. Khailany, and S. W. Keckler, “Simba: Scaling deep-learning inference with multichip-module-based architecture,” in _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ , ser. MICRO ’52. New York, NY, USA: Association for Computing Machinery, 2019, p. 14–27. [Online]. Available: https://doi.org/10.1145/3352460.3358302 

- [48] Y. Shen, T. Ji, M. Ferdman, and P. Milder, “Medusa: A scalable interconnect for many-port dnn accelerators and wide dram controller interfaces,” in _2018 28th International Conference on Field Programmable Logic and Applications (FPL)_ , 2018, pp. 101–1014. 

- [49] H. Tao, M. M. Hameed, H. A. Marhoon, M. Zounemat-Kermani, S. Heddam, S. Kim, S. O. Sulaiman, M. L. Tan, Z. Sa’adi, A. D. Mehr, M. F. Allawi, S. Abba, J. M. Zain, M. W. Falah, M. Jamei, N. D. Bokde, M. Bayatvarkeshi, M. Al-Mukhtar, S. K. Bhagat, T. Tiyasha, K. M. Khedher, N. Al-Ansari, S. Shahid, and Z. M. Yaseen, “Groundwater level prediction using machine learning models: A comprehensive review,” _Neurocomputing_ , vol. 489, pp. 271–308, 2022. [Online]. Available: 

   - https://www.sciencedirect.com/science/article/pii/S092523122200282X 

- [50] J. Weng, S. Liu, V. Dadu, Z. Wang, P. Shah, and T. Nowatzki, “Dsagen: Synthesizing programmable spatial accelerators,” in _2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)_ , 2020, pp. 268–281. 

- [51] Xilinx. (2022) Xilinx Deep Learning Unit (DPU). [Online]. Available: https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Deep-LearningProcessor-Unit 

214 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:27:55 UTC from IEEE Xplore.  Restrictions apply. 

