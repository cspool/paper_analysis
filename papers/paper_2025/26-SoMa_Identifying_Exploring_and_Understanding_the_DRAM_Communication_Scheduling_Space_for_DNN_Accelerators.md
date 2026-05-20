2025 IEEE International Symposium on High Performance Computer Architecture (HPCA) 

**==> picture [29 x 29] intentionally omitted <==**

**==> picture [29 x 29] intentionally omitted <==**

**==> picture [28 x 28] intentionally omitted <==**

# SoMa: Identifying, Exploring, and Understanding the DRAM Communication Scheduling Space for DNN Accelerators 

Jingwei Cai _[†]_ , Xuan Wang _[‡§]_ , Mingyu Gao _[†¶♠]_ , Sen Peng _[‡§]_ , Zijian Zhu _[†]_ , Yuchen Wei _[†]_ , Zuotong Wu _[‡§]_ , and Kaisheng Ma _[†∗]_ Tsinghua University _[†]_ , Xi’an Jiaotong University _[‡]_ , IIISCT _[§]_ , Shanghai AI Laboratory _[¶]_ Shanghai Qi Zhi Institute _[♠]_ Corresponding Author _[∗]_ 

_{_ caijw21,gaomy,zhuzj23,weiyc22,kaisheng _}_ @tsinghua.edu.cn, _{_ xuanwang,3123353083 _}_ @stu.xjtu.edu.cn 

_**Abstract**_ **—Modern Deep Neural Network (DNN) accelerators are equipped with increasingly larger on-chip buffers to provide more opportunities to alleviate the increasingly severe DRAM bandwidth pressure. However, most existing research on buffer utilization still primarily focuses on single-layer dataflow scheduling optimization. As buffers grow large enough to accommodate most single-layer weights in most networks, the impact of singlelayer dataflow optimization on DRAM communication diminishes significantly. Therefore, developing new paradigms that fuse multiple layers to fully leverage the increasingly abundant onchip buffer resources to reduce DRAM accesses has become particularly important, yet remains an open challenge.** 

**To address this challenge, we first identify the optimization opportunities in DRAM communication scheduling by analyzing the drawbacks of existing works on the layer fusion paradigm and recognizing the vast optimization potential in scheduling the timing of data prefetching from and storing to DRAM. To fully exploit these optimization opportunities, we develop a Tensor-centric Notation and its corresponding parsing method to represent different DRAM communication scheduling schemes and depict the overall space of DRAM communication scheduling. Then, to thoroughly and efficiently explore the space of DRAM communication scheduling for diverse accelerators and workloads, we develop an end-to-end scheduling framework, SoMa, which has already been developed into a compiler for our commercial accelerator product. Compared with the stateof-the-art (SOTA) Cocco framework, SoMa achieves, on average, a 2.11** _×_ **performance improvement and a 37.3% reduction in energy cost simultaneously. Then, we leverage SoMa to study optimizations for LLM, perform design space exploration (DSE), and analyze the DRAM communication scheduling space through a practical example, yielding some interesting insights. Moreover, SoMa has been open-sourced at https://github.com/SETScheduling-Project/SoMa-HPCA2025.** 

## I. INTRODUCTION 

In order to process a variety of tasks with better performance and accuracy, DNNs are rapidly becoming more complex and heavy [13], [14], [19], [41], [51]. Accelerators [23], [25], [26], [31], [34], [38] with more computing units, larger buffers, and higher memory bandwidth have been developed to accelerate these DNN workloads. 

However, under modern semiconductor process, the rate of increase in DRAM bandwidth lags behind the rate of transistor density growth, which is a long-standing issue [46], [54]. This disparity is even more pronounced in DNN accelerators [46], 

[60], which are more specialized and have a larger proportion of dedicated computing elements than traditional chips like CPU. Thus, DRAM communication is increasingly becoming a performance bottleneck in DNN computation [46], [60]. To mitigate this bottleneck, accelerators are equipped with increasingly larger on-chip buffers [18], [23], [25], [31], [34], [38], which provide opportunities to optimize DRAM communication by exploiting reuse opportunities in DNNs. 

Several works leverage buffer resources to reduce DRAM accesses under the paradigm of “layer fusion” [3], [7], [8], [29], [33], [37], [49], [60]. This approach involves buffering the feature maps (fmaps) produced by earlier layers on-chip, allowing subsequent consuming layers to read them directly. This strategy avoids the overhead of first writing back to DRAM and then reading from it, thereby reducing DRAM access costs. This optimization paradigm holds immense potential; for instance, Cocco [49] achieved performance improvements ranging from 1.89% to 50.33% by merely exploring the option of which layers to fuse. Beyond this, there are numerous dimensions worth exploring under this paradigm, such as execution order and execution granularity. However, like Cocco, most existing studies [3], [37] have only focused on a small portion of this optimization space. Therefore, we believe that the complex optimization dimensions within the layer-fusion paradigm have yet to be clearly delineated or defined, much less thoroughly explored and understood in the context of the entire layer-fusion optimization space. 

While reducing DRAM access is an important approach for optimizing DRAM communication, we identify another optimization approach that has been largely overlooked in the field of DNN scheduling: prefetching and delayed storing, i.e., adjusting the timing of fetching/storing data from/to DRAM to be sometime earlier/later. We focus on this approach and believe it has potential based on an insightful observation: in modern DNN networks, the ratio of DRAM bandwidth demand to computing demand varies significantly across different layers (see Fig. 3). After layer fusion, the overall ratio of DRAM bandwidth demand to computing demand for different tiles (computing unit) varies even more (see Fig. 3). This observation indicates that with the application of layer fusion, DRAM bandwidth usage during the entire computing 

979-8-3315-0647-6/25/$31.00 ©2025 IEEE DOI 10.1109/HPCA61900.2025.00048 

533 

process becomes very uneven—sometimes leading to congestion due to high demand and sometimes causing bandwidth resource waste due to low demand. This motivates us to apply prefetching-and-delayed-storing techniques to alleviate the uneven DRAM communication load. However, choosing the appropriate timing for prefetching and storing is a nontrivial problem. Clearly defining, thoroughly exploring, and understanding this paradigm is a significant challenge. 

“Layer Fusion” and “Prefetching and Delayed Storing” each have their own optimization spaces, but they are not independent and are intricately coupled. This is reflected in the following points: 1) both paradigms trade buffer usage for DRAM communication optimization, leading to competition for buffer usage, and 2) layer fusion affects the types and quantities of data that need to communicate with (i.e., prefetch from and store to) DRAM. Therefore, we define the complex space formed by these two paradigms as the DRAM Communication Scheduling Space. 

After identifying the challenges and optimization potential of DRAM communication scheduling, we make the following contributions to identify, explore, and understand DRAM Communication Scheduling Space. 

We first introduce a Tensor-centric Notation with two categories and a total of six attributes to encode the scheduling schemes in the DRAM Communication Scheduling Space; then we show how to parse each encoded scheme into actual hardware behaviors. Based on this notation, we define and illustrate the DRAM Communication Scheduling Space and the complex trade-offs behind it. Existing works can be described using our notation, representing only a small subset within the space we have defined. To the best of our knowledge, this work is _**the first**_ to comprehensively define and analyze the DRAM Communication Scheduling Space. 

To thoroughly and structurally explore the DRAM Communication Scheduling Space defined by the Tensor-centric notation, we develop an end-to-end framework, SoMa, which employs a Buffer Allocator, a two-stage simulated annealing (SA) exploration engine, and an accurate simulator to conduct a structured and efficient exploration of the space. We have successfully built a complete compilation flow based on SoMa, from model input to instruction generation, for an accelerator approaching mass production. 

Then, we conduct extensive experiments on different workloads (including CNNs and LLMs), hardware configurations, and batch sizes, demonstrating an average of a 2.11 _×_ performance improvement and a 37.3% reduction in energy costs compared to the SOTA framework, Cocco. We analyzed the experimental performance of LLMs and uncovered several intriguing phenomena: (1): For the decode stage, the optimization potential of DRAM scheduling is minimal due to its extremely low compute density, which imposes a pure DRAM bandwidth demand on the accelerator. (2): For the decode stage, increasing the batch size does not consistently improve computational utilization. As the batch size grows, the increasing size of the KV cache becomes comparable to that of the weights, diminishing the benefits of further increases in 

**==> picture [190 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
WL0<br>Core<br>PE Array<br>Core<br>… Vector<br>Unit<br>Core<br>OL0<br>(a) DNN Accelerator (b) Computing Core<br>AL0<br>DRAM<br>Global Buffer<br>**----- End of picture text -----**<br>


Fig. 1. DNN Accelerator Template 

batch size for improving compute density. In addition, we use SoMa to explore and analyze the architectural design space, gaining some interesting insights. For example, with small batch sizes, DRAM bandwidth plays an irreplaceable role. However, with SoMa, the importance of the buffer becomes increasingly prominent as the batch size increases. Moreover, we present a practical execution graph comparison between Cocco and SoMa to enhance understanding of the trade-offs underlying the DRAM Communication Scheduling Space. 

## II. HARDWARE BASIS 

In this section, we introduce a generic large-scale DNN accelerator template (Fig. 1) and the corresponding abstract instruction system adopted in this paper, which represents many mainstream commercial DNN accelerators [6], [16], [24]–[26], [34], [38]. 

As shown in Fig. 1(a), the template primarily consists of DRAM, a Global Buffer (GBUF), and several cores. The GBUF is shared among all cores. As shown in Fig. 1(b), each core has private small buffers/register files ( _WL_ 0 _, AL_ 0 _, OL_ 0) for rapid access by computing units. The PE Array is used for computing GEMM/Conv operations, and the Vector Unit is designed for computing other vector/scalar operations, such as element-wise addition, pooling, layer normalization, etc. 

Although specific instructions vary significantly among these accelerators, they still share apparent common patterns. Based on these common patterns, we abstract three instructions: load, store, and compute. The “load” and “store” instructions refer to moving data from DRAM to the GBUF and from the GBUF to DRAM, respectively. The “compute” instruction refers to the operations performed on a tensor/vector. In accelerators, a tensor is often divided into smaller tensors, which are sequentially processed by the core group. Each small tensor is further split into smaller subtensors for parallel processing by the cores within the core group. The specific operations involved include loading ifmaps and weights from GBUF into the local buffer of each core, performing the computations, and then writing the computed ofmaps back to the GBUF. Since these instructions typically occur in sets and are synchronized, and as this study focuses on optimizing DRAM communication, we abstract them into a single “compute” instruction. In our DRAM-COMPUTE diagram (e.g., Fig. 4 Bottom), “load & store” instructions and “compute” instructions can be respectively represented by 

534 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [204 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
Overlap<br>Overlap<br>Width (W)<br>Tile 1 Tile 3 Tile 1 Tile 3 Tile 1 Tile 3<br>Layer A Layer B Layer C<br>DRAM IA1 WA WB WC IA2 OC1 IA3 OC2 IA4 OC3 OC4<br>COMPUTE A1 B1 C1 A2 B2 C2 A3 B3 C3 A4 B4 C4<br>Height (H)<br>**----- End of picture text -----**<br>


Fig. 2. A Practical Layer-fusion Group ( _LG_ ) Example 

tensor blocks in the DRAM row and the COMPUTE row. The start and end of any instruction can serve as markers for the beginning of another instruction (Fig. 4 Right). 

## III. IDENTIFY DRAM COMMUNICATION OPTIMIZATION OPPORTUNITIES 

The GBUF plays a crucial role in optimizing DRAM communication within DNN accelerators, and research on how to use it to optimize DRAM communication has been a hot topic since the advent of DNN accelerators [10]–[12], [26], [35]. However, most current studies focus on optimizing dataflow for small-scale DNN accelerators, specifically on how to tile a single layer into small parts and adjust their computing order to optimize DRAM communication [20], [21], [27], [32], [36], [42], [48], [52], [55], [57]. 

As DNN accelerators have evolved, architects have equipped them with increasingly larger GBUFs, with capacities reaching tens or even hundreds of megabytes [2], [16], [24], [25], [31], [34]. Such large buffers can accommodate most individual layers of most networks, significantly reducing the effects of optimizing single-layer dataflow for DRAM communication. Therefore, developing new techniques to fully leverage the rapidly growing on-chip buffer resources to optimize the increasingly bottlenecked DRAM bandwidth is crucial and offers significant opportunities. Next, we will analyze the challenges and optimization opportunities in DRAM communication scheduling. 

## _A. Layer Fusion_ 

Layer fusion is an important paradigm for using buffers to optimize DRAM communication [3], [37], [49], [59]. Fig. 2 illustrates the computation and DRAM access behavior of a simple three-layer network under layer fusion, where multiple fused layers are computed in a fine-grained manner sequentially, relying on on-chip buffers for switching fmaps. The optimization dimensions within this paradigm are numerous and complex, but have not been clearly depicted and analyzed. A straightforward optimization dimension is which layers to fuse, which affects both DRAM access savings and buffer occupancy. This is the primary focus of most existing studies [49], [59]. Another evident optimization dimension is the computing granularity of each fused layer group, i.e., the size of each computing tile (Fig. 2). This affects buffer occupancy, halo overlap overhead, and the effectiveness of intra-core optimizations (detailed analysis is in Sec. IV-A1). 

**==> picture [204 x 168] intentionally omitted <==**

**----- Start of picture text -----**<br>
Normalized Operations Normalized Operations<br>(a) Layers of ResNet-50 (b) Layers of Transformer-Large<br>More Spread Out More Spread Out<br>Normalized Operations Normalized Operations<br>(c) Tiles of ResNet-50 (d) Tiles of Transformer-Large<br>Normalized DRAM Access Normalized DRAM Access<br>Normalized DRAM Access Normalized DRAM Access<br>**----- End of picture text -----**<br>


Fig. 3. (a) and (b) show the normalized DRAM access and the normalized operation number for each layer in ResNet-50 and Transformer-Large, respectively (each point represents a layer). (c) and (d) show the normalized DRAM access and the normalized operation number for each smallest computing unit (Tile) of ResNet-50 and Transformer-Large, respectively, scheduled using the SOTA Cocco Framework (each point represents a Tile). _**The darker the color, the more identical overlapped points there are**_ . The normalization method involves dividing the value of each point by the maximum value among all points (DRAM access and operations are independently normalized). We use the default edge accelerator and batch size 1, as introduced in Sec. VI-A 

DeFENIS [37] has analyzed this dimension but has not jointly explored the above two dimensions. Most other studies address this dimension using heuristic rules [49], [59], overlooking optimization opportunities within this space. We can see that even these two dimensions lack systematic joint exploration, let alone additional dimensions (introduced in Sec. IV-A1) that have been overlooked by existing works. 

## _B. Prefetching and Delayed Storing_ 

Besides layer fusion, we have identified another optimization opportunity overlooked by existing literature: prefetching and delayed storing. We argue this paradigm also has great potential based on the following insightful observation. 

As shown in Fig. 3(a) and (b), in modern DNNs, the DRAM access demand and computing requirements vary significantly across different layers. Unfortunately, after layer fusion, the variance in the memory access to computation ratio for each tile becomes even more pronounced (see Fig. 3(c) and (d)), which are more spread out along the axes compared to their counterpart in Fig. 3(a) and (b). This indicates a larger number of DRAM-access-intensive and compute-intensive tiles, with fewer tiles having balanced demands. Specifically, this is because, within a fused layer group, the first tile of each layer with weights needs to load weights (e.g., _A_ 1 _, B_ 1 _, C_ 1 in Fig. 2), which often results in high DRAM demand, corresponding to the points near the Y-axis in Fig. 3(c) and (d). The subsequent tiles do not need to load weights (e.g., _A_ 2 _, B_ 2 _, C_ 2 in Fig. 2). Additionally, many fmaps’ DRAM access requirements are eliminated as a result of fusion (e.g., the ofmaps of _Ai_ and _Bi_ ). Consequently, many tiles even have no DRAM access demand (e.g., _B_ 2 _, B_ 3 _, B_ 4), corresponding to the points near the X-axis in Fig. 3(c) and (d). 

535 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [488 x 221] intentionally omitted <==**

**----- Start of picture text -----**<br>
D Pooling<br>A B C DRAM         To decide.<br>E Conv/GEMM<br>Topology COMPUTE A1 A2 B C1 E1 D1 C2 E2 D2 END<br>Computing Order [ A, B, C, E, D ] FLG1 of LG1 FLG2 of LG1 LG2<br>Layer-fusion BUFFER<br>Related Fusion Cut SetFine-grained  { 1, 2 } [ A 1| B |2 C, E, D ] DecideTo DecideTo DecideTo DecideTo DecideTo DecideTo DecideTo DecideTo DecideTo<br>Attributes<br>Tiling Number A:2, B:1, [C, E, D]:2<br>(LFA) OA2<br>IB<br>DRAM Cut Set { 2 } [ A, B C, E, D ]2 OA1 OA1 OC1 IE1 ID1 OC2 IE2 ID2<br>Time<br>from/toTensors IA1, IA2, WA, WB, OB, WD, OD1,  (a) Parsing LFA<br>DRAM IC1, IC2, WE, OE1, OE2, OD2<br>DRAM TensorDRAM Order IWA1, WD, IC2A, I, OA2E1, W, OB, WD1, OE, OE2, OB, ID2C1,  DRAMCOMPUTE IA1 WA IA2A1 A2 WB WE B OB IC1 WC1D EIC21 OE1 D1 OC2D1 E2 ODE22 OD2 END<br>Load & Store BUFFER OB IC1 OE1 OD1 OD1 OD1<br>Related (A1, A2), (A1, B), (A1, B), (B, C1),  IA2 IA2 WE WE WE WE WE WE OD2<br>Attributes Living  (B, D2), (B, C1), (C1, E1), (C1,END),  WA WA WB IC2 IC2 IC2 IC2 OE2 OE2<br>(DLSA) Duration (C1, E2), (E1, D1), (D1, E2),  IA1 OA2 IB WD WD WD WD WD WD<br>OA1 OA1 OC1 IE1 ID1 OC2 IE2 ID2<br>(E2, END), (D2, END) Time<br>(b) Parsing DLSA<br>**----- End of picture text -----**<br>


Fig. 4. Parsing an Example Encode of a Five-Layer Network into Actual Scheduling Schemes. The DRAM, COMPUTE, and BUFFER in the right part show the DRAM access, workload computation, and buffer usage, respectively. We use _Mi_ to represent the _i_ th tile of layer _M_ , and _I/OMi_ to represent the ifmaps and ofmaps of _Mi_ . _WM_ represent the weights of layer _M_ . In the BUFFER, blocks with the same background color represent the same data. 

This severe imbalance between DRAM access and compute demands makes the overlap between DRAM access and compute under the traditional double-buffer strategy (prefetching data in the previous tile and storing data in the next tile) insufficient, especially in the context of layer fusion. For example, Fig. 2 shows the DRAM communication under the traditional double-buffer strategy. As seen, there is a waste of DRAM bandwidth, and the computing resources are severely stalled. To better demonstrate the severity and prevalence of this challenge, we analyze the workloads, batch sizes, and platforms in Sec. VI-A. With the SOTA Cocco Scheduling Strategy, the DRAM and computation utilization rates for the case in Fig. 2(c) are 52.69% and 62.64%, respectively, while in Fig. 2(d) they are 72.45% and 45.84%. These utilization are defined as the ratio of the sum of all DRAM tensors/computing tiles time to the total runtime. This result indicates that neither resource is fully utilized, leaving significant opportunities for overlap. 

Based on the above observation and analysis, we find that controlling the timing of data prefetching and storing to utilize idle DRAM bandwidth and alleviate peak-time pressure has great potential. For example, suppose that there are two consecutive identical Layer-fusion Groups ( _LG_ and _LG[′]_ ) (Fig. 2 shows _LG_ , and _LG[′]_ is not shown). By prefetching _LG[′]_ ’s _WA[′][, W] B[ ′]_[, and] _[ W][ ′] C_[during the DRAM idle time corresponding] to _B_ 3 _, B_ 4, and _C_ 4 in _LG_ , the computing stall at the start of _LG[′]_ can be resolved. Additionally, starting to load _IA_ 3 and _IA_ 4 in _LG_ earlier can also erase the stalls before _A_ 3 and _A_ 4 in _LG_ . However, achieving precise and automatic control over prefetching and delayed storing is non-trivial. Clearly defining, thoroughly exploring, and understanding this paradigm is a significant challenge. 

## _C. Combine Them Together_ 

“Layer Fusion” and “Prefetching and Delayed Storing” are two optimization paradigms with complex interrelationships, which lie in two aspects. The first aspect lies in the fact that both optimization paradigms inherently trade buffer usage for DRAM communication optimization, leading to competition for buffer resources between the two. Additionally, the layer fusion choice affects DRAM communication requirements, which in turn impacts the optimization space of prefetching and delayed storing. Therefore, these two paradigms exhibit complex interactions within the DRAM Communication Scheduling Space. Efficiently and structurally exploring this space presents a major challenge. Addressing this challenge and gaining a deep understanding of these interactions form the key objectives of this paper. 

## IV. TENSOR-CENTRIC NOTATION 

## _A. Encoding Format and Parsing Methods_ 

In this section, through a practical example of a five-layer network ( _A_ to _E_ ) shown in Fig. 4, we explain how the proposed Tensor-centric Notation translates layers into finegrained tensor computations, DRAM accesses, and buffer usage, as well as the trade-offs associated with different encoding choices. 

As shown in Fig. 4, the notation has six attributes, which can be divided into two categories: Layer-Fusion-related Attributes (LFA) and DRAM-Load-and-Store-related Attributes (DLSA). Consequently, the overall parsing process is also divided into two stages. The first stage involves parsing the LFA to determine: 1) the computing granularity and order, as well as the buffer occupancy of tensors reused on-chip, and 2) all tensors requiring DRAM interaction. The second 

536 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

stage involves parsing the DLSA to derive the specific timing and buffer occupancy details for all tensors that must be loaded/stored from/to DRAM. 

_1) Layer-Fusion-related Attributes:_ LFA includes _**Computing Order, Fine-grained Layer-fusion Cut (FLC), Tiling Number,**_ and _**DRAM Cut**_ , which we will discuss in sequence. 

The first attribute, Computing Order, arranges these layers into a serial sequence, representing their coarse-grained execution order. Thus, a valid Computing Order cannot have any dependency that goes from right to left, as this would result in a scenario where the data needed by a layer calculated earlier is not yet computed. We can see an example of the Computing Order in Fig. 4; swapping the order of _D_ and _E_ remains valid, but swapping _A_ and _B_ does not. 

The second attribute, Fine-grained Layer-fusion Cut (FLC), records the cut locations, which cut layer sequences into Fine-grained Layer-fusion Groups (FLGs). All FLCs together constitute the FLC Set ( _{_ 1 _,_ 2 _}_ in Fig. 4). Each FLG possesses an attribute, Tiling Number (the third attribute), which determines the computing granularity of the FLG. Layers within an FLG are processed continuously at the granularity of the tile rather than completely finishing the first layer before starting the second, thereby saving on-chip buffer overhead. For the example in the left top of Fig. 4, the layers are cut into three FLGs ([ _A_ ] _,_ [ _B_ ] _,_ [ _C, E, D_ ]) with Tiling Numbers 2 _,_ 1 _,_ 2, respectively. Within the FLG [ _C, D, E_ ], each layer is divided into two tiles, which are then processed sequentially. Given the Tiling Number, we use a heuristic strategy to partition each layer into computing tiles along the multiple dimensions. Specifically, it prioritizes tiling the batch dimension since it does not produce halo overlap costs, followed by the height and width of the ofmaps, keeping them as equal as possible to reduce overlap. The reason for not splitting the channel dimension is that splitting the channel would prevent the next layer from accessing all channels, making it impossible to fuse more than two layers. For example, in Fig.2, with a batch size of 1 and a Tiling Number of 4, we split the height and width dimensions each by 2 (the channel dimension is not shown). It is worth emphasizing that for intermediate layers involving operations like convolutions or poolings, which produce halo overlaps, the size of each tile may be larger than 1 _/Tile_ of the size of the fmaps (e.g., _LayerA_ and _B_ in Fig. 2). The specific method for determining each tile’s size of intermediate layers considering halo overlap influence has already been proposed in Cocco [49] and DeFENIS [37], so we directly adopt their methods. The trade-off associated with the Tiling Number primarily concerns the balance between buffer usage, halo overlap overhead, and computing efficiency within the core array. The finer-grained tiles imply reduced buffer usage, but if they involve layers that produce overlaps, they may incur more extra computation and memory accesses. Additionally, the computing efficiency within the core array may decrease, as smaller tiles imply reduced on-chip reuse opportunities [57]. 

By parsing the above three attributes, we can derive the entire computation sequence (the COMPUTE row in Fig. 4). We define certain specific FLCs as DRAM Cuts, which 

cut layer sequences Layer-fusion Groups (LGs). Dependencies between different LGs require data to be sent to DRAM and then loaded back for computation. All DRAM Cuts together constitute the DRAM Cut Set ( _{_ 2 _}_ in Fig. 4). For example, _C_ depends on _B_ , and there is a DRAM cut between them. Therefore, _C_ can only load ifmaps from DRAM after _B_ has sent its ofmaps to DRAM. Thus, all the requests for each tile’s interaction with DRAM can be determined as follows: if a layer has weights, it has a weight-load request. For example, all layers except _C_ (pooling layer has no weights) require their weights to be loaded from DRAM (represented as _WM_ in the left part of Fig. 4). If a layer has forward dependencies that span different LGs (or its input is the overall network input), then the ifmaps of its related tiles need to be loaded from DRAM ( _IMi_ in the left part of Fig. 4). If it has backward dependencies that span different LGs (or its output is the overall network output), then the ofmaps of its related tiles need to be written back to DRAM ( _OMi_ in the left part of Fig. 4). The trade-off involved in DRAM Cuts primarily concerns the balance between buffer requirements and the volume of DRAM access. Generally speaking, the more fused layers there are (the fewer the DRAM Cuts), the lower the number of DRAM accesses will be, but the demand for buffer capacity will increase. 

The remaining fmaps corresponding to the dependencies that do not cross DRAM cut can be directly reused on-chip. The buffer occupancy duration for such data ranges from the production of the ofmaps tile to the consumption of the tile. For the example in Fig. 4(a), there is an FLC between _A_ and _B_ , thus the ofmaps of _A_ are stored on-chip. Fig. 4(a) only shows the buffer allocation for on-chip data transfers, while all tensors related to DRAM communication are not displayed (to determine in the DRAM Load & Store Phase). 

Since the DRAM Cut Set is a subset of the FLC Set, each LG contains one or more FLGs. When data dependencies exist between different FLGs, the producing layers in the former FLG need to aggregate the ofmaps from different tiles of the layers before they can be used by the consuming layers in the latter FLG. Once the computation for an FLG is completed, the weights of the layers in the FLG can be released, freeing up some buffer space. For the example in Fig. 4, the weights of _A_ can be released after all its tiles are computed. _B_ must wait until _A_ to finish computing all its ofmaps before it can start its computation. The above analysis demonstrates that the FLC primarily involves a trade-off in buffer occupation. Specifically, the FLC can free up some buffer space occupied by weights at the cost of fmaps accumulation. 

_2) DRAM-Load-and-Store-related Attributes:_ DLSA includes _**DRAM Tensor Order**_ and _**Living Duration**_ , which will be introduced in serial. 

The DRAM Tensor Order attribute determines the access sequence for all DRAM tensors, such as the tensors in dashed squares which is in the middle of Fig. 4 Left. 

Every DRAM tensor has an adjustable Living Duration attribute, which is a 2-element tuple ( _Start, End_ ). _Start_ and _End_ are tile IDs. This attribute has dual implications: first, 

537 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

within this time frame, the required buffer is allocated to this tensor. For the example in Fig. 4, the Living Duration of _WE_ is ( _B, D_ 2). Therefore, _WE_ lives in buffer from Tile _B_ to _D_ 2. Second, it schedules the timing for loading and storing. Specifically, for ifmaps and weights, the _End_ is fixed at the next tile of the last tile that requires them, indicating when they can be released. The _Start_ indicates when this tensor can start being loaded from DRAM. For ofmaps, the _Start_ is fixed at the tile that produces them, indicating that once this tile finishes, storing to DRAM can begin. The _End_ signifies the tile by which the transfer must be completed; otherwise, the tile should stall. For example, the _End_ of _OE_ 1 is _D_ 1, so if its store is not completed, _D_ 1 cannot start. Although _Start_ indicates when loading or storing can begin, the actual start time depends on whether the required data is ready and whether the loads and stores preceding this tensor in the DRAM Tensor Order have been completed. For example, the _Start_ of _IC_ 2 is _C_ 1, but the loading of the preceding _WD_ takes a long time, causing the actual load of _IC_ 2 to begin only in the middle of _E_ 1. 

## _B. Space Size Comparison_ 

As introduced above, each scheme is encoded by six variable attributes. Thus, our Tensor-centric Notation constructs a sixdimensional vast optimization space. In contrast, if we map the representable scheduling schemes of the SOTA Cocco [49] into our notation, only Computing Order and DRAM Cut can change, with the FLC Set being identical to the DRAM Cut Set. The other four attributes are determined by heuristic strategies, either explicitly or implicitly, making its explorable optimization space much smaller than ours. Moreover, DeFENIS [37] proposes a simulator and demonstrates the performance and energy efficiency of: 1) different LGs with heuristically determined Tile Numbers, and 2) the same LGs with Tile Numbers enumerated at certain intervals. They analyze some characteristics of these two attributes individually. However, they do not jointly explore the space composed of these two attributes, let alone the other four dimensions. As a result, the schemes it touches are far fewer than ours. 

## V. SOMA FRAMEWORK 

## _A. SoMa Overview_ 

As shown in Fig. 5, SoMa is an end-to-end DNN scheduling framework. SoMa takes as inputs: 1) hardware configuration, such as the number and organization of cores, DRAM bandwidth, buffer size, etc.; 2) framework configurations, such as the optimization goal and searching hyperparameters, etc.; 3) a DNN model description file generated by high-level frameworks like PyTorch After scheduling, SoMa outputs: 1) instructions; 2) reports on energy costs and latency; 3) a detailed scheduling scheme. 

The optimization objective of SoMa is _Energy[n] ×Delay[m]_ , where _n_ and _m_ are adjustable parameters that prioritize energy efficiency or performance according to different needs. _Energy_ and _Delay_ refer to the total energy costs and latency for processing a batch of samples, respectively. Thus, small 

**==> picture [218 x 263] intentionally omitted <==**

**----- Start of picture text -----**<br>
Hardware Framework DNN<br>INPUT<br>Parameters Configs Model<br>LFA EXPLORATION STAGE<br>Model Buffer  SA<br>Parser Budget Controller LFA Heuristic<br>Encode DLSA<br>LFA Gen Strategy<br>Complete<br>Encode<br>Cost &<br>Explored<br>Buffer scheme Cost Analyzer<br>Allocator<br>Cost &  Explored Analyzed<br>Explored  Results Scheme Cost<br>scheme<br>SA Evaluator<br>Controller<br>DRAM<br>IR DLSA Gen Access<br>Generator CompleteEncode Cost Analyzed Compute<br>Scheme<br>Analyzer Array<br>Instruction Cost Scheduler<br>Generator DLSA EXPLORATION STAGE &Evaluator<br>Energy&Latency Schedule<br>Instructions OUTPUT<br>Report Scheme<br>**----- End of picture text -----**<br>


Fig. 5. SoMa Framework 

batch sizes can evaluate SoMa’s effectiveness in latencycentric scenarios, while larger batch sizes can be used for throughput-centric scenarios. 

Fig. 5 shows that the key exploration process is governed by three main components: the Buffer Allocator, the LFA Exploration Stage, and the DLSA Exploration Stage. The Buffer Allocator controls the outermost iteration, each involving a complete two-stage top-down exploration. Based on the respective effects and buffer usage of the two stages in the previous iteration, the Buffer Allocator adjusts the buffer budget for the two stages competing for buffer usage in the current iteration. The two-stage top-down exploration involves separately varying and searching LFA and DLSA in each stage, with each stage using SA for independent searches. 

The reason for using a two-stage search with a Buffer Allocator is that: 1) as analyzed in Sec. III-C, a single change in LFA can significantly impact DRAM tensors (e.g., adding or removing a DRAM Cut or changing the Tiling Number), thus previously optimized DLSA attributes may become suboptimal or even invalid. This makes it difficult to retain good solutions obtained by varying DLSA attributes while continuously varying LFA. 2) The two-stage exploration with a Buffer Allocator can produce strong synergistic effects. DRAM access has a significant impact on performance and energy costs. Therefore, we observe that the first stage’s optimization tends to minimize fmaps-related DRAM access, which reduces the optimization difficulty and increases the potential for the second stage optimization. This is mainly because a) the large proportion of weights is beneficial for optimization, as weights have fewer dependency constraints 

538 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

and can be adjusted more freely, while fmaps have more complex dependencies with a narrower range of adjustments; b) more extensive layer fusion and reduced DRAM access result in purer compute time, providing greater optimization opportunities for the second stage. Therefore, we found that dividing the optimization into two stages can produce good synergistic effects. The only risk is that the first stage might occupy too much buffer space, excessively limiting the optimization space for the second stage. We address this potential risk through the Buffer Allocator. 

The optimal results can be directly output from the Buffer Allocator as scheduling schemes and reports, or they can be converted into a more easily parsable intermediate representation (IR) through our IR Generator module. This IR can then be fed into the Instruction Generator module to generate actual instructions. 

## _B. Buffer Allocator_ 

In the first iteration, we conduct a complete two-stage search, with the only constraint being that the buffer usage does not exceed the hardware buffer capacity. We record the maximum buffer usage of the scheme explored in the first stage ( _Buffermax_ ), as well as the best overall encoding scheme and its cost ( _Costbest_ ). In subsequent iterations, the buffer usage limit for the first stage is reduced by _a_ % (10% used in the following experiments) of _Buffermax_ each time (solutions exceeding this limit are deemed invalid), and the overall cost ( _Costtemp_ ) is recorded. If _Costtemp_ is better than _Costbest_ , _Costbest_ and the optimal encoding scheme are updated accordingly. The iteration stops when the costs of the optimal solutions found in two consecutive iterations do not exceed _Costbest_ . The rationale behind using this iteration to allocate buffers between the two stages for overall optimization is that while the performance of both stages improves with increased buffer size, the rate of improvement slows as buffer size grows. Therefore, adjusting the buffer allocation in small increments helps effectively find the sweet spot that maximizes the combined performance of both stages. 

## _C. LFA & DLSA Exploration Stage_ 

Both LFA and DLSA employ SA to explore this space. The key factors in SA are the initial solution, cooling schedule, and operators. The initial solution and operators are discussed in the following sections, while the cooling schedule is described here. Starting from an initial solution, each iteration randomly selects an operation to modify the encoding and evaluates it. If the new scheme’s cost ( _c[′]_ ) is higher than the previous _c−c[′]_ cost ( _c_ ), it is accepted with probability _p_ = _e cTn_ , where _Tn_ is the temperature at iteration _n_ . Otherwise, the scheme is always accepted. The temperature at iteration _n_ is given by _Tn_ = _T_ 0 1+1 _−αN[n] N[n]_[,][where] _[T]_[0][is][the][initial][temperature][and] _[α]_[is] the cooling rate. The total number of iterations is _N_ = _βX_ . For the first stage, _β_ and _X_ are set to 100 and the number of layers, respectively. For the second stage, they are set to 1000 and the number of DRAM tensors, respectively. We also support setting an additional termination time. Once this 

time is reached, the algorithm performs _Y_ more iterations, accepting only improved solutions. 

_1) LFA Exploration Stage:_ In this stage, The initial solution consists of each layer forming its own independent LG and FLG (e.g., both _FLG_ and _LG_ are 1 _,_ 2 _,_ 3 _,_ 4 as shown in Fig. 4), meaning no fusion is applied. The tile number is set to the minimum granularity, corresponding to the size required for the core array to perform parallel computation. Then, the SA operators transform the LFA, while the DLSA is determined using a classical double-buffer strategy (as introduced in Sec. III-C). The specific operators are as follows: 

**Change Computing Order** : Randomly select a layer and change its order to another valid location. **Change Tiling Number** : Randomly select an FLG and multiply or divide its Tiling Number by 2. 

**Add/Delete An FLC** : Randomly add or delete an element in FLC Set. Specifically, adding an FLC means cutting an FLG into two FLGs with the same Tiling Number attribute as the original FLG. Removing an FLC means merging two FLGs into one, with the new FLG’s Tiling Number inherited probabilistically based on the layer count ratio of the original two FLGs. 

**Add/Delete A DRAM Cut** : Randomly add or delete an element in the DRAM Cut Set. The added element must be in the FLC Set. 

_2) DLSA Exploration Stage:_ In this stage, the initial solution adopts the best scheme explored by the previous stage, with the LFA attribute remaining constant. The SA controller then primarily focuses on searching within the DLSA for the DRAM tensors corresponding to this selected LFA. The specific operators are introduced as follows: 

**Change DRAM Tensor Order** : Randomly select a DRAM tensor and change its order to another valid location. 

**Change Living Duration** : Randomly select a DRAM tensor and randomly change its _Start_ (for ifmaps and weights) or _End_ (for ofmaps). For example, in Fig. 4(b), by reducing the _Start_ of _WB_ by 1 from _B_ to _A_ 2, the STALL between _A_ 2 and _B_ can be eliminated, and _WB_ will be included in the buffer associated with _A_ 2. 

Notably, in each operation, the probability of selecting a DRAM tensor is proportional to its size since larger tensors generally have a greater impact on performance and buffer utilization, warranting more transformation opportunities. _D. Evaluator_ 

In this section, we introduce an accurate evaluator, capable of evaluating various scheduling schemes, described using our Tensor-centric Notation across different hardware configurations in terms of energy cost and latency. 

The evaluation process follows a local-to-global approach, first assessing each computing tile and DRAM load/store request (DRAM tensor) individually and then conducting an overall assessment. 

For each computing tile (e.g., _A_ 1 in Fig. 4), the ifmaps and weights have been prefetched into the GBUF, and the ofmaps are written back to the GBUF. From a macro perspective, the Core Array Scheduler explores how to further 

539 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

divide this tile into sub-tiles for computation by each core (as introduced in Sec. II), aiming to maximize parallelism and data reuse. The corresponding Evaluator assesses each interaction between GBUF and L0 buffers, as well as computations, while accounting for dependencies to evaluate overall performance and energy consumption. The corresponding energy costs and computing time of the searched optimal scheme are taken as the tile’s energy costs and computing time. As this area of research is well-established [20], [21], [28], [32], [36], [42], [52], [55], [57], we adopt a classic scheduler and evaluator for this purpose [32], [42]. 

Each DRAM communication tensor’s energy costs are calculated by summing the products of read and write data volumes with their respective unit energy costs. The read and write times are calculated by dividing the data volumes by the respective bandwidths. 

The total energy cost is calculated by summing up the energy costs of the above sub-components, similar to existing classical works [32], [42], [57]. The total computing time is derived based on the evaluated times of all computing tiles and DRAM tensors using the following method: 

For each DRAM tensor, it can start execution only when the following three conditions are met: 1) the preceding DRAM tensor has been completed; 2) for ifmaps or weights, their _Start_ must be smaller than or equal to the current tile ID; and 3) for ofmaps, it must wait until its generating computing tile has finished. For example, in Fig. 4, although _IC_ 2’s _Start_ is _C_ 1, the preceding DRAM tensor ( _WD_ ) is not completed until _E_ 1, so it can only start at the middle of _E_ 1. Moreover, _WB_ has a _Start_ of _B_ . Although the previous DRAM tensor ( _IA_ 2) has already been completed, it must still wait until _A_ 2 finishes before it can begin. 

Each computing tile can start execution only when the following conditions are met: 1) all required data (ifmaps, weights, etc.) are ready. For the example in Fig. 2, _A_ 1, _B_ 1, and _C_ 1 cannot follow their respective preceding computing tile immediately because the required data are not ready at the end of the preceding computing tile. 2) All DRAM tensors with _End_ less than or equal to the tile must be completed. For the instance in Fig. 4, _D_ 1 cannot follow _E_ 1 because _OE_ 1’s _End_ is _D_ 1, and _D_ 1 cannot start until _OE_ 1 has finished execution. 

## _E. Generality and Portability of SoMa_ 

The proposed encoding and SoMa framework exhibit excellent generality for two main reasons: 1) the template depicted in Fig. 1 is highly general, encompassing many accelerators from both the industry [9], [25], [26], [34], [53] and academia [12], [17], [30]; 2) The hardware behavior information encoded by our notation is very general (i.e., computing, data loading, and storing, instruction dependency, etc.). 

Our SoMa also possesses excellent portability, due not only to the aforementioned reasons but also because our framework is designed with a robust modular architecture. This design allows for easy adaptation to different accelerators, which may have distinct core micro-architectures, by simply replacing the relevant Core Array Scheduler & Evaluator modules and 

Instruction Generation module. We have developed a comprehensive compilation flow for our accelerator [1], which can serve as a concrete example for porting to other accelerators. 

## _F. Open-source of SoMa_ 

We have now open-sourced files and documentation for key stages to illustrate the entire SoMa Compiler workflow at this link [1]. In the future, we plan to set up a small-scale cloud platform. This platform will allow users not only to access the open-sourced SoMa scheduler but also to modify or even replace our scheduler (as long as the output is converted to IR format), enabling it to be translated into instructions that can run on the chip. Given that many existing accelerators (e.g., TPU [26]) do not offer such low-level API access, we believe this platform can significantly advance related research. 

## VI. EVALUATION 

## _A. Experiment Setup_ 

_1) Hardware Configuration:_ To comprehensively evaluate the effects of SoMa, we utilize both edge and cloud hardware platforms: we set the edge computing power to 16 TOPS (referencing Qualcomm Snapdragon 8 Gen 3’s 15 TOPS [44] and Apple A15’s 15.8 TOPS [4], A16’s 17 TOPS [5]), and the cloud to 128 TOPS (referencing NVIDIA Orin’s 138 TOPS [39] and TPU V4i’s 136 TOPS [25]). Buffer and DRAM bandwidth are set to 8MB and 32MB, and 16GB/s and 128GB/s, respectively, based on our DSE results in Fig. 7. This configuration allows accelerators to achieve outstanding performance with reasonable hardware resources. We also thoroughly tested SoMa’s performance under different buffer sizes and DRAM bandwidths (as shown in Fig. 7). The default process technology is TSMC 12nm, with an operating frequency of 1GHz. All unit energy costs for different operations required by the Evaluator (as introduced in Sec. V-D) are obtained from the actual RTL code synthesis and simulation during our accelerator development. The optimization goal is set as Energy[1] _×_ Delay[1] , as both energy costs and performance are crucial in inference accelerators [16], [22]. 

_2) Workload:_ To comprehensively evaluate SoMa effects and analyze the trade-offs behind different DRAM communication scheduling schemes, we scale batch size from 1 to 64, covering latency-sensitive scenarios to throughputcentric scenarios as introduced in Sec. V-A. In our experiments, ResNet-50 [19], ResNet-101 [19], Inception-ResNetv1 (IRes) [47], Randwire [56] and GPT-2 [45] are chosen as workloads. ResNet-50 is chosen because it takes classical residual structures widely employed in many DNNs. ResNet101 is chosen because it has a similar structure to ResNet-50 but with a larger number of layers. The Inception-ResNet-v1 and Randwire are selected to represent the DNNs with wider and more complex structures. GPT-2 is selected to represent language-processing DNNs. Since GPT-2 has various versions for different scenarios, we use GPT-2-Small with a token length of 512 (prefill 512 and decode the 513th) for the edge platform, and GPT-2-XL with a token length of 1024 (prefill 1024 and decode the 1025th) for the cloud platform. 

540 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [488 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
Core Array Energy DRAM Energy Computing Resources Utilization (Performance) Average Buffer Utilization Theoretical Maximum Computing Resources Utilization<br>1 1<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0 0<br>1 1<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0 0<br>BS = 1 BS = 4 BS = 16 BS = 64 BS = 1 BS = 4 BS = 16 BS = 64 BS = 1 BS = 4 BS = 16 BS = 64 BS = 1 BS = 4 BS = 16 BS = 64 BS = 1 BS = 4 BS = 16 BS = 64 BS = 1 BS = 4 BS = 16 BS = 64<br>ResNet-50 ResNet-101 Inception-ResNet-v1 RandWire GPT2-Prefill GPT2-Decode<br>Edge<br>Normalized  Energy cost Utilization<br>Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2<br>Cloud<br>Normalized  Energy cost Utilization<br>Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2 Cocco Ours_1 Ours_2<br>**----- End of picture text -----**<br>


Fig. 6. Overall Comparisons between Cocco and SoMa (Ours). Ours 1 and Ours 2 demonstrate how the efficient solution is progressively optimized through the first and second stages. **Computing Resources Utilization** equals to _Util_ (Actual Evaluated Latency), where _Util_ ( _t_ ) = Total Number of Operations in Network _/_ (Peak Hardware Computing Power _×_ t). Thus, _the utilization can be viewed as a measure of_ _**performance**_ . **Average Buffer Usage** is calculated as[�] _T ile_[(][Buffer][Usage] _[×]_[Computing] Time _T ile_ ) _/_ Total Comp Time. The **Theoretical Maximum Computing Resources Utilization** (represented by Blue Diamonds) is used to measure the theoretical maximum optimization possible in the second stage. Specifically, it is _Util_ ( _MIN {_[�] _T ile_[Computing] Time _T ile,_[�] _T ensor_[DRAM] Access Time _T ensor}_ ), without considering dependencies (e.g., _Util_ ( _MIN {_ ( _IA_ 1 + _..._ + _OC_ 4) _,_ ( _A_ 1 + _..._ + _C_ 4) _}_ ) in Fig. 2). This means the theoretical optimization upper limit in the second stage is achieved when all DRAM tensors or computing tiles are continuously executed without stall. BS represents batch size. For the edge platform, the workload is GPT-2-Small, while for the cloud platform, it is GPT-2-XL. 

ResNet-50 and GPT-2 are chosen as the default workloads for the discussion due to their representativeness and popularity in image and language processing, respectively. 

_3) Baseline:_ We select the SOTA Cocco scheduling framework [49] as our baseline, as it explores the layer-fusion space while adopting mainstream tiling and prefetch-and-delayedstoring strategies. 

## _B. Overall Comparisons_ 

Fig. 6 shows that after the first stage, SoMa, on average, improves 1.82 _×_ performance and reduces 37.3% energy costs compared to Cocco. The second stage, on average, further improves performance by 1.16 _×_ over the first stage. It can be observed that the schemes found in the second stage are very close to the theoretical maximum optimization value (blue diamonds), with an average difference of only 3.1%. Thus, after optimization through the two stages, the bestexplored solution improves 2.11 _×_ performance and reduces 37.3% energy costs. 

SoMa improves performance by 2.15 _×_ , 2.18 _×_ , 2.01 _×_ , 2.61 _×_ , 2.55 _×_ and 1.14 _×_ and reduces energy cost by 39.5%, 41.0%, 46.0%, 47.1%, 47.0%, and 3.1% on ResNet-50, ResNet-101, Inception-ResNet-v1, RandWire, GPT2-Prefill, and GPT2-Decode compared to Cocco, respectively. This demonstrates that SoMa can consistently improve performance and energy efficiency on various DNNs. **However, compared to other networks, including GPT-2-Prefill, SoMa demonstrates almost no optimization effect in GPT-2-Decode, and the overall computational utilization remains extremely low. This is because the decode stage exhibits significantly lower compute density [43], with its latency primarily dominated by weight and KV cache loading. Moreover, another interesting phenomenon is that the computational utilization of GPT-2 does not increase linearly with the batch size; instead, the growth rate gradually diminishes. For instance, after SoMa optimization, the utilization rates** 

**for GPT-2-Small-Decode and GPT-2-XL-Decode across batch sizes of 1, 4, 16, and 64 are 0.66%, 2.03%, 4.26%, 5.84% and 0.60%, 1.90%, 4.13%, 5.83%, respectively. This is because, as the batch size grows, the increasing size of the KV cache becomes comparable to or even exceeds that of the weights, diminishing the benefits of further increases in batch size for improving compute density.** 

_1) Analysis of the First Stage of SoMa:_ In the first stage of SoMa, we observe significant reductions in Core Array Energy and DRAM Energy by 34.8% and 44.3%, respectively. This reduction in Core Array Energy is mainly due to the flexibility of our approach in adjusting the Tiling Number during exploration, unlike Cocco’s more conservative approach that selects each tile size based only on the basic parallelism requirements of the computing units [49] (detailed analysis and example can be found in Sec. VII-B1). Therefore, our approach often has fewer tiles within the buffer constraint (with an average total number of computing tiles per network being 7962 for Cocco and 751 for ours), with each tile being larger, allowing more optimization and reuse opportunities in the Core Array Scheduler. The decrease in DRAM Energy is because SoMa allows for a smaller number of LGs per network (an average of 2.5) compared to Cocco (an average of 13.0), meaning it can fuse more layers. The main reasons for this difference are: 1) Cocco’s finer-grained tiles result in significant accumulated backtracking halo overlap costs when dealing with convolutional and pooling layers (as introduced in Sec. III-A and Fig. 2). 2) Our approach can use FLCs (with an average of 3.9 FLGs per network) instead of DRAM Cuts to free some weights (as introduced in Sec. IV-A1) and adjust the Tiling Number. Specifically, shuffling weights can save buffer space, enabling the fusion of more layers. Moreover, since different FLGs can have different Tiling Numbers, the FLC can switch the computing granularity between two FLGs without accessing DRAM (detailed analysis in Sec. VII-B1). 

541 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [488 x 298] intentionally omitted <==**

Fig. 7. Design Space Exploration over DRAM Bandwidth and Buffer Size for the 16TOPS Edge DNN Accelerator. We present the latency corresponding to the optimal schemes explored using Cocco and SoMa (Ours) under different networks, batch sizes, DRAM bandwidth, and buffer sizes. In the results of Ours, we highlighted the hardware configurations with the same minimum value (accurate to two decimal places) using a red envelope curve. 

The performance improvement also largely stems from the above-analyzed factors, as coarser-grained tiles provide more on-chip reuse opportunities, better overlapping of computation, and GBUF access time, while Cocco may sometimes suffer from GBUF load and store stalls. Additionally, the significant reduction in DRAM accesses naturally reduces overall latency. 

_2) Analysis of the Second Stage of SoMa:_ The performance improvement brought by the second stage mainly comes from effectively exploiting the DRAM usage imbalance opportunities described in Sec. III-B. Specifically, by intelligently prefetching and delaying storing, many DRAM idle periods can be utilized, reducing computation stalls and DRAM bandwidth waste. In Sec. VII-B2, we demonstrate this optimization through a simple practical example. 

## VII. DISCUSSION 

_A. Design Space Exploration_ 

In Fig. 7, we conduct a DSE for a 16 TOPS accelerator across different workloads and batch sizes, focusing on buffer size and DRAM bandwidth. This exploration yields some interesting insights as follows. 

_1) Insight 1:_ **When dealing with small batch sizes (especially 1), even with SoMa, DRAM bandwidth plays a more decisive role in performance compared to buffer** 

**size. However, the impact of buffer size becomes more significant as the batch size increases.** 

For small batch sizes (e.g., Ours-Batch size = 1 in Fig. 7), the greater influence of DRAM bandwidth is evident as increasing buffer size does not significantly reduce latency, even with SoMa. However, increasing DRAM bandwidth results in a noticeable reduction in latency. This is because: 1) the fmaps sizes are small, so a minimal buffer can almost entirely store data on-chip, making further increases in buffer size unnecessary; 2) weights need to be loaded regardless, but a small batch size means less computation, providing fewer opportunities for SoMa to use prefetching and delayed storing to hide data transfer times. Consequently, weight loading times can severely stall computation and dominate latency. Thus, increasing DRAM bandwidth can directly reduce this latency and significantly enhance overall performance. 

As the batch size increases, the impact of buffer size becomes more significant, as evidenced by the greater reduction in latency with larger batch sizes when increasing buffer size (e.g., Ours-Batch size = 4 and 16 in Fig. 7). This is because: 1) the fmaps sizes increase, allowing a larger buffer to better exploit reuse opportunities; 2) additionally, with more computation, the longer computing times provide more opportunities for SoMa to use the buffer for prefetching and delayed storing to hide data transfer times. 

542 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [475 x 381] intentionally omitted <==**

Fig. 8. Comparison of Practical Execution Graphs among the Schemes Explored by Cocco (top), the First Stage (middle), and the Second Stage (bottom) of SoMa on default edge-side accelerator. We also point out all DRAM Cuts, FLCs, and their corresponding Tiling Numbers. In the running graph for one block of GPT-2-XL-Prefill, we have highlighted the main matrix multiplication layers, while certain element-wise layers (such as transpose, softmax, add, and layer normalization) are not explicitly marked. Q, K, and V represents Query, Key, and Value, respectively. 

## _2) Insight 2:_ **Having both a large buffer and high DRAM bandwidth is often wasteful and offers limited performance gains compared to configurations with either high DRAM bandwidth or moderate DRAM bandwidth paired with a large buffer.** 

In Ours in Fig. 7, we observe that many configurations in the lower right corner exhibit similar performance (highlighted by a red envelope curve). Among these points, the ones in the farthest lower right corner are equipped with both the large DRAM bandwidth and buffer sizes, which is quite wasteful. Specifically, we find that for all workloads, there is a noticeable point where increasing DRAM bandwidth beyond a certain threshold yields diminishing returns, as the dominant factor for latency becomes computing time. This lower triangle (red envelope curve) in SoMa, a feature not observed in Cocco, indicates that with the efficient buffer utilization in SoMa, onchip buffers can adequately compensate for DRAM bandwidth. 

While HBM provides high bandwidth density in the current era, its high cost deters many companies [15], [50]. Therefore, moderately increasing buffer size and opting for lower-cost DRAM models could be a viable differentiation strategy, and SoMa can be a significant aid for this. 

## _B. Lessons From a Practical Example_ 

Fig. 8 shows the actual execution graph of scheduling schemes explored by Cocco, SoMa’s first stage, and SoMa’s second stage, respectively. For ResNet-50, compared to the baseline, the first stage of SoMa achieves an average of 1.57 _×_ performance improvement and a 36.1% reduction in energy cost. The second stage achieves an additional average of 1.25 _×_ performance improvement over Stage 1, resulting in a total performance gain of 1.96 _×_ over Cocco. Next, we will analyze the reasons for SoMa’s performance and energy efficiency improvements using this practical example, thereby enhancing 

543 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

the understanding of the DRAM Communication Scheduling Space. 

_1) Lessons From the First Stage:_ From this example, we can make two observations: 1) **adaptively adjusting the Tiling Number based on network characteristics and buffer size is crucial for exploiting buffer potential and improving performance; 2) FLC can free weights and aggregate fmaps to adjust the Tiling Number without accessing DDR, which can further enhance buffer utilization efficiency and leverage the potential of dynamically adjusting the Tiling Number.** Next, we will analyze the origins of these two insightful observations in detail, using the example in Fig. 8 as a reference. 

First, by comparing the Tiling Numbers in Cocco and the first stage of SoMa, we find that Cocco’s Tiling Number is larger, while SoMa’s Tiling Number can be as low as 1. This is mainly because Cocco uses a heuristic strategy to set the Tiling Number based on the Core Array’s parallelism requirements (Kernel-Channel (KC) parallelism [25], [26], [34], [38], [40], [50]), a common approach in many studies [58], [59]. Consequently, larger kernel and channel dimensions result in a higher Tiling Number. For example, in ResNet-50, the Tiling Number (16) for later layers with larger kernel and channel dimensions is higher than that of earlier layers (8). However, this strategy is often conservative, leading to higher Tiling Numbers. A larger Tiling Number increases backtracking halo overlap costs and limits the Core Array Scheduler’s ability to find on-chip reuse opportunities, which reduces performance and energy efficiency. In contrast, SoMa better adapts to buffer size and network structure to control the Tiling Number. For example, in ResNet-50, the fmap size decreases as the network deepens, so the buffer consumption for a smaller Tiling Number also decreases accordingly. Thus, SoMa intelligently allows the Tiling Number to decrease monotonically with network depth (from left to right), rather than increasing as in Cocco. Additionally, we observe that in SoMa, the last LG of ResNet-50 and the first FLG of GPT2-XL-Prefill both have a Tiling Number of 1, allowing for the immediate disposal of weights after processing each tile (or layer). This conserves buffer space and facilitates further fusion. 

Second, in both ResNet-50 and GPT-2-XL-Prefill, it happens that the first two LGs in the Cocco scheme and the first two FLGs in the SoMa scheme consist of identical layers. We observe that SoMa efficiently uses the FLC to clear weights and adjust tile sizes without incurring the DRAM access overheads seen in the Cocco scheme. Specifically, in ResNet50, if the Tiling Number for the first FLG equals 2 (matching the next FLG), the buffer is insufficient for the first FLG. In GPT-2-XL-Prefill, only when the Tiling Number for the first FLG is 1 can it promptly clear weights to accommodate Q, K, and V on chip. 

_2) Lessons From the Second Stage:_ The improvement of the second stage is attributed to the clever adjustment of the access order and living duration of DRAM tensors. In this stage, SoMa properly performs precise surgical strikes on some key 

tensors to reduce the total computing stall. 

It can be observed that in both examples, SoMa’s second stage effectively prefetches and delays storing DRAM tensors. ResNet-50, however, has more DRAM tensors, making adjustments more complex; therefore, we primarily use it as the example for analysis. 

As shown by the black solid arrows near the diagonally striped regions in Fig. 8, DRAM tensors that cause computing stalls are adjusted to DRAM-free periods. More specifically, the weight of the first layer of _LG_ 2 is prefetched several tiles earlier, while the ofmaps of the last layer of _FLG_ 2 is delayed by one tile and swaps positions with the ifmaps of the first layer of _LG_ 2. This adjustment successfully utilizes DRAM idle time to eliminate the computing stall near the border between _FLG_ 2 and _LG_ 2, while considering dependencies. 

Additionally, it can be observed that _LG_ 2 contains many chunky weights (with the three largest weights up to 2304KB with INT8 precision). Given the limited 8MB buffer, prefetching these weights far in advance is impractical. However, simply pushing all weights forward would not maximally eliminate computing stalls. Therefore, SoMa cleverly selects to move the two relatively larger weight tensors at the front of _LG_ 2 to the DRAM idle time at the back of _FLG_ 2 (marked by dashed black arrows). The remaining weights are not altered in their DRAM Access Order but are pushed forward by changing their Living Duration. This approach maximizes the elimination of computing stalls within _LG_ 2. Although some computing stalls still appear, the maximum buffer usage near these stalls has already reached 8MB, indicating that further prefetching is not feasible. 

## VIII. CONCLUSION 

In this work, we first analyze various existing paradigms that use buffers to optimize DRAM communication, identifying challenges in the current layer fusion paradigm and uncovering previously overlooked opportunities for prefetching and delayed storing. Based on these observations, we introduce a Tensor-centric Notation and a matching parsing method to describe DRAM communication scheduling schemes, thereby defining the DRAM Communication Scheduling Space. We then propose a two-stage scheduling framework, SoMa, to structurally explore this space. Experimental results demonstrate that, compared to the SOTA Cocco framework, SoMa efficiently explores the broader optimization space defined by our notation, fully utilizing the buffer’s potential to optimize DRAM communication. Moreover, we leverage SoMa to conduct several case studies, yielding interesting insights into architecture design and the trade-offs underlying the DRAM Communication Scheduling Space. 

## IX. ACKNOWLEDGMENT 

This research was partially supported by Dushi Program from Tsinghua University. 

544 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] “Soma open-source framework,” https://github.com/SET-SchedulingProject/SoMa-HPCA2025. 

- [2] D. Abts, G. Kimmell, A. Ling, J. Kim, M. Boyd, A. Bitar, S. Parmar, I. Ahmed, R. DiCecco, D. Han, J. Thompson, M. Bye, J. Hwang, J. Fowers, P. Lillian, A. Murthy, E. Mehtabuddin, C. Tekur, T. Sohmers, K. Kang, S. Maresh, and J. Ross, “A software-defined tensor streaming multiprocessor for large-scale machine learning,” in _Proceedings of the 49th Annual International Symposium on Computer Architecture_ , ser. ISCA ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 567–580. [Online]. Available: https://doi.org/10.1145/3470496.3527405 

- [3] M. Alwani, H. Chen, M. Ferdman, and P. A. Milder, “Fused-layer CNN accelerators,” in _49th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2016, Taipei, Taiwan, October 15-19, 2016_ . IEEE Computer Society, 2016, pp. 22:1–22:12. [Online]. Available: https://doi.org/10.1109/MICRO.2016.7783725 

- [4] Apple, “Apple a15,” https://en.wikipedia.org/wiki/Apple A15, 2021. 

- [5] Apple, “Apple a16,” https://en.wikipedia.org/wiki/Apple A16, 2022. 

- [6] P. Bannon, G. Venkataramanan, D. D. Sarma, and E. Talpes, “Computer and redundancy solution for the full self-driving computer,” in _2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, August 18-20, 2019_ . IEEE, 2019, pp. 1–22. [Online]. Available: https://doi.org/10.1109/HOTCHIPS.2019.8875645 

- [7] J. Cai, Y. Wei, Z. Wu, S. Peng, and K. Ma, “Inter-layer scheduling space definition and exploration for tiled accelerators,” in _Proceedings of the 50th Annual International Symposium on Computer Architecture, ISCA 2023, Orlando, FL, USA, June 17-21, 2023_ , Y. Solihin and M. A. Heinrich, Eds. ACM, 2023, pp. 13:1–13:17. [Online]. Available: https://doi.org/10.1145/3579371.3589048 

- [8] J. Cai, Z. Wu, S. Peng, Y. Wei, Z. Tan, G. Shi, M. Gao, and K. Ma, “Gemini: Mapping and architecture co-exploration for large-scale dnn chiplet accelerators,” in _2024 IEEE International Symposium on HighPerformance Computer Architecture (HPCA)_ . IEEE, 2024, pp. 156– 171. 

- [9] Cambricon, “Mlu290-m5,” https://www.cambricon.com/index.php?m= content&c=index&a=lists&catid=340. 

- [10] T. Chen, Z. Du, N. Sun, J. Wang, C. Wu, Y. Chen, and O. Temam, “Diannao: A small-footprint high-throughput accelerator for ubiquitous machine-learning,” in _Proceedings of the 19th International Conference on Architectural Support for Programming Languages and Operating Systems_ , ser. ASPLOS ’14. New York, NY, USA: Association for Computing Machinery, 2014, p. 269–284. [Online]. Available: https://doi.org/10.1145/2541940.2541967 

- [11] Y. Chen, J. S. Emer, and V. Sze, “Eyeriss: A spatial architecture for energy-efficient dataflow for convolutional neural networks,” in _43rd ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2016, Seoul, South Korea, June 18-22, 2016_ . IEEE Computer Society, 2016, pp. 367–379. [Online]. Available: https://doi.org/10.1109/ISCA.2016.40 

- [12] Y. Chen, T. Luo, S. Liu, S. Zhang, L. He, J. Wang, L. Li, T. Chen, Z. Xu, N. Sun, and O. Temam, “Dadiannao: A machine-learning supercomputer,” in _47th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2014, Cambridge, United Kingdom, December 13-17, 2014_ . IEEE Computer Society, 2014, pp. 609–622. [Online]. Available: https://doi.org/10.1109/MICRO.2014.58 

- [13] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in _Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers)_ , J. Burstein, C. Doran, and T. Solorio, Eds. Association for Computational Linguistics, 2019, pp. 4171–4186. [Online]. Available: https://doi.org/10.18653/v1/n19-1423 

- [14] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” 2021. 

- [15] dramexchange, “Dram price,” https://www.dramexchange.com/. 

- [16] A. Firoozshahian, J. Coburn, R. Levenstein, R. Nattoji, A. Kamath, O. Wu, G. Grewal, H. Aepala, B. Jakka, B. Dreyer, A. Hutchin, U. Diril, K. Nair, E. K. Aredestani, M. Schatz, Y. Hao, R. Komuravelli, K. Ho, S. Abu Asal, J. Shajrawi, K. Quinn, N. Sreedhara, P. Kansal, 

   - W. Wei, D. Jayaraman, L. Cheng, P. Chopda, E. Wang, A. Bikumandla, A. Karthik Sengottuvel, K. Thottempudi, A. Narasimha, B. Dodds, C. Gao, J. Zhang, M. Al-Sanabani, A. Zehtabioskuie, J. Fix, H. Yu, R. Li, K. Gondkar, J. Montgomery, M. Tsai, S. Dwarakapuram, S. Desai, N. Avidan, P. Ramani, K. Narayanan, A. Mathews, S. Gopal, M. Naumov, V. Rao, K. Noru, H. Reddy, P. Venkatapuram, and A. Bjorlin, “Mtia: First generation silicon targeting meta’s recommendation systems,” in _Proceedings of the 50th Annual International Symposium on Computer Architecture_ , ser. ISCA ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3579371.3589348 

- [17] M. Gao, J. Pu, X. Yang, M. Horowitz, and C. Kozyrakis, “TETRIS: scalable and efficient neural network acceleration with 3d memory,” in _Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS 2017, Xi’an, China, April 8-12, 2017_ , Y. Chen, O. Temam, and J. Carter, Eds. ACM, 2017, pp. 751–764. [Online]. Available: https://doi.org/10.1145/3037697.3037702 

- [18] L. Gwennap, “Tenstorrent scales ai performance: New multicore architecture leads in data-center power efficiency,” 2020. [Online]. Available: https://www.linleygroup.com/mpr/article.php?id=12287 

- [19] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016_ . IEEE Computer Society, 2016, pp. 770–778. [Online]. Available: https://doi.org/10.1109/CVPR.2016.90 

- [20] K. Hegde, P. Tsai, S. Huang, V. Chandra, A. Parashar, and C. W. Fletcher, “Mind mappings: enabling efficient algorithm-accelerator mapping space search,” in _ASPLOS ’21: 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Virtual Event, USA, April 19-23, 2021_ , T. Sherwood, E. D. Berger, and C. Kozyrakis, Eds. ACM, 2021, pp. 943–958. [Online]. Available: https://doi.org/10.1145/3445814.3446762 

- [21] Q. Huang, A. Kalaiah, M. Kang, J. Demmel, G. Dinh, J. Wawrzynek, T. Norell, and Y. S. Shao, “Cosa: Scheduling by constrained optimization for spatial accelerators,” in _48th ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2021, Valencia, Spain, June 14-18, 2021_ . IEEE, 2021, pp. 554–566. [Online]. Available: https://doi.org/10.1109/ISCA52012.2021.00050 

- [22] INTEL, “Goya inference platform white paper,” https://habana.ai/wpcontent/uploads/2021/10/Habana-GOYA-Inference-PerformanceWhitepaper-Nov20.pdf. 

- [23] M. James, M. Tom, P. Groeneveld, and V. Kibardin, “ISPD 2020 physical mapping of neural networks on a wafer-scale deep learning accelerator,” in _ISPD 2020: International Symposium on Physical Design, Taipei, Taiwan, March 29 - April 1, 2020, delayed to September 20-23, 2020_ , W. Swartz and J. Lienig, Eds. ACM, 2020, pp. 145–149. [Online]. Available: https://doi.org/10.1145/3372780.3380846 

- [24] Y. Jiao, L. Han, R. Jin, Y. Su, C. Ho, L. Yin, Y. Li, L. Chen, Z. Chen, L. Liu, Z. He, Y. Yan, J. He, J. Mao, X. Zai, X. Wu, Y. Zhou, M. Gu, G. Zhu, R. Zhong, W. Lee, P. Chen, Y. Chen, W. Li, D. Xiao, Q. Yan, M. Zhuang, J. Chen, Y. Tian, Y. Lin, W. Wu, H. Li, and Z. Dou, “A 12nm programmable convolution-efficient neuralprocessing-unit chip achieving 825tops,” in _2020 IEEE International Solid- State Circuits Conference, ISSCC 2020, San Francisco, CA, USA, February 16-20, 2020_ . IEEE, 2020, pp. 136–140. [Online]. Available: https://doi.org/10.1109/ISSCC19947.2020.9062984 

- [25] N. P. Jouppi, D. H. Yoon, M. Ashcraft, M. Gottscho, T. B. Jablin, G. Kurian, J. Laudon, S. Li, P. C. Ma, X. Ma, T. Norrie, N. Patil, S. Prasad, C. Young, Z. Zhou, and D. A. Patterson, “Ten lessons from three generations shaped google’s tpuv4i : Industrial product,” in _48th ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2021, Valencia, Spain, June 14-18, 2021_ . IEEE, 2021, pp. 1–14. [Online]. Available: https://doi.org/10.1109/ISCA52012.2021.00010 

- [26] N. P. Jouppi, C. Young, N. Patil, D. A. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P. Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, 

545 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter performance analysis of a tensor processing unit,” in _Proceedings of the 44th Annual International Symposium on Computer Architecture, ISCA 2017, Toronto, ON, Canada, June 24-28, 2017_ . ACM, 2017, pp. 1–12. [Online]. Available: https://doi.org/10.1145/3079856.3080246 

- [27] S. Kao, G. Jeong, and T. Krishna, “Confuciux: Autonomous hardware resource assignment for DNN accelerators using reinforcement learning,” in _53rd Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2020, Athens, Greece, October 1721, 2020_ . IEEE, 2020, pp. 622–636. [Online]. Available: https: //doi.org/10.1109/MICRO50266.2020.00058 

- [28] S. Kao and T. Krishna, “GAMMA: automating the HW mapping of DNN models on accelerators via genetic algorithm,” in _IEEE/ACM International Conference On Computer Aided Design, ICCAD 2020, San Diego, CA, USA, November 2-5, 2020_ . IEEE, 2020, pp. 44:1–44:9. [Online]. Available: https://doi.org/10.1145/3400302.3415639 

- [29] S.-C. Kao, S. Subramanian, G. Agrawal, A. Yazdanbakhsh, and T. Krishna, “Flat: An optimized dataflow for mitigating attention bottlenecks,” in _Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ , 2023, pp. 295–310. 

- [30] D. Kim, J. Kung, S. M. Chai, S. Yalamanchili, and S. Mukhopadhyay, “Neurocube: A programmable digital neuromorphic architecture with high-density 3d memory,” in _43rd ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2016, Seoul, South Korea, June 18-22, 2016_ . IEEE Computer Society, 2016, pp. 380–392. [Online]. Available: https://doi.org/10.1109/ISCA.2016.41 

- [31] S. Knowles, “Graphcore,” in _IEEE Hot Chips 33 Symposium, HCS 2021, Palo Alto, CA, USA, August 22-24, 2021_ . IEEE, 2021, pp. 1–25. [Online]. Available: https://doi.org/10.1109/HCS52781.2021.9567075 

- [32] H. Kwon, P. Chatarasi, M. Pellauer, A. Parashar, V. Sarkar, and T. Krishna, “Understanding reuse, performance, and hardware cost of DNN dataflow: A data-centric approach,” in _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2019, Columbus, OH, USA, October 12-16, 2019_ . ACM, 2019, pp. 754–768. [Online]. Available: https://doi.org/10.1145/3352460. 3358252 

- [33] J. Lee, D. Shin, J. Lee, J. Lee, S. Kang, and H. Yoo, “A full HD 60 fps CNN super resolution processor with selective caching based layer fusion for mobile devices,” in _2019 Symposium on VLSI Circuits, Kyoto, Japan, June 9-14, 2019_ . IEEE, 2019, p. 302. [Online]. Available: https://doi.org/10.23919/VLSIC.2019.8778104 

- [34] H. Liao, J. Tu, J. Xia, and X. Zhou, “Davinci: A scalable architecture for neural network computing,” in _2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, August 18-20, 2019_ . IEEE, 2019, pp. 1–44. [Online]. Available: https://doi.org/10.1109/HOTCHIPS.2019.8875654 

- [35] D. Liu, T. Chen, S. Liu, J. Zhou, S. Zhou, O. Teman, X. Feng, X. Zhou, and Y. Chen, “Pudiannao: A polyvalent machine learning accelerator,” _SIGPLAN Not._ , vol. 50, no. 4, p. 369–381, mar 2015. [Online]. Available: https://doi.org/10.1145/2775054.2694358 

- [36] L. Lu, N. Guan, Y. Wang, L. Jia, Z. Luo, J. Yin, J. Cong, and Y. Liang, “TENET: A framework for modeling tensor dataflow based on relation-centric notation,” in _48th ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2021, Valencia, Spain, June 14-18, 2021_ . IEEE, 2021, pp. 720–733. [Online]. Available: https://doi.org/10.1109/ISCA52012.2021.00062 

- [37] L. Mei, K. Goetschalckx, A. Symons, and M. Verhelst, “Defines: Enabling fast exploration of the depth-first scheduling space for dnn accelerators through analytical modeling,” in _2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 2023, pp. 570–583. 

- [38] T. Norrie, N. Patil, D. H. Yoon, G. Kurian, S. Li, J. Laudon, C. Young, N. P. Jouppi, and D. A. Patterson, “Google’s training chips revealed: Tpuv2 and tpuv3,” in _IEEE Hot Chips 32 Symposium, HCS 2020, Palo Alto, CA, USA, August 16-18, 2020_ . IEEE, 2020, pp. 1–70. [Online]. Available: https://doi.org/10.1109/HCS49909.2020.9220735 

- [39] Nvdia, “Hardware for self-driving cars,” https://www.nvidia.com/en-us/ self-driving-cars/drive-platform/hardware/. 

- [40] Nvidia, “Nvdla deep learning accelerator,” http://nvdla.org., 2017. 

- [41] OpenAI, “Gpt-4 technical report,” 2023. 

- [42] A. Parashar, P. Raina, Y. S. Shao, Y. Chen, V. A. Ying, A. Mukkara, R. Venkatesan, B. Khailany, S. W. Keckler, and J. S. Emer, 

“Timeloop: A systematic approach to DNN accelerator evaluation,” in _IEEE International Symposium on Performance Analysis of Systems and Software, ISPASS 2019, Madison, WI, USA, March 24-26, 2019_ . IEEE, 2019, pp. 304–315. [Online]. Available: https://doi.org/10.1109/ISPASS.2019.00042 

- [43] P. Patel, E. Choukse, C. Zhang, A. Shah,[´] I. Goiri, S. Maleki, and R. Bianchini, “Splitwise: Efficient generative llm inference using phase splitting,” in _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 2024, pp. 118–132. 

- [44] Qualcomm, “Qualcomm snapdragon 8gen3,” https://en.wikipedia.org/ wiki/List of Qualcomm Snapdragon systems on chips#Snapdragon 8/8s Gen 3 (2024), 2024. 

- [45] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever _et al._ , “Language models are unsupervised multitask learners,” _OpenAI blog_ , vol. 1, no. 8, p. 9, 2019. 

- [46] Z. Sun, S. Kvatinsky, X. Si, A. Mehonic, Y. Cai, and R. Huang, “A full spectrum of computing-in-memory technologies,” _Nature Electronics_ , pp. 1–13, 2023. 

- [47] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, “Inception-v4, inception-resnet and the impact of residual connections on learning,” in _Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, February 4-9, 2017, San Francisco, California, USA_ , S. P. Singh and S. Markovitch, Eds. AAAI Press, 2017, pp. 4278–4284. [Online]. Available: http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/ view/14806 

- [48] Z. Tan, H. Cai, R. Dong, and K. Ma, “Nn-baton: DNN workload orchestration and chiplet granularity exploration for multichip accelerators,” in _48th ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2021, Valencia, Spain, June 14-18, 2021_ . IEEE, 2021, pp. 1013–1026. [Online]. Available: https://doi.org/10.1109/ISCA52012.2021.00083 

- [49] Z. Tan, Z. Zhu, and K. Ma, “Cocco: Hardware-mapping coexploration towards memory capacity-communication optimization,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1_ , ser. ASPLOS ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 69–84. [Online]. Available: https://doi.org/10.1145/3617232.3624865 

- [50] J. Vasiljevic, L. Bajic, D. Capalija, S. Sokorac, D. Ignjatovic, L. Bajic, M. Trajkovic, I. Hamer, I. Matosevic, A. Cejkov, U. Aydonat, T. Zhou, S. Z. Gilani, A. Paiva, J. Chu, D. Maksimovic, S. A. Chin, Z. Moudallal, A. Rakhmati, S. Nijjar, A. Bhullar, B. Drazic, C. Lee, J. Sun, K. Kwong, J. Connolly, M. Dooley, H. Farooq, J. Y. T. Chen, M. Walker, K. Dabiri, K. Mabee, R. S. Lal, N. Rajatheva, R. Retnamma, S. Karodi, D. Rosen, E. Munoz, A. Lewycky, A. Knezevic, R. Kim, A. Rui, A. Drouillard, and D. Thompson, “Compute substrate for software 2.0,” _IEEE Micro_ , vol. 41, no. 2, pp. 50–55, 2021. [Online]. Available: https://doi.org/10.1109/MM.2021.3061912 

- [51] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in _Proceedings of the 31st International Conference on Neural Information Processing Systems_ , ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 6000–6010. 

- [52] R. Venkatesan, Y. S. Shao, M. Wang, J. Clemons, S. Dai, M. Fojtik, B. Keller, A. Klinefelter, N. R. Pinckney, P. Raina, Y. Zhang, B. Zimmer, W. J. Dally, J. S. Emer, S. W. Keckler, and B. Khailany, “Magnet: A modular accelerator generator for neural networks,” in _Proceedings of the International Conference on Computer-Aided Design, ICCAD 2019, Westminster, CO, USA, November 4-7, 2019_ , D. Z. Pan, Ed. ACM, 2019, pp. 1–8. [Online]. Available: https://doi.org/10.1109/ICCAD45719.2019.8942127 

- [53] O. Wechsler, M. Behar, and B. Daga, “Spring hill (NNP-I 1000) intel’s data center inference chip,” in _2019 IEEE Hot Chips 31 Symposium (HCS), Cupertino, CA, USA, August 18-20, 2019_ . IEEE, 2019, pp. 1–12. [Online]. Available: https://doi.org/10.1109/HOTCHIPS.2019.8875671 

- [54] W. A. Wulf and S. A. McKee, “Hitting the memory wall: implications of the obvious,” _SIGARCH Comput. Archit. News_ , vol. 23, no. 1, pp. 20–24, 1995. [Online]. Available: https://doi.org/10.1145/216585.216588 

- [55] Q. Xiao, S. Zheng, B. Wu, P. Xu, X. Qian, and Y. Liang, “HASCO: towards agile hardware and software co-design for tensor computation,” in _48th ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2021, Valencia, Spain, June 14-18, 2021_ . IEEE, 2021, pp. 1055–1068. [Online]. Available: https://doi.org/10.1109/ISCA52012.2021.00086 

546 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

- [56] S. Xie, A. Kirillov, R. Girshick, and K. He, “Exploring randomly wired neural networks for image recognition,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2019, pp. 1284–1293. 

- [57] X. Yang, M. Gao, Q. Liu, J. Setter, J. Pu, A. Nayak, S. Bell, K. Cao, H. Ha, P. Raina, C. Kozyrakis, and M. Horowitz, “Interstellar: Using halide’s scheduling language to analyze DNN accelerators,” in _ASPLOS ’20: Architectural Support for Programming Languages and Operating Systems, Lausanne, Switzerland, March 16-20, 2020_ , J. R. Larus, L. Ceze, and K. Strauss, Eds. ACM, 2020, pp. 369–383. [Online]. Available: https://doi.org/10.1145/3373376.3378514 

- [58] S. Zheng, X. Zhang, L. Liu, S. Wei, and S. Yin, “Atomic dataflow based graph-level workload orchestration for scalable DNN accelerators,” in _IEEE International Symposium on HighPerformance Computer Architecture, HPCA 2022, Seoul, South Korea, April 2-6, 2022_ . IEEE, 2022, pp. 475–489. [Online]. Available: https://doi.org/10.1109/HPCA53966.2022.00042 

- [59] S. Zheng, X. Zhang, D. Ou, S. Tang, L. Liu, S. Wei, and S. Yin, “Efficient scheduling of irregular network structures on cnn accelerators,” _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ , vol. 39, no. 11, pp. 3408–3419, 2020. 

- [60] S. Zheng, S. Chen, P. Song, R. Chen, X. Li, S. Yan, D. Lin, J. Leng, and Y. Liang, “Chimera: An analytical optimizing framework for effective compute-intensive operators fusion,” in _2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 2023, pp. 1113–1126. 

## APPENDIX 

## _A. Abstract_ 

This appendix provides guidance on accessing and using the SoMa framework (introduced in Sec. V) to replicate the key results shown in Fig. 6 and Fig. 7. It also presents the architecture of the end-to-end compiler built on SoMa, detailing the inputs and outputs of each major stage (from model input to final instructions). Other experiments, which also involve similar steps and analyses, are omitted here for the sake of brevity. 

_2) Software dependencies:_ A C++ compilation environment with support for the C++ 17 standard is required. Linux is recommended. It is recommended to use “GNU make” to build the program. Additionally, the following Python packages are required for reproducing Fig. 7: “pandas”, “matplotlib”, “seaborn”, and “numpy”. 

## _D. Installation_ 

For artifact evaluation, start by downloading the artifact from Zenodo: 

- $ wget -O SOMA_AE.zip https://zenodo.org/records /14599935/files/SOMA_AE.zip?download=1 

- $ unzip SOMA_AE.zip 

Our SoMa exploration framework is in “SOMA”. We use ‘build.sh‘ to build the SoMa framework and create the result directory. 

$ cd SOMA 

- $ ./build.sh 

The executable target will be generated at “./build/soma”, and the result directories will be “results/overall” and “results/dse”. 

You can install the needed Python packages using pip with the following commands. 

- $ pip install -r requirements.txt 

Or you can use conda to install with the following commands. 

- $ conda install --file requirements.txt 

## _E. Experiment workflow_ 

_1) Overall and DSE:_ Once the SoMa framework is built, you can reproduce the Overall (Fig. 6) and DSE (Fig. 7) experiments with the command below. 

## $ ./run.sh --eta 

## _B. Artifact check-list (meta-information)_ 

- **Algorithm: Simulated Annealing** 

- **Program: C++, Shell, Python (only for data collection)** 

- **Compilation: by Makefile** 

- **Hardware: Recommend a server with 96+ cores and at least 1GB RAM per core.** 

- **Metrics: Cost function** _E×D_ **is employed in all experiments.** 

- **Experiments: reproduce Fig. 7 and Fig. 6.** 

- **How much disk space required (approximately)?: 1GB** 

- **How much time is needed to prepare workflow(approximately)?: Several minutes at most.** 

- **How much time is needed to complete experiments (approximately)?: For all 432 experiments (96 for Fig. 6 and 332 for Fig. 7), it takes about 2 days on a 192core Intel Xeon Platinum 8260. Most experiments (95%) are completed within 3.5 hours, while the remaining ones, mainly experiments with batch=64, require the full 2 days to finish.** 

- **Publicly available?: Yes** 

- **Code licenses (if publicly available)?: AGPL-3.0 License** 

- **Archived (provide DOI)?: 10.5281/zenodo.14599935** 

## _C. Description_ 

_1) How to access:_ The artifact is uploaded to Zenodo:10.5281/zenodo.14599935 

The “run.sh” takes parameters from “args.txt” as input for each soma instance, including compute power, DRAM bandwidth, storage directory, and the random seed. For each configuration, both our method and the baseline use the same seed. By default, the “run.sh” utilizes all CPU cores, with each core running a separate SoMa process. Each SoMa process outputs the corresponding results and logs to “results/overall” and “results/dse”. When using all 192 cores, it takes around 2 days to run on an Intel Xeon Platinum 8260 server. Due to the long runtime, we recommend using tools like “nohup” or “screen” to prevent disconnection due to inactivity, ensuring that all experiments complete successfully. 

After all experiments are completed, we use “get results.sh” to extract the results from the raw outputs. 

- $ ./get_results.sh 

“get results.sh” will use Python scripts under folder “pyscripts” to generate four files: “overall.csv”, “stats.log”, “dse.csv”, and “Fig7 heatmaps DSE.svg”. “overall.csv” contains all the data presented in Fig. 6, while “stats.log” includes all the data analyzed and calculated in the Sec. VI-B. “dse.csv” contains all the data related to Fig. 7, and “Fig7 heatmaps DSE.svg” reproduces Fig. 7. 

547 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

_2) Comparison with Baselines:_ The comparison with the baseline in Fig. 6 can be found in “stats.log”. For detailed data of each case, you can refer to “overall.csv”. The results of the DSE experiment are available in “Fig7 heatmaps DSE.svg”, with the detailed data in “dse.csv”, which is also humanreadable, just like “overall.csv”. 

_3) SoMa Compiler Workflow:_ In “Compiler-IR”, we present open-sourced files that showcase the workflow of the end-toend SoMa-based compiler developed for our high-performance commercial AI accelerator, the ZEBU FPGA-based Verification Platform, and the corresponding results. For related information, please refer to “Compiler-IR/Readme.md”. (This section is for material demonstration only and does not require execution or reproduction.) 

While we are currently unable to release the full source code of the whole compiler due to IP flow restrictions, we believe the provided scheduling engine at the core of SoMa, along with the materials and documentation, effectively offers a clear understanding of the entire workflow of the SoMabased compiler. Additionally, we are committed to establishing a small-scale cloud platform after the chip tape-out and related testing are completed. This platform will allow users to access the open-sourced compiler based on SoMa, with the flexibility to modify or even replace our scheduler (as long as the output is converted into IR format), enabling translation into chip-executable instructions. (This commitment has also been included in Sec. V-F of the paper.) 

## _F. Evaluation and expected results_ 

After executing “./get results.sh”, you can use the following script to compare the results with the expected ones: 

$ compare.sh 

“compare.sh” uses the “diff” command to compare “overall.csv”, “stats.log”, and “dse.csv”. The “Fig7 heatmaps DSE.svg” file may not be byte-for-byte identical due to font or library version differences, but the data used, namely “dse.csv”, should be identical. If all files match, it will output a message like “All files match the expected results!”. Otherwise, it will report “Some files do not match the expected results.” and highlight the differences. 

## _G. Methodology_ 

Submission, reviewing and badging methodology: 

- https://www.acm.org/publications/policies/artifactreview-badging 

- http://cTuning.org/ae/submission-20201122.html 

- http://cTuning.org/ae/reviewing-20201122.html 

548 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:30:09 UTC from IEEE Xplore.  Restrictions apply. 

