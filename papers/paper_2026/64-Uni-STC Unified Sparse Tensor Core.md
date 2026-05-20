2026 IEEE International Symposium on High-Performance Computer Architecture (HPCA) 

**==> picture [107 x 35] intentionally omitted <==**

# Uni-STC: Unified Sparse Tensor Core 

Haocheng Lian[1] , Qiyue Zhang[1] , Xinran Zhao[1] , Meichen Dong[1] , Yijie Nie[1] , Zhengyi Zhao[1] , Junzhong Shen[2] , Wei Guo[2] , Chun Huang[2] , Bingcai Sui[2] and Weifeng Liu[1] 

1. Super Scientific Software Laboratory, Department of CST, China University of Petroleum-Beijing, Beijing, China 

2. National University of Defense Technology, Changsha, China 

_{_ haocheng.lian, qiyue.zhang, xr.zhao, meichen.dong, yijie.nie, zhengyi.zhao _}_ @student.cup.edu.cn 

_{_ shenjunzhong, wineer guowei, chunhuang, bingcaisui _}_ @nudt.edu.cn and weifeng.liu@cup.edu.cn 

_**Abstract**_ **—Modern processors are increasingly adopting tensor cores as key computational units. Compared to existing designs for dense and structured sparsity, recent dual-side sparse tensor cores have evolved to support general sparsity. However, existing methods still face limitations on generality (incomplete sparse kernel support prevents broad applicability) and performance (outer-product/row-row schemes yield unsatisfactory hardware utilisation, data reuse, and energy efficiency).** 

**In this paper, we propose Uni-STC, a unified sparse tensor core that delivers high-performance dataflows for four key sparse kernels: sparse matrix-vector multiplication (SpMV), sparse matrixsparse vector multiplication (SpMSpV), sparse matrix-multiple vector multiplication (SpMM), and sparse general matrix-matrix multiplication (SpGEMM). To efficiently support these diverse sparse workloads, we first introduce BBC, a unified sparse format co-designed with Uni-STC’s dataflow. We then design UniSTC’s architecture supporting (1) fine-grained task partitioning to improve resource utilisation, (2) parallel sparse-tile processing to enhance data reuse, and (3) a dynamic network to reduce intermediate data movement and energy consumption. Evaluated across 2893 SuiteSparse and 302 DLMC matrices, Uni-STC demonstrates significant improvements, outperforming the stateof-the-art RM-STC with a 2.21** _×_ **geomean speedup and 2.96** _×_ **higher energy efficiency.** 

## I. INTRODUCTION 

In the past decade, tensor cores may be the most innovative data-level parallelism technology on modern processors. Compared to classic vector SIMD units, tensor cores can complete matrix-matrix multiplication (GEMM) far more efficiently in both throughput and energy. Driven by such demand from high performance scientific and AI workloads, modern mainstream GPUs [9], [71], CPUs [4] and TPUs [35], [36] are already equipped with tensor cores of various precisions, sizes, and structured sparsity capabilities. 

As sparse matrix computations are one of the major parallel computing patterns [1], designing sparsity-aware architectures received much attention [59], [66], [87]. Domain-specific architectures (DSAs) accelerating sparse computations, as well as sparse tensor cores (STCs) able to replace tensor cores in GPUs (the focus of our work), are representative directions. However, despite these advances, they still face significant limitations in terms of generality and performance. 

From the perspective of generality, modern scientific computing and AI applications are exhibiting an increasing demand for diverse sparse computation patterns [25], [53], [56], [69], with the main operations covering combinatorial 

TABLE I: A brief comparison of DS-STC [78], [92], RMSTC [30] and Uni-STC (our work proposed in this paper). 

|**STC**|**Sparse kernel**|**Datafow**|**Task of one cycle**|
|---|---|---|---|
|DS-STC|SpGEMM|Outer-product|Vector mul. vector<br>to update a matrix|
|RM-STC|SpGEMM|Row-row|Scalars mul. vectors<br>to update vectors|
|Uni-STC<br>(this work)|SpMV, SpMSpV,<br>SpMM & SpGEMM|Outer-product<br>plus segmented<br>dot-product|A group of parallel<br>vector mul. vector<br>to update scalars|



applications of multiple sparse kernels. Unfortunately, the limited functional support of existing sparsity-aware architectures constrains their use in wider real-world applications. 

From the perspective of performance, the existing architectures utilising outer-product [63], [78], [92] and row-row [30], [87], [93] dataflows often adopt coarse task partitioning, which results in suboptimal MAC utilisation. These architectures also continuously transmit intermediate products over large-scale networks, leading to high energy consumption. 

Although the goals are explicitly specified, simultaneously improving generality and performance remains challenging. Software-only interface expansion may address generality, but often leaves hardware capabilities underutilised, highlighting the need for hardware-software co-design [64], [66]. First, it is essential to devise a single sparse format that can efficiently support a variety of sparse kernels. Second, a unified architecture must be able to generate fine-grained tasks to utilise hardware resources, schedule tasks in parallel to increase data reuse, and manage data movement to reduce energy consumption. Finally, the architectural design requires rigorous validation using a large number of sparse matrices, various sparse kernels and real-world applications. 

In this paper, we propose Uni-STC, a unified sparse tensor core that brings high performance to complete sparse kernels, including sparse matrix-vector multiplication (SpMV), sparse matrix-sparse vector multiplication (SpMSpV), sparse matrixmultiple vector multiplication (SpMM), and sparse general matrix-matrix multiplication (SpGEMM). Uni-STC works on a fundamental sparse format called Bitmap-Bitmap-CSR (BBC) that combines compressed sparse row (CSR) arrays and two-level bitmap information. In addition, Uni-STC includes three newly designed functional units: tile multiply scheduler 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

(TMS), dot-product generator (DPG), and segmented dotproduct unit (SDPU). These units take sparse tiles from the BBC format as input, split and recombine them into small dot-product tasks, schedule them for data reuse, execute the dot-products with fewer data movements, and finally save the output in the BBC format. 

Compared to two state-of-the-art STC studies dual-side sparse tensor core (DS-STC) [78], [92] and row-merge sparse tensor core (RM-STC) [30], the Uni-STC emphasizes (1) the support of more complete sparse kernels, (2) the combination of various dataflows for generating fine-grained tasks, and (3) the increase of data-level parallelism in a single cycle. Table I gives a brief comparison of DS-STC, RM-STC and Uni-STC. 

We evaluate Uni-STC with all 2,893 SuiteSparse matrices across the four sparse kernels (SpMV, SpMSpV, SpMM, SpGEMM), 302 DLMC matrices for DNN inference, and an Algebraic MultiGrid (AMG) solver for application-level testing. Simulation results show Uni-STC achieves geometric mean speedups of 3 _._ 35 _×_ and 2 _._ 21 _×_ over DS-STC and RMSTC at the kernel level, accompanied by energy reductions of 1 _._ 97 _×_ and 1 _._ 27 _×_ , leading to energy efficiency gains of 7 _._ 05 _×_ and 2 _._ 96 _×_ . Despite an 18% area overhead in its dedicated modules compared to the state-of-the-art RM-STC, Uni-STC retains application-level speedups of 1 _._ 43 _×_ on DNNs and 1 _._ 92 _×_ on the AMG solver, enabled by its kernel performance. This work makes the following contributions: 

- We propose BBC, a unified format that supports softwarehardware collaborative computing for the four sparse kernels, while reducing storage overhead and mitigating complex hardware decoding. 

- We design the Uni-STC architecture to support the four sparse kernels, optimizing resource utilisation, data reuse, and energy efficiency by featuring three novel functional units: TMS, DPG and SDPU. 

- We conduct evaluation covering the performance, energy, and area of Uni-STC. Results demonstrate performance improvement and energy reduction over state-of-the-art designs with acceptable area overhead. 

## II. BACKGROUND 

## _A. CSR and Bitmap Storage Formats_ 

Sparse matrices typically employ compressed storage formats to save memory and enhance computational throughput. The CSR format is prevalent due to its simplicity and efficient row-wise access to nonzero elements. Alternatively, bitmapbased representations are favoured for smaller matrices, offering a compact layout that facilitates rapid element retrieval. Fig. 1 depicts a 4 _×_ 4 sparse matrix alongside its CSR and bitmap representations, highlighting their distinct storage and indexing mechanisms. 

## _B. Sparse Kernels_ 

In contrast to dense operations, sparse computations involve a diverse array of operand types, where inputs and outputs vary in both sparsity (dense or sparse) and dimensionality (vector or matrix). Fig. 2 lists these combinations into four fundamental 

**==> picture [253 x 86] intentionally omitted <==**

**----- Start of picture text -----**<br>
Matrix: CSR Format:<br>a b RowPtr 0 2 3 4 6<br>ColIdx 0 2 1 3 0 3<br>c Val a b c d e f<br>d Bitmap Format:<br>e f Mask 1 0 1 0 0 1 0 0 0 0 0 1 1 0 0 1<br>Val a b c d e f<br>**----- End of picture text -----**<br>


Fig. 1: An example of the CSR and Bitmap formats. 

**==> picture [253 x 100] intentionally omitted <==**

**----- Start of picture text -----**<br>
SpMV SpMSpV<br>K =4 N =1 N =1 K =4 N =1 N =1<br>Matrix  A Vector  x Vector  y Matrix  A Vector  x Vector  y<br>SpMM SpGEMM<br>K =4 N =4 N =4 K =4 N =4 N =4<br>Matrix  A Matrix  B Matrix  C Matrix  A Matrix  B Matrix  C<br>=4 =4 =4 =4 =4 =4<br>M K M M K M<br>=4 =4 =4 =4 =4 =4<br>M K M M K M<br>**----- End of picture text -----**<br>


Fig. 2: Sparse kernels SpMV, SpMSpV, SpMM and SpGEMM. 

**==> picture [253 x 153] intentionally omitted <==**

**----- Start of picture text -----**<br>
(1) Dot-product dataflow K =4 N =4 N =4<br>for m  in  0 ... M<br>  for n  in  0 ... N<br>for k  in  0 ... K<br>C [ m , n ] +=  A [ m , k ] *  B [ k , n ]<br>Matrix  A Matrix  B Matrix  C<br>(2) Outer-product dataflow K =4 N =4 N =4<br>for k  in  0 ... K<br>  for m  in  0 ... M<br>for n  in  0 ... N<br>C [ m , n ] +=  A [ m , k ] *  B [ k , n ]<br>Matrix  A Matrix  B Matrix  C<br>(3) Row-row dataflow K =4 N =4 N =4<br>for m  in  0 ... M<br>  for k  in  0 ... K<br>for n  in  0 ... N<br>C [ m , n ] +=  A [ m , k ] *  B [ k , n ]<br>Matrix  A Matrix  B Matrix  C<br>=4 =4 =4<br>M K M<br>=4 =4 =4<br>M K M<br>=4 =4 =4<br>M K M<br>**----- End of picture text -----**<br>


Fig. 3: Three fundamental dataflows for matrix multiplication: dot-product, outer-product and row-row. 

TABLE II: Sparse kernels in different applications. 

||**SpMV**|**SpMSpV**|**SpMM**|**SpGEMM**|
|---|---|---|---|---|
|**GNN**|||✓|✓|
|**AMG**|✓|||✓|
|**BFS**|✓|✓|||



kernels—SpMV, SpMSpV, SpMM, and SpGEMM—that serve as cornerstones for scientific computing and AI workloads. 

## _C. Dataflows_ 

Matrix multiplication primarily relies on three fundamental dataflows: (1) the dot-product (DotP) dataflow, which computes a single element of _C_ by multiplying a row of _A_ with a column of _B_ ; (2) the outer-product (OutP) dataflow, which updates the whole _C_ by multiplying a column of _A_ with a row of _B_ ; and (3) the row-row dataflow, which generates a row of _C_ by scaling rows of _B_ with scalar elements from a row of _A_ . Fig. 3 provides a schematic illustration of these mechanisms. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

TABLE III: Task sizes at different levels in STCs (64 MACs). 

|**Task**<br>**Level**|**Task**<br>**Name**|**Task Size (**_M × N × K_**)**|**Task Size (**_M × N × K_**)**|**Task Size (**_M × N × K_**)**|**Task Size (**_M × N × K_**)**|
|---|---|---|---|---|---|
|||**NV-DTC**<br>**[60]**|**DS-STC**<br>**[78], [92]**|**RM-STC**<br>**[30]**|**Uni-STC**<br>**(ours)**|
|T1|**MMA**<br>**instruction**|16_×_16_×_16||||
|T2|**Machine**<br>**instruction**|8_×_8_×_4|16_×_16_×_1|8_×_16_×_2|None|
|T3<br>T4|**Tile**<br>**Vt**|4_×_4_×_4<br>8_×_8_×_1<br>8_×_4_×_2<br>N|||4_×_4_×_4<br>114|



**==> picture [253 x 215] intentionally omitted <==**

**----- Start of picture text -----**<br>
DS-STC RM-STC Uni-STC<br>Matrix  A<br>MAC: 2x2 MAC: 2x2 MAC: 1x4<br>SpMV<br>Multiply<br>dense<br>vector  x #Cycles: 6 #Cycles: 6 #Cycles: 3<br>MAC Util.: 41.67% MAC Util.: 41.67% MAC Util.: 83.33%<br>SpMSpV<br>Multiply<br>vector sparse x #Cycles: 4 #Cycles: 5 #Cycles: 2<br>MAC Util.: 37.50% MAC Util.: 30.00% MAC Util.: 75.00%<br>SpMM<br>Multiply<br>dense<br>matrix  B #Cycles: 12 #Cycles: 12 #Cycles: 10<br>MAC Util.: 83.33% MAC Util.: 83.33% MAC Util.: 100%<br>SpGEMM<br>Multiply<br>matrix sparse B #Cycles: 9 #Cycles: 11 #Cycles: 7<br>MAC Util.: 69.44% MAC Util.: 56.82% MAC Util.: 89.29%<br>**----- End of picture text -----**<br>


Fig. 4: Schematic dataflow comparison of DS-STC, RM-STC, and Uni-STC across the four kernels, assuming a MAC array size of 4. Solid and dashed black boxes demarcate the data access windows for the first and final execution cycles, respectively; red slashes highlight ineffective memory accesses. For DS-STC and RM-STC, black dots signify accessed elements, while orange lines trace the per-cycle execution trajectory. 

## III. MOTIVATION 

_A. Challenge 1: Acceleration of sparse applications_ 

_1) Demand for generality:_ As summarized in Table II, realworld applications frequently require a combination of sparse kernels. For instance, Graph Neural Networks (GNNs) [25], [69] use both SpMM and SpGEMM for node information propagation and aggregation. Similarly, Algebraic Multigrid (AMG) solvers [53] and Breadth-First Search (BFS) algorithms [56] depend on multiple sparse kernels for convergence and traversal efficiency. This workload diversity underscores the critical need for accelerators capable of supporting a comprehensive suite of sparse computations. 

_2) Unified data structure:_ Implementing a unified data structure is a necessary condition for effectively supporting multiple sparse kernels. This structure eliminates costly online format conversions between kernels, supporting a unified dataflow in hardware design to enhance generality. However, 

**==> picture [253 x 146] intentionally omitted <==**

**----- Start of picture text -----**<br>
(0%, 25%] utilisation (25%, 50%] utilisation<br>(50%, 75%] utilisation (75%, 100%] utilisation<br>175 140 1200 60<br>150 120 1000 50<br>125 100 800 40<br>1007550 806040 600400 3020<br>25 20 200 10<br>0 0 0 0<br>NV DS RM Uni NV DS RM Uni NV DS RM Uni NV DS RM Uni<br>consph shipsec1 crankseg_2 cant<br>n=83.3K, nnz=6.0M  n=140.9K, nnz=7.8M  n=63.8K, nnz=14.1M  n=62.5K, nnz=4.0M<br>#inter-prod/blk=164.9 #inter-prod/blk=189.5 #inter-prod/blk=198.5 #inter-prod/blk=280.2<br>3530 7060 7060 16001400<br>25 50 50 1200<br>201510 403020 403020 1000800600400<br>5 10 10 200<br>0 0 0 0<br>NV DS RM Uni NV DS RM Uni NV DS RM Uni NV DS RM Uni<br>opt1 pdb1HYS pwtk gupta3<br>n=15.4K, nnz=1.9M  n=36.4K, nnz=4.3M  n=217.9K, nnz=11.6M  n=16.8K, nnz=9.3M<br>#inter-prod/blk=506.4 #inter-prod/blk=517.2 #inter-prod/blk=548.3 #inter-prod/blk=1154.1<br>#Cycles (in Millions)<br>**----- End of picture text -----**<br>


Fig. 5: STCs’ SpGEMM performance on eight representative matrices in Table VII ( _C_ = _A_[2] ). This figure shows the results with color-coded blocks, which display the proportion of cycles with varying utilisation rates within the total cycles. 

designing such a unified structure is challenging because of the sparse kernels variety and the hardware constraints. 

Given the limited generality of existing accelerators, accelerating real-world sparse applications requires a unified framework that integrates a common data structure, software algorithms, and a sparse tensor core. 

Understanding the inefficiency of existing STCs requires examining their decomposition of large tasks into multiple layers. As shown in Table III, we organize the computation into a four-level task hierarchy (T1–T4): 

- (T1) The matrix multiply-accumulate (MMA) instruction task: A 16( _M_ ) _×_ 16( _N_ ) _×_ 16( _K_ ) matrix multiplication corresponding to a warp MMA (WMMA) instruction on an A100 GPU. 

- (T2) Machine instruction task: A task corresponding to a Parallel Thread Execution (PTX) instruction from the compiler, which follows a predefined, multi-cycle execution flow. 

- (T3) Tile task: A sub-task generated by partitioning a T2 task based on the STC’s per-cycle throughput. For sparse computation, it is designed to support hardwarelevel concatenation. 

- (T4) Vector task: A fine-grained task derived from a T3 task, whose length is determined by the STC’s ability to merge adjacent intermediate products. 

Specifically, fixed-size T2 tasks are well-suited for regular sparsity but struggle with unstructured patterns. The unpredictable locations of nonzeros in such cases lead to inefficient memory accesses and significant throughput degradation. Fig. 4 illustrates how fixed task partitioning can degrade throughput. In each cycle, DS-STC forms an outer-product task from a half-column of _A_ and a half-row of _B/x_ , whereas RM-STC generates multiple ‘scalar _×_ vector’ tasks from two half-row vectors. This rigid selection frequently causes inefficient data accesses (marked by red slashes), resulting in lower MAC utilisation compared to Uni-STC. Our quantitative analysis in Fig. 5 further emphasizes this performance gap. For 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
M =0 M =0<br>can not<br>K =0 concat M =1 M ≠1 ...<br>M =2 M ≠2<br>M =3 ... M =0 M =2<br>can not K =1 ... N N =0 N =3<br>concat M =7 M =7 K =0,2 K =1,2,3<br>Outer-product Row-row Dot-product<br>(DS-STC) (RM-STC) (Uni-STC)<br>**----- End of picture text -----**<br>


Fig. 6: Restrictions of different STC on task concatenation. 

real-world matrices, NVIDIA dense tensor core (NV-DTC) offers only limited sparsity support, with MAC utilisation falling below 25% in 84.34% of cycles. Although DS-STC and RM-STC demonstrate higher efficiency than NV-DTC, their utilisation remains suboptimal. We therefore identify two primary challenges to enhancing STC MAC utilisation: task scheduling and task concatenation. 

## _B. Challenge 2: Task scheduling_ 

_1) Inefficiency of data gathering:_ As shown in Fig. 4, DSSTC and RM-STC achieve transient high MAC utilisation by gathering sparse matrices into dense vectors. However, they suffer from frequent low-utilisation phases (indicated by red slashes in Fig. 4). These phases, stemming from ineffective accesses, lead to 61.68% and 62.78% of cycles operating below 50% utilisation (Fig. 5). Furthermore, because their T3 task dimensions are rigidly tailored to specific sparsity patterns, efficiency degrades significantly when handling diverse realworld patterns, such as long rows in matrix _A_ . 

_2) Insufficient parallelism within STC:_ The proportion of low-utilisation cycles in DS-STC and RM-STC significantly surpasses the 15.82% baseline achieved in Uni-STC. This stems from their lack of a load-aware task execution mechanism. Given the inherent difficulty in minimizing low-load tasks, a paradigm shift from gathering data to gathering tasks (aggregating multiple low-load tasks) is essential. However, existing architectures lack the workload-aware design necessary to implement this shift, which hinders overall utilisation. 

Therefore, it is necessary to bypass T2 task partitioning, integrate task-load awareness into STC, and support parallel task execution. 

## _C. Challenge 3: Task concatenation_ 

_1) Coarse Task Granularity:_ The limited proportion of high-utilisation cycles in DS-STC and RM-STC (approximately 20% in the red region of Fig. 5) stems from their coarse task granularity. Specifically, for tasks in the 50-75% utilisation range (the yellow region), these architectures lack a mechanism to further partition and reorganize them to better fit the MAC array dimensions. Therefore, T3 tasks need to be further broken down. 

_2) Concatenating restrictions:_ However, as shown in Fig. 6, merely refining task granularity is insufficient to resolve the utilisation bottleneck. DS-STC and RM-STC, employing outer-product and row-row dataflows respectively, adhere to 

**==> picture [253 x 126] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Register (16,384 x 32-bit)<br>or or or or or or or<br>A : dense/sparse matrix B : dense/sparse matrix/vector  C : dense/sparse matrix/vector<br>L0 instruction cache Meta buffer  (144B)<br>Warp scheduler<br>Dispatch unit<br>Register file Tile multiply  Dot  Segmented<br>INT 32FP 32 Uni- scheduler (TMS) generators product  dot product unit<br>FP 64 STC  (DPG) 0~7 (SDPU)<br>LD/ST SFU<br>(b) GPU SM (c) Unified Sparse Tensor Core (Uni-STC)<br> Network  (1KB)<br>(2KB) B<br>Tile queue  Buffer  Shuffle  Network C<br>A<br>Dot product queue<br> Network A Final accumulator<br>**----- End of picture text -----**<br>


Fig. 7: (a) Uni-STC’s supported data types; (b) Uni-STC’s position in GPU SM; and (c) Uni-STC’s architecture, highlighting three core components: TMS, DPG and SDPU. 

rigid 2D or 3D structural layouts (consistent with T3 task definitions in Table III). Such rigidity limits task concatenation flexibility: DS-STC cannot concatenate tasks at different positions along the _K_ -dimension, whereas RM-STC only permits concatenation along the _N_ -dimension. Consequently, even with fine-grained tasks, these spatial constraints prevent efficient packing and leave the hardware underutilised. 

Therefore, adopting a least-constrained dot-product method for task refinement offers a more promising solution. 

## _D. Uni-STC Design Principles_ 

Addressing these challenges, we formulate three design principles for Uni-STC: 

- 1) Unify data structure and architecture to support diverse sparse kernels. 

- 2) Offload T1 task execution to the STC while augmenting scheduling capabilities. 

- 3) Decompose T3 tasks into fine-grained vector tasks to enhance task concatenation efficiency. 

## IV. UNI-STC ARCHITECTURE 

As shown in Fig. 7, to overcome the limitations of existing STCs, we propose Uni-STC, a unified architecture designed to replace the original GPU tensor cores and support various sparse kernels. It comprises three functional units: the Tile Multiply Scheduler (TMS), the Dot Product Generator (DPG), and the Segmented Dot Product Unit (SDPU). Operationally, the TMS first decomposes T1 tasks into T3 tasks for the DPGs. The DPGs then subsequently partition these into fine-grained T4 tasks, which are ultimately concatenated and executed by the SDPU. 

## _A. Task Generation Using TMS and DPG_ 

To support diverse sparse patterns and kernels, Uni-STC’s fundamental working unit is the 4 _×_ 4 _×_ 4 T3 task, derived from the decomposition of a larger 16 _×_ 16 _×_ 16 T1 task. This design choice is motivated by three key considerations: 

(1) Mitigating inefficiency from real-world sparsity: Tasks defined with _K_ = 1 (DS-STC) or _K_ = 2 (RM-STC) lead to numerous low-utilisation cycles when handling patterns such 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 133] intentionally omitted <==**

**----- Start of picture text -----**<br>
Tile multiply scheduler (TMS)<br>non-zero tileT3 task 0 15 2 36 7 nnz_r = 2 0  1 3   3 k 0   pos 0 ！<br>[4] 8 9 A B<br>CDE F nnz_c = 2 CD 3 1 0 C<br>0 2 0 2<br>4 6 nnz_r = 3 4 6 2   B 0   1<br>Bitmap  A 8 A nnz_c = 2 8 A 2   7 0   D<br>6 7 nnz_r = 2 6 7 2   A 1   0 ！<br>A B A B<br>nnz_c = 2 2   6 1   2<br>1 3 nnz_r = 1 1 3 1   A 1   4<br>Bitmap  B nnz_c = 2 1   8 1   6<br>1 Outer-product unit 2 Sequencer 3  Dispatcher<br>=0<br>k order<br>=1<br>k<br>=2<br>k Tile queue<br>=3<br>k<br>Dot product generators(DPGs)<br>**----- End of picture text -----**<br>


Fig. 8: TMS component and its subsequent modules. 

TABLE IV: Trade-offs of T3 task sizes on cycle count, the number of DPGs to saturate SDPU, and network scale to route tiles and nonzeros. The 4 _×_ 4 _×_ 4 size is the best among the three, as it avoids excessive DPG counts and routing overhead. 

|**Task**<br>**size**|**#Cycles**|**#DPGs to**<br>**saturate SDPU**|**Network scale to route**|**Network scale to route**|
|---|---|---|---|---|
||||**tiles**|**nonzeros**|
|2_×_2_×_2|1|32-64 (high)|64_×_#DPGs (high)|4_×_4|
|4_×_4_×_4|1|8-16|16_×_#DPGs|16_×_16|
|8_×_8_×_8|_≥_2 (high)|2-4 (low)|4_×_#DPGs|64_×_64 (high)|



as long rows or long columns (e.g., matrix crankseg_2 in Fig. 5) or nonzeros concentrated near the diagonal (matrix cant). To achieve stable utilisation across such diverse structures, we adopt a symmetric configuration with _M_ = _N_ = _K_ . 

(2) Facilitating a unified data structure: To meet the unified data structure requirement outlined in Section III-D while avoiding complex hardware decoders, we select symmetric tile dimensions. This symmetry allows both operands to share identical bitmap encoding logic. 

(3) Balancing resource utilisation and timing: Table IV compares the 4 _×_ 4 _×_ 4 configuration with alternative tile sizes. A 2 _×_ 2 _×_ 2 design incurs excessive resource overhead, requiring 32-64 DPGs and a much larger routing network. Conversely, an 8 _×_ 8 _×_ 8 size fails to meet timing constraints ( _≥_ 2 cycles), suffers from limited parallelism (2-4 DPGs, denoted as low), and has high routing costs. The chosen 4 _×_ 4 _×_ 4 configuration strikes an balance, avoiding the resource overhead of smaller tiles and the timing violations of larger ones. 

During computation, a 16 _×_ 16 matrix block is partitioned into 16 4 _×_ 4 tiles. A two-level bitmap encodes this structure to steer the pipeline: the top-level bitmap (marking tiles) guides the TMS in generating T3 tasks, while the bottom-level bitmap (marking elements) directs the DPG to generate T4 tasks. 

_1) Tile multiply scheduler (TMS) in Fig. 8:_ 1 Generation of T3 tasks. The TMS generates a four-layer intermediate product bitmap using an outer-product approach, where each position in the bitmap represents a T3 task. For instance, the top-left position in the _K_ = 0 bitmap (marked by a green ‘0’) signifies the T3 task _C_ 00+ = _A_ 00 _× B_ 00. 

2 Task ordering. Task ordering for batched T3 tasks substantially impacts data reuse and energy consumption. For in- 

**==> picture [253 x 309] intentionally omitted <==**

**----- Start of picture text -----**<br>
Dot product generator (DPG) 0<br>non-zeroelement 10 00 0 01 1 intermediateproducts  0 1T4 task<br>0 0 0 0 2 3 4<br>0 0 0 0 1  1 5<br>1 C1CC 1 1111 11 Previous C 6 7 00<br>Bitmap  A00 1 1 1<br>display<br>1 in hex<br>1 1 F 9 21 3 F 49<br>4<br>E 8 6E7 8<br>Bitmap  B00 1 1 Hex encoded C00   Index  C00<br>1 1 2 K -index 3 Index<br>1 Outer-product unit encoder composer<br>Fig. 9: DPG component and its adjacent modules.<br>Dot-product Outer-product Row-row<br>50<br>25<br>0<br>7<br>4<br>1<br>7<br>4<br>1<br>12<br>6<br>0<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16<br>#Nonzeros of 4(M) × 4(N) matrix C<br>=0<br>k<br>=1<br>k<br>=2<br>k<br>Tile queue Fill<br>unit (SDPU)<br>Dot product queue<br>=3<br>k<br>Tile multiply scheduler (TMS)  Segmented dot product<br>rate (%)<br>Data reuse<br>#Tasks (parallel)<br>#Tasks (aligned)<br>Conflict rate (%)<br>**----- End of picture text -----**<br>


Fig. 10: Comparison of dot-product, outer-product and rowrow ordering methods (assuming Uni-STC can complete eight T3 tasks per cycle). The metrics are: (1) data reuse rates for matrices _A_ and _B_ , calculated as 1 _−_ TheoreticalActual AccessesAccesses[,][(2)] average parallel tasks per cycle, (3) average aligned tasks per cycle, and (4) average write conflict rate. 

stance, at layer _K_ = 0, parallel execution of _T_ 00 _, T_ 01 _, T_ 10 _, T_ 11 fetches tiles ( _A_ 0 _, A_ 1 _, B_ 0 _, B_ 1) only once, whereas sequential execution would double read volume. To identify the most effective strategy, we evaluated dot-product, outer-product, and row-row orders based on parallelism, _K_ -dimension alignment, and write conflicts (defined as[#] # _[Con] T otalCycles[f][lictC][y][cles]_[).][As][shown][in] Fig. 10, the outer-product strategy is superior, achieving high parallelism (avg. 4.54 tasks), a 47.38% peak reuse rate through effective _K_ -alignment, and low write conflicts (e.g., 6.2% peak at #Nonzeros=6), thereby mitigating bottlenecks. 

Additionally, we implement an adaptive intra-layer task ordering mechanism. The system dynamically selects a columnmajor order when nonzero rows outnumber nonzero columns, and a row-major order otherwise, enhancing data reuse across diverse workloads. 

3 Task dispatch. The TMS enqueues generated T3 tasks into the Tile queue. In the event of a write conflict (e.g., the T3 task marked by the red box and exclamation mark in Fig. 8), the Tile queue employs round-robin arbitration to stall the conflicting T3 task, forcing the corresponding DPG to wait one cycle before execution. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

_2) Dot-product generators (DPGs) :_ The DPG’s workflow begins with a T3 task. 1 First, it applies an outer-product method to the bottom-level bitmaps to generate four intermediate bitmap layers. 2 These layers are then overlaid, creating a map where the 4-bit value at each position encodes the indexmatching results for a sparse vector dot-product. 

3 Next, the DPG combines this overlaid map with the structural layout of tile _C_ to generate 8-bit T4 task codes. Concurrently, it extracts the required operand vectors from tiles _A_ and _B_ for subsequent concatenation. For instance, in Fig. 9, the value ‘49’ in the orange box signifies the following: the upper nibble ‘4’ denotes the accumulation target (4th nonzero in tile _C_ ), while the lower nibble ‘9’ encodes the sparse dot-product pattern (0x1001). Thus, the T4 task ‘49’ corresponds to: _C_ 0 _,_ 0[4] += _A_ 1 _,_ 0 _× B_ 0 _,_ 3 + _A_ 1 _,_ 3 _× B_ 3 _,_ 3. 

4 Multiple T4 tasks from a DPG are filled into the Dotproduct queue in a _Z_ -shaped pattern, as depicted in Fig. 9. 

This ordering is critical for minimizing data movement. When vector tasks are concatenated, the required broadcast range for any nonzero is minimized. Specifically: (1) For matrix _A_ , an element is broadcast to a compact group of only 5 (4 + 1) adjacent multipliers, as our scheduling limits its reuse to at most two consecutive vector tasks (length _≤_ 4). (2) For matrix _B_ , the _Z_ -shaped fill order ensures an element is broadcast to a slightly wider range of 9 (4+4+1) multipliers, because two tasks requiring the same _B_ data are separated by at most one intervening task. This localized data forwarding is highly efficient; alternative strategies, such as an _N_ -shaped fill order, were tested and found to be inferior for most matrices. 

The aforementioned process of task dispatch and vector concatenation relies on simple prefix sums and shift units. These components are commonly employed in prior works [21], [87], and are therefore omitted for brevity. 

Uni-STC’s default configuration of 8 DPGs is driven by a sensitivity study on Energy Efficiency Density (EED) and alignment with hardware resource budgets. The EED analysis, presented in Fig. 22, shows that increasing the DPG count from 4 to 8 benefits SpMM and SpGEMM, whereas a further increase to 16 yields diminishing returns and introduces higher overheads, particularly for SpMV and SpMSpV. Moreover, the 8-DPG configuration aligns with existing tensor core resource budgets. Because each T3 task is constrained to at most 64 intermediate products, Uni-STC can flexibly scale its precision from 256 MACs@FP16 to 64 MACs@FP64 within the same hardware footprint. This is accomplished while retaining sufficient task concatenation capability to achieve significant performance gains. 

## _B. Segmented Dot Product Unit (SDPU)_ 

To facilitate parallel execution of multiple T4 tasks, we introduce the SDPU. As illustrated in Fig. 11(a), T4 tasks generated by DPG 0 are compactly concatenated for batched processing within the SDPU. Fig. 11(b) is a merge-forward structure, which dynamically configures any four adjacent multipliers into a complete binary tree. This design yields two key benefits. First, it enables the compact, parallel computation 

**==> picture [253 x 251] intentionally omitted <==**

**----- Start of picture text -----**<br>
Segmented dot product unit (SDPU)<br>21 3F 49 78 6E<br>a b<br>1 1 2 3 4 1 2 1 1 2 3<br>1 3 7 3 1 3 3<br>ctrl<br>1 10 3 1 6<br>fwd a sum<br>(a) (b)<br>Fig. 11: SDPU component and its preceding modules.<br>Control by TMS Control by  8 DPGs Synchronize<br>64x5 MUX<br>... ... array ( A ) ... ...<br>A  &  B 2 ⋅ 16x 8 tile task 2 ⋅ 8 ⋅ 4x8 64x9 MUX dot product 8 ⋅ 16x16 16x 8  C<br>tiles networks queue networks array ( B ) queue network network tiles<br>( A  &  B ) ( A  &  B ) ( C ) ( C )<br>T ask Generation Task Concatenation Execution & Write  C<br>(DPG) 0<br>Dot product queue<br>Dot product generator<br>T3 tasks<br>concat vectors<br>**----- End of picture text -----**<br>


Fig. 12: Internal pipeline and datapath in Uni-STC. 

of multiple T4 tasks. Second, it facilitates the pre-merging of up to four partial products before they are written out, which significantly reduces write traffic to the result matrix _C_ . 

## _C. Internal Pipeline and Datapath_ 

As shown in Fig. 12, to meet the 1.5 GHz target frequency (A100), Uni-STC implements a three-stage internal pipeline that uses Tile and Dot-product queues to manage task lifecycles, thereby decoupling control and data flows. 

_1) Three-Stage Pipeline:_ The execution flow, triggered by the issuance of a UWMMA instruction (see Section IV-E), consists of three main stages: 

- Stage 1: Task Generation. Acting as the controller, the TMS fetches the top-level bitmap from the Meta Buffer (144B) and generates T3 tasks, which are dispatched into the Tile queue. 

- Stage 2: Task Concatenation. Eight DPGs operate in parallel, utilising underlying bitmaps to populate the Dotproduct queue with T3 and T4 task codes, as well as network control signals. These signals are used to acquire operands from the Matrix _A_ buffer (2KB) and registers. 

- Stage 3: Execution & Write _C_ . The SDPU pops a batch of merged T4 tasks, performs segmented dot-products, accumulates results in an accumulator buffer (1KB), and updates registers. 

Notably, the Tile and Dot-product queues store only control information rather than the numerical values of matrices _A_ and _B_ . This design choice is driven by two factors: first, to minimize the area overhead associated with wide datapaths; and second, to accommodate potential latency, as values may not be available in the registers or buffers during the first two pipeline stages. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

_2) Datapath:_ Prior studies (e.g., RM-STC [30]) have established that on-chip network scale and data traffic are the primary drivers of energy consumption in STCs. While previous sections have demonstrated how Uni-STC mitigates data traffic—specifically through reuse-aware scheduling in the TMS and partial product pre-merging in the SDPU—the scale of the interconnect remains a critical efficiency bottleneck. Therefore, this section shifts focus to the other factor: optimizing Uni-STC’s network scale to reduce energy per bit. 

As shown in Fig. 12, Uni-STC employs a two-layer network for data access. The outer layer, controlled by the TMS, uses three dedicated 16 _×_ 8 networks to forward tiles for matrices _A_ , _B_ , and _C_ . For matrix _C_ , since the SDPU output can be directly partitioned, each tile is handled by a dedicated 16 _×_ 16 network, and with 8 DPGs in parallel, this results in an 8 _·_ 16 _×_ 16 network structure. For matrices _A_ and _B_ , each first passes through a dedicated 4 _×_ 8 network into the dot product queue. Subsequently, two sets of MUX arrays—64 _×_ 5 for _A_ and 64 _×_ 9 for _B_ —select the corresponding vectors from the queue. This hierarchical network design eliminates the need to implement separate 64 _×_ 256 networks for matrices _A_ , _B_ , and _C_ , achieving reductions in energy per bit of 7 _._ 16 _×_ , 5 _._ 33 _×_ , and 2 _._ 83 _×_ , respectively. 

Additionally, Uni-STC employs a dynamic DPG activation mechanism to optimize energy efficiency. By calculating the prefix sums of intermediate products at the Tile queue head, the TMS determines the number of DPGs required to saturate the SDPU. The control logic then power-gates any redundant DPGs and their associated datapaths—including the input networks for matrices _A_ and _B_ (2 _·_ 8 _·_ 4 _×_ 8) and the output network for matrix _C_ (8 _·_ 16 _×_ 16). This selective gating, which assumes wake-up latency is hidden by look-ahead scheduling, enables energy savings of up to 2 _._ 83 _×_ compared to an alwayson approach (see Section VI-C). 

## _D. BBC Format_ 

Guided by the design principles from Section III-D, we propose the BBC format, a hierarchical data structure. Its outer layer uses the CSR format to organize submatrices, while its inner layer employs a two-level bitmap to manage elements within each sparse submatrix. Fig. 13 illustrates this format with a downsized 8 _×_ 8 matrix, where each 4 _×_ 4 submatrix is subdivided into four 2 _×_ 2 blocks. 

The second-level index of the BBC format, ValPtr Lv2, is provided directly to Uni-STC, enabling the TMS to control the forwarding of corresponding tile data. This design choice is motivated by a trade-off between hardware and software costs. Unlike RM-STC, which requires a hardware decoder consuming 16.67% of the area overhead, BBC enables direct execution. We offload indexing to a one-time software encoding. This approach incurs negligible storage overhead—no more than 0.3% within the BBC format, translating to just 0.015% of the total die area—while eliminating the costly hardware decoder. 

Additionally, the two-level bitmap structure can be used directly by TMS without decoding. Converting a 4 _×_ 4 submatrix 

**==> picture [253 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
RowPtr 0 1 3 ColIdx 0 0 1 BlkPtr 0 1 2 4<br>a b<br>c d BitMap_Lv1<br>1 0 0 0 1 1 ValPtr_Lv1 0 4 6 10<br>+<br>0 0 1 0 0 0<br>ValPtr_Lv2 0 0 0 2<br>BitMap_Lv2<br>g i 1 1 1 0 0 1 0 1<br>h j 1 1 1 0 1 0 0 1<br>e<br>f Value a b c d e f g h i j<br>**----- End of picture text -----**<br>


Fig. 13: Downsized BBC format for an 8 _×_ 8 matrix. At the top level, RowPtr and ColIdx use CSR to locate nonzero 4 _×_ 4 submatrices. The sparsity pattern within these submatrices is then described by a two-level bitmap: BitMap Lv1 identifies which 2 _×_ 2 blocks contain nonzero elements, and BitMap Lv2 specifies the exact location of the nonzero elements within those blocks. All nonzero elements are stored in the Value array. They are accessed using a two-level pointer where ValPtr Lv1 provides the base address for a 4 _×_ 4 submatrix and ValPtr Lv2 provides the offset for a specific 2 _×_ 2 block. 

within the DPG into four row or column vectors accounts for approximately 6.6% of the total area overhead. The primary cost is the one-time offline construction of the BBC format. However, this cost is amortized across multiple invocations and can be entirely eliminated for frequently used matrices by saving and reloading them via implemented file I/O function. 

## _E. Hardware Integration with GPU_ 

To integrate Uni-STC as a coprocessor in the GPU Streaming Multiprocessor (SM) and bypass the T2 task partitioning stage, we require micro-architectural adjustments to the SM in two parts: instruction issue and data interaction. 

(1) Instruction issue: This requires two control logic modifications: updating the instruction decoder to parse Uni-STC’s opcodes, and extending the warp scheduler to dispatch the decoded instructions. Both modifications incur negligible area and energy overhead. 

(2) Data interaction: Uni-STC interfaces with core SM components solely via the register file, a design that leverages the high-bandwidth operand collector interfaces of modern SM90+ architectures (e.g., Hopper and Blackwell). For earlier generations like Ampere, however, the register-file ports must be widened to provide the necessary bandwidth: up to 16 FP64 source and 4 FP64 destination operands per thread, per cycle. 

With these adjustments, Uni-STC operates as an independent computational unit within the SM. The following subsections detail the instruction set, execution lifecycle and control interaction. 

## _F. Instruction Set_ 

Table V summarizes the Uni-STC instruction set (UWMMA), which follows WMMA semantics and includes the cycle ranges for FP64 operations. Data types are categorized by suffixes: ‘ _i_ ’ for 8-bit indexes, ‘ _b_ ’ for 16-bit bitmaps, and ‘ _v_ ’ for 64-bit values. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**Algorithm 1:** SpMV/SpMSpV with Uni-STC@FP64 

TABLE V: Uni-STC@FP64 instruction set (UWMMA). 

|**Operation**|**Operation**|**Registers per threads**|**Cycles**|
|---|---|---|---|
|Load|Meta<br>(MV)|_A_16_b_1_, A_16_b_2_, X_16_b,_<br>_A_4_b_1_/A_4_i_1_, A_4_b_2_/A_4_i_2|1|
||Meta<br>(MM)|_A_16_b, B_16_b, C_16_b_,<br>_A_4_b/A_4_i, B_4_b/B_4_i, C_4_b/C_4_i_|1|
||_A_|_Av_(0_∼_7)|2|
|T3 Task<br>Generate|MV|Use meta data saved in buffer|1_∼_4|
||MM|Use meta data saved in buffer|1_∼_8|
|Calculate<br>and Store|MV|_Av_2(0_∼_7)_, Xv, Yv_|1_∼_8|
||MM|_Bv_(0_∼_7)_, Cv_(0_∼_7)|1_∼_64|



To comply with the operand limits of PTX instructions (e.g., the ‘mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64’ variant allows a maximum of 20 FP64 register operands per thread) and better aligns with the properties of sparse kernels, we choose to store both the block values of matrix _A_ and the corresponding block structures within Uni-STC’s internal buffers. Integrating the UWMMA instruction set and this data handling approach necessitates compiler modifications. 

## _G. Execution Lifecycle and Control Interaction_ 

Uni-STC executes sparse kernels through a coordinated UWMMA instruction sequence. This lifecycle relies on interaction with the SM and internal state registers to achieve asynchronous task generation and synchronous computation: 

(1) Operand collection. The cycle begins with stc.load instructions. The SM uses the operand collector to fetch numerical data or metadata from the register files and stores them into Uni-STC’s internal buffers (Matrix A Buffer or Meta Buffer). This phase is synchronous and memory-bound. 

(2) Asynchronous task generation. Upon issuing a stc.task instruction, the Uni-STC transitions its state register from IDLE to BUSY. This triggers the TMS and DPGs to begin processing metadata and filling the two task queues. This asynchronous process allows the SM to immediately retire the stc.task_gen instruction and proceed with other work, effectively hiding the task generation latency. (3) Synchronized computation. The stc.numeric instruction initiates computation on the SDPU by first checking the flag register: 

- Stall (BUSY): If the flag is BUSY, indicating insufficiently populated task queues, the pipeline stalls. 

- Execute (READY): Once the DPGs populate the queues, the flag transitions to READY. The SDPU then begins execution, consuming T4 tasks, performing segmented dot-products, and accumulating the results. 

(4) Completion. When the batch of T4 tasks is fully processed, the flag returns to IDLE, enabling the results to be written back to the register files. 

1: _laneid ← threadIdx.x_ &31 

2: _row ← warpRowId_ [ _warpid_ ] 3: _start ← warpIndex_ [ _warpid_ ] 4: _end ← warpIndex_ [ _warpid_ + 1] 5: _ry ←_ 0 6: _j ← start_ 7: **while** _j < end_ **do** 8: _a_ 4 _b ← load bitmap_ ( _laneid_ ) 9: _a_ 4 _i ← load offset_ ( _laneid_ ) 10: _rxb ← load bitmapx_ ( _laneid_ ) 11: % _stc.load.meta mv A_ 16 _b_ [ _j_ ] _, A_ 16 _b_ [ _j_ + 1] _, rxb, a_ 4 _b, a_ 4 _i_ 12: % _stc.task gen.mv_ // TMS and DPG generate T3 and T4 tasks 13: **for** _i ←_ 0 _→_ 15 **do** 14: _rA_ [ _i_ ] _← load value A_ ( _A val_ + _a_ 4 _i, a_ 4 _b, laneid, i_ ) 15: **end for** 16: % _stc.load.a rA_ [0 _∼_ 7]// Load 16 _×_ 16 block data of matrix _A_ 17: _rx ← load value x_ ( _laneid_ ) 18: % _stc.numeric.mv rA_ [8 _∼_ 15] _, rx, ry_ // SDPU execute T4 tasks 19: _j_ + = 2 20: **end while** 21: _shfl gather_ ( _ry_ ) 22: _write back_ ( _ry, row, laneid_ ) 

**Algorithm 2:** SpMM/SpGEMM with Uni-STC@FP64 

|**lg**|**orithm 2:**SpMM/SpGEMM with Uni-STC@FP64|
|---|---|
|1: <br>2: <br>3:|_warpid ←threadIdx.x >>_ 5<br> _laneid ←threadIdx.x_&31<br> _row ←warpRowId_[_warpid_]|
|4: <br>5:<br>6:|**for** _j ←Arow_<br>_ptr_[_row_] _→Arow_<br>_ptr_[_row_+ 1] **do**<br>_Acol ←Aci_[_j_]<br>_A_16_b ←A_16_b_<br>_ptr_[_j_]|
|7:<br>8:|_Av_[8] _←load_<br>_v_(_row, Acol, laneid_)<br>_Abi ←load_<br>_bi_(_row, Acol, laneid_)|
|9:|% _stc.load.a Av_[0 _∼_7]// Load 16_×_16 block data of matrix _A_|
|10:<br>11:|**for** _Bj ←Brow_<br>_ptr_[_Acol_] _→Brow_<br>_ptr_[_Acol_+ 1] **do**<br>_Bcol ←Bcol_<br>_idx_[_Bj_]|
|12:|_B_16_b ←B_16_b_<br>_ptr_[_Bj_]|
|13:<br>14:<br>15:|**if** _A_16_b × B_16_b_ **and** _bfind_(_Bcol_) **then**<br>_C_16_b ←Ccol_<br>_idx_[_bfind_<br>_result_]<br>_Bbi ←load_<br>_bi_(_Acol, Bcol, laneid_)|
|16:<br>17:<br>18:|_Cbi ←load_<br>_bi_(_row, Bcol, laneid_)<br>% _stc.load.meta_<br>_mm A_16_b, B_16_b, C_16_b, Abi, Bbi, Cbi_<br>% _stc.task_<br>_gen.mm_// TMS and DPG generate T3 and T4 tasks|
|19:<br>20:<br>21:|_Bv_[8] _←load_<br>_v_(_Acol, Bcol, laneid_)<br>_Cv_[8] _←load_<br>_v_(_row, Bcol, laneid_)<br>% _stc.numeric.mm Bv_[0 _∼_7]_, Cv_[0 _∼_7]// SDPU execution|
|22:<br>23:|_accumulate_<br>_c_(_row, Bcol, laneid, Cv_)<br>**end if**|
|24:|**end for**|
|25:|**end for**|



## V. UNI-STC DATAFLOW 

This section details the software-hardware co-design of UniSTC, focusing on the dataflow from both the software and hardware perspectives. 

## _A. Software Dataflow_ 

Based on the BBC format and the UWMMA instruction set, we design the four sparse kernels at the software level. The implementation of SpMV and SpMSpV is presented in Algorithm 1. During execution, Uni-STC computes the multiplication of two blocks of matrix _A_ and corresponding vectors, accumulating the results in each thread’s ry register. Finally, shfl_gather is used to accumulate the results in the first 16 threads, and then written back to global memory. For SpMM and SpGEMM, detailed in Algorithm 2, the dataflow leverages the first-level CSR structure within the BBC format. This structure facilitates the scheduling of T1 tasks through a rowby-row outer product formulation ( _Ci∗_ + = _Aik × Bk∗_ ). 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [516 x 197] intentionally omitted <==**

**----- Start of picture text -----**<br>
original matrix  B compressed matrix  B MAC Utilisation Proportions100<br>DS-STC RM-STC row 1row 2 Uni-STC 0 (0) 80<br>#Cycles   8   #Cycles   6   8 #Cycles  4   1 C [0][0] A [0][0] B [0][0] 6040 50.00<br>UtilisationMAC 37.50%  UtilisationMAC 50.00%  UtilisationMAC 75.00% 23 C00  [m][n] += ∑  k A ⋅ 00 B [m][k] 00  [k][n]  200 0.0025.0025.00<br>gather<br>original matrix  A row 6 compressed matrix  A layer  k =0 layer  k =1 MAC Utilisation Intervals (%)cycle 1<br>Step 1 Step 2 4 11 22 8 2121 Step 1 Step 2 4 (0)819415   (0) at  Write conflict 15on  C 14[0][0]: k  = 0, 1511 (1)2  428 412  42  4 1 12 2  16 1 30<br>col 6 Step 3 Step 4 4 4 11 ！ 12  (2) atone postponedone executed,   k 6 = 2 (0) (1) (2)SDPU(3) (4) (5) (6) (7)<br>MAC Utilisation Proportions100 scatter 4 4 MAC Utilisation Proportions100 scatter 4 4 (5)0 1 2 3 layer  k =2 layer  k =3 scatter<br>(2) (3) (4) (6) (7)<br>80 75.00 80 4 1 2 1 0<br>60 60 50.00 C [1][0] A [1][2] B [2][0] (5)6 (6)1 (7)3<br>4020 0.00 0.00 25.00 4020 16.67 16.6716.67 CCC101010 [0][1] += [1][0] += [0][0] +=  AAA121212 [0][1]×[1][0]×[0][1]× BBB202020 [1][1][1][0][1][0] 231 (3)(0)44 (1)(4)11 (5)(2)22 (0)2 (1)4<br>0 0<br>MAC Utilisation Intervals (%) matrix  C MAC Utilisation Intervals (%) matrix  C ( Intermediate products are written in the middle of each block )Each block represents a 2×2×2 T3 task matrix  C<br>(0,25](25,50](50,75](75,100]<br>(0,25](25,50](50,75](75,100] (0,25](25,50](50,75](75,100]<br>Proportion of Total Cycles (%)<br>gather<br>network: 16 × 64 network: 4 × 4 × 8 network: 8 × 4 × 4<br>Proportion of Total Cycles (%) Proportion of Total Cycles (%)<br>**----- End of picture text -----**<br>


Fig. 14: Comparison of DS-STC, RM-STC and our Uni-STC on a downsized 8( _M_ ) _×_ 8( _N_ ) _×_ 8( _K_ ) T1 task. 

The ‘warpRow’, ‘warpIndex’, and ‘warpRowId’ variables are used in the preceding algorithms to implement a static load-balancing scheme, which configures the data processing range ofeach warp. 

For dense computations, the structural information of dense vector and matrix is stored in GPU memory (a total of 96B). This information is loaded into registers with a single read operation at the start of the computation. 

## _B. Hardware Dataflow_ 

The hardware dataflow of an STC is defined by the interplay between its task preparation method and its computational unit architecture, which ultimately dictates performance. To illustrate the resulting differences, we present a case study in Fig. 14 that compares three STCs processing a downsized 8( _M_ ) _×_ 8( _N_ ) _×_ 8( _K_ ) T1 task. The comparison focuses on two key stages: task preparation and task execution. For a fair comparison, each STC is equipped with 16 multipliers and their associated adders. 

_1) Task preparation:_ The goal of task preparation is to decompose large T1 tasks into smaller T3 tasks compatible with the computational units. DS-STC and RM-STC achieve this using a hybrid software-hardware approach that reduces hardware overhead. As illustrated in Fig. 14, this process begins in software, where the compiler expands a T1 task into intermediate T2 sub-instructions, represented by the redhighlighted box. This stage leverages the GPU front-end’s skipping mechanism for coarse-grained sparsity support. Subsequently, in hardware, any T2 task that still exceeds the computational unit’s capacity is further subdivided into T3 tasks for sequential execution. 

Although this collaborative method reduces hardware overhead, its core limitation is that T2 task splitting is rigidly tied to the computational unit’s structure. Within STC, there is a lack of mechanisms to address the load imbalance of T2 tasks caused by irregularity, which typically results in relatively low 

MAC utilisation. In contrast, Uni-STC adopts a more flexible strategy. Although it initially divides T1 tasks into fixed-size T3 tasks, it provides a dynamic task fusion mechanism to mitigate load imbalance. 

Fig. 14 highlights the four-layer T3 tasks for Uni-STC, where the diagram’s notation is interpreted as follows. The number in the center of each block denotes the count of intermediate products, the number in the upper-left corner identifies the assigned DPG, and the green blocks signify multiple T3 tasks that are concurrently executed on the SDPU during the first cycle. 

_2) Task execution:_ Achieving effective task fusion is nontrivial. As illustrated in Fig. 14, the approaches in DS-STC and RM-STC suffer from two key inefficiencies related to the MAC array. First, T2 tasks can be too small to fully utilise the array’s resources, leading to wasted performance (e.g., RM-STC). Second, even sufficiently large tasks may have shapes that are incompatible with the array, which prevents the concatenation of multiple T3 tasks and thus causes inefficiency (e.g., DS-STC). This architectural challenge is compounded by the complexity of implementing a hardware-based, multidimensional knapsack solver on resource-constrained STCs. 

Uni-STC addresses these fusion challenges by decomposing T3 tasks into even finer-grained vector dot-product operations (T4 tasks). As shown in Fig. 14, a 2( _M_ ) _×_ 2( _N_ ) _×_ 2( _K_ ) T3 task is broken down into 1( _M_ ) _×_ 1( _N_ ) _×_ 2( _K_ ) T4 tasks. The concatenation of these vector tasks is accomplished using simple prefix sums and shift units, thereby accelerating computation on the SDPU. Consequently, this approach boosts Uni-STC’s utilisation to 75%, a significant improvement over the 50% of RM-STC and 37.5% of DS-STC. 

In summary, Uni-STC adopts a software-hardware codesigned dataflow: BBC and UWMMA express and schedule the four kernels in software, while the hardware dataflow (TMS _→_ DPG _→_ SDPU) enables efficient task preparation and execution to improve utilisation under irregular sparsity. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 123] intentionally omitted <==**

**----- Start of picture text -----**<br>
BSR BSR BBC CSR<br>(16x16) (4x4) (this work) (baseline)<br>1<br>10<br>0<br>10<br>1<br>10<br>2<br>10<br>[2 [0] 2 [1] )[2 [1] 2 [2] )[2 [2] 2 [3] )[2 [3] 2 [4] )[2 [4] 2 [5] )[2 [5] 2 [6] )[2 [6] 2 [7] )[2 [7] 2 [8] )<br>Non-zero elements per block<br>Space<br>Reduction<br>**----- End of picture text -----**<br>


Fig. 15: Space reduction of the three formats BSR (4 _×_ 4), BSR (16 _×_ 16) and our BBC over the baseline CSR. 

## VI. EVALUATION 

## _A. Experimental Setup_ 

On the dataset side, we evaluate SpMV, SpMSpV, and SpMM using all 2893 matrices from SuiteSparse [10], and SpGEMM ( _C_ = _A_[2] ) using its 2126 square matrices. For DNN inference, we evaluate ResNet-50 and Transformer [74] models using the 302 weight matrices from DLMC [23] at 70% and 98% sparsity. Additionally, input vectors for SpMSpV are randomly generated with 50% sparsity, and the number of columns in matrix _B_ for SpMM is set to 64. 

On the software side, we compare our BBC format with the conventional CSR and BSR (with block sizes of 4 _×_ 4 and 16 _×_ 16) to assess the memory efficiency derived from its unique sparse matrix structure. 

On the hardware side, we build upon Accel-Sim [38] with added support for asynchronous memory access, integrating our STC simulator to support GAMMA [93], SIGMA [66], Trapezoid [87], NV-DTC [60] (A100’s original Tensor Core), DS-STC [78], [92], RM-STC [30], and our work Uni-STC. 

To rigorously evaluate the architectural benefits of Uni-STC under configurations ‘64 MAC@FP64 and 128 MAC@FP32’, we establish a fair comparison by aligning the theoretical compute throughput of all designs. To this end, we adopt SIGMA’s PE design and scale the MAC arrays of all evaluated architectures, including GAMMA and Trapezoid, accordingly. 

We assess three key metrics: performance, energy, and area. Performance is measured using a unified software invocation of a T1 task with dimensions 16( _M_ ) _×_ 16( _N_ ) _×_ 16( _K_ ). Energy consumption is extrapolated from register activity following the Sparseloop methodology [80]. Uni-STC’s chip area is analyzed using yosys [79], FreePDK45 [62], and CACTI7 [3]. 

## _B. Data Structure Comparison_ 

Fig. 15 compares the memory overhead of our BBC format against the conventional CSR and BSR (with the block sizes of 4 _×_ 4 and 16 _×_ 16) across all 3195 test matrices. The memory usage of the BBC format shrinks as the number of nonzeros per block (NnzPB) increases, becoming the most efficient for 2585 matrices (where NnzPB _>_ 3.57) and delivering savings of up to 15 _._ 26 _×_ over CSR. Conversely, the BSR format typically requires more storage than CSR. 

TABLE VI: Comparison of STCs. MMA instruction task size: 16 _×_ 16 _×_ 16, MAC array size: 128@FP32 or 64@FP64. 

**==> picture [261 x 256] intentionally omitted <==**

**----- Start of picture text -----**<br>
T3 Task Size<br>T4 Task Size<br>STC (128 or 64 MACs)<br>( M × N × K )<br>( M × N × K )<br>GAMMA [93] 16  × (8 or 4)  ×  1<br>SIGMA [66] 1  × (8 or 4)  ×  16<br>TrIP: 16  ×  (4 or 2)  ×  2<br>Trapezoid [87] TrGT: 16  ×  4  ×  (2 or 1) Same as T3<br>TrGS: 8  ×  4  × (4 or 2) Task Size<br>NV-DTC [60] (8 or 4)  ×  4  ×  4<br>DS-STC [78], [92] 8  × (16 or 8)  ×  1<br>RM-STC [30] (16 or 8)  ×  4  ×  2<br>Uni-STC (this work) 4  ×  4  ×  4 1  ×  1  ×  4<br>GAMMA SIGMA Trapezoid Uni-STC (this work)<br>NV-STC DS-STC RM-STC<br>1.0 1.0 1.0<br>A Dense A 50% Sparse A 80% Sparse<br>0.8 0.8 0.8<br>0.6 0.6 0.6<br>0.4 0.4 0.4<br>0.2 0.2 0.2<br>0.0 0.0 0.0<br>10 20 30 40 50 60 70 80 90 10 20 30 40 50 60 70 80 90 10 20 30 40 50 60 70 80 90<br>B-sparsity(%) B-sparsity(%) B-sparsity(%)<br>MAC Utilisation (%)<br>**----- End of picture text -----**<br>


Fig. 16: MAC utilisation for GAMMA, SIGMA, Trapezoid, DS-STC, RM-STC and Uni-STC (128 MAC@FP32). 

The one-time format conversion overhead is modest, comparable to the execution time of a few hundred SpMV operations. On a 64-core AMD EPYC 7702 CPU, this conversion takes less than 1000 ms, while on an NVIDIA A100 GPU, the overhead is less than 100 ms. This initial cost can be effectively amortized and becomes negligible in iterative applications such as GNN training and linear solvers. 

## _C. Hardware Comparison_ 

Table VI details the configurations of all evaluated STCs. For multi-mode architectures like SIGMA and Trapezoid, we select their best-performing configurations. Since our implementations of GAMMA, SIGMA, and Trapezoid are specifically adapted for a fair throughput comparison, which do not accurately reflect the original designs. As their energy consumption and energy efficiency are both lower than RMSTC, in this section, our analysis against these three architectures focuses solely on performance. 

_1) Comparison using random matrices:_ Following the methodology of RM-STC, we first evaluate MAC utilisation using random 8192 _×_ 8192 matrices with varying sparsity. 

As shown in Fig. 16, Uni-STC achieves average speedups of 1 _._ 67 _×_ , 1 _._ 73 _×_ , and 1 _._ 13 _×_ over GAMMA, SIGMA, and Trapezoid, respectively. The performance gain over GAMMA stems from Uni-STC’s ability to bypass empty rows, a task difficult for GAMMA’s blocking approach. The advantage over SIGMA is due to its effective handling of dual-sided sparsity, whereas SIGMA’s modes are either limited to single-sided sparsity or incur high transmission overhead. The speedup 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [516 x 214] intentionally omitted <==**

**----- Start of picture text -----**<br>
DS-STC (baseline, always 1.0 ×  in the sub-figures) RM-STC Uni-STC (this work)<br>6 3 3<br>4 2 2<br>2 1 1<br>0 0 0<br>4 4 10<br>2 2 5<br>0 0 0<br>21 8 12<br>14 6 8<br>4<br>7 2 4<br>0 0 0<br>SpMV SpMSpV SpMM SpGEMM Inference (Dense) Inference (Sparse)<br>Speedup<br>Energy<br>Reduction<br>Energy<br>Efficiency<br>consph crankseg_2 shipsec1 cant opt1 pdb1HYS pwtk gupta3 consph crankseg_2 shipsec1 cant opt1 pdb1HYS pwtk gupta3 consph crankseg_2 shipsec1 cant opt1 pdb1HYS pwtk gupta3 consph shipsec1 crankseg_2 cant opt1 pdb1HYS pwtk gupta3 ResNet50-1 ResNet50-2 ResNet50-3 ResNet50-4 Transformer-1 Transformer-2 Transformer-3 Transformer-4 ResNet50-1 ResNet50-2 ResNet50-3 ResNet50-4 Transformer-1 Transformer-2 Transformer-3 Transformer-4<br>**----- End of picture text -----**<br>


Fig. 17: Comparison of speedup, energy consumption, and energy efficiency of four sparse kernels, as well as ResNet50 and Transformer inference on DS-STC, RM-STC, and Uni-STC. The value after the model name denotes the layer number. Among them, the four sparse kernels use 64 MAC@FP64, and the DNN inference uses 128 MAC@FP32. 

**==> picture [253 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
Read A Read B Write C<br>1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0<br>DS-STCRM-STCUni-STC DS-STCRM-STCUni-STC DS-STCRM-STCUni-STC DS-STCRM-STCUni-STC<br>consph shipsec1 crankseg_2 cant<br>1.0 1.0 1.0 1.0<br>0.5 0.5 0.5 0.5<br>0.0 0.0 0.0 0.0<br>DS-STCRM-STCUni-STC DS-STCRM-STCUni-STC DS-STCRM-STCUni-STC DS-STCRM-STCUni-STC<br>Normalized Energy<br>**----- End of picture text -----**<br>


Fig. 18: Energy consumption of I/O (reading _A_ and _B_ , and writing _C_ ) in SpGEMM on the eight matrices. 

**==> picture [253 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
DS-STC RM-STC Uni-STC (this work)<br>1.0<br>0.5<br>0.0 2 0 2 4 6<br>0.10 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0<br>0.05<br>0.00 consph2 shipsec1 crankseg_20 cant 2 opt1 pdb1HYS4 pwtk 6 gupta3<br>Traffic<br>(a) Network<br>Scale<br>(b) Average<br>**----- End of picture text -----**<br>


Fig. 19: The data traffic and average network scale when writing matrix _C_ . 

against Trapezoid is attributed to Uni-STC’s global task scheduling, which avoids the potential load imbalances found in Trapezoid’s grouped MAC array design. 

Uni-STC also demonstrates superior MAC utilisation compared with NV-DTC, DS-STC, and RM-STC by factors of 

TABLE VII: Information of the eight representative matrices. The column #inter-prod/blk represents the average number of intermediate products per T1 task during SpGEMM computation, with a maximum value of 16 _×_ 16 _×_ 16 = 4096. 

|**Matrix** _A_|**n(**_A_**)**|**nnz(**_A_**)**|**plot**|**nnz(**_C_**)**|**#inter-prod/blk**|
|---|---|---|---|---|---|
|consph|83K|6.0M||26.5M|164.9|
|shipsec1|140K|7.8M||24.1M|189.5|
|crankseg<br>2|64K|14.1M||104.6M|198.5|
|cant|62K|4.0M||17.4M|280.2|
|opt1|15K|1.9M||8.2M|506.4|
|pdb1HYS|36K|4.3M||19.6M|517.2|
|pwtk|218K|11.6M||32.8M|548.3|
|gupta3|17K|9.3M||270.9M|1154.1|



2 _._ 89 _×_ , 1 _._ 89 _×_ , and 1 _._ 39 _×_ , respectively. This superiority stems from its finer-grained task parallelism and stronger sparsity adaptation. In contrast, NV-DTC lacks sparsity adaptation, DSSTC’s performance is constrained by dual-sided sparsity, and RM-STC is particularly sensitive to the sparsity of matrix _A_ . 

In dense computation scenarios, all DTC/STCs achieve 100% MAC utilisation, but their energy consumption varies. Normalizing to NV-DTC, our Uni-STC achieves a 0 _._ 94 _×_ energy reduction, outperforming both DS-STC (0 _._ 67 _×_ ) and RM-STC (0 _._ 83 _×_ ). This advantage arises because DS-STC and RM-STC incur additional overhead for data reuse and intermediate transfers. In contrast, Uni-STC activates only two DPGs, preserving a data movement pattern consistent to NV- 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [516 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
DS-STC RM-STC Uni-STC (this work)<br>100<br>50<br>0<br>20<br>0 2 [0] 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] 2 2 2 1 2 [0] 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [4] 2 [5] 2 [6] 2 [7] 2 [8] 2 [9] 2 [10] 2 [11] 2 22 12 [0] 2 [1] 2 [2] 2 [3] 2 [4] 2 [5] 2 [6] 2 [7] 2 [8] 2 [9] 2 [10] 2 [11]<br>SpMV SpMSpV SpMM SpGEMM<br>Mac<br>Utilisation<br>Energy<br>Efficiency<br>**----- End of picture text -----**<br>


Fig. 20: Performance distribution of the three STCs and four kernels on matrices from SuiteSparse. The x-axis denotes the average number of intermediate products per T1 task. Energy efficiency analyses use DS-STC as the baseline, and is calculated as ‘speedup _×_ energy reduction’. 

DTC. The minor additional energy in Uni-STC is attributed to its task scheduling within the TMS and DPG. The breakeven point is reached when matrix _A_ is dense and matrix _B_ ’s sparsity is below 85%, at which point Uni-STC’s energy efficiency becomes comparable to that of NV-DTC. 

_2) Comparison using real world matrices:_ To further highlight the performance differences among STCs arising from real-world sparse patterns, we select eight matrices from SuiteSparse, as listed in Table VII, to compare the four sparse kernels, and we use DLMC model data to evaluate the inference effects of both dense and sparse weights. Fig. 17 presents the speedup, energy reduction, and energy efficiency of UniSTC and RM-STC, normalized to DS-STC as the baseline. The results consistently show that Uni-STC’s superior performance and lower energy consumption translate to significantly higher overall energy efficiency. 

For SpMV and SpMSpV kernels: About performance, (1) In SpMV, the MAC array structures of DS-STC and RM-STC limit their utilisation to below 12.5% and 25%, respectively. In contrast, Uni-STC’s fine-grained task parallelism yields speedups of 5 _._ 21 _×_ over DS-STC and 2 _._ 74 _×_ over RM-STC. (2) In SpMSpV, RM-STC’s MAC utilisation drops below 12.5% as the input vector _x_ becomes sparser. Uni-STC uses the SDPU to achieve speedups of 5 _._ 25 _×_ and 5 _._ 50 _×_ . About energy, (1) For SpMV, Uni-STC reduces energy by 2 _._ 76 _×_ compared to DS-STC and 1 _._ 01 _×_ compared to RM-STC by reusing vector _x_ data and minimizing intermediate product transfers, delivering average energy efficiency gains of 14 _._ 34 _×_ and 2 _._ 77 _×_ . (2) For SpMSpV, the energy reduction further improves to 3 _._ 06 _×_ and 1 _._ 72 _×_ , achieving average energy efficiency gains of 15 _._ 97 _×_ and 9 _._ 41 _×_ . 

For SpMM, SpGEMM, and DNN inference (with convolution treated as SpGEMM), Uni-STC consistently outperforms the baselines. DS-STC exhibits poor energy efficiency due to its coarse-grained partitioning and lack of task parallelism. In comparison, Uni-STC achieves energy efficiency gains of 1 _._ 74 _×_ , 2 _._ 21 _×_ , 1 _._ 37 _×_ , and 1 _._ 51 _×_ over RM-STC for these four kernels, respectively. About performance, (1) For SpMM and dense DNN inference, Uni-STC’s fine-grained partitioning, which leverages sparsity in matrix _A_ , delivers 1 _._ 53 _×_ and 1 _._ 35 _×_ speedups over RM-STC, which is constrained by a fixed 4-cycle task execution. (2) For SpGEMM and sparse DNN inference, Uni-STC adapts to the sparse distribution to 

TABLE VIII: Comparison of performance ( _P_ ), energy consumption ( _E_ ), and energy efficiency ( _E × P_ ) of STCs on the SuiteSparse Matrix Collection. 

|**Compared**<br>**With**|**SpMV**<br>_P_<br>_E_<br>_E × P_|**SpMV**<br>_P_<br>_E_<br>_E × P_|**SpM**|
|---|---|---|---|
|||_E × P_|_P_<br>_E_|
|**DS-STC**<br>64 MAC@FP64<br>Av<br>Ma|er<br>**3.76**<br>2.02|7.59|**4.18**<br>3.1|
||x<br>**16.00**<br>5.47|27.06|**28.76**<br>6.7|
|**DS-STC**<br>128 MAC@FP32<br>Av<br>Ma|er<br>3.58<br>**2.79**|**9.89**|**4.18**<br>**4.2**|
||x<br>**16.00**<br>**7.41**|**30.79**|**28.76**<br>**9.1**|
|**RM-STC**<br>64 MAC@FP64<br>Av<br>Ma|er<br>1.47<br>1.00|1.48|3.39<br>1.9|
||x<br>3.96<br>2.71|5.07|13.99<br>4.7|
|**RM-STC**<br>128 MAC@FP32<br>Av<br>Ma|er<br>1.39<br>1.37|1.91|3.39<br>2.6|
||x<br>3.33<br>3.67|6.68|13.99<br>6.5|
||**SpMM**<br>_P_<br>_E_<br>_E × P_||**SpGE**|
|||_E × P_|_P_<br>_E_|
|**DS-STC**<br>64 MAC@FP64<br>Av<br>Ma|er<br>**3.07**<br>1.51|**4.17**|2.40<br>1.9|
||x<br>**8.00**<br>5.61|**20.66**|**16.00**<br>5.6|
|**DS-STC**<br>128 MAC@FP32<br>Av<br>Ma|er<br>2.09<br>**1.89**|3.77|**2.50**<br>**2.5**|
||x<br>**8.00**<br>**6.60**|15.94|**16.00**<br>**7.2**|
|**RM-STC**<br>64 MAC@FP64<br>Av<br>Ma|er<br>2.52<br>0.77|1.84|1.45<br>1.3|
||x<br>7.15<br>1.80|9.19|5.20<br>3.5|
|**RM-STC**<br>128 MAC@FP32<br>Av<br>Ma|er<br>2.44<br>0.94|2.29|1.23<br>1.7|
||x<br>7.18<br>2.09|12.48|3.40<br>4.9|



maintain speedups of 1 _._ 88 _×_ and 1 _._ 48 _×_ , whereas RM-STC struggles with dual-matrix sparsity. About energy, as illustrated in Fig. 18 and 19, Uni-STC’s energy savings are substantial. It reduces the energy for writing matrix _C_ by 6 _._ 5 _×_ compared to DS-STC, resulting in lower overall consumption than both DS-STC and RM-STC. This reduction is primarily driven by two factors: a smaller dynamic network scale (a 2 _._ 36 _×_ contribution) and reduced data traffic from the SDPU (an additional 2 _._ 75 _×_ contribution). 

Moreover, the different energy efficiency improvements on ResNet50 and Transformer demonstrate Uni-STC’s ability to perceive sparse loads: (1) In ResNet50, because the images are usually sparse after preprocessing, Uni-STC consumes more energy to enable multiple DPG to improve the throughput of SDPU. (2) In Transformer, because the load is relatively dense, Uni-STC activates only a single DPG in most cycles, saving nearly 2 _×_ energy consumption compared to RM-STC. 

We extend our comparison to all SuiteSparse matrices for four key kernels. As detailed in Table VIII, Uni-STC consistently achieves higher energy efficiency than the stateof-the-art RM-STC. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [253 x 220] intentionally omitted <==**

**----- Start of picture text -----**<br>
SpMV SpGEMM All<br>6<br>5 4.84<br>4.15 3.98<br>4<br>32 1.78 1.42 2.02 1.54 2.32 1.49 2.07 2.63 2.46<br>1 1.12 0.84 1.06<br>0<br>SIGMA GAMMA RM-STC Trapezoid Uni-STC (ours)<br>Fig. 21: Speedup on AMG compared to DS-STC.<br>DS-STC RM-STC Uni-STC (4) Uni-STC (8) Uni-STC (16)<br>16<br>12<br>8<br>4<br>0<br>12<br>8<br>4<br>0<br>SpMV SpMSpV SpMM SpGEMM<br>(vs. DS-STC)<br>Geomean Speedup<br>EED<br>(FP32)<br>EED<br>(FP64)<br>**----- End of picture text -----**<br>


Fig. 22: Comparison of energy efficiency density (EED) normalized to DS-STC. 

We measure computational density by calculating the average number of intermediate products contained within each T1 task. Fig. 20 illustrates the performance of the three STCs as a function of this density. For extremely sparse matrices, most T1 tasks complete in a single cycle. Consequently, the MAC utilisation across the three STCs is nearly identical, and Uni-STC conserves energy by activating only a single DPG. As block density increases, Uni-STC activates more DPGs to boost MAC utilisation, yielding higher performance in SpMM and SpGEMM. When matrices become even denser, the MAC utilisation for all STCs approaches saturation, at which point Uni-STC again saves energy by deactivating most DPGs. 

## _D. Case study: AMG_ 

This experiment adapts an existing AMG solver [14], [53], a key tool in scientific computing, by substituting its original FP64 dense Tensor Core calculations with our STC designs. We then quantitatively evaluate the speedup achieved on the SpMV and SpGEMM kernels, using the DS-STC as the baseline. 

As illustrated in Fig. 21, Uni-STC demonstrates superior performance. In contrast, other STCs—while effective on random matrices—are hampered by the irregularity of realworld sparse patterns, such as elements being concentrated on the diagonal or within specific rows and columns. For SpMV, architectural limitations in MAC arrays constrain DSSTC, SIGMA, GAMMA, and RM-STC, impeding effective acceleration. For SpGEMM, the absence of fine-grained task partitioning restricts gains for DS-STC, GAMMA, and RMSTC. Similarly, SIGMA achieves only marginal SpGEMM improvements; despite its focus on data reuse, it suffers from suboptimal MAC utilisation. Finally, although Trapezoid achieves a 4 _._ 15 _×_ SpMV speedup via dot-product acceleration, 

TABLE IX: Area breakdown of the core modules in UniSTC. The percentage represents the total area for a projected deployment of 432 Uni-STCs (4 per SM _×_ 108 SMs) on an NVIDIA A100 GPU, relative to its 826 _mm_[2] die area. 

|**Module Name**|**Area (**_mm_2**)**|**Percentage (%)**|
|---|---|---|
|Benes & MUX networks|0.002|0.1|
|TMS & DPG|0.012|0.6|
|Extra adders in SDPU|0.018|0.94|
|Meta data buffer (144B)|0.0005|0.03|
|Accumulate buffer (1KB)|0.003|0.15|
|Matrix _A_ buffer (2KB)|0.007|0.3|
|**Total Overhead**|**0.0425**|**2.12**|



real-world irregularity exacerbates load imbalances across its PE rows, limiting it to a modest 1 _._ 06 _×_ speedup for SpGEMM. Conversely, Uni-STC effectively mitigates these irregularities, delivering notable speedups of 4 _._ 84 _×_ for SpMV and 2 _._ 46 _×_ for SpGEMM. 

## _E. Energy Efficiency Density_ 

We introduce the Energy Efficiency Density (EED) metric to holistically evaluate Uni-STC and guide the determination of the optimal number of DPGs. This metric quantifies the trade-offs among performance, energy consumption, and area, and is defined as the normalized energy efficiency per area:[Reduction] _EED_ =[S][p][eedu] Area[p] _[×]_[Ener] Overhead[gy] . A higher EED value signifies greater energy efficiency achieved per unit of area. 

Fig. 22 presents a detailed comparative analysis of the EED for the three STCs, revealing that Uni-STC consistently outperforms both DS-STC and RM-STC across the evaluated workloads. The analysis shows contrasting trends as the number of DPGs increases from 4 to 16: the EED for SpMV and SpMSpV gradually decreases, while for SpMM and SpGEMM, it conversely exhibits an upward trend. This tradeoff analysis identifies DPG=8 as an balanced configuration. At this setting, the EED for SpMM and SpGEMM nearly matches that of the DPG=16 configuration, representing a significant 1 _._ 37 _×_ improvement over DPG=4. Concurrently, the EED reduction for SpMV and SpMSpV is minimal—only 1 _._ 1 _×_ lower than at DPG=4. Based on this evidence, we establish DPG=8 as the default configuration for Uni-STC. 

## _F. Area Analysis and Time Budget_ 

We synthesize the Uni-STC@FP64 (configured with 8 DPGs) using Yosys [79] and the FreePDK45 library [62]. Synthesis results indicate that the critical path lies within the “Execution & Write _C_ ” stage, which satisfies the 1.5 GHz timing constraint. Regarding area estimation, the buffers in Uni-STC are modeled using CACTI 7 [3] at 45 nm and scaled to 7 nm technology. For logic area, we aggregate the TMS and DPG due to their structural similarities. Furthermore, since the SDPU is derived from the original Tensor Core with additional adders, we only account for its incremental overhead. Table IX details the area breakdown of these specific modules. Ideally, the total area overhead for 432 Uni-STC units is approximately 2.12% of the 826 mm[2] die area of an NVIDIA A100 GPU [60]. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

## VII. RELATED WORK 

## _A. Sparse Kernel Acceleration_ 

Prior works have adopted diverse strategies to accelerate **SpMV** . On heterogeneous multi-core CPU and GPU platforms, Speculative Segmented Sum [49] achieves higher throughput using speculative computation, CSR5 [47] fosters greater parallelism with tiling, HASpMV [42] leverages heterogeneity-aware formats to improve memory access, TileSpMV [57] promotes data locality through tiled processing, TileSpMSpV [34] uses adaptive kernels on GPUs, and DASP [52] utilises regularized tensor core for acceleration. For HBM-equipped FPGAs, Serpens [72] and Cuper [89] mitigates memory access conflicts through customized dataflows and reordering. In distributed environments, DistSpMV Balanced [55] reconciles computation and communication by means of graph partitioning. 

To accelerate **SpMM** , researchers have pursued several optimization directions. The first is data-centric, VEGETA [33] and TB-STC [44] support hybrid sparsity formats, ASADI [41] applies diagonal compression, and Avalanche [6] improves access patterns via data reordering. The second direction optimizes the computation flow, SPADE [20] and HotTiles [19] reduce data transfer and adapt to sparsity, while GROW [31] balances data locality with parallelism. The third direction leverages specific hardware features, Eureka [22] utilises tensor cores, and Leda [88] optimizes dataflows on FPGAs. 

For **SpGEMM** , various approaches have been proposed to improve performance. In hardware architecture and dataflows, OuterSPACE [63] pioneered the outer-product approach. Trapezoid [87] designs specialized dataflows, NeuraChip [69] uses hash-based decoupling, and TaskFusion [16] enhances data sharing. Other targeted improvements include SGCN [90] enhancing format support, HIRAC [67] improving locality, S2TA [50] supporting dual-sided sparsity, and GoSPA [11] applying intersection computation. Pattern-based and tilelevel optimisations represent another significant direction, explored in works such as SPAGHETTI [29], SpArch [94], DRT [61], GAMMA [93], HARP [39], Tailors [84], DSSTC [78], and RM-STC [30]. For sparse ML workloads, many works have investigated adaptive dataflow strategies, including FEATHER [73], Sparseloop [80], Flexagon [59], ACES [51], FEASTA [95], SPADA [43], CANDLES [24], Sparse Tensor Core [96], SparTen [21], and ExTensor [28]. Acceleration on CPUs and GPUs has also been extensively studied. Liu et al. [46], [48] proposed a foundational four-stage framework. HASpGEMM [8] improves load balancing on heterogeneous cores, while GPU-specific works exploit hardware registers [45]. In particular, TileSpGEMM [58] adopts tiled execution to enhance locality and alleviate load imbalance. Furthermore, approaches like IA-SpGEMM [82], [83] focus on input-aware method selection to adapt to matrix sparsity. 

Several studies propose a unified design to accelerate **multiple kernels** . Early efforts combine pairs of sparse kernels, where VIA [65] improves index matching for SpMV and SpMM, and PruneGNN [25] includes units for both SpMM 

and SpGEMM. Griffin [68] later expands this scope by optimizing resource reuse across dense and sparse matrices. Building on this trend, KAMI [76] unifies dense GEMM with sparse SpMM and SpGEMM, and Siracusa et al. [70] propose a versatile multi-lane architecture. 

**Bitmap-based** compression reduces indexing and bandwidth overhead. This technique is used by SMASH [37] to compress metadata and by Buluc¸ et al. [5] to cut bandwidth. More recent works adapt it for modern hardware, SpInfer [15] and BerryBees [56] design Tensor Core aware encodings, while AmgT [53] uses a bitmap driven format to accelerate both SpMV and SpGEMM. 

## _B. Other Sparsity-Aware Optimisation_ 

Sparsity is also exploited in **PIM** and **ReRAM** accelerators. In the PIM domain, early works like GaaS-X [7] optimize data representation for graph SpMV. Subsequent efforts include SpaceA [81], a dedicated SpMV accelerator, and more recently, pSyncPIM [2], which implements partial synchronous execution. Similarly, ReRAM-based approaches have evolved. Yang et al. [86] leverage activation and weight sparsity. Recently, AmgR [14] and ReCG [13] exploit inmemory computation to further improve performance and energy efficiency of sparse linear solvers. 

In machine learning (ML) workloads, **pruning** is widely applied. Foundational works from Han et al. [26] and Yu et al. [91] propose compressed DNN schemes, while SCNN [64] provides an early accelerator for CNNs. Subsequent efforts optimize data handling. Hanson et al. [27] and Lew et al. [40] improve data reuse, Feinberg et al. [17] reorder weights, Jang et al. [32] search nonzeros, and SIGMA [66] constructs reduction trees. Another direction addresses dynamic sparsity, where DPACS [18], SOFA [77], and Sparse-DySta [12] handle various dynamic patterns, and TensorDash [54] leverages input sparsity. Moreover, SpAtten [75] prunes tokens and heads, and HuffDuff [85] enhances mobile sparse accelerator efficiency. 

## VIII. CONCLUSION 

This paper presents Uni-STC, a unified sparse tensor core accelerating a comprehensive set of sparse kernels. Leveraging the novel BBC format, Uni-STC dynamically generates finegrained tasks, schedules them to improve data reuse, and executes concatenated dot-products. This approach optimizes hardware utilisation while reducing intermediate data movement. Evaluations confirm that Uni-STC delivers significant speedup and energy savings over state-of-the-art designs. 

## IX. ACKNOWLEDGMENTS 

We are very grateful to all reviewers for their invaluable comments and to the shepherd for the constructive guidance. Weifeng Liu is the corresponding author of this paper. This work is partially supported by the National Natural Science Foundation of China (U23A20301, 62372467 and 62202481). We also thank the researchers at the Beijing Institute of Open Source Chip for our helpful discussions. Finally, we appreciate Xin Shi and Yuxiang Pu for their help in the implementation and verification of the Uni-STC. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] K. Asanovic, R. Bodik, J. Demmel, T. Keaveny, K. Keutzer, J. Kubiatowicz, N. Morgan, D. Patterson, K. Sen, J. Wawrzynek, D. Wessel, and K. Yelick, “A view of the parallel computing landscape,” _Communications of the ACM_ , 2009. 

- [2] D. Baek, S. Hwang, and J. Huh, “psyncpim: Partially synchronous execution of sparse matrix operations for all-bank pim architectures,” in _ISCA_ , 2024. 

- [3] R. Balasubramonian, A. B. Kahng, N. Muralimanohar, A. Shafiee, and V. Srinivas, “Cacti 7: New tools for interconnect exploration in innovative off-chip memories,” _TACO_ , 2017. 

- [4] A. Biswas, “Sapphire rapids,” _HCS_ , 2021. 

- [5] A. Buluc¸, S. Williams, L. Oliker, and J. Demmel, “Reduced-bandwidth multithreaded algorithms for sparse matrix-vector multiplication,” in _IPDPS_ , 2011. 

- [6] G. Byeon, S. Kim, H. Kim, S. Han, J. Kim, P. Nair, T. Kang, and S. Hong, “Avalanche: Optimizing cache utilization via matrix reordering for sparse matrix multiplication accelerator,” in _ISCA_ , 2025. 

- [7] N. Challapalle, S. Rampalli, L. Song, N. Chandramoorthy, K. Swaminathan, J. Sampson, Y. Chen, and V. Narayanan, “Gaas-x: Graph analytics accelerator supporting sparse data representation using crossbar architectures,” in _ISCA_ , 2020. 

- [8] H. Cheng, W. Li, Y. Lu, and W. Liu, “Haspgemm: Heterogeneity-aware sparse general matrix-matrix multiplication on modern asymmetric multicore processors,” in _ICPP_ , 2023. 

- [9] J. Choquette, “Nvidia hopper gpu: Scaling performance,” in _HCS_ , 2022. 

- [10] T. A. Davis and Y. Hu, “The university of florida sparse matrix collection,” _ACM Trans. Math. Softw._ , 2011. 

- [11] C. Deng, Y. Sui, S. Liao, X. Qian, and B. Yuan, “Gospa: An energyefficient high-performance globally optimized sparse convolutional neural network accelerator,” in _ISCA_ , 2021. 

- [12] H. Fan, S. I. Venieris, A. Kouris, and N. Lane, “Sparse-dysta: Sparsityaware dynamic and static scheduling for sparse multi-dnn workloads,” in _MICRO_ , 2023. 

- [13] M. Fan, X. Cheng, D. Yang, Z. Jin, and W. Liu, “Recg: Reramaccelerated sparse conjugate gradient,” in _DAC_ , 2024. 

- [14] M. Fan, X. Tian, Y. He, J. Li, Y. Duan, X. Hu, Y. Wang, Z. Jin, and W. Liu, “Amgr: Algebraic multigrid accelerated on reram,” in _DAC_ , 2023. 

- [15] R. Fan, X. Yu, P. Dong, Z. Li, G. Gong, Q. Wang, W. Wang, and X. Chu, “Spinfer: Leveraging low-level sparsity for efficient large language model inference on gpus,” in _EuroSys_ , 2025. 

- [16] Z. Fan, Q. Zhang, P. Abillama, S. Shoouri, C. Lee, D. Blaauw, H.S. Kim, and D. Sylvester, “Taskfusion: An efficient transfer learning architecture with dual delta sparsity for multi-task natural language processing,” in _ISCA_ , 2023. 

- [17] B. Feinberg, B. C. Heyman, D. Mikhailenko, R. Wong, A. C. Ho, and E. Ipek, “Commutative data reordering: A new technique to reduce data movement energy on sparse inference workloads,” in _ISCA_ , 2020. 

- [18] Y. Gao, B. Zhang, X. Qi, and H. K.-H. So, “Dpacs: Hardware accelerated dynamic neural network pruning through algorithm-architecture codesign,” in _ASPLOS_ , 2023. 

- [19] G. Gerogiannis, S. Aananthakrishnan, J. Torrellas, and I. Hur, “Hottiles: Accelerating spmm with heterogeneous accelerator architectures,” in _HPCA_ , 2024. 

- [20] G. Gerogiannis, S. Yesil, D. Lenadora, D. Cao, C. Mendis, and J. Torrellas, “Spade: A flexible and scalable accelerator for spmm and sddmm,” in _ISCA_ , 2023. 

- [21] A. Gondimalla, N. Chesnut, M. Thottethodi, and T. N. Vijaykumar, “Sparten: A sparse tensor accelerator for convolutional neural networks,” in _MICRO_ , 2019. 

- [22] A. Gondimalla, M. Thottethodi, and T. N. Vijaykumar, “Eureka: Efficient tensor cores for one-sided unstructured sparsity in dnn inference,” in _MICRO_ , 2023. 

- [23] Google, “Deep learning matrix collection(dlmc),” 2020. [Online]. Available: https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz 

- [24] S. Gudaparthi, S. Singh, S. Narayanan, R. Balasubramonian, and V. Sathe, “Candles: Channel-aware novel dataflow-microarchitecture codesign for low energy sparse neural network acceleration,” in _HPCA_ , 2022. 

- [25] D. Gurevin, M. Shan, S. Huang, M. A. Hasan, C. Ding, and O. Khan, “Prunegnn: Algorithm-architecture pruning framework for graph neural network acceleration,” in _HPCA_ , 2024. 

- [26] S. Han, X. Liu, H. Mao, J. Pu, A. Pedram, M. A. Horowitz, and W. J. Dally, “Eie: Efficient inference engine on compressed deep neural network,” in _ISCA_ , 2016. 

- [27] E. Hanson, S. Li, H. H. Li, and Y. Chen, “Cascading structured pruning: Enabling high data reuse for sparse dnn accelerators,” in _ISCA_ , 2022. 

- [28] K. Hegde, H. Asghari-Moghaddam, M. Pellauer, N. Crago, A. Jaleel, E. Solomonik, J. Emer, and C. W. Fletcher, “ExTensor: An Accelerator for Sparse Tensor Algebra,” in _MICRO_ , 2019. 

- [29] R. Hojabr, A. Sedaghati, A. Sharifian, A. Khonsari, and A. Shriraman, “Spaghetti: Streaming accelerators for highly sparse gemm on fpgas,” in _HPCA_ , 2021. 

- [30] G. Huang, Z. Wang, P.-A. Tsai, C. Zhang, Y. Ding, and Y. Xie, “Rm-stc: Row-merge dataflow inspired gpu sparse tensor core for energy-efficient sparse acceleration,” in _MICRO_ , 2023. 

- [31] R. Hwang, M. Kang, J. Lee, D. Kam, Y. Lee, and M. Rhu, “Grow: A row-stationary sparse-dense gemm accelerator for memory-efficient graph convolutional neural networks,” in _HPCA_ , 2023. 

- [32] J.-W. Jang, S. Lee, D. Kim, H. Park, A. S. Ardestani, Y. Choi, C. Kim, Y. Kim, H. Yu, H. Abdel-Aziz, J.-S. Park, H. Lee, D. Lee, M. W. Kim, H. Jung, H. Nam, D. Lim, S. Lee, J.-H. Song, S. Kwon, J. Hassoun, S. Lim, and C. Choi, “Sparsity-aware and re-configurable npu architecture for samsung flagship mobile soc,” in _ISCA_ , 2021. 

- [33] G. Jeong, S. Damani, A. R. Bambhaniya, E. Qin, C. J. Hughes, S. Subramoney, H. Kim, and T. Krishna, “Vegeta: Vertically-integrated extensions for sparse/dense gemm tile acceleration on cpus,” in _HPCA_ , 2023. 

- [34] H. Ji, H. Song, S. Lu, Z. Jin, G. Tan, and W. Liu, “Tilespmspv: A tiled algorithm for sparse matrix-sparse vector multiplication on gpus,” in _ICPP_ , 2023. 

- [35] N. P. Jouppi, D. H. Yoon, M. Ashcraft, M. Gottscho, T. B. Jablin, G. Kurian, J. Laudon, S. Li, P. Ma, X. Ma, T. Norrie, N. Patil, S. Prasad, C. Young, Z. Zhou, and D. Patterson, “Ten lessons from three generations shaped google’s tpuv4i: Industrial product,” in _ISCA_ , 2021. 

- [36] N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P. luc Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon, “In-datacenter performance analysis of a tensor processing unit,” in _ISCA_ , 2017. 

- [37] K. Kanellopoulos, N. Vijaykumar, C. Giannoula, R. Azizi, S. Koppula, N. M. Ghiasi, T. Shahroodi, J. G. Luna, and O. Mutlu, “Smash: Codesigning software compression and hardware-accelerated indexing for efficient sparse matrix operations,” in _MICRO_ , 2019. 

- [38] M. Khairy, Z. Shen, T. M. Aamodt, and T. G. Rogers, “Accel-sim: An extensible simulation framework for validated gpu modeling,” in _ISCA_ , 2020. 

- [39] J. Kim, M. Jang, H. Nam, and S. Kim, “Harp: Hardware-based pseudotiling for sparse matrix multiplication accelerator,” in _MICRO_ , 2023. 

- [40] J. S. Lew, Y. Liu, W. Gong, N. Goli, R. D. Evans, and T. M. Aamodt, “Anticipating and eliminating redundant computations in accelerated sparse training,” in _ISCA_ , 2022. 

- [41] H. Li, Z. Li, Z. Bai, and T. Mitra, “Asadi: Accelerating sparse attention using diagonal-based in-situ computing,” in _HPCA_ , 2024. 

- [42] W. Li, H. Cheng, Z. Lu, Y. Lu, and W. Liu, “Haspmv: Heterogeneityaware sparse matrix-vector multiplication on modern asymmetric multicore processors,” in _CLUSTER_ , 2023. 

- [43] Z. Li, J. Li, T. Chen, D. Niu, H. Zheng, Y. Xie, and M. Gao, “Spada: Accelerating sparse matrix multiplication with adaptive dataflow,” in _ASPLOS_ , 2023. 

- [44] J. Liu, S. Zeng, J. Zhao, L. Ding, Z. Wang, J. Li, Z. Zhu, X. Ning, C. Zhang, Y. Wang, and G. Dai, “Tb-stc: Transposable block-wise n:m structured sparse tensor core,” in _HPCA_ , 2025. 

- [45] J. Liu, X. He, W. Liu, and G. Tan, “Register-aware optimizations for parallel sparse matrix-matrix multiplication,” _International Journal of Parallel Programming_ , 2019. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

- [46] W. Liu and B. Vinter, “An efficient gpu general sparse matrix-matrix multiplication for irregular data,” in _IPDPS_ , 2014. 

- [47] W. Liu and B. Vinter, “Csr5: An efficient storage format for crossplatform sparse matrix-vector multiplication,” in _ICS_ , 2015. 

- [48] W. Liu and B. Vinter, “A framework for general sparse matrix-matrix multiplication on gpus and heterogeneous processors,” _Journal of Parallel and Distributed Computing_ , 2015. 

- [49] W. Liu and B. Vinter, “Speculative segmented sum for sparse matrixvector multiplication on heterogeneous processors,” _Parallel Computing_ , 2015. 

- [50] Z.-G. Liu, P. N. Whatmough, Y. Zhu, and M. Mattina, “S2ta: Exploiting structured sparsity for energy-efficient mobile cnn acceleration,” in _HPCA_ , 2022. 

- [51] X. Lu, B. Long, X. Chen, Y. Han, and X.-H. Sun, “Aces: Accelerating sparse matrix multiplication with adaptive execution flow and concurrency-aware cache optimizations,” in _ASPLOS_ , 2024. 

- [52] Y. Lu and W. Liu, “Dasp: Specific dense matrix multiply-accumulate units accelerated general sparse matrix-vector multiplication,” in _SC_ , 2023. 

- [53] Y. Lu, L. Zeng, T. Wang, X. Fu, W. Li, H. Cheng, D. Yang, Z. Jin, M. Casas, and W. Liu, “Amgt: Algebraic multigrid solver on tensor cores,” in _SC_ , 2023. 

- [54] M. Mahmoud, I. Edo, A. H. Zadeh, O. M. Awad, G. Pekhimenko, J. Albericio, and A. Moshovos, “Tensordash: Exploiting sparsity to accelerate deep neural network training,” in _MICRO_ , 2020. 

- [55] H. Mi, X. Yu, X. Yu, S. Wu, and W. Liu, “Balancing computation and communication in distributed sparse matrix-vector multiplication,” in _CCGrid_ , 2023. 

- [56] Y. Niu and M. Casas, “Berrybees: Breadth first search by bit-tensorcores,” in _PPoPP_ , 2025. 

- [57] Y. Niu, Z. Lu, M. Dong, Z. Jin, W. Liu, and G. Tan, “Tilespmv: A tiled algorithm for sparse matrix-vector multiplication on gpus,” in _IPDPS_ , 2021. 

- [58] Y. Niu, Z. Lu, H. Ji, S. Song, Z. Jin, and W. Liu, “Tilespgemm: A tiled algorithm for parallel sparse general matrix-matrix multiplication on gpus,” in _PPoPP_ , 2022. 

- [59] F. M. noz Mart´ınez, R. Garg, M. Pellauer, J. L. Abell´an, M. E. Acacio, and T. Krishna, “Flexagon: A multi-dataflow sparse-sparse matrix multiplication accelerator for efficient dnn processing,” in _ASPLOS_ , 2023. 

- [60] Nvidia, “NVIDIA A100 Tensor Core GPU Architecture,” White Paper, 2020. [Online]. Available: https://images.nvidia.com/aem-dam/en-zz/ Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf 

- [61] T. O. Odemuyiwa, H. Asghari-Moghaddam, M. Pellauer, K. Hegde, P.A. Tsai, N. C. Crago, A. Jaleel, J. D. Owens, E. Solomonik, J. S. Emer, and C. W. Fletcher, “Accelerating sparse data orchestration via dynamic reflexive tiling,” in _ASPLOS_ , 2023. 

- [62] C. Oliveira, M. T. Moreira, R. Guazzelli, and N. L. V. Calazans, “Ascend-freepdk45: An open source standard cell library for asynchronous design,” _ICECS_ , 2016. 

- [63] S. Pal, J. Beaumont, D.-H. Park, A. Amarnath, S. Feng, C. Chakrabarti, H.-S. Kim, D. Blaauw, T. Mudge, and R. Dreslinski, “Outerspace: An outer product based sparse matrix multiplication accelerator,” in _HPCA_ , 2018. 

- [64] A. Parashar, M. Rhu, A. Mukkara, A. Puglielli, R. Venkatesan, B. Khailany, J. Emer, S. W. Keckler, and W. J. Dally, “Scnn: An accelerator for compressed-sparse convolutional neural networks,” _ACM SIGARCH Computer Architecture News_ , 2017. 

- [65] J. Pavon, I. V. Valdivieso, A. Barredo, J. Marimon, M. Moreto, F. Moll, O. Unsal, M. Valero, and A. Cristal, “Via: A smart scratchpad for vector units with application to sparse matrix computations,” in _HPCA_ , 2021. 

- [66] E. Qin, A. Samajdar, H. Kwon, V. Nadella, S. Srinivasan, D. Das, B. Kaul, and T. Krishna, “Sigma: A sparse and irregular gemm accelerator with flexible interconnects for dnn training,” in _HPCA_ , 2020. 

- [67] H. Shabani, A. Singh, B. Youhana, and X. Guo, “Hirac: A hierarchical accelerator with sorting-based packing for spgemms in dnn applications,” in _HPCA_ , 2023. 

- [68] J. H. Shin, A. Shafiee, A. Pedram, H. Abdel-Aziz, L. Li, and J. Hassoun, “Griffin: Rethinking sparse optimization for deep learning architectures,” in _HPCA_ , 2022. 

- [69] K. Shivdikar, N. B. Agostini, M. Jayaweera, G. Jonatan, J. L. Abell´an, A. Joshi, J. Kim, and D. Kaeli, “Neurachip: Accelerating gnn computations with a hash-based decoupled spatial accelerator,” in _ISCA_ , 2024. 

- [70] M. Siracusa, V. Soria-Pardos, F. Sgherzi, J. Randall, D. J. Joseph, M. M. Planas, and A. Armejach, “A tensor marshaling unit for sparse tensor algebra on general-purpose processors,” in _MICRO_ , 2023. 

- [71] A. Smith and N. James, “Amd instinct™mi200 series accelerator and node architectures,” _HCS_ , 2022. 

- [72] L. Song, Y. Chi, L. Guo, and J. Cong, “Serpens: A high bandwidth memory based accelerator for general-purpose sparse matrix-vector multiplication,” in _DAC_ , 2022. 

- [73] J. Tong, A. Itagi, P. Chatarasi, and T. Krishna, “Feather: A reconfigurable accelerator with data reordering support for low-cost on-chip dataflow switching,” in _ISCA_ , 2024. 

- [74] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in _NIPS_ , 2017. 

- [75] H. Wang, Z. Zhang, and S. Han, “Spatten: Efficient sparse attention architecture with cascade token and head pruning,” in _HPCA_ , 2021. 

- [76] H. Wang, Y. Du, S. Li, X. Tian, Q. Sun, and W. Liu, “Kami: Communication-avoiding general matrix multiplication within a single gpu,” in _SC_ , 2025. 

- [77] H. Wang, J. Fang, X. Tang, Z. Yue, J. Li, Y. Qin, S. Guan, Q. Yang, Y. Wang, C. Li, Y. Hu, and S. Yin, “Sofa: A compute-memory optimized sparsity accelerator via cross-stage coordinated tiling,” in _MICRO_ , 2024. 

- [78] Y. Wang, C. Zhang, Z. Xie, C. Guo, Y. Liu, and J. Leng, “Dual-side sparse tensor core,” in _ISCA_ , 2021. 

- [79] C. Wolf, “Yosys open synthesis suite,” https://yosyshq.net/yosys/. 

- [80] Y. N. Wu, P.-A. Tsai, A. Parashar, V. Sze, and J. S. Emer, “Sparseloop: An analytical approach to sparse tensor accelerator modeling,” in _MICRO_ , 2022. 

- [81] X. Xie, Z. Liang, P. Gu, A. Basak, L. Deng, L. Liang, X. Hu, and Y. Xie, “Spacea: Sparse matrix vector multiplication on processing-in-memory accelerator,” in _HPCA_ , 2021. 

- [82] Z. Xie, G. Tan, W. Liu, and N. Sun, “Ia-spgemm: An input-aware autotuning framework for parallel sparse matrix-matrix multiplication,” in _ICS_ , 2019. 

- [83] Z. Xie, G. Tan, W. Liu, and N. Sun, “A pattern-based spgemm library for multi-core and many-core architectures,” _TPDS_ , 2022. 

- [84] Z. Y. Xue, Y. N. Wu, J. S. Emer, and V. Sze, “Tailors: Accelerating sparse tensor algebra by overbooking buffer capacity,” in _MICRO_ , 2023. 

- [85] D. Yang, P. J. Nair, and M. Lis, “Huffduff: Stealing pruned dnns from sparse accelerators,” in _ASPLOS_ , 2023. 

- [86] T.-H. Yang, H.-Y. Cheng, C.-L. Yang, I.-C. Tseng, H.-W. Hu, H.S. Chang, and H.-P. Li, “Sparse reram engine: joint exploration of activation and weight sparsity in compressed neural networks,” in _ISCA_ , 2019. 

- [87] Y. Yang, J. S. Emer, and D. Sanchez, “Trapezoid: A versatile accelerator for dense and sparse matrix multiplications,” in _ISCA_ , 2024. 

- [88] E. Yi, J. Bai, Y. Nie, D. Niu, Z. Jin, and W. Liu, “Leda: Leveraging tiling dataflow to accelerate spmm on hbm-equipped fpgas for gnns,” in _ICCAD_ , 2024, pp. 215:1–215:9. 

- [89] E. Yi, Y. Duan, Y. Bai, K. Zhao, Z. Jin, and W. Liu, “Cuper: Customized dataflow and perceptual decoding for sparse matrix-vector multiplication on hbm-equipped fpgas,” in _DATE_ , 2024, pp. 1–6. 

- [90] M. Yoo, J. Song, J. Lee, N. Kim, Y. Kim, and J. Lee, “Sgcn: Exploiting compressed-sparse features in deep graph convolutional network accelerators,” in _HPCA_ , 2023. 

- [91] J. Yu, A. Lukefahr, D. Palframan, G. Dasika, R. Das, and S. Mahlke, “Scalpel: Customizing dnn pruning to the underlying hardware parallelism,” in _ISCA_ , 2017. 

- [92] C. Zhang, Y. Wang, Z. Xie, C. Guo, Y. Liu, J. Leng, G. Sun, Z. Ji, R. Wang, Y. Xie, and R. Huang, “Dstc: Dual-side sparsity tensor core for dnns acceleration on modern gpu architectures,” _IEEE Transactions on Computers_ , 2024. 

- [93] G. Zhang, N. Attaluri, J. S. Emer, and D. Sanchez, “Gamma: Leveraging gustavson’s algorithm to accelerate sparse matrix multiplication,” in _ASPLOS_ , 2021. 

- [94] Z. Zhang, H. Wang, S. Han, and W. J. Dally, “Sparch: Efficient architecture for sparse matrix multiplication,” in _HPCA_ , 2020. 

- [95] K. Zhong, Z. Zhu, G. Dai, H. Wang, X. Yang, H. Zhang, J. Si, Q. Mao, S. Zeng, K. Hong, G. Zhang, H. Yang, and Y. Wang, “Feasta: A flexible and efficient accelerator for sparse tensor algebra in machine learning,” in _ASPLOS_ , 2024. 

- [96] M. Zhu, T. Zhang, Z. Gu, and Y. Xie, “Sparse tensor core: Algorithm and hardware co-design for vector-wise sparse neural networks on modern gpus,” in _MICRO_ , 2019. 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

## **A. Artifact Appendix** 

## **A.1 Abstract** 

This artifact appendix describes the experimental workflow to reproduce the results presented in the paper “Uni-STC: Unified Sparse Tensro Core” (Paper #313). We provide a containerized environment (Docker) pre-installed with the simulators, scripts, and small-scale datasets. The experiments are categorized into two levels: Fast Verification (approx. 5 hours) for functional validation and Complete Verification (approx. 75 hours) for full reproduction. 

## **A.2 Artifact check-list (meta-information)** 

- **Program:** Python 3.9, Bash Scripts, C++ Simulators. 

- **Compilation:** GCC 9+, OpenMP 4.5+, OpenCV 4.x. 

## **A.4 Installation** 

## **A.4.1 Deployment** 

**1. Download and Decompress.** Download `HPCA-Pap313-AE.tar` from the link[3] . 

**2. Load and Start Container.** Load the image into your local Docker registry and launch the container in the background. Note that if you encounter permission errors, please prepend `sudo` . 

```
$dockerload<HPCA-Pap313-AE.tar
```

```
#Optional:removethetarfiletosavespace
$rmHPCA-Pap313-AE.tar
```

```
$dockerrun-itd--nameHPCA-Pap313hpca-pap313-ae:v2
```

- **Data set:** SuiteSparse Matrix Collection (2,800+ matrices) and DLMC. 

- **Run-time environment:** Ubuntu 22.04 LTS (via Docker). 

- **Hardware:** X86-64 CPU, _≥_ 64 GB DRAM. 

- **Storage:** _≥_ 150 GB (Fast Mode) / _≥_ 500 GB (Complete Mode). 

- **Experiments:** Format overhead analysis, Performance comparison, AMG solver, and Energy Efficiency Density. 

- **Prepare workflow time:** 3 hours to download a 40GB Image. 

## **A.4.2 Initialization** 

Access the container, upgrade python package and execute the initialization script. This script compiles the simulator binaries and checks library integrity. 

```
$dockerexec-itHPCA-Pap313/bin/bash
```

- **Execution time:** Fast mode: 5 hours; complete mode: 75 hours. 

- **Publicly available:** Yes. 

- **Workflow automation framework used:** Yes. 

## **A.3 Description** 

## **A.3.1 How to access** 

We provide a persistent artifact package hosted on Google Drive, which includes: 

1. **Docker Image** ( `HPCA-Pap313-AE.tar`[1] ): Contains the OS, dependencies, small data set, simulators, and plotting scripts. 

2. **Full Dataset** ( `matrix.7z`[2] ): The complete SuiteSparse collection required for complete verification. 

```
(container)$cd/root
```

```
#upgradepackageandcompile
(container)$pip3installpipsetuptoolswheel-U
(container)$pip3installquickstart-rhy-U
(container)$./init.sh
```

**Expected Output:** The initialization is successful if the following logs appear: 

```
[INFO]CompileResNet50(sparse)Succeeded!
[INFO]CompileResNet50(dense)Succeeded!
```

```
[INFO]CompileSimulator(Scheduler=8)Succeeded!
```

## **A.3.2 Hardware dependencies** 

To fully reproduce the results reported in the paper, we recommend the following hardware configuration: 

- **Processor:** X86-64 CPU with at least 16 cores. 

- **Memory:** Minimum 64 GB DRAM is required to load large matrices in the complete dataset. 

- **Disk: 100 GB** for the docker image and fast verification. **600 GB** for the full dataset decompression.ss 

## **A.3.3 Software dependencies** 

The artifact is encapsulated in a Docker container to ensure environment consistency. The host machine requires: 

- **OS:** Linux (Ubuntu 20.04/22.04 recommended). 

## **A.5 Experiment workflow** 

We provide a unified automation tool `qrun` to manage experiments. All commands should be executed in the `/root/Sim` directory: 

## `(container)$ cd /root/Sim` 

_**Note on Pre-computed Results.**_ To enable rapid inspection, we have pre-packaged execution logs and generated figures. This allows the subsequent verification instructions to complete in under **10 minutes** . 

If you prefer to execute the full simulation from scratch to verify the functional reproduction, please clean the pre-existing data using the following commands: 

- **Docker Engine:** Version _≥_ 20.10. 

Inside the container, the environment is pre-configured with: 

- **Compilers:** GCC 11.4, CMake 3.22. 

- **Python Env:** Python 3.10 with necessary libraries. 

- **OpenCV:** Version 4.x for image processing. 

```
#removefiguresandexecutionlogs
(container)$rm/root/Sim/fig/*
(container)$cd/root/Sim/dist&&rmtransformer*.csv
spmv/*.csvspmm/*.csvspmspv/*.csvspgemm/*.csvai/*
```

```
(container)$cd/root/Sim&&rmresnet50/dense/*.csv
reset50/sparse/*.csv
```

1 `https://drive.google.com/file/d/1o_ pdtPdox7aEdRE2e4GtbEPiMFGpPHCu` 

2 `https://drive.google.com/file/d/ 1Pp3BBOvU8nGoB12bb4o3wZs41twiXwXM` 

- 3 `https://drive.google.com/file/d/1o_ pdtPdox7aEdRE2e4GtbEPiMFGpPHCu` 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

## **A.5.1 Part 1: Fast Verification (L1)** 

_Estimated Time: ∼5 hours — Storage: No extra storage required._ 

This mode uses small-scale datasets included in the image to reproduce key figures (Fig. 15–19, 21). 

## **A.6.1 Result Inspection** 

## **Option 1: Export to Host (Recommended)** 

For the best viewing experience and to facilitate comparison with the paper, we recommend copying all generated figures to host. Execute the following command on your **host terminal** : 

- **Task 1.1:** Format Overhead (Fig. 15) 

```
$dockercpHPCA-Pap313:/root/Sim/fig./uni-stc-results
```

## `(container)$ qrun format` 

_Explanation:_ Evaluates the storage compression ratio of the BBC format across varying sparsity levels. 

- **Task 1.2:** Hardware Comparison (Fig. 17, 18, 19) 

_Explanation:_ This will create a folder named `uni-stc-results` in your current directory containing all generated `.png` files. 

## **Option 2: In-Terminal Preview** 

For users employing modern terminal emulators capable of image rendering (e.g., **Kitty** , **iTerm2** , or **Ghostty** ), you can preview results directly inside the container without exporting. 

## `(container)$ qrun run-sample` 

```
#Insidethecontainer
```

_Explanation:_ Runs SpMV, SpMSpV, SpMM and SpGEMM kernels on representative matrices. Measures performance and energy. 

- **Task 1.3:** Random SpGEMM Evaluation (Fig. 16) 

## `(container)$ qrun spgemm2` 

- **Task 1.4:** AMG Application (Fig. 21) 

```
(container)$qrunrun-amg
```

## **A.5.2 Part 2: Complete Verification (L2)** 

_Estimated Time: ∼75 hours — Storage: ∼500GB required._ This mode downloads the full SuiteSparse collection[4] to reproduce the remaining distribution figures (Figures 20 and 22). 

**Step 1: Mount Dataset.** Download `matrix.7z` on your **host** machine, copy it to the container, and extract it. 

```
#OnHostMachine
```

```
$dockercpmatrix.7zHPCA-Pap313:/root
```

## `# On Container` 

```
(container)$cd/root
```

```
(container)$7zzxmatrix.7z
```

```
(container)$mvmatrix/*/matrix
```

## **Step 2: Execution.** 

- **Task 2.1:** Full Dataset Distribution (Fig. 20) 

```
(container)$qrunrun-all#Takes~24hours
```

- **Task 2.2:** Energy Efficiency Density (Fig. 22) 

```
(container)$qruneed#Takes~48hours
```

## **A.6 Evaluation and expected results** 

Upon completion of the experiments, all generated charts are stored in the container directory `/root/Sim/fig/` . We provide two methods to inspect these results. 

4 `https://drive.google.com/file/d/` 

```
(container)$qsicat/root/Sim/fig/15.png
```

## **A.6.2 Detailed Analysis** 

We outline the specific observations required to validate the artifacts below. **Note:** The simulator provided in this artifact is a lightweight version extracted from Accel-Sim to facilitate rapid verification. As it excludes power modeling for register I/O, the observed energy savings for Uni-STC may be _higher_ than the conservative figures reported in the paper. 

- **Fig. 15 (Format Overhead):** Verify that the BBC format spacereduction (y-axis) _increases_ as the density (x-axis) increases. 

- **Fig. 16 (Random SpGEMM Performance):** Uni-STC should demonstrate performance that is _equal to or greater than_ other baseline hardwares. 

- **Fig. 17 & 20 (Overall Performance & Efficiency):** 

   - **Fig. 17 (Representative):** Confirm that Uni-STC achieves the highest values in speedup, energy reduction, and area efficiency. 

   - **Fig. 20 (Full Dataset):** Confirm that these performance gains are consistent across the full SuiteSparse collection (2,800+ matrices). 

- **Fig. 18 (Energy Breakdown):** Verify that Uni-STC achieves the _lowest total energy consumption_ . Observe that the energy consumption is balanced across the three internal operations (Fetch, Schedule, Compute), showing similar values. 

- **Fig. 19 (Traffic & Network Scale):** Verify that Uni-STC incurs the _lowest data traffic_ compared to other architectures. Confirm that Uni-STC supports the required enabled network scale as depicted in the figure. 

- **Fig. 21 (AMG Solver):** Uni-STC should exhibit a higher speedup ratio compared to other baseline hardwares. 

- **Fig. 22 (Scalability - EED):** Compare the Energy Efficiency Density (EED) between Uni-STC(8) and Uni-STC(4): For SpMV / SpMSpV, Uni-STC(8) is slightly _lower_ than UniSTC(4). For SpMM / SpGEMM, Uni-STC(8) is _higher_ than Uni-STC(4). 

## **A.7 Methodology** 

Submission, reviewing and badging methodology: 

- `https://www.acm.org/publications/policies/artifa ct-review-and-badging-current` 

- `https://cTuning.org/ae` 

```
1Pp3BBOvU8nGoB12bb4o3wZs41twiXwXM
```

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 04,2026 at 04:43:32 UTC from IEEE Xplore.  Restrictions apply. 

