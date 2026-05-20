# **Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture** 

## Yu Gong 

Lingyi Huang 

Electrical and Computer Engineering Electrical and Computer Engineering Rutgers University Rutgers University Piscataway, New Jersey, USA Piscataway, USA yg430@soe.rutgers.edu lingyi.huang@rutgers.edu 

## Haodong Chang 

Electrical and Computer Engineering Texas A&M University College Station, USA haodong@tamu.edu 

Rongjian Liang Cheng Yang Zhexiang Tang Nvidia Electrical and Computer Engineering Electrical and Computer Engineering Austin, USA Rutgers University Rutgers University rliang@nvidia.com Piscataway, USA Piscataway, USA cy411@scarletmail.rutgers.edu zhexiang.tang@rutgers.edu 

Jiang Hu 

Electrical and Computer Engineering Texas A&M University College Station, USA Computer Science and Engineering Texas A&M University College Station, USA jianghu@tamu.edu 

## **Abstract** 

Tiled architectures have emerged as a compelling platform for scaling deep neural network (DNN) execution, offering both compute density and communication efficiency. To harness their full potential, effective inter-layer scheduling is crucial for managing operation order, memory behavior, and compute resource coordination. However, current schedulers often fall short due to three persistent issues: incomplete treatment of core design factors, limited flexibility in handling diverse workload structures, and reliance on heuristic search algorithms with poor convergence. 

In this work, we trace these limitations to the absence of a unified and expressive scheduling representation. We introduce _Crane_ , a framework that addresses these gaps through a hierarchical tableformat abstraction capable of encoding rich scheduling semantics. Crane supports both inference and training workloads, and reformulates scheduling as a mathematically structured optimization problem, enabling more complete and efficient exploration of the scheduling space. Evaluations show that Crane reduces energydelay product by up to 21 _._ 01× and improves scheduling speed by at least 2 _._ 82× over state-of-the-art baselines. 

## **CCS Concepts** 

## • **Computer systems organization** → **Neural networks** . 

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License. _MICRO ’25, Seoul, Republic of Korea_ 

© 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1573-0/25/10 https://doi.org/10.1145/3725843.3756023 

Bo Yuan 

Electrical and Computer Engineering Rutgers University Piscataway, USA bo.yuan@soe.rutgers.edu 

## **Keywords** 

Inter-layer Scheduling, Deep Neural Networks, Tiled Architecture 

## **ACM Reference Format:** 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan. 2025. Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture. In _58th IEEE/ACM International Symposium on Microarchitecture (MICRO ’25), October 18–22, 2025, Seoul, Republic of Korea._ ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3725843.3756023 

## **1 Introduction** 

Deep neural network (DNN) hardware accelerators have been widely adopted in real-world applications for energy-efficient execution. As the computational and storage demands of DNNs continue to grow, tiled architectures—composed of Network-on-Chip (NoC)-connected hardware tiles—have emerged as a scalable and flexible solution for large-scale model processing. In these architectures, each tile typically integrates a processing element (PE) array, a global buffer, and an NoC router, enabling efficient intraand inter-tile communication. This structural design supports highperformance execution and has been widely adopted in both industry [17, 22, 24, 35] and academia [3, 5, 8, 10, 11, 16, 20, 31, 36, 42]. 

To fully utilize the computing power and memory capacity of connected hardware tiles, an efficient scheduling scheme—responsible for mapping computational workloads to hardware resources—is critical. DNN scheduling is generally divided into intra-layer and inter-layer scheduling. Intra-layer scheduling focuses on mapping individual layers to one or more hardware tiles and has been extensively studied in the literature [7, 11, 12, 28, 32, 33, 37], with a 

1250 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

wide range of solutions and notations proposed. In contrast, interlayer scheduling determines the computation order, memory access patterns, and resource allocation across multiple layers and tiles, aiming to maximize hardware utilization and energy efficiency. As application scenarios diversify and DNN architectures grow increasingly complex, inter-layer scheduling has attracted growing research attention [5, 9, 13, 38, 40]. 

Despite recent efforts, existing inter-layer schedulers still face several fundamental limitations (Section 2.2). First, with respect to the four key design factors in inter-layer scheduling—namely execution scheme, fusion strategy, recomputation, and batch splitting—even state-of-the-art solutions fail to comprehensively explore all of them, thereby limiting performance. In particular, the lack of integration of recomputation as a core design factor for training optimization significantly constrains the ability of current schedulers to support efficient training. Second, even within the limited scope of design factors they incorporate, existing inter-layer schedulers suffer from restricted scheduling flexibility—such as constrained processing orders for cross-layer sub-batches, support limited to linear chain-structured workloads, or even a complete inability to handle training workloads. Third, the underlying search engines of most existing inter-layer schedulers rely on heuristic approaches, which essentially sample the search space with slow convergence. This leads to incomplete and inefficient exploration, resulting in suboptimal scheduling performance and slower scheduling speed. 

To address these limitations, we first analyze and identify their root causes —chiefly, the absence of a proper representation framework. We then distill three essential lessons that such a representation must satisfy: rich expressiveness, topological flexibility, and mathematical structuredness (Section 3.1). Guided by these insights, we propose _Crane_ (Section 4,5,6), a novel inter-layer scheduling framework designed to support both inference and training on tiled architectures. Crane is built upon an efficient table-format hierarchical representation that comprehensively captures key design factors, enables flexible scheduling across diverse workloads, and formulates the scheduling problem as a structured optimization task that can be thoroughly and efficiently solved by mixed-integer linear programming (MILP) solver. Evaluation results (Section 7) show that Crane achieves a 1 _._ 13× to 21 _._ 01× reduction in energydelay product costs and delivers at least a 2 _._ 82× scheduling speedup compared to state-of-the-art solutions. 

## **2 Background & Motivation** 

## **2.1 Inter-layer Scheduling** 

In general, inter-layer scheduling for tiled architecture is typically structured around the following four components: 

**1) Execution Scheme.** This decision variable defines how the DNN workload consisting of multiple layers is mapped temporally and spatially across multiple computing cores. In general, three execution patterns are commonly adopted in practice: _Sequential:_ Processes the model layer by layer in sequence; therefore, different layers are executed at different time steps, and each layer fully utilizes all the computational resources and on-chip memory ; _Pipeline:_ Coordinates several dependent layers to be processed concurrently in a pipelined fashion, sharing the hardware tiles across layers; 

|**Inter-layer**|**Design**|**Search**|**Support**|**Search Space**|**Schedule**|
|---|---|---|---|---|---|
|**Scheduler**|**Factors**|**Alg.**|**Training**|**(Training)**|**Flexible**|
|MBS|F+B|Greedy<br>Slow|Yes<br>Limited|O(2_𝑛_log_𝑚_)<br>Sampled|Yes|
|Tangram|F+B|DP<br>Slow|Hypothesized|O(2_𝑛_log_𝑚_)<br>Thoroughly|Yes|
|Checkmate|R|MILP<br>Fast|Yes<br>Limited|O(_𝑛_2)<br>Thoroughly|Batch<br>-level Only|
|TileFlow|E+F+B<br>(Partially)|GA<br>Slow|No|N/A|Branchless<br>Only|
|SET|E+F+B|SA<br>Slow|Hypothesized|O(9_._899_𝑛_log_𝑚_)<br>Sampled|Tied to<br>Batch-level|
|**Crane**|**E+F+R+B**|MILP<br>Fast|Yes|O(_𝑚_4_𝑛_−1 log_𝑚_)<br>Thoroughly|**Yes**|



**Table 1: Comparison of various inter-layer schedulers. Some notes: i) E, F, R and B denote execution scheme, layer fusion, recomputation, and sub-batch splitting, respectively. ii) The search spaces is for** _𝑛_ **-layer model with batch size of** _𝑚_ **. iii) Tangram and SET are not designed for training and do not report any training results. We hypothesize their potential training support by directly applying their inference schedules to the backward pass. iv) DP: Dynamic Programming; GA: Genetic Algorithm; SA: Simulated Annealing; MILP: Mixed-Integer Linear Programming v) The resource binding and loop ordering of TileFlow are manually fixed and only loop tiling is explored in [21].** 

_Parallel:_ Allows multiple layers to be processed simultaneously without needing to account for dependencies among them. 

**2) Fusion Strategy.** This is another important design factor that determines how to directly transfer the output data from previous layers to the latter ones without expensive off-chip memory (DRAM) access. As the strategy aims at reducing data movement, fusion [2, 18, 26, 41] is typically considered along with the execution pattern. In the scenario of sequential, the intermediate results are calculated and retained within local hardware tiles; while in the scenario of pipeline, the mapped group of tiles for one layer sends the output data to another tile group allocated for the next layer. 

**3) Recomputation Scheme.** This strategy decides the protocol that when and which intermediate results should be temporally discarded and recomputed in the future as needed. By trading additional computation for reduction in storage, the recomputation scheme aims to effectively free up memory capacity – a critical advantage in various memory-constrained scenarios. In practice, the most common application of this strategy is in DNN training [4, 15, 19, 39], where DRAM capacity becomes a major bottleneck, especially as memory consumption for activation scales with batch size. Applying recomputation in this context enables the training of larger and more complex models using larger batch sizes without requiring extra hardware resources. 

**4) Batch Splitting Plan.** This strategy determines how to partition a batch of data into smaller subsets, which are then processed sequentially to complete the computation for each layer [6, 14, 18, 34]. Batch splitting can be applied in both inference and training scenarios, particularly when memory capacity is insufficient to process an entire batch at once. By dividing the batch into manageable pieces, this approach allows the utilization of limited memory resources while still leveraging batch processing benefits. 

1251 

Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

## **2.2 Limitations of Previous Works** 

Motivated by the critical role of inter-layer scheduling in enabling efficient DNN inference and training, a set of inter-layer schedulers have been proposed in recent years. Table 1 summarizes the most relevant works, highlighting the design factors they explore, the search algorithms they employ, and their target deployment scenarios. Based on this summary, we identify several key limitations: 

**Challenge #1. Incomplete Exploration of Design Factors.** None of the existing works comprehensively and systematically explore all four key design factors. As shown in Table 1, even stateof-the-art inter-layer schedulers such as SET [5] and TileFlow [40] lack support for recomputation schemes (R), resulting in limited or no applicability to training workloads. On the other hand, while Checkmate [15] incorporates recomputation strategies, it does not account for the other design factors (such as E and F) and is incompatible with schedulers like SET due to fundamental differences in search algorithms and representation frameworks. Additionally, Checkmate’s exploration is restricted to the batch level rather than the sub-batch level (B), significantly constraining its scheduling granularity and design space. As a result, existing inter-layer scheduling approaches cannot offer efficient, unified solutions – particularly for training scenarios. For example, as illustrated in Fig. 1, in ResNet-50 training with a batch size of 64, prior works can only optimize either DRAM data access (e.g., SET) or capacity requirements (e.g., Checkmate), but not both simultaneously. 

**Challenge #2. Constrained Scheduling Flexibility.** Even when optimizing solely for inference scenarios—where recomputation (R) is not required—existing schedulers such as SET and TileFlow, which consider P+F+B, still suffer from limited scheduling flexibility. Specifically, SET enforces that the processing order of sub-batches across layers is strictly tied to the batch-level execution pattern. For example, when the execution scheme for 3- sub-batch Layer-A, Layer-B and Layer-C is set to a pipeline pattern, the identified sub-batch-level processing order can only be _𝐴_ 1 → ( _𝐴_ 2 _, 𝐵_ 1) →( _𝐴_ 3 _, 𝐵_ 2 _,𝐶_ 1) →( _𝐵_ 3 _,𝐶_ 2) → _𝐶_ 3. Alternative scheduling options, such as _𝐴_ 1 _𝐴_ 2 →( _𝐴_ 3 _, 𝐵_ 1) →( _𝐵_ 2 _𝐵_ 3 _,𝐶_ 1 _𝐶_ 2) → _𝐶_ 3, are never explored (see Fig. 15(a) for a practical example). Evidently, this rigid constraint significantly narrows the design space and may miss more efficient scheduling solutions. Although TileFlow overcomes the rigid batch-level scheduling constraint by allowing layer partitioning along arbitrary dimensions, it has two key limitations. 1) It is limited to optimizing linear, chain-structured workloads, such as GEMM or convolution chains. This limitation stems from its tile-centric, layer-splitting representation, which cannot model control-flow structures like branches. 2) It cannot support training workloads. Partitioning along non-batch dimensions preserves correctness only in forward propagation; backward propagation—critical for training—requires additional halo exchanges, global reductions, and synchronized statistics. 

**Challenge #3. Incomplete and Inefficient Search.** Most stateof-the-art inter-layer schedulers rely on heuristic search algorithms, leading to both insufficient solution quality and long runtime. This limitation manifests in two ways. 1) _Incomplete scheduling space coverage_ : the search procedures do not comprehensively explore the full scheduling space but instead rely on **sampling** -based heuristics such as simulated annealing (SET) or genetic algorithms (TileFlow). 

**==> picture [242 x 60] intentionally omitted <==**

**----- Start of picture text -----**<br>
Baseline SET MBS Checkmate Ours<br>20<br>15<br>10<br>5<br>0<br>Data Access (GB) Required DRAM Capacity (GB)<br>**----- End of picture text -----**<br>


**Figure 1: Required data access and DRAM capacity for training ResNet-50 with a batch size of 64. While SET and MBS reduce data access through batch splitting (B) and layer fusion (F), the overall DRAM capacity requirement remains high. Checkmate significantly lowers DRAM capacity by introducing recomputation (R), while having high data access. Crane effectively reduces both data access and DRAM capacity through comprehensive optimization strategies.** 

These methods inherently risk missing globally optimal solutions. 2) _Long scheduling duration_ : due to their stochastic nature, these heuristics converge slowly, resulting in long search times even for moderately sized workloads. For example, scheduling Inception inference with a batch size of 128 on 144 hardware tiles takes over two hours using the SET framework on an AMD EPYC 7402P CPU. 

## **3 Crane Preview: Philosophy & Contributions 3.1 Lessons Learned: Representation Matters** 

As outlined in Section 2.2, existing inter-layer schedulers suffer from several critical limitations. Our in-depth analysis reveals that the root cause lies in the lack of a proper representation framework: 

_**Lesson #1. Representation should provide rich expressiveness.**_ For Challenge #1, the reason prior works cannot fully explore all four design factors is that their underlying representations lack the expressiveness needed to support such exploration. For example, automatic exploration of recomputation strategies requires fine-grained tracking of memory consumption across all time steps and layers—something that the resource allocation (RA) tree-based notation used in SET cannot provide. 

_**Lesson #2. Representation should exhibit topological flexibility.**_ For Challenge #2, the inherent topology of the representations used in prior works limits the flexibility of scheduling. For example, the construction process of the ratio-tree in SET inherently enforces repeated execution patterns across the same node, which imposes rigid scheduling constraints. In the case of TileFlow, its tile-centric tree representation works well for simple, linear chains of computation. However, it becomes significantly challenging to construct tile trees that accurately describe more complex computation flows involving multiple fan-in and fan-out structures. 

_**Lesson #3. Representation should support mathematical structuredness.**_ For Challenge #3, the representation fundamentally shapes the form and tractability of the optimization problem. Existing works fail to produce well-structured objectives—they lack essential mathematical properties such as continuity, differentiability, convexity, and linear-discrete structure. As a result, the scheduling problems cannot be formulated for efficient, principled optimization and must instead rely on heuristic, sampling-based search with slow convergence and no guarantee of solution quality. 

1252 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

## **3.2 Key Contributions of Crane: Table-format Hierarchical Representation** 

Grounded in the representation-centric design philosophy outlined above, the core innovation of Crane lies in developing an effective representation that satisfies the key requirements of expressiveness, flexibility, and mathematical structuredness. More specifically: 

**Contribution #1.** First, to address Challenge #2, in Section 4 we identify and generalize a novel hierarchical representation, where the execution scheme at each hierarchical level is modeled as a subset of a pipeline scheme. This hierarchical structure offers sufficient topological flexibility to accommodate diverse scheduling behaviors and is proven to abstract the execution scheme of any model whose computational graph forms a directed acyclic graph (DAG), given sufficient hierarchical depth. 

**Contribution #2.** Then, to address Challenge #1, in Section 5.1, 5.2, 5.3 and 5.5, we introduce a table-format notation to capture subbatch execution and memory status at each level of the hierarchy. These tables explicitly record the effects of scheduling decisions on both workload execution and memory usage. As a result, the table-format representation offers sufficient expressiveness to fully describe the detailed impact of all four key design factors. 

**Contribution #3.** Finally, to address Challenge #3, in Section 5.4, 5.5 we show that the structured nature of the table-format representation enables the scheduling problem to be formulated as a MILP. The linear constraints and clearly defined decision variables of the table representation allow the use of mature MILP solvers, enabling thorough exploration of the scheduling space with significantly faster convergence compared to heuristic methods. 

**==> picture [194 x 311] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 Tiles Tiles Tiles<br>2 2<br>Scheduling 1 2 1 2<br>Time 1 Time 1 Time<br>2<br>Sequential Parallel Pipeline<br>State-1 State-2 State-1<br>State-3 State-2 State-3<br>Derivation 2<br>1 2<br>1<br>Pipeline format :State-1 :State-2 :State-3<br>Figure 2: Derive states of execution scheme from pipeline pattern.<br>Tiles<br>Block  A Scheduling 3 4<br>1 2 3 4 1 2 1 23 4<br>Time<br>A 4<br>4<br>Scheduling Derivation A 4<br>Tiles A<br>4 Pipeline format State-1 State-2 State-3<br>A 4<br>A<br>Active State<br>Time<br>Block  A<br>1 2 3<br>Scheduling Pipeline format<br>Derivation<br>Tiles<br>2 3 3<br>2 3 Active 1 2 3<br>1 1 2 3 State 1 1 2<br>State-1 State-2 State-3 State-4 State-5<br>Time<br>**----- End of picture text -----**<br>


**Figure 2: Derive states of execution scheme from pipeline pattern.** 

**Figure 3: The states of complex execution scheme of a 4-layer model can be derived from pipeline pattern in a hierarchical manner.** 

## **4 Hierarchical Representation of Scheduling 4.1 Intuitive Glance** 

To efficiently explore the vast scheduling space, we propose a hierarchical representation of the scheduling schemes. The core of this representation is the concept of _**Execution State**_ , which captures the pattern of active layer execution – that is, which layers are simultaneously involved in computation at each processing stage. 

Example 1. Take the scheduling of a 2-layer model as an example. As shown in Fig. 2, when the two layers are processed sequentially, the execution involves exactly two distinct States: initially, only Layer-1 is active (State-1), followed by a stage where only Layer2 is active (State-3). In contrast, when the layers are processed concurrently (i.e., in parallel), the execution involves only a single State – both Layer-1 and Layer-2 are active simultaneously (State-2). Finally, in the case of pipeline processing, the execution progresses through three distinct States, involving Layer-1 alone, then both layers concurrently, and finally Layer-2 alone – that is, all three States (State-1, State-2, and State-3) occur over time. 

Notably, from the earlier example, we observe that the pipeline pattern naturally encompasses all _execution states_ that also appear in the sequential and parallel patterns. This suggests that _pipeline scheduling may serve as a unifying structure, capable of capturing the full range of active layer configurations encountered across different scheduling strategies._ However, due to the simplicity of the 2-layer model, this observation may not generalize. To further examine its 

validity, we consider a more complex example involving a deeper model with a more intricate sub-batch-level scheduling scheme. 

Example 2. As illustrated in Fig. 3, consider a 4-layer model whose scheduling behavior cannot be easily categorized into any single basic pattern. To analyze it, we group Layer-1 through Layer-3 into a single _block_ A, effectively transforming the model into a 2-block structure: block A followed by Layer-4 (treated as a standalone block). At this level of abstraction, the overall schedule clearly follows a pipeline pattern between block A and Layer-4. All execution states involved in this 2-block view correspond to standard pipeline-derived states. Looking inside block A, its internal scheduling consists of four distinct execution states: (1) a state where only Layer-1 is active; (2) a state where Layer-1 and Layer-2 are active concurrently; (3) a state where Layer-2 and Layer-3 are active concurrently; and (4) a state where only Layer-3 is active. When we arrange Layer-1 through Layer-3 in a pure pipeline fashion, we observe five distinct pipeline-derived execution states. Crucially, the four states actually used within block A are all included in this set of five, reinforcing the observation that pipeline-derived execution states are expressive enough to represent even irregular or non-canonical schemes. 

**Hierarchical Generalization.** This hierarchical interpretation naturally extends to deeper models with increasingly complex scheduling behavior. By recursively grouping subsets of layers 

1253 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture 

into higher-level blocks, the overall scheduling can be abstracted into multiple hierarchical levels. While the interaction between blocks at each level may vary in form, we consistently observe that the execution states involved – regardless of complexity – are drawn from, or are subsets of, those generated by canonical pipeline scheduling. This reinforces the role of the pipeline pattern not only as a representational baseline, but as a structural foundation for expressing general inter-layer scheduling schemes. 

## **Table 2: Notation Description.** 

|**Notation**|**Description**|
|---|---|
|_𝐵𝑆_<br>_𝐵𝑆𝑠𝑢𝑏_<br>_𝑑𝑚,𝑗_<br>_𝐿_<br>_𝑁_|Batch size<br>Sub-batch size<br>Dependency from Layer-_𝑚_to Layer-_𝑗_<br>Total number of layers in the model<br>Number of sub-blocks in a composite block|
|L<br>_𝐵_<br>_𝐵𝑖_<br>C_𝐵_<br>L_𝐵_<br>H<br>S_𝐵_<br>J_𝑖_<br>_𝑠𝑖_<br>A_𝐵_′<br>LJ_𝑖_<br>A_ℓ_|Set of all layers in the model<br>A block (basic or composite)<br>The_𝑖_-th sub-block of block_𝐵_<br>Ordered sequence of sub-blocks within block_𝐵_<br>Set of layers contained in block_𝐵_<br>Hierarchical block structure over L<br>State index set of block_𝐵_:{1_, . . . ,_2_𝑁_−1}<br>Sub-blocks active in state-_𝑖_<br>State workload: number of sub-batches processed in state-_𝑖_<br>Set of states where sub-block_𝐵_′ is active<br>Layers active in state-_𝑖_<br>States where layer_ℓ_is active|



**Observation 1.** With sufficient hierarchical abstraction, complex inter-layer scheduling schemes in deep models can be consistently represented by execution states derived from the pipeline pattern. This highlights pipeline scheduling as a unifying structural basis for expressing general execution behavior across all levels in models. 

## **4.2 Formal Notation** 

We now formalize the core concepts introduced in Section 4.1. **Definition 1** ( **Block** ) **.** Let L = {1 _,_ 2 _, . . . , 𝐿_ } be the set of all layers in the model. A _block 𝐵_ is a structural scheduling unit defined recursively as follows: 

- (i) _𝐵_ is a **basic block** if it corresponds directly to a consecutive subset of layers from L, and contains no sub-blocks. 

- (ii) _𝐵_ is a **composite block** if it consists of an ordered sequence of sub-blocks C _𝐵_ = { _𝐵_ 1 _, 𝐵_ 2 _, . . . , 𝐵𝑁_ }, where each _𝐵𝑖_ is itself a block (either basic or composite). 

The set of layers associated with block _𝐵_ , denoted L _𝐵_ , is defined recursively as: 

**==> picture [183 x 28] intentionally omitted <==**

A block is said to be _top-level_ if the associated layer set spans the entire model, i.e., L _𝐵_ = L. 

**Definition 2** ( **State** ) **.** Let _𝐵_ be a block with an ordered sequence of sub-blocks C _𝐵_ = { _𝐵_ 1 _, 𝐵_ 2 _, . . . , 𝐵𝑁_ }, where each _𝐵𝑖_ is a block (either basic or composite). The execution of block _𝐵_ proceeds through a set of pipeline-derived execution states, indexed by S _𝐵_ = {1 _,_ 2 _, . . . ,_ 2 _𝑁_ − 1}. 

Each state index _𝑖_ ∈S _𝐵_ is associated with: 

**==> picture [178 x 9] intentionally omitted <==**

**==> picture [163 x 28] intentionally omitted <==**

- (ii) a non-negative scalar _𝑠𝑖_ ∈ R≥0, called the _state workload_ , representing the number of sub-batches processed during State- _𝑖_ . These scalars will participate in a later formulation of scheduling constraints. 

**Definition 3** ( **Involved State Set** ) **.** Let _𝐵_ be a block with subblock sequence C _𝐵_ = { _𝐵_ 1 _, 𝐵_ 2 _, . . . , 𝐵𝑁_ } and state index set S _𝐵_ = {1 _,_ 2 _, . . . ,_ 2 _𝑁_ −1}. The _involved state set_ of an entity specifies indices of all states where that entity is active during execution of _𝐵_ . 

- (i) For a sub-block _𝐵_[′] ∈C _𝐵_ , the involved state set is: 

**==> picture [96 x 10] intentionally omitted <==**

- (ii) For a layer _ℓ_ ∈L _𝐵_ , let L J _𝑖_ =[�] _𝐵_[′] ∈J _𝑖_[L] _𝐵_[′][be the set of layers] active in state _𝑖_ . The involved state set of layer _ℓ_ is: 

**==> picture [94 x 10] intentionally omitted <==**

**Constraint 1** ( **Computation-Completeness** ) **.** Let _𝐵_ be a block with state set S _𝐵_ = {1 _,_ 2 _, . . . ,_ 2 _𝑁_ − 1}, and let _𝑠𝑖_ ∈ R≥0 denote the state workload for each state _𝑖_ ∈S _𝐵_ . Let L _𝐵_ be the set of all layers contained in block _𝐵_ , and let A _ℓ_ ⊆S _𝐵_ denote the involved state set for layer _ℓ_ ∈L _𝐵_ . 

The state workloads must satisfy the following constraint: 

**==> picture [144 x 25] intentionally omitted <==**

where _𝐵𝑆_ is the total batch size and _𝐵𝑆_ sub is the sub-batch size. This ensures that each layer processes the full batch exactly once, distributed across the states in which it is active. 

Building on the above definitions, we now establish the expressive capacity of the hierarchical scheduling abstraction. Specifically, we prove that any valid inter-layer execution pattern in a model whose computational graph forms a DAG—including architectures with branching and merging structures—can be represented using a hierarchical block composition with pipeline-derived states. 

**Theorem 1** ( **Universality of Hierarchical Block Representation** ) **.** Any valid inter-layer scheduling behavior of a model with a DAG structure can be represented using a hierarchical block composition equipped with pipeline-derived execution states and associated workloads. 

Proof. Let computational graph of model be a DAG _𝐺_ = (L _, 𝐸_ ), where L is set of layers and _𝐸_ denotes data dependencies. We construct a hierarchical representation by traversing _𝐺_ topologically. 

**Step 1:** At every vertex with in-degree or out-degree exceeding one (branching or merging points), we partition _𝐺_ and encapsulate each linear segment (a path of vertices with in-degree and outdegree equal to one) into a _basic block_ . Each branching or merging vertex, together with its directly connected linear segments, forms a higher-level _composite block_ . Recursively composing these blocks yields the top-level hierarchical block _𝐵_ ⊤. 

**Step 2:** Consider any composite block consisting of ordered subblocks C _𝐵_ = { _𝐵_ 1 _, . . . , 𝐵𝑁_ }, with canonical pipeline state set S _𝐵_ = {1 _, . . . ,_ 2 _𝑁_ − 1}. By assigning _𝑠𝑖_ , we can flexibly represent diverse execution schemes—sequential, pipelined, parallel, or combinations thereof. For example, in a _fan-out_ scenario (where sub-block _𝐵_ 1 

1254 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

produces data consumed by subsequent branches), assigning _𝑠_ 1 = _𝑠𝑁_ +1 = _𝐵𝑆_ / _𝐵𝑆_ sub and setting other _𝑠𝑖_ = 0 demonstrates parallel execution. In a _fan-in_ scenario (where sub-block _𝐵𝑁_ merges earlier branches), assigning _𝑠𝑁_ −1 = _𝑠_ 2 _𝑁_ −1 = _𝐵𝑆_ / _𝐵𝑆_ sub captures merging behavior. These examples show workload assignments across state indices can express any valid sub-block execution pattern. 

**Step 3:** By recursively applying Steps 1 and 2 across the DAG, both linear and branched structures are covered. As the same hierarchical abstraction and scheduling semantics apply uniformly to basic and composite blocks, the top-level block _𝐵_ ⊤ fully captures the execution behavior of the original model. Hence, all legal inter-layer schedules can be represented via hierarchical block abstraction. □ 

_**Remark**_ 1 ( **Expressiveness Across DAG-Based Architectures** ) _._ Architectures such as self-attention layers, ResNet bottlenecks with skip connections, and Inception modules in GoogleNet are all representable as feed-forward DAGs. Therefore, they fall within the scope of Theorem 1 and are fully expressible using the proposed hierarchical block abstraction. 

Having established that any valid inter-layer execution behavior can be represented through hierarchical blocks and pipeline-derived states, we now formulate the scheduling problem as a constrained optimization. This formulation captures both structural choices and numerical decisions: 

**Problem 1** ( **Inter-layer Scheduling** ) **.** Let L = {1 _,_ 2 _, . . . , 𝐿_ } be the set of all layers in the model. The inter-layer scheduling problem seeks to jointly determine: 

- a hierarchical block structure H , which recursively partitions L into nested blocks; and 

- a collection of state workloads { _𝑠𝑖_ } _𝐵_ ∈H _,𝑖_ ∈S _𝐵_ , where each _𝑠𝑖_ ∈ R≥0 corresponds to a pipeline-derived execution state _𝑖_ ∈S _𝐵_ of block _𝐵_ , 

so as to minimize the total execution cost: 

**==> picture [172 x 23] intentionally omitted <==**

Here, C(H _,_ { _𝑠𝑖_ }) denotes the energy-delay product (EDP) of model execution, and A _ℓ_ is involved state set of layer _ℓ_ , as defined earlier. 

_**Remark**_ 2 ( **Addressing Challenge #2: Flexible and General Scheduling** ) _._ The hierarchical block abstraction, through its pipelinederived states and flexible workload assignments, supports diverse sub-batch execution patterns—sequential, pipelined, or parallel. It also generalizes across various architectures with branches or skip connections. These capabilities directly address Challenge #2 by broadening the feasible scheduling space. 

The formulation above reveals that solving the scheduling problem involves two types of decisions: determining the workload configuration within each block, and selecting the hierarchical organization of blocks across the model. These two components are structurally decoupled – each block’s schedule is governed by its local execution states, while the block hierarchy implicitly defines inter-block dependencies. In this paper Section 5 focuses on intra-block scheduling, and Section 6 explores construction and optimization of hierarchical block structure. 

## **5 Table-format Intra-Block Scheduling 5.1 Representation of Execution Scheme** 

**Recording State Information Using the Scheduling Table.** As analyzed above, the execution scheme of a block can be represented through combinations of execution states and their associated workload values { _𝑠𝑖_ }. To facilitate visibility and manipulation of this execution structure, we introduce a table-based representation, called the _Scheduling Table (ScT)_ . This provides a structured and cumulative view of how sub-batch workloads are distributed across execution states and sub-blocks, offering a more intuitive and analyzable form than listing state variables directly. 

**Definition 4** ( **Scheduling Table (** ScT **)** ) **.** For a block _𝐵_ with subblock sequence C _𝐵_ = { _𝐵_ 1 _, 𝐵_ 2 _, . . . , 𝐵𝑁_ } and pipeline-derived state set S _𝐵_ = {1 _, . . . ,_ 2 _𝑁_ − 1}, the _scheduling table_ ScT ∈ R[(][2] _[𝑁]_[−][1][)×] _[𝑁]_ records the cumulative number of sub-batches processed by each sub-block across states. 

- (i) ScT has 2 _𝑁_ − 1 rows and _𝑁_ columns. Row _𝑖_ corresponds to State- _𝑖_ , and column _𝑗_ corresponds to sub-block _𝐵 𝑗_ ∈C _𝐵_ . 

- (ii) The entry ScT _𝑖,𝑗_ denotes the total number of sub-batches processed by sub-block _𝐵 𝑗_ from State-1 through State- _𝑖_ , inclusive. 

- (iii) Let A _𝐵 𝑗_ ⊆S _𝐵_ denote the involved state set of sub-block _𝐵 𝑗_ . The cumulative processed sub-batches for _𝐵 𝑗_ up to State- _𝑖_ is 

**==> picture [84 x 23] intentionally omitted <==**

**Constraint-Form Equivalence of Definition 5.** As illustrated above, ScT _𝑖,𝑗_ are derived according to Definition 5. To enable integration with our MILP formulation, we now express an equivalent set of constraints – Eqs. 1 through 6 – for computing ScT in a constraint-based format. Eq. 1 ensures that all entries in ScT are non-negative integers. Eqs. 2 and 3 define boundary conditions for execution: Eq. 2 corresponds to the stage range before any subblock _𝐵 𝑗_ becomes active (i.e., A _𝐵 𝑗_ ∩{1 _, . . . ,𝑖_ } = ∅), while Eq. 3 corresponds to the point after _𝐵 𝑗_ has completed execution (i.e., A _𝐵 𝑗_ ∩{1 _, . . . ,𝑖_ } = A _𝐵 𝑗_ ). Eq. 4 enforces the monotonicity of accumulated sub-batch processing for each sub-block. Eq. 5 encodes data dependencies using the binary variable _𝑑𝑚,𝑗_ ∈D, where _𝑑𝑚,𝑗_ = 1 if sub-block _𝐵 𝑗_ depends on the output of sub-block _𝐵𝑚_ , and _𝑑𝑚,𝑗_ = 0 otherwise. This dependency set D is fixed once the model architecture is given, and its inclusion ensures that dependent sub-blocks are not assigned to overlapping states in a parallel pattern. Finally, Eq. 6 describes the cumulative accumulation of sub-batch workloads across states, consistent with the semantics of ScT. 

**==> picture [203 x 72] intentionally omitted <==**

## **5.2 Representation of Fusion Strategy** 

While determining the state workloads { _𝑠𝑖_ } and constructing the scheduling table ScT fully specifies the execution scheme of a block, it does not capture memory-related behavior – particularly those associated with _fusion strategies_ . These strategies directly affect 

1255 

Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

**==> picture [506 x 213] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 Tiles # of Processed Sub-batches 1 2 … N<br>Block  B ……N2 Mapping RRR DRAM… RRR ……… RRR RepresentationState-Based State-1s1 State-2ss22 State-N … sssNNN State-2N-1s2N-1 Time State 1~2N-1State 1~NState 1~2State 1 s! + ⋯+ sBSs!/"sBS+ s!= BS#$%" /BS#$% ………… s"BS+ ⋯+ s/ s BS0" #$% # … ……… BS/BS s 00##$%<br>DNN Model Tiled Architecture Execution Scheme ScT<br>Figure 4: State-based representation of execution scheme for mapping multiple layers onto a tiled architecture. The number of hardware tiles<br>allocated to different layers within the same state is proportionally determined based on their computa tional costs.<br>DNN Model<br>Tiles # of Processed<br>Maximum Index oStored Sub-batches f  1 43 1 2 Sub State 1 -batches 1 3 02 Min of Stored . Index  1 2<br>Layer Fusion Dataflow in SRAMMinimum Index ofStored Sub-batches DNN Model 2 21 State1(0:3] State3MeT( 0:3] State State 1State 1~~23 ScT 33 03 SubState 2S- tate1batches SS0 DD3 SS0 DD0<br>1 (0:3] in SRAM 2 (0:3] in DRAM 0 3 0 0<br>DRAM PEs in Tiles SRAM in Tiles PEs in  Tiles DRAM State 3 S D S D<br>3 3 3 0<br>Index: [1][2][3]Input of  1 Processing Processing  1 Writing Index: [1][2][3]Output of  1 Processing Processing  2 Writing Index: [1][2][3]Output of  2 MeT<br>State1 State 3<br>**----- End of picture text -----**<br>


**Figure 5: Example that how** MeT **and** ScT **jointly and implicitly capture the impact of fusion strategy on memory status. The 2-layer model is a top-level block with each layer as a sub-block. In State-1, all 3 sub-batches of Layer-1’s input data are transferred from DRAM to PEs for computation. The resulting outputs are then stored in on-chip SRAM, avoiding costly DRAM writes. Consequently, SRAM holds 3 sub-batches of outputs (indexed as 1, 2, 3) for Layer-1, while DRAM stores none. This is reflected in** ScT1 _,_ 1 = 3 **, indicating that 3 sub-batches are processed in State-1. Meanwhile,confirming no data is stored in DRAM. In State-3, the 3 sub-batches stored in SRAM are directly sent to PEs for Layer-2 processing, eliminating** MeT _[𝑆]_ 1 _,_ 1[=][0] **[ defines the stored sub-batch range in SRAM as]**[(][0][:][3][]] **[, and]**[ MeT] _[𝐷]_ 1 _,_ 1[=][3] **[ defines the DRAM range as]**[(][3][:][3][]] **[,] DRAM access. After computation, the 3 sub-batches of Layer-2’s output data are stored in DRAM, and Layer-1’s intermediate results are evicted from SRAM. This is represented by** MeT _[𝐷]_ 3 _,_ 2[=][ 0] **[ and]**[ ScT][3] _[,]_[2][=][ 3] **[, defining the sub-batch range of Layer-2’s output in DRAM as]**[(][0 : 3][]] **[. Additionally,]** MeT **in SRAM nor DRAM, precisely describing the final memory state.** _[𝑆]_ 3 _,_ 2[=][ 3] **[ indicates no Layer-2’s output remains in SRAM]**[(][3] _[,]_[ 3][]] **[. Similarly,]**[ MeT] _[𝑆]_ 3 _,_ 1[=][ MeT] _[𝐷]_ 3 _,_ 1[=][ 3] **[ confirms that Layer-1’s outputs are stored neither]** 

memory reuse, data lifetimes, and intermediate storage requirements, which are not encoded in ScT alone. 

To model this dimension, we introduce the _Memory Table_ MeT, which tracks the _lifetime of intermediate data_ in memory. Specifically, MeT captures the allocation and deallocation dynamics of sub-batch-level intermediate results for each execution unit (e.g., layer or sub-block), offering a structured and interpretable view of how fusion affects memory usage. Each fusion decision (the “ _action_ ”) introduces implicit changes to memory behavior (the “ _impact_ ”), influencing how long intermediate results must be stored and when memory can be released. 

**Definition 5** ( **Memory Table (** MeT **)** ) **.** For a block with sub-block sequence C _𝐵_ = { _𝐵_ 1 _, 𝐵_ 2 _, . . . , 𝐵𝑁_ } and state set S _𝐵_ = {1 _, . . . ,_ 2 _𝑁_ − 1}, the _memory table_ MeT ∈ R[(][2] _[𝑁]_[−][1][)×] _[𝑁]_[×][2] tracks the memory status of intermediate data for each block or sub-block across states. 

(i) MeT has 2 _𝑁_ −1 rows and _𝑁_ columns, where row _𝑖_ corresponds to State- _𝑖_ , and column _𝑗_ corresponds to block _𝐵 𝑗_ ∈C _𝐵_ . 

- (ii) Each entry MeT _𝑖,𝑗_ is a tuple (MeT _𝑖,𝑗[𝐷][,]_[ MeT] _𝑖,𝑗[𝑆]_[)][, where:] 

   - MeT _[𝐷]_[the][lower][(open)][bound][of][the][sub-batch] _𝑖,𝑗_[denotes] 

   - range stored in **DRAM** for sub-block _𝐵 𝑗_ at State- _𝑖_ ; 

   - MeT _[𝑆]_[the][lower][(open)][bound][of][the][sub-batch] _𝑖,𝑗_[denotes] 

   - range stored in **SRAM** for sub-block _𝐵 𝑗_ at State- _𝑖_ . 

As defined above, MeT tracks the lower bound of sub-batches stored in memory. Meanwhile, ScT _𝑖,𝑗_ monitors the number of subbatches processed for block _𝐵 𝑗_ from State-1 to State- _𝑖_ , effectively representing the upper (closed) bound of that range. Therefore, ScT and MeT together define the range of sub-batches stored in SRAM and DRAM, denoted as (MeT _𝑖,𝑗[𝑆][,]_[ ScT] _[𝑖,𝑗]_[]][ and][ (][MeT] _𝑖,𝑗[𝐷][,]_[ ScT] _[𝑖,𝑗]_[]][,] respectively. Since the essence of layer fusion is to allocate intermediate data in SRAM to reduce costly DRAM accesses, MeT and ScT jointly capture the impact of fusion on memory behavior. 

**Construction of** MeT **.** In general, MeT is derived by following the construction rules. Eq. 7 ensures that the lower bounds of the sub-batches stored in memory are non-negative integers. Eq. 9 enforces that these lower bounds cannot exceed the upper bound represented by ScT _𝑖,𝑗_ , which tracks the number of sub-batches processed. Furthermore, Eq. 8 enforces a non-decreasing order for the lower bounds as the state progresses, reflecting the policy that earlier sub-batches are discarded first when memory (SRAM or DRAM) capacity is insufficient to store all sub-batches. This is based on the observation that newly generated data is more likely to be required by future computations, whereas previously generated sub-batches can be discarded temporarily. When sub-block _𝐵 𝑗_ depends on the data from sub-block _𝐵𝑚_ , Eq. 10 introduces a constraint to ensure that, if the output of sub-block _𝐵𝑚_ is required in memory, the necessary data is available in the previous state. Specifically, since ScT _𝑖_ −1 _,𝑗_ represents the upper bound of the sub-batch index processed by sub-block _𝐵 𝑗_ in State-( _𝑖_ − 1), the corresponding lower bound in State- _𝑖_ must be less than or equal to MeT _𝑖[𝑆]_ −1 _,𝑚_[and] 

1256 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

**==> picture [241 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
MeT [FW]<br>1 Min. Index  DRAM<br>of Stored  1 2<br>Sub-batches Output of  1<br>DNN Model2 State 3 …… S D S D Output of Index: [3]Index: [3]2 B2R2 : Backward: Recomputation<br>3 2 3 2<br>Forward<br>Step 1<br>Output of Output of Index: [3]Index: [3]DRAM 12 LossB2B1(2:3](2:3] Equ.14 # of Processed Sub-batchesState 1~2State 1~3State 1ScT [BW1] B2111 B1001 Sub-batchesMin. Index of Stored State 3MeT …… [BW1] S1 B2 D1 S1 B1 D1<br>Step 2<br>ScT [BW2] MeT [BW2]<br>DRAM (0:2]R1 B1(0:2] # of Processed Sub-batches R1 R2 B2 B1 Sub-batchesMin. Index of Stored  R1 R2 B2 B1<br>Index: [1][2]Input of  1 (0:2]R2LossB2(0:2] State 1~7Status-1Status-7State 1 …… 12 02 02 02 Status-7State 7 …… S2 D2 S2 D2 S2 D2 S2 D2<br>**----- End of picture text -----**<br>


**Figure 6: Recomputation Example. Consider a 2-layer model treated as a top-level block, with each layer as a sub-block. Suppose** ScT3 _,_ 1 = ScT **cating that the sub-batch range stored in DRAM for both Layer-1 and** 3 _,_ 2 = 3 **. After the forward pass, we have** MeT _[𝐷]_ 3 _,_ 1[=][ MeT] _[𝐷]_ 3 _,_ 2[=][ 2] **[, indi-] Layer-2 is** (2 _,_ 3] **.In Step 1, these stored activations are used for backward propagation over sub-batch** (2 _,_ 3] **, where the loss flows through Layer-2 and then Layer-1. This step involves only the backward pass and aligns with the optimization of** ScT **and** MeT **as defined in Eq. 14.In Step 2, recomputation is required for the evicted sub-batch** (0 _,_ 2] **. The input data for Layer-1 over this range is fetched from DRAM, and forward recomputation is performed for both layers, followed by the backward pass. This combined forward-backward workload also fits within the scheduling and memory optimization framework of** ScT **and** MeT **.** 

MeT _𝑖[𝐷]_ −1 _,𝑚_[. This ensures that sub-block] _[ 𝐵][𝑚]_[’s output is available for] sub-block _𝐵 𝑗_ when needed. Eq. 11 and 12 define the maximum storage capacity for SRAM and DRAM, respectively. In these equations, _𝑉𝑗_ represents the size of one sub-batch of sub-block _𝐵 𝑗_ ’s output, and ScT _𝑖,𝑗_ − MeT _𝑖,𝑗[𝑆]_[and][ ScT] _[𝑖,𝑗]_[−][MeT] _𝑖,𝑗[𝐷]_[represent the number of] sub-batches stored in SRAM and DRAM, respectively. These constraints ensure that the total data stored in memory does not exceed the available capacity of SRAM ( _𝐶𝑎𝑝[𝑆]_ ) and DRAM ( _𝐶𝑎𝑝[𝐷]_ ). 

**==> picture [209 x 100] intentionally omitted <==**

## **5.3 Representation of Recomputation Scheme** 

As analyzed in Section 2.1, recomputation is a critical decision variable in inter-layer scheduling, particularly for DNN training. Next we introduce how to use ScT and MeT to describe the recomputation process during training. In general, determining a recomputation strategy requires answering two key questions. 

_Question #1 (Where/Which): Which sub-blocks’ activations should be discarded and recomputed?_ 

_Question #2 (How): How can forward recomputation be coordinated with backward pass in the context of sub-batch-based processing?_ 

Notably, the scheduling space for answering Question #1 is vast. As discussed in Section 2.2, there are approximately 2[200] possible checkpoint choices for recomputation in a 4-sub-batch-based ResNet-50. Fortunately, this extensive search space can be effectively integrated into the construction and optimization of ScT and MeT after the forward pass, as these two tables precisely track the sub-batches stored in SRAM and DRAM. In other words, once ScT and MeT are determined, the location and amount of activations to be recomputed are automatically identified. 

Answering Question #2 is more challenging due to the interaction between forward recomputation and backward propagation, creating a bi-directional processing flow. To address this, we propose splitting the process into two distinct phases, each of which can be effectively described using ScT and MeT. 

**Step-1: Backward Pass-only Pre-processing.** As illustrated in Fig. 6, after forward propagation, a new pair of ScT and MeT tables are constructed to describe the action of backward propagating the sub-batches currently stored in DRAM. The goal in this phase is to consume as many of the DRAM-stored activations of sub-block _𝐵𝑚_ required by sub-block _𝐵 𝑗_ for the backward pass as possible. No further backward computation can occur in _𝐵 𝑗_ until forward recomputation in _𝐵𝑚_ generates the required data. The benefit of this ”pre-processing" arrangement is that it simplifies the data dependency of backward processing in each sub-block from two sources (stored activations and recomputed activations) to a single source (recomputed activations only). This ensures that the recomputation in Step-2 can always be performed prior to the backward computation, allowing ScT and MeT to represent the coupled recomputation and backward pass. 

To ensure the success of this arrangement, at the end of forward pass (State-(2 _𝑁_ − 1)), the amount of stored activation results for sub-block _𝐵𝑚_ should be no less than that of sub-block _𝐵 𝑗_ . This brings a new constraint when constructing MeT for forward pass: 

**==> picture [189 x 13] intentionally omitted <==**

After optimizing forward-specific MeT (denoted as MeT _[𝐹𝑊]_ ) with constraints described in Eq. 7-12 and Eq. 13, ScT for Step-1, denoted as ScT _[𝐵𝑊]_[1] , can be constructed using the same method as described in Section 5.1, as the processing in Step-1 is also one-directional. Specifically, Eq. 1-6 still serve as the constraints for table construction. The only difference is that Eq. 2 and 3 are replaced by the following constraint, considering the consumption of stored activation incurred by pre-processing (note that the indices of sub-batches in Eq. 14 are offset for consistency with physical meaning of ScT _𝑖,𝑗_ ): 

**==> picture [232 x 34] intentionally omitted <==**

Then, the corresponding MeT (MeT _[𝐵𝑊]_[1] ) can be derived from ScT _[𝐵𝑊]_[1] , constrained by Eq. 7-12. 

**Step-2: Forward Recomputation-then-Backward Pass.** After Step-1, another pair of tables, denoted as ScT _[𝐵𝑊]_[2] and MeT _[𝐵𝑊]_[2] , will be constructed. As illustrated in Fig. 6, the two tables in Step-2 consist of 2 _𝐿_ layers — the first and last _𝐿_ layers correspond to the recomputation phase and the backward pass phase, respectively. 

1257 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture 

Notably, because in Step-2, recomputation is only performed to recover the previously discarded sub-batches of activation data, Eq. 3 is replaced with the following constraint: 

**==> picture [210 x 13] intentionally omitted <==**

where MeT _[𝐷,𝐹𝑊]_ 2 _𝑁_ −1 _,𝑗_[is the index of the first sub-batch of activation] stored for sub-block _𝐵 𝑗_ in State-(2 _𝑁_ − 1) during forward propagation, minus 1 (to account for the open bound), and equivalently represents the index of the last sub-batch discarded. 

## **5.4 Latency & Energy Evaluation** 

Upon representing the execution scheme, fusion strategy, and recomputation scheme using ScT and MeT, we can construct performance model to evaluate latency and energy for a given scheduling configuration, preparing for scheduling space exploration. 

**Latency Evaluation.** The latency performance is modeled based on delays incurred from PE computation and data traffic. 

_Computation-incurred Latency._ At the hardware tile granularity, the computation-incurred latency of each state is determined by the workload and tile utilization. The workload of sub-block _𝐵 𝑗_ (which could be a single layer or multiple layers within a block) for processing one sub-batch, denoted as Workload _𝑗_ , is the number of FLOPs required by that sub-block. To evaluate hardware utilization, we follow the method adopted in SET: For _𝑇𝑖,𝑗_ tiles allocated for sub-block _𝐵 𝑗_ in State- _𝑖_ , the corresponding utilization ratio _𝑢𝑖,𝑗_ is calculated by mapping the four dimensions (dim _𝑞_ ) of one sub-batch onto the four factors of _𝑇𝑖,𝑗_ ( _𝑘𝑖,𝑗_[1][×] _[ 𝑘] 𝑖,𝑗_[2][×] _[ 𝑘] 𝑖,𝑗_[3][×] _[ 𝑘] 𝑖,𝑗_[4][=] _[𝑇][𝑖,𝑗]_[)][and] calculating the product of utilization across these dimensions: 

**==> picture [159 x 31] intentionally omitted <==**

Then the computation latency of sub-block _𝐵 𝑗_ in State- _𝑖_ is as: 

**==> picture [165 x 20] intentionally omitted <==**

where _𝑃_ denotes the computing power per unit time for each tile. Since computation-incurred latency for a DNN in one state is determined by the bottleneck sub-block in that state (as sub-blocks are processed concurrently within one state), and sub-block execution in different states happens sequentially, the overall computationincurred latency _𝐿_ comp for the entire model can be calculated as: 

**==> picture [175 x 19] intentionally omitted <==**

Note that Eq. 18 describes the latency evaluation for forward propagation. It can be easily extended for backward propagation by including the computation of gradients. 

_Data Traffic-incurred Latency._ From the perspective of a hardware tile, the latency incurred by transferring its required input data consists of three parts: the cost incurred by reading data from DRAM, SRAM, and other hardware tiles. Since the tiled architecture uses NoC as a unified fabric to transfer data, the data traffic-incurred latency can be evaluated as: 

**==> picture [236 x 35] intentionally omitted <==**

**==> picture [194 x 128] intentionally omitted <==**

**----- Start of picture text -----**<br>
Inference OptimizationSchedule  Backward<br>Forward ExplorationIntra-layer  Forward ScT ... ScT ScT<br>Forward TrainingBackward ExplorationIntra-layer  𝑩𝐒𝐮𝐛1 ScT ...... 𝑩 ScT 𝐒𝐮𝐛4 ...... 𝑩 ScT 𝐒𝐮𝐛6 Select  𝑩𝐒𝐮𝐛2 Top K3 𝑩 Based on  𝐒𝐮𝐛4 𝑩𝐒𝐮𝐛5 Cost  comp Output<br>Select  Top K1 Based on  Cost  comp<br>Input 𝑩𝐒𝐮𝐛𝟏 v 1 2 𝑩𝐒𝐮𝐛𝟏 ... 𝑩𝐒𝐮𝐛4 ... 𝑩𝐒𝐮𝐛6 ... Optimal Sub-batch Size 𝑩𝐒𝐮𝐛3<br>𝑩𝐒𝐮𝐛2 DNN Model3 Select  MeT Top K ... 2 Based on  MeT ... Cost MeT traffic Select  MeT Top K4 Based on  MeT MeT Cost traffic tiles t<br>𝑩𝐒𝐮𝐛3 R R 𝑩𝐒𝐮𝐛2 𝑩𝐒𝐮𝐛4 𝑩𝐒𝐮𝐛6 𝑩𝐒𝐮𝐛3 𝑩𝐒𝐮𝐛4 𝑩𝐒𝐮𝐛5 Optimal ScT<br>𝑩𝑩𝐒𝐮𝐛5𝐒𝐮𝐛4 ArchitectureTiled R R ... ... DRAM Cap. t<br>Sub-batch Size Candidates 𝑩𝐒𝐮𝐛6 SRAM/DRAM Capacity v ExplorationIntra-layer  𝑩𝐒𝐮𝐛3 v Optimal MeTSRAM Cap. t<br>**----- End of picture text -----**<br>


**Figure 7: Overall intra-block exploration and optimization process.** 

where _𝑉𝑚_ is the storage size for one sub-batch of data and the corresponding weight, and _𝐵𝑊𝑁_ is the bandwidth of NoC. Here, for data movement from sub-block _𝐵𝑚_ to sub-block _𝐵 𝑗_ in State- _𝑖_ , Dep _𝑚,𝑖,𝑗[𝐶]_[,][ Dep] _𝑚,𝑖,𝑗[𝑆]_[, and][ Dep] _𝑚,𝑖,𝑗[𝐷]_[represent the amount of data trans-] ferred from other hardware tiles, SRAM, and DRAM, respectively. Since ScT and MeT record all the computing and memory statuses for all sub-blocks, the amount of these three types of data transfer can be easily calculated from the two tables. Additionally, _𝐻𝐶_ , _𝐻𝑆_ , and _𝐻𝐷_ represent the hop counts for transferring these data through NoC, while _𝐵𝑊𝐷_ represents the bandwidth of DRAM. The latency for data sourced from DRAM is included in Eq. 19 since it is transferred via NoC but must first be read from the DRAM. **Energy Evaluation.** Energy costs are evaluated similarly by considering the consumption due to computation and data traffic. _Computation-incurred Energy Consumption._ Eq. 20 describes the evaluation model for computation-incurred energy cost ( _𝐸_ comp). Here, _𝐸_ comp, unit is the unit energy consumption for each operation, and _[𝑠][𝑖]_[Workload] _𝑢𝑖,𝑗[𝑗]_ represents the equivalent number of FLOPs required for computation, adjusted for the tile utilization. 

**==> picture [185 x 23] intentionally omitted <==**

_Data Traffic-incurred Energy Consumption._ For the tiled architecture, the energy consumption due to data traffic has two sources: data movement through NoC and from/to DRAM. Eq. 21 describes the evaluation of the total energy cost incurred by transferring data at the tile level. Here, _𝐸_ NoC, unit and _𝐸_ DRAM, unit are the unit energy consumption per hop and access, respectively. 

**==> picture [210 x 35] intentionally omitted <==**

## **5.5 Overall Exploration & Optimization Process** 

Based on the table-format schedule representations and the modeled cost function for hardware performance, we are now ready to describe the automatic exploration process for inter-layer scheduling. Note that selecting the proper batch splitting plan ( _𝐵𝑆_ sub), as another important decision factor affecting the overall scheduling scheme, will be integrated into the search process. 

Fig. 7 shows the overall exploration for inter-layer scheduling. Given the pre-determined batch size _𝐵𝑆_ by the workload, we first enumerate all its factors as the candidates for _𝐵𝑆_ sub. Then, for each 

1258 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

possible _𝐵𝑆_ sub, the corresponding ScT is built with the constraints described by Eq. 1 - 6. Since all constraints are linear, we can apply piecewise linear approximation and solve the problem using an MILP solver to find the corresponding _𝑠𝑖_ ’s that minimize the computation-related EDP, a bilinear term that can be linearized via McCormick envelope [25]: 

**==> picture [175 x 15] intentionally omitted <==**

where _𝑝_ and _𝑞_ are hyper-parameters setting the importance of latency and energy, respectively. With the above-identified _𝑠𝑖_ ’s, we can specify ScT’s and further calculate the corresponding minimized EDP cost for each _𝐵_ sub candidate. After preserving _𝐾_ 1 possible _𝐵_ sub’s with the smallest _𝐾_ 1 _𝐿_ comp _[𝑝]_[×] _[ 𝐸][𝑞]_ comp[’s, we further use] them and the corresponding _𝑠𝑖_ ’s to build _𝐾_ 1 MeT’s. Again, an MILP solver is applied to minimize the data traffic-related EDP cost: 

**==> picture [212 x 20] intentionally omitted <==**

The candidate list for _𝐵_ sub is then further shrunk by preserving only _𝐾_ 2 _𝐵_ sub’s that correspond to the smallest _𝐿_ traffic _[𝑝]_[×] _[𝐸][𝑞]_ traffic[’s. Note] that for the training workload, another round of table construction and _𝐵_ sub candidate reduction are needed. 

Finally, intra-layer scheduling is applied to the remaining _𝐾_ 2 candidates to update total latency and energy consumption after considering the optimization within each hardware tile as follows: 

**==> picture [226 x 52] intentionally omitted <==**

where _𝛼𝑖,𝑗_ is the factor considering the actual hardware utilization for processing sub-block _𝐵 𝑗_ in State- _𝑖_ . _𝛽𝑖,𝑗_ and _𝛾𝑖,𝑗_ are the corresponding data traffic-related latency and energy consumption associated with intra-layer scheduling. Here, _𝛼𝑖,𝑗_ , _𝛽𝑖,𝑗_ , and _𝛾𝑖,𝑗_ are obtained from the intra-layer scheduler. Thanks to the abstraction of hardware tiles in inter-layer scheduling, the existing intra-layer schedulers can be easily applied to our framework as a plug-in. 

_**Remark**_ 3 ( **Addressing Challenge #1: Unified Representation of Core Scheduling Factors** ) _._ The ScT and MeT table formats, together with sub-batch size selection ( _𝐵𝑆_ sub), provide a unified and extensible representation of inter-layer scheduling behavior. These tables explicitly record execution states and memory status, which are determined by four design factors (E,P,R,B). This joint representation ensures that all key design factors are consistently captured within a single framework—directly addressing Challenge #1. 

_**Remark**_ 4 ( **Addressing Challenge #3: Structured and Exhaustive Scheduling via MILP** ) _._ The table-based formulation using ScT and MeT naturally leads to an MILP, which encodes scheduling constraints with high structural regularity. This enables exhaustive exploration of the scheduling space using off-the-shelf MILP solvers, offering both fast convergence and globally optimal solutions for intra-block scheduling—effectively addressing Challenge #3. 

## **6 Hierarchical Structure Optimization** 

After optimizing intra-block scheduling, the next step is to optimize hierarchical block structure. This involves refining the arrangement and organization of blocks to improve the overall scheduling efficiency. To that end, we propose an iterative two-step solution: 

**Step-1: Graph Partition.** In this step, the DNN model is partitioned based on the dependency relationships between layers in its computational graph. Specifically, layers are grouped into blocks by examining these dependencies. If layers form a sequential dependency chain, where each layer directly depends on the previous one and only outputs its result to the next, they are combined into a single block. For example, Layer-2 in Fig. 8 solely depends on Layer-1, and Layer-3 solely depends on Layer-2, so Layers-2 and -3 are grouped into Block-B. When a layer or block has multiple dependencies or outputs its results to multiple layers or blocks, a higher-level block is formed to capture these relationships. Additionally, each layer can also function as a single block, enabling flexible sub-batch size optimization. Following these principles, Block-A and Block-C are constructed to contain Layer-1 and Layer4, respectively, while Block-D is formed by consolidating Block-A, B, and C, thus capturing all inter-block dependencies. 

**Step-2: Gradual Partition.** This step optimizes overall scheduling scheme by incrementally updating and optimizing lower-level blocks. Initially, the workload of each partitioned block is set as the corresponding FLOP counts, without considering tile utilization. Similarly, the initialization of data traffic for each block does not account for impact of potential optimization in lower-level blocks. The optimization process begins with top-level block, for example, Block-D in Fig. 8. From the perspective of Block-D, lower-level blocks (Block-A, B, and C) serve as component layers. We optimize the inter-layer scheduling for Block-D using the process illustrated in Fig. 7, where the cost evaluation is based on the FLOPs and data traffic information of component blocks. After finishing the schedule optimization for Block-D, the newly obtained optimal cost is compared with the initial cost (initially set to zero). If the difference exceeds a pre-set threshold _𝜃_ , the scheduling for Block-A, B, and C will be re-optimized. For each lower-level block, the batch size, allocated hardware tiles, and available SRAM capacity may be updated after optimizing the top-level block. These updated parameters are then used in the schedule optimization for the lower-level block. 

As shown in Fig. 8, cost-driven block partitioning process is used to identify optimal block structure for each lower-level block. Consider Block-B as an example. Since it consists of two layers, two possible scenarios are evaluated: (1) Layer-2 and Layer-3 are treated as separate blocks without forming a higher-level block, (2) they form a hierarchical nested block. After optimizing for both cases, block structure with lower cost is selected as optimal structure for these layers. This gradual partitioning is performed for all hierarchical nested block structures until reaching bottom level. 

**Iterative Update.** Once the optimization of lower-level blocks is complete, the workload and data traffic for each block are determined and passed back to the higher-level block. Based on this updated information, the scheduling for the higher-level block is re-optimized. This iterative procedure continues until a predefined convergence condition for the top-level block is met. 

## **7 Evaluation** 

## **7.1 Validation of Cost Evaluator** 

Crane is implemented in C++ for comprehensive scheduling exploration. To validate its cost model, we compare it to SET, which reports less than 3% deviation from cycle-accurate simulations of 

1259 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture 

**==> picture [380 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 1 A WorkloadWorkloadWorkloadABC, Traffic, Traffic, TrafficABC Optimization of Block-DSchedule  CostNew<  − Cost 𝜽 Prev Yes Step-2Output Tiles9 D<br>2 2 Workload & Traffic Iterative Update NoLower Block Opt.Gradual Partition 6<br>4 3 z D 4 C3 B 23 B E2 2 Optimization of Block-ESchedule 3 Schedule F3 Optimization of Block-FSchedule  z Lower Cost WorkloadTiles2 223 B Traffic33 B 31 A B C State<br>Step-1 Block-B G H B Optimization of Block B State Scheduling of Nested<br>Graph Partition Block-A Blocks<br>Block-C<br>**----- End of picture text -----**<br>


**Figure 8: Exploration and optimization process using nested block-based representation. The schedule optimization module refers to Fig. 7.** 

multi-tile compute and memory access behavior using zsim [30] and DRAMSim2 [29]. Both Crane and SET employ brute-force intralayer scheduling. During validation, individual inter-layer scheduling is disabled and replaced with a common manually configured scheme, and hardware settings match those of SET. We evaluate various DNN models (VGG, ResNet-50, Transformer-Large, InceptionV3) across different batch sizes. Results show Crane achieves 1% deviation from SET and 4% transitive deviation from cycle-accurate simulation baselines, confirming high fidelity of our cost model. 

## **7.2 Evaluation on Inference** 

**Baseline.** We compare our approach with Tangram and SET by enabling their respective inter-layer scheduling processes. Unlike Crane, which supports both inference and training, these frameworks are designed for inference. To ensure a fair comparison, we disable Crane’s exploration and optimization for recomputation and the backward pass, focusing only on inference performance. 

**Hardware Configuration.** For fair comparison, we follow SET by adopting the NVDLA-style architecture as the hardware tile for both 16-tile edge-side and 144-tile cloud-side platforms, operated at 1GHz clock frequency with TSMC 12nm process. Each tile is equipped with _𝑃_ = 1024 int8 MACs and a 1 MB SRAM. The energy consumption for each MAC operation is _𝐸𝑐𝑜𝑚𝑝,𝑢𝑛𝑖𝑡_ = 0 _._ 018 pJ/op. We adopt a meshed NoC to connect the tiles, with a bandwidth of _𝐵𝑊𝑁𝑜𝐶_ = 24 GB/s, and the unit energy consumption for each hop is _𝐸𝑁𝑜𝐶,𝑢𝑛𝑖𝑡_ = 0 _._ 7 pJ/bit. Following SET, the DRAM bandwidth is set as _𝐵𝑊𝐷_ = 0 _._ 5 GB/TOPs (i.e., 16 GB/s, 144 GB/s for edge-side and cloud-side architecture, respectively), and the corresponding unit energy consumption is _𝐸𝐷𝑅𝐴𝑀,𝑢𝑛𝑖𝑡_ = 7 _._ 5 pJ/bit. DRAM capacity is not constrained here since inference is not sensitive to it. The energy consumption for various register, buffer and SRAM sizes associated with the intra-layer exploration is obtained via ARM Memory Compiler [1]. To explore the search space, we set the hyperparameter in our framework as _𝐾_ 1 = ⌈0 _._ 5 × _𝑁𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒_ ⌉, and _𝐾_ 2 = ⌈0 _._ 2 × _𝑁𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒_ ⌉, where _𝑁𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒_ is the candidate amount of the _𝐵𝑆𝑢𝑏_ . _𝜃_ is configured as 2%× _𝐶𝑜𝑠𝑡𝑃𝑟𝑒𝑣_ . The parameters _𝑝_ = 1 _,𝑞_ = 1 are specified in the cost function. 

**Workloads & Performance.** We benchmark ResNet-50 and GoogleNet on ImageNet, as well as Transformer-Large (12 layers, 16 heads, sequence length 512, hidden dimension 1024), GPT-2 (16 layers, 16 heads, sequence length 1024, hidden dimension 1024), and OPT-6.7B (32 layers, 32 heads, sequence length 2048, hidden dimension 4096) with various batch sizes ( _𝐵𝑆_ ). These models are evaluated as workloads on both edge and cloud-side platforms. Fig. 9 shows the latency and energy performance achieved by the optimal scheduling strategies from schedulers. It is seen that Crane substantially 

outperforms Tangram and SET. Compared with Tangram, Crane reduces latency by 1 _._ 70× – 3 _._ 30×, cuts energy consumption by 0 _._ 84× – 1 _._ 49×, and lowers EDP by 1 _._ 87× – 4 _._ 20×. In comparison to SET, Crane achieves latency reductions of 1 _._ 12× – 3 _._ 02× (averaging 1 _._ 64×), decreases energy consumption by 1 _._ 01× – 1 _._ 38× (averaging 1 _._ 21×), and reduces EDP by 1 _._ 13× – 4 _._ 17× (averaging 1 _._ 84×). 

**Comparison with TileFlow.** We also evaluate the performance of Crane against TileFlow on a 4 × 4 tiled architecture with a batch size of 64. Fig. 10 shows Crane outperforms TileFlow significantly: for ResNet-50, it achieves 28% lower latency, 39% lower energy consumption, and a 56% reduction in EDP. For BERT-Base, the corresponding reductions are 13%, 17%, and 28%, respectively. 

## **7.3 Evaluation on Training** 

**Baseline.** To evaluate training performance of Crane, we compare its results with those of MBS [23], an inter-layer scheduling method specialized for DNN training. MBS employs batch-splitting and layer fusion strategies to optimize data traffic and accelerate training on tiled accelerators. Additionally, since SET and Tangram are specifically designed for inference and its source code does not support training workloads, and in particular, it lacks support for recomputation, we estimate the potential performance of a hypothetical training-oriented SET and Tangram by applying the scheduling optimization from inference-only scheduling separately to forward and backward passes. The total estimated cost is then obtained by summing the resulting latency and energy metrics. 

**Hardware Configuration.** For fair comparison, we follow the settings of MBS and configure the target computing platform with two tiled systolic array. Each hardware tile features a MAC array size of _𝑃_ = 16384 (128 × 128) and 10 MB of SRAM. For highbandwidth data movement, the on-chip NoC bandwidth ( _𝐵𝑊𝑁𝑜𝐶_ ) is set to 100 GB/s, with energy consumption of _𝐸𝑁𝑜𝐶,𝑢𝑛𝑖𝑡_ = 0 _._ 7 pJ/bit. Additionally, a 32 GB HBM2 off-chip memory provides a bandwidth of _𝐵𝑊𝐷_ = 300 GB/s, consuming _𝐸𝐷𝑅𝐴𝑀,𝑢𝑛𝑖𝑡_ = 3 _._ 9 pJ/bit [27]. Our evaluation follows the intra-layer computation scheme proposed in MBS. The search hyper-parameters are configured as _𝐾_ 3 = ⌈0 _._ 1 × _𝑁𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒_ ⌉ and _𝐾_ 4 = ⌈0 _._ 05 × _𝑁𝐶𝑎𝑛𝑑𝑖𝑑𝑎𝑡𝑒_ ⌉, while other parameters adhere to the configurations in Section 7.2. For the comparison with the hypothetical training-oriented SET, the cloud-side architecture is utilized and the configurations remain the same except DRAM size is set as 128 GB for training procedure. 

**Workloads & Performance.** For the training evaluations with MSB, we select the ResNet-50, ResNet-101, Inception-V3, and InceptionV4 models. These models are trained on the ImageNet dataset using a batch size of 256. As illustrated in Fig. 11, the scheduling generated by our framework outperforms MBS’s solution with respect to 

1260 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

**==> picture [506 x 70] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.61.41.20.80.60.40.21 32.521.510.5 1.61.41.20.80.60.40.21 3.532.521.510.5<br>0 0 0 0<br>B = 1ResNet50B = 8 B = 64 B = 1GoogleNetB = 8 B = 64 B = 1Transformer-LargeB = 8 B = 64 B = 1 GPT-2B = 8 B = 64 B = 1OPT-6.7BB = 8 B = 64 B = 1ResNet50B = 8 B = 64 B = 1GoogleNetB = 8 B = 64 B = 1Transformer-LargeB = 8 B = 64 B = 1 GPT-2B = 8 B = 64 B = 1OPT-6.7BB = 8 B = 64<br>(a) Edge-Side Architecture (b) Cloud-Side Architecture<br>TangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCrane TangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCraneTangramSETCrane<br>Normalzied Energy  Consumption Normalized Latency Normalzied Energy  Consumption Normalized Latency<br>**----- End of picture text -----**<br>


**Figure 9: Performance of inter-layer schedulers for inference workloads. Compared with Tangram, Crane reduces energy-delay product (EDP) by** 1 _._ 87× **–** 4 _._ 20× **(averaging** 3 _._ 03× **) across all platforms, models, and batch sizes, and by** 1 _._ 13× **–** 4 _._ 17× **(averaging** 1 _._ 84× **) compared with SET. The average runtime speedup for scheduling achieved by Crane is** 2 _._ 82× **and** 156 _._ 20× **compared to SET and Tangram, respectively.** 

**==> picture [242 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
TileFlow Ours<br>1<br>0.5<br>0<br>Latency Energy EDP Latency Energy EDP<br>ResNet50 Bert-Base<br>Normalized  Performance<br>**----- End of picture text -----**<br>


**Figure 10: Crane achieves 28%, 39%, and 56% reductions over TileFlow in latency, energy and EDP, respectively, on ResNet50. On Bert-Base, Crane delivers 13%, 17%, and 28% reductions in the same metrics.** 

**==> picture [218 x 121] intentionally omitted <==**

**----- Start of picture text -----**<br>
4<br>Latency Energy Consumption DRAM Traffic<br>3<br>2<br>1<br>0<br>ResNet50 ResNet101 InceptionV3 InceptionV4<br>(a) Comparison with MBS<br>12<br>9 Latency Energy Consumption DRAM Traffic<br>6 Out-of-memory<br>error in SET and<br>3 Tangram<br>0<br>SET Tangram SET Tangram SET Tangram SET Tangram<br>ResNet50 Transformer-large GPT-2 OPT-6.7B<br>(b) Comparison with hypothetical training-oriented SET and Tangram<br>(Times)<br>Normalized  Reduction<br>(Times)<br>Normalized  Reduction<br>**----- End of picture text -----**<br>


**Figure 11: Performance of inter-layer schedulers for training workload. (a) Crane outperforms MBS with** 1 _._ 42× **–** 2 _._ 22× **lower DRAM data traffic and** 3 _._ 62× **–** 5 _._ 36× **lower EDP. (b) Compared to trainingoriented SET, Crane reduces DRAM traffic by** 2 _._ 76× **–** 2 _._ 97× **and EDP by** 11 _._ 01× **–** 21 _._ 01× **; against Tangram, the reductions are** 4 _._ 81× **–** 5 _._ 72× **and** 45 _._ 43× **–** 64 _._ 73× **, respectively. Notably, SET and Tangram fail to schedule OPT-6.7B due to out-of-memory, while Crane successfully generates a DRAM-feasible schedule.** 

latency, energy consumption, DRAM data traffic, and total EDP cost per training step, achieving reduction of 1 _._ 64×, 2 _._ 54×, 1 _._ 67×, and 4 _._ 18× on average across various models. These advancements are attributed to two primary factors: 1) MBS adopts a layer sequential processing pattern, whereas Crane applies a combination of three patterns for scheduling; and 2) MBS relies on a heuristic approach to determine batch splitting and layer fusion settings, while our framework optimizes all the four design factors of scheduling. 

To compare with hypothetical training-oriented SET and Tangram, we select ResNet-50 on ImageNet, Transformer-Large, GPT-2, and OPT-6.7B as models for training evaluation results. The batch size for training is 128. As shown in Fig. 11, Crane reduces latency by 5 _._ 16×–6 _._ 96×, energy consumption by 2 _._ 15×–3 _._ 04×, and DRAM traffic by 2 _._ 76×–2 _._ 97× over SET, achieving 11 _._ 01×–21 _._ 01× lower EDP. 

**==> picture [194 x 71] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.2<br>1 Energy Latency EDP<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>32 16 8 4 2 1<br>Sub-batch Size<br>Normalized  Performance<br>**----- End of picture text -----**<br>


**Figure 12: Energy, latency, and EDP of ResNet-50 training under varying sub-batch sizes, with a fixed total batch size of 32. A subbatch size of 32 (i.e., no batch splitting) results in much higher costs, highlighting the importance of effective batch splitting plan (B).** 

**==> picture [242 x 85] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.5 Without Recomputation With Recomputation<br>1<br>0.5<br>0 VGG16 ResNet50 Transformer-Large InceptionV4 VGG16 ResNet50 Transformer-Large InceptionV4<br>DRAM Capacity RequirementMemory Capacity Requirement DRAM AccessData Access<br>Figure 13: With recomputation enabled, Crane largely reduces<br>DRAM capacity requirement with minor overhead (Batch size= 64).<br>Normalized Performance<br>**----- End of picture text -----**<br>


Compared to Tangram, Crane reduces latency by 7 _._ 86×–10 _._ 26×, energy by 5 _._ 69×–6 _._ 41×, and DRAM traffic by 4 _._ 81×–5 _._ 72×, leading to 45 _._ 43×–64 _._ 73× EDP savings. Notably, the hypothetical trainingoriented SET and Tangram cannot explore the inter-layer schedule for OPT-6.7B training due to an out-of-memory error. This occurs because the memory consumption for training this model exceeds DRAM capacity, while their searched schedule lacks the ability to explore recomputation opportunities that could reduce memory usage. In contrast, Crane successfully generates an optimized schedule that enables training within the given DRAM budget. 

## **7.4 Ablation Study & Analysis** 

**Impact of Batch Splitting (B).** As shown in Figure 12, using no batch splitting (i.e., a sub-batch size of 32) results in higher energy, latency, and EDP compared to configurations with smaller subbatches, demonstrating the effectiveness of batch-level partitioning. Additionally, the results reveal a trade-off between computation and data movement overhead. Smaller sub-batches improve onchip buffer utilization and allow more aggressive layer fusion, thus reducing memory access cost. However, they also lead to lower PE utilization and repeated loading of weight data, which increases computation overhead. Crane’s search process can systematically explore this trade-off to identify an optimal sub-batch configuration. 

1261 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Crane: Inter-Layer Scheduling Framework for DNN Inference and Training Co-Support on Tiled Architecture 

**==> picture [242 x 205] intentionally omitted <==**

**----- Start of picture text -----**<br>
Latency Energy Consumption EDP<br>1<br>0.5<br>0<br>Pipeline Sequential Flexible Scheme Pipeline Sequential Flexible Scheme<br>ResNet50 Transformer-large<br>Figure 14: When exploring execution scheme is enabled, Crane iden-<br>tifies better scheduling solution with reduced latency, energy and<br>EDP. Model inference on cloud architecture with batch size 64.<br>16 16<br>15 15<br>14 14<br>13 13<br>12 12<br>11 11<br>10 10<br>9 9<br>8 8<br>7 7<br>6 6<br>5 5<br>4 4<br>3 3<br>2 2<br>1 1<br>0 10000 20000 30000 40000 50000 0 10000 20000 30000 40000 50000<br>Time(cycles) Time(cycles)<br>(a) Execution Graph of SET (b) Execution Graph of Crane<br>Normalized Performance<br>Tiles Tiles<br>**----- End of picture text -----**<br>


**Figure 14: When exploring execution scheme is enabled, Crane identifies better scheduling solution with reduced latency, energy and EDP. Model inference on cloud architecture with batch size 64.** 

**Figure 15: Tile-Time schedule for Inception inference (Crane vs SET).** 

**==> picture [242 x 73] intentionally omitted <==**

**----- Start of picture text -----**<br>
25 1.2 30 1.2<br>20 1 25 1<br>15 0.8 20 0.8<br>0.6 15 0.6<br>10 0.4 10 0.4<br>5 Cost Runtime 0.2 5 Cost Runtime 0.2<br>0 0 0 0<br>4 6 8 10 12 0 5 10 15 20<br>𝜽 (Percentage of Previous Cost ) 𝑲𝟐<br>(a) Sensitivity of  𝜽 (b) Sensitivity of  𝑲𝟐<br>Iteration<br>Cost (ms x mJ) Cost(ms x mJ)<br>Normalized Runtime Normalized Runtime per<br>**----- End of picture text -----**<br>


**Figure 16: Runtime and EDP cost analysis of** _𝜃_ **and** _𝐾_ 2 **. Increasing** _𝜃_ **and decreasing** _𝐾_ 2 **reduce runtime but raise cost.** 

high-level and lower-level blocks—achieves fast convergence, unlike the slower, randomized simulated annealing used in SET. 

**Sensitivity Analysis.** The parameters _𝐾_ 1– _𝐾_ 4 and _𝜃_ affect runtime and scheduling quality: _𝐾_ 1– _𝐾_ 4 control the number of subbatch candidates retained per iteration (impacting per-step runtime), while _𝜃_ sets the total number of iterations. Fig. 16 shows for VGG inference on an edge platform, increasing these values improves scheduling but increases runtime. Notably, the same _𝐾_ 1– _𝐾_ 4 and _𝜃_ are used across all experiments in Section 7.2 and 7.3. 

## **8 Conclusion** 

**Impact of Recomputation (R).** Fig. 13 shows that enabling recomputation in Crane reduces DRAM capacity requirements by 2 _._ 2× on average, with only a modest 0 _._ 125× increase in data access overhead. This demonstrates the value of integrating recomputation strategies into the scheduling framework for training efficiency. 

**Impact of Execution Scheme (E).** Fig. 14 highlights the importance of exploring execution schemes. With this exploration enabled, Crane achieves average reductions of 2 _._ 6× in latency, 1 _._ 8× in energy, and 4 _._ 7× in EDP compared to the case turning off execution scheme search. These gains stem from mitigating data reloading overheads in sequential scheduling and reducing pipeline bubbles in fully pipelined execution. 

**Finer-Grained Sub-batch Optimization Analysis.** We use an inference example of Inception-ResNet-V1 model on edge-sided accelerator with _𝐵𝑆_ = 2 and _𝐵𝑆𝑠𝑢𝑏_ = 1 to demonstrate how the scheduling derived from our framework eliminates bubble overhead, as shown in Fig. 15. Each colored block represents a layer, with the horizontal axis representing the execution time and the vertical axis indicating the allocated tiles for that layer. Unlike SET, which restricts the search space to coarse-grained sub-batch scheduling, Crane expands the space to support flexible mappings across subbatches of different layers. This broader exploration enables higher hardware performance through finer-grained scheduling. 

**Runtime Analysis.** The end-to-end runtime of all frameworks is evaluated on an AMD EPYC 7402P CPU. Compared to SET and Tangram, Crane achieves 2 _._ 82× and 156 _._ 20× speedup, respectively. This substantial gain—despite Crane’s larger search space—is enabled by three key factors: (1) For a given _𝐵_ sub, each block’s optimization is formulated as an MILP problem solvable by efficient solvers; (2) The hierarchical search space is effectively pruned by (a) restricting sub-batch exploration to Top- _𝐾_ 1, _𝐾_ 2 candidates based on _𝐶𝑜𝑠𝑡_ comp and _𝐶𝑜𝑠𝑡_ traffic, and (b) applying cost-driven block refinement to avoid enumerating poor nested structures; (3) The deterministic, hierarchical refinement process—alternating between 

We present Crane, a unified inter-layer scheduling framework for tiled architectures that supports both inference and training. By leveraging a hierarchical table-format representation, Crane captures essential design factors, enables flexible scheduling, and transforms scheduling into a structured optimization problem. Experimental results show substantial gains over existing works. 

## **Acknowledgments** 

This work was supported in part by the National Science Foundation under Grants CCF-2529764, CCF-2425399 and CCF-2529763, and by a Hans Fischer Senior Fellowship at the Technical University of Munich. 

## **References** 

- [1] [n. d.]. _ARM Downloads Beta - Artisan_ . https://developer.arm.com/downloadsbeta/search?term=artisan 

- [2] Manoj Alwani, Han Chen, Michael Ferdman, and Peter Milder. 2016. Fused-Layer CNN Accelerators. In _2016 49th Annual IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 1–12. 

- [3] Chen Bai, Xuechao Wei, Youwei Zhuo, Yi Cai, Hongzhong Zheng, Bei Yu, and Yuan Xie. 2024. Klotski v2: Improved DNN Model Orchestration Framework for Dataflow Architecture Accelerators. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ (2024). 

- [4] Olivier Beaumont, Lionel Eyraud-Dubois, and Alena Shilova. 2021. Efficient Combination of Rematerialization and Offloading for Training DNNs. _Advances in Neural Information Processing Systems_ 34 (2021), 23844–23857. 

- [5] Jingwei Cai, Yuchen Wei, Zuotong Wu, Sen Peng, and Kaisheng Ma. 2023. InterLayer Scheduling Space Definition and Exploration for Tiled Accelerators. In _Proceedings of the 50th Annual International Symposium on Computer Architecture_ . 1–17. 

- [6] Hongzheng Chen, Cody Hao Yu, Shuai Zheng, Zhen Zhang, Zhiru Zhang, and Yida Wang. 2024. Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ . 1095–1111. 

- [7] Yonggan Fu, Yongan Zhang, Yang Zhang, David Cox, and Yingyan Lin. 2021. Auto-NBA: Efficient and Effective Search over the Joint Space of Networks, Bitwidths, and Accelerators. In _International Conference on Machine Learning_ . PMLR, 3505–3517. 

- [8] Mingyu Gao, Jing Pu, Xuan Yang, Mark Horowitz, and Christos Kozyrakis. 2017. TETRIS: Scalable and Efficient Neural Network Acceleration with 3D Memory. 

1262 

MICRO ’25, October 18–22, 2025, Seoul, Republic of Korea 

Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang, Zhexiang Tang, Jiang Hu, and Bo Yuan 

   - In _Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems_ . 751–764. 

- [9] Mingyu Gao, Xuan Yang, Jing Pu, Mark Horowitz, and Christos Kozyrakis. 2019. TANGRAM: Optimized Coarse-Grained Dataflow for Scalable NN Accelerators. In _Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems_ . 807–820. 

- [10] Raveesh Garg, Hyoukjun Kwon, Eric Qin, Yu-Hsin Chen, Tushar Krishna, and Liangzhen Lai. 2024. PipeOrgan: Efficient Inter-operation Pipelining with Flexible Spatial Organization and Interconnects. _arXiv preprint arXiv:2405.01736_ (2024). 

- [11] Kartik Hegde, Po-An Tsai, Sitao Huang, Vikas Chandra, Angshuman Parashar, and Christopher W Fletcher. 2021. Mind Mappings: Enabling Efficient AlgorithmAccelerator Mapping Space Search. In _Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ . 943–958. 

- [12] Qijing Huang, Minwoo Kang, Grace Dinh, Thomas Norell, Aravind Kalaiah, James Demmel, John Wawrzynek, and Yakun Sophia Shao. 2021. Cosa: Scheduling by Constrained Optimization for Spatial Accelerators. In _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 554–566. 

- [13] Qijing Huang, Po-An Tsai, Joel S Emer, and Angshuman Parashar. 2024. Mind the Gap: Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 150–166. 

- [14] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, and Zhifeng Chen. 2019. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. _Advances in neural information processing systems_ 32 (2019). 

- [15] Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Joseph Gonzalez, Kurt Keutzer, and Ion Stoica. 2020. Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization. _Proceedings of Machine Learning and Systems_ 2 (2020), 497–511. 

- [16] Norman P. Jouppi, Doe Hyun Yoon, Matthew Ashcraft, Mark Gottscho, Thomas B. Jablin, George Kurian, James Laudon, Sheng Li, Peter Ma, Xiaoyu Ma, Thomas Norrie, Nishant Patil, Sushma Prasad, Cliff Young, Zongwei Zhou, and David Patterson. 2021. Ten Lessons From Three Generations Shaped Google’s TPUv4i: Industrial Product. In _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 1–14. 

- [17] Norman P. Jouppi, Cliff Young, Nishant Patil, David Patterson, Gaurav Agrawal, Raminder Bajwa, Sarah Bates, Suresh Bhatia, Nan Boden, Al Borchers, Rick Boyle, Pierre-luc Cantin, Clifford Chao, Chris Clark, Jeremy Coriell, Mike Daley, Matt Dau, Jeffrey Dean, Ben Gelb, Tara Vazir Ghaemmaghami, Rajendra Gottipati, William Gulland, Robert Hagmann, C. Richard Ho, Doug Hogberg, John Hu, Robert Hundt, Dan Hurt, Julian Ibarz, Aaron Jaffey, Alek Jaworski, Alexander Kaplan, Harshit Khaitan, Daniel Killebrew, Andy Koch, Naveen Kumar, Steve Lacy, James Laudon, James Law, Diemthu Le, Chris Leary, Zhuyuan Liu, Kyle Lucke, Alan Lundin, Gordon MacKean, Adriana Maggiore, Maire Mahony, Kieran Miller, Rahul Nagarajan, Ravi Narayanaswami, Ray Ni, Kathy Nix, Thomas Norrie, Mark Omernick, Narayana Penukonda, Andy Phelps, Jonathan Ross, Matt Ross, Amir Salek, Emad Samadiani, Chris Severn, Gregory Sizikov, Matthew Snelham, Jed Souter, Dan Steinberg, Andy Swing, Mercedes Tan, Gregory Thorson, Bo Tian, Horia Toma, Erick Tuttle, Vijay Vasudevan, Richard Walter, Walter Wang, Eric Wilcox, and Doe Hyun Yoon. 2017. In-datacenter Performance Analysis of A Tensor Processing Unit. In _Proceedings of the 44th annual international symposium on computer architecture_ . 1–12. 

- [18] Sheng-Chun Kao, Xiaoyu Huang, and Tushar Krishna. 2022. DNNFuser: Generative Pre-trained Transformer as a Generalized Mapper for Layer Fusion in DNN Accelerators. _arXiv preprint arXiv:2201.11218_ (2022). 

- [19] Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, and Zachary Tatlock. 2020. Dynamic Tensor Rematerialization. _arXiv preprint arXiv:2006.09616_ (2020). 

- [20] Hyoukjun Kwon, Prasanth Chatarasi, Michael Pellauer, Angshuman Parashar, Vivek Sarkar, and Tushar Krishna. 2019. Understanding Reuse, Performance, and Hardware Cost of DNN Dataflow: A Data-Centric Approach. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ . 754–768. 

- [21] Yao Liang and Guangyu Sun. 2023. TileFlow: A Fine-Grained Spatial Scheduler for DNN Training. https://github.com/pku-liang/TileFlow. Accessed: 2025-06-20. 

- [22] Heng Liao, Jiajin Tu, Jing Xia, Hu Liu, Xiping Zhou, Honghui Yuan, and Yuxing Hu. 2021. Ascend: A Scalable and Unified Architecture for Ubiquitous Deep Neural Network Computing: Industry Track Paper. In _2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 789–801. 

- [23] Sangkug Lym, Armand Behroozi, Wei Wen, Ge Li, Yongkee Kwon, and Mattan Erez. 2019. Mini-Batch Serialization: CNN Training with Inter-layer Data Reuse. _Proceedings of Machine Learning and Systems_ 1 (2019), 264–275. 

- [24] Stefano Markidis, Steven Wei Der Chien, Erwin Laure, Ivy Bo Peng, and Jeffrey S Vetter. 2018. Nvidia Tensor Core Programmability, Performance & Precision. In _2018 IEEE international parallel and distributed processing symposium workshops (IPDPSW)_ . IEEE, 522–531. 

_programming_ 10, 1 (1976), 147–175. 

   - [26] Linyan Mei, Koen Goetschalckx, Arne Symons, and Marian Verhelst. 2023. DeFiNES: Enabling Fast Exploration of the Depth-First Scheduling Space for DNN Accelerators Through Analytical Modeling. In _2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)_ . IEEE, 570–583. 

   - [27] Mike O’Connor, Niladrish Chatterjee, Donghyuk Lee, John Wilson, Aditya Agrawal, Stephen W Keckler, and William J Dally. 2017. Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems. In _Proceedings of the 50th Annual IEEE/ACM International Symposium on Microarchitecture_ . 41–54. 

   - [28] Brandon Reagen, José Miguel Hernández-Lobato, Robert Adolf, Michael Gelbart, Paul Whatmough, Gu-Yeon Wei, and David Brooks. 2017. A Case for Efficient Accelerator Design Space Exploration via Bayesian Optimization. In _2017 IEEE/ACM International Symposium on Low Power Electronics and Design (ISLPED)_ . IEEE, 1–6. 

   - [29] Paul Rosenfeld, Elliott Cooper-Balis, and Bruce Jacob. 2011. DRAMSim2: A cycle accurate memory system simulator. _IEEE computer architecture letters_ 10, 1 (2011), 16–19. 

   - [30] Daniel Sanchez and Christos Kozyrakis. 2013. ZSim: Fast and accurate microarchitectural simulation of thousand-core systems. _ACM SIGARCH Computer architecture news_ 41, 3 (2013), 475–486. 

   - [31] Yakun Sophia Shao, Jason Clemons, Rangharajan Venkatesan, Brian Zimmer, Matthew Fojtik, Nan Jiang, Ben Keller, Alicia Klinefelter, Nathaniel Pinckney, Priyanka Raina, Stephen G. Tell, Yanqing Zhang, William J. Dally, Joel Emer, C. Thomas Gray, Brucek Khailany, and Stephen W. Keckler. 2019. Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture. In _Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture_ . 14–27. 

   - [32] Jianfeng Song, Rongjian Liang, Yu Gong, Bo Yuan, and Jiang Hu. 2024. DiMOSparse: Differentiable Modeling and Optimization of Sparse CNN Dataflow and Hardware Architecture. In _2024 Design, Automation & Test in Europe Conference & Exhibition (DATE)_ . IEEE, 1–6. 

   - [33] Jianfeng Song, Rongjiang Liang, Bo Yuan, and Jiang Hu. 2024. DiMO-CNN: Deep Learning Toolkit-Accelerated Analytical Modeling and Optimization of CNN Hardware and Dataflow. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ (2024). 

   - [34] Linghao Song, Xuehai Qian, Hai Li, and Yiran Chen. 2017. PipeLayer: A Pipelined Reram-Based Accelerator for Deep Learning. In _2017 IEEE international symposium on high performance computer architecture (HPCA)_ . IEEE, 541–552. 

   - [35] Jasmina Vasiljevic, Ljubisa Bajic, Davor Capalija, Stanislav Sokorac, Dragoljub Ignjatovic, Lejla Bajic, Milos Trajkovic, Ivan Hamer, Ivan Matosevic, Aleksandar Cejkov, Utku Aydonat, Tony Zhou, Syed Zohaib Gilani, Armond Paiva, Joseph Chu, Djordje Maksimovic, Stephen Alexander Chin, Zahi Moudallal, Akhmed Rakhmati, Sean Nijjar, Almeet Bhullar, Boris Drazic, Charles Lee, James Sun, Kei-Ming Kwong, James Connolly, Miles Dooley, Hassan Farooq, Joy Yu Ting Chen, Matthew Walker, Keivan Dabiri, Kyle Mabee, Rakesh Shaji Lal, Namal Rajatheva, Renjith Retnamma, Shripad Karodi, Daniel Rosen, Emilio Munoz, Andrew Lewycky, Aleksandar Knezevic, Raymond Kim, Allan Rui, Alexander Drouillard, and David Thompson. 2021. Compute Substrate for Software 2.0. _IEEE micro_ 41, 2 (2021), 50–55. 

   - [36] Ofri Wechsler, Michael Behar, and Bharat Daga. 2019. Spring Hill (NNP-I 1000) Intel’s Data Center Inference Chip. In _2019 IEEE Hot Chips 31 Symposium (HCS)_ . IEEE Computer Society, 1–12. 

   - [37] Yannan Nellie Wu, Po-An Tsai, Angshuman Parashar, Vivienne Sze, and Joel S Emer. 2022. Sparseloop: An analytical Approach to Sparse Tensor Accelerator Modeling. In _2022 55th IEEE/ACM International Symposium on Microarchitecture (MICRO)_ . IEEE, 1377–1395. 

   - [38] Qingcheng Xiao, Size Zheng, Bingzhe Wu, Pengcheng Xu, Xuehai Qian, and Yun Liang. 2021. HASCO: Towards Agile Hardware and Software Co-design for Tensor Computation. In _2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture (ISCA)_ . IEEE, 1055–1068. 

   - [39] Jianhao Zhang, Shihan Ma, Peihong Liu, and Jinhui Yuan. 2024. Coop: Memory is not a Commodity. _Advances in Neural Information Processing Systems_ 36 (2024). 

   - [40] Size Zheng, Siyuan Chen, Siyuan Gao, Liancheng Jia, Guangyu Sun, Runsheng Wang, and Yun Liang. 2023. TileFlow: A Framework for Modeling Fusion Dataflow via Tree-Based Analysis. In _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture_ . 1271–1288. 

   - [41] Shixuan Zheng, Xianjue Zhang, Daoli Ou, Shibin Tang, Leibo Liu, Shaojun Wei, and Shouyi Yin. 2020. Efficient Scheduling of Irregular Network Structures on CNN Accelerators. _IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems_ 39, 11 (2020), 3408–3419. 

   - [42] Jinming Zhuang, Zhuoping Yang, Shixin Ji, Heng Huang, Alex K Jones, Jingtong Hu, Yiyu Shi, and Peipei Zhou. 2024. SSR: Spatial Sequential Hybrid Architecture for Latency Throughput Tradeoff in Transformer Acceleration. In _Proceedings of the 2024 ACM/SIGDA International Symposium on Field Programmable Gate Arrays_ . 55–66. 

- [25] Garth P McCormick. 1976. Computability of global solutions to factorable nonconvex programs: Part I—Convex underestimating problems. _Mathematical_ 

1263 

