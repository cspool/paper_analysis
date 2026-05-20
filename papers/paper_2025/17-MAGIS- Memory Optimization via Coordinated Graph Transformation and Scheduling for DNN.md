# MAGIS **: Memory Optimization via Coordinated Graph Transformation and Sc heduling for DNN** 

Renze Chen 

Renze Chen Zijian Ding[∗] crz@pku.edu.cn bradyd@cs.ucla.edu Peking University University of California, Los Angeles China United States Chengrui Zhang Jingwen Leng zhangchr@stu.pku.edu.cn leng-jw@cs.sjtu.edu.cn Peking University Shanghai Jiao Tong University China China 

Chengrui Zhang zhangchr@stu.pku.edu.cn Peking University China 

## Size Zheng 

zhengsz@pku.edu.cn Peking University China 

Xuanzhe Liu xzl@pku.edu.cn Peking University China 

Yun Liang[†] ericlyun@pku.edu.cn Peking University China 

## **Abstract** 

Recently, memory consumption of Deep Neural Network (DNN) rapidly increases, mainly due to long lifetimes and large shapes of tensors. Graph scheduling has emerged as an effective memory optimization technique, which determines the optimal execution, re-computation, swap-out, and swap-in timings for each operator/tensor. However, it often hurts performance significantly and can only manipulate tensors’ lifetimes but not shapes, limiting the optimization space. We find that graph transformation, which can change the tensor shapes and graph structure, creates a new tradeoff space between memory and performance. Nevertheless, graph transformation are applied separately so far, with primary focus on optimizing performance and not memory. 

In this paper, we propose MAGIS, a DNN memory optimization framework that coordinates graph transformation with graph scheduling. MAGIS uses a hierarchical tree to represent Fission Transformation (F-Trans), a type of transformation which can effectively reduce tensor shapes in a sub-graph. To keep the complexity low, we build a light-weight search space based on graph structure analysis. MAGIS decomposes graph scheduling into graph transformation and re-ordering and designs an incremental scheduling 

∗Work done while the author was a student at Peking University. †Corresponding author. 

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for thirdparty components of this work must be honored. For all other uses, contact the owner/author(s). _ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA_ © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0386-7/24/04. https://doi.org/10.1145/3620666.3651330 

algorithm to alleviate the scheduling overhead after each graph transformation step to efficiently coordinate them. Experimental results show that compared to state-of-theart works, MAGIS only uses 15%∼85% of their peak memory usage with the same latency[1] constraint and obtains a better Pareto boundary in dual-objective optimization of memory and performance. Our code is now available at https://github.com/pku-liang/MAGIS. 

## **ACM Reference Format:** 

Renze Chen, Zijian Ding, Size Zheng, Chengrui Zhang, Jingwen Leng, Xuanzhe Liu, and Yun Liang. 2024. MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN. In _29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS ’24), April 27-May 1, 2024, La Jolla, CA, USA._ ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3620666.3651330 

## **1 Introduction** 

As deep neural networks (DNNs) become more complex in terms of topology and size, the memory consumption of DNNs keeps growing, which poses great challenges for both training and inference. The memory consumption turns out to be more important when larger models come to stage [7, 12, 51]. The memory consumption increase can be attributed to two main factors. First, there are numerous tensors with **long lifetimes** , such as model parameters [7, 12, 15, 40, 51], activations during the training’s forward pass [5, 10, 38, 42], and intermediate tensors in complex networks [44, 73, 75]. Second, many tensors have **large shapes** , including large batch sizes for efficient training/inference, long sequence lengths in language models [7, 12, 51], and high resolutions in image-related models [21, 40, 45]. 

> 1In this paper, the terms "performance" and "latency" are interchangeably used, both referring to the time taken by a DNN to complete one inference/training epoch. 

607 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

Optimizing memory usage for DNNs becomes crucial for both server and mobile computing devices. GPUs, for instance, NVIDIA GeForce RTX 3090, provide only tens to dozens of GB of memory, while the large-batch training or inference sometimes requires several tens or even hundreds of GB of memory. Memory optimization is beneficial for executing large DNN, enabling co-location of multiple tasks in memory [32], and reducing cross-card communications in distributed learning. Similarly, mobile CPUs such as Qualcomm Snapdragon 888 provide only a few tens of GB of memory and many background applications may reside in memory, which greatly limits the space for DNN. Memory optimization is beneficial for deploying DNNs on mobile devices without consuming too much background memory. 

Graph scheduling is a class of widely used memory optimization techniques for DNNs, mainly including rematerialization [5, 10, 17, 18, 24, 27–29, 37, 38, 47], swapping [5, 20, 22, 30, 37–39, 41, 42, 57], and re-ordering [3, 22, 58, 72]. Its core idea is to manipulate the lifetimes of tensors by scheduling when each operator/tensor computes, evicts, recomputes, offloads, and reloads, thereby reducing the peak amount of tensors simultaneously residing in memory. However, because of the overhead introduced by re-computation or data transfer, it frequently leads to a notable reduction in performance. Moreover, although it operates the lifetimes of tensors, it does not affect the tensor shapes, which limits its potential optimization space. 

On the other hand, graph transformation is a class of optimization techniques based on equivalent transformations of graphs. Existing works have achieved good results in optimizing DNN performance [25, 26, 54, 56, 62]. They employ rule-based sub-graph substitution technique, which can be roughly divided into two types: Aggregation Transformation (A-Trans), like Figure 1 (a), which enhances hardware utilization to improve performance by aggregating small operators into larger ones at the cost of temporally increased memory usage; Interim Transformation (I-Trans), such as Figure 1 (b), which generally exploits algebraic equivalence to provide opportunities for other graph transformations. In addition, we find that the dual of A-Trans, which we call **Fission Transformation (F-Trans)** , like Figure 1 (c), can effectively reduce memory at the cost of lower hardware utilization by splitting some large operators into smaller ones and executing only one of the split parts at a time. 

However, graph transformation for memory optimization poses two main challenges. **(1) Complexity introduced by F-Trans.** On one hand, F-Trans leads to rapid growth in the size of the graph (as shown in Figure 1 (c), where the number of nodes almost doubles after transformation), which hinders subsequent optimization; on the other hand, F-Trans itself has a vast search space, as it can be applied to almost every sub-graph. **(2) Correlated graph transformation and graph scheduling.** Graph transformation involves a trade-off between memory and performance (e.g., 

**==> picture [242 x 98] intentionally omitted <==**

**----- Start of picture text -----**<br>
Add Add<br>Slice Matmul Slice<br>Conv2d Conv2d Conv2d Conv2d<br>Matmul Matmul Concat Concat Slice Slice Slice<br>Input Node X W1 W2 X W1 W2 X1 X2 X3 W X1 X2 X3 W<br>(a) Aggregation Trans.<br>Output Node Add Add Add Add Concat Add<br>Matmul W2 Matmul W2 Matmul<br>Add Add Matmul W1 Matmul W1 Matmul<br>A B C A B C X Slice X Slice<br>(b) Interim Trans. (c) Fission Trans.<br>**----- End of picture text -----**<br>


**Figure 1.** Examples of graph transformations. **(a)** and **(b)** are transformations borrowed from TASO [25], which are used to optimize performance. **(c)** is the dual of Aggregation Trans. and can effectively trade memory with performance. 

A-Trans trades memory for performance, and F-Trans does the opposite), but the final memory usage and performance are also traded by graph scheduling. This necessitates the need for efficient coordinated optimization between graph transformation and scheduling, which is challenging since both of them are complicated optimization. 

To tackle these challenges, we propose MAGIS, a DNN memory optimization framework through coordinated graph transformations and scheduling. To address the complexity problem of F-Trans, we propose Fission Hierarchy Tree (FTree) to express the graph structure after F-Trans, without actually transforming the graph into a complex structure. Although such design somehow limits the search space, it keeps the complexity low, making it easier for subsequent transformation and scheduling to search for better solutions. We then propose analytic methods to select proper sub-graphs and dimensions for F-Trans to construct a light-weight F-Tree, effectively reducing the search space of F-Trans. 

To address the second challenge, our goal is to alleviate the complexity of graph scheduling after each graph transformation step. We firstly decompose re-materialization and swapping into graph transformations and re-ordering, where re-materialization and swapping are two important scheduling techniques which can trade memory with performance, while re-ordering is a scheduling method that optimize memory without hurting performance. Such decomposition moves the memory & performance trade-off completely to the transformation phase, and the scheduling phase can only focus on memory optimization through re-ordering. It makes the scheduling after each graph transformation step much simpler, and fuses the memory & performance trade-offs into the unified search space of graph transformation. Then, we design an incremental graph scheduling algorithm that efficiently obtains a new schedule based on the previous schedule and the current transformation, further reducing scheduling time. 

Our contributions can be summarized as follows: 

- We design and implement MAGIS, a memory optimization framework based on coordinated graph transformation and graph scheduling. 

608 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

- We formalize graph fission transformation, represent it based on hierarchy tree, and use graph analysis to reduce its search space. 

- We propose transformations and algorithms that efficiently coordinate graph transformation and graph scheduling for memory optimization. 

We compare MAGIS with state-of-the-art graph schedulingbased memory optimization frameworks on various DNNs. Experimental results demonstrate that MAGIS can optimize original peak memory usage to 15%∼50% with no more than 10% latency overheads. Compared to state-of-the-art methods, MAGIS can optimize peak memory to only 15%∼85% of theirs with the same latency constraint, and can achieve a 1.25× speedup over them under the same memory constraint, obtaining a better Pareto boundary in dual-objective optimization of memory and latency. Our code is now available at https://github.com/pku-liang/MAGIS. 

## **2 Background & Motivation** 

**Table 1.** Notations 

|**Notation**|**Description/Defnition**|
|---|---|
|V(_𝐺_), E(_𝐺_)<br>operators, dependencies of_𝐺_<br>D(_𝐺_), T(_𝐺_)<br>dimension graph, dominator tree of_𝐺_<br>cost(_𝐺_),cost(_𝑣_)<br>execution latency of_𝐺_and_𝑣_∈V(_𝐺_)<br>size(_𝑣_) or |_𝑣_|<br>output tensor size of operator_𝑣_<br>_𝐺._pre(_𝑣_)_,𝐺._suc(_𝑣_)<br>predecessors, successors of_𝑣_∈V(_𝐺_)<br>_𝐺._anc(_𝑣_)_,𝐺._des(_𝑣_)<br>ancestors, descendants of_𝑣_∈V(_𝐺_)<br>inps(_𝐺_),outs(_𝐺_)<br>inputs, outputs of_𝐺_<br>_𝐺._sub(_𝑆_) or_𝐺_[_𝑆_]<br>sub-graph of_𝐺_induced from_𝑆_⊆V(_𝐺_)<br>_𝐺._inps(_𝑆_)<br>nodes consumed by_𝑆_⊆V(_𝐺_) from outside<br>_𝐺._outs(_𝑆_)<br>nodes produced by_𝑆_⊆V(_𝐺_) for outside||



## **2.1 Computation Graph** 

**Graph Structure.** DNN during training or inference process is often represented as "computation graph" _𝐺_ (abbreviated as "graph"). _𝑉_ = V( _𝐺_ ) is the set of operators, each of which has several input tensors and one output tensor, and _𝐸_ = E( _𝐺_ ) ⊆ _𝑉_ × _𝑉_ is the set of data dependencies between operators. ( _𝑣_ 1 _, 𝑣_ 2) ∈ _𝐸_ means that the output tensor of _𝑣_ 1 is one of the input tensors of _𝑣_ 2. Related notations used in this paper are shown in Table 1. In cases where there is no ambiguity, we use xxx( _𝑣_ ) as an abbreviation for _𝐺._ xxx( _𝑣_ ). Some notations can be derived from other notations, for example, _𝐺._ inps( _𝑆_ ) = ([�] _𝑣_ ∈ _𝑆[𝐺.]_[pre][(] _[𝑣]_[))\] _[𝑆]_[, and] _[𝐺.]_[outs][(] _[𝑆]_[)][=] (outs( _𝐺_ ) ∪[�] _𝑣_ ∈V( _𝐺_ )\ _𝑆[𝐺.]_[pre][(] _[𝑣]_[)) ∩] _[𝑆]_[. A node] _[ 𝑢]_[dominates] node _𝑣_ if every path from the entry node to _𝑣_ must go through _𝑢_ ; and then _𝑢_ is _𝑣_ ’s dominator. The intermediate dominator of a node _𝑣_ is the dominator of _𝑣_ that is dominated by all the dominators of _𝑣_ except _𝑣_ itself. The dominator tree [4] is the tree where each node’s parent is its intermediate dominator in the graph. A computation graph usually has many input nodes (e.g., input tensor, label tensor, and weight tensors), so 

the dominator tree we use here usually takes the input tensor as the entry. Note that for _𝑇_ = T ( _𝐺_ ), _𝑇_ itself is also a graph, and the operations in Table 1 are also applicable to it. For example, the set of child nodes of a node _𝑣_ in _𝑇_ is _𝑇._ suc( _𝑣_ ). The nodes of _𝑇_ also belong to _𝐺_ , i.e., V(T ( _𝐺_ )) ⊆V( _𝐺_ ). 

**Execution Latency.** In single machine situation (e.g., single-card GPU), the operators in the graph are generally executed in order, and the order _𝑠_ = ( _𝑣_ 1 _, 𝑣_ 2 _, ..., 𝑣𝑛_ ) must satisfy the data dependencies between operators. The graph execution latency can be estimated as the sum of the latency of the operators: cost( _𝐺_ ) ≈[�] _𝑣_ ∈V( _𝐺_ )[cost][(] _[𝑣]_[)][.] 

**Memory Usage.** Given a topo-order _𝑠_ = ( _𝑣_ 1 _, 𝑣_ 2 _, ..., 𝑣𝑛_ ), assuming that _𝑖_ is the timestamp when the _𝑖[𝑡ℎ]_ operator is finished, we can calculate the lifetime of the output tensor of each operator _𝑣𝑖_ : the start timestamp is _𝑆𝑖_ = _𝑖_ − 1, and the free timestamp is _𝐹𝑖_ = max _𝑣𝑗_ ∈ _𝑠𝑢𝑐_ ( _𝑣𝑖_ ) _𝑗_ . Based on the lifetime of each tensor, the set of tensors that are active during the execution of _𝑣𝑖_ is _𝐴𝑖_ = { _𝑣 𝑗_ | _𝑆 𝑗_ ≤ _𝑖_ ≤ _𝐹 𝑗_ }. Then the **active memory usage** during _𝑣𝑖_ ’s execution is _𝑀𝑖_ =[�] _𝑢_ ∈ _𝐴𝑖_[|] _[𝑢]_[|][, and] the **peak memory usage** during the execution of graph _𝐺_ is: _𝑀𝑝𝑒𝑎𝑘_ = max _𝑖 𝑀𝑖_ . We define **memory hot-spots** as the set of tensors that contribute to the peak memory usage, that is, the tensors that are active when peak memory usage is reached: _𝐻_ =[�] { _𝐴𝑖_ | _𝑖_ ∈{1 _,_ 2 _, ...,𝑛_ } ∧ _𝑀𝑖_ = _𝑀𝑝𝑒𝑎𝑘_ }. 

## **2.2 Graph Scheduling and Transformation** 

Graph scheduling is a class of widely used DNN memory optimization techniques, which manipulates the lifetimes of tensors to schedule when to execute (re-ordering [3, 58]), evict & re-compute (re-materialization [5, 10, 18, 24, 27, 37, 38]), and offload & reload (swapping [5, 20, 22, 37, 38, 41, 42, 57]) each operator/tensor without influencing tensor shapes. 

Graph transformation is a class of techniques to optimize computation graphs by mutating their structures while preserving semantics. Existing works [25, 26, 56, 62] mainly optimize latency via rule-based sub-graph substitution, which can be categorized into two types: Aggregation Transformation (A-Trans), aggregating small operators into larger ones to trade memory for latency; Interim Transformation (I-Trans), mostly based on algebraic equivalence to provide opportunities for other transformations. 

## **2.3 Motivation** 

We find that appropriate graph transformations can also improve the memory usage of graphs. For example, as shown in Figure 2 (c), splitting operators reduces peak memory usage at the cost of more operator calls and decreased hardware utilization. With the help of graph transformation, memory optimization of DNNs can be greatly enhanced. For example, in Figure 2 (a), there’s a simplified graph structure commonly observed in DNN training or some DNNs with long skipconnections [23, 44, 73, 75]. It has a peak memory usage of 1056 since 33 tensors with size 32 are alive when computing the 33-th operator, which exceeds the memory limit of 

609 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

**==> picture [504 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) 32 32 32…… × 32 1056 peak memory usage 32 32 32 32…… × 32 32 (d) Slicing 4 × 32 × 56 peak memory usage44 free memory for async swap 32 × Concat<br>32 8 8 …… 8 8 8 8 …… 8 8 32<br>32 × 96 peak memory usage4 free memory for async swap 32 ×<br>(b) 32 32 …… 32 32 32 32 …… 32 32 Extern Storage<br>Extern Storage (e) 4 × 2 × 32 × 52 peak memory usage48 free memory for async swap 32 ×<br>Slicing Concat<br>(c) 16 Slicing × 32 × 98 peak memory usage 32 × Concat 32 8 4 …… 4 4 4 4 …… 4 8 32<br>32 2 2 …… 2 2 2 2 …… 2 2 32<br>Extern Storage<br>**----- End of picture text -----**<br>


**Figure 2.** Motivation examples with memory limit of 100. **(a)** Without any optimization. **(b)** Using swapping. **(c)** Using fission transformation. **(d)(e)** Using fission transformation and swapping. 

100. In Figure 2 (b), although graph scheduling alone can restrict memory usage to 100 by swapping temporally unused tensors into external storage , it causes long latency due to data transfer. However, incorporating graph transformations, as shown in Figure 2 (d), more memory is saved and asynchronous swapping can be utilized to hide data transfer latency. Although the hardware utilization decreases, the latency penalty can be compensated by the efficiency gain provided by asynchronous swapping in this case. 

We name the transformation used in Figure 1 (c) and Figure 2 (c) (d) (e) as **Fission Transformation** (F-Trans), which is the dual of A-Trans and can effectively optimize the memory usage by splitting operators. However, the existing graph transformation techniques based on rule-based sub-graph substitution [25, 26, 56, 62] can not be used for F-Trans. First, F-Trans often greatly increases the graph complexity, hindering subsequent optimization. Second, F-Trans involves a vast search space, since it can be applied to almost any sub-graph. For example, Figure 2 (e) uses two different F-Trans, and even for such a simple network in this example, the search space for feasible F-Trans is huge. Finding efficient ways to represent and search for F-Trans is a challenge. 

In addition, coordinating graph transformations with graph scheduling is critical for optimizing memory usage with graph transformations. Figure 2 (c) shows that applying graph transformations alone can optimize memory usage, but excessively fine-grained operator splitting may result in high performance costs. Instead, combining graph transformation and graph scheduling as in Figure 2 (d) can significantly reduce memory usage and achieve shorter latency by jointly balancing the memory and performance trade-offs of both transformation and scheduling. 

## **3 Design Overview** 

Figure 3 shows the overall design of MAGIS. It accepts a DNN graph and outputs the optimized graph and schedule. MAGIS has four main components: M-State, M-Analyzer, M-Rules, and M-Optimizer. 

**==> picture [242 x 156] intentionally omitted <==**

**----- Start of picture text -----**<br>
Computation Graph Fission Hierarchy Tree  M-State<br>A B C 4 (F-Tree) Best Schedule<br>0 1 2 5 7 9 13 A n=1 (0, 1, 2, 4, 5, 7, … , 14)<br>D 10 12 14 B n=1 D n=4 Simulation/Profile Result<br>3 6 8 11 C n=2 Peak Mem: … Latency: …<br>Input Graph Simulator & Profiler M-Rules ( § 5)<br>Initial Scheduling<br>TASO Transform Rules<br>M-Analyzer ( § 4) M-Optimizer ( § 6) (A-Trans & I-Trans)<br>D-Graph Analysis M-State Apply Transform Rules Fission Hierarchy Tree<br>Mutation Rules<br>F-Tree Construction Incremental Scheduling Scheduling-based Rules<br>(Re-mat. & Swapping)<br>If F-Tree needs update Best M-State<br>**----- End of picture text -----**<br>


**Figure 3.** Overview of MAGIS. 

M-State represents the optimization status, including computation graph, fission hierarchy tree (F-Tree), best schedule, and simulation & profile result. F-Tree represents the hierarchical search space of fission transformation (F-Trans), where a node with _𝑛_ = 1 represents a potential sub-graph & dimension candidate for F-Trans, and a node with _𝑛 >_ 1 represents a sub-graph already been split via F-Trans along some dimension into _𝑛_ parts. M-Analyzer generates the search space of fission transformation (F-Trans), by constructing the fission hierarchy tree (F-Tree) according to the computation graph. M-Optimizer coordinates the graph transformations (including F-Trans) and scheduling to optimize the latency & memory. M-Rules provide the transformations for M-Optimizer, including "TASO rules" used in previous works [25, 26, 56, 62], F-Tree mutation rules for manipulating F-Tree to reflect F-Trans applications on the graph (§5.1) , and scheduling-based rules decomposed from graph scheduling. Note that F-Trans is decoupled as F-Tree and mutation rules applied on the F-Tree. These rules are integrated with others (e.g., TASO rules, scheduling-based rules) in M-Rules, forming a unified optimization space explored by the M-Optimizer. 

610 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

**==> picture [505 x 276] intentionally omitted <==**

**----- Start of picture text -----**<br>
X 𝑣0 W1 𝑣1 W2 𝑣2 W3 𝑣3 Node Shape ⟨𝑣0, 1⟩ ⟨𝑣1, 1⟩ ⟨𝑣2, 1⟩ ⟨𝑣3, 1⟩ ⟨𝑣0, 2⟩<br>MM 𝑣4 MM 𝑣5 MM 𝑣6 𝑣1, 𝑣𝑣20, 𝑣3 [[HN,, h T,, C C]] ⟨𝑣4, 1⟩ ⟨𝑣5, 1⟩ ⟨𝑣6, 1⟩ ⟨𝑣4, 2⟩ ⟨𝑣5, 2⟩ ⟨𝑣6, 2⟩ ⟨𝑣4, 3⟩ ⟨𝑣5, 3⟩ ⟨𝑣6, 3⟩<br>⟨𝑣7, 1⟩ ⟨𝑣7, 2⟩ ⟨𝑣7, 3⟩ ⟨𝑣7, 4⟩<br>BMM 𝑣7 𝑣4, 𝑣𝑣57, 𝑣6 [[NN,, H H,, T T,, h T]] ⟨𝑣8, 1⟩ ⟨𝑣8, 2⟩ ⟨𝑣8, 3⟩ ⟨𝑣8, 4⟩<br>SM 𝑣8 𝑣8 [N, H, T, T] ⟨𝑣9, 1⟩ ⟨𝑣10, 2⟩ ⟨𝑣9, 2⟩ ⟨𝑣9, 3⟩ ⟨𝑣9, −1⟩<br>MM: Matmul W4 𝑣10 BMM 𝑣9 𝑣9 [N, H, T, h] ⟨𝑣11, 1⟩ ⟨𝑣11, −1⟩ ⟨𝑣11, 2⟩<br>BMM: Batch MM 𝑣10 [C, H, h] ⟨𝑣12, 1⟩ ⟨𝑣12, 2⟩<br>SM: Softmax MM+ 𝑣𝑣1112 𝑣𝑣1112 [[NN,, T T,, C C]] Batch dimension Head dimension Sequence dimension<br>(a) Graph  𝐺 of self-attention (b) Shapes of each node in  𝐺 (c) Some sub-graphs of  𝐺 ’s D-Graph<br>Figure 4.  Example of D-Graph.  𝑁,𝑇,𝐶, 𝐻,ℎ represents batch-size, seq-len, hidden-dim, num-heads, head-dim respectively.<br>Input of  𝑆 Output of  𝑆 S Slice C Concat + Add 4 M-Analyzer<br>... 0 𝐷 ⟨𝑣0, 1⟩ S ... 0 S (D-Graph)InIn this section,andand usewewe itwilltowilltoto firstdefineintroduceF-Trans.defineintroduceF-Trans.introduceF-Trans.F-Trans. DimensionThenThen we proposeGraphGraph<br>1 𝑆 3 ⟨𝑣3, 1⟩ 3 1 3’ F-Tree as an abstraction of the optimization space/state of F-<br>⟨𝑣2, 1⟩ 4 S 2 S 4’ Trans, and provide an algorithm to construct a light-weight<br>2 4 ⟨𝑣4, 1⟩ F-Tree considering F-Trans only on some sub-graphs that<br>6 5 5’ 6’<br>⟨𝑣6, 1⟩ are selected based on dominator tree and memory hot-spots.<br>5 6 ⟨𝑣5, 1⟩ 8 7 7’ 8’<br>7 8 C C 4.1 Dimension Graph<br>+<br>⟨𝑣7, 1⟩ ⟨𝑣8, −1⟩<br>Intuitively, an F-Trans splits a sub-graph along a "dimension"<br>(a) ... ... ... (b) (c) ... ... ...<br>**----- End of picture text -----**<br>


(D-Graph)InIn this section,andand usewewe itwilltowilltoto firstdefineintroduceF-Trans.defineintroduceF-Trans.introduceF-Trans.F-Trans. DimensionThenThen we proposeGraphGraph F-Tree as an abstraction of the optimization space/state of F- Trans, and provide an algorithm to construct a light-weight F-Tree considering F-Trans only on some sub-graphs that are selected based on dominator tree and memory hot-spots. 

Intuitively, an F-Trans splits a sub-graph along a "dimension" running through it. Therefore, we propose Dimension Graph (D-Graph) to identify the graph-level dimensions. 

**Figure 5.** F-Trans _𝑓_ = ( _𝑆, 𝐷,𝑛_ ) ( _𝑛_ = 2) in graph _𝐺_ , which is simplified from the training-graph of an MLP. **(a)** Sub-graph _𝑆_ = { _𝑣_ 3 _, 𝑣_ 4 _, 𝑣_ 5 _, 𝑣_ 6 _, 𝑣_ 7 _, 𝑣_ 8}. **(b)** D-Graph _𝐷_ , which represents the batch-dim of _𝑆_ ’s activation. **(c)** Result graph after F-Trans. 

Given a graph _𝐺_ where _𝑣_ ∈V( _𝐺_ ) has _𝑠𝑣_ dimensions in its output tensor and _𝑟𝑣_ reduce-axes in its computation, we define D-Graph _𝐷_ = D( _𝐺_ ) where for each _𝑣_ ∈V( _𝐺_ ) and _𝑖_ = − _𝑟𝑣, ...,_ −2 _,_ −1, 1 _,_ 2 _, ...,𝑠𝑣_ , there’s ⟨ _𝑣,𝑖_ ⟩∈V( _𝐷_ ). For each ( _𝑢, 𝑣_ ) ∈E( _𝐺_ ), if the _𝑖[𝑡ℎ]_ dimension of _𝑢_ and _𝑗[𝑡ℎ]_ dimension of _𝑣_ correspond to the same spatial-axis[2] , then there’s (⟨ _𝑢,𝑖_ ⟩ _,_ ⟨ _𝑣, 𝑗_ ⟩) ∈E( _𝐷_ ); and if the _𝑖[𝑡ℎ]_ dimension of _𝑢_ corresponds to the _𝑗[𝑡ℎ]_ reduce-axis of _𝑣_ ’s computation, then there’s (⟨ _𝑢,𝑖_ ⟩ _,_ ⟨ _𝑣,_ − _𝑗_ ⟩) ∈E( _𝐷_ ). For instance, a MatMul operator _𝑐_ (expressed as _𝑐_ [ _𝑚,𝑛_ ] =[�] _𝑘[𝑎]_[[] _[𝑚,𝑘]_[]][×] _[ 𝑏]_[[] _[𝑘,𝑛]_[]][, where] _𝑚,𝑛_ are _𝑐_ ’s dimensions, and _𝑘_ is the reduce-axis) with its inputs _𝑎,𝑏_ ∈ pre( _𝑐_ ) provides connections (⟨ _𝑎,_ 1⟩ _,_ ⟨ _𝑐,_ 1⟩), (⟨ _𝑎,_ 2⟩ _,_ ⟨ _𝑐,_ −1⟩), (⟨ _𝑏,_ 1⟩ _,_ ⟨ _𝑐,_ −1⟩), (⟨ _𝑏,_ 2⟩ _,_ ⟨ _𝑐,_ 2⟩) ∈E( _𝐷_ ). 

MAGIS takes a computation graph as input. The graph and its initial schedule are analyzed by M-Analyzer, which constructs the F-Tree, outputs the initial M-State and sends the M-State to the M-Optimizer. M-Optimizer applies M- Rules to produce new M-States by mutating some sub-graphs or sub-F-trees. Note that the rules will not choose the subgraph spanning the boundary of the sub-graphs affected by F-Trans (the sub-graph belonging to the F-Tree node with _𝑛 >_ 1) for transformation. This is because for a region _𝑅_ already affected by F-Trans, the rules will not transform the sub-graph _𝑆_ that partly intersects with _𝑅_ , as some nodes of _𝑆_ will be split during execution while some not. It then performs fast incremental scheduling on these new graphs, utilizing the mutated graph region of the transformation and prior schedules, to quickly derive near-optimal schedules and associated profile results. Effective M-States are iteratively fed back to M-Optimizer. Besides, if a graph transformation is applied on a sub-graph that has not been affected by F-Trans, M-optimizer will query M-Analyzer to update the F-Tree in the new M-States. 

**Example.** Figure 4 (a) illustrates graph _𝐺_ , extracted from a transformer block [55], with shapes detailed in part (b). Part (c) depicts some sub-graphs of D( _𝐺_ ), like one with batchdimensions from tensors excluding _𝑣_ 1 _, 𝑣_ 2 _, 𝑣_ 3 _, 𝑣_ 10, one with head-dimensions from tensors excluding _𝑣_ 0 _, 𝑣_ 12, and another with sequence-dimensions from tensors except _𝑣_ 1 _, 𝑣_ 2 _, 𝑣_ 3 _, 𝑣_ 10. 

## **4.2 Fission Transformation** 

With the help of D-Graph, we can define an F-Trans of graph _𝐺_ as _𝑓_ = ( _𝑆, 𝐷,𝑛_ ), where _𝑆_ ⊆V( _𝐺_ ), _𝐷_ is the D-Graph to split sub-graph _𝐺_ [ _𝑆_ ] along, _𝑛_ is the fission number. It has the following constraints: **(1)** _𝐺_ [ _𝑆_ ] is weakly connected. **(2)** _𝐺_ [ _𝑆_ ] is convex: _𝐺._ inps( _𝑆_ ) ∩[�] _𝑣_ ∈ _𝐺._ outs( _𝑆_ ) _[𝐺.]_[des][(] _[𝑣]_[)][=][∅][.] **[ (3)]**[ The] graph after fission has no redundant computation, requiring 

The remainder of this paper is structured as follows: §4 introduces M-Analyzer of MAGIS, §5 discusses M-Rules, and §6 details M-Optimizer. 

> 2Here we do not consider spatial-axis with sliding-window, such as the height axis of a 3 × 3 convolution; we will improve it in future work. 

611 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

## **Algorithm 1:** M-Analyzer: F-Tree Construction 

||**Algorithm 1:**M-Analyzer: F-Tree Construction|**Algorithm 1:**M-Analyzer: F-Tree Construction|
|---|---|---|
||**input :**graph:_𝐺_; max-level:_𝐿_||
||**output:**fssion hierarchy tree:_𝐹_||
|**1**|_𝐹_:= ∅;||
|**2**|_𝐻_:=MemoryHotspots(_G_);||
|**3 **|**for**_𝐷_∈_connected components of_ D(_𝐺_) **do**||
|**4**|_𝐺_′ :=subgraph of_𝐺_induced from_𝐷_;||
|**5**|_𝑇_:= T(_𝐺_′);||
|**6**|_𝑠_:=GetScores(_𝐺_′_,𝑇, 𝐻_);||
|**7**|_𝑠𝑚𝑎𝑥_=max_𝑣_∈V(_𝐺_′)_𝑠_[_𝑣_];||
|**8**|**if**_𝑠𝑚𝑎𝑥_≤0**then continue**;||
|**9**|**for**_𝑖_∈{1_,_2_, ..., 𝐿_} **do**||
|**10**|_𝑉_:= {_𝑣_∈V(_𝐺_′) | _𝑖_/_𝐿_≤_𝑠_[_𝑣_]/_𝑠𝑚𝑎𝑥<_ (_𝑖_+1)/_𝐿_};||
|**11**|**for**_𝑣𝑑𝑜𝑚_∈{_𝑣_∈_𝑉_|_𝑇._des(_𝑣_) ∩_𝑉_= ∅} **do**||
|**12**||_𝑆_:=_𝑇._des(_𝑣𝑑𝑜𝑚_) \ {_𝑣𝑑𝑜𝑚_};|
|**13**||_𝐷_′ :=subgraph of_𝐷_induced from_𝑆_;|
|**14**||_𝑓_:= (_𝑆, 𝐷_′_,_1);|
|**15**||**if** _𝑓_is valid **then** _𝐹_:=_𝐹_∪{_𝑓_};|
||||
||||
|**16 **|**return**_𝐹_;||



that ∀ _𝑣_ ∈ _𝑆_ , there’s exact one _𝑖_ ∈ Z s.t. ⟨ _𝑣,𝑖_ ⟩∈V( _𝐷_ ), and ∀( _𝑢, 𝑣_ ) ∈E( _𝐺_ [ _𝑆_ ]), ∃ _𝑖, 𝑗_ ∈ Z s.t. (⟨ _𝑢,𝑖_ ⟩ _,_ ⟨ _𝑣, 𝑗_ ⟩) ∈E( _𝐷_ ). 

Given an F-Trans _𝑓_ = ( _𝑆, 𝐷,𝑛_ ) of _𝐺_ , the result graph after F-Trans is a graph with _𝑛_ split parts of _𝐺_ [ _𝑆_ ]. ∀ _𝑢_ ∈ _𝐺._ inps( _𝑆_ ), if ∃ _𝑖 >_ 0 s.t. ⟨ _𝑢,𝑖_ ⟩∈V( _𝐷_ ), then _𝑢_ will be sliced for each split part, otherwise shared by them. ∀ _𝑣_ ∈ _𝐺._ outs( _𝑆_ ), if ∃ _𝑖 >_ 0 s.t. ⟨ _𝑣,𝑖_ ⟩∈V( _𝐷_ ), then _𝑣_ will be computed by merging the related outputs of split parts, otherwise reducing them. Note that, the split parts are executed sequentially to save memory by timely freeing intermediate tensors of each part at the cost of lower hardware utilization (e.g., parallelism, locality) due to smaller operator shapes. 

**Example.** Figure 5 demonstrates an example of F-Trans _𝑓_ = ( _𝑆, 𝐷,𝑛_ ) with _𝑛_ = 2. _𝑣_ 1 is a weight tensor, so there’s no ⟨ _𝑣_ 1 _,𝑖_ ⟩∈V( _𝐷_ ); so in the result graph _𝑣_ 1 is shared by each split part. Other inputs, _𝑣_ 0 and _𝑣_ 2, are sliced for each part. _𝑣_ 8 is the gradient of _𝑣_ 1, computed by adding along batch-dim, so ⟨ _𝑣_ 8 _,_ −1⟩∈V( _𝐷_ ); so in the result graph _𝑣_ 8 is computed by adding the outputs of each split part. Other outputs, _𝑣_ 6 and _𝑣_ 7, are computed by concatenating the outputs of each part. 

## **4.3 Fission Hierarchy Tree** 

Directly applying F-Trans to a graph will significantly increase the complexity, especially when the fission number is large. Since each F-Trans divides a graph into several isomorphic sub-graphs, we can save only one of them. Instead of transforming the original graph directly, we construct a fission hierarchy tree (F-Tree). Each tree-node in the F- Tree records a F-Trans _𝑓_ = ( _𝑆, 𝐷,𝑛_ ). For any tree-node _𝑓_ = ( _𝑆, 𝐷,𝑛_ ) and its parent _𝑓_[′] = ( _𝑆_[′] _, 𝐷_[′] _,𝑛_[′] ), we have _𝑆_ ⊆ _𝑆_[′] . Figure 3 displays an example of F-Tree, where each node represents a sub-graph surrounded by a dashed box in the 

left-side graph and the _𝑛_ next to the node is the fission number. When _𝑛_ = 1, it indicates that the node is an fission candidate, and when _𝑛 >_ 1, it indicates that the subgraph of the node has been split into _𝑛_ parts by F-Trans. Such abstraction significantly reduces the complexity of subsequent graph transformation and scheduling. 

However, the search space for F-Trans on graph _𝐺_ is still large, reaching up to _𝑂_ (2[|V(] _[𝐺]_[)|][2] ) since almost any convex sub-graph can be a fission candidate. Indeed, arbitrarily applying F-Trans does not guarantee peak memory reduction. Effective memory saving can be achieved only when F-Trans targets sub-graphs containing memory hot-spots (§2.1). 

**Analysis.** For an F-Trans _𝑓_ = ( _𝑆, 𝐷,𝑛_ ) of graph _𝐺_ , with memory hot-spots as _𝐻_ and _𝐼_ = _𝐺._ inps( _𝑆_ ). _𝑀_ 0 and _𝑀𝑓_ represent the peak memory usages before and after F-Trans, shown in Equation (1). Since inputs _𝐼_ reside in memory when executing split sub-graphs, _𝑀𝑓_ should combine their sizes � _𝑣_ ∈ _𝐼_[|] _[𝑣]_[|][ with][ �] _𝑣_ ∈ _𝐻_ \ _𝑆_[|] _[𝑣]_[|][ (sizes of memory hot-pots beyond] _𝑆_ ) into[�] _𝑣_ ∈( _𝐻_ \ _𝑆_ )∪ _𝐼_[|] _[𝑣]_[|][. The peak memory reduction after F-] Trans, i.e., _𝑀_ 0 − _𝑀𝑓_ , is shown in Equation (2). 

**==> picture [233 x 32] intentionally omitted <==**

**Metric.** We can observe that to make _𝑀_ 0 − _𝑀𝑓_ larger, we need to ensure that _𝑆_ includes more memory hot-spots, while _𝐼_ consumes less memory. To minimize input memory usage of F-Trans, we select a node and consider the subgraph dominated by it as the fission candidate, ensuring the sub-graph has only one entry node[3] . We define a metric called "memory heat", representing the total size of hot-spots in a sub-graph dominated by a node. Given the graph _𝐺_ with dominator tree _𝑇_ = T ( _𝐺_ ) and memory hot-spots _𝐻_ , we calculate _𝑣_ ’s memory heat with Equation (3), where _𝐻_ ∩ _𝑇._ des( _𝑣_ ) are the memory hot-spots dominated by _𝑣_ . We then assign a score for each node _𝑣_ as shown in Equation (4), estimating the potential peak memory reduction after F- Trans on the sub-graph dominated by _𝑣_ , where the first term is the reduction of the sizes of memory hot-spots, and the second term is the sizes of input nodes which should reside in memory during the execution of each split part after F- Trans. We typically set _𝑛_ = 2 to ensure that just splitting the sub-graph into two parts also yields benefits. 

**==> picture [235 x 28] intentionally omitted <==**

**Algorithm.** Based on the metrics discussed above, we propose Algorithm 1 to construct an F-Tree. The main idea is identifying nodes with scores (Equation (4)) distributed in different intervals, since a higher score indicates more peak memory reduction of F-Trans, but may also imply larger latency overhead. The hyper-parameter _𝐿_ controls the number of intervals and the F-Tree’s max-level. This algorithm 

> 3Strictly, weight tensors may also be input nodes, as discussed in §2.1 

612 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

**==> picture [504 x 81] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Memory Hot-spot (b)  0 (c)  (d)  �0 0 (e)<br>1 21 node 0 1 2 3 1 21 (�0, �, 1)<br>2 19 heat 12 11 10 5 �2 2 19<br>score 6 5.5 5 2.5 (�2, �, 1)<br>0 1 2 3 4 65 7 8 9 11 1015 17 184 3 8 20 4 3 10 3 11 3 12 2 … … �1110 1015 �174 184 3 �83 20 (�3, �, 1)<br>17 18 19 21 12 16 5 6 7 9 12 16 5 6 7 9 (�10, �, 1)<br>1.5 1.5 0.5 1 … (�4, �, 1)<br>10 11 12 13 14 15 16 20 13 14 13 14<br>**----- End of picture text -----**<br>


**Figure 6.** Example of F-Tree construction based on Algorithm 1 (with _𝐿_ = 5). Each tensor has a size of 1. **(a)** _𝐺_[′] in Algorithm 1 line 4. **(b)** Dom _𝑇_ = T ( _𝐺_[′] ). **(c)** Scores calculated based on Equation (3) (4), where nodes in orange boxes are selected dominators ( _𝑣𝑑𝑜𝑚_ in Algorithm 1 line 11). **(d)** Selected sub-graphs ( _𝑆_ in Algorithm 1 line 12). **(e)** Constructed F-Tree. 

inputs graph _𝐺_ and max-level _𝐿_ , iterating over connected components _𝐷_ of D( _𝐺_ ) (line 3), extracting sub-graph _𝐺_[′] and its dominator tree _𝑇_ (lines 4-5), then calculating scores based on Equation (3) (4) (line 6). Upon obtaining the maximum score _𝑠𝑚𝑎𝑥_ (line 7), it segments [0 _,_ 1] into _𝐿_ intervals, selecting nodes in different intervals based on normalized scores _𝑠_ [ _𝑣_ ]/ _𝑠𝑚𝑎𝑥_ (lines 10-11), and generating fission candidates from sub-graphs dominated these nodes (lines 12-15). The F-Tree is constructed from these sub-graphs. 

**Example.** Figure 6 gives an example of F-Tree construction for a computation graph simplified from the training graphs of various models. For demonstration, we only show one connected component _𝐷_ ∈D( _𝐺_ ) here. Part (a) is the _𝐺_[′] in Algorithm 1 at line 4. Part (b) shows dominator tree _𝑇_ = T ( _𝐺_[′] ). Part (c) shows the calculated results of heat and score based on Equation (3) (4). Here _𝐿_ = 5, so there are 5 normalized score intervals [0.2, 0.4), [0.4,0.6), [0.6,0.8), [0.8,1), [1,1], and the nodes in dashed boxes are selected. Part (d) shows the selected sub-graph nodes as fission-candidates. Part (e) shows the finally constructed F-Tree. 

## **5 M-Rules** 

M-Rules in MAGIS borrow the rules of Aggregation Transformation (A-Trans) and Interim Transformation (I-Trans) from previous works like TASO [25], shown by Figure 1 (a) (b). We call these TASO Rules, which can be used to optimize latency. Beside of these, in this section, we will introduce F-Tree Mutation Rules and Scheduling-based Rules to further optimize memory and latency. 

## **5.1 Fission Hierarchy Tree Mutation Rules** 

All tree-nodes _𝑓_ = ( _𝑆, 𝐷,𝑛_ ) of the initial F-Tree constructed by Algorithm 1 have _𝑛_ = 1. We refer them as disabled nodes, whose sub-graphs have not performed F-Trans. Node with _𝑛 >_ 1 is called enabled node, which means its sub-graph has already performed F-Trans and is split into _𝑛_ parts. The F-Tree Mutation Rules mainly change the _𝑛_ of the F-Tree node to apply F-Trans to the graph. They include: 

- **Enabling Rule.** It enables a disabled leaf node of F- Tree or a parent node of an enabled node without enabled ancestors, as shown in Figure 7 (a). 

- **Lifting Rule.** It disables an enabled node without enabled ancestors and enables its parent node, as shown in Figure 7 (b). 

- **Disabling Rule.** It disables an enabled node that has no enabled descendant node, as shown in Figure 7 (c). 

- **Mutating Rule.** It increases an enabled node’s fission number _𝑛_ to the next number that can divide the dimension length, as shown in Figure 7 (d). 

With the help of M-Analyzer and above rules, we decouple F-Trans into F-Tree construction before optimization phase and F-Tree mutation during optimization phase. It can be observed that we actually start enabling leaf nodes first and gradually move towards nodes closer to the root. Since applying fission on the nodes closer to the root has a greater impact on memory and latency, we start from the leaves for smaller mutation steps and smoother search. 

## **5.2 Scheduling-based Rules** 

We introduce two additional operators, Store and Load, to represent swapping behaviour in graph scheduling. Based on this, we add four rules as follows: 

- **Re-materialization Rule.** It separates one user B from an operator A with multiple users and lets it use a recalculated operator A’, as shown in Figure 8 (a) (b). 

- **De-re-materialization Rule.** It is the dual of the rematerialization rule and combines two operators A and A’ of the same type with the same inputs into a single operator, as shown in Figure 8 (c) (d). 

- **Swapping Rule.** It inserts Store and Load between an operator A and one of its users B to represent that A will be swapped-out to external storage first, and then swapped-in when B needs to use it, as shown in Figure 8 (e). 

- **De-swapping Rule.** It is the dual of the swapping rule and removes Store and Load between two operators, as shown in Figure 8 (f). 

With the help of the rules above, we can decompose graph scheduling into graph transformation and re-ordering, where transformation phase decides what operators need to be recomputed / swapped, and re-ordering decides when to recompute / swap. Then the trade-offs between memory and latency can be moved to graph transformation phase, and 

613 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

**==> picture [505 x 61] intentionally omitted <==**

**Figure 7.** Illustrations of F-Tree Mutation Rules. **(a)** Enable an F-Tree node. **(b)** Lift an F-Tree node. **(c)** Disable an F-Tree node. **(d)** Increase the fission number _𝑛_ (with dimension length _𝑁_ = 12). 

**==> picture [504 x 49] intentionally omitted <==**

**----- Start of picture text -----**<br>
* C * B * * C * B * * C * B * * C * B * *A A’* * A * *A A’* * A * *LoadB Load*<br>AX * A X* A’ * X1 A X2 * * XA1 A’X2 * X* X * * X1 X2 * * X1 X2 * * BA * StoreA * StoreA * *A *<br>(a)  Re-mat. Rule 1  (b)  Re-mat. Rule 2 (c)  De-Re-mat. Rule 1  (d)  De-Re-mat. Rule 2  (e)  Swapping Rule (f)  De-Swapping Rule<br>**----- End of picture text -----**<br>


**Figure 8.** Scheduling-based Rules, representing the transformations decomposed from graph scheduling. The edges marked with an asterisk (*) represent zero or multiple edges 

**Algorithm 2:** M-Optimizer: Incremental Scheduling 

||**Algorithm 2:**M-Optimizer: Incremental Scheduling|**Algorithm 2:**M-Optimizer: Incremental Scheduling|
|---|---|---|
||**input**<br>**:**old, new graph:_𝐺𝑜𝑙𝑑,𝐺𝑛𝑒𝑤_;||
||old mutated sub-graph nodes:_𝑆𝑜𝑙𝑑_;||
||schedule of old graph:_𝜓𝑜𝑙𝑑_||
||**output**<br>**:**schedule of new graph:_𝜓𝑛𝑒𝑤_||
|**1 **|**function**GetRescheduleInterval(_𝐺,𝑆,𝜓_)**:**||
|**2**|**function**ExtendBound(_𝑖,𝑑_)**:**||
|**3**|ˆ_𝑛_=∞;_𝑣_:=_𝜓_[_𝑖_];_𝑙_:=0;||
|**4**|**while**_𝑙<_20∧_(_ˆ_𝑛>_10∨nw(_𝑣_) _<_4_)_∧nw(_𝑣_) _<_ ˆ_𝑛_**do**||
|**5**||ˆ_𝑛_:=nw(_𝑣_);_𝑖_:=_𝑖_+_𝑑_;_𝑣_:=_𝜓_[_𝑖_];_𝑙_:=_𝑙_+1;|
|**6**|**return**_𝑖_;||
|**7**|_𝐼𝑆_:= {_𝑖_| _𝑖_=1_, ...,_|_𝜓_| if_𝜓_[_𝑖_] ∈_𝑆_};||
|**8**|**return**ExtendBound(min_𝐼𝑆,_−1),ExtendBound(max_𝐼𝑆,_1);||
|**9**|beg,end:=GetRescheduleInterval(_𝐺𝑜𝑙𝑑,𝑆𝑜𝑙𝑑,𝜓𝑜𝑙𝑑_);||
|**10**|_𝑆𝑛𝑒𝑤_:= V(_𝐺𝑛𝑒𝑤_) \ (_𝜓𝑜𝑙𝑑_[:beg] ∪_𝜓𝑜𝑙𝑑_[end:]);||
|**11**|Ψ:= {DpSchedule(_𝑆_) | _𝑆_∈GraphPartition(_𝑆𝑛𝑒𝑤_)};||
|**12 **|**return**Merge(_𝜓𝑜𝑙𝑑_[:beg]_,_MergeSubSched(Ψ)_,𝜓𝑜𝑙𝑑_[end:]);||



graph scheduling phase only needs to consider re-ordering that generally has no effect on total execution latency. Such decomposition makes the scheduling after each graph transformation step much simpler. 

**Heuristic.** Considering the Re-materialization Rule and Swapping Rule can be applied to almost any operator, resulting in a large search space that slows down optimization, in the actual sub-graph pattern-matching process, these two rules can be selectively applied, _filtering out sub-graphs that do not contain memory hot-spots._ 

## **6 M-Optimizer** 

In this section, we first introduce incremental scheduling, to efficiently generate the optimal schedule for the transformed graph using information from the mutated sub-graph and the previous schedule. We then present the top-level search algorithm, which prioritizes M-States based on both memory and latency and transforms current best M-States using M- Rules to generate new M-States. 

## **6.1 Incremental Scheduling** 

To obtain memory usage and performance of a graph, we need to perform graph scheduling. Performing full graph scheduling after each graph transformation is expensive. To address this issue, we design an incremental scheduling algorithm that determines the subset of the graph that needs to be rescheduled based on the previous scheduling and the subgraph scope impacted by the previous graph transformation. This approach allows us to perform scheduling only on the necessary sub-graphs, reducing the overhead of scheduling. Algorithm 2 presents the details. It first obtains the sequence of operators that need to be rescheduled in the original graph by using GetRescheduleInterval (line 9). Next, the corresponding sub-graph _𝑆𝑛𝑒𝑤_ is obtained for this sequence in the new graph (line 10), which is then partitioned into several sub-graphs that can be independently scheduled using GraphPartition (line 11). The scheduling of each subgraph is performed using the dynamic programming-based algorithm in previous work [3] (line 11), and finally, the resulting schedules are combined to form the schedule for the new graph, which is integrated with the schedule for the original graph (line 12). 

GetRescheduleInterval is a crucial processes in Algorithm 2, designed to find the interval in the original schedule that needs to be rescheduled. The interval should not be too small, otherwise the rescheduled result would be suboptimal or even incorrect. Also, the interval should not be too large, otherwise the rescheduling process will consume too much time. Trading between the optimization quality and time cost is important. 

We introduce narrow waist (NW) value nw( _𝑣_ ) of a node _𝑣_ to solve it. For a graph _𝐺_ and a node _𝑣_ ∈V( _𝐺_ ), nw( _𝑣_ ) is defined as |V( _𝐺_ )| −| _𝐺._ anc( _𝑣_ )| −| _𝐺._ des( _𝑣_ )| − 1, i.e., |V( _𝐺_ ) \ _𝐺._ anc( _𝑣_ ) \ _𝐺._ des( _𝑣_ )| − 1. The NW value can be used to measure the number of nodes that are independent of the given node. A lower nw( _𝑣_ ) implies that more nodes are dependent on _𝑣_ and _𝑣_ depends on more nodes, which makes 

614 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

**Algorithm 3:** M-Optimizer: Search Algorithm 

||**Algorithm 3:**M-Optimizer: Search Algorithm|
|---|---|
||**input**<br>**:**input graph_𝐺_; memory constraint_𝑀_;|
||F-Tree max-level_𝐿_;|
||**output**<br>**:**optimized M-State_𝜇𝑏𝑒𝑠𝑡_|
|**1 **|**function** BetterThan(_𝜇_1_, 𝜇_2_,𝛿_=1)**:**|
|**2**|**return** (max(_𝜇_1_._mem_, 𝑀_)_, 𝜇_1_._lat) _<_|
||(max(_𝛿_×_𝜇_2_._mem_, 𝑀_)_,𝛿_×_𝜇_2_._lat);|
|**3 **|**function**GraphHash(_𝐺_)**:**|
|**4**|**for**_𝑣_∈topo-order(_𝐺_) **do**|
|**5**|_𝑥𝑣_:=hash(hash(_𝑣_) ⊕(�<br>_𝑢_∈_𝐺._pre(_𝑣_) _𝑥𝑢_));|
|**6**|**return**hash(�<br>_𝑣_∈_𝐺𝑥𝑣_);|
|**7**|_𝜇𝑏𝑒𝑠𝑡_:=InitState(_𝐺_);_𝑋_:= ∅;|
|**8**|_𝑄_:=PriorityQueue({_𝜇𝑏𝑒𝑠𝑡_}_,_ BetterThan);|
|**9 **|**while**_𝑄_≠∅**do**|
|**10**|_𝜇_:=_𝑄._pop();_𝑥_:=GraphHash(_𝜇.𝐺_);|
|**11**|**if**_𝑥_∈_𝑋_**then continue**;|
|**12**|_𝑋_:=_𝑋_∪{_𝑥_};|
|**13**|**if** _𝜇’s F-Tree needs update_**then**|
|**14**|_𝜇_:=Analyze(_𝜇, 𝐿_); # Algorithm1|
|**15**|**for**_𝜇_′ ∈ApplyTransformRules(_𝜇_)**do**|
|**16**|_𝜇_′ :=ApplyIncrementalSchedule(_𝜇_′); # Algorithm2|
|**17**|**if**<br>BetterThan(_𝜇_′_, 𝜇𝑏𝑒𝑠𝑡_)**then** _𝜇𝑏𝑒𝑠𝑡_:=_𝜇_′ ;|
|**18**|**if**<br>BetterThan(_𝜇_′_, 𝜇𝑏𝑒𝑠𝑡,_1_._1)**then** _𝑄._push(_𝜇_′) ;|
|||
|**19 **|**return**_𝜇𝑏𝑒𝑠𝑡_;|



_𝑣_ a suitable dividing point for topological ordering problem. Specifically, all the nodes that _𝑣_ depends on should be scheduled before _𝑣_ , and all the nodes that are dependent on _𝑣_ should be scheduled after _𝑣_ , providing a natural partition of the scheduling problem. Also, after we find the optimal schedules separately for _𝐺._ anc( _𝑣_ ) and _𝐺._ des( _𝑣_ ), the peak memory consumption is guaranteed to be less than _𝑀𝑜𝑝𝑡_ +[�] _𝑣_ ∈V( _𝐺_ )\ _𝐺._ anc( _𝑣_ )\ _𝐺._ des( _𝑣_ )[|] _[𝑣]_[|][, where] _[ 𝑀] 𝑜𝑝𝑡_[represents] the peak memory achieved under the optimal scheduling of _𝐺_ . If nw( _𝑣_ ) = 0, then the scheduling problem for the graph can be divided into two completely independent subproblems at _𝑣_ . We design a heuristic algorithm based on the NW value to select interval whose boundary NW values are as small as possible (line 2-6), where the constants 20 _,_ 10 _,_ 4 are empirical hyper-parameters which perform well in practical. The idea behind GraphPartition is to use nodes with nw( _𝑣_ ) ≤ 1 as dividing points to partition each connected component of the given graph into multiple sub-graphs. 

## **6.2 Top-level Search Algorithm** 

MAGIS adopts a greedy search algorithm to optimize graphs. There are two modes of optimization supported by MAGIS: optimizing latency given memory limit or optimizing memory given latency limit. Algorithm 3 shows the search algorithm for the former mode. 

The inputs of Algorithm 3 consist of a graph _𝐺_ , a given memory limit _𝑀_ , and F-Tree max-level _𝐿_ . We first schedule and analyze the given graph to obtain an initial M-State (line 

**Table 2.** Workloads for Evaluation 

|Name|Batch|Other Confguration|
|---|---|---|
|ResNet-50[19]|64|image-size=224|
|BERT-base[12]|32|sequence-length=512|
|ViT-base[15]|64|image-size=224,patch-size=16|
|U-Net[45]|32|image-size=256|
|U-Net++[73]|16|image-size=256|
|GPT-Neo-1.3B[6]|32|sequence-length=512|
|BTLM-3B[13]|32|sequence-length=512|



9). Then we construct a priority queue for storing M-State (line 10) where the priority is determined by the BetterThan function (line 1-4) that compares latency first when both M- States satisfy the memory limit _𝑀_ ; otherwise, it compares memory (note that we compare ( _𝑎,𝑏_ ) _<_ ( _𝑐,𝑑_ ) with lexicographical order). We then iteratively pop an M-State _𝜇_ (line 12) and apply M-Rules to generate a series of new M-State (line 17). The Analyze function (line 16) will update the F- Tree in M-State _𝜇_ if its previously mutated sub-graph is not influenced by F-Trans. We perform incremental scheduling on the newly generated M-State _𝜇_[′] . Then we will push _𝜇_[′] to queue if it’s not worse than _𝜇𝑏𝑒𝑠𝑡_ in a relaxed condition (controlled by a small coefficient _𝛿_ , empirically set to 1 _._ 1). To prevent redundant search, we borrow the idea of WeisfeilerLehman Test [48] to hash a given graph (line 5-8, line 12-14), where ⊕ means bytes concatenation operation. 

To reduce the overhead of performance measurement, we implement a simulator with an operator performance cache. It saves the actual execution latency of operators, and uses a simulation approach to obtain the overall performance and memory usage of the whole graph with a schedule. When considering asynchronous swapping, re-ordering involving Store/Load operators can also slightly affect latency. To address this, our re-ordering strategy is to place the Store as early as possible and place the Load as late as the data transfer latency can be just hidden. 

## **7 Evaluation** 

## **7.1 Experiment Setup** 

We use rustworkx [52] to implement MAGIS’s graph data structure. We implement a code generation backend to generate Python code calling PyTorch APIs based on the graph and schedule. We use PyTorch’s CUDA Stream API to implement asynchronous Store and Load. The data is swapped between GPU memory and CPU memory. Although our current implementation targets NVIDIA GPU, our methods can be easily ported to other platforms. 

Our main baselines for comparison are: (1) PyTorch [36]: unoptimized graphs are directly converted into PyTorch code after simple topo-order scheduling, acting as the baseline for memory usage and execution latency. Note that basic memory saving are applied for this baseline, that is, future-unused tensors are deleted immediately. (2) POFO [5]: state-of-theart work for memory optimization of networks with simple 

615 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

structures and linearly connected cells, considering both re-materialization and swapping. We use the open-sourced implementation of POFO[4] . (3) DTR [27]: state-of-the-art work using re-materialization technology for memory optimization of arbitrary networks. We use the implementation of DTR in MegEngine [1] (its eager mode and PyTorch both call cuBLAS & cuDNN for computation on NVIDIA GPUs with the same performance). (4) XLA [46]: state-of-the-art DNN compiler using a greedy re-materialization algorithm for memory optimization. (5) TVM [9] (Relay [43]): stateof-the-art DNN compiler, performing basic memory saving to reclaim future-unused tensors. (6) Torch-Inductor [2] (TI): state-of-the-art DNN compiler leveraging OpenAI Triton [50], performing basic memory saving to recycle tensors that are no longer used in the future. 

Table 2 shows the workloads we use for evaluation. We select the training processes of the following networks as experiment workloads: (1) Classic CNN classification network: ResNet [19], with linear inter-cell connection and simple intra-cell structure. (2) Classic transformer networks: BERT [12] and ViT [15], with linear inter-cell connection and complicated intra-cell structure. (3) Image segmentation networks with long skip-connections: U-Net [45] and U-Net++ [73], with complicated inter-cell connections (UNet++ is even more complex than U-Net) and simple intracell structure. (4) Large language models: GPT-Neo-1.3B [6] and BTLM-3B [13], with much larger weights and deeper structures compared with classic transformer networks. Note that the workloads diversely span from language models to vision models, from large models to small models. The data type is bf16 for GPT-Neo & BTLM, and tf32 for others. 

The platform we use for our experiments is an Intel workstation equipped with 20 Intel(R) Xeon(R) Silver 4210R CPUs, an NVIDIA GeForce RTX 3090 GPU, CUDA version 11.6, cuDNN version 8.4.0, PyTorch version 2.1.0, MegEngine version 1.12.3, TensorFlow version 2.15.0, and TVM version 0.14.0. The max-level parameter _𝐿_ of Algorithm 3 is 4 by default. For every optimization process, we run MAGIS with a time budget of 3 minutes. For each baseline, we first use TASO rules (mainly the A-Trans rules which merge operations like the QKV-projections in a transformer-block into a single operation and split the result later) to optimize the network to ensure a fair comparison. We measure the peak memory usage of the optimization results of MAGIS, PyTorch, POFO, and TI via torch.cuda.max_memory_allocated; for DTR, we use megengine.get_max_allocated_memory; for XLA, we use tf.config.experimental.get_memory_info; for TVM, we hack the memory allocation information of its memory planner. Note that, since baseline PyTorch cannot run the workload settings of GPT-Neo and BTLM in the experiment platform because of out-of-memory, we measure its latency and peak memory using MAGIS’s simulator. 

> 4https://gitlab.inria.fr/hiepacs/rotor/-/tree/offload 

**==> picture [241 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Latency Overhead < 10%  OOM<br>1<br>0.8<br>0.6 OOM<br>0.4<br>0.2<br>0<br>ResNet (b64) BERT (b32) ViT (b64) UNet (b32) UNet++ (b16) GPT-Neo (b32) BTLM (b32)<br>(b) Latency Overhead < 5% 1 OOM<br>0.8<br>0.6 OOM<br>0.4<br>0.2<br>0<br>ResNet (b64) BERT (b32) ViT (b64) UNet (b32) UNet++ (b16) GPT-Neo (b32) BTLM (b32)<br>MAGIS POFO DTR XLA TVM TI<br>Memory Ratio<br>Memory Ratio<br>**----- End of picture text -----**<br>


**Figure 9.** Peak memory ratio compared to un-optimized PyTorch (lower is better). "OOM" means the memory usage exceeds the memory limit of our experiment platform. 

**==> picture [242 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Memory Ratio < 80%  FAILURE<br>0.4<br>0.3<br>0.2<br>0.1<br>0<br>ResNet (b64) BERT (b32) ViT (b64) UNet (b32) UNet++ (b16) GPT-Neo (b32) BTLM (b32)<br>(b) Memory Ratio < 40%  FAILURE<br>0.8<br>0.6<br>0.4<br>0.2<br>0<br>ResNet (b64) BERT (b32) ViT (b64) UNet (b32) UNet++ (b16) GPT-Neo (b32) BTLM (b32)<br>MAGIS POFO DTR XLA TVM TI<br>Latency Overhead<br>Latency Overhead<br>**----- End of picture text -----**<br>


**Figure 10.** Latency overhead compared to PyTorch without optimization (lower is better). "FAILURE" means the memory ratio cannot be optimized to meet the constraint. 

## **7.2 Experiment Results** 

## **7.2.1 Memory Optimization with Latency Constraint.** 

We first evaluate the memory optimization effects of MAGIS and baselines under 10% and 5% latency overhead constraints. Results are shown in Figure 9. With 10% latency overhead limit, MAGIS optimizes peak memory to 15%∼60% of PyTorch’s, outperforming other baselines (60% at best). TVM and TI only perform basic memory saving like the PyTorch baseline, so their optimized memory ratios are near to 100%. MAGIS’s memory is 15%∼80% of POFO’s, 20%∼85% of DTR’s, and 15%∼70% of XLA’s. At 5% latency overhead limit, MAGIS optimizes peak memory to 25%∼70% of PyTorch’s, 25%∼80% of POFO’s, 35%∼80% of DTR’s, and 25%∼80% of XLA’s. 

For ResNet, when the latency overhead limit is 10% (5%), MAGIS’s peak memory is around 80%∼85% (75%∼80%) of POFO & DTR & XLA. MAGIS’s results are closed to baselines’, mainly because ResNet has a simple structure, and the benefits brought by our methods are limited. 

For BERT and ViT, MAGIS achieves 50%∼70% (65%∼75%) of the baselines’ memory at 10% (5%) latency overhead constraint. The relative results of MAGIS are better than ResNet 

616 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

**==> picture [506 x 91] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) ResNet-50 (batch=64) (b) BERT-base (batch=32) (c) UNet (batch=32) (d) GPT-Neo (batch=32)<br>1.2 0.8 1 0.7<br>MAGIS 0.81 0.6 0.8 0.60.5<br>POFO 0.6 0.4 0.6 0.4<br>DTR 0.4 0.4 0.3<br>XLA 0.2 0.2 0.2 0.2<br>0 0.1<br>TVMTI -0.2-0.4 0 0.2 0.4 0.6 0.8 1 -0.20 0 0.2 0.4 0.6 0.8 1 -0.20 0 0.2 0.4 0.6 0.8 1 0 0 0.2 0.4 0.6 0.8 1<br>Memory Ratio Memory Ratio Memory Ratio Memory Ratio<br>Latency Overhead<br>Latency Overhead Latency Overhead Latency Overhead<br>**----- End of picture text -----**<br>


**Figure 11.** Latency & memory curves of MAGIS and baselines. MAGIS can achieve Pareto optimal in almost all cases. 

**==> picture [242 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.6<br>0.5 ViT (batch=64, patch-size=16)<br>0.4<br>0.3 MAGIS<br>0.2 POFO<br>0.1 POFO (factor=32)<br>0 POFO (factor=16)<br>0 0.2 0.4 0.6 0.8 1 POFO (factor=8)<br>Memory Ratio<br>Latency Overhead<br>**----- End of picture text -----**<br>


**Figure 12.** Comparing MAGIS with POFO. The network used by POFO has been pre-processed with micro-batching (with different factors). 

due to more intra-cell complexity of transformer networks. MAGIS performs better on ViT than on BERT due to shorter sequence length of ViT. Sequence length has a larger impact on latency than on peak memory, making it more challenging to optimize the memory under a given latency constraint with longer sequence. For UNet and UNet++, MAGIS achieves 15%∼35% (25%∼70%) of baselines’ memory at 10% (5%) latency overhead constraint. MAGIS performs better on these two networks compared to other networks due to more complex inter-cell structures which provide more optimization space for graph transformation. For GPT-Neo and BTLM, MAGIS maintains ≤40% (≤60%) of PyTorch’s memory at 10% (5%) latency overhead limit. Only XLA avoids OOM among baselines for GPT-Neo. All baselines are OOM for BTLM under both constraints. 

**7.2.2 Latency Optimization with Memory Constraint.** We then conduct experiments to compare the latency optimization effects of MAGIS and other works under 80% and 40% peak memory limits of un-optimized PyTorch. Results are shown in Figure 10. TVM & TI cannot optimize all the workloads into 80% memory ratio. POFO almost cannot optimize UNet & UNet++. DTR’s processes for UNet++, GPT-Neo, and BTLM take too long with a 40% memory limit, and XLA also cannot optimize these workloads under such constraints. We mark these cases as "FAILURE" in the figure. With an 80% limit, MAGIS reduces latency overhead to ≤5%, better than POFO (≤40% for BTLM, ≤20% for others), DTR (≤15%), and XLA (≤20%). At 40% memory limit, MAGIS maintains ≤15% overhead, while POFO stays at ≤40%, DTR reaches ≤45% for ResNet/BERT/ViT and ≤70% for UNet, and XLA caps at 

≤70%. Similar to the previous experiments, MAGIS performs the best on UNet/UNet++, followed by ResNet/BERT/ViT, achieving a 1.25× speedup over DTR for UNet under the 40% memory limit. Among the baselines, only POFO can optimize GPT-Neo and BTLM under the 40% memory limit, but with much higher latency overhead than MAGIS. Note that although TASO rules bring some peak memory overhead, the overhead is small (around 5% on average) since these rules only enlarge local memory footprint. The baselines that fail to meet the memory constraints in Figure 10 still fail without applying TASO rules. 

**7.2.3 Trade-off Curves of Latency & Memory.** We plot the memory & latency trade-off curves in Figure 11. Note that XLA, TVM, and TI may achieve lower latency than the PyTorch baseline when there’s no memory constraint, resulting in points below the horizontal line. When memory ratio is 1, MAGIS is faster than PyTorch but slower than XLA, TVM, and TI, due to MAGIS currently not implementing the sophisticated compilation optimizations like operator fusion as these compilers do. From the results, it can be observed that MAGIS’s curve remains mostly below the baselines’ curve. This indicates that we have achieved a better Pareto boundary in the dual-objective optimization of memory and latency, which means that, given a latency constraint, MAGIS achieves lower memory consumption, or given a memory constraint, it achieves lower latency. 

We observe XLA’s curve is nearly linear but experiences substantial latency overhead under low memory limits, since when memory limit is tight, re-computing one operator might depend on another operator re-materialization. The re-materialization used by DTR is better than XLA’s greedy algorithm, enabling a near-linear trade-off between memory and latency even under tight memory limits. POFO’s curve is also near-linear as it also adopts swapping, which balances memory and latency in a near-linear ratio. When the memory constraint it not tight, MAGIS’s curve is near-linear with a slope lower than baselines since it also employs graph transformations such as F-Trans to balance memory and latency. However, under strict memory limits, MAGIS’s curve becomes increasingly steep because even F-Trans incurs large overhead to optimize memory within tight constraint, caused by poor locality of on-chip memories due to small operators split from F-Trans. 

617 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

**==> picture [506 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Latency Overhead < 10% (b) Latency Overhead < 5% (c) Memory Ratio < 80% (d) Memory Ratio < 40%<br>1 1 0.12 0.3<br>0.9 0.1 0.25<br>0.9<br>0.8 0.08 0.2 naïve-fission<br>0.7 0.8 0.06 0.15 naïve-sch-rule<br>0.6 0.04 0.1 max-level=2<br>0.5 0.7 0.02 0.05 max-level=4<br>0.4 0.6 0 0 max-level=8<br>0 30 60 90 120 150 180 0 30 60 90 120 150 180 0 30 60 90 120 150 180 0 30 60 90 120 150 180<br>Elapsed Optimization Time (secs) Elapsed Optimization Time (secs) Elapsed Optimization Time (secs) Elapsed Optimization Time (secs)<br>Memory Ratio Memory Ratio<br>Latency Overhead Latency Overhead<br>**----- End of picture text -----**<br>


**Figure 13.** Heuristic breakdown of MAGIS when optimizing BERT workload in 3 minutes with the constraints used in §7.2.1 and §7.2.2. The diamond "⋄" in a curve is the time point after which its optimization result meets the constraint. The square "□" in a curve is the time point with the best optimization result. 

**==> picture [241 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
35 1.1<br>30 speedup geomean<br>25 1<br>20<br>0.9<br>15<br>10 0.8<br>5<br>0 0.7<br>1 11 21 31 41 51 61 71 81 91 101 0 10 20 30 40 50 60 70 80 90 100<br>(a) Time cost speedup  (b) Optimization quality<br>**----- End of picture text -----**<br>


**Figure 14.** Comparison between incremental scheduling (IS) and full scheduling (FS). **(a)** Speedup of IS’s scheduling time relative to FS’s scheduling time. **(b)** Schedule result quality of IS compared to FS. 

|**Total **|**Trans.**<br>**Sched.**<br>**Simul.**|**Hash**|**Filtered**|**Others**|
|---|---|---|---|---|
|Count<br>/|7148<br>924<br>924|7148|6224|/|
|Cost (secs)<br>60|2.52<br>3.70<br>8.71|44.82|/|0.25|



**Figure 15.** Optimization time cost breakdown of MAGIS when optimizing ViT (batch 64) in 1 minutes. "Filtered" means the duplicated graphs filtered by hash test. 

**7.2.4 Comparison to Micro-batching.** We examine the effect of graph transformation on memory optimization by integrating it into baselines. We select ViT as workload and POFO as target baseline. We focus on F-Trans due to its substantial memory impact. We apply micro-batching to ViT, dividing the whole graph (factors: 32, 16, 8) along batchdimension to simulate a simple F-Trans. The split sub-graph are fed to POFO, and execution latency is multiplied by the sub-graph count. 

Figure 12 reveals that graph transformation optimization enhances POFO’s performance under stringent memory constraints. Under different memory limits, POFO performs best with different factors. This indicates that there are different trade-off spaces between graph transformation and scheduling. MAGIS outperforms both optimized and original POFO due to better coordinating transformation and scheduling. 

**7.2.5 Heuristic Ablation.** The heuristics used in MAGIS main include: **H1** ) F-Tree construction (Algorithm 1) discussed in §4.3; **H2)** heuristic used for schedule-based rules mentioned in §5.2; **H3)** hyper-parameter _𝐿_ (Algorithm 1 

and 3) to control the max-level of F-Tree. We conduct a breakdown experiment with five settings: _○_[1] **naïve-fission** : disabling **H1** by randomly selecting valid sub-graph & dimension for F-Trans; _○_[2] **naïve-sch-rules** : disabling **H2** by matching schedule-based rules on the whole graph; _○_[3] / _○_[4] / _○_[5] **max-level=2/4/8** : setting hyper-parameter _𝐿_ as 2/4/8 (default is _𝐿_ = 4 for other settings). We evaluate them on BERT workload under constraints used in §7.2.1 and §7.2.2 with a time budget of 3 minutes. Figure 13 depicts the curves of their elapsed optimization time and the historical best results during searching. 

**naïve-fission** performs the worst due to limited F-Trans optimization of memory, causing up to 70% and 45% higher peak memory consumption and 10%-12% higher latency overhead than the best setting. **naïve-sch-rule** outperforms **naïve-fission** due to enabling **H1** . But it lags behind others since it disables **H2** , slowing down search convergence and making it challenging to find better results within the time budget. Settings excluding **naïve-fission** and **naïve-schrule** generally yield better outcomes, with **max-level=4** exhibiting the best overall performance. **max-level=2** restricts the F-Trans search space due to shorter F-Tree, thereby reducing optimization potential. Conversely, **max-level=8** expands the search space, slowing search and making optimization more difficult; in (d), **max-level=8** is even inferior to **naïve-sch-rule** (with _𝐿_ = 4). 

**7.2.6 Optimization Time.** Figure 15 illustrates the time costs of different processes in a 1-minute ViT (batch 64) training optimization using MAGIS. Processes include Transformation ("Trans."), Scheduling ("Sched."), Simulation ("Simul."), Hash Test ("Hash"). "Filtered" indicates the number of graphs filtered out after hash test. "Trans." contributes a minor 2.52s overhead, while "Sched." stands at 3.7s. "Simul.", necessitated by operator performance data collection, exhibits the highest average overhead at 8.71s. "Hash" incurs the highest total overhead (44.82s), mainly due to filtering duplicate graphs, effectively reducing other processes’ overhead. 

## **7.3 Evaluation of Incremental Scheduling** 

We evaluate incremental scheduling (IS) against full scheduling (FS) in terms of speed for 10 randomly generated DNNs 

618 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

**==> picture [242 x 81] intentionally omitted <==**

**----- Start of picture text -----**<br>
15<br>UNet (batch=32)<br>10<br>PyTorch<br>5<br>M Aa G iaS 1IS-1<br>0 M Aa G iaS 2IS-2<br>0 100 200 300 400 500<br>Execution Timeline (ms)<br>Memory Usage (GB)<br>**----- End of picture text -----**<br>


**Figure 16.** Execution time & memory usage for UNet. 

with structures resembling NASNet [75]. Using TASO’s [25] graph transformation rules, we conduct 100 rounds of transformations (10 rounds per DNN) after an initial scheduling. Both IS and FS employ the DP algorithm from [3] (DpSchedule in Algorithm 2). Figure 14 (a) illustrates IS’s speed advantage over FS, achieving a speedup of 4 ∼ 30× (10× in average) across 100 tests. Figure 14 (b) presents the optimization quality of IS, measured as the ratio of peak memory usage optimized by IS to that optimized by FS. In 94 out of 100 tests, IS attains the same level of optimality as FS. 

## **7.4 Case Study** 

We use UNet as a case study to demonstrate the optimization effect. Figure 16 depicts UNet’s training time & memory with PyTorch, MAGIS-1 (memory limited at 80% of PyTorch’s peak), and MAGIS-2 (limited at 60%). Both PyTorch and MAGIS-1 display initial memory increase followed by decrease due to activation saving during forward phase and activation releasing during backward phase. MAGIS-1 has lower peak memory thanks to re-materialization, swapping, and F- Trans, but incurs higher latency. MAGIS-2 exhibits dual memory peaks, caused by a F-Trans covering the whole graph. This reduces peak memory further compared to MAGIS-1, yet increases latency overhead. 

## **8 Related Work** 

In this section, we briefly introduce the related work of MAGIS, mainly including techniques of graph scheduling (re-materialization, swapping, and re-ordering) as well as graph transformation. Some other DNN compilers are also discussed in this section. 

**Re-materilization** evicts some intermediate tensors and re-computing them later when needed. It was first applied in deep learning by [10, 17, 18]. Graph-theoretic analysis is used in [28, 29]. Checkmate [24] uses Integer Programming (IP) for optimization. DTR [27] uses heuristic strategies to optimize re-mat. of dynamic graphs. MONeT [47] co-optimize re-mat. and operator implementations. 

**Swapping** stores some tensors on external storage and reloads them later when needed. vDNN [42], Capuchin [38], and SuperNeurons [57] use it for DNN training on GPUs. SwapAdvisor [22] co-optimizes re-ordering, memory allocation, and swapping. TFLMS [30] represents swapping by special operators and control-flow edges. POET [37] uses 

IP to combine re-mat. and swapping for training on mobile devices. POFO [5] uses Dynamic Programming (DP) to combine re-mat. and swapping. ZeRO-Offload [41] combines swapping with distributed training. AutoTM [20] and ZeROInfinity [39] use persistent memory as external storage. 

**Re-ordering** finds proper topo-order of DNNs to optimize memory. Serenity [3] uses DP for optimization. SwapAdvisor [22] considers both re-ordering and swapping. HMCOS [58] hierarchically searches optimal ordering. Zhong et al. [72] use IP with variable pruning to speedup optimization. 

**Graph Transformation** originates from compiler’s super optimization [34]. It gradually optimizes the graph with a sub-graph mutated at each step. MetaFlow [26] uses backtracking algorithm, and TenSAT [62] employs equality saturation [60] for searching. TASO [25] generates transformation rules automatically based on program synthesis. PET [56] proposed partial equivalent transformation. Unity [54] integrates distributed parallel optimization into graph transformation. Turner et al. [53] combine graph transformation with neural architecture search. Compared to previous work, MAGIS can trade the latency and memory optimization. Regarding transformation types, MAGIS investigates the formalization and search for fission transformation. We also propose the re-materialization and swapping rules derived from graph scheduling, enhancing the coordination between graph transformation and scheduling. 

**Other DNN Compilers.** Besides the works mentioned above, many other DNN compilers have been proposed [2, 8, 9, 11, 14, 16, 31, 33, 43, 46, 49, 50, 59, 61, 63–71, 74] in recent years. For example, AutoTVM [11], FlexTensor [70], Ansor [64], and Roller [74] automatically generate/explore tuning space of a single operator or a small sub-graph; UNIT [59], AMOS [66], and TensorIR [16] automatically map operators onto hardware accelerators with specialized tensor instructions; Rammer [33], HFuse [31], and IOS [14] fuse parallel operators to increase hardware utilization; DNNFusion [35], AStitch [71], and Apollo [63] fuse chained operators to reduce data movement; BOLT [61], Chimera [69], SET [8], TileFlow [67], and Welder [49] additionally explore fusion space for compute-intensive operators. 

## **9 Conclusion** 

We propose MAGIS, a DNN optimizer for memory & latency with a systematic design of fission transformation effective coordination between graph transformation and scheduling. Experimental results show that compared to state-of-the-art methods, MAGIS only uses 15% ∼ 85% memory with same latency constraint and obtains a better memory & latency Pareto boundary. 

## **Acknowledgments** 

This work is supported in part by the National Natural Science Foundation of China (NSFC) under grant No.U21B2017. 

619 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

Renze Chen et al. 

## **References** 

- [1] Megengine: A fast, scalable and easy-to-use deep learning framework. https://github.com/MegEngine/MegEngine, 2020. 

- [2] torch.compiler - PyTorch 2.1 documentation. https://pytorch.org/docs/ 2.1/torch.compiler, 2023. 

- [3] Byung Hoon Ahn, Jinwon Lee, Jamie Menjay Lin, Hsin-Pai Cheng, Jilei Hou, and Hadi Esmaeilzadeh. Ordering Chaos: Memory-Aware Scheduling of Irregularly Wired Neural Networks for Edge Devices. _MLSys_ , 2:44–57, 2020. 

- [4] V Aho Alfred, S Lam Monica, and D Ullman Jeffrey. _Compilers Principles, Techniques & Tools_ . pearson Education, 2007. 

- [5] Olivier Beaumont, Lionel Eyraud-Dubois, and Alena Shilova. Efficient Combination of Rematerialization and Offloading for Training DNNs. In _NIPS_ , volume 34, pages 23844–23857, 2021. 

- [6] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with MeshTensorflow, 2021. 

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, and Amanda Askell. Language models are few-shot learners. _NIPS_ , 33:1877–1901, 2020. 

- [8] Jingwei Cai, Yuchen Wei, Zuotong Wu, Sen Peng, and Kaisheng Ma. Inter-layer Scheduling Space Definition and Exploration for Tiled Accelerators. In _ISCA_ , pages 1–17, 2023. 

- [9] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. Tvm: An Automated End-to-End Optimizing Compiler for Deep Learning. In _OSDI_ , pages 578–594, 2018. 

- [10] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost, 2016. 

- [11] Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. Learning to optimize tensor programs. _NIPS_ , 31, 2018. 

- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. _NAACL_ , pages 4171–4186, 2019. 

- [13] Nolan Dey, Daria Soboleva, Faisal Al-Khateeb, Bowen Yang, Ribhu Pathria, Hemant Khachane, Shaheer Muhammad, Zhiming, Chen, Robert Myers, Jacob Robert Steeves, Natalia Vassilieva, Marvin Tom, and Joel Hestness. Btlm-3b-8k: 7b parameter performance in a 3b parameter model, 2023. 

- [14] Yaoyao Ding, Ligeng Zhu, Zhihao Jia, Gennady Pekhimenko, and Song Han. Ios: Inter-operator scheduler for cnn acceleration. _MLSys_ , 3:167– 180, 2021. 

- [15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, and Sylvain Gelly. An image is worth 16x16 words: Transformers for image recognition at scale. _ICLR_ , 2020. 

- [16] Siyuan Feng, Bohan Hou, Hongyi Jin, Wuwei Lin, Junru Shao, Ruihang Lai, Zihao Ye, Lianmin Zheng, Cody Hao Yu, Yong Yu, and Tianqi Chen. TensorIR: An Abstraction for Automatic Tensorized Program Optimization. In _ASPLOS_ , pages 804–817, 2023. 

- [17] Andreas Griewank and Andrea Walther. Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. _TOMS_ , 26(1):19–45, 2000. 

- [18] Audrunas Gruslys, Rémi Munos, Ivo Danihelka, Marc Lanctot, and Alex Graves. Memory-efficient backpropagation through time. _NIPS_ , 29, 2016. 

- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _CVPR_ , pages 770–778, 2016. 

- [20] Mark Hildebrand, Jawad Khan, Sanjeev Trika, Jason Lowe-Power, and Venkatesh Akella. Autotm: Automatic tensor movement in heterogeneous memory systems using integer linear programming. In _ASPLOS_ , pages 875–890, 2020. 

- [21] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models. In _NIPS_ , volume 33, pages 6840–6851. Curran Associates, Inc., 2020. 

- [22] Chien-Chin Huang, Gu Jin, and Jinyang Li. Swapadvisor: Pushing deep learning beyond the gpu memory limit via smart swapping. In _ASPLOS_ , pages 1341–1355, 2020. 

- [23] Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely Connected Convolutional Networks. In _CVPR_ , 2018. 

- [24] Paras Jain, Ajay Jain, Aniruddha Nrusimha, Amir Gholami, Pieter Abbeel, Kurt Keutzer, Ion Stoica, and Joseph E. Gonzalez. Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization. _MLSys_ , 2020. 

- [25] Zhihao Jia, Oded Padon, James Thomas, Todd Warszawski, Matei Zaharia, and Alex Aiken. TASO: optimizing deep learning computation with automatic generation of graph substitutions. In _SOSP_ , pages 47– 62, 2019. 

- [26] Zhihao Jia, James Thomas, Todd Warszawski, Mingyu Gao, Matei Zaharia, and Alex Aiken. Optimizing DNN Computation with Relaxed Graph Substitutions. _MLSys_ , 1:27–39, 2019. 

- [27] Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, and Zachary Tatlock. Dynamic Tensor Rematerialization. _ICLR_ , 2021. 

- [28] Ravi Kumar, Manish Purohit, Zoya Svitkina, Erik Vee, and Joshua Wang. Efficient Rematerialization for Deep Networks. _NIPS_ , 32, 2019. 

- [29] Mitsuru Kusumoto, Takuya Inoue, Gentaro Watanabe, Takuya Akiba, and Masanori Koyama. A Graph Theoretic Framework of Recomputation Algorithms for Memory-Efficient Backpropagation. In _NIPS_ , volume 32, 2019. 

- [30] Tung D. Le, Haruki Imai, Yasushi Negishi, and Kiyokuni Kawachiya. Tflms: Large model support in tensorflow by graph rewriting, 2019. 

- [31] Ao Li, Bojian Zheng, Gennady Pekhimenko, and Fan Long. Automatic Horizontal Fusion for GPU Kernels. In _CGO_ , 2020. 

- [32] Gangmuk Lim, Jeongseob Ahn, Wencong Xiao, Youngjin Kwon, and Myeongjae Jeon. Zico: Efficient {GPU} memory sharing for concurrent {DNN} training. In _ATC_ , pages 161–175, 2021. 

- [33] Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, and Lidong Zhou. Rammer: Enabling Holistic Deep Learning Compiler Optimizations with {rTasks}. In _OSDI_ , pages 881–897, 2020. 

- [34] Henry Massalin. Superoptimizer: a look at the smallest program. _ASPLOS_ , 15(5):122–126, 1987. 

- [35] Wei Niu, Jiexiong Guan, Yanzhi Wang, Gagan Agrawal, and Bin Ren. DNNFusion: accelerating deep neural networks execution with advanced operator fusion. In _PLDI_ , pages 883–898, 2021. 

- [36] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, Soumith Chintala, H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché Buc, E. Fox, and R. Garnett. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In _NIPS_ , 2019. 

- [37] Shishir G. Patil, Paras Jain, Prabal Dutta, Ion Stoica, and Joseph E. Gonzalez. POET: Training Neural Networks on Tiny Devices with Integrated Rematerialization and Paging. In _ICML_ , 2022. 

- [38] Xuan Peng, Xuanhua Shi, Hulin Dai, Hai Jin, Weiliang Ma, Qian Xiong, Fan Yang, and Xuehai Qian. Capuchin: Tensor-based gpu memory management for deep learning. In _ASPLOS_ , pages 891–905, 2020. 

- [39] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. Zero-infinity: Breaking the gpu memory wall for extreme 

620 

ASPLOS ’24, April 27-May 1, 2024, La Jolla, CA, USA 

MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN 

scale deep learning. In _SC_ , pages 1–14, 2021. 

- [40] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 

- [41] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. ZeRO-Offload: Democratizing Billion-Scale Model Training. In _ATC_ , pages 551–564, 2021. 

- [42] Minsoo Rhu, Natalia Gimelshein, Jason Clemons, Arslan Zulfiqar, and Stephen W. Keckler. vDNN: Virtualized deep neural networks for scalable, memory-efficient neural network design. In _MICRO_ , pages 1–13, 2016. 

- [43] Jared Roesch, Steven Lyubomirsky, Marisa Kirisame, Logan Weber, Josh Pollock, Luis Vega, Ziheng Jiang, Tianqi Chen, Thierry Moreau, and Zachary Tatlock. Relay: A high-level compiler for deep learning, 2019. 

- [44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-Resolution Image Synthesis With Latent Diffusion Models. In _CVPR_ , pages 10684–10695, 2022. 

- [45] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In _MICCAI_ , pages 234–241, 2015. 

- [46] Amit Sabne. Xla: Compiling machine learning for peak performance. 2020. 

- [47] Aashaka Shah, Chao-Yuan Wu, Jayashree Mohan, Vijay Chidambaram, and Philipp Kraehenbuehl. Memory Optimization for Deep Networks. In _ICLR_ , 2022. 

- [48] Nino Shervashidze, Pascal Schweitzer, Erik Jan Van Leeuwen, Kurt Mehlhorn, and Karsten M. Borgwardt. Weisfeiler-lehman graph kernels. _JMLR_ , 12(9), 2011. 

- [49] Yining Shi, Zhi Yang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Ziming Miao, Yuxiao Guo, Fan Yang, and Lidong Zhou. Welder: Scheduling Deep Learning Memory Access via Tile-graph. In _OSDI_ , 2023. 

- [50] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In _MAPL_ , pages 10–19, 2019. 

- [51] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, MarieAnne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. 

- [52] Matthew Treinish, Ivan Carvalho, Georgios Tsilimigkounakis, and Nahum Sá. rustworkx: A high-performance graph library for python. _JOSS_ , 7(79):3968, 2022. 

- [53] Jack Turner, Elliot J. Crowley, and Michael O’Boyle. Neural Architecture Search as Program Transformation Exploration. _ASPLOS_ , 2021. 

- [54] Colin Unger, Zhihao Jia, Wei Wu, Sina Lin, Mandeep Baines, Carlos Efrain Quintero Narvaez, Vinay Ramakrishnaiah, Nirmal Prajapati, Pat McCormick, and Jamaludin Mohd-Yusof. Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization. In _OSDI_ , pages 267–284, 2022. 

- [55] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _NIPS_ , pages 5998–6008, 2017. 

- [56] Haojie Wang, Jidong Zhai, Mingyu Gao, Zixuan Ma, Shizhi Tang, Liyan Zheng, Yuanzhi Li, Kaiyuan Rong, Yuanyong Chen, and Zhihao Jia. PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections. In _OSDI_ , pages 37–54, 2021. 

- [57] Linnan Wang, Jinmian Ye, Yiyang Zhao, Wei Wu, Ang Li, Shuaiwen Leon Song, Zenglin Xu, and Tim Kraska. Superneurons: dynamic GPU memory management for training deep neural networks. In _PPoPP_ , pages 41–53, 2018. 

- [58] Zihan Wang, Chengcheng Wan, Yuting Chen, Ziyi Lin, He Jiang, and Lei Qiao. Hierarchical memory-constrained operator scheduling of 

   - neural architecture search networks. In _DAC_ , pages 493–498. Association for Computing Machinery, July 2022. 

- [59] Jian Weng, Animesh Jain, Jie Wang, Leyuan Wang, Yida Wang, and Tony Nowatzki. UNIT: Unifying Tensorized Instruction Compilation. In _CGO_ , pages 77–89, 2021. 

- [60] Max Willsey, Chandrakana Nandi, Yisu Remy Wang, Oliver Flatt, Zachary Tatlock, and Pavel Panchekha. egg: Fast and Extensible Equality Saturation. _POPL_ , 5:1–29, 2021. 

- [61] Jiarong Xing, Leyuan Wang, Shang Zhang, Jack Chen, Ang Chen, and Yibo Zhu. Bolt: Bridging the Gap between Auto-tuners and Hardwarenative Performance. _MLSys_ , 4, April 2022. 

- [62] Yichen Yang, Phitchaya Mangpo Phothilimtha, Yisu Remy Wang, Max Willsey, Sudip Roy, and Jacques Pienaar. Equality Saturation for Tensor Graph Superoptimization. _MLSys_ , 2021. 

- [63] Jie Zhao, Xiong Gao, Ruijie Xia, Zhaochuang Zhang, Deshi Chen, Lei Chen, Renwei Zhang, Zhen Geng, Bin Cheng, and Xuefeng Jin. Apollo: Automatic Partition-based Operator fusion through Layer by Layer Optimization. _MLSys_ , 4, 2022. 

- [64] Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, et al. Ansor: Generating high-performance tensor programs for deep learning. In _OSDI_ , pages 863–879, 2020. 

- [65] Size Zheng, Renze Chen, Yicheng Jin, Anjiang Wei, Bingyang Wu, Xiuhong Li, Shengen Yan, and Yun Liang. Neoflow: A flexible framework for enabling efficient compilation for high performance dnn training. _TPDS_ , 33(11):3220–3232, 2021. 

- [66] Size Zheng, Renze Chen, Anjiang Wei, Yicheng Jin, Qin Han, Liqiang Lu, Bingyang Wu, Xiuhong Li, Shengen Yan, and Yun Liang. Amos: Enabling automatic mapping for tensor computations on spatial accelerators with hardware abstraction. In _ISCA_ , pages 874–887, 2022. 

- [67] Size Zheng, Siyuan Chen, Siyuan Gao, Liancheng Jia, Guangyu Sun, Runsheng Wang, and Yun Liang. TileFlow: A Framework for Modeling Fusion Dataflow via Tree-based Analysis. In _MICRO_ , 2023. 

- [68] Size Zheng, Siyuan Chen, and Yun Liang. Memory and Computation Coordinated Mapping of DNNs onto Complex Heterogeneous SoC. In _DAC_ , pages 1–6, 2023. 

- [69] Size Zheng, Siyuan Chen, Peidi Song, Renze Chen, Xiuhong Li, Shengen Yan, Dahua Lin, Jingwen Leng, and Yun Liang. Chimera: An Analytical Optimizing Framework for Effective Compute-intensive Operators Fusion. In _HPCA_ , pages 1113–1126, 2023. 

- [70] Size Zheng, Yun Liang, Shuo Wang, Renze Chen, and Kaiwen Sheng. Flextensor: An automatic schedule exploration and optimization framework for tensor computation on heterogeneous system. In _ASPLOS_ , pages 859–873, 2020. 

- [71] Zhen Zheng, Xuanda Yang, Pengzhan Zhao, Guoping Long, Kai Zhu, Feiwen Zhu, Wenyi Zhao, Xiaoyong Liu, Jun Yang, Jidong Zhai, Shuaiwen Leon Song, and Wei Lin. AStitch: enabling a new multidimensional optimization space for memory-intensive ML training and inference on modern SIMT architectures. In _ASPLOS_ , 2022. 

- [72] Shuzhang Zhong, Meng Li, Yun Liang, Runsheng Wang, and Ru Huang. Memory-aware Scheduling for Complex Wired Networks with Iterative Graph Optimization. In _ICCAD_ , 2023. 

- [73] Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, and Jianming Liang. Unet++: A nested u-net architecture for medical image segmentation. In _DLMIA_ , pages 3–11, 2018. 

- [74] Hongyu Zhu, Ruofan Wu, Yijia Diao, Shanbin Ke, Haoyu Li, Chen Zhang, Jilong Xue, Lingxiao Ma, Yuqing Xia, Wei Cui, Fan Yang, Mao Yang, Lidong Zhou, Asaf Cidon, and Gennady Pekhimenko. {ROLLER}: Fast and Efficient Tensor Compilation for Deep Learning. In _OSDI_ , pages 233–248, 2022. 

- [75] Barret Zoph, Vijay Vasudevan, Jonathon Shlens, and Quoc V. Le. Learning Transferable Architectures for Scalable Image Recognition. In _CVPR_ , 2018. 

621 

