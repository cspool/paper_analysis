# <span id="page-5-0"></span>4.2 Two-Level Inductive Operator Scheduling

The scheduling algorithm minimizes the end-to-end execution time of a DL model by deciding the number of future operators to preload before or during each operator's on-chip execution (i.e., *preload number*). Each preload number represents a trade-off point between on-chip execution speed and HBM bandwidth utilization (Figure 4).

The optimization space is exponential to the number of operators. Suppose there are N operators in a DL model, and the on-chip memory can fit at most K operators. Each operator's execution can overlap with 1 to K operators' preload. Thus, there are  $O(K^N)$  combinations of preload numbers for all operators. For example, for IPU-POD4 (3.5GB on-chip memory) and OPT-30B, each identical layer has 84 operators and  $K \geq 28$ , forming up to  $28^{84}$  combinations.

We develop an O(KN)-time algorithm for this problem. The insight is, as operators in a DL model typically execute in a sequential order due to data dependency, instead of exploring all combinations of preload numbers, we can exploit the execution order and inductively derive the optimal preload number for each operator.

We can either start from the first operator and find the optimal preload number for each succeeding operator, or start from the last operator and schedule each preceding operator. For each operator, we explore all possible preload numbers based on the already scheduled operators, and we pick the preload number that minimizes the "start-to-current" or "current-to-end" execution time. As both induction directions are equivalent, we focus on the second one.

The base case of induction is trivial, as the last operator has no succeeding operators to preload (i.e., preload number is always 0). For the inductive step, we show an example in Figure 10. In Figure 10 (a), Elk has finished scheduling all operators after Op5. Then, Elk schedules the execution of Op5 in Figure 10 (b). Elk enumerates all possible preload numbers for Op5. For each preload number, Elk invokes the cost-aware on-chip memory allocation algorithm (§4.3) to determine the execution/preload space sizes for the involved operators and the estimated execution time of Op5.

For example, preload number 0 means we do not overlap Op5's execution with any preload. The execution time of Op5 is minimized, but the overall execution time from Op5 to the end of the model is sub-optimal. For preload numbers 1 and 2, Elk overlaps the

execution of 0p5 with the preload of 0p6, or the preloads of both 0p6 and 0p7. Though 0p5's execution time is longer, the overall execution times are better than preload number 0. As preload number 1 yields the lowest current-to-end time, Elk selects it for 0p5.

After scheduling Op5's execution, ELK schedules its preload to occur just before its execution or before Op6's preload, whichever is earlier, to preserve data dependency. Scheduling the preceding operator (Op4) will depend on Op5's preload time, which is estimated as the maximum of (1) the HBM access time from a roofline model [60] and (2) the interconnect transfer time from the cost model in §4.3.

Our algorithm has O(KN) complexity as we iterate through N operators with up to K preload numbers per operator. The algorithm provably finds the end-to-end plan with the shortest total time, assuming it can obtain the optimal execution time for each preload number. Lemma 4.1 and Theorem 4.2 formalize the algorithm.

<span id="page-5-3"></span>Lemma 4.1 (Base case). Given a model with N operators, for each operator i, let  $T^i_{s-pre}$  and  $T^i_{e-pre}$  be the start and end time of operator i's preload. Let  $T^i_{s-exe}$  and  $T^i_{e-exe}$  be the start and end time of operator i's execution. Let  $T_{start} = T^1_{s-pre}$  and  $T_{end} = T^N_{e-exe}$  be the start and end time of the model execution. Then, for operator N, preload number 0 minimizes  $T_{end} - T^N_{s-exe}$ .

PROOF. Since operator N is the last operator, the only possible preload number is 0.

<span id="page-5-4"></span>Theorem 4.2 (Inductive step). Let  $1 \le i < N$ . Suppose we have minimized  $T_{end} - T_{s-exe}^{i+1}$ . Then, there exists a preload number p whose  $T_{s-exe}^i$  minimizes  $T_{end} - T_{s-exe}^i$ , or maximizes  $T_{s-exe}^i$ . Specifically, we have  $T_{e-exe}^i = \min(T_{s-exe}^{i+1}, T_{s-pre}^{i+p+1})$ , and  $T_{s-exe}^i = T_{e-exe}^i - L_{exe}^i$  where  $L_{exe}^i$  is the execution time of operator i derived by the cost-aware memory allocation algorithm in §4.3.

PROOF. First, to prove  $T_{e-exe}^i = \min(T_{s-exe}^{i+1}, T_{s-pre}^{i+p+1})$  for any preload number p, we have (1) Opi must finish execution before  $\operatorname{Op}(i+1)$  starts execution, e.g.,  $T_{e-exe}^i \leq T_{s-exe}^{i+1}$ ; and (2) Opi's execution can be overlapped with the preload of the next p operators, which implies  $\operatorname{Opi}$ 's execution must finish before the preload of  $\operatorname{Op}(i+p+1)$ , e.g.,  $T_{e-exe}^i \leq T_{s-pre}^{i+p+1}$ . Next, we prove the existence of  $T_{s-exe}^i$  that minimizes  $T_{end} - T_{s-exe}^i$ . Suppose by contradiction that  $T_{end} - T_{s-exe}^{i+1}$  is minimized but there is no  $T_{s-exe}^i$  that minimizes  $T_{end} - T_{s-exe}^i$ . Since our inductive step explored all  $T_{s-exe}^i$  values by enumerating all possible preload numbers, the only possible case is that we must explore more preload numbers to find the global  $\max(T_{s-exe}^i)$ , or the global  $\max(T_{s-exe}^i)$  is greater than  $T_{s-exe}^{i+1}$ , which means  $T_{s-exe}^{i+1}$  can be larger. This is a contradiction since  $T_{end} - T_{s-exe}^{i+1}$  is already minimized, e.g.,  $T_{s-exe}^{i+1}$  is already maximized.

### <span id="page-5-1"></span>4.3 Cost-Aware On-chip Memory Allocation

In §4.2, when scheduling an operator, ELK needs to optimize the performance for each preload number. Given the currently executing operator and a set of operators to be preloaded, ELK defines a two-level tradeoff space between the execution/communication time and the memory consumption.

First, there are two types of *intra-operator tradeoffs*: (1) For the currently executing operator, ELK trades memory space for execution time (§3.1). (2) For each preloaded operator, ELK trades

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Figure 11: Tradeoff between time overhead & memory usage.

<span id="page-6-1"></span>![](_page_6_Figure_4.jpeg)

Figure 12: Cost model accuracy of different operators and inter-core transfer, for different tile shapes. Each point is the measured vs. predicted per-core execution or transfer time.

memory space for this operator's inter-core data exchange overhead (§3.3). Second, there is an inter-operator tradeoff: as operators have different memory-time tradeoffs, we allocate more memory to operators that benefit more from a larger execution/preload space.

Elk explores the two-level space in two stages. First, for each operator, Elk finds all Pareto-optimal tradeoff plans between time and memory. Second, Elk jointly determines the execution/preload space sizes of all operators based on the Pareto-optimal plans and the total on-chip memory capacity.

**Intra-operator tradeoff for on-chip execution** (Tradeoff 1 in

Figure 11). For the currently executing operator, there are many

partition plans to partition its computation into tiles, each runs on

one core (Figure 3). Elk integrates existing compiler techniques to enumerate all partition plans of an operator given its operator type and tensor shapes [7, 16, 34, 46, 70, 74]. These techniques represent each plan as a list of integers (see examples in §5) and check if a plan is compatible with the target hardware (e.g., not using more cores than available, not overflowing the SRAM). For each plan, Elk estimates its execution time using a cost model and its execution space using the tile size. ELK examines all plans to find the ones on the Pareto-optimal curve, where each plan either runs faster than any other plans that use the same or less memory, or uses less memory than any others with the same or less execution time. Cost model for execution time. As DL workloads have predictable execution patterns [4, 16, 34, 38, 61, 74], ELK uses an accurate cost model to quickly estimate the performance of per-core execution and inter-core transfer. For each operator type (e.g., MatMul), we randomly generate tiles with varied shapes, and run each tile using one core on the target device. Then, we fit a linear tree model [10] using the tile shapes as inputs and the profiled execution times as outputs. For inter-core transfer, we fit a model for each network link using transfer volumes as inputs and transfer times as outputs. For each partition plan, ELK determines the tile-to-core mapping and orchestrates the inter-core transfer (e.g., the source/destination cores and intermediate hops of each transfer, see §5). Elk uses the per-link cost model and the communication pattern to estimate the total transfer time. Figure 12 shows that ELK can accurately predict

the execution and transfer times of an IPU chip. Elk can use different cost models [4, 34, 38, 74] for different hardware platforms.

MICRO '25, October 18-22, 2025, Seoul, Republic of Korea

**Intra-operator tradeoff for preloading** (Tradeoff 2 and 3 in Figure 11). For each preloaded operator, its partition plan is already decided in a previous step of the inductive operator scheduling (§4.2). This execute-state plan is chosen for execution speed, which may use more memory space. As the operator is not currently executing, ELK assigns a memory-efficient preload-state plan. To start execution, a data distribution phase transforms the operator from preload- to execute-state by distributing the required data via the interconnect (e.g., Figure 3 (c)). It saves this operator's preload space at the cost of extra inter-core data exchange overhead, compared to broadcasting the required data at preload time following the execute-state plan (e.g., Figure 3 (b)).

Each execute-state plan may have many preload-state plans, by configuring how much data is broadcasted on preload. On preload, if 4 cores share a data piece, we can evenly split it into 1, 2, or 4 chunks, and broadcast each chunk to 4, 2, or 1 cores. Each core receives 1,  $\frac{1}{2}$ , or  $\frac{1}{4}$  of the data on preload (this decides preload space size), and fetches the rest 0,  $\frac{1}{2}$ , or  $\frac{3}{4}$  on data distribution. ELK finds the Pareto-optimal preload-state plans of each preloaded operator, by estimating their preload space sizes and data distribution times. Inter-operator tradeoff. With limited on-chip memory, Elk jointly trades off memory allocation among the executing and preloaded operators. It minimizes the total time, which is determined by (1) execution times, (2) data-distribution times, (3) interconnect contention overhead due to overlapped preload and execution, and (4) memory access contention overhead between local SRAM accesses and inter-core accesses<sup>2</sup>. To estimate the contention overhead on each interconnect link, Elk divides total traffic by link bandwidth.

As enumerating all possible plan combinations is impractical (e.g.,  $O(P^K)$  combinations for K operators each with P plans), ELK uses a heuristic based on each operator's memory-cost efficiency. ELK starts with each operator's fastest plan as the currently selected plan. This combination of plans requires the most execution/preload space, so the total space requirement may exceed the memory capacity. Elk then iteratively searches for the best combination of plans whose total memory requirement can fit into the on-chip memory, at the cost of slightly increasing the total execution time.

For each search step, ELK examines the next plan with a smaller memory footprint along the Pareto-optimal curve for each operator. ELK selects the most "cost-effective" operator whose next plan has the largest ratio  $\Delta = \frac{\text{reduced space size}}{\text{increased time}}$  compared to the currently seincreased time lected plan. For example, in Figure 11, 0p5 is the executing operator. Op6 and Op7 are preloaded operators. ELK updates the current plan for Op7 and proceed to the next search step. Elk stops when the total memory requirement does not exceed the available capacity.

In the worst case, Elk needs to examine all Pareto-optimal plans for all operators. Hence, it has O(PK)-time complexity for K operators to fit on-chip and P plans per operator. Combined with §4.2, the complexity is  $O(PK^2N)$  for N operators (K is also the number of possible preload numbers of each operator).

<span id="page-6-2"></span> $<sup>^2\</sup>mathrm{For}$  some ICCA chips where local SRAM accesses are blocked by inter-core accesses (e.g., IPU), we estimate access contention overhead using the inter-core access time.

<span id="page-7-2"></span>![](_page_7_Figure_2.jpeg)

Figure 13: Reorder preloads to allow larger execution space.

<span id="page-7-3"></span>![](_page_7_Figure_4.jpeg)

Figure 14: The generation of candidate preload orders.

#### <span id="page-7-0"></span>4.4 Preload Order Permutation

ELK allows operators to be preloaded in a different order than the execution order. This has two benefits.

First, reordering helps mitigate interconnect contention. As the interconnect traffic pressure fluctuates (see §3.3), the reordering opportunistically reschedules heavy preload traffic to avoid "rush hours" on the interconnect.

Second, by reordering the preload of some large operators to a later time, we can save more space for execution by reducing the lifespans of their large memory footprints in the on-chip SRAM. For instance, in Figure 13, 0p6 requires more preload space than 0p7. If we preload in order, the execution space is 1/2 of the total on-chip memory at time  $t_1$ . If we reorder their preloads, the execution space is 5/6 of the total memory at  $t_1$ .

As large models consist of thousands of operators, it is unrealistic to test all preload orders (there are N! orders given N operators). However, most of the orders are invalid, as they overflow the onchip memory. If we delay an operator's preload to a late time, its execution will also be delayed. As operators are executed in order, future operators also cannot execute until this delayed operator completes execution, even if they have already been preloaded into the on-chip memory. Since there is no free space to preload more operators, and the preloaded operators cannot free their space until executed, the on-chip memory will overflow. In practice, Elk only needs to explore a reasonable amount of valid preload orders.

Generate valid preload orders. ELK enumerates all valid preload orders by scanning through all operators following the inductive operator scheduling order (§4.2) and incrementally picking the next operator to preload in each step.

Figure 14 shows an example of a DL model with 9 operators. In the first step (Figure 14 (a)), ELK picks the last operator to preload. We can only fit two operators into the on-chip memory, so either Op8 or Op9 can be the last operator to preload. This generates two branches for the next step.

In the second step (Figure 14 (b)), ELK iterates through both branches and picks the second-to-last operator to preload for each branch. In the upper branch, Op8 is already preloaded. If we choose Op6 as the second-to-last operator to preload, both Op7 and Op9 need

<span id="page-7-4"></span>![](_page_7_Figure_14.jpeg)

Figure 15: The abstracted device programming model of Elk.

to be preloaded before Op6. This implies all three operators, Op6, Op7, and Op9, must stay on-chip together because their memory cannot be freed up until Op6 is executed. In our example, as the memory cannot fit all three operators, we can only choose Op7 or Op9. Similarly, in the lower branch, we can only choose Op7 or Op8, and we do not consider the space requirement of Op9 because it can be preloaded after we free up Op7 and Op8's memory.

ELK repeats the above process and generates a suffix tree of all valid preload orders. Given N operators in a model, if we can fit at most K operators on-chip, our search tree has  $O(K^N)$  leaves, compared to the original O(N!) search space.

**Prune the valid order search space.** Given the unique characteristics of LLMs, Elk can further prune the candidate orders while still being able to find a near-optimal order.

First, many operators, such as softmax, preload little or no data from HBM, as they perform in-place computations on the intermediate output. For example, OPT-30B [66] has 2,269 operators, but 289 of them contribute 99.8% HBM load volume. Since the remaining 1,980 operators preload little or no data from HBM, reordering their preloads will have negligible performance benefits. *Thus, ELK focuses on reordering only the preloads of operators with high HBM load volume.* In practice, we only reorder the preload of operators whose tensor sizes are above average (e.g., for LLM decoding, the average size is model size divided by operator count). For smaller operators that often preload little or no data from HBM, we preload them in order (i.e., 0p i will be the i'th preloaded operator).

Second, an LLM consists of identical transformer layers. *ELK only reorders the preloads within one layer*, and applies the same order to identical layers. With these rules, ELK prunes the search space from  $O(K^N)$  to  $O(C^H)$ . H is the number of HBM-heavy operators per layer, so H << N ( $H \le 6$  in most transformer models). C is the maximum number of HBM-heavy operators per layer that can fit on-chip, so C << K and  $C \le H$ .

For each generated preload order, Elk invokes the operator scheduling pass in §4.2, forming a  $O(C^HPK^2N)$  search space. Elk picks the best end-to-end plan among all preload orders.

#### <span id="page-7-1"></span>4.5 Mapping to Hardware

The execution plan generated by ELK specifies all operator's preload order and each operator's partition plans. ELK maps the plan to an abstracted programming model, which can be applied to generic ICCA chips with off-chip memory. As shown in Figure 15, ELK abstracts two key device functions that are generated during compilation. (1) preload\_async(op=i) commands all cores to request Opi's data from HBM based on the preload-state partition plan. (2) execute(op=i) runs Opi on all cores based on the execute-state plan.

For the example in Figure 15, preload\_async(op=2) requests HBM controllers to deliver Op2's data to each core's SRAM, following the *preload-state plan* (see §4.3). When the data delivery completes, the controllers will append a done\_preload\_op\_2 tag to the end of the delivered data in each core's SRAM.

Then, execute(op=2) will run in 3 steps when it is called. First, it waits until preload\_async(op=2) completes, by verifying the value of done\_preload\_op\_2 tag in each core's SRAM. Second, each core calls distribute\_data to copy shared data from peers, transforming from preload-state to execute-state plan. Third, each core calls local\_execute to compute a tile following the execute-state plan.

To summarize, the hardware enforces three rules for preload\_async and execute calls, using one-way synchronization. (1) An invocation of execute blocks all future preload\_asyncs and executes, until the invoked execute finishes. This enforces the operator execution order and specifies which preload\_asyncs can overlap with an execute. (2) To enforce the preload order, all preload\_asyncs execute sequentially. (3) preload\_async(op=i) does not block any execute except execute(op=i), as an operator must preload before execution.

#### <span id="page-8-0"></span>5 Implementation details

ELK compiler framework. We implement ELK as a generic compiler framework that can support different ICCA chip implementations. Most ELK components are hardware-agnostic.

(1) ELK frontend takes a DNN model from ML frameworks like Py-Torch [48] as input. The model is first converted into an ONNX graph [1], which represents all operators in the model as a directed acyclic graph. ELK obtains layer information, operator definitions [56], and tensor shapes from the ONNX graph. ELK can support most DL models representable as an ONNX graph.

(2) The execution plan generation (§4.2–§4.4) takes operator partition plans as inputs. ELK supports single-operator partition plans generated by compilers that use different parallel execution models [34, 40, 74]. In our experiments, we use the plans generated with the recent compute-shift execution model proposed in [34], as it represents the state of the art for operator execution on ICCA chips.

For each operator, we enumerate all possible partition plans by representing each plan as a list of integers. For instance, <90,9> evenly slices each dimension of a 2-dimension operator into 90 and 9 parts, forming 90×9=810 tiles. For each plan, Elk decides the mapping of each tile to each core. It uses different mapping strategies for different network topologies. Elk currently targets ICCA chips with two popular network topologies: all-to-all network and mesh network. For chip with all-to-all network, Elk sequentially maps all tiles, as core locations do not impact the inter-core data transfer cost. For chip with *N*-dimensional mesh network, Elk chooses from plans that partition an operator along at most *N* dimensions, so it can map each partitioned dimension to a mesh dimension. Then, Elk uses dimension-order routing [22, 28] to maximize the all-reduce bandwidth. Besides the two topologies that are used by most ICCA chips today, Elk is scalable to support other topologies.

Based on the partition, mapping, and routing information, Elk's cost model estimates each plan's compute, memory, and interconnect costs. Using the costs of all plans for all operators, Elk runs the scheduling, allocation, and reordering procedures in 4.2-4.4 to trade-off among performance factors and compose an optimized

<span id="page-8-1"></span>Table 2: DL models used in our evaluation.  $\underline{C}$ : max number of HBM-heavy operators per layer that fit on-chip.  $\underline{H}$ : number of HBM-heavy operators per layer.  $\underline{P}$ : max number of plans per operator.  $\underline{K}$ : max number of operators that fit on-chip.  $\underline{N}$ : total number of operators. We calculate C and K using the on-chip memory capacity of real IPU-POD4 as an example.

| Name            | Description                                   | C | H | P   | K   | N    |
|-----------------|-----------------------------------------------|---|---|-----|-----|------|
| Llama2-13B [53] | Large language model (LLM)                    | 6 | 6 | 66  | 88  | 1928 |
| Gemma2-27B [51] | LLM with Grouped-Query<br>Attention (GQA) [5] | 6 | 6 | 206 | 128 | 2216 |
| OPT-30B [66]    | LLM                                           | 5 | 6 | 58  | 46  | 2269 |
| Llama2-70B [53] | LLM with GQA [5]                              | 6 | 6 | 168 | 86  | 3808 |
| DiT-XL [44]     | Diffusion transformer                         | 4 | 4 | 123 | 136 | 1521 |

<span id="page-8-2"></span>![](_page_8_Figure_13.jpeg)

Figure 16: ELK compile time for varied model/batch sizes.

end-to-end execution plan. The execution plan generation in Elk is implemented in 2.5K lines of code (LoC) of Python.

(3) The code generation in ELK generates the kernel code for computing each tile and the inter-core data transfer operations, based on the target hardware and selected partition plans. For compute, ELK uses code templates from vendor-provided libraries [21]. For inter-core transfer, ELK reserves an 8KB buffer in each core's 624KB SRAM to buffer incoming data, which improves the transfer granularity and performance. The code generation in ELK is developed in 4K LoC of Python and C++.

Scalability of ELK. ELK prunes the search space of a large model to  $O(C^H PK^2 N)$  complexity. We list the complexity factors for different models in Table 2, all using batch size 32 and sequence length 2048. As model size grows, N scales sub-linearly, while C, H, P, and K change independently. Thus, ELK's search space size scales sub-linearly with the DL model size.

ELK can generate an end-to-end plan for an LLM on ICCA chip like IPU-POD4 in 5 minutes using a 32-core AMD EPYC 7543 CPU (see Figure 16). On each CPU thread, ELK can test a candidate preload order in seconds. As ELK prunes the number of preload orders (e.g., 720 for Llama2-70B), the compilation finishes in minutes. **Emulation framework.** As the ICCA chip we can access (IPU-POD4) does not have HBM, we build an emulation framework using a real IPU-POD4. The pod has 4 IPU MK2 chips with a total of 5,888 cores, 3.5GB on-chip memory, and 640GB/s inter-chip bandwidth. By default, we use model parallelism [40] across the four chips, since it incurs little inter-chip communication overhead, because the activation tensor to be reduced across chips is usually small. To obtain HBM access latencies, our framework uses an acknowledged memory simulator [32]. We evenly slice each tensor across all HBM modules to balance traffic, and sequentially place tensors in HBM. The HBM can easily saturate its bandwidth when ELK sequentially reads data at tensor granularity (tensor sizes range from 43 to 219 MB). Based on the tensor placement, we generate memory traces of all tensors to obtain HBM latencies from the memory simulator.

The framework then executes the end-to-end plan generated by ELK on IPU-POD4, where it computes each tile on each core

<span id="page-9-2"></span>![](_page_9_Figure_2.jpeg)

Figure 17: The per-token serving latency of various models and batch sizes on 4 ICCA chips with 16TB/s HBM.

and moves shared data between cores based on the partition plans selected by Elk. To emulate HBM accesses, one core acts as an HBM controller to broadcast "HBM data" to other cores and apply HBM latencies by delaying each broadcast. To synchronize execution with preload, the "controller" core also appends done\_preload\_op\_i tags (i.e., arbitrary constants, see §4.5) to the end of broadcasted data, allowing receiver cores to check whether a preload has finished.

As Elk homogeneously partitions each tensor to cores following common operator tiling strategies [12, 18, 74], all cores receive tensor tiles of the same size during each preload. Thus, we emulate the interconnect traffic caused by preload by using one "controller" core to broadcast data to all cores. The broadcast saturates the interconnect and the inbound links on receiver cores, emulating the contention between inter-core data sharing and operator preload. Simulation framework. To conduct sensitivity analysis and design space exploration, we build an event-driven simulator for ICCA chips, which simulates all cores, network links, and off-chip HBM accesses. For each core, we simulate a local SRAM, a compute pipeline, and a network agent that sends/receives data to/from other cores. For each network link, we model its latency and bandwidth [3, 6]. Based on the execution plan generated by ELK, we derive the simulation events at tile granularity, including computing a tile on a core, transferring a tile over a specific network link, and fetching a tile from the off-chip HBM. Each core/link maintains its event queue to execute its events sequentially. For an all-to-all network, we model HBM controllers as dedicated nodes in the network (see §2.1). For a mesh network, we attach HBM controllers to the edges of the mesh grid. To simulate a multi-chip system, we track the in-flight inter-chip transfer events and cap their total bandwidth. We also use our real IPU-based emulator to validate our simulator.

#### 6 Evaluation

With our emulation framework, we show that on average, ELK achieves (1) **94.84**% of the performance of an ideal roofline design (§6.2), (2) **89.52**% inter-core interconnect bandwidth utilization, and almost ideal HBM and FLOPS utilization relative to the roofline (§6.3). With our simulator, (3) we demonstrate ELK enables design space exploration for scaling compute, communication, and off-chip memory accesses for ICCA chips. We report our insights in §6.4.

#### <span id="page-9-0"></span>6.1 Experimental Setup

**Workloads.** We examine the inference decoding phase of differently sized LLMs (see Table 2), using varied batch sizes and sequence lengths. We also test a stable diffusion model (see Figure 23) and LLM training (see Figure 24).

**Emulator setup.** We emulate 4 HBM3E modules [39] per ICCA chip, following a state-of-the-art (SOTA) GPU [41]. With 4 ICCA chips, we have 16TB/s total HBM bandwidth.

Simulator setup. We simulate 4 chips and 16TB/s HBM bandwidth by default. The configuration (compute and local SRAM) of each core and the latency/bandwidth of each network link are the same as the emulator setup by default, and packets that share one NoC link are scheduled sequentially. We simulate both all-to-all and 2D mesh networks. For all-to-all network, we follow the IPU-POD4 architecture [29]. For mesh network, each core can simultaneously communicate with all its neighbors (up to 4 in a 2D mesh) [7].

**Baselines.** As there is no open-sourced compiler for ICCA chip with HBM, we conduct an ablation study by creating two baselines that extend SOTA compilers for ICCA chips [34] to support HBM, and an ELK variant that disables preload reordering (§4.4). We also compare ELK to an ideal roofline. In brief, we compare these designs:

- Basic: The design follows existing DL compilers to optimize on-chip execution. It maximizes the execution space and uses the remaining space to preload the next operator.
- Static: Following the SOTA compiler T10 [34] developed for ICCA chips, we extend it to jointly optimize on-chip execution and off-chip loading. First, it follows SambaNova [46] to preload multiple operators in advance, by reserving a preload space. Then, it find the fastest execution plan for each operator given the remaining execution space size. We further improve the design by finding the best static preload and execution space sizes for the entire DL model (the sizes will not change throughout the model execution). When preloading a set of operators, all operators use either the preload-state plan with the largest memory footprint or the plan with the smallest footprint, whichever is faster.
- ELK-Dynamic (ELK-Dyn): A partial design of ELK, which optimizes the preload-execution overlap (§4.2) and on-chip memory allocation (§4.3). This design represents ELK's performance without preload order permutation (§4.4).
- ELK-Full: The full ELK design, which enables all optimizations, including the preload order permutation (§4.4).
- Ideal: The theoretical roofline performance, where each of preload
  and execution has its own interconnect (i.e., no interconnect contention) and full-sized on-chip memory (i.e., no memory space
  contention). Each operator uses the minimum preload space to\nemulate the benefits of maximum preload number, and the data
  distribution phase has zero latency to emulate the benefits of
  maximum preload space per operator.

