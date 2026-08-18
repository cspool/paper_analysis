# Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

Jingkui Yang\*‡#, Fangxin Liu†‡#⊠, Xin Ju\*‡, Ning Yang†, Chenyang Guan†, Junjie Wang†,
Zongwu Wang†‡, Mei Wen\*, Jian Liu§, Li Jiang†‡⊠, and Haibing Guan†

\*National University of Defense Technology, Changsha, China

†Shanghai Jiao Tong University, Shanghai, China

‡Shanghai Qi Zhi Institute, Shanghai, China

§Beijing University of Aeronautics and Astronautics, Beijing, China

⊠: yangjingkui2001@nudt.edu.cn, liufangxin@sjtu.edu.cn, ljiang\_cs@sjtu.edu.cn

Abstract—Sparse tensor computation is a critical primitive across many domains. The highly irregular structure of sparse matrices limits the performance and efficiency of sparse tensor computation on conventional platforms, motivating extensive efforts on specialized hardware accelerators. However, existing accelerators typically rely on rigid execution dataflows such as inner-product, outer-product, or row-based schemes. Each dataflow is optimized for a particular sparsity pattern and fails to deliver robust performance across the wide diversity of real workloads.

Although sparsity reduces computation and memory cost, effectively exploiting it on hardware remains challenging because sparsity patterns vary widely and often change at runtime. Recent accelerators attempt to balance efficiency and generality by introducing architectural flexibility, but fixed-dataflow designs degrade under pattern shifts, while flexible designs support multiple modes only at the cost of higher complexity and static configuration. To address these limitations, we propose Harmonia, a hierarchical scheduling approach that allows a sparse accelerator to efficiently adapt to different sparsity patterns. Harmonia first uses a lightweight offline model to derive a nearoptimal initial tiling strategy and dataflow mapping. Both tile shape and dataflow mode are configurable to match diverse sparsity characteristics. At runtime, Harmonia detects the current sparsity pattern and dynamically selects the most effective dataflow mode based on tile-level observations. The underlying hardware supports this adaptive execution with fast, lowcost reconfiguration and load balancing. Extensive evaluations demonstrate that Harmonia delivers an average of  $1.75 \times$  higher performance and  $2.47\times$  better energy efficiency compared to state-of-the-art accelerators, while maintaining robust and stable throughput across highly variable runtime sparsity patterns.

Index Terms—Sparse Matrix Multiplication, Dynamic Scheduling, Dataflow Architecture, Runtime Adaptation

## I. INTRODUCTION

Sparse tensor computation has emerged as a foundational primitive across diverse workloads, powering deep neural networks [1]–[3], scientific simulations [4], [5], and large-scale graph analytics [6], [7]. To exploit sparsity's potential

![](_page_0_Figure_11.jpeg)

Fig. 1. Design-space trade-off between flexibility and hardware efficiency in sparse accelerators. Fixed-dataflow accelerators achieve high efficiency for specific sparsity patterns by tailoring PE micro-architecture, but sacrifice flexibility. Adaptive-dataflow designs improve flexibility by decoupling multipliers and adders, yet incur low efficiency due to complex data routing. The versatile PE-row architecture offers a balanced point by localizing flexibility within PE rows while preserving overall structural regularity.

for reducing computation and memory traffic, numerous accelerators have been proposed, ranging from fixed-dataflow architectures [8]–[13] optimized for specific sparsity patterns, to versatile architectures [14]–[17] capable of supporting multiple sparse execution modes. Despite this progress, a fundamental tension persists between efficiency and flexibility, as illustrated in Fig.1: specialized designs achieve high utilization within narrow operating regimes but fail to generalize, while more flexible designs broaden applicability at the cost of area, energy, and control complexity. This trade-off limits the scalability of sparse acceleration and motivates architectures that are both flexible and efficient.

Fixed-dataflow accelerators (Fig.1 (a)), such as SIGMA [8] and HighLight [9], bind PE microarchitectures tightly to assumed sparsity patterns. Once the pattern or density deviates from design-time expectations, utilization collapses; for example, SIGMA's Flex-DPE array drops below 10% utilization under high sparsity because most MAC units process zeros. On the other hand, flexible accelerators (Fig.1 (c)), such as Flexagon [14], expand support for multiple dataflows through reconfigurable routing, buffering, and control. However, this

<sup>#</sup> Both authors contributed equally to this work.

<sup>™</sup> Corresponding authors: liufangxin@sjtu.edu.cn, ljiang\_cs@sjtu.edu.cn. Work done during an internship at Shanghai Jiao Tong University & Shanghai Qi Zhi Institute.

flexibility is achieved through heavy interconnects and complex data steering; Flexagon dedicates more than 75% of its area to non-compute structures, leading to substantial efficiency loss.

Versatile architectures like Trapezoid [16] and VersaAccel [18] represent a promising middle ground (Fig. 1 (b)), retaining a homogeneous PE array while enabling lightweight reconfiguration of data distribution and partial-sum handling. These designs achieve near-constant efficiency across a broad sparsity range, offering a strong hardware foundation for universal sparse computation. However, their flexibility remains hardware-centric: while they support multiple dataflows, they lack a system mechanism to select the appropriate dataflow at runtime. As sparsity varies across tiles, the system experiences load imbalance, uncoordinated memory traffic, and degraded reuse. These issues reveal a key gap between hardware capability and system adaptivity, highlighting the need for a runtime scheduling layer that actively guides versatile hardware.

Unfortunately, existing scheduling methods still fail to fill this gap, as summarized in TABLE I. Most approaches optimize only one layer of the sparse execution hierarchy, without offering end-to-end coordination: (1) Intra-tile dataflow selection (e.g., Misam [19]) adapts to local reuse and sparsity variation but ignores tile-to-tile dependencies and shared memory pressure. (2) Inter-tile data orchestration (e.g., HYTE [20]) optimizes tiling and data movement at the global level but assumes fixed local dataflows, leaving no room for runtime adaptation. (3) Cross-layer analytical modeling (e.g., Vesper [17]) links global and local behavior through static operation-intensity analysis, but its uniform-sparsity assumption leads to large prediction errors for irregular workloads. As a result, existing work is either locally adaptive or globally fixed, making it unable to turn flexible hardware into truly adaptive system-level behavior.

This paper introduces Harmonia, a hierarchical scheduling framework that coordinates hardware flexibility with system-level adaptivity. The design recognizes that hardware flexibility alone is insufficient for sparse acceleration without a coupled static-dynamic scheduling stack to guide resources at runtime. By introducing a software-hardware scheduling interface that leaves the compute datapath unchanged, Harmonia transforms a homogeneous accelerator into a logically heterogeneous architecture through integrated offline modeling and runtime feedback.

Our main contributions are summarized as follows:

- **Hierarchical Coupling Analysis.** We characterize how dataflow, tiling, and traversal decisions interact across layers, and we build a statistical model that captures these dependencies to guide joint optimization.
- Unified Scheduling Framework. We develop a static-dynamic scheduling framework that combines global analytical planning with lightweight runtime profiling, enabling tile-level refinement of dataflow and data distribution.
- Adaptive Runtime Substrate. We design a reconfigurable dataflow substrate that supports fast mode switch-

![](_page_1_Figure_8.jpeg)

Fig. 2. Comparison of representative sparse dataflows. The upper diagrams illustrate nonzero traversal patterns for Inner Product (InP), Rowbased (Row), and Outer Product (OutP) dataflows. The lower table qualitatively compares reuse efficiency and control complexity. "Index matching cost" denotes the control overhead for nonzero alignment, and "Psum merge overhead" represents reduction effort for partial sums.

ing and low-cost metadata updates, allowing homogeneous hardware to act as a logically heterogeneous execution engine.

We evaluate Harmonia across DNN inference, scientific computing, and graph workloads. Harmonia delivers  $1.75\times$  higher performance and  $2.47\times$  better energy efficiency on average, while maintaining stable performance under runtime sparsity changes.

#### II. BACKGROUND

Sparse matrix multiplication computes  $C = A \times B$ , where  $A \in \mathbb{R}^{M \times K}$  and  $B \in \mathbb{R}^{K \times N}$ . Unlike dense GEMM, sparse matrix multiplication introduces irregular nonzero traversal and index alignment, making the choice of dataflow (i.e., the order of the three nested loops) crucial to both data reuse and control complexity [17], [21].

#### A. Versatile Sparse Accelerators

As shown in Fig. 2, sparse matrix multiplication is commonly executed using one of three dataflows: (1) Inner Product (InP) dataflow computes each output element  $C_{m,n}$  via a row–column dot product. It provides strong output reuse but weak input reuse since each column of B must be repeatedly fetched. (2) Outer Product (OutP) dataflow generates a partial-sum matrix for each k by multiplying a column of A with a row of B. It maximizes input reuse but requires substantial psum merging. (3) Row-based (Row) dataflow multiplies a nonzero  $A_{m,k}$  with the full row  $B_{k,:}$ , offering moderate reuse with reduced merging overhead.

The complementary characteristics of sparse dataflows reveal a key insight: no single mapping achieves high reuse and low overhead simultaneously. This motivates versatile sparse

TABLE I

COMPARISON OF REPRESENTATIVE SPARSE SCHEDULING APPROACHES ACROSS INTRA-TILE AND INTER-TILE OPTIMIZATION DIMENSIONS.

HARMONIA UNIQUELY INTEGRATES BOTH STATIC AND DYNAMIC COORDINATION TO ACHIEVE FULL-LAYER ADAPTIVITY.

| Category                | Representative  | Intra-tile (Dataflow) |              |              | Inter-tile (Tiling) |              |              | Runtime      | Limitation                        |
|-------------------------|-----------------|-----------------------|--------------|--------------|---------------------|--------------|--------------|--------------|-----------------------------------|
|                         |                 |                       |              | Adaptivity   |                     |              |              |              |                                   |
| Versatile Accelerator   | Trapezoid [16]  | <b>√</b>              | <b>√</b>     | X            | X                   | Х            | X            | Х            | Hardware-oriented only            |
| Dataflow Selection      | Misam [19]      | ✓                     | $\checkmark$ | ✓            | X                   | X            | X            | $\checkmark$ | Local optimization only           |
| Data Orchestration      | HYTE [20]       | X                     | ✓            | X            | $\checkmark$        | $\checkmark$ | $\checkmark$ | $\checkmark$ | Lacks intra-tile feedback         |
| Analytical Modeling     | Vesper [17]     | ✓                     | $\checkmark$ | ✓            | $\checkmark$        | X            | $\checkmark$ | ×            | Static analytical estimation      |
| Hierarchical Scheduling | Harmonia (Ours) | $\checkmark$          | $\checkmark$ | $\checkmark$ | $\checkmark$        | $\checkmark$ | $\checkmark$ | $\checkmark$ | Dynamic, cross-layer coordination |

accelerators, which support multiple dataflow templates within a unified hardware substrate.

Previous efforts such as Flexagon [14] introduced multitemplate switching across the array, enabling each workload to select a dataflow offline. Building on this direction, Trapezoid [16] proposed a finer-grained approach that reconfigures only two specific components rather than the entire array. It maintains a homogeneous PE structure and reconfigures the Distribution Network (DN) [22] for input broadcasting and the Merge-Reduction Network (MRN) [14] for partial-sum aggregation. By adjusting these routing paths, Trapezoid transitions among InP- and Row-like behaviors while maintaining near-constant efficiency across a wide sparsity range, from dense DNN layers to highly sparse SpGEMM.

This hardware-level flexibility allows Trapezoid to implement multiple sparse dataflows on the same PE array. However, this flexibility exists only at the structural level and is not coupled with system-level adaptivity. As a result, Trapezoid can switch among dataflow templates, but it cannot adapt these choices to runtime sparsity changes, causing its potential efficiency to remain underutilized.

#### B. Sparse Scheduling Strategies

The performance of sparse acceleration depends not only on hardware flexibility but also on how dataflows and tiles are scheduled at runtime [23], [24]. Prior work explores three scheduling directions (intra-tile, inter-tile, and cross-layer), as summarized in TABLE I.

1) Intra-tile Dataflow Selection: With the emergence of flexible accelerators, researchers have investigated runtime adaptivity inside each tile. Systems such as **Misam** [19] and **Spada** [15] dynamically choose dataflows (e.g., InP, OutP, Row) based on the sparsity characteristics of the tile.

Misam formulates dataflow selection as a lightweight classification task: it uses decision-tree models [25] to predict an appropriate dataflow from simple matrix statistics and then reconfigures hardware logic accordingly. Spada takes a different approach by proposing window-adaptive dataflows. Its execution windows of size  $(\alpha \times \beta)$  can emulate InP-, OutP-, or Row-style reuse, enabling a trade-off between generality and efficiency.

However, these approaches remain purely local. Their decisions are made solely from tile-level sparsity patterns and do not coordinate across tiles. As a result, when sparsity

irregularity spans multiple tiles or layers, local decisions no longer align with global optimization, leading to suboptimal end-to-end utilization.

2) Inter-tile Data Orchestration: Since on-chip SRAM cannot hold entire matrices, sparse workloads are typically partitioned into tiles to improve data reuse and bandwidth efficiency [26]. Early approaches such as Tailors [27] adopt static tiling via offline overbooking, partitioning tiles based on average sparsity observations. Dynamic schemes such as DRT [24] and HARP [28] instead reconfigure tiles at runtime, but they suffer from high control overhead and limited scheduling flexibility. More recent work, HYTE [20], introduces a hybrid strategy that combines static analysis with lightweight runtime refinement. By profiling memory access patterns on the fly, HYTE adaptively adjusts tile boundaries and traversal order, achieving a balance between compile-time predictability and runtime responsiveness.

However, HYTE's adaptivity remains restricted to the intertile level. It assumes a fixed intra-tile dataflow and does not model its interaction with global tiling and reuse patterns. As a result, HYTE provides dynamic coordination across tiles but keeps tile-internal execution static, leaving finer-grained adaptation opportunities unexploited.

3) Cross-layer Analytical Modeling: To address the coupling between intra- and inter-tile scheduling, **Vesper** [17] introduces a unified analytical framework that models computation and memory behavior together. By evaluating operational intensity (OI) under various dataflow, tiling, and loop configurations, it searches for a global optimum through offline analysis. This cross-layer approach provides strong designtime interpretability and represents an important step toward hierarchical scheduling.

However, Vesper's static nature limits runtime adaptability. It relies on the often unrealistic assumption of uniform sparsity across tiles and provides no runtime feedback to correct deviations from its analytical predictions. As a result, although it establishes a theoretical link between intra- and inter-tile scheduling, it cannot dynamically close the optimization loop during execution.

In summary, existing scheduling strategies target different layers but remain isolated. The lack of a unified framework that combines static global planning with dynamic local adaptation remains the main obstacle to fully exploiting the

![](_page_3_Figure_0.jpeg)

Fig. 3. Motivation for hierarchical scheduling in Sparse Matrix-Sparse Matrix Multiplication (SpMSpM). (a)(b) Tile shape fundamentally changes reuse and buffer behavior, causing different dataflows to become optimal under different (K, N) dimensions. The tile shape  $(64 \times K \times N)$  varies while keeping the operation amount constant. (c)(d) Tile occupancy reshapes buffer pressure and psum behavior, shifting the optimal dataflow as tiles grow or shrink. We keep the tile shape fixed as a square and scale the tile size uniformly from  $1 \times$  to  $256 \times$  the PE array size. (e) Static analytical models fail to predict these effects: tile-level sparsity variation causes large deviations between estimated and actual latency/traffic. (f) Latency comparison of OutP and Row dataflows. The best dataflow shifts from OutP to Row as the sparsity pattern changes.

potential of versatile sparse accelerators.

#### III. CHALLENGE AND MOTIVATION

Prior scheduling strategies [15], [17], [19], [20], [24], [27]–[30] have made significant progress in accelerating sparse matrix multiplication. However, most approaches treat sparse scheduling as a set of isolated subproblems, such as selecting the dataflow within a tile or orchestrating tiles across memory. They often ignore the interdependence among hierarchical parameters, including tile shape, tile occupancy, and traversal order, which jointly determine performance.

As a result, even versatile accelerators with flexible dataflow and PE configurations cannot fully exploit their potential when faced with complex, non-uniform sparsity patterns. Coordinated scheduling across layers is required to bridge this gap. In this section, we examine the key challenges of sparse scheduling, which motivate the design of Harmonia.

#### A. Layer Interdependence

Existing approaches assume intra- and inter-tile scheduling can be optimized independently, but this assumption breaks down under sparse and irregular workloads. Inter-tile parameters, such as tile shape and occupancy, directly affect reuse opportunities in each tile, altering the optimal intra-tile dataflow.

We study a  $16 \times 16$  PE array with 16 KB local buffers and 1 MB on-chip SRAM using a cycle-accurate sparse dataflow simulator. The following results illustrate the strong impact of inter-tile configurations on intra-tile performance.

**Tile Shape Alters the Optimal Dataflow.** As shown in Fig. 3 (a) and (b), we utilize ResNet-0.1 (representing a ResNet-50 network pruned to an overall weight density of 0.1 [16]) and Llama-0.2 (a LLaMA-7B model pruned to a 0.2 density) to evaluate InP, OutP, and Row-based dataflows across various tile shapes while keeping the total amount of

computation constant. Detailed generation methodologies for all evaluated workloads are deferred to Section VI. Results show that the best dataflow depends heavily on tile dimensions: OutP dataflow performs best when K is small and N is large, but its performance degrades sharply with large K due to buffer overflows, while InP and Row-based dataflows benefit from higher K through increased reuse and PE-level parallelism. Some shapes (e.g.,  $64 \times 128 \times 64$ ) favor Row-based dataflow over OutP dataflow, demonstrating that intratile dataflow choice cannot be made in isolation.

Tile Occupancy Shifts SRAM Access Behavior. We evaluate the bcsstk10.mtx and email.mtx [31] workloads across tile sizes from  $16 \times 16$  ( $1 \times$  the PE array size) to  $256 \times 256$  ( $256 \times$  the PE array size) under InP, OutP, and Row-based dataflows (Fig.3 (c) and (d)). All dataflows initially benefit from higher occupancy, as small tiles repeatedly reload A, B, and C fragments, causing excessive SRAM traffic. OutP dataflow reaches minimum traffic earlier but worsens for very large tiles due to partial sums overflowing the local buffer. InP dataflow benefits from larger K, but large tiles eventually increase reload cost. Row-based dataflow is most tolerant to large tiles, processing one row at a time, though extremely large tiles amplify redundant B accesses. Overall, tile occupancy fundamentally reshapes dataflow efficiency, and no single dataflow consistently minimizes SRAM access.

**Insight 1:** Inter-tile parameters fundamentally reshape intratile reuse, thereby reversing dataflow performance. This observation implies that sparse scheduling cannot be decomposed into isolated layers and instead requires explicit modeling of hierarchical coupling.

#### B. Static Analytical Model Limitation

Prior cross-layer scheduling frameworks (e.g., Vesper [17]) attempt to derive a "globally optimal" schedule using static analytical models. These models assume uniform sparsity and

rely on global statistics, such as average sparsity or expected reuse, to estimate latency and memory traffic. In practice, however, sparse matrices exhibit significant intra- and intertile variation, causing static predictions to diverge sharply from actual execution.

Fig.3(e) illustrates this using the <code>bcsstk10.mtx</code> workload under a fixed  $64 \times 64$  tiling. The static model predicts OutP dataflow as optimal, estimating balanced latency and minimal SRAM access. However, actual execution reveals:

- Latency: OutP performs poorly compared to analytical predictions and is outperformed by Row-based dataflow (1.6× the latency of Row).
- **SRAM traffic**: InP achieves the true minimum, while the model incorrectly favors OutP.
- Prediction error: SRAM traffic for InP and Row is overestimated by 3.7× and 3.1×, respectively.

These discrepancies stem from uneven nonzero distributions: some tiles are dense enough to overflow local buffers under OutP, causing frequent SRAM spills, while sparse tiles under InP fail to fully utilize PE parallelism. Fig. 3(f) demonstrates that the best dataflow shifts from OutP to Row as the sparsity pattern changes in <code>email.mtx</code>, further validating this observation. By relying on global averages, static models overlook these tile-level variations, leading to suboptimal dataflow selection, buffer misalignment, and performance loss.

**Insight 2:** Sparse workloads exhibit highly uneven, tile-level sparsity variations. Efficient scheduling requires dynamic feedback rather than relying solely on static estimates.

## C. Toward Adaptive Runtime Substrate

While static models fail to capture tile-level irregularities, purely software-driven dynamic scheduling is also limited, as it lacks visibility into hardware signals such as SRAM pressure, psum spills, or fine-grained reuse. Without this information, runtime decisions remain coarse or conservative. We envision a lightweight runtime substrate with three key capabilities:

- Multi-level hardware feedback. Hardware exposes finegrained signals, including memory pressure, reuse counters, and psum merging latency, to capture execution dynamics beyond static models.
- **Fine-grained runtime reconfiguration.** Using compact metadata updates, the runtime can adjust tile traversal order, remap partitions, or switch intra-tile dataflows with minimal overhead.
- Cross-layer signal translation. The substrate interprets low-level counters into high-level scheduling hints, enabling globally coherent decisions without large SRAM overhead or costly profiling.

This substrate transforms hardware flexibility into dynamic system intelligence, enabling efficient adaptation to irregular sparse behavior.

**Insight 3:** Combining valuable aspects of prior scheduling approaches with lightweight profiling hardware enables systems to fully exploit architectural flexibility, delivering robust performance in the presence of varying sparsity profiles.

![](_page_4_Figure_14.jpeg)

Fig. 4. Overview of the Harmonia Hierarchical Scheduling Framework. Harmonia integrates (a) a *Static Analytical Layer* for offline tiling and dataflow planning, (b) a *Dynamic Profiling Layer* for runtime tuning and scheduling, and (c) an *Adaptive Hardware Substrate* for execution and feedback.

#### IV. SCHEDULING FRAMEWORK

#### A. Harmonia Overview

Observations in Section III reveal a *Sparsity-Dataflow Affinity*, where specific sparsity patterns favor distinct intra-tile dataflows and inter-tile configurations. Exploiting this affinity at runtime requires tight coupling across the software-hardware boundary to maintain low hardware overhead. Consequently, Harmonia employs a three-layer hierarchy comprising static modeling, online profiling, and adaptive hardware feedback. This structure progressively refines the execution plan by incorporating real-time architectural signals to compensate for initial static inaccuracies.

- (a) Static Analytical Layer. The static layer determines the initial configuration and allocates tiles to SRAM, aiming to maximize operation intensity (Ops/Byte). Using coarse-grained descriptors such as matrix shape, global sparsity, and datatype, it sets baseline parameters including tile shape, occupancy bounds, loop order, and SRAM allocation. While not fully optimal, it establishes a robust performance floor and a strong starting point for runtime refinement.
- **(b) Dynamic Profiling Layer.** This layer samples row/column densities and clustering patterns for tiles in SRAM to refine PE-row assignments, buffer usage, and dataflow selection. It adjusts tile shapes, resource allocation (DN/MRN routing), and execution order to maximize per-tile throughput, constrained by the global structure from the static layer.
- (c) Adaptive Hardware Substrate. This substrate powers the *Dynamic Tuning Layer* (detailed in Section IV-D), using low-cost hardware feedback (e.g., SRAM pressure counters, psum spill flags, merge-depth monitors, and PE stall indicators) to correct runtime deviations. Based on these signals, this tuning layer performs fine-grained adjustments, such as switching dataflow modes, resizing tiles, or reshaping merge configurations. A lightweight cost model ensures adjustments only occur when performance gains exceed reconfiguration overhead, preserving stability.

In summary, Harmonia implements a static-dynamic codesign pipeline: the static layer provides global planning, the dynamic layer adapts to observed sparsity, and the hardware feedback layer corrects residual mismatches, achieving robust and high-performance sparse execution.

## B. Static Analytical Layer

The static (offline) analytical layer performs device-aware global planning using only coarse-grained descriptors (matrix shape, global density, datatype) and hardware parameters (PE array size, SRAM capacity). Decisions are independent of pertile nnz and ensure buffer feasibility under plausible sparsity patterns. It sets baseline tile shapes, occupancy bounds, loop order, and SRAM allocation.

**Design Objective.** Given hardware constraints (on-chip SRAM capacity  $S_{SRAM}$ , PE array size P), the static layer maximizes operation intensity (OI):

$$OI = \frac{OPs}{Bytes_{loaded}}$$
 (1)

where OPs is the number of operations on the SRAM-resident block and Bytes<sub>loaded</sub> is off-chip data transferred. Maximizing OI improves reuse and reduces memory traffic, providing a robust starting point for runtime refinement.

**Notation and Feasibility Constraints.** Let the global matrices be  $A \in \mathbb{R}^{M \times K}$ ,  $B \in \mathbb{R}^{K \times N}$ , and  $C \in \mathbb{R}^{M \times N}$ . A candidate SRAM-resident block is parameterized by tile sizes  $(T_M, T_K, T_N)$ . The number of blocks per dimension is

$$n_M = \left\lceil \frac{M}{T_M} \right\rceil, \quad n_K = \left\lceil \frac{K}{T_K} \right\rceil, \quad n_N = \left\lceil \frac{N}{T_N} \right\rceil.$$

Assume global densities  $\rho_A$ ,  $\rho_B$  (fraction of nonzeros in A, B). Under the independence approximation, the expected nnz in a tile is

$$\mathbb{E}[\mathrm{nnz}_A^{\mathrm{tile}}] = \rho_A \cdot T_M T_K, \qquad \mathbb{E}[\mathrm{nnz}_B^{\mathrm{tile}}] = \rho_B \cdot T_K T_N,$$

and the (approximate) expected useful OPs in a tile is proportional to  $\mathbb{E}[\operatorname{nnz}_A^{\operatorname{tile}}] \cdot \frac{\mathbb{E}[\operatorname{nnz}_B^{\operatorname{tile}}]}{\sigma^{T_K}}$ , i.e., roughly  $\rho_A \rho_B T_M T_K T_N$ .

A conservative buffer feasibility constraint enforces that the expected SRAM footprint (including A, B, and psum storage) does not exceed  $S_{\text{SRAM}}$  with a safety margin. Let  $s_{\text{val}}$  be bytes per value (e.g., 4 for FP32). Then we require:

$$s_{\text{val}}(\mathbb{E}[\text{nnz}_A^{\text{tile}}] + \mathbb{E}[\text{nnz}_B^{\text{tile}}]) + s_{\text{psum}} \cdot T_M T_N \leq \beta S_{\text{SRAM}},$$
 (2)

where  $s_{\rm psum}$  is bytes per partial-sum (typically  $s_{\rm val}$ ) and  $0 < \beta < 1$  is a safety factor (e.g.,  $\beta = 0.8$ ) that leaves headroom for index storage, metadata, and runtime variability.

Static Layer Decision Variables. The static layer produces:

- Block-level plan: block size  $(T_M, T_K, T_N)$ , inter-block loop order (M, K, N), and SRAM partitioning ratio among A, B, and psum buffers.
- Baseline tile configuration: PE-level tile shape, tile occupancy upper bound for overflow safety, and initial intra-tile dataflow (InP/Row/OutP) based on global matrix dimensions and densities.

**Selection Criterion.** For each candidate  $(T_M, T_K, T_N)$  that satisfies the feasibility constraint (2), the static stage computes an estimated operation intensity:

$$\widehat{\mathrm{OI}}(T_M, T_K, T_N) \; = \; \frac{\widehat{\mathrm{OPs}}(T_M, T_K, T_N)}{\widehat{\mathrm{Bytes}}(T_M, T_K, T_N)}$$

## Algorithm 1 Static block selection (offline)

**Require:** global sizes (M, K, N), densities  $(\rho_A, \rho_B)$ , SRAM  $S_{\text{SRAM}}$ , PE config

**Ensure:** baseline  $(T_M, T_K, T_N)$ , traversal order, SRAM partition

- 1: generate candidate tile shapes  $\mathcal{T}$  (prune by divisibility and PE alignment)
- 2: for each  $(T_M, T_K, T_N) \in \mathcal{T}$  do
- 3: compute  $\mathbb{E}[\operatorname{nnz}_A^{\operatorname{tile}}], \mathbb{E}[\operatorname{nnz}_B^{\operatorname{tile}}]$
- 4: estimate Bytes and OP
- 5: if feasibility constraint (2) satisfied then
  - compute OI
- 7: else

6:

- 8: discard candidate
- 9: end if
- 10: end for
- 11: select candidate maximizing  $\widehat{OI}$ ; choose traversal order by simple heuristics on (M, K, N)
- 12: Partition SRAM  $S_A, S_B, S_C$  based on marginal-OI gain

where OPs and Bytes are estimated from global densities and datatype sizes (including index overhead for the chosen sparse representation). The chosen block shape maximizes OI subject to hardware constraints and practical PE-mapping considerations (e.g., divisibility by PE-row length).

Heuristics for Traversal Order and SRAM Allocation. Traversal order is chosen to balance A residency and B streaming:

- Small M, N (C fits in SRAM): prefer k-outer order to keep C resident while streaming A/B blocks.
- Small K: prefer order emphasizing M/N reuse.

SRAM is partitioned as  $S_A, S_B, S_C$  such that  $S_A + S_B + S_C \leq S_{\text{SRAM}}$ , with ratios reflecting their expected contribution to operation intensity.

**Example.** For a workload with large M, K, N and  $\rho_A = \rho_B = 0.1$  on a  $16 \times 16$  PE array, enumerating candidate tiles shows  $(T_M, T_K, T_N) = (64, 128, 64)$  achieves robust  $\widehat{\text{OI}}$  for InP/Row/OutP and satisfies (2) with  $\beta = 0.8$ . This tuple serves as the baseline block shape for online refinement. Algorithm 1 summarizes the static search.

**Role and Limitations.** The static layer provides a *safe* and *high-quality* starting point: it reduces the online search space, avoids infeasible tiles, and maximizes OI based on coarse statistics. However, it cannot capture per-tile variations; the online layer refines decisions, and the dynamic tuner corrects residual mismatches at runtime.

## C. Dynamic Profiling Layer

Once a block resides in on-chip SRAM, Harmonia can observe sparsity characteristics unavailable during offline modeling. The dynamic profiling layer performs lightweight pertile analysis to refine the baseline configuration, aiming to reduce per-tile latency and maximize overall throughput. Since the number of operations per tile is fixed, runtime optimization

focuses on improving PE utilization, mitigating routing pressure, preventing buffer spills, and avoiding worst-case dataflow behavior.

Runtime Information. Tile metadata reveals the exact nonzero count, allowing the computation of tile density ρtile = nnz/(TMTK) and the nonzero distribution across rows/columns. A lightweight sample of the stationary matrix A (performed once per block) estimates row-level variance, nonzero clustering, and potential PE-row workload imbalance. These statistics guide refinements in tile shape, dataflow selection, and resource allocation.

Tile Shape Refinement. Starting from the offline tile shape, Harmonia adjusts only dimensions exhibiting strong sparsity deviation:

- Low density: expand the tile to improve PE utilization and reuse.
- High density: shrink the tile to avoid psum spills, reduce merge depth, and lower routing pressure.
- Clustered sparsity: align tile boundaries with nonzero clusters to prevent local hotspots.

These changes are lightweight, modifying only tile slicing and dispatch logic without hardware reconfiguration.

Dataflow Selection (InP/Row/OutP). Harmonia evaluates three intra-tile dataflows per tile:

- InP: favors moderate sparsity with uniform per-row density, enabling strong B-row reuse and high parallelism.
- Row: best for high row-level variance or clustered A; keeps A rows stationary to stabilize PE load and psum merge depth.
- OutP: suitable for uniform A and low-density B; minimizes merge depth and routing contention.

Decision criteria combine tile-level density, B-side reuse, and structural properties of A observed via runtime sampling.

Limitations of Predictive Online Scheduling. Even with exact tile-level sparsity, the dynamic profiling layer cannot capture truly runtime phenomena, such as:

- merge-tree collisions and psum propagation delays,
- congestion in the DN/MRN routing network,
- instantaneous SRAM pressure or peak buffer usage,
- stalls from irregular producer–consumer alignment.

As a result, highly irregular tiles may still underperform relative to predictions. This motivates the Dynamic Tuning Layer (Section IV-D), which leverages runtime feedback to enact structural corrections to dataflow, tile shape, and resource allocation.

## *D. Dynamic Tuning Layer*

Operating directly within the Adaptive Hardware Substrate, the Dynamic Tuning layer is the final stage of Harmonia's scheduler. It reacts to *actual hardware behavior* observed during tile execution. Its goals are to (1) prevent performance collapse under highly irregular sparsity, (2) correct mispredictions from the Online Profiling layer, and (3) provide tile-level robustness via lightweight, feedback-driven adaptation.

This layer consists of three components: runtime anomaly detection, corrective actions for dataflow and tile shape, and a cost-aware switching model with stability mechanisms.

Runtime Feedback Signals. The PE array provides lightweight counters and flags:

- SRAM pressure & psum spills: indicate mismatches between sparsity and chosen dataflow (e.g., InP overflows psum buffers under extreme sparsity; Row-based reloads excessive B rows when locality is low).
- MRN merge depth / stalls: reveal deviations from expected merge-tree behavior, signaling dataflow mismatch.
- PE stall cycles: reflect operand unavailability, DN congestion, or merge backpressure, capturing aggregate microarchitectural impact.

These signals allow Harmonia to detect pathological tiles that deviate from online profiling predictions.

Corrective Actions. Upon detecting anomalies, Harmonia applies targeted adjustments to the tile shape or switches the dataflow to restore efficient execution.

- *(1) Dataflow Switching:* Dynamic Tuning first attempts to resolve anomalies by switching the intra-tile dataflow:
  - InP → Row / OutP: If merge depth is too low or PE utilization drops, switch to Row for highly imbalanced A rows, or OutP for near-uniform sparsity.
  - Row → OutP / InP: If repeated B row loads cause high SRAM pressure, switch to OutP if merge depth is low, or InP if the tile is dense.
  - OutP → Row / InP: If merge depth exceeds expectations, choose Row if spills stem from locality, or InP if caused by high density.

This enables the hardware to exhibit *logical heterogeneity*, adapting tile-by-tile.

- *(2) Tile Shape Scaling (Micro–Retiling):* If dataflow switching is insufficient, Harmonia adjusts the tile size locally:
  - Shrink: triggered by spills, SRAM pressure, or deep merge backpressure; reduce K to lower merge depth, or M/N to limit buffer and DN load.
  - Grow: triggered by consistently low density without anomalies; increases reuse and reduces control overhead.

Micro-retiling is confined within blocks, incurs negligible software overhead, and avoids re-tiling the entire matrix.

Cost Model and Switching Principle. Switching a dataflow or resizing a tile incurs a structural overhead, such as pipeline flush, DN/MRN reprogram and buffer reset. Harmonia triggers a change only if:

Gain 
$$> \alpha \cdot \text{Cost}$$
,  
where  $\begin{array}{c} \text{Gain} = T_{\text{before}} - T_{\text{after}}, \\ \text{Cost} = T_{\text{reconfig}} + T_{\text{flush}} + T_{\text{buf\_reset}} \end{array}$  (3)

The factor α tunes aggressiveness: larger for irregular workloads, smaller for regular workloads.

Stability Mechanisms and Worst-Case Bounds. Harmonia incorporates architectural safeguards to prevent oscillation and guarantee convergence during rapid sparsity fluctuations.

![](_page_7_Figure_0.jpeg)

Fig. 5. Architecture overview of Harmonia. The only hardware additions are lightweight feedback counters, a reconfiguration engine, and a tiling controller, which together form the SW-HW scheduling interface. These components expose runtime signals and enable low-cost dataflow and tile-shape reconfiguration, allowing the hierarchical scheduler to adapt execution on a per-tile basis.

These mechanisms maintain O(1) complexity and ensure deterministic hardware behavior:

- Convergence & Average Case: Feedback collection logic is decoupled from the execution datapath, making hardware profiling overhead negligible. A reconfiguration event involves a pipeline flush, DN/MRN reprogramming, and buffer resets, requiring only 20 to 50 cycles. Cycleaccurate simulations show these adjustments result in total stalls of less than 1%.
- Oscillation Avoidance: A hardware-based hysteresis
  mechanism prevents unstable reconfigurations. Anomaly
  counters must exceed predefined thresholds for T consec-\nutive cycles (typically 2 to 4) before triggering a tuning\nevent, effectively absorbing transient sparsity fluctuations.
- Worst-Case Guarantee: Harmonia strictly bounds the worst-case degradation penalty. If the analytical cost model (Equation (3)) evaluates that Gain ≤ Cost, or if a triggered switch fails to improve performance, the hardware absorbs a maximum penalty of exactly one reconfiguration delay (50 cycles) before falling back and reverting to the static baseline.

**Summary.** Dynamic Tuning does not change global plans but corrects tile-level deviations, stabilizing PE utilization and mitigating sparsity-induced bottlenecks. It ensures Harmonia achieves *robust performance across highly irregular sparsity patterns*, completing the three-layer hierarchy.

