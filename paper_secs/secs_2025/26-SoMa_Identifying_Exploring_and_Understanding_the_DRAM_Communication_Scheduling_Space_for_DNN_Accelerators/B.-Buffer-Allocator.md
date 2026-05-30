# B. Buffer Allocator

In the first iteration, we conduct a complete two-stage search, with the only constraint being that the buffer usage does not exceed the hardware buffer capacity. We record the maximum buffer usage of the scheme explored in the first stage ( $Buffer_{max}$ ), as well as the best overall encoding scheme and its cost  $(Cost_{best})$ . In subsequent iterations, the buffer usage limit for the first stage is reduced by a% (10%) used in the following experiments) of  $Buffer_{max}$  each time (solutions exceeding this limit are deemed invalid), and the overall cost  $(Cost_{temp})$  is recorded. If  $Cost_{temp}$  is better than  $Cost_{best}$ ,  $Cost_{best}$  and the optimal encoding scheme are updated accordingly. The iteration stops when the costs of the optimal solutions found in two consecutive iterations do not exceed  $Cost_{best}$ . The rationale behind using this iteration to allocate buffers between the two stages for overall optimization is that while the performance of both stages improves with increased buffer size, the rate of improvement slows as buffer size grows. Therefore, adjusting the buffer allocation in small increments helps effectively find the sweet spot that maximizes the combined performance of both stages.

#### C. LFA & DLSA Exploration Stage

Both LFA and DLSA employ SA to explore this space. The key factors in SA are the initial solution, cooling schedule, and operators. The initial solution and operators are discussed in the following sections, while the cooling schedule is described here. Starting from an initial solution, each iteration randomly selects an operation to modify the encoding and evaluates it. If the new scheme's cost (c') is higher than the previous cost (c), it is accepted with probability  $p = e^{\frac{c-c}{cT_n}}$ , where  $T_n$ is the temperature at iteration n. Otherwise, the scheme is always accepted. The temperature at iteration n is given by  $T_n = T_0 \frac{1 - \frac{n}{N}}{1 + \alpha \frac{n}{N}}$ , where  $T_0$  is the initial temperature and  $\alpha$  is the cooling rate. The total number of iterations is  $N = \beta X$ . For the first stage,  $\beta$  and X are set to 100 and the number of layers, respectively. For the second stage, they are set to 1000 and the number of DRAM tensors, respectively. We also support setting an additional termination time. Once this time is reached, the algorithm performs Y more iterations, accepting only improved solutions.

1) LFA Exploration Stage: In this stage, The initial solution consists of each layer forming its own independent LG and FLG (e.g., both FLG and LG are 1, 2, 3, 4 as shown in Fig. 4), meaning no fusion is applied. The tile number is set to the minimum granularity, corresponding to the size required for the core array to perform parallel computation. Then, the SA operators transform the LFA, while the DLSA is determined using a classical double-buffer strategy (as introduced in Sec. III-C). The specific operators are as follows:

**Change Computing Order:** Randomly select a layer and change its order to another valid location.

**Change Tiling Number:** Randomly select an FLG and multiply or divide its Tiling Number by 2.

**Add/Delete An FLC**: Randomly add or delete an element in FLC Set. Specifically, adding an FLC means cutting an FLG into two FLGs with the same Tiling Number attribute as the original FLG. Removing an FLC means merging two FLGs into one, with the new FLG's Tiling Number inherited probabilistically based on the layer count ratio of the original two FLGs.

**Add/Delete A DRAM Cut:** Randomly add or delete an element in the DRAM Cut Set. The added element must be in the FLC Set.

2) DLSA Exploration Stage: In this stage, the initial solution adopts the best scheme explored by the previous stage, with the LFA attribute remaining constant. The SA controller then primarily focuses on searching within the DLSA for the DRAM tensors corresponding to this selected LFA. The specific operators are introduced as follows:

**Change DRAM Tensor Order**: Randomly select a DRAM tensor and change its order to another valid location.

Change Living Duration: Randomly select a DRAM tensor and randomly change its Start (for ifmaps and weights) or End (for ofmaps). For example, in Fig. 4(b), by reducing the Start of  $W_B$  by 1 from B to  $A_2$ , the STALL between  $A_2$  and B can be eliminated, and  $W_B$  will be included in the buffer associated with  $A_2$ .

Notably, in each operation, the probability of selecting a DRAM tensor is proportional to its size since larger tensors generally have a greater impact on performance and buffer utilization, warranting more transformation opportunities.

#### D. Evaluator

In this section, we introduce an accurate evaluator, capable of evaluating various scheduling schemes, described using our Tensor-centric Notation across different hardware configurations in terms of energy cost and latency.

The evaluation process follows a local-to-global approach, first assessing each computing tile and DRAM load/store request (DRAM tensor) individually and then conducting an overall assessment.

For each computing tile (e.g.,  $A_1$  in Fig. 4), the ifmaps and weights have been prefetched into the GBUF, and the ofmaps are written back to the GBUF. From a macro perspective, the Core Array Scheduler explores how to further

divide this tile into sub-tiles for computation by each core (as introduced in Sec. II), aiming to maximize parallelism and data reuse. The corresponding Evaluator assesses each interaction between GBUF and L0 buffers, as well as computations, while accounting for dependencies to evaluate overall performance and energy consumption. The corresponding energy costs and computing time of the searched optimal scheme are taken as the tile's energy costs and computing time. As this area of research is well-established [20], [21], [28], [32], [36], [42], [52], [55], [57], we adopt a classic scheduler and evaluator for this purpose [32], [42].

Each DRAM communication tensor's energy costs are calculated by summing the products of read and write data volumes with their respective unit energy costs. The read and write times are calculated by dividing the data volumes by the respective bandwidths.

The total energy cost is calculated by summing up the energy costs of the above sub-components, similar to existing classical works [32], [42], [57]. The total computing time is derived based on the evaluated times of all computing tiles and DRAM tensors using the following method:

For each DRAM tensor, it can start execution only when the following three conditions are met: 1) the preceding DRAM tensor has been completed; 2) for ifmaps or weights, their Start must be smaller than or equal to the current tile ID; and 3) for ofmaps, it must wait until its generating computing tile has finished. For example, in Fig. 4, although IC2's Start is C1, the preceding DRAM tensor (WD) is not completed until E1, so it can only start at the middle of E1. Moreover, W<sup>B</sup> has a Start of B. Although the previous DRAM tensor (IA2) has already been completed, it must still wait until A<sup>2</sup> finishes before it can begin.

Each computing tile can start execution only when the following conditions are met: 1) all required data (ifmaps, weights, etc.) are ready. For the example in Fig. 2, A1, B1, and C<sup>1</sup> cannot follow their respective preceding computing tile immediately because the required data are not ready at the end of the preceding computing tile. 2) All DRAM tensors with End less than or equal to the tile must be completed. For the instance in Fig. 4, D<sup>1</sup> cannot follow E<sup>1</sup> because OE1's End is D1, and D<sup>1</sup> cannot start until OE<sup>1</sup> has finished execution.

