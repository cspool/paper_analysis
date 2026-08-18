# Algorithm 1: Dequant Instruction Reorder

```
Input : Sliced INST list I; uncovered dequant list D; max
         partial buffer size Bmax
  Output: Optimized INST list after reordering
1 // Phase 1: Pre-scan to identify potential slots
2 S ← Dict // Valid slot positions, idle window
3 // Phase 2: Move dequant instructions forward
4 foreach (posdequant, instdequant) in reverse(D) do
5 extra buf ← Bmax − #P artialSum(instdequant);
6 pos tmp ← posdequant; candidate ← None;
7 while pos tmp < |I| − 1 do
8 if pos tmp ∈ S.keys and S[pos tmp] > 0 then
9 candidate ← pos tmp;
10 instnext ← I[pos tmp + 1];
11 // Calculate burden based on instruction type
12 if instnext.type = Compute then
13 burden ← #P artialSum(instnext);
14 else if instnext.type = ReadData then
15 burden ← 0;
16 else if instnext.type ∈
          {ReadScale, Dequant, W riteBack} then
17 burden ← Bmax + 1 // Blocking
18 extra buf ← extra buf − burden;
19 if extra buf < 0 then
20 return; // Cannot move further
21 pos tmp ← pos tmp + 1;
22 if candidate ̸= None then
23 Move(instdequant, candidate);
24 Update S[candidate]; CheckValid();
```

ing additional partial results in SRAM. To ensure the partial results fit within the buffer, we track the remaining partialbuffer capacity as we move a dequantization instruction. When it moves past a compute operation, we subtract the amount of partial result produced by that operation from the remaining capacity. Movement stops when capacity is exhausted.

*Rule 3: Constraint of available idle window.* We will also track the dequantization instructions already hidden in each free slot. A free slot will turn invalid when the idle window is fully utilized by these dequantizations.

#### *D. Dequantization Instruction Reordering Procedure*

For simplicity and to limit the overhead of instruction reordering, we restrict each reordering process to a single iteration over the weight matrix and apply Alg. 1 to the corresponding sliced instruction list. As shown in the algorithm, before reordering any dequantization instructions, we first scan for all candidate free slots created by DRAM row changes and DRAM reads, and record the number of idle cycles available in each slot (line 2). To maximize utilization of these slots, we process dequantization instructions in reverse order (line 4). For each dequantization, we attempt to place it at the latest candidate slot it can reach without exceeding the extra buffer capacity reserved for partial QGroup sums (extra buf), as indicated in line 5-6. This prevents a later dequantization from occupying a slot that could better serve an earlier one, thereby preserving more movement flexibility for preceding dequant operations. When moving a dequantization across other instructions, we apply the principles discussed in the previous section. These constraints either consume the remaining  $extra_buf$  (line 13/15) or may prevent further movement entirely (line 17&20). Once a valid slot is found, the dequantization is relocated accordingly (line 23). We then update the remaining idle cycles of that slot and mark it invalid if its idle window has been fully consumed (line 24).

It should be noted that, in weight-only quantization, the dequantization will provide the dequantized weight value for computation. As a result, moving dequantization over a computation will require storing the weight value used in the computation, the amount of which is much more than the partial sum. The available moving range is much smaller under the same partial buffer, as a result, we only implement this technique in weight-activation cases.

#### VIII. COMPILATION SPACE EXPLORATION

#### A. Guidelines for Compilation Strategy Selection

As discussed in Sec. IV-A, *FlexQ-NDP* formulates low-bit FP computation on NDP as a compilation space exploration problem. Given the complexity of the compilation space, we first analyze the compatibility between NDP compilation strategies and different QConfigs, and summarize two practical guidelines to enable compilation-space pruning and efficient search.

Guideline-1: Operator partition should consider the size of QGroup. When partitioning a quantized matrix across PUs, additional padding may be introduced because the matrix is organized into QGroups under the group-wise quantization scheme. For example, when a 704-wide matrix dimension is partitioned across 32 PUs, each PU can evenly process 22 elements. However, if the QGroup size is misaligned with this partitioning (e.g., 64), a single QGroup spans three PUs, and the last PU must process elements belonging to two QGroups with different scales. As a result, this PU is forced to perform dequantization while other PUs are still executing MAC, leading to divergent instruction streams across PUs and incurring non-uniform control overhead.

Guideline-2: Buffer Size Allocation. Since scales also need to be fetched from DRAM and stored in buffer during computation, how to divide the SRAM buffer between values and scales becomes a critical problem. For example, for QConfigs with larger QGroups (e.g., group size 128), more buffer space tends to be allocated to values, as the number of scales is relatively small. Conversely, for smaller QGroups (e.g., group size 16), a larger portion of the buffer is allocated to scales. Furthermore, the proposed scale-value interleaved FP layout and dequantization-hiding technique are also tightly coupled to the buffer allocation. For example, constructing an itBlock requires determining the number of QGroups due to the scale-buffer capacity (Equ. 2), and dequantization instruction reordering is constrained by the available buffer size ( $extra_buf$  in Alg. 1).

#### B. Compilation Space Encoding Method

FlexQ-NDP constructs and encodes the compilation space for low-bit FP in the following aspects.

**Operator Partition:** We follow the definition in prior work [63] to specify how each dimension of the operator is partitioned across hardware levels:  $Partition = \{Part_{Ch,Ra,De,Bk}^{M,K,N}\}$ , which represents the partition number of the M/N/K dimensions at the Channel/Rank/Device/Bank hierarchy levels and is constrained by hardware parallelism. It should be noted that the term "device" refers to the DRAM modules within a rank that share the command and address bus while maintaining independent data buses. This concept is seldom used in the GDDR family. However, we include it here for the compatibility with other memory types.

We **Buffer** allocation: use a four element to describe the size of allocated buffer: quadruple  $(Val\_Buf, Scale_A\_Buf, Scale_W\_Buf, Dequant\_Buf).$ We fix the buffer size for final result to the bitwidth of a DRAM column (e.g., 32B), according to previous practice in [63]. The sum of these buffers is limited by the total available buffer size  $Total\_Buf$ , and the granularity of allocation is also set to 32B. For the weight-only quantization, Scale\_A\_Buf is set to zero, and Dequant\_Buf is parsed as the buffer size to store the dequantized weight data.

**Loop Permutation:** Here we take the loop tiling on the K dimension as an example. The granularity of the inner loop (K inner) is set to the number of values that can be accommodated in a DRAM column:

$$K_{Tile} = Tile_{Col} \times \#Value/Col$$
 (10)

Based on the tiling scheme, we allow swapping the order of M dimension and N dimension, and form two candidate loop orders: {Order1=M $\rightarrow$ K Outer $\rightarrow$ N $\rightarrow$ K Inner} and {Order2=N $\rightarrow$ K Outer $\rightarrow$ M $\rightarrow$ K Inner}. To maximize data reuse, we will store weight in the value buffer for Order1, and reserve input in the value buffer for Order2. To ensure no buffer miss will be triggered during the inner loop, we restrict the choice of  $K\_Tile$  by  $K\_Tile \leq Data\_Buf$ .

**DRAM Mapping:** As stated in Sec. VI, the column number of scale region  $Col_S$  can vary under the restriction of  $Scale_W\_Buf$ . Each  $Col_S$  corresponds to an optimal layout.

# Algorithm 1: Dequant Instruction Reorder

```
Input : Sliced INST list I; uncovered dequant list D; max
         partial buffer size Bmax
  Output: Optimized INST list after reordering
1 // Phase 1: Pre-scan to identify potential slots
2 S ← Dict // Valid slot positions, idle window
3 // Phase 2: Move dequant instructions forward
4 foreach (posdequant, instdequant) in reverse(D) do
5 extra buf ← Bmax − #P artialSum(instdequant);
6 pos tmp ← posdequant; candidate ← None;
7 while pos tmp < |I| − 1 do
8 if pos tmp ∈ S.keys and S[pos tmp] > 0 then
9 candidate ← pos tmp;
10 instnext ← I[pos tmp + 1];
11 // Calculate burden based on instruction type
12 if instnext.type = Compute then
13 burden ← #P artialSum(instnext);
14 else if instnext.type = ReadData then
15 burden ← 0;
16 else if instnext.type ∈
          {ReadScale, Dequant, W riteBack} then
17 burden ← Bmax + 1 // Blocking
18 extra buf ← extra buf − burden;
19 if extra buf < 0 then
20 return; // Cannot move further
21 pos tmp ← pos tmp + 1;
22 if candidate ̸= None then
23 Move(instdequant, candidate);
24 Update S[candidate]; CheckValid();
```

ing additional partial results in SRAM. To ensure the partial results fit within the buffer, we track the remaining partialbuffer capacity as we move a dequantization instruction. When it moves past a compute operation, we subtract the amount of partial result produced by that operation from the remaining capacity. Movement stops when capacity is exhausted.

*Rule 3: Constraint of available idle window.* We will also track the dequantization instructions already hidden in each free slot. A free slot will turn invalid when the idle window is fully utilized by these dequantizations.

#### *D. Dequantization Instruction Reordering Procedure*

For simplicity and to limit the overhead of instruction reordering, we restrict each reordering process to a single iteration over the weight matrix and apply Alg. 1 to the corresponding sliced instruction list. As shown in the algorithm, before reordering any dequantization instructions, we first scan for all candidate free slots created by DRAM row changes and DRAM reads, and record the number of idle cycles available in each slot (line 2). To maximize utilization of these slots, we process dequantization instructions in reverse order (line 4). For each dequantization, we attempt to place it at the latest candidate slot it can reach without exceeding the extra buffer capacity reserved for partial QGroup sums (extra buf), as indicated in line 5-6. This prevents a later dequantization from occupying a slot that could better serve an earlier one, thereby preserving more movement flexibility for preceding dequant operations. When moving a dequantization across other instructions, we apply the principles discussed in the previous section. These constraints either consume the remaining  $extra_buf$  (line 13/15) or may prevent further movement entirely (line 17&20). Once a valid slot is found, the dequantization is relocated accordingly (line 23). We then update the remaining idle cycles of that slot and mark it invalid if its idle window has been fully consumed (line 24).

It should be noted that, in weight-only quantization, the dequantization will provide the dequantized weight value for computation. As a result, moving dequantization over a computation will require storing the weight value used in the computation, the amount of which is much more than the partial sum. The available moving range is much smaller under the same partial buffer, as a result, we only implement this technique in weight-activation cases.

#### VIII. COMPILATION SPACE EXPLORATION

#### A. Guidelines for Compilation Strategy Selection

As discussed in Sec. IV-A, *FlexQ-NDP* formulates low-bit FP computation on NDP as a compilation space exploration problem. Given the complexity of the compilation space, we first analyze the compatibility between NDP compilation strategies and different QConfigs, and summarize two practical guidelines to enable compilation-space pruning and efficient search.

Guideline-1: Operator partition should consider the size of QGroup. When partitioning a quantized matrix across PUs, additional padding may be introduced because the matrix is organized into QGroups under the group-wise quantization scheme. For example, when a 704-wide matrix dimension is partitioned across 32 PUs, each PU can evenly process 22 elements. However, if the QGroup size is misaligned with this partitioning (e.g., 64), a single QGroup spans three PUs, and the last PU must process elements belonging to two QGroups with different scales. As a result, this PU is forced to perform dequantization while other PUs are still executing MAC, leading to divergent instruction streams across PUs and incurring non-uniform control overhead.

Guideline-2: Buffer Size Allocation. Since scales also need to be fetched from DRAM and stored in buffer during computation, how to divide the SRAM buffer between values and scales becomes a critical problem. For example, for QConfigs with larger QGroups (e.g., group size 128), more buffer space tends to be allocated to values, as the number of scales is relatively small. Conversely, for smaller QGroups (e.g., group size 16), a larger portion of the buffer is allocated to scales. Furthermore, the proposed scale-value interleaved FP layout and dequantization-hiding technique are also tightly coupled to the buffer allocation. For example, constructing an itBlock requires determining the number of QGroups due to the scale-buffer capacity (Equ. 2), and dequantization instruction reordering is constrained by the available buffer size ( $extra_buf$  in Alg. 1).

#### B. Compilation Space Encoding Method

FlexQ-NDP constructs and encodes the compilation space for low-bit FP in the following aspects.

**Operator Partition:** We follow the definition in prior work [63] to specify how each dimension of the operator is partitioned across hardware levels:  $Partition = \{Part_{Ch,Ra,De,Bk}^{M,K,N}\}$ , which represents the partition number of the M/N/K dimensions at the Channel/Rank/Device/Bank hierarchy levels and is constrained by hardware parallelism. It should be noted that the term "device" refers to the DRAM modules within a rank that share the command and address bus while maintaining independent data buses. This concept is seldom used in the GDDR family. However, we include it here for the compatibility with other memory types.

We **Buffer** allocation: use a four element to describe the size of allocated buffer: quadruple  $(Val\_Buf, Scale_A\_Buf, Scale_W\_Buf, Dequant\_Buf).$ We fix the buffer size for final result to the bitwidth of a DRAM column (e.g., 32B), according to previous practice in [63]. The sum of these buffers is limited by the total available buffer size  $Total\_Buf$ , and the granularity of allocation is also set to 32B. For the weight-only quantization, Scale\_A\_Buf is set to zero, and Dequant\_Buf is parsed as the buffer size to store the dequantized weight data.

**Loop Permutation:** Here we take the loop tiling on the K dimension as an example. The granularity of the inner loop (K inner) is set to the number of values that can be accommodated in a DRAM column:

$$K_{Tile} = Tile_{Col} \times \#Value/Col$$
 (10)

Based on the tiling scheme, we allow swapping the order of M dimension and N dimension, and form two candidate loop orders: {Order1=M $\rightarrow$ K Outer $\rightarrow$ N $\rightarrow$ K Inner} and {Order2=N $\rightarrow$ K Outer $\rightarrow$ M $\rightarrow$ K Inner}. To maximize data reuse, we will store weight in the value buffer for Order1, and reserve input in the value buffer for Order2. To ensure no buffer miss will be triggered during the inner loop, we restrict the choice of  $K\_Tile$  by  $K\_Tile \leq Data\_Buf$ .

**DRAM Mapping:** As stated in Sec. VI, the column number of scale region  $Col_S$  can vary under the restriction of  $Scale_W\_Buf$ . Each  $Col_S$  corresponds to an optimal layout.

