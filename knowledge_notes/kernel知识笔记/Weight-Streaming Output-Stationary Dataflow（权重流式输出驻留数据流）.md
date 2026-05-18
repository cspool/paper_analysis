## Weight-Streaming Output-Stationary Dataflow（权重流式输出驻留数据流）

术语是什么？通过联网搜索让回答具体和精准。

Weight-Streaming Output-Stationary Dataflow是RPU论文中TMAC（Tile Multiplier-Accumulator）采用的dataflow策略。在这种dataflow中：(1) **Weight-streaming**: 权重元素从off-chip memory→on-chip buffer→Stream Decoder→TMAC以streaming方式依次流入，每个weight element被消费后立即被下一个weight element替换（无on-chip weight reuse）；(2) **Output-stationary**: partial sum（output的partial accumulation）驻留在TMAC的local accumulator register中，跨多个weight tile累积直到一个stripe完成。Activation在streaming和output之间扮演中间角色——broadcast across TMAC columns但局限在一个stripe内（64 BF16 values），不跨stripe复用。这种dataflow是三种经典systolic-array dataflow（weight-stationary, output-stationary, input-stationary）的混合变体，专为LLM decode的VMM优化。

从kernel调度角度拆解：

Weight-Streaming Output-Stationary dataflow的伪代码（per TMAC, per stripe）：

```
// === TMAC Dataflow (8×8 MAC array) ===
// Activation: broadcast across columns (spatial reuse)
// Weight: stream through tiles (no on-chip reuse)
// Partial sum: stationary in accumulator (temporal accumulation)

// For one stripe (8 tile-columns, T tile-rows per column):
for col in 0..7:  // tile columns in stripe
    // === Activation loading (once per column) ===
    act_elements[0..7] = activation_register_file[col*8 : (col+1)*8]
    // act_elements stay in column broadcast registers
    
    // Reset column accumulators
    accumulators[col][0..7] = 0  // FP32
    
    for row in 0..T-1:  // tile rows
        // === Weight streaming (per tile) ===
        weight_tile = stream_decoder.decode_next()  // 8×8 BF16
        // weight_tile elements stream through MAC array:
        // - 8 columns × 8 rows = 64 MACs in parallel
        // - Each row receives different weight element
        // - All rows in same column receive same activation element
        
        // Parallel MAC (64 MACs/cycle):
        for r in 0..7, c in 0..7:
            accumulators[col][r] += act_elements[c] * weight_tile[r][c]
            // act_elements[c]: broadcast across column c
            // weight_tile[r][c]: element-wise streamed
            // accumulators[col][r]: output-stationary (stays in local reg)
    
    // === Column tree-sum reduction (after all tile-rows) ===
    for r in 0..7:
        output[col] = tree_sum_3stage(accumulators[col][0..7])
        // 3-stage reduction: 8→4→2→1
        // Stage 1: (A0+A1), (A2+A3), (A4+A5), (A6+A7)
        // Stage 2: (A01+A23), (A45+A67)
        // Stage 3: A0123 + A4567

// After all 8 columns: output[0..7] ready
```

Dataflow选择理由（vs alternatives）：
- **Weight-stationary**: weight需要保持在MAC array中跨多个activation复用——在decode VMM中每个weight仅被使用一次（BS=1），无复用机会
- **Input-stationary (activation stationary)**: 需要full activation vector on-chip → 大activation（K-dim~16K for Llama3-70B）无法全部on-chip
- **Output-stationary**: partial sum保持在local accumulator，最小化writeback bandwidth——decode VMM中output vector size（N/C）相对较小（per-core shard of N-dim），适合on-chip保持

术语一般如何实现？如何使用？

在RPU中，TMAC硬件直接实现weight-streaming output-stationary dataflow：1024-bit compute bus每cycle deliver一个8×8 BF16 weight tile（64 elements）到TMAC；activation register file保持64 BF16 activation values（8 columns × 8 rows broadcast）；每个MAC单元的FP32 accumulator独立保持partial sum，直至stripe结束tree-sum reduction到local register file。Dataflow由RPU ISA指令驱动compiler决定，运行时TMAC硬件自动执行fixed streaming schedule。

涉及论文标题：
- RPU - A Reasoning Processing Unit

