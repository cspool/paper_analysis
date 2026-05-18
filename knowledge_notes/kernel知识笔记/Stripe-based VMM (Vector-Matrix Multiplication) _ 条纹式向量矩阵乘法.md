## Stripe-based VMM (Vector-Matrix Multiplication) / 条纹式向量矩阵乘法

术语是什么？通过联网搜索让回答具体和精准。

Stripe-based VMM是RPU论文提出的一种将Vector-Matrix Multiplication (O=V×W, V∈R^{1×K}, W∈R^{K×N})分块执行的策略，专为低延迟LLM decode的分布式VMM设计。一个stripe定义为8个垂直堆叠的tile（一个tile=8列×K行weight子矩阵）跨所有weight shard列。VMM按stripe粒度执行：先处理一个stripe内所有列tile的tile-rows（column-first traversal），做完列级tree-sum reduction后写入local register，再进入下一stripe。Stripe-based执行在三种常见VMM遍历策略中取得平衡：(1) Column-first (inner-product风格)需要full activation vector on-chip；(2) Row-first (outer-product风格)产生高partial sum writeback bandwidth；(3) Stripe-based最小化on-chip bandwidth需求，并使计算与通信fine-grained overlap（处理当前stripe时network buffer收集下一stripe所需activation shard）。

从kernel调度角度拆解：

Stripe-based VMM的伪代码执行流程（以O = V × W, C个core column-sharded的分布式场景为例）：

```
// VMM Stripe-based Kernel
// W sharded across C cores: each core stores W_i ∈ R^{K×N/C}
// Core i computes O_i = V × W_i (output fragment)

// === Stripe Definition ===
// Stripe: 8 vertically stacked tiles spanning all columns of W_i
// Tile: 8 columns × K rows of W_i (actually: 8×8 weight sub-matrix per TMAC cycle)
// Activation shard per stripe: 64 BF16 values

// === For each stripe s in W_i ===
for stripe_s = 0 to (N/C / 64) - 1:
    // Phase 1: Wait for activation shard
    activation_shard = network_buffer.read_valid(64 BF16 values)
    write_to_activation_register_file(activation_shard)  // 64 entries, 8 per tile column

    // Phase 2: Column-first traversal within stripe
    // (Process tile-column 0 first, then tile-column 1, ..., tile-column 7)
    for tile_col in 0..7:
        // Phase 2a: Tile-row iteration (weight-streaming)
        for tile_row in 0..(K/8)-1:
            weight_tile = stream_decoder.decode(memory_buffer.read())
            // weight_tile is 8×8 BF16, broadcast to all TMAC columns
            TMAC.compute(activation_shard[tile_col], weight_tile)
            // 64 MACs/cycle: activation[col] broadcast across 8 columns,
            // weight elements stream in element-wise

        // Phase 2b: Column-wise tree-sum reduction
        // After all tile-rows in this column processed:
        partial_sums[0..7] = TMAC.accumulators[tile_col][0..7]
        output[tile_col] = tree_sum_3stage(partial_sums)  // 3-stage adder tree

    // Phase 3: Write back stripe results
    local_register_file.write(output[0..7])
    // Meanwhile, network_buffer asynchronously collects next stripe's activation

// === After all stripes: output fragment O_i ready ===
// O_i forwarded to downstream cores via Network DMA
```

Striping的关键特性：
1. **最小化on-chip bandwidth**：每次仅需64 BF16 activation on-chip（vs full K-length vector for inner-product）
2. **低writeback pressure**：仅stripe结束时writeback一次（8个output values），而不是每tile writeback
3. **通信-计算overlap**：处理当前stripe时network异步收集下一stripe activation
4. **高度适配weight-streaming dataflow**：weight tiles按stripe内行列顺序依次stream through TMAC

术语一般如何实现？如何使用？

Stripe-based VMM通过RPU compiler生成的ISA指令实现。Compiler在将torch.nn.Linear lowering为三阶段micro-kernel时，Looping阶段内嵌stripe iteration logic：Loading stage配置DMA从HBM-CO pre-read weight tiles for next stripe → Loop stage drive TMAC execute current stripe (column-first traversal + tree-sum) → Launch stage forward activation to next core。TMAC hardware直接支持stripe-based dataflow——64-entry activation register file、3-stage tree-sum reduction、per-column accumulator management。Pipeline Arbiter管理stripe间的activation和weight buffer entry同步。

涉及论文标题：
- RPU - A Reasoning Processing Unit

