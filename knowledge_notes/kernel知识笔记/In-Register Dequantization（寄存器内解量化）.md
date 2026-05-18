## In-Register Dequantization（寄存器内解量化）

术语是什么？通过联网搜索让回答具体和精准。
In-register dequantization是Quantix fused kernel中使用的技术，将non-uniform quantized weights的所有dequantization操作完全在GPU register file内完成，无需任何global memory或shared memory的中间读写。该技术通过bit concatenation（1-bit+2-bit→3-bit index）和centroid indexing（3-bit index查per-row FP16 centroids）两条寄存器内路径重建FP16权重，避免了SqueezeLLM等传统方法中"dequantize→写回内存→从内存读回→matmul"的多级内存路径和cache-unfriendly centroid pointer chasing。Ablation study显示in-register dequantization是Quantix最大性能贡献组件——移除后性能降至完整版本的约40%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
In-register dequantization在fused kernel的一个TC tile (16×16)内的执行流程：

```
// ===== 寄存器布局 =====
// 一个TC tile各thread负责4 pairs (8个元素/thread)
// Pair的结构: row0 element1, row0 element2, row8 element1, row8 element2
// 原因: Tensor Core MMA的ldmatrix指令需要交错来自不同行的数据

// W1' (1-bit) segment registers: 128 bits/thread
//  R1_low:   bit positions 0-31 (32个1-bit values for W1')
// W2' (2-bit) segment registers: 256 bits/thread
//  R2_low, R2_high: bit positions 0-63, 64-127 (32个2-bit values for W2')
// Centroids C registers: 8个FP16 per row × 2 rows = 16个FP16 registers
```

```
// ===== Step 1: Bit Concatenation =====
// 处理4个pairs (i=0,1,2,3)，每个pair含2行×2列=4个元素
for pair_idx in range(4):
    // 从W1'和W2'提取对应bits
    bit1_row0_col0 = extract_bit(R1, pair_idx*2+0)     // 1-bit
    bit2_row0_col0 = extract_2bits(R2, pair_idx*2+0)    // 2-bit
    
    // Concatenate: [1-bit] + [2-bit] → [3-bit]
    index_3bit_row0_col0 = (bit1_row0_col0 << 2) | bit2_row0_col0
    // 例如: [1] + [10] → [110] = 6 (binary)
    
    // 8个3-bit indices打包到1个32-bit register
    // Register layout: [row0_pair0 | row0_pair1 | row8_pair0 | row8_pair1 | ...]
```

```
// ===== Step 2: Centroid Indexing =====
// 每行有2^k个FP16 centroids (k=3时8个)
// Centroids C均已在registers中（从shared memory加载）
// 例如: Row 0's centroids = [33.14, -48.24, 1.32, 0.90, -7.82, 53.13, 73.96, -27.63]

// Step 2a: Extract individual 3-bit index from packed 32-bit register R
// qi = (R >> (3*i)) & 0x7
// 避免条件分支的bitwise操作
for i in range(8):
    qi = (R >> (3 * i)) & 0x7  // 提取第i个3-bit index
    
// Step 2b: Centroid lookup (register-to-register)
// 用qi索引centroids数组
    w_deq[i] = centroids_row[i//4][qi]
    // 论文未明确说明centroid lookup的具体register-level实现
    // 可能使用PRMT（permute）指令或conditional select
```

```
// ===== 为什么In-Register关键 =====
// Naive方法 (SqueezeLLM等):
//   W† = C[Wq]  → pointer chasing: Wq读取→地址计算→memory load→返回→存入memory
//   latency: global memory load (数百cycles)
// In-register方法:
//   qi = (R >> (3*i)) & 0x7 → 1条shift + 1条AND (各1 cycle)
//   centroid select from register → ~1-5 cycles
//   latency: 数cycle vs 数百cycles
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
In-register dequantization在Quantix fused kernel中以CUDA PTX实现。关键点：
1. 数据准备：bit-divided和bit-mapped的W1'/W2' segments通过cp.async从global预取到shared memory，再通过ld.shared加载到registers
2. Dequantization在CUDA cores上执行（ALU指令：shift, AND），无需shared memory参与
3. 重建的FP16 weights直接在registers中被Tensor Core MMA消费，中间不写入任何内存
4. 寄存器布局精心设计以匹配Tensor Core的ldmatrix指令对数据interleaving的要求（如图7所示row0/row8交替）
5. 该技术要求centroids也在registers中——每行8个FP16 centroids (3-bit)，共需16个FP16 registers（2行），对于32-wide的warp register file可承受
6. 限制：过大batch时register pressure增加可能导致spilling，影响ALU utilization——论文观察到batch≥32时ALU utilization下降即因register spilling

涉及论文标题：
- High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

