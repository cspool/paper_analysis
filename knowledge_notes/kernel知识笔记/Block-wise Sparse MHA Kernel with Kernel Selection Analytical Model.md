## Block-wise Sparse MHA Kernel with Kernel Selection Analytical Model

术语是什么？通过联网搜索让回答具体和精准。

Block-wise Sparse MHA Kernel是STOF框架中针对GPU优化的sparse multi-head attention fused kernel，以OT (OuterTile)为粒度partition Q/K/V张量，利用two-level sparse storage format跳过无效计算，将QK^T GEMM、mask application、Softmax和PV GEMM全部融合在单一kernel中执行。与之互补的是Row-wise Kernel（以Q行为粒度partition，warp内shuffle通信，适合小seq_len+高稀疏率场景）。Kernel Selection Analytical Model（公式1）基于valid OT ratio和seq_len自动选择row-wise或block-wise kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Kernel Selection公式和核心优化：

```
// ===== Kernel Selection Analytical Model (STOF Eq. 1) =====
threshold = (load_row_ptr[ceil(seq_len/16)] / (ceil(seq_len/16))²) 
          - (τ / (log₂(ceil(seq_len/16)))²)
// τ = 1.2 (empirically set)
// 第一项 = valid OT ratio per row
// 第二项 = log penalty（压制extreme sparse长序列场景）
//
// threshold < 0 → row-wise kernel (warp内shuffle, 零warp间sync)
// threshold ≥ 0 → block-wise kernel (partition到SMEM利用memory hierarchy)

// ===== Block-wise Kernel Key Optimizations =====
// 1. Q Register Resident: Q_i保持在register跨KV tile循环复用（vs FA2需每次SMEM读）
// 2. Async Data Copying: cp.async异步加载V_j，与Q_i×K_Tj GEMM重叠
// 3. OT-level Compute Skipping: 仅加载valid OTs，Bigbird (80.8% sparsity)仅~19.2% OTs计算
// 4. IT-level Fine-grained Mask: 对part OTs用bitmap_mask做per-element mask (POPC/shift)
// 5. SMEM Double Buffering: K_Tj/V_j共享同一SMEM物理区域
// 6. Tensor Core Alignment: 8×8 IT对齐mma.m16n8k16 operand tile

// Performance: (batch=16, seq_len=4096, sliding window 93.8% sparsity)
// STOF block-wise vs FA2: 4.8× on A100
// STOF block-wise vs FlexAttention: 4.9× on A100
```

Kernel selection实例：
```
// 场景1: BERT-Base, batch=1, seq_len=128, causal (50% sparsity)
// OT grid: 2×2, load_row_ptr[2]=3 → threshold≈0.75-1.2=-0.45<0
// → row-wise kernel: warp内shuffle, 零warp间sync

// 场景2: BERT-Base, batch=16, seq_len=4096, Bigbird (80.8% sparsity)
// OT grid: 64×64, load_row_ptr[256]≈12/row
// → block-wise kernel: SMEM hierarchy利用, Q register resident
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

STOF实现：block-wise kernel基于FA2的CuTe结构，扩展引入two-level storage format和对应优化，约2,500 LOC C/CUDA。通过torch/cpp_extension JIT编译为.so。CuTe提供tile-level抽象（TiledMMA、TiledCopy等），允许类型安全地组合tensor core操作、shared memory staging和global memory access。block_size/num_warps/num_stages等launch配置通过AutoTune搜索。

涉及论文标题：
- Accelerating Sparse Transformer Inference on GPU

