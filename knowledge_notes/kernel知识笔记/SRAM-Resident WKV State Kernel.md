## SRAM-Resident WKV State Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
**原始 RWKV 论文（EMNLP 2023）**首次引入 custom CUDA kernel 用于 WKV 计算，以解决串行扫描在标准深度学习框架中的低效问题。Eagle/Finch 训练时沿用的 custom CUDA kernel，核心设计选择：**不沿时间维度并行**（尽管 WKV 可通过 associative scan 做 time-parallel），而是**沿非时间维度并行**，将 recurrent state 操作保持在 GPU SRAM 中。原始 RWKV 的 WKV kernel 面向向量 state（head=1），计算量较小（仅 d 维逐元素操作）；Eagle/Finch 的矩阵 state（head=64）大幅增加了 SRAM 驻留需求。原理是：time-parallel 的 associative scan 虽并行度高，但每次迭代需要从 HBM 读取中间结果→SRAM 计算→写回 HBM，memory bandwidth 成为瓶颈；非时间维度并行将 state s∈R^{(D/h)×(D/h)} 驻留在 SRAM 中，每时间步仅读写 token 输入/输出（远小于 state 矩阵），memory 开销显著降低。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Finch WKV kernel 执行流程：
```
// 配置: batch=8, D=4096, head=64, h=64
// State: s ∈ R^{B×h×64×64} = 8×64×64×64×4B ≈ 8MB (SRAM-resident)

For t = 0 to seq_len-1:
  // 沿非时间维并行: batch和head维度
  parallel_for (b, head):
    // 1. 从HBM加载当前token (小数据量): x_t ∈ R^D
    //    → ddlerp Token Shift (SRAM内计算)
    //    使用LoRA: tanh(x@A)@B, A∈R^{D×32}, B∈R^{32×D}
    
    // 2. WKV计算 (SRAM内, state在SRAM中驻留):
    k_t, v_t ← Token Shift + Linear
    wkv = (u⊙k_t^T)⊗v_t + s[b,head]    // 矩阵乘: k_t^T·v_t
    
    // 3. 更新state (仍在SRAM):
    w_t = exp(-exp(lora_d(ddlerp_d(·))))
    s[b,head] = diag(w_t)·s[b,head] + k_t^T@v_t
    
    // 4. 输出 (receptance + SiLU gate):
    o_t = LayerNorm(r_t @ wkv)
    o_t = SiLU(g_t)⊙o_t
    
  // 5. 写入HBM: o_t ∈ R^{B×h×D/h}
```
对比 time-parallel (associative scan) 方案:
```
// 需要反复HBM↔SRAM传输state中间结果
For each pair of adjacent sequence elements (parallel scan tree):
  Load s_left from HBM → SRAM → merge s_left+s_right → write s_merged to HBM
// 每层scan tree depth=log(T), 每层都需全量state读写
// 总HBM传输量: O(T×log(T)×D²/h)  vs SRAM方案: O(T×D)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源在 RWKV-LM (https://github.com/RWKV/RWKV-LM)。纯 PyTorch 也有 time-parallel 实现（基于 GLA 方法，https://github.com/RWKV/RWKV-infctx-trainer）。性能：16k 序列 Finch kernel 比 Flash Attention v2 快 4.2×，比 Mamba 省 17% 内存、比 Flash Attention 省 40% 内存（A100 80GB, batch=8, D=4096, head=64）。论文指出该 kernel 还有进一步优化空间（algorithmic improvements），留待未来工作。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

---
