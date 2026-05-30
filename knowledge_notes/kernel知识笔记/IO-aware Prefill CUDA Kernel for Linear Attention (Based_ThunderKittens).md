## IO-aware Prefill CUDA Kernel for Linear Attention (Based/ThunderKittens)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IO-aware prefill CUDA kernel是ThunderKittens框架中为Based线性注意力架构设计的自定义CUDA kernel，通过warp-register分区存储矩阵值recurrent state（KV-state ∈ R^{d×d̃}，d̃≈273 for Taylor 2nd-order feature map）避免HBM↔SRAM反复传输，实现IO最优的prefill计算。核心策略：将KV-state矩阵分片存储在各warp的register file中（而非global memory），在prefill阶段沿序列维度扫描时仅在register中累加更新state，最终一次性写回HBM。JRT论文扩展此kernel支持Prefix Linear Attention (PLA)：第一次调用fnbased(k_e,v_e)用非因果sum计算encoder KV-state存于寄存器A0/A1/A2（对应Taylor 0/1/2阶项），第二次调用fnbased(q_d,k_d,v_d)从该register状态续算decoder输出写SRAM→HBM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// ThunderKittens Based kernel寄存器管理 (per warp):
Register A0: stores 0th-order KV-state contribution  // Σ 1·v
Register A1: stores 1st-order KV-state contribution  // Σ k·v (linear term)
Register A2: stores 2nd-order KV-state contribution  // Σ vec(k⊗k)·v (quadratic term)

// JRT-RNN扩展 (Algorithm 2 in JRT paper):
1. 初始化SRAM buffer和register fragments
2. fnbased(k_e, v_e):  // Encoder prefill
   - 使用非因果sum (而非causal cumsum)
   - 不乘queries (与原Based kernel不同)
   - KV-state = Σ_{j=1}^{M} (k_e[j]^T v_e[j])
   - 结果存于寄存器A0/A1/A2
3. fnbased(q_d, k_d, v_d):  // Decoder prefill  
   - 从encoder初始化的register state续算
   - KV-state_dec = encoder_state + Σ_{j=1}^{i} (k_d[j]^T v_d[j])
   - K-state_dec = encoder_k_sum + Σ_{j=1}^{i} k_d[j]
   - y_i = (q_d[i]·KV-state_dec) / (q_d[i]·K-state_dec) 写入SRAM
4. Store y from SRAM → HBM

// FLOPS (per layer, B=batch, N=seqlen, H=heads, D=head_dim, d=model_dim):
// Causal LA: 2BNHD(feature map) + 4BNHdD(KV dot+cumsum+Q dot+D sum)
// PLA add:   BMHD(k_e feature map) + 3BMHdD(k_e·v_e + D sum + state merge)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/HazyResearch/ThunderKittens（Based kernel），PLA扩展：https://github.com/HazyResearch/prefix-linear-attention。性能（H100）：N=32768/B=16时JRT-RNN CUDA 5.6ms vs FA2 107.8ms (19.2× faster)，vs FLA Triton 123.7ms (22.0× faster)。JRT-Prompt CUDA (2N prefill): 9.0ms → 11.9× > FA2, 13.7× > FLA。PLA decode每token O(1)无修改。ThunderKittens framework仅~282-316 lines per kernel (vs Triton 89-104)，实现14×于FLA Triton的线性注意力加速。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---
