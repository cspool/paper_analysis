## Flash Linear Attention (FLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flash Linear Attention (FLA) 是一款用于高效线性注意力模型的开源库（https://github.com/fla-org/flash-linear-attention），由 Songlin Yang 等人开发，使用纯 PyTorch 和 Triton 实现。核心目标：为各类线性注意力/RNN-like 架构提供统一的、硬件高效（GPU Tensor Core 优化）的 Triton kernel 实现。支持的模型包括：RetNet, GLA, Mamba/Mamba2, DeltaNet, GatedDeltaNet, HGRN2, RWKV6/7, GSA 等 20+ 种线性注意力架构。FLA 提供三种计算模式：(1) Parallel — O(L²) 自注意力风格（用于短序列训练验证）；(2) FusedRecurrent — O(L) 纯 recurrent（共享内存/寄存器内维护 hidden state）；(3) FusedChunk/Chunk — O(LC) chunk-wise（推荐训练模式，chunk 内 MatMul + chunk 间 recurrent）。A100 上 benchmark 显示 FusedChunk 在 seq_len=16K 时比 FlashAttention 快约 10×（forward pass）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
FLA 在 AHN 论文中的使用流程：

```
用户在 LLaMA-Factory 中使用 FLA 训练 AHN 的过程:
1. 安装 FLA:
   pip install flash-linear-attention

2. 在模型代码中引入 FLA layer:
   from fla.layers import GatedDeltaNet, DeltaNet, Mamba2

3. 构建 AHN 模块（以 AHN-GDN 为例）:
   class AHN_GatedDeltaNet(nn.Module):
     def __init__(self, hidden_size, num_heads, head_dim):
       self.W_alpha = nn.Linear(hidden_size, num_heads)   # α gate
       self.W_beta = nn.Linear(hidden_size, num_heads)    # β gate
       self.W_gamma = nn.Linear(hidden_size, num_heads)   # γ gate
       self.W_o = nn.Parameter(head_dim, head_dim)        # output projection
       # FLA的GatedDeltaNet kernel处理底层Triton实现

     def forward(self, k_exit, v_exit, h_old, x_t_W):
       alpha = self.W_alpha(x_t_W)  # (B, num_heads)
       beta = self.W_beta(x_t_W)
       # 调用FLA的fused_recurrent kernel:
       h_new = fla.ops.fused_recurrent_gated_delta(
         k_exit, v_exit, alpha, beta, h_old
       )
       return h_new

4. FLA内部Triton kernel执行:
   - FusedChunk模式: 将序列切分为chunks(如512 tokens/chunk)
   - Chunk内: 使用Tensor Core MatMul计算alpha*beta*k^T*v
   - Chunk间: 使用shared memory recurrent传递hidden state
   - WY representation: 将Householder product分解为MatMul-friendly形式
   - 输出: 每token的压缩记忆更新结果

5. 训练循环:
   - 初始化h_0 = 0 (每个head的H×H零矩阵)
   - 每个training step: 输入batch → FLA fused_chunk_forward → h更新
   - Backward: Triton自动生成反向kernel
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
安装：`pip install flash-linear-attention`，要求 PyTorch >= 2.5, Triton >= 3.0。使用场景：任何需要将线性注意力/RNN-like 模型高效运行在 GPU 上的项目，特别是长序列训练。FLA 的 fused_recurrent 模式对推理至关重要——无需 materialize 完整 attention matrix，hidden state 驻留在 shared memory/寄存器中。AHN 论文通过 FLA 实现 AHN-Mamba2/AHN-DN/AHN-GDN 三种实例化的高效训练和推理。局限性：Triton kernel 需针对特定 GPU 架构调优，跨平台（AMD/Intel GPU）兼容性仍在完善。

涉及论文标题：
- Artificial_Hippocampus_Networks_for_Efficient_Long-Context_Modeling

---
