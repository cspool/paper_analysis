## FlexAttention（编译器驱动的Attention Kernel生成编程模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlexAttention 是 PyTorch 2.5+ 引入的编译器驱动的 attention kernel 生成编程模型，由 Meta 提出并集成到 PyTorch 核心。其核心思想是：将 attention 变体的差异抽象为两个用户可定义的 PyTorch callable——`score_mod`（score modification，修改 score 矩阵中的值，如添加 Alibi positional bias）和 `mask_mod`（attention mask，指定哪些 score 设为 -inf，如 causal mask）——然后通过 `torch.compile` 编译栈（TorchDynamo + TorchInductor）自动捕获这些函数并翻译为 Triton 代码块，动态注入到手写的 Triton attention kernel 模板（forward/backward/decoding）中，生成针对特定 attention 变体的优化 fused kernel。用户仅需 5-10 行 PyTorch 代码即可获得与手写 FlashAttention kernel 竞争的性能（0.68x-1.43x vs FAv2），对 FlashAttention 不支持的变体加速 5.49x-8.00x（vs SDPA itemized mask fallback）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
FlexAttention 的编译流程：

1. **前端编程模型**：用户用 PyTorch 定义 `mask_mod(b, h, q_idx, kv_idx) -> bool` 和 `score_mod(score, b, h, q_idx, kv_idx) -> score`。例如 causal mask: `lambda b, h, q, kv: q >= kv`，Alibi bias: `lambda score, b, h, q, kv: score + alibi_bias[h] * (q - kv)`。
2. **图捕获（TorchDynamo）**：`torch.compile` 的 TorchDynamo 拦截 `flex_attention(query, key, value, block_mask=block_mask)` 调用，捕获 score_mod 和 mask_mod 的 PyTorch 计算图。
3. **Lowering（TorchInductor）**：TorchInductor 将捕获的子图翻译为 Triton 原语操作（element-wise arithmetic, comparison, lookup），生成 Triton code blocks。
4. **模板注入**：生成的 Triton 代码块被动态注入到 3 个手写的 Triton attention kernel 模板中——forward（在线 softmax + tiled QK^T GEMM + tiled PV GEMM）、backward（通过 torch.autograd 生成的 score_mod/mask_mod 反向计算）和 decoding（q_len=1 推理专用）。
5. **运行时 kernel 执行**：生成的 Triton kernel 在 GPU 上执行，结合 BlockMask 的 block sparsity 跳过低效计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlexAttention 已合并入 PyTorch 主仓库（https://github.com/pytorch/pytorch），在 PyTorch 2.5+ 中作为 `torch.nn.attention.flex_attention` 模块可用。attention-gym（https://github.com/pytorch-labs/attention-gym）提供示例和可视化工具。

基本使用：
```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=S, KV_LEN=S)
output = flex_attention(query, key, value, block_mask=block_mask)
```

高级使用（组合多个 mask + score modification）：
```python
from torch.nn.attention.flex_attention import and_masks

combined_mask = and_masks(causal_mask, sliding_window_mask)
block_mask = create_block_mask(combined_mask, B, H, S, S)

def alibi_score_mod(score, b, h, q_idx, kv_idx):
    return score + alibi_bias[h] * (q_idx - kv_idx)

output = flex_attention(query, key, value, block_mask=block_mask, score_mod=alibi_score_mod)
```

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
