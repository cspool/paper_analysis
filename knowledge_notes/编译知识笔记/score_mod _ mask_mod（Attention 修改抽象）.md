## score_mod / mask_mod（Attention 修改抽象）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
`score_mod` 和 `mask_mod` 是 FlexAttention 的两个核心用户编程接口，用于统一表达各种 attention 变体。两者均接收 batch index `b`、head index `h`、query position `q_idx` 和 key-value position `kv_idx` 四个位置参数：

- **mask_mod**: `(b, h, q_idx, kv_idx) -> bool`。返回 `True` 表示保留该 score，`False` 表示将该 score 设为 `-inf`（完全屏蔽）。适用于因果 mask（`q_idx >= kv_idx`）、滑动窗口（`q_idx - kv_idx <= window`）、文档 mask（`doc_id[q_idx] == doc_id[kv_idx]`）等 boolean 模式。
- **score_mod**: `(score, b, h, q_idx, kv_idx) -> score`。接收当前 score 标量值，返回修改后的值。适用于 Alibi bias（`score + bias[h] * (q_idx - kv_idx)`）、soft-capping（`tanh(score) * cap`）等连续值修改。

为什么要区分两者？虽然 mask_mod 是 score_mod 的特例（乘以 0 或 1），但分离出 mask_mod 有两个原因：(1) score_mod 对每个 score 标量执行昂贵的修改，而 mask_mod 转换为 score_mod 需要额外的乘法开销；(2) mask_mod 提供了额外的语义信息——某些 score 计算可以被完全跳过（block sparsity optimization）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 FlexAttention 编译流程中：
1. 用户定义 mask_mod 和 score_mod 为纯 PyTorch 函数。
2. TorchDynamo 在 `torch.compile` 期间捕获这两个函数的计算图。
3. TorchInductor 将两个函数的图分别翻译为 Triton 代码块。
4. 编译时，`create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN)` 通过 `torch.vmap` 对所有的 (q_block, kv_block) pair 批量评估 mask_mod，生成 BlockMask。
5. 运行时，score_mod 代码块注入到所有 visible block 的 QK^T GEMM 之后；mask_mod 代码块仅注入到 Partial Block（部分被 mask 的 block），Full Block 跳过 mask_mod 以节省 15% 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
示例——定义 PrefixLM（causal + prefix 组合）：
```python
def causal_mask(b, h, q, kv):
    return q >= kv

def prefix_mask(b, h, q, kv):
    return kv < prefix_len  # prefix tokens visible to all

from torch.nn.attention.flex_attention import or_masks
prefix_lm_mask = or_masks(causal_mask, prefix_mask)

def alibi_score_mod(score, b, h, q, kv):
    return score + bias[h] * (q - kv)
```

涉及论文标题：
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
