## Block-wise Attention Mask for Diffusion dLLM (M_BD / M_OBC / M_BC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-wise Attention Mask是Fast-dLLM v2中用于block diffusion训练的自定义注意力掩码，将noised序列x_t和clean序列x_0沿sequence维度拼接（总长2L），应用hybrid attention pattern M_full ∈ {0,1}^{2L×2L}，分解为四个子掩码区域：

M_full = [[M_BD, M_OBC], [0, M_BC]]

其中：
- **M_BD (Block-diagonal mask)**：x_t内部的块内双向自注意力。同一block内token互相可见，支持block内的masked token refinement。矩阵仅对角block为1。
- **M_OBC (Offset block-causal mask)**：x_t → x_0的跨序列注意力。每个noised token可attend到前面block的clean token，保持块间causal conditioning。矩阵为上三角block结构。
- **M_BC (Block-causal mask)**：x_0内部的自回归式注意力。clean token可attend到同block及之前block的token，保持AR-like progression。
- **0 (左下角)**：x_0不能attend到x_t（clean不应看到noise），保证训练的信息流向正确。

推理时简化为：已解码block（x_0的前缀）作为cached prefix只读，当前block x_t^b双向自注意力+对prefix的causal attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 训练时序列拼接
Input = concat([x_t, x_0])           # 总长2L
# x_t: noised sequence (部分[MASK])
# x_0: clean sequence (原始token)

# Attention mask M_full ∈ {0,1}^{2L×2L}
M_BD[i][j] = 1 iff block(i)==block(j)  # x_t内部块内双向
M_OBC[i][j] = 1 iff block(j) < block(i) # x_t看clean历史块
M_BC[i][j] = 1 iff block(j) <= block(i)  # x_0内部块因果

# 推理时简化mask:
# 已解码块作为prefix → 缓存K/V → 仅当前noised block计算
# 当前block: bidirection自注意力 + causal attend to prefix
```

使用PyTorch flex-attention实现，避免手动构造完整2L×2L的mask矩阵（内存O(L²)），而是通过自定义score_mod函数在attention计算时动态决定哪些position pair可见。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现基于PyTorch的`torch.nn.attention.flex_attention.flex_attention` API（或旧版`F.scaled_dot_product_attention`的custom mask参数）。自定义`score_mod`函数实现四种mask逻辑。key insight：这个mask设计与AR模型的causal attention高度接近（仅将同一block内从causal改为bidirectional），因此预训练AR模型只需少量微调即可适应——这是Fast-dLLM v2仅需~1B tokens微调的关键原因。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM
