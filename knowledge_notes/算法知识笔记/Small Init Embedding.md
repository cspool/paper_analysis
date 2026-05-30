## Small Init Embedding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Small Init Embedding 是 RWKV 原始论文提出的一种嵌入层初始化策略：将 token embedding 矩阵初始化为极小值（U(±1e-4) 均匀分布，而非标准的 N(0, 0.02) 正态分布），并在 embedding 后立即加一个额外的 LayerNorm。论文观察到标准 Transformer 训练初期 embedding 矩阵变化缓慢，模型难以从初始噪声状态快速脱离。极小初始化使 embedding 值接近零，经过 LayerNorm 后因输入值小而梯度方向变化剧烈——一步微小的参数更新即可产生大幅方向改变，加速收敛。实验验证（Figure 9）：使用 small init emb 的训练 loss 下降速度和最终收敛均优于标准初始化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Standard (GPT/BERT): Embedding ~ N(0, 0.02), 无额外 LayerNorm
x = Embedding(token_ids)          # ~N(0,0.02), 初始值分散
x = x + PositionalEncoding        # 或 RoPE
→ Transformer blocks...

# RWKV Small Init Embedding:
x = Embedding(token_ids)          # ~U(±1e-4), 初始值接近零
x = LayerNorm(x)                  # 额外 LayerNorm：放大梯度方向变化
→ RWKV blocks...
```
关键机制：当 embedding 值极小时，LayerNorm 的输入均值和方差接近零，微小的参数梯度变化经过 LayerNorm 的除法（除以接近零的标准差）后被显著放大，导致 embedding 快速重组到有意义的表示空间。论文附注指出实验中使用的是 U(±1e-4) 而非 RWKV 实际使用的 N(0, 1e-4)，但差异可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`nn.Embedding(vocab_size, dim).weight.data.uniform_(-1e-4, 1e-4)`，后接 `nn.LayerNorm(dim)`。适用于深度 post-LN 架构的训练加速，尤其当 embedding 维度较大时效果显著。论文通过对比实验（batch size=400）验证了小初始化嵌入 + 额外 LayerNorm 相比标准正态初始化的 loss 收敛加速效果。该策略随后被 Eagle/Finch 等 RWKV 后续版本继承。

涉及论文标题：
- RWKV__Reinventing_RNNs_for_the_Transformer_Era

---
