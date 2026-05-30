## Model Growth Initialization (Layer Stacking / 模型增长初始化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model Growth Initialization 是 LongCat-Flash 使用的大模型参数初始化策略。核心思想：先训练一个小规模的"前身模型"（predecessor model），然后通过 layer stacking（层堆叠）将小模型扩展为目标大模型。具体地，LongCat-Flash 先训练一个 14-layer（half-scale）的模型（与目标模型架构完全一致），然后用 expansion rate r=2 将 14 layers 堆叠为 28 layers，作为 560B target model 的初始化。

公式：$$L_{\text{target}} = \underbrace{L_{\text{small}} \circ L_{\text{small}} \circ \cdots \circ L_{\text{small}}}_{r}$$ 其中 $L_{\text{small}}$ 是从 token embedding 到 final hidden states 的变换，$L_{\text{target}}$ 是堆叠 r 份复本后的大模型变换。

LongCat-Flash 实验（Figure 5b）显示 model growth 初始化的典型 loss 轨迹：初期 loss 短暂上升（因参数翻倍导致的不一致）→ 随后加速收敛 → 最终 outperform random initialization baseline。推测两个因素：(1) 小模型收敛更快→提供更高质量的初始参数；(2) Growth 操作作为 implicit regularization 防止参数坍塌。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Model Growth Pipeline

# Step 1: 训练 Predecessor Model
small_model = ScMoE(n_layers=14, d_model=6144, ...)
small_model.train(data=first_segment_of_20T_tokens)
checkpoint = save(small_model)  # 保留 optimizer state, LR schedule, sample counter

# Step 2: 构造 Target Model (Layer Stacking, r=2)
target_model = ScMoE(n_layers=28, d_model=6144, ...)  # 架构与 small_model 一致，仅 depth 翻倍
for i in range(14):
    target_model.layers[2*i] = copy(small_model.layers[i])
    target_model.layers[2*i + 1] = copy(small_model.layers[i])
# Embedding/Unembedding: 直接继承

# Step 3: 恢复状态继续训练
target_model.load_optimizer_state(checkpoint.optimizer_state)  # optimizer state 被扩展
target_model.load_lr_schedule(checkpoint.lr_schedule)
target_model.load_sample_counter(checkpoint.sample_counter)  # 从 predecessor 的训练进度继续
target_model.train(data=remaining_tokens)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 与相关方法（Net2Net, bert2BERT, LiGO, SOLAR）相比，LongCat-Flash 使用最简单的 layer stacking (Du et al., 2024; Kim et al., 2023)，即直接复制层而非训练更复杂的 growth operator。
2. 关键实践：(1) 保留所有训练状态（optimizer states, LR schedule, sample counter）而非仅保留模型参数；(2) Expansion rate r=2 (depth doubling) 而非更大的 r——过大的 r 可能导致更严重的初期性能退化；(3) Over-optimizing predecessor 会降低 target model 的 token efficiency——需在适当时间点执行 growth（LongCat-Flash 在 tens of billions tokens 后执行）。
3. Predecessor 架构必须与 target 一致（相同的 d_model, expert 数量, MLA 配置等），仅 depth 不同。

涉及论文标题：
- LongCat-Flash Technical Report
