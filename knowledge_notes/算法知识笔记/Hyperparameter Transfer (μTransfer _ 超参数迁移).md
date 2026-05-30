## Hyperparameter Transfer (μTransfer / 超参数迁移)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hyperparameter Transfer 是 LongCat-Flash 使用的大模型超参数选择策略。核心思想：在小 proxy model 上搜索最优超参数（初始化方差 σ² 和学习率 η），然后通过理论推导的 scaling rules 将这些超参数迁移到 target 大模型，避免在大模型上直接进行昂贵的超参数搜索。

LongCat-Flash 采用 Standard Parameterization (SP) 下的 "Adam LR Full Align" scaling rules [Everett et al., 2024]。对于 width scaling factor $s = n_{\text{target}}/n_{\text{proxy}} = 6144/768 = 8$，迁移规则：
- Embedding layer: $\sigma^2$ 和 $\eta$ 直接迁移（不变）
- Hidden/Unembedding layers: $\sigma^2_{\text{target}} = \sigma^2_{\text{proxy}}/s$, $\eta_{\text{target}} = \eta_{\text{proxy}}/s$
- 所有其他属性（depth, sparsity, batch size）在迁移中保持不变

LongCat-Flash 选择 proxy width=768（s=8），认为这个比例在计算效率和迁移精度间取得了最佳平衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LongCat-Flash 中：(1) 用 s=8 的 proxy 模型搜索最优 σ² 和 η；(2) 按 Table 1 的规则逐层映射到 target model；(3) 在 proxy 上训练极小规模（<1B activated params）即可完成搜索，计算开销远小于 target-scale 搜索。

涉及论文标题：
- LongCat-Flash Technical Report
