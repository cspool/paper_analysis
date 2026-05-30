## Expert Popularity-based GPU Placement

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Popularity-based GPU Placement 是一种 MoE 推理的 GPU 显存分配优化策略：在模型加载（initialization）阶段，使用 calibration data 对各 expert 的激活频率进行离线 profiling，然后按热门度降序将尽可能多的 expert 放入 GPU memory（不超过显存容量），其余 expert 放在 CPU memory。目标是最大化 GPU expert cache 的 hit rate——即推理时所需 expert 已在 GPU 显存中的概率。

Fiddler 在 Mixtral-8x7B (256 experts) 上的 profiling 数据显示：expert 热门度分布相对均衡（mean=0.71, std=0.08, 25th percentile=0.67, 75th percentile=0.76），但仍有足够差异使得热门度导向放置比随机放置提升 hit rate 3-5 个百分点（Env1: 25.2% vs 21.9% random, Env2: 53.0% vs 48.8% random）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Fiddler 中的 Expert Popularity-based GPU Placement 流程：

```
┌─────────────────────────────────────────────────────────┐
│ Offline Profiling (一次性)                                │
│   1. 加载 Mixtral-8x7B + ShareGPT calibration data       │
│   2. 前向推理 calibration 样本，记录各 expert 的激活次数   │
│   3. 输出: popularity[layer][expert] ∈ [0, 1]           │
│      (value = 激活次数 / 最热门 expert 激活次数)           │
│   4. 全局排序: 所有 256 experts 按 popularity 降序排列     │
├─────────────────────────────────────────────────────────┤
│ Initialization (每次加载模型)                              │
│   1. 计算 GPU 可容纳的 expert 数:                         │
│      N_gpu = floor((VRAM - non_expert_size) / expert_size)│
│   2. 选择 top-N_gpu 热门 expert 放入 GPU memory           │
│   3. 其余 256 - N_gpu expert 放入 CPU pinned memory       │
├─────────────────────────────────────────────────────────┤
│ Runtime (每次推理)                                        │
│   is_at_gpu(layer, expert) = expert in top-N_gpu         │
│   → Strategy (a) 概率 = hit rate                         │
│   → Strategy (b)/(c) 概率 = 1 - hit rate                 │
└─────────────────────────────────────────────────────────┘
```

与随机放置的 hit rate 对比（256 experts total）：
| Environment | GPU experts | Random hit rate | Popularity hit rate | Gain |
|-------------|-----------|-----------------|---------------------|------|
| Env1 (24GB) | 56/256 | 21.9% | 25.2% | +3.3pp |
| Env2 (48GB) | 125/256 | 48.8% | 53.0% | +4.2pp |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Calibration data**：论文使用 ShareGPT 对话数据集做 expert 热门度 profiling（Appendix C）
- **Robustness**：论文在 LMSYS-Chat-1M 不同数据集上验证了方法的鲁棒性（Fiddler 仍比 llama.cpp 快 1.56×）
- **Assumption**：expert 选择基于 token 特征，expert 热门度在不同输入领域间近乎通用（引用自 Mixtral 和 OpenMoE 论文），因此 offline profiling 数据可在不同下游任务间复用
- **对比其他方法**：
  - llama.cpp 按层分配 GPU/CPU（前 ngl 层全 GPU），不区分 expert 热门度
  - Mixtral-Offloading 使用 LRU cache（runtime 动态），Fiddler 使用 static popularity（init-time 固定）——两者正交互补
- **与 Algorithm 1 的关系**：Popularity placement 提高 is_at_gpu() 返回 true 的概率（即 Strategy (a) 的使用频率），减少进入 Strategy (b)/(c) 的需要

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
