## Image Contribution (IC)（图像贡献度指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Image Contribution (IC) 是 MMTok 提出的量化评估指标，衡量视觉信息对多模态任务答案的相对贡献。定义为 IC = (Perf_All - Perf_0) / Perf_0，其中 Perf_All 是使用全部 vision tokens 时的性能，Perf_0 是完全不提供 vision tokens（text-only）时的性能。IC 越高，说明该 benchmark 越依赖视觉信息；IC 越低，则该任务主要通过语言先验/文本信息即可解决，vision token selection 的效果难以体现。MMTok 发现 LLaVA-1.5-7B 在 MMMU 上 IC=0.089、ScienceQA 上 IC=0.094 —— 即仅用文本就能达到 90%+ 的全 token 性能，因而在这些低 IC 数据集上评估 token selection 方法会严重低估差异。MMTok 据此筛选出 5 个高 IC 数据集（MMB IC=2.35, POPE IC=0.92, MME IC=0.92, SEED-I IC=0.79, GQA IC=0.64）和 LLaVA-NeXT 额外 TextVQA (IC=0.62) 用于核心评估，使实验结果更能反映 token selection 方法的真实差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IC 计算流程：
```
# 对每个 benchmark dataset D
# Step 1: Full token 性能
Perf_All = evaluate(VLM, D, use_all_vision_tokens=True)

# Step 2: Zero token (text-only) 性能
Perf_0 = evaluate(VLM, D, use_all_vision_tokens=False)
          # 仅输入 text tokens, vision 侧置零或完全移除

# Step 3: 计算 IC
IC = (Perf_All - Perf_0) / Perf_0

# Step 4: 分类 — 高 IC 数据集用于 token selection 评估
if IC > threshold:   # 论文未明确给出阈值, 从 Table 4 看 ~0.4+
    mark_as_high_IC(D)
```

应用示例（LLaVA-1.5-7B, Table 4）：
| Dataset | Perf_All | Perf_0 | IC | 分类 |
|---------|----------|--------|-----|------|
| MMB | 64.7 | 19.33 | 2.347 | High IC |
| POPE | 85.9 | 44.64 | 0.924 | High IC |
| MME | 1862 | 970.89 | 0.918 | High IC |
| SEED-I | 66.14 | 37.03 | 0.786 | High IC |
| GQA | 61.9 | 37.65 | 0.644 | High IC |
| TextVQA | 58.2 | 41.66 | 0.397 | Borderline |
| SQA | 69.5 | 63.51 | 0.094 | Low IC |
| MMMU | 36.3 | 33.33 | 0.089 | Low IC |

在 Low IC 任务上，即使保留 0 个 vision token，性能下降也不显著；因此 token selection 方法的差异被压缩。MMTok 建议仅在高 IC 任务上评估 token selection 质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IC 实现简单：对每个 benchmark，运行两次完整评估（all tokens + text-only），计算相对增益。在 Lmms-eval 框架中可通过配置 `--model_args` 控制视觉 token 的提供方式。IC 的使用场景：(1) Benchmark 筛选：在评估新 token selection 方法前，先计算各 benchmark 的 IC，仅在高 IC 任务上比较；(2) 方法诊断：若方法在低 IC 任务上也表现好，说明方法可能通过更好的 vision-unrelated 决策而非更好的 vision token selection 获得提升；(3) 任务分析：帮助研究者理解哪些 VLM 任务真正需要视觉信息，指导 VLM 架构设计。局限性：IC 依赖于具体的 VLM（不同模型对零 vision token 的鲁棒性不同），需要 per-model 计算。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs
