## Vision-to-Text Information Aggregation / 视觉到文本信息汇聚现象

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vision-to-Text Information Aggregation 是 TimeViper 在 Hybrid Mamba-Transformer MLLM 中发现的 token 信息流现象：随着 LLM 层深度增加，视觉 token 中的信息逐步从 vision tokens 汇聚到 instruction tokens（指令中心任务如 MCQ/TVG）或直接贡献到 response tokens（视觉中心任务如 VDC）。在深层 layer，vision tokens 几乎 100% 冗余——完全移除所有 vision tokens 也不影响模型性能。该现象由 information blocking 实验揭示：通过修改 attention mask 阻断 vision→instruction (V2I) 或 vision→response (V2R) 信息流，观察各层各任务的性能变化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Information Blocking Experiment (3 token types: vision, instruction, response)
# Block V2I (vision -> instruction):
# [X_0^{l+1}, X_1^{l+1}, Y^{l+1}] = [[1,0,0],[0,1,0],[1,1,1]] @ [X_0^l, X_1^l, Y^l]
# Block V2R (vision -> response):
# [X_0^{l+1}, X_1^{l+1}, Y^{l+1}] = [[1,0,0],[1,1,0],[0,1,1]] @ [X_0^l, X_1^l, Y^l]
```
Annotations: Instruction-centric tasks (MCQ, TVG) → 浅层阻断 V2I 性能急剧下降，深层几乎无影响（信息已转移）；Vision-centric tasks (VDC) → 阻断 V2R 浅层急剧下降；所有任务深层完全 dropping vision tokens 无性能损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
观察到的现象直接指导 TransV 设计：(1) 浅层保留较多 vision tokens (50% uniform dropping)，因为 vision tokens 在此阶段仍重要；(2) 深层激进 dropping (90%)，信息已转移至 instruction tokens；(3) 通过 cross-attention 显式执行信息转移而非依赖模型隐式学习。这一现象与 Transformer-based MLLM 类似发现一致（LLaVA-Mini, PDrop），但 TimeViper 首次在 hybrid Mamba-Transformer 中验证并利用。方法学源自 What's in the Image (Kaduri et al., CVPR 2025)。

涉及论文标题：
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
