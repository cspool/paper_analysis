## Expert Activation Matrix（专家激活矩阵）

术语是什么？
Expert Activation Matrix是MoE-Infinity系统中用于记录每个request历史expert激活情况的数据结构。它是一个request-level的矩阵，行对应历史请求，列对应experts，矩阵元素表示某请求过去是否激活了某expert（binary或hit count）。新请求到达时，系统基于该矩阵做expert prediction→同步prefetch predicted experts→执行forward。FineMoE指出该矩阵的局限性：request-level粗粒度聚合冲淡了iteration-level清晰模式（entropy分析显示coarse-grained patterns比fine-grained更难预测）。

从系统架构角度拆解术语：
```
# MoE-Infinity Expert Activation Matrix：
# 行=历史requests, 列=experts (所有layers的expert被展平)
         E0_L0  E1_L0  ...  E0_L1  E1_L1  ...
Req_1      1      0    ...    0      1    ...
Req_2      0      1    ...    1      0    ...
...

# 新请求进入时：
# 1. 查找该请求的历史expert激活记录
# 2. 预测该请求可能激活的expert
# 3. 同步prefetch predicted experts（blocking forward）
# 4. 执行MoE forward
```
对比FineMoE的Expert Map：Map Store记录per-iteration、per-layer的完整概率分布（非binary hit），并存储semantic embedding和trajectory用于相似度检索。

术语一般如何实现？
在MoE-Infinity中用Python字典/张量存储request-to-expert激活映射。每次inference完成后更新该request的激活记录。FineMoE实验中在评测前准备MoE-Infinity对应历史矩阵以公平比较。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
