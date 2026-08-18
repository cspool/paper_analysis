## Time to Solution（TTS，求解时间指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Time to Solution（TTS）是随机/随机化求解器（如 Ising 机、量子退火器、Tabu）的标准性能指标：在目标置信水平下求解给定实例的期望时间。随机求解器每次运行有成功概率 p，需多次重复；TTS 把"成功概率"与"每次运行时间"折算成期望时间，TTS 越小越好。SATIC 论文用 95/100 独立 repeats 成功的置信水平（即求解概率 ≥0.95），以迭代次数（硬件调用数）计 TTS 作为主要成本度量之一（避免单次墙钟测量的噪声与实现差异），并同时报告端到端 runtime 分解。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
TTS 计算流程（SATIC 评估方法）：
```
# 每实例：120 repeats × 50K 迭代预算
for rep in 1..120:
    (sol, ok) ← run_solver(instance, budget=50K)     # ok = 任一迭代 CheckSolution 通过
    if ok: success_reps += 1
p_success ← success_reps / 120
# TTS（迭代口径，95% 置信）：
#   TTS ≈ t_iter * ln(0.05) / ln(1 - p_success)   # 期望求解时间；p<0.95 时按目标置信外推
# 报告平均迭代次数找解（比较 baseline 时用，因每迭代时间近似恒定）
```
比较结果：SATIC++ 平均 9.4 迭代解出 10 个基准实例（最接近 baseline 约 250 迭代，D-Wave EID 常触 500 迭代上限失败）；TTS 图（Fig.13）显示 SATIC++ 在 UF 系列上远低于 D-Wave EID 与 SATIC。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：重复运行统计成功概率 + 迭代/时间计量，按置信水平折算（对数公式）；SATIC 以迭代数（硬件调用）为 TTS 单位并补充端到端 runtime（Software+Hardware Prep+Hardware+PCIe+Preprocessing 分解）。使用：随机求解器研究的标准对比指标（论文引用 [59][60]）；配合 solved instances / successful repeats / iteration count 一起评估编译器质量与可扩展性。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
