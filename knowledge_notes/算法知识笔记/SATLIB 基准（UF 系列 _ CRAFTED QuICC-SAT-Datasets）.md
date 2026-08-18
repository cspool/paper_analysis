## SATLIB 基准（UF 系列 / CRAFTED QuICC-SAT-Datasets）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SATLIB 是 SAT 研究领域公开的基准库（Hoos & Stützle，SAT 2000，http://www.satlib.org），提供大量随机与结构化的 SAT 实例，是 SAT 求解器评测的标准资源。其 UF 系列（Uniform Random 3-SAT）是位于相变区的随机 3SAT 实例（如 UF20：20 变量/91 子句、UF50：50/218、UF75：75/325、UF100：100/430、UF125：125/538、UF150：150/645、UF175：175/753、UF200：200/860、UF225：225/960、UF250：250/1065），naming 按变量数区分，公认难解。QuICC-SAT-Datasets（UMD-ARLIS，https://github.com/UMD-ARLIS/QuICC-SAT-Datasets）是另一组面向量子启发求解器评估的 SAT 基准（含 quiet planting 型 CRAFTED 与 AI planning 型 CRAFTED）。SATIC 论文将 SATLIB UF 系列作为 unseen（未见）测试集验证泛化性，将 CRAFTED 系列作为 seen（定制压力）测试集。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
基准使用流程（SATIC 评估 pipeline）：
```
# 批次（batch）= 同配置（k, n, m）的实例组；实例 = 单个 SAT 问题
for batch in [Batch-4-50-500, ..., UF20, UF50, ..., UF250]:   # Table III
    for inst in batch:                          # 每批 ≥50 实例
        for rep in 1..120:                      # repeats ≥100 保证统计
            (sol, ok) = SATIC_compile_and_run(inst, max_iter=50K)  # 迭代=硬件调用数
            if ok: repeats_success += 1
        solved[inst] = (repeats_success > 0)    # 任一次成功即算解出
    batch_solved = all(solved[inst] for inst)
```
SATLIB UF 系列被用作 unseen 测试（训练/调参时未见），CRAFTED seen 系列用于探测容量上限（Ratio 列 = QUBO 规模/45-spin 容量，如 Batch-4-150-1570 达 73.1）。评估结果：SATIC++ 在 UF175（23× 容量）之前全解，UF200/225/250 解 98/97/92。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SATLIB 提供 DIMACS CNF 格式文件（网页/镜像下载），UF 系列来自 uniform random generator 的固定种子生成；QuICC-SAT-Datasets 提供 GitHub 仓库（按 quiet planting、AI planning 等类别组织，含 generator 与实例）。使用：评测器按批次-实例-重复的三层结构运行（论文定义 batch/instance/repeats/iteration count 指标），用 TTS 与 solved 比例比较不同编译器/分解器（SATIC vs SATIC++ vs D-Wave EID）；也是很多经典与量子求解器论文的标准对照。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
