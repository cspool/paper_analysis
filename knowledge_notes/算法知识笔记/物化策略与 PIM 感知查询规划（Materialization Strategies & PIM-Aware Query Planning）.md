## 物化策略与 PIM 感知查询规划（Materialization Strategies & PIM-Aware Query Planning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DBMS 查询执行中的中间结果物化策略与查询规划启发式。传统（CPU 中心）列存 DBMS：select 输出 index array（随选择性线性增长）或 bitvector/bitmask（定长）；用位图可简化算子间流水（AND 合并两个过滤），但其他算子不接受位图输入时被迫物化（materialization）。CPU 侧启发式：①高选择性谓词先做（减少后续算子处理记录数）；②最小化哈希表大小（塞进 cache）；③左深 join 树。论文发现这些启发式在 BLIMP（bank 级 PIM）上失效或反转：(1) 哈希表 >几 KB 后大小不再影响 PIM 性能（probe 是随机 row buffer 访问、命中率低），只有选择性驱动 join 排序——CPU 推荐的 SDC 序在 BLIMP 上是 6 种序中第 4 差（比最优 CSD 慢 23%）；(2) 物化时机决定 compute domain 转换与 relayout 次数——Early Mat.（全部在 BLIMP 物化）、Hybrid Mat.（只在 domain 转换前物化）、Late Mat.（host 物化），bitvector 定长 vs value array 随选择性线性增长使成本随选择性带变化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
示例查询（SSB Q1.1 SF100）`WHERE Fizz < 25 AND Buzz BETWEEN 1 AND 3` 的五种计划：①Host——两 PIMDT 列先整列 relayout 回 host 再评估（最贵）；②Isolated——两个 BLIMP select 各自隔离执行、结果分别 relayout 回 host 由 host 做 AND（多数 PIM 研究做法；relayout 浪费严重）；③Early Mat.——两个 select 输出数组都留在 bank 内，再派发一个 BLIMP 逻辑 AND 算子合并，最后才回 host（relayout 最少但第一个过滤未让第二个过滤少处理记录）；④Hybrid Mat.——第一个过滤输出 bitvector 留在 PIM，第二个过滤并行处理时隐式 AND，只在 domain 转换前物化（最优区域）；⑤Late Mat.——同 ④ 但全部由 host 物化（适合高选择性，value array 小）。伪代码级：
```
# PIM 感知规划（论文方法）
plan = []
for op in query.operators (PIMDT 列上):
    if op 支持在 BLIMP 执行:
        plan.append(dispatch_blimp(op))          # 预处理→relayout→执行→部分物化/原位保留
    else: plan.append(dispatch_host(op))          # 回退 host
join_order = sort(joins, key=selectivity_asc)     # 低选择性优先（PIM 主驱动）
materialize = "late" if 整体高选择性 else "hybrid" # bitvector 定长 vs value array 线性
```
结果：PIM-optimal 比 Isolated 快 3.2×（平均 22% 时间省在 relayout）、比 CPU-optimal 快平均 28%（最大 40% Q3.3）；bushy join 树（Q4.3）需多次重建哈希表、重复 relayout+build+broadcast，不适合 PIM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文用手工按 PIM 启发式构造查询计划（自动规划器留作 future work），查询执行器支持"PIMDT 列上的算子→BLIMP 工作流、否则回退 host"；解析与分析沿用常规 DBMS（仅额外解析 PIMDT 列约束）。CPU 侧基线用 DuckDB 生成计划（含 bushy join 的 Q4.3）与手调 C++ 单块 kernel。该策略源于列存数据库物化研究（Abadi et al. 2007 物化策略、Vetica late materialization），在 PIM 上下文中重新评估其价值。

涉及论文标题：
- Taking Analytic Databases to the Bank
