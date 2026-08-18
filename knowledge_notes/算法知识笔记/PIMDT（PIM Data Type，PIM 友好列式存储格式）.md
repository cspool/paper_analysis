## PIMDT（PIM Data Type，PIM 友好列式存储格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIMDT 是论文为 BLIMP（bank 级 PIM）OLAP 数据库提出的列式存储数据类型：把指定的数据库列以 "PIM 友好" 布局常驻存储——整字（如 64-bit）落在单个 DRAM bank 内、BLIMP 核可直接经本地 row buffer 访问，从而避免查询时的软件 relayout。其他 "host 列" 保持原布局。通过 SQL 列约束声明，如 `Bar bigint NOT NULL PIMDT(BLIMP)`。设计约束：(1) 只能用于定长类型（可变长字符串、blob 与 PIMDT 不兼容，因其必须能按元素边界快速 chunk 化）；(2) 列上的算子必须 PIM-amenable（数据并行、可向量化、无跨数据依赖）；(3) 更新/插入需按字节重排（每个插入字节一次写）；(4) 查询时按 configurable chunk 大小把 PIMDT 数据均分到各 bank，满足 32MB bank 容量约束（除列数据外还要容纳算子指令、输入与输出数据）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文流程（SSB SF100）：LINEORDER 的外键列（lo_orderdate、lo_partkey、lo_suppkey、lo_custkey）与过滤列（lo_quantity、lo_discount）声明为 PIMDT 常驻；查询时存储管理器只把"预铺好"的 PIMDT 列分区 relayout 载入各 bank（无需整列查询时重排）→ 算子内核就地执行 → 结果位图/部分物化留在 bank 内链给下一算子 → 最后 host 取回。伪代码级（chunk 化与加载）：
```
# 查询执行前（存储管理器）
for col in query.PIMDT_columns:
    chunks = chunk_by_element_boundary(col, bank_capacity=32MB - reserved)
    # 每 chunk 的元素在单 bank 内连续，host 只做一次 relayout 载入
# 执行时：每 bank 载入 kernel + PIMDT chunk + 辅助数据 → BLIMP 核计算
```
选择 PIMDT 列的判据类似传统索引创建决策：列被查询频率（存储权衡）与 PIM 适性（运行时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文在存储管理层实现 PIMDT（storage manager 判断 offload 所需数据、减少 relayout 依赖、复用已 relayout 数据），查询解析器需解析 PIMDT 列约束语义；无法在 PIM 执行的算子回退 host（此时 PIMDT 列需 relayout 回 host 布局）。端到端效果：PIM 感知规划（含 PIMDT + 晚物化 + 低选择性优先 join 序）比隔离算子外推快 3.2×；隔离计划平均 22% 查询时间在 relayout。论文未声明 PIMDT 实现开源（论文未明确说明）；同组 BLIMP 框架 dovedevic/blimp（https://github.com/dovedevic/blimp）含 /relayout 例程。

涉及论文标题：
- Taking Analytic Databases to the Bank
