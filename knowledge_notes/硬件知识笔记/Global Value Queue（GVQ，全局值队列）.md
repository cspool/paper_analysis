## Global Value Queue（GVQ，全局值队列）

术语解释
<GVQ 是全局值预测器的核心数据结构：按程序顺序保存近期（投机或已提交）指令的输出值，供全局预测器按"距离"取 base 值并与 stride 相加生成预测值。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：① 全局值预测的预测公式是"预测值 = 值队列中距离 d 处的 base 值 + stride"——这个"值队列"就是 GVQ，它把程序顺序上不同动态指令的输出串成一个序列，从而表达"指令间的值相关性"；② gDiff 的 GVQ 记录近期 dispatch 指令的投机结果（由 OoO 引擎产生），PC 索引表项的 distance index 指向队列中该指令的 base 值位置；③ gDiff 的弱点：投机值可能因分支/值误预测而错误，且无 squash 策略导致错误历史长期滞留队列、污染后续预测；④ EgDiff 把 GVQ 逻辑分为两段：投机段（32 项，与 LQ 大小一致、索引同步——两者都是循环缓冲、同锁步分配与 squash）用于预测，非投机段（32 项，匹配预测器 order）保存已提交值用于更新预测器；⑤ 预测值与 base 值就绪状态由条目字段跟踪（value、distance、diff、VP、misp、available、wake）。
- 网页佐证：GVQ 概念源自 gDiff（ISCA 2003，Zhou et al., ACM DL "Detecting global stride locality in value streams"）；EgDiff 论文（IEEE CAL 2025 及本 ISCA'26 版）对其重设计（投机/非投机双段）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件角色：GVQ 是 OoO 后端的一条附加缓冲，与 LQ 同步更新：① renamed load 指令 dispatch 进 LQ 时，在投机段分配条目（结构与 LQ 对齐）；② load 完成执行、更新 LQ 时，对应投机条目同步写入投机结果；③ commit 时，已提交值按 FIFO 顺序从投机段传播到非投机段；④ pipeline squash 时，投机段中比 squash 点年轻的条目全部失效，非投机段保持不动。
- 预测/延迟预测流程（论文表 I 例子）：① 指令 2 的 base 值位于条目 5（尚未就绪）→ 条目 2 存 distance=3、diff=20、VP=0，并把"2"写入条目 5 的 Wake 字段；② 指令 5 结果就绪写入 GVQ → 系统读条目 5 的 Wake 找到条目 2 → 校验条目 2 的 distance 仍匹配（3）→ 用新 base + diff 计算预测值、写入 PRF 并广播 wakeup 唤醒依赖指令；③ 若条目 2 在条目 5 之前就完成（Avail 置 1），唤醒时忽略，避免过时绑定复活；绑定阶段目标条目已被占则丢弃新绑定（保持一对一）。
- 面积：GVQ 64 项 × 141 bit（64-bit value + VP/misp/available/distance/diff/wake 元数据）= 1.1KB；随机访问循环缓冲（类似 load queue），预测读单条目、更新至多两位置（distance 与 distance-1），无相联查找。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：随机访问循环缓冲（FIFO 语义），投机段与 LQ 同构对齐（同大小、同分配/回滚时机）；非投机段接收提交值；每项含 value 与元数据字段（VP、misp、available、distance、diff、wake index）；物理成本用 CACTI 7.0 @22nm 评估（预测表+GVQ 合计 0.078mm²，占处理器面积 0.22%）。
- 使用方式：在 gem5 O3 CPU（src/cpu/valuepred/ 框架）中实现"GVQ 与 LQ 锁步"的同步逻辑（dispatch 分配、执行更新、commit 传播、squash 回滚），配合 EgDiff 预测表与距离轮询更新算法跑 SPEC CPU 2017；论文未提供公开代码（无法确认）。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
