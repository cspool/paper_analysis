## gDiff 全局值预测器与 EgDiff（全局 stride 局部性）

术语解释
<gDiff 是 ISCA 2003 提出的基于全局值历史（跨动态指令）的 stride 值预测器：PC 索引预测表每项存多个 global diff（stride）+ distance index，用全局值队列（GVQ）中指定距离处的前序指令输出作 base 值，预测值 = base + stride。EgDiff 是本文重设计版本：加置信机制、非投机更新、延迟预测与距离轮询，把存储降 95.8% 并达到 99%+ 精度。>

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- gDiff 逻辑链：① 观察：指令 I 的值可能与程序顺序上前序某条指令 J 的输出存在稳定 stride 关系（全局 stride 局部性，ISCA 2003 Zhou et al., "Detecting global stride locality in value streams"，源自 Bodine 2002 论文）；② 硬件结构：PC 索引预测表（每项 n 个 stride diff + distance index）+ GVQ（记录近期 dispatch 指令的投机结果）；③ 预测：dispatch 时 PC 取 n 个 diff 与 distance index → 按 distance 从 GVQ 取 base 值 → 预测值 = base + diff；④ 更新（投机更新）：投机结果就绪后，计算与 GVQ 前序值的 delta 序列，与 n 个 diff 比较，匹配位成为新 distance index。
- gDiff 的四大失败点（本文 §II-D）：a) 精度不足 99%（误预测 squash 惩罚吃掉收益）；b) 投机更新污染——误预测分支/值的未提交结果污染 GVQ 与预测表（no-squash）；c) 依赖局部预测器补历史（局部预测器低置信时不输出，全局历史不完整），且对局部局部性强指令冗余；d) 存储大——order-32 每项 32 个 diff，4K 项达 1.03MB。
- EgDiff 的对应重设计：a) aggressive confidence（tag 索引 + usefulness bits + last misprediction + FPC）；b) 非投机更新（GVQ 分投机/非投机段，只用 commit 值更新预测器，squash 只清投机段）；c) deferred prediction（base 不可用暂缓、Wake 绑定、base 就绪反向触发）；d) distance polling（每项只留 1 个 diff + 5-bit distance，轮询收敛）。结果：EgDiff 4K 项 44KB（-95.8%）、精度 >99%、覆盖 25.87%、平均 IPC 提升 4.37%；与 EVES hybrid 19KB 达 6.16%（>32KB EVES 单独 4.81%），无限混合 7.02%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件架构中的角色：全局值预测器是 OoO 核后端/rename 阶段的附加硬件（预测表 + GVQ），预测值注入 PRF 与投机 GVQ，验证在 LQ 完成路径、更新在 commit 路径。
- 具体流程（EgDiff 全生命周期，论文 §III-B）：① 预测：PC 哈希索引 44KB 预测表 → tag 匹配且 FPC 饱和(7) → 取 64-bit diff + 5-bit distance → 从投机 GVQ 取 distance 处 base 值 → 预测值 = base + diff 写入 PRF（唤醒依赖）与投机 GVQ；② 验证：load 完成，LQ 结果与投机 GVQ 预测值比对，匹配则跳过 PRF 回写，不匹配置 misp、触发 21 cycle pipeline squash、回滚投机 GVQ；③ 更新：commit 值 FIFO 传播到非投机段，按 Algorithm 1（距离轮询）更新——δa = v_actual − NVQ[d].value 与预测 δp 比较，匹配则 FPC/u 递增，不匹配则 distance-1 重算 diff 并重置计数器（d 失效则重置默认 n）。
- 与 gDiff 的硬件差异（Fig.6 vs Fig.1）：表项从"32 diff + distance index"变为"1 diff + distance"（88 bit/项 vs 多 KB/项）；GVQ 从单一投机队列变为"32 投机 + 32 非投机"双段（141 bit/项，共 1.1KB）；新增 Wake 字段支持延迟预测的单绑定反向触发。
- 示例（表 I）：指令 2 的 base 在条目 5（未就绪）→ 指令 2 存 distance/diff、VP=0，并把自身索引写入条目 5 的 Wake；条目 5 结果就绪 → 按 Wake 找到指令 2 → 校验 distance 仍匹配 → 用新 base + diff 生成延迟预测写入 PRF。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：预测表为直接索引 SRAM（类似单个 TAGE 组件表），GVQ 为随机访问循环缓冲（FIFO 语义，类似 load queue），预测单次读取、更新至多访问两位置（distance 与 distance-1），无相联查找；面积/能耗用 McPAT（处理器整体）与 CACTI 7.0 @22nm（预测表+GVQ：0.078mm² = 0.22% 处理器面积、能耗 0.07%）。
- 使用方式（复现）：在 gem5 O3 CPU（本论文 gem5 20.0+，src/cpu/valuepred/ 框架）实现 gDiff baseline 与 EgDiff，按 Table II 配置 4GHz OoO 核（ROB/IQ/LQ/SQ 192/64/32/32、L1D 32KB 3 cycle、L2 256KB 12 cycle、L3 2MB 32 cycle、DDR4-2400 双通道、21 cycle 最小误预测惩罚），跑 SPEC CPU 2017 ref 输入、ARMv8-A -O3、SE mode、Simpoints 3.2（150M 指令 = 50M warmup + 100M 统计），统计 IPC speedup/覆盖率/误预测率；EgDiff 本身未找到公开代码（无法确认是否开源）。
- 网页佐证：gDiff 源自 ISCA 2003（ACM DL，"Detecting global stride locality in value streams"）；EgDiff 早期版本发表于 IEEE CAL 2025（"EgDiff: An Enhanced Global Load Value Predictor"，DOI 10.1109/LCA.2025.3590382）；gem5 值预测框架（xs-gem5 文档：VPUnit 的 predict/update/specUpdate/squash 接口、CompositeValuePredictor）。

涉及论文标题：
- Revisiting Global Value Prediction: A Resurgent Complement to Local Predictors
