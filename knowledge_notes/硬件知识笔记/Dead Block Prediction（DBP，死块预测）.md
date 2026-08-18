## Dead Block Prediction（DBP，死块预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DBP 预测哪些 cache 块在淘汰前不会再被引用（dead block），用于替换策略（优先淘汰死块）、bypass（死块不分配）或省电。早期工作用 counter/signature-based 预测器（Lai et al.），后续提出 sampling-based（Khan et al.）等方法降低开销。Bumper 的讨论（Section IV/VIII）：可以把 DBP 训练成"useful 行判别器"用作 IFU 预取过滤器，但移动应用 code/data footprint 巨大（0.5–2.3MB / 2.0–15.0MB），所需跟踪结构开销不切实际；且误判代价不对称——把 useful 行误判为 dead 并过滤掉会严重损失性能。Bumper 的 commit hint 可视为"零风险的死后判定"：不做预测，等首个提交证据出现再提升，无预测器、无训练、存储仅 422B。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DBP 在替换策略中的流程：行命中/插入时查询预测器 → 若预测 dead 则降低其替换优先级（或 bypass）→ 预测错误时性能受损。Bumper 的对照流程：行以 RRPV=3 插入 → 行内首条指令提交（事实信号，非预测）→ 提升 RRPV=0；对 useless 行的"预测"是 100% 准确的（永不提升即自然被淘汰）。两者的本质区别：DBP 预测未来，Bumper 依据过去。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：计数器/PC 签名表或采样器跟踪块复用模式；应用于 cache replacement（与 RRIP 等结合）、bypass 与 power gating。Bumper 论文结论：对大规模 footprint 的移动负载，DBP 类方案风险高、开销大，而 commit 驱动的方案（Bumper）以极低代价达成同等目的。

R-Max 补充视角（ISCA'26，Perfect Dead Block 计数）：R-Max 的 dead block counter 是 DBP 的 oracle 版本——不是预测、而是由 Bélády's MIN 离线处理访存流后精确得到（Alg.2：对每个标 prefetch 的访问，向后累计同地址 hold 访问数直到下一 prefetch 标记或记录结束，即块在被逐出前还会被 demand 命中的次数）。仿真中 demand 命中递减计数器，归零=块死亡 → 触发定向预取替换死块。R-Max 与 Dead Block Correlating Prefetcher（DBCPS）类似地"用死块作预取触发"，但加了完美死块预测与完美未来数据预测；λ Queue 处理"被预取但因访存重排被逐出、计数器未归零"的块（地址+计数器暂存，不占缓存容量）。
涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
- R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance
