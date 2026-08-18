## SBRB（Squashed-Branch Reuse Buffer，被 squash 分支复用缓冲）

术语解释
SBRB 是放在取指（fetch）单元中、按"动态分支实例"索引的小型缓冲：分支在错误路径上被执行后把 outcome 存入，重取指时若命中且置信高，就用该 outcome 覆盖默认分支预测器的预测，从而直接消除误预测。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SBRB 是"fetch 阶段 squashed-branch reuse"的具体实现结构。逻辑链：(1) 处理器中一条分支误预测会 squash 其后数百条指令，其中不少是控制独立（CI）指令，包括一些已经执行完的更年轻分支；(2) 这些被 squash 分支的 outcome 在正确路径上重取指时依然有效（对 CIDI 分支而言），可用来覆盖预测器；(3) 关键难点是重取指时如何把 resolved path 上的动态分支与其 squashed path 上的"对应实例"对齐——本文用不变签名解决：key = fold32(PC) XOR sig（PC 折叠 64→32 位后与 32-bit 签名异或）；(4) 分支在 execute 阶段按 key 访问 SBRB：miss 则分配并写入 outcome，hit 则更新 outcome（同一动态分支被多次 squash 时会发生）；(5) 分支在 fetch 阶段被预测时按 key 查 SBRB：命中且该分支在 BTB 中的置信计数器为高，则用 SBRB outcome 作为最终预测，覆盖 TAGE-SC-L 的输出。本文默认配置：256 项、4-way 组相联、每项 30 bit（valid:1, lru:2, tag:26, outcome:1），仅 960 B；设计空间探索显示 4-way 与全相联在 256 项处收敛到无界 SBRB 的峰值性能。SBRB 天然跨多次 squash 保留 outcome（乱序窗口内可能发生多次 squash），这是相对 SYRANT 的 SBL 只保留最近一次 squash 的优势之一。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SBRB 位于 fetch 单元，参与两个阶段：预测（fetch）与执行回写（execute）。具体流程（br1 误预测场景）：fetch 预测 br2/br3（TAGE-SC-L）→ br1 误预测广播 squash（R10K branch mask）→ deferred squash 让 rename 之后的 squashed-path 指令继续执行 → 其中 br3 执行完，按 key(br3) 把 outcome 写入 SBRB → 从分支 checkpoint 恢复 Signature Stack（sig 回到 br1 之前的值）→ 从正确目标重取指 → 再次取到 br3 时按 key(br3) 查 SBRB 命中，且 BTB 置信计数 >3 → 用 squashed outcome 覆盖预测器 → br3 不再误预测。执行阶段访问用的 key 可以由 BQ 条目携带（BQ 每项增位），或由解析分支用其 checkpoint 中的 SS 重算（本文采用后者）。伪代码（预测路径与执行路径）：
```
// fetch 阶段（分支被预测时）
key = fold32(PC) ^ sig
if (SBRB.hit(key) && BTB.confidence(PC) > threshold)
    prediction = SBRB.outcome(key)      // 覆盖默认预测器
else
    prediction = TAGE_SC_L.predict()

// execute 阶段（分支执行后）
key = fold32(PC) ^ sig
if (SBRB.hit(key)) SBRB.outcome(key) = actual_outcome
else SBRB.allocate(key, actual_outcome) // 可能需要替换
```
Annotations：key 由 PC 与签名合成，签名来自 Signature Stack（LFSR 维护）；threshold=3（3-bit 饱和计数器的上半态）；置信训练见 BTB 置信计数器机制（仅在 squashed outcome 可用且与默认预测不同时 +1/-1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上把 SBRB 当普通 cache 结构组织：32-bit key 拆成 index 与 tag，每项 valid/lru/tag/outcome。容量与相联度经设计空间探索确定（256 项、4-way 收敛）。论文验证了 LFSR 签名是完整标识符的有效代理：SBRB-literal-stack（256 项字面栈 → SHA-256 → CRC 缩减 32-bit 签名）与 SBRB 性能几乎无差别。论文还验证了签名本身不可或缺：PC-only key 的配置性能明显退化。SBRB 与 MSSR/RI 的 squash reuse 目标不同但可组合：SBRB 在 fetch 阶段消除误预测，MSSR/RI 在 rename 阶段复用寄存器结果、降低 penalty。注意 SBRB 依赖编译器生成的循环信息（LIT/LIS）与延迟 squash 机制配合才能收集足够的 outcome。开源情况：论文基于 NC State 自研 RISC-V 超标量模拟器评估，SBRB 实现未开源（联网搜索无公开仓库）。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
