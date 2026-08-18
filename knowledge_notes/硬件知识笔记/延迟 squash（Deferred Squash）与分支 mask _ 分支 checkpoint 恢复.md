## 延迟 squash（Deferred Squash）与分支 mask / 分支 checkpoint 恢复

术语解释
延迟 squash 是本文配套的误预测恢复机制：检测到误预测后立即清 fetch~dispatch 段，但 rename 之后已分配资源的 squashed-path 指令继续派发/发射/执行，直到第一条 resolved-path bundle 到达 rename 才最终 squash——为 SBRB 争取更多"执行完的 squashed 分支"。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 立即 squash（baseline，MIPS R10K branch mask 构造）：每条被取的分支在全局 branch mask 中占 1 bit 并对应一个 checkpoint；每条指令继承当前 branch mask 副本，从而知道所有比自己老且未解析的分支；分支解析时广播 one-hot mask + 正确/错误信号：正确则所有指令清该 bit，错误则 (a) 全局 branch mask 恢复到该分支自身的 mask，(b) Free List head / ROB/LQ/SQ/BQ tail / 重命名表 / SS 从该分支 checkpoint 恢复，(c) squash fetch~dispatch 段，(d) IQ 与执行通道中 branch mask 含该 bit 的指令自无效化；(2) 延迟 squash：立即执行 (a)(b)(c) 中的大部分但**不清 dispatch 段**（rename 已为 dispatch bundle 分配了物理寄存器与 ROB/LQ/SQ 项，回滚指针并不干扰），1-bit 状态机 idle → delayed squash，并记录误预测分支的 one-hot mask 与其"更老分支 mask"；当第一条 resolved-path bundle 到达 rename 时：清 dispatch 段、执行 (d) 自无效化、状态回 idle；(3) 延迟期其他分支 brY 解析的交互：brY 比 brX 年轻（brX 的 mask 中无 brY 位）则被静默；更老且正确则照常清 bit；更老且错误则 brX 先按 resolved-path 到达的方式终结 squash，再对 brY 执行立即恢复步骤，brY 接替状态机、状态保持 delayed squash。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
延迟 squash 的作用是扩大 SBRB 的原料供给：误预测检测到最终 squash 之间多出 fetch→rename 的 8 个周期（默认 12 级流水），让 squashed-path 分支继续执行完并写 SBRB。流程例子：br1 在 execute 解析为误预测 → (a)(b) 立即恢复、fetch 从正确目标重取 → rename 之后的 br3 继续 issue/execute → br3 执行完按 key 写 SBRB（若此时尚未被清）→ 8 周期后第一条 resolved-path bundle 到 rename → 清 dispatch 并自无效化 IQ 中 br1 更年轻指令（br3 若还在执行通道也在此刻被清）→ 完成。效果数据（Table IV）：squash 阴影平均 17.68 条分支，误预测检测时完成 4.25（24%），延迟后完成 8.19（45%）。性能影响：immediate squash 下 SBRB 收益从 4.43% 降至 2.22%；且 delayed squash 对"无 SBRB 的 baseline"本身平均 +1.55%（阴影中更多 CI load 执行完、提前发起 cache miss）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 恢复以 checkpoint 为单位（本文 64 个分支 checkpoint），SS 也随 checkpoint 复制（每 checkpoint 一份 SS 副本，2.9KB 总开销），退休 SS 走 BQ 冗余位维护；(2) 延迟 squash 不延迟 resolved-path 的 fetch 与执行（rename 之前的 squashed-path 指令已清，rename 处有 8 周期空档可让 resolved-path bundle 无阻塞进入）；(3) 正确性靠 branch mask 位测试保证：只有"比误预测分支年轻"的指令才会被延迟清掉，更老指令不受影响。通用背景：rename 阶段必须恢复 RAT 才能给 resolved-path 指令改名，这是延迟 squash 的硬边界——本机制正好把延迟窗口定义为"检测点到 resolved-path 首条指令到 rename"的自然间隔，不需要额外结构即可放行阴影执行。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
