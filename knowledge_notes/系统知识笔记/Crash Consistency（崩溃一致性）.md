## Crash Consistency（崩溃一致性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Crash consistency 指系统在任意时刻断电后，持久化状态仍能恢复到某个一致、可用的检查点，而不会出现部分写入导致的损坏。对 EHS 而言，由于断电频繁，每次 checkpoint 本身必须 failure-atomic：任何中途断电都不能留下"半个 checkpoint"。本论文的核心难点正是"加密粒度 vs 原子性"矛盾——AES-XTS 以 16B 块为单位加密，64B 页横跨 4 个 XTS 块；EHS 只保证 2B 原子写，若页只持久化了一部分，剩下的密文无法解密。论文 [66] 指出这是加密 NVM 的通用难题。
- MANATEE 的解决方案：①把 V_backup 阈值调高，保证每次 page manager 调用（含 4×16B 块凑成 64B 页的原子 flush）在执行期间不会被断电打断（能量缓冲覆盖该操作）；②完成标志位机制：checkpoint 最后一步把 NVM 中标志位翻转为 1，恢复时若标志位为 0 则判定 checkpoint 不完整并 abort。
从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：store 指令更新 SPM 页 → WTQ 记录 (页号, frame 号) → 断电信号到来 → 允许当前 page manager 原子操作完成（若中断点：①提供页号/颜色后，SPM 未变、BT 可恢复；②BT 命中后 store 前，SPM 已空、需按 BT 重取；③store 完成但 WTQ 未更新——这三处都因 failure-atomic 保障被规避）→ 按 WTQ 加密持久化脏页 → 翻转完成标志位 → 休眠。恢复按标志位决定继续或 abort。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MANATEE 用"提高 V_backup 阈值 + 页粒度原子 flush（SPM 内部缓冲 4 个 16B 密文块，凑齐 64B 一次写入 NVM）+ 完成标志位"三件套实现；对比的 NVSRAM/Mapi-Pro 用整 SPM checkpoint（开销大），传统加密 NVM 方案需事务机制或硬件支持（不适合能量受限 EHS）。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
