## Re-Order Buffer（ROB，重排序缓冲）与 hint 传播

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ROB 是乱序执行核中按程序序记录所有未提交指令（uop）的环形缓冲，负责顺序提交（retire）、精确异常、误预测恢复与寄存器重命名回收。Bumper 基线为 620-entry ROB、10-wide retire（Table III）。Bumper 对 ROB 的扩展：给每个 ROB 项（每 uop）加 1-bit send_hint 标志（共 80B 开销）——L1I 行被读取时若其 send_hint 位为 1，则随行内首条指令/首条 uop 把标志置入 ROB 项并随指令传播；当该指令（或首条 uop）在 ROB 头按序 retire 时，检测到标志则把该指令的 VA（恢复路径上本就维护的地址）送回 IFU 生成提升 hint。这是"提交信息回传缓存层次"的最小代价实现：不在 retire 处直连 L2C（退休点无 PA），而是借道 ROB 标志 + 恢复路径 VA。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Bumper 的 hint 传播流程（Fig.9）：L1I fill 置 send_hint=1 → 行内首条 uop 读出时把标志复制进其 ROB 项 → 指令乱序执行、写回 → 在 ROB 头顺序 retire 时检测标志置位 → 发 VA 回 IFU（②）→ IFU 经 HL1Q 机会式访问 ITLB 翻译（③④）→ L1I tag 访问清除 send_hint（⑤）→ HL2Q 仲裁后以 PA 提升 L2C（⑥⑦⑧）。每行 L1I 生命周期内至多一次信号，带宽开销可忽略（L1I 请求 +0.4% 平均）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：环形 SRAM/CAM，每项含 uop 类型、目标寄存器、完成位、异常/分支元数据与 PC/VA；Intel/ARM 高端核均有 ROB 或等价 order-preserving 结构（论文允许用等价结构替代）。Bumper 只加 1 bit/uop，是最便宜的"跨流水线元数据传播"载体——对比方案是给每条提交指令都向 L2C 发 hint（带宽/能耗不可接受）。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
