## Delayed Write Verification（延迟写验证）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Delayed Write Verification（延迟写验证）是 HBM-CASO 解决"写验证"问题的技术：现代 HBM 用 advanced SysECC 取代 CRC 后，片上没有资源解码这种更复杂的纠错码来逐写验证传输数据；HBM-CASO 改为把一批写操作的奇偶校验符号用 XOR 累加成一个 64-bit 结果，批末与内存控制器侧同步累加的结果比较，验证整批写的传输完整性。逻辑链（两个关键观察）：(1) 写操作不像读操作那样面临超出 ECC 纠错能力的风险——控制器保存着原始未损坏副本，出错可重传；(2) 写操作常批量发生且不在程序执行关键路径上、延迟不影响执行。因此采用"批量检测"策略：每 64B 写在 HBM 侧生成 8B 片上 parity（4B regional + 4B CRC），两个 Accumulation Unit（均匀分布在两个 pseudo-channel，不跨 PC）对每 PC 的 2B regional + 2B CRC parity 做 bitwise XOR 累加进 64-bit 寄存器；控制器并行执行相同累加；批末（32~64 次传输，burst 长度 256~512）把控制器侧 64-bit 结果随最后一条写传入 HBM 比较；不匹配则整批重传（控制器写缓冲需扩展到 batch 大小）。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在内存控制器 + HBM 的协作数据路径上运转：控制器收到一批写请求 → 逐条生成 global parity 并 XOR 累加进 64-bit 寄存器 → 写数据传输进 HBM → HBM ODECC 生成 regional parity、Accumulation Unit XOR 累加（2B regional + 2B CRC per PC）→ 批末控制器把 64-bit XOR 结果附在最后一条写上传输（优化：在写-读 turnaround 时隙传输，因结果由外围电路生成而非内部阵列读出，可直接发出）→ HBM 比较两侧累加结果 → 匹配则批完成、不匹配则控制器重发整批。性能特征：(1) 验证推迟到最后一条写，但不占关键路径（写不阻塞程序执行）；(2) 带宽开销可忽略——64-bit XOR 只占一次 burst，而整批 32~64 次传输；(3) 保护强度：64-bit 累加冲突率 2^-64，远强于 baseline 每 32B 用 16-bit CRC 的局部保护；(4) 重传代价可控：DRAM 错误率通常 <10^-8，即使异常升高到 10^-4 也可通过调整 batch size 吸收。片上硬件：两个 Accumulation Unit（XOR 累加器 + 64-bit 寄存器），与 Merging Unit 合计仅 +61 cells/+17μm²。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HBM 侧两个 per-pseudo-channel 的 XOR 累加器（2B regional + 2B CRC parity 对逐写异或进 64-bit 寄存器）；控制器侧一个同构累加器 + 批末比较逻辑 + 扩展写缓冲（hold 整批直到验证通过）。使用场景：任何"片上无法解码更强系统 ECC、但系统希望用更强 ECC"的写验证场景；本质是"检测 + 重传"（与 CRC/链路重传语义一致，但把检测域从单写扩展到整批）。评估：论文用 Ramulator2 扫 batch size 16~256 与 BER 10^-8/10^-6/10^-4（Figure 9），最坏情况（高错误率+大批量+写密集）开销约 3%；传输 UE%（Table IV）G-mode 全 0、R-mode 仅 1SAF 下 0.001%（对比 baseline 最高 0.096%）。此技术可推广到其它 ECC 方案（Hamming/残差码），因为合并与累加只依赖线性性质。论文未明确说明累加器与比较逻辑的具体门级实现。

涉及论文标题：
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
