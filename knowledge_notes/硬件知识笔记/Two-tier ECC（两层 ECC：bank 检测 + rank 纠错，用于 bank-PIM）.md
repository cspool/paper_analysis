## Two-tier ECC（两层 ECC：bank 检测 + rank 纠错，用于 bank-PIM）

术语解释
论文为 host 控制型 bank-PIM 定制的两层可靠性结构：第一层在 bank 内用 on-die CRC 只做检测（detect-only），最大化多比特错误覆盖；第二层在 rank 级用 chipkill ECC 做纠错，仅在出错或写时访问，避免频繁跨 chip 重编码与写放大。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
传统 rank-PIM 把计算放 rank 级以自动获得 rank 级 ECC；bank-PIM 若只靠 bank 内 on-die ECC（DDR5 SEC(136,128) 只纠单比特、HBM3 RS(19,17) 受故障隔离边界约束）则多比特错误检测不足 → SDC。论文的两层 ECC：(1) 近 bank 第一层——on-die CRC（CRC8(136,128) 用 8-bit 冗余匹配 DDR5 配置，CRC16(144,128) 用 16-bit 匹配 HBM3 配置），PIM 访问时纯 detect-only（检测覆盖率最大化，CRC8 比 RS8 检测好一个数量级、CRC16 在等冗余下超 RS16 一个数量级以上）；非 PIM 全 rank 访问时同一 CRC 切为单比特纠错模式（保留 on-die ECC 处理 VRT 的原始用途），模式由 PIM/non-PIM 配置寄存器信号选择、零额外周期。(2) rank 级第二层——chipkill ECC（每 rank 2 冗余 chip）：CRC 检出错误时经 DDR5 ALERT_n 引脚通知内存控制器，控制器对出错 all-bank 命令涉及的 bank 顺序读、重构 rank 级码字、chipkill 纠错、写回并重试 PIM 命令（硬件状态机或带 fence 的软件 handler 实现）。因为目标 PIM 应用读为主、写缓冲在 PIM SRAM，第二层只在纠错与 host 写时参与，避免频繁 RMW。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件运转流程（一次 PIM GEMV 出错）：bank 读码字 → bank 对共享 ECC 引擎的 CRC 解码器 detect-only 检错 → 检出错误断言 ALERT_n → 内存控制器（或软件 handler）切 PIM→host 模式（约 37.5ns 切换）→ 对涉及 bank 顺序发单 bank 读重构 rank codeword → chipkill ECC 纠正 → 按故障 chip 位取反写回（Codeword Flip）→ 重试 PIM 命令；仍失败升级为单比特纠错模式解码，再失败 host 模拟执行 + 页面退役。纠错延迟：单 chip 单 bank 基线 63.75ns（17 个单 bank 请求），nominal VRT 率下 99.9998% 触发访问落此路径，最坏多 chip 141.25ns 且概率 <0.2%。可靠性收益：等冗余下比 DDR5 bank-PIM baseline SDC 低 400×、比 HBM3 bank-PIM（on-die RS(19,17)）SDC 低 2–10×、总失败率低 30–80K×（rank 纠错把 DUE 压低，而 HBM3 bank-PIM 的 DUE 高）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bank 侧为每 bank 一个 CRC 解码器（bank 对共享一个 ECC 引擎，~2.6% 面积开销，关键路径 ~13 级门），rank 侧复用 host 正常访问的 rank 级 ECC。使用：通用两层 ECC（如 XED）直接搬 bank-PIM 会因 VRT 频繁触发 rank 纠错而性能大跌（10,000 VRT 单元 → >20% 损失），必须配 Codeword Flip 掩蔽 VRT（见"Codeword Flip"条目）；扩展性：论文指出两层 ECC + Codeword Flip 可用于任意 rank 配置（含 LPDDR 系，CRC16 匹配 LPDDR5x DLEP 的 16-bit on-die 冗余），UpMem 类非 host 协调架构需额外协调机制。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory
