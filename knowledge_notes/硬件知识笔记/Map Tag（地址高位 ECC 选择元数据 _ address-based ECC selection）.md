## Map Tag（地址高位 ECC 选择元数据 / address-based ECC selection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Map Tag 是 RangeGuard 的内存控制器机制：把未使用的物理地址高位（如 bit[57:54]）编码为"保护模式选择元数据"，让控制器在读写时识别每个数据区域应使用的 RangeMap/子映射（或传统 SEC-DED 精确纠错模式），无需额外引脚或 sideband 信号。逻辑链：现代 DNN 同时使用 FP32/BF16/FP8/INT8 等多种数值格式、且不同张量（权重 vs 激活、早期 vs 晚期 block）值分布差异大 → 单一 RangeMap 无法适配 → 控制器需按数据区域切换保护模式 → 利用"超出已实现物理内存容量"的高位地址位做 Map Tag（这些位不影响真实内存位置，只作控制元数据），复用 Virtualized ECC（Yoon & Erez ASPLOS 2010）的 address-based ECC selection 思想，实现 per-region 灵活保护而不干扰地址翻译与既有 ECC 操作。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
内存控制器数据通路（写/读）：写路径——按 Map Tag 选 RangeMap 子映射 → 每值映射为 RID（32/16/8-bit 值分别 4/2/1-bit RID）→ 同一 32-bit 区域的值 RID 打包进一个 ECC 符号 → RS 编码生成 16-bit parity → 丢弃显式 RID、存数据+parity；读路径——按 Map Tag 选同一子映射 → 重新生成候选 RID → 与存储 parity 过 RS 解码 → 纠正错误 RID 符号 → 被纠值替换为范围代表值；需要精确恢复的区域（如不可容忍近似的关键元数据）Map Tag 选择 SEC-DED 模式。多格式适配例子：16-bit 值用 4-bit RID、8-bit 值用 2-bit RID、4-bit 值用 1-bit RID，使一个 32-bit 区域的全部 RID 恰占一个 ECC 符号——这样 DRAM 常见的 32-bit 对齐 burst 故障表现为单个符号错误，可用单符号纠错吸收；RID 位宽更小时剩余纠错能力可处理多个故障。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：控制器内多实例 RangeMap（flip-flop 表）并行驻留 + 地址高位解码器选路；Map Tag 写入由操作系统按数据区域属性（格式/张量类型）配置，与地址翻译共存。使用要点：Map Tag 位必须选在"已实现物理容量之外"的地址位（否则改变内存位置）；多个格式可同时受保护（论文 Fig.7 每个 RangeMap 实例含 32/16/8-bit 子映射）；代价是多个子映射实例的硬件开销（论文面积报告按 RangeMaps 单列）。这是 RangeGuard 以"最小硬件改动"支持多格式/多分布的关键机制。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs
