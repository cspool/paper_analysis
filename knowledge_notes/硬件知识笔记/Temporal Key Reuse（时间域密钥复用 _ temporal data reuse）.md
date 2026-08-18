## Temporal Key Reuse（时间域密钥复用 / temporal data reuse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Temporal Key Reuse 是 FlashTFHE（ISCA'26）提出的加速器数据复用策略，与空间（spatial）架构的"BSK 单遍流经 PE 阵列、边流边弃"相反：外层循环每次只从 DRAM 载入一个 BSK（bootstrapping key）chunk（≤0.8MB chip-wide，小到可常驻片内），把同一 chunk 在整批（最多 48 个）并行 bootstrapping 之间按 round-robin 顺序时间复用后再驱逐。核心效果是把"计算吞吐"与"DRAM 带宽"解耦——同一 chunk 反复喂给宽流水线多次，加宽 coefficient 并行不再按比例抬高带宽需求。
- 支撑条件：GLWE 累加 buffer 移出 PE 阵列、放进每 core 的稠密 SRAM（9.2MB Acc buffer），即放弃 output-stationary 执行——partial sums 不再要求驻留在单个 PE 里，正是这个驻留约束挡住了空间架构的 BSK 局部性。论文用伪代码描述调度（Figure 11）：
```
for bsk_chunk in BSK:                    # 外层：载入一个 chunk 片内复用
    parallel_for core in num_cores:
        for i, decomp_glwe in rr_ctxts[core]:   # 内层：round-robin 遍历在飞 ciphertext
            acc += VecMAC(FFT(decomp_glwe), bsk_chunk)   # 累计到 SRAM Acc buffer
    # I-FFT 只在所有 chunk 累计完后触发（SRAM-resident accumulator 使然）
```

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 FlashTFHE 的运转流程：multi-bit TFHE 的 BSK 可达数 GB（GPT-2 decoder layer 的 key 4.7GB），空间架构必须为每次 bootstrapping 从 DRAM 单遍重取整份 BSK，加宽 PE 则带宽线性暴涨（两栈 HBM2E 在 16 coefficients/cycle/PE 即饱和）；时间域复用一个 BSK chunk 摊薄到整批 ciphertext，使单 core MAC 吞吐达到 512 coefficients/cycle（同带宽约束下为空间架构的 8×），两栈 HBM2E（819GB/s）仍可支撑 8 core。ciphertext 并行也变成软件可控的 batch size（round-robin 数量 12 最优），无需复制整套 FFT 流水线与随 N 线性增长的 in-flight 状态（N=65536 时单 R2MDC 流水线 ~59.5MB，时间域设计消除该复制）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要素：①片内 GGSW buffer（0.8MB）存当前 chunk；②SRAM-resident GLWE accumulator（9.2MB/core 组），I-FFT 延迟到所有 chunk 累计完后触发；③round-robin 调度器把 ciphertext 轮流喂给 BRU 流水线；④adaptive batching 由编译器按程序实际并行度调节 batch 大小。使用/评估：论文 Figure 8 对比显示 temporal 设计在相同带宽下吞吐 2×、8 ciphertext 时同面积吞吐 2.63×；Figure 15 显示 round-robin ciphertext=12 时吞吐最大且 buffer 最小。该策略是"空间 vs 时间"数据复用范式在 FHE 加速器上的首次系统性对比，也是 FlashTFHE 相对 Morphling/Matrix/Strix 的核心区别。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
