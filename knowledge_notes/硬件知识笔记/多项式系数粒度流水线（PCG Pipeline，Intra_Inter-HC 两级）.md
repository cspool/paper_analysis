## 多项式系数粒度流水线（PCG Pipeline，Intra/Inter-HC 两级）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PCG（Polynomial Coefficient-Grained，多项式系数粒度）流水线是 CASCADE 的细粒度流水模型：以"单个多项式系数"而非"整个 RLWE 密文"为流水单位，系数算完立即流入下一级。两级：Intra-HC Pipeline——单个 HC 内功能单元之间流式传输中间结果，不等整个 RLWE 在一个阶段处理完；Inter-HC Pipeline——上游 HC 算完一个多项式系数即经 D2D 送给下游 HC，不等整个 RLWE 完成。目标：重叠执行、让所有功能单元保持忙碌、最小化缓冲内存占用（CASCADE 每 HC 内部 buffer 仅 1 MB）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转例子（一次 HMUX 的系数粒度流水）：RLWE 多项式的第 j 个系数经 Rotation→Decomp→FFT→VMA→IFFT 逐级流动，第 j 个系数的 FFT 结果算完时第 j−1 个系数已在 VMA 乘加、第 j−2 个已在 IFFT——五级深度重叠；Inter-HC 侧，上游 HC 完成 HMUX_i 一个系数的 IFFT 后立刻经 D2D 发给下游 HC 做 HMUX_{i+1} 的 Rotation。效果：一个 HMUX 的时延 ≈ 最长流水级（而非五级之和），稳态下每个 HC 每周期都在处理不同 HMUX/不同系数，配合 intra-HC batching（注入多个 RLWE 密文）避免功能单元气泡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：HC 内各功能单元（Rotation/Decomp/FFT/VMA/IFFT）之间用 streaming datapath 连接；Inter-HC 靠 D2D 链路 + 输入/输出 double buffer（128 KB 各）实现"接收下一密文与处理当前密文"重叠。使用：为支撑跨 HMUX 流水线并行（n 个 HMUX 的 ACC 依赖链），系数粒度把"密文级流水"进一步细化为"系数级流水"，是 CASCADE 高利用率（91.03%-97.18%）的来源之一；代价是 D2D 通信粒度变细（更频繁的跨 chiplet 传输），因此需配合 Interleaved-Fusion 映射把部分 HMUX 融合在本地执行以控制 D2D 流量。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
