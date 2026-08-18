## 双链部分和累加数据通路（Adaptive Dual-Chain Accumulation Datapath）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 双链部分和累加数据通路是 UNICORE 为可组合 PE 在不同位宽模式下高效累加部分和而设计的自适应行内累加结构：每行内实现两条独立累加链（左 slice 链 S_L、右 slice 链 S_R），根据运行模式动态调整。W4A4 模式（所有 PE 贡献同一结果）下两条链输出经最终加法器求和产生统一输出 S_+*；W8A8 模式（融合 PE 成为单个宽 MAC、内部进位链激活）下两条链输出直接拼接产生宽位宽结果 S_||*。由于 PE 间部分和连接固定，切换不需要多路选择器或复杂路由逻辑，位宽自适应零 mux 开销。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 运转流程：每个 PE 完成 S-FPMA 乘法与补偿后，乘积转 sign-magnitude（经 shifter），专用加法器将其与传入的两条链部分和之一混合格式累加（2's complement 部分和 + sign-magnitude 乘积），生成新部分和沿行内链传播到下一 PE；W4A4 时 S_L 与 S_R 独立传播（两条独立 MAC 的各自部分和），行尾经末级加法器合并；W8A8 时两条链的输出拼接为宽部分和。相比"每模式一套累加链 + mux 选择"的方案，双链方案用固定连线实现了两种模式的累加语义。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：SpinalHDL RTL 的行内两条固定累加链 + 行尾求和/拼接逻辑；Registers 排除在面积分解外（不同 systolic dataflow 需要不同 buffering），UNICORE 组合逻辑与寄存器开销均最低。作用：配合 S-FPMA 位宽融合，使同一 systolic 行在 W4A4/W8A8（及混合 W4A8）间切换时无需重构累加网络，维持高利用率。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
