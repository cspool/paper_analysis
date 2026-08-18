## 块级异步 softmax（Block-centric Asynchronous Softmax，SCU）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 块级异步 softmax 是 TAGT 的 Specific Computing Unit（SCU）实现的 softmax 硬件机制，用于克服标准 softmax 的全局数据依赖（需要全序列最大值/求和）导致的低并行。核心思想是把 softmax 拆成"块内并行 + 块间按 TID 异步累加"：Block Partition Module (BPM) 把 FAU 流入的原始注意力分数流切成固定大小块（S_i）入 Input Queue；Element-wise Computing Unit (ECU) 并行计算每块的局部分子 O_j = Σ_{i∈S_j} e^{s_i-φ}·v_i 与局部分母 l_j = Σ_{i∈S_j} e^{s_i-φ}（φ 为统一最大值，由 Param Loader 提供）；带 {TID, IsFirstBlock, IsLastBlock, O_i, l_i} 标签发往 Reduction Unit。RU 由 Dispatcher + 64 个 Core Reduction Unit（CRU，各含 Comparator、TID_Tag/O_accum/l_accum 寄存器、Merge ALU）+ Shared Divider Pool 组成：Dispatcher 广播 TID，CRU 并行比对——Hit 则 Merge ALU 异步累加（O_accum_new=O_accum+O_j），Miss 则分配空闲 CRU 初始化；处理 IsLastBlock 的 CRU 触发最终归一化 Output=O_final/l_final（Divider Pool 仲裁分配除法器），随后清空状态接新任务。Activation Unit（AU）做 ReLU 等非线性。
- 与软件 online softmax（FlashAttention 的逐行 m/l 状态机）的区别：online softmax 是软件顺序更新 running max/sum；块级异步 softmax 是硬件把分数流按块并行计算局部分量、再以 TID 为键在 CRU 上异步归约，消除跨块同步。注意本机制是 TAGT 专用硬件实现，非 GPU kernel。

从硬件架构角度拆解术语，比如术语在硬件架构中发挥作用的流程例子。通过联网搜索让回答具体和精准。
- 数据流：FAU 的 UPE 算出原始分数流 → SCU 入口 BPM 分段 → 块入 Input Queue → 空闲 ECU 取块（Param Loader 提供 φ、TID、IsFirst/LastBlock 标志）并行算 (O_j, l_j) → 打包 {TID, flags, O_j, l_j} → Dispatcher 广播 TID → 64 CRU 并行 Hit/Miss 匹配并异步累加 → 目标块的 IsLastBlock 包到达时请求 Divider Pool → 归一化 O_final/l_final → AU（ReLU）→ 送 VPE 聚合 value。
- 并行效果：多个 ECU 同时处理不同块、多个 CRU 同时累加不同 TID 的部分和、除法与累加解耦（CRU 清空后立即接新任务），从而把 softmax 的全局依赖变成细粒度异步流水。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：SCU 为 TAGT 内专用硬件模块（1 个 SCU，与 16 FAU 协作），RTL 综合到 Alveo U280 @280MHz；BPM/Input Queue/ECU 阵列/RU（Dispatcher+64 CRU+Divider Pool）/AU 组成（Fig.8b）。与 FAU 的 UPE/VPE 形成"分数流式路径（UPE→SCU）+ 贡献聚合路径（SCU→VPE）"的融合注意力流水，全程不物化 N×N 注意力矩阵。
- 使用：作为 TAGT 注意力计算的归一化引擎，支撑 TDS 稀疏注意力（每目标顶点 O(m·log_m N) 个分数）在硬件上的高并行执行；配合 FAU 的 Reconfigured MM 模式跑 FFN。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
