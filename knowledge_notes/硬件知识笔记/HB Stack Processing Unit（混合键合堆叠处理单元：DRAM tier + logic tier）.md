## HB Stack Processing Unit（混合键合堆叠处理单元：DRAM tier + logic tier）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HybridSpec 的 HB 栈是"高带宽-小容量"处理单元：face-to-face 堆叠的 DRAM tier（数据存储 + 带宽源）与 logic tier（计算 + 控制）经 hybrid bonding via 直接互连（2-3µm pitch）。逻辑层含与 DRAM die 对齐的 logic block：每 block 有输入/输出激活缓冲、控制器、分布式计算阵列（array element 含小本地缓冲 + HB 控制器 + MAC），块间相邻互连做通信；logic tier 还有全局控制器、指令/地址/任务缓冲与外部接口。流片原型参数：408mm²@400MHz，DRAM 侧 4×2.5GB die（80 I/O group × 256-bit、10GB、4TB/s、0.88pJ/bit），logic 侧 4 block ×（140KB activation SRAM + 512KB weight SRAM + 80×64 FP16/BF16 MAC）。用途：跑投机解码的 draft 模型（内存受限、需要高带宽、容量需求小）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
数据流两级：(1) inter-logic block：TP 权重切分（输出/输入通道交替），计算 tile 化后与 ring-based 通信流水重叠（先算可独立部分、边算边传部分和，图 7），隐藏大部分通信延迟；(2) intra-logic block：Opr1×Opr2=Res 模式、Opr2（权重/KV）stationary——Opr2 从 HB DRAM 直接映射上计算阵列（weight-stationary GEMM / KV-cache stationary attention），Opr1 从激活缓冲读出并广播到各 array element，element 内 adder tree 累加部分和、块级缓冲聚合。运行流程（一次 draft decode 迭代）：draft KV 从 DRAM tier 经 HB I/O 进各 block → 权重驻留 SRAM、激活广播 → MAC 阵列算 GEMV/attention → 部分和经块间 ring 通信聚合 → 输出写回/送出栈。执行模型：draft 模型 + 其 KV cache 常驻 HB 栈，目标/草稿 prefill 与验证都在 XPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DRAM 工艺造 DRAM tier、先进逻辑工艺造 logic tier，face-to-face Cu-Cu 键合（对准精度 + CMP 平坦化），logic tier 到 substrate 用 wire bonding；流片验证（Fig.13 floorplan/via 布局/TEM）。使用方式：承接"高带宽、低容量、内存受限"的工作负载（draft decode），与 XPU+LPDDR5X（大容量 target）组成模型级异构系统；模拟评估用 silicon-derived 参数注入扩展 SplitwiseSim 的事件驱动模拟器。设计取舍：单层堆叠容量小（≤20GB 级），不适合存放随请求增长的目标 KV cache（HB-ATTEN baseline 的教训）。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
