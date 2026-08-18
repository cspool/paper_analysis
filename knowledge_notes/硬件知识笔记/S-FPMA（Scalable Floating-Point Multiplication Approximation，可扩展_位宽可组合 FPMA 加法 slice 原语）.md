## S-FPMA（Scalable Floating-Point Multiplication Approximation，可扩展/位宽可组合 FPMA 加法 slice 原语）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- S-FPMA 是 UNICORE 提出的可动态融合的 FPMA 计算原语：把 FPMA（浮点乘法近似，基于 Mitchell 对数近似 log2(1+M)≈M 用整数加法替代浮点乘法）分解为一组统一的 4-bit 加法 slice，宽精度通过进位链级联 slice 构造（W8A8=2 slice、W16A16=4 slice）。由于 FPMA 本质上是对拼接 exponent-mantissa 域的整数加法，其数据通路天然可分解为定宽加法器；而乘法器的部分积结构不可线性组合（O(n²)），这是 S-FPMA 与 bit-composable 乘法器（Bit-Fusion 风格）的根本区别。每个 4-bit slice 独立处理一段 exponent-mantissa、减去 bias，并纳入细粒度（FG）与粗粒度（CG）补偿：FG 补偿在 slice LSB 侧拼接恢复缺失低阶位，W8 融合模式中 FG 只留在最低 slice（细粒度细节沿加法链传播）、CG 利用 B 的空闲位参与粗粒度尾数位加法。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 在 UNICORE PE 中运转：每个 PE 为双 slice S-FPMA——独立模式（W4A4）左右 slice 各做独立低比特 MAC（各产生 E4M4 乘积并分别累加，两条独立部分和 S_L/S_R）；融合模式（W8A8）两 slice 经内部 carry chain 组合成单个宽 MAC（处理 E4M3 操作数，进位从低 slice 流入高 slice），FG 补偿保留在最低 slice、CG 1-bit 补偿以 carry-in 注入最低 slice，乘积指数 E5 驱动双 shifter（低 4 位控制右移器、E5−δ 驱动左移器），裁剪拼接后送入累加。W16A16 时四个 slice 级联、进位逐 slice 传播、补偿保持对齐。因为融合后的数据通路仍是一次加法，面积与时延随位宽严格线性 O(n) 增长（如 W8A8 PE 计算密度 3.51、W4A4 7.02、W16A16 相对 Tender 高 5.26×）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：SpinalHDL 描述 RTL（S-FPMA 加法 slice + 进位链 + FG/CG 补偿 LUT/进位注入），Synopsys Design Compiler TSMC 28nm@1GHz 综合；面积分解显示 Composable Mult 不再是主导（对比乘法器型 baseline 最高占 PE 面积 83.6%），UNICORE PE 面积降 13.6%–43%、GEMM 面积降 19%–49%。使用：同一硬件在 W4A4/W4A8/W8A8/W16A16 四种模式间切换（精度相关 mux 由 Control Unit 控制选择独立/融合模式），无需复制 FPU 或加宽乘法器阵列，实现多精度统一 GEMM 引擎。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore（Hardware/UniCore）。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
