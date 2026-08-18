## Unified Format Converter（统一格式转换器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Unified Format Converter 是 UNICORE 每个 PE 阵列前的硬件模块，承担双重角色：(1) 把低比特浮点输入无损转换为内部 E3M2 扩展格式，消除 subnormal（尾数左移归一化、指数同步补偿），使所有操作数满足 FPMA 对数近似的正常数假设；(2) 解码 DynFP 量化格式——根据每 group 的格式索引从 LUT 查询 E/M 布局、Z 值（负零重映射）与 I-flag（空位插入）映射到等效 E3M2 表示。所有张量仍以原始低比特格式存储与传输，E3M2 扩展仅存在于计算数据通路内，保留低比特存储的系统级收益。负零（E=0,M=0,S=1）由多路选择器检测并选择预定义 Z 值；格式索引在 1 cycle 内解码。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 运转流程：Weight Buffer/Unified Buffer 流出的 DynFP4/DynFP3 权重与激活进入转换器 → 以 (格式索引, 数值) 为索引查 LUT 得到 E3M2 正常数（普通值 (−1)^S·2^(Φ−B)·(1+M)；subnormal E=0,M≠0 → (−1)^S·2^(1−B)·M 左移归一化；负零 → Z 值）→ 输出送入 PreAdd（激活做指数偏置校正 T=A−B）或直接驻留 PE 列。Z 值在计算新权重张量时由离线贪心搜索确定的格式参数加载进转换器。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：SpinalHDL RTL 的小型 LUT + 多路选择器（负零检测 mux）+ 归一化移位逻辑；格式参数（E/M 布局、Z、I）存于 LUT，运行时 1 cycle 解码。元数据开销：权重每 group 4-bit 格式索引 + 8-bit scale（有效位宽 4.375 bits，比 MXFP4 高 2.9%），K/V 每 group 1-bit 索引；scale 因子存 GEMM 阵列 Rescale 单元的 8-bit 寄存器（占 GEMM 面积 0.73%）。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
