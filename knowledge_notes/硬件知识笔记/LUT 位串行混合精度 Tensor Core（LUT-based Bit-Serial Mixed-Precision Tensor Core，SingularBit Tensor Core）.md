## LUT 位串行混合精度 Tensor Core（LUT-based Bit-Serial Mixed-Precision Tensor Core，SingularBit Tensor Core）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LUT 位串行混合精度 Tensor Core 是 SingularBit 加速器的计算核心：一个 32×32 PE 阵列，用查找表（LUT）+ 位串行（bit-serial）方式在单一数据通路上执行 1–4 bit 混合精度权重 × FP16 激活的矩阵乘。它建立在低比特 LLM 推理的 LUT 计算范式之上（LUT Tensor Core [51]、SLIM-LLaMA [52]、ISSCC 数字 CIM [53]）：把低位宽权重乘法转化为"激活部分和查表 + 取反 + 移位累加"。核心设计：activation loader 对 4 个输入通道预计算 FP16 部分和并只生成 8 个 LUT 条目（0 通道编码符号、1–3 通道预计算全部幅度组合，大幅减少表项数）；计算时权重 1–3 位经多路选择器（multiplexer）选对应部分和、0 位决定是否取反；SingularBit-W Compute FSM 提供当前 bit 位置（activation loader 据此对 FP16 部分和的指数加 bit offset 实现位串行），weight loader 按 rank 广播权重位与 group scale；intra-group accumulator 做 FP16 组内累加、inter-group accumulator 按组 scale 聚合，双级累加支持 group 量化。相比传统多精度核（需独立数据通路或重配置 [49][50]），本设计用同一硬件路径处理 1–4 bit，位宽直接换算成延迟/能耗。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一个 4-bit rank 区域的矩阵乘，Fig.8/9）：①activation loader 读 4 个输入通道的 FP16 激活，预计算 8 个部分和（0 通道作符号位、1–3 通道幅度组合），按当前 bit offset 调整指数后广播到 PE 行；②weight loader 从（ARB-LLM 式）$\hat{U}$/$\hat{V}^T$ 的二进制基表示读出当前 rank 的 1–4 bit 权重位与行/列缩放因子，广播到 PE 列；③每个 PE 用多路选择器按 1–3 位选部分和、0 位取反，得到该 bit 的乘加结果；④intra-group 累加器组内 FP16 累加（一个量化组共享 scale），inter-group 累加器乘组 scale 后聚合输出；⑤FSM 推进 bit 位置、weight loader 在 group 边界同步发 scale，全部 bit 累加完即得 $X\hat{U}$ 或 $X\hat{V}^T$ 结果。结果示例（论文表 VII）：1× Tensor Core 307.3 mW/5.58 mm²（PE Array 168.1 mW/2.73 mm² 占 54.7%/48.9%、Buffer 99.2 mW/1.94 mm²、GeMV/SIMD 16.6 mW/0.82 mm²、Control 23.4 mW/0.09 mm²）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：作为 ASIC 逻辑综合实现（28nm CMOS @1GHz，论文未指明 EDA 工具）；配套 LUT Tensor Core 开源工作可参考（https://github.com/... LUT Tensor Core，论文未给链接）。使用方式：权重以 ARB-LLM 二进制基格式（$\sum_i\alpha_{r,i}\alpha_{c,i}B_i$）常驻 DRAM，decode 时按位流式加载；LUT 预计算只做一次、跨 PE 行复用削减算术能量（论文归因于"预计算 group 缩放因子乘法与四通道累加、跨 PE 行复用"）；每个 tensor core 还配局部 GeMV 单元（矩阵-向量）与 SIMD 单元（bias/非线性），由 core scheduler 管理数据移动与 buffer。论文数据：tensor core 直接消费低比特值（无需反量化），是 79% 系统能耗下降与 5.3× reasoning speedup 的关键部件。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
