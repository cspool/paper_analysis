## AI Cube（AIC）与 AI Vector（AIV）单元

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DaVinci AI Core 内的两类计算单元（ENEC 论文 Figure 2 称 decoupled AIC and AIV）：
- AI Cube（AIC）：面向矩阵运算优化的单元，提供稠密线性代数的高吞吐（如 FP16 一次执行 16×16 矩阵乘加），操作数来自 L0A/L0B（fractal 格式 FRACTAL_ZZ/FRACTAL_ZN），结果入 L0C（FRACTAL_NZ）；灵活性有限，只擅长规整矩阵乘。
- AI Vector（AIV）：面向大规模向量化运算的单元，支持 gather、归约、逐元素操作（Add/Exp/LayerNorm/Softmax），操作数必须在 Unified Buffer 中（32 字节对齐）；每个 AI Core 集成 1 个 AIC + 2 个 AIV（910B2：24 AIC + 48 AIV）。
ENEC 论文的架构分析：压缩计算是"向量逐元素操作，不能直接利用 Cube"——因此 ENEC 把压缩/解压 kernel 只放在 AIV 上，与 AIC 的模型算子并行（协作推理流水线）。AIV 的限制（限制传统压缩算法）：无条件分支、无 scatter/gather 指令、整数算术指令集少、32 字节段内禁止 SIMD、无轻量线程间同步。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AIV 执行 ENEC 压缩的流水（每块 16384 元素）：
```
UB ← MTE2(HBM 权重块)
AIV: 拆分指数 → 分支无关变换(加/乘/移位) → 分组位宽阈值(bitwise OR) → lane folding 打包(OR+shift)
UB → MTE3 → HBM 压缩流；同时 AIC 并行执行上一层模型 GEMM（L0A/L0B→L0C）
```
Annotations：AIV 与 AIC 解耦（各自有独立缓冲与指令流），靠队列同步；ENEC 让下一层解压在 AIV、当前层前向在 AIC 重叠，隐藏解压时间。IDD-Scan 就是为了绕过 AIV 的 32 字节段内 SIMD 禁令而设计的（见 kernel调度层条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AIC 用 Cube 指令（矩阵乘），AIV 用向量指令（AscendC 表达，经 CANN 编译）；数据由 MTE 搬运。使用：AIC 承担 GEMM/卷积等稠密计算，AIV 承担逐元素/规约/访存整形（softmax、layernorm、压缩/解压）；ENEC 是"AIV 承担非矩阵类运行时计算"的典型例子——压缩/解压吞吐 263-523 GB/s 全部来自 48 个 AIV 的向量化。局限：ENEC 明确说明 AIV API 缺乏位操作指令集（INT8/FP8 相关变换无法实现），是它不处理 INT8/FP8 权重的原因之一。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
