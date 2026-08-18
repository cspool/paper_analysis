## Epilogue Unit（EU，后处理单元：冲突无关查找 + 加法树归约）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVA 的专用后处理单元，承接 VQ-GEMM 产出输出码本（OC）后的两步操作：(1) 冲突无关查找——每周期并行读 v=32 个 WI 索引，从 OC 取对应 FP16；(2) add-only 归约——垂直 32 输入加法树（单码本执行）或对角累加（多码本 C0-C3 输出级并行）。每个 EU 只含 32 个加法器，但每个加法器累加 d=8 个点积结果，故 32 个加法器处理 32×d/C=128 个权重元素，等价于常规 VQ 解码的 128 次 MAC——以加法器替代 MAC 且无冲突、无需乘法器。EU 是 decode 的关键路径（VQ-GEMM 256 cycles vs EU 4096 cycles，N=4096），与 GEMM 流水重叠，可扩展数量（4 EU 使 64GB/s 带宽饱和，再增仅增能耗，面积开销 3.5%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# EU 处理一块（v=32, d=8, N=4096）：
idx[0..31] = read WI(v×N tile)        # 并行读 32 索引
val[i] = OC_bank[idx[i]]              # 冲突无关查找（OC 行 → 独立 bank）
sum = adder_tree(val[0..31])          # 垂直 32 输入加法树（单码本）
# 多码本：对角累加跨 C0-C3，输出级并行
```
Table X 扩展性：EVA EU-4×1 2.12×、EU-32×1 16.95×、EU-32×4 64.84× speedup（vs 全冲突 VQ 1.00×），归一化阵列面积仅 1.01×/1.05×/1.18×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RTL 加法器树 + 多 bank OC SRAM（每 bank 一行，32 bank×1 FP16=64B/cycle；4 EU 时 32 bank×1×4=256B/cycle），与 GEMM 阵列输出直连（无片外往返、避免带宽争用）；EVA 配置 4×32-input 加法树。使用方式：LLM decode 阶段承接查找+归约；与 GEMM 流水重叠使 EU 接近峰值利用率；DSE（Fig. 8）决定 EU 数与带宽匹配——4 EU 在 500MHz/64GB/s 下每周期处理 4×v=128 索引，正好吃满带宽。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
