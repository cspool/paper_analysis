## LUT 查找表量化（非解析量化 / Codebook-based Lookup）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
非解析（non-analytic）量化去掉闭式量化函数，直接用 k-means 等最小化重建误差（MSE）学习映射，典型代表是查找表（LUT）量化：权重→码本索引，decode 时 1-to-1 查表重建（硬件加速器 GOBO、FIGLUT、LUT Tensor Core、LUT-DLA；软件框架 AQLM/QuiP#）。优点：表示灵活、2-bit 级精度仍高（Table VI：EVA 2-bit 下游平均比 LLM.265 VB 高 19pp）；缺点：不规则、非合并的查表访存导致 bank 冲突、并行化困难，且硬件需复制（duplication）或广播（broadcast）码本到多 PE——FIGLUT 广播 16×32×(8×16bit) 带宽、LUT-DLA 复制 256×(16×16bit) 寄存器、GOBO 复制 768×(8×16bit)，码本有效规模被限制在 ≤16 条目（Table I）。EVA 定位为第一个架构级 VQ/查表 LLM 推理加速器，核心洞察是把"查权重码本"变成"查输出码本"（见 VQ-GEMM 条目），从根上消除冲突且无需复制/广播。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 常规 LUT decode（FIGLUT 式）：激活作输入、权重模式作索引取预计算部分和
for each tile: partsum = LUT[act_pattern][weight_idx]   # 广播/复制到多 PE，同 bank 冲突时串行
# EVA 式：先 O = X·B（GEMM 产出全部"输入×centroid"点积的输出码本），再 y = Lookup(O, I)
```
查找次数减少、全部规则化、跨 bank 无冲突、每次访问带宽从 d 个 FP16 降为 1 个 FP16。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GOBO（寄存器复制）、LUT-DLA（复制）、FIGLUT（BCQ 二元编码 + 4-input LUT 广播）、LUT Tensor Core；软件侧 AQLM/QuiP#（PyTorch）。使用方式：用于低比特权重推理，尤其 2-bit 级；EVA 的对比基线 FIGLUT 在 32×32 阵列上 decode 利用率仅 4.34%、吞吐 44.49 GOPs（2.82× SA），而 EVA 以"输出码本查找"达 498.49 GOPs（31.64× SA）——查表对象（查 WC vs 查 OC）是硬件效率的分水岭。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
