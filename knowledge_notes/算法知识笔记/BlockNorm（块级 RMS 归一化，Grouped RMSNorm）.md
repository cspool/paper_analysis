## BlockNorm（块级 RMS 归一化，Grouped RMSNorm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BlockNorm 是 LoKA Mods 的核心组件：把 GEMM 后的归一化改为沿特征维按固定块（如 256 元素）独立做 RMS 归一化，公式 RMSNorm((Wx+b).view(-1,BlockN)).view(B,N)，数学上等价于无参数 Grouped RMSNorm。设计动机：把归一化直接融合进 GEMM epilogue（输出 tile 还在片上 L1/L2/寄存器时完成），避免 HBM 往返；但标准归一化沿特征维求全局统计，与 GEMM 输出的物理数据布局错位，无法片上完成。
- 为什么用 RMS 而非 LayerNorm：RMSNorm 只按激活 L2 范数归一化、不做均值相减，避免低精度下相近数值相减的灾难性相消（mean cancellation）误差；BlockNorm 把全局单统计量拆成块内独立统计量，一个离群值不再压制全部特征，解耦特征子空间、增加表示自由度（类比 GroupNorm 分通道归一化）。
- 两种形态：Case 1 大 batch 小输出维——整行特征可放单个 thread block，统计全本地计算，行为与标准归一化一致，激活/量化/反量化可一并融合；Case 2 小 batch 大输出维——单 block 装不下整行，RMS 统计需跨 block 同步，抵消融合收益；缩小 batch tile 又引入 SM wave quantization 与 W 矩阵 L2 命中率下降。最终取舍：放松数学等价性，用固定块（train/test 一致）规避全局同步，鲁棒适配任意 shape。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算流程（一次 GEMM+归一化+激活融合）：FP8 GEMM 输出 y∈R^{B×N}（tile 在片上）→ y.view(B, N/BlockN, BlockN) → 每块独立算 RMS：rms_b = sqrt(mean(block²)+ε)，out_b = y_b / rms_b（可选乘缩放参数，无参数版即纯归一化）→ 紧跟 Hard Swish out·ReLU6(out+3)/6 与反量化 → 写回 HBM。块内全部在寄存器/SMEM 完成，无全局同步。
- 与标准 RMSNorm 对比：全局 RMSNorm 用单一统计量（一个离群值压低所有特征）；BlockNorm 每 256 元素独立归一化。论文实验（Wukong 生产模型，BlockNorm 256 vs RMSNorm）收敛到相同 loss，说明块内归一化保持模型质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：融合进 GEMM kernel 的 epilogue（类似 epilogue fusion，但论文强调应用在低精度上下文）；PyTorch 侧以 reshape+分组 RMS 实现（.view(-1,BlockN) 后逐块归一化再 view 回 (B,N)）。使用要点：块大小 train/test 严格一致以保证一致性；块足够大（如 256）时收敛对块大小不敏感；与 Hard Swish、量化/反量化同 kernel 融合最大化效率。关联：设计上对齐 MX（Microscaling）硬件标准的块共享缩放思路，避免全局同步开销；参考 GroupNorm、pRMSNorm（用 6.25% 神经元估 RMS）等先例。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
