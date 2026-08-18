## Index Counter（索引计数器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Index Counter 是 OASIS 加速器 PE Line 内的高并行度部件（§IV-B），计算 WAQ LUT-GEMM 中"拼接索引的分布"——即统计每个唯一拼接索引（concat_idx = 激活索引 ∥ 权重索引，nW+nA bit）在 K 个输入通道上的出现次数 count[j]。原理：把每个拼接索引解码为 one-hot 向量（2^(nW+nA) 宽），多个 one-hot 组成矩阵，用 bit counter 做行求和得到各索引的出现次数；例如 2-bit 拼接索引 '01' 解码为 [0,0,1,0]，第 i 行的行和即索引 i 的出现次数。这些 count 作为加权和的权重，与 Cartesian Product LUT 条目相乘累加（MAC Tree）完成沿 K 的归约。硬件配置：每条 PE Line 含 32 个 16-input Index Counter（16-input 指同时接收 16 个拼接索引），每芯片 16 PE Line × 32 = 512 个；单条 16-in 计数器面积 2.71×10⁻¹ mm²/功耗 6.14×10⁻² W 级（Table II，按每 line 计 2.71×10⁻¹）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Index Counter 在 OASIS 主分支流水中的位置（图9a，图8 步骤④）：Concat Unit 产出拼接索引流 → 16 个拼接索引并行进入一个 16-input Index Counter → 各自解码为 one-hot 向量（并排成 one-hot 矩阵）→ bit counter 逐行求和得 count[0..2^(nW+nA)−1] → count 与 LUT 条目送入 MAC Tree 做加权和。示例（nW=nA=1，K=6，论文 Fig.6）：拼接索引序列如 [01,11,01,10,01,00] 解码成 4 行 one-hot 矩阵后行求和得 count={00:1, 01:3, 10:1, 11:1}，Y[n]=1·LUT[0]+3·LUT[1]+1·LUT[2]+1·LUT[3]。设计动机：把"K 次 FP16 乘法加法"转成"一次计数 + 最多 2^(nW+nA) 次查表加权和"，计数本身是纯位运算（one-hot 解码 + 加法树），面积/功耗远低于 FP16 MAC，可大规模并行（4096 个 Concat → 32 个计数器）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每个输入拼接索引经 decoder 转 one-hot（2^(nW+nA)-bit），16 路 one-hot 按位对齐后由行求和网络（bit counter/加法树）累加；为满足时序与面积约束取 16-input 粒度，每条 PE Line 布置 32 个以并行处理整层 K。使用场景：LUT-based 量化 GEMM 的通用配套部件——任何"索引流 → 直方图/计数 → 加权和"的归约范式；对比 GPU 上的 histogram/segmented reduction kernel，Index Counter 是其在加速器中的硬连线版本。无公开 RTL；论文用 TSMC 28nm 综合得到面积/功耗（Table II）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
