## Student Float (SF4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Student Float (SF4) 是一种基于 Student's t-distribution 推导的 4-bit 查找表（lookup-based）量化数据类型。与 Normal Float (NF4) 假设权重服从正态分布不同，SF4 基于对 30+ DNN 的大规模 profiling 发现——大多数 DNN 的权重和激活分布由 Student's t-distribution（自由度 ν≈5）最优近似。SF4 通过将概率质量等分为 16 份，经 t-distribution 分位数函数 Q_S(p;ν) 映射，归一化到 [-1,1]，生成 16 个量化层级。具体导出流程（Algorithm 1）：(1) 设定 δ = 0.5×(1/32 + 1/30)；(2) 在概率空间等距生成 p₁=δ,...,p₈=0.5,...,p₁₆=1-δ（固定 p₈=0.5 确保零点无损表示，对称侧多分配值以适配现代激活函数的正偏特性）；(3) s̃ᵢ = Q_S(pᵢ; ν=5)，经 t-distribution 分位数函数映射；(4) sᵢ = s̃ᵢ / maxᵢ|s̃ᵢ| 归一化到 [-1,1]。SF4 的 16 个量化层级为固定值，可作为 NF4 的直接替代品用于 weight-only PTQ。当 ν→∞ 时，t-distribution 收敛到正态分布，SF4 收敛到 NF4。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SF4 在 PTQ pipeline 中的角色是码本（codebook）：

```
# === 离线阶段：SF4 码本生成（一次性） ===
ν = 5  # 基于 30+ DNN profiling 最频自由度
δ = 0.5 * (1/32 + 1/30) ≈ 0.0323
p = [δ, ...均匀间距..., 0.5, ...均匀间距..., 1-δ]  # 16 个概率值
s̃ = Q_S(p; ν)    # t-distribution 分位数函数
s = s̃ / max|s̃|    # 归一化到 [-1,1]
# s = [s₁, s₂, ..., s₁₆] 即为 SF4 的 16 个量化层级

# === 推理/量化阶段：Block-wise SF4 量化 ===
W_flat = W.reshape(-1)
blocks = W_flat.reshape(B, 128)    # block size 128
for b in 1..B:
    w_max[b] = max(|blocks[b,:]|)
    for i in 1..128:
        x = blocks[b,i] / w_max[b]      # 归一化到 [-1,1]
        idx = argminⱼ |x - sⱼ|           # 最近邻 SF4 层级
        Ŵ[b,i] = w_max[b] * s_{idx}     # 解码

# 存储格式：4-bit index + per-block FP16 w_max
# 推理时查表解码：Ŵ = w_max × SF4_table[index]
```

与 NF4 对比：NF4 的概率空间划分基于 Gaussian 分位数（假设 N(0,σ²)），SF4 基于 t-distribution 分位数（假设 t(ν=5)）。论文实验证明 SF4 在 LLaMA2-7B 上平均 LAMBADA 准确率比 NF4 高 0.76%。SF4 对不同自由度的敏感度较低——ν=4/5/6 均接近最优——因此论文固定 ν=5 作为通用选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SF4 实现为修改版 Intel Neural Compressor 库（论文实验平台）中的查找表量化后端。在 QLoRA 风格的 weight-only PTQ 中，SF4 可以直接替换 NF4：只需替换 16 个码本值，block-wise absmax 归一化和查表解码流程完全相同。开源代码位于 https://github.com/cornell-zhang/llm-datatypes。由于 SF4 是纯查找表格式（类似 NF4），其硬件实现需要浮点查找表和高精度 MAC 单元（论文未为 SF4 设计专用 MAC），因此 SF4 主要用于：(a) weight-only 内存绑定推理（memory-bound inference），查表解码开销可忽略；(b) 作为高精度参考格式指导硬件高效数据类型（如 E2M1）的设计。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
