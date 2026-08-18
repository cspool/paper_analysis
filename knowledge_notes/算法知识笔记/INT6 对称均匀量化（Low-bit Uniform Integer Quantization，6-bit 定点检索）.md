## INT6 对称均匀量化（Low-bit Uniform Integer Quantization，6-bit 定点检索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT6 是对称均匀线性量化：把浮点值 x 以动态缩放因子 α=max(|x|) 线性映射到 6-bit 有符号整数区间 [-31,31]，Q_x=round(31·x/α)（S=31，无零点和无裁剪）。ParetoES 用它做稀疏 embedding 检索的低比特格式，与 Ultra-CSR 编码结合使有效内存带宽较 FP32 提升 6×（每非零从 32-bit 降到 6-bit）。位宽权衡：INT6 保持 Recall@100 不变，而 5-bit/4-bit 分别掉 Recall 最多 10%/32.5%。与 AccelES 的关键差异：ParetoES 不裁剪（no clipping）——裁剪虽可保 Recall 但会压制高范数分量、扭曲方向相似度（余弦/内积语义），不裁剪保留角度缩放是更保几何的选择。混合精度策略：聚类全精度浮点执行，聚类后向量/质心/查询一次性量化到 6-bit，使在线检索全程整数内积。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化的张量计算：
```
# 离线：对每个稀疏向量 x（或质心/查询），动态缩放
alpha = max(|x_j| for j in nnz(x))          # 每向量一个缩放因子
Q_x[j] = round(31 * x[j] / alpha),  for j in nnz(x)   # -> [-31, 31]
# 在线：内积在量化域近似
<A_i, v> ≈ (alpha_i * alpha_v / 961) * sum_j (Q_A[i,j] * Q_v[j])
# 因排序只需相对大小，实际检索直接比较 sum_j Q_A[i,j]*Q_v[j]（尺度因子单调共享）
```
pipeline 位置：离线端（聚类之后、ReSparse 之后/同时）量化全部矩阵元素并连同缩放因子编码进 Ultra-CSR packet；在线端查询向量同样量化后下发 FPGA。INT6 使每 512-bit HBM packet 容纳 30 个非零（FP32 下仅 ~11），DSP 单周期可做 3 个 6-bit 乘法（相对 FP32 并行度 2.7×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：对称量化在 FPGA DSP 上以 6-bit 整数乘法直接执行（Alveo U280 DSP48 支持小位宽乘法复用）；缩放因子 α 存每行/每向量头（Ultra-CSR 元数据），检索排序只依赖量化内积的相对大小。使用场景：任何内存受限的稀疏检索/SpMV 加速器（AccelES-INT6 同方案，FPGA32 为 FP32 对比）。注意事项：无裁剪依赖 α=max 的动态缩放捕获全局动态范围，若个别向量有极端 outlier 会导致该向量其余元素量化分辨率下降（论文未展开讨论此 trade-off）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
