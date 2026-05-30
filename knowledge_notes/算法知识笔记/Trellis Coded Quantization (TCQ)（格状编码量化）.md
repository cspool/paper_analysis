## Trellis Coded Quantization (TCQ)（格状编码量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Trellis Coded Quantization (TCQ) 由 Marcellin 和 Fischer (1990) 首次提出，将 Trellis Coded Modulation 的概念应用于量化。TCQ 定义了一个 (L, k, V) trellis——一个具有 2^L 个节点的有向图，每个节点有 2^{kV} 条入边和出边，每个节点关联一个值 ∈ R^V（形成 codebook C ∈ R^{2^L × V}）。对长度为 T 的序列 S ∈ R^T，每连续 V 个元素分配给 trellis 上的一个节点，约束是分配的节点形成一条 walk。重建序列 Ŝ 由 walk 中节点值的拼接给出。由于相邻元素由 2^{kV} 条边连接，只需存储来自哪条边（k 比特），而非整个 codebook 索引。在可加失真度量（如 MSE）下，最优 Ŝ 可通过 Viterbi 算法在 O(2^L T) 时间内找到——复杂度与比特率 k 无关，与序列维度 T 线性。这使 TCQ 可实现超高维量化（维度 > 100），克服了 Vector Quantization（VQ）的指数复杂度瓶颈（VQ 需 O(2^{kd} d) 时间和空间）。对于 i.i.d. 高斯源，随 L 增大，TCQ 高效逼近无限长度失真率 D_R（2-bit 时 D_R=0.063，QTIP 256D TCQ 达 0.069 MSE，而 QuIP# 8D VQ 达 0.089，标量 Lloyd-Max 达 0.118）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TCQ 在 QTIP 中的算法 pipeline（以量化一个 256 维权重序列为例，L=16, k=2, V=1）：
```
输入: 序列 S ∈ R^{256}, (L=16, k=2, V=1) bitshift trellis G, codebook C ∈ R^{2^16}
1. 定义价值函数: V_t(x) = min{ Σ_{i=1}^t ||C_{x_i} - s_i||² | x_1..x_t 是 G 上路径, x_t=x }
2. 初始化: V_1(x) = ||C_x - s_1||², ∀x ∈ [0, 2^L)
3. for t = 2 to T:
     for each node y ∈ [0, 2^L):
       V_t(y) = min_{(x,y)∈G} [V_{t-1}(x) + ||C_y - s_t||²]
       记录回溯指针 ptr_t(y) = argmin_x
4. 反向回溯: x*_T = argmin_x V_T(x), 然后 x*_{t-1} = ptr_t(x*_t)
5. 输出 Ŝ = [C_{x*_1}, ..., C_{x*_T}], 编码为起始状态(L bits) + 每步 k 比特 = L + kT bits
```
关键：V_t(y) 的更新仅考虑 G 中 (x,y) 有边的节点对，bitshift trellis 中每个节点仅 2^{kV}=4 条入边，故每步 O(2^L × 2^{kV}) = O(2^{L+kV}) = O(2^{18}) ≈ 262K 操作，总复杂度 O(2^L T) ≈ O(16.8M)，远小于暴力搜索 O(2^{kT}) = O(2^{512})。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TCQ 在 LLM 量化中的使用：(1) 离线阶段——用 Incoherence Processing (RHT) 使权重近似 i.i.d. 高斯 → BlockLDLQ 框架中每 T_x×T_y 权重块作为序列用 TCQ 量化 → 存储编码比特；(2) 推理阶段——bitshift trellis 解码 + compute-based codes 即时生成权重 → GEMV。TCQ 的线性复杂度突破了 VQ 的维度限制，使有效维度从 ≤8 提升到 256+。开源实现：https://github.com/Cornell-RelaxML/qtip（QTIP 论文）。TCQ 也可用于其他需要高维量化的场景（如压缩感知、图像编码等），原始 Marcellin & Fischer (1990) 论文主要针对语音/图像压缩。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
