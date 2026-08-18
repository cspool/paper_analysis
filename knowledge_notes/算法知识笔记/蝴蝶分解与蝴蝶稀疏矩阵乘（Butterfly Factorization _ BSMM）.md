## 蝴蝶分解与蝴蝶稀疏矩阵乘（Butterfly Factorization / BSMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 蝴蝶分解是把稠密矩阵 W∈R^(n×n) 近似为 log2 n 个块对角稀疏因子矩阵的乘积 W≈B_n^(1)·B_n^(2)···B_n^(log n) 的矩阵分解：第 k 个因子 B_n^(k) 对固定距离 2^k 的索引对做结构化两两混合（2 参数 2×2 混合参数化下每因子 2n 参数、总参数 2n log2 n，相对稠密 n×n 的压缩比 2 log2 n / n）。乘法（BSMM，butterfly-sparse matrix multiplication）只需 O(n² log n) 复杂度，比稠密投影 O(n³) 低一个数量级。数学根源是 Cooley-Tukey FFT 的蝴蝶图——DFT 矩阵可分解为 log n 层稀疏因子，每层做固定距离的加/减乘混合（radix-2 蝴蝶 a±b·ω）；把该结构推广到一般线性变换即为 butterfly factorization（Tri Dao 等 ICML 2019 / Monarch ICML 2022 系列）。本地知识库旁证：FWHT/Fast Hadamard Transform 条目同样用 in-place butterfly 结构（log₂n 层、每层 n/2 对 (a+b, a-b)）；Block-Sparse Attention 条目用固定 butterfly sparsity pattern 逼近任意稀疏矩阵。
- MLX 论文对先验蝴蝶稀疏的批评：全局蝴蝶分解应用于整个投影矩阵，大 d 时分解问题复杂度高、收敛难、近似误差大；且 GPU 上蝴蝶 kernel 运算强度低（bandwidth-bound）却远低于 CUDA 带宽 roofline（多级 strided/shuffle 重排破坏局部性 + stage-wise 依赖与批量同步/tile 规则执行错配，执行单元不匹配——只能跑 CUDA core 而非 TensorCore）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BSMM 的蝴蝶层级计算（n=8 的 3 层分解，radix-2 双路混合）：
```
# 输入向量 x ∈ R^8，因子层 k=0,1,2（stride = 2^k）
for k in 0..2:                              # 每层 O(n)
    stride = 2**k
    for i in 0..n-1 step 2*stride:
        for j in i..i+stride-1:
            a, b = x[j], x[j+stride]
            x[j], x[j+stride] = a + b, a - b   # 蝴蝶对混合（2 参数 2×2 混合）
# 总复杂度 O(n log n)；作为权重 W 的分解时，逐因子乘输入 = BSMM
```
MLX 把 BSMM 表达为三层嵌套循环映射到空间阵列：最内层 i2 在 4×4 网格上全展开（64 输出元素并发）、中层 i1 在 PE 内本地执行、外层 i0 作为数据流图迭代由片上序列器驱动；蝴蝶层的确定性 stride（±2,±4,±8,...）直接映射为 skip-hop 网格的跳距，PE_x 把部分和路由给消费 PE_{x+s}，多个 BSMM 层并发执行形成严格分层片上流水（Fig.10）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：(1) 算法：把 QKV/FFN 投影权重离线分解成蝴蝶因子（全局版 Monarch 式 O(D log D)，或 MLX 分层版 O((D²/B)·log B)），推理时输入逐因子乘；(2) 软件：PyTorch 层用矩阵分块 + 分层交换实现 BSMM kernel（H100 上 prefill 结合 FFT-CMP 2.72× vs eager / 1.64× vs FlashAttention2）；(3) 硬件：MLX 空间阵列用 CDC/tagged-block/skip-hop 路由执行蝴蝶层折叠流水，相对先验稀疏加速器（SpAtten/DOTA/Sanger/ViTALiTy/BitVert）最多 5.8× 加速，与 FABNet（FPGA 蝴蝶加速器）重实现对比 1.19-1.30× 端到端加速、1.14× LUT 开销。局限：蝴蝶分解是近似（有精度损失）、需离线分解开销、GPU 上受执行单元限制收益打折。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
