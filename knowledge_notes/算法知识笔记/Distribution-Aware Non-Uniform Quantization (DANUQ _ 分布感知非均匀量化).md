## Distribution-Aware Non-Uniform Quantization (DANUQ / 分布感知非均匀量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DANUQ (Distribution-Aware Non-Uniform Quantization) 是 FedWSQ 提出的新型非均匀量化方法，专为联邦学习中 LMPU (Local Model Parameter Update) 的通信压缩设计。与需要学习量化参数（scale, zero-point）的传统方法不同，DANUQ 使用**固定的预计算量化级别（QLs）**，无需每轮传输额外的量化参数。核心设计分两层：(1) **Scaling**：用 LMPU 的标准差 σ 而非 absmax 作为 scale factor，因为 σ 对 outlier 更稳健且与 N(0,1) 假设一致。Global scaling vector s_g 通过 EMA 在 server 端跨 client 聚合并广播，保证量化一致性。(2) **QLs 设计**：假设归一化后 LMPU ∼ N(0,1)，最小化期望量化误差 E[(x - q)^2] = Σ_{r=0}^R ∫_{u_r}^{u_{r+1}} (x - q_r)^2 φ(x) dx，通过暴力搜索在离散化搜索空间中枚举所有可能 QLs 组合，找到使该目标最小的 {q_1, ..., q_R}。预计算的最优 QLs 为：1-bit: [-0.798, 0.798]；2-bit: [-1.224, 0, 0.765, 1.724]；4-bit: 16 个非均匀间隔值。DANUQ 不需要 backpropagation 学习量化参数，因此在 FL 场景中不引入额外的通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DANUQ 量化 LMPU 的具体流程（Client-side）：

```python
# Input: ΔW_i ∈ R^P (full-precision LMPU for layer l)
#        s_{g,l} (global scale for layer l, received from server)
#        QLs = [q_0, q_1, ..., q_R] (pre-computed optimal quantization levels for B-bit)

# Step 1: Normalize by global scale
ΔW_norm = ΔW_i / s_{g,l}    # assume ~ N(0,1) after normalization

# Step 2: Build quantization boundaries
u_0 = 0                      # fixed q_0 = 0 for B >= 2
for r = 1 to R:
    u_r = (q_{r-1} + q_r) / 2   # boundary between q_{r-1} and q_r
u_{R+1} = +inf

# Step 3: Non-uniform quantization by nearest-neighbor mapping
# For the symmetric half [0, +inf):
for each element x in abs(ΔW_norm):
    if x < u_1:
        idx = 0
    elif u_1 <= x < u_2:
        idx = 1
    ...
    elif u_R <= x:
        idx = R
    q_x = sign(x) * QLs[idx]   # restore sign

# For 1-bit (special case: omit q_0=0 constraint):
# QLs = [-0.798, 0.798], boundary at 0
# idx = 0 if x < 0 else 1, q_x = QLs[idx]

# Step 4: Build output
ΔW̄_i = q_x_values                # B-bit indices into QL lookup table
s_i = std(ΔW_i)                  # local scale (transmitted alongside)
return (ΔW̄_i, s_i)
```

**Annotations**: QLs 在 [0, +∞) 非均匀分布：密集区域（均值附近 ∼0）分配更细粒度 QLs，稀疏区域（尾部 ∼2.5+）分配更粗粒度。1-bit 特例省略 q_0=0 约束以允许两个 QLs 对称放置（≈ ±0.798）。Dequantization 在 server 端通过查表 + 乘 s_{g,l} 恢复 full-precision。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DANUQ 的实现分两阶段：(1) **离线预计算 QLs**：给定 B-bit，在经验合理的搜索范围内（如 [-3, 3] for N(0,1)），将 QL 空间离散化为候选网格，枚举所有满足排序约束 q_0 < q_1 < ... < q_R 的组合，评估 Eq.(10) 的目标函数，选最小化的配置。使用并行处理加速。QLs 计算一次后便固定，所有 client 和所有通信轮次复用。(2) **在线量化/反量化**：Client 端将归一化的 LMPU 值通过二分查找或阈值比较映射到最近 QL，传输 QL 索引（B-bit 整数）和 per-layer scale s_i（float 向量）。Server 端通过 QL 查找表 + 乘 scale 恢复全精度值。与 FedPAQ (uniform quantization + absmax + 概率舍入) 相比，DANUQ 的主要优势是：(a) "标准差 scale"代替 "absmax"避免了 outlier 导致的大部分值 underflow；(b) "非均匀 QLs"在概率密度高的区域提供更高精度。在使用上，DANUQ 与 WS 协同工作：WS 稳定训练过程使 LMPU 统计更接近正态分布，DANUQ 利用这一分布特性进行高效压缩。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
