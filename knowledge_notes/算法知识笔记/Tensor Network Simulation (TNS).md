## Tensor Network Simulation (TNS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tensor Network Simulation（张量网络模拟）是另一种量子线路模拟范式：与 SVS 类似做矩阵向量乘，但引入 bond dimension D——线路中任意矩阵允许的最大秩；若某矩阵秩超过 D，则把其 D 之外的特征值置 0 做低秩近似（截断）。D 越小近似越强但内存/计算越省，使 TNS 能模拟比 SVS 更宽更深的电路（论文模拟到 40 qubit），代价是输出有近似误差。TNS 对低纠缠电路（自然低 D）尤其高效。
- 本论文把 TNS 作为 TUSQ 的第二个验证后端：TUSQ 的所有组件（ECM+DFTT）都只依赖"矩阵向量乘 + 从向量采样"，因此可直接叠在张量网络模拟器上。TNS+TUSQ vs 未优化 TNS（CUDA-Q tensornet-mps）对 40-qubit QFT/Adder/QAOA(p=2)（bond dimension=16、100k shots、α=0.01、β=100）平均加速 248.39×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TNS 张量计算（低秩近似）：M ≈ M_D = Σ_{i=1}^D σ_i u_i v_i†（截断奇异值分解，σ_i 为前 D 大奇异值），从而把 2^n 维态矢量压缩为 O(n·D²) 规模的张量网络（MPS/MPS 形式）；门的应用变成张量收缩。
- TNS+TUSQ 流水线：① ECM 在 CPU 预采样 ER、合并/剪枝电路实例（与 SVS 场景相同）；② 对每个剩余电路用 TNS 计算输出向量（cuTensorNet v2.9.1 后端）；③ DFTT 用张量网络态在树上的计算/uncompute 复用共享前缀；④ 输出按频率加权采样并平均。
- 实测对比：Unopt TNS vs TNS+TUSQ 时间（秒）：QFT40 1119642→3444、Adder40 628889→2625、QAOA40(p=2) 158407→805；未优化 TNS 在 40 小时超时内未完成（按 100/1k/10k shots 外推）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：NVIDIA cuTensorNet（cuQuantum SDK 组件，v2.9.1）提供 GPU 张量网络模拟 kernel；CUDA-Q 通过 `--target tensornet-mps` 或 `tensornet-mps` flag 调用；本论文的 baseline 用 CUDA-Q 0.11.0 + tensornet-mps 做未优化 TNS。使用：TNS 对 SVS 内存 O(2^n) 不可行的大电路（>30 qubit）是替代方案，但需接受 bond dimension 截断带来的近似；TUSQ 证明冗余消除优化与其正交可叠加。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
