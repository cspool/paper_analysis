## Effective Step Size in Quantized Weights（量化权重有效步长）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Effective Step Size（有效步长）衡量单次参数更新中权重的变化幅度。对于全精度权重，effective step size ≈ |μ·g|（LR × 梯度），与 LR 高度相关，因此调度 LR 可直接控制 coarse-to-fine 优化。但对于 QAT 中的量化权重 w_q，由于其 effective step size `|Δw_q^t| = |w_q^t - w_q^{t-1}|` 具有离散特性：`|Δw_q^t| ≈ δ^t·I[w_d^t ≠ w_d^{t-1}]`（δ^t 为量化级别间距），即要么为零（未发生 transition），要么为 δ^t（若发生 transition）。因此量化权重的 average effective step size 主要由发生 transition 的权重数量（即 TR）而非 LR 决定。这一洞察是 TR 调度技术的基础——用调度 TR 替代调度 LR，以直接控制量化权重的 coarse-to-fine 优化进程。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化权重 effective step size 的推导（Eq.6-9）：
```
给定: w_q^t = δ^t·w_d^t, w_q^{t-1} = δ^{t-1}·w_d^{t-1}
假设: δ^t ≈ δ^{t-1}（单个 update 内 scaling factor 变化极小）
      且单次 transition 仅跨越一个量化级别（通常情况）

多 bit 量化（round function, Eq.7 → Eq.9）:
|Δw_q^t| = |δ^t w_d^t - δ^{t-1} w_d^{t-1}| 
         ≈ δ^t·|w_d^t - w_d^{t-1}|
         = δ^t·I[w_d^t ≠ w_d^{t-1}]   （transition 时跨一级别）

二值量化（sign function, Eq.8）:
|Δw_q^t| = ½|δ^t w_d^t - δ^{t-1} w_d^{t-1}| 
         ≈ ½δ^t·I[w_d^t ≠ w_d^{t-1}]  （w_d∈{-1,1}）

推论: Average effective step size ≈ δ^t·(transitions/N) = δ^t·k^t
```
关键实验证据（Table S6, CIFAR-100, ResNet-20 W2A2）：在训练后期将 final target TR 从 0（无 transition）递增到 1e-3（较多 transition），average effective step size 从 0 增至 5e-4，test accuracy 从 65.61%±0.21 降至 62.12%±0.70，精度标准差从 0.21 增到 0.70。验证了减小后期 transition/effective step size 对 QAT 收敛稳定的关键性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Average effective step size 在论文中用作监控指标而非训练目标：在每个 iteration 计算 `mean(|w_q^t - w_q^{t-1}|)` 并记录到训练曲线。论文通过对比 LR 调度（Fig. 1c 蓝线）和 TR 调度（Fig. 1c 红线）下的曲线验证 TR 调度有效性——TR 调度下 effective step size 平滑衰减到零，LR 调度下噪声大且训练后期不收敛。作为 TR 的变体，论文也讨论了直接调度 effective step size（Eq. S3: k^t = Σ|w_q^t(i)-w_q^{t-1}(i)|/N），但由于不同层的量化权重 scale（δ^t）不同，需逐层搜索初始 target value 的超参，而 TR 调度（使用 w_d 计数）与 scale 无关，更具通用性和易用性。

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---
