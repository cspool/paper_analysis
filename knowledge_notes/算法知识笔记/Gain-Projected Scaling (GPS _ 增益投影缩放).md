## Gain-Projected Scaling (GPS / 增益投影缩放)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPS（Gain-Projected Scaling）是 PTQ4ARVG 提出的首个基于数学优化（而非经验设计）的量化 scaling 策略。核心思想是通过 Taylor 展开量化损失，将 scaling 对量化的影响量化为 gain function g(s)，然后通过求导得到闭式最优解。具体步骤：(1) 将激活-权重量化损失 E(x,W) 分解为激活量化损失 E_x 和权重量化损失 E_W 的上界；(2) 用 Taylor 展开近似 E_x 和 E_W（以 MSE 替代 Hessian）；(3) 引入 per-channel scaling factor s，分析 scaling 后 E'_x < E_x（激活量化损失降低）和 E'_W > E_W（权重量化损失增加）；(4) 定义 scaling gain g(s) = E_x - E'_x - (E'_W - E_W)；(5) 对 g(s) 求导得闭式解 s_i = s_k · √(Σ_j |ΔW_{i,j}·x_i|) / √(Σ_j |W_{i,j}·Δx_i|)，其中 s_k 为激活 range 最大通道的 scaling factor。GPS 是 Equivalent Scaling 的数学优化版本，与 SmoothQuant 等经验方法不同，GPS 提供了理论保证的最优解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 RAR-B 的 qkv 线性层（X∈R^{T×n}, W∈R^{n×m}）为例：

```
输入: 激活 X (校准数据), 权重 W, bit-width b
输出: 最优 scaling factor s ∈ R^n

# Step 1: 准备量化误差数据
X_q = uniform_quantize(X, b)    # 模拟量化
W_q = uniform_quantize(W, b)
ΔX = |X - X_q|                   # 激活量化误差 (per-channel)
ΔW = |W - W_q|                   # 权重量化误差 (per-channel)

# Step 2: 找激活 range 最大的通道
R_x[i] = max(X[:,i]) - min(X[:,i]), i=1..n
k = argmax(R_x)                  # 激活 range 最大的通道索引

# Step 3: 计算 s_k (使该通道激活和权重 range 对齐)
s_k = sqrt(R_x[k] / R_W[k])

# Step 4: 闭式解计算其余 scaling factors (Eq. 16)
for i = 1 to n:
    if i != k:
        num = sum_{j=1}^{m} |ΔW[i,j] * X[i]|   # 权重量化误差 × 激活值
        den = sum_{j=1}^{m} |W[i,j] * ΔX[i]|    # 权重值 × 激活量化误差
        s[i] = s_k * sqrt(num / den)

# Step 5: 应用等效缩放并融合 (Eq. 2)
X' = X / s                       # 激活除以 s (推理时不执行)
W' = W * s.unsqueeze(1)          # 权重乘以 s (离线融合到 AdaLN 权重)
```

GPS 的关键数学性质：(1) 当 s>1 且 s_i > s_j（基于 Remark 1 的统计观察），scaling 使激活量化损失降低而权重量化损失增加；(2) g(s) 是凸函数，求导后得到的 s_i 是全局最优解；(3) GPS 仅应用于 qkv 和 fc1 层（scaling factor 可被吸收的层），且 scaling factor 离线融合实现零推理开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPS 的实现关键是：(1) Remark 1 的统计验证——当 R_x[i] > R_x[j] 时，超过 98% 的通道满足 s_i > s_j，超过 99.5% 的通道满足 R_x[i]/s_i > R_x[j]/s_j，这保证了 scaling 后激活 range 的相对顺序不变；(2) 逐通道独立计算 s_i，无需迭代训练或反向传播（vs OmniQuant 需数小时训练）；(3) 计算复杂度低——仅需一次前向量化获取 ΔX 和 ΔW，然后逐通道计算 s_i；(4) GPS 可作为 plug-and-play 组件集成到任意量化框架。PTQ4ARVG 论文的实验表明，GPS 在 RAR-B W6A6 上将 FID 从 SmoothQuant 的 63.77 降至 36.51（-42.7%），优于所有经验 scaling 方法（OS+, RepQ*, SQ+RepQ*）。开源代码：https://github.com/BienLuky/PTQ4ARVG。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models
