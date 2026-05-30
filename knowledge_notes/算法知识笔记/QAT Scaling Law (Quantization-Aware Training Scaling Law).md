## QAT Scaling Law (Quantization-Aware Training Scaling Law)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QAT 缩放定律是描述量化感知训练中量化误差 δ_p 随模型参数量 N、训练数据量 D 和量化粒度 G 变化的数学关系。与现有方法（通过 EPM 修改 Chinchilla 的 N 项）不同，QAT 缩放定律将量化误差建模为 Chinchilla loss 的独立加项：L(N, D, G) = A/N^α + B/D^β + E + δ_p(N, D, G)。其中 δ_p(N, D, G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}。使用对数项 log₂(G) 满足边界条件 G=1（无量化）时 δ_p=0。拟合参数 k, γ_N, γ_D, γ_G 均为正数，表明：δ_p 随 N 增大而减小（大模型更鲁棒）、随 D 增大而增大（更多训练数据放大全精度 vs 量化差距）、随 G 变粗而增大（粗粒度量化误差更大）。γ_N, γ_D, γ_G 的大小反映量化误差对各自变量的敏感度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QAT 缩放定律的建立流程：
```
# Step 1: BF16 基线训练 + Chinchilla 拟合
for each (N, D):
    model = Llama3Style(N)
    loss_bf16[N,D] = train_bf16(model, D)
fit L_bf16(N,D) = A/N^α + B/D^β + E  # 用 Huber loss + L-BFGS, 约束 α=β

# Step 2: W4A4 QAT 实验
for each (N, D, G) in grid:  # N∈{74,145,297,595}M, D∈{10,20,50,100}B, G∈{32,64,128,256,per-token/channel}
    model = Llama3Style(N)
    loss_W4A4[N,D,G] = train_W4A4(model, D, G)
    δ_W4A4[N,D,G] = loss_bf16[N,D] - loss_W4A4[N,D,G]  # ground truth

# Step 3: 拟合 δ_p 缩放定律
fit δ_p(N,D,G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}
# 80 次实验数据，Huber loss + L-BFGS

# Step 4: 外推验证
predict δ_p for 973M model at 100B/200B tokens → compare with actual
```

EPM 推导：eff(C) = [A / (A + k · D^{γ_D} · (log₂(G))^{γ_G} · N^{α-γ_N})]^{1/α}。当 α > γ_N 时（W4A4 满足），eff(C) 随 N 增大而减小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缩放定律的实用价值：(1) 预测任意 (N,D,G) 组合的量化误差，指导模型设计和训练策略；(2) 通过 EPM 评估 W4A4 vs W8A8 的 cost-accuracy trade-off（EPM > 0.5 时 4-bit 更优）；(3) 量化误差分解（W4A16/W16A4）揭示权重 vs 激活的贡献，指导优化方向；(4) 结合 FC2 瓶颈分析，通过混合精度消除主要误差源。论文使用 PyTorch + OLMo2 训练框架，未开源专用代码。拟合使用 Huber loss（对 outlier 鲁棒）+ L-BFGS 优化器。

涉及论文标题：
- Scaling Law for Quantization-Aware Training
