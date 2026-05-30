## Learning Rate Scaling Law for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Learning Rate (LR) Scaling Law for MoE 是 Joint MoE Scaling Laws 论文提出的 MoE 训练超参数指导公式，根据模型规模和 expert 数自动选择最优 peak learning rate：

$$LR(N_{act \setminus e}, E) = \exp(8.39 - 0.81 \ln(N_{act \setminus e}) - 0.25 \ln(E))$$

其中 N_act\e 为不含 embedding 的 active parameters。该公式揭示两个关键趋势：(1) 更大模型 → 更低 LR（系数 -0.81），(2) 更多 expert → 更低 LR（系数 -0.25）。此前文献对 MoE LR 方向存在分歧——Dai et al. (2024) 用更低 LR，Zoph et al. (2022) 用更高 LR——该公式通过实验证据解决了这一分歧：更低的 LR 有利于 MoE 训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LR Scaling Law 的拟合和使用流程：

```
# 1. 网格搜索：针对不同 (N_act\e, E) 组合训练小模型
configs = [(N_1, E_1), (N_2, E_2), ...]  # 覆盖参数空间
candidate_LRs = [1e-5, 3e-5, 1e-4, ...]

for (N, E) in configs:
    for lr in candidate_LRs:
        train_model(N, E, lr, D=small_dataset)
        record loss(N, E, lr)

# 2. 确定每个 config 的 optimal LR
LR_opt[N, E] = argmin_lr loss(N, E, lr)

# 3. 最小二乘拟合（log-log 空间）
# ln(LR) = c0 + c1·ln(N_act\e) + c2·ln(E)
# → c0=8.39, c1=-0.81, c2=-0.25

# 4. 使用：给定 N, E 直接计算 optimal LR
lr = exp(8.39 - 0.81*ln(N_act\e) - 0.25*ln(E))
```

验证：E={1,8} 上拟合，E=4（插值）和 E=32（外推）上验证。Ablation 移除 E 项后 E=32 外推明显 suboptimal。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 公式使用 ln(LR) 的线性预测而非直接预测 LR，避免 Kaplan et al. (2020) 公式在 N>10^10 时预测负 LR 的问题——exp(负值) 始终为正
- 论文验证了 E 项的必要性（Fig.8 ablation）
- 适用场景：任何需要在不同 MoE 配置间做公平比较的训练实验
- 限制：公式基于 ≤5B total params 的实验拟合，大模型外推需谨慎

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
