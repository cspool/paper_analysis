## Calibration Module in MoE Extension (MoE 扩展中的校准模块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Calibration Module 是 MoExtend 在新增 expert 后为保持原有 MoE 输出分布一致性而引入的轻量校正网络。当 MoE 层从 m 个 expert 扩展为 m+1 个 expert 后，softmax 概率分布会因分母增大而整体缩小：s(x)_j' = e^{f(x)_j} / (Σ_{h=1}^m e^{f(x)_h} + e^{f(x)_{m+1}}) ≤ s(x)_j。这意味着原有 expert 的输出权重被"稀释"，即使原有 expert 参数未变，前向传播的特征分布也会漂移，导致一定程度的已有知识遗忘。Calibration Module 通过为每个 expert 添加一个可学习的校正因子 s_c(x) 来修正此效应。

MoExtend 中采用的 Type2(a) 结构：两个线性层 + GELU 激活函数组成的轻量网络，输出作为每个 expert 的校正因子（加法模式）。初始化策略：第一层正态初始化，第二层零初始化——确保训练初期 s_c(x)=0，模型输出与未加 calibration 时一致。

从算法pipeline角度拆解术语：

**带 Calibration 的 MoE 前向计算：**
```
# 输入 x，原有 m experts + 1 new expert
logits = x @ W_new          # [B, m+1]
probs = softmax(logits)       # [B, m+1]

# 获取 top-K expert 的权重和索引
weights, indices = top_k(probs, K)

# 对每个选中的 expert 计算输出，并施加 calibration
output = 0
for j in range(K):
    expert_idx = indices[j]
    w = weights[j]
    expert_out = FFN[expert_idx](x)
    
    # Calibration: 每个 expert 有独立的 calibration 模块
    calib = sc(x)  # 两层 GELU 网络，输出标量
    
    # 加法校正模式（Type2 a）
    output += w * (1 + calib) * expert_out

# output 即为 MoE 层的最终输出
```

注释：
- s_c(x) = W_1(GELU(W_2(x)))，其中 W_1 零初始化，W_2 正态初始化
- 初始状态 s_c(x)=0，MoE(x) = Σ s(x)_j · FFN(x)_j（与原始一致）
- 训练后 s_c(x) 学习出对概率缩放的补偿

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **结构设计空间**（MoExtend 实验结论）：
  - Type1（简单可学习参数 1×m）：加法模式用 Zero 初始化、乘法模式用 One 初始化
  - Type2（两层 GELU 网络）：加法模式用 Zero+Normal 初始化（最优）、乘法模式导致梯度爆炸
  - **Type2(a) 加法模式为最优**：POPE 84.3, MME 1571.0, SQA 73.4, VQA^T 55.7
- **关键设计原则**：初始化必须使 s_c(·) 初始输出为零（加法模式）或一（乘法模式），确保训练初期不干扰模型前向输出，避免异常 loss
- **可扩展性**：Calibration 的概念不仅限于 MoE 扩展——任何涉及模型结构修改后需要"输出分布对齐"的场景都可用类似设计（如 model merging、架构搜索后的 fine-tuning）

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension
