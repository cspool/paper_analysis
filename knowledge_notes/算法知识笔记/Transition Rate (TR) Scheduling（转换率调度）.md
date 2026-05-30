## Transition Rate (TR) Scheduling（转换率调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transition Rate（TR，转换率）调度是 Lee et al. 提出的专用于 QAT 的训练调度技术。核心动机：传统 QAT 使用学习率（LR）调度来间接控制量化权重的更新幅度，但量化权重仅在潜权重（latent weights）越过 quantizer 的 transition point 时才改变离散级别，其 effective step size 与 LR 相关性弱，导致后期训练不稳定。TR 定义为每次迭代中发生离散级别变化的量化权重占总权重的比例：`k^t = Σᵢ I[w_d^t(i) ≠ w_d^{t-1}(i)] / N`，其中 `w_d` 为离散权重（round/sign 函数输出）。TR 调度通过设定目标 TR `R^t`（通常用 cosine scheduler 从初始值 `λ√b_w` 衰减到零），然后用 TALR 自适应调整潜权重更新步长，使得实际 running TR `K^t`（EMA 平滑后）跟踪目标 TR。这实现了对量化权重的"粗到细"控制——初期高 TR 允许充分探索，后期 TR 趋近零保证收敛稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TR 调度在 QAT 中的每步迭代（来自论文 Algorithm 1）：
```
输入: 总迭代数 T, 目标 TR R^t, momentum m=0.99
输出: 训练后的量化模型

for t in 1..T:
    # Step 1: 前向传播 — 潜权重 w 经 quantizer 转为量化权重 w_q
    w_n = clip(γ·w/s, α, β)          # 归一化 (Eq.1)
    w_d = round(w_n)                   # 离散化 (Eq.2)
    w_q = w_d/γ                        # 定标反归一化 (Eq.3)
    # 用 w_q 计算前向输出和 loss
    
    # Step 2: 反向传播 — STE 通过 discretizer 回传梯度
    计算 gradient term g^t（取决于优化器类型）
    
    # Step 3: 计算当前 TR (Eq.5)
    k^t = count(w_d^t != w_d^{t-1}) / N  # 发生变化的量化权重占比
    
    # Step 4: 估计 running TR (Eq.10)
    K^t = m·K^{t-1} + (1-m)·k^t        # EMA 平滑
    
    # Step 5: 调整 TALR (Eq.11)
    U^t = max(0, U^{t-1} + η(R^t - K^t))  # 加法反馈更新，η=U^0
    
    # Step 6: 更新潜权重 (Eq.12)
    w^{t+1} = w^t - U^t·g^t            # 用 TALR 代替 LR
```
关键设计：初始 target TR = λ√b_w（λ=TR factor, b_w=量化位宽），位宽越大 transition points 越多，需更高初始 TR。TR momentum m=0.99 平滑单步噪声。TALR 更新因子 η = U^0（初始 TALR），使调整步长与初始值成比例。TR 调度仅用于量化潜权重；未量化参数（第一层/最后一层）仍用传统 LR。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) TR 使用离散权重 `w_d`（integer）而非量化权重 `w_q` 来计数 transition——因为 `w_q = δ^t·w_d` 中 scaling factor δ^t 可能变化，使用 `w_q` 会误检 transition；(2) 当 TR 调度启用时，weight quantizer 的 scale parameter s 固定不变——否则 s 变化也会触发 transition，干扰 TR 控制；(3) 支持的 optimizer 类型：SGD、Adam、AdamW、NAdam、Adamax、RMSProp、Adagrad（Table S5 验证），仅需将 g^t 的计算替换为对应优化器规则；(4) 训练时间仅增加约 2%（Table S7）；(5) 在极低位宽（2-bit/binary）和轻量模型（MobileNetV2）上增益更显著（+6.7% 精度）；(6) TR factor λ 在 {5e-3, 1e-3, 5e-4, 1e-4} 中搜索，λ∈[4e-3, 6e-3] 对 ResNet-20 W2A2 均优于 LR baseline。开源：https://cvlab.yonsei.ac.kr/projects/TRS/

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---
