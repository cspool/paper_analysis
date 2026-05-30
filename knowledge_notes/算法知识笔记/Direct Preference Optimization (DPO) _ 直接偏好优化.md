## Direct Preference Optimization (DPO) / 直接偏好优化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Direct Preference Optimization (DPO) 是 Rafailov et al. (2023, NeurIPS) 提出的一种用于对齐 LLM 输出与人类偏好的后训练方法。核心思想：将 RLHF 中两步流程（训练 reward model → RL 优化 policy）合并为单步直接优化。DPO 基于一个关键数学洞察——在 Bradley-Terry 偏好模型下，最优 policy π* 与 reward 函数 r 之间存在双射映射：r(x,y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x)。利用此关系，可以直接在偏好数据上优化 policy，无需显式训练 reward model 或执行 RL。DPO 损失函数：L_DPO(π_θ; π_ref) = -E_{(x,y_w,y_l)~D}[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]，其中 y_w 为 preferred response，y_l 为 dis-preferred response，π_ref 为冻结的参考模型（通常是 SFT 后的模型），β 为 KL 散度惩罚系数。σ 是 sigmoid 函数。损失函数直观含义：增大 preferred 与 dis-preferred 之间的相对对数概率差，同时 β 约束 policy 不偏离 π_ref 太远。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DPO 训练 pipeline（基于 TPO 论文中的使用方式）：
```
输入: 偏好数据集 D = {(x, y_w, y_l)}, 参考模型 π_ref (SFT checkpoint)
输出: 对齐后的 policy π_θ

For each batch (x, y_w, y_l) in D:
    # 1. 前向传播计算 log-probabilities
    log_p_w = log π_θ(y_w | x)      # preferred 的对数概率
    log_p_l = log π_θ(y_l | x)      # dis-preferred 的对数概率
    log_p_w_ref = log π_ref(y_w | x)
    log_p_l_ref = log π_ref(y_l | x)

    # 2. 计算 log-ratio (implicit reward)
    ratio_w = log_p_w - log_p_w_ref
    ratio_l = log_p_l - log_p_l_ref

    # 3. DPO 损失
    L_DPO = -log σ(β * (ratio_w - ratio_l))

    # 4. 可选: SFT 辅助损失 (TPO 中使用)
    L_SFT = -log_p_w

    # 5. 联合损失
    L = L_DPO + α * L_SFT

    # 6. 反向传播
    θ ← θ - η * ∇_θ L
```
TPO 论文中 LongVA-TPO 使用 β=0.3, α=0.5, lr=4×10⁻⁶；LLaVA-Video-TPO 使用 β=0.2, α=1, lr=3×10⁻⁷。β 越大，policy 偏离 π_ref 的惩罚越重，训练越保守。训练 1 epoch，约 4 小时（8×A100, batch_size=64）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPO 的标准实现在 HuggingFace TRL 库中：`DPOTrainer` 类。使用流程：(1) 准备偏好数据集，每行包含 prompt/chosen/rejected 三个字段；(2) 加载 SFT checkpoint 作为 reference model（冻结，不参与梯度更新）；(3) 配置 β 超参数和训练参数；(4) 调用 DPOTrainer.train()。开源实现：TRL (https://github.com/huggingface/trl)。DPO 相比 RLHF+PPO 的优势：无需训练单独 reward model、无需 RL 算法（更稳定）、单步训练（更快）、内存开销更低（只需两份模型：policy + reference，而非 policy + reward + value + reference）。局限性：(1) 偏好数据质量要求高——噪声偏好对会直接误导优化方向；(2) offline 性质——无法像 online RLHF 一样从模型自身采样中学习；(3) 对 β 等超参数敏感。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models
