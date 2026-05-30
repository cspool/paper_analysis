## Model Merging / Weight Averaging (模型合并 / 权重平均)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Model Merging（模型合并，也称 Weight Averaging / 权重平均）是一种训练免（training-free）的多模型融合技术，通过直接对多个模型的权重进行算术平均来获得一个统一的通用模型。其理论基础是线性模式连通性（Linear Mode Connectivity）：从同一预训练初始化出发、分别在不同分布或任务上微调的模型，其权重往往位于同一平坦的损失盆地（loss basin）内，因此简单的线性插值不会导致损失剧烈上升。Li et al. (ECCV 2024) 证明了这一原理在 Multi-Target Domain Adaptation 中的有效性：对不同目标域分别做源自由域自适应（Source-Free Domain Adaptation）得到多个模型，通过 midpoint weight averaging 合并为一个统一模型，无需额外训练。合并操作分为两部分：(1) 模型参数合并 — 简单线性平均 θ_merged = (θ_1 + θ_2 + ... + θ_k) / k；(2) 归一化统计量合并 — 使用 Gaussian prior 从各模型的 BN 统计量估计合并后的 BN 均值和方差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 HDRQ 论文中的模型合并流程（语义分割，两个目标域）为例：
```
# 步骤 1: 源预训练模型 → 单目标域自适应
θ_src = pretrained_ResNet101_on_ImageNet()
θ_G→C = HRDA_adapt(θ_src, target=Cityscapes)   # GTA→Cityscapes
θ_G→I = HRDA_adapt(θ_src, target=IDD)          # GTA→Indian Driving

# 步骤 2: 各域独立 HDRQ 量化
BN_fold(θ_G→C); BN_fold(θ_G→I)                # BN 折叠
θ_C_q = HDRQ_quantize(θ_G→C, bit=4)            # Hessian + Distance Reg
θ_I_q = HDRQ_quantize(θ_G→I, bit=4)

# 步骤 3: Noise-Sampling Rounding + Merging
for k in range(30):                              # 30 次噪声采样
    ε_C, ε_I ~ U[-Δ/2, Δ/2]
    w_merged_k = midpoint_avg_noisy(θ_C_q, θ_I_q, ε_C, ε_I)
    score_k = cosine_similarity(w_merged_k - θ_src, θ_C_q - θ_I_q)
w_merged = argmax_k(score_k)                    # 选择最优样本

# 步骤 4: 多目标域推理
y_C = merged_model(x_C)                          # Cityscapes 上推理
y_I = merged_model(x_I)                          # IDD 上推理
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
模型合并的实现非常简单：(1) 使用 PyTorch 的 state_dict() 获取各模型参数字典；(2) 对各层参数执行逐元素的加权平均；(3) 对 BN 统计量特殊处理（Li et al. 用 Gaussian prior 建模，HDRQ 通过预先 BN folding 规避此问题）。合并的关键前提条件是模型从同一预训练 checkpoint 出发且在各自微调中保持相同架构。HDRQ 揭示了 quantized 模型合并的特有问题：量化引入的离散化噪声破坏权重对齐，增大 Error Barrier。其解决方案（Hessian Regularization + Distance Regularization）在量化阶段即保证合并兼容性，使合并后精度损失从 baseline QDrop 的约 4 mIoU 降至近零。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
