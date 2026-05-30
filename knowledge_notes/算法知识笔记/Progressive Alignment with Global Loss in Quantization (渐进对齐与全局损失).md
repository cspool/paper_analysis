## Progressive Alignment with Global Loss in Quantization (渐进对齐与全局损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
渐进对齐与全局损失是 QuEST 的三阶段优化策略的统称：(1) 阶段一 TLA（Temporal Layer Alignment）：独立微调时间嵌入层，利用时序信息独立于图像输入且在模型早期确定的特性；(2) 阶段二 CMA（Critical Module Alignment）：在 TLA 冻结后微调注意力相关层；(3) 全局损失 L_G = E_t[||O(x_t; w) - Õ(x_t; w, s)||²]，在 CMA 阶段叠加，为所有未选中的层提供网络级梯度信号。渐进式设计的原因：时间嵌入和注意力层功能不重叠，且时间信息先于空间/语义信息处理，因此应先对齐时间嵌入再对齐注意力。全局损失的重要性：仅用局部损失（TLA+CMA w/o L_G）FID 为 8.99（TLA）和 6.41（CMA）；添加 L_G 后分别改善至 6.41 和...（CMA+L_G 效果未单独给出但联合使用最显著）。有趣的是，**仅使用全局损失**会导致性能退化 7.13 FID，说明局部+全局的组合是必需的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三阶段联合优化的最终目标函数：
```
argmin_{w_l} (L_TLA + L_CMA + 2*L_G),  l ∈ C_TE ∪ C_A

其中：
L_TLA = Σ_{l∈C_TE} E_t[||FP_time_embed(t) - Q_time_embed(t)||²]
         → 微调：w_l (l∈C_TE), s_l (l∈C_TE)

L_CMA = Σ_{l∈C_A} E_t[||FP_attn(z_l) - Q_attn(z̃_l)||²]
         → 微调：w_l (l∈C_A), ŝ = s \ s_l (l∈C_TE)

L_G   = E_t[||FP_final(x_t; w) - Q_final(x_t; w̃, s)||²]
         → 微调：w_l (l∈C_TE∪C_A), s (全部激活量化参数)

# 渐进执行
# Step 1: TLA 独立训练（w_TE, s_TE）
# Step 2: CMA + 2*L_G 联合训练（w_A, ŝ）
# 注意：Step 2 中权重 w_A 梯度来自 (L_CMA + 2*L_G)，
#       ŝ 梯度来自 (L_CMA + 2*L_G)，
#       这使未选中层的量化参数 s 通过 L_G 获得间接优化
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现细节：(1) 权重 L_G 系数为 2（相对 L_TLA 和 L_CMA 各 1），实验通过网格搜索确定；(2) 网络级训练（network-wise training）——所有激活量化参数一次性优化，而非逐层/逐块重建（PTQ 方式），显著节省时间；(3) Adam 优化器，权重学习率 1e-5，量化参数学习率 1e-4（量化参数需要更大学习率因为对其初始估计更粗糙）；(4) 2300 次迭代完成（vs EfficientDM 的 32000 次）；(5) 该策略可扩展到 Stable Diffusion 等更大模型——Full-finetune OOM，但 QuEST 可在 48GB GPU 上完成；(6) 仅 L_G 不足以获得好性能（原因：全局损失信号对深层参数太弱，需要局部对齐的强梯度信号）。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
