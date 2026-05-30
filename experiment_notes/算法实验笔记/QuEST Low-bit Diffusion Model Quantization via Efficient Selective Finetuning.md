## QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuEST 是一种面向扩散模型低比特量化的参数高效选择性微调方法。核心 pipeline：(1) 发现扩散模型中激活值分布不均衡（数值大量集中在零附近，但有稀疏的大值对生成质量重要），导致低比特量化失败；(2) 提出通过权重微调来调整激活分布，使其更向量化友好——理论证明（Theorem 3.2）通过将大量化扰动Δ分解为K个小扰动ε=Δ/K，说明微调权重可使模型对量化扰动更鲁棒；(3) 识别两类关键层：时间嵌入层（Property ❶：时序信息对量化至关重要）和注意力相关层（Property ❷：FeedForward等层对位宽降低特别敏感，6比特即失败）；(4) 采用数据无关方式构建校准集（128-256样本/时间步，从随机高斯噪声xT采样）；(5) 选择性渐进微调：先 TLA（Temporal Layer Alignment）微调时间嵌入层权值w^l和激活量化参数s^l，再 CMA（Critical Module Alignment）微调注意力相关层权值和对所有未更新的量化参数s进行优化，最后用全局损失 L_G（量化模型与全精度模型最终输出MSE）监督；(6) 总损失：argmin(L_TLA + L_CMA + 2L_G)，仅微调不足7%的参数；(7) 使用单组量化参数覆盖所有时间步，无需按时间步分别量化。前向伪代码：t → TimeEmbed(t, w_TE_finetuned) → 各层注入 → Attention(Q,K,V with w_finetuned) → FFN(sensitive, activations cautiously quantized) → 计算 L_TLA (TE层输出MSE) + L_CMA (注意力层输出MSE) + L_G (最终输出MSE) → Adam更新部分w和s。量化函数：x̂ = clamp(round(x/s) + Z; qmin, qmax), x̃ = (x̂ - Z)*s。
  - 实验比较：(a) W8A8/W4A8/W4A4 下 QuEST vs PTQ4DM、Q-Diffusion、PTQ-D、EfficientDM 在 LSUN-Bedrooms（LDM-4）和 LSUN-Churches（LDM-8）上的 FID/sFID；(b) ImageNet 256×256 上 FID/sFID/IS 对比；(c) Stable Diffusion v1.4 文本到图像生成的 CLIP Score 对比；(d) 与 TFMQ-DM 对比；(e) CIFAR10 低分辨率对比；(f) 消融：TLA vs TLA+CMA vs TLA+CMA+LG 组件贡献；(g) 全局损失消融 w/ vs w/o L_G；(h) 效率对比 QuEST vs EfficientDM vs Full-finetune (时间/显存/迭代数/FID)；(i) LoRA 集成消融；(j) 预计算时间嵌入对比。

- 硬件平台是什么，配置是什么。
  - NVIDIA A6000 GPU（48GB）；Stable Diffusion 实验在单卡完成。环境：Python，PyTorch，CUDA，Linux。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LDM-4（LSUN-Bedrooms 256×256）、LDM-8（LSUN-Churches 256×256）、LDM-4（ImageNet 256×256）、Stable Diffusion v1.4（512×512）
  - 数据集：LSUN-Bedrooms、LSUN-Churches、ImageNet（条件生成）、COCO2014 prompts（文本到图像，10000 条验证集）
  - Benchmarks/Metrics：FID（Fréchet Inception Distance）、sFID（spatial FID）、IS（Inception Score）、CLIP Score（ViT-B/16 backbone）
  - Samplers：DDIM（LDMs，20/200/500 步），PLMS（Stable Diffusion，50 步）
  - 评估：50000 张采样图像，官方评估脚本
  - 超参数：Adam optimizer, lr_w=1e-5（权重微调）, lr_s=1e-4（量化参数微调），校准集 256 样本/时间步（Stable Diffusion 128 样本/时间步）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/hatchetProject/QuEST
  - 张量计算流程：(1) 输入 x_T ∼ N(0,I) 采样高斯噪声，通过全精度模型前向获取中间激活作为校准目标；(2) 量化：W̃ = clamp(round(W/s_w) + Z_w; qmin, qmax)，x̃_l = clamp(round(x_l/s_l^a) + Z_l; qmin, qmax)；(3) TLA 微调：for l in C_TE: O_TE(l) = FP_model(t; w_l), Õ_TE(l) = Q_model(t; w̃_l, s_l) → loss = MSE(O_TE, Õ_TE)，反向传播更新 w_l 和 s_l；(4) CMA 微调：for l in C_A: O_attn(l) = FP_model(z_l; w_l), Õ_attn(l) = Q_model(z̃_l; w̃_l, ŝ) → loss = MSE(O_attn, Õ_attn)，更新 w_l 和 未在 TLA 中更新的 s；(5) 全局损失：L_G = MSE(FP_model(x_t; w), Q_model(x_t; w̃, s))；(6) 最终优化：argmin_{w_l} (L_TLA + L_CMA + 2L_G), l ∈ C_TE ∪ C_A。
