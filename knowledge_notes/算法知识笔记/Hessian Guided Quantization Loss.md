## Hessian Guided Quantization Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hessian Guided Quantization Loss 是 BRECQ 提出的量化质量评估指标，通过二阶 Taylor 展开估计量化引入的任务损失。E[L(Ŵ)] - E[L(W)] ≈ (1/2)·ε^T·H(W)·ε。Hessian 近似为 Fisher Information Matrix (FIM)，FIM 对角元近似为输出梯度平方：H_i ≈ (∂L/∂O_i)²。最终 L_Hessian = Σ_i (Ô_i - O_i)² · (∂L/∂O_i)²。该近似的成立条件：(1) 模型完美拟合真实分布（FIM ≈ Hessian）；(2) 任务损失为交叉熵（Bartlett 第二恒等式）。当模型拟合不佳或用于检测/分割任务时近似误差显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
L_ce = CrossEntropy(O_hat, labels)
grad = dL_ce/dO_fp                              # 输出梯度
H_brecq = grad^2                                # FIM 对角近似
L = sum((O_hat - O_fp)^2 * H_brecq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BRECQ Hessian loss 在 APHQ-ViT 论文中被系统分析其局限性：(1) 蒸馏损失（KL 散度）不满足 Bartlett 恒等式；(2) Table 4 显示 MSE 在某些 ViT 架构上优于 BH（ViT-B: MSE=73.79 vs BH=66.62）；(3) 无法泛化到检测/分割。APHQ-ViT 的 APH loss 通过有限差分直接计算 Hessian 对角元规避这些近似误差。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 提出 attention-aware Hessian 替代传统的 layer-wise Hessian H=2E[XX^T]。对 W_V 的 Hessian 定义为 H_V=2E[XA^TAX^T]（Equation 18），通过 attention map A 将 Q 和 K 投影的信息耦合进 Hessian。对 W_Q/W_K 的损失函数分别通过 E[K^TK] 和 E[Q^TQ] 注入跨投影信息，形式上仍使用 H=2E[XX^T] 作为基础 Hessian 但损失计算为 tr(E[K^TK]·ΔW·E[XX^T]·ΔW^T)。这种 attention-aware Hessian 的核心动机是：标准 H=2E[XX^T] 仅考虑输入 X 的统计特性，忽略了 attention 模块内 Q/K/V 之间的交互，而 H_V 通过 A=softmax(QK^T/√d) 天然编码了这种交互。

BiLLM 将 Hessian 矩阵用于两个关键环节：(1) Salient 权重识别：s_i = w_i²/[H^{-1}]_{ii}²，通过 Hessian 逆的对角元加权限定每个权重元素对层输出的二阶影响，比仅用 |w| 判断显著性更准确（捕获了权重值小但对输出敏感的权重）；(2) Block-wise OBC 误差补偿：H 的 Cholesky 分解 H_chol = Cholesky((H+λI)^{-1}) 用于计算补偿公式，其中 Hessian 由校准数据 X 计算：H = 2XX^T（L2 误差 Hessian）。λ 为正则化项防止 Hessian 矩阵奇异。

FOEM 将 Taylor 展开 δE = g δw^T + ½ δw H δw^T 的一阶项 g 从假设为零改为显式保留：通过 g ≈ β(W−𝕎)H（β=0.1）近似梯度，代入 Lagrangian 约束优化后 H 和 H^{-1} 在代数中自动消去。这揭示了二阶-only 建模（如 BRECQ Hessian loss 和 GPTQ）在逐列累积补偿场景下的系统性误差。

GuidedQuant (ICML 2025) 进一步将该思路推广到现代 LLM（70B）：(1) 使用 block-diagonal Fisher 近似替代 BRECQ 的对角近似，保留每个 output channel 内 d_in×d_in 的 Fisher block；(2) 引入 averaging approximation，将 d_out 个 Fisher block 按 g 组（g=2~4）平均，使存储从 Θ(d_in² d_out) 降至 Θ(d_in² g)；(3) 将 guided Hessian H̄_k 直接集成到现有 layer-wise output-based PTQ 方法（LNQ、QTIP、SpinQuant）中作为直接 plugin。
