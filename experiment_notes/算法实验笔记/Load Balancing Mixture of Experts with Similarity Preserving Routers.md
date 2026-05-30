## Load Balancing Mixture of Experts with Similarity Preserving Routers

- 属于算法pipeline的实现是什么？实验比较什么？
  - SIMBAL（SIMilarity-preserving routers for MoE load BALancing）提出一种新的 MoE 负载均衡辅助损失 L_orth = ||R^T R - I_E||_1，通过鼓励 router 权重矩阵 R ∈ R^{D_M × E} 逼近正交矩阵来保持 token 间成对相似性。结合 Saxe et al. 2014 的正交初始化，SIMBAL 使得相似 token 获得相似的 expert 分布，减少 expert 间知识冗余。核心设计选择：(1) 使用 loss-based 方法替代显式正交参数化（QR分解），避免大模型训练中的计算开销和数值不稳定；(2) 损失函数数据集无关且计算便宜，对 batch size 不敏感；(3) 提出 Pairwise Expert Similarity (PES) 指标量化 expert 冗余度。
  - 实验比较：
    - SIMBAL vs LBL (Load Balancing Loss, Fedus et al. 2022) 为主比较
    - 两种模型规模：MoE-M (230M active/627M total) 和 MoE-L (761M active/3.14B total)
    - 无负载均衡 baseline (no loss) 验证 collapse 避免
    - Loss-Free (LF) balancing [Wang et al. 2024] 组合实验（附录A.1）
    - 评估指标：Validation Perplexity、收敛速度、PES、SEU、Router Entropy、Router Gram L2 距离
    - 下游 benchmark：ARC Challenge/Easy、HellaSwag、PIAQ、WinoGrande、GLUE
    - 推理时 expert pruning 协同实验 [Szatkowski et al. 2024]
  - 结果：SIMBAL 比 LBL 快 36% 收敛；MoE-M perplexity 13.685 vs 14.086，MoE-L 8.304 vs 8.517；MoE-L avg benchmark 45.19% vs 43.28%；PES 显著更低 (0.0044 vs 0.0255)；推理 pruning 下 7.4% speedup

- 硬件平台是什么，配置是什么。
  - MoE-M 训练：8× NVIDIA A100 40GB GPUs，Distributed Data Parallelism (DDP)
  - MoE-L 训练：8× AMD MI300X 192GB accelerators，DDP
  - 软件环境：PyTorch (bfloat16 训练精度)，基于 OLMo 开源代码库
  - 推理 pruning 实验：论文未明确说明具体 GPU 型号

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：Transformer backbone + RMSNorm + SwiGLU activations + RoPE + Z-loss (1e-5)
    - MoE-M：DM=768, Depth=8, Heads=8, DF(expert)=768, 32 experts, top-4, 230M active/627M total, RoPE θ=1e4, Peak LR=5e-4
    - MoE-L：DM=1536, Depth=12, Heads=12, DF(expert)=1536, 32 experts, top-4, 761M active/3.14B total, RoPE θ=1e5, Peak LR=3e-4
    - Dense baselines：Dense-M (230M) 和 Dense-L (761M)
    - 所有 FFN 层替换为 MoE 层
  - 训练配置：AdamW optimizer, weight decay=0.01, linear warmup (2000 steps) + cosine decay, bfloat16
  - 数据集：DCLM-pool-400m-1x [Li et al. 2025]，cl100k_base tokenizer (tiktoken)，77M tokens 验证集
  - 训练量：MoE-M 19.9B tokens, MoE-L 78.6B tokens
  - LBL baseline：loss coefficient 0.01

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 基于 OLMo 开源代码库 (https://github.com/allenai/OLMo)。LBL 参考 lucidrains/st-moe-pytorch。
  - SIMBAL loss 伪代码（来自论文 Appendix A.3 Figure 6）：
```python
def simbal_loss(router_linear, p=1):
    w = router_linear.weight           # [E, D_M]
    w_ortho = torch.matmul(w, w.T)     # Gram matrix R^T R
    eye = torch.eye(w.shape[0], device=w.device)
    loss = torch.norm(w_ortho - eye, p=p)  # ||R^T R - I||_1
    return loss
```
  - LBL baseline 伪代码：
```python
def balance_loss(gates):
    # gates: [batch_size, num_tokens, num_experts]
    expert_mask = gates > 0.0
    f_i = reduce(expert_mask.float(), "b t e -> b e", "mean")
    P_i = reduce(gates, "b t e -> b e", "mean")
    loss_per_batch = num_experts * torch.sum(f_i * P_i, dim=-1)
    return loss_per_batch.mean()
```
  - SIMBAL 单 token 张量流：x ∈ R^D_M → Router R (near-orthogonal) → scores ∈ R^E → softmax → top-4 experts → SwiGLU FFN per expert → weighted sum output
  - PES 指标：C_expert(x) = (2/(N(N-1))) Σ_i Σ_{j>i} cos(f_i(x), f_j(x))，PES 越低表示 expert 多样性越高、冗余越低
