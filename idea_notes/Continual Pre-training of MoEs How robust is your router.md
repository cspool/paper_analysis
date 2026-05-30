## Continual Pre-training of MoEs How robust is your router

- baseline方法是什么？
  Baseline 包含两层：(1) **密集 Transformer CPT**：使用 replay + LR re-warming + re-decaying 对 dense decoder-only transformer 进行持续预训练（Ibrahim et al., 2024），从衰减 checkpoint 恢复后用 Cosine Annealing 重新 warming + decaying；(2) **Full Re-training**：将 FineWeb 和下游数据（Stack/German）混合后从头完整训练。Baseline 在 MoE 场景下的核心缺陷：没有针对 MoE 路由算法的 CPT 行为分析——不清楚路由算法在分布偏移下是否会加剧遗忘、是否能维持负载均衡、现有的 dense CPT 策略（replay + LR re-warming/decaying）对 MoE 是否同样有效。

  **Baseline 全栈执行例子（以 570M dense transformer + 64×A100, FineWeb→German CPT 为例）**：
  - **算法层**：输入 batch 含 1024 个 sequence（seq_len=2048），通过 24 层 Llama3-style decoder-only transformer，每层经过 Multi-Head Self-Attention (16 heads) + GEGLU FFN（intermediate=2816→output=1024）。CPT 时从衰减 checkpoint (η=3e-5) 开始，用 Cosine Annealing re-warm 到 η_max=3e-4 再 decay 到 η_min=3e-5。40% replay：每 batch 中 410 samples 来自 FineWeb，614 samples 来自 German CC。
  - **系统框架层**：基于 GPT-NeoX 训练框架，64×A100 GPU，data parallel + ZeRO-1。ZeRO-1 将 optimizer states (AdamW m, v) 分片到 64 张 GPU，每张 GPU 持有 1/64 的 optimizer states。前向：每张 GPU 独立计算 full batch slice 的前向 loss。反向：每张 GPU 计算梯度后 AllReduce 聚合，然后各 GPU 用本地 optimizer state 分片更新参数，再 AllGather 参数。
  - **Kernel层**：标准 PyTorch 操作，包括 cuBLAS GEMM (FFN MatMul)、Flash Attention (self-attention)、LayerNorm、GeLU 激活。论文未明确说明 kernel 细节。
  - **硬件层**：64×NVIDIA A100 GPU（80GB 或 40GB SXM），NVLink + NVSwitch 互联，每步耗时约 880ms（dense），MFU 约 111 TFLOPs。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文首次系统研究 MoE CPT 的完整行为，通过三项关键贡献解决 baseline 在 MoE 场景下的未知问题：

  **(1) 构建四种 MoE 架构的 CPT 实验矩阵 → 填补 MoE CPT 知识空白**
  - 系统对比了当前 SOTA 的两种路由算法（PBTk: z-loss + aux loss; SBTk: Sinkhorn-Knopp 迭代近似线性分配问题）和两种架构（Granular: 31 routed + 1 shared, K=3; Switch: 8 routed, K=1, full FFN）
  - 训练规模：570M active / 2B total，各训练 600B tokens（400B FineWeb + 200B Stack/German），严格 overtraining regime (>10× Chinchilla optimal)
  - 所有 MoE 与 FLOP-matched dense baseline 对比，确保公平

  **(2) 证明 MoE 路由算法对分布偏移具有"令人惊讶的鲁棒性" → 消除对 CPT 破坏路由负载均衡的担忧**
  - SBTk MoEs 在分布偏移时 MRI 几乎不变（因为显式的 Sinkhorn balancing 步骤强制均衡）
  - PBTk MoEs 经历短暂的 MRI spike（分布偏移后），但在 500 steps 内恢复至比 SBTk 更低的 MRI 水平
  - 两种路由算法最终的负载均衡均优于或等于 full re-training baseline
  - 提出 Maximum Routing Imbalance (MRI) 指标作为 MoE 推理延迟的代理：MRI(t,j) = max_i [∑_x 𝟙{i∈I_k(x)} / |B|]，独立于具体硬件部署

  **(3) 建立 MoE CPT 最佳实践 → CPT MoE 可匹配 full re-training 性能且大幅降低成本**
  - Infinite LR schedule (CosineInf) + replay 是 MoE CPT 的最佳组合：在 FineWeb→Stack (30% replay) 和 FineWeb→German (40% replay) 两个任务上，CPT MoE 的验证 loss、English/German/Code benchmark 均匹配或超越 full re-training baseline
  - CPT 仅消耗 full re-training 约 1/3 的计算成本
  - MoE 在 CPT 期间保持对 FLOP-matched dense 的 sample efficiency 优势
  - 路由行为分析发现：CPT 期间路由决策变化主要发生在早期层 (layers 0-2) 和后期层 (layers 13-23)；0% replay 的 checkpoint 在早期层变化最大且遗忘最多，说明早期层的剧烈路由变化与遗忘相关

  **论文方法全栈执行例子（以 Granular PBTk MoE, FineWeb→German, 40% replay + CosineInf 为例）**：
  - **算法层**：输入的 1024 个 sequence（2048 tokens/seq = 2,097,152 tokens/batch）进入 24 层 MoE transformer。每层 MoE block：Self-Attention → Router (W_r: 1024→31, linear projection + softmax) → Top-3 expert selection → shared expert (GEGLU, intermediate=704) + 3 selected experts (GEGLU, intermediate=704) 的加权组合。Router 输出通过 z-loss (coeff=0.001) 和 aux loss (coeff=0.01) 惩罚大 logit 和负载不均衡。CPT 从 CosineInf schedule 的 η_const=1.65e-4 平滑过渡继续训练 95,370 steps。Replay: 每 batch 410 FineWeb + 614 German CC tokens。
  - **系统框架层**：GPT-NeoX + Megablocks grouped GEMM kernel。Megablocks 将同一 batch 中不同 token 被路由到不同 expert 的 FFN 计算打包为单次 grouped GEMM：将 batch 中所有 tokens 按 target expert 分组，同一 expert 的 tokens 拼接为连续矩阵块，一次性完成 batched MatMul，避免逐个 expert 的小矩阵乘法开销。
  - **Kernel层**：论文未明确说明 Megablocks 的详细 kernel 参数，但 grouped_gemm 仓库提供 CUDA kernel 实现。Megablocks 的基本原理：将稀疏 MoE 的多个 expert FFN 的 GEMM 操作融合为一次 grouped GEMM，减少 kernel launch overhead 和内存碎片。
  - **硬件层**：64×A100，data parallel + ZeRO-1。Granular MoE 每步约 1680ms（dense 的 ~2×），forward ~485ms，backward ~1091ms，MFU 约 78 TFLOPs。Sinkhorn 版本因迭代求解额外增加约 110ms/步。
