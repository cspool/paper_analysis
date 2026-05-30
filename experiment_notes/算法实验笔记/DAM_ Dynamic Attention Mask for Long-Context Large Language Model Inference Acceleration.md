## DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  DAM 提出一种免微调的动态稀疏注意力机制，通过两阶段流程为每个 attention map 生成自适应稀疏 mask，保留跨层、跨 head 的异构 attention 模式。Stage 1：冻结的预训练模型在 Pattern Capture Length (PCL) 范围内处理输入序列，提取完整 attention map → 对累积的注意力分数计算均值 → Box-Cox 变换（λ=0.5）放大中小注意力值 → 归一化（减去全局最小值） → 以阈值 τ=0.3 进行二值化生成 "true mask" → 通过结构模式匹配（匹配 score 阈值 μ=0.8）识别对角线模式 P_diag,r 和垂直模式 P_vert,c → 对超过 PCL 的长序列，将匹配到的结构模式外推生成 "extended mask"。Stage 2：将生成的动态稀疏 mask 在 softmax 之前应用于 attention score，mask 位置设为 -∞ 使注意力概率为 0，将 FLOPs 复杂度从 O(L²) 降至 O(sL)（s 为每个 query 保留的平均 key 数，s ≪ L）。实验比较 LongEval（长度 3K-104K tokens）和 LV-Eval（16K-256K tokens，含单跳 QA 如 cmrc-mixup 和多跳 QA 如 dureader-mixup）上的检索准确率/评分，以及在 LLaMA 3.2 1B/3B 和 Vicuna 7B 上与 Full Attention（Original）、FlashAttention、MoA、StreamingLLM、H2O 的 GPU 内存、吞吐量、平均时延对比。

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100 40GB（LongEval 评测）；2× NVIDIA H100 80GB（LV-Eval 评测）；1× NVIDIA A100 40GB（效率评测，即表 1 的 Memory/Throughput/Latency）。原始 LLaMA 3.2 3B 在单卡 A100 40GB 上处理超 4K 序列即 OOM。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.2-1B-Instruct、LLaMA-3.2-3B-Instruct（用于可扩展性分析）；Vicuna-7B（效率评测）。Attention map 捕获数据集：Multi-News（大规模多文档摘要数据集）。Benchmark：LongEval（100 data items per length level，以行级 key-value 检索精度衡量）、LV-Eval（含 single-hop: cmrc-mixup, multifieldqa-en-mixup, multifieldqa-zh-mixup；multi-hop: dureader-mixup, loogle-CR-mixup, loogle-MR-mixup, hotpotwikiqa-mixup, lic-mixup）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/HanzhiZhang-Ulrica/DAM。算法 pipeline 如下：

  **Stage 1 — Mask 生成（离线，基于冻结模型）**：
  1. 在 Multi-News 数据集上运行冻结的 LLaMA 模型，对长度不超过 PCL（L=512）的序列提取各层各 head 的完整 attention map A_{ℓ,h,i,j}。
  2. 计算跨 batch 的累积注意力与 token 位置计数，求均值：\bar{A}_{ℓ,h,i,j} = A_{ℓ,h,i,j} / (C_{ℓ,h,i,j} + ε)。
  3. Box-Cox 变换（λ=0.5）：B_{ℓ,h,i,j} = (X^{0.5} - 1) / 0.5，其中 X = max(\bar{A}, ε)。
  4. 归一化：B^*_{ℓ,h,i,j} = B_{ℓ,h,i,j} - min_{all}(B)，得到 \tilde{A}_{ℓ,h,i,j}。
  5. True mask 生成（阈值 τ=0.3）：m_{i,j} = 1 if \tilde{A}_{ℓ,h,i,j} ≥ τ else 0。
  6. 结构模式匹配（阈值 μ=0.8）：对每个 true mask M_{ℓ,h}，与模式池 P = {P_diag,r} ∪ {P_vert,c} 中的每个模式 P_k 计算匹配分数 γ_k = Σ_{i,j} M·P_k / Σ_{i,j} P_k。若 γ_k ≥ μ，该模式被匹配。
  7. 扩展 mask：\tilde{M}_{ℓ,h} = Σ_{P_k: γ_k≥μ} P_k，二值化：\tilde{M}^{(i,j)} = 1 if 该位置任一匹配模式为 1 else 0。

  **Stage 2 — 推理时 mask 应用**：
  若 S ≤ L：直接使用 true mask M_{ℓ,h}。
  若 S > L：将 PCL 范围内的 M_{ℓ,h} 与外推的扩展 mask 组合为 S×S 的完整 mask \tilde{M}_{ℓ,h}。
  在 softmax 前应用：A'_{ℓ,h} = (QK^T / √d_k) ⊙ \tilde{M}_{ℓ,h}，mask 位置设为 -∞，最终 O' = softmax(A')V。

  DAM 不修改模型权重，不引入额外训练，与基于 tile 的 GPU 执行兼容（未来可与 FlashAttention 等 memory-efficient kernel 融合）。
