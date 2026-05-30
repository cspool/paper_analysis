## Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **DefaultMoE**：一种轻量级 MoE 训练方法，在保持前向稀疏激活（仅 Top-K experts 计算前向）的同时，通过为未被激活的 expert 提供 **default vector**（EMA 历史输出的指数移动平均），使得 Router 在反向传播时收到来自所有 N 个 experts 的 **dense gradient**，而非仅有被选中的 K 个 experts 的梯度。具体机制包括：(1) 为每个 expert i 维护 EMA buffer Ê_i = β·Ê_i^{(t-1)} + (1-β)·E_i(x)，记录 expert 历史输出的期望值；(2) 前向传播时，对非激活 expert 用 Ê_i 替代真实 E_i(x)，计算 y = Σ π_i · (E_i(x) if i∈TopK else Ê_i)；(3) 反向传播时，路由器收到所有 N 个 experts 的梯度信号 ∂y/∂π = [Ê_1, ..., E_i(x) for i∈TopK, ..., Ê_N]^T。实验比较：DefaultMoE vs Standard TopK MoE、SparseMixer、ReMoE、Loss-Free Balancing，在 1.96B 总参数 MoE（8c1/8c2/32c1/32c2/32c4 多种配置）上训练 160B tokens，对比 pretraining PPL、收敛速度（token-to-target-PPL）、下游 12 个 benchmark（LogiQA, MathQA, MMLU, OpenBookQA, Lambada, SocialIQA, HellaSwag, ARC, Winogrande, PubMedQA, BoolQ, PIQA, SciQ）、以及 Router gradient 与 dense gradient 的相似度。

- 硬件平台是什么，配置是什么。
  AWS 集群 64 GPUs（论文具体 GPU 型号未明确说明，训练代码和配置推断为 A100/H100 级别 GPU）。单 GPU throughput 测试使用 1024 和 2048 hidden dim 的模型分别在 1 GPU 上测试。7.33B 参数模型 per-node throughput 约 1393 tokens/sec（TopK）vs 1391 tokens/sec（DefaultMoE）。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 Llama 架构的 MoE Transformer，使用 SwiGLU FFN、16 attention heads (dim 64)、LayerNorm、RoPE、DeepNet 初始化。总参数量 1.96B（其中 366M 非 MoE 参数，1.6B MoE 参数），active params 根据配置变化（8c1: 565M, 8c2: 764M, 32c1: 416M, 32c2: 466M, 32c4: 565M）。也测试了 hidden dim=512 (557M) 到 hidden dim=2048 (7.33B) 的模型。使用 Llama3 tokenizer。数据集：FineWeb-Edu 和 FineWeb。训练 160B tokens（≈283 tokens/param 的 overtraining 比例）。Benchmarks：LogiQA, MathQA, MMLU, OpenBookQA, LAMBADA, SocialIQA, HellaSwag, ARC (Easy+Challenge), Winogrande, PubMedQA, BoolQ, PIQA, SciQ，使用 lm-eval-harness 评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码已开源：https://anonymous.4open.science/r/default-moe-6C74/，配置文件：https://anonymous.4open.science/r/default-moe-6C74/configs/default-moe-2B.yml。训练基于 gpt-neox + Megablocks + liger kernel (Triton)。核心算法流程：

  伪代码（DefaultMoE 单层前向+反向）：
  ```
  # 前向传播
  Input: token x, router params W (d_token -> N), experts E_0..E_{N-1}
  1. logits = W @ x                          # [N], router logits
  2. pi = Softmax(logits)                    # [N], expert weights
  3. A = TopK(pi, K)                         # indices of K selected experts
  4. for i in A:
       y_i = E_i(x)                          # compute only K expert outputs
  5. for i not in A:
       y_i = EMA_buffer[i]                   # use stored EMA default vector
  6. y = sum(pi_i * y_i for i in 0..N-1)    # weighted sum of all N outputs
  7. for i in A:
       EMA_buffer[i] = beta * EMA_buffer[i] + (1-beta) * mean(E_i(x) over batch)
     # EMA update only for activated experts in this batch

  # 反向传播
  8. dL/d(pi_i) = dL/dy * y_i               # dense gradient: ALL experts contribute
     - Activated experts: y_i = E_i(x)      [true output]
     - Non-activated experts: y_i = EMA_buffer[i]  [default vector]
  9. dL/dW = dL/d(pi) @ d(pi)/dW             # router gets gradient from all N experts
  ```

  张量计算示意（batch_size=B, N=8 experts, K=1 active, hidden=H）：
  ```
  x: [B, H]
  W: [N, H]
  pi: [B, N] = Softmax(x @ W^T)
  TopK mask: [B, N] with exactly K ones per row
  
  # Standard TopK MoE forward
  y_topk: [B, H] = sum_i( mask[b,i] * pi[b,i] * E_i(x[b]) )
  
  # Default MoE forward (dense sum with EMA substitution)
  y_default: [B, H] = sum_i( pi[b,i] * (mask[b,i]*E_i(x[b]) + (1-mask[b,i])*EMA[i]) )
  # Note: EMA[i] is [H], broadcast across batch
  
  # EMA update (only for activated experts)
  for i where mask[:,i] has any True:
    activated_outputs = E_i(x[mask[:,i]])  # [num_activated_i, H]
    EMA[i] = beta * EMA[i] + (1-beta) * mean(activated_outputs, dim=0)
  
  # Router gradient comparison
  # TopK gradient:
  dL/dW[i,:] = dL/dy * (1/B) * sum_b( mask[b,i] * pi[b,i] * x[b] * E_i(x[b]) )
  # DefaultMoE gradient (dense):
  dL/dW[i,:] = dL/dy * (1/B) * sum_b( pi[b,i] * x[b] * (mask[b,i]*E_i(x[b]) + (1-mask[b,i])*EMA[i]) )
  ```
  
  关键超参数：β=0.9（8c1/8c2）、β=0.65（32c1）、β=0.95（32c2）、β=0.999（32c4）。使用 weighted update（按 router probability 加权更新 EMA）后 β 不再敏感。EMA 初始化为零。
