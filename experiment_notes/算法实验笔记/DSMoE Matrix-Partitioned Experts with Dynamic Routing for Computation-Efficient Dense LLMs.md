## DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **DSMoE (Dynamic Sparse Mixture-of-Experts)**，一种将预训练 Dense 模型的 FFN 层矩阵沿 intermediate 维度划分为多个 expert 块，并通过 sigmoid 门控 + straight-through estimator + 稀疏损失实现动态、输入自适应的稀疏激活的方法。三个核心模块：
  
  **FFN Partitioning**：将 SwiGLU FFN 中的上投影矩阵 U、门控矩阵 W、下投影矩阵 V 沿 intermediate 维度均等划分为 n 组（如 LLaMA-7B 中 D=11008 划分为 8×1376），每组构成一个 expert。划分后所有 expert 输出之和在数学上等价于原始 FFN 输出。
  
  **Straight-Through Estimator**：前向传播时通过阈值 τ=0.5 的阶跃函数 G(x) 控制稀疏激活；反向传播时通过 S(x)=sg(G(x))+x-sg(x) 允许梯度穿过未激活 expert 的门控参数 Y_i（公式16），使非激活 expert 也能根据输出 o_i 是否有益于降低损失来更新路由参数，解决"死 expert"问题。
  
  **Sparse Loss**：L = L_LM + (1/LN) Σ G(σ(ĥY_n))，L1 范数惩罚门控激活值，与门控梯度形成对抗效应，鼓励模型主动抑制不重要 expert 的输出。不引入传统 MoE 的 load balancing loss。
  
  实验比较：
  - **Perplexity (Table 1)**：DSMoE vs LLM-Pruner (channel-wise/block-wise)、SparseGPT (非结构化剪枝)、LLaMA-MoE (传统 MoE top-k)，在 LLaMA-1B (激活参数 735M) 和 LLaMA-7B (激活参数 3.93B) 两档
  - **Downstream Benchmarks (Table 2)**：10 个下游任务（HellaSwag/LAMBADA/PIQA/SIQA/StoryCloze/Winogrande + GSM8K/NaturalQs/TriviaQA/WebQs），zero-shot 和 5-shot
  - **Ablation: Straight-Through Estimator (Table 3)**：有 S(x) vs 无 S(x)（仅用 G(x)），PPL 从 7.41 退化至 12.75
  - **Ablation: Piecewise Function G(x) (Fig. 2)**：训练时不使用 G(x)（用连续 sigmoid），推理时加阈值，PPL 随 τ 增大急剧上升
  - **Layer-wise Activation Patterns (Fig. 3)**：热力图分析各层专家激活数分布，发现 W 形激活模式
  - **Threshold Sweep (Table 4)**：τ=0.2~0.8 下 PPL 与激活参数比例的关系

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号和具体硬件配置。论文提及使用 LLaMA-1B 和 LLaMA-7B 模型进行继续预训练，训练 10B tokens 数据，batch size=32，sequence length=1024，learning rate=2e-5，但未披露使用的 GPU 类型、数量和显存配置。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - **LLaMA-1B**（基于 LLaMA-3.2-1B）：d=2048, D=8192, 总参数 1.24B, 激活参数 735M (8 experts, D=1024×8)
  - **LLaMA-7B**（基于 LLaMA-2-7B）：d=4096, D=11008, 总参数 6.74B, 激活参数 3.93B (8 experts, D=1376×8)
  
  数据集：
  - 继续预训练数据：Fineweb-edu (通用)、OpenWebMath (数学)、StarCoder (代码)、Cosmopedia (合成数据)，混合后总计 10B tokens，tokenizer 限制最大长度 1024
  
  Benchmarks：
  - 验证集 PPL（从各数据集随机采样 5000 条非重叠样本）
  - Zero-shot: HellaSwag, LAMBADA, PIQA, SIQA, StoryCloze, Winogrande
  - 5-shot: GSM8K (exact match), NaturalQs (exact match), TriviaQA (exact match), WebQs (exact match)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：**论文未提供公开开源代码仓库**。PapersWithCode 页面标注 "No code implementations yet"。论文对应的 arXiv ID 为 2502.12455。

  **DSMoE 算法 Pipeline（基于 PyTorch 伪代码）**：

  ```
  # === 符号 ===
  # x: [B, d] 输入 hidden states
  # U_list, W_list, V_list: 各 n 个切片的 FFN 矩阵
  #   U_i: [d, D/n], W_i: [d, D/n], V_i: [D/n, d]
  # Y: [d, n] 门控网络参数矩阵
  # tau: 激活阈值 (默认 0.5)
  
  def dsmo_e_ffn_forward(x, U_list, W_list, V_list, Y, tau=0.5, training=True):
      """
      DSMoE FFN layer forward pass
      x: [B, d] - input hidden states after attention
      """
      n = len(U_list)  # number of experts
      B, d = x.shape
      
      # Step 1: Compute gate logits and sigmoid
      gate_logits = x @ Y  # [B, n]
      gate_probs = sigmoid(gate_logits)  # [B, n], values in (0, 1)
      
      # Step 2: Piecewise gating with Straight-Through Estimator
      if training:
          # STE: forward uses hard threshold, backward passes gradient
          gate_hard = gate_probs.clone()
          gate_hard[gate_hard <= tau] = 0.0  # G(x) in forward
          gate_values = gate_hard + gate_probs - gate_probs.detach()  # S(x) = sg(G) + x - sg(x)
      else:
          # Inference: only hard threshold
          gate_values = gate_probs.clone()
          gate_values[gate_values <= tau] = 0.0
      
      # Step 3: Compute expert outputs
      outputs = []
      for i in range(n):
          # SwiGLU FFN for expert i
          # o_i = (act(x @ W_i) ⊙ (x @ U_i)) @ V_i
          gate_part = silu(x @ W_list[i])  # [B, D/n]
          up_part = x @ U_list[i]          # [B, D/n]
          expert_out = (gate_part * up_part) @ V_list[i]  # [B, d]
          outputs.append(expert_out)
      
      # Step 4: Weighted sum with gating
      # h = Σ o_i * S(σ(x @ Y_i))
      h = sum(
          outputs[i] * gate_values[:, i:i+1]  # [B, d] * [B, 1]
          for i in range(n)
      )  # [B, d]
      
      # Step 5: Activation count normalization
      # Scale by n / num_active to maintain output norm
      active_mask = (gate_probs > tau).float()  # [B, n]
      num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
      h = h * (n / num_active)  # [B, d]
      
      return h, gate_values, num_active.mean()
  
  # === Loss Computation ===
  def compute_loss(lm_loss, gate_values_list, L, N):
      """
      Total loss = Language Modeling Loss + Sparse Loss
      gate_values_list: list of gate_values from each layer
      L: number of Transformer layers
      N: number of experts per layer
      """
      sparse_loss = 0.0
      for gate_vals in gate_values_list:
          # L1 norm on gated activations (gate_vals already thresholded)
          sparse_loss += gate_vals.sum()
      sparse_loss = sparse_loss / (L * N)
      return lm_loss + sparse_loss
  ```

  **张量计算流程（以 LLaMA-7B, d=4096, n=8, D/n=1376 为例）**：
  
  1. 输入 x: [B, 4096]
  2. 门控计算：x @ Y: [B, 4096] × [4096, 8] → [B, 8]，sigmoid 后得到每个 expert 的激活概率
  3. 硬阈值：STE 前向将 ≤0.5 的值置零，反向保持梯度流
  4. Expert 计算：每个 expert i 执行 x @ W_i [B, 1376] ⊙ x @ U_i [B, 1376] → intermediate [B, 1376]，再 @ V_i [1376, 4096] → [B, 4096]
  5. 加权求和：Σ o_i · gate_i → [B, 4096]
  6. 归一化：× n / num_active → 最终输出 [B, 4096]
