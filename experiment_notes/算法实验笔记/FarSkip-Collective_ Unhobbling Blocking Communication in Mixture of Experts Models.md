## FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - FarSkip-Collective 提出一种修改 MoE 模型架构连接性的方法，通过使计算能够在通信进行期间使用"过时"（outdated）或"部分"（partial）激活值来消除阻塞通信模式。核心实现包括两部分：
    1. **FarSkip-Collective 架构修改**：修改 Transformer 层的残差连接，使得下一子块的计算输入 $o_k^*$ 不再等待当前子块的完整输出 $o_k$。提出两种变体：
       - (8a) "Outdated"：$o_k^* = o_{k-1}$，直接使用上一层的完整输出
       - (8b) "Partial"：$o_k^* = o_{k-1} + f_k^*(o_{k-1}^*)$，使用当前子块中不依赖通信的部分计算结果
       对于 Attention 子块输入，使用 partial activation：$\text{attn-in}_k = o_{k-2} + \text{attn-out}_{k-1} + \text{shared-exp-out}_{k-1}$（缺失 routed-exp-out_{k-1}，使得 Combine 通信可被重叠）。对于 MLP 子块输入，使用 outdated activation：$\text{mlp-in}_k = o_{k-1}$（使 Dispatch 通信可被重叠）。
    2. **FCSD（FarSkip-Collective Self-Distill）**：通过 KL 散度知识蒸馏将原始模型转化为 FarSkip-Collective 模型。以原始模型为 teacher，FarSkip 修改后的模型为 student，使用 KL 散度 loss $\mathcal{L}_{KD}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} [\sum_t KL(q(\cdot \mid x, y_{<t}) \parallel p_{\theta}(\cdot \mid x, y_{<t}))]$ 训练。训练配方：AdamW + cosine-annealing LR scheduler + 1000-step warmup，batch-size 从 $\{2^{16}, 2^{17}, 2^{18}\}$ 中 sweep 选择，learning rate 从 {2e-5, 4e-5, 8e-5} 中 sweep 选择，最多训练 10B tokens，使用 MBPP+ 作为 early stopping 验证集（patience=20 evals, delta=2%）。
  - 实验比较：FCSD 蒸馏的 FarSkip-Collective 模型 vs 原始模型 vs SFT baseline，在 11 个下游评测上对比准确性（Tab. 1）。蒸馏方法消融：KL vs KL+Inter.L2 vs SFT vs KL+Embed Freeze vs 不同 batch-size（Tab. 2）。层数消融：不同比例（50%/75%/90%/100%）和不同位置（从首层/从末层）的 FarSkip 层替换（Fig. 3）。Pretraining from scratch：从头预训练 FarSkip 架构 vs 常规架构的 loss curve 对比（Fig. 8, Tab. 4）。

- 硬件平台是什么，配置是什么。
  - 训练：1× AMD MI325X 8GPU 机器（单节点）；多节点扩展：4 节点 × 4×MI325X（每节点 8GPU），节点间 400Gbps 互联。
  - 推理：1× AMD MI300X 8GPU 机器；多节点：2 节点系统，8×400Gbs NIC 互联。
  - 软件环境：PyTorch（torch.dist async_op + CUDA Stream），Megatron-LM（训练），vLLM & SGLang（推理），HIP/CUDA-graphs。

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-V2-Lite (16B-A3B, 64 experts), Qwen-3-30B MoE (30B-A3B), Llama-4-Scout (109B-A17B), DeepSeek-V2 (235B), DeepSeek-V3 (671B, 用于训练的缩短版 L=6 约 71B)。
  - 训练数据：GenQA [43] 和 Infinity Instruct [22] 的 SFT 数据，最多 10B tokens。
  - 下游评测 Benchmark（11 个）：PIQA, ARC-Easy, ARC-Challenge, HellaSwag, CommonsenseQA, WinoGrande, HumanEval+, MMLU, OpenBookQA, GSM-8K, MBPP+。
  - Pretraining 评测：ARC-C, ARC-E, BoolQ, HellaSwag, MMLU, OpenBookQA, PIQA, SCIQ, WinoGrande。
  - 性能指标：通信重叠率（overlap %），端到端加速比（speed-up），Time-To-First-Token (TTFT)，Time-Between-Tokens (TBT)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文声明 "We plan to open-source our implementation and modified model checkpoints"，截至分析时未在 web search 中发现公开代码仓库。
  - **算法 pipeline 解释**：

  **FarSkip-Collective 架构修改**（以单个 MoE Transformer 层为例）：

  原始 MoE 层前向（常规连接性）：
  ```
  # 输入: o_{k-1} (上一层输出)
  # Step 1: Attention sub-block
  attn_out = Attention(LayerNorm(o_{k-1}))
  o_k_attn = o_{k-1} + attn_out         # 残差连接

  # Step 2: MoE gating (阻塞等待 o_k_attn)
  gate_scores = Router(LayerNorm(o_k_attn))

  # Step 3: Dispatch all-to-all (阻塞通信)
  tokens = AllToAllDispatch(o_k_attn, gate_scores)  # 暴露通信气泡

  # Step 4: Routed experts + Shared experts
  routed_out = RoutedExperts(tokens)
  shared_out = SharedExperts(o_k_attn)

  # Step 5: Combine all-to-all (阻塞通信)
  combined = AllToAllCombine(routed_out)  # 暴露通信气泡

  # Step 6: 最终输出
  o_k = o_k_attn + shared_out + combined
  ```

  FarSkip-Collective 修改后的前向（Section 4.1 训练执行顺序）：
  ```
  # 输入:
  #   o_{k-2}: 上上层输出
  #   attn_out_{k-1}: 上一层 attention 输出
  #   shared_out_{k-1}: 上一层 shared expert 输出
  #   routed_out_{k-1}: 上一层 routed expert 输出 (待 Combine)

  # Step 1: Attention part (a) — q, k, v 准备
  #   attn-in_k = o_{k-2} + attn_out_{k-1} + shared_out_{k-1}  (partial)
  q, k, v = MLA_prepare(attn-in_k)     # 不依赖 routed_out_{k-1}

  # Step 2: 同步上一层的 Combine (如果上一层是 FarSkip MoE 层)
  WaitCombine(routed_out_{k-1})         # 此时 Combine 已被重叠

  # Step 3: MoE gating
  gate_scores = Router(LayerNorm(o_{k-1}))

  # Step 4: 异步启动 Dispatch
  DispatchAsync(tokens, gate_scores)    # async_op=True, 立即返回

  # Step 5: Attention part (b) — core attention + output projection
  #   Dispatch 在后台运行，与 attention 计算重叠
  attn_out_k = MLA_core_attn(q, k, v)

  # Step 6: 同步 Dispatch，执行 routed experts
  WaitDispatch()
  routed_out_k = RoutedExperts(dispatched_tokens)

  # Step 7: 异步启动 Combine
  CombineAsync(routed_out_k)            # 后台运行

  # Step 8: Shared experts (与 Combine 重叠)
  shared_out_k = SharedExperts(o_{k-1})

  # 最终输出在下一层同步 Combine 时获取
  ```

  **重叠窗口条件**（Eq. 9）：
  $$T_{\text{Dispatch}} + T_{\text{Combine}} \le T_{\text{overlappable}} = T_{\text{layer}} - (T_{\text{Routed Experts}} + T_{\text{Gate}})$$

  **FCSD 训练伪代码**：
  ```python
  # 加载原始模型作为 teacher（冻结）
  teacher = load_checkpoint("original_moe")
  teacher.eval()
  for p in teacher.parameters():
      p.requires_grad = False

  # 初始化 FarSkip-Collective student（与 teacher 参数形状相同，仅连接性不同）
  student = convert_to_farskip(teacher)  # 修改 skip connections
  student.train()

  # Sweep: batch_size ∈ {2^16, 2^17, 2^18}, lr ∈ {2e-5, 4e-5, 8e-5}
  optimizer = AdamW(student.parameters(), lr=best_lr)
  scheduler = CosineAnnealingLR(optimizer)

  for step in range(max_steps):
      x = next_batch()  # SFT data (GenQA + Infinity Instruct)
      with torch.no_grad():
          teacher_logits = teacher(x)
      student_logits = student(x)
      loss = KL_divergence(teacher_logits, student_logits)
      loss.backward()
      optimizer.step()
      scheduler.step()

      # Early stopping: 每 1000 steps 评估 MBPP+
      if step % 1000 == 0:
          mbpp_score = evaluate_mbpp_plus(student)
          if is_instability(mbpp_score, patience=20, delta=0.02):
              break
  ```

  **训练中的 Sequence Number Hijacking**（反向传播）：
  在反向传播中，FarSkip 使用 PyTorch autograd 的 Sequence Number 机制重新排序计算优先级。默认 autograd 按节点创建顺序（与正向相同顺序）处理就绪节点。FarSkip 重新分配 Sequence Number，将子块反向计算节点优先级提高，将通向通信输入的节点优先级降低，使得在通信等待期间先执行子块计算，最大化重叠窗口。

  **Pretraining from scratch 结果**（Fig. 8, Tab. 4）：
  从头预训练 DeepSeek-V2-Lite 架构（16B, 64 experts）50B tokens，FarSkip vs Regular：
  - 最终 training loss: 2.205 vs 2.187（最后 50 步平均）
  - 下游评测平均分: 54.7 vs 54.4
