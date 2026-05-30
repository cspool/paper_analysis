## TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

- 属于算法pipeline的实现是什么？实验比较什么？
  TransPrune 是一种训练无关（training-free）的 LVLM 视觉 token 剪枝方法，属于 within-LLM pruning。核心实现包括两个互补的 token 重要性评估准则：
  (1) **Token Transition Variation (TTV)**：测量每个 token 在 self-attention 和 FFN 模块中表征的变化。定义幅度变化 m(F, T_in) = ||T_out||₂ / ||T_in||₂，方向变化 d(F, T_in) = cosine_similarity(T_out, T_in)。TTV = Softmax(1 - |d|) · m，对 self-attention 和 FFN 的 TTV 求和得到每层总 TTV。
  (2) **Instruction-Guided Attention (IGA)**：计算 instruction token 对 image token 的 attention 权重平均值，引入任务相关的语义监督。
  (3) **Accumulation Mechanism**：由于 TTV 模式在各层间不稳定，对中间层（7-12）的 TTV 进行跨层累积，在每个 pruning layer 做出更精确的剪枝决策。最终 Score = α·TTV + (1-α)·IGA，α=0.5，pruning 在 layers 7, 9, 12 执行。

  实验比较：(1) 与 within-LLM 方法（FastV、TopV、PDrop、ShortV、SparseVLM）在 LLaVA-v1.5-7B 上的 8 个 benchmark 对比（MME^P、VQA^v2、Seed^I、TextVQA、SQA^I、POPE、GQA、MMB^en），分为 ~40-50% TFLOPs（TransPrune-High: 1.56 TFLOPs / 40.8%）和 ~25-35% TFLOPs（TransPrune-Low: 1.19 TFLOPs / 31.2%）两档（Table 1）；(2) LLaVA-NeXT-7B 上的同类对比，TransPrune-High 达 8.33 TFLOPs (40.0%)（Table 2）；(3) Qwen2.5-VL-7B 上与 FastV 的对比（Table 3）；(4) 与 projector-based 方法（VisionZip、CDPruner）的联合剪枝效果（Table 4-5）；(5) Video-LLaVA 上视频 benchmark（TGIF、MSVD）的泛化实验（Table 6）；(6) 消融实验：TTV-only 有效性（Table 7）、不同 layer 选择（Table 9）、浅层 vs 深层 accumulation（Table 10）、accumulation 机制有无（Table 11）、magnitude vs direction 贡献（Table 12）、α 参数影响（Table 13）；(7) 延迟（ms）和显存（GB）的实际测量对比（Table 8）。

- 硬件平台是什么，配置是什么。
  所有实验在 **A100 GPU (40GB)** 上进行。推理时使用 **FlashAttention** 进行高效 attention 计算。TransPrune 的 TTV 计算仅需模块输入/输出，IGA 仅计算 instruction→image token 的 attention（非完整 attention map），因此与 FlashAttention 兼容。

- 模型是什么。数据集和bench分别是什么。
  模型：**LLaVA-v1.5-7B**、**LLaVA-NeXT-7B**、**Qwen2.5-VL-7B**（不同架构验证泛化性）。视频模型：**Video-LLaVA**。
  数据集/Benchmark：**MME、MMBench(MMB^en)、SEED(Seed^I)、ScienceQA(SQA^I)、VQA-v2、POPE、GQA、TextVQA**（共 8 个），覆盖 perception、reasoning、VQA 任务。视频 benchmark：**TGIF、MSVD**。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明代码将在接收后开源于 https://github.com/liaolea/TransPrune（截至分析时尚未完全公开）。
  
  算法 Pipeline 伪代码：

  ```
  # 输入: 视觉 tokens T_I (shape: [N, d]), instruction tokens T_inst (shape: [L, d])
  # 超参: accumulation_layers A = {7,8,9,10,11,12}, pruning_layers P = {7,9,12}, α=0.5

  retained_indices = range(N)  # 初始保留所有 token

  for each transformer layer l = 1 to max_layer:
      # 前向传播（FlashAttention）
      T_out = TransformerLayer_l(T_in)
      
      if l in A:  # accumulation layer
          # 计算 TTV
          T_attn_out = SelfAttention(T_in[retained_indices])
          T_ffn_out = FFN(T_attn_out)
          
          d_attn = cosine_similarity(T_attn_out, T_in[retained_indices])  # [N_retained]
          m_attn = ||T_attn_out||_2 / ||T_in[retained_indices]||_2       # [N_retained]
          TTV_attn = Softmax(1 - |d_attn|) * m_attn
          
          d_ffn = cosine_similarity(T_ffn_out, T_attn_out)
          m_ffn = ||T_ffn_out||_2 / ||T_attn_out||_2
          TTV_ffn = Softmax(1 - |d_ffn|) * m_ffn
          
          TTV[l] = TTV_attn + TTV_ffn  # 存储当前层 TTV
      
      if l in P:  # pruning layer (e.g., 7, 9, 12)
          # 累积 TTV
          TTV_acc = sum(TTV[j] for j in A where j <= l)   # Equation (4)
          
          # 计算 IGA（用下一层 l+1 的 attention）
          A_inst2img = softmax(Q_inst[l+1] @ K_img[l+1].T / sqrt(d))
          IGA = mean(A_inst2img, dim=instruction)          # Equation (5), [N_retained]
          
          # 组合得分
          Score = α * TTV_acc + (1-α) * IGA                # Equation (6)
          
          # 剪枝：保留得分最高的 K 个 token
          keep_count = schedule[l]  # TransPrune-High/Low 预设保留数
          retained_indices = topk(Score, keep_count)
  ```

  TTV 关键张量计算（Equation 1-3）：
  - m(F, T_in) = ||T_out||₂ / ||T_in||₂ （幅度变化率）
  - d(F, T_in) = (T_out · T_in) / (||T_out||₂ · ||T_in||₂) （方向余弦相似度）
  - TTV(F, T_I) = Softmax(1 - |d(F, T_I)|) · m(F, T_I) （Equation 2）
  - TTV_l(T_I) = TTV(Attention, T_I) + TTV(FFN, T_I) （Equation 3，每层汇总）

  额外 FLOPs 开销（Equation 7）：主要由 TTV 的 L2 norm 和 cosine similarity 计算、IGA 的 instruction-visual attention 组成，开销与 stage 数 s 和 token 维度 d 线性相关（O(sd)），对比 baseline 总计算量可忽略（Table 8 显示 TransPrune 延迟最低 111.4ms，显存 14.82GB）。
