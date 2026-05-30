## See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：ECRD（Evidence-Constrained Reweighting Decoding），一种 training-free、plug-and-play 的推理时解码框架，不修改模型权重，仅在 test time 介入 token selection 过程。包含两个核心组件：(1) **Distribution Supervisor（分布监督器）**：维护一个文本证据池（textual evidence pool），对 base LVLM 的 top-k 候选 token 计算证据诱导分布 r_i(w)，然后与 base 分布 p_i(w) 通过自适应权重 α_i = p_{(1)}（base 模型 top-1 概率）进行协商混合，得到 p_i^{mix}(w)。当 base 模型置信度高时（p_{(1)} 大），保持 base 分布主导；当 base 分布分散时（p_{(1)} 小，hallucination-prone），证据权重增大。(2) **Visual Decider（视觉裁决器）**：由 GRIT（基于 Qwen2.5-VL-3B 的视觉定位模型）实例化，当协商分布的 margin Δ_i = p_{(1)}^{mix} - p_{(2)}^{mix} ≤ δ 且 k* > 1 时触发，读取图像和当前推理前缀，输出一个候选 token w* 和一条人类可读的微观察证据句 E_i，强制提交 w* 并将 E_i 追加到证据池。证据池仅包含文本（坐标仅用于可解释性，不参与 scoring），避免反复编码图像裁剪。

  实验比较：
  (a) TreeBench 上 vs base LVLMs（Qwen2.5-VL 7B/32B/72B、LLaVA-OneVision 7B/72B、InternVL3 8B/38B/78B），ECRD 在各 backbone 和 scale 上一致提升 +4.5~+10.9 个点 overall accuracy；(b) vs RL-based 视觉定位推理模型 DeepEyes-7B、Pixel-Reasoner-7B、TreeVGR-7B；(c) vs training-free baselines Woodpecker、ViperGPT、ControlMLLM、beam search、self-consistency、diverse sampling；(d) vs VDGD（Visual Description Grounded Decoding，ECRD 的前身方法）；(e) RH-Bench 上 Reasoning/Perception/RH-AUC；(f) V*Bench、MathVista、ChartQA、OCRBench、HallusionBench 五个通用多模态 benchmark；(g) Ablation: base only vs +VDGD vs +supervisor vs +supervisor+Qwen2.5-VL-3B decider vs +supervisor+GRIT-3B decider (full ECRD)；(h) 不确定性阈值 δ 的 cost-accuracy trade-off 分析（Fig. 3）；(i) 定性分析：supervisor 重新加权解决歧义 vs visual decider 中链注入视觉证据 vs visual decider 直接输出最终答案。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA H20-NVLink GPU。所有测试在 H20 上进行：t_0（δ=0 时每问题平均时间）在 V*Bench 8.98s、MathVista 12.92s、ChartQA 9.76s、OCRBench 3.24s、HallusionBench 11.67s。l_0（单次 visual decider 调用全局平均延迟）在 1.12-1.46s 之间。Latency model: T(δ) ≈ t_0 + l_0 · r(δ)，其中 r(δ) 为每问题平均 decider 调用次数。

- 模型是什么。数据集和bench分别是什么。
  模型：Base LVLMs 包括 Qwen2.5-VL 系列（7B/32B/72B）、LLaVA-OneVision 系列（7B/72B）、InternVL3 系列（8B/38B/78B）。Visual Decider 为 GRIT-3B（基于 Qwen2.5-VL-3B，针对视觉定位优化）。Ablation 中还测试了 Qwen2.5-VL-3B 作为 decider 的变体。Base 模型和 decider 均冻结，不做任何 fine-tuning。Private model 参考线：GPT-4o、o3、Gemini-2.5-Flash、Gemini-2.5-Pro。

  Benchmarks：(i) TreeBench — 评估"thinking with images"，分为 Perception（Attr./Mater./Phys./ObjRet./OCR）和 Reasoning（Persp./Order./Cont.&Oc./Contain./Compar.），metric 为 answer accuracy；(ii) RH-Bench — 评估 Reasoning/Perception 和 RH-AUC（平衡推理长度和幻觉的指标）；(iii) V*Bench — 视觉搜索引导的多模态能力（Attr./Spatial/Overall）；(iv) MathVista — 视觉上下文中的数学推理；(v) ChartQA — 图表问答；(vi) OCRBench — OCR 能力；(vii) HallusionBench — 语言幻觉与视觉错觉诊断。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/uuuuZYC/See-It-Say-It-Sorted

  ECRD 算法 pipeline 伪代码（以单步解码为例）：

  ```
  # ===== 初始化 =====
  E_0 = {d_global}  # 证据池初始化为全局图像描述
  prefix = [instruction_tokens, image_tokens]

  # ===== 逐步解码循环 =====
  for step i in 0..max_len:
      # 1. Base LVLM forward: 获取 next-token 分布
      logits_i = frozen_LVLM(prefix)  # shape: [vocab_size]
      p_i = softmax(logits_i)         # base 分布

      # 2. Knee truncation: 选择 top-k* 候选集
      p_sorted = sort(p_i, descending=True)
      k* = argmax_k (p_sorted[k] - p_sorted[k+1])  # 式(2)
      C_i = {w_1, ..., w_k*}  # 候选 token 集，式(3)

      # 3. 证据评分: 对每条证据计算 mean-over-prefix 概率
      for each evidence sentence E_j in E_i:  # E_j = (e_1,...,e_L)
          q_Ej(w) = (1/L) * sum_{t=1..L} p_VLM(w | e_{<t})  # 式(5)
          # 即 evidence sentence 每个 prefix 下 token w 的平均条件概率
      S_i(w) = -log( (1/N) * sum_{j} q_Ej(w) )  # 式(6)
      # 证据池中 N 条证据的平均支持度（log 空间）

      # 4. 证据诱导分布: 仅在 C_i 内归一化
      r_i(w) = softmax_{w in C_i}(-S_i(w))  # 式(7)

      # 5. 质量匹配缩放: r_i 的总 mass 匹配 p_i 在 C_i 内的 mass
      mass_p = sum_{w in C_i} p_i(w)
      mass_r = sum_{w in C_i} r_i(w)
      r_tilde_i(w) = r_i(w) * (mass_p / mass_r)  # for w in C_i，式(8)

      # 6. 协商混合: base + evidence 的自适应融合
      alpha_i = max(p_i)  # top-1 概率作为自适应权重，式(11)
      p_mix_i(w) = alpha_i * p_i(w) + (1-alpha_i) * r_tilde_i(w)  # w in C_i, 式(10)
      p_mix_i(w) = alpha_i * p_i(w)                          # w not in C_i

      # 7. 不确定性检测: 决定是否调用 visual decider
      margin = max(p_mix_i) - second_max(p_mix_i)  # 式(12)
      if k* > 1 and margin <= delta:  # delta=0.08
          # 触发 visual decider
          w_star, evidence_sentence = GRIT(image, prefix_tail, C_i)
          # GRIT 读图 + 当前文本前缀 + 候选集，输出最佳 token 和证据句
          x_i = w_star              # 强制提交 decider 选择的 token
          E_{i+1} = E_i ∪ {evidence_sentence}  # 追加到证据池，式(13)
      else:
          x_i = argmax(p_mix_i)     # 直接取混合分布 top-1
          E_{i+1} = E_i             # 证据池不变

      prefix = concat(prefix, x_i)  # 更新解码前缀

  # ===== 终止 =====
  # 当生成 EOS token 或达到 max_len 时停止
  # 最终 answer 从完整 prefix 中提取
  ```

  张量计算关键维度：
  - logits_i: [1, vocab_size]（如 Qwen2.5-VL-7B 的 vocab 约 152k）
  - C_i: [1, k*]，k* 由 knee truncation 动态决定，通常为个位数
  - q_Ej(w): 对每条证据句子 E_j 的每个 prefix 位置计算 p_VLM(w|e_{<t})，对 |E_i| 条证据取平均。证据评分 O(k*|E_i|)，由于 k* 为个位数且 |E_i| 增长缓慢，overhead 很小
  - alpha_i: 标量，p_{(1)} ∈ [0,1]
  - p_mix_i: [1, vocab_size]，仅在 C_i 内的 mass 被重新分配
  - visual decider 调用延迟: l_0 ≈ 1.12-1.46s/call（H20 GPU）
  - GRIT 输出: w* ∈ C_i（单个 token），E_i = 一句自然语言证据（约 20-50 tokens）+ 可选坐标标注

  关键设计理念：
  - **Training-free**: LVLM 和 GRIT decider 完全冻结，无需任何 fine-tuning/RL/偏好优化
  - **Cost-aware**: visual decider 仅在 k*>1 且 margin≤δ 时触发，δ=0.08 时 r(δ) 在低个位数，达到 accuracy-cost elbow
  - **Textual evidence**: 证据池仅存文本，无需反复编码图像裁剪，后续 token 可直接引用之前的微观察，形成跨步长 evidence reuse
  - **自适应权重 alpha_i = p_{(1)}**: 当 base 模型自信时（分布尖锐），alpha→1 保持 base 主导；当 base 模型犹豫时（分布平坦），alpha 小，证据权重增大，精准在 hallucination-prone 步干预
