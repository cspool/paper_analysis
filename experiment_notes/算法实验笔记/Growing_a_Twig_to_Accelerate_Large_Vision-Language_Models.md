## Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：TwigVLM/TwigVLM++——在冻结的base VLM早期层上"生长"一个轻量twig block，通过两种策略实现推理加速：(1) **Twig-guided Token Pruning (TTP)**：在prefilling阶段，利用twig最后一层（靠近prediction head）的attention map指导视觉token剪枝，替代传统方法中早期层不敏感的attention信号。具体流程：输入tokens X经过base VLM前K层得X^(K)，再经twig block得最后twig层的attention map A^(K+T)，用该attention map选择top-R最重要的视觉tokens保留，其余丢弃。引入**FinalWipe**策略在Kf层后移除所有视觉tokens，使平均保留token数 R̄=[M×K+R×(Kf-K)]/L，在固定R̄下允许更大的R。(2) **Self-Speculative Decoding (SSD)**：在decoding阶段，以浅层子网络Ms（前K层+twig）为draft model自回归生成候选tokens，以深层子网络Mb（完整base VLM）为target model并行验证。draft每步预测5个tokens后触发验证（含early-exit:概率<θ=0.6时停止），接受匹配的tokens并追加一个bonus token。
  TwigVLM++扩展：(i) **Multi-head twig架构**：解耦D-Head（标准next-token prediction）和P-Head（专用于token重要性评分），P-Head通过可学习gating投影Gq/Gk调制自注意力层的Q/K计算重要性分数s=1/H·Σσ((Gq(xq)⊙q̃)(Gk(Xk)⊙K̃)^T/√dh)；(ii) **两阶段训练**：Stage-1用L_NTP+α·L_PredKL+γ·L_AttnKL训练twig，Stage-2用GRPO式RL仅训练P-Head，reward为pruned输入下对参考答案的mean log-probability，配合动态pruning ratio schedule（候选集R={64,...,192}，annealing分布逐渐偏向小R）；(iii) **Tree-based SSD**：draft model构建token tree（expansion width E=10, selection width K=10, depth D=4），target model用tree attention并行验证多条候选路径。

  实验比较：(a) 主实验 —— LLaVA-1.5-7B在6个benchmark(GQA/MMB/MME/VQA^T/SQA^I/VQA^V2)三个pruning ratio(66.7%/77.8%/88.9%)下 vs FastV/SparseVLM/PDrop/MustDrop/VisionZip/FasterVLM；(b) LLaVA-NeXT-7B同样benchmark对比；(c) Qwen2.5-VL-7B在image benchmark(GQA/MME/MMB/SQA^I/VQA^T/VQA^V2/MMStar)和video benchmark(OCRBench/Blink/VideoMME/EgoSchema/MVB)下对比；(d) 生成速度对比 —— TextVQA(短response, S̄≈10)和MM-Vet(长response, S̄≈100)上的RelSpd；(e) 消融 —— 视觉token选择注意力源、加速策略组合(TTP/SSD)、twig block初始化、twig层数T、pruning位置K、FinalWipe位置Kf；(f) TwigVLM++消融 —— Stage-1 head/loss组合、Stage-2 static vs dynamic pruning ratio、RL训练数据量；(g) LLaVA-1.5-13B扩展实验；(h) Token acceptance rate分析、data efficiency训练数据比例实验。

- 硬件平台是什么，配置是什么。
  8×NVIDIA A100 GPU服务器。训练TwigVLM的LLaVA-1.5-7B twig block约10 GPU hours（占base VLM训练时间的~10%），TwigVLM++约20%时间。推理使用相同硬件配置。

- 模型是什么。数据集和bench分别是什么。
  模型：Base VLM包括LLaVA-1.5-7B、LLaVA-NeXT-7B、Qwen2.5-VL-7B、LLaVA-1.5-13B。Twig配置：T=3 twig layers，pruning位置K=2，FinalWipe位置Kf=24。
  训练数据：LLaVA-665K（用于LLaVA-1.5和LLaVA-NeXT的twig训练），MAmmoTH-VL-10M中5M单图样本（用于Qwen2.5-VL的twig训练）。Stage-2 RL仅用50K SFT样本。
  Benchmarks：GQA、MMBench(MMB)、MME、TextVQA(VQA^T)、ScienceQA-IMG(SQA^I)、VQA-v2(VQA^V2)、MM-Vet、MMStar、OCRBench、Blink、VideoMME、EgoSchema、MVBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/MILVLG/twigvlm (Apache 2.0 License)

  算法 pipeline 伪代码（TwigVLM推理流程，论文Algorithm 1）：

  ```
  # bVLM: 完整 base VLM M_b
  # twig: twig block (T transformer layers)
  # K: 共享的低层数
  # K_f: FinalWipe层位置
  # R: 剪枝后保留的visual token数
  # delta: maximum draft token length (default 5)
  # theta: 停止draft的置信度阈值 (default 0.6)
  def sVLM_forward(tokens):
      X_k = bVLM.forward_low_layers(tokens, k=K)
      prob, Attn_last = twig.forward(X_k)
      a_i = argmax(prob)
      return X_k, prob, Attn_last, a_i

  def TwigVLM_inference(img, ques):
      draft_toks = []
      final_resp = []
      # Prefilling阶段：sVLM前向
      X_k, _, Attn, a_i = sVLM_forward((img, ques))
      draft_toks.append(a_i)
      # TTP: 用twig最后层的attention剪枝visual tokens
      # X_k_b: bVLM的共享token latent
      X_k_b = pruning(X_k, Attn, r=R)  # 按Eq.(5): P(X^(K)_Mb, A^(K+T)_Ms, R)
      # SSD循环: draft→verify迭代
      while EOS_TOKEN not in final_resp:
          X_k, prob, _, a_i = sVLM_forward(a_i)
          draft_toks.append(a_i)
          X_k_b = concat(X_k_b, X_k, axis=1)
          # 停止draft并触发验证
          if len(draft_toks) >= delta or prob < theta:
              # FinalWipe: 移除K_f层后的所有visual tokens
              tgt_probs = bVLM.forward_high_layers(
                  X_k_b, k=K, final_wipe=K_f)
              # 验证draft tokens
              right_toks = [a for a, p in zip(draft_toks, tgt_probs[:-1])
                            if argmax(p) == a]
              right_toks.append(argmax(tgt_probs[-1]))
              final_resp.extend(right_toks)
              draft_toks = []
              X_k_b = None
              a_i = final_resp[-1]
      return final_resp
  ```

  TwigVLM++ P-Head评分计算（Eq.7）：
  ```
  # 输入: X^(K+T) — twig最后一层SA层的输入
  # Q, K = X^(K+T)·W_q, X^(K+T)·W_k  (标准自注意力投影)
  # q̃ ∈ R^{H×d_h}: query向量(最后textual token位置)
  # K̃ ∈ R^{H×M×d_h}: key矩阵(visual token位置)
  # G_q, G_k: 可学习gating投影(Linear+nonlinear activation)
  scores_m = []
  for h in range(H):
      gated_q = G_q(x_q)^{(h)} ⊙ q̃^{(h)}     # element-wise gating
      gated_k = G_k(X_k)^{(h)} ⊙ K̃^{(h)}
      scores_h = softmax(gated_q @ gated_k.T / sqrt(d_h))
      scores_m.append(scores_h)
  s = mean(scores_m, dim=0)  # 最终token重要性分数 ∈ R^M
  # 用s替代原attention map进行pruning: Eq.(8)
  ```

  TwigVLM++ Stage-2 RL (GRPO-style, Eq.12-15)：
  ```
  # π_θ: P-Head产生的token重要性分布
  # 对每个样本采样 G=32 个pruning action a_i
  for a_i in range(G):  # 每个action: 无放回采样R个visual token位置
      π_i = π_θ
      for j in range(R):
          a_j ~ Categorical(π_i)          # 按当前分布采样
          π_i[a_j] = 0                     # 移除已选位置
          π_i = π_i / sum(π_i)            # 重归一化
      # reward = pruned输入下生成参考答案的mean log-prob
      r_i = (1/S) * Σ log p_Mb(y*_j | X̂, y*_{<j})  # Eq.(13)
  # Group-level advantage normalization
  Â_i = (r_i - mean({r})) / std({r})                # Eq.(14)
  # 纯on-policy更新 (importance ratio = 1)
  L_stage2 = (1/G) * Σ Â_i * log π_θ(a_i)           # Eq.(15)
  ```

  算法 pipeline 全栈执行流程（以 LLaVA-1.5-7B base + TwigVLM 为例）：
  - 算法层：Image → CLIP Vision Encoder → 576 visual tokens + text prompt tokens → 拼接输入 → base VLM前2层 → twig block (3层, 初始化为VLM第3-5层权重) 得attention map → TTP按R=41剪枝保留top visual tokens → 剩余VLM层(3-32层)处理pruned序列(含FinalWipe在24层移除所有visual tokens), 同时twig作为draft model自回归生成候选tokens → base VLM(target model)并行验证并接受匹配tokens → 生成答案。
  - 系统框架层：基于HuggingFace Transformers，复用base VLM权重初始化twig，仅训练twig block（冻结前K层和剩余层）。训练用LLaVA-665K SFT数据+AR loss。推理时draft和target model共享前K层KV-cache。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准FlashAttention（v2.3.2），利用GPU并行计算能力通过SSD将decode阶段从逐token串行转为批token并行验证。Tree-based SSD使用tree attention（topology-aware causal mask替代标准causal mask）。
  - 硬件架构层：8×NVIDIA A100 GPU训练，推理可用单卡A100。
