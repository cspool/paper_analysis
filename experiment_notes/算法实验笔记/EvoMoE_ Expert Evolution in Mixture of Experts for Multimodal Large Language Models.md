## EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是针对 MLLM 的 MoE-tuning 框架的两项算法创新：(1) **Expert Evolution**：通过动态平衡先验经验与梯度更新，从单个可训练 FFN 专家逐步演化出多个多样化专家，解决 MoE-tuning 中复制初始化导致的 expert uniformity 问题；(2) **Dynamic Token-aware Router (DTR)**：用两个 hypernetwork（分别处理视觉和文本 token）动态生成每个 router 的 up-sampling 和 down-sampling 层参数，实现基于模态和 token 内在值的自适应专家分配，解决线性路由器的 router rigidity 问题。

  实验比较：(a) EvoMoE vs MoE-LLaVA（MoE-tuning baseline）、Qwen-MoE 等 sparse MLLM，在 0.5B/1.6B/1.8B/2.7B/7B 多种 LLM 规模上对比；(b) EvoMoE 仅激活 top-1 expert vs MoE-LLaVA 激活 top-2 experts，激活参数更少但性能更优；(c) 消融实验：DTR vs Linear Router、Expert Evolution vs Replication Init、不同 DTR 架构设计（single router / modality-specific router / shared router / HyperNet）、不同 expert diversity 策略（Noise、Dropout、Contrastive Loss、Local Loss）；(d) MoE 策略探索：w/o first MoE layer、Shared Expert、不同阶段可训练参数、Alternating vs All-Layer MoE placement；(e) Evolution β 值的选择（固定 vs 随机采样）。

- 硬件平台是什么，配置是什么。
  8x NVIDIA A100-80G GPU，使用 DeepSpeed ZeRO-2（Stage I、II）和 ZeRO-2_offload（Stage III），精度 Bf16。Qwen2-0.5B 训练 25h，StableLM-1.6B 训练耗时未明确给出（Stage I 6h + Stage II 12h + Stage III 7h = 25h 为 Qwen-1.8B 配置），各模型 batch size 均为全局 64（Stage II/III，gradient accumulation 2），文本最大长度 2048。

- 模型是什么。数据集和bench分别是什么。
  模型：基于 MoE-LLaVA 和 LLaVA 1.5 框架，LLM 包括 Qwen2-0.5B、StableLM-1.6B、Qwen-1.8B、Phi2-2.7B、OpenChat-7B，视觉编码器为 CLIP-L 和 SigLIP-L，图像投影层为 MLP with GeLU，输入分辨率 336×336（部分实验 384×384）。MoE 配置：4 个 FFN expert，top-1 路由，alternating MoE layer placement（非全部层）。Stage II 仅训练 FFN1（Expert 1），其他 experts 由 Expert 1 演化生成且冻结；Stage III 仅训练 DTR（hypernetworks + final linear layer），experts 冻结。

  数据集：Stage I（Warm-up）使用 MIMIC-IT、LRV、SViT、LVIS 混合数据；Stage II（Expert Evolution）和 Stage III（DTR）使用 LLaVA-mix-665k。
  
  Benchmarks：Image QA — VQA-v2、GQA、ScienceQA (SQA)、TextVQA；Multi-modal Understanding Toolkits — POPE、MME、MMBench。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供 EvoMoE 开源代码链接，arXiv 页面无 GitHub 仓库。底层框架 MoE-LLaVA（https://github.com/PKU-YuanGroup/MoE-LLaVA）和 LLaVA 1.5（https://github.com/haotian-liu/LLaVA）是开源的。核心算法流程：

  **Expert Evolution（Stage II）**：
  ```
  # 输入：Stage I 输出的 FFN 参数 θ_1（Expert 1），N=4 个 experts
  # 每步训练时，对 Expert n (n=2,3,4) 执行演化：
  for each training step:
      # 仅训练 θ_1（Expert 1），计算梯度 ∇θ_1
      ∇θ_1 = backward(L_regressive + α * L_aux)
      update θ_1 with optimizer

      # 对其他 experts 进行演化：
      for n in [2, 3, 4]:
          β_n = random_sample(range_n)  # range_2=[0.9,0.99], range_3=[0.8,0.89], range_4=[0.7,0.79]
          θ_n ← β_n · θ_1 + (1 - β_n) · ∇θ_1  # EMA 形式的参数演化

  # 关键：θ_2, θ_3, θ_4 不参与梯度计算，仅由 θ_1 通过不同 β 演化而来
  ```

  **DTR Forward Pass（Stage III）**：
  ```
  # 输入：经过 MSA 后的 visual tokens V ∈ R^{P×C} 和 text tokens T ∈ R^{M×C}
  # 仅 H_V, H_T（各含两个 MLP layer）和最终 linear layer φ 可训练

  # Hypernetwork 生成动态参数：
  for τ in {V, T}:
      z' = MSA(LN(z_prev)) + z_prev  # MSA block 输出
      Θ_up^τ, Θ_down^τ = H^τ(z')     # Hypernetwork 生成 up/down 投影矩阵
      
      # Token-aware routing：
      E^τ = Θ_up^τ · SwiGLU(Θ_down^τ · z')  # 动态投影 + SwiGLU 激活
      ρ^τ = φ(E^τ)                            # 最终 linear router 预测 top-k expert 概率

  # 对每个 token，选择 top-1 expert 执行 FFN：
  z_out = FFN_{selected_expert}(LN(z'))
  ```

  训练总损失：L_total = L_regressive + α · L_aux，其中 α=0.001，L_aux 为负载均衡损失（鼓励 experts 均匀处理 token）。
