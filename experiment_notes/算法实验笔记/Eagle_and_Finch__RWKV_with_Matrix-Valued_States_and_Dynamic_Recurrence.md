## Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Eagle（RWKV-5）和Finch（RWKV-6）两种新型RNN架构，沿袭RWKV-4的线性注意力思路。核心创新：(1) Eagle引入multi-headed matrix-valued states（WKV state从向量d扩展为矩阵(D/h)×(D/h)），用LayerNorm（等效GroupNorm）替代分母归一化，添加SiLU attention gating，移除receptance的Sigmoid激活，改进参数初始化；(2) Finch在Eagle基础上引入data-dependent dynamic recurrence：Token Shift从静态lerp升级为ddlerp（data-dependent linear interpolation），通过LoRA（λ + tanh(xA)B）使token shift量依赖输入内容；decay rate w从固定learned vector变为时间可变的w_t = exp(-exp(d_t))，其中d_t也由LoRA生成。同时引入新tokenizer（RWKV World Tokenizer, V=65536, Trie-based greedy matching）和新数据集（RWKV World v2, 1.12T tokens, 70% English + 15% multilingual + 15% code）。
  实验对比：(a) multilingual benchmarks (LAMBADA Multilingual, XCOPA, XNLI, PAWS-X, XStoryCloze, xWinogrande) 和 English benchmarks (LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, GLUE, WinoGrande, SciQ, COPA) 对比Pythia/Mamba/RWKV-4/Llama-2/Falcon/Mistral/MPT等；(b) MQAR合成任务对比RWKV-4和Mamba；(c) PG19长上下文loss vs position对比RWKV-4；(d) Bamboo长上下文推理benchmark；(e) 速度和内存benchmark对比Mamba和Flash Attention (A100 80GB)；(f) 多模态：VisualRWKV (GQA, ScienceQA-IMG, Text-VQA, POPE)、Music modelling、AudioRWKV (AudioSet mAP)；(g) 架构消融（170M参数在Pile上训练330B tokens对比Mamba/RWKV-4/Pythia）；(h) DDLerp消融；(i) AlignBench中文对齐、MTBench、Self-Learning Capability、零样本推理。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA A100 80GB GPU（Eagle 0.4B: 24 GPUs, 1.5B/3B: 48 GPUs; Finch 1.6B/3B同配置），Eagle 7B使用64×H800 GPU。训练精度bfloat16（WKV计算用float32保证数值稳定性）。优化器Adam（β1=0.9, β2=0.99, weight decay=0.001仅用于linear和embedding）。pretraining context length=4096。学习率：linear 10-step warmup (20%→100%) + cosine decay。详细超参数见表17（max LR: 0.4B=4e-4, 1.5B=3e-4, 3B=2e-4, 7B=1.5e-4; micro batch size: 0.4B/1.5B=8, 3B=4, 7B=9; global batch size: 7B=2359296）。速度benchmark: 单张A100 80GB, batch size=8, model dim=4096, head size=64。

- 模型是什么。数据集和bench分别是什么。
  模型：Eagle (RWKV-5) 0.4B (L24/D1024), 1.5B (L24/D2048), 3B (L32/D2560), 7B (L32/D4096); Finch (RWKV-6) 1.6B (L24/D2048), 3B (L32/D2560)。参数公式：#Params_E = 13D²L+14DL+4D+2DV, #Params_F = 13D²L+464DL+4D+2DV (V=65536)。Head dim恒为64，h=D/64。Finch的LoRA矩阵: A∈R^{D×32}, B∈R^{32×D} (A_ω∈R^{D×64}, B_ω∈R^{64×D})。内部状态大小: L(2D+D²/h)=66DL（约5×DL RWKV-4的5.2倍）。Channel Mixing hidden dim从4D缩减至3.5D（Eagle）以保持等参数关系。
  训练数据集：RWKV World v2（1.12T tokens，70% English + 15% multilingual + 15% code），组件包括Wikipedia/SlimPajama/peS2o/BigPatent/Pile of Law/StarCoder/OSCAR23.01/TED2020/PhilPapers/Books3/Gutenberg/OpenSubtitles等（见表9）。Benchmark：(1) LM Evaluation Harness: 多语言(LAMBADA-M, XCOPA, XNLI, PAWS-X, xStoryCloze, xWinogrande) + 英语(LAMBADA, HellaSwag, PIQA, ARC-E/C, GLUE, WinoGrande, SciQ, COPA, OpenBookQA, HeadQA, ReCoRD)；(2) MQAR合成；(3) PG19测试集长上下文；(4) Bamboo长上下文推理（9/10任务有效）；(5) VisualRWKV: GQA, ScienceQA-IMG, Text-VQA, POPE；(6) AUDIOSET mAP；(7) AlignBench, MTBench；(8) 自学习能力(SLC Score); (9) 零样本（Aggression, MathQA, Sarcasm等13个数据集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源（Apache 2.0）。训练代码：https://github.com/RWKV/RWKV-LM，推理代码：https://github.com/RWKV/ChatRWKV，time-parallel训练：https://github.com/RWKV/RWKV-infctx-trainer，模型权重：https://huggingface.co/RWKV。

  Eagle (RWKV-5) Time Mixing算法pipeline (single head, recurrent form):
  ```
  # 参数: W_r, W_k, W_v, W_g ∈ R^{D/h × D/h} (投影矩阵)
  #        μ_r, μ_k, μ_v, μ_g ∈ R^{D/h} (learnable token-shift系数)
  #        ω ∈ R^{D/h} (trainable log-decay), u ∈ R^{D/h} (boost)
  #        s_t ∈ R^{(D/h)×(D/h)} (matrix-valued recurrent state)
  #        W_o ∈ R^{(D/h)·h × D} (output projection)

  Input: x_t ∈ R^D (当前token embedding)
         x_{t-1} (前一token, 用于token-shift)
         s_{t-1} ∈ R^{(D/h)×(D/h)} (上一时刻的WKV state)

  # Step 1: Token Shift (per-head lerp)
  r_t = (x_t + (x_{t-1} - x_t) ⊙ μ_r) @ W_r      # ⊙ element-wise
  k_t = (x_t + (x_{t-1} - x_t) ⊙ μ_k) @ W_k
  v_t = (x_t + (x_{t-1} - x_t) ⊙ μ_v) @ W_v
  g_t = (x_t + (x_{t-1} - x_t) ⊙ μ_g) @ W_g
  # r_t, k_t, v_t, g_t ∈ R^{D/h}

  # Step 2: Decay rate (固定, per-channel)
  w = exp(-exp(ω))  # w ∈ (0,1)^{D/h}, contraction matrix

  # Step 3: WKV computation (加权key-value矩阵)
  #   当前token (特殊处理, boost u):
  wkv_cur = diag(u) · k_t^T · v_t     # ∈ R^{(D/h)×(D/h)}
  #   历史state:
  wkv = wkv_cur + s_{t-1}

  # Step 4: 更新state (decay + accumulate)
  s_t = diag(w) · s_{t-1} + k_t^T · v_t

  # Step 5: Receptance + LayerNorm + SiLU gating + Output
  o_t = LayerNorm(r_t @ wkv)          # ∈ R^{D/h}, LN per head (=GroupNorm, h groups)
  o_t = concat(SiLU(g_t) ⊙ o_t)      # 拼接所有head
  output = o_t @ W_o                  # ∈ R^D

  Return: output, s_t, x_t (for next time step)
  ```

  Finch (RWKV-6) Time Mixing算法pipeline (single head, recurrent form):
  ```
  # 额外参数: A_□ ∈ R^{D×32}, B_□ ∈ R^{32×D} for □∈{r,k,v,g,d}
  #           (decay的A_ω∈R^{D×64}, B_ω∈R^{64×D} 加倍)
  #           μ_x ∈ R^D (ddlerp内的token-shift系数)
  #           λ_□ ∈ R^D (LoRA bias)

  Input: x_t, x_{t-1}, s_{t-1} (同Eagle)

  # Step 1: Data-Dependent Token Shift (ddlerp)
  #   LoRA对token差值进行数据依赖调制:
  lora_□(x) = λ_□ + tanh(x @ A_□) @ B_□   # ∈ R^{D/h}
  ddlerp_□(a,b) = a + (b-a) ⊙ lora_□(a + (b-a) ⊙ μ_x)

  r_t = ddlerp_r(x_t, x_{t-1}) @ W_r
  k_t = ddlerp_k(x_t, x_{t-1}) @ W_k
  v_t = ddlerp_v(x_t, x_{t-1}) @ W_v
  g_t = ddlerp_g(x_t, x_{t-1}) @ W_g

  # Step 2: Data-Dependent Time-Varying Decay
  d_t = lora_d(ddlerp_d(x_t, x_{t-1}))    # ∈ R^{D/h}
  w_t = exp(-exp(d_t))                    # 时间可变decay, ∈ (0,1)^{D/h}

  # Step 3: WKV computation
  wkv_cur = diag(u) · k_t^T · v_t
  wkv = wkv_cur + s_{t-1}

  # Step 4: 更新state (data-dependent decay)
  s_t = diag(w_t) · s_{t-1} + k_t^T · v_t

  # Step 5: Output (同Eagle)
  o_t = LayerNorm(r_t @ wkv)
  output = concat(SiLU(g_t) ⊙ o_t) @ W_o

  Return: output, s_t, x_t
  ```

  Channel Mixing（Eagle和Finch共享, 同RWKV-4但hidden dim减至3.5D）:
  ```
  r'_t = lerp_{r'}(x'_t, x'_{t-1}) @ W_{r'}     # ∈ R^D
  k'_t = lerp_{k'}(x'_t, x'_{t-1}) @ W_{k'}     # ∈ R^{3.5D}
  v'_t = ReLU(k'_t)^2 @ W_{v'}                   # squared ReLU激活
  o'_t = σ(r'_t) ⊙ v'_t                          # sigmoid gating

  # Finch中token-shift也升级为ddlerp:
  r'_t = ddlerp_{r'}(x'_t, x'_{t-1}) @ W_{r'}
  k'_t = ddlerp_{k'}(x'_t, x'_{t-1}) @ W_{k'}
  ```

  训练时并行化：WKV计算在时间维度可通过associative scan或FlashAttention类技术并行化。论文选择沿非时间维度并行，使用custom CUDA kernel将state操作保持在SRAM中。另外提供纯PyTorch time-parallel实现（基于GLA方法）。

  关键设计差异RWKV-4→5→6：
  - RWKV-4: 向量state s∈R^D, head size=1, scalar decay, Sigmoid receptance, 有分母归一化
  - Eagle: 矩阵state s∈R^{(D/h)×(D/h)}, head size=64, per-channel decay w, LayerNorm替代分母, SiLU gating, 无Sigmoid receptance
  - Finch: 矩阵state + data-dependent w_t和ddlerp token-shift

  推理效率：O(1) per-token time, O(D²/h) memory per layer for state。Finch训练时16k序列比Flash Attention快4.2×，比Mamba省17%内存、比Flash Attention省40%内存（A100 80GB, batch=8, D=4096, head=64）。
