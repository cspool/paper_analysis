## UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  UniQL提出统一的后训练量化（PTQ）与结构化剪枝联合压缩框架，核心由四个算法组件构成：
  （1）**伪逆无关的MLP结构化权重排序**（Pseudo-inverse-free structured weight-sorting）：对MLP层的up/gate/down投影矩阵，通过校准数据集采集中间激活，计算通道相关性矩阵的ridge leverage scores（λ=1），按得分排序通道。无需求解Moore-Penrose伪逆，避免O(n³)复杂度和FP64精度要求，相比MoDeGPT达到22×加速（19min vs 7h3min）。
  （2）**量化感知SVD分解**（Quantization-aware SVD）：对MHSA的value-output权重组执行两次连续SVD分解（C^{1/2}W_v = U_vΣ_vV_v^T，然后SVD(Σ_vV_v^T W_o) = UΣV^T），排序eigenvectors。关键创新：将长尾特征值对角阵Σ融合到U（W = (UΣ)V），使得每列的σ_i充当该列量化组的scaling factor，避免低比特量化（INT4）对数值分布的敏感性。不融合Σ时4-bit 25%剪枝精度仅为60.2%，融合后提升至67.7%（+7.5%）。
  （3）**状态感知SSM权重排序**（State-aware weight sorting）：针对Mamba块，将SSM计算拆分为输入掩码M（B和C权重）与状态H（z/x/o权重）两个子公式。B-C排序考虑输入依赖的离散化Δ^g通过广播外积(ΔB)^g，计算多SSM头相关性的范数得分。z-x-o排序直接从SSM状态H^T H收集相关性计算ridge leverage scores。
  （4）**Masked LoRA微调**：在已排序但未剪枝的模型上使用LoRA（r=8，α=16），每次训练步随机采样全局剪枝率P_t ∈ [P_15, P_20, ...]，层间剪枝率由Block Influence (BI) scores分配。训练数据集Alpaca，5 epochs，单张GPU一次完成。最终仅需一次云上压缩即可支持设备端0%-35%的可配置剪枝率。
  
  实验比较：结构化剪枝baseline（MoDeGPT、SVD-LLM）在FP16和4-bit下的zero-shot准确率；PTQ baseline（TRT-AWQ、TAO-HQQ、GPTQ）在W4A16下的准确率和模型尺寸；单轮自适应剪枝（one-pass）vs 需要多次运行才能支持不同剪枝率的baseline。

- 硬件平台是什么，配置是什么。
  云端压缩：单块NVIDIA A6000 GPU，48GB显存。
  边缘推理：NVIDIA Jetson Orin Nano 8GB（统一内存架构）。
  云端推理延迟测量：A6000 GPU（1k prefill + 1k generation tokens）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B、Llama-3.1-8B、Qwen-2.5-7B（Transformers）；Mamba2-8B（SSM）；Nemotron-H-8B、Bamba-v2-9B（Mamba-Transformer Hybrid）。
  数据集/benchmark：zero-shot五任务平均——HellaSwag（length-normalized acc）、PIQA（acc）、ARC-easy（acc）、ARC-challenge（length-normalized acc）、WinoGrande（acc）。附加MMLU（5-shot）、MBPP+（0-shot coding）。
  校准数据集：WikiText-2（BI score + PTQ校准）、Alpaca（权重排序 + masked LoRA微调）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码和量化模型已开源：https://github.com/enyac-group/UniQL
  
  算法pipeline全过程（以Llama-3.1-8B MLP层为例，D_h=4096, D_int=14336）：
  ```
  # Step 1: 收集中间激活并排序（Algorithm 1）
  For i in calibration_samples:
      X_int[i] = σ(X_h[i] @ W_g) ⊙ (X_h[i] @ W_u)  # shape [128, 14336]
  C = mean_i(X_int[i]^T @ X_int[i])                   # [14336, 14336]
  s = diag(C @ (C + λI)^{-1})                          # ridge leverage scores
  S_m = I[:, argsort(s)]                              # 排序矩阵 [14336, 14336]
  # 重排权重
  W_u' = W_u @ S_m     # [4096, 14336], 列按重要性降序
  W_g' = W_g @ S_m
  W_d' = S_m^T @ W_d   # [14336, 4096], 行按重要性降序

  # Step 2: Masked LoRA微调
  # 计算BI scores分配层间剪枝率
  s_layer[l] = 1 - E[x_l^T y_l / (||x_l|| ||y_l||)]
  P_layer = L * P_avg * softmax(-s / 0.1)
  # 训练循环
  For each step t:
      随机采样 P_t ∈ {P_15, P_25, P_35}
      mask = top-k(W channels by P_t)  # 去除排名最低的通道
      loss = LoraFT(W_masked, input)
      W = W + ΔW_LoRA

  # Step 3: GPTQ量化 (W4A16, group_size=128)
  For each column i:
      s = max(|W_col_i|) / (2^3 - 1)     # INT4 max=7
      W_q_col_i = clamp(round(W_col_i / s), -8, 7)

  # Step 4: 设备端自适应剪枝
  # 加载INT4权重 → 在线解包 → 去除末尾通道 → 重新打包为INT32
  D'_int = D_int * (1 - p%)  # p=0,15,25,35
  # 保留前D'列的通道，丢弃末尾通道
  ```
  关键张量操作：MLP排序将SiLU-gated MLP中间维度D_int的通道按重要性重排，剪枝时只需减少D_int维度而不改变隐藏层维度D_h，保证各层适配。8B模型在FP16下从16GB压缩至4.1GB（4×压缩），35%剪枝后仅2.8GB（5.7×压缩）。
