## EfficientQAT Efficient Quantization-Aware Training for Large Language Models

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出EfficientQAT，一种高效的量化感知训练框架，分两阶段：(1) Block-AP (Block-wise training of All Parameters)：逐block训练所有权重和量化参数（步长s、零点z）；(2) E2E-QP (End-to-End training of Quantization Parameters)：冻结量化权重，仅端到端训练步长s。实验对比三类方法：
  - PTQ：GPTQ、AWQ、OmniQuant、AutoRound、ApiQ、CBQ、QuIP#（向量量化）、AQLM（向量量化）
  - QAT：LLM-QAT、BitDistiller、PB-LLM、DB-LLM
  - Q-PEFT：QLoRA、QA-LoRA、PEQA、IR-QLoRA
  评估指标：5个zero-shot常识推理任务（WinoGrande、PIQA、HellaSwag、Arc-Easy、Arc-Challenge）平均准确率、WikiText2和C4困惑度、MMLU（5-shot）指令微调准确率、MME/MMBench/MM-Vet/ScienceQA多模态评估。

- **硬件平台是什么，配置是什么。**
  单张NVIDIA A100-80GB GPU。Block-AP阶段Llama-2-7B需8.5GB显存、13B需10.3GB、70B需29.9GB。E2E-QP阶段4/3/2-bit分别需7.0/6.4/5.6GB（7B）、11.7/10.6/9.1GB（13B）、48.4/42.0/34.2GB（70B）。推理加速测试使用BitBLAS在A100-80GB上评估INT2矩阵向量乘法加速比。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：Llama-2（7B/13B/70B）、Llama-3（8B/70B）用于量化对比；Llama-1（7B/13B）用于指令微调；LLaVA-1.5（7B/13B）用于多模态指令微调
  - 数据集：RedPajama（4096样本，Block-AP上下文长度2048、E2E-QP上下文长度4096）；Alpaca用于指令微调（源长度384、目标长度128、10000步、batch size 16）
  - Benchmark：lm-evaluation-harness v0.4.2（5个zero-shot常识推理任务）、WikiText2和C4困惑度（上下文长度2048）、MMLU（5-shot）、MME/MMBench/MM-Vet/ScienceQA（多模态）

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/OpenGVLab/EfficientQAT

  **量化与反量化过程**（均匀量化，per-group共享步长s和零点z）：
  ```
  W_int = clamp(round(W / s) + z, 0, 2^N - 1)   # N-bit量化
  W_hat = (W_int - z) * s                         # 反量化，用于前向计算
  ```
  其中W为FP16/BF16全精度权重，s为FP16步长，z为N-bit零点。每g个权重共享一组(s, z)。

  **Block-AP阶段**（逐block训练所有参数）：
  ```
  for each transformer_block in model:
      for epoch in [1, 2]:
          for batch in calibration_data:            # RedPajama 4096 samples
              # 前向：对block内所有Linear层执行量化+反量化
              W_int = clamp(round(W/s) + z, 0, 2^N-1)
              W_hat = (W_int - z) * s
              output = transformer_block(input, W_hat)
              loss = MSE(output, fp16_block_output)  # 重建损失
              # STE反向传播，更新 W, s, z
              W_grad = STE_gradient(loss, W_hat)
              s_grad = compute_s_gradient()           # 详见Eq.(3)
              z_grad = compute_z_gradient()           # 详见Eq.(4)
              W -= lr_W * W_grad                      # lr_W=2e-5(2-bit)/1e-5(3/4-bit)
              s -= lr_s * s_grad                      # lr_s=1e-4
              z -= lr_z * z_grad
  ```

  **E2E-QP阶段**（端到端仅训练步长s）：
  ```
  freeze(W_quantized)  # 冻结Block-AP产出的量化权重
  for batch in training_data:                         # RedPajama 4096 samples, ctx=4096
      W_hat = (W_quantized - z) * s                   # 仅反量化，无量化过程
      output = model(input, W_hat)
      loss = cross_entropy(output, labels)            # 语言模型损失
      s_grad = (W_quantized - z) * loss_grad          # ∂W_hat/∂s = W_q - z
      s -= lr_s * s_grad                              # lr_s=2e-5(2-bit)/1e-5(3-bit)
  ```

  **训练效率**：Llama-2-70B的2-bit量化仅需41 GPU小时（单A100-80GB），Block-AP 26.6h + E2E-QP 14.3h。对比DB-LLM需82h、BitDistiller需64h、LLM-QAT需900h（均需≥4 GPU）。平均位宽计算公式：bits/param = N + (N+16)/g，其中g为group size。g=64时2-bit量化为2.28 bits/param。

  **推理加速**：使用BitBLAS在A100-80GB上测试，INT2矩阵向量乘法加速比约2.9x-4.4x（vs FP16 linear layer）。兼容MLC-LLM、AWQ、Marlin、T-MAC等推理框架。

- 属于算法pipeline的实现是什么？实验比较什么？
  BinaryDM 提出了基于量化感知训练（QAT）的扩散模型权重二值化方法，将DM权重推至1-bit极限。核心实现包含两个组件：(1) Evolvable-Basis Binarizer (EBB)，通过双基二值化+正则化驱动演化到单基全二值化，增强初期表征能力；(2) Low-rank Representation Mimicking (LRM)，通过PCA低秩投影对齐全精度和二值化DM的中间表征，改善优化方向。实验比较了vanilla baseline（基础sign二值化+LSQ激活量化）vs BinaryDM，以及对比了多种SOTA方法：LSQ、Q-Diffusion、EfficientDM、Q-DM、TDQ、ReActNet、INSTA-BNN、BI-DiffSR，覆盖W1A32/W1A8/W1A4多种位宽配置。

- 硬件平台是什么，配置是什么。
  Intel Xeon Gold 6336Y 2.40GHz CPU + NVIDIA A100 40GB GPU。实际硬件推理效率测试使用 Qualcomm Snapdragon 855 Plus 和 Larq 通用部署库，测得单次卷积 BinaryDM 38.2ms vs FP 176.4ms（4.62×加速）。

- 模型是什么。数据集和bench分别是什么。
  模型：pixel-space DDIM（CIFAR-10 32×32）和 latent-space LDM（LDM-4/LDM-8，基于U-Net+spatial transformer的噪声估计网络）。数据集：CIFAR-10 32×32（无条件）、LSUN-Bedrooms 256×256（无条件）、LSUN-Churches 256×256（无条件）、FFHQ 256×256（无条件）、ImageNet 256×256（条件生成）。评估指标：IS、FID、sFID、Precision-and-Recall，使用ADM TensorFlow评估套件，随机生成50000样本计算。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/Xingyu-Zheng/BinaryDM（ICLR 2025）。基于 latent-diffusion 和 stable-diffusion 代码库构建。

  算法pipeline（QAT二值化训练与推理流程）：
  1. **初始化**：从预训练全精度DM加载权重，初始化为带可学习标量的二值化权重 w^bi = σ * sign(w)，σ_0 = ||w||/n
  2. **第一阶段（EBB多基+正则化）**：
     - 前向二值化：w_EBB^bi = σ_I * sign(w) + σ_II * sign(w - σ_I * sign(w))
     - 卷积计算：o = σ_I * (a ⊗ sign(w)) + σ_II * (a ⊗ sign(w - σ_I * sign(w)))，⊗为仅含加法的卷积（XNOR+popcount）
     - LRM低秩对齐：对全精度DM中间表征ε̂_θi^FP计算协方差 C_i = (hw)⁻² * ε̂ * ε̂^T，特征分解取前⌈c/4⌉列特征向量E_i，投影R_i^FP = ε̂^FP * E_i，R_i^bi = ε̂^bi * E_i
     - 总损失：L_total = L_simple + (9e-2)/N * Σ σ_II + (1e-4)/M * Σ ||R_i^FP - R_i^bi||
     - STE近似sign函数梯度反向传播
  3. **第二阶段（单基全二值化）**：
     - σ_II→0后，移除高阶项：w^bi = σ_I * sign(w)
     - 继续LRM蒸馏训练，投影矩阵固定不变
  4. **位置选择**：EBB仅应用于首尾各6层（约15%参数），中间层使用vanilla binarizer，减少过渡阶段的不稳定性
  5. **推理**：W1A4时将4-bit激活分解为4个1-bit激活+偏置项，基于Larq W1A1算子实现
