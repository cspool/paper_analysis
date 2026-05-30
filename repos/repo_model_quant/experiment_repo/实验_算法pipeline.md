# 实验_算法pipeline

## Training Dynamics Impact Post-Training Quantization Robustness

- 属于算法pipeline的实现是什么？实验比较什么？
  研究训练过程中的超参数（学习率、学习率调度策略、weight decay、weight averaging）对模型 PTQ 鲁棒性的影响。核心发现：量化误差主要由学习率衰减驱动，而非训练数据规模；较大的稳定学习率、weight averaging（LAWA/model soup）和更大的 weight decay 均能降低 PTQ 误差。实验比较：(a) 不同开源模型训练轨迹的量化误差演变（OLMo/OLMo2/SmolLM3/Apertus/OpenSci/Amber），涵盖 3-bit 和 4-bit GPTQ；(b) 受控实验：不同 token 预算、学习率大小（1e-3 到 6e-3）、WSD vs cosine 调度、不同 weight decay（λ）、AdamW vs AdamC 优化器下的量化误差；(c) LAWA weight averaging vs 中间 cooldown 的性能对比；(d) 量化误差与 loss landscape 几何性质（Hessian sharpness/trace）的关联。

- 硬件平台是什么，配置是什么。
  自训练实验：最多 8 块 NVIDIA A100-80GB GPU。下游评估使用 vLLM。开源模型评估使用 HuggingFace Transformers + GPTQModel/GPTQ backend。

- 模型是什么。数据集和bench分别是什么。
  开源模型：OLMo-1B/7B、OLMo2-1B/7B/13B/32B、SmolLM3-3B、Apertus-8B、OpenSci-1.3B、Amber-7B。自训练模型：Pythia-160M 和 70M 参数 Transformer（Vaswani 架构），在 FineWebEdu 上训练最多 100B tokens（seq_len=2048, batch_size=0.5M tokens）。校准数据集：C4（GPTQ 量化，group_size=128）。评估：held-out RefinedWeb（validation loss）；下游 benchmark：ARC-Challenge、ARC-Easy、OpenbookQA、PIQA、HellaSwag、WinoGrande、MathQA、PubMedQA、SciQ、Social IQa、CommonsenseQA、MMLU（12 个任务，5-shot）。评估框架：LM-eval-harness + vLLM。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确给出自研代码仓库。量化使用 GPTQModel（https://github.com/modelcloud/gptqmodel）和 HuggingFace 内置量化后端。预训练使用 plainLM（https://github.com/Niccolo-Ajroldi/plainLM）。Hessian 分析使用 PyHessian。

  核心分析 pipeline 伪代码：
  ```
  def analyze_ptq_robustness(checkpoints, lr_schedule, quant_method="GPTQ"):
      results = []
      for step, ckpt in enumerate(checkpoints):
          # Step 1: 量化模型
          W = ckpt.weights                        # FP16/BF16 weights
          W_q = quantize(W, method=quant_method)  # 3-bit or 4-bit GPTQ
          W_hat = dequantize(W_q)                 # reconstruct from scales + lowbit

          # Step 2: 计算相对交叉熵损失
          ce_full = cross_entropy(f(X; W), y)
          ce_quant = cross_entropy(f(X; W_hat), y)
          rel_ce = (ce_quant / ce_full) - 1.0

          # Step 3: 记录学习率状态
          lr = lr_schedule[step]
          results.append((step, lr, rel_ce, ce_full))

      return results

  # WSD 调度下的发现:
  # - 稳定阶段（constant lr=3e-3）：量化误差温和上升 ~1%
  # - 衰减阶段（lr→0）：量化误差急剧飙升 ~20%
  # - 峰值学习率越大（6e-3 > 3e-3 > 1e-3），衰减后量化误差越小
  # - 不同 token 预算（10B-100B）在 cooldown 后有可比的量化误差
  ```

  Weight averaging 介入方式（LAWA）：
  ```
  def lawa_quantization(checkpoints, window=5):
      # 沿训练轨迹滑动平均
      avg_weights = mean(checkpoints[-window:])  # 最近 K 个 checkpoint 均匀平均
      W_q = quantize(avg_weights, method="GPTQ", bits=3)
      # LAWA 的量化误差可比肩甚至优于 lr cooldown 后的模型
      return W_q
  ```

  Hessian 几何分析 pipeline：
  ```
  def hessian_analysis(W, val_data):
      # Hutchinson estimator for Hessian trace
      z = rademacher_sample(dim=W.numel())
      Hv = hessian_vector_product(loss, W, z)
      trace_est = z @ Hv                          # unbiased trace estimator

      # Power iteration for max eigenvalue (sharpness)
      lambda_max = power_iteration(loss, W, n_iters=50)

      # 发现：lr 衰减时 sharpness 和 trace 同时激增
      # 较大峰值 lr → 更平坦的 loss basin → 更低量化误差
      return lambda_max, trace_est
  ```


## Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 *aespa*（attention-centric efficient and scalable post-training quantization algorithm）算法，是一种面向超大规模 Transformer 模型的 PTQ 方案。核心策略是：逐层（layer-wise）执行量化以保持效率，但以注意力输出（attention-wise）重构为目标来考虑跨层依赖。具体实现分两步：(1) 使用 Z-FOLD 结合提出的 Hessian 计算量化参数（scale 和 zero-point）；(2) 使用 AdaRound 基于提出的精炼损失函数优化 weight-rounding policy。精炼量化目标分别为：W_V 最小化 `tr(ΔW_V·E[XA^TAX^T]·ΔW_V^T)`、W_Q 最小化 `tr(E[K^TK]·ΔW_Q·E[XX^T]·ΔW_Q^T)`、W_K 最小化 `tr(E[Q^TQ]·ΔW_K·E[XX^T]·ΔW_K^T)`。通过预计算 E[XX^T]、E[XA^TAX^T]、E[K^TK]、E[Q^TQ]，每轮迭代仅需 O(d_h d^2) FLOPs，相比传统 block-wise 方法 O(B d_h L·max{d,L})，在 OPT-125M 上约快 10 倍。仅做 weight-only 量化（激活保持 FP16）。实验比较：(a) block-wise PTQ：BRECQ、OmniQuant、AffineQuant；(b) layer-wise PTQ：RTN、OPTQ、Z-FOLD。在 WikiText-2、C4、PTB 上评估 PPL，并用 ARC-c/e、HellaSwag、MMLU 评估零样本推理性能。精度涵盖 INT2/3/4/6，特别在 INT2 精度下优势显著（如 OPT-6.7B INT2，aespa PPL=15.71 vs OmniQuant=4900+）。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA A100 GPU（80 GB），LLaMA2 实验使用 NVIDIA H100 GPU。校准数据集使用从 C4 随机采样 128 段 2048 token 的序列。量化处理时间：OPT-125M 约 5 分钟完成 INT2 量化（BRECQ 需要 ~2 小时），OPT-1.3B 约 1.24 小时（BRECQ 需要 ~10.7 小时）。

- 模型是什么。数据集和bench分别是什么。
  模型：OPT（125M/350M/1.3B/2.7B/6.7B/13B/30B）、BLOOM（560M/1.1B/1.7B/3B/7.1B）、LLaMA（7B/13B/30B）、LLaMA2（7B/13B）。校准数据集：C4（128 segments × 2048 tokens）。评估基准：WikiText-2、C4、PTB（perplexity）、零样本任务 ARC-c、ARC-e、HellaSwag、MMLU。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/SamsungLabs/aespa（NeurIPS 2024 官方实现，CC BY-NC 4.0，Python，5 commits），包含 quantize.py、quantizer.py、quant_utils.py、aespa.py、main.py 及定制模型文件（modeling_llama_custom.py、modeling_bloom_custom.py）。

  算法 pipeline 伪代码（参考论文 Algorithm 1 及 Table 4）：
  ```
  def aespa_quantization(W, X_calib):
      # Step 1: Pre-compute Hessian matrices
      H_xx = E[XX^T]                              # for all layers
      H_v = E[X A^T A X^T]                        # for W_V only
      E_ktk = E[K^T K]                            # for W_Q
      E_qtq = E[Q^T Q]                            # for W_K

      # Step 2: Determine quantization parameters (scale, zero-point)
      for each layer l:
          if l in {W_Q, W_K, W_V}:
              H = corresponding_attention_hessian  # use attention-aware Hessian
          else:
              H = H_xx                            # standard Hessian for FFN/out-proj
          # Z-FOLD: optimize step size s to minimize tr(ΔW·H·ΔW^T)
          s = argmin_s tr(ΔW(s)·H·ΔW(s)^T)

      # Step 3: Initialize W_int via OPTQ (optional but empirically faster)
      W_int = OPTQ_round(W, s)

      # Step 4: Optimize weight-rounding policy via AdaRound with proposed losses
      for iter in range(2000):
          for W_Q:  loss = tr(E[K^T K]·ΔW_Q·H_xx·ΔW_Q^T) + λ·round_reg
          for W_K:  loss = tr(E[Q^T Q]·ΔW_K·H_xx·ΔW_K^T) + λ·round_reg
          for W_V:  loss = tr(ΔW_V·H_v·ΔW_V^T) + λ·round_reg
          for other: loss = tr(ΔW·H_xx·ΔW^T) + λ·round_reg
          update W_int via gradient descent (lr=0.015)

      return W_int, s
  ```
  张量计算示例（W_V 量化，基于 Equation 17-18）：预计算阶段对校准数据全量计算 `H_V = 2 * mean(X @ A^T @ A @ X^T)`，形状为 [d, d]。每次迭代计算损失 `loss = sum((ΔW_V @ H_V) ⊙ ΔW_V)`，无需执行 attention forward pass，复杂度 O(d_h d^2) 与校准数据量无关。

## Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Granular-DQ，一种 patch-wise、layer-invariant 的动态量化方法，包含两个顺序步骤：(1) Granularity-Bit Controller (GBC) 构建粗到细的多粒度层次表示，按每个 patch 对整张图像的贡献比例分配 bit-width；(2) Entropy-to-Bit (E2B) 机制基于像素熵统计对高 bit patch 进行细粒度 bit-width 自适应调整，配合 Adaptive Threshold Calibration (ATC) 利用 EMA 动态校准熵阈值。实验比较基线：与全精度模型及 PAMS、CADyQ、CABM、AdaBM、RefQSR 对比 PSNR/SSIM 和 FAB (Feature Average Bit-width)；消融研究 GBC、E2B、ATC 各自贡献及候选 bit 配置和阈值数量的影响。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 GPUs，PyTorch 框架实现。训练时 LR RGB patch 随机裁剪至 48×48（CNN）或 64×64（Transformer），batch size 16，300K iterations，初始 lr=2×10^{-4}，250K iterations 后减半。

- 模型是什么。数据集和bench分别是什么。
  CNN 模型：SRResNet、EDSR、IDN；Transformer 模型：SwinIR-light、HAT-S。训练集：DIV2K（800 样本，×2 和 ×4 SR）。评估 benchmark：Urban100、Test2K、Test4K（源自 DIV8K 经 bicubic 下采样）。指标：PSNR、SSIM（重建精度）、FAB（量化效率）、BitOPs（计算复杂度）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/MmmingS/Granular-DQ.git。算法流程：
  1. 输入图像 X → GBC 编码器 E 提取 D 层多粒度特征 Z={Z_1,...,Z_D}（分辨率递减，Z_1 最细粒度，Z_D 最粗粒度）。
  2. 所有粒度特征 GroupNorm + 平均池化到 Z_D 分辨率 → concat → GAP → 通道统计量 S。
  3. 线性层 W_g ∈ R^{(N×D)×N} 作用于 S 生成门控 logits G，对每个 patch X_i 使用 Gumbel-Softmax 采样门控索引 θ_i = argmax_n(g_{i,n} + σ_n)，计算门控分数 p_i（patch 贡献概率），映射到候选 bit code b_n ∈ {4,6,8}。
  4. E2B：对训练集所有 LR patch 计算像素熵 H（基于 Gaussian 加权核密度估计），按升序排序得到 H。插入分位数阈值 t1=0.5, t2=0.9 将 H 划分为 3 个子区间，对应 bit codes [4,5,8]。对 GBC 分配高 bit 的 patch，据其熵 E 落入区间决定适配 bit-width。
  5. ATC：训练首 epoch 用 EMA (γ=0.9997) 动态校准阈值 t^(j) = t^(j-1)·γ + Norm(E)·(1-γ)。
  6. 量化器：QuantSR 作为候选量化方案，权重统一 8-bit 线性量化。仅使用 L1 loss 训练。Transformer 的 attention block 保持全精度。

## Task-Specific Zero-shot Quantization-Aware Training for Object Detection

- 属于算法pipeline的实现是什么？实验比较什么？
  提出首个面向目标检测的task-specific Zero-shot Quantization（ZSQ）框架，包含两个阶段：(1) **Task-Specific Calibration Set Synthesis**：使用Adaptive Label Sampling从预训练检测网络中以零样本方式重建目标类别、位置和尺寸分布，结合task-specific检测损失L_detect（含L_category、L_box、L_conf）与task-agnostic先验损失L_prior（BNS对齐或Patch Similarity Entropy）合成带标注的校准集；(2) **Task-Specific QAT with Distillation**：联合KL散度知识蒸馏（L_KD）、特征级MSE蒸馏（L_feat）和task-specific检测训练损失（L_detect）微调量化网络。实验在W8A8/W6A6/W4A8/W5A5/W4A4多种位宽下比较：YOLOv5-s/m/l、YOLO11-s/m/l、CNN-backbone Mask R-CNN、Swin-T/S Transformer-backbone Mask R-CNN。对比方法包括LSQ、LSQ+（real-data QAT）以及Genie、ZeroQ（task-agnostic ZSQ），均在MS-COCO 2017和Pascal VOC验证集上用mAP/mAP50评估。

- 硬件平台是什么，配置是什么。
  YOLOv5/YOLO11实验：2× NVIDIA GeForce RTX 4090 GPU；Mask R-CNN实验：4 GPU；ViT实验：8 GPU。实现基于PyTorch框架。

- 模型是什么。数据集和bench分别是什么。
  模型：YOLOv5-s/m/l、YOLO11-s/m/l（单阶段检测器）；Mask R-CNN + ResNet backbone（CNN两阶段检测器）；Mask R-CNN + Swin-T/S backbone（Transformer两阶段检测器）。数据集与Bench：MS-COCO 2017验证集（mAP/mAP50）、Pascal VOC验证集（mAP）。所有模型使用预训练FP32权重作为teacher初始化。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/DFQ-Dojo/dfq-toolkit
  
  算法Pipeline（以YOLOv5-s W6A6 MS-COCO为例）：
  
  **Stage I — Task-Specific Calibration Set Synthesis：**
  1. 初始化输入x ∈ R^{N×3×160×160}，每个像素从高斯噪声N(0,1)采样。随机生成单目标标签y（category ∼ U(0,C)，bbox中心∼U(W/2,1-W/2)，bbox宽高∼U(0.2,0.8)）。
  2. Adaptive Label Sampling循环（Algorithm 1）：每固定间隔用预训练teacher ϕ(θ)对当前x做前向推理 → 取conf > conf_thresh的高置信度预测作为new_tgts → 计算IOU(new_tgts, 当前tgts) → 添加不与现有标签重叠的新标签 → 移除未被teacher检测到的旧标签 → 确保每张图至少保留一个标签。
  3. 固定采样得到的标签y，重新初始化高斯噪声x ∈ R^{N×3×640×640}，用task-specific损失优化：
     L_total = α_prior · L_prior(x) + α_detect · L_detect(ϕ(x), y) + L_reg(x)
     其中L_prior为BNS alignment loss（CNN模型）或Patch Similarity Entropy loss（Transformer模型），L_detect = L_category + L_box + L_conf，L_reg = α_TV·L_TV + α_l2·||x||₂²。
  4. 优化2500次迭代（YOLOv5），Adam优化器，lr=1e-2，余弦退火，使用Cutout数据增强。生成2k张合成校准样本。

  **Stage II — QAT with Task-Specific Distillation：**
  1. 对全精度网络ϕ(θ)的所有内部层（除首尾层外）附加LSQ量化器，使用per-tensor symmetric quantization，量化公式：w_int = clip(⌊w_fp/s⌉, -2^{b-1}, 2^{b-1}-1)，ŵ_fp = w_int × s。
  2. 对每个合成样本(ẍ_i, ŷ_i)计算三项损失：
     - L_KD = (τ²/N)·Σ KL(z^F(ẍ_i;θ), z^Q(ẍ_i;θ'))：预测匹配KL散度蒸馏，τ为温度
     - L_feat = (1/(NL))·Σ||f_l^F(ẍ_i;θ) - f_l^Q(ẍ_i;θ')||₂²：特征级MSE蒸馏，L为蒸馏层数
     - L_detect = L_category + L_box + L_conf：task-specific检测损失
  3. 总损失：L^Q = β_KL·L_KD + β_feat·L_feat + β_detect·L_detect
  4. Adam优化器训练QAT，YOLOv5 lr=1e-4，超参{β_detect, β_KL, β_feat} = {0.04, 0.1, 1.0}。量化scale因子s通过反向传播联合学习。
  
  关键数值结果：YOLOv5-l W6A6 mAP=45.1%（超越full-data LSQ 43.3%达+1.8pp），使用仅1/60训练数据（2k vs 120k）；收敛速度可达LSQ的16×。

## SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  提出SYNQ，一个面向Zero-shot Quantization（ZSQ）的合成感知微调框架，无需任何真实训练数据即可对预训练模型进行量化。核心包含三个创新组件：(1) **高斯低通滤波器（Low-pass Filter）**：对合成数据集在频域应用高斯低通滤波去除高频噪声，使合成样本的幅度分布更接近真实图像；(2) **类激活图对齐（CAM Alignment）**：通过MSE损失对齐预训练模型与量化模型的Grad-CAM显著性图，确保量化模型基于正确图像区域进行预测；(3) **困难样本仅用软标签（Soft Labels for Difficult Samples）**：根据预训练模型预测概率定义样本难度δ(x_i,θ)=1-q(x_i;θ)，对难度超过阈值τ的样本仅使用KL散度（软标签），不施加交叉熵损失（硬标签），防止错误标签误导训练。实验比较W4A4和W3A3量化下的Top-1准确率，对比方法包括GDFQ、ARC、Qimera、ARC+AIT、IntraQ、AdaSG、AdaDFQ、HAST、TexQ、PLF。

- 硬件平台是什么，配置是什么。
  所有实验在配备Intel Xeon Silver 4214和NVIDIA RTX 3090的工作站上完成。实现基于PyTorch和TorchVision库，Python语言。

- 模型是什么。数据集和bench分别是什么。
  模型：ResNet-20（用于CIFAR-10和CIFAR-100）、ResNet-18、ResNet-50、MobileNetV2（用于ImageNet）；ViT模型包括DeiT-Tiny、DeiT-Small、Swin-Tiny、Swin-Small（均预训练于ImageNet）。数据集：CIFAR-10、CIFAR-100、ImageNet（ILSVRC 2012），仅用于评估（不参与训练）。评估指标：Top-1准确率。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/snudm-starlab/SynQ
  
  算法Pipeline（以ResNet-18 W3A3 ImageNet为例）：
  1. **合成数据集生成（Step 1）**：初始化5120个服从高斯噪声的样本{x_i}及随机类别标签{y_i}。迭代优化最小化Inception Loss L_IL（交叉熵）与Batch Normalization Statistics Loss L_BNS（BN层running mean和std的L2距离），生成与原始分布相似的合成样本。baseline进一步集成TexQ的calibration center synthesis和HAST的hard sample generation/promotion。
  2. **低通滤波（Idea 1）**：对每个合成样本x_i应用高斯低通滤波：x_i^F = F^{-1}(G ⊙ F(x_i))，其中G_{uv} = exp(-D(u,v)²/(2D₀²))，D_0为截止频率超参数（搜索范围{20,40,60,80,100}）。F和F^{-1}分别为FFT和逆FFT。
  3. **量化初始化**：使用RTN（Round-To-Nearest）方案将全精度模型θ量化为θ^q。
  4. **微调量化模型（Step 2，100 epochs）**：对每个滤波后样本x_i^F，计算三项损失：
     - KL散度：KL(q(x_i^F;θ) || q(x_i^F;θ^q))（知识蒸馏，始终应用）
     - CAM对齐损失：L_CAM = ||S^θ(x_i^F) - S^θ^q(x_i^F)||_F²，其中S^θ为Grad-CAM生成的显著性图（Idea 2）
     - 交叉熵损失：仅当δ(x_i^F;θ) ≤ τ时（τ搜索范围{0.5,0.55,0.6,0.65,0.7}）施加λ_CE·CE(q(x_i^F;θ^q), y_i)（Idea 3）
     总损失：L_SYNQ = KL + 1_{δ≤τ}·λ_CE·CE + λ_CAM·L_CAM
  5. **优化器**：SGD with momentum=0.9, weight decay=1e-4。Batch size：CIFAR-10/100为256，ImageNet为16。初始学习率在{1e-4,1e-5,1e-6}中搜索，每epoch衰减0.1。
  
  Grad-CAM显著性图计算：S^θ(x_i) = ReLU(Σ_k α_k·A^{k;θ}(x_i))，其中α_k = (1/(W_k H_k)) Σ_{w,h} ∂y^{y_i}/∂A^{k;θ}_{wh}(x_i)为第k通道激活A^{k;θ}对真实类别预测分数的平均梯度权重。

## SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出SliM-LLM，一个基于salience驱动的group-wise混合精度PTQ框架。核心包含：(1) **Salience-Determined Bit Allocation (SBA)**：基于group内平均salience排序，通过双指针搜索最小化输出KL散度来优化bit-width分配。(2) **Salience-Weighted Quantizer Calibration (SQC)**：通过引入calibration参数τ，在三倍标准差规则筛选的salient权重子集上优化加权量化误差，增强对局部重要权重的感知。SliM-LLM以GPTQ为backbone，SliM-LLM⁺以OmniQuant为backbone（仅用SBA，保留learnable weight clipping替代SQC）。实验比较2/3/4-bit weight-only量化下WikiText2和C4 perplexity（per-group size=128），zero-shot任务（PIQA, ARC-e, ARC-c, BoolQ, HellaSwag, Winogrande），以及MMLU、MathQA。对比方法包括RTN、GPTQ、AWQ、QuIP、PB-LLM、OmniQuant、AffineQuant、APTX、LLM-MQ。

- 硬件平台是什么，配置是什么。
  量化在单张NVIDIA A800-80GB GPU上完成（SliM-LLM无梯度，SliM-LLM⁺使用AdamW优化器）。部署测试同样在A800上使用修改版AutoGPTQ进行。量化框架基于GPTQ (Frantar et al., 2022) 和 OmniQuant (Shao et al., 2023)，PyTorch实现。

- 模型是什么。数据集和bench分别是什么。
  模型：OPT (1.3B, 2.7B, 6.7B, 13B, 30B, 66B)，LLaMA-1 (7B, 13B, 30B, 65B)，LLaMA-2 (7B, 13B, 70B)，LLaMA-3 (8B, 70B)，Gemma2-9B，Mixtral 8×7B，Vicuna-13B（对话评估），LLaVA-Next-8B（多模态评估）。校准数据集：从WikiText2随机选取128个样本，每个2048 tokens。评估数据集：WikiText2、C4（perplexity）；PIQA、ARC-e/ARC-c、BoolQ、HellaSwag、Winogrande（zero-shot）；MMLU、MathQA；AI2D、ChartQA、DocVQA、MMBench（VLM评估）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/Aaronhuang-778/SliM-LLM
  
  算法Pipeline（以LLaMA-7B 2-bit量化为例）：
  1. **校准数据采集**：从WikiText2选取128个2048-token样本，前向传播收集每层输入激活x_F。
  2. **Hessian计算**（逐层）：H = (1/P) Σ x_F^[k] x_F^[k]^T，计算Cholesky分解 H^in = Cholesky((H + λI)^(-1))。
  
  3. **SBA (Salience-Determined Bit Allocation)**，对每层权重W ∈ R^{n×m}（group_size=128，共k=m/128个group）：
     ```
     # 计算每个group的平均salience
     for each group g_i (i=0..k-1):
         S[i] = mean(W_g^2 / [diag(H^in)]_g^2)
     # 按salience排序groups
     sort groups by S descending
     # 双指针搜索最优混合精度比例
     for p = 1 to ceil(k/2):
         将p个最低salience的group量化为1-bit，p个最高salience量化为3-bit，其余2-bit
         计算KL_div(xW^T || xŴ_q^T)
         选择KL_div最小的p*作为最优配置
     ```
     约束条件: |G_{N-1}| = |G_{N+1}|（即1-bit和3-bit group数量相等，维持2-bit平均位宽）
  
  4. **SQC (Salience-Weighted Quantizer Calibration)**，对每个group:
     ```
     # 用3-σ规则筛选salient元素
     w_s = {w | w < μ-3σ 或 w > μ+3σ}  # 约占group内1%元素
     w_us = 其余元素
     # 在[1-λ, 1+λ]内搜索最优τ（λ=0.1, n=50 candidates）
     for τ in linearly spaced [0.9, 1.1] (50 steps):
         Δ = τ(w_max - w_min) / (2^b - 1)
         z = -⌊τ w_min / Δ⌋
         ŵ = fakequant(W, b, Δ, z)
         loss = ||w_s - ŵ_s||₂² + ||w_us - ŵ_us||₂²
     选择最小化loss的τ*, Δ*, z*
     ```
  
  5. **GPTQ Error Compensation**（逐列）：
     ŵ_q^b = fakequant(W_{:,b:b+β}, g_b, Δ*, z*)
     E = (W_{:,b:b+β} - ŵ_q^b) / diag(H^in_{b:b+β,b:b+β})
     W_{:,b+β:} = W_{:,b+β:} - E · H^in_{b:b+β,b+β:}

  6. **SliM-LLM⁺变体**：SBA保持不变，量化器部分用OmniQuant的Learnable Weight Clipping (LWC)和Learnable Equivalent Transformation (LET)替代SQC，使用AdamW优化器进行梯度优化。

## I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Shift-Uniform-Log2 Quantizer (SULQ) 解决log2量化器对post-Softmax激活的量化低效问题，以及三阶段Smooth Optimization Strategy (SOS) 解决不同量化粒度下loss landscape的粗糙和放大问题。实验在ImageNet上比较ViT-S/B、DeiT-T/S/B、Swin-S/B在W3A3/W4A4/W6A6下的Top-1准确率，在COCO上比较Mask R-CNN和Cascade Mask R-CNN（Swin-T/S为主干）的W4A4检测/分割AP。对比方法包括PTQ4ViT、BRECQ、QDrop、PD-Quant、RepQ-ViT、FQ-ViT、APQ-ViT、Ranking-ViT、EasyQuant、NoisyQuant、Bit-shrinking等。

- 硬件平台是什么，配置是什么。
  单张NVIDIA 3090 GPU。框架为PyTorch，预训练模型来自Timm库。训练时间约31分钟（DeiT 3-bit）。

- 模型是什么。数据集和bench分别是什么。
  模型：ViT-S、ViT-B、DeiT-T、DeiT-S、DeiT-B、Swin-S、Swin-B（ImageNet分类）；Swin-T、Swin-S作为Mask R-CNN和Cascade Mask R-CNN的主干（COCO检测/分割）。数据集：ImageNet（分类），COCO 2014（检测+分割）。从数据集中随机选取1024张图片作为校准集。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/zysxmu/IaS-ViT
  
  算法Pipeline：
  1. **量化器配置**：对所有权重和矩阵乘法的输入做均匀量化（channel-wise权重，layer-wise激活），post-Softmax激活用SULQ，LayerNorm和Softmax保持全精度。
  2. **SULQ量化器**（替换标准log2量化器）：
     ```
     输入: X (post-Softmax激活), bit-width b, shift η
     X_q = clamp(round((-log2(X + η) - min_val) / s), 0, 2^b - 1)
     反量化: X_deq = 2^{-round(s * (X_q - z))} - η
     ```
     SULQ通过添加shift bias η后接log2变换再均匀量化，使量化区间完整覆盖输入域，解决标准log2量化器的"quantization inefficiency"（大量值被clamp到远端）。
  3. **SOS三阶段优化**（Block-wise reconstruction objective L_l = ||X_l - X̄_l||_2）：
     - Stage 1：全精度权重 + post-LayerNorm激活用channel-wise量化，其他激活用layer-wise量化 → 在平滑低loss的landscape下优化
     - Stage 2：通过scale reparameterization将channel-wise量化器无缝转换为layer-wise：调整LayerNorm的affine参数 β̃=(β+s⊙r₂)/r₁, γ̃=γ/r₁ 以及下一层权重 W̃_{:,j}=r₁⊙W_{:,j}, b̃_j=b_j-(s⊙r₂)W_{:,j}
     - Stage 3：量化所有权重，在量化激活+量化权重下再微调恢复性能
  4. **训练超参**：Adam优化器，权重lr=4e-5（cosine衰减），weight decay=0，量化参数校准后固定不优化。ImageNet batch_size=64，6-bit用200 iterations，其他用1000 iterations。η通过grid search选取最小化量化误差的值。

## Scheduling Weight Transitions for Quantization-Aware Training

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出了一种 Transition Rate（TR）调度技术，替代 QAT 中传统的学习率（LR）调度。核心实现是：在每个训练迭代中计算量化权重的 TR（发生离散级别变化的权重占比），用指数移动平均估计 running TR，然后通过 Transition-Adaptive Learning Rate（TALR）自适应地调整潜权重的更新步长，使得 running TR 匹配目标 TR。实验比较：（1）plain optimizer（SGD/Adam/AdamW，使用传统 LR 调度）vs 论文方法的 variants（SGDT/AdamT/AdamWT，使用 TR 调度），在 ImageNet、CIFAR-10/100、MS COCO 上的分类/检测精度对比；（2）不同类型调度器（step decay vs cosine annealing）下 plain vs TR 调度器的鲁棒性对比；（3）不同优化器（SGD, Adam, NAdam, Adamax, AdamW, RMSProp, Adagrad）下 TR 调度的泛化能力。

- 硬件平台是什么，配置是什么。
  4 × NVIDIA A5000 GPU（ImageNet 训练用时测量，Table S7）。CIFAR 实验使用论文未具体指明 GPU 型号的训练平台。

- 模型是什么。数据集和bench分别是什么。
  模型：MobileNetV2、ResNet-18/20/34/50、ReActNet-18（binary quantization specialized architecture）、DeiT-T/S（ViT-based）。数据集与 Benchmark：ImageNet（ILSVRC2012，top-1 validation accuracy）、CIFAR-10/100（top-1 test accuracy）、MS COCO 2017（RetinaNet 目标检测，AP/AP50/AP75/APS/APM/APL）。量化位宽覆盖：W1A1（binary）、W2A2、W3A3、W4A4。所有模型从 pretrained full-precision 权重初始化，第一层和最后一层不量化。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://cvlab.yonsei.ac.kr/projects/TRS/

  **算法 Pipeline：TR 调度 QAT 的每一步迭代（来自 Algorithm 1）：**

  Step 1 — 正向传播：潜权重 w 经 normalizer `w_n = clip(γ·w/s, α, β)` 归一化，再经 discretizer `w_d = round(w_n)` 转为离散整数值，最后经 fixed de-normalizer `w_q = w_d/γ` 输出量化权重（γ、α、β 为位宽常量，s 为可学习的 scale 参数，TR 调度时 weight quantizer 的 s 固定不变）。

  Step 2 — 计算当前 TR `k^t = Σᵢ I[w_d^t(i) ≠ w_d^{t-1}(i)] / N`，即发生离散级别变化的量化权重占总权重的比例。

  Step 3 — 估计 running TR `K^t = m·K^{t-1} + (1-m)·k^t`，使用 momentum m=0.99。

  Step 4 — 调整 TALR `U^t = max(0, U^{t-1} + η(R^t - K^t))`，其中 R^t 是目标 TR（由 scheduler 如 cosine decay 衰减），η = U^0（初始 TALR 值）。

  Step 5 — 逆/反向传播：gradient term g^t 用 STE 通过 discretizer 回传到潜权重，g^t 取决于优化器类型（SGD 用一阶矩，Adam 用动量归一化梯度）。

  Step 6 — 更新潜权重 `w^{t+1} = w^t - U^t·g^t`。注意这里用的是 TALR U^t 而非固定 LR。

  关键设计：初始 target TR = λ·√b_w，其中 λ 是 TR factor（超参，如 5e-3），b_w 是权重量化位宽；target TR 按 cosine scheduler 衰减到零；η 等于初始 TALR，使调整步长与初始值成比例。量化器基于修改版 LSQ（multi-bit）和 ReActNet（binary）。

## SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  SLiM 是一个 one-shot 压缩框架，将硬件友好的量化（SLiM-Quant）、半结构化稀疏（Wanda 2:4 sparsity）和基于显著性的低秩近似（SLiM-LoRA）整合为统一 pipeline。实验比较：（1）zero-shot 下游任务平均准确率（MMLU、Piqa、Arc-Easy、Arc-Challenge、WinoGrande、OpenBookQA）；（2）WikiText2 语言建模困惑度；（3）NVIDIA RTX 3060 和 A100 GPU 上的逐层推理加速比；（4）端到端内存缩减比；（5）floating-point operation（FLOP）缩减比。Baseline 包括：SparseGPT+OPTQ、Wanda+Group AbsMax/AWQ/OmniQuant/AffineQuant、JSQ、L²QER、Magnitude Pruning、MaskLLM。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 3060（consumer GPU，加速比实验）、NVIDIA A100-40GB（data center GPU，加速比和主要实验）、NVIDIA H100（微调开销实验，单卡）。所有实验运行于 University of Toronto。

- 模型是什么。数据集和bench分别是什么。
  模型：OPT 家族（125M, 350M, 1.3B, 2.7B, 6.7B, 13B）、LLaMA-2（7B, 13B）、LLaMA-3.1-405B（仅加速比）。数据集与 Benchmark：zero-shot 下游任务（MMLU, Piqa, Arc-Easy, Arc-Challenge, WinoGrande, OpenBookQA）使用 Language Model Evaluation Harness（lm-eval-harness）；WikiText2 用于困惑度评估；C4 数据集（128 条序列用于校准，300,000 tokens 用于可选微调）；SlimPajama 作为备选校准数据集。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Mohammad-Mozaffari/slim

  **SLiM 三阶段 compression pipeline（逐层执行）：**

  **阶段一：SLiM-Quant（概率量化）**
  输入：权重矩阵 W ∈ R^{d_in × d_out}，量化位宽 q（通常 q=4）
  1. 构建权重绝对值直方图 f_abs（bin 数 = max(512, min(d_in*d_out/1000, 20000))）
  2. 多网格搜索最优 scaling factor α：
     - 低分辨率网格：在 [0, max(|W|)] 范围内取 10 个均匀样本，计算每个 α 的 E(α) = E_quant(α) + E_clip(α)
       - E_quant(α) = ∫_0^α f_abs(x) |α × round(x/α) × 2^{1-q} - x|² dx
       - E_clip(α) = ∫_α^∞ f_abs(x) |α - x|² dx
     - 在最低误差 α_low 附近高分辨率细化搜索
  3. 最优 α* = argmin_α E(α)
  4. W^Q = round(clip(W/α*)) × 2^{q-1}
  可选 SLiM-Quant^O（activation-aware）：对 1% 最高显著性的 channel（saliency = |diag(x_mean) × W|），scale up 权重 × s，scale down 对应激活 ÷ s，降低输出误差。

  **阶段二：Sparsification（剪枝）**
  使用 Wanda 在量化权重 W^Q 上施加 2:4 半结构化稀疏或 50% 非结构化稀疏：
  对于每行 weight w_i ∈ R^{d_out}，重要性 score_ij = |w_ij| × ||x_j||_2，保留 score 最高的 50%（2:4 模式：每 4 个连续元素保留 2 个）。

  **阶段三：SLiM-LoRA（显著性低秩适配）**
  1. 计算压缩误差 E_C = W^C - W（其中 W^C = W + E_Q + E_S，E_Q 为量化误差，E_S 为稀疏误差）
  2. 构建 saliency 函数 F(W) = diag(x)W，其中 x ∈ R^{d_in} 为校准集输入的平均绝对值（+ min(|x|) 避免零元素）
  3. 计算误差显著性 S_C = diag(x) × E_C
  4. SVD 分解：S_C = U Σ V^T，取 rank r = 0.1 × d 得到 L̃ = U_r Σ_r^{1/2}, R̃ = Σ_r^{1/2} V_r^T
  5. 逆 saliency 变换：L = diag(1/x) × L̃，R = R̃
  最终近似：W ≈ W^C + LR

  **阶段四（可选）：低秩适配器量化 + PEFT 微调**
  - 对 LR 适配器使用 AbsMax group quantization（group size=128，4-bit）压缩至 4×
  - 冻结量化稀疏权重，仅微调低秩适配器（300K tokens C4，batch size 64，seq len 1024）
  - 量化适配器微调使用 STE（straight-through estimator）+ Triton 自定义量化/反量化 kernel
  - 优化器：AdaFactor + 线性 LR schedule，BF16 精度

  **推理加速路线：**
  - 量化稀疏矩阵乘法使用 Sparse Marlin kernel（集成 vLLM）
  - 低秩适配器使用 Dense Quantized Marlin（量化时）或 PyTorch kernel（全精度时）
  - 小 batch size decode 模式
  - 加速比：RTX 3060 上最高 4.3×，A100 上最高 3.8×（逐层测量）
  - 内存缩减：SLiM^Q 达 0.23×（vs dense），SLiM 达 0.33×（含全精度适配器）

## SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出两种新颖的通信量化技术来压缩 ShardedDP 中的权值和梯度通信——(1) **Quantization on Weight Differences (qWD)**：不直接量化权值，而是对当前迭代与前次迭代间的权值差值做 INT4 量化，利用差值分布更均匀且范围更小的特性降低量化误差；(2) **Two-Level Gradient Smooth Quantization (TLq-HS)**：对 intra-node 梯度 all-to-all 通信使用 INT8 量化（降低误差），对 inter-node all-to-all 通信使用 INT4 量化（大幅压缩带宽），并在量化前施加 Hadamard Transform 平滑梯度中的 outlier。
  - 实验比较：Baseline（BF16/FP32 混合精度 Megatron-LM 全精度训练）vs ZeRO++ 类似策略（直接 4-bit 量化权值 qW + 两级均 4-bit 量化梯度 ULq）vs qWD 单独 vs TLq 单独 vs TLq-HS 单独 vs SDP4Bit（qWD + TLq-HS 组合），测量 validation loss（准确率）和 E2E TFLOPs throughput（加速比）。

- 硬件平台是什么，配置是什么。
  - **平台1**：16 节点，每节点 4× NVIDIA A100-SXM4-40GB，100 Gbps Slingshot10 互联（低带宽 inter-node）。
  - **平台2**：16 节点，每节点 8× NVIDIA H800-SXM5-80GB，8 条 InfiniBand 链路共 3.2 Tbps（高带宽 inter-node）。
  - 最大规模 128 GPUs。

- 模型是什么。数据集和bench分别是什么。
  - 模型：GPT 系列（125M, 350M, 1.3B, 2.7B, 6.7B, 13B, 18B 参数），配置详见 Table 7（hidden size 768→6144, layers 12→40）。
  - 数据集：The Pile（800GB），每轮 80,000 iterations（处理超 40B tokens），验证 loss 使用 The Pile 验证集。
  - Benchmark：E2E validation loss（准确率指标）和 E2E TFLOPs throughput（加速比指标），通过 Megatron-LM 内置 loss logging 和 throughput timer 收集。Wall-clock time vs. loss 曲线也作为综合指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源仓库：https://github.com/hanlin-lu/SDP4Bit（Apache-2.0），基于 Megatron-LM 实现，亦有 ByteDance-Seed 官方 fork：https://github.com/ByteDance-Seed/SDP4Bit。
  - 算法pipeline详细说明：

  **1) Quantization on Weight Differences (qWD)**：
  每轮迭代中，每个 GPU 持有完整模型权重 `w_model`（BF16）和分片的 main weights `w_main[p]`（FP32，用于优化器状态）。优化器更新 main weights 后：
  ```
  # 计算权值差值（FP32 -> BF16）
  d[p] = w_main[p] - w_model[p]

  # INT4 对称线性量化（group-wise, group_size=2048）
  for each group of 2048 elements in d[p]:
      s = max(abs(group))           # scale factor
      d_q = round(clip(group, -s, s) / s * 7)  # map to {-7, ..., +7}
      store(s, d_q)

  # AllGather 量化后的差值（通信量 ≈ 4 bit/elem）
  d_q_global = AllGather(d_q[p])

  # 反量化并更新模型权值
  for each group:
      d_deq = d_q_global * s / 7
      w_model = w_model + d_deq
  ```
  关键优势：(a) 权值差值的数值范围比权值本身小得多（`||δw|| < ||w||`），INT4 量化误差更小——论文通过直方图（Fig. 4）展示差值分布更均匀且范围更窄；(b) 理论保证：weight difference 量化兼容 biased compressor（如 top-k sparsifier），而直接量化权值时使用 biased compressor 会导致收敛失败（Counterexample 4.1 证明 ternary quantizer 直接量化权重时 SGD 卡在初始值不动）。

  **2) Two-Level Gradient Smooth Quantization (TLq-HS)**：
  梯度同步采用两次 all-to-all 替代传统的 reduce-scatter（沿用 ZeRO++ 的通信模式），但使用两级精度 + Hadamard 平滑：
  ```
  # Step 1: Hadamard Transform 平滑 outlier（32x32 矩阵在线旋转）
  g_hat = H @ grad @ H.T   # H 是 32x32 Walsh-Hadamard matrix

  # Step 2: INT8 量化 → Intra-node AlltoAll → 反量化 → 局部 reduce
  qg_8bit = round(clip(g_hat, -s8, s8) / s8 * 127)  # INT8, group_size=512
  list_qg8 = IntraAlltoAll(qg_8bit)   # 仅节点内通信（NVLink/NVSwitch 高带宽）
  # 反量化后做 local reduce（省略 Hadamard 逆向，因 H·H=I 自动抵消）
  g_local_reduced = sum([dequantize(x) for x in list_qg8])

  # Step 3: Hadamard → INT4 量化 → Inter-node AlltoAll → 反量化 → 最终 reduce
  g_hat_reduced = H @ g_local_reduced @ H.T
  qg_4bit = round(clip(g_hat_reduced, -s4, s4) / s4 * 7)  # INT4, group_size=128
  list_qg4 = InterAlltoAll(qg_4bit)   # 跨节点通信（InfiniBand/Slingshot 低带宽）
  g_reduced = sum([dequantize(x) for x in list_qg4])
  g_final = H @ g_reduced @ H.T        # 最终逆变换恢复原始梯度
  ```
  优化技巧（Section 3.3）：
  - 利用 `H·H=I` 在 Step 2 省略 Hadamard 逆向（intra-node dequant 后无需再 transform）
  - 利用 `Σ H·g_i = H·Σ g_i` 将 inter-node dequant 后的 Hadamard 移到最终 reduction 之后，将 transform 次数从 3 降低到 2
  - 将 Hadamard + quantization/dequantization 融合为单个 CUDA kernel

  **3) 训练运行时优化（Section 3.3）**：
  - **Buffer reuse**：Megatron-LM 维持完整 model weights，无需额外 buffer 存储历史权重用于差值计算
  - **Hadamard kernel fusion**：Hadamard transform 与 (de)quantization 融合为单个 CUDA kernel，利用 shared memory 局部性将 overhead 降低到近乎零

  **训练命令示例（GPT-1.3B on 32 A100）**：
  ```bash
  python pretrain_gpt.py \
    --num-layers 24 --hidden-size 2048 --num-attention-heads 16 \
    --seq-length 2048 --micro-batch-size 2 --global-batch-size 256 \
    --train-iters 80000 --lr 2e-4 --min-lr 2e-5 \
    --lr-decay-style cosine --lr-warmup-iters 2000 \
    --optimizer adam --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 \
    --fp16 --use-distributed-optimizer \
    --quantized-weights --weight-quantization-bits 4 --wq-group-size 2048 \
    --quantized-gradients --gradient-quantization-bits-intra 8 --gq-group-size-intra 512 \
    --gradient-quantization-bits-inter 4 --gq-group-size-inter 128 \
    --hadamard-transform --gradient-alltoall-pipeline 4 \
    --no-async-tensor-model-parallel-allreduce
  ```

  **核心结果**：
  - GPT-6.7B validation loss 与全精度 baseline 几乎重合（Fig. 1），最大 loss 增加仅 0.24%
  - GPT-18B on 128 H800: 4.08× E2E throughput speedup（59.2 vs 14.5 TFLOPs）
  - 低带宽网络下加速比更大：6.7B on A100 Slingshot 达 3.44×（37.1 vs 10.8 TFLOPs）
  - 收敛保证：Theorem 4.1 证明达到与标准 SGD 相同的 O(1/√T) 收敛率，且放宽了 QSDP 对 Polyak-Łojasiewicz 条件和对特定 quantizer 的依赖

## RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：RoSTE 是一种量化感知监督微调（QA-SFT）算法，将 4-bit 权重量化、激活量化和 KV cache 量化与监督微调结合在单一训练阶段。核心算法 pipeline：(1) **Bilevel Optimization Formulation**：上层子问题通过 STE (Straight-Through Estimator) 优化量化后的权重矩阵以最小化 SFT loss；下层子问题通过最小化 weight-activation quantization error surrogate loss 来选择最优旋转矩阵。(2) **Adaptive Rotation Strategy**：对每一层，在 identity 矩阵 I（无旋转）和 random Walsh-Hadamard 矩阵 H 之间做离散搜索，选择使量化误差更低的选项。旋转矩阵 R 作用于线性层：`LIN_i(X; W_i, R_i) = σ(Q_x(X R_i) Q_w(R_i^T W_i))`，利用正交变换 R_i R_i^T = I 保持计算不变性。旋转分为可离线吸收的 between-block rotation R1/R2 和在线旋转 R3/R4。(3) **RoSTE 训练循环（Algorithm 1）**：外层 K 次迭代（论文设 K=1）交替执行 rotation configuration search（逐层比较 I vs H 的量化误差 E(12)）和内层 T 步 QAT via STE（`w^{t+1} = w^t - η g_ste^t`，其中 `g_ste^t = (⟨Q_x(Rx_t) | Q_w(Rw^t)⟩ - y_t) R^T Q_x(Rx_t)`）。量化方案：非对称均匀量化（asymmetric uniform quantizer），per-token activation quantization + per-channel weight quantization。旋转矩阵使用 fast Hadamard CUDA kernel 实现高效在线旋转。
  - 实验比较：(a) **Exp.1**：Pythia 1B/6.9B 和 Qwen2.5 0.5B/7B 在 Reddit TL;DR Summarization 任务上对比 RoSTE vs PTQ baselines（RTN, GPTQ, QuaRot, SpinQuant on fine-tuned models）和 QAT baseline（STE without rotation），W4A4KV4 及 W4A8KV4 配置，评价指标 ROUGE-1/2/L/LSum；(b) **Exp.2**：Llama 3.1 8B 在 Tulu 3 SFT mixture 上训练，6 个下游任务评估（TruthfulQA, MMLU-Pro, BigBenchHard, AGIEval, GSM8K, MATH），W4A4KV4 及 W4A8KV4 配置；(c) 消融实验：旋转策略对比（No Rotation / Complete Rotation / Adaptive Rotation (RoSTE)）；(d) 理论验证：量化误差随训练步数的变化轨迹（Fig. 4）、激活 outlier 分布可视化（Fig. 3, 6, 7）；(e) 与 QLoRA、LLM-QAT、DuQuant 的额外比较（Table 7-9）；(f) 训练开销统计（Table 10：training time + peak GPU memory）。

- 硬件平台是什么，配置是什么。
  - 8× NVIDIA A100 GPUs 集群。CUDA 环境论文未详细说明版本号。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Pythia 1B/6.9B (Biderman et al., 2023), Qwen2.5 0.5B/7B (Yang et al., 2024), Llama 3.1 8B (Dubey et al., 2024)。
  - Exp.1 数据集：Reddit TL;DR Summarization dataset (Huang et al., 2024)，训练集 117k 样本，评估用 TL;DR test dataset，指标 ROUGE-1/2/L/LSum (Lin, 2004)。
  - Exp.2 数据集：Tulu 3 SFT mixture dataset (Lambert et al., 2024)，训练集 100k 样本。评估使用 EleutherAI LM Evaluation Harness (Gao et al., 2021)，benchmarks：TruthfulQA (6-shot, Acc mc1), MMLU-Pro (0-shot, EM), BigBenchHard (3-shot, EM), AGIEval (0-shot, Acc), GSM8K (8-shot, EM), MATH (4-shot, EM)。
  - 量化配置：权重 W4/W4（4-bit），激活 A4/A8（4/8-bit），KV cache KV4/KV8（4/8-bit），asymmetric uniform quantizer，per-token activation + per-channel weight 量化组，clipping factor ∈ {1, 0.95, 0.9}。
  - 校准：从 fine-tuning dataset 抽取 n=128 样本计算量化误差 E(12)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/OptimAI-Lab/RoSTE
  - **算法伪代码（对应 Algorithm 1 + 全文实现细节）**：
    ```
    输入: 预训练权重 {W_i^pt}_{i=0}^{ℓ-1}, 学习率 η, 校准样本 D_cal (n=128), SFT 数据集 D_sft
    输出: 量化微调模型 m_Q(·; W^{KT}, R^{K-1})

    // Step 1: 修改 normalization layer（吸收 mean subtraction + scale/shift 到相邻权重矩阵）
    for each normalization layer:
      if LayerNorm: absorb mean subtraction into prev weight, absorb scale/bias into next weight
      if RMSNorm: absorb scale into next weight

    // Step 2: 初始化
    W^0 = {W_i^pt}_{i=0}^{ℓ-1}

    // Step 3: RoSTE 外层循环 (论文设 K=1)
    for k = 0, ..., K-1:
      // -- Lower level: Rotation Configuration --
      计算 E_all_I = E(W^{kT}, {I}_{i=0}^{ℓ-1})   // 全部无旋转的量化误差
      计算 E_all_H = E(W^{kT}, {H}_{i=0}^{ℓ-1})   // 全部旋转的量化误差
      for each layer/module i = 0, ..., ℓ-1:
        比较 layer-wise quantization error:
          若 E_i(I) < E_i(H): R_i^k = I  (no rotation)
          否则: R_i^k = H  (random Walsh-Hadamard rotation)
          其中 H = H_diag · Diag(s), H_diag ∈ R^{d×d} 为 Walsh-Hadamard 矩阵, s ∈ {-1,1}^d 随机 sign vector

      // -- Upper level: QAT Stage via STE --
      for t = 0, ..., T-1:
        采样 mini-batch ξ ⊆ D_sft
        对每个 linear layer i (forward pass):
          X_in = input activation
          // 在线旋转（若 R_i 未被 merge 进权重）
          X_rot = X_in · R_i                  // 对非 mergeable rotation (R_3, R_4)
          X_q = Q_x(X_rot)                    // per-token asymmetric 量化
          W_rot = R_i^T · W_i                 // 若 R_i mergeable，在训练前已完成
          W_q = Q_w(W_rot)                    // per-channel asymmetric 量化
          output = σ(X_q · W_q)               // INT4 matmul via fast Hadamard kernel

        计算 SFT loss L = -E_i[Σ_t log P(y_{i,t} | x_i, y_{i,<t}; m_Q)]

        // Backward via STE
        梯度近似：∂Q_w(R_i^T W_i) / ∂W_i ≈ R_i  (STE: 量化器当作恒等)
        更新：W^{kT+t+1} = W^{kT+t} - η ∇_W L_SFT(m_Q(·; W, R^k); ξ)

      // 旋转矩阵吸收（merge offline rotations into weights for inference）
      merge R_1, R_1^T, R_2, R_2^T, R_4^T into corresponding weight matrices
      keep R_3, R_3^T, R_4 as online fast Hadamard kernel rotations
    ```
  - **量化误差计算 E(12)**（用于 rotation selection）：
    ```
    E({W_i}, {R_i}) = Σ_{i=0}^{ℓ-1} ||Q_w(R_i^T W_i) - R_i^T W_i||^2
                    + (1/n) Σ_{i=0}^{ℓ-1} Σ_{j=0}^{n-1} ||Q_x(X_{i,j} R_i) - X_{i,j} R_i||^2
    ```
    对 n=128 个校准样本计算 weight quantization error + activation quantization error，逐层逐 sample 求和。
  - **关键旋转位置**（Fig. 5）：R1（between-block, offline mergeable）作用于 Q/K/V projection、Up/Gate projection、O projection、Down projection、embedding、lm_head；R2（in-block, offline mergeable）作用于 Value projection 和 O projection（MHSA 内）；R3（in-block, online）作用于 Query 和 Key（消除 KV cache outliers）；R4（in-block, online）作用于 Down projection（MLP 内）。

## QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuantSparse 是一个统一的后训练量化（PTQ）+ 注意力稀疏化框架，针对 Video Diffusion Transformer（如 HunyuanVideo-13B、Wan2.1-1.3B/14B）。核心算法 pipeline 包含两大组件：(1) **Multi-Scale Salient Attention Distillation (MSAD)**：在校准阶段，通过 Global Guidance（对 Q/K 做 average pooling stride=s 下采样后计算低分辨率 attention，MSE 蒸馏 FP 与量化模型的 attention map）和 Local Guidance（根据 token saliency 分布选择 top-k salient queries 做高分辨率 attention 蒸馏）两种互补机制，高效对齐量化前后的 attention 分布（内存从 O(L²) 降至 O(L̃²+kL)）。(2) **Second-Order Sparse Attention Reparameterization (SSAR)**：在推理阶段，利用量化噪声在扩散过程中呈缓变随机过程的特性，发现二阶残差 Δ̃_quant^(t)=Δ_quant^(t)−Δ_quant^(t-1) 在不同时间步间具有显著高于一阶残差的时间稳定性；缓存参考时间步的一阶+二阶残差，在稀疏 attention 输出上叠加缓存残差近似全 attention；进一步对二阶残差做 SVD 投影到 top-r 主成分上抑制时间方差。整体流程：校准阶段 block-wise PTQ → MSAD 优化量化参数（minimize L_quant + λ_global L_global + λ_local L_local）→ 推理阶段量化稀疏 attention + SSAR 残差校正（cache interval=5）。量化粒度：channel-wise weight + dynamic token-wise activation，uniform symmetric quantization。稀疏 mask 使用 SVG（SparseVideoGen）的静态 spatial-temporal pattern。
  - 实验比较：(a) QuantSparse vs 纯量化 baseline（PTQ4DiT, Q-DiT, SmoothQuant, QuaRot, ViDiT-Q, Q-VDiT）在 W6A6/W4A8 下；(b) QuantSparse vs 纯量化+稀疏化 naive 组合（QuaRot+DFT/Jenga/SVG, Q-VDiT+DFT/Jenga/SVG）在多种 density 下；(c) 多指标评估：CLIPSIM, VQA, FlowScore（绝对质量）+ PSNR, SSIM, LPIPS（与 FP16 的差异）；(d) VBench benchmark 8 维度评估（IQ, AQ, MS, DD, BC, SuC, ScC, OC）；(e) 消融：MSAD（global/local 各自贡献）、SSAR（无/first-order/second-order/SVD 各级贡献）、cache interval、attention density、pooling stride s、salient token k、λ 权重、SVD rank r、full attention distillation vs MSAD 效率对比；(f) 效率分析：模型存储、显存消耗、DiT 推理时间、端到端加速比；(g) 与其他加速技术组合：SageAttention + TeaCache；(h) 校准资源开销。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA A800 80GB GPU，CUDA 12.4。INT 矩阵乘法使用 CUTLASS 基于 PyTorch 实现。校准和推理均在单 A800 上完成。

- 模型是什么。数据集和bench分别是什么。
  - 模型：HunyuanVideo-13B (Kong et al. 2024)、Wan2.1-1.3B (Wan et al. 2025)、Wan2.1-14B (Wan et al. 2025)。图像生成扩展：Hunyuan-DiT 1.5B (Li et al. 2024c)。采样步数 50。
  - Calibration：20 个随机生成样本，每个 transformer block 训练 15 epoch。
  - 评估数据：OpenSORA prompt sets（与 Zhao et al. 2024, Feng et al. 2025c 相同）。
  - Benchmarks：CLIPSIM (Wu et al. 2021), VQA (Wu et al. 2023), FlowScore (Liu et al. 2024b), PSNR/SSIM/LPIPS (Zhang et al. 2018), VBench 8 维度 (Huang et al. 2024b)。
  - Baseline 量化方法：PTQ4DiT (Wu et al. 2024), Q-DiT (Chen et al. 2024), ViDiT-Q (Zhao et al. 2024), Q-VDiT (Feng et al. 2025c), SmoothQuant (Xiao et al. 2023a), QuaRot (Ashkboos et al. 2024)。
  - Baseline 稀疏化方法：DiTFastAttn/DFT (Yuan et al. 2024, cache-based), Jenga (Zhang et al. 2025d, dynamic pattern), SparseVideoGen/SVG (Xi et al. 2025, static pattern)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/wlfeng0509/QuantSparse（MIT License，代码尚未发布，README 声明"will be released soon"）
  - 算法 pipeline 张量计算流程（以 HunyuanVideo-13B W4A8 + 15% attention density 为例）：

  **阶段一：校准（Calibration）**
  1. 从预训练 FP16 video DiT 加载权重。初始化量化参数 {s, z}：channel-wise scale for weights, token-wise dynamic scale for activations。
  2. 采样 20 个 calibration samples，block-wise 逐 block 优化：
     a. FP 前向：X ∈ R^(L×d_in) → Q_fp = X W_q^T, K_fp = X W_k^T, V_fp = X W_v^T（FP16）。
     b. 计算 FP attention: A_fp = softmax(Q_fp K_fp^T / √d_k) ∈ R^(h,L,L)。
     c. 计算 token saliency: s_j = Σ_h Σ_i A_fp[h,i,j]，选 top-k=256 salient queries I。
     d. 量化前向：Q_quant = Q(X) Q(W_q)^T, K_quant = Q(X) Q(W_k)^T, V_quant = Q(X) Q(W_v)^T。其中 Q(·): X_Q = clip(⌊X/s⌋+z, 0, 2^b-1), dequant = s·(X_Q−z)。
     e. Global Guidance: Q̃ = AvgPool(Q_quant, s=128), K̃ = AvgPool(K_quant, s=128) → A_global = softmax(Q̃ K̃^T / √d_k) → L_global = MSE(A_global^FP, A_global^quant)。
     f. Local Guidance: A_local = softmax(Q_quant[I,:] K_quant^T / √d_k) → L_local = MSE(A_local^FP, A_local^quant)。
     g. Total loss: L_distill = L_quant + λ_global·L_global + λ_local·L_local（Wan2.1: λ=1e-4, HunyuanVideo: λ_global=1.0, λ_local=1e2）。
     h. AdamW 优化 s, z（scale LR=5e-2, channel-wise scale/rotation matrix LR=5e-3, cosine scheduler）。
  3. 吸收量化参数到权重中，得到量化模型 M_quant。

  **阶段二：推理（Inference）**
  1. 加载 M_quant，输入 prompt P，初始化缓存 C = {Δ_quant^(t_ref), Δ̃_quant^(t_ref)}。
  2. for t = 0 to T-1（T=50 去噪步）:
     a. 量化稀疏 attention：Q_quant = Q(X_t) Q(W_q)^T, K_quant = Q(X_t) Q(W_k)^T → A_s,q^(t) = SparseAttention(Q_quant, K_quant, V_quant; M) where M 为 SVG spatial-temporal mask（density=15%/25%）。
     b. if t - t_ref ≤ τ（τ=5，cache refreshing interval）:
        - 复用缓存残差: Δ_curr = Δ_quant^(t_ref) + Δ̃_quant^(t_ref)。
     c. else:
        - 计算全 attention A_full^(t)（无 mask）。
        - 更新一阶残差: Δ_quant^(t) = A_full^(t) − A_s,q^(t)。
        - 更新时间步 t-1 全 attention A_full^(t-1) 及其稀疏版 A_s,q^(t-1)。
        - 计算二阶残差: Δ̃_quant^(t) = Δ_quant^(t) − Δ_quant^(t-1)。
        - SVD 投影: SVD(Δ̃_quant) = S U V^T → Δ̃_quant_proj = S_{:,:r} U_{:r,:r} V^T_{:,:r}（r=16）。
        - 更新缓存 C = {Δ_quant^(t), Δ̃_quant_proj}, t_ref = t。
     d. 精炼 attention: Ã^(t) = A_s,q^(t) + Δ_curr（稀疏输出 + 缓存残差 = 近似全 attention）。
     e. Ã^(t) @ V_quant → 后续 transformer block 计算。
  3. 解码 latent → 输出视频 Y。

  **关键张量形状**（HunyuanVideo-13B, 720×1280p, frames=60）：
  - X ∈ R^(L×d_in), L ≈ 10^4+ tokens
  - A_full ∈ R^(h×L×L) → 单层 attention 矩阵内存 ~6.82GB（不可承受）
  - MSAD Global: Q̃,K̃ ∈ R^(L̃×d_k), L̃ = L/s² → attention ~O(L̃²)，s=128 时仅需 ~0.14GB
  - MSAD Local: k=256 个 salient queries → attention ~O(kL)，高效
  - SSAR cache: Δ_quant + Δ̃_quant（first+second order），与 first-order 缓存等量存储

## Quamba2 A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Quamba2 是一个针对 Selective State Space Models（Mamba1/Mamba2）的后训练量化（PTQ）框架，支持 W8A8、W4A8、W4A16 三种 bit-width 配置，以及 W4A{8/16}-mixed 混合精度。核心算法 pipeline：(1) **Sort-and-cluster**：利用 SSM 的 channel order preserving 属性，offline 校准各 channel 的最大值后对 head 内 channel 排序，对 head 和 channel 分别聚类（m=4 组 head, n=4 组 channel），为每个 head×channel 组计算独立 scaling factor 量化 x_t 到 8-bit；(2) **Per-state-group quantization**：利用 SSM 的 state persistence 属性（B 和 C 中激活的 state group 在时间步和样本间保持一致），对 B_t 和 C_t 按 state group 分别量化（每组一个 scaling factor）；(3) **Cluster-aware weight reordering**：根据 sort-and-cluster 的排序/聚类索引，offline 重排 input projection、causal convolution、normalization 和 output projection 的权重，保证 SSD 计算保持 channel order 从而输出不变；(4) **Offline Hadamard matrix fusion**：将 Hadamard 矩阵 offline 融合到 input/output projection 权重中（W_out^H = H_n W_out H_n^T, W_in^H = W_in H_n^T），配合 online Hadamard transform 实现 compute-invariance；(5) **Head-to-toe quantization**：从 embedding 层到 SSM blocks 到 lm_head 全量化，embedding 用 per-token quantization，lm_head weight 用 per-group quantization；(6) **W4AX-mixed**：进化搜索（population=40, generations=5）自动识别敏感 block 分配 W4A16，其余用 W4A8。伪代码：calibration set（Pile 512句）→ calibrate x channel max → sort channels → cluster heads(m) → cluster channels per head(n) → quantize x_t with m×n scales → quantize B_t/C_t per state group → reorder weights offline → fuse Hadamard offline → GPTQ on 4-bit projection weights → W4AX evolutionary search。
  - 实验比较：(a) Quamba2 vs Quamba vs MambaQuant 在 Mamba1 1.4B/2.8B 和 Mamba2 1.3B/2.7B/8B 上的零样本准确率（6 任务平均）；(b) W8A8/W4A8/W4A16 latency 对比（TPOT/TTFT, A5000 + Orin Nano 8G）；(c) MMLU 5-shot 评估 W4A8 vs W4A16 vs W4AX-mixed；(d) 消融：sort-and-cluster、per-state-group、Hadamard、GPTQ 各组件贡献；(e) embedding/lm_head 量化消融；(f) mixed-precision handcrafted vs auto search 对比；(g) batch size scaling（b=1/32/64/128/256）TPOT 对比；(h) 能效分析（J/req, tokens/GW）；(i) Pareto front: accuracy vs latency vs memory 与 QuaRot/Llama2/Llama3-QServe 对比。

- 硬件平台是什么，配置是什么。
  - Cloud: NVIDIA A5000 GPU 24GB。Edge: NVIDIA Orin Nano 8G。Latency profiling：warm-up iterations + 100 iterations 平均。CUDA kernel 基于 CUTLASS 实现。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba1（1.4B, 2.8B），Mamba2（130M, 370M, 1.3B, 2.7B, 8B）。Calibration: Pile dataset 随机 512 句（fixed seed）。
  - Benchmarks：LM-EVAL 框架。Zero-shot: LAMBADA, HellaSwag, PIQA, ARC-easy, ARC-challenge, WinoGrande（5 次平均）。MMLU 5-shot（57 学科）。Generation: Natural Questions (exact match), SquadV2 (F1)。
  - Baseline：Quamba (Chiang et al. 2025), MambaQuant (Xu et al. 2025)。对比 Transformer 量化：QuaRot (Llama2), QServe (Llama3 W4A8KV4)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/enyac-group/Quamba（论文声明 will be released）
  - 算法 pipeline 张量计算流程（以 Mamba2 W4A8 为例）：
    1. **Calibration**：从 Pile 采样 512 句 → 前向传播 → 记录每层 x 各 channel 的 max(|x_c|) → 按 max 降序排列 channel → 对 head 聚类（m=4 组）→ 对每组 head 内 channel 聚类（n=4 组）→ 记录 sort/cluster indices。同时记录 B/C 的 state group 激活模式。
    2. **Weight offline processing**：(a) 根据 cluster indices 重排 input proj weights W_in 的列和 causal conv1d weights 的 channel；(b) 重排 norm weights；(c) 重排 output proj weights W_out 的行；(d) Hadamard fusion: W_in^H = W_in @ H_n^T, W_out^H = H_n @ W_out @ H_n^T；(e) GPTQ 优化 4-bit 量化权重。
    3. **Inference（单 token 前向）**：
       - u_t ∈ R^D → W4A16/W4A8 input projection: x_t, B_t, C_t, Δ_t = (W_in^H)^T @ u_t（权重 4-bit, 激活 16-bit 或 4-bit weight×8-bit act）
       - Online Hadamard: x_t^H = H_n @ x_t
       - Sort-and-cluster: 按 sort/cluster indices 重排 x_t^H → 分组 → 每组内 quantize: x̄_t^s = clamp(round(x_t^H / s_{m,n}), -127, 127)（8-bit）
       - Per-state-group 量化 B_t/C_t: B̄_t^g = clamp(round(B_t / s_g), -127, 127)（每组 state group 一个 scale）
       - Causal conv1d: y_conv = conv1d_8bit(x̄_t^s, W_conv_8bit)
       - SSD scan: h_t = A_t @ h_{t-1} + B̄_t^g @ x̄_t^s, y_ssd = C̄_t^g @ h_t（8-bit states）
       - Online Hadamard: y^H = H_n @ (y_ssd ⊙ SiLU(z_t))
       - Output projection: y_out = (W_out^H)^T @ ȳ^H（权重 4-bit, 激活 8-bit）
    4. **W4AX 混合精度搜索**：evolutionary search (pop=40, gen=5) → 每代保留 top 50% → 10 mutation + 10 crossover → 最终每层选 W4A8 或 W4A16。

## QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QeRL 将 NVFP4 权重量化与 LoRA 低秩适配结合，用于 LLM 强化学习（RL）训练。核心 pipeline：(1) 用 AWQ 对预训练 LLM 权重做 NVFP4 后训练量化（calibration: OpenThoughts-114k 的 256 序列×2048 tokens）；(2) 在量化模型 Q/K/V/O/gate/up/down 层添加 LoRA adapter（rank=32，约 1% 可训参数）；(3) 通过 GRPO 或 DAPO 进行 RL 训练。(4) 创新 AQN 机制：高斯噪声融入 RMSNorm scale 参数，按指数衰减调度（σ_start=1e-2, σ_end=5e-4, K=10 阶段），实现从探索到利用的动态过渡。伪代码：每步确定 stage k → σ(k)=σ_start×(σ_end/σ_start)^((k-1)/(K-1)), stage 0 无噪声 → 注入噪声到旧策略 π_θold ← π_θ+N(0,σ²) → rollout 生成 G 个候选 → 计算 group relative advantage → 用 GRPO/DAPO 目标更新 LoRA。前向推理用 Marlin kernel 加速 NVFP4×BF16 矩阵乘法，梯度仅回传 LoRA 层。
  - 实验比较：(a) QeRL vs BF16 LoRA vs BF16 Full FT vs QLoRA(NF4+LoRA) 在 GSM8K 准确率；(b) BigMath 训练的 MATH500/AIME24/25/AMC23 上比较；(c) 量化格式消融 NVFP4 vs MXFP4 vs NF4；(d) noise scheduler 消融 exponential vs linear vs cosine vs logarithmic；(e) LoRA rank 消融 16/32/64/128；(f) 学习率消融；(g) rollout 吞吐量/内存对比（batch=2/4/8）。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 80GB GPU。速度测试在单 H100，最终评估模型在 8×H100 训练。vLLM 引擎用于 rollout（memory utilization: 3B=0.20, 7B=0.30, 14B=0.45, 32B=0.40）。环境：CUDA≥12.4.1, Linux, 64GB RAM。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-3B/7B/14B/32B-Instruct（基础通用模型，未做数学微调）
  - 训练数据：GSM8K（7500 样本, generation number=8）, BigMath（122000 样本, generation number=16, difficulty level 3-5 或 4-5）
  - Benchmarks：GSM8K, MATH500, AIME 2024, AIME 2025, AMC 23, 均报 Pass@1
  - RL 算法：GRPO（GSM8K 训练）, DAPO（BigMath 训练）
  - RL 超参数：AdamW-8bit, LR=1e-5(QeRL/QLoRA)/5e-6(LoRA BF16), Batch=128, Samples per prompt=8(GSM8K)/16(BigMath), Max response=4096/8192, temperature=1.0, Clip(0.2,0.28)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/NVlabs/QeRL (Apache 2.0)
  - 张量计算流程：(1) NVFP4 量化权重 \tilde{W} + FP32 全局 scale S_FP32 + FP8(E4M3) block-wise scale S_E4M3（block=16 元素），dequant: \hat{W}=S_FP32·(S_E4M3⊙\tilde{W})；(2) LoRA: ΔW=BA, B∈R^{d×r}, A∈R^{r×k}, r=32；(3) AQN: Z_noisy∈R^{1×d}∼N(0,σ²I) 并入 RMSNorm: RMSNorm_noise(x)=w_noise⊙x/√(mean(x²)+δ)，w_noise=Z_noise+w（等价变换为乘法噪声 (Z_noise/w+I)⊙\hat{W}）；(4) 前向：x_attn=RMSNorm_noise(x)·\hat{W}_{q,k,v}+LoRA_output, x_ffn=RMSNorm_noise(x)·\hat{W}_{gate,up}+LoRA_output；(5) 仅更新 LoRA 参数 A,B，量化权重冻结。使用方法：`python quantize_nvfp4.py --model Qwen/Qwen2.5-7B-Instruct` → `bash training/dapo_qwen2.5-7b_nvfp4_single_gpu.sh`。

## QT-DoG: Quantization-aware Training for Domain Generalization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QT-DoG 将量化感知训练（QAT）作为隐式正则化器用于域泛化（Domain Generalization, DG）。核心思想是：权重量化会引入均匀分布的量化噪声 Δ（量化误差范围 [−s/2, +s/2]），该噪声作为隐式正则项，推动优化过程趋向损失景观中的平坦极小值（flatter minima），从而减少对源域的过拟合并提升 OOD 泛化能力。量化公式：w̄ = ⌊clip(w/s, −Q_N, Q_P)⌉，w_q = w̄ × s。采用 LSQ (Learned Step Size Quantization) 作为主要量化方法，每通道独立学习 scaling factor s，所有层除最后一层外均量化至低比特。在训练进行到一定步数后（DomainNet: 8000 步，其余数据集: 2000 步）启动量化。EoQ (Ensemble of Quantization) 训练 5 个独立初始化的量化模型，通过 bagging 方式（平均 softmax 输出）集成预测：ŷ = argmax_k Softmax((1/E) Σ f(x; w_q^i))。
  - 实验比较：(1) QT-DoG (单模型 7-bit) vs ERM, IRM, Group DRO, Mixup, MLDG, CORAL, MMD, Fish, Fishr, SWAD, MIRO, CCFP, ARM, VREx, RSC, Mixstyle, SagNet 等 DG 方法——在 DomainBed 五大数据集上比较 OOD 准确率；(2) EoQ (5 模型集成) vs ERM Ensemble, DiWA, EoA, DART；(3) QT-DoG + CORAL / + MixStyle 组合实验；(4) 不同量化方法消融：LSQ vs INQ (QAT) vs OBC (PTQ)；(5) WILDS 数据集实验 (Amazon, Camelyon)；(6) ViT 实验 (DeiT-Small) 和 CLIP 实验 (ViT-B/16)；(7) Bit precision 分析 (8/7/6/5/4/3/2-bit)；(8) ResNeXt-50-32x4d 更大预训练数据集实验；(9) 量化步数消融 (1000/2000/3000/4000)；(10) Channelwise vs Layerwise scaling factor 消融；(11) 统一噪声注入消融。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A100 GPU，Python 3.8.16，PyTorch 1.10.0，Torchvision 0.11.0，CUDA 12.1。CPU 推理延迟测试使用 AMD EPYC 7302 处理器（全精度 34.28ms vs INT8 21.02ms for ResNet-50）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet-50（25M 参数，ImageNet 预训练，主要骨干）；ResNeXt-50-32x4d（25M 参数，Instagram 1B 图像预训练）；DeiT-Small（ViT，Domain Generalization 实验）；CLIP ViT-B/16（CLIP-based DG 实验）；DistilBERT（WILDS text 任务）。
  - 数据集：DomainBed 基准——PACS (4 domains, 9991 samples, 7-class)、VLCS (4 domains, 10729 samples, 5-class)、OfficeHome (4 domains, 15588 samples, 65-class)、TerraIncognita (4 domains, 24788 samples, 10-class)、DomainNet (6 domains, 586575 samples, 345-class)；WILDS 基准——Amazon (review 分类, 10th percentile acc)、Camelyon17 (病理切片分类, average acc)。
  - Benchmark metric：out-of-domain accuracy（每个 domain 轮流作为目标域，其余为源域，三次独立运行取平均 ± 标准误），使用 DomainBed 训练-域验证协议（Gulrajani & Lopez-Paz, 2021）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://saqibjaved1.github.io/QT_DoG/（论文声称将开源代码、环境配置、复现指令）。
  - 软件依赖：DomainBed 框架（Gulrajani & Lopez-Paz, 2021），PyTorch，LSQ (Esser et al., 2020)，GradCAM 可视化使用 pytorch-grad-cam。
  - 算法 pipeline 核心流程（QT-DoG with LSQ, ResNet-50, 7-bit）：
    1. 训练设置：采用 DomainBed 默认超参——batch_size=32（per-domain），Adam optimizer，lr=5e-5，无 weight decay。ImageNet 预训练 ResNet-50 初始化。每个域轮流作为目标域，其余为源域，20% 源域样本用作验证集。
    2. 前 2000 步（DomainNet 为 8000 步）：正常全精度 ERM 训练。
    3. 第 2000 步起：对除最后一层外的所有层启用 LSQ 量化——
       伪代码：
       ```
       for each layer with weight W:
         s = learnable_parameter(init_value)  # per-channel scaling factor
         W_bar = round(clip(W / s, -Q_N, Q_P))  # Q_N=2^(b-1), Q_P=2^(b-1)-1 for signed b-bit
         W_q = W_bar * s                          # quantized weight
         # Forward: y = x @ W_q
         # Backward: STE (Straight-Through Estimator) through round()
       ```
    4. 每 300 步在源域验证集上评估，选择最佳模型。
    5. 推理时使用量化权重 W_q（INT7 格式），模型体积压缩约 4.6x（25M FP32 → ~5.4M INT7 等效）。
    6. EoQ：独立训练 5 个模型（不同随机种子），集成输出为各模型 softmax 概率的平均：
       ```
       y_hat = argmax(mean([softmax(f(x; W_q^i)) for i in 1..E], dim=0))
       ```
       5 个 7-bit 量化模型总参数量 ≈ 1.1x 全精度单模型，显著低于 DiWA（60 个全精度模型）和 EoA（6 个全精度模型）。
  - 关键发现：7-bit 为最优比特精度（PACS 87.8% vs ERM 84.7%）；QAT (LSQ/INQ) 有效而 PTQ (OBC) 无效（无训练过程无法寻找平坦极小值）；量化在 2000 步时引入效果最佳。

## QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QWHA 提出一种基于 Walsh-Hadamard Transform (WHT) 的量化感知 PEFT 适配器（WHA），配合量化感知初始化方案（AdaAlloc + Refinement）。核心设计分为三部分：
    (1) **WHA 适配器设计**：权重更新公式 ΔW = F H^{-1}，其中 H 是固定的 WHT 矩阵（±1 元素，通过 Kronecker 积 H_N = H_2 ⊗ H_{2^{n-1}} 递归构造），F = Scatter(c, E) 是可训练的稀疏系数矩阵。WHA 相比 LoRA 具有 full-rank 表示能力（rank ≈ min(d_in, d_out)），仅使用单个变换（而非传统 FT-based adapter 的双变换），且 WHT 的方形波基函数相比 DCT/DHT 的正弦基函数更擅长捕获量化误差的异常值（outlier），能集中更多能量在少量系数中（Pareto hill index η 最小）。
    (2) **AdaAlloc 参数选择**：按输出通道的激活误差大小按比例分配参数预算 p_i = p · ||(ΔW_Q X)_{i,:}||_F^t / Σ_j ||(ΔW_Q X)_{j,:}||_F^t，保证每个通道至少分配若干参数（维持 full rank），同时在每个通道内选择 |v B^{-1}| 最大的位置（即 |(ΔW_Q H)_{i,j}| 最大的系数），既减少量化误差又维持 fine-tuning 能力。
    (3) **Refinement 值精化**：对已选参数位置，通过最小二乘法重新投影优化参数值：x^* = v B'^T (B' B'^T)^{-1}，其中 v = (ΔW_Q)_{i,:} R, B = H^{-1} R, R = U Σ^{1/2}（XX^T 的矩阵平方根），B' 由选中索引对应的 B 行组成。此步骤使选中的 basis vectors 能补偿未选中向量的影响，大幅降低层输出误差（例如 Key 投影：Refinement 前 0.62 → 后 0.27，缩小 2.3x）。
  - 实验比较：(1) Main evaluation: QWHA vs CLoQ (LoRA-based QA-PEFT), SHiRA (sparse adapter), LoCA (DCA-based), SSH (DHA-based), GPTQ_MagR (quantized only) — 在 LLaMA-3.1-8B / LLaMA-3.2-3B / Mistral-7B-v0.3 上，4/3/2-bit 量化，CSQA 和 GSM8k benchmark；(2) Ablation on adapter type: WHA vs DCA vs DHA vs Sparse（均使用 AdaAlloc + Refinement）；(3) Ablation on parameter selection: Random vs Magnitude vs LoCA vs SSH vs AdaAlloc（均使用 WHA + Refinement）；(4) QWHA vs CLoQ 的 accuracy vs parameter budget 曲线（Figure 6）；(5) 训练效率：各方法在 Alpaca 训练时间对比（batch size 1/2/4/8/16），WHT 1D vs 2D 训练时间对比；(6) 消融：温度 t (0.25/0.5/1.0/1.5/2.0)，量化 group size (32/64/128/256)。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB GPU。训练批次大小 4-16（GSM8K 用 batch=4, Alpaca 用 batch=6）。校准集：128 条 WikiText-2 序列（长度 2048），用于量化和适配器初始化的 calibration。PyTorch 框架 + AdamW optimizer。量化方案：GPTQ + MagR，group size 64，适配器应用于所有线性层（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA-3.1-8B（32 layers, d=4096），LLaMA-3.2-3B（28 layers, d=3072），Mistral-7B-v0.3（32 layers, d=4096）。
  - 数据集（微调）：Stanford-Alpaca（52k 指令微调样本）；GSM8k（数学推理训练集）。
  - Benchmark（评估）：
    - CSQA (CommonsenseQA, Zero-shot)：覆盖 7 个多选题基准——ARC-Challenge, ARC-Easy, BoolQ, HellaSwag, OpenBookQA, PiQA, WinoGrande。使用 lm-evaluation-harness 评测。
    - GSM8k (Zero-shot CoT)：算术推理，测试集 zero-shot chain-of-thought。
  - 量化校准集：WikiText-2，128 条序列 × 2048 token。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/vantaa89/qwha
  - 软件依赖：PyTorch, Transformers (HuggingFace), fast-hadamard-transform (Dao-AILab, https://github.com/Dao-AILab/fast-hadamard-transform), GPTQ (Frantar et al., 2023)
  - 算法 pipeline 核心流程（QWHA, 4-bit 量化, LLaMA-3.2-3B, P(r=64)）：
    1. **量化阶段**：
       - 使用 GPTQ + MagR 对预训练权重 W_0 做 4-bit 量化，group size 64：
         ```
         s = max(|W_group|) / (2^(b-1) - 1)  # per-group quantization scale
         W_Q_tilde = clamp(round(W_0 / s) - z, 0, 2^b - 1)  # INT4
         W_Q = (W_Q_tilde + z) * s  # dequantized float
         ΔW_Q = W_0 - W_Q  # quantization error
         ```
    2. **Calibration 阶段**（收集激活统计，用于初始化）：
       - 128 条 WikiText-2 序列前向传播，收集各层激活 X
       - 计算 Hessian 平方根：XX^T = U Σ U^T → R = U Σ^{1/2}
       - 预计算 B = H^{-1} R（H 为 WHT 矩阵，通过 fast-hadamard-transform kernel 计算）
    3. **QWHA 初始化阶段（Algorithm 1）**：
       ```
       for each layer i with linear weight W_Q:
         # Step 1: AdaAlloc - 通道级参数分配
         for each output channel j in 0..d_out-1:
           error_j = ||(ΔW_Q X)_{j,:}||_F  # 各通道的输出误差
         p_j = floor(p * error_j^t / sum_k(error_k^t))  # t=1 默认
         余数分配给最小分配的通道，确保 sum(p_j) = p
         
         # Step 2: Per-channel parameter selection
         for each output channel j:
           v = (ΔW_Q)_{j,:} R  # 投影的量化误差
           dense_sol = v B^{-1} = (ΔW_Q H)_{j,:}  # 稠密解
           # 选取 |dense_sol| 最大的 p_j 个位置作为 E_j
           E_j = TopK_pj_Index(|dense_sol|)
           
           # Step 3: Value Refinement
           B' = B[E_j, :]  # 选中行
           c_j = v B'^T (B' B'^T)^{-1}  # 最小二乘精化
         
         # 构建稀疏矩阵 F
         F = Scatter(c, E)  # F[E[l,0], E[l,1]] = c[l]
    4. **Fine-tuning 阶段**：
       ```
       for each training step:
         for each linear layer:
           # WHA 前向传播
           ΔW = F H^{-1}  # F 为稀疏，H^{-1} 通过 fast Hadamard kernel
           Y = (W_Q + α * ΔW) X  # α_effective ≈ 1.0 (α_explicit=4000/d_in)
         # 反向传播：仅更新系数 c（F 中的非零值），E 和 H 固定
         # Loss = cross_entropy(logits, labels)
         # Optimizer: AdamW, lr = 3e-5 (LLaMA-3.2-3B, 4-bit, Alpaca)
       ```
    5. **推理阶段**：
       - WHA 适配器的额外推理：ΔW X = F (H^{-1} X)，通过 fast Hadamard kernel 实现，仅用加法和减法（无矩阵乘法）
       - 合并：Y = W_Q X + ΔW X。
       - 推理吞吐：184.6 tokens/sec (batch=128, prefill=2048, gen=64)，仅比 LoRA (188.1) 低 1.9%，远优于 DCA/DHA (92.4 tokens/sec, 下降 50.9%)

## QERA: an Analytical Framework for Quantization Error Reconstruction

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QERA 提出两个分析解（analytical solution）来求解量化误差重建（Quantization Error Reconstruction, QER）问题——给定量化权重 W̃，寻找最优低秩项 C_k = A_k B_k 使得层输出误差最小化：(1) QERA-exact (Theorem 1)：最小化 E[||y - ỹ||²] 的精确闭式解 C_k = (R_{XX}^{1/2})⁻¹ · SVD_k(R_{XX}^{1/2} (W - W̃))，其中 R_{XX} = E[x^T x] 为输入自相关矩阵；(2) QERA-approx (Theorem 2)：在"不同嵌入维度不相关"假设下的高效近似解 C_k = S⁻¹ · SVD_k(S (W - W̃))，其中 S = diag(√E[x₁²], ..., √E[x_m²]) 为对角标度矩阵。QERA 对量化函数 q(·) 无约束，可结合任意量化方法使用。QERA-approx 解释了 LQER 启发式方法的成功，并解决了 LQER 中校准样本数与恢复性能不一致的问题。
  - 实验比较：(1) QPEFT 实验：QERA-approx vs Full Fine-tuning, LoRA, QLoRA, LoftQ (5-iter) — 在 RoBERTa-base @ GLUE (4/3/2-bit, rank 8/64) 和 LLaMA-2-7B/LLaMA-3.1-8B @ SlimPajama & GSM8K (4/2-bit, rank 8/64) 上比较微调准确率/困惑度和收敛速度；(2) PTQ 实验：QERA-exact & QERA-approx vs BF16, w-only, ZeroQuant-V2, LQER, HQQ — 在 TinyLlama-1.1B, Gemma-2-2B, Phi-3.5-mini, LLaMA-2-7B/13B, LLaMA-3.1-8B/70B 上比较 WikiText2 PPL 和 6 个下游任务 (ARC, BoolQ, CommonSenseQA, Winogrande, MMLU, BBH) 准确率，以及 Vicuna-7b-v1.5 @ AlpacaEval 2.0 的 Win Rate；(3) 消融：模型输出误差 vs 权重逼近误差、LoftQ 迭代数 vs 模型输出误差、校准集大小对 LQER vs QERA 性能的影响、Assumption 1 的 R_{XX} 非对角元验证。

- 硬件平台是什么，配置是什么。
  - QPEFT 实验：4× NVIDIA A100 80GB GPU，AMD EPYC 64-Core Processor，1024GB RAM，总计约 2100 GPU hours。
  - PTQ 实验：8× NVIDIA A6000 48GB GPU，AMD EPYC 256-Core Processor，1024GB RAM，总计约 4500 GPU hours。
  - 矩阵平方根计算使用 SciPy 的 blocked Schur algorithm，在 CPU 上执行（FP64）。自相关矩阵外积在 FP32 累积，FP64 计算平方根。

- 模型是什么。数据集和bench分别是什么。
  - 模型：RoBERTa-base (QPEFT)；LLaMA-2-7B/13B, LLaMA-3.1-8B/70B, TinyLlama-1.1B, Gemma-2-2B, Phi-3.5-mini (PTQ)；Vicuna-v1.5-7B (指令跟随评估)。
  - 数据集（微调）：GLUE benchmark (MNLI, QNLI, RTE, SST-2, MRPC, CoLA, QQP, STS-B)；SlimPajama（连续预训练）；GSM8K（监督微调）。
  - 数据集（校准）：WikiText2（用于 RoBERTa-base QPEFT 校准）；论文中 QPEFT 校准集来自预训练数据集；PTQ 校准集论文未明确指定具体数据集名。
  - Benchmark（PTQ 评估）：WikiText2 (perplexity), ARC (challenge), BoolQ, CommonSenseQA, Winogrande, MMLU, BigBench-Hard (BBH)；使用 lm-evaluation-harness 评测。AlpacaEval 2.0 (GPT4-Turbo 作为 evaluator, length-controlled win rate)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/ChengZhang-98/QERA。
  - 软件依赖：PyTorch, Transformers, PEFT, Accelerate, SciPy (blocked Schur algorithm), lm-evaluation-harness, Evaluate, AlpacaEval 2.0。
  - 量化格式：4-bit 使用 QLoRA 的 4-bit floating point (PEFT 实现)；3/2-bit 使用 emulated MXINT (block size=32/16)。
  - 核心伪代码（QERA-approx 初始化，PyTorch-like）：
    ```
    # W: pretrained weight [m, n]
    # q(), dq(): quantize, dequantize functions
    # X_calib: calibration dataset of input vectors
    # k: target rank

    def qera_approx_init(W, q, dq, X_calib, k):
        # Step 1: Compute activation statistics
        s_sq = torch.zeros(m)
        for x in X_calib:        # x shape: [1, m]
            s_sq += x.square().squeeze(0)
        s_sq /= len(X_calib)
        S = torch.diag(torch.sqrt(s_sq))  # shape: [m, m]

        # Step 2: Quantize and compute scaled error
        W_q = q(W)
        W_tilde = dq(W_q)
        E = W - W_tilde                # weight quantization error
        Q = S @ E                       # scaled error

        # Step 3: Truncated SVD on scaled error
        U, Sigma, Vt = torch.svd(Q)
        U_k = U[:, :k]
        Sigma_k = Sigma[:k]
        Vt_k = Vt[:k, :]

        # Step 4: Unscale to get low-rank terms
        A_k = torch.inverse(S) @ U_k   # shape: [m, k]
        B_k = torch.diag(Sigma_k) @ Vt_k  # shape: [k, n]

        return A_k, B_k

    # Forward pass (same for QERA-exact and QERA-approx):
    # y = x @ (W_tilde + A_k @ B_k)
    # At inference: pre-merge C_k = A_k @ B_k into W_tilde
    ```
  - QERA-exact 伪代码（区别于 approx 在于使用 R_{XX} 代替 S）：
    ```
    def qera_exact_init(W, q, dq, X_calib, k):
        # Step 1: Compute autocorrelation matrix
        R = torch.zeros(m, m, dtype=torch.float64)
        for x in X_calib:
            R += (x.T @ x).to(torch.float64)   # outer product
        R /= len(X_calib)
        R_sqrt = matrix_sqrt(R)                 # blocked Schur, CPU

        # Step 2-4: Same as approx but with R_sqrt instead of S
        W_q = q(W); W_tilde = dq(W_q)
        Q = R_sqrt @ (W - W_tilde)
        U, Sigma, Vt = torch.svd(Q)
        A_k = torch.inverse(R_sqrt) @ U[:, :k]
        B_k = torch.diag(Sigma[:k]) @ Vt[:k, :]
        return A_k, B_k
    ```
  - 张量计算流程：给定线性层 y = xW（x ∈ R^m, W ∈ R^{m×n}），量化 W → W̃ = dq(q(W))，QERA 寻找 C_k = A_k B_k（rank k << min(m,n)）使得 ||ỹ - y||₂ 最小化。QERA-exact 通过 R_{XX}^{1/2} 将最小化层输出误差转化为标准 SVD 低秩逼近问题：min ||R_{XX}^{1/2}(W̃ + C_k - W)||_F²。QERA-approx 在 Assumption 1（E[x_i x_j]=0, i≠j）下将 R_{XX} 简化为对角矩阵 S²，大幅降低计算开销。推理时 y = x(W̃ + A_k B_k)，低秩项可预合并进 W̃ 不引入额外推理开销。


## QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QA-LoRA 提出分组量化 + 低秩适应的联合方法，核心在于平衡量化与适应的自由度。具体设计：(1) 对预训练权重 W ∈ R^{D_in × D_out} 的每一列划分为 L 组（组大小 g = D_in / L），每组独立使用缩放因子 α_{l,j} 和零点 β_{l,j} 进行 INT4/INT3/INT2 量化；(2) 输入 x 经过 AvgPool/group-sum 聚合操作 A(x)，将维度从 D_in 降至 L；(3) LoRA 适配器 A ∈ R^{L × D_int}，B ∈ R^{D_int × D_out}，其中 A 的行数与分组数 L 对齐，不再对每行自由优化，而是组内共享；(4) 前向传播：y = W̃^T x + s · A(x)^T · A^T B^T；(5) 微调后通过更新零点矩阵 B' = B - s · (L1 L2) ⊘ A 将 LoRA 权重合并到量化模型中，保持 INT 格式用于推理。此设计实现微调时使用量化权重节省显存和时间，推理时直接使用 INT 格式无需 PTQ。
  - 实验比较：QA-LoRA vs QLoRA vs QLoRA w/ GPTQ vs PEQA，在不同模型规模（LLaMA 7B/13B/33B/65B）、不同量化位宽（INT4/INT3/INT2）、不同微调数据集（Alpaca 52K、FLAN v2 320K、Self-instruct、Longform、Chip2）下比较 MMLU 0-shot/5-shot 准确率、CommonSense QA（HellaSwag、PIQA、WinoGrande、ARC-e、ARC-c、BoolQ、OBQA）0-shot 准确率。消融实验：分组大小（g=32/64/128）、数据集大小（160K-480K）对 MMLU 准确率的影响。

- 硬件平台是什么，配置是什么。
  - GPU: Tesla V100。7B/13B/33B 模型使用 1 块 V100，65B 模型使用 2 块 V100。训练步数 Alpaca 10K、FLAN v2 20K。batch size 16。paged AdamW optimizer，max gradient norm 0.3，constant LR schedule，7B/13B LR=2e-5，33B/65B LR=1e-5。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA（7B、13B、33B、65B）和 LLaMA2（7B、13B）。
  - 数据集：微调用 Alpaca（52K）、FLAN v2（320K subset）、Self-instruct、Longform、Chip2。
  - Benchmark：MMLU（0-shot 和 5-shot，57 个语言任务含 STEM、Humanities、Social Science、Other），CommonSense QA（HellaSwag、PIQA、WinoGrande、ARC-easy、ARC-challenge、BoolQ、OpenBookQA），使用 lm-eval-harness 评测。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/yuhuixu1993/qa-lora（论文中提供）。
  - 量化方法：采用 GPTQ 进行预训练权重的分组不对称量化（group size=32），act-order=false，true-sequential=true。支持 INT4/INT3/INT2，但方法框架也兼容其他 PTQ 方法（如 AWQ、SPQR）。
  - 核心伪代码（Algorithm 1，PyTorch-like）：
    ```
    # D_in, D_out, D_int: dimensions
    # L: number of quantization groups (D_in // L = group size)
    # s: adaptation coefficient; N: bit width
    
    QA = nn.AvgPool1d(D_in // L)          # 组内平均聚合
    lora_A = nn.Parameter(torch.empty((D_int, L)))
    lora_B = nn.Parameter(torch.empty((D_out, D_int)))
    
    def qalora_forward(x, W, lora_A, lora_B):
        W_tilde = pre_quantization(W, alpha, beta)
        result = x @ W_tilde
        result += (QA(x) * (D_in // L)) @ lora_A.T @ lora_B.T * s
        return result
    
    def pre_quantization(W, alpha, beta):
        # alpha: shape (L, D_out), beta: shape (L, D_out)
        W_hat = torch.round(W / alpha) + beta
        return alpha * (W_hat - beta)
    
    def merge_with_quantization(beta, lora_A, lora_B):
        # 合并 LoRA 到零点矩阵，保持 INT 格式
        beta_new = beta - s * (lora_B @ lora_A).T / alpha
        return beta_new
    ```
  - 张量计算流程：给定输入 x ∈ R^{D_in} 和分组量化权重 W̃ = [α_{l,j} · ⌊(w_{i,j} - β_{l,j}) / α_{l,j}⌉ + β_{l,j}]（其中 l = ⌊i/g⌋ 为组索引），聚合操作 A(x) 对每组内 g 个元素求和输出 L 维向量。forward 计算 y_j = Σ_i x_i · W̃_{i,j} + s · Σ_k (Σ_{r=1}^g x_{(k-1)g+r}) · a_{k,mid} · b_{mid,j}。合并推理时，只需更新 β'_{l,j} = β_{l,j} - s · (Σ_{mid} b_{mid,j} · a_{l,mid}) / α_{l,j}，Ŵ 和 α 不变，模型仍为 INT 格式。

## Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Q-resafe——量化感知的安全修补框架，在已量化的 LLM 上通过三个关键步骤恢复安全能力：(1) **安全修补数据集构建**：使用预量化全精度 LLM 作为教师，对校准数据集的每个 prompt x，生成 preferred response y_w ~ π_W(·|x)（全精度模型）和 dispreferred response y_l ~ π_Q⁰(·|x)（量化后模型），构建 DPO 偏好三元组 (x, y_w, y_l)，无需人工标注；(2) **DPO 对齐损失**：L = -E_{(x,y_w,y_l)} log σ(β log(π_Q(y_w|x)/π_Q⁰(y_w|x)) - β log(π_Q(y_l|x)/π_Q⁰(y_l|x)))，以量化模型 π_Q⁰ 为参考模型；(3) **周期性安全关键权重识别与选择性更新**：每 K 次迭代使用 SNIP score I(W_ij, x) = |W_ij · ∇_{Q_ij} L(x)|，对校准集 D_calib 取均值得到 SafeScore(Q)，选择 top-τ 百分比的权重作为安全关键权重，通过掩码矩阵 M_Q 只更新这些权重，其余权重保持不变，更新约束为 Q = Q⁰ + Quant(M_Q ⊙ AB)（LoRA 低秩分解 + 再量化）。算法 1 给出了完整流程。
  - 实验比较（安全风险评估）：对四种代表性量化方法（AWQ (PTQ w/o FT)、AQLM (PTQ w/ FT)、LLM-QAT (QAT w/ FT)、QLoRA (QAT w/ LoRA FT)）在 INT4/INT8/3-bit/2-bit 下，使用三种风险等级校准数据集（Risk-I: UltraChat benign、Risk-II: AOA indirectly harmful、Risk-III: AdvBench directly harmful），评估 ASR↓、MT-Bench↑、AlpacaEval↑。
  - 实验比较（安全修补）：Q-resafe vs 各 baseline 量化方法在三种风险数据集上的安全恢复效果，包含消融实验（τ 从 0.0 到 1.0 的安全关键权重比例、SFT vs DPO vs Q-resafe 的对比、2-bit 到 8-bit 的 bit-width 影响、LLM.int8()/NF4/FP4 的方法泛化性、不同解码策略 τ/top-k/top-p 下的鲁棒性）。

- 硬件平台是什么，配置是什么。
  - 4 × NVIDIA A100 40GB GPU。框架：PyTorch + HuggingFace Transformers。预训练模型权重从 HuggingFace Hub 获取。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-2-7B-Chat（安全对齐对话强）、Gemma-7B-Instruct（结构化任务强），均为经过 instruction tuning 和 RLHF 的安全对齐模型。
  - 校准数据集：AdvBench（520 条有害指令，覆盖亵渎、威胁、错误信息等）、UltraChat（大规模多域安全对话）、AOA（绝对服从 agent 提示 + 10 条 AdvBench 样本构造的间接有害数据集）、Alpaca-cleaned（用于安全关键权重识别的消融实验）。
  - 安全评估 Bench：ASR（Attack Success Rate，响应对有害指令的攻击成功率↓），辅助指标包括 HarmBench 分类器和 Harmfulness Score (1-5 GPT-4 评分)。编码策略攻击评估用 ASR_Decoding（变 temperature/top-k/top-p 采样）。
  - 效用评估 Bench：MT-Bench（160 题，8 领域，双轮对话，GPT-4 评分 1-10）、AlpacaEval（805 题，单轮，GPT-4 对比 text-davinci-003 的 win rate）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：项目主页 https://thecommonirin.github.io/Qresafe/，代码仓库 https://github.com/Thecommonirin/Qresafe。论文声明发布所有评估模型和修改后的 Q-Resafe benchmark。
  - **算法 pipeline 伪代码（对应 Algorithm 1）**：
    1. 构建安全修补数据集 D_patch：
       for each prompt x in D_calib:
           y_w ~ π_W(·|x)   # 全精度模型生成 preferred response
           y_l ~ π_Q⁰(·|x)  # 量化模型生成 dispreferred response
           D_patch ← D_patch ∪ {(x, y_w, y_l)}
    2. 初始化：量化权重 Q⁰ (来自 AWQ/AQLM/LLM-QAT/QLoRA)，LoRA 矩阵 A∈R^{d_in×r}, B∈R^{r×d_out}（r=128），超参数 τ=0.6, K=1000, η=5e-6, β=0.01。
    3. for t = 0 to T-1:
         if t % K == 0:  # 周期性重新识别安全关键权重
           对于每层权重矩阵 W，计算 SNIP score:
             SafeScore(Q^t) = E_{x∈D_calib} |W_ij · ∇_{Q_ij} (-log p(y|x))|
           M_Q = 1[SafeScore(Q^t) ∈ Top-τ]  # 选 top-τ% 权重
           (M_A, M_B) = MapMask(M_Q)        # 映射到 LoRA 维度的掩码
         # 选择性 SGD 更新（仅更新安全关键权重对应的 LoRA 行/列）
         A^{t+1} = M_A ⊙ (A^t - η∇_A L_DPO) + (1-M_A) ⊙ A^t
         B^{t+1} = M_B ⊙ (B^t - η∇_B L_DPO) + (1-M_B) ⊙ B^t
         Q^{t+1} = Q⁰ + Quant(A^{t+1} B^{t+1})  # LoRA 更新后再量化到同精度
    4. 输出：安全修补后的量化 LLM π_{Q^T}
  - **张量计算示例**：对 Llama-2-7B-Chat 的某层权重 W∈R^{4096×4096}，INT4 量化后 Q⁰∈Q^{4096×4096}，LoRA r=128 时 A∈R^{4096×128}, B∈R^{128×4096}，参数量仅为全量微调的 6.25%。SNIP 计算：对校准 batch 的每个 token 计算交叉熵损失 L，反向传播得 ∇_Q L，逐元素乘 |Q_ij · ∇_{Q_ij} L|，跨 batch 求均值排序，取 top-60%(τ=0.6) 的权重索引生成 M_Q∈{0,1}^{4096×4096}。MapMask 将 M_Q 中有 1 的行/列映射为 M_A∈{0,1}^{4096×128}（对应行有 1 则整行标记）和 M_B∈{0,1}^{128×4096}（对应列有 1 则整列标记）。更新时非掩码位置的 LoRA 参数保持为零初始化不变。


## RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：RaBitQ 是一种新的随机化量化方法，用于高维向量的 ANN 搜索距离估计。核心设计包括：(1) **码本构造**：先归一化数据向量到单位超球面，构造 2^D 个双值向量组成的确定码本 C = {±1/√D}^D（超立方体顶点），再采样随机正交矩阵 P 旋转码本得到 C_rand = {Px | x∈C}，消除对特定向量的偏好；(2) **量化编码**：对每个数据向量 o，计算 o'=P^{-1}o，取 o' 各维度的符号位构成 D-bit 字符串 x̄_b∈{0,1}^D 作为量化码，其中 1 对应 +1/√D、0 对应 -1/√D，O(D) 时间；(3) **无偏距离估计器**：⟨o,q⟩ ≈ ⟨ō,q⟩/⟨ō,o⟩，其中 ō=Px̄ 为量化向量，⟨ō,o⟩ 可预计算（期望值约 0.8），⟨ō,q⟩ = ⟨x̄,q'⟩，q'=P^{-1}q。证明该估计器无偏且具有严格的 O(1/√D) 概率误差界（渐近最优）；(4) **高效计算**：查询时对 q' 做随机化均匀标量量化（B_q=4 bit），⟨x̄_b,q̄_u⟩ 通过 bitwise-and + popcount 或 FastScan SIMD 批量实现。
  - 实验比较（距离估计精度与效率）：RaBitQ vs PQ、OPQ、LSQ，变长量化码（padding 0 或调整 M），六个数据集上评估 average relative error、maximum relative error 和 time per vector。
  - 实验比较（ANN 查询性能）：RaBitQ-batch + IVF vs OPQx4fs-batch + IVF vs HNSW，评估 Recall、Average Distance Ratio 和 QPS，K=100，对比 re-ranking 参数的影响（OPQ: rerank=500/1000/2500，RaBitQ: error-bound-based 无参数）。
  - 实验比较（参数验证）：ε₀（置信区间参数）从 0.0 到 4.0 的 recall 曲线；B_q（查询量化位数）从 1 到 8 的 average relative error 曲线；无偏性验证（10^7 样本对的线性回归）。
  - 实验比较（索引阶段时间）：RaBitQ (117s) vs PQ (105s) vs OPQ (291s) vs LSQ (>24h timeout)，GIST 数据集 32 线程。

- 硬件平台是什么，配置是什么。
  - AMD Threadripper PRO 3955WX @3.9GHz（Zen2 微架构，支持 AVX2 SIMD），64GB RAM。C++ 由 g++ 9.4.0 编译，-Ofast -march=core-avx2，Ubuntu 20.04 LTS。查询时间单线程评估，索引时间 32 线程评估。所有方法优化至 AVX2 SIMD 指令集。

- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型——用于压缩高维向量和加速 ANN 查询。六个公开真实数据集：
    | 数据集 | 规模 | D | Query Size | 类型 |
    |---|---|---|---|---|
    | MSong | 992,272 | 420 | 200 | Audio |
    | SIFT | 1,000,000 | 128 | 10,000 | Image |
    | DEEP | 1,000,000 | 256 | 1,000 | Image |
    | Word2Vec | 1,000,000 | 300 | 1,000 | Text |
    | GIST | 1,000,000 | 960 | 1,000 | Image |
    | Image | 2,340,373 | 150 | 200 | Image |
  - 评价指标：Average Relative Error、Maximum Relative Error、Time per Vector（距离估计）；Recall@100、Average Distance Ratio、QPS（ANN 查询）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/gaoj0017/RaBitQ
  - RaBitQ 量化 pipeline：
    ```
    # === Index Phase ===
    # 输入：N 个 D 维数据向量 o_r，聚类数 nlist=4096
    # 1. KMeans 聚类：{c_k}, 将向量分配到桶
    # 2. 采样随机正交矩阵 P（随机高斯矩阵 + QR 分解）
    # 3. 对每个聚类 k:
    #    o = (o_r - c_k) / ||o_r - c_k||          # 单位化
    #    o' = P^{-1} @ o                           # 逆变换
    #    x̄_b[i] = 1 if o'[i] > 0 else 0           # 逐维度符号 → D-bit 字符串
    #    precompute: ||o_r - c_k||, <ō,o> = (1/√D)·Σ|o'[i]|
    # 存储：量化码 x̄_b (D bits) + ||o_r-c_k|| + <ō,o>

    # === Query Phase (单向量) ===
    # 输入：查询向量 q_r
    # 1. 选最近 nprobe 个聚类质心
    # 2. 对每个选中聚类 k:
    #    q = (q_r - c_k) / ||q_r - c_k||
    #    q' = P^{-1} @ q
    #    q' 随机化均匀标量量化 → q̄_u (B_q=4-bit unsigned integers)
    #    对每个候选数据向量：
    #      <x̄_b, q̄_u> = Σ_j 2^j · popcount(x̄_b & q̄_u^{(j)})  # bitwise 分解
    #      <ō,q> = (2Δ/√D)·<x̄_b,q̄_u> + (2v_l/√D)·popcount(x̄_b) - (Δ/√D)·Σq̄_u[i] - √D·v_l
    #      estimator = <ō,q> / <ō,o>                         # 无偏估计 ⟨o,q⟩
    #      dist_est² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·estimator
    #      error_bound = √((1-<ō,o>²)/<ō,o>²) · (ε₀/√(D-1))
    #      if dist_est² - error_bound > best_exact_dist²: 距离下界 > 当前最优 → 剪枝
    ```

    # === Query Phase (batch, FastScan SIMD) ===
    # 对批处理 32 个量化码：
    #   D 位字符串拆成 D/4 个 4-bit 子段
    #   预计算 D/4 个 LUT，每个 LUT 含 2^4=16 个值
    #   LUT 加载到 AVX2 256-bit 寄存器（每寄存器 2 个 LUT）
    #   SIMD shuffle → 查表 + 累加，一周期处理 32 个向量
    ```
  - 关键张量计算：
    - 码本向量：x ∈ C = {±1/√D}^D，最终码本 C_rand = P·C（不显式物化，维护矩阵 P 即可）
    - 量化码：x̄_b ∈ {0,1}^D，关系 x̄[i] = (2·x̄_b[i] - 1)/√D
    - 查询量化：q̄ = Δ·q̄_u + v_l·1_D，q̄_u[i] = ⌊(q'[i]-v_l)/Δ + u_i⌋, u_i~Uniform(0,1)
    - 内积分解：⟨x̄,q̄⟩ = (2Δ/√D)·⟨x̄_b,q̄_u⟩ + (2v_l/√D)·Σx̄_b[i] - (Δ/√D)·Σq̄_u[i] - √D·v_l
    - 距离估计器：⟨o,q⟩ ≈ ⟨ō,q⟩/⟨ō,o⟩ (无偏), 误差界 O(1/√D) w.h.p.
    - bitwise 实现：⟨x̄_b,q̄_u⟩ = Σ_{j=0}^{B_q-1} 2^j · popcount(x̄_b AND q̄_u^{(j)})
  - IVF 集成：聚类数 4,096，每个聚类独立归一化和量化。查询时选最近 nprobe 个聚类，对聚类内所有向量基于 error bound 判定是否需要 re-rank（精确距离计算）。无 re-ranking 参数需手工调参。
  - 关键参数：ε₀=1.9（控制置信区间，固定无需调参），B_q=4（查询量化位宽，固定无需调参），默认量化码长度 = ceil(D/64)×64 bits。

## Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：扩展 RaBitQ 量化方法，使其支持中等压缩率（B bits/dim, e.g. 4-8 bits, 对应 4x-8x 压缩率）。核心设计：(1) **新码本构造**：从 B-bit 无符号整数网格 G = {-(2^B-1)/2 + u | u=0,...,2^B-1}^D 中取向量，归一化后用随机正交矩阵 P 旋转，得到码本 G_r = {P·y/||y|| | y∈G}。该码本由随机旋转的单位向量组成，继承了 RaBitQ 的无偏估计和渐近最优误差界；(2) **量化编码算法 (Algorithm 1)**：对每个数据向量 o，计算 o' = P^{-1}o，通过枚举至多 D·2^{B-1} 个 critical values（使用最小堆维护），每次 O(1) 更新时间，找到码本中最近向量 ȳ，编码为无符号整数向量 ȳ_u = ȳ + (2^B-1)/2 · 1_D，总复杂度 O(2^B·D log D)；(3) **距离估计**：使用无偏估计器 ⟨ō,q⟩/⟨ō,o⟩ 估计内积 ⟨o,q⟩，⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u, q'⟩ - (2^B-1)/2 · Σq'[i])，其中 q'=P^{-1}q；(4) **两阶段距离比较**：利用量化码的最高有效位（等价于原始 RaBitQ 的二进制码 ȳ₀）先通过 FastScan SIMD 批量估计粗略距离，若足以判定该候选非 NN 则剪枝，否则访问剩余位 ȳ_last 增量计算高精度距离 ⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩。
  - 实验比较（距离估计精度）：RaBitQ(ext) vs RaBitQ(pad)、SQ（均匀标量量化）、LVQ（逐向量标量量化）、PQ/OPQ（k=8），B=1~10 bits/dim，六个数据集上评估 average relative error 和 maximum relative error。
  - 实验比较（ANN 查询性能）：RaBitQ(ext) + IVF vs LVQ + IVF（最竞争 baseline），B=3,4,5,7,8,9，评估 Recall、Average Distance Ratio、QPS，K=100。还包含可扩展性验证（MSMARCO ~100M 向量）、无偏性验证（线性回归拟合 slope=1, intercept=0）、经验公式常数测量（c_ε=5.75）。

- 硬件平台是什么，配置是什么。
  - 两台 Intel Xeon Gold 6418H @4.0GHz（Sapphire Rapids 架构，48 cores/96 threads），1TB RAM。C++ 源码由 GCC 11.4.0 编译，-Ofast -march=native，Ubuntu 22.04 LTS。搜索性能单线程评估，索引时间多线程评估。所有方法均优化至 AVX512 SIMD 指令。

- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型——该方法用于压缩高维向量和加速 ANN 查询。数据集均为公开真实数据集：
    | 数据集 | 规模 | 维度 | 类型 |
    |---|---|---|---|
    | MSong | 992,272 | 420 | Audio |
    | Youtube | 999,000 | 1,024 | Video |
    | OpenAI-1536 | 999,000 | 1,536 | Text (text-embedding-3-large) |
    | OpenAI-3072 | 999,000 | 3,072 | Text (text-embedding-3-large) |
    | Word2Vec | 1,000,000 | 300 | Text |
    | GIST | 1,000,000 | 960 | Image |
    | MSMARCO | 113,520,750 | 1,024 | Text (Cohere embed-english-v3) |
  - 评价指标：Average Relative Error, Maximum Relative Error（距离估计）；Recall, Average Distance Ratio, QPS（ANN 查询）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/VectorDB-NTU/Extended-RaBitQ
  - Extended RaBitQ 量化 pipeline：
    ```
    # === Index Phase (压缩数据向量) ===
    # 输入：N 个 D 维数据向量 o_r，bit 数 B
    # 1. 计算全局质心 c = mean(o_r)
    # 2. 采样随机正交矩阵 P（用随机高斯矩阵 + QR 分解）
    # 3. 中心化并归一化: o = (o_r - c) / ||o_r - c||
    # 4. 变换: o' = P^{-1} @ o   (每个向量右乘 P^{-1})
    # 5. Algorithm 1 量化编码（对每个向量 o'）:
    #    t=0, v_max=0, t_max=0
    #    初始化 y_cur, <y_cur,o'>, ||y_cur|| with t=0
    #    while 存在未枚举的 critical value:
    #      t = 下一个最小 critical value (来自 minheap, O(log D))
    #      更新 y_cur（仅一个维度变化，O(1)）
    #      更新 <y_cur,o'> 和 ||y_cur||（O(1)）
    #      if <y_cur,o'>/||y_cur|| > v_max:
    #        v_max = <y_cur,o'>/||y_cur||, t_max = t
    #    ȳ = round(t_max · o')   (逐维度最近整数, 裁剪到 [-2^{B-1}+0.5, 2^{B-1}-0.5])
    #    ȳ_u = ȳ + (2^B-1)/2 · 1_D   (存储为 B-bit 无符号整数)
    #    存储: ||o_r-c||, 1/(||ȳ_u||·<ō,o>), ȳ_u

    # === Query Phase (估计距离) ===
    # 输入：查询向量 q_r
    # 1. 变换: q' = P^{-1} @ ((q_r-c)/||q_r-c||)
    # 2. 对每个候选数据向量:
    #    # 第一阶段：仅用最高有效位 ȳ₀（等价 RaBitQ 二进制码）
    #    # FastScan SIMD 批量计算 <ȳ₀, q'>
    #    dist_est_1 = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·(<ō,q>_est_1)
    #    # 其中 <ō,q>_est_1 = (2/√D)·<ȳ₀,q'> - (1/√D)·Σq'[i]，再除以 <ō,o>
    #    if dist_est_1 下界 > 当前最优距离:
    #      剪枝，跳过该候选
    #    else:
    #      # 第二阶段：访问剩余位 ȳ_last，增量计算
    #      <ȳ_u,q'> = 2^{B-1}·<ȳ₀,q'> + <ȳ_last,q'>
    #      <ō,q>_est_full = (1/||ȳ||)·(<ȳ_u,q'> - (2^B-1)/2 · Σq'[i])
    #      dist_est_full 用于最终 NN 选择
    ```
  - 关键张量计算（以 B=4, 8x 压缩为例）：
    - 原始 RaBitQ：ō₀ = P·(2/√D · x̄_b - 1/√D · 1_D)，其中 x̄_b ∈ {0,1}^D
    - Extended RaBitQ：ō = P·(ȳ/||ȳ||)，ȳ ∈ {-(2^B-1)/2 + u}^D，码本大小 2^{B·D}
    - B=1 时退化为原始 RaBitQ（码本 = 超立方体顶点 ±1/√D）
    - 误差经验公式：ε < 5.75 · 2^{-B} / √D（>99.9% 置信度）
  - IVF 集成：聚类数 = 4,096（百万级）/ 262,144（亿级 MSMARCO），每个聚类用本地质心中心化，扫描最近 nprobe 个聚类中的所有向量。

## PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PassionSR 是面向 one-step diffusion (OSD) 图像超分模型的 post-training quantization (PTQ) 方法。核心包含三个设计：(1) **模型简化**——将 OSEDiff 的 DAPE-CLIPEncoder 分支替换为基于空字符串的常数 embedding，得到仅包含 UNet 和 VAE 的 PassionSR-FP（参数减少 27.13%，操作减少 6.25%，性能几乎无损）；(2) **Learnable Boundary Quantizer (LBQ)**——使用可训练的上下界参数 B_l, B_u 替代传统 fixed-range 量化器，通过 fake quantization 模拟量化误差，量化与反量化公式为 X_c = Clip(X, B_l, B_u), α = (B_u-B_l)/(2^N-1), β = B_l, X_I = round((X_c-β)/α), X_q = α·X_I+β；(3) **Learnable Equivalent Transformation (LET)**——在线性层、卷积层和注意力矩阵乘法中引入逐通道可学习 scale factor s 和 offset δ，通过等效变换（Linear: W̃ = s⊙W, X̃ = (X-δ)⊘s, B̃ = B+δW；Attention: Q̃ = Q⊘s, K̃ = s⊙K）重新分布激活值以抑制离群值，s 和 δ 可融入前序层或权重/偏置中，无额外推理开销；(4) **Distributed Quantization Calibration (DQC)**——将标定过程分为两阶段：Stage 1 仅训练 LET 的 scale/offset，Stage 2 重新初始化 LBQ 并联合训练，从而稳定训练、加速收敛并降低 GPU 显存。
  - 实验比较：PassionSR (W8A8/W6A6) vs **MaxMin** (传统 min-max 量化)、**LSQ** (learned step size QAT)、**Q-Diffusion** (多步扩散量化 PTQ)、**EfficientDM** (QALoRA-based 量化微调)。所有对比方法均基于 PassionSR-FP backbone 进行量化。在 RealSR、DRealSR、DIV2K val 三个数据集上评估 PSNR/SSIM/LPIPS/DISTS（参考 IQA）和 NIQE/MUSIQ/MANIQA/CLIP-IQA（非参考 IQA）。消融实验验证 LBQ、LET、DQC 各组件的贡献。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明 GPU 型号。消融实验表 4 显示 GPU 显存占用为 28-40 GB，推测使用 NVIDIA A100 (40GB) 或类似高端 GPU。软件环境：CUDA 11.8 + PyTorch 2.0.1。标定训练使用单卡（CUDA_VISIBLE_DEVICES="0"）。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：以 OSEDiff (NeurIPS 2024) 为 backbone，简化后得到 PassionSR-FP（仅含 UNet + VAE，FP32 下参数 949M、操作 4,240G）。量化设置：PassionSR-U（仅 UNet 量化）和 PassionSR-UV（UNet+VAE 量化），W8A8 和 W6A6 两种精度。
  - **标定数据集**：从 DIV2K train 中随机裁剪 500 对 128×128 LR-HR 图像。
  - **测试数据集**：RealSR、DRealSR、DIV2K val。全尺寸图像评估。
  - **评估指标**：PSNR、SSIM、LPIPS、DISTS（参考 IQA）；NIQE、MUSIQ、ManIQA、CLIP-IQA（非参考 IQA）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/libozhu03/PassionSR（CVPR 2025），含 PTQ 标定脚本 `ptq_quantize_single.py`、推理脚本 `inference_single.py`、评估脚本 `measure.py`、YAML 配置及预训练模型链接（Google Drive）。
  - PassionSR 量化 pipeline（以 OSEDiff 的 UNet + VAE 为例，标定 W8A8）：
    ```
    # === Step 1: 模型简化 ===
    # 将 OSEDiff 的 text embedding 分支替换为空字符串的 ClipEncoder 输出常数
    # 删除 DAPE 模块，仅保留 UNet + VAE → PassionSR-FP

    # === Step 2: 构建 LBQ 量化器 ===
    # 对每层权重 W 和激活 X，定义可训练上下界 B_l, B_u
    # 前向: X_c = clamp(X, B_l, B_u)
    #       alpha = (B_u - B_l) / (2^bit - 1)
    #       X_q = alpha * round((X_c - B_l) / alpha) + B_l
    # STE 反向传播梯度

    # === Step 3: LET 等效变换 ===
    # 对 Linear 层 (X ∈ R^{N×C_in}, W ∈ R^{C_in×C_out}):
    #   s = exp(param_s), delta = param_delta  # 可学习参数
    #   W_tilde = s ⊙ W                          # 按元素乘
    #   X_tilde = (X - delta) ⊘ s               # 按元素除
    #   B_tilde = B + delta @ W
    # 对 Conv 层: 沿 channel 维度应用相同变换
    # 对 Attention (Q,K,V 矩阵乘法):
    #   Q_tilde = Q ⊘ s, K_tilde = s ⊙ K
    #   P_q = Softmax(Q_a1(Q_tilde) · Q_a2(K_tilde^T))
    # 变换后 s,δ 合并入前层/权重，无额外推理开销

    # === Step 4: DQC 两阶段标定 ===
    # Stage 1: 冻结 LBQ，仅训练 LET 的 s 和 δ
    #   for epoch in range(2):
    #       for X_lr, X_hr in calib_loader:
    #           Y_q = quantized_forward(X_lr)   # LBQ(fixed) + LET(trainable)
    #           Y_fp = fp_forward(X_lr)
    #           loss = MSE(Y_q, Y_fp)           # 模块级逐层标定
    #           loss.backward()                 # 仅更新 LET 参数
    # Stage 2: 重新初始化 LBQ，联合训练 LBQ + LET
    #   for epoch in range(2):
    #       for X_lr, X_hr in calib_loader:
    #           Y_q = quantized_forward(X_lr)   # LBQ+LET 均可训练
    #           loss_unet = ||I(Z_lq, ε_q) - I(Z_l, ε_fp)||_2  # latent space MSE
    #           loss_vae_e = ||V_qe(X_fp) - V_fpe(X_fp)||_2
    #           loss_vae_d = ||V_qd(X_q) - V_fpd(X_fp)||_2
    #           loss.backward()                 # 更新 LBQ 和 LET 参数

    # 标定完成，scale/offset 合并入权重，得到 INT8 推理模型
    # 推理: HR ≈ VAE_decoder(UNet_int8(VAE_encoder_int8(LR)))
    ```

## ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：ParoQuant 是一种后训练权重量化（PTQ）方法，结合**hardware-efficient independent Givens rotations**（独立 Givens 旋转）与 **channel-wise scaling**（逐通道缩放），在量化前对权重进行可学习变换以抑制离群值（outliers）。核心包含三个设计：(1) **Scaled Pairwise Rotation Transform**——将权重矩阵按 group size=128 分组，每组内先应用逐通道缩放（diag(α)·W）拉平各通道幅值，再依次应用 K=8 个独立旋转（每个旋转由最多 64 对互不重叠的 Givens 旋转组成），使同一量化组内动态范围收窄；(2) **两阶段逐层优化**——Stage 1 用 AdamW 优化旋转角度 θ 和缩放因子 α 以最小化量化层输出误差 ||Q(l)(X') - l(X)||，Stage 2 采用 QAT-like 方式微调权重和量化参数 (s, z) 进一步消除残留离群值；(3) **算法-系统联合设计**——旋转核与反量化 GEMM kernel 融合，推理时对激活应用逆变换 T^{-1} = (R_1^{-1}...R_K^{-1})·diag(1/α)，计算仅涉及成对向量化乘加指令。
  - 实验比较：ParoQuant (W4A16, INT4, group=128) vs **AWQ** (channel-wise scaling, grid search)、**EfficientQAT** (layer-wise fine-tuning, SOTA 线性量化)、**QTIP** (随机 Hadamard + trellis 量化, SOTA 向量量化)、**QuIP#** (Hadamard + lattice codebook)、**OmniQuant** (可学习平滑参数)、**SpinQuant** (可学习旋转)。在 LLaMA-2-7B、LLaMA-3-8B/70B、LLaMA-3.1 Instruct 8B、DeepSeek-R1-Distill-LLaMA-3.1-8B、Qwen3-1.7B/4B/8B/14B 上评估 WikiText2/C4 困惑度、MMLU-Pro/GPQA Diamond/AIME-24/AIME-25（推理任务）、BoolQ/ARC-C/ARC-E/HellaSwag（非推理任务）。消融实验验证各组件（scaling, rotations, Stage 2 fine-tuning）的贡献，以及校准集大小（128-2048）、旋转数量（2-8）的影响。

- 硬件平台是什么，配置是什么。
  - 训练/量化优化：单张 NVIDIA H200 GPU。PyTorch 2.8.0 + Transformers 4.55.2 + Datasets 3.6.0。校准集 2048 samples × 2048 tokens（均匀采样自 WikiText2、C4、RedPajama），验证集 64 samples from Pile。Batch size=16（70B 模型减半），AdamW 优化器，learning rate: 旋转角度和缩放 0.05、权重 10^{-5}、scales/zero-point 10^{-6}，cosine 衰减至 1/20。量化 LLaMA-3-8B 耗时约 9 小时。
  - 推理测速：NVIDIA RTX A6000 (48GB)、RTX 6000 Ada (48GB)、RTX 4090 (24GB)，batch size=1 decode，PyTorch 2.6.0 + torch.compile max-autotune + CUDA Graphs。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LLaMA-2-7B、LLaMA-3-8B、LLaMA-3-70B、LLaMA-3.1-Instruct-8B、DeepSeek-R1-Distill-LLaMA-3.1-8B、Qwen3-1.7B/4B/8B/14B（Base 预训练模型）。所有线性层权重被量化为 W4A16（INT4 group=128）。
  - **校准数据集**：WikiText2 + C4 + RedPajama 各 1/3，共 2048 样本 × 2048 tokens。验证集：Pile 64 样本。
  - **PPL 评估**：WikiText2、C4（test split，LLaMA-3/Qwen3 context 8192，LLaMA-2 context 4096）。
  - **推理 Benchmark**：MMLU-Pro (12k samples，seed=42)、GPQA Diamond (198 samples，seeds=42/0/1)、AIME-24 (30 samples，seeds=42/0/1)、AIME-25 (30 samples，seeds=42/0/1)。使用 Lighteval 0.8.1 + vLLM 0.10.1。
  - **非推理 Benchmark**：BoolQ、ARC-Challenge、ARC-Easy、HellaSwag，使用 lm-eval-harness 0.4.9.1，batch size=32。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/z-lab/paroquant（MIT License，ICLR 2026 接收），PyPI `pip install paroquant`，HuggingFace 上提供已量化模型（如 z-lab/Meta-Llama-3-8B-PARO、z-lab/DeepSeek-R1-Distill-Llama-8B-PARO）。
  - ParoQuant W4A16 量化 pipeline（以 LLaMA-3-8B 单层为例，W ∈ R^{C_in×C_out}，group=128）：
    ```
    # === Stage 0: 分组与配对选择 (Algorithm A1) ===
    # 将 W 沿 channel 维度按 group size g=128 切分为子矩阵
    # 对每个 group (128×C_out)，独立选择 K=8 个 independent rotations，
    #   每个 rotation 选 N=64 对互不重叠的 channel pairs
    for each group W_g ∈ R^{128×D}:
        P_all = shuffle({(i,j) | 1≤i<j≤128})  # 所有可能的配对
        A[i,j] = 1 for i≠j  # 全局可用性矩阵
        for r = 1 to K:
            A_rot = copy(A)  # 当前 rotation 内的可用性
            for each pair (i,j) in P_all:
                if |P_r| ≥ N: break
                if A_rot[i,j]==0: continue
                append (i,j) to P_r
                A_rot[i,:]=0; A_rot[:,i]=0  # 禁用通道 i
                A_rot[j,:]=0; A_rot[:,j]=0  # 禁用通道 j
                A[i,j]=0; A[j,i]=0  # 全局禁用

    # === Stage 1: 优化旋转和缩放 (Algorithm A2) ===
    # 对每个 decoder layer l，逐层优化：
    for each layer l with calibration input X' (已量化的前层输出):
        Y = l(X)  # 原始层输出作为标签
        for each linear in l:
            # 对每个 group 应用 scaled pairwise rotation:
            α = ones(128)       # channel-wise scaling (初始化为1)
            θ = zeros(K×N)     # rotation angles (初始化为0)
            T(W_g) = (∏_{t=1}^{K} R(P_t, θ_t)) · diag(α) · W_g
            # R(P_t, θ_t) 是第 t 个 independent rotation，
            #   由 |P_t| 个互不重叠的 Givens 旋转组成
            #   每个 Givens 旋转: 
            #     W'[i,:] = cosθ·W[i,:] - sinθ·W[j,:]
            #     W'[j,:] = sinθ·W[i,:] + cosθ·W[j,:]
        
        # 量化变换后的权重:
        s = (max(T(W)) - min(T(W))) / 15   # INT4 量化步长
        z = -round(min(T(W)) / s)           # 零点
        W_q = clamp(round(T(W)/s) + z, 0, 15)
        
        # 优化目标: min_{α,θ} ||l'(X') - Y|| (SmoothL1Loss)
        #   where l' 使用量化后的权重，推理时对激活应用 T^{-1}:
        #     T^{-1}(X) = X · diag(1/α) · (R_1^{-1}) · ... · (R_K^{-1})
        #     R_t^{-1}: 逆序 Givens 旋转，角度取 -θ
        optimize α, θ with AdamW (lr=0.05, 10 epochs)

    # === Stage 2: 权重和量化参数微调 ===
    # Stage 1 后大部分离群值已消除，但仍可能有孤立离群值
    # 进一步微调权重 W 和量化参数 (s, z):
    for each layer l:
        optimize W, s, z with AdamW (lr=1e-5 for W, lr=1e-6 for s/z, 10 epochs)
        # 损失函数同 Stage 1, SmoothL1Loss

    # === 推理时 (Inference) ===
    # 对每个 linear layer Y = X·W + b:
    # 1. 应用逆变换到激活: X' = T^{-1}(X)
    #    - 在 fused CUDA kernel 中完成 (见 kernel调度)
    # 2. INT4 GEMM: Y = dequant(W_q) @ X'^T + b
    #    - 使用 AWQ 的 W4A16 GEMM kernel
    ```
  - 最终效果：在推理任务上 ParoQuant 平均精度仅下降 0.9%，相比 AWQ 提升 2.4%、比 EfficientQAT 提升 6.3%，且仅比 AWQ 慢约 10%，比 QTIP 快约 25%。



## PT²-LLM Post-Training Ternarization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PT²-LLM 是一种后训练三值量化（Post-Training Ternarization）框架，将 LLM 权重量化到 {−1, 0, +1} 三值（等效 1.58-bit）。核心包含三个组件：(1) **非对称三值量化器（Asymmetric Ternary Quantizer, ATQ）**——引入逐行偏移 μ 捕获权重非对称分布，通过两阶段无训练优化：**Iterative Ternary Fitting (ITF)** 交替求解最优三值网格参数 (α, μ)（闭式解）和最优三值赋值 T（逐元素弹性舍入），迭代到收敛（~10 轮）；**Activation-aware Grid Alignment (AGA)** 利用校准数据激活统计量 C = Σ XX^T，以输出误差 E_x = ||WX - ŴX||² 为目标闭式求解更优的网格参数 (α, μ)，冻结 T 避免过拟合。(2) **结构相似性重排序（Structural Similarity-based Reordering, SSR）**——在 GPTQ 逐块量化框架中，每次选择下一块时基于残差矩阵的列间余弦相似度选取 top-k 最相似的列组成量化块，使块内权重分布更紧凑、方差更小，抑制离群值影响。
  - 实验比较：PT²-LLM (1.58-bit) vs **GPTQ** (2-bit) vs **AWQ** (2-bit) vs **QuIP** (2-bit) vs **Slim-LLM** (混合精度 2-bit SOTA) vs **PB-LLM** (1.7-bit，位宽最接近的 baseline)，在 LLaMA-7B/13B/65B、LLaMA-2-7B/13B/70B、LLaMA-3-8B、Qwen3-14B-Base 上比较 WikiText2/C4 困惑度（PPL）和 7 个零样本 QA 任务（PIQA, ARC-e, ARC-c, HellaSwag, Winogrande, OBQA, BoolQ）的准确率。消融实验验证 ITF、AGA、SSR 各自贡献，以及校准集大小和类型的影响。额外比较压缩时间、模型大小（Size Reduction）、推理吞吐（llama.cpp 上 prefill/decode/end-to-end 吞吐对比 2-bit 模型）。

- 硬件平台是什么，配置是什么。
  - 单张 Nvidia A800-80GB GPU。PTQ 校准阶段：WikiText2（128 个随机 2048-token 片段作为校准集），block size=128，无需训练或梯度反传。压缩 LLaMA-7B 耗时约 32 分钟。推理测速：llama.cpp（https://github.com/ggml-org/llama.cpp）在 Nvidia A800 GPU 上，序列长度 128（prefill）、256（decode）、128+256（end-to-end）。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LLaMA-1（7B, 13B, 65B）、LLaMA-2（7B, 13B, 70B）、LLaMA-3（8B）、Qwen3-14B-Base。所有线性层权重被量化。
  - **校准数据集**：WikiText2（128 个 2048-token 片段）。
  - **评估数据集（PPL）**：WikiText2、C4。
  - **评估 Benchmark（零样本 QA）**：PIQA、ARC-easy、ARC-challenge、HellaSwag、Winogrande、OpenBookQA、BoolQ。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/XIANGLONGYAN/PT2-LLM（Apache 2.0，论文被 ICLR 2026 接收，代码和量化模型标记为将发布，截至检查尚未上传完整实现）。arXiv: 2510.03267。
  - PT²-LLM 量化流程（以 LLaMA-7B 单层线性层 W∈R^{n×m} 为例，block size=128）：
    ```
    # === SSR: Structural Similarity-based Reordering ===
    # 输入: W ∈ R^{n×m}, block_size k=128
    # 逐块处理，每次根据残差矩阵列间余弦相似度选块：
    for col_start in range(0, m, k):
        # 计算残差矩阵的列均值参考向量
        w_bar = mean(W_remaining, axis=1)  # (n,)
        # 计算剩余每列与 w_bar 的余弦相似度
        sim_j = (W_remaining[:,j] · w_bar) / (||W_remaining[:,j]||₂ * ||w_bar||₂)
        # 选 top-k 相似列作为当前量化块
        B = top_k_columns(sim_j, k)
        # 对该块执行 ATQ 量化 + GPTQ 误差补偿
        quantize_and_compensate(B)
    
    # === ATQ: Asymmetric Ternary Quantizer (对每个 block) ===
    # 输入: W_block ∈ R^{n×k}, X_calib ∈ R^{B×L×k}
    # Step 1: 非对称初始化
    μ = row_mean(W_block)                          # (n,): 逐行均值
    W_tilde = W_block - μ                           # 中心化
    Δ = 0.75 * row_mean(|W_tilde|)                 # (n,): 阈值估计
    T_ij = 1 if W_tilde_ij > Δ_i else (-1 if W_tilde_ij < -Δ_i else 0)
    α = Σ_j(T_ij * W_tilde_ij) / Σ_j(|T_ij|)       # (n,): 最优缩放
    
    # Step 2: ITF — 迭代三值拟合 (约10轮收敛)
    while T != T_prev:
        # 闭式求解最优网格 (α*, μ*)
        α* = (m*(W∘T)1 - (T1)∘(W1)) / (m*(T∘T)1 - (T1)²)
        μ* = ((T∘T)1∘(W1) - (T1)∘((W∘T)1)) / (m*(T∘T)1 - (T1)²)
        # 弹性舍入更新 T
        Z_ij = (W_ij - μ*_i) / α*_i
        T_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
    
    # Step 3: AGA — 激活感知网格对齐
    C = Σ_b Σ_l X_{bl} X_{bl}^T                        # 激活协方差
    d = 1^T C 1                                        # 标量
    v = T C 1                                           # (n,)
    α* = (d*(W∘T)S1 - v∘(WS1)) / (d*T²S1 - v²)         # 激活感知闭式解
    μ* = (T²S1∘(WS1) - v∘((W∘T)S1)) / (d*T²S1 - v²)
    # T 冻结不更新，避免过拟合
    
    # 输出: Ŵ = α* T + μ*  (每行仅3个可能值: {-α_i+μ_i, μ_i, α_i+μ_i})
    ```
  - 关键公式：量化后权重 Ŵ = αT + μ，其中 α 为逐行缩放因子、μ 为逐行偏移、T∈{−1,0,+1}^{n×m}。存储时仅需保存 α（n个float）、μ（n个float）和 T（n×m 个 2-bit 索引），理论位宽 ≈ 1.58 bit/权重。

## PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PTQ1.61 是一种极低位（1.61-bit）后训练量化（PTQ）方法，包含三个核心创新：(1) **一维结构化掩码（Structured Mask）**——基于输入激活的通道级幅值，通过数学推导证明量化误差上界与输入激活通道幅值强相关，提出按通道保留 top-20% 显著权重为 4-bit、其余二值化为 1-bit，掩码仅额外引入 0.0002-bit/权重；(2) **分块缩放因子优化（Block-wise Scaling Factors Optimization）**——将缩放因子设为可学习参数，联合 MSE loss（幅值差距）和负对数余弦相似度 loss（方向偏差）进行分块优化，考虑行间隐式相关性和角度偏差；(3) **量化预处理（Quantization Preprocessing）**——使用轻量级 restorative LoRA（rank=64, 20K steps）在预训练数据集 RedPajama 上微调，将显著权重的分布从散乱模式转化为行集中模式，使模型更适合逐通道 PTQ。
  - 实验比较：PTQ1.61 vs **PB-LLM**（1.7+1 bit，10% 8-bit + 非结构化掩码）vs **BiLLM**（1+1.1 bit，多组二值化 + 非结构化掩码）vs **OmniQuant**（2-bit）vs **AWQ**（2-bit）vs **GPTQ**（2-bit）vs **QuIP**（2-bit），在 LLaMA/LLaMA-2/LLaMA-3/OPT 系列模型上比较 WikiText2/C4 困惑度（PPL）和 8 个推理 benchmark 的零样本准确率。消融实验验证结构化掩码、可学习缩放因子和量化预处理各自贡献。

- 硬件平台是什么，配置是什么。
  - 2 张 Nvidia A800 GPU。PTQ 阶段：校准集来自 WikiText2（128 个随机 2048-token 片段），分块训练 20 epochs，batch size=1。量化预处理阶段：LoRA rank=64，20K steps，单张 A100 GPU 耗时 <1.2 小时。整体 PTQ1.61 在 LLaMA-7B 上总耗时约 2h，GPU 内存 15GB；LLaMA-13B 上约 4.2h，GPU 内存 19GB。使用 lm-evaluation-harness 工具包进行推理评估。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LLaMA-1（7B, 13B, 30B, 65B）、LLaMA-2（7B, 13B, 70B）、LLaMA-3（8B）、OPT（2.7B, 6.7B, 13B）。所有线性层权重均被量化。
  - **数据集/Benchmark（语言生成）**：WikiText2、C4（困惑度 PPL 评估）。
  - **数据集/Benchmark（推理）**：PIQA、ARC-e、ARC-c、HellaSwag、Winogrande、Race、LAMBADA（使用 lm-evaluation-harness）；MMLU、GSM8K、LongBench（附录）。
  - **数据集（预处理）**：RedPajama（LLaMA 系列的预训练数据集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/zjq0455/PTQ1.61
  - PTQ1.61 量化流程（以 LLaMA-7B 单层线性层 W∈R^{4096×4096} 为例）：
    ```
    输入: 预训练权重 W, 校准数据 X (128×2048 tokens)
    阶段1 - 量化预处理（可选）:
      W_preprocessed = W + LoRA(W, RedPajama)  // rank=64, 20K steps
      // 目标: 将显著权重转化为行集中分布模式
    
    阶段2 - 结构化掩码生成:
      对每层计算输入激活 X 的通道幅值 ||x_i||  // x_i ∈ R^n
      选择 top-20% 通道作为显著通道 → mask ∈ {0,1}^{4096×1}  // 一维掩码
      // 仅额外 0.0002-bit/权重
    
    阶段3 - 分块优化量化:
      for each transformer block:
        // 显著通道量化
        W_q[salient] = round(W[salient] / S_q) + Z_q  // 4-bit 量化, Eq.(1)
        // 非显著通道二值化
        W_q[non-salient] = α * sign(W[non-salient])   // 1-bit, Eq.(2)
        // α 为可学习缩放因子, 初始化 α^w = ||w||_1 / n_w
        for epoch = 1 to 20:
          前向: X_q → Block_quant → output_q
          损失: L = ||output_fp - output_q||_2  // MSE
                 + (-log(cos_sim(output_fp, output_q)))  // NLC loss, Eq.(6)
          更新: α = AdamW(L, lr=5e-4 或 1e-3)  // 分块优化, Eq.(7)
        end for
      end for
    输出: 量化权重 W_q (平均 1.61-bit)
    ```
  - 量化误差上界推导（Section 3.2, Eq.(4)）：E = |X(W_q^T - W^T)| ≤ Σ_i (|x_i| * Σ_j |w_{i,j}^q - w_{i,j}|)，证明第 i 通道的量化误差上界与输入激活幅值 |x_i| 和权重行量化误差乘积成正比。由于激活幅值约为权重的 1000 倍（尤其 top-20% 通道），因此保护高激活通道对应的权重行可最大程度降低量化误差上界。
  - 分块优化目标函数（Eq.(7)）：min_{α_s, α_r} E(F(W_q'), X) + E(F(W_q'), X_q)，其中第一分支减轻量化误差传播（量化和全精度 block 输出对比），第二分支量化同一输入对量化前后 block 的输出差异。W_q' 为考虑缩放因子的反量化权重（Eq.(9)）。
  - 推理时内存对比：LLaMA-7B 量化后 PTQ1.61 仅需 1.41GB（vs PB-LLM 2.36GB, BiLLM 1.83GB），因为无需加载额外的非结构化掩码。

## PARQ Piecewise-Affine Regularized Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PARQ 是一种 QAT 方法，核心贡献包括：(1) **凸分段仿射正则化（PAR）**——定义 PAR(w)=max_k{a_k(|w|-q_k)+b_k}，其中斜率 a_k 满足 0≤a_0<a_1<...<a_m=+∞，具有严格递增斜率的分段仿射凸函数，在非光滑点（±q_k）处产生聚类效应；(2) **AProx（Aggregate Proximal Gradient）算法**——用累积步长 γ_t=Ση_s 缩放 proximal map 替代标准 Prox-SGD 中逐次缩放的 proximal map，使软量化（slanted segments）随训练逐步收敛到硬量化（hard quantization），解决了 Prox-SGD 中随 η_t→0 的 diminishing regularization 问题；(3) **PARQ 实用实现**——使用 LSBQ（Least Squares Binary Quantization）在线估计目标量化值 Q，通过独立的逆斜率 schedule ρ_t^{-1}（cosine decay 或 sigmoid decay）从软量化渐近到硬量化，无需预先指定 λ 和 {a_k}。
  - 实验比较：PARQ vs **STE/BinaryConnect**（Courbariaux et al. 2015，全程使用硬量化映射）vs **BinaryRelax**（Yin et al. 2018，非凸正则化，slanted segment 斜率逐步减小至 0），在 5 种位宽（ternary T、1-4 bits）下的分类准确率比较。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号和硬件配置。使用 PyTorch 框架进行训练，训练设备为标准 GPU（论文作者来自 Meta FAIR，使用公司内部 GPU 集群）。开源代码 https://github.com/facebookresearch/parq 可在单 GPU 或多 GPU 环境运行。

- 模型是什么。数据集和bench分别是什么。
  - **模型**: ResNet-20、ResNet-56（CIFAR-10 实验）；ResNet-50（ImageNet 实验）；DeiT-Ti（5M 参数）、DeiT-S（22M 参数）、DeiT-B（86M 参数）（ImageNet 实验）。所有模型权重均被量化，卷积/注意力 block 权重 per-channel 量化（row-wise over tensors）。对 DeiT，embedding、layer normalization 参数和最终 projection 权重保持全精度。
  - **数据集/Benchmark**: CIFAR-10（图像分类）、ImageNet（ILSVRC 2012，图像分类）。每项实验 3 次随机种子取平均，报告 mean ± std。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：PyTorch 包 https://github.com/facebookresearch/parq，实现 PARQ 及多种主流 QAT 方法（STE/BinaryConnect、BinaryRelax），可复现论文结果。
  - PARQ 算法核心流程（Algorithm 1）：
    ```
    输入: w¹ ∈ R^d, 量化比特数 n, 步长 {η_t}_{t=1}^T, 逆斜率 schedule {ρ_t^{-1}}_{t=1}^T
    初始化: u¹ = w¹
    for t = 1, 2, ..., T-1 do
        u^{t+1} = u^t - η_t ∇f(w^t, z^t)        // 前向步: 在量化参数 w^t 处计算梯度
        Q^{t+1} = LSBQ(u^{t+1}, n)               // 在线估计目标量化值
        w^{t+1} = prox_PARQ(u^{t+1}, Q^{t+1}, ρ_t) // 聚集 proximal 映射
    end for
    输出: w^T
    ```
  - LSBQ 估计 Q：对于 n-bit 量化，将 u∈R^d 近似为 w_i = Σ_{j=1}^n v_j s_j(u_i)，其中 v_j 为递减的正标量，s_j(u_i)∈{-1,1} 为二进制函数。解通过 greedy foldable representation 获得：s_j(u_i)=sgn(u_i-Σ_{ℓ=1}^{j-1} v_ℓ s_ℓ(u_i))。Q={±q_1,...,±q_m} 为 v_j 的组合（如 q_m=v_1+...+v_n），|Q|=2^n。
  - prox_PARQ 结构（图 9a）：与 AProx 的渐进性不同，PARQ 使用独立斜率 ρ_t。ρ_t^{-1} 从 1 单调递减到 0（cosine decay 或 sigmoid），使得 proximal map 从接近 identity（训练初期，slope≈1）逐步过渡到硬量化（训练末期，slope→∞）。这避免了因为有限训练迭代次数导致 γ_t 不够大的问题。
  - AProx 理论核心：与 Prox-SGD 的区别在于，Prox-SGD 中的 u^{t+1}=w^t-η_t g^t（w^t 已含过往正则化贡献），使用近端正则化 η_t λ Ψ 平衡单步梯度；而 AProx 中的 u^{t+1}=u^t-η_t g^t（u^t 仅累加梯度），使用聚集正则化 γ_t λ Ψ 平衡所有过往梯度。AProx 等价于 ProxConnect（Dockhorn et al. 2021），但通过凸 PAR（而非任意单调 proximal map）提供了更强的收敛保证。证明了最后迭代（last-iterate）收敛 O(ln(t)/√t)，而非仅平均迭代收敛。
  - 训练细节：ResNet 使用 SGD（momentum=0.9, weight decay=2e−4），200 epochs，lr=0.1 在 epoch 80/120/150 除以 10；DeiT 使用 AdamW（lr=5e−4, weight decay=0.05），300 epochs，最后 20 epochs 将 lr 固定在 1e−8。DeiT 使用 RandAugment、mixup 和 CutMix（不含 repeated augmentation）。


## Optimal and Approximate Adaptive Stochastic Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出三个 ASQ (Adaptive Stochastic Quantization) 算法：(1) **QUIVER**——利用预处理数组（β/γ 累积和）使 C[k,j] 可 O(1) 计算，证明 C 满足 quadrangle inequality 从而使 DP 矩阵成为 totally monotone matrix，应用 SMAWK 算法以 O(d) 时间找到行最小值，将 ASQ 动态规划从 O(s·d²) 时间/O(d²) 空间优化到 O(s·d) 时间/O(s·d) 空间；(2) **Accelerated QUIVER**——推导 s=3 时中间量化值的闭式解（C²[k,j] 可通过 b*_{k,j} 公式 O(1) 计算），每次 SMAWK 调用跳过两个量化值而非一个，将 SMAWK 调用次数从 s-2 减半至 ⌊s/2⌋-1，速度提升最高 5.4×；(3) **Apx. QUIVER**——将量化值候选集离散化为 m+1 个均匀网格点，用 histogram-style 预处理计算 C_m[k,j]，运行复杂度 O(d + m·s)，提供保证 AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)。
  - 实验比较：(a) Exact 对比：Accelerated QUIVER vs ZipML 精确解，不同 d (2¹⁰ 到 2²³) 和 s (2 到 16) 下的 runtime 和 vNMSE；(b) Approximate 对比：Apx. QUIVER vs ZipML-CP Uniform、ZipML-CP Quantiles、ZipML 2-Apx、ALQ；(c) 分布泛化：LogNormal、Normal、Exponential、TruncNorm、Weibull 五种分布；(d) 加速比消融：Accelerated QUIVER vs QUIVER 在不同 s 和 d 下的加速比。

- 硬件平台是什么，配置是什么。
  - AWS g4dn.4xlarge EC2 实例，custom Intel Cascade Lake CPU，64 GB RAM，Ubuntu 22.04 OS。GPU 排序/量化的额外开销测量使用同实例上配备的 NVIDIA T4 GPU，PyTorch v2.1.2，CUDA toolkit v12.3。

- 模型是什么。数据集和bench分别是什么。
  - 本研究不涉及具体 ML 模型，而是以合成向量作为输入。向量条目为 i.i.d. 采样自五种分布：**LogNormal(0,σ²)**、**Normal(0,1)**、**Exponential(1)**、**Truncated Normal(μ=0,σ²=1,a=-1,b=1)**、**Weibull(1,1)**。这些分布被已有工作报道为能刻画 DNN gradients、模型权重和激活值的分布特征。向量维度 d 范围 2¹⁰ 到 2²⁴（约 16M）。评估指标为 **vNMSE**（vector normalized MSE = E[‖X-X̂‖²]/‖X‖²）和 **runtime**。每个实验 5 次随机种子取平均。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：C++ 实现 https://github.com/ranbenbasat/QUIVER。
  - QUIVER 算法核心流程（Algorithm 1）：
    ```
    输入: 已排序向量 X ∈ R^d, 量化值个数 s
    输出: 最优量化值集合 Q ⊆ X, |Q|=s
    
    1. Preprocess(X): 计算 β_j=Σ_{i=1}^j x_i, γ_j=Σ_{i=1}^j x_i², ∀j∈[1,d]
       // 使 C[k,j] = -x_j·x_k·(j-k) + (x_j+x_k)·(β_j-β_k) - (γ_j-γ_k) 可O(1)计算
    2. for j=2 to d: MSE[2,j] = C[1,j]  // 初始化 (i=2 行)
    3. for i=3 to s:
         // 隐式定义矩阵 A[k,j] = MSE[i-1,k] + C[k,j]
         // C 满足 quadrangle inequality → A 是 totally monotone
         K[i,·] = SMAWK(A)  // O(d) 时间找到每列的行最小值索引
         MSE[i,j] = MSE[i-1, K[i,j]] + C[K[i,j], j], ∀j∈[i..d]
    4. Q = {x₁, x_d}, j = d
    5. for i=s down to 3:
         j = K[i, j]; Q = Q ∪ {x_j}
    6. return Q
    ```
  - Accelerated QUIVER 关键加速（s=3 闭式解）：
    ```
    // C²[k,j] = C[k, b*_{k,j}] + C[b*_{k,j}, j]  在 O(1) 计算
    b*_{k,j} = ⌈(j·x_j - k·x_k - (β_j - β_k)) / (x_j - x_k)⌉
    
    // 推导：对区间 [x_k,x_j] 中间插入 q，Q(q) 为两段方差和
    // dQ/dq = Σ_{x∈[x_k,q]}(x-x_k) - Σ_{x∈(q,x_j]}(x_j-x)
    // 极小值在导数从负变正处 → b*_{k,j} 闭式解
    // 每次 SMAWK 调用跳过两个量化值，调用次数减半
    ```
  - Apx. QUIVER 关键流程：
    ```
    输入: X, s, m       // m 为离散网格划分数
    1. δ = (x_d - x_1)/m
    2. 收集直方图: A_ℓ = count(x in [s_ℓ, s_{ℓ+1}]), ∀ℓ∈[0,m-1]
    3. 计算累积量: α_ℓ, β_ℓ, γ_ℓ (O(m) 扫描)
    4. 用 C_m[k,j] = -s_j·s_k·(α_j-α_k) + (s_j+s_k)·(β_j-β_k) - (γ_j-γ_k)
       替代 C[k,j]，其余与 QUIVER 相同
    5. 复杂度: O(d + m·s)
    6. 近似保证: AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)
    ```
  - 使用示例（C++ 命令行）：
    ```bash
    # 编译
    g++ -O3 -std=c++17 quiver.cpp -o quiver
    # 运行：对 1M 维 LogNormal 向量计算最优 4-bit 量化值
    ./quiver --input vector_1M.txt --s 16 --algorithm accelerated
    # 近似量化：m=400 离散网格，6ms 完成
    ./quiver --input vector_1M.txt --s 16 --algorithm approximate --m 400
    ```

## Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **OBR (Optimal Brain Restoration)**——一个训练无关（training-free）的统一框架，通过 Group Error Compensation 在剪枝和量化之间计算最优补偿，调和两者对权重分布的冲突需求。完整流程为：(1) Hadamard Rotation 平滑权重 outliers → (2) 剪枝生成稀疏 mask → (3) **OBR for Pruning**：将剪枝损失的信息从 pruned elements (eviction set E₁) 通过 Hessian 传播补偿到 unpruned elements (retain set R₁)，即 Δw_{R₁}^{prune} = -H_{R₁R₁}^{-1} H_{R₁E₁} w_{E₁} → (4) **OBR for Quantization**：将 unpruned elements 按比例 α 分为 eviction set E₂（前 α 比例）和 retain set R₂，补偿量化误差，即 Δw_{R₂}^{quant} = -H_{R₂R₂}^{-1} H_{R₂E₂} (w̄_{E₂} - quant(w̄_{E₂})) → (5) RTN/GPTQ 量化得到最终 W4A4KV4 + 50% sparse 权重。
  - 实验比较：OBR_RTN 和 OBR_GPTQ vs. **QuaRot (quant-only)** W3A4KV4 baseline、**QuaRot+WANDA** (naive combination)、**SparseGPT+GPTQ** (strong joint pruning+quantization baseline)。指标为 WikiText2 perplexity 和 PIQA/BoolQ/HellaSwag/ARC-easy/ARC-challenge/WinoGrande 零样本准确率。额外比较：(a) 不同 bit-width (W4A8KV8, W4A16KV16)；(b) SpinQuant 和 FlatQuant 旋转矩阵；(c) 2:4/4:8 半结构化稀疏；(d) BitNet-2B-4T 对比；(e) 纯剪枝/纯量化单任务扩展；(f) 不同 calibration 数据集 (C4)。INT4 2:4 sparse GEMM kernel 在实际 GPU 上的 latency/FLOPs/TOPS。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100-SXM4-80GB GPU（模型压缩 calibration）。INT4 2:4 sparse GEMM kernel 效率测试在同一 A100 GPU 上进行（利用 Ampere 架构的 native INT4 sparse GEMM 支持）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：**Llama2** (7B/13B/70B)、**Llama3** (8B/70B)、**Qwen2.5-Instruct** (7B/32B)、**Qwen2.5-Instruct** (1.5B/3B，BitNet 对比)。
  - Calibration 数据集：128 samples from **WikiText-2**，sequence length 2048（默认）；也测试 **C4** calibration。
  - 评估数据集/bench：**WikiText-2** test set (perplexity)；**PIQA、BoolQ、HellaSwag、ARC-Easy、ARC-Challenge、WinoGrande**（零样本常识推理，使用 lm-eval-harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub **https://github.com/csguoh/OBR**，HuggingFace **https://huggingface.co/HangGuo/OBR**。LICENSE：QuaRot backbone Apache 2.0，SpinQuant backbone CC-BY-NC 4.0，FlatQuant backbone MIT。
  - 软件环境：Python 3.10, PyTorch, HuggingFace Transformers (Qwen2.5 需 4.45.0), fast-hadamard-transform, CUTLASS (kernel)。
  - OBR 算法 Pipeline 伪代码：

```
输入: Hadamard-rotated 权重矩阵 W ∈ ℝ^{C_out × C_in}, 
      Hessian 近似 H = 2XX^T ∈ ℝ^{C_in × C_in}, 分区比例 α
输出: 低比特稀疏权重 Ŵ ∈ ℤ^{C_out × C_in}

// Step 1: 剪枝
M ∈ {0,1}^{C_out × C_in} = prune(W)    // 使用 WANDA/SparseGPT/magnitude mask
W^{prune} ← W ⊙ M

// Step 2: OBR 误差补偿
ΔW^{OBR} ← 0
for c = 1 ... C_out do                    // 逐行处理 (row-wise decoupling)
    // === OBR for Pruning ===
    R₁ ← {i | M_{c,i} = 1}              // unpruned indices
    E₁ ← {j | M_{c,j} = 0}              // pruned indices
    b₁ ← H_{R₁E₁} · W_{c,E₁}^T          // 从 Hessian 提取子矩阵×剪枝权重
    Δw_{R₁}^{prune} ← -H_{R₁R₁}^{-1} · b₁  // 闭式解: 补偿到 unpruned slots
    w̄ ← W_{c,R₁}^{prune} + Δw_{R₁}^{prune} // 补偿后的稀疏权重

    // === OBR for Quantization ===
    e^{quant} ← w̄ - quantize(w̄)          // 量化误差向量
    t ← ⌊α · |R₁|⌋                        // 按照比例 α 切分
    E₂ ← {r₁, ..., r_t}                  // 前 α 比例 → eviction set
    R₂ ← {r_{t+1}, ..., r_{|R₁|}}        // 剩余 1-α → retain set
    b₂ ← H_{R₂E₂} · e_{E₂}^{quant}^T
    Δw_{R₂}^{quant} ← -H_{R₂R₂}^{-1} · b₂  // 闭式解: 补偿量化误差

    // 合并补偿
    ΔW_{c,R₁}^{OBR} += Δw_{R₁}^{prune}
    ΔW_{c,R₂}^{OBR} += Δw_{R₂}^{quant}
end for

// Step 3: 量化
W^{quant} ← W^{prune} + ΔW^{OBR}
Ŵ ← quantize(W^{quant})                  // RTN 或 GPTQ quantizer
```

  - 关键张量计算与直觉：
    - **二阶 Hessian 目标**：min E[ΔL] ≈ ½ vec(ΔW) H_full vec(ΔW)^T，H_full ≈ I ⊗ H 逐行解耦后变为 C_out 个独立子问题 min ½ Σ_i Δw_i H Δw_i^T。
    - **Group Error Compensation 闭式解**：将 Δw 分为 retain set R 和 eviction set E，令 e_E 为 E 上的压缩误差，则 min_{Δw_R} ½[Δw_R e_E] [H_{RR} H_{RE}; H_{ER} H_{EE}] [Δw_R^T e_E^T] 的闭式解为 Δw_R^* = -H_{RR}^{-1} H_{RE} e_E。Hessian 作为"桥梁"将误差从 E 传播到 R。
    - **计算复杂度**：需要逐行求解线性系统 H_{RR}^{-1} b，对 7B 模型约需 2 小时（单 A100）。
  - 使用示例（QuaRot + Llama2-7B）：
    ```bash
    cd ./QuaRot
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --rotate \
      --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip --ppl_eval
    ```

 One-Line Revolution for Generative AI Model Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **LPCD (Layer-Projected Coordinate Descent)**——一种将 layer-wise PTQ 扩展到任意子模块量化的统一框架。核心流程为两步交替：(1) **Relaxation Step**：在固定其他 block 的条件下，对当前 block 做无约束连续优化（求闭式解或梯度下降近似），得到松弛后的全精度权重；(2) **Projection Step**：用标准 layer-wise PTQ 投影器（RTN 的 Π^(d) 或 GPTQ 的 Π^(a)）将松弛解投影回量化域。LPCD 统一了已有的 QEP（单步 weight-side LPCD）和 LoaQ（单步 augmented submodule LPCD），并自然扩展到三个 Transformer 子模块：**QK Module**（grouped-query attention 的 Q/K 投影）、**VO Module**（Value-Output 聚合）、**Up-Down Module**（MLP 的 Up/Down 投影）。QEP 扩展还包括 activation quantization、KV-cache quantization、orthogonal rotation matrices 和 LoRA-based error compensation。
  - 实验比较：LPCD-based submodule quantization vs. **QEP** 和 **LoaQ** 两种 error compensation baseline，分别在基础 quantizer **RTN** 和 **GPTQ** 上叠加。量化位宽：INT4、INT3、INT2（per-channel weight quantization）。指标为 WikiText-2 perplexity (PPL) 和 ARC-Easy/PIQA 零样本平均准确率。模型包括 LLaMA2-7B/13B、LLaMA3-8B、Qwen3-8B/14B。Figure 1 展示各层 output MSE 对比。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 GPU（TSUBAME 4.0 超级计算机）。单卡运行量化流程。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA2-7B、LLaMA2-13B、LLaMA3-8B、Qwen3-8B、Qwen3-14B（HuggingFace Transformers 实现，含 Qwen3-4B 和 LLaMA3.2-1B 用于超参网格搜索）。
  - 数据集/bench：**WikiText-2**（perplexity 评估）；**ARC-Easy** 和 **PIQA**（零样本准确率，使用 lm-eval-harness）；**C4** 和 **WikiText-2**（calibration 数据，最终使用 2048 tokens / 256 sequences 随机采样自 WikiText-2 以减轻过拟合）。
  - 量化配置：INT4/INT3/INT2 per-channel weight-only quantization；最后 2 层跳过量化（因激活 outliers 频率高）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确提供开源链接。
  - 软件环境：Python 3.12.11, PyTorch 2.4.0, HuggingFace Transformers 4.55.3。
  - 算法pipeline 核心伪代码（以 VO Module 为例，一轮 LPCD 更新 W_V 和 W_O）：

    ```
    # 输入: 已量化的残差流 R̂, 全精度 R
    #       全精度注意力权重 S^(h), 量化注意力权重 Ŝ^(h)
    #       量化校准特征 X̂, 全精度特征 X
    #       上一轮量化权重 Ŵ_V^(g), Ŵ_O

    # === Step 0: 初始化 ===
    # 先用 LoaQ 量化每个子模块作为初始点

    # === Step 1: Value Relaxation (固定 Ŵ_O, 更新 Ŵ_V) ===
    # 对每个 group g:
    #   Y = concat_h(S^(h) X W_V^(g)) W_O + R    # 全精度目标
    #   Ŷ_{¬g} = 不含 group g 贡献的量化输出
    #   y* = vec(Y - Ŷ_{¬g})
    #   Z_V^(g) = Σ_{h∈H_g} (Ŵ_O^(h)T ⊗ Ŝ^(h) X̂)
    #
    #   闭式解（内存不可行时用梯度下降近似）:
    #   vec(W̄_V^(g)) = (Z_V^(g)T Z_V^(g))^{-1} Z_V^(g)T y*

    # === Step 2: Value Projection ===
    #   Ŵ_V^(g) = Π_Q^(w)(W̄_V^(g))   # 用 RTN 或 GPTQ 投影

    # === Step 3: Output Relaxation (固定 Ŵ_V, 更新 Ŵ_O) ===
    #   Ĥ = concat_h(Ŝ^(h) (X̂ Ŵ_V^(g(h))))
    #   W̄_O = (ĤT Ĥ)^{-1} ĤT (Y - R̂)   # 闭式可解（ĤTĤ 规模可控）

    # === Step 4: Output Projection ===
    #   Ŵ_O = Π_Q^(w)(W̄_O)
    ```

  - 对 QK Module 和 Up-Down Module 的 Relaxation Step 因设计矩阵过大（如 QK 的 Z_Q ∈ R^{T² × (D_model d_k)}，Up 的 Z_U ∈ R^{T D_model × (D_model D_up)}），不显式构造矩阵，改为梯度下降近似求解（Adam, bs=8, 40 epochs, cosine LR 起始 1e-5）。
  - 对 Down Step 和 O-Step，因设计矩阵规模可控（仅依赖 head dim 或 D_up），直接用闭式解。
  - LPCD 在 LoaQ 量化结果之上运行（LoaQ 作为 LPCD 的初始化），先 LoaQ 后 LPCD。超参 α ∈ [0,1] step 0.1, β ∈ [0,1] step 0.05 通过小模型（Qwen3-0.5B, LLaMA3.2-1B）网格搜索确定后迁移到大模型。

## Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **TTT (Test-Time Training) 层**——一种将 RNN 隐藏状态建模为机器学习模型本身、更新规则为自监督学习梯度步的序列建模层。两个实例：**TTT-Linear**（隐藏状态为线性模型 `f(x)=Wx`）和 **TTT-MLP**（隐藏状态为两层 MLP，hidden dim 4×，GELU 激活，含 LN 和残差连接）。关键创新包括：mini-batch TTT（b=16）实现内循环并行化；dual form 将梯度计算转化为矩阵乘法以提高 GPU/TPU 硬件利用率；可学习的 reconstruction views（θ_K, θ_V, θ_Q）；可学习的初始权重 θ_init = W_0；可学习的逐 token 学习率 η(x) = η_base · σ(θ_lr · x)。论文在 Mamba backbone（含时序卷积）和 Transformer backbone 下评估。
  - 实验比较：TTT-Linear 和 TTT-MLP vs. **Transformer**（Llama-based Transformer++，含 RoPE、SwiGLU、RMSNorm）和 **Mamba**（现代 RNN baseline），在 125M/350M/760M/1.3B 四个规模下，使用 matched training FLOPs。消融实验展示从 linear attention 逐步加入 learnable W_0、LN+residual in f、mini-batch TTT(b=16 vs b=T)、learnable η、Mamba backbone 的改进过程（Table 1）。额外消融 mini-batch size b 对 perplexity 和 wall-clock time 的影响（Figure 7）。

- 硬件平台是什么，配置是什么。
  - 训练：TPU v5e-256 pod
  - 推理延迟评测：NVIDIA A100 GPU 80G HBM，PCIe 连接
  - Transformer 推理 baseline 使用 vLLM serving 系统

- 模型是什么。数据集和bench分别是什么。
  - 模型规模：125M（12层/d=768）、350M（24层/d=1024）、760M（24层/d=1536）、1.3B（24层/d=2048）参数
  - 数据集：**The Pile**（标准 2k 和 8k 上下文实验）、**Books3**（Pile 子集，用于长上下文 1k-32k 实验）
  - 评估指标：perplexity（PPL）、scaling law 曲线（FLOPs vs. PPL）、每 token 平均 PPL 随 token index 变化、wall-clock latency（forward/prefill 和 generate/decode）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：JAX 实现 https://github.com/test-time-training/ttt-lm-jax，PyTorch 实现 https://github.com/test-time-training/ttt-lm-pytorch。基于 EasyLM 框架。
  - 算法 pipeline 核心伪代码（TTT-Linear 单层，第一个 mini-batch，dual form）：

    ```
    # 输入: X = [x_1,...,x_b] ∈ R^{d×b} (mini-batch of tokens)
    # 可学习参数: θ_K, θ_V, θ_Q ∈ R^{d×d} (reconstruction views)
    #             θ_init = W_0 ∈ R^{d×d} (初始权重)
    #             θ_lr ∈ R^d (学习率参数)
    # 超参: η_base = 1.0 (TTT-Linear), b = 16

    # Step 1: 生成 training view, label view, test view
    X̂ = θ_K @ X          # training view, 低秩投影
    Y  = θ_V @ X          # label view
    X̄ = θ_Q @ X          # test view

    # Step 2: 自监督损失 (MSE reconstruction)
    # ℓ(W_0; x_t) = ||W_0 x̂_t - y_t||²
    # 对每个 token 的梯度: G_t = ∇ℓ(W_0; x_t) = 2(W_0 x̂_t - y_t) x̂_t^T

    # Step 3: mini-batch 更新 (dual form, 无需显式计算 G_t)
    # W_b = W_0 - η Σ_{t=1}^b G_t = W_0 - 2η (W_0 X̂ - Y) X̂^T
    W_b = W_0 - 2 * η * (W_0 @ X̂ - Y) @ X̂.T

    # Step 4: 输出 token 计算 (dual form)
    # z_t = W_t x̄_t = (W_0 - η Σ_{s=1}^t G_s) x̄_t
    # Z = [z_1,...,z_b] = W_0 X̄ - 2η (W_0 X̂ - Y) mask(X̂^T X̄)
    # 其中 mask 是上三角 mask（类似 attention mask, 但用 0 替代 -∞）
    Δ = (W_0 @ X̂ - Y) * mask(X̂.T @ X̄)    # mask 为上三角 1/下三角 0
    Z = W_0 @ X̄ - 2 * η * Δ
    ```

  - 内循环/外循环双层训练：
    - **内循环 (TTT)**：对每个序列，从 W_0 开始，对每个 mini-batch 计算梯度并更新 W。目标是最小化 reconstruction loss ℓ(W; x_t) = ||f(θ_K x_t; W) - θ_V x_t||²。
    - **外循环 (常规训练)**：优化 θ_rest（网络其余参数）、θ_K, θ_V, θ_Q, θ_init, θ_lr。目标是最小化 next-token prediction loss。训练配置遵循 Chinchilla recipe（AdamW, cosine schedule, warmup, weight decay 0.1, gradient clipping 1.0, mixed precision）。
  - 时间/空间复杂度：每个 token O(d²)（与序列长度 T 无关），dual form 将内循环 mini-batch 计算转化为 matmul 以利用 TensorCores。

## MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MagR 是一种基于 channel-wise ℓ∞-regularized 最小二乘优化的权重预处理技术，通过 Proximal Gradient Descent（近端梯度下降）配合 ℓ₁-ball 投影来迭代减少预训练权重的最大幅度，从而缩小量化步长 δ、降低量化误差。MagR 作为非线性预处理变换，不需要在推理时对特征做逆变换。MagR + RTN、MagR + OPTQ、MagR + OPTQ†（加30轮 coordinate descent）以及 MagR + QuIP 作为复合方案。
  - 实验比较：在 LLaMA1（7B–65B）和 LLaMA2（7B–70B）上，对 W2A16、W3A16、W4A16（per-channel）和 W2A16g128、W3A16g128、W4A16g128（per-group）配置进行 weight-only quantization，对比 RTN、OPTQ、AWQ、OmniQuant、QuIP。指标为 WikiText2/C4 perplexity 和 PIQA/ARC-Easy/ARC-Challenge/Winogrande 零样本准确率。还报告了 MagR 预处理耗时和总量化耗时。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A100 GPU（80GB 显存）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA1（7B, 13B, 30B, 65B）和 LLaMA2（7B, 13B, 70B），使用 HuggingFace 实现。
  - 数据集/bench：WikiText2 和 C4（语言生成 perplexity 评估，context length 2048），128 个 calibration samples；PIQA、ARC-Easy、ARC-Challenge、Winogrande（零样本任务，使用 lm-eval-harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/AozhongZhang/MagR
  - 算法流程（per-channel MagR for one linear layer）：
    1. 输入预训练权重 W_hat ∈ R^{m×n}，Hessian H = XᵀX ∈ R^{m×m}（X 为 calibration 特征矩阵），迭代次数 K，步长 η = 1/λ_max(H)，惩罚参数 α。
    2. 初始化 W⁰ = W_hat。
    3. 对 k = 0,...,K-1：
       - 梯度下降步：V^k = W^k - η · H · (W^k - W_hat)
       - 近端算子（列级 ℓ₁-ball 投影）：W^{k+1} = V^k - ηα · proj_{‖·‖₁≤1}(V^k/(ηα))
    4. 返回预处理后的权重 W = W^K。
    5. 在预处理后的 W 上应用标准 uniform quantizer（含可选的 δ 缩放因子 β ≤ 1）：δ = β · (max(w)−min(w))/(2^b−1)。
  - 核心思想：特征矩阵 X 近似秩亏（见表2，fraction rank 均值 70%–84%），因此存在无数 w 满足 Xw ≈ Xw_hat，可在保持层输出的前提下大幅降低 w 的 ℓ∞ 范数（最大幅度），缩小量化步长。
  - ℓ₁-ball 投影用 O(m log m) 的排序+软阈值算法（Algorithm 2/3）。
  - per-group 扩展：将 V ∈ R^{m×n} reshape 为 R^{d×((m/d)·n)} 后独立做 ℓ₁-ball 投影。
  - 参数设置：K=150，α=10⁻³（per-channel）/ 10⁻⁴（per-group），β ∈ [0.8, 0.95] 取决于 bit-width。

## LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LoTA-QAF 是一种面向量化 LLM 的量化感知微调（QAF）方法，包含三个核心组件：i) 三元适配器（Ternary Adaptation, TA），训练 `A_T ∈ {-1,0,1}^{D_in×r}` 和 `B_T ∈ {-1,0,1}^{r×D_out}`，通过辅助矩阵 `ΔW = A_T B_T` 和阈值 ω 生成三元矩阵 `Ŵ ∈ {-1,0,1}^{D_in×D_out}`（`Ŵ_ij = sign(ΔW_ij) · I_{|ΔW_ij|>ω}`），直接在量化网格内调整量化权重 `W_int`；ii) TA-based lossless merging 机制，通过 `W'_int = W_int + Ŵ` 和 `z' = z + sμ`（μ 为偏移因子）将适配器无损合并到量化权重和零点因子中；iii) t-SignSGD 优化器，使用基于符号梯度的更新和动态百分位阈值 σ_t 选择性地更新三元适配器权重（`A_{T,t+1} = clip(A_{T,t} - sign(g_t) · I_{|g_t|>max(τ,σ_t)}, -1, 1)`）。
  - 实验比较两种微调范式：i) performance-recovery（Alpaca 数据集微调后在 MMLU 5-shot 上评估，对比 GPTQ 量化基线、GPTQ+LoRA（4+16bit）和 QA-LoRA）；ii) task-specific（在 GSM8K、SQL generation、ViGGO 三个任务上微调评估）。

- 硬件平台是什么，配置是什么。
  - 所有实验在一张 NVIDIA A800 GPU 上运行。
  - 推理效率测试使用 Llama 3.1 8B 模型，4-bit/2-bit 用 TritonV2QuantLinear kernel，3-bit 用 TorchQuantLinear kernel。Batch size 8-128，最大推理长度 512 tokens。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama 3.1 8B、Qwen 2.5 14B、Qwen 2.5 32B、Llama 3.3 70B。
  - 量化方式：GPTQ 非对称量化，Llama 8B 和 Qwen 14B 用 group size 64，Qwen 32B 和 Llama 70B 用 group size 128。校准数据使用 C4 数据集 1024 样本。
  - 数据集：Alpaca（performance-recovery 微调）；GSM8K（7.47k 训练/1.32k 测试）、SQL generation（30k 训练/1 测试）、ViGGO（5.1k 训练/1.08k 测试）（task-specific 微调）。
  - Benchmark：MMLU（5-shot，含 Humanities/STEM/Social/Other 四类）；GSM8K（0-shot）、SQL（0-shot）、ViGGO（0-shot）；使用 lm-eval 框架评估 MMLU，使用 HALO 的自定义评估框架评估 task-specific 任务。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码在 github.com/KingdalfGoodman/LoTA-QAF，MIT 协议，包含 LoTA_QAF_main.py（训练/评估）、LoTA/layer.py（CustomLoraLinear 实现三元适配）、LoTA/adapter.py（LTA 推理加载）、LoTA/lota_merge.py（合并逻辑）、t_signSGD.py（优化器）等。
  - 算法 pipeline（张量计算）：
    1. 量化预训练权重：`W_q = s · clamp(round((W - z)/s), 0, 2^N-1) + z`，得到 `W_int`、`s`、`z`
    2. 初始化三元适配器：`A_T` 用 Kaiming normal 初始化后通过阈值 0.75·mean(|A_T|) 三值化为 `{-1,0,1}`；`B_T` 初始化为零
    3. 前向传播：`ΔW = A_T @ B_T`（整数矩阵，元素 ∈ [-r, r]）→ `Ŵ_ij = sign(ΔW_ij) · I_{|ΔW_ij|>ω}` → `W'_int = W_int + Ŵ`（含边界检查防止溢出 [0, 2^N-1]）→ `μ = mean(ΔW - ω·Ŵ)` → `z' = z + s·μ` → `y = (s·W'_int + z')^T · x`
    4. 反向传播（t-SignSGD）：计算梯度 g_t = ∇_{A_T} L → 确定动态阈值 σ_t（top-5% 梯度幅值，线性衰减至 0.01%）→ `A_{T,t+1} = clip(A_{T,t} - sign(g_t) · I_{|g_t|>max(τ,σ_t)}, -1, 1)`
    5. 推理时无损合并：直接使用 `W'_int` 和 `z'` 替代 `W_int` 和 `z`，无需额外适配器计算，保持低比特推理效率（相比 LoRA 的 16-bit 适配器，LoTA 合并后速度提升 1.7x-2.0x）
  - 超参数：rank r=64 (8B/14B) 或 32 (32B/70B)；ω = 0.75r (Alpaca/GSM8K/SQL) 或 0.875r (ViGGO)；σ_t 初始 top-5%，前 80% 训练线性衰减至 0.1%，后 20% 固定 0.01%；优化器 paged AdamW，max grad norm 0.3，batch size 64，source length 1024，target length 256

## Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：HDRQ（Hessian and Distance Regularizing Quantization），一种面向多目标域自适应的合并友好型 PTQ 方法，包含三个核心组件：i) Noise-based Hessian Regularization：将量化误差建模为均匀噪声 ε∼U[-Δ/2, Δ/2]，向权重中加入采样噪声 w+ε 替代确定性量化值 ŵ 进行重建训练，隐式惩罚损失曲面的尖锐曲率，使权重收敛到更平坦的局部极小值，降低合并时的 error barrier；ii) Weight Distance Regularization：通过最小化量化后权重与源模型权重的 ℓ₂ 距离（||w_src − w_tar||₂），利用三角不等式（||w_tar1 − w_tar2|| ≤ ||w_src − w_tar1|| + ||w_src − w_tar2||）间接控制不同域自适应权重之间的差异，确保合并兼容性；iii) Noise-Sampling-Based Rounding：在合并阶段，对量化权重加入采样噪声后再取整（I_merged = ⌊(I₁·Δ₁+ε₁ + I₂·Δ₂+ε₂)/(Δ₁+Δ₂)⌉），通过 cosine similarity 筛选最优噪声样本，解决浮点域合并与整数域合并间的舍入歧义问题。整体流程：源预训练模型 → 单目标域自适应（HRDA/SHOT）→ HDRQ 量化（block-wise reconstruction 20000 迭代，Adam with cosine annealing LR=0.001，λ=5e-2，最后 3500 迭代切换到 fake quantization）→ 模型合并（midpoint weight averaging + noise sampling rounding）→ 多目标域统一模型。
  - 实验比较：在 Semantic Segmentation（GTA→Cityscapes/IDD，HRDA+ResNet-101）和 Image Classification（Office-Home 四域，SHOT+ResNet-50）两个多目标域自适应任务上，对比 BRECQ 和 QDrop 两种 PTQ baseline，涵盖 W6A6、W4A4、W8A8、W8A4、W4A8、W4A4、W3A3 多种 bit-width。指标为 mIoU（分割）和 Harmonic Mean Accuracy（分类）。额外包含 ablation study（逐组件增量消融：Baseline QDrop → +Noise-based quantization → +Distance regularization）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号和配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet-101（语义分割 backbone）+ 简单卷积头（HRDA 架构）；ResNet-50（图像分类 backbone，SHOT 架构）。
  - 数据集/benchmark：
    - 语义分割：源域 GTA 合成数据集（Richter et al., 2016），目标域 Cityscapes（Cordts et al., 2016）和 Indian Driving Dataset / IDD（Varma et al., 2019）。指标 mIoU，30 次采样平均。
    - 图像分类：Office-Home 数据集（Venkateswara et al., 2017），四域（Real/Art/Clipart/Product），一域作源、三域作目标，四种源域配置 R→A,C,P / A→R,C,P / C→R,A,P / P→R,A,C。指标 Harmonic Mean Accuracy。
  - 单目标域自适应方法：HRDA（Hoyer et al., 2022，语义分割），SHOT（Liang et al., 2020，图像分类）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源代码链接。发表于 ICML 2025，作者机构为 POSTECH。
  - 算法 pipeline（以 ResNet-50 单层 block-wise reconstruction 为例）：
    1. **源模型域自适应**：对源预训练模型 θ₀ 各自适应到目标域，得到 θ₁（如 Real→Clipart）和 θ₂（如 Real→Product）。
    2. **HDRQ 量化（各域独立执行）**：
       - 对 θ₁ 的每个 block（含 BN 折叠后的卷积层），计算量化步长 Δ = (max(w)−min(w))/(2^b−1)。
       - 噪声量化模拟：w_hat = clamp(⌊w/Δ⌉, −2^{b-1}, 2^{b-1}−1)·Δ，噪声 ε = w − w_hat ∼ U[-Δ/2, Δ/2]，训练使用 w+ε。
       - Block-wise reconstruction：minimize ||F_block(w+ε, x) − F_block(w_orig, x)||₂² + λ·||w_src − (w+ε)||₂²，其中 λ=5e-2 为距离正则化系数。
       - 最后 3500 迭代切换到 fake quantization（确定性 ŵ），学习率衰减到很小。
    3. **合并（Noise Sampling Rounding）**：
       - 采样多组噪声 ε₁^k, ε₂^k ∼ U[-Δ/2, Δ/2]，对每组 k 计算 I_merged^k = ⌊(I₁·Δ₁+ε₁^k + I₂·Δ₂+ε₂^k)/(Δ₁+Δ₂)⌉。
       - 计算 cos_sim(vec(w_merged^k − w_tar), vec(w_tar1 − w_tar2))，选最高相似度的样本。
       - 最终合并权重：w_merged = (w_tar1_quant + w_tar2_quant) / 2（midpoint averaging，此时 step sizes 相同）。
    4. 推理：合并后的统一量化模型直接在各目标域上推理，无需额外的适配器或变换。
  - 张量计算核心（Hessian Regularization 的数值效果）：E[L(ŵ)] ≈ E[L(w) + ½·εᵀ·∇²_w L(w)·ε]，均匀噪声 ε 使损失函数隐式惩罚 ∇²_w L(w) 的大特征值，引导到平坦区域。

## LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

- 属于算法pipeline的实现是什么？实验比较什么？
  - **LogART**：首个将对数域可学习舍入（Learnable Logarithmic Rounding, LLR）集成到 PTQ 中的方案。核心组件：
    1. **LLR（Learnable Logarithmic Rounding）**：在基-2对数域中将 RTN 替换为 floor + 可学习变量 R，通过 sigmoid 函数 σ(R) 使每个 weight 的舍入决策 soft 化。损失函数 = 逐层/逐块重建误差 (Frobenius 范数) + 正则化项（鼓励 σ(R) → 0 或 1）。梯度链：∂L/∂W̃ → ∂W̃/∂Q_W → ∂Q_W/∂R，其中包含指数项 2^{-Q_W} 和对数缩放因子 s·ln2。
    2. **OHS（Optimized Hyperparameter Search）**：三级搜索策略 —— (a) ABS: tensor-wise 非对称边界搜索（无需校准数据），基于 max/min weight 分配不同数量的正负码字；(b) SFS: block-wise 缩放因子搜索（通过最小化块级重建误差搜索最优 s_of 抵御 outlier）；(c) DBS: block-wise 动态基搜索（自适应分配 n₁:n₂ 的 base-√2 和 base-2 比例）。
    3. **Dynamic Base Quantizer**：大值用 base-√2（细粒度），小值用 base-2（粗糙但硬件友好），比例由 DBS 按分布感知方式搜索。量化码本含 n₁ 个 base-√2 码字和 n₂ 个 base-2 码字。
    4. **Asymmetric Quantizer**：首次为对数域设计非对称量化，通过自适应边界 l_a 为正值和负值分配不同码字数，解决 LLM 中常见的非对称 weight 分布。
    5. **Outlier-Resilient Quantizer**：引入可搜索超参数 s_of 替代 max(|W|) 来确定量化范围，实现自适应极值裁剪。
    6. **HAF（Hardware Approximation Function）**：用 K-term Signed Dyadic Expansion (SDE) 近似 √2（如 √2 ≈ 2⁰ + 2⁻¹），将乘 √2 替换为 shift-add 操作。HAF 嵌入 LLR 前向传播，近似误差在优化过程中被吸收为噪声。
  - 实验比较的 baselines：
    - LLM: GPTQ（linear/RTN）、BRECQ（linear/optimization）、AffineQuant（linear）、aespa（linear）
    - CNN: AdaRound（linear/learnable rounding）、BRECQ（linear）、FlexRound（linear）、LogNet（log/RTN）、SLogII（log）
    - Vision Transformer: BRECQ、APHQ（linear）、AdaLog（linear weight + log activation）、LogNet、SLogII

- 硬件平台是什么，配置是什么。
  - 单块 NVIDIA RTX 5090D GPU（32 GB）用于所有量化实验。
  - AE 硬件评估：Synopsys Design Compiler，28nm UMC 工艺，250 MHz，0.9V。

- 模型是什么。数据集和bench分别是什么。
  - LLM: OPT-125M, OPT-1.3B, OPT-6.7B, LLaMA2-7B, LLaMA3-8B；评估数据集 WikiText-2 (PPL) 和 C4 (PPL)
  - CNN: ResNet18, ResNet50, MobileNetV2；评估数据集 ImageNet (Top-1 Accuracy)
  - Vision Transformer: ViT-Small, ViT-Base, DeiT-Tiny, DeiT-Base；评估数据集 ImageNet (Top-1 Accuracy)
  - 校准数据：LLM 从 WikiText-2 或 C4 随机采样 32 段 2048 token；Vision 从 ImageNet 随机采样 2048 张无标签图片

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/logart-lab/logart
  - 算法 pipeline 伪代码：
  ```
  # LogART 量化流程（per-channel weight quantization）
  # 输入: FP16 weight W, calibration data X, bitwidth N
  
  # Step 1: OHS - 搜索最优超参数
  For each weight channel:
      # ABS: 基于 max/min 计算非对称边界 l_a（无需校准数据）
      w_h = max(w_max, -w_min); w_l = min(w_max, -w_min)
      l_a = floor(d_a / 2)  # d_a 由 w_h, w_l, threshold t 决定
  
  For each block (e.g., attention module):
      # SFS + DBS: 联合搜索 s_of 和 n₁:n₂ 以最小化块级重建误差
      # 校准集上的前向传播，搜索使 ||ΔW·X||_F² 最小的配置
      argmin_{s_of, n₁, n₂} E[||(W - W̃)X||_F²]
  
  # Step 2: LLR - 可学习对数舍入
  Initialize R = 0  # 每元素一个可学习变量
  For iter in 1..max_iters (LLM: 500, Vision: 2000):
      # Soft quantize (Eq. 12, 17):
      Q_W = clamp( floor(-log_B(|W| / (s_of * S))) + σ(R), 0, U )
      W̃ = S * sign(W) ⊙ B^{-Q_W}  # B ∈ {2, √2} per-element
  
      # HAF: 硬件近似的 √2 用 SDE 替代
      If B == √2:  # 仅对 base-√2 的奇数 Q_W 元素
          W̃' = W̃ * (1 + (γ - 1) * M)  # M = (Q_W mod 2) ⊙ [B == √2]
  
      # Loss:
      L_recon = tr(ΔW · E[XX^T] · ΔW^T)  # 逐层重建
      L_reg = λ * Σ(1 - |2σ(R) - 1|^β)   # 鼓励 hard rounding
      L = L_recon + L_reg
  
      # Gradient descent on R (Eq. 29):
      ∂L/∂R = 2s·ln2 · M_c ⊙ 2^{-Q_W} ⊙ sign(W) ⊙ [(WX - W̃X)X^T] ⊙ σ'(R)
             + λ · ∂f_reg/∂R
      R = R - lr * ∂L/∂R
  
  # Step 3: Hard rounding
  Q_W_hard = clamp( floor(-log_B(|W| / (s_of * S))) + round(σ(R)), 0, U )
  ```
  - 量化位宽：3-bit 和 4-bit 权重量化（weight-only）；支持与任意激活量化方法（SmoothQuant, AdaLog, QuaRot 等）组合
  - 优化器：Adam，CosineAnnealingLR scheduler，lr 从 0.05 衰减到 0.015，rounding loss weight λ=1

## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- 属于算法pipeline的实现是什么？实验比较什么？
  - **KIVI**：免调优的 2bit KV Cache 非对称量化算法。核心设计：
    1. **Key Cache per-channel 量化**：分析发现 key cache 中少数固定 channel 具有极大 magnitude outlier，per-channel 量化可以将误差限制在每个 channel 内部，不影响其他正常 channel。实现上，由于 per-channel 量化跨 token，无法直接 append 到流式 KV cache，因此将 key cache 分为 grouped 部分（每 G 个 token 一组做 group-wise per-channel 量化）和 residual 部分（保留 FP16，最多 R 个 token）。
    2. **Value Cache per-token 量化**：value cache 无 outlier pattern，但由于 attention output 是 value cache 的加权求和（权重为稀疏 attention score），per-token 量化将误差限制在每个 token 内部，保证重要 token 不受其他 token 量化影响。实现上同样分为 grouped 和 residual 两部分。
    3. **Full precision sliding window**：residual 部分（最多 R 个 token）保持在 FP16，形成局部全精度滑动窗口。这对 GSM8K 等困难任务至关重要。
    4. 量化方式采用 group-wise round-to-nearest（公式：Q(X) = ⌊(X - z_X)/s_X⌉, X' = Q(X)·s_X + z_X），group size G=32，residual length R=128。
  - 实验比较：
    - **不同量化配置的 fake quantization 对比**：2bit (K per-channel, V per-token) vs 2bit (K per-token, V per-token) vs 2bit (K per-channel, V per-channel) vs 2bit (K per-token, V per-channel) vs 4bit per-token vs 16bit baseline
    - **KIVI-2 / KIVI-4 vs 16bit baseline**：在 Llama-2-7B/13B、Falcon-7B、Mistral-7B 上全面对比
    - **Ablation**：group size G∈{32, 64, 128}、residual length R∈{32, 64, 96, 128}
    - **Efficiency**：KIVI vs FP16 baseline 的峰值内存和吞吐量对比（ShareGPT 真实 workload）
    - **Long context**：LongBench 8 个子任务 + NIAH

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A100 GPU（80GB）
  - 论文未明确说明具体的 CPU/内存配置

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Llama-2-7B、Llama-2-13B（multi-head attention）
    - Llama-2-7B-Chat、Llama-2-13B-Chat
    - Falcon-7B（multi-query attention，KV cache 仅单头）
    - Mistral-7B（multi-head attention）
    - Llama-3-8B-Instruct（group query attention，KV 8头）
    - Mistral-7B-Instruct-v0.2（group query attention，32K context）
    - LongChat-7B-v1.5-32K（32K context）
  - 数据集/Benchmark：
    - **LM-Eval**（normal context）：CoQA（EM accuracy）、TruthfulQA（BLEU）、GSM8K（EM accuracy）
    - **LongBench**（long context）：Qasper（F1，Single-Doc QA）、QMSum（ROUGE，Summarization）、MultiNews（ROUGE，Summarization）、TREC（Classification，Few-shot）、TriviaQA（F1，Few-shot）、SAMSum（ROUGE，Few-shot）、LCC（Similarity，Code Completion）、RepoBench-P（Similarity，Code Completion）
    - **Needle-in-a-Haystack (NIAH)**：passkey retrieval，使用 Paul Graham Essays 填充背景，7-digit passkey
    - **ShareGPT**：真实 LLM serving workload 效率测试，平均 prompt 长度 161、输出长度 338

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/jy-yuan/KIVI
  - **算法 pipeline 详解**：

  **Prefill 阶段（伪代码）**：
  ```
  输入: X ∈ R^{l_prompt × d}
  1. X_K = X·W_K, X_V = X·W_V  // 计算 key/value
  2. X_V_g = X_V[:l_prompt-R], X_V_r = X_V[l_prompt-R:]  // split value
  3. Q(X_V_g) = GroupQuant(X_V_g, dim=token, G=32)  // per-token group quant
  4. Q(X_K_g), X_K_r = KeyQuant(X_K)  // per-channel group quant + residual
  5. KV cache = {Q(X_K_g), X_K_r, Q(X_V_g), X_V_r}  // 存储量化缓存
  6. return X_K, X_V  // 传给下一层的是全精度
  ```

  **KeyQuant 函数**：
  ```
  procedure KeyQuant(X_K ∈ R^{l×d}):
    r = l % R           // 不能被R整除的余数
    X_K_g = X_K[:l-r]   // grouped 部分
    X_K_r = X_K[l-r:]   // residual 部分（FP16）
    Q(X_K_g) = GroupQuant(X_K_g, dim=channel, numGroup=l//G)  // 沿channel维度分组量化
    return Q(X_K_g), X_K_r
  ```

  **Decoding 阶段（伪代码）**：
  ```
  输入: KV cache, t ∈ R^{1×d}
  1. t_Q = t·W_Q, t_K = t·W_K, t_V = t·W_V
  2. X_K_r = Concat([X_K_r, t_K], dim=token)  // 新token加入residual
  3. X_V_r = Concat([X_V_r, t_V], dim=token)
  4. if len(X_K_r) == R:  // residual满了，量化并移入grouped
       Q(X_K_r) = KeyQuant(X_K_r)
       Q(X_K_g) = Concat([Q(X_K_g), Q(X_K_r)], dim=token)
       X_K_r = empty
  5. if len(X_V_r) > R:
       Q_outdated = GroupQuant(X_V_r[:-R], dim=token, G=32)
       Q(X_V_g) = Concat([Q(X_V_g), Q_outdated], dim=token)
       X_V_r = X_V_r[-R:]
  6. A = Concat([t_Q·Q(X_K_g)^T, t_Q·X_K_r^T], dim=token)  // tiled matmul
  7. A_g = Softmax(A)[:-R], A_r = Softmax(A)[-R:]
  8. t_O = A_g·Q(X_V_g) + A_r·X_V_r  // 混合精度 attention output
  9. return t_O
  ```

  **张量计算关键**：
  - Key cache: X_K ∈ R^{l×d}，沿 channel(dim) 维度分组量化，每 G=32 个 token 一组
  - Value cache: X_V ∈ R^{l×d}，沿 token(dim) 维度分组量化，每 G=32 个 channel 一组
  - Attention score: 使用 tiled matrix multiplication 分别计算 grouped quantized 部分和 residual FP16 部分，Concat 后 Softmax
  - Attention output: 按 residual 划分享 softmax 权重 A_g/A_r，分别与 quantized value 和 FP16 value 做矩阵乘法后求和

## KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - **KBVQ-MoE**：第一个专门为 MoE 架构设计的向量量化（VQ）框架，包含两个创新模块：
    - **IDRE（Input-driven Redundancy Elimination）**：KLT 引导的 SVD 分解。先对输入激活做 Karhunen–Loève Transform（KLT）构建输入相干基（input coherence basis），将各 expert 权重投影到此空间形成统一表示 `W̄`，再对 `W̄` 做 SVD 提取 top-k 主导共享分量 `W_share`（保留全精度），剩余 expert-specific 分量 `W_quant` 交给 VQ 量化。KLT 确保提取方向同时对齐输入能量和跨 expert 权重能量。
    - **BCOS（Bias-Corrected Output Stabilization）**：对 expert-specific 权重做 VQ 量化后，以 channel-wise affine compensation（scale s + bias b）校正量化输出，使得校正后的每个 channel 的 mean/variance 与 FP16 输出对齐。s 和 b 通过 MMSE 闭式解计算：`s_j ≈ σ_{y_j}/σ_{ŷ_j} - 1`, `b_j = μ_{y_j} - (1+s_j)μ_{ŷ_j}`。
  - 实验比较：
    - 与 RTN、GPTQ（scalar quantization）、MoEQuant（MoE 专用量化）、Direct VQ（直接向量量化）比较，在 2-bit 和 3-bit 下的 WikiText2 perplexity 和 7 个零样本任务（ARC-E, ARC-C, HellaSwag, LAMBADA-openai, LAMBADA-standard, PIQA, WinoGrande）的 Avg Acc。
    - Plugin 实验：将 IDRE+BCOS 作为插件集成到 GPTVQ 和 VPTQ 中对比性能提升。
    - 消融实验：KLT vs 无 KLT 的 SVD；不同 SVD 截断秩 k/n 比例；IDRE 和 BCOS 各自贡献；BCOS 中 mean 和 variance 校正各自贡献。
    - 与 MoE 结构压缩方法（Sub-MoE, D2-MoE, EAC-MoE）在 Mixtral-8×7B 上的公平对比。
    - 更具挑战性 benchmark：MMLU, MathQA, GSM8K, HumanEval。
    - 解码速度测试：BF16 vs Quantized 的 tokens/s 加速比。

- 硬件平台是什么，配置是什么。
  - 量化实验：NVIDIA RTX A6000 GPU
  - MoE 压缩方法对比实验：NVIDIA RTX A100 GPU, PyTorch 2.1
  - 解码速度测试：论文未明确说明测试 GPU，仅报告 Qwen1.5-MoE-A2.7B 在 1k input tokens 下 BF16 为 22.31 tokens/s，2-bit quantized 为 35.24 tokens/s（加速 1.58×）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, Mixtral-8x7B, DeepseekV2-Lite（以及 DeepSeekMoE-16B 用于 challenge benchmark）
  - Calibration 数据集：RedPajama 数据集，random seed 42，随机采样 256 条，sequence length 4096
  - Perplexity 评测：WikiText2，sequence length 4096
  - 零样本评测（7 个数据集）：Arc-Challenge, Arc-Easy, HellaSwag, LAMBADA-openai, LAMBADA-standard, PIQA, WinoGrande
  - 挑战性 benchmark：MMLU, MathQA, GSM8K, HumanEval
  - 评测工具：LM-Evaluation-Harness (v0.4.0)

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：论文未明确说明开源链接
  - KBVQ-MoE 算法完整流程（Algorithm 1 from Appendix A.8）：
    ```
    === Pre-Process: KLT-guided SVD (IDRE) ===
    输入: expert weights {W^(i)}_{i=1..n}, 输入激活 X
    
    1. 计算输入协方差: C_X = 1/(B-1) X^T X ∈ R^{ic×ic}
    2. 特征分解: C_X = U_KLT Λ_KLT U_KLT^T
    3. 输入相干基: U_X = U_KLT Λ_KLT^{1/2}
    4. for i=1 to n:
         W̃^(i) = W^(i) U_X          # 投影到输入相干空间
    5. 堆叠所有 expert: W̄ = [W̃^(1); ...; W̃^(n)] ∈ R^{(n·oc)×ic}
    6. SVD: W̄ = (U Σ V^T)^T
    7. 选 top-k: U_k = U_{:,1:k}, V_k = V_{:,1:k}, Σ_k = Σ_{1:k,1:k}
    8. 按 expert 划分 V_k: V_k = [Σ_k V_k^(1); ...; Σ_k V_k^(n)]
    9. for i=1 to n:
         U_share = U_X^{-1} U_k              # ic×k 共享映射
         W_share^(i) = (U_share (V_k^(i))^T)^T   # oc×ic 共享分量
         W_quant^(i) = W^(i) - W_share^(i)       # 残差（expert-specific）
    
    === Quantization: Vector Quantization of W_quant ===
    11. for i=1 to n:
         将 W_quant^(i) 划分为 d 维子向量 {z}
         用 K-means++ 初始化 codebook C = {c_1,...,c_K}
         训练 VQ codebook via K-means (100 iterations)
         for each sub-vector z:
           q = argmin_j ||z - c_j||^2
           z_q = c_q
         得到 W_quant,VQ^(i)
    
    === Post-Process: Bias Correction (BCOS) ===
    13. 定义量化权重: Ŵ^(i) = W_share^(i) + W_quant,VQ^(i)
    14. 从 calibration data 估计 per-channel 统计量:
          μ_y, σ_y  (原始输出 y = W^(i)x)
          μ_ŷ, σ_ŷ  (量化输出 ŷ = Ŵ^(i)x)
    15. 计算校正参数:
          s_j = σ_{y_j} / σ_{ŷ_j} - 1
          b_j = μ_{y_j} - (1+s_j) μ_{ŷ_j}
    16. 校正输出: y_corr = (1+s) ⊙ (Ŵ^(i)x) + b
    ```
  - 关键超参数：
    - SVD 截断秩 k: 推荐 k = ic/128（full rank 的 1/128），此时平均 bit-width 增加约 0.08 bits
    - VQ 子向量长度 d: 设置为 4
    - K-means: K-means++ 初始化，100 iterations
    - BCOS 额外参数: 每层仅 2·oc 个参数（scale + bias per channel），推理时额外 FLOPs < 0.1%
  - 压缩效果：Qwen1.5-MoE-A2.7B 在 2-bit 下压缩率 87%，有效位宽 2.08 bits；实际存储从 27.9GB(FP16) 降至 4.3GB

## Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations

## Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations

- 属于算法pipeline的实现是什么？实验比较什么？
  - **BOF4**：基于改进 Lloyd's EM 算法，最小化 block-wise absmax 量化后网络权重的端到端 MSE 或 MAE，计算信息论最优的 4-bit 量化码本（codebook）。关键创新在于 centroid 更新公式考虑了 block maximum 的分布权重（MSE 用 `w_b^max` 的平方加权，MAE 用 `w_b^max` 加权），而非直接最小化归一化权重的量化误差。
  - **BOF4-S**：将 block-wise absmax normalization 改为 signed absmax normalization，即归一化常数取 signed absolute block maximum（保持符号而非取绝对值）。这使得归一化后只需固定 2 个 reconstruction level（0 和 1），而非 3 个（-1, 0, 1），降低量化误差。
  - **OPQ（Outlier-Preserving Quantization）**：混合精度策略，将 outlier weights 以 bfloat16 + 64-bit 位置索引单独存储。Outlier 判定基于 `|w_{b,i}| > σ_b * F_M^{-1}(q)`，其中 σ_b 为 block 内标准差，F_M^{-1}(q) 为绝对 block maxima 分布的 q-分位数（q=0.95）。Outlier 在量化前替换为 0，改善归一化后权重分布与理论分布的吻合度。
  - 实验比较：与 NF4（QLoRA 中的 NormalFloat）和 AF4（AbnormalFloat）比较以下指标：
    - (a) 合成数据（Gaussian weights）的 MAE/MSE 随 block size I 变化曲线
    - (b) 真实 LLM 权重的 MAE、MSE、perplexity（WikiText-2, LAMBADA）
    - (c) 下游任务准确率（MMLU, ARC-Challenge, HellaSwag, PIQA, SIQA, WinoGrande）
    - (d) QLoRA 微调后指令跟随（IFEval）和代码生成（HumanEval+, MBPP+）的准确率

- 硬件平台是什么，配置是什么。
  - 微调：1× NVIDIA A100 40GB，每轮 < 8 小时
  - 评估：NVIDIA RTX 3080 10GB 或 A100 40GB
  - 推理运行时测试：NVIDIA RTX 4070 Ti Super

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-3.1（3B, 8B）、Llama-3.2（3B, 8B）、Qwen-2.5（0.5B, 3B, 7B）、Mistral-7B-v0.3
  - 数据集：WikiText-2（perplexity，rolling log-likelihood，max seq len 2048）、LAMBADA（perplexity）
  - Benchmark：MMLU（few-shot）、ARC-Challenge、HellaSwag、PIQA、SIQA、WinoGrande
  - 微调数据集：Unnatural Instructions（指令跟随）、Magicoder-OSS-Instruct-75K（代码生成）
  - 超参数：AdamW optimizer, lr=4e-5, β1=0.9, β2=0.999, batch size=16, 1875 steps, max_grad_norm=0.3, LoRA dropout=0.1, 不做 double quantization

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/ifnspaml/bof4
  - **Block-wise absmax 量化流程**：
    ```
    # Step 1: Block partitioning
    W_flat = W.reshape(-1)  # flatten weight tensor
    blocks = W_flat.reshape(B, I)  # B blocks, each size I
    
    # Step 2: Block-wise normalization
    for b in 1..B:
        w_max[b] = max_i |blocks[b, i]|           # absmax (BOF4)
        # 或 w_max[b] = blocks[b, argmax_i |blocks[b,i]|]  # signed (BOF4-S)
        x[b, i] = blocks[b, i] / w_max[b]          # normalize to [-1, 1]
    
    # Step 3: Scalar quantization
    for b in 1..B:
        for i in 1..I:
            l = argmin_ℓ |x[b,i] - x̂(ℓ)|           # nearest codebook entry
            Ŵ[b, i] = w_max[b] * x̂(l)              # decode
    ```
  - **BOF4 EM 算法（MSE 优化，Monte-Carlo 方法）**：
    ```
    # Initialize codebook x̂(1..16), fix x̂(8)=0, x̂(1)=-1, x̂(16)=1
    # Sample W ~ N(0,1) with shape B×I
    repeat until convergence:
        # 1. Assignment (nearest neighbor)
        for each block b, weight i:
            x = w[b,i] / w_max[b]
            ℓ = argmin_j |x - x̂(j)|
            assign x to region R_ℓ
        
        # 2. Centroid update (MSE) - modified from standard Lloyd's
        for ℓ in 2..15 (skip fixed levels):
            # Collect all x_k in R_ℓ across all blocks, with w_k = w_max of their block
            # Weighted mean (Eq. 6):
            x̂(ℓ) = Σ_k (w_k² * x_k) / Σ_k (w_k²)
    ```
  - **BOF4 EM 算法（MAE 优化）**：centroid 改为 weighted median（Eq. 8）：`x̂(ℓ) = median_W(x_1..x_K; w_1..w_K)`，即加权绝对偏差最小的点。
  - **BOF4 vs BOF4-S 的区别**：BOF4 固定 `x̂(1)=-1, x̂(8)=0, x̂(16)=1`（3 个固定值）；BOF4-S 固定 `x̂(8)=0, x̂(16)=1`（2 个固定值），因为 signed normalization 后归一化权重分布只在 x=1 有一个离散概率质量 `1/I`。
  - **OPQ 算法流程**：
    ```
    # Step 1: Block-level standard deviation
    for b in 1..B:
        σ_b = std(blocks[b, :])  # corrected sample std
    
    # Step 2: Outlier detection
    threshold = F_M^{-1}(0.95)  # 95th percentile of absolute block maxima
    for each weight w[b,i]:
        if |w[b,i]| > σ_b * threshold:
            mark as outlier -> store in bfloat16 + 64-bit position index
            replace w[b,i] = 0 in tensor
    
    # Step 3: Quantize non-outlier weights with BOF4(-S) as usual
    # Step 4: Decoding: read outlier positions, restore bfloat16 values
    ```
  - **码本示例**（BOF4-S MSE, I=64）：`x̂ = [-0.8568, -0.6693, -0.5235, -0.4005, -0.2911, -0.1900, -0.0939, 0.0, 0.0888, 0.1795, 0.2743, 0.3760, 0.4887, 0.6189, 0.7791, 1.0]`

## GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

- 属于算法pipeline的实现是什么？实验比较什么？
  - **GuidedQuant**：一种 PTQ 方法，利用 end loss 梯度信息构建 layer-wise quantization objective，通过 block-diagonal Fisher 近似保留同一 output channel 内的 cross-weight dependencies。同时引入 averaging approximation 将 Hessian 矩阵按 output channel 分组平均，大幅减少内存和计算开销。
  - **LNQ**：一种新的 Layer-wise Non-uniform Quantization 算法，采用 alternating minimization（交替优化 codebook 和 assignment），codebook 使用闭式解（least squares），assignment 使用 cyclic coordinate descent（CD）优化，保证目标函数单调递减。
  - 实验比较三类量化格式：(1) weight-only scalar (uniform/non-uniform)、(2) weight-only vector、(3) weight-and-activation。GuidedQuant 作为 plugin 应用于 LNQ、QTIP、SpinQuant，分别与 SqueezeLLM、GPTVQ 1D、QuIP、AQLM、QTIP、SpinQuant 等 baseline 比较 perplexity（WikiText2/C4）和下游任务准确率（zero-shot/few-shot）。

- 硬件平台是什么，配置是什么。
  - GPU：RTX 4090（吞吐测试）、RTX 6000 Ada（量化过程）、RTX 3090（吞吐测试）、A100（梯度缓存）
  - 量化时使用 1×RTX 6000 Ada 或 1-8×R6A 并行，梯度缓存使用 1-6×A100；Hessian 缓存使用 4×RTX 6000 Ada

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-2（7B/13B/70B）、Llama-3（8B/70B）
  - 校准数据：RedPajama（1024 sentences × 4096 tokens for weight-only；WikiText2 128 sentences × 2048 tokens for weight-and-activation）
  - 评估数据集：WikiText2（perplexity）、C4（perplexity）
  - 下游任务：BoolQ、PIQA、SIQA、HellaSwag、WinoGrande、ARC-easy、ARC-challenge、OBQA（8 zero-shot tasks avg）、MMLU（5-shot）

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/snu-mllab/GuidedQuant
  - **GuidedQuant 算法流程（Algorithm 1）**：
    1. 将每层 output channels 划分为 g 组：`J_k = {(d_out/g)*(k-1)+1, ..., (d_out/g)*k}`
    2. 用一次反向传播计算 end loss 对各层输出的梯度，按组平均平方梯度：`s_k = (1/|J_k|) * sum_{j in J_k} (∂ℓ/∂z_j)^2`
    3. 逐层逐组计算加权 Hessian：`H̄_k = X^T * Diag(s_k) * X`（shape: d_in × d_in）
    4. 调用 base quantizer Q（如 LNQ、QTIP GPTQ 等）量化该组内的权重：`Ŵ[:, J_k] = Q(H̄_k, W[:, J_k])`
  - **LNQ 算法流程（Algorithm 2）**：
    1. Cholesky 分解：`H = LL^T`
    2. 对每个 output channel j 交替优化：
       - Codebook 闭式解：`c = (P^T H P)^{-1} P^T H w_j`（least squares，用 torch.linalg.lstsq）
       - Assignment 优化：cyclic CD，对每个输入维度 i 坐标下降：
         `Ŵ_i = Round(W_i - H_i,others/H_ii * (Ŵ_others - W_others))`
       - CD 保证单调递减，T 为交替迭代次数，K 为 CD 循环轮数
  - **三种 integration 方式**：
    - LNQ + GQuant：将 layer-wise Hessian `X^TX` 替换为每组的 `H̄_k`
    - QTIP + GQuant：将 BlockLDLQ 的 Hessian 替换为 `H̄_k`
    - SpinQuant + GQuant：将 GPTQ weight quantizer 的 Hessian 替换为 `H̄_k`
  - 超参数：g=4 (7B/13B), g=2 (70B) for weight-only；g=1 for weight-and-activation; LNQ 中 T=2, K=4 (7B/13B); T=1, K=4 (70B)

## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MxMoE 是一个面向 MoE 模型的混合精度量化框架，核心包含两部分：(1) **硬件感知的 bitwidth 分配**：对 MoE block 中每个 expert 的每个 linear block（gate_proj, up_proj, down_proj），在由参数量化敏感度 Δ_{i,j,k}、expert 激活频率和硬件资源构成的多维设计空间中，通过 ILP（整数线性规划）求解最优混合精度量化方案。目标函数为 min L^r · T^{1-r}，其中 L 为量化输出扰动（Euclidean distance）、T 为基于 tile 级 profiling 的执行时间，r 为精度-性能权衡超参数（weight-only 极低比特 r=1，weight-activation r=0.75）。求解后使用 randomized Hadamard 变换 + GPTQ-based 量化完成权重 quantization；(2) **自动混合精度 Group-GEMM kernel 生成**：micro-kernel specialization + resource configuration + tile scheduling。量化方案以 linear-block 为粒度（而非 expert 级），校准使用 WikiText2 的 128 sequences × 4096 tokens。
  - 实验比较：
    - Weight-only quantization（2.25-bit / 3.25-bit 平均位宽）：MxMoE vs GPTQ（含相同 random Hadamard 变换预处理），在 DeepSeek-V2-Lite、Qwen1.5-MoE、Qwen2-MoE、Mixtral-8×7B 上对比 WikiText2 perplexity 和 7 个零样本任务（AC/AE/HS/LO/LS/PQ/WG）的 Avg Acc
    - Weight-activation quantization（5-bit 平均位宽）：MxMoE vs QuaRot（4-bit uniform），r=0.75
    - Ablation：linear-block vs expert-level bitwidth allocation granularity；超参数 r 对 accuracy-performance tradeoff 的影响
    - 性能分析：MoE block 计算吞吐量（memory-bound 512 tokens / compute-bound 8192 tokens），混合精度 vs uniform precision vs full-precision

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA RTX 4090 GPU（24GB 显存）

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-V2-Lite（16B 总参，2.4B 激活，64+2 experts, TopK=6）、Qwen1.5-MoE（14.3B 总参，2.7B 激活，60+4 experts, TopK=4）、Qwen2-MoE-Instruct（54.2B 总参，7B 激活，64+8 experts, TopK=8）、Mixtral-8×7B-Instruct-v0.1（46.7B 总参，12.9B 激活，8 experts, TopK=2）
  - 校准数据：WikiText2 训练集，128 条 sequence，每条长度 4096
  - Perplexity 评测：WikiText2
  - 零样本评测：Arc-Challenge, Arc-Easy, HellaSwag, LAMBADA-openai, LAMBADA-standard, PIQA, WinoGrande
  - Expert 激活频率统计：HumanEval-X 数据集

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/cat538/MxMoE
  - 算法 pipeline 核心流程（伪代码）：
    ```
    === 阶段 1: 离线校准与统计收集 ===
    输入: MoE 模型 (M 层), 校准数据 X_cal (128 seqs × 4096 tokens)

    1. 对每个 MoE block 的每个 linear-block (expert i ∈ [1,E], block j ∈ [1,N]):
       for each supported quantization scheme k ∈ S:
           计算量化扰动 Δ_{i,j,k}:
             W_q = GPTQ_quant(W_{i,j}, scheme k)
             Ô = MoE_block_forward(X_cal, with W_q at (i,j), rest FP16)
             O = MoE_block_forward(X_cal, all FP16)
             Δ_{i,j,k} = ||Ô - O||₂

    2. 统计 expert 激活频率 f_i = P(expert i | token)

    === 阶段 2: ILP 求解混合精度方案 ===
    3. 构建 ILP:
       minimize L^r · T^{1-r}
       where:
         L = Σ_{i,j,k} Δ_{i,j,k} · x_{i,j,k}
         T = (1/P) · Σ_{i,j,k,t} c_{i,j,k,t} · y_{i,j,k,t} · x_{i,j,k}
       s.t.:
         x_{i,j,k} ∈ {0,1}, Σ_k x_{i,j,k} = 1  (每 linear block 一方案)
         y_{i,j,k,t} ∈ {0,1}, Σ_t y_{i,j,k,t} = 1  (每 linear block 一 tile 配置)
         Σ_{i,j,k} W_{i,j,k} · x_{i,j,k} ≤ M  (内存预算)

    === 阶段 3: 量化执行 ===
    4. 按 ILP 输出 {x_{i,j,k}} 逐 linear block 量化:
       - randomized Hadamard 变换 (incoherence processing, 来自 QuaRot)
       - GPTQ per-channel/per-group symmetric/asymmetric min-max quantization
       - 权重离线完成; 激活运行时动态量化
    ```
  - 关键参数：S = {W2A16, W4A16, W4A4, W4A4-g128, W8A8, ...}；平均位宽 2.25/3.25（weight-only）或 5（weight-activation）；r=1 或 0.75
  - 核心结果：2.25-bit 下 WikiText2 PPL 比 GPTQ 低 2.4（DeepSeekV2-Lite）；W5A5 速度比 full-precision 快 3.4×，比 uniform W8A8 快 29.4%

## Focused Quantization for Sparse CNNs

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 **Focused Quantization (FQ)**，一种针对稀疏 CNN 的混合量化策略，将 shift quantization（权重量化为 2 的幂次值 `{0,±1,±2,±4,...}`）与 recentralized quantization 结合。核心创新：**(1) Recentralized Quantization**：对稀疏层的权重分布拟合高斯混合模型（GMM，2 个分量），用 EM 算法找到高概率密度区域，独立对每个区域做 shift quantization（先减均值除以标准差，shift quantize 后再反变换），使量化层级集中匹配权重分布。**(2) Wasserstein 分离判定**：用 2-Wasserstein 距离衡量两个高斯分量的分离程度，当 `W(c₁,c₂) < w_sep`（默认 2.0）时退化为普通 shift quantization，自适应选择量化策略。**(3) MDL 视角优化**：将量化建模为最小描述长度（MDL）优化，误差代价 `L_E` 为交叉熵，复杂度代价 `L_C` 为 KL 散度。**(4) 完整压缩流水线**：Dynamic Network Surgery 细粒度剪枝 → FQ 量化 → INQ（增量量化，逐步增加量化比例 25%→50%→75%→87.5%→100%，每步 fine-tune 3 epochs, LR=0.001，最后一步 10 epochs）→ Huffman 编码。

  实验对比：
  - Baselines：TTQ（三元量化）、INQ（shift quantization, 2/3/5 bit）、ADMM（极低比特）、ABC-Net（5 bases 二值卷积）、LQ-Net（可学习量化, 2 bit）、D&Q（蒸馏+量化）、Coreset-Based Compression、ThiNet（filter 剪枝）、Clip-Q（剪枝+量化+权重共享）
  - 评估配置：5-bit FQ、7-bit FQ
  - 评估指标：Top-1/Top-5 准确率、压缩率 CR（×）、模型大小（MB）、Sparsity（%）、logic gate 数量（硬件效率）
  - 消融实验：Wasserstein 分离阈值 w_sep 从 1.0 到 3.5 以 0.1 递增（CIFAR-10 9层 CNN，每个值训练 100 次）
  - 渐进量化消融：逐步量化 weights（5-bit FQ）→ activations（8-bit integer）→ BN parameters（16-bit integer）

- **硬件平台是什么，配置是什么。**
  训练平台论文未详细说明。硬件资源评估针对自定义加速器，使用 3×3 卷积、padding=1、8×8×100 输入激活、8×8×100 输出，估算双输入逻辑门数下界（unrolled architecture, same throughput）。FPGA 加速器生成参见配套工作 [24] "Automatic generation of multi-precision multi-arithmetic CNN accelerators for FPGAs"（ICFPT 2019）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：ResNet-18、ResNet-50、MobileNet-V1、MobileNet-V2；CIFAR-10 快速分类器（9 层 CNN，用于 w_sep 消融）
  - 数据集/Benchmark：ImageNet（ILSVRC 2012）用于主要评估；CIFAR-10 用于超参数消融
  - 剪枝方法：Dynamic Network Surgery [6]

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/deep-fry/mayo（Mayo 框架）。

  **FQ 算法pipeline核心流程（逐层处理）**：

  ```
  # ===== 阶段1: 剪枝后权重预处理 =====
  # 输入: W ∈ R^{Cout, Cin, Kh, Kw}  (已由 Dynamic Network Surgery 剪枝)
  # z_θ ∈ {0,1}: pruning mask

  # ===== 阶段2: FQ Recentralized Quantization (逐层) =====
  # Step 2.1: 拟合高斯混合模型
  θ_nonzero = {w ∈ W | w ≠ 0}  # 非零权重集合
  初始化: μ_- = mean({θ < 0}), σ_- = std({θ < 0})
          μ_+ = mean({θ > 0}), σ_+ = std({θ > 0})
          λ_- = λ_+ = 0.5

  # EM 算法求 MLE
  repeat until convergence:
      # E-step: 计算每个权重属于各分量的后验概率
      γ_c(θ) = λ_c * N(θ|μ_c, σ_c) / Σ_j λ_j * N(θ|μ_j, σ_j)
      # M-step: 更新参数
      N_c = Σ_θ γ_c(θ)
      μ_c = Σ_θ γ_c(θ) * θ / N_c
      σ_c^2 = Σ_θ γ_c(θ) * (θ-μ_c)^2 / N_c
      λ_c = N_c / |θ|

  # Step 2.2: 分量分配与量化
  for each weight θ:
      m_θ = argmax_c λ_c * N(θ|μ_c, σ_c)  # 选择最可能的分量
      # Recentralize: 归一化到零均值
      θ_norm = (θ - μ_{m_θ}) / σ_{m_θ}
      # Shift quantize: 量化为 2 的幂
      θ_hat_norm = Q^{shift}_{n,b}(θ_norm)  # n-bit shift quantization
      # De-normalize
      Q[θ] = z_θ * α * (θ_hat_norm * σ_{m_θ} + μ_{m_θ})

  # Step 2.3: Wasserstein 分离判定
  # 归一化方差
  σ²_global = Var(θ_nonzero)
  W(c₁,c₂) = ((μ₊-μ₋)² + (σ₊-σ₋)²) / σ²_global
  if W(c₁,c₂) < w_sep (default 2.0):
      退化为 shift quantization (精度高 1 bit，因不需要 m_θ bit)
  ```
  
  最终量化后值形式：`Q_c^{rec}[θ] = Q^{shift}_{n,b}[(θ-μ_c)/σ_c] * σ_c + μ_c`
  
  其中 `Q^{shift}_{n,b}[v] = s * 2^{e-b}`（s ∈ {-1,0,1}, e ∈ [0,2^k-1], b 为逐层 bias）。
  
  乘法被替换为 bit-shift：`x * (s * 2^{e-b}) = s * (x << (e-b))`（或 `>>` 当 e<b 时）。

  **硬件实现优化**：
  - μ₊, μ₋ 量化为最近的 2 的幂次值
  - σ₊ 和 σ₋ 约束为相等，可融合到逐层缩放因子 α 中
  - α 可融入 BN 融合，消除推理时乘法
  - 5-bit FQ 内部使用 3-bit 无符号 shift quantization（1 bit sign + 1 bit component selection + 3 bit shift value = 5 bit total）
  - 最终 dot-product 仅含 bit-shift 和整数加法，无浮点乘法

## FlatQuant: Flatness Matters for LLM Quantization

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 FLATQUANT（Fast and Learnable Affine Transformation），一种新的 PTQ 方法，通过为每个线性层学习最优仿射变换来增强权重和激活的平坦度（flatness），从而降低量化误差。核心创新包括三个组件：**(1) 可学习仿射变换（LT）**：使用 Kronecker 乘积 P = P₁ ⊗ P₂ 构造两个轻量矩阵替代完整的大矩阵 P ∈ R^{n×n}，将内存开销降至 n/2 倍、计算节省 √n/2 倍（取 n₁=n₂=√n 时最优）；**(2) 可学习逐通道缩放（PS）**：在预量化变换前引入 diag(c) 缩放向量，可融合到前层 LayerNorm 或线性层中消除推理开销；**(3) 可学习裁剪阈值（LCT）**：在仿射变换后对权重和激活应用 sigmoid 后的裁剪阈值 α_w, α_a ∈ (0,1)。训练采用逐块 PTQ 方式，MSE 损失在 128 条校准数据（WikiText-2，2048 tokens/条）上优化 15 epochs，使用 AdamW（LR=5e-3，cosine annealing），batch size=4。使用 SVD 分解 + AMP 训练实现 50% 训练时间缩减。

  实验对比：
  - Baselines：SmoothQuant、OmniQuant、AffineQuant、QuaRot、SpinQuant、QUIK-4B
  - 量化配置：W4A4（RTN 和 GPTQ 两种 weight quantizer），W4A4KV4，W3A3KV3（极端低比特），weight-only（W4A16/W3A16），KV cache only（K2-4b + V2-4b）
  - 评估指标：WikiText-2/C4 perplexity；ARC-C/ARC-E/HellaSwag/LAMBADA/PIQA/Winogrande 零样本准确率；MT-Bench 多轮对话
  - 消融实验：LT/PS/LCT 各组件贡献（LLaMA-3-8B，RTN baseline PPL 1266.60 → LT only 8.50 → +PS 7.95 → +LCT 6.98）；校准集泛化（WikiText2/C4/Pile）；裁剪策略对比（变换前 vs 变换后 vs QuaRot 固定阈值）；混合精度方案
  - 架构泛化：Qwen-2.5-Instruct（7B/32B）、DeepSeek-V3-Base（671B MoE）、DeepSeek-R1

- **硬件平台是什么，配置是什么。**
  校准：单卡 GPU，LLaMA-3-8B 约需 26GB GPU 内存、0.9 小时（AMP+SVD 训练）。推理速度测试：NVIDIA RTX 3090 GPU，prefill seq_len=2048，decode 256 tokens。FP32 全精度训练备选方案需约 35.4GB 内存、2.2 小时。70B 模型校准时间约数小时。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：LLaMA-2（7B/13B/70B）、LLaMA-3（8B/70B）、LLaMA-3.1-8B-Instruct、Qwen-2.5-Instruct（7B/32B）、DeepSeek-V3-Base（671B MoE）、DeepSeek-R1
  - 校准数据：WikiText-2（128 segments, 2048 tokens each，默认）；消融中使用 C4、Pile
  - Perplexity：WikiText-2、C4
  - 零样本常识推理：ARC-Challenge、ARC-Easy、HellaSwag、LAMBADA、PIQA、Winogrande（lm-eval-harness）
  - 多轮对话：MT-Bench（GPT-4o 评估）
  - MoE 评估：C-Eval、MMLU（DeepSeek-V3）、AIME2024（DeepSeek-R1）

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/ruikangliu/FlatQuant。基于 HuggingFace Transformers + PyTorch 实现。

  **算法pipeline核心流程（LLaMA 架构，逐 block 量化）**：

  **Step 1：构建 Kronecker 仿射变换矩阵**
  对 hidden_dim=n 的线性层，分解 n = n₁ × n₂（取 n₁+n₂ 最小且 n₁≤n₂），构造 P₁ ∈ R^{n₁×n₁}、P₂ ∈ R^{n₂×n₂}，则 P = P₁ ⊗ P₂。
  对于 LLaMA-2-7B 的 hidden_dim=4096，最优分解 (n₁,n₂)=(64,64)；对于 intermediate_dim=11008，分解为 (64,172)。

  **Step 2：前向传播中的量化线性层**
  原始：Y = X W^T
  量化后（以单个线性层为例）：
  ```
  X̃ = reshape(X, [k, n₁, n₂])          # k=tokens, n₁×n₂=n
  X' = P₁^T ×₁ X̃ ×₂ P₂                 # 仿射变换，平滑激活分布
  X'_q = Q(X')                          # per-token 对称量化到 INT4
  W̃ = reshape(W, [m, n₁, n₂])
  W' = P₁^{-1} ×₁ W̃ ×₂ (P₂^{-1})^T      # 逆变换权重（离线预计算）
  W'_q = Q(W')                          # per-channel 对称量化到 INT4
  Y = X'_q W'_q^T                       # INT4 matmul (CUTLASS kernel)
  ```

  **Step 3：训练优化（逐 Transformer block）**
  对第 l 个 block，优化参数 Θ = {P₁, P₂, c, α_a, α_w}：
  ```
  min_Θ || F_l(X) - F̂_l(X; Θ) ||_F^2
  ```
  其中 F̂_l 将 block 内所有线性层替换为 Step 2 的量化版本。
  - P₁, P₂ 使用 Cayley 参数化保证正交性，SVD 求逆（P^{-1}=VΣ^{-1}U^T），AMP 训练
  - c 为逐通道缩放因子，训练后融合到前层权重/layer norm
  - α_w, α_a 经 sigmoid 后用于裁剪量化范围

  **Step 4：Transformer 集成**
  - Self-Attention：4 个变换矩阵 {P_a, P_o, P_h, P_v} — P_a 用于 Q/K/V 投影输入，P_o 用于输出投影输入，P_h/P_v 用于 per-head KV cache 变换（不分解，因 head dim 较小），P_o 与 P_v 融合减少开销
  - FFN：2 个变换 {P_ug, P_d} — P_ug 用于 gate+up 投影输入，P_d 用于 down 投影输入
  - 保留原始 LayerNorm（而非 QuaRot 的 RMSNorm 修改），使各 block 可学习独立的仿射变换

  **关键结果**：
  - LLaMA-3-70B W4A4：RTN 准确率下降 <1%（Avg 79.01 vs FP16 79.95），超越 SpinQuant 7.5%
  - LLaMA-2-70B W4A4 RTN WikiText-2 PPL：3.55（FP16 3.32，仅 +0.23）
  - 在线变换仅占 FP16 模型 FLOPs 的 2.61%，额外内存 3.41MB（LLaMA-2-7B）


## First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 FOEM（First-Order Enhanced Method），一种改进 GPTQ 的 PTQ 方法。核心创新是在量化误差补偿中显式引入一阶梯度项。GPTQ 仅用二阶泰勒展开（假设一阶项为零），但 FOEM 发现逐列量化过程中，对已量化列的补偿会导致剩余未量化 latent weights 偏离 full-precision 权重，产生不可忽略的一阶梯度。FOEM 通过 g(W) ≈ β(W − 𝕎)H 近似梯度（无需反向传播），代入 Lagrangian 求解后 Hessian 项自动消去，仅增加轻量权重差分运算。实验对比：
  - Baseline：FP16（全精度上限）、RTN（round-to-nearest）、GPTQ、GPTAQ
  - 量化配置：weight-only（W4A16、W3A16，group size 128），weight-activation（W4A4KV4 结合 SpinQuant 预训练旋转矩阵）
  - 评估指标：WikiText2/C4 perplexity（PPL），PIQA/ARC-Easy/ARC-Challenge/HellaSwag/Winogrande/BoolQ 零样本准确率，5-shot MMLU
  - 消融：β ∈ {0.1, 0.2, ..., 1.0} 灵敏度分析（W3A16 Llama3-8B），β≤0.5 持续提升，β>0.5 急剧退化
  - 架构泛化：SSM 模型 Mamba-1.4B（W3A16）

- **硬件平台是什么，配置是什么。**
  量化校准：单卡 NVIDIA A800-80GB GPU。70B 模型评估需 2× A800 GPU。推理速度测试使用 vLLM 部署。校准数据：C4 数据集随机 128 条序列，序列长度 2048。β=0.1（所有实验默认）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：Llama2-7B/13B、Llama3-8B/70B、Llama3.2-1B/3B、Qwen3-8B、Phi-1.5B、Mistral-7B、Mamba-1.4B
  - 校准数据：C4（128 samples, seq_len=2048）
  - Perplexity：WikiText2、C4
  - 零样本常识推理：PIQA、Winogrande、ARC-Easy、ARC-Challenge、HellaSwag、BoolQ
  - 知识推理：5-shot MMLU

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/Xingyu-Zheng/FOEM（AAAI 2026）。基于 gptqmodel 库集成。

  **算法pipeline核心流程（Algorithm 1：FOEM 单层量化）**：

  输入：FP 权重 W（m×n），校准输入 X，block size B
  1. 计算 Hessian：H = XX^T
  2. Cholesky 分解：L = Inverse_Cholesky(H + λI)，T = L^T（上三角）
  3. 保存原始权重副本：𝕎 ← W
  4. 按 block（size B）迭代：
     a. 对 block 内每列 j（j=i,...,i+B-1）：
        - 量化：Q_{:,j} ← quant(W_{:,j})
        - 一阶增强误差：E_{:,j-i} ← ((W_{:,j} − Q_{:,j}) − β(W_{:,j} − 𝕎_{:,j})) / T_{jj}
        - 补偿当前 block 内后续列：W_{:,j:(i+B)} ← W_{:,j:(i+B)} − E_{:,j-i} · T_{j,j:(i+B)}^T − β(W_{:,j} − 𝕎_{:,j})
     b. 补偿 block 外后续列：W_{:,(i+B):} ← W_{:,(i+B):} − E · T_{i:(i+B),(i+B):}^T

  **与 GPTQ 的核心差异**：
  - GPTQ 的补偿项：δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}（仅二阶）
  - FOEM 的补偿项：δw = −((w_q − ŵ_q) − β(w_q − 𝕎e_q^T))/T_{qq} · T_{q,q:} − β(W − 𝕎)
    额外减去 β(W_{:,j} − 𝕎_{:,j}) 项和分子中的 β(w_q − 𝕎e_q^T) 项，来自一阶梯度的近似
  - 梯度近似原理：g(W) ≈ (W − 𝕎)H，代入 Lagrangian 解后 H 和 H^{−1} 自动消去，无需显式计算
  - 额外开销：仅权重差分运算，无矩阵乘法。Llama3-8B 量化时间：GPTQ 825.50s，FOEM 828.90s（仅 +0.4%）


## FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出FedWSQ，结合 Weight Standardization (WS) 梯度过滤和 Distribution-Aware Non-Uniform Quantization (DANUQ) 两个算法pipeline组件。WS在local training中通过连续投影（投影到 span{w˜_n,m, 1}^⊥）过滤掉与WSP向量对齐的分量和mini-batch梯度均值分量，从而缓解非i.i.d.数据导致的client drift。DANUQ基于标准正态分布先验，通过暴力搜索预先计算出最小化期望量化误差的最优量化级别（QLs），量化LMPU时使用共享global scaling vector（EMA更新），避免传输额外量化参数。论文对比实验包括：
  - 全精度FL方法：FedAvg、FedProx、FedAvgM、FedADAM、FedDyn、FedMLB、FedLC、FedNTD、FedSmoo、FedDecorr、FedWon、FedRCL、FedACG
  - 量化FL方法：FedPAQ（1-bit uniform quantization）、FedHQ+（4-bit/1-bit）
  - NUQ方法消融：NF（NormalFloat）、FP（Floating Point）vs DANUQ
  - 比特策略消融：FBA（固定比特分配）和DBA（动态比特分配，每轮随机1/2/4-bit，期望2.3bits）
  - 评估指标：CIFAR-10/100、Tiny-ImageNet测试集准确率（1000轮后）、收敛曲线、loss landscape Hessian top eigenvalue

- **硬件平台是什么，配置是什么。**
  NVIDIA RTX 4090 GPU。PyTorch框架实现。SGD优化器，初始学习率0.1，weight decay 0.001，指数衰减因子0.995。100个clients，5%参与率。每轮local training 5个epoch，batch size使每个local epoch含10次迭代。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：默认ResNet-18（将BN替换为GN），WS应用于每个GN层前。额外测试ShuffleNet、VGGNet-9、SqueezeNet、MobileViT作为backbone验证泛化性
  - 数据集：CIFAR-10（10类）、CIFAR-100（100类）、Tiny-ImageNet（200类）
  - 非i.i.d.设置：Dirichlet分布 α∈{0.1, 0.3, 0.6}，α越小数据异质性越高

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/Seongyeol-kim/FedWSQ

  **算法pipeline核心流程（参考Algorithm 1伪代码）**：

  1. Server端每轮采样clients S_t，广播GMP W_g^{t-1} 和 global scaling vector s_g^{t-1}
  2. Client端local training（K步迭代）：
     - Forward: 对每层权重向量w_{n,m}应用WS：w̃_{n,m} = (ρ/σ(w_{n,m})) · (I - P_1) w_{n,m}
       - (I - P_1): 减去均值（投影到span{1}^⊥），去除DC分量
       - 除以σ并缩放：标准化到零均值单位方差，乘以ρ控制scale
     - Backward: 梯度经双重投影过滤 ∂L/∂w_{n,m} = (ρ/σ) · (I - P_1)(I - P_{w̃_{n,m}}) ∂L/∂w̃_{n,m}
     - Optimizer step: W_i^k ← W_i^{k-1} - η∇f_i(W_i^{k-1})
  3. Client端量化LMPU ΔW_i = W_i^K - W_g：
     - 逐层归一化：ΔW_{i,l} / s_{g,l}（除以global scale，假设归一化后∼N(0,1)）
     - DANUQ量化：将归一化值映射到预计算的最优QLs
       - 1-bit QLs: [-0.798, 0.798]（省略q_0=0约束）
       - 2-bit QLs: [-1.224, 0, 0.765, 1.724]
       - 4-bit QLs: [-2.654, -1.974, -1.508, -1.149, -0.834, -0.544, -0.269, 0, 0.230, 0.465, 0.708, 0.966, 1.248, 1.568, 1.968, 2.649]
       - Quantization boundaries: u_r = (q_{r-1} + q_r)/2，将[0, +∞)分成R+1个区间
       - 量化规则：x ∈ [u_r, u_{r+1}) → q_r
     - 传输量化后的 ΔW̄_i 和 local scale vector s_i（未经量化）
  4. Server端dequantize并聚合：
     - Dequantize: Δ_i^t ← (ΔW̄_i^t, s_i^t)，还原为全精度
     - Aggregate: Δ^t ← Σ_{i∈S_t} h_i Δ_i^t
     - Update GMP: W_g^t ← W_g^{t-1} + Δ^t
     - Update global scale: s_g^t ← (1-β)s_g^{t-1} + β·(1/|S_t|)·Σ_{i∈S_t} s_i^t，β=0.1

  **DANUQ QLs预计算原理**：
  目标是最小化 E[(Δw - Δw̄)^2] = Σ_{r=0}^R ∫_{u_r}^{u_{r+1}} (x - q_r)^2 p(x) dx，其中p(x)为N(0,1)的PDF。由于closed-form解难以获得（含高斯积分和误差函数），采用暴力搜索在合理范围内离散搜索最优QLs。搜索空间限制在经验范围内，使用并行处理加速。

  **FBA/DBA混合精度策略**：
  - FBA: 每个client固定比特宽度（从{1,2,4}中选择）
  - DBA: 每轮每个client随机分配比特宽度∼Uniform{1,2,4}，期望约2.3bits


## Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - **Student Float (SF4)**：基于 Student's t-distribution 推导的理论最优 4-bit 查找表数据类型。对 30+ DNN 进行 weight/activation 分布 profiling，发现大多数 DNN 分布由 Student's t-distribution（自由度 ν≈5）最优近似，而非正态分布。SF4 将概率质量均匀划分 16 份（固定 p₈=0.5 确保零无损表示），经 t-distribution 分位数函数 Q_S(p;ν) 映射后归一化到 [-1,1]，作为 NF4 的高精度替代品。
  - **Supernormal Support**：针对 E2M1 FP4 因正负零冗余浪费 1/16 值的问题，将负零重映射为额外超常值。提出两个变体：(a) super-range (SR) — 在分布边缘分配一个点扩展动态范围；(b) super-precision (SP) — 在分布内部增加一个点提升精度。同时将 SP 扩展到 APoT4。
  - 实验比较 baselines：
    - Lookup: NF4（Normal Float, 假设正态分布）
    - Integer: INT4, INT3
    - FP4 variants: E2M1-I (Intel), E2M1-B (bitsandbytes), E2M1 (standard), E3M0
    - Alternative: APoT4
  - 评估场景：weight-only PTQ（block size 128, RTN ± MSE clipping）、W4A4（± SmoothQuant）、GPTQ 比较、subchannel block size sweep (16/32/64/128/256)、3-bit 格式、CNN/ViT vision models

- 硬件平台是什么，配置是什么。
  - 量化实验：基于修改版 Intel Neural Compressor 库 + PyTorch；GPU 具体型号论文未明确说明
  - 硬件评估：Synopsys Design Compiler，TSMC 28nm 工艺，SystemVerilog RTL 综合 MAC 单元

- 模型是什么。数据集和bench分别是什么。
  - LLM：Mistral-7B, LLaMA2-7B, OPT-1B, OPT-6.7B, Phi-2, BLOOM-7B, Yi-6B, LLaMA-7B (多语言)
  - Vision：ResNet18, ResNet50, DenseNet121, ViT-B-16
  - 数据集/Benchmark：
    - LAMBADA（准确率 ↑）、WikiText-2（困惑度 ↓）
    - 零样本常识推理：HellaSwag, Winogrande, PIQA, BoolQ, ARC-c
    - 多语言 LAMBADA（EN/FR/DE/IT/ES）、ImageNet-1K（vision, Top-1 acc）
  - 校准方法：RTN（round-to-nearest）和 MSE clipping calibration；sub-channel block size 128；总计 4000+ 数据点

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/cornell-zhang/llm-datatypes
  - **SF4 导出算法（Algorithm 1）**：
    ```
    # 离线一次性计算
    δ = 0.5 * (1/32 + 1/30) = 0.0323
    # 在概率空间均匀分布 16 个点，p₈=0.5（零无损）
    p₁=δ, p₂,..., p₇, p₈=0.5, p₉,..., p₁₅, p₁₆=1-δ
    # 通过 t-distribution (ν=5) 分位数函数映射
    s̃ᵢ = Q_S(pᵢ; ν=5)   # Q_S = Student's t 分位数函数
    sᵢ = s̃ᵢ / maxᵢ|s̃ᵢ|    # 归一化到 [-1, 1]
    # s₁...s₁₆ 即为 SF4 的 16 个量化层级
    ```
  - **SF4 量化流程（类似 NF4/QLoRA 的 block-wise quantization）**：
    ```
    # Block-wise weight-only quantization
    W_flat = W.reshape(-1)
    blocks = W_flat.reshape(B, 128)     # block size 128
    for b in 1..B:
        w_max[b] = maxᵢ |blocks[b,i]|
        for i in 1..128:
            x = blocks[b,i] / w_max[b]       # normalize to [-1,1]
            idx = argminⱼ |x - sⱼ|           # nearest SF4 codebook entry
            Ŵ[b,i] = w_max[b] * s_{idx}      # decode
    # 存储：每个 weight 4-bit index + 每 block 一个 FP16/BF16 w_max
    ```
  - **Supernormal E2M1 原理**：
    ```
    # 标准 E2M1: 1 sign + 2 exp + 1 mantissa → 正值: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # 负零 = 冗余，浪费 6.25% (1/16) 位数空间

    # Super-precision (SP): 负零 → 5.0
    # 正值: {0, 0.5, 1, 1.5, 2, 3, 4, 5, 6}  — 分布内部增加 1 个层级

    # Super-range (SR): 负零 → 8.0
    # 正值: {0, 0.5, 1, 1.5, 2, 3, 4, 6, 8}  — 扩展动态范围
    ```
  - **APoT4 SP 原理**：APoT4 = (-1)^S (2^E + 2^Ẽ)，其中 E∈{0, 2⁻¹, 2⁻², 2⁻⁴}、Ẽ∈{0, 2⁻³}。SP 变体在同样的集合框架下复用负零位增加一个额外求和组合。
  - 核心洞察：E2M1 的形状分段逼近 SF4，这解释了为何 E2M1 比 INT4 精度更高——它对分布中心的密集区域分配了更多量化层级。

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

## YOCO (You Only Cache Once): Decoder-Decoder Architectures for Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  YOCO 提出了 decoder-decoder 架构替代传统 decoder-only Transformer，核心实现包含两个组件：(1) **Self-Decoder**：使用高效自注意力（gated retention 或 sliding-window attention），仅需 O(1) 常量 KV cache 内存；(2) **Cross-Decoder**：通过 cross-attention 复用 Self-Decoder 生成的全局 KV cache，使整体模型仅需 O(N) 而非 O(NL) 缓存。Self-Decoder 占前 L/2 层，Cross-Decoder 占后 L/2 层。实验比较了：(a) 与 OpenLLaMA-3B、StableLM-3B 的 LM Eval Harness 零样本下游任务性能（1T/1.6T tokens 训练）；(b) 从 160M 到 13B 的 scaling curves（对比 Llama-Transformer、YOCO_gRet、YOCO_SWA）；(c) 1M 上下文长度的 needle retrieval 和长序列 PPL；(d) 推理效率：GPU memory、prefill latency、throughput（32K-1M 长度，H100-80GB）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100-80GB GPU（推理 profiling 实验）。训练平台论文未明确说明（使用内部 CUBE 分布式训练系统，GPU 集群由 Ben Huntley 维护）。Triton kernel 实现基于 FLA (Flash-Linear-Attention) 库。

- 模型是什么。数据集和bench分别是什么。
  模型：3B YOCO 主模型（hidden=3072, layers=26, query heads=24, KV heads=8 with GQA, non-embedding params=2.8B）；scaling 模型从 160M 到 13B（7 种尺寸）。
  数据集：训练语料类似 StableLM-3B-4E1T 的 curated corpus，tokenizer 为 tiktoken-c1100k_base。
  Benchmark：LM Eval Harness（ARC-C, ARC-E, BoolQ, HellaSwag, OBQA, PIQA, Winogrande, SciQ）、Needle-in-a-Haystack（1M 长度）、Multi-Needle Retrieval（128K 长度）、长序列 NLL（book + repository-level code, >1M tokens）。Scaling 曲线使用 validation loss。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://aka.ms/YOCO。基于 FLA（https://github.com/sustcsonglin/flash-linear-attention）实现 gated retention 的 Triton kernel。

  算法pipeline（YOCO 推理流程）：
  1. **输入**：序列 x = [x_1, ..., x_N] ∈ R^{N×d_model}，输入 embedding X^0
  2. **Self-Decoder 前向**（L/2 层，gated retention 或 sliding-window attention）：
     - Gated Retention（默认，推理用 recurrent 模式）：
       - Q_n = (X_n W_Q) ⊙ Θ, K_n = (X_n W_K) ⊙ Θ̄, V_n = X_n W_V
       - γ_n = sigmoid(X_n W_γ)^{1/τ}
       - S_n = γ_n S_{n-1} + K_n^T V_n  （recurrent state update, O(1) memory）
       - gRet(X_n) = Q_n S_n
     - 或 Sliding-Window Attention：每个 query 仅关注窗口大小 C 内的 key
  3. **全局 KV Cache 生成**：K̂ = LN(X^{L/2}) W_K, V̂ = LN(X^{L/2}) W_V（单层全局缓存）
  4. **Cross-Decoder 前向**（L/2 层）：
     - Q̂^l = LN(X^l) W_Q^l
     - Y^l = Attention(Q̂^l, K̂, V̂) + X^l  （cross-attention，复用共享 KV cache）
     - X^{l+1} = SwiGLU(LN(Y^l)) + Y^l
  5. **Prefill 优化**：Cross-Decoder 的 cross-attention 仅依赖 K̂, V̂，prefill 阶段可在 Self-Decoder 完成后提前退出，仅需 L/2 层前向计算
  6. **输出**：X^L → softmax classifier → next-token prediction

  关键张量计算（gated retention, recurrent mode, 单 head）：
  - S_0 = 0 ∈ R^{d×d}
  - 对 timestep n=1..N：K_n ∈ R^d, V_n ∈ R^d, γ_n ∈ R
    - S_n = γ_n · S_{n-1} + K_n^T · V_n  （outer product, O(d²) state）
    - O_n = Q_n · S_n  （vector-matrix product, O(d²)）
  - 推理时仅维护 S_n 为中间状态，不存储 per-token KV cache

## DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  DMQ 提出了针对扩散模型后训练量化（PTQ）的异常值抑制方法，核心包含两个组件：(1) **Learned Equivalent Scaling (LES)**：学习通道级缩放因子 τ ∈ R^{Cin}，通过双向重分布异常值（Y = (X/τ)(τ^T ⊙ W)），最小化量化输出与原始输出的 MSE 来平衡权重和激活之间的量化难度。引入 **Adaptive Timestep Weighting**，基于各时间步的累积损失动态调整权重 λ_{t_i} = (1 - Λ_{t_i}/ΣΛ_{t'})^α，优先优化量化误差小但对最终质量影响关键的早期去噪步。(2) **Power-of-Two Scaling (PTS)**：针对 skip connection 等层中的极端异常值，使用通道级 2 的幂次缩放因子 δ，通过 bit-shift 操作高效处理（Y ≈ s^X s^W · Σ X̃ · (W̃ ≪ δ)），配合 **Voting Algorithm** 从校准集中通过统计共识选择鲁棒的 δ 因子。
  实验比较了 W8A8、W4A8、W4A6 量化配置下的无条件生成（FFHQ、LSUN-Bedroom、LSUN-Church）、条件生成（ImageNet）和文本引导生成（MS-COCO + Stable Diffusion v1.4），对比方法包括 Q-Diffusion、PTQD、EDA-DM、TFMQ-DM。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练/推理所用 GPU 型号。实验使用 PyTorch 框架，基于 LDM（latent-diffusion）和 Stable Diffusion 官方实现。Section E 中自定义 CUDA kernel 在 GPU 上实现了 W4A8 GEMM 的 5.17× 加速（vs PyTorch FP32 GEMM at M=3072）。

- 模型是什么。数据集和bench分别是什么。
  模型：LDM-8（LSUN Church 256×256）、LDM-4（LSUN Bedroom 256×256 / FFHQ 256×256 / ImageNet 256×256）、Stable Diffusion v1.4（text-to-image, 512×512）。数据集：FFHQ 256×256、LSUN-Bedrooms 256×256、LSUN-Churches 256×256、ImageNet 256×256、MS-COCO（text prompts）。评估指标：FID、sFID、IS（条件生成）、LPIPS、SSIM、PSNR、CLIP Score（文本引导生成）。采样使用 DDIM sampler，无条件/条件生成用 20 步，文本引导生成用 50 步。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/LeeDongYeun/dmq。基于 LDM（https://github.com/CompVis/stable-diffusion）和 guided-diffusion（https://github.com/openai/guided-diffusion）构建。

  算法pipeline（后训练量化流程）：
  1. **校准数据收集**：使用 DDIM sampler（无条件/条件 20 步，文本引导 25 步），每步采样 n=256 个数据点，总计 N=T×n 个校准数据点（无条件 5120，类条件 10240，文本引导 6400）
  2. **LES 学习阶段**：逐层优化通道级缩放因子 τ：
     - 前向：X̂ = X/τ, Ŵ = τ^T ⊙ W
     - 量化：X̂_q = MinMaxQ(X̂), Ŵ_q = MinMaxQ(Ŵ)
     - 损失：L_i = ||X_i W - Q(X̂_i) Q(Ŵ)||²
     - 自适应时间步加权：λ_{t_i} = (1 - Λ_{t_i}/ΣΛ_{t'})^α，Λ_t ← 0.95Λ_t + 0.05·E[L_i]
     - 总损失：L = (1/B) Σ λ_{t_i} L_i
     - 迭代次数：4000-6000，batch size=32（无条件/条件）/8（Stable Diffusion）
  3. **权重精炼**：使用 BRECQ 进行 block-wise 权重量化重建（Adaround 自适应舍入）
  4. **PTS 因子选择**（仅 skip connection 层）：
     - 候选选择：对每个校准样本 i 和通道 k，评估 δ ∈ {0,1,...,D}，选最小化量化误差的 δ*_{i,k}
     - 投票：δ_k^{mode} = mode({δ*_{i,k}})，一致性 r_k = Σ1{δ*_{i,k}=δ_k^{mode}}/N
     - 阈值化：若 r_k > κ(=0.85)，δ_k = δ_k^{mode}；否则 δ_k = 0（不缩放）
  5. **推理融合**：
     - LES：τ 融合到权重（τ^T ⊙ W 预计算）和激活 scale（τ ⊙ s^X 预计算），零推理开销
     - PTS：激活量化 X̃ = clamp(⌊X / (2^δ ⊙ τ ⊙ s^X)⌉, l, u)，权重加载时执行 Ŵ_{kj}^{shifted} = Ŵ_{kj} ≪ δ_k，矩阵乘 Y ≈ s^X s^W · Σ X̃ · Ŵ^{shifted}

  关键张量计算示例（W4A8 skip connection 层）：
  - 权重 W ∈ R^{Cin×Cout}，激活 X ∈ R^{B×Cin}
  - LES 融合后：Ŵ = τ^T ⊙ W（预计算），激活量化 scale = τ ⊙ s^X（预计算）
  - PTS 融合后：X̃ = round(X / (2^δ ⊙ τ ⊙ s^X))，Ŵ^{shifted} = Ŵ ≪ δ（bit-shift at kernel load time）
  - 输出：Y ≈ s^X · s^W · (X̃ @ Ŵ^{shifted})

## D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

- 属于算法pipeline的实现是什么？实验比较什么？
  D²-DPM 提出了针对扩散模型后训练量化（PTQ）的"双重去噪"（Dual Denoising）机制，在不重新训练的情况下精确保修正量化噪声对噪声估计网络的不利影响。核心实现包含两个关键步骤：(1) **时间步感知的量化噪声建模（TSQNM）**：利用高斯联合分布建模量化输出与量化噪声之间的关系，通过 BRECQ 校准数据在每个时间步估计联合分布参数（均值 μ 和方差 Σ），在推理时根据量化输出条件化地预测量化噪声的均值和协方差；(2) **双重去噪**：提出 S-D²（随机双重去噪）和 D-D²（确定性双重去噪）两种变体，分别从量化输出中减去估计的量化噪声或量化噪声均值，恢复扩散噪声分布，并修正 SDE 采样方程中的 drift coefficient 和 diffusion coefficient。实验比较了全精度 FP32 baseline、PTQ4DM、Q-diffusion、PTQD 等方法在 W8A8 和 W4A8 量化配置下的生成质量。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号或具体硬件配置。实验使用 PyTorch 框架执行，量化工具链基于 BRECQ 和 Adaround 构建。性能指标使用 BOPs（Bit Operations）衡量理论加速比，W8A8 实现 11.67× BOPs 降低和 3.99× 体积压缩，W4A8 实现 23.33× BOPs 降低和 7.95× 体积压缩。论文未提供在真实硬件上的 wall-clock 延迟测量。

- 模型是什么。数据集和bench分别是什么。
  模型：LDM-4 和 LDM-8（Latent Diffusion Models, Rombach et al. 2022），基于 U-Net + spatial transformer 的噪声估计网络。数据集：ImageNet 256×256（条件生成，classifier-free guidance scale=3.0/1.5）、LSUN-Bedrooms 256×256（无条件生成）、LSUN-Churches 256×256（无条件生成）。评估指标：FID、sFID、Inception Score (IS)、Precision、Recall（使用 OpenAI ADM TensorFlow 评估器，生成 50000 样本计算），以及 Size (MB) 和 BOPs (T) 作为效率指标。采样参数配置：条件生成 {scale=3.0, η=0.0|1.0, steps=20} / {scale=1.5, η=0.0|1.0, steps=250}，无条件生成 {η=0.0|1.0, steps=200}。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/TaylorJocelyn/D2-DPM（AAAI 2025）。基于 BRECQ PTQ 框架和 LDM 代码库构建。

  算法pipeline（双重去噪后训练量化流程）：

  1. **校准数据收集**：用全精度模型 `model_fp` 在 M 步逆扩散中收集校准样本 `{(x_t, t, c)^i}`，输入 BRECQ
  2. **PTQ 量化**：`model_q = BRECQ(model_fp, q_params, calibration_data)` —— 使用 Adaround 作为权重量化器，首尾层固定 8-bit，其余层量化至目标位宽
  3. **量化噪声建模**（TSQNM）：用 `model_fp` 和 `model_q` 推理收集 S×T 组量化输出-噪声对 `{(ε̂, Δε)^i}`。对每个时间步 t，假设元素间不相关且各向同性：
     - 估计联合高斯分布参数（4个对角矩阵/标量）：μ̂_ε(t), μ_Δ(t), σ²_ε̂(t), σ_Δ²(t), 以及交叉协方差 σ_ε̂Δ(t)
     - 存储为 μ[T×2], Σ[T×4]
  4. **推理时条件化噪声预测**：在采样时间步 t，用 `model_q(x_t)` 得到量化输出 ε̂_θ^(t)，通过 TSQNM 计算：
     - μ_{Δε|ε̂=ε̂_θ^(t)} = (σ_ε̂Δ/σ²_ε̂) · (ε̂_θ^(t) - μ_ε̂) + μ_Δ  （条件均值）
     - σ²_{Δε|ε̂=ε̂_θ^(t)} = σ²_Δ - σ²_ε̂Δ / σ²_ε̂  （条件方差）
  5. **双重去噪**（两种变体）：
     - **S-D²（随机）**：采样 z ~ N(0, I)，计算 Δε' = μ_{Δε|ε̂} + σ_{Δε|ε̂} · z，恢复 ε' = ε̂_θ^(t) - Δε'，代入标准 SDE 采样
     - **D-D²（确定性）**：仅减去条件均值 ε' = ε̂_θ^(t) - μ_{Δε|ε̂}，额外方差 σ²_Δ 被吸收到扩散项中：g'(t) = √(g²(t) - g⁴(t)·σ²_Δ(t)/σ²_t)
  6. **DDIM 采样更新**：x_{t-1} = √α_{t-1} · (x_t - √(1-α_t)·ε')/√α_t + √(1-α_{t-1} - |Σ_t|^{1/d})·ε' + Σ_t^{1/2}·ε_t，其中 Σ_t 被调整以吸收 D-D² 中的额外方差

  关键张量计算示例（以 W4A8 量化，时间步 t≈0.5T，batch element 为例）：
  - 量化输出 ε̂_θ^(t) ∈ R^{4×64×64}（LDM-4 latent 空间）
  - 条件均值 μ_{Δε|ε̂} ∈ R^{4×64×64}，逐元素 / 逐通道计算均值和方差的 element-wise 校正
  - 条件方差 σ²_{Δε|ε̂}：假设各向同性简化为标量，用于 S-D² 中的噪声采样或 D-D² 中的扩散项调整

## Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

- 属于算法pipeline的实现是什么？实验比较什么？
  Squat（EdgeQAT）提出了面向边缘设备SLM的粗粒度QAT框架，核心包含三个组件：(1) **Entropy-Guided & Distribution-Aligned Distillation**：熵损失 L_E 最大化量化后 query/key 的信息熵（等价于最小化量化误差），分布损失 L_D 通过余弦相似度对齐量化注意力图与FP16注意力图，解决量化自注意力模块的表征退化；(2) **Token Adaptive Quantization**：基于每个token对初始token的平均注意力分数评估重要性，TopK选择 ρ 比例的重要token分配8-bit、其余分配4-bit，通过TCLM模块实现动态分组+拼接+分别量化；(3) **Adaptive Training Pipeline**：FP16教师模型蒸馏量化学生模型，总损失 L_total = L_distill + r_E·L_E + r_D·L_D。
  实验比较了NIPQ、PACT、LLM-QAT三种QAT baseline，覆盖W8A8、W4A8、W4A4三种位宽配置。在BLiMP零样本评估和(Super)GLUE微调评估上验证精度，在OnePlus 11和Raspberry Pi 5上验证硬件加速。

- 硬件平台是什么，配置是什么。
  - 训练：论文未明确说明训练GPU型号（基于PyTorch框架）。
  - 推理延迟测试：OnePlus 11（Snapdragon 8 Gen 2，全部核心多线程），Raspberry Pi 5（BCM2712四核Arm Cortex A76，四核全用）。延迟基于1000次迭代取平均，输入序列长度128。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-58M（BabyLLaMA架构）、GPT2-97M。
  数据集：预训练数据来自BabyLLaMA工作[46]并经regex清洗，BPE tokenizer（vocab=16000）。
  Benchmark：BLiMP（零样本评估，含BLiMP Main 12个子集+BLiMP Supplement 5个子集）、(Super)GLUE（微调评估，11个子任务：CoLA, SST-2, MRPC, QQP, MNLI, MNLIm, QNLI, RTE, BoolQ, MultiRC, WSC）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/shawnricecake/squant

  **算法pipeline（QAT训练流程）：**

  1. **对称逐层量化前向（Preliminary）**：
     - 权重量化：Q(w) = clip(round(w/α_w), -2^{b_w-1}, 2^{b_w-1}-1)，ŵ = Q(w)·α_w
     - 激活量化：Q(x) = clip(round(x/α_x), -2^{b_x-1}, 2^{b_x-1}-1)，x̂ = Q(x)·α_x
     - 线性层：F_Linear(x, w) = α_x·α_w·[Q(x) × Q(w)]
     - 反向传播使用STE近似梯度

  2. **熵引导优化（Entropy Loss）**：
     - 假设query q ~ N(μ_q, σ_q²)，key k ~ N(μ_k, σ_k²)
     - 熵 H(q) = ½log(2πeσ_q²)，H(k) = ½log(2πeσ_k²)
     - 损失：L_E = -log(Σ_{l=1}^L Σ_{h=1}^H log(1 + σ_q²·σ_k²))
     - 最大化熵等价于最小化量化误差（MOE ≈ MAE for Gaussian）

  3. **分布对齐优化（Distribution Loss）**：
     - L_D = log(Σ_{l=1}^L Σ_{h=1}^H (attn_q · attn_f) / (||attn_q||₂ · ||attn_f||₂))
     - 对齐量化注意力图与FP16注意力图的余弦相似度

  4. **Token自适应量化（Token Adaptive Quantization）**：
     ```
     输入: activations x ∈ R^{N×d}, attention map attn, important ratio ρ
     1. scores = attn[:, 0]  // 每个token对初始token的平均注意力
     2. threshold = TopK(scores, Int(ρ*N))  // Heapsort取第k大
     3. for i = 0 to N-1:
     4.     if scores[i] >= threshold:
     5.         x_8bit.append(x[i])  // 重要token → 8-bit
     6.     else:
     7.         x_4bit.append(x[i])  // 非重要token → 4-bit
     8.   x_q = concat(layer_wise_quant8(x_8bit), layer_wise_quant4(x_4bit))
     9.   output = MKMP_multiplier(x_q, w_q)  // 混合精度MAC
     ```

  5. **蒸馏训练总损失**：
     - L_distill = (1-γ)·L_CE + γ·τ²·L_KL
     - L_total = L_distill + 0.5·L_E + 1.0·L_D

  **关键结果**：LLaMA-58M W4A8 BLiMP avg=69.4%（FP16=69.7%，仅↓0.3%），W4A4 avg=67.8%。GPT2-97M W4A4 BLiMP avg=69.2%（FP16=69.9%）。OnePlus 11上GPT2-97M INT4加速2.26×，Raspberry Pi 5上2.37×。

## GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 GPTAQ，一种改进 GPTQ 的 finetuning-free 量化方法。核心创新是**非对称校准（Asymmetric Calibration）**：GPTQ 每层独立最小化 `||ŵX - wX||²`（对称校准），GPTAQ 改为最小化 `||ŵX - wX̃||²`，其中 X̃ 是前一层全精度模型的输出激活，X 是前一层量化后的实际激活。这显式补偿了前层量化误差累积导致的输入激活偏差。方法基于 Optimal Brain Compression (OBC) 框架推导出闭式解，Δw 包含两项：(1) 与 GPTQ 相同的量化误差补偿项 `−((ŵ_q − w_q)/H_{qq}^{-1})·H_{q,:}^{-1}`；(2) 新增的残留误差补偿项 `rX^T H_{-q}^{-1}`。为实现高效计算，提出四个优化步骤：任意顺序处理列（支持行并行）、残差分解（将 R 分解为逐神经元分量避免重复计算）、Cholesky 重构化（数值稳定 + 矩阵融合）、Lazy-Batch 更新（提高 GPU 利用率）。仅需比 GPTQ 多约 20 行代码。

  实验对比：
  - Vision Transformer（W4A4, W2A4）：DeiT-S/B 上与 PTQ4ViT、APQ-ViT、PD-Quant、RepQ-ViT、GPTQ 对比
  - Language Transformer（W4A4, W2A4）：LLaMA2-7B/13B/70B、LLaMA3-8B/70B 上与 OmniQuant、QLLM、DuQuant、QuaRot+GPTQ、SpinQuant+GPTQ 对比
  - Weight-Only Quantization（3-bit per-group）：LLaMA2-7B/13B、LLaMA3-8B-Instruct 上与 AWQ、GPTQ 对比
  - 超大模型：EVA-02（90% ImageNet top-1）和 LLaMA3.1-405B 的 W4A4 量化
  - 评估指标：ImageNet Top-1 准确率（Vision）、WikiText2/C4 Perplexity（Language）、PiQA/ARC-E/ARC-C/HellaSwag/Winogrande/BoolQ 零样本准确率
  - 消融实验：ΔW 两项的各自贡献（仅第一项=GPTQ、仅第二项、两项联合=GPTAQ）、激活量化顺序（A→W vs W→A）
  - 效率分析：P 矩阵计算的并行 vs 非并行实现延迟对比、GPTQ vs GPTAQ 逐层延迟对比

- **硬件平台是什么，配置是什么。**
  单张 NVIDIA A100 GPU（所有量化校准实验）。PyTorch 2.4.1-cu12.4。HuggingFace Transformers 框架。GPU Hours 报告：LLaMA3-8B QuaRot+GPTQ 需 0.2h，QuaRot+GPTAQ 需 0.3h；LLaMA3-70B QuaRot+GPTQ 需 1.8h，QuaRot+GPTAQ 需 2.7h。SpinQuant 需额外 4-28 GPU-hours（8×A100）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：DeiT-S、DeiT-B（Vision Transformer）；LLaMA2-7B/13B/70B、LLaMA3-8B/70B、LLaMA3.1-405B（Language Transformer）；EVA-02（Huge Vision Transformer）
  - 校准数据：ImageNet 训练集 128 samples（Vision）；WikiText2 训练集 128 sequences × 2048 tokens（Language）；C4 128 sequences（Weight-Only）
  - Perplexity：WikiText2、C4
  - 零样本推理：PiQA、ARC-Easy、ARC-Challenge、HellaSwag、Winogrande、BoolQ
  - ImageNet Top-1 准确率（Vision）

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  开源代码：https://github.com/Intelligent-Computing-Lab-Yale/GPTAQ

  **算法pipeline核心流程（Algorithm 1：GPTAQ 单层量化）**：

  输入：FP 权重 W (m×n)，校准输入 X，FP 输入 X̃（来自全精度前向），Block size B
  ```
  1. H ← X X^T                                    # 计算 Hessian
  2. ΔX X^T ← (X̃ - X) X^T                        # 输入偏差 × 激活转置
  3. L ← Inverse_Cholesky(H + λ₁I)               # Cholesky 分解
  4. P ← ((ΔX X^T L) ⊙ M_U) L^T                  # Theorem 4.2：并行计算 P 矩阵
     其中 M_U 是严格上三角掩码矩阵（对角线以上为1）
  5. Q ← 0_{m×n}, E ← 0_{m×B}                    # 初始化量化权重和误差缓冲
  6. for i = 0, B, 2B, ... do                     # 按 block 迭代
  7.     for j = i, i+1, ..., i+B-1 do             # block 内逐列量化
  8.         Q_{:,j} ← quant(W_{:,j})              # 量化当前列
  9.         E_{:,j-i} ← (W_{:,j} - Q_{:,j}) / L_{jj}  # GPTQ 误差项
  10.        W_{:,j:(i+B)} ← W_{:,j:(i+B)} - E_{:,j-i} L_{j,j:(i+B)}^T  # 补偿 block 内
  11.    end for
  12.    # Lazy-batch 更新 block 外列
  13.    W_{:,(i+B):} ← W_{:,(i+B):} - E·L_{i:(i+B),(i+B):}^T          # GPTQ 项
  14.                  + W_{:,i:(i+B)} P_{i:(i+B),(i+B):}              # GPTAQ 新增项
  15. end for
  ```

  **与 GPTQ 的核心差异**：
  - GPTQ 的补偿项：δw = −(w_q − ŵ_q)/L_{qq} · L_{q,q:}（仅量化误差补偿）
  - GPTAQ 的补偿项：在第 14 行新增 `W_{:,i:(i+B)} P_{i:(i+B),(i+B):}` 项，补偿前层量化导致的输入激活偏差累积
  - P 矩阵通过 Theorem 4.2 的并行公式一次性计算：`P = ((ΔX X^T L) ⊙ M_U) L^T`，利用 CUDA 优化，<1ms 完成（vs 非并行实现 >10⁴× slower）

  **残差分解原理（Step 2）**：
  - 直接法需要每次迭代重新计算 R = W X̃ − W X，复杂度 O(mnk)，k >> n 时极慢
  - 分解法：R = Σ_{q=1}^n W_{:,q} ΔX_{q,:}，预计算一次后，第 q 次迭代仅关注第 q 个神经元的残差分量 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}`，复杂度降至 O(mn)

  **整模型量化流程（Algorithm 2）**：
  ```
  for i = 1 to b-th block:
      Move block[i] to GPU                     # 每次仅一个 block 在 GPU
      X̃ ← block[i](X̃)                          # 全精度前向，缓存各层 FP 输入
      if AQ enabled: enable activation quantization
      for each layer in block[i]:
          Compute H and ΔX X^T for layer
          Run GPTAQ Algorithm 1 for layer
          Quantize layer weights
      X ← block[i](X)                          # 量化 block 输出
      Move block[i] to CPU
  ```

  **关键结果**：
  - W4A4 DeiT-S：72.8%（GPTQ 71.9%，+0.9%）；W2A4 DeiT-S：46.8%（GPTQ 38.4%，+8.4%）
  - W4A4 LLaMA3-70B WikiText2 PPL：6.93（GPTQ 9.44，↓2.51）；QuaRot+GPTAQ avg accuracy 69.1%（QuaRot+GPTQ 62.4%，+6.7%）
  - W4A4 LLaMA3.1-405B WikiText2 PPL：3.48（GPTQ 5.82，↓2.34）
  - W4A4 EVA-02 ImageNet Top-1：88.30%（GPTQ 86.48%，+1.82%；FP16 90.05%）
  - 额外延迟：GPTAQ 比 GPTQ 多 30-40%（大维度时），小维度时 <10%


## GPTVQ: The Blessing of Dimensionality for LLM Quantization

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出 GPTVQ，一种将 GPTQ 框架扩展到非均匀向量量化（VQ）的后训练量化方法。核心实现：(1) **GPTVQ 算法**：将权重矩阵按 d 维列组逐块量化（2D VQ 为默认配置），使用 Hessian 加权误差补偿（沿用 GPTQ 的 Cholesky 分解 + lazy batch update），量化误差沿 d 维坐标累积后一次性更新剩余权重；(2) **EM 初始化**：用加权马氏距离或 k-Means++ 初始化 codebook，E-step 用 Hessian 加权的距离函数（公式 5）找最优质心，M-step 用 Moore-Penrose 伪逆闭式解更新质心；(3) **Codebook update**：GPTVQ 结束后通过梯度下降进一步最小化层输出 MSE（公式 7）更新 codebook 值；(4) **Blockwise data normalization**：在 codebook 初始化前对权重子行按 log-scale 做 per-block max 归一化（4-bit 缩放因子），改善 VQ 误差；(5) **4-bit codebook 量化**：将 codebook 进一步量化到 INT4，通过预缩放 + EM 初始化 + GPTQ 内逐组缩放实现极小精度损失。

  实验对比：
  - Weight-only 量化 baseline：RTN、GPTQ、AWQ、OmniQuant（均匀量化，group size 128/64）
  - VQ baseline：AQLM、QuIP#（其他向量量化方法）
  - 位宽配置：2.125/2.25/3.125/4.125 bpv，1D/2D/4D VQ
  - 模型：Llama-1 (7B/13B/30B/65B)、Llama-2 (7B/13B/70B)、Llama-3 (8B/70B)、Mistral-7B-v0.1、Mixtral-MoE-8x7B-v0.1、BLOOM-560M（消融）
  - 评估指标：WikiText2 perplexity、PIQA/ARC-easy/ARC-challenge/BoolQ/HellaSwag/Winogrande 零样本准确率平均
  - 消融：EM 初始化方法（Mahalanobis vs k-Means++）、EM 迭代次数（10-100）、Codebook update 有无、scaling block size、codebook SVD vs INT8 量化 vs 无压缩
  - 与 LoRA 结合：GPTVQ + LoRA adapter（frozen/trained），对比 QLoRA/LoftQ，评估 WikiText2 PPL + GSM8k 准确率

- **硬件平台是什么，配置是什么。**
  量化校准：单张 NVIDIA H100 GPU。Llama-v2-7B 量化时间约 30 分钟 - 1 小时，Llama-v2-70B 约 3-11 小时（vs AQLM 的 35 小时 on H100）。移动端推理：Snapdragon X Elite 平台，Windows + Clang 18.1 with Polly。校准数据：WikiText2 训练集 128 sequences × 2048 tokens。与 AQLM 对比时使用 SlimPajama 校准集（4096 samples × 2048 tokens）。

- **模型是什么。数据集和bench分别是什么。**
  - 模型：Llama-1 (7B/13B/30B/65B)、Llama-2 (7B/13B/70B)、Llama-3 (8B/70B)、Mistral-7B-v0.1、Mixtral-MoE-8x7B-v0.1、BLOOM-560M
  - 校准数据：WikiText2 训练集（128 sequences, 2048 tokens，默认）；SlimPajama（4096 samples × 2048 tokens，AQLM 对比）
  - Perplexity：WikiText2（validation set, sequence length 2048）
  - 零样本任务：PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、Winogrande（LLM-evaluation-harness）
  - GSM8k（LoRA adapter 实验）
  - 对 Llama3 零样本平均省略 BoolQ（对齐 Huang et al. 2024 协议）

- **开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。**
  论文声明 GPU kernel 代码 "will be made available in the future"（尚未开源）。算法伪代码见论文 Algorithm 1 (GPTVQ) 和 Algorithm 2 (QuantGroup)。

  **GPTVQ 算法核心流程（Algorithm 1, 论文第 4 页）**：

  输入：权重矩阵 W ∈ R^{r×c}，逆 Hessian H^{-1}，block size B，VQ dimensionality d，质心数 k，group size l（假设每 column 一个 group）

  ```
  1. N_b ← c/B                     # 总 block 数
  2. m ← l/r                       # group 的 column 数
  3. Q ← 0_{r,c}                   # 量化后权重
  4. E ← 0_{r,c}                   # 误差矩阵
  5. N_g ← rc/l                    # 总 group/codebook 数
  6. C_g ← 0_{d,k}, g=1,...,N_g    # codebook 初始化
  7. H^{-1} ← Cholesky(H^{-1})^T   # Cholesky 分解得到上三角矩阵
  8. for i = 0, B, 2B, ..., N_b·B do
  9.     if i % m == 0 then        # 新 group 开始时
  10.        g ← i/m               # 当前 group 索引
  11.        C_g ← init_codebook(W_{:, i:i+m-1})  # EM 初始化 codebook
  12.    end if
  13.    Q_{:, i:i+m-1} ← QUANTGROUP(W_{:, i:i+m-1})  # 量化当前 group
  14.    W_{:, i+B:} ← W_{:, i+B:} − E · [H^{-1}]_{i:i+B, i+B:}  # lazy update
  15. end for
  ```

  **QuantGroup 子算法（Algorithm 2, 论文第 4 页）**：

  ```
  1. function QUANTGROUP(W)  # W ∈ R^{r×m}
  2.   for j = 0, d, 2d, ..., l do
  3.     P = j, ..., j+d-1   # 当前 d 维列的索引
  4.     Q_{:,P} ← VQ_quant(W_{:,P}, C_g)  # 用 C_g 中最优质心量化
  5.     E_{:,P} ← (W_{:,P} - Q_{:,P})[H^{-1}]_P  # 计算误差
  6.     U ← Σ_{p=0}^{d-1} E_{:,j+p} [H^{-1}]_{p, j+d-1:B}  # 累积误差
  7.     W_{:, j+d-1:B} ← W_{:, j+d-1:B} - U  # 补偿剩余权重
  8.   end for
  9. end function
  ```

  **Codebook 初始化 EM 算法**：

  目标：min_{I, c} Σ_{m=0}^{k} Σ_{i∈I_m} (x^{(i)} - c^{(m)})^T D^{(i)} (x^{(i)} - c^{(m)})
  - D = diag(1/[H^{-1}]_{11}, ..., 1/[H^{-1}]_{cc})（Hessian 加权）
  - E-step：固定质心 c^{(m)}，为每个 d 维向量 x^{(i)} 分配最优质心（公式 5）
  - M-step：固定分配，闭式解 c^{(m)} = (Σ D^{(i)})^{+} Σ D^{(i)} x^{(i)}（Moore-Penrose 伪逆）

  **Codebook Update（附录 A）**：
  初始化完成后，固定 codebook 索引，用梯度下降（PyTorch）最小化 ||WX - Q(C)X||²_F，其中 Q(C) 是 codebook C 的查找操作。每步更新 C 后重建 Q 并继续。

  **推理时的解压缩流程（移动端 CPU）**：
  1. DRAM → SoC cache：加载 VQ 编码 (indices + LUT + scale)
  2. TBL 指令解码 6-bit index → 8-bit signed int（每个维度 1 条指令，2D VQ 需 2 条）
  3. 逐元素 scale × decoded_int → 反量化到 native data type
  4. 矩阵-向量乘法（SIMD 加速）

  **Bits per value 计算**：bpv = log₂(k)/d + kdb_c/l，k=质心数，d=VQ 维度，b_c=codebook bit-width，l=共享同一 codebook 的权重数

  **关键结果**：
  - Llama-v2-70B W2@g128：GPTVQ 2D PPL 4.72 vs OmniQuant 6.55（↓1.83）
  - GPTVQ 2D 4D at 3.125 bpv：WikiText2 PPL 5.83/7.00（Llama-v3-8B）vs FP16 6.14
  - 压缩时间：Llama-v2-7B 2.5h vs AQLM no BFT 18.3h（7.3× 加速）
  - 与 LoRA 结合：GPTVQ 4D 2.125 bpv + LoRA-trained Llama-v2-7B WikiText2 PPL 5.83 vs LoftQ NF2 20.9 (GSM8k)


## Hymba: A Hybrid-head Architecture for Small Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - **Hymba**：一种融合 attention 和 SSM（Mamba）的 hybrid-head 并行架构。核心创新包括四部分：(1) **Hybrid-Head Module**：在同一层内并行放置 attention heads 和 SSM heads，两者处理相同输入，输出经 learnable β₁, β₂ 归一化重缩放后取均值融合；(2) **KV Cache 优化**：仅 3 层使用 global attention（首/中/末层），其余层使用 sliding window attention (SWA)；相邻层间共享 KV cache（cross-layer KV sharing）；(3) **Meta Tokens**：128 个可学习 token 前置到输入序列，作为 learned cache initialization 引导 attention 关注有意义 token，减轻 attention sink 现象；(4) **Scaling**：模型从 300M 消融扩展到 1.5B（32 layers, hidden=1600, 25 attn heads, 5 query groups），训练 1.5T tokens。
  - 实验比较包括：(a) 与 SOTA sub-2B 小模型对比（SmolLM2-1.7B、Qwen2.5-1.5B、Llama-3.2-1B/3B、Phi-1.5、h2o-danube2、OpenELM 等）；(b) Apple-to-apple 架构对比：同参数/同数据/同训练 recipe 下对比 Llama3、Mamba2、Mamba2 w/ FFN、Samba（sequential Mamba-Attn）在 300M 和 1B 两个规模的各项任务表现；(c) Needle-in-a-Haystack 长上下文检索对比；(d) Instruction-tuned 模型对比（Llama-3.2-1B-Instruct、Qwen2.5-1.5B-Instruct、SmolLM-1.7B-Instruct）；(e) 消融实验：Attention/SSM head 比例、parallel vs sequential fusion、local/global attention ratio、KV cache sharing、meta tokens、fusion strategy（mean vs concat）。

- 硬件平台是什么，配置是什么。
  - 训练：128× NVIDIA A100 GPU（pretrain 1.5B 模型 1.5T tokens）
  - 推理吞吐测试：NVIDIA A100 GPU，sequence length=8K，batch size=128，PyTorch（OOM 时减半 batch size 直到不 OOM）
  - 后训练（SFT + DPO）：论文未明确说明 GPU 数量/型号

- 模型是什么。数据集和bench分别是什么。
  - 模型：Hymba-125M (24 blocks, hidden=512, 8 attn heads)、Hymba-350M (32 blocks, hidden=768, 12 attn heads)、Hymba-1.5B (32 blocks, hidden=1600, 25 attn heads, 5 query groups)
  - 训练数据：DCLM-Baseline-1.0 + SmolLM-Corpus + NVIDIA 内部高质量数据集（Hymba-1.5B: 1.5T tokens total，其中 public data only 版本为 DCLM 1T + SmolLM 500B）
  - Ablation (300M)：100B tokens 训练，序列长度 1K/2K；FineWeb 数据集（A.3 apple-to-apple at 300M）
  - Apple-to-apple (1B)：100B tokens SmolLM-Corpus
  - 评估 Benchmark：
    - Commonsense Reasoning: MMLU (5-shot), ARC-Easy/C (0-shot), PIQA (0-shot), HellaSwag (0-shot), Winogrande (0-shot), OBQA (0-shot), TruthfulQA (0-shot), SIQA (0-shot), LAMBADA (0-shot)
    - Recall-Intensive: SQuAD-C (1-shot), SWDE
    - Language Modeling: WikiText-2 perplexity, LMB perplexity
    - Instruction-tuned: GSM8K (5-shot), GPQA (0-shot), IFEval, BFCLv2 (Berkeley Function-Calling Leaderboard), RoleBench
  - 评估框架：lm-evaluation-harness（主评估），HuggingFace/LightEval（小模型评估）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源模型：HuggingFace 上发布 Hymba-1.5B-Base 和 Hymba-1.5B-Instruct
  - 基于 Mamba 和 Llama 架构实现，使用 PyTorch 框架

  **Hymba Hybrid-Head 算法pipeline（单层前向）**：

  ```
  # 输入: X ∈ R^{N×d_model}（原始 token 序列，N=tokens, d_model=hidden_dim）
  # 先 prepend meta tokens R ∈ R^{m×d_model}, m=128
  X̃ = concat([R, X], dim=0)  # X̃ ∈ R^{(N+m)×d_model}

  # Step 1: 输入投影
  # W_in_proj = [W^Q, W^K, W^V, W^{SSM}, W^G]
  Q = W^Q @ X̃  # attention queries
  K = W^K @ X̃  # attention keys
  V = W^V @ X̃  # attention values
  X_ssm = W^{SSM} @ X̃  # SSM input features
  G = W^G @ X̃  # SSM gates

  # Step 2a: Attention head 输出（sliding window 或 global）
  # 滑动窗口时 causal mask 限制为 window_size 内
  M_attn = softmax(Q @ K^T / √d_head) @ V    # Y_attn = M_attn @ X̃
  # 若使用 GQA: K, V 的 head 数少于 Q 的 head 数

  # Step 2b: SSM head 输出（Mamba-style, 逐 token recurrent）
  # B = W_B @ X_ssm, C = W_C @ X_ssm
  # Δ = Softplus(W_Δ @ X_ssm)
  for i in 1..N+m:
      # Discretize continuous SSM
      Ā_i = exp(Δ_i ⊗ A)  # A ∈ R^{d_state×d_state}, Δ_i ∈ R^{d_inner}
      B̄_i = Δ_i ⊗ B_i
      # Recurrent update
      h_i = Ā_i ⊙ h_{i-1} + B̄_i ⊙ X_ssm[i]   # h ∈ R^{d_inner×d_state}
      y_i = C_i @ h_i
  Y_ssm = G ⊙ Y  # element-wise gate

  # Step 3: 融合（归一化 + 重缩放 + 平均）
  Y_attn_norm = norm(Y_attn)
  Y_ssm_norm = norm(Y_ssm)
  Y_fused = β₁ ⊙ Y_attn_norm + β₂ ⊙ Y_ssm_norm
  # β₁, β₂ ∈ R^{d_model} 是可学习 per-channel 缩放向量

  # Step 4: 输出投影
  Y = W_out_proj @ Y_fused

  # 注：实际实现中，每层有多个 attention heads 和 SSM heads，
  # 如 1.5B: 25 attn heads × (d_head=64), SSM heads 占据剩余维度
  # attn:mamba 参数比约 1:5.23（最终配置含 GQA 和 KV sharing 后）
  ```

  **KV Cache 优化配置**：
  - 仅第 1 层、中间层、最后 1 层使用 global full attention（共 3 层）
  - 其余 29 层使用 sliding window attention（window_size=1024）
  - 每 2 个连续层共享同一 KV cache（cross-layer KV sharing）
  - 结果：8K 序列下 cache size 从 Llama 的 414.7MB 降至 39.4MB（10.5× reduction）

  **Meta Tokens 推理流程**：
  ```
  # 离线预计算（仅一次）
  K_meta = W^K @ R     # meta tokens 的 K
  V_meta = W^V @ R     # meta tokens 的 V
  X_ssm_meta = W^{SSM} @ R  # meta tokens 的 SSM 输入
  # 存储这些值作为 "learned cache initialization"

  # 在线推理
  X̃ = concat([K_meta_cache, K_input], dim=0)  # 在 K cache 维度
  # 后续计算同上述 pipeline，meta tokens 部分的 K/V/SSM 状态从预计算值加载
  ```

  **训练配置**：
  - LR scheduler: Warmup-Stable-Decay (WSD)，warmup=1% steps，stable peak lr=3e-3，decay to 1e-5 over 20% steps
  - Sequence length: 2K（最后 100B tokens 增至 8K，同步调整 ROPE base）
  - Batch size: 2M tokens
  - 后训练：FFT (lr=5e-5) → DPO (lr=3e-6)，LMFlow toolkit，packed samples (block_size=8192 for SFT, 2048 for DPO)

  **关键结果**：
  - Hymba-1.5B avg accuracy 61.06% vs SmolLM2-1.7B 60.04%（+1.02%），cache 79MB vs 1573MB（19.91× reduction），throughput 664 vs 238 tok/s（2.79×）
  - 超越 Llama-3.2-3B：avg +1.32%，cache 11.67× smaller，3.49× faster
  - Apple-to-apple 1B avg 54.57% vs Llama3 52.82%（+1.75%），vs Samba 52.83%（+1.74%）

## DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  DartQuant 提出基于分布的旋转矩阵校准方法用于 LLM Post-Training Quantization（PTQ），由三部分组成：(1) **Rotational Distribution Calibration**：将旋转矩阵优化重新定义为将激活变换到最适合量化的分布；(2) **Whip loss**：`Whip = Σ exp(-|x_i|)`，驱动旋转后激活趋向均匀分布，减少量化误差；(3) **QR-Orth**：通过 QR 分解保证正交性（`R = QR(Z)`，优化隐参数 Z 替代直接在 Grassmannian 流形上优化 R），避免 Cayley SGD 等复杂黎曼优化器。
  实验比较 RTN、SmoothQuant、GPTQ、OmniQuant、QuaRot、SpinQuant、OSTQuant 在 4-8-16、4-4-16、4-4-4 比特设置下的 WikiText2/C4/PTB PPL 和 9 项零样本任务准确率，以及旋转矩阵优化的 GPU 时间和内存开销。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU 服务器（主实验和表 3 时间/内存对比）。单卡 NVIDIA RTX 3090（展示 70B 校准可行性，约 3 小时完成，23.47 GiB 内存）。

- 模型是什么。数据集和bench分别是什么。
  模型: Llama-2 (7B/13B/70B)、Llama-3 (8B/70B)、Mixtral-8×7B (MoE)、DeepSeek-MoE-16B。
  数据集和 benchmark: PPL 用 WikiText2、C4、PTB；零样本用 LAMBADA、HellaSwag、PIQA、WinoGrande、OpenBookQA、SIQA、MMLU、ARC-Easy、ARC-Challenge。
  校准集: 128 samples from WikiText2，sequence length 2048。GPTQ weight reconstruction 使用相同校准集，per-token asymmetric 激活量化。Token sampling ratio 为 10%。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源: https://github.com/CAS-CLab/DartQuant.git

  **DartQuant 核心算法（Algorithm 1）：**
  ```
  输入: LLM model, 校准序列 S, 隐参数 Z_0 ∈ R^{n×n}, 最大迭代 T, 学习率 η
  输出: 旋转矩阵 R ∈ R^{n×n}

  1. X ← LLM(S)                     // 前向传播收集所有指定层的激活
  2. X ← token_sampling(X)          // 随机采样 10% token 以减少计算量
  3. Z ← Z_0                        // 初始化隐参数为随机 Hadamard 矩阵
  4. for k = 0 to T do
  5.     R ← qr_decomposition(Z)    // QR-Orth: 通过 QR 分解获得正交旋转矩阵
  6.     O ← X @ R                  // 激活旋转: O = XR
  7.     L ← Whip(O)                // Whip loss = Σ_{i=1}^{C_in} exp(-|o_i|)
  8.     Z ← Z - η ∂L/∂Z            // 标准 SGD/Adam 更新隐参数（无需约束优化器）
  9. end for
  ```

  **旋转矩阵在 Transformer 中的融合（Computational Invariance）：**
  - R_1（可学习，DartQuant 优化）：右乘 W_q, W_k, W_v, W_up, W_gate；R_1^T 左乘 W_out, W_down, W_embedding；R_1 右乘 W_lm_head
  - R_2（可学习，DartQuant 优化）：插入 W_v 和 W_o 之间，R_2 融入 W_v，R_2^T 融入 W_o
  - R_3（在线 Hadamard）：在 attention score 计算中在线执行，抵消 KV cache 量化损失（因 RoPE 存在无法融合入权重）
  - R_4（在线 Hadamard）：在 FFN down-projection 前在线执行（因 gating 机制无法融入 W_up/W_gate）

  **Whip Loss 机制：** 激活向量 x ∈ R^{C_in}，Whip = Σ exp(-|x_i|)。该函数在零附近有较大梯度，将接近零的小值推开；在 norm-invariance 约束（||Rx|| = ||x||）下，小值被迫增大 → outliers 被迫减小以保持 L2 范数不变 → 整体分布趋向均匀。灵感来自 Laplace→Uniform 的 CDF 变换：U_X(x) = τ[exp(x/b)-1] for x≤0。

  **QR-Orth vs Cayley SGD 计算复杂度：**
  Cayley SGD 额外计算量约 6n³（矩阵乘法+投影），QR-Orth 仅需 QR 分解约 4/3 n³。100 步 SGD 耗时：QR-Orth 5.7h vs Cayley 8.2h（1.44× 加速）；QR-Orth SGD 仅 6 步即达到 Cayley SGD 100 步同等效果（41× effective 加速）。

  **推理流程：** 所有权重 INT4 存储，激活在矩阵乘法前量化为 INT4 → TensorCore INT4×INT4 矩阵乘产生 INT32 → 立即转换为 FP16（含 scale）。R1, R2 预融合无推理开销，R3, R4 使用快速 Hadamard kernel 在线计算。

  **关键结果：** 70B 模型校准时间从 SpinQuant 42.9 GPU-hours → DartQuant 0.91 GPU-hours（47× 加速），内存从 238.89 GiB → 23.47 GiB（10× 节省）。首次在单卡 RTX 3090 上完成 70B 旋转校准（~3h）。Llama-3-70B w4a4kv16 零样本 avg loss 仅 3.31%（vs SpinQuant 6.64%, OSTQuant 4.76%）。

## LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - **LoftQ**：一种量化 + LoRA 联合初始化的框架，通过交替优化（量化与 SVD 低秩近似）来近似原始高精度预训练权重，解决 QLoRA 中量化误差导致 LoRA 初始化偏移的问题。核心流程：给定预训练权重 W，交替执行：(1) 量化当前残差 Q_t = q_N(W - A_{t-1}B_{t-1}^T)，(2) 对量化残差做 SVD 获得 top-r 低秩近似 A_t B_t^T。最终输出量化 backbone Q_T 和 LoRA 适配器初始化 A_T, B_T，满足 W ≈ Q_T + A_T B_T^T。LoftQ 与量化函数无关，支持 NormalFloat (NF2/NF4) 和 Uniform quantization。
  - 实验比较 QLoRA（量化 backbone + 零初始化 LoRA）、Full-precision LoRA（16-bit backbone + LoRA）、Full fine-tuning（全参数微调）作为基线。覆盖 2-bit、4-bit、混合精度（前几层 4-bit + 剩余 2-bit）多种精度级别。评估 encoder-only（DeBERTaV3-base）、encoder-decoder（BART-large）、decoder-only（LLAMA-2-7b/13b）三类模型。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU（所有训练和推理实验）
  - 量化时间测试：Intel Xeon CPU E5-2650 v4 @ 2.20GHz

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeBERTaV3-base（encoder-only, ~183M）、BART-large（encoder-decoder, ~406M）、LLAMA-2-7b、LLAMA-2-13b
  - 数据集与 Benchmark：
    - NLU: GLUE benchmark（MNLI, QNLI, RTE, SST-2, MRPC, CoLA, QQP, STS-B），SQuADv1.1，ANLI
    - 摘要生成：XSum, CNN/DailyMail（评估 ROUGE-1/2/L）
    - NLG: WikiText-2（评估 Perplexity），GSM8K（评估数学推理 accuracy）
  - LoRA rank：DeBERTaV3 用 rank 16/32，BART 用 rank 8/16，LLAMA-2 用 rank 64

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/yxli2123/LoftQ
  - 模型开源：https://huggingface.co/LoftQ
  - 基于 HuggingFace Transformers + PyTorch 实现

  **LoftQ 核心算法（对单个权重矩阵 W ∈ R^{d1×d2}）：**

  ```
  # 输入: 预训练权重 W, 目标 rank r, N-bit 量化函数 q_N, 交替步数 T
  A_0 = 0  # ∈ R^{d1×r}
  B_0 = 0  # ∈ R^{d2×r}

  for t = 1 to T:
      # Step 1: 量化残差（减去上一步的低秩近似）
      residual = W - A_{t-1} @ B_{t-1}^T   # ∈ R^{d1×d2}
      Q_t = q_N(residual)                    # N-bit 量化, Q_t ∈ R_N^{d1×d2}

      # Step 2: SVD 分解量化残差，取 top-r 分量
      R_t = W - Q_t                         # 量化误差矩阵
      U, Σ, V^T = SVD(R_t)                  # Σ = diag(σ_1, σ_2, ..., σ_d)
      # 取 top-r:
      A_t[:, i] = sqrt(σ_i) * U[:, i]       # i = 1..r
      B_t[:, i] = sqrt(σ_i) * V[:, i]       # i = 1..r

  输出: Q_T (量化 backbone), A_T, B_T (LoRA 适配器初始化)
  ```

  **关键特例 T=1**：Q_1 等于 QLoRA 的量化权重（因 A_0B_0^T=0），A_1B_1^T 是量化残差 W-Q_1 的 top-r SVD。仅 T=1 即可显著减轻量化差异，更多迭代进一步缩小初始化差距。

  **LoRA Fine-tuning 使用方式**：
  - 存储：Q_T 编码为整数矩阵 M（通过公式 X_INT = round((2^N-1) F(X_HP))）和查找表 T
  - 初始化：backbone 用整数矩阵 M（freeze），LoRA 适配器用 A_T, B_T（可训练）
  - 前向：Y = X · dequant(T, M) + X · A_T B_T^T
  - 推理时 adapter 可 merge 回 backbone：W_final = dequant(M) + A_T B_T^T

  **计算成本**：LoftQ 逐权重矩阵独立执行，可并行化。例如 LLAMA-2-13b 单矩阵 (5120×5120, T=5, NF4) 耗时 43s（CPU），总量化时间可接受。

  **关键结果**：
  - DeBERTaV3-base 2-bit Uniform：MNLI-m 88.0%（QLoRA 79.9%, +8.1%），CoLA 60.5（QLoRA N.A.）
  - BART-large 4-bit NF4 rank=8 XSum Rouge-1 44.08（QLoRA 42.91, +1.17）
  - LLAMA-2-7b 2-bit WikiText-2 PPL 7.85（QLoRA N.A.，不收敛）
  - LLAMA-2-13b 2/4-bit 混合精度 GSM8K 38.1%（纯 2-bit QLoRA N.A.）
  - 4-bit 场景接近 Full fine-tuning；T 不敏感（T=1~10 均可），T=1 已有显著增益

## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Mamba 提出一种选择性状态空间模型（S6），核心是将 SSM 的连续参数（Δ, B, C）由静态改为输入依赖（input-dependent），使模型能沿序列维度"选择性地"传播或遗忘信息。具体包括：i) 选择机制（Selection Mechanism）：将 Δ = τ_Δ(Parameter + s_Δ(x)), B = s_B(x), C = s_C(x) 参数化为输入 x 的函数，其中 s_B(x) = Linear_N(x), s_C(x) = Linear_N(x), s_Δ(x) = Broadcast_D(Linear_1(x)), τ_Δ = softplus；ii) Mamba 架构：将 H3 的 SSM 块与传统 MLP 块合并为同质化单一模块（图3），使用 gate 分支 + SiLU 激活 + 卷积 + 选择性 SSM 的主分支，无 attention 甚至无传统 MLP 块，扩展因子 E=2，两个 Mamba 块匹配一个 Transformer（MHA+MLP）的参数数（≈12D²）；iii) 硬件感知并行扫描算法（见 kernel调度 层）。
  - 实验比较：在语言、DNA、音频三个模态上与以下 baseline 对比：
    - 语言建模（Pile, GPT2/NeoX tokenizer）：vs Transformer (GPT3)、Transformer++（LLaMa 风格，RoPE + SwiGLU + RMSNorm + 高 LR）、Hyena、H3++、RWKV、RetNet；下游零样本评估 vs Pythia、OPT、GPT-Neo、RWKV
    - DNA 建模（HG38 human genome）：vs Transformer++、HyenaDNA
    - 音频建模（YouTubeMix piano, SC09 speech）：vs SaShiMi (S4+MLP UNet)、WaveNet、SampleRNN、WaveGAN、DiffWave
    - 合成任务：Selective Copying（序列长度 4096）和 Induction Heads（训练长度 256，测试外推至 1M）
  - 指标：perplexity（语言/DNA）、bits per byte（音频）、下游准确率（LAMBADA/HellaSwag/PIQA/Arc-E/Arc-C/WinoGrande）、物种分类准确率、FID/IS/mIS/AM（语音生成质量）、训练速度（scan 速度 vs FlashAttention-2）、推理吞吐量、内存消耗

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB PCIe GPU
  - 训练配置：语言模型 125M–1.3B（Chinchilla 缩放律，总计 2.5B–26B tokens），大模型下游评估延伸至 2.8B 参数、300B tokens；DNA 模型 ~250K–40M 参数、上下文长度 1024–1M；音频模型 ~3.5M–24M 参数
  - 优化器：AdamW，β=(0.9, 0.95)，weight decay 0.1，gradient clip 1.0，cosine LR schedule with linear warmup
  - 混合精度：BF16

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba（125M/350M/760M/1.3B/1.4B/2.8B 用于语言；~250K–40M 用于 DNA；~3.5M–24M 用于音频）。所有模型使用 real-valued diagonal SSM（S4D-Real 初始化），状态维度 N=16，Δ 投影维度 R=64（默认为 D 的小比例）。
  - 数据集：
    - 语言：The Pile（800GB），GPT2 tokenizer（缩放律）/ GPT-NeoX tokenizer（下游评估）
    - DNA：HG38 human genome（~4.5B 碱基对训练集）
    - 音频：YouTubeMix（4 小时钢琴独奏，16000Hz，mu-law 8-bit 编码）；SC09（1 秒语音片段，16000Hz，数字"zero"~"nine"）
  - Benchmark：
    - 语言：Pile 验证集 perplexity + 零样本：LAMBADA, HellaSwag, PIQA, Arc-Easy, Arc-Challenge, WinoGrande（使用 EleutherAI lm-evaluation-harness）
    - DNA：HG38 验证集 perplexity + Great Apes 五物种分类（{human, chimpanzee, gorilla, orangutan, bonobo}，共享 99% DNA）
    - 音频：YouTubeMix bits per byte (BPB)；SC09 无条件生成 NLL, FID, IS, mIS, AM

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/state-spaces/mamba
  - 算法 pipeline（选择性 SSM 前向传播，以单通道为例）：

  ```
  输入: x ∈ R^{B×L×D}  (batch, sequence length, model dimension)
  输出: y ∈ R^{B×L×D}

  参数:
    A ∈ R^{D×N}  (diagonal state matrix, N ≈ 16)
    Δ 投影权重: W_Δ ∈ R^{D×R}, b_Δ ∈ R^D  (R=64 典型值)
    B 投影权重: W_B ∈ R^{D×N}
    C 投影权重: W_C ∈ R^{D×N}
    输入投影: W_in ∈ R^{D×2ED}, W_out ∈ R^{ED×D}  (E=2)

  前向过程 (per Mamba block):
    1. 输入投影 (gate + main branch):
       x_proj = x @ W_in^T  →  R^{B×L×2ED}
       x_gate, x_main = split(x_proj, dim=-1)  → 各 R^{B×L×ED}

    2. 1D 卷积 (short convolution on main branch):
       x_conv = SiLU(Conv1d(x_main))  →  R^{B×L×ED}

    3. 选择性 SSM (S6) — 对每个通道独立执行:
       Δ = softplus(W_Δ @ x_conv + b_Δ)  →  R^{B×L×D}
       B = W_B @ x_conv  →  R^{B×L×N}
       C = W_C @ x_conv  →  R^{B×L×N}

       离散化 (Zero-Order Hold):
       Ā = exp(Δ ∘ A)  →  R^{B×L×D×N}  (∘ 表示 broadcast element-wise)
       B̄ = Δ ∘ B       →  R^{B×L×D×N}  (一阶近似)

       并行关联扫描 (recurrent form, fused in SRAM):
       h_0 = 0
       h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_conv_t  (⊙ = element-wise)
       y_ssm_t = C_t ⊙ h_t

    4. Gate:
       y = (y_ssm * SiLU(x_gate)) @ W_out^T  →  R^{B×L×D}

    5. Residual connection + LayerNorm (如前文所述可选)
  ```

  - 核心洞察：
    - **选择性**：Δ 控制"关注当前输入 vs 保持历史状态"的平衡（广义 RNN gating），大 Δ≈reset 并关注新输入，小 Δ≈忽略当前输入保持历史。B 和 C 的选择性提供更细粒度的输入→状态、状态→输出控制
    - **时间复杂度**：训练 O(BLDN) 比 Transformer O(BL²D) 在长序列上更高效；自回归推理 O(1) 每步，无需 KV cache
    - **Theorem 1**：当 N=1, A=-1, B=1 时，选择性 SSM 退化为经典 gated RNN: g_t = σ(Linear(x_t)), h_t = (1-g_t)h_{t-1} + g_t·x_t

  - 关键结果：
    - Mamba-2.8B 在零样本评估中平均 accuracy 63.3%，超过 Pythia-6.9B (61.7%) 和同规模所有 baseline
    - Mamba-1.4B 生成吞吐量 5× 于同规模 Transformer
    - Induction Heads 训练长度 256 时完美外推至 ≥1M 长度（4000× 训练长度）
    - DNA 1M 上下文下 Great Apes 分类达 81.31% (7M 参数模型)
    - SC09 语音生成 FID 0.67 (24.3M)，超越 WaveGAN (2.03)、DiffWave (1.92) 等 GAN/扩散方法

## MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - MicroMix 提出基于 Microscaling (MX) 数据格式的混合精度量化算法，支持 MXFP4 (E2M1)、MXFP6 (E3M2/E2M3)、MXFP8 (E4M3/E5M2) 三种精度，对每个线性层自适应分配三种精度通道比例。核心算法机制：(1) 通过排列（permutation）将激活通道按绝对值均值升序重排；(2) 定义量化阈值 T(4) 和 T(6)（基于 INT8 量化误差上界推导，确保 MXFP 量化误差不超 INT8 上界），将超过阈值的元素分配到更高精度；(3) 离线计算每层的 p4/p6/p8 比例和排列 σ。推理时激活 online 执行 fused reorder-and-quantize，权重 offline 预量化。GEMM 使用 CUTLASS MXFP kernel，各精度分组独立计算后拼接。
  - 实验比较了 QuaRot (W4A4)、QUIK (mixed 4/8)、Atom (mixed 4/8 with 128 INT8 channels)、FlatQuant (W4A4)、AMXFP4 (MXFP4 only)、INT6 baseline，以及 FP16 参考。比较维度：(a) zero-shot accuracy (ARC_C, BoolQ, Lambada, PIQA, Winogrande); (b) 5-shot MMLU; (c) WikiText2 PPL; (d) 代码生成 (HumanEval, MBPP); (e) 数学推理 (GSM8K, MATH, MMLU-STEM, CMATH); (f) 单 kernel 延迟; (g) prefill/decode 端到端性能; (h) 峰值内存占用。
- 硬件平台是什么，配置是什么。
  - NVIDIA RTX 5070Ti Laptop GPU（Blackwell）、RTX 5090（Blackwell）、RTX PRO 6000（Blackwell），均支持 FP4 Tensor Cores。Blackwell FP4 Tensor Core 吞吐为 FP16 的 4×、FP8/INT8 的 2×。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama3.1-8B、Qwen2.5-32B、Qwen2.5-Coder-14B/32B-Instruct、Qwen2.5-Math-7B-Instruct、Mixtral-8x7B-v0.1-Instruct。
  - 校准数据集：WikiText2、Pile、C4（校准采样 32 条，覆盖 batch size 8/16/32/64 和 sequence length 512/1024/2048/4096）。
  - Zero-shot benchmarks (lm-eval)：ARC_C、BoolQ、Lambada、PIQA、Winogrande。
  - 5-shot：MMLU。
  - PPL：WikiText2。
  - 代码 benchmarks：HumanEval、HumanEval+、MBPP、MBPP+。
  - 数学 benchmarks：GSM8K、MMLU-STEM、CMATH、MATH。
- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/lwy2020/MicroMix
  - 算法流程（伪代码）：
    ```
    # === 离线校准阶段 ===
    for each linear layer k in model:
      # 用校准数据前向传播，收集该层输入激活 X^k ∈ R^{L×I}
      M^k[j] = (1/L) * Σ_i |X^k[i,j]|  for j = 1..I  # 通道绝对值均值
      σ^k = argsort(M^k)  # 升序排列索引

      # 计算阈值：T(n) = 2^b * 2^(n-1)/q_max * max(|X|)/254
      # 对 MXFP4 (E2M1): n=4, b=1, q_max=6
      T(4) = 2^1 * 2^3/6 * max(|X|)/254 = max(|X|)/95.25
      # 对 MXFP6 (E3M2): n=6, b=3, q_max=28
      T(6) = 2^3 * 2^5/28 * max(|X|)/254 = max(|X|)/27.8

      # 分组：按重排后的值划分
      X_sorted = X^k[:, σ^k]
      G4 = X_sorted[:, 0:p4*I]  where |x| ≤ T(4)
      G6 = X_sorted[:, p4*I:(p4+p6)*I]  where T(4) < |x| ≤ T(6)
      G8 = X_sorted[:, (p4+p6)*I:I]  where |x| > T(6)

      # 存储配置 (p4^k, p6^k, p8^k, σ^k)

    # === 量化阶段 ===
    # 权重量化（离线，一次性）：
    W^k_reordered = W^k[σ^k, :]  # 按激活排列重排
    for block in W^k_reordered.reshape(-1, 32):  # block_size=32
      s = 2^{floor(log2(max(|block|))) - b}  # E8M0 scale
      Q(block) = round(clip(block/s, -q_max, q_max))

    # 激活量化（在线，fused reorder-and-quantize kernel）：
    X_quant = fused_reorder_and_quantize(X^k, σ^k, p4^k, p6^k, p8^k)
    # 输出三组 MX 格式张量 [G4_mxfp4, G6_mxfp6, G8_mxfp8]

    # === GEMM 推理 ===
    # 各分组独立执行 CUTLASS MXFP GEMM
    Y4 = MXFP4_GEMM(G4_mxfp4, W4_mxfp4)  # FP4 Tensor Core, MMA fused dequant
    Y6 = MXFP6_GEMM(G6_mxfp6, W6_mxfp6)  # FP6 MMA on Tensor Cores
    Y8 = MXFP8_GEMM(G8_mxfp8, W8_mxfp8)  # FP8 MMA on Tensor Cores
    Y = concat_and_reorder_back(Y4, Y6, Y8, σ^k)  # 恢复原通道序，输出 BF16
    ```
  - 关键结果：Llama3.1-8B 平均 5.51 bits，zero-shot avg 71.56 (FP16: 73.03)，MMLU 62.65 (FP16: 65.24)。Qwen2.5-32B 平均 5.22 bits，zero-shot avg 75.20 (FP16: 75.55)，MMLU 81.79 (FP16: 83.32)——近乎无损。Mixtral-8x7B 精度 drop <0.4 分，执行时间从 5m18s 降至 2m03s。

## MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出两种后训练（post-training）专家级稀疏化技术——(1) **Expert Pruning（专家剪枝）**：逐层使用小规模校准数据集（C4/MATH），枚举所有保留 r 个专家的组合，以最小化 Frobenius 范数量化重构损失 ‖F'(x, C) − F(x)‖_F 为目标，保留最优专家子集，永久丢弃不重要专家，减少模型参数量和部署内存。(2) **Dynamic Expert Skipping（动态专家跳过）**：在推理时，根据路由权重比值 w_{e1}/w_{e0} 与逐层阈值 β（校准集上该比值的中位数）的比较，动态决定是否跳过次优专家，减少每个 token 激活的专家数，提升推理速度。两者可组合使用。
  - 实验比较：
    - Baseline 1: Wanda（2:4 结构化稀疏，约 50% 参数减少）
    - Baseline 2: Random Expert Pruning（随机丢弃专家）
    - Baseline 3: Frequency-based Expert Pruning（基于激活频率丢弃专家）
    - 比较 r=6（保留6专家/丢弃2专家）和 r=4（保留4专家/丢弃4专家，vs 原始8专家）
    - 评估指标：8 项 EleutherAI LM Harness 零样本任务平均准确率、GSM8K 5-shot 准确率、MATH 零样本准确率、峰值 GPU 内存使用量（MB）、token 生成加速比

- 硬件平台是什么，配置是什么。
  - 推理部署：NVIDIA A100-80G GPU。原始 Mixtral 8x7B (bf16) 需 2 张 A100-80G；剪枝 r=6 或 r=4 后仅需 1 张 A100-80G
  - 微调：16 张 A100-80G GPU（MetaMathQA 微调，900 步，lr=2e-5，cosine scheduler）
  - 推理速度测试基于修改版 AutoGPTQ 脚本

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral 8x7B、Mixtral 8x7B Instruct（MoE 架构，每层 8 个专家，top-2 路由，总参数量 47B，专家占 45B 即 ~96%）
  - 校准数据集：C4（通用任务剪枝，128 条序列 × 2048 tokens）；MATH 训练集（领域特定任务剪枝，128 条序列 × 2048 tokens）
  - Benchmark（通用）：EleutherAI LM Harness 8 项零样本任务——ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande
  - Benchmark（领域特定）：GSM8K（5-shot）、MATH（零样本）
  - 微调数据集：MetaMathQA

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/Lucky-Lance/Expert_Sparsity
  - 依赖框架：HuggingFace Transformers
  - 算法 pipeline（Expert Pruning — 逐层枚举剪枝）：
    ```
    # Step 1: 校准数据前向传播，缓存每层输入-输出对
    for layer l in 1..L:
        for batch in calibration_data:
            x[l], F_l(x[l]) = forward_and_cache(layer_l, x)
    
    # Step 2: 逐层枚举剪枝
    for layer l in 1..L:
        best_loss = inf
        for C in combinations({expert_0,...,expert_{n-1}}, r):
            F'(x,C) = Σ_{j:0→r-1} w̃_{e_j} · E_{e_j}(x)
            其中 w̃_{e_j} = w_{e_j} / Σ_{m=0}^{r-1} w_{e_m}
            loss = ‖F'(x[l],C) − F_l(x[l])‖_F
            if loss < best_loss: best_experts[l] = C
    
    # Step 3: 修改 config，加载保留的专家
    # 复杂度: C(n,r) 枚举 × L 层; Mixtral 8x7B r=6 约 30min
    ```
  - 算法 pipeline（Dynamic Expert Skipping — 在线跳过）：
    ```
    # 校准阶段：逐层计算跳过阈值 β
    for layer l in 1..L:
        ratios = []  # 收集路由权重比
        for each token x in calibration_data:
            w[e0], w[e1] = top2 routing weights
            ratios.append(w[e1] / w[e0])
        β[l] = median(ratios)  # 中位数 → 跳过概率约 50%
    
    # 推理阶段：动态跳过
    for each token x in generation:
        e0, e1 = top2 routing indices
        if w[e1] < β[l] * w[e0]:
            y = E_{e0}(x)  # 仅用 top-1 专家
        else:
            y = w̃[e0]·E_{e0}(x) + w̃[e1]·E_{e1}(x)  # 用 top-2
    ```
  - 关键结果：r=6（24% 参数减少）→ 1.19-1.20× 加速，平均性能下降 ~2.9 点；r=4（48% 参数减少）→ 1.27× 加速，平均性能下降 ~7.1 点；组合剪枝+动态跳过 r=4 → 1.33× 加速。r=6 时单张 80G GPU 可部署 bf16 Mixtral 8x7B。领域特定校准（MATH 替代 C4）可大幅提升数学任务剪枝效果（GSM8K 5-shot r=6: 41.02→51.25）。微调后 r=7 的剪枝模型可超越原始 8-expert 模型。

## MobiLlama Small Language Model tailored for edge devices

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MobiLlama 是一种 0.5B（及 0.8B）参数的 SLM（Small Language Model），核心创新在于**共享 FFN（Feed-Forward Network）设计**：不同于传统 Transformer 每层一个独立 MLP 块（含多个 FFN 层），MobiLlama 让所有 22 层 Transformer block 共享同一个 MLP 块。在常规设计中（如 large-base），FFN 层占 65% 的总参数量，通过共享 FFN 可将参数量减少约 60%，使得在保持 22 层 + hidden dim 2048 的高容量配置下，总参数仅 0.5B（与低容量 baseline 相同）。0.8B 版本通过拓宽共享 FFN（hidden dim 2532, intermediate 11080）获得更高精度。
  - 实验比较：
    - **Baseline 对比（同训练budget）**：baseline1（22 层/1024 hidden/0.54B）、baseline2（8 层/2048 hidden/0.52B）、large-base（22 层/2048 hidden/1.2B）在 100B tokens 上预训练，4 benchmarks（HellaSwag, TruthfulQA, MMLU, Arc_C）对比。
    - **SOTA 对比**：与 gpt-neo-125m、tiny-starcoder、cerebras-gpt-256m、opt-350m、megatron-gpt2-345m、LiteLlama、gpt-sw3-356m、pythia-410m、xglm-564m、Lamini-GPT-LM 等 <1B 模型在 9 benchmarks 对比。
    - **Efficiency 对比**：在 RTX2080Ti GPU（bf16）、i7 CPU（4bit GGUF）、Snapdragon-685 手机（4bit GGUF）三个平台上对比 Llama2 7B、Phi2 2.7B、large-base 1.2B，指标含 Avg Tokens/Sec、Avg Memory Consumption、Avg Battery Consumption。
    - **Slicing 对比**：与 SliceGPT 30% 参数 sliced 的 OPT-1.3B/6.7B、Llama-2-7B、Phi2-2.7B 在 4 benchmarks 对比。
    - **多模态评估**：MobiLlama-V 0.8B（CLIP+LLM）在 GQA、SQA、TextQA、MME 上评估。

- 硬件平台是什么，配置是什么。
  - 预训练：20 个 GPU 节点，每节点 8×NVIDIA A100（80GB），800 Gbps 互联，NVLink + 2 port 200 Gb/s (4× HDR) InfiniBand。吞吐约 14k-15k tokens/sec/GPU。
  - 部署测试：PC with RTX 2080Ti GPU（bf16 部署）、Laptop with i7 CPU（4bit GGUF）、Smartphone with Snapdragon-685 processor（4bit GGUF）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：MobiLlama 0.5B（hidden size 2048, 22 layers, 32 heads, intermediate 5632, max seq len 2048, vocab 32000, RMSNorm, RoPE, SwiGLU）；MobiLlama 0.8B（hidden size 2532, intermediate 11080, 其余同 0.5B）；large-base 1.2B（22 layers, hidden 2048, 独立 FFN per layer）。
  - 数据集：预训练用 LLM360 Amber dataset（1.2T tokens），含 Arxiv 30B、Book 28.9B、C4 197.7B、Refined-Web 665B、StarCoder 291.9B、StackExchange 21.8B、Wikipedia 23.9B。
  - Benchmark：HellaSwag（10-shot）、TruthfulQA（0-shot）、MMLU（5-shot）、Arc_Challenge（25-shot）、CrowsPairs（0-shot）、PIQA（0-shot）、Race（0-shot）、SIQA（0-shot）、Winogrande（5-shot），共 9 个 benchmarks。多模态：GQA、SQA、TextQA、MME。
  - 评估框架：Analysis-360 framework（基于 lm-evaluation-harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/mbzuai-oryx/MobiLlama（完整训练数据pipeline、训练代码、模型权重、300+ checkpoints、评估代码）
  - 算法 pipeline 核心（共享 FFN 设计）：
    ```
    # 传统 Transformer (如 large-base):
    for layer in 1..L:
        # 每层有独立的 MHA + MLP
        h = h + MHA(LayerNorm(h), layer_id=layer)       # 独立 attention
        h = h + MLP[layer](LayerNorm(h))                 # 独立 FFN，MLP[layer] 是第 layer 层的参数
        # MLP 通常含 3 个 FFN (gate, up, down): W_gate, W_up, W_down

    # MobiLlama (共享 FFN):
    shared_MLP = MLP(W_gate_shared, W_up_shared, W_down_shared)  # 仅一份 FFN 参数
    for layer in 1..L:
        h = h + MHA(LayerNorm(h), layer_id=layer)       # 每层独立 attention（含 Q/K/V/O proj）
        h = h + shared_MLP(LayerNorm(h))                # 所有层共享同一 MLP
    ```
  - 参数量分析：在 large-base 中，FFN 参数占 65%（W_gate/W_up/W_down），attention 占 30%（Q/K/V/O proj），heads 占 5%。通过共享 FFN，整体参数从 1.2B 降至 0.5B（减少约 60%）。
  - 训练超参数：AdamW（β1=0.9, β2=0.95），初始 LR=3e-4，cosine schedule 衰减至 3e-5，weight decay=0.1，gradient clipping=1.0，warmup=2000 steps，batch size=800（160×5），Flash-Attention 加速。
  - 0.8B 版本：在 0.5B 基础上 widening shared FFN（hidden dim 2048→2532, intermediate 5632→11080），其余架构不变。
  - 多模态 MobiLlama-V：CLIP visual encoder 桥接 MobiLlama decoder，在 665k vision-language instruction 数据上端到端微调。

## Scaling Law for Quantization-Aware Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出统一的 QAT（Quantization-Aware Training）缩放定律，将 W4A4 量化误差 δ_p 建模为模型参数量 N、训练数据量 D、量化粒度 G 三个变量的函数：δ_p(N, D, G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}。在 Chinchilla 缩放定律 L(N,D) 的基础上叠加该量化误差项。关键发现：(1) δ_p 随 N 增大而减小、(2) 随 D 增大而增大、(3) 随 G 变粗而增大。进一步将 W4A4 量化误差分解为权重量化误差 δ_{W4A16} 和激活量化误差 δ_{W16A4} 两个分量，发现激活量化误差是主要瓶颈（尤其在 FC2 Proj 输入层，因其来自 SwiGLU 输出，峰度高达 89 远高于其他层）。提出混合精度方案：对 FC2 输入用 8-bit 量化，可消除激活瓶颈，使权重和激活误差贡献趋于均衡（ratio R 从 1.67 降至 0.85–1.10）。
  - 实验比较：(1) 与现有 QAT 缩放定律 [Frantar et al. 2025, Kumar et al. 2024] 对比，ours 将 N、D、G 统一建模为单条曲线（vs baseline 需为每种 G 单独拟合），W4A16 相对误差从 19.3% → 5.2%，W4A4 从 8.5% → 4.7%；(2) 消融实验：在 W4A4 精度下去除 D 项，预测相对误差从 4.7% 升至 8.6%（W4A4）和 5.2% 升至 13.8%（W4A16）；(3) INT4 vs FP4 量化精度对比（INT4 略优于 FP4）；(4) 不同量化器对比（AbsMax/LWC/LSQ 权重量化差异 <0.003，LAC 在 G>256 时显著优于 AbsMax 激活量化）；(5) FC2 8-bit 混合精度消融：G=32 量化误差降 20.5%，G=256 降 42.9%；(6) 973M 模型外推验证：缩放定律对更大模型和数据量准确外推。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 GPU，总计 268 次 QAT 实验消耗 276K GPU-hours
  - 训练框架：PyTorch，基于 OLMo2 训练超参数（AdamW β=(0.9, 0.95), weight decay 0.1, gradient clip 1.0, cosine LR schedule, warmup 500 steps, sequence length 2048）
  - 混合精度训练：BFloat16

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama3-style decoder-only Transformer，参数规模 N ∈ {74M, 145M, 297M, 595M} + 973M（外推验证），架构含 GQA + SwiGLU + RMSNorm + RoPE。详细架构配置：74M (12 layers/768 hidden/16 attn heads/4 KV heads), 145M (12/1024/16/4), 297M (12/1536/24/6), 595M (24/1536/24/6), 973M (16/2048/32/8)。Chinchilla 缩放定律拟合额外引入 2.8B/6.5B/12.7B 模型（OLMo-2 官方 release）。
  - 数据集：OLMo2-Mix-1124 pretraining 数据集，训练 token 数 D ∈ {10B, 20B, 50B, 100B} + 200B（外推验证）
  - Benchmark/评估指标：smoothed training loss 作为验证损失的无偏估计（与 Chinchilla 一致），量化误差 δ_p 定义为 loss_bf16 − loss_W4A4

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明代码开源。实验基于 PyTorch + HuggingFace Transformers，使用 OLMo2 开源训练框架。

  **算法 pipeline（QAT 训练 + 缩放定律拟合流程）：**

  ```
  # === Step 1: 全精度 Chinchilla 缩放定律拟合 ===
  # 用 BF16 模型训练数据拟合 L(N,D) = A/N^α + B/D^β + E
  # 使用 Huber loss + L-BFGS 优化器估计 (E, A, α, B, β)
  # 约束 α = β (与 Chinchilla 原文一致)
  # 输入: 145M-12.7B 模型的训练 loss，输出: 拟合参数

  # === Step 2: W4A4 QAT 训练 ===
  for each (N, D, G) in experiment_grid:
      model = Llama3Style(N)  # N ∈ {74M, 145M, 297M, 595M, 973M}
      for step in range(D):
          # 前向传播（量化插入在 Linear 层前后）:
          X_int4 = quantize_activation(X, G, AbsMax/LAC)
          W_int4 = quantize_weight(W, G, AbsMax)
          Y = INT4_GEMM(X_int4, W_int4) * s_X * s_W  # Fake-quantized forward
          loss = CrossEntropy(Y, labels)
          # STE 反向传播（通过量化器的直通估计器）
          loss.backward()
          optimizer.step()
      # 记录最终量化误差:
      δ_W4A4 = loss_bf16 - loss_W4A4  # 量化误差 ground truth

  # === Step 3: W4A16/W16A4 解耦训练（误差分解用） ===
  for each (N, D, G):
      # W4A16: 仅权重量化，激活保持 BF16
      # W16A4: 仅激活量化，权重保持 BF16
      # 分别记录 δ_W4A16 和 δ_W16A4
  # 验证 δ_W4A4 ≈ k · (δ_W4A16 + δ_W16A4)，k≈0.906

  # === Step 4: 缩放定律拟合 ===
  # δ_p(N,D,G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}
  # 用 80 次 W4A4 QAT 实验数据拟合 k, γ_N, γ_D, γ_G
  # 使用 Huber loss + L-BFGS

  # === Step 5: FC2 瓶颈分析与混合精度 ===
  # 分析各层 kurtosis → FC2 Proj 输入 kurtosis=89（vs 其他层 <10）
  # 原因: FC2 输入来自 SwiGLU 的 gating + 非线性变换，产生 outlier
  # 方案: FC2 Proj 输入保持 8-bit 量化，其余保持 4-bit
  # 效果: δ_W16A4(FC2 8-bit) 与 δ_W4A16 的 ratio R ∈ [0.85, 1.10]
  ```

  **量化器实现细节：**
  - 权重 AbsMax：s = M / max(|W|), W_int = clamp(round(W/s), -2^{b-1}, 2^{b-1}-1)
  - 激活 AbsMax（G<256）：同上，按 group 计算
  - 激活 LAC（G≥256）：s = M / (max(|X|)·γ)，γ 为可学习 clipping factor，同 group index 共享
  - W4A4 使用 INT4（16 个可表示值），优于 FP4 E2M1（15 个可表示值），尤其在 per-channel/token 粒度下差距 0.015 loss

  **关键结果：** 
  - 量化误差趋势：N 从 74M→594M，δ_{W4A4} 平均降 34%；D 从 10B→100B，δ_{W4A4} 平均升 22%；G 从 finest→coarsest，δ 差 0.037（约半数粗粒度误差）
  - 误差分解：δ_{W4A16} 对 D 的敏感度 γ_D=0.1610 远大于 δ_{W16A4} 的 γ_D=0.0331；δ_{W16A4} 对 G 的敏感度 γ_G=0.9812 远大于 δ_{W4A16} 的 γ_G=0.3533
  - 激活量化误差始终大于权重量化误差（ratio R>1），但随着 D/N 增大差距缩小
  - FC2 8-bit 后，W4A4 EPM（Effective Parameter Multiplier）提升 0.06–0.14，W4A4 EPM 始终 >0.5（即 W4A4 优于此 W8A8 的 cost-accuracy trade-off）
  - 973M/200B tokens 外推验证缩放定律准确预测趋势

## Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 MoDiff（Modulated Diffusion）框架，通过 modulated quantization（调制量化）和 error-compensated modulation（误差补偿调制）两个创新机制加速扩散模型的生成过程。核心思想是对相邻时间步之间的激活差值（temporal difference a_t − a_{t+1}）而非原始激活值进行量化——因为差值范围远小于原始激活（约10×），从而可以用更低 bit-width 的量化达到同等精度。实验将 MoDiff 应用于 Q-Diffusion（Q-Diff）和动态逐通道量化（LCQ）两种 baseline PTQ 方法，比较在 CIFAR-10、LSUN-Churches、LSUN-Bedrooms 上不同激活 bit-width（8/6/4/3/2-bit）下的生成质量（IS、FID、sFID）和理论计算量（GBops）。同时在 Stable Diffusion v1.4（MS-COCO）、DiT-XL/2（ImageNet）、SDXL-Turbo（few-step）上验证泛化性。还包含 DDPM、DPM-Solver、PLMS 等不同 sampler 的兼容性实验。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体硬件平台。效率评估使用 DeepSpeed 工具统计每个去噪步骤的单图理论二元运算次数（BOPs/GBops），不报告实际 wall-clock time 加速。论文明确说明硬件实现是未来工作方向。

- 模型是什么。数据集和bench分别是什么。
  模型：DDIM（CIFAR-10，100步）、Latent Diffusion Model LDM-4（LSUN-Bedrooms，500步）和 LDM-8（LSUN-Churches，200步）、Stable Diffusion v1.4（MS-COCO 2014，50步，DPM solver）、DiT-XL/2（ImageNet 256×256，50步，Transformer架构）、SDXL-Turbo（few-step，2/4/8步，结合 MixDQ）。数据集：CIFAR-10（32×32）、LSUN-Bedrooms（256×256）、LSUN-Church-Outdoor（256×256）、MS-COCO 2014、ImageNet 256×256。评估指标：IS、FID、sFID（基于50k生成图像），Precision/Recall，以及 GBops。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/WeizhiGao/MoDiff

  算法 pipeline 伪代码（以单层线性算子 A^(l) 为例）：

  ```
  # === 初始化 (t=T, 第一步: Warm-up with full precision) ===
  a_hat[T] = Q(a_T)                          # 量化原始激活 (Eq.8)
  o_hat[T] = A(a_hat[T])                     # 量化后计算输出 (Eq.9)

  # === 迭代 (t = T-1 到 1): Error-Compensated Modulation ===
  for t in range(T-1, 0, -1):
      # Step 1: 误差补偿激活重建 (Eq.13)
      a_hat[t] = Q(a_t - a_hat[t+1]) + a_hat[t+1]

      # Step 2: 调制量化计算 (Eq.14)
      o_hat[t] = A(Q(a_t - a_hat[t+1])) + o_hat[t+1]
  ```

  关键张量计算与实现细节：
  - 标准 PTQ：各时间步独立量化原始激活 a_t → Q(a_t) → A(Q(a_t))，误差独立、各步不共享信息
  - MoDiff 调制计算：利用线性算子 A 的线性性，将 o_t = A(a_t) 等价重写为 o_t = A(a_t − a_{t+1}) + o_{t+1}
  - 调制量化：对差值 Δ_t = a_t − a_{t+1} 量化，其范围约 10× 小于原始激活，同等 bit-width 下量化误差显著降低
  - 误差补偿：用 â_{t+1} 替代 a_{t+1}，使上一步量化误差 e_{t+1} = a_{t+1} − â_{t+1} 被自动注入到下一步差值计算中补偿。Theorem 4.4 证明标准调制误差累积呈 2^{T−k} 指数增长，误差补偿调制呈 (2c)^{T−k} (c<1/2) 指数衰减
  - 与缓存方法关系：当激活差值范围低于阈值时，Q 可分配 0-bit（即跳过计算），此时 MoDiff 退化为 DeepCache 等缓存方法的超集
  - 实现要点：(1) 移除所有应用 MoDiff 的层的 bias 项；(2) 第一步 Warm-up 使用全精度激活；(3) 逐层重构校准数据集；(4) 逐层而非逐 block 重构以保证稳定性
  - 额外内存开销：CIFAR-10 单图生成时仅为 35–39 MB（含中间变量 â_t 和 ô_t），batch size 增大后仍可控

## OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：OmniQuant 是一种可微分后训练量化（PTQ）方法，在冻结原始 FP16 权重的前提下，引入少量可学习量化参数（Learnable Weight Clipping 的 γ/β 和 Learnable Equivalent Transformation 的 s/δ/s_a），通过 block-wise 量化误差最小化用 SGD 优化，求得最优量化配置。包含两个核心模块：
    1. **Learnable Weight Clipping (LWC)**：学习权重截断强度 γ ∈ [0,1], β ∈ [0,1]（Eq.2），通过 `h = (γ max(W) - β min(W)) / (2^N - 1)` 动态调整量化步长，将 MinMax 量化推广到可学习版本；γ=1, β=1 时退化为 MinMax。
    2. **Learnable Equivalent Transformation (LET)**：在 attention 和 FFN 层中学习通道级缩放 s 和偏移 δ（Eq.3-5），将激活量化的难度等效迁移到权重量化上；同时将等效变换扩展到 Q/K 的矩阵乘法（Eq.5），使 KV cache 也可被量化。
  - 所有可学习参数量化后可融合进权重，推理时不引入额外计算或参数。
  - 实验比较：
    - Weight-only quantization（W2A16, W3A16, W4A16，含 per-channel 和 group-wise g128/g64 变体）：对比 RTN、GPTQ、AWQ，指标为 WikiText2/PTB/C4 perplexity。
    - Weight-activation quantization（W6A6, W4A4）：对比 SmoothQuant、Outlier Suppression+ (OS+)、RPTQ、LLM-QAT，指标为 PIQA、ARC-e、ARC-c、BoolQ、HellaSwag、Winogrande 零样本任务准确率，以及 WikiText2/C4 perplexity。
    - 指令微调模型（LLaMA-2-chat）W3A16g128：在 Vicuna-Bench 上对比 RTN、AWQ（GPT-4 评估，win rate）。
    - MMLU 零样本评估（Table A16）。
    - 真实设备加速：通过 MLC-LLM 部署在 A100-80G 上，测试 weight memory、running memory 和 token/s 吞吐量。

- 硬件平台是什么，配置是什么。
  - 量化训练：单卡 NVIDIA A100-40G GPU，LLaMA-7B W4A4 约 1.6 小时，LLaMA-65B W4A4 约 14.4 小时
  - 部署测试：单卡 NVIDIA A100-80G GPU，通过 MLC-LLM 评测推理吞吐和显存占用
  - 校准数据：WikiText2 中随机 128 个 2048-token 段落，batch size=1
  - 训练配置：AdamW 优化器（weight decay=0），LWC 学习率 5e-3，LET 学习率 1e-2，默认 20 epochs（W2A16 用 40 epochs）

- 模型是什么。数据集和bench分别是什么。
  - 模型：OPT（125M–66B）、LLaMA-1（7B–65B）、LLaMA-2（7B–70B）、LLaMA-2-chat（7B/13B，指令微调版）、Falcon-180B
  - 校准数据集：WikiText2（128 segments × 2048 tokens）
  - 评估数据集（perplexity）：WikiText2、PTB、C4
  - 评估 benchmark（零样本准确率）：PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、Winogrande、MMLU
  - 生成质量评估：Vicuna-Bench（80 个问题，GPT-4 评分）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/OpenGVLab/OmniQuant
  - 部署框架：https://github.com/mlc-ai/mlc-llm
  - 基于 PyTorch + HuggingFace Transformers 实现，lm-eval-harness 用于零样本评估

  **OmniQuant 核心算法（伪代码）：**

  ```
  Input: calibration dataset X, pre-trained LLM model M
  Output: quantized model

  X_fp = X_q = X  # full-precision and quantized model inputs

  for each transformer block B_i in M:  # block-wise calibration
      X_fp = B_i(X_fp)
      init LWC parameters Θ1 = {γ=1, β=1}
      init LET parameters Θ2 = {s=SmoothQuant_init, δ=OS+_init, s_a=1}

      for k in range(epochs):
          for (x_q, x_fp) in (X_q, X_fp):
              # Eq.(3): tilde_X = (X - δ) ⊘ s, tilde_W = s ⊙ W, tilde_B = B + δW
              # Eq.(5): tilde_Q = Q ⊘ s_a, tilde_K^T = s_a ⊙ K^T
              B_i' = LET(B_i, Θ2)

              # Eq.(2): h = (γ*max(W)-β*min(W))/(2^N-1)
              #          W_q = clamp(round(W/h)+z, 0, 2^N-1)
              B_i' = Quantize(B_i', Θ1)

              x_q' = B_i'(x_q)
              loss = ||x_fp - x_q'||^2
              loss.backward()
              update Θ1, Θ2 via AdamW

      # Fuse and finalize
      B_i = LET(B_i, Θ2)  # absorb scaling into weights
      B_i = Quantize(B_i, Θ1)
      X_q = B_i(X_q)

  return quantized model M
  ```

  **LET 等效变换详解**：
  - 线性层（Eq.3）：Y = XW + B = [(X-δ)⊘s] · [s⊙W] + [B+δW]，s, δ ∈ R^{1×Cin} 为通道级 scale/shift
  - Attention（Eq.5）：P = Softmax(Q@K^T) = Softmax((Q⊘s_a) @ (s_a⊙K^T))，s_a ∈ R^{1×Cout}
  - 融合：tilde_X 中的 s, δ 吸收到前一层 LayerNorm/Linear 中；tilde_W 中的 s 和 tilde_Q/tilde_K 中的 s_a 融入原始权重矩阵
  - 应用位置（四对 LET）：[ln1, (q_proj,k_proj,v_proj)]、[v_proj,out_proj]、[Q,K]、[ln2,fc1]（第二层 FFN 除外）

  **LWC vs PACT/LSQ**：
  - PACT 直接学习绝对截断阈值，LSQ 学习绝对 scale/zero-point；当 LET 每轮改变权重分布时两者收敛失败（Table A14, Figure A5）
  - LWC 学习相对截断强度 γ/β ∈ [0,1]，处理权重分布变化时更稳定

  **校准效率**：LLaMA-7B 在 W3A16 下仅需 16 个样本即可收敛（Table A11）；校准数据集切换为 C4/Pile 时 perplexity 波动仅 0.0006-0.17（Table A10）。

## PB-LLM Partially Binarized Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PB-LLM 提出部分二值化 LLM 权重矩阵的方法，核心包括：(1) **部分二值化权重矩阵**——通过 magnitude 或 Hessian 指标按元素粒度检测 salient weights（显著权重），将小比例（如 5%-30%）salient weights 保留在高位宽（如 8-bit），其余 90%+ 权重二值化为 ±1（乘以 optimal scaling factor）；(2) **PB-GPTQ（PTQ 方法）**——将 GPTQ 的 Hessian 引导迭代量化扩展到部分二值化场景，逐列量化权重矩阵，对 unsalient 权重二值化、salient 权重高比特量化后，计算 Hessian 补偿并应用到剩余列；(3) **QAT 方法**——冻结 salient weights 保持全精度（不参与训练），对 residual binary weights 使用可解析推导的 optimal scaling factor α* = ||w_F||_1 / n（column-wise L1 norm 平均）来最小化 L2 binarization error。
  - 实验比较：(a) 5 种已有 binarization 方法（BNN, XNOR, Bi-Real, ReCU, FDA）直接应用于 OPT-1.3B 的效果（均在随机猜测以下）；(b) PB-GPTQ vs RTN 在不同 salient fraction（50%/20%/10%/5%）下的 C4 perplexity，以及 Magnitude vs Hessian 检测准则对比；(c) PB-GPTQ layer-wise vs group-wise (g=128) 对比；(d) PB-LLM QAT vs LLM-QAT、SmoothQuant、RTN、PB-GPTQ 在 LLaMA-7B 上的 7 个零样本常识推理任务（BoolQ, PIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OBQA）和 perplexity（C4, WikiText2, PTB）；(e) 训练效率：PB-LLM 仅需 1-10K iterations 恢复性能，而 LLM-QAT 需要 100K iterations。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号。使用 PyTorch 框架进行训练和评估。QAT 训练使用 AdamW 优化器（zero weight decay），batch size=1 per GPU，learning rate=2e-5，cosine learning rate decay。由于计算资源限制，方法论探索主要使用 OPT-1.3B，评估扩展到 LLaMA-7B。

- 模型是什么。数据集和bench分别是什么。
  - 模型：**OPT-1.3B**（方法论探索和消融实验）、**LLaMA-7B**（主要实验结果）。
  - 训练数据：**RedPajama-simple-1B**（RedPajama-1T 的 0.1% 子集，包含 Commoncrawl, C4, GitHub, Wikipedia, Books3, ArXiv, Stackexchange）。
  - 评估数据集/benchmark：(a) 7 个零样本常识推理任务：**BoolQ, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OBQA**（使用 lm-eval-harness）；(b) Perplexity：**WikiText2, C4, PTB**。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub **https://github.com/hahnyuan/PB-LLM**。
  - 算法 pipeline 核心流程：

  **部分二值化权重矩阵格式**：
  ```
  W ∈ R^{d_o × d_i} (全精度预训练权重)

  # Step 1: Salient Weight Detection
  For PTQ (Hessian 准则):
    H = 2 X X^T  (Hessian 矩阵，X 为 calibration 特征)
    v_i = w_i^2 / [H^{-1}]_{ii}^2  (saliency metric)
    选 saliency top-k% 为 salient weights W^{sal}

  For QAT (Magnitude 准则):
    按 |w_i| 排序，选 top-k% 为 salient weights W^{sal}

  # Step 2: 部分二值化
  W^{unsal} → sign(W^{unsal}) → {-1, +1}
  W^{sal} → 保留高比特 (asymmetric per-channel INT8)

  存储: N_bit ≤ 1 * r_binary + 8 * (1 - r_binary) + 1 (bitmap)
  10% salient → 最多 2.7-bit 等效量化
  ```

  **PB-GPTQ (PTQ) 算法流程**：
  ```
  输入: W ∈ R^{d_o × d_i}, calibration data X, salient fraction k%
  1. 计算 H = 2 X X^T, H^{-1}
  2. 逐列迭代量化:
     For column q in [1, d_i]:
       # 识别当前列 salient/unsalient
       W_q^{sal}, W_q^{unsal} = split_by_saliency(W[:, q])

       # 二值化 unsalient: α_q = mean(|W_q^{unsal}|), Ŵ_q^{unsal} = α_q * sign(W_q^{unsal})
       # 量化 salient: Ŵ_q^{sal} = MinMaxQuant(W_q^{sal}, bit=8)

       # GPTQ Hessian 误差补偿
       δ = (W[:, q] - Ŵ[:, q]) / [H^{-1}]_{qq} * (H^{-1})_{:, q}
       W[:, q+1:] += δ
  3. 输出: partially-binarized 权重矩阵 Ŵ
  ```

  **QAT 训练流程**：
  ```
  1. Salient Weights Frozen:
     W^{sal} = top-k% by |W|  (一次性检测)
     训练全程 freeze W^{sal}，仅更新 W_F^{unsal}

  2. Optimal Scaling Factor (column-wise):
     w̄_B = sign(w_F)
     α* = ||w_F||_1 / n  (闭式解, minimize ||w_F - α w̄_B||_2^2)
     前向: y = W^{sal} x + α* · sign(W_F^{unsal}) x

  3. 反向 (STE): ∂L/∂x = ∂L/∂sign(x) if |x| ≤ 1 else 0

  4. 训练: AdamW, lr=2e-5, cosine decay, 10K iters
  ```

  关键结果：PB-LLM 30% salient (等效 ~3.7 bit) 在 LLaMA-7B 上 Avg 66.9 vs FP 68.7；PB-LLM 10% salient (等效 ~1.7 bit) Avg 60.6。QAT 可大幅恢复 PTQ 性能（PB-GPTQ 10% Avg 36.5 → PB-LLM 10% Avg 60.6）。

## PM-KVQ Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PM-KVQ 是一种面向 long-CoT LLM 的 post-training KV Cache 混合精度量化方法，包含三个核心组件：(1) **Progressive Quantization（渐进量化）**——初始以 16-bit 存储 KV Cache，当显存耗尽时通过 bit-width shrinking 逐步将已有 KV Cache 降位宽（16→8→4→2 bit），同时新 token 继续以当前最高位宽存储，等价右移操作 `X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b` 等效于先反量化再量化；(2) **Block-wise Memory Allocation（块级内存分配）**——用一阶 Taylor 近似估计每个 transformer block 的 KV Cache 对 b-bit 量化的敏感度 `s_{i,b} = ||G_{K_i} ⊙ (K_i - Q_b(K_i))||_1 + ||G_{V_i} ⊙ (V_i - Q_b(V_i))||_1`，将 bit-width 分配形式化为 Integer Programming 问题由 CVXPY 求解，为更敏感的 block 分配更高位宽；(3) **Calibration with Positional Interpolation（位置插值校准）**——用短上下文校准数据（2048 tokens）配合 RoPE 位置插值 `cos(s·mθ_i)` 近似长上下文数据分布（s=4 嵌入 8192 上下文），避免因 RoPE 低频通道周期超过校准数据长度导致校准偏差。同时继承 KIVI 的 channel-wise reparameterization（将 Key Cache outlier 迁移到 Query）、首 token INT16 保留和 128 token 滑动窗口。
  - 实验比较：PM-KVQ vs **RotateKV**（uniform bit-width + Hadamard rotation）、**MiKV**（heavy-hitter oracle + mixed precision）、**KIVI**（per-channel Key + per-token Value），在 4-bit（DeepSeek-LLaMA-8B）和 2-bit（其他 LLM）下比较数学推理（AIME-2024/2025 pass@1 和 Voting）和编程（LiveCodeBench pass@1、CMIMC-2025 pass@1 和 Voting）。消融实验：(a) 三种 bit-width shrinking 策略对比（Direct Right Shift / Modified Right Shift / Equivalent Right Shift）；(b) 不同校准长度和位置插值因子对比。

- 硬件平台是什么，配置是什么。
  - 性能评估：8×A100-80G GPU 服务器（fake quantization，非真实量化推理）。单 GPU 设定：DeepSeek-Qwen-7B 用 1×4090-24G、DeepSeek-LLaMA-8B 用 1×4090-24G、DeepSeek-Qwen-14B 用 1×A100-40G、DeepSeek-Qwen-32B 用 1×A100-80G、QwQ-32B 用 1×A100-80G、DeepSeek-LLaMA-70B 用 1×A100-80G。

- 模型是什么。数据集和bench分别是什么。
  - 模型：**DeepSeek-R1-Distill-Qwen-7B/14B/32B**、**DeepSeek-R1-Distill-LLaMA-8B/70B**、**QwQ-32B**。所有模型使用 GQA 或 MHA attention。
  - 校准数据集：**RedPajama arXiv 子集**，512 条样本，每条 2048 tokens。位置插值因子 s=4（有效上下文 8192 tokens），α 参数在 [0,1] 以 grid size 20 搜索最小化 self-attention 重建损失。
  - 评估 Benchmark：**AIME-2024**（30 题）、**AIME-2025**（30 题）——数学竞赛 pass@1 和 Voting（16 次采样）；**CMIMC-2025**——数学竞赛 pass@1 和 Voting（16 次采样）；**LiveCodeBench**（2025.1.1-4.6 题目）——代码生成 pass@1（4 次采样）。采样参数：temperature=0.6, top-p=0.95, max output length=32768 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub **https://github.com/thu-nics/PM-KVQ**。
  - PM-KVQ 算法完整 Pipeline 伪代码：

  ```
  === 阶段 1: 离线校准与块级内存分配 ===
  输入: long-CoT LLM (N 个 transformer blocks), 校准数据 D_cal (512 seqs × 2048 tokens)

  # Step 1.1: 块级敏感度分析
  for i = 1 to N:  # 每个 transformer block
      用 D_cal 做前向传播，计算 KV Cache 和梯度
      for b in B (可选位宽集合，如 {2,4} 或 {4,8}):
          对 K_i, V_i 做 b-bit fake quantization → Q_b(K_i), Q_b(V_i)
          计算敏感度: s_{i,b} = ||G_{K_i} ⊙ (K_i - Q_b(K_i))||_1
                            + ||G_{V_i} ⊙ (V_i - Q_b(V_i))||_1

  # Step 1.2: Integer Programming 求解 bit-width 分配
  构建 ILP:
    minimize  Σ_i Σ_b x_{i,b} · s_{i,b}
    subject to:
      Σ_b x_{i,b} = 1, ∀i   # 每个 block 分配一个 Fbit
      Σ_i Σ_b x_{i,b} · (Mem(Q_b(K_i)) + Mem(Q_b(V_i))) ≤ M  # 显存约束
      x_{i,b} ∈ {0,1}
  调用 CVXPY 求解 → 得到每个 block i 的目标 Fbit b_i^*

  # Step 1.3: 位置插值校准
  对 RoPE 旋转矩阵引入位置缩放因子 s=4:
    [K̃_{m,i}; K̃_{m,i+d/2}] = [cos(s·mθ_i) -sin(s·mθ_i); sin(s·mθ_i) cos(s·mθ_i)] [K_{m,i}; K_{m,i+d/2}]
  用校准数据计算 channel-wise reparameterization factor:
    λ_i = (max_m K_{m,i})^α
  其中 α 在 [0,1] 以 grid size 20 搜索最小化 self-attention 重建损失

  === 阶段 2: 推理时渐进量化 ===
  # 对每个 transformer block i，目标 Fbit = b_i^*
  # 设当前 KV Cache 位宽为 current_bit = 16
  # 预留显存 budget = b_i^* 对应最大上下文长度所需显存

  for each decoding step t:
      # 新 token 的 K, V 以 current_bit 存储
      K_new, V_new = current_token_key_value

      if memory_usage + new_token_memory > memory_budget:
          # 位宽缩减: current_bit 降一档 (16→8, 8→4, 4→2)
          new_bit = current_bit // 2
          # Equivalent Right Shift: 等效于反量化→量化
          X_new_bit = ((2^{2*new_bit} - 2^{new_bit} + 1) * (X_old + 2^{new_bit-1})) >> (3*new_bit)
          # 零点和缩放因子更新: Z_b = Z_{2b}, S_b = (2^b + 1) S_{2b}
          current_bit = new_bit

      # 存储新 token KV 到 cache
      KV_cache.append(K_new, V_new, bit=current_bit)

      # 注意力计算（混合精度）:
      # - 首 token 始终 INT16
      # - 最近 128 token 始终 INT16
      # - 其余按 current_bit × Fbit 的渐进量化结果
      attention_output = mixed_precision_attention(Q, KV_cache)
  ```

  - 位宽缩小策略核心张量计算（Equivalent Right Shift）：
    - **16→8 bit**: `X_8 = ((2^{16} - 2^8 + 1)(X_16 + 2^7)) >> 24`，`Z_8 = Z_16`, `S_8 = (2^8 + 1)S_16`
    - **8→4 bit**: `X_4 = ((2^8 - 2^4 + 1)(X_8 + 2^3)) >> 12`，`Z_4 = Z_8`, `S_4 = (2^4 + 1)S_8`
    - **4→2 bit**: `X_2 = ((2^4 - 2^2 + 1)(X_4 + 2^1)) >> 6`，`Z_2 = Z_4`, `S_2 = (2^2 + 1)S_4`
    - 等效性证明：先将 2b-bit 量化整数反量化为浮点 → 再重新量化为 b-bit，等价于上述整数移位操作

  - 关键超参数：
    - 量化方式：asymmetric group-wise quantization，group size=128
    - Fbit 设置：DeepSeek-LLaMA-8B 用 4-bit，其他 LLM 用 2-bit
    - 可选位宽集合 B：DeepSeek-LLaMA-8B 用 {4, 8}，其他 LLM 用 {2, 4}
    - 首 token：INT16 保留
    - 滑动窗口：最近 128 tokens INT16（继承自 KIVI/SKVQ）

  - 核心结果：PM-KVQ 在 2-bit 下比 KIVI 提升最高 8% pass@1（数学推理），Voting 提升最高 15.56%（7B 级模型）、17.78%（10B-32B 级模型）。70B 级 LLaMA-70B 在 AIME-2024 上 pass@1 从 KIVI 的 51.88% 提升到 64.79%。

## PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PMQ-VE 是一个面向多帧视频增强模型的 PTQ 量化框架（W/A 联合量化），包含粗-细两阶段过程：(1) **BMFQ（Backtracking-based Multi-Frame Quantization）**——粗阶段：对多帧激活张量 X∈R^{N×C×H×W} 进行 per-frame 独立量化，为每帧 X_i 搜索最优裁剪边界 (lb_i, ub_i)。采用百分位数初始化（lb∈[p0.1, p10], ub∈[p90, p99.9]），通过回溯搜索算法（BTBI）在候选空间中递归评估量化误差，剪枝后回溯以寻找最优解；(2) **PMTD（Progressive Multi-Teacher Distillation）**——精阶段：层次化蒸馏框架，训练低比特模型时使用 FP 教师和中间比特教师（如 INT8）共同监督。损失函数 L_PMTD = (L_INT + α(t)·L_FP) / (1+α(t))，α(t) 线性增长，L_INT 和 L_FP 各自包含输出重建损失（L2）和中间特征匹配损失（MSE, λ=5）。量化函数为 fake quantization（uniform quantizer + STE），量化 Linear 和 MatMul 层。
  - 实验比较：PMQ-VE (W2A2, W4A4) vs **OpenVINO** [Gorbachev et al. 2019], **TensorRT** [Vanholder 2016], **SNPE** [Ignatov et al. 2018], **Percentile** [Li et al. 2019], **MinMax** [Jacob et al. 2018], **NoisyQuant** [Liu et al. 2023], **DBDC+Pac** [Tu et al. 2023], **2DQuant** [Liu et al. 2024]，在三个视频增强任务（STVSR, VSR, VFI）的多个 benchmark 上比较 PSNR、SSIM、LPIPS、NIQE。

- 硬件平台是什么，配置是什么。
  - 8 张 NVIDIA V100 GPU。PyTorch 框架实现。训练：Adam 优化器，初始 lr=2×10^-4，Cosine Annealing，20000 次迭代。初始化和蒸馏微调阶段 batch size 分别为 8 和 2 per GPU。数据增强：随机裁剪、旋转、翻转。

- 模型是什么。数据集和bench分别是什么。
  - 模型/backbone：**RSTT** [Geng et al. 2022]（STVSR 任务）、**MIA** [Zhou et al. 2024]（VSR 任务）、**EMA-VFI** [Zhong et al. 2024]（VFI 任务）。均为 Transformer-based 视频增强模型。
  - 数据集/benchmark：**Vimeo-90K** [Xue et al. 2019] 训练集用于所有任务训练；**Vid4** [Liu & Sun 2013] 和 Vimeo-90K 测试集（含 Vimeo-Fast/Medium/Slow）作为评估 benchmark。评估指标：PSNR、SSIM（Y 通道 YCbCr 色彩空间）、LPIPS [Zhang et al. 2018]、NIQE [Mittal et al. 2012]。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/xiaoBIGfeng/PMQ-VE（论文承诺开源），PyTorch 实现。
  - BMFQ 伪代码（BTBI 算法）：
    ```
    输入: 多帧激活张量 X ∈ R^{N×C×H×W}, 步长 ΔL, ΔU, 阈值 ε
    输出: 每帧最优边界 (lb_i*, ub_i*), i=1..N
    
    For each frame i in 1..N:
        X_i = X[i,:,:,:]
        lb_0 = p_0.1(X_i), ub_0 = p_99.9(X_i)   // 百分位初始化
        visited = {}, error_min = ∞
        
        Function Backtrack(lb, ub):
            if (lb, ub) in visited or out of [p0.1,p10]×[p90,p99.9]:
                return
            visited = visited ∪ {(lb, ub)}
            X_q = Quantize(X_i, lb, ub)           // clamp + round + dequantize
            err = ||X_i - X_q||_2
            if err > error_min + ε: return        // 剪枝
            if err < error_min:
                error_min = err; lb_i* = lb; ub_i* = ub
            foreach (δ_l, δ_u) in {±ΔL, ±ΔU}:
                Backtrack(lb+δ_l, ub+δ_u)
        
        Backtrack(lb_0, ub_0)
    ```
  - PMTD 蒸馏流程：
    ```
    // 训练 4-bit 模型
    // Teacher 1: FP32 教师模型; Teacher 2: INT8 中间教师模型
    for t in 1..T:
        // 前向传播
        out_student = QuantizedModel_4bit(x)
        out_int8 = INT8_Teacher(x)           // 中间教师输出
        out_fp = FP32_Teacher(x)             // FP 教师输出
        
        // 损失计算
        L_rec_int = ||out_student - out_int8||^2   // L2 重建损失
        L_feat_int = MSE(feat_student, feat_int8)  // 特征匹配
        L_INT = L_rec_int + λ * L_feat_int         // λ=5
        
        L_FP = L_rec_FP + λ * L_feat_FP
        
        α(t) = min(1, t/T_warmup)           // 线性增长权重
        L = (L_INT + α(t)*L_FP) / (1+α(t))
        
        // 反向传播（STE 梯度）
        L.backward()
        optimizer.step()
    ```
  - 量化公式（fake quantization）：
    ```
    x_clip = clamp(x, lb, ub) = min(max(x, lb), ub)
    Δ = (ub - lb) / (2^N - 1)
    x_int = round((x_clip - lb) / Δ)
    x̂ = x_int * Δ + lb
    // 梯度通过 STE: ∂x̂/∂x = 1 if x∈[lb,ub] else 0
    ```
  - 消融实验效果（STVSR 2-bit 设置）：Baseline（无任何核心模块）12.67dB → +Per-Frame Quantization 19.64dB → +BMFQ 27.56dB → +PMTD 30.33dB。

## PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PTQ4ARVG 是首个面向 ARVG（AutoRegressive Visual Generation）模型家族的 training-free PTQ 框架，包含三个核心组件：
    (1) **GPS (Gain-Projected Scaling)**：通过 Taylor 展开量化激活和权重的量化损失，定义 scaling gain（激活量化损失减少量减去权重量化损失增加量），对 gain 函数求导得到闭式最优 scaling factor：s_i = s_k · √(Σ_j |ΔW_{i,j}·x_i|) / √(Σ_j |W_{i,j}·Δx_i|)，其中 s_k 为最大激活 range 通道的 scaling factor，由 s_k = √(R_x^k / R_W^k) 确定。GPS 是首个基于数学优化的量化 scaling 策略，无需训练。
    (2) **STWQ (Static Token-Wise Quantization)**：利用 ARVG 的固定 token 长度和跨样本位置不变分布两大特性，离线分配 per-token 量化参数。对 AdaLN 模块沿 token 序列分配量化参数；对线性层的 sink token（首 token）和 normal token 分别分配量化参数。使用百分位数（percentile）校准确保精度，无在线校准开销。
    (3) **DGC (Distribution-Guided Calibration)**：基于 Mahalanobis 距离 ρ(x) = √((x-u)^T S^{-1} (x-u)) 评估每个样本对分布熵的贡献，选择 top 50% 高熵样本作为校准集，消除样本间分布不匹配导致的量化参数校准偏差。
  - 实验比较：PTQ4ARVG vs SmoothQuant、OS+、RepQ*、OmniQuant（training-based）、QuaRot（rotation-based）、SVDQuant（low-rank decomposition），在 W8A8 和 W6A6 两种位宽下，对 VAR（d16/d20/d24/d30）、RAR（B/L/XL/XXL）、PAR（XL-4×/XXL-4×/3B-4×/3B-16×）、MAR（B/L/H）四个 ARVG 模型家族进行量化，比较 FID、sFID、IS、Precision 四个图像生成质量指标。额外消融验证 GPS/STWQ/DGC 各自贡献，对比 GPS vs 其他 scaling 方法（SmoothQuant, RepQ*, OS+, SQ+RepQ*），对比 STWQ vs DTWQ（dynamic token-wise quantization）的精度和 speedup，对比 DGC vs random/uniform 采样的校准效果。W4A8 实验展示更低 bit-width 下的优势。在 RTX 3090 上部署 8-bit 量化模型评估真实加速比和内存节省。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX 3090 用于实际部署评估（测量推理延迟、峰值内存和加速比）。校准和量化过程在 GPU 上完成（基于 baseline 对比中 OmniQuant 需 A100-80G 的说明，推断使用高性能 GPU 进行量化实验）。
  - 校准集：128 张 ImageNet 图像（DGC 从更大校准池中选择 top 50% 高熵样本）。
  - 量化配置：W8A8 和 W6A6，所有线性层和矩阵乘法均量化（含 KV cache）。权重采用 channel-wise 非对称量化，激活采用 layer-wise 非对称量化（经 STWQ 后为 per-token static）。scaling factor 离线融合到网络权重中，推理时零额外开销。
  - 评估套件：ADM's TensorFlow evaluation suite *guided-diffusion* 计算 FID/sFID/IS/Precision。生成 50K 张 ImageNet 图像评估质量。
  - 实际部署测试：标准 CUDA kernel 部署 decoder 网络，batch size=100，不同 token 序列长度（64/128/256/512/1024）下测试延迟和峰值内存。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：VAR（d16, d20, d24, d30，最大 2B 参数）、RAR（B, L, XL, XXL，最大 1.5B 参数）、PAR（XL-4×, XXL-4×, 3B-4×, 3B-16×，最大 3B 参数）、MAR（B, L, H，最大 1B 参数）。所有预训练模型来自各官方仓库。
  - **数据集**：ImageNet（ILSVRC 2012, Deng et al. 2009），用于生成 50K 图像评估质量，128 张图像用于校准。
  - **Benchmark/指标**：FID↓（Fréchet Inception Distance）、sFID↓、IS↑（Inception Score）、Precision↑。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/BienLuky/PTQ4ARVG
  - PTQ4ARVG 量化流程（以 RAR-B 单层线性层为例，W6A6）：
    ```
    输入: 预训练 ARVG 模型, 校准数据 X_cal (128 张 ImageNet)
    输出: W6A6 量化模型

    === Step 1: DGC - 分布引导校准集选择 ===
    计算校准池中所有样本的均值 u 和协方差 S
    for each sample x in calibration pool:
        ρ(x) = sqrt((x-u)^T · S^{-1} · (x-u))  // Mahalanobis 距离
    选择 ρ(x) 最大的 top 50% 样本作为校准集 X_DGC

    === Step 2: GPS - 增益投影缩放 ===
    for each linear layer (qkv, fc1) in ARVG blocks:
        // 量化当前权重和激活，计算量化误差
        X_q = Q(X), W_q = Q(W)
        ΔX = X - X_q, ΔW = W - W_q

        // 搜索激活 range 最大的通道 k
        R_x = max(X, dim=0) - min(X, dim=0)   // per-channel activation range
        k = argmax(R_x)

        // 计算 s_k：使该通道激活和权重 range 对齐
        s_k = sqrt(R_x^k / R_W^k)

        // 计算剩余通道的 scaling factors (闭式解, Eq. 16)
        for i = 1 to n (n = input channels):
            if i != k:
                s_i = s_k * sqrt(Σ_{j=1}^m |ΔW_{i,j} · x_i|) /
                           sqrt(Σ_{j=1}^m |W_{i,j} · Δx_i|)

        // 应用等效缩放 (Eq. 2)，缩放因子离线融合到权重
        X' = X ⊘ s        // activation 逐通道除以 s
        W' = s ⊙ W        // weight 逐通道乘以 s

    === Step 3: STWQ - 静态逐 token 量化 ===
    for each block in decoder:
        // (a) AdaLN 模块：沿 token 序列分配 per-token 量化参数
        for t = 1 to T (fixed token length):
            δ_AdaLN[t] = PercentileCalib(X_AdaLN[:, t, :])

        // (b) 线性层 (qkv, fc1, etc.)：sink token 与 normal token 分别量化
        for each linear layer input X_lin ∈ R^{T×n}:
            X_sink = X_lin[0, :]               // 首 token (sink, 含条件信息)
            X_normal = X_lin[1:, :]            // 其余 token
            δ_sink = PercentileCalib(X_sink)
            δ_normal = PercentileCalib(X_normal)

    === Step 4: 量化推理 (无在线校准开销) ===
    for each inference step (token generation):
        // 使用 Step 3 中预设的静态 per-token 量化参数 δ
        X_int = clamp(round(X / δ) + z, 0, 2^b - 1)
        // INT 矩阵乘法 + 反量化
        Y = dequant(X_int · W_int, δ_x, δ_w)
    ```
  - GPS 数学推导核心（Taylor 展开 + 求导得闭式解）：
    - 量化损失分解（Eq. 3）：E(x,W) ≤ E_x + E_W = E[L(x̂,W) − L(x,W)] + E[L(x,Ŵ) − L(x,W)]
    - Taylor 展开（Eq. 4-6）：E_W ≈ ½·ΔW^T·H^(W)·ΔW，E_x ≈ ½·Δx^T·H^(x)·Δx，用 MSE 近似 Hessian
    - 引入 scaling 后（Eq. 11-12）：E'_x < E_x（激活量化损失降低），E'_W > E_W（权重量化损失增加）
    - 缩放增益函数（Eq. 13）：g(s₂) = g_x − g_W ∝ s₂²/s₁²·W²Δx² + s₁²/s₂²·ΔW²x²
    - 对 s₂ 求导得闭式解（Eq. 14-16）：∂g/∂s₂ = 0 ⇒ s₂ = s₁·√(Σ|ΔW·x|)/√(Σ|W·Δx|)
  - 推理加速实测（8-bit RAR-L, RTX 3090, batch=100, seq_len=256）：PTQ4ARVG 达 2.87× speedup（FP: 3722ms → Quant: 1297ms），峰值内存从 4241MB 降至 2186MB（1.94× 压缩）。对比 QuaRot 因在线 Hadamard 旋转导致 0.70× slowdown（6062ms vs FP 3722ms）。

## Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Q-VDiT 是一种面向视频生成 Diffusion Transformer (V-DiT) 的后训练量化（PTQ）框架。核心包含两个组件：(1) **Token-aware Quantization Estimator (TQE)**——利用量化误差比原始权重具有更低信息熵的理论（Theorem 3.2，H(Δ)≤H(W)），使用两组低维向量参数 α∈R^{d_in} 和 β∈R^{d_out} 进行低秩误差估计（rank=1），从 token 维度和 feature 维度正交地补偿量化误差。修订后的前向计算：X·W^T ≈ Q̂(X)·Q̂(W)^T + Δ̂·β，其中 Δ̂_{[f_i+1:f_i+s,:]} = (M_i ⊙ Q̂(X)_{[f_i+1:f_i+s,:]})·α，M∈R^t 是 token-aware 缩放因子，对每帧 token 进行选择性缩放，M_i = η_i/ω_i 由量化误差权重 η_i 和显著度量 ω_i 初始化（Eq. 9）；(2) **Temporal Maintenance Distillation (TMD)**——构建 FP 模型中帧间相似度分布 D^{FP}_i = softmax([T^{FP}_{i,1},...,T^{FP}_{i,t}]) 作为先验，用 KL 散度对齐量化模型的帧间时序分布 D^{Q}_i，使每个 frame 的优化考虑整体视频特征分布，弥补 MSE loss 无法捕捉帧间信息的问题。总损失 L_total = L_task + γ·L_temporal，γ=100。
  - 实验比较（VBench benchmark，Tab. 1）：Q-VDiT (W4A6/W3A8/W3A6) vs Q-DiT、PTQ4DiT、SmoothQuant、Quarot、EfficientDM、SVDQuant、ViDiT-Q。8 个维度：Imaging Quality、Aesthetic Quality、Motion Smoothness、Dynamic Degree、Background Consistency、Subject Consistency、Scene Consistency、Overall Consistency。Q-VDiT 在 W3A6 下 Scene Consistency 达 23.40，超过 SOTA（ViDiT-Q 11.99 / EfficientDM 12.04）近 1.9×。W3A8 下 Overall Consistency 达 22.39 vs SOTA 18.53。
  - 实验比较（OpenSORA prompt set 多指标评估，Tab. 2）：CLIPSIM、CLIP-Temp、VQA-Aesthetic、VQA-Technical、ΔFLOW Score、Warping Error。W4A6 下几乎无损（VQA-Aesthetic 67.05 vs FP 66.91）。W3A6 下 VQA-Technical 从 SOTA 29.58 提升到 59.10，CLIPSIM 从 0.1768 提升到 0.1785。
  - 实验比较（Higher bits, Tab. 3）：W8A8/W6A6/W4A8 vs Q-Diffusion、Q-DiT、PTQ4DiT、SmoothQuant、Quarot、ViDiT-Q。W4A8 下 VQA-Aesthetic 达 71.32，甚至超过 FP 模型（66.91）。
  - 实验比较（Latte model on UCF-101, Tab. 6）：Naive-PTQ、ViDiT-Q vs Q-VDiT。指标：FVD、FVD-FP16、CLIPSIM、CLIP-Temp、VQA-Aesthetic、VQA-Technical、ΔFLOW Score、Temporal Flickering。W3A6 下 VQA-Aesthetic 达 19.79 vs ViDiT-Q 16.83。
  - 消融实验（Tab. 4）：PTQ4DiT baseline → +TQE(w/o M) → +TQE(w M) → +TMD → 完整 Q-VDiT，各组件均正向贡献。γ 超参数不敏感（Fig. 6, γ ∈ {0.1,1,10,100,500,1000} 均显著优于 PTQ4DiT baseline）。
  - 定性对比（Fig. 5, Fig. 7-15）：W3A6 下其他方法无法生成有意义图像，Q-VDiT 仍能生成清晰图像且帧间有显著连贯运动变化。

- 硬件平台是什么，配置是什么。
  - GPU 显存校准阶段约 16600-19460 MB（Tab. 5），具体 GPU 型号论文未明确说明。W8A8 校准耗时约 12.5-12.9 小时。推理效率（Tab. 7）：W4A8 下相比 FP16 达到 2.40× 显存节省和 1.35× 推理加速。使用 LoRunner Kernel（来自 SVDQuant）将 TQE 低秩分支（rank=1）与量化 kernel 融合，额外延迟仅约 5%。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Open-SORA (HPC-AI, 2024) —— 基于 PixArt-α 架构的视频 DiT；Latte (Ma et al., 2024) —— class-conditioned 视频 DiT（UCF-101 训练）。仅量化线性层权重（channel-wise quantization），激活使用动态 token-wise quantization。位宽：W4A6、W3A8、W3A6（主打 hard settings），以及 W8A8、W6A6、W4A8（higher bit settings）。
  - **校准数据集**：Open-Sora 提供的 10 个 prompt，均匀选择 50 个去噪步。校准迭代：6-8 bit 为 5k iters，4-bit 为 10k iters，3-bit 为 15k iters；batch size=4；学习率 lr=1e-6（权重量化参数），lr=1e-5（TQE 参数）。
  - **Benchmark**：VBench（8 维度综合视频生成评估）；OpenSORA prompt set（CLIPSIM、CLIP-Temp、DOVER VQA-Aesthetic/Technical、ΔFLOW Score、Warping Error）；UCF-101（FVD、FVD-FP16、Temporal Flickering）。
  - **推理设置**：Open-SORA 用 100-step DDIM，CFG scale 4.0；Latte 用 20-step DDIM solver，CFG scale 7.0。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/wlfeng0509/Q-VDiT（MIT License，ICML 2025），基于 Open-Sora v1.0 和 ViDiT-Q，含 calibrate 和 inference 脚本。
  - Q-VDiT 量化 pipeline（以 Open-SORA 单层 Linear 为例，W4A6）：
    ```
    # === 输入: FP 模型权重 W ∈ R^{d_out×d_in}, 视频 latent Z ∈ R^{n×d}, n=s×t ===

    # === Step 1: TQE 参数初始化 (Eq. 8-9) ===
    for each linear layer with weight W:
        α ∈ R^{d_in}: Kaiming normal init
        β ∈ R^{d_out}: zero init (LoRA convention)
        M ∈ R^{t}: token-aware scaling init
          for frame i in [0..t-1]:
            η_i = exp[1-ρ(X_i, Q̂(X)_i)] / Σ_v exp[1-ρ(X_v, Q̂(X)_v)]
            ω_i = Σ_τ |X_{i,τ}| / Σ_v Σ_τ |X_{v,τ}|
            M_i = η_i / ω_i

    # === Step 2: 量化前向传播 (TQE, Eq. 8) ===
    for each linear layer:
        Q_X = quantize(X, s_x, z_x)    # token-wise dynamic quantization
        Q_W = quantize(W, s_w, z_w)    # channel-wise quantization

        # TQE 误差补偿: r=1 低秩近似
        for frame i in [0..t-1]:
            f_i = i * s
            Δ̂[f_i:f_i+s, :] = (M_i ⊙ Q_X[f_i:f_i+s, :]) @ α   # ∈ R^{s×1}

        Y = Q_X @ Q_W^T + Δ̂ @ β^T                              # ∈ R^{n×d_out}

    # === Step 3: TMD 损失计算 (Eq. 13-15) ===
    # S^{FP}, S^{Q} ∈ R^{n×d}, n=s×t
    for frame i, j in [0..t-1]:
        T_fp[i,j] = cosine_sim(S_fp_i, S_fp_j)   # ρ(·,·)
        T_q[i,j]  = cosine_sim(S_q_i, S_q_j)
    for frame i in [0..t-1]:
        D_fp_i = softmax([T_fp[i,0], ..., T_fp[i,t-1]])  # ∈ R^t
        D_q_i  = softmax([T_q[i,0],  ..., T_q[i,t-1]])
    L_temporal = Σ_i KL(D_fp_i || D_q_i)

    # === Step 4: 校准训练 ===
    for iter in 1..total_iters:
        Y_q = quantized_forward(X_calib)   # TQE 修正前向
        Y_fp = fp_forward(X_calib)         # 全精度参考

        L_task = ||Y_q - Y_fp||²           # Eq. (11)
        L_temporal = compute_TMD(Y_q, Y_fp) # Eq. (15)
        L_total = L_task + 100 * L_temporal # γ=100

        # 更新 TQE (α,β,M) 和量化参数 (s,z)
        optimizer.step(L_total)

    # === Step 5: 推理 ===
    # LoRunner Kernel 融合: TQE branch (r=1) + quant GEMM
    # Y = fused_quant_linear(X)  # 额外延迟<5%
    ```
  - 关键理论（Theorem 3.2）：量化误差 Δ = W − Q̂(W) 的信息熵 H(Δ) ≤ H(W)，因为 round-to-nearest 舍入的 decimal truncation 是 surjection (Lemma A.1)。因此 Δ 可在 rank=1 空间估计，参数从 d_out×d_in 降到 d_out+d_in。
  - TMD 梯度分析（Eq. 16-18）：∂L_temporal/∂S^{Q}_i 包含 Σ_j [·T^{Q}_{i,j} + ·T^{Q}_{j,i}] 双向项，且 ∂L_temporal/∂T^{Q}_{i,j} = Σ_k D^{FP}_{i,k}·D^{Q}_{i,j} − D^{FP}_{i,j}（Eq. 17），确保任意帧对的相关性受所有帧共同影响。

## QTIP: Quantization with Trellises and Incoherence Processing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QTIP 是一种 weight-only 后训练量化（PTQ）方法，使用 **Trellis Coded Quantization (TCQ)** 替代 Vector Quantization (VQ) 实现超高维量化（有效维度 256，VQ 被限制在 ≤8 维）。核心设计包含三部分：(1) **Bitshift Trellis**——硬件友好的 trellis 结构，节点 i→j 有边当且仅当 j = (i·2^{kV} mod 2^L) + c，第 t 组权重仅依赖连续 L-bit 窗口，支持并行解码且无需存储 trellis 图结构；(2) **Compute-based Random Gaussian Codes**——三种免查找表或混合计算码：**1MAD**（MAD+LCG+4×8-bit 求和，2 指令生成近似高斯值）、**3INST**（LCG+XOR FP16 magic number 尾数/指数位，3 ALU 指令，m1+m2 近似高斯）、**HYB**（hash+2^Q×2 LUT 2D codebook，2 指令/权重摊销）；(3) **Incoherence Processing**——Random Hadamard Transform (RHT) 使权重近似 i.i.d. 高斯分布，匹配 TCQ 的源编码假设。QTIP 作为 QuIP# BlockLDLQ 框架的 drop-in 替换量化器，将 Tx×Ty 权重块作为高维序列用 Viterbi 算法（O(2^L T) 时间）量化。Algorithm 4 的近似 tail-biting trellis 使编码比特数与硬件字长对齐。
  - 实验比较：(1) QTIP 纯计算码 (1MAD/3INST, L=16, V=1, Tx=Ty=16) vs QuIP#、AQLM——Llama 2 (7B/13B/70B) 上 2/3/4-bit perplexity（Table 3），无 fine-tuning 即超越含 fine-tuning 的 QuIP# 和 AQLM；(2) QTIP HYB 混合码 (L=16, V=2, Q=9, 2KiB codebook) vs QuIP#、AQLM、GPTVQ-2D、PV-Tuning——Llama 1/2/3/3.1/3.2 全系列 perplexity 和 zeroshot 准确率；(3) 消融：L (8/10/12/16) vs MSE（Table 10）、V (1/2/4) vs MSE（Table 11）、tail-biting 近似 vs 最优解（Table 2）；(4) i.i.d. 高斯源量化失真（Table 1）：Lloyd-Max SQ 0.118 vs QuIP# E8P VQ 0.089 vs QTIP 256D TCQ 0.069 vs D_R 0.063。

- 硬件平台是什么，配置是什么。
  - GPU：RTX 6000 Ada (960GB/s 显存带宽) 用于推理吞吐评测；RTX 3090、RTX A6000 Ampere 用于跨平台解码速度（Table 17）。Together AI 提供计算资源。PyTorch + HuggingFace Transformers。解码利用 16×16 MMA tile 进行矩阵乘法。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama 1 (7B/13B/30B/65B)、Llama 2 (7B/13B/70B)、Llama 3 (8B/70B)、Llama 3.1 Instruct (8B/70B/405B)、Llama 3.2 Instruct (1B/3B)。
  - 校准数据：RedPajama（Hessian 生成：Llama 1/2 用 6144 seq × 2048 tokens，Llama 3 用 4096 seq × 8192 tokens，405B 用 2048 seq × 8192 tokens）。
  - 评估：Wikitext2、C4（perplexity, OPTQ 方式）；LM Eval Harness（zeroshot: ARC-C, ARC-E, BoolQ, PiQA, WinoGrande, HellaSwag）；推理吞吐 (tokens/s, batch_size=1 decode)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Cornell-RelaxML/qtip
  - 依赖：PyTorch, HuggingFace Transformers, QuIP# BlockLDLQ, fast-hadamard-transform, Neil Sloane Hadamard 矩阵, RedPajama。
  - 算法核心（QTIP + BlockLDLQ, Algorithm 5 伪代码）：
    ```
    输入: W ∈ R^{m×n}, H ∈ R^{n×n}, T_x, T_y, L, k, V, code C.
    Ŵ ← 0_{m,n}
    LDL^T ← T_y-block LDL decomposition of H
    A ← L - I
    for j = n/T_y-1 down to 0:
      x ← W_{:,jT_y:(j+1)T_y} + (W_{:,jT_y:} - Ŵ_{:,jT_y:}) A_{jT_y:(j+1)T_y}
      x ← x.reshape(m/T_x, T_x T_y)
      x̂ ← Viterbi(x, (L,k,V) bitshift trellis, C)  # 逐行 TCQ
      Ŵ_{:,jT_y:(j+1)T_y} ← x̂.reshape(m, T_y)
    输出: Ŵ
    ```
  - Viterbi 量化：在 bitshift trellis G 上最小化 Σ ||C_{x_i} - s_i||²，动态规划 V_t(y) = min_{(x,y)∈G} V_{t-1}(x) + ||C_y - s_t||²，O(2^L T) 时间。
  - 1MAD 码：x ← (ax+b) mod 2^32 (LCG) → x ← sum of four 8-bit unsigned ints → (x-510)/147.8 → 近似 N(0,1)。2 inst: MAD + vabsdiff4。
  - 3INST 码：x ← (ax+b) mod 2^32 → XOR bottom 16 bits with magic FP16 m's mantissa/exp/sign → XOR top 16 bits → m1+m2 → 近似高斯。3 inst: MAD + lop3 + add。
  - HYB 码：x ← x²+x mod 2^32 (hash) → idx = (x>>(15-Q)) & (2^Q-1) → v = C[idx] (2^Q×2 LUT) → sign-flip v[1] via XOR bit 15。摊销 2 inst/weight。C 可 fine-tune。
  - 关键：QTIP HYB codebook 仅 2KiB (2^9×2)，比 AQLM 的 1MiB 小 512×，可放入 L1 cache。解码仅需 ≤4 GPU 指令/权重，达到 >80% 峰值显存带宽。

## QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuEST 是一种面向扩散模型低比特量化的参数高效选择性微调方法。核心 pipeline：(1) 发现扩散模型中激活值分布不均衡（数值大量集中在零附近，但有稀疏的大值对生成质量重要），导致低比特量化失败；(2) 提出通过权重微调来调整激活分布，使其更向量化友好——理论证明（Theorem 3.2）通过将大量化扰动Δ分解为K个小扰动ε=Δ/K，说明微调权重可使模型对量化扰动更鲁棒；(3) 识别两类关键层：时间嵌入层（Property ❶：时序信息对量化至关重要）和注意力相关层（Property ❷：FeedForward等层对位宽降低特别敏感，6比特即失败）；(4) 采用数据无关方式构建校准集（128-256样本/时间步，从随机高斯噪声xT采样）；(5) 选择性渐进微调：先 TLA（Temporal Layer Alignment）微调时间嵌入层权值w^l和激活量化参数s^l，再 CMA（Critical Module Alignment）微调注意力相关层权值和对所有未更新的量化参数s进行优化，最后用全局损失 L_G（量化模型与全精度模型最终输出MSE）监督；(6) 总损失：argmin(L_TLA + L_CMA + 2L_G)，仅微调不足7%的参数；(7) 使用单组量化参数覆盖所有时间步，无需按时间步分别量化。前向伪代码：t → TimeEmbed(t, w_TE_finetuned) → 各层注入 → Attention(Q,K,V with w_finetuned) → FFN(sensitive, activations cautiously quantized) → 计算 L_TLA (TE层输出MSE) + L_CMA (注意力层输出MSE) + L_G (最终输出MSE) → Adam更新部分w和s。量化函数：x̂ = clamp(round(x/s) + Z; qmin, qmax), x̃ = (x̂ - Z)*s。
  - 实验比较：(a) W8A8/W4A8/W4A4 下 QuEST vs PTQ4DM、Q-Diffusion、PTQ-D、EfficientDM 在 LSUN-Bedrooms（LDM-4）和 LSUN-Churches（LDM-8）上的 FID/sFID；(b) ImageNet 256×256 上 FID/sFID/IS 对比；(c) Stable Diffusion v1.4 文本到图像生成的 CLIP Score 对比；(d) 与 TFMQ-DM 对比；(e) CIFAR10 低分辨率对比；(f) 消融：TLA vs TLA+CMA vs TLA+CMA+LG 组件贡献；(g) 全局损失消融 w/ vs w/o L_G；(h) 效率对比 QuEST vs EfficientDM vs Full-finetune (时间/显存/迭代数/FID)；(i) LoRA 集成消融；(j) 预计算时间嵌入对比。

- 硬件平台是什么，配置是什么。
  - NVIDIA A6000 GPU（48GB）；Stable Diffusion 实验在单卡完成。环境：Python，PyTorch，CUDA，Linux。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LDM-4（LSUN-Bedrooms 256×256）、LDM-8（LSUN-Churches 256×256）、LDM-4（ImageNet 256×256）、Stable Diffusion v1.4（512×512）
  - 数据集：LSUN-Bedrooms、LSUN-Churches、ImageNet（条件生成）、COCO2014 prompts（文本到图像，10000 条验证集）
  - Benchmarks/Metrics：FID（Fréchet Inception Distance）、sFID（spatial FID）、IS（Inception Score）、CLIP Score（ViT-B/16 backbone）
  - Samplers：DDIM（LDMs，20/200/500 步），PLMS（Stable Diffusion，50 步）
  - 评估：50000 张采样图像，官方评估脚本
  - 超参数：Adam optimizer, lr_w=1e-5（权重微调）, lr_s=1e-4（量化参数微调），校准集 256 样本/时间步（Stable Diffusion 128 样本/时间步）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/hatchetProject/QuEST
  - 张量计算流程：(1) 输入 x_T ∼ N(0,I) 采样高斯噪声，通过全精度模型前向获取中间激活作为校准目标；(2) 量化：W̃ = clamp(round(W/s_w) + Z_w; qmin, qmax)，x̃_l = clamp(round(x_l/s_l^a) + Z_l; qmin, qmax)；(3) TLA 微调：for l in C_TE: O_TE(l) = FP_model(t; w_l), Õ_TE(l) = Q_model(t; w̃_l, s_l) → loss = MSE(O_TE, Õ_TE)，反向传播更新 w_l 和 s_l；(4) CMA 微调：for l in C_A: O_attn(l) = FP_model(z_l; w_l), Õ_attn(l) = Q_model(z̃_l; w̃_l, ŝ) → loss = MSE(O_attn, Õ_attn)，更新 w_l 和 未在 TLA 中更新的 s；(5) 全局损失：L_G = MSE(FP_model(x_t; w), Q_model(x_t; w̃, s))；(6) 最终优化：argmin_{w_l} (L_TLA + L_CMA + 2L_G), l ∈ C_TE ∪ C_A。

## QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuIP# 是一种 weight-only 后训练量化（PTQ）方法，由三大技术组件构成：(1) **Randomized Hadamard Transform (RHT) 非相干处理**：用随机符号向量 S_U, S_V 和 Hadamard 矩阵对权重矩阵 W 和 Hessian H 做双边共轭变换——Ŵ = Had(S_U · Had(S_V · W^T)^T)，Ĥ = Had(S_V · Had(S_V · H)^T)——使权重趋于高斯分布（亚高斯），消除离群值，实现 μ-incoherent 保证，复杂度 O(n log n)（相比 QuIP Kronecker 方法的 O(n√n)）；(2) **BlockLDLQ + E8P 格基码书向量量化**：基于 g-block LDL 分解 H = L^T D L，对非相干化后的权重矩阵按 g=8 列分块自适应舍入——Ŵ_k = Q(W_k + (W_{:(k-1)} - Ŵ_{:(k-1)})A_k)——其中 Q 为 2-bit E8P 向量量化器。E8P 基于 E8 格（8 维最高密度球填充，kissing number 最优），通过符号翻转对称性将 2^16 条码字压缩为 2^8 条源码书（仅 1KiB），支持快速推理。高比特下使用 Residual Vector Quantization (RVQ) 逐残差量化（如 4-bit = 2×2-bit E8P, 3-bit = 2-bit E8P + 1-bit E8）；(3) **层间微调**：先在各 Transformer Block 内微调未量化层补偿已量化层（MSE loss），再端到端微调所有未量化参数（layernorms、S_U、S_V、LM head），优化 CrossEntropy loss。sign vectors 以 FP16 存储。<br>算法伪代码：QuIP#(W, H) → IP-RHT: Ŵ,Ĥ,S_U,S_V ← Had(S_U·Had(S_V·W^T)^T) → BlockLDLQ: Ŵ ← Q_blocks(W+(W-Ŵ)(L^T-I)) using E8P codebook → FineTune: Adam optimize S_U,S_V,layernorms per block then end-to-end.<br>推理伪代码（Algorithm 2）：y ← Had(S_V ⊙ x) → y ← decompress_multiply(Ŵ, C, y) → y ← Had(S_U ⊙ y) → output y，其中 decompress_multiply 用 E8P CUDA kernel 从压缩码书解码权重并与激活做 MMA。
  - 实验比较：(a) Llama 1 (7B/13B/30B/65B) 和 Llama 2 (7B/13B/70B) 在 2/3/4 bit 下与 OmniQuant、AWQ、QuIP、AQLM 的 Wikitext2 和 C4 困惑度对比；(b) Llama 2 在 ARC-C/ARC-E/PIQA/WinoGrande 上的 Zeroshot 精度对比；(c) Llama 1/2 的 bit scaling 行为（3-bit 超越理论无损 4-bit 线）；(d) 消融实验：RHT vs Kronecker（QuIP vs QuIP# no FT & no E8）、E8P vs 半整数格 vs D4 格 vs K-Means、有/无微调；(e) Mixtral 8x7B（MoE）和 Falcon 180B 上的泛化性验证；(f) 推理吞吐：RTX 4090 上 2/4-bit Llama 模型生成速度（tok/s）及峰值显存带宽利用率，与 AQLM/FP16 对比。

- 硬件平台是什么，配置是什么。
  - 量化实验：NVIDIA A100 GPU（多卡节点），Llama 2 70B 无微调 <10 GPU-hours，含微调约 100 GPU-hours（不含 Hessian 生成）；Hessian 生成：RedPajama 1T 数据集 6144 条序列 × 模型原生上下文长度（Llama 1=2048, Llama 2=4096）。
  - 推理性能测试：NVIDIA RTX 4090（1TB/s 峰值显存带宽），FlashAttention 库 Llama 实现；A6000 上 QuIP# 吞吐约为 QuIP 的 2 倍。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama 1 (7B, 13B, 30B, 65B)、Llama 2 (7B, 13B, 70B)、Mixtral 8x7B (MoE)、Falcon 180B
  - 数据集/Benchmark：(a) 困惑度：Wikitext2、C4（OPTQ 采样函数）；(b) Zeroshot：LM Eval Harness（Gao et al., 2023）评测 ARC-Challenge、ARC-Easy、PIQA、WinoGrande、BoolQ；(c) 校准/微调数据：RedPajama 1T 数据集，256 训练序列 + 128 验证序列；
  - 量化位宽：2-bit (E8P), 3-bit (E8P 2-bit + E8 1-bit RVQ), 4-bit (2× E8P 2-bit RVQ)
  - 超参数：E8P scale ρ=0.9；微调：Adam optimizer, lr=5×10^-5（权重）/ 5×10^-4（sign vectors for 2-bit），batch size 8（block内）/ 1（端到端），5 epochs（160 steps），序列长度 = 模型原生上下文（70B 端到端用 3072 避免 OOM）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Cornell-RelaxML/quip-sharp；预量化模型：https://huggingface.co/relaxml
  - 张量计算流程（以 Llama 2 7B 2-bit QuIP# 为例）：
    1. **Hessian 生成**：从 RedPajama 采样 6144 条序列，对每个线性层计算代理 Hessian H = E_x[xx^T]。
    2. **IP-RHT 非相干处理**（Algorithm 3）：采样随机符号向量 S_V ∼ U{±1}^n, S_U ∼ U{±1}^m → Ŵ ← Had(diag(S_U) · Had(diag(S_V) · W^T)^T)，其中 Had 为 Fast Walsh-Hadamard Transform（O(n log n)，仅 ±1 运算无浮点乘法）→ Ĥ ← Had(diag(S_V) · Had(diag(S_V) · H)^T)。输出 Ŵ, Ĥ, S_U, S_V。对非 2 的幂次维度：分解 n = p × q（p 为最大 2 的幂次，q 已知 Hadamard 矩阵存在），使用 V = H_p ⊗ H_q。
    3. **BlockLDLQ 自适应舍入**（Section 4.1）：对 Ĥ 做 g-block LDL 分解 Ĥ = L^T D L → 设置 U = L^T - I → 按 g=8 列分块迭代舍入 Ŵ_k = Q_E8P(Ŵ_k + (Ŵ_{:(k-1)} - Ŵ̂_{:(k-1)}) · A_k)，其中 A_k 为 U 的第 k 个 8 列块，Q_E8P 为 E8P 2-bit 向量量化。
    4. **E8P 解码**（Section 4.2）：每个 16-bit 码字编码一个 8 维向量——8 bits 查源码书 S（256 个 |D̂_8| 绝对值条目），7 bits 控制 7 个符号翻转（第 8 个符号由奇偶性推断），1 bit 控制 ±1/4 偏移。解码：c = S[code[0:8]] → 符号翻转 parity 恢复 → v ∈ E8 + 1/4。
    5. **RVQ 高比特扩展**（Section 4.3）：4-bit 量化 = 两次 2-bit E8P 残差量化——δ_1 = Q_E8P(Ŵ) · s_1, δ_2 = Q_E8P((Ŵ - δ_1)/s_2) · s_2, Ŵ̂ = δ_1 + δ_2。
    6. **层间微调**（Algorithm 5）：对每个 Decoder Block D ∈ M：Y ← D(X) → 对每层 L ∈ D 按顺序量化 → 冻结 L 的量化权重 → Adam 优化 D 以最小化 MSE(D(X_train), Y_train) → X ← Y。全部 Block 完成后，端到端微调剩余参数（layernorms, S_U, S_V, LM head），最小化 CrossEntropy(M(D_train), C_train)，使用验证集早停。
    7. **推理**（Algorithm 2）：输入激活 x → y ← Had(S_V ⊙ x)（FWHT）→ y ← E8P_decode_matvec(Ŵ, C, y)（CUDA kernel, MMA Tensor Core 指令）→ y ← Had(S_U ⊙ y) → 输出。

## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuaRot 通过随机 Hadamard 变换消除 LLM 激活值中的离群值 (outlier)，实现端到端 4-bit 量化（权重、激活值、KV cache）。核心 pipeline 分为两阶段：(1) **Stage 1 权重修改**：利用计算不变性（computational invariance）将随机 Hadamard 矩阵 Q 融入权重矩阵，消除跨层激活值离群值；在 FFN 的 down-projection 前和注意力模块内部插入在线 Hadamard 变换，消除层内激活值离群值；(2) **Stage 2 量化**：用 GPTQ 或 RTN 对权重进行 per-column 对称量化（INT4），激活值在线 per-token 对称量化（INT4），KV cache 用 asymmetric group-wise 量化（group size=128）。最终所有矩阵乘法均在 INT4 下完成，无需保留任何高精度通道或 outlier feature。伪代码核心流程：(a) 生成随机 Hadamard Q = H_d diag(s), s∈{±1}^d → (b) 离线权重修改：W_gate ← Q^T diag(α) W_gate, W_up ← Q^T diag(α) W_up, W_down ← H W_down Q → (c) 在线推理：X_norm = RMSNorm(X) → X_q = round(clip(X_norm/s_x, -7, 7)) → Y_int = CUTLASS_INT4_GEMM(X_q, W_q) → Y = dequant(Y_int, s_x, s_w) → (仅 W_down 前) X_h = Hadamard(X) → X_hq = quant(X_h) → Y_int = GEMM(X_hq, W_down_q) → Y = dequant(...) → YQ（旋转后的输出）。
  - 实验比较：(a) 4-bit 量化后的 WikiText-2 困惑度 vs SmoothQuant, OmniQuant, QUIK-4B, Atom-128G（Table 1）；(b) 零样本任务精度（PIQA, WinoGrande, HellaSwag, Arc-Easy, Arc-Challenge, LAMBADA）vs FP16 baseline（Table 2）；(c) RTN vs GPTQ 权重量化消融（Table 3）；(d) group-wise 量化不同 group size (64/128/256) 消融（Table 4）；(e) KV cache 不同 bit-width 组合消融 (2/3/4-bit for K and V)（Table 6）；(f) Random Orthogonal vs Hadamard 消融（Table 8）；(g) FP16 vs FP32 Hadamard 变换精度消融（Table 10）；(h) 4-bit weight-only 量化消融（Table 7）；(i) LLAMA-3 和 Phi-3-mini 模型扩展实验（Table 11-13）。

- 硬件平台是什么，配置是什么。
  - 量化准备（离线）：单张 NVIDIA A100 GPU。LLAMA2-70B 模型修改耗时 5 分钟，GPTQ 量化耗时 2 小时。校准集：WikiText-2 训练集 128 样本（sequence length=2048）。
  - 性能评估：NVIDIA RTX 3090 GPU。CUDA 12.1。PyTorch 框架 + Hugging Face Transformers。CUTLASS 库做 INT4 TensorCore GEMM。FlashInfer 库做量化 KV cache attention。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLAMA-2 家族（7B/13B/70B），LLAMA-3（8B/70B），Phi-3-mini-4k-instruct
  - 数据集：WikiText-2 训练集用于 GPTQ 校准（128 samples, seq=2048）
  - Benchmarks：WikiText-2 困惑度（语言生成质量）；六项零样本任务——PIQA, WinoGrande, HellaSwag, Arc-Easy, Arc-Challenge, LAMBADA（使用 LM Evaluation Harness 默认参数）
  - 量化配置：GPTQ（默认）或 RTN 权重量化 + per-token 对称激活量化（clipping ratio=0.9）+ asymmetric group-wise KV cache 量化（group size=128, clipping ratio=0.95）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/spcl/QuaRot
  - 算法 pipeline 张量计算流程（LLAMA-2 FFN 层 + Attention）：
    1. **离线权重修改（Stage 1）**：
       - 生成随机 Hadamard: Q = H_d diag(s), s_i ∈ {±1} 随机采样, H_d 为 Walsh-Hadamard 矩阵（O(d log d) 变换）
       - RMSNorm 吸收: α 融合到相邻权重 → W_gate = diag(α) W_gate, W_up = diag(α) W_up
       - 跨层旋转: W_gate ← Q^T W_gate, W_up ← Q^T W_up, W_down ← H W_down Q（H 为精确 Hadamard 矩阵）
       - 注意力 head-wise: W_v ← W_v (I⊗H_{d_h}), W_out ← H (I⊗H_{d_h}) W_out，其中 H = (I⊗H_{d_h})(H_{n_h}⊗I)
       - W_k, W_q 不做离线修改（因 RoPE 存在），改为在线 head-wise Hadamard 旋转
    2. **前向推理（Stage 2）**：
       - RMSNorm（无 scale, FP32）：x_norm = x / ||x||
       - 激活量化（per-token symmetric INT4）：s_x = max(|x_norm|, dim=row) × 0.9 / 7, x_q = round(clip(x_norm/s_x, -7, 7))
       - INT4 GEMM (W_gate/W_up): Y_int32 = x_q × W_q^T（TensorCore, INT32 accumulator）
       - Dequant: Y_fp16 = (Y_int32 ⊙ s_x^T ⊙ s_w) / scale_factor → cast to FP16
       - SiLU gate: Y_gate ⊙ σ(Y_up)（FP16 element-wise）
       - 在线 Hadamard（仅 W_down 前）：Y_h = Walsh-Hadamard(Y_fp16)（O(d log d), FP16 或 FP32）
       - 再次量化 + INT4 GEMM (W_down) + dequant → 旋转后输出 YQ
    3. **Attention 模块（量化 KV cache）**：
       - Q/K/V projection（INT4 GEMM, 同上）
       - Post-RoPE 在线 head-wise Hadamard：Q_h = Q (I⊗H_{d_h}), K_h = K (I⊗H_{d_h})（head-wise Walsh-Hadamard, O(d_h log d_h) per head）
       - KV cache 量化：K_q = round(clip((K_h - z)/s_k, 0, 15))（asymmetric group-wise, group=128）
       - Attention 计算：P = softmax(Q_h K_h^T / √d_h) → Y = P V_h（FlashInfer 实现，online softmax + 反量化）
       - 在线 Hadamard head 变换（out-projection 前）：Z_h = Z (H_{n_h}⊗I)（reshape + Walsh-Hadamard）
       - Out-projection: INT4 GEMM (W_out) + dequant
    4. 使用方法：`python quarot.py --model meta-llama/Llama-2-70b-hf` → Stage 1 权重融合 → GPTQ 量化 → 保存量化模型 → 推理加载

## QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QuantCache 是一个针对 Diffusion Transformers (DiTs) 视频生成的 training-free 联合优化推理加速框架，包含三层算法 pipeline：(1) **Hierarchical Latent Caching (HLC)**：基于 inter-step feature divergence 自适应决定缓存刷新策略。对每个 timestep t 和 layer l，计算 timestep-wise feature divergence score D_t^(l) = ||p_t^(l) - p_{t-k}^(l)||_1 / k · ||∇_t m_t^(l)||（p_t^(l) 为 layer l 在 timestep t 的激活，k 为上次缓存步，∇_t m_t^(l) 为帧间梯度），根据 D_t^(l) 与阈值 δ_1、δ_2 的关系分三档决定缓存刷新间隔 τ_t^(l) ∈ {τ_max, τ_mid, τ_min}；(2) **Adaptive Importance-Guided Quantization (AIGQ)**：权重量化方面，通过评估每层的 numerical error、perceptual distortion 和 temporal dynamics 计算 sensitivity，在总 bit-width 预算 B_total 约束下迭代分配 precision（Σ_l B(l) ≤ B_total），并引入 channel-balancing mechanism（scaling 修正静态 imbalance + rotation 修正动态 timestep 变化）减少量化 outlier；激活量化方面，提出 timestep-wise content-adaptive bit allocation function bit-width(t) ∈ {Bit_max, Bit_mid, Bit_min}，基于 timestep 冗余度 D_t 自适应调节激活精度；(3) **Structural Redundancy-Aware Pruning (SRAP)**：在线计算相邻层 feature 的 cosine similarity S_t^(l,l+1) = ⟨p_t^(l), p_t^(l+1)⟩ / (||p_t^(l)|| ||p_t^(l+1)||)，当 S > τ_high 时完全跳过 layer l+1 的计算，当 τ_low ≤ S ≤ τ_high 时以概率 P_base 剪枝，当 S < τ_low 时不剪枝。同时 track 累积 feature variation V_t = Σ||p_t - p_{t-i}||_1 动态调整剪枝概率：V_t 低时增加剪枝（精细 refine 阶段），V_t 高时减少剪枝（剧烈变化阶段）。三层联合优化：HLC 消除跨 timestep 冗余计算，AIGQ 按 feature sensitivity 动态降精度，SRAP 在同一 timestep 内剪枝冗余层。伪代码：每个 timestep t → 计算 D_t^(l) → HLC 决定是否刷新缓存 τ_t^(l) → AIGQ 按 bit-width(t) 量化权重/激活 → 计算 S_t^(l,l+1) → SRAP 决定是否跳过 layer l+1 → 仅对非缓存/非剪枝层执行 full compute。
  - 实验比较：(a) VBench 质量对比：QuantCache W8A8/W4A6 vs Open-Sora FP16, Q-diffusion, Q-DiT, PTQ4DiT, SmoothQuant, Quarot, ViDiT-Q 在 8 维度（Motion Smoothness, BG Consistency, Subject Consistency, Aesthetic Quality, Imaging Quality, Dynamic Degree, Scene Consistency, Overall Consistency）；(b) CLIP+DOVER 质量对比：CLIPSIM, CLIP-Temp, VQA-Aesthetic, VQA-Technical 四项指标；(c) Ablation study: Baseline(无优化) vs +HLC vs +HLC+AIGQ vs +HLC+AIGQ+SRAP 各组件贡献（speedup 从 1.00× → 4.12× → 6.33× → 6.72×）；(d) Speedup 对比：QuantCache vs Open-Sora, T-Gate, PAB, ViDiT-Q, AdaCache-slow, AdaCache-fast（1.00×∼2.24× vs 6.72×）。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A800-80GB GPU，CUDA 12.1。开发了 optimized GEMM CUDA kernels，通过 kernel fusion 将量化过程与 rotation transformations 及 intermediate feature caching 融合。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Open-Sora 1.2（DiT-based video generation model），生成 64-frame, 512×512 resolution 视频，100 timesteps（denoising steps）。
  - Benchmarks：(1) VBench benchmark suite（8 个评估维度：Motion Smoothness, Background Consistency, Subject Consistency, Aesthetic Quality, Imaging Quality, Dynamic Degree, Scene Consistency, Overall Consistency）；(2) CLIP 指标（CLIPSIM, CLIP-Temp，衡量 text-video alignment 和 temporal semantic consistency）；(3) DOVER 视频质量评估（VQA-Aesthetic, VQA-Technical）。
  - Baseline 方法：Q-diffusion (ICCV 2023), Q-DiT (2024), PTQ4DiT (NeurIPS 2025), SmoothQuant (ICML 2023), Quarot (NeurIPS 2024), ViDiT-Q (ICLR 2025), T-Gate (TMLR 2025), PAB (2024), AdaCache (2024)。
  - Quantization scheme：uniform min-max quantization，per-channel weight quantization + dynamic per-layer activation quantization（激活量化参数 online 计算）。混合精度权重量化 offline 确定（small calibration dataset）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/JunyiWuCode/QuantCache（论文声明 code and models will be available）
  - 算法 pipeline 张量计算流程（以 Open-Sora 1.2, W4A6 + HLC + SRAP 为例）：
    1. **初始化/Calibration**：加载 Open-Sora 1.2 预训练权重（FP16）→ 用 small calibration dataset 前向传播 → 记录每层 weight sensitivity（numerical error + perceptual distortion + temporal dynamics）→ offline 确定混合精度 bit-width 分配：关键层（高 sensitivity）→ Bit_max (e.g. 8-bit)，次要层 → Bit_mid (e.g. 6-bit)，冗余层 → Bit_min (e.g. 4-bit)。同时记录 channel-balancing scaling factors（offline 融合到前层权重）。
    2. **Per-timestep 推理循环**（共 100 timesteps，以 timestep t 为例）：
       - Step 1 — HLC 决策：计算 D_t^(l) = ||p_t^(l) - p_{t-k}^(l)||_1 / k · ||∇_t m_t^(l)|| → 如果 D_t^(l) < δ_1 且上次缓存未过期(τ_t^(l)=τ_max) → 直接复用 cached feature，跳过 layer l 的完整计算。如果 D_t^(l) ≥ δ_2 → τ_t^(l) = τ_min（频繁刷新）。
       - Step 2 — AIGQ 量化（对非缓存层）：计算 timestep 冗余度 D_t（从 D_t^(l) 聚合）→ 确定激活 bit-width: D_t ≥ θ_2 → Bit_min (aggressive quant)；θ_1 ≤ D_t < θ_2 → Bit_mid；D_t < θ_1 → Bit_max。Weights: W̄ = clamp(round(W / s_W) + z_W, 0, 2^b_W - 1)（per-channel）。Activations: X̄ = clamp(round(X / s_X) + z_X, 0, 2^b_X - 1)（dynamic per-layer, online compute s_X = (max(X) - min(X))/(2^b_X - 1)）。Channel balancing: X_balanced = R @ (S ⊙ X)（S 为 scaling 修正矩阵，R 为 rotation 矩阵）。
       - Step 3 — Transformer block 计算：对每个 layer l（STA = Spatial-Temporal Attention, CA = Cross-Attention, FFN = Feed-Forward Network）→ 加载量化权重 W̄^(l) 和量化激活 X̄^(l) → 执行低精度 GEMM → dequant 输出。
       - Step 4 — SRAP 剪枝（同 timestep 内）：计算相邻层 cosine similarity S_t^(l,l+1) = ⟨p_t^(l), p_t^(l+1)⟩ / (||p_t^(l)|| · ||p_t^(l+1)||) → 如果 S > τ_high → 跳过 layer l+1（feature copy forward）。同时计算 V_t = Σ ||p_t - p_{t-i}||_1 → V_t < δ_low（精细 refine）→ 增加剪枝概率；V_t > δ_high（剧烈变化）→ 减少剪枝。
    3. **去噪输出**：经上述优化后的 DiT 前向计算 → 输出预测噪声 ε_θ(x_t, t) → 更新 x_{t-1} → 循环至 t=0 → VAE decoder 生成视频帧。

## SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 SPR²Q，一种针对 Mamba 架构图像超分辨率模型的低比特后训练量化（PTQ）方法。包含两个核心组件：(1) **Pre-Quantization Fine-tuning with Fused Rectifier (PQFR)**：在量化前将低秩 rectifier 模块（ΔW = BA，A∈ℝ^{r×d_in}，B∈ℝ^{d_out×r}）学习到的权重增量融合到骨干网络，注入补偿信息，联合优化 rectifier 参数 (A,B) 和量化器参数 (a,b)；(2) **Static Priority-Based Rectifier Routing (SPR²)**：构建 N=4 个 rectifier 组成的 rectifier group，通过动态门控网络 g_i 加权聚合训练后，离线校准得到静态路由表（SPR²Q Table），推理时每个模块从路由表中检索最优增量并融合，不引入额外推理开销。
  - 实验比较：(a) 与 SOTA Mamba 量化方法对比（PTQ4VM, Quamba, MambaQuant）在 4-bit 和 2-bit 精度下；(b) 与 SwinIR Transformer 量化方法对比（2DQuant, FIMA-Q, APHQ-ViT）验证跨架构泛化性；(c) 消融实验：组件消融（PQFR→+RGT→+OSRC 逐步增益）、rectifier rank（r=2/4/8/16）、rectifier group size（N=2/4/8）；(d) 极端 1-bit 量化评估；(e) 实际效率：模型尺寸压缩（4-bit:2.51×, 2-bit:2.81×）和 FLOPs 加速（4-bit:3.44×, 2-bit:4.15×）。

- 硬件平台是什么，配置是什么。
  - NVIDIA RTX 4090 GPU。基于 PaddlePaddle 深度学习框架实现。

- 模型是什么。数据集和bench分别是什么。
  - 主模型：MambaIRv2-light（Mamba-based SR backbone）。跨架构泛化模型：SwinIR-light（Transformer-based）。
  - 训练集：DF2K（DIV2K + Flickr2K）。
  - 评估 Benchmark：Set5（5张）、Set14（14张）、B100（100张）、Urban100（100张）、Manga109（109张）。
  - 评估指标：PSNR 和 SSIM（在 YCbCr 空间的 Y 通道上测量）。
  - 缩放因子：×2 和 ×4。量化精度：4-bit、2-bit（主实验）、1-bit（极端实验）。
  - 对比 Baseline：PTQ4VM、Quamba、MambaQuant（Mamba 量化 SOTA）；2DQuant、FIMA-Q、APHQ-ViT（Transformer 量化 SOTA）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：论文未明确提供开源代码链接。论文声明基于 PaddlePaddle 框架实现。
  - 算法 pipeline 张量计算流程（SPR²Q, 4-bit, MambaIRv2-light, ×2 SR）：

  **阶段一 — Rectifier Group Training（动态路由训练, 12,000 iterations, batch=8）：**
  ```
  # 输入：预训练的 MambaIRv2-light 权重 W (frozen)
  # 初始化：N=4 个 rectifier {(A_i, B_i)}, A_i∈ℝ^{r×d_in}, B_i∈ℝ^{d_out×r}, r=8
  # 初始化：轻量门控网络 G（输出 N 维权值 g_i），量化器裁剪界 (a, b)
  # 优化器：Adam, lr=1e-2, Cosine Annealing

  for each training iteration:
      X_lr = sample_batch(DF2K)  # 低分辨率输入图像
      
      # 前向传播
      for each Mamba module l in model:
          # Step 1: 门控计算
          g = G(X_lq)  # g ∈ ℝ^N, softmax 归一化
          
          # Step 2: rectifier 加权聚合
          ΔW_fused = Σ_{i=1}^{N} g_i · (B_i @ A_i)  # 融合后增量
          
          # Step 3: 权重更新 + 伪量化
          W' = W + ΔW_fused                     # 融合 rectifier 补偿
          Ŵ' = clip(W', a, b)                   # 裁剪到 [a, b]
          s = (b - a) / (2^n - 1)               # 量化步长, n=4
          W_q' = round((Ŵ' - a) / s) · s + a    # 量化-反量化
          
          # Step 4: 量化权重前向计算
          Y = X_q @ W_q'                         # 线性变换输出
      
      # Step 5: Loss 计算
      L_pixel = || f_q(x) - y_FP ||_1           # 像素级重建 loss
      L_feature = Σ_{l=1}^{L} || φ_l(f_q(x)) - φ_l(f_FP(x)) ||_2²  # 逐块特征对齐 loss
      L = L_pixel + λ · L_feature               # 混合损失
      
      # Step 6: 反向传播 (STE 梯度估计)
      ∂L/∂A_i = g_i · B_i^T @ ∂L/∂W'           # STE 近似通过 round()
      ∂L/∂B_i = g_i · ∂L/∂W' @ A_i^T
      ∂L/∂(a,b) = ∂L/∂W_q' · ∂W_q'/∂(a,b)      # 裁剪界梯度 (Eq. 8)
      update(A_i, B_i, a, b, G) via Adam
  ```

  **阶段二 — Offline Static Routing Calibration（500 iterations）：**
  ```
  # 输入：训练好的 rectifier 组 {(A_i, B_i)} 和预训练权重 W（均 frozen）
  # 目标：为每个模块学习最优静态门控权重 ĝ

  for each Mamba module l:
      # 优化 ĝ ∈ ℝ^N（本文用梯度下降法, Eq. 12）
      ĝ = argmin_g L(f(X, Q_{a,b}(W + Σ g_i · (B_i@A_i))))
      # 收集 ĝ 并构建 SPR²Q Table
      SPR2Q_Table[l] = Σ ĝ_i · (B_i @ A_i)  # 预计算最优增量
  ```

  **阶段三 — 推理（零额外交付）：**
  ```
  for each Mamba module l:
      # 从 SPR²Q Table 检索该模块的最优增量
      ΔW_opt = SPR2Q_Table[l]
      
      # 权重融合（offline，实际推理前完成）
      W_final = W + ΔW_opt                    # 补偿后的权重
      
      # 量化（offline 完成）
      W_q_final = Q_{a,b}(W_final)            # 4-bit/2-bit 量化权重
      
      # 推理时直接加载量化权重进行前向计算
      Y = X @ W_q_final                       # 无额外门控、无动态路由
  ```

  **关键数值结果（4-bit, ×2）：** Set5 PSNR=37.72（vs PTQ4VM 37.17, MambaQuant 36.67）；Urban100 PSNR=31.53（vs PTQ4VM 30.47, MambaQuant 28.08）。MambaIRv2-light 从 3.01MB→1.20MB(4-bit, 2.51×), 1.07MB(2-bit, 2.81×)；FLOPs 从 75.6G→22.0G(4-bit), 18.2G(2-bit)。
  - 推理阶段无额外计算开销：所有 rectifier 参数离线融合，SPR²Q Table 在推理前已固化，模型结构与原始 MambaIRv2-light 完全一致。

## SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SageAttention，一种面向Attention的INT8后训练量化（PTQ）方法，即插即用加速推理。核心技术：(1) Smooth K — 对K矩阵按token维度减去均值mean(K)消除channel-wise outlier，数学上等价于原始softmax（$q(K-\text{mean}(K))^\top = qK^\top - q\cdot\text{mean}(K)^\top$），不影响attention score；(2) INT8 per-token/per-block量化Q和K，利用INT8 Tensor Core加速$QK^\top$ Matmul；(3) FP16 accumulator for PV Matmul — 保留P和V在FP16精度，使用FP16 accumulator代替FP32 accumulator，2×加速$PV$ Matmul而零精度损失；(4) Adaptive Quantization — 对每个layer自动在SAGEAttn-B（QK INT8 per-block + PV FP16）和SAGEAttn-vB（全INT8，PV也INT8量化）中选择，cosine similarity > 99.8%的层选vB（约4%更快），其余选B（更准确）。
  - 实验比较：speed对比FlashAttention2、xformers、Torch Attention（TOPS和真实模型延迟）；accuracy对比FP16 full-precision attention；quantization method对比FlashAttention3 FP8版本、per-token/per-block/per-tensor INT8量化。End-to-end评估覆盖Llama2-7B（WikiText perplexity, LAMBADA, MMLU accuracy）、CogvideoX（CLIPSIM, CLIP-T, VQA-a, VQA-t, FScore）、Unidiffuser/UltraPixel（FID, sFID, CLIP, ImageReward）、TIMM（ImageNet, Sketch, ImageNet-r accuracy）、Llava1.6（TextVQA, POPE, VQAv2）。消融实验包括smooth K的overhead（<0.2%）和adaptive quantization的收益（+11.7% OPS）。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA RTX 4090（PCIE 5.0, 16-core Xeon 6430 CPU, 120GB DDR4 RAM）；NVIDIA RTX 3090（16-core Xeon 8358P CPU, 80GB DDR4 RAM）。
  - 软件环境: Ubuntu 22.04, torch 2.4.0+cu121, triton-nightly (20240816), python 3.11, gcc/g++ 9。

- 模型是什么。数据集和bench分别是什么。
  - 模型: Llama2-7B（text2text）, CogvideoX（text2video）, Unidiffuser（text2image）, UltraPixel（text2image）, TIMM vit_base_patch16_224（image classification）, Llava1.6（visual QA）。
  - 数据集/benchmark: WikiText（perplexity）, LAMBADA（contextual understanding）, MMLU（knowledge）；Open-Sora prompt sets（video generation）；COCO 2014val（前256条annotations用于图像生成FID/sFID计算）；ImageNet, ImageNet-Sketch, ImageNet-Rendition（分类）；TextVQA, POPE, VQAv2（VQA）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接: https://github.com/thu-ml/SageAttention（MIT License）。
  - 算法pipeline（SAGEAttn-B, Algorithm 1）:
    1. Preprocessing: K_smooth = K - mean(K)  # mean over token dim, shape 1×d
    2. Quantization: (δ_Q, Q̂) = ψ_Q(Q/√d), (δ_K, K̂) = ψ_K(K_smooth)  # INT8 per-block
    3. Tiling: 将Q̂分为T_m=N/b_q块{Q̂_i}，K̂和V分为T_n=N/b_kv块{K̂_j},{V_j}
    4. For i in [1, T_m] (并行于SMs):
       For j in [1, T_n]:
         S_i^j = Matmul(Q̂_i, K̂_j^T) × δ_Q[i] × δ_K[j]  # INT8 Tensor Core, dequant via scale multiplication
         (m_i^j, P̃_i^j) = online_softmax(m_i^{j-1}, S_i^j)  # FP16
         O_i^j = diag(e^{m_i^{j-1}-m_i^j})O_i^{j-1} + Matmul(P̃_i^j, V_j, accum=FP16)  # FP16 accumulator
       O_i = diag(l_i^{T_n})^{-1}O_i^{T_n}
    5. Q,K量化粒度选择: per-token (SAGEAttn-T) 或 per-block (SAGEAttn-B)。P使用per-block（因P每行max=1，静态scale=1/127），V使用per-channel（解决channel-wise outlier）。
    6. Adaptive: 对每layer测试SAGEAttn-vB cosine sim，若>99.8%则用vB（全INT8，快4%），否则用B（FP16 PV accumulator）。


## Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 Sherry，一种 1.25-bit 三值量化方法，核心实现：(1) **3:4 细粒度结构化稀疏**：在每连续 4 个权重中强制恰好 3 个非零（±1），将 4 个权重打包为 5 bits（排列数 N_perm = C(4,3) × 2³ = 32，饱和 5-bit 索引），实现等效 1.25 bit/weight；(2) **Arenas（Annealing Residual Synapse）模块**：训练时注入异构梯度 Y = X·Q(W) + λ_t·X·W，其中 λ_t 为 annealing coefficient 在训练结束时退火至零，防止梯度同质化导致的 weight trapping 和表示坍缩。实验比较：(1) zero-shot 基准精度对比（PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande），baseline 包括 TWN、Spectra、BitNet、TernaryLLM、LLM-QAT、ParetoQ、Tequila；(2) CPU 推理吞吐量（tokens/s）和模型大小（MB）对比（BitNet I2_S、Tequila TL2）。

- 硬件平台是什么，配置是什么。
  训练平台：论文未明确说明 GPU 配置。推理效率评估：Intel i7-14700HX CPU，测量 tokens/s 和模型大小；AngelSlim 框架层面在 Apple M4 和 MediaTek Dimensity 9500 上额外评估了边缘设备推理效率。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.2 1B 和 3B（BF16 baseline）。数据集与 Benchmark：PIQA、ARC-Easy（ARC-e）、ARC-Challenge（ARC-c）、HellaSwag（HelS）、WinoGrande（WinG），均为 zero-shot 评估。训练数据：论文未明确说明 Sherry 的训练数据集规模和构成。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Tencent/AngelSlim（sherry 分支），复现模型权重：https://huggingface.co/MoraxGeo/Sherry-3B-1.25bit-per-channel

  **算法 Pipeline：Sherry 1.25-bit 三值量化 QAT（结合 3:4 稀疏 + Arenas 模块）：**

  **Step 1 — 量化前向传播（带 3:4 结构化稀疏）：**
  给定权重组 W = [w₁, w₂, w₃, w₄]（连续 4 个权重为一组）：
  ```
  # 标准三值量化函数（per-weight）:
  Q(w_i) = α · sign(w_i) · I[|w_i| ≥ Δ]   # 输出 ∈ {+α, 0, -α}

  # 3:4 结构化稀疏约束（per-group of 4）:
  # 在每组 4 个权重中，保证恰好 1 个为 0，3 个非零（±1）
  # 实现方式：计算 |w_i| 的排序，将最小 |w_i| 的权重量化为 0
  for each group g of 4 weights:
      scores = [|w_1|, |w_2|, |w_3|, |w_4|]
      prune_idx = argmin(scores)
      for i in [0..3]:
          if i == prune_idx:
              Q(w_i) = 0
          else:
              Q(w_i) = α · sign(w_i)
  ```

  **Step 2 — Arenas 模块前向注入（并行于量化路径）：**
  ```
  # 量化路径（3:4 sparsified）:
  Y_q = X · Q(W)   # 矩阵乘法用三值权重 + 3:4 稀疏

  # Arenas 残差路径（注入连续潜权重的异构梯度）:
  Y_res = λ_t · X · W   # W 为全精度潜权重, λ_t 随时间退火

  # 总前向输出:
  Y = Y_q + Y_res = X · Q(W) + λ_t · X · W
  ```

  **Step 3 — 5-bit 打包与推理（training-free，仅推理时）：**
  ```
  # 每组 4 个三值权重（含 3:4 稀疏）打包为 5-bit 索引:
  # N_perm = C(4,3) × 2^3 = 4 × 8 = 32 种可能排列
  # 5 bits 恰好编码 32 种排列 → 完美饱和 5-bit 索引空间

  for each group of 4 ternary weights:
      # 编码：3 个非零位置的组合（C(4,3)=4 种）× 
      #        每个非零位置的符号（±1，2³=8 种）
      group_idx = encode_sparse_ternary(w_0, w_1, w_2, w_3)
      # group_idx ∈ [0, 31], 存储为 5-bit
      # 存储开销：4 weights / 5 bits = 1.25 bits/weight
  ```

  **Step 4 — 训练时 Arenas 退火与梯度流：**
  ```
  # λ_t 调度: 从 λ_0（初始值，论文未给出具体值）退火至 0
  # 梯度流（通过 STE）:
  ∂L/∂W = ∂L/∂Y · (λ_t + ∂Q(W)/∂W)
  # 当 λ_t > 0 时，∂L/∂X 不会坍缩为低秩（Arenas 注入全秩残差信号）
  # 当 λ_t → 0 时，模型收敛为纯三值量化+3:4 稀疏，推理零额外开销
  ```

  **关键设计对比：**
  - 2-bit 打包策略（如 BitNet）：每权重 2 bits → 浪费 0.42 bits（32/16=2x 开销），每 4 权重 8 bits
  - 1.67-bit 打包策略（如 Tequila）：3 权重打包为 5 bits → SIMD 不友好的 3-way pattern
  - Sherry 1.25-bit：4 权重打包为 5 bits → SIMD 友好的 4-way pattern，完美对齐现代 CPU 的 128/256-bit 向量寄存器

## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

- 属于算法pipeline的实现是什么？实验比较什么？
  ResQ 是一种后训练量化（PTQ）方法，通过 PCA 识别激活中方差最高的低秩子空间（rank r = d/8），将子空间内系数保持 8-bit 高精度，其余量化到 4-bit；并在每个子空间内应用不变随机旋转抑制 outliers，将权重、激活和 KV cache 均量化到 W/A/KV=4/4/4。实验比较 ResQ 与 RTN、GPTQ、SmoothQuant+、QUIK、QuaRot、SpinQuant 在 Wikitext 困惑度、0-shot common sense reasoning（8 任务）、MMLU、GSM8K、LongBench（qmsum/samsum/repobench-p）、MMMU（多模态）上的表现。ResQ 相比 SpinQuant 在 Wikitext 困惑度上降低 4-33%，0-shot 精度提升 0.1-5.4%，无额外训练。

- 硬件平台：单张 NVIDIA A100 80GB GPU 用于量化和评估（Meta-Llama-3-70B 使用 4 张 GPU 评估）；NVIDIA RTX 3090 用于硬件加速测试。

- 模型：Llama 2 (7B, 13B)、Meta-Llama-3 (8B, 70B)、Llama 3.2 (1B, 3B)、Qwen2.5 (0.5B, 3B, 72B)、Qwen2-VL (2B, 7B Instruct)。数据集与 benchmark：Wikitext（困惑度）、ARC-c/e, BoolQ, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande（0-shot common sense reasoning）、MMLU（语言理解）、GSM8K 5-shot（数学推理）、samsum/qmsum/repobench-p from LongBench（对话摘要和代码补全）、MMMU（多模态理解）。校准数据使用 Wikitext 512 随机样本获取投影矩阵，GPTQ 使用 128 随机样本。

- 开源情况：代码开源 https://github.com/utkarsh-dmx/project-resq。基于 HuggingFace Transformers + PyTorch 实现，使用 CUDA 11.8 + CUTLASS 进行 INT4/INT8 GEMM 操作，评估使用 lm_evaluation_harness v0.4.5 和 LongBench。

- 算法 pipeline 详解（张量计算级别）：
  给定激活 X∈R^{n×d} 和权重 W∈R^{d×d}：
  1. **PCA 投影矩阵构造**：U = PR，其中 P 由 X 协方差矩阵 XX^T 的特征向量按特征值递增排列组成（后 r 列为高精度子空间 P_h，前 d-r 列为低精度子空间 P_l），R 为随机正交矩阵（Hadamard 或随机旋转）。
  2. **投影与量化**：X_q = Q_L(X·U_l) + Q_H(X·U_h)，W_q = Q_L(U_l^T·W) + Q_H(U_h^T·W)，其中 Q_L 为 4-bit 量化，Q_H 为 8-bit 量化。
  3. **输出计算**：X_q·W_q = Q_L(XU_l)·Q_L(U_l^T·W) + Q_H(XU_h)·Q_H(U_h^T·W)，交叉项因正交性消失。
  4. **推理时投影融合**：U_A 通过右乘 o_proj/down_proj 权重矩阵融入前一层；U_B/U_C 处理注意力块内 KV cache 量化投影；U_D 为 Hadamard 矩阵，通过快速 Hadamard 变换实现；U_C 因 RoPE 存在需运行时显式计算（8-bit 量化）。
  5. **理论保证**：Theorem 4.2 证明 PCA 基选择最小化量化误差上界。

  校准流程：从 Wikitext 采样 512 条校准数据 → 前向传播收集各层激活 X → 对 X 做 PCA 得特征向量 P → 生成随机正交旋转矩阵 R → 构造 U=PR → 将 U_A/U_B/U_C/U_D 融合到权重 → 用 GPTQ 对权重做进一步优化量化。Meta-Llama-3-8B 完整流程在单张 A100 上耗时 35 分钟。

## SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  SpQR提出一种新的混合稀疏-量化压缩格式，将LLM权重量化到3-4 bits的同时保持near-lossless精度（<1% perplexity损失）。核心包含两个创新：(1) **敏感权重异常值检测与隔离**：基于Optimal Brain Surgeon框架的封闭形式敏感度准则 s_ij = (w_ij - quant(w_ij))² / (2[H⁻¹]_jj)，在GPTQ逐列量化过程中动态计算每个权重的敏感度，将超过阈值τ的高敏感度权重（约1%）保留为16-bit异常值。(2) **双层量化（Bilevel Quantization）**：使用极小group size（β₁=8-32）进行分组量化，并将第一层量化统计量（scale和zero-point）本身再以相同算法做第二层量化（β₂=16），从而在低bit-width下保持高精度。实验比较SpQR vs GPTQ和RTN (round-to-nearest) baseline在3-bit和4-bit配置下的WikiText2、C4、Penn Treebank perplexity，以及五任务zero-shot accuracy（WinoGrande, PiQA, HellaSwag, ARC-easy, ARC-challenge）。同时进行ablations：bilevel quantization vs 16-bit statistics、unstructured outliers vs row outliers vs column outliers、GPTQ activation order heuristic效果。

- 硬件平台是什么，配置是什么。
  量化和评估主要在单张NVIDIA A100-80GB GPU上完成，部分实验在NVIDIA A6000（48GB）上进行。量化实施采用PyTorch实现。推理速度测试在A100 GPU上进行，batch size=1的token-by-token生成模式。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA (7B, 13B, 30B, 65B)，Falcon (7B, 40B)，OPT (6.7B, 13B, 30B, 66B)。校准数据集：对LLaMA使用RedPajama数据集（LLaMA训练数据的公开复刻），对Falcon使用RefinedWeb数据集。量化的校准样本数为128个2048-token序列。评估数据集：WikiText2、C4、Penn Treebank（perplexity）；WinoGrande、PiQA、HellaSwag、ARC-easy、ARC-challenge（zero-shot accuracy，使用LM Evaluation Harness）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/Vahe1994/SpQR（论文引用"to be integrated into github.com/TimDettmers/bitsandbytes"）。
  
  ```bash
  # 以LLaMA-30B near-lossless配置为例：
  python main.py $MODEL custom --custom_data_path=$DATA \
      --wbits 4 --groupsize 16 --perchannel \
      --qq_scale_bits 3 --qq_zero_bits 3 --qq_groupsize 16 \
      --outlier_threshold 0.1 \
      --fit_quantizer_without_outliers --permutation_order act_order
  ```
  
  SpQR算法Pipeline（以W ∈ R^{d_out × d_in} 的3-bit base量化 + 1% outliers为例）：
  
  ```
  Input: W ∈ R^{m×n} (weight matrix), X ∈ R^{n×d} (calibration data, 128×2048 tokens)
  Output: Q (quantized weights), S_q/Z_q (first-level quantized scales/zeros), 
          S_s/Z_s/S_z/Z_z (second-level quantized statistics), W_sparse (CSR outlier matrix)
  
  1. 计算Hessian: H = 2XXᵀ, Hⁱᶜ = Cholesky((H + λI)⁻¹)
  
  2. 逐列（逐β₁ group）处理权重矩阵，对每column group i=1, β₁, 2β₁, ..., n:
     
     a) 检测异常值（outliers子程序）:
        E_base = error(W[:,j], Hⁱᶜ[:,j])           # 所有(beta1列)权重的L2 error
        for each column j in the group:
            E_ol = error(W[:, loo], Hⁱᶜ_loo,loo)   # leave-one-out error
            if E_base - E_ol > τ:                   # 标记为outlier
                O = O ∪ {j}
     
     b) 在排除outlier的情况下拟合group-wise quantizer:
        ŝ, ẑ = fit_statistics(W_group, O)           # bilevel quantization
     
     c) 量化非outlier权重:
        Q[:,j] = quantize(W[:,j], ŝ, ẑ)
        ŵ_q = dequantize(Q[:,j], ŝ, ẑ)
     
     d) 误差补偿（GPTQ风格）:
        E[:,j] = (W[:,j] - ŵ_q) / Hⁱᶜ[j,j] · (1 - is_outlier(W[:,j]))
        W[:,j:i+β₁] -= E · Hⁱᶜ[j,(j:i+β₁)]
        W[:,(i+β₁):n] -= E · Hⁱᶜ[(i:i+β₁),(i+β₁):n]
  
  3. 收集outlier矩阵为CSR格式:
     W_sparse = gather_outlier_matrix(W, O)  # row-first, col-second排序
     
     存储格式：
     - 每个outlier: 16-bit value + 16-bit column index = 32 bits
     - 每行: 一个32-bit row pointer (cumulative outlier count)
  
  4. 收集量化统计量为双层结构:
     S_q, Z_q: first-level 3-bit quantized scales & zero-points (每组β₁=16权重)
     S_s, Z_s: second-level scales (量化ŝ的scale), 每组β₂=16个first-level统计量
     S_z, Z_z: second-level zeros (量化ẑ的zero), 每组β₂=16个first-level统计量
     
     每256权重（β₁×β₂=16×16）的内存布局：
     - 256个3-bit weight codes
     - 16个3-bit scales + 16个3-bit zero-points
     - 4个16-bit second-level statistics scalars
  
  平均bits数计算：b̄ = b_w + (b_s+b_z)/β₁ + 64/(β₁β₂) + 32·r_o
  例：b_w=3, b_s=b_z=3, β₁=16, β₂=32, r_o=0.4% → b̄ = 3 + 6/16 + 64/512 + 0.128 = 3.63 bits/param
  ```

## SqueezeLLM Dense-and-Sparse Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  Post-training weight-only quantization framework with two novel techniques: (i) **sensitivity-based non-uniform quantization** using Fisher information (diagonal Hessian approximation) to weight k-means clustering — centroids are pulled closer to weights with higher second-order sensitivity to the final loss, achieving far better perplexity than uniform quantization at equal bitwidth; (ii) **Dense-and-Sparse decomposition** that extracts 0.45% weight values (0.05% most sensitive + 0.4% outliers) into a sparse FP16 matrix stored in CSR format, with the remaining 99.55% dense matrix quantized at 3-4 bits with significantly contracted value range. Both dense and sparse components participate in inference via separate but concurrently-launched custom CUDA kernels.

  实验比较：Perplexity on C4 and WikiText2 (LLaMA 7B/13B/30B/65B, LLaMA2 7B/13B/70B, OPT 1.3B/2.7B/6.7B/13B/30B) against RTN, GPTQ (with/without activation ordering, with/without grouping g128), AWQ (g128), SpQR, QuIP, and OmniQuant; MMLU zero-shot and 5-shot accuracy on Vicuna v1.1/v1.3 (7B/13B/33B) vs AWQ; instruction-following ability via GPT-4 pairwise scoring vs GPTQ/AWQ.

- 硬件平台是什么，配置是什么。
  量化：NVIDIA A100-80G (Fisher information computation via gradient backpropagation) + Intel Xeon Gold 6126 48-core (sensitivity-weighted k-means clustering)。推理延迟评测：NVIDIA A6000 GPU (primary, using Torch CUDA profiler for 128/1024 token generation), also A100 GPU (kernel-only matrix-vector runtime benchmark)。

- 模型是什么。数据集和bench分别是什么。
  Models: LLaMA (7B/13B/30B/65B), LLaMA2 (7B/13B/70B), OPT (1.3B/2.7B/6.7B/13B/30B), Vicuna v1.1 (7B/13B), Vicuna v1.3 (7B/13B/33B)。Datasets: C4 and WikiText2 (perplexity, chunk size 2048), MMLU benchmark (zero-shot and 5-shot), Vicuna evaluation (80 sample questions, GPT-4 pairwise scoring with order randomization, 160 total queries)。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/SqueezeAILab/SqueezeLLM (MIT license)。

  算法Pipeline全流程（以LLaMA-7B 3-bit量化为例）：

  **Step 1: Fisher信息矩阵计算（Sensitivity Estimation）**
  ```
  # 对calibration dataset D (100 samples from C4或Vicuna training set)
  for each sample d in D:
      loss = cross_entropy(model(d), labels)
      compute gradients g_d = ∂loss/∂W for all weight matrices
  F = diagonal( (1/|D|) * Σ_d (g_d ⊙ g_d) )  # per-weight Fisher diagonal
  ```
  Fisher计算资源：7B模型~0.3分钟(A100)，65B模型~2.5分钟(A100)。峰值内存需求：7B=33GB, 13B=61GB, 30B=149GB, 65B=292GB。

  **Step 2: Sensitivity-weighted K-means Clustering（Per-channel, per weight matrix）**
  ```
  量化目标函数: argmin_Q Σ_{i=1}^{N} F_ii * (w_i - Q(w_i))^2
  # Fisher对角线值F_ii充当每个权重w_i的importance weight
  # 非均匀量化: 每个输出channel有独立LUT (2^b个FP16 centroid)
  
  for each output channel c in weight matrix W (shape [out_c, in_c]):
      k = 2^b  # e.g., k=8 for 3-bit
      centroids = k-means++初始化(W[c, :], k)
      用F[c, :]作为权重执行weighted k-means:
          repeat until convergence:
              assignment[i] = argmin_j ||w_i - centroid_j||^2
              centroid_j = (Σ_{i ∈ cluster_j} F_ii * w_i) / (Σ_{i ∈ cluster_j} F_ii)
      输出: LUT[c] = {c_0, ..., c_{k-1}} (FP16 values), indices[c] ∈ [0, k-1]^in_c (b-bit每个)
  ```
  K-means耗时：7B=11min, 13B=17min, 30B=45min, 65B=80min (Xeon Gold 6126 48核)。

  **Step 3: Dense-and-Sparse Decomposition**
  ```
  for each weight matrix W:
      # 识别outliers (基于百分位阈值)
      T_min = percentile(W, τ_low)    # τ_low ≈ 0.2%
      T_max = percentile(W, 1 - τ_high) # τ_high ≈ 0.2%
  
      # 识别sensitive values (基于Fisher信息排名)
      top_k_sensitive = topk_indices(F, k=0.05% of total elements)
  
      # 稀疏矩阵S = outliers ∪ sensitive values (去重)
      S_indices = {i | W_i < T_min or W_i > T_max or i ∈ top_k_sensitive}

      # 存储S为CSR格式 (FP16)
      S_csr.values = W[S_indices]             # FP16
      S_csr.col_indices = column indices       # int16
      S_csr.row_ptrs = row boundary offsets    # int32

      # 密集矩阵D = W中非S元素 (99.55%的参数)
      D = W.copy(); D[S_indices] = 0
  
      # 对D执行Step 2的sensitivity-weighted k-means
      D_indices, LUTs = weighted_kmeans_quantize(D, F, b=3)
  
  存储格式:
  - Dense分量: 3-bit indices + per-channel 8-entry LUT (8×FP16 per channel)
  - Sparse分量: CSR格式带FP16 values
  - 总avg bits ≈ 3.24 bit (3-bit dense + ~0.24 bit sparse overhead for 0.45% sparsity)
  ```

  **Step 4: 推理Forward Pass (GPU kernel)**
  ```
  def forward_layer(W_indices_3bit, LUTs, S_csr, activation_X):
      # X: activation vector (FP16), shape [in_features]
      # 两个kernel fused在单次launch中:
      
      # Dense部分: LUT-based dequant + matvec
      # 每个thread block加载index → LUT lookup → FP16 multiply-accumulate
      Y_dense = lut_dequant_matvec(W_indices_3bit, LUTs, X)
      
      # Sparse部分: Balanced CSR SpMV
      # 每线程10个non-zero元素 (balanced kernel避免row-skew问题)
      Y_sparse = balanced_csr_matvec(S_csr, X)
      
      return Y_dense + Y_sparse
  ```
  全部计算保持FP16精度，activations不量化。非均匀量化的LUT dequantization开销很小（相比uniform quantization增加仅~10% latency）。

  **关键结果（LLaMA-7B, C4 perplexity）**:
  | 方法 | Avg Bits | PPL (C4) | Speedup vs FP16 | Mem (GB) |
  |------|----------|----------|-----------------|----------|
  | FP16 Baseline | 16 | 7.08 | 1.0x | 12.7 |
  | RTN 3-bit | 3 | 28.26 | 2.3x | 2.9 |
  | GPTQ 3-bit (no group) | 3 | 9.55 | 2.3x | 2.9 |
  | SqueezeLLM dense-only | 3.02 | 7.75 | 2.1x | 2.9 |
  | GPTQ 3-bit (g128) | 3.24 | 7.89 | 0.2x (permutation overhead) | 3.0 |
  | AWQ 3-bit (g128) | 3.24 | 7.90 | 2.0x | 3.0 |
  | **SqueezeLLM 0.45%** | **3.24** | **7.56** | **1.9x** | **3.1** |

  Dense-only SqueezeLLM 3-bit already outperforms GPTQ g128 (7.75 vs 7.89), demonstrating that sensitivity-based non-uniform quantization alone is more effective than group-wise uniform quantization. Adding 0.45% sparsity further reduces the gap from FP16 to only 0.48 PPL points.


## S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出S²Q-VDiT，一个面向视频扩散模型（V-DMs）的后训练量化（PTQ）框架。核心包含两个创新组件：(1) **Hessian-aware Salient Data Selection (SDS)**：联合评估扩散信息量（diffusion salience）和量化敏感度（quantization salience）两个维度，通过Hessian矩阵近似计算每个校准样本的重要性分数并进行min-max归一化后的乘积作为统一salience得分，从候选池中筛选高质量校准数据。扩散信息量C_diff = ||x_t - x_{t-1}||²/||x_t||²衡量相邻去噪步的表示变化；量化敏感度C_quant = ||x_t^T x_t||_2基于Levenberg-Marquardt近似的Hessian矩阵衡量样本对量化扰动的敏感度。(2) **Attention-guided Sparse Token Distillation (STD)**：利用V-DMs中空间-时间注意力固有的稀疏模式，通过注意力图中token-wise的注意力分布计算每个token的重要性权重λ_j = normalize(Σ_{h,i} A_{h,i,j})，将量化损失重加权为L_quant = (1/n) Σ_j λ_j||θ^f(x_{j,:}) - θ^q(x_{j,:})||²，使模型在优化中聚焦高影响力token。

  实验比较W4A6（4-bit权重6-bit激活）和W4A4（4-bit权重4-bit激活）两种量化设置下的视频生成质量。对比方法包括：Q-DiT、PTQ4DiT、ViDiT-Q（扩散模型PTQ）；SmoothQuant、QuaRot（LLM PTQ baseline）。评估指标：VBench的8个维度（Imaging Quality, Aesthetic Quality, Motion Smoothness, Dynamic Degree, Background Consistency, Subject Consistency, Scene Consistency, Overall Consistency）和EvalCrafter的5个指标（CLIPSIM, CLIP-Temp, VQA-Aesthetic, VQA-Technical, FLOW Score）。

- 硬件平台是什么，配置是什么。
  所有校准实验在单张NVIDIA A800 GPU上完成。量化部署效率测试同样在单张NVIDIA A800 GPU上使用CUDA实现（基于ViDiT-Q和FlatQuant的CUDA kernel）。量化框架基于PyTorch实现。

- 模型是什么。数据集和bench分别是什么。
  模型：CogVideoX-2B、CogVideoX-5B (Yang et al., 2024)、HunyuanVideo-13B (Kong et al., 2024)。校准数据：10个随机prompt生成候选校准样本，最终每个方法选取40个样本（trade-off 性能vs校准时间后选择）。校准数据集尺寸对比实验（20/40/80样本）。评估benchmark：VBench (Huang et al., 2024) 和 EvalCrafter (Liu et al., 2024)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码（论文声明）：https://github.com/wlfeng0509/s2q-vdit
  
  算法Pipeline（以CogVideoX-2B W4A6量化为例）：

  **阶段1：Hessian-aware Salient Data Selection (SDS)**
  ```
  # 输入：预训练V-DM模型θ，候选prompt集合P，总去噪步数T
  # 输出：校准数据集D_calib，大小为N=40

  for each prompt p in P:
      for each timestep t in [1..T]:
          x_t = 模型在prompt p、timestep t的隐层表示  # x_t ∈ R^{n×d}, n=s×t
          计算扩散salience: C_diff(x_t) = ||x_t - x_{t-1}||² / ||x_t||²
          计算量化salience: C_quant(x_t) = ||x_t^T x_t||_2  # Levenberg-Marquardt Hessian近似
      end
  end

  # min-max归一化到[0,1]
  C̅_diff = (C_diff - C_diff_min) / (C_diff_max - C_diff_min)
  C̅_quant = (C_quant - C_quant_min) / (C_quant_max - C_quant_min)

  # 联合salience分数（乘积形式，算术-几何平均不等式确保双高才得分高）
  for each candidate x_t:
      C_sample(x_t) = C̅_diff(x_t) · C̅_quant(x_t)
  end

  # 按C_sample降序排列，选Top-N样本构成D_calib
  D_calib = top_N_samples_by_C_sample
  ```

  **阶段2：Block-wise Post-Training Quantization with STD**
  ```
  # 量化方案：uniform per-channel weight量化 + dynamic per-token activation量化
  # 使用channel-wise scale、rotation matrix、learnable clipping threshold
  # 基于GPTQ weight quantizer

  # 量化参数：对称量化
  # x_int = clamp(round(x/Δ), -2^{N-1}, 2^{N-1}-1), Δ = max(|x|)/(2^{N-1}-1)

  for each transformer block in model:
      # 前向计算FP模型的attention map用于后续STD
      A = block.attention(x)  # A ∈ R^{H×n×n}, H为head数

      # 计算每个token的重要性权重λ_j
      for each token j in [1..n]:
          S_j = Σ_{h,i} A_{h,i,j}  # token j从所有token和head获得的attention权重之和
      end
      λ_j = (S_j - min(S)) / (max(S) - min(S)) * (λ_max - λ_min) + λ_min

      # Block-wise量化优化（30个样本，15 epochs，AdamW optimizer，cosine LR）
      for epoch in [1..15]:
          for each sample x in D_calib（随机选30个）:
              # 量化损失：重加权的MSE
              L_quant = (1/n) Σ_{j=1}^{n} λ_j · ||θ^f_block(x_{j,:}) - θ^q_block(x_{j,:})||²
              # 更新量化参数：diag-balancing scale (lr=5e-3), rotation matrix (lr=5e-3),
              # learnable clipping threshold (lr=5e-2)
          end
      end

      # 吸收量化参数到权重（weight folding），无额外推理负担
      # 激活量化采用online dynamic quantization
  end
  ```

  **阶段3：部署推理**
  ```
  # W4A6量化后推理
  # 权重：per-channel INT4存储，推理时dequantize到FP16
  # 激活：per-token online dynamic INT6量化
  # CUDA实现基于ViDiT-Q [62] 和 FlatQuant [47]
  # 结果（CogVideoX-5B）：模型存储 2.633GB (3.94×压缩)，推理显存 10.145GB (1.56×节省)，延迟 203.2s (1.28×加速)
  ```

  **消融实验关键结果**：
  - SDS vs 随机采样：W4A4 CogVideoX-2B下SDS的Imaging Quality=52.95±0.69（方差仅0.69），而ATOP仅51.65±1.76（方差1.76），证明SDS在性能和稳定性上均优于随机采样。
  - STD vs 无STD：λ_min=0.5时获得最佳性能平衡，所有λ_min取值均能提升量化性能，证明STD的鲁棒性。
  - 校准数据量：40样本 vs 20样本显著提升，40→80样本提升微小，选40为统一设置。
  - 校准资源：W4A4 CogVideoX-2B下GPU Memory 35.68GB, GPU Time 2.88h, 相比PTQ4DiT仅增加2GB和0.63h。

## DiJiang: Efficient Large Language Models through Compact Kernelization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Frequency Domain Kernelization (FKA) —— 通过加权Quasi-Monte Carlo采样和DCT（离散余弦变换）将预训练vanilla Transformer的softmax注意力替换为线性复杂度的核化注意力，仅需少量fine-tuning即可将二次注意力O(n²d)降至线性O(nmd)（设m=d）。核心步骤：(1) 基于Bochner定理将Gaussian核（softmax注意力的等价形式）转为积分形式；(2) 用加权Quasi-Monte Carlo（PFF→WPFF）替代Monte Carlo采样以提升近似效率（O(1/m) vs O(1/m^{-0.5})）；(3) 用DCT系数矩阵C替换随机投影进行频域映射（WDCF），将复杂度从O(m)降至O(log m)。
  - 实验比较：(a) 不同模型规模（70M～2.8B Pythia）fine-tuning性能与训练时间；(b) 跨模型泛化（OPT-350M, TinyLLaMA-1.1B, LLaMA2-7B）；(c) 与Linformer/Performer/RetNet/Cosformer等线性注意力方法对比；(d) 推理吞吐与显存对比；(e) 注意力图可视化分析。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A800（训练时间测量及推理吞吐评估均使用A800）。
  - 推理评估token长度为2048。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Pythia（70M/160M/410M/1B/1.4B/2.8B系列，从HuggingFace EleutherAI checkpoint fine-tune）、OPT-350M、TinyLLaMA-1.1B、LLaMA2-7B。
  - 训练数据集：The Pile（825 GiB英语语料，22个子集）。
  - 评测benchmark：(小模型) PIQA, WinoGrande, WSC, ARC-E, ARC-C, LogiQA；(7B模型) 额外包含 SIQA, BoolQ, HellaSwag, MMLU, NQ, COPA, Race-Middle。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码已公开 https://github.com/YuchuanTian/DiJiang
  - 算法pipeline（Algorithm 1 + 核心公式）：

  **推理阶段——FKA前向计算（Equation 13）**：
  ```
  输入: Q, K, V ∈ R^{n×d}  (n=tokens, d=head_dim, 设定m=d)
  1. 构建 DCT 系数矩阵 C ∈ R^{m×d}:
     C[j1,j2] = s_j1 * s_j2 * Σ_i1 Σ_i2 cos(π(2i1+1)j1/(2d)) * cos(π(2i2+1)j2/(2d))
     其中 s_j = sqrt(1/d) 若 j=0, 否则 sqrt(2/d)
  2. 随机采样对角矩阵 T = diag(t_1,...,t_m), t_i ~ U(0,1) 服从逆累积分布
  3. 计算 WDCF 特征映射（Equation 12）:
     φ_WDCF(x) = D ⊙ exp(T · C · x^T)  对 x ∈ {q_i, k_i}
     其中 D ∈ R^m 为可学习权重
  4. 计算线性注意力（Equation 13）:
     FKA(Q,K,V) = φ_WDCF(Q) · φ_WDCF(K)^T · V
                 = φ(Q)_{n×m} × (φ(K)^T)_{m×n} × V_{n×d}
                 = φ(Q) × (φ(K)^T × V)    # 先算后两项 O(nmd)
                 → 输出 O ∈ R^{n×d}
  ```

  **训练/微调阶段（Algorithm 1）**：
  ```
  输入: 少量训练数据 x_i, 预训练Transformer模型 M
  1. 初始化每层的 DCT系数C, 权重D, 对角矩阵T
  2. 将每层的 Attention(Q,K,V)=softmax(QK^T)V 替换为 FKA(Q,K,V)=φ_WDCF(Q)·φ_WDCF(K)^T·V
  3. 得到变换后模型 M_FKA
  4. repeat:
       a. 从 x_i 随机采样mini-batch
       b. 用 M_FKA 前向传播
       c. 按loss和梯度更新 M_FKA 中的可学习参数
     until convergence
  输出: 高效语言模型 M_FKA
  ```

  - 关键设计：(a) 使用DCT替代FFT因为DCT在实数域操作，更少计算量且更硬件友好；(b) 设置m=d避免增加计算复杂度；(c) 借鉴RetNet的gating机制增强DiJiang；(d) 训练仅需原始Pythia约1/16的训练时间，DiJiang-7B仅需40B tokens训练（LLaMA2-7B用2T tokens）。

  **主要实验结果**：
  - DiJiang-410M vs Pythia-410M: 平均benchmark 0.456 vs 0.454，训练6.6天 vs 105.8天（~1/16），推理787 tokens/s vs 203 tokens/s（~3.9×）。
  - DiJiang-7B vs LLaMA2-7B: 平均benchmark 0.557 vs 0.565，训练数据40B tokens vs 2000B tokens（~1/50）。
  - DiJiang-2.8B vs Pythia-2.8B: 平均0.473 vs 0.478，训练37.1天 vs 593.3天（~1/16），推理284 tokens/s vs 34 tokens/s（~8.4×）。
  - 对比其他线性注意力（Pythia-410M fine-tuning）：DiJiang 0.4567（最佳），Performer 0.4183，Cosformer 0.4047，Linformer 0.3982，RetNet 0.3843。

## PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  论文系统性地研究了构建高性能小语言模型（1B-1.5B参数级）的三个核心维度：神经架构设计、参数初始化策略和模型优化策略。
  - **神经架构——Compact Tokenizer**：发现从大模型（100k词表）直接继承tokenizer对小模型不友好，embedding+head层占38.19%参数。通过频率分析发现top-48k词汇覆盖97.86%训练语料，压缩至48k词表使embedding+head参数占比降至18.07%，性能最优。
  - **神经架构——Architecture Tweak**：在1B参数量约束下探索depth/width/FFN expansion rate的影响。Spearmanr系数显示depth与性能相关性最高（0.528），expansion rate无明显线性关系。推荐20层depth、expansion rate 2.77的配置，兼顾性能与推理速度（V100上29.49 tokens/s）。
  - **参数初始化——Parameter Inheritance**：从PanGu-π-7B通过learnable binary mask继承参数。关键发现：(1) Layer Selection：中间层冗余度高，首尾2-3层对性能至关重要；(2) Intra-layer Selection：数据驱动的learnable mask优于L1/L2/Taylor等启发式方法。
  - **模型优化——Multi-round Training**：小模型容量有限导致严重catastrophic forgetting。提出基于loss的概率采样策略（p_i = exp(l_i)/Σ_j exp(l_j)），50%采样率的第二轮训练即可获得主要收益。同时探索batch size与learning rate缩放关系（推荐r=0.5，batch size < 4M为安全范围）。
  - 实验比较：(1) 消融实验在50B tokens子集上验证各组件，用ARC-E/HellaSwag/C3评估；(2) PanGu-π-1B Pro vs 原版PanGu-π-1B（平均提升8.87）；(3) PanGu-π-1.5B Pro vs Qwen-1.8B/Phi2-2.7B/Open-LLaMA-3B等SOTA小模型（在C-Eval/CMMLU/MMLU/AGI-Eval/BoolQ/AX-b/PIQA/EPRSTMT/XSum/C3十个benchmark上全面对比）。

- 硬件平台是什么，配置是什么。
  - 训练：华为昇腾910 (Huawei Ascend 910) 集群
  - 推理速度测试：单卡NVIDIA V100 GPU，FP16精度，batch size 20，测试生成510个新token（前缀2 tokens）的端到端速度
  - 实现框架：PyTorch，基于LLaMA-like架构

- 模型是什么。数据集和bench分别是什么。
  - 模型：PanGu-π-1B Pro（depth=21/width=1792/vocab=48k/expansion=2.77，总~1B参数）和 PanGu-π-1.5B Pro（depth=22/width=2048/vocab=48k，总~1.5B参数）。架构基于LLaMA-like Transformer，从PanGu-π-7B通过learnable binary mask继承参数
  - 预训练数据：1.6T tokens，中英文~1:1比例，来源为互联网多元语料；扩展版本PanGu-π-1.5B Pro*使用6T tokens
  - Benchmarks：使用OpenCompass框架评估。Examination: C-Eval、CMMLU；Knowledge: MMLU；Reasoning: AGI-Eval、BoolQ、AX-b、PIQA；Understanding: EPRSTMT、XSum、C3。消融实验使用ARC-Easy、HellaSwag、C3

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/YuchuanTian/RethinkTinyLM
  - 算法Pipeline核心四步骤：

  **Step 1 — Compact Tokenizer构建**
  ```
  输入: 大词表 V_large (100883 vocab), 训练语料 D (1.6T tokens)
  1. 统计 D 中每个vocabulary的出现频率 freq[v] for v in V_large
  2. 按频率降序排序 vocabularies，计算累积覆盖率:
     cum_coverage(k) = Σ_{i=1}^{k} freq[v_i] / Σ_{i=1}^{N} freq[v_i]
  3. 选择最小 k 使得 cum_coverage(k) > 97% → k=48k
  4. 使用 SentencePiece BPE 在 D 上训练 48k tokenizer
  5. 小模型使用新tokenizer，embedding层: 48k × d_model, head层: d_model × 48k
  输出: 紧凑tokenizer，参数占比从 ~38% 降至 ~18%
  ```
  关键公式：PEHL = (2 × V × d_model) / total_params，推荐 PEHL < 20%

  **Step 2 — 架构配置搜索与选择**
  ```
  约束: total_params ≈ 1B, vocab_size = 48k 固定
  搜索空间: depth ∈ [9,40], width ∈ [1280,2560], expansion_rate ∈ [1.0,4.0]
  过程:
  for each (depth, width, expansion_rate) in sampled_configs(30):
      model = build_llama_like(depth, width, expansion_rate, vocab=48k)
      train(model, data=5B_tokens)
      metrics = evaluate(model, [ARC-E, HellaSwag, C3])
      record(depth, width, expansion_rate, metrics.avg)
  相关性分析:
  Spearmanr(depth, performance) = 0.528  # 强正相关
  Spearmanr(width, performance) = -0.528  # 强负相关（因为width与depth在固定参数下互斥）
  Spearmanr(expansion_rate, performance) ≈ 0  # 无明显线性关系
  选择: depth=21, width=1792, expansion_rate=2.77 (PanGu-π-1B Pro)
        depth=22, width=2048, expansion_rate=2.77 (PanGu-π-1.5B Pro)
  ```

  **Step 3 — 参数继承 (Learnable Mask Pruning)**
  ```
  输入: 大模型权重 W_large ∈ R^{d_large × ...}（PanGu-π-7B），目标架构 A_small
  1. Layer Selection:
     对 W_large 的每一层 i ∈ [1, L_large]:
         测量跳过该层后的性能下降 Δperf(i)
     发现: 前2-3层和最后几层Δperf大（关键层），中间层Δperf小（冗余）
     策略: 保留 L_small 层 = 前3层 + 后3层 + 中间均匀采样(L_small-6)层
  2. Intra-layer Selection (Learnable Mask):
     对每层参数 W ∈ R^{d_out × d_in}:
         初始化二值mask M ∈ {0,1}^{d_out × d_in}，通过Gumbel-Sigmoid可微近似
         mask训练: min L_task(f_M ⊙ W_large) + λ · ||M||_1
         f_M = σ((log(u) - log(1-u) + log(α)) / τ)，τ anneal至0
     提取: W_small = extract_submatrix(W_large, where M=1)
  3. 证明有效性（Table 5）:
     Learnable mask: Avg = 48.08（最优）> Taylor: 47.90 > L2: 47.00 > L1: 46.06 > Base(随机初始化): 42.06
  输出: 初始化的tiny model权重 W_small，具有大模型的表征能力
  ```

  **Step 4 — Multi-round Training**
  ```
  输入: 预训练数据 D, 模型参数 θ, 训练轮数 R=2, 采样率 r=0.5
  Round 1:
     将 D 随机均分为 K=8 个part: D = {P_1, P_2, ..., P_K}
     顺序训练: for i=1..K: θ ← SGD_step(θ, P_i)
     记录每个batch的loss: L_k = {l_1, l_2, ..., l_N_k} for each P_k
  Round 2:
     对每个 P_k:
         计算归一化采样概率: p_i = exp(l_i) / Σ_{j=1}^{N_k} exp(l_j) 
         采样 r × N_k 个batch（困难样本被采样概率更高）
     合并采样数据为 D' = {sampled_batches}
     继续训练: for batch in D': θ ← SGD_step(θ, batch)
  ```
  效果（Table 6-7）：Single round Avg=51.61 → Two round r=50% Avg=54.46 (+2.85) → Three round Avg=54.44（饱和）

  训练超参数：Optimizer AdamW (β1=0.9, β2=0.95)；LR Cosine decay, initial LR=2e-4；Batch size 2M tokens；Weight decay 0.1


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

