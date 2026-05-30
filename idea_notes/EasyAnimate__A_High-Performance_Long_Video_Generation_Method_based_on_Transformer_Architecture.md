## EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture

- baseline方法是什么？
  Baseline 方法是基于 Diffusion Transformer 的 3D Full Attention 视频生成模型（如 CogVideoX、HunyuanVideo、OpenSora），使用 CLIP + T5 双文本编码器，DDPM 采样，以及按固定分辨率和帧数训练的 naive training strategy。

  Baseline（以典型 3D Full Attention DiT + CLIP/T5 为例）全栈执行例子：
  - 算法层：文本 prompt → CLIP 编码（限制 77 tokens）+ T5 编码 → 拼接视频 tokens → 48 层 DiT Attention（每层对所有 F×H×W tokens 做 3D full attention，计算复杂度 O((F·H·W)^2)）→ 多步 DDPM denoising → 3D VAE decode → 视频帧序列。生成 1024×1024×49 frames时，单卡 A100 需约 30 分钟。
  - 系统框架层：PyTorch 分布式训练（DDP/FSDP），naive training 下不同分辨率/帧数样本 token 数差异大，导致 GPU 利用率不均（部分 GPU 闲置等待）。
  - 编译框架层：论文未明确说明
  - kernel 调度层：FlashAttention 用于标准 3D attention，但无针对视频注意力的特殊优化
  - 硬件架构层：NVIDIA A100 GPU 集群

  Baseline 的缺陷：
  1. **3D Full Attention 计算复杂度随序列长度二次增长**：对于高分辨率长视频（1024×1024×49 frames），F×H×W tokens 产生的序列长度极大，full attention 的 O(N^2) 计算和 O(N^2) 显存需求使训练/推理成本极高。Spatial-temporal decoupled attention 虽降低复杂度但显著损害生成质量（受限于 3D 感受野）。
  2. **Naive training 导致 GPU 利用率不均**：不同分辨率和帧数的视频 token 数不同，在同一 batch 中导致不同 GPU 处理不同数据量，部分 GPU 提前完成后空闲等待，训练吞吐量低。
  3. **CLIP/T5 文本编码器能力有限**：CLIP 限制输入 77 tokens，T5 对复杂场景和细粒度文本理解不足，导致文本-视频语义对齐差。
  4. **生成视频与人类偏好偏差**：大规模 web 数据训练的扩散模型在美学质量、文本遵循度上不足，现有 reward 相关方法仅用于 U-Net + DDPM 架构，在 DiT + rectified flow 架构上未探索。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：EasyAnimate 通过四项设计分别解决上述缺陷：

  (1) **Hybrid Windows Attention** → 解决 O(N^2) 复杂度问题。提出 6 方向滑动窗口注意力（fhw/fwh/hfw/hwf/wfh/whf），将注意力头分为 6 组，每组沿不同 3D 方向重排 token 序列后执行滑动窗口注意力，仅需一次 FlashAttention 调用。然后将 window attention 层与 full attention 层交替排布（window attention 放在中间层 12-36），在保留全局上下文的条件下大幅降低计算量。1024 分辨率下训练加速 22.39%，推理加速 25.53%。

  (2) **Training with Token Length** → 解决 GPU 利用率不均问题。将相似 token 数的视频分组到同一训练 step，如 512^2×49 frames 与 768^2×21 frames token 数相近，同组训练。每次迭代训练的 token 数从 6.17M 提升到 13.63M（+120.91%）。

  (3) **Qwen2-VL-7B 文本编码器** → 解决文本理解不足。使用 MLLM 替代 CLIP/T5，支持多语言和长文本输入，通过 RMSNorm + FC 层处理文本特征以对齐视频特征 L2 norm。VBench Total Score 从 80.42% 提升到 81.57%。

  (4) **Reward Backpropagation + Rectified Flow** → 解决人类偏好偏差。使用 HPSv2.1 + MPS 可微分 reward model 组合，通过 LoRA 微调 DiT。关键适配：K=10（因 rectified flow 下梯度 norm 比 DDPM 小，仅优化最后一步不稳定）、F=1（因果 VAE 的首帧解码能力足够，多帧导致 dynamics 损失和 reward hacking）。VBench Total Score 从 81.57% 提升到 83.42%。

  对比 baseline 的全栈执行例子（EasyAnimate + Qwen2-VL + HWA + Reward BP）：
  - 算法层：文本 prompt（多语言）→ Qwen2-VL-7B 提取倒数第二层 hidden features → RMSNorm → FC 线性变换对齐 → 拼接视频 noised latents → 48 层 MMDiT（层 1-12 用 3D full attention 建立全局上下文 → 层 12-36 用 6-direction sliding window attention 降低计算量 → 层 36-48 用 3D full attention 维持稳定性）→ Rectified Flow 快速采样（比 DDPM 少步数）→ 3D Causal VAE 逐帧因果解码（缓存前帧 latent）→ 高质量视频帧序列。后训练阶段通过 Reward BP + LoRA 微调，每步只优化最后 K=10 denoising 步骤、只计算第一帧 reward。
  - 系统框架层：PyTorch 分布式训练 + TTL 策略（按 token 数分组每步样本，均衡 GPU 负载）
  - 编译框架层：论文未明确说明
  - kernel 调度层：FlashAttention（window attention 仍兼容 FlashAttention sliding window 参数）
  - 硬件架构层：NVIDIA A100 GPU 集群，训练耗时 1024×1024 下 59.79s/iter（vs full attention 77.04s/iter），推理 21.32s/iter（vs 28.63s/iter）
