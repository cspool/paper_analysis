## SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

- baseline方法是什么？
  Baseline 方法分为两类：
  
  **(A) 动作条件视频生成模型**：AVDC（image-space diffusion，仅在 image space 做 video policy）、AnimateDiff（个性化 T2I 动画扩展）、SEINE（short-to-long 生成过渡帧）、iVideoGPT（交互式自回归 transformer）。这些模型通过在大规模数据上预训练（类比 slow learning）来构建世界模型，但受限于 context window（如仅 16 帧/4秒），无法记忆超出当前窗口的轨迹，导致长视频中 temporally distant frames 不一致。
  
  **(B) 长视频生成模型**：Streaming-T2V（conditional attention + appearance preservation module + video enhancer）是最先进的 long video generation 方法。它通过 anchoring 一个 anchor frame 来保持全局 context，配合 conditional attention 逐 chunk 生成。然而其 appearance preservation module 仅使用单一 anchor frame，无法存储完整的情节记忆（episodic memory），导致回访先前场景时一致性差。
  
  全栈执行例子（以 Streaming-T2V 为例）：
  - 算法pipeline层：给定 text prompt → 生成首帧 → 用 conditional attention module 以 anchor frame + 前 chunk 条件生成后续 chunk → 逐 chunk 串联成 long video。仅使用最后一个 chunk 和 anchor frame，超出窗口的内容被遗忘。
  - 系统框架层：PyTorch + Diffusers 生态。模型为 latent video diffusion model（基于预训练 text-to-video model），使用 3D UNet denoiser。推理时在 latent space 操作，不经过 pixel 编解码循环。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 VAE encoder/decoder + UNet 推理 kernel，FP16 推理。
  - 硬件架构层：论文中 baseline 实验均在 V100 GPU 上运行（统一比较），AVDC 需 image-space diffusion 故显存和推理时延更高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 SLOWFAST-VGEN，通过三层设计解决上述缺陷：
  
  **(1) Slow Learning — Masked Conditional Video Diffusion（解决 baseline A 的 slow learning 质量问题）**：基于 ModelScopeT2V 修改，对前序 chunk 做 masked conditioning（past frames 保持 clean 不做 denoising loss），生成后续 chunk。配合自采集的 200k 多场景 action-video 数据集，大幅提升动作条件生成的 FVD（514 vs baseline 最高 782）。
  
  **(2) Fast Learning — TEMP-LORA（解决 baseline B 的记忆存储问题）**：核心创新。在推理过程中，每当生成新 chunk（输入 X_i + action C_i → 输出 Y_i），将 X_i 和 Y_i 的 latent 拼接后添加噪声，通过去噪训练更新 TEMP-LORA 参数 Θ_i。与 Streaming-T2V 的 anchor frame 不同，TEMP-LORA 参数中存储了整个生成轨迹的情节记忆。关键设计选择：
  - 对拼接序列**全加噪**（不含干净条件帧），使模型学习整个 trajectory 而非局部 transition
  - 训练时**不含文本条件**，专注于轨迹记忆
  - 遵循 local learning rule：ΔW 仅依赖当前迭代的局部 input-output 对
  
  **(3) Slow-Fast Learning Loop（解决长时规划任务中需跨 episode 泛化的问题）**：内层 fast learning 循环在每个 episode 上快速适配并积累 TEMP-LORA 参数；外层 slow learning 循环固定 TEMP-LORA，利用多 episode 的 (input, output, Θ) 数据更新核心权重 Φ，实现从单 episode 记忆到跨 episode 技能泛化。
  
  全栈执行例子（SLOWFAST-VGEN）：
  - 算法pipeline层：输入初始帧 X_0 + action 序列 → 逐 chunk 生成：Y_i = (Φ + Θ_i)(X_i, C_i) → 拼接 X_i' = X_i ⊕ Y_i → 加噪去噪 train TEMP-LORA 更新 Θ_{i+1} → Θ 参数累积整个 trajectory 的情节记忆 → 后续 chunk 生成时 Θ 保留了之前场景信息（如回访 Loc1 时场景一致）。可生成长达 1000 帧无明显退化。
  - 系统框架层：基于 ModelScopeT2V（latent video diffusion，CLIP text encoder + VAE + 3D UNet with spatial-temporal blocks）。Slow learning 阶段冻结 VAE/CLIP，仅训练 UNet (Φ)。Fast learning 阶段在 LoRA 低秩矩阵 (Θ, rank=32) 上做推理时训练，不修改 Φ。Video Planning 采用 UPDP：ChatGPT→子目标分解→逐 chunk 生成 video→逆动力学模型→action 执行。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 diffusion model UNet 推理 + LoRA 低秩矩阵乘加。TEMP-LORA 训练时额外一次前向+反向传播（单 V100，仅更新 LoRA 参数）。推理 overhead 仅 +6.8% 时延（12.93s→13.81s），显存增加 +3.7%（9579MB→9931MB）。
  - 硬件架构层：64×V100 训练，1×V100 推理。SCuts 从 0.89（Streaming-T2V）降至 0.37，FVD 从 782 降至 514，SRC 从 91.02 提升至 93.71。

  关键差异对比：
  | 维度 | Baseline (Streaming-T2V) | SLOWFAST-VGEN (Ours) |
  |------|--------------------------|----------------------|
  | 慢学习 | Text-to-video 预训练 | Masked conditional video diffusion + 200k 多场景 action-video 数据 |
  | 快学习 | 无（仅 anchor frame + conditional attention） | TEMP-LORA 参数存储全轨迹情节记忆 |
  | 记忆范围 | 单一 anchor frame（丢失中间轨迹） | 完整 trajectory（逐 chunk 累积 Θ） |
  | 记忆形式 | 图像像素（anchor frame 始终可见） | 低秩参数（ΔW = AB^T, r=32） |
  | 训练/推理 | 纯推理（无推理时训练） | 推理时 fast learning + 可选 slow-fast loop fine-tuning |
  | 学习回环 | 无 | Inner fast learning + Outer slow learning |
  | SCuts | 0.89 | 0.37 |
  | 最长生成 | 未明确（chunk-by-chunk 但内容漂移） | 1000 帧无明显退化 |
