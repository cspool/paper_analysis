## SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SLOWFAST-VGEN，一种双速学习系统，包含：(1) Slow Learning——基于 ModelScopeT2V 的 masked conditional video diffusion model，对前序视频 chunk 和语言 action 做条件生成后续 chunk；(2) Fast Learning——推理阶段的 TEMP-LORA 模块，通过将输入输出 latent 拼接后加噪去噪训练 LoRA 参数，在参数中存储情节记忆；(3) Slow-Fast Learning Loop——内层 fast learning 循环适配每个 episode 并积蓄 TEMP-LORA 参数，外层 slow learning 循环利用多 episode 数据更新核心权重。

  实验比较：
  - 视频生成质量：FVD、PSNR、SSIM、LPIPS vs AVDC/Streaming-T2V/Runway Gen-3 Turbo/AnimateDiff/SEINE/iVideoGPT
  - 长视频一致性：SCuts (PySceneDetect 场景切换数)、SRC (Scene Revisit Consistency，回访场景余弦相似度)
  - 长时规划：RLBench 机器人操作（物体归位距离）和 Minecraft 游戏导航（到预定义路径点距离）的 Dist 与 FVD

- 硬件平台是什么，配置是什么。
  慢学习训练：约 64 张 V100 GPU，batch size 128。推理和快学习：单张 V100 GPU。训练时冻结 VAE 和 CLIP Encoder，仅训练 UNet。

- 模型是什么。数据集和bench分别是什么。
  模型：基于预训练 ModelScopeT2V（latent video diffusion + 3D UNet + 时空 attention blocks）修改。UNet 参数记为 Φ（slow learning weights），TEMP-LORA 低秩矩阵 Θ 作为 fast learning weights，最终权重 W' = Φ + Θ。Slow learning rate 5e-6，fast learning rate 1e-4，LoRA rank 32，Adam 优化器，context window 32 frames。
  数据集：自采集 200k 视频配语言 action 标注，涵盖 5 个领域：Unreal Engine（Google 3D Tiles/Unreal City Sample/购买素材，Python 脚本自动化 agent 控制）、Minecraft（手动游戏录制键盘鼠标）、EPIC-KITCHENS（第一人称厨房日常）、Robot（OpenX-Embodiment + Metaworld + RLBench）、Driving（HDD + Unreal Engine 生成）。测试集从自采数据集中预留。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未明确说明代码开源（项目网站 slowfast-vgen.github.io 有 Code 链接但截至检索时未公开）。

  算法 pipeline（三层结构）：

  **第1层 — Slow Learning (Masked Conditional Video Diffusion)**：
  给定 fp 帧过去帧和 fg 帧待生成帧：
  ```
  z_{t,:fp} = z_{0,:fp}                                   # 条件帧不加噪
  z_{t,fp:(fp+fg)} = sqrt(ᾱ_t)·z_{0,fp:(fp+fg)} + sqrt(1-ᾱ_t)·ε  # 生成帧加噪
  z_t = concat(z_{t,:fp}, z_{t,fp:(fp+fg)})               # 拼接送入 UNet
  loss = ||ε - ε_Φ(z_t[fp:(fp+fg)], t, c)||²              # 仅在后 fg 帧计算 loss
  ```
  条件 c 为 CLIP 编码的语言 action text。VAE 编码视频到 latent，VAE 冻结。

  **第2层 — Fast Learning (TEMP-LORA)**：
  推理时逐 chunk 生成，每轮迭代 i：
  ```
  // 生成当前 chunk
  Y_i = (Φ + Θ_i)(X_i, C_i)              # X_i 是上一轮输出，C_i 是当前 action
  
  // 训练 TEMP-LORA 存储情节记忆
  X_i' = concat(X_i, Y_i)                 # 拼接输入和输出 latent
  z_t^{i'} = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε  # 全序列加噪（不保留 clean 条件帧）
  loss_Θ = ||ε - ε_{Φ+Θ_i}(z_t^{i'}, t)||²  # 全序列计算 loss，无文本条件
  Θ_{i+1} = Θ_i - α·∇_Θ loss_Θ           # 更新 LoRA 参数
  ```
  核心设计：抛弃原始 TEMP-LORA 的 input→output 格式，对拼接的全序列加噪去噪训练，强调记忆整个轨迹而非关注即时转换。

  **第3层 — Slow-Fast Learning Loop**：
  ```
  while not converged:
      D_s = ∅
      for each (x, episode) in D:
          初始化 Θ_0^e
          for i in 0..I-1:
              D_s = D_s ∪ {X_i^e, X_{i+1}^e, Θ_i^e}  # 收集 input/output/TEMP-LORA
              固定 Φ，更新 Θ_i^e（fast learning）
      for {X_i^e, X_{i+1}^e, Θ_i^e} in D_s:
          Φ_i^e = Φ + Θ_i^e
          基于 Φ_i^e(X_i^e) 和 ground-truth X_{i+1}^e 计算 loss
          固定 Θ_i^e，更新 Φ（slow learning）
  ```

  视频规划（Video Planning）：采用 UPDP 框架，将任务规划形式化为 text-conditioned video generation。ChatGPT 分解任务为子目标 → 每个子目标生成 video chunk → 逆动力学模型将连续帧转为可执行 action。

  关键张量维度（推断自论文描述）：
  - latent z_t: fp+fg 帧，每帧 latent 维度取决于 VAE 压缩率
  - LoRA ΔW = AB^T: A∈R^{m×r}, B∈R^{n×r}, r=32
  - 推理开销：wo TEMP-LORA 约 12.93s/sample, w TEMP-LORA 约 13.81s/sample（+6.8%）; 显存 9579MB vs 9931MB（+3.7%）
  - 可生成最长 1000 帧无明显失真和退化
