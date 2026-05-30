## Owl-1__Omni_World_Model_for_Consistent_Long_Video_Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：Owl-1 (Omni World ModeL) —— 一个基于世界模型的长视频生成框架，通过建模底层世界演化来产生长期一致的长视频。核心创新包括：
  (1) **Omni World Model（全向世界模型）**：构建闭环的 state-observation-dynamics 三元组 (Eq. 1-3)，模拟世界的演化过程。latent state variable s_t 编码当前时刻和历史信息，通过 state decoder D（视频扩散模型）解码为显式视频观测 o_t；从观测和状态预测未来世界动态 d_t；动态再反过来更新状态变量 s_{t+1} = g(s_t, d_t)，形成自回归闭环。
  (2) **Comprehensive Condition from Latent State**：latent state s_t 由所有历史观测推导而来（Eq. 4），作为下一轮生成的综合条件，相比传统方法仅用 last-frame 条件，具有更长的时序感受野和长期一致性。
  (3) **Anticipation of Future Dynamics**：显式预测未来世界动态 d_t（文本形式），嵌入到状态演化中，解决传统方法重复生成同质内容的问题，提升内容的多样性和可控性。
  (4) **Multi-Stage Training（三阶段训练）**：
    - Stage 1 (Alignment)：冻结视频扩散模型，仅训练LMM，用MSE loss对齐 latent state s_t 与视频扩散模型 text encoder 的文本特征（Eq. 7）。
    - Stage 2 (Generative Pretraining)：联合微调LMM和视频扩散模型，用latent state s_t 替代原始text condition输入视频扩散模型进行denoising训练（Eq. 8）。
    - Stage 3 (World Model Training)：在带dense caption的长视频数据上微调，加入world dynamics prediction，用next-token prediction teacher-forcing监督（Eq. 9）。

  实验比较：
  (a) **VBench-I2V（Table 1）** —— 2s短视频，8个维度评估。对比 VideoCrafter-I2V、ConsistI2V、SEINE-512x512、I2VGen-XL、Animate-Anything、SVD-XT-1.0、DynamiCrafter-1024。Owl-1 Total Score=89.15，在 Motion Smoothness (98.92) 和 Temporal Flickering (98.69) 上表现优异。
  (b) **VBench-Long（Table 2）** —— 7s长视频，16个维度评估。对比 Mira、OpenSoraPlan、OpenSora、Mochi-1、CogVideoX、Kling、Vchitect-2.0、Gen-3、MiniMax。Owl-1 在 Subject Consistency (98.29) 和 Background Consistency (98.61) 上为 best/open-source best。
  (c) **定性可视化（Figure 4, 5, 6）** —— 展示 8s 通用视频生成和 24s (3 scenes × 8s) 世界模型驱动的长视频生成效果，验证 state variable 在跨场景 transitions 时的 consistency。

- 硬件平台是什么，配置是什么。
  训练平台：8 × NVIDIA A800 GPUs（每张80G显存）。Stage 1 (Alignment) 训练1天，Stage 2 (Generative Pretraining) 训练5天，Stage 3 (World Model Training) 训练1天，共约7天。框架：PyTorch。

- 模型是什么。数据集和bench分别是什么。
  模型：LMM = Chameleon model [31]，Video Diffusion Model = DynamiCrafter-1024 [35]。LoRA fine-tuning LMM（rank=8），全参数微调视频扩散模型。总可训练参数约2B（LMM LoRA ~798M + DynamiCrafter全参 ~1.2B）。Learnable state queries s_t 长度128。每视频分割为4s clips作为observation o_t，每clip采样2帧输入LMM。
  数据集：
  - General video generation: WebVid（400K random samples, 10M+ videos, 52K hours）+ Panda70m（2M random samples, 70M videos, 平均8s）
  - Dense video captioning: ActivityNet Captions（20K videos, 100K captions, 平均120s）+ Vript（12K high-res videos, 400K segments, 密集script标注）
  Benchmarks：
  - VBench-I2V: 8 dimensions（Video-Image Subject/Background Consistency, Subject/Background Consistency, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality, Temporal Flickering）
  - VBench-Long: 16 dimensions（含Subject Cons., Background Cons., Temporal Flickering, Motion Smoothness, Dynamic Degree, Aesthetic Quality, Imaging Quality, Object Class, Multiple Objects, Human Action, Color, Spatial Relationship, Scene, Appearance Style, Overall Consistency）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：Code https://github.com/huang-yh/Owl（论文明确提供GitHub链接）。Chameleon LMM和DynamiCrafter-1024均为公开模型。

  算法pipeline伪代码（Owl-1 推理生成流程）：

  ```
  # 输入: starting image I, text description d_0, num_clips N
  # 输出: long video consisting of N clips

  def owl1_long_video_generation(I, d_0, N):
      # Step 1: 用图像扩散模型生成首帧
      first_frame = SD2.1_v(I, d_0)

      # Step 2: 初始化latent state
      s_0 = LMM.encode_state(I, d_0, learnable_queries)  # shape: [128, dim]

      # Step 3: 生成首段视频clip
      o_0 = VideoDiffusion.decode(state=s_0, image_cond=first_frame, prev_obs=None)
      # VideoDiffusion: 以 s_0 替代原始text condition
      # noise_pred = epsilon_theta(o_{t,m}, m, s_0, None)
      # 通过L2 denoising loss优化后采样得到o_0

      # 构建初始LMM输入序列
      seq = [I_tokens, d_0_tokens, s_0_queries, o_0_visual_tokens, d_0_tokens]

      video_clips = [o_0]

      for t in range(1, N):
          # Step 4: 预测世界动态 (Eq. 2)
          d_t = LMM.predict_dynamics(s_{t-1}, o_{t-1})
          # d_t = f(s_{t-1}, o_{t-1}), teacher-forcing style next-token pred

          # Step 5: 更新latent state (Eq. 3)
          s_t = LMM.update_state(s_{t-1}, d_t)
          # s_t = g(s_{t-1}, d_t), 聚合历史信息

          # Step 6: 解码为视频观测 (Eq. 1)
          # 若跨场景切换则丢弃image_cond，仅用s_t作为条件
          if scene_transition:
              o_t = VideoDiffusion.decode(state=s_t, image_cond=None,
                                          prev_obs=o_{t-1})
          else:
              last_frame = o_{t-1}.last_frame()
              o_t = VideoDiffusion.decode(state=s_t, image_cond=last_frame,
                                          prev_obs=o_{t-1})
              # noise_pred = epsilon_theta(o_{t,m}, m, s_t, o_{t-1})

          # Step 7: 追加到序列
          o_t_tokens = VQVAE.encode(uniform_sample_key_frames(o_t, K=2))
          seq.extend([s_t_queries, o_t_tokens, d_t_tokens])
          video_clips.append(o_t)

      return concat(video_clips)
  ```

  张量计算流程（前向pass）：
  - LMM输入序列 Seq = [..., s_t_queries (128×dim), o_t_vq_tokens (2帧×N_tokens/帧), d_t_text_tokens, ...]
  - LMM对序列做causal self-attention，输出updated s_{t+1} queries 和 predicted d_t tokens
  - Video Diffusion Model接收 s_t (128×dim) 作为cross-attention condition，替代原始text embedding
  - Denoising: 从随机噪声 z_T 开始，每一步 z_{m-1} = denoise(z_m, m, cross_attn(s_t), concat_image_cond(o_{t-1})), 最终 z_0 = o_t
