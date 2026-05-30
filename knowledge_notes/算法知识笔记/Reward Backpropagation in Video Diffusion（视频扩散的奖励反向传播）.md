## Reward Backpropagation in Video Diffusion（视频扩散的奖励反向传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reward Backpropagation 是一种利用可微分 reward model 直接优化扩散模型采样过程的后训练方法。与 RL-based 方法（如 DDPO, DPO）将采样视为 MDP 并用 policy gradient 优化不同，reward backpropagation 直接通过 reward model 反向传播梯度到扩散模型的去噪步骤。算法核心：从文本条件 c 生成视频通过采样过程 sample(theta, c, x_T)，解码后过 reward model R 计算分数，优化目标为 L(theta) = -E_c[R(sample(theta, c, x_T), c)]。为节省显存，只对最后 K 步保留计算图。EasyAnimate 针对 rectified flow + 3D Causal VAE 的关键适配：(1) K=10（rectified flow 下梯度 norm 小于 DDPM），(2) F=1（仅解码首帧计算 reward），(3) 使用 LoRA 微调，(4) HPSv2.1 + MPS 组合最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def reward_backprop(dit_lora, vae, reward_models, prompt, K=10, F=1):
    c = qwen2vl_encode(prompt)
    z_T = torch.randn(latent_shape)

    # Phase 1: T->K detach
    z = z_T
    for t in range(T, K, -1):
        v_pred = dit_lora(z, t/T, c)
        z = (z + v_pred * (1/T)).detach()

    # Phase 2: K->0 with grad
    for t in range(K, 0, -1):
        v_pred = dit_lora(z, t/T, c)
        z = z + v_pred * (1/T)  # 保留计算图

    # Phase 3: decode first frame + reward
    video_f1 = vae.decode(z[0:1])  # F=1
    reward = reward_hps(video_f1, prompt) + reward_mps(video_f1, prompt)
    loss = -reward
    loss.backward(); lora_optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
先驱工作 AlignProp/DRaFT（文生图, K=1 DDPM）和 VADER（视频, 多帧）未适配 DiT+rectified flow。EasyAnimate 首次将该方法适配到 DiT+rectified flow+3D causal VAE 架构。消融显示：(1) K=1 时训练不稳定，reward 骤降；(2) F>1 时视频 dynamics 退化和 reward hacking（背景 artifacts）；(3) HPSv2+MPS 组合 VBench Total Score 83.42%，Aesthetic Quality 69.48 为所有模型最高。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
