## DiT（Diffusion Transformer，扩散 Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DiT 是用 Transformer 取代 U-Net 卷积骨干的扩散模型（Peebles & Xie, ICCV 2023, arXiv:2212.09748）。与传统 U-Net 扩散模型相比，DiT 把潜空间表示切成 patch 序列当作 token，用多个 attention + FFN block 堆叠建模，并通过 adaLN（adaptive LayerNorm）或 in-context 方式把 timestep 条件注入每一层。本论文语境下 DiT 是 VLA 系统中的动作规划器：给定视觉/语言/动作多模态 token 与噪声动作，迭代去噪输出 7-DoF 动作。与图像生成 DiT 的关键差异（论文 TABLE II）：动作规划 DiT 的计算层级为 trajectory-iteration-model 三级（图像 DiT 只有 iteration-model 两级）、模态多出 action/state 等、输入 token 长度仅 10^1–10^2（图像 DiT 为 10^2–10^3），这些差异使图像 DiT 的优化方法（如 Δ-DiT、BlockDance 式背景/轮廓跳过）无法直接迁移。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
动作规划 DiT 的推理 pipeline：一个任务轨迹 = 数百次动作推理（LIBERO-Long 平均 376 次）；每次动作推理 = 10–50 个去噪步（Dita 约 50 步、π0.5 约 10 步、GR00T N1.5 约 4 步）；每个去噪步 = 多个 attention + FFN block。伪代码（单动作生成）：
```
x_T ~ N(0, I)                          # 随机噪声动作
for t in [T, T-1, ..., 1]:
    h = concat(vision_tokens, language_tokens, action_tokens, x_t)
    for block in blocks:
        q,k,v = h @ Wq, h @ Wk, h @ Wv        # QKV 投影
        h = softmax(q @ k^T / sqrt(d)) @ v    # 自注意力
        h = FFN(h)                            # 两个 Linear + GELU
    eps = noise_prediction_head(h)
    x_{t-1} = (x_t - (1-alpha_t)/sqrt(1-bar_alpha_t) * eps) / sqrt(alpha_t) + sigma_t * z
action = x_0                              # 7 DoF: 平移3 + 旋转3 + 夹爪1
```
该 pipeline 的三个冗余来源（本论文核心观测）：轨迹级相邻动作高一致（55.2% 旋转变化 <2°、97.2% 平移 <1cm）、迭代级相邻步 attention/FFN 特征 >98% 相似且每步重复加载权重（60.1% 重复外部访存）、模型级 91.7% 多模态输入每步不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：本论文在 PyTorch 中基于 Dita 模型（arXiv:2503.19757）的 DiT action planner 实现，部署基线为 NVIDIA A40 GPU（INT8/PTQ4DiT），约 300M 参数 + 50 步的配置在 A40 上仅 2.6Hz。使用：作为 VLA 的动作生成端（vision-language 语义推理端可卸载云端、动作端要求 50–200Hz 实时频率），评估于 LIBERO-Long（20Hz 控制频率、最多 520 环境步）、CALVIN、SimplerEnv。Web 补充：DiT 原文 arXiv:2212.09748。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
