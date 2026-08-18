## VLA（Vision-Language-Action）模型与 DiT 动作规划器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLA 是把视觉、语言、动作三模态统一建模的具身智能模型（代表：RT-2、OpenVLA、Physical Intelligence π0.5、NVIDIA GR00T N1.5、Figure Helix）。主流 VLA 系统由两部分组成：vision-language model 做语义推理（对时延不敏感、可卸载云端），DiT 做动作规划（要求高动作频率，服务机器人 ≥50Hz、工业 ≥200Hz）。DiT 动作规划器的算法源头是 Diffusion Policy（Chi et al., RSS 2023 / IJRR 2025, arXiv:2303.04137）：把 visuomotor policy 表示为条件去噪扩散过程，动作序列经 receding-horizon 执行。本论文的动作输出为 7 DoF：平移向量 (ΔX, ΔY, ΔZ)（笛卡尔偏移）、旋转向量 (ΔΦ, ΔΘ, ΔΨ)（各轴朝向偏移）、夹爪状态 g（开/合）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VLA 闭环控制 pipeline（本论文 Fig.1）：观测（视觉帧 + 语言指令 + 当前状态）→ VLM 语义推理 → 多模态 token 输入 DiT 动作规划器 → 迭代去噪输出下一动作 → 机器人执行 → 环境新观测 → 循环。动作频率 = 1 / 单动作推理时延；任务执行时间 ≈ 动作总数 × 单动作时延（本论文评估中 actuation 时间 <0.3% 被忽略）。LIBERO 默认控制频率 20Hz、每个任务最多 520 环境步，因此 GPU 上动作频率只有几 Hz 时单个任务需数分钟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：常用开源 VLA（OpenVLA 7B、π0.5、GR00T N1.5）均在 GPU 上以自回归 VLM + 扩散动作头运行。本论文对 Dita（DiT 约 100M）、π0.5（DiT 约 300M、10 去噪步）、GR00T N1.5（约 500M、4 去噪步）三个模型评估 DiTPA 框架的通用性。使用场景：机械臂操作（抓取、开关、桌面整理）、导航 VLA（如 NaVILA）。Web 补充：Diffusion Policy arXiv:2303.04137；LIBERO benchmark（Liu et al., NeurIPS 2023）为 130 任务、5 套件、20Hz 控制频率。

- M100 补充视角（ISCA'26，车规 VLA 推理）：M100 SoC/NPU 面向自动驾驶（AD）、LLM 与智能人机交互三大域，VLA 是端到端 AD 的前沿（视觉感知、环境理解、动作规划），是 M100 设计的重要驱动之一。MindVLA 为理想自研下一代 AD 模型，集成 LLM 组件 + MoE（8 专家）transformer；UniAD 作为端到端 AD 基准（感知+预测+规划），其大量 query token（如 TrackFormer 900 query）提供充足并行机会，契合数据流架构。论文未涉及 DiT 动作规划器或机器人 VLA 的部署细节。
涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
