## Complementary Learning System in Video Generation (视频生成中的互补学习系统)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
互补学习系统（Complementary Learning System, CLS）是认知科学理论（McClelland et al., 1995），认为人脑由两个互补的学习机制组成：新皮层（neocortex）负责慢学习——通过大量经验逐步构建世界模型，支持泛化和决策；海马体（hippocampus）负责快学习——从单次经验快速编码情节记忆，支持快速适应和一致性保持。SlowFast-VGen 将此理论映射到视频生成系统：慢学习 = masked conditional video diffusion pre-training（类比新皮层），快学习 = TEMP-LORA 推理时训练（类比海马体）。数学对应：W' = W + ΔW = W_slow + W_fast = Φ + Θ，其中 Φ 为慢学习预训练权重，Θ 为快学习 LoRA 参数。LoRA 的 local learning rule（Δc(t) = x^μ(t)·y^μ(t)）与神经科学的 Hebbian-like local learning 对应。Slow-Fast Learning Loop 进一步模拟海马体-新皮层的记忆巩固过程：快速编码 → 离线整合 → 抽象为通用知识。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# CLS 映射到 SlowFast-VGen 的计算框架
# 
# 生物学 (McClelland et al. 1995):
#   新皮层: 慢速学习 → 结构化世界知识 → 泛化
#   海马体: 快速学习 → 情节记忆编码 → 快速适应
#   记忆巩固: 海马记忆 → 离线 → 新皮层整合
#
# SlowFast-VGen 计算类比:
#   Slow Learning (Φ):   预训练视频扩散模型 → 通用世界动力学
#   Fast Learning (Θ):   TEMP-LORA 推理时更新 → 单 episode 轨迹记忆
#   Slow-Fast Loop:      内层-快速编码 → 数据收集 → 外层-慢整合

# Local Learning Rule 对应:
# 生物学: Δc(t) = x^μ(t)·y^μ(t)   (Palm, 2013)
# LoRA:   W' = W + ΔW = Φ + Θ
#         Θ 更新: ΔΘ ∝ z_{0,i-1} ⊕ z_{0,i}  (仅依赖局部 input-output)
```

Annotations:
- CLS 理论解释了为什么仅 slow learning（预训练）不足以生成一致长视频：缺乏海马体式快速记忆机制，模型无法记忆超出当前 context window 的轨迹
- LoRA 的 local update 机制天然适配情节记忆：每个 episode step 的 ΔW 仅依赖于当前迭代的 input-output 对
- Slow-Fast Loop 的"离线整合"阶段对应：外层循环在任务完成后利用所有 episode 数据统一更新 Φ

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CLS 在 SlowFast-VGen 中作为理论框架指导系统设计，而非直接实现的计算组件。具体映射：(1) Slow Learning → 200k 视频数据上预训练 MCVD，构建通用世界模型（覆盖 Unreal/Minecraft/Kitchen/Robot/Driving 五大场景）；(2) Fast Learning → TEMP-LORA (rank=32) 推理时存储 episode 记忆，不修改慢学习权重；(3) Memory Consolidation → Slow-Fast Learning Loop 将多 episode 的 TEMP-LORA 记忆整合到核心权重 Φ。该框架的关键洞察：现有视频生成模型（WorldDreamer, Pandora, iVideoGPT 等）仅实现慢学习（预训练），缺失快学习能力，导致长视频一致性差。CLS 理论为"为什么需要双速学习"提供了认知科学基础。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation
