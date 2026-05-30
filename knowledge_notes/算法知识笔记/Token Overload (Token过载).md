## Token Overload (Token过载)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Overload（Token 过载）是 D-CoDe 论文识别并命名的另一个核心挑战。它指：视频输入产生的 visual tokens 数量远超静态图像（即使经过压缩），超过了图像预训练 VLM 的有效处理容量，导致模型无法全面理解这些 token 中的信息。具体表现为性能随 token 增加先提升后饱和（plateau）——多余的 token 不仅无益甚至可能引入干扰。论文通过 EgoSchema 10-frame 实验（Figure 2b）量化了这一效应：随着保留的 visual token 数量增加（通过不同 top-k activation retention ratio 控制），baseline（vanilla LLaVA-NeXT）的 accuracy 先上升后趋于平台，而 Question Decomposition 变体的 accuracy 持续增长且与 baseline 的差距不断拉大。Token Overload 的本质不是"token 太多算不动"，而是"模型无法从超量 token 中提取全部有效信息"——即模型的理解容量成为瓶颈，而非计算容量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Token Overload 的量化机制：
```
# === Token Overload 现象（Figure 2b） ===
# 实验设置: EgoSchema, 10 input frames, LLaVA-NeXT 7B
# 变量: 不同 top-k retention ratio → 控制的 visual token 数量

retention_ratios = [0.1, 0.2, 0.3, ..., 1.0]
baseline_acc = [38.2, 40.1, 41.5, 42.0, 42.3, 42.3, 42.2, ...]  # 饱和
decomp_acc = [40.5, 43.2, 45.8, 47.5, 49.0, 50.2, 51.0, ...]     # 持续增长
# gap = decomp_acc - baseline_acc 随 token 增加不断扩大

# 原因分析:
# Baseline: LLaVA_NeXT(F_final, Q)
#   模型试图从大量 visual tokens 中一次性提取所有相关信息
#   → 注意力分散 → 超出模型理解容量 → 性能饱和
#
# D-CoDe: LLaVA_NeXT(F_final, A_sub_1, ..., A_sub_n, Q)
#   子问题引导模型每次关注视频的一个具体方面
#   → 注意力聚焦 → 多次 pass 覆盖全部语义 → 性能持续提升
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Token Overload 是一个概念性术语（问题定义）。D-CoDe 通过 Question Decomposition 来缓解：将复杂问题分解为聚焦子问题，引导模型每次关注视频的不同语义方面，从而在多个 pass 中"消化"超量 token，避免单次 pass 中的注意力分散。论文未讨论其他缓解 Token Overload 的方法（如增加 context length、使用 memory bank 等），因为这些方法通常需要训练或架构修改，而 D-CoDe 的目标是 training-free。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition
