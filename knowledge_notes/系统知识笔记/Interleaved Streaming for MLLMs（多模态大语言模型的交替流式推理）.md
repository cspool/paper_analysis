## Interleaved Streaming for MLLMs（多模态大语言模型的交替流式推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Interleaved Streaming 是当前 MLLM 流式视频理解的主流范式，表现为"mini-batch offline"式交替：先编码一段视频帧（prefill），再生成一段文本（decode），然后编码下一帧，再生成下一段文本，如此循环。代表性工作包括 LiveCC、VideoLLM-Online、Flash-VStream、TimeChat-Online 等。其根本局限在于：MLLM 的全局连续位置编码要求文本生成完成后才能确定下一视觉 token 的位置索引→prefill 和 decode 无法并行→在 safety-critical 实时场景（如辅助导航）中可能因生成长文本而错过危险检测。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Interleaved Streaming 执行流程（Wait-K 策略）
# 假设: m_i 视觉 token/frame, K=3 文本 token/step
#
# Position IDs: 全局连续 [0, 1, 2, ..., E_0, E_0+1, ...]
#       E_i = 上一轮最大位置索引

Step 0: [Prefill]  frame_0 → vision tower → V_0 tokens (pos: 0..m_0-1)
        [Decode]   generate A_0[0..2] (pos: m_0..m_0+2)
        → E_0 = m_0 + 2

Step 1: [Prefill]  frame_1 → vision tower → V_1 tokens (pos: E_0+1..E_0+m_1)
        [Decode]   generate A_1[0..2] (pos: E_0+m_1+1..E_0+m_1+3)
        → E_1 = E_0 + m_1 + 3

Step 2: [Prefill]  frame_2 (pos: E_1+1..)
        ...  # 严格串行交替

# 关键问题: V_{i+1} 的起始位置依赖 k_i（文本生成长度）
# k_i 不可预知 → V_{i+1} prefill 必须等 A_i decode 完成
# Attention 路径: ...A_i[t-1]→V_{i+1}[0]→V_{i+1}[1]...→A_i+1[0]
# 视觉 token 插入打断文本序列的连续 attention
```

Interleaved Streaming 的两个核心缺陷：
1. **文本连贯性破坏**：视觉 token 插入文本序列之间，打断连续的语义流→BLEURT 从 Offline 53.21 骤降至 44.11，流利度从 4.84 降至 2.84。
2. **对调度扰动极度敏感**：帧到达率或生成速率波动时（Random schedule），BLEURT 进一步降至 40.56，出现重复、碎片化、截断等严重质量问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Speak While Watching 论文通过在 Qwen2.5-VL 的原始连续位置编码上配置 `QWEN2_5_VL_VARIANT=interleave` 实现 Interleaved Streaming baseline。使用 wait-K=3 策略。这是论文中的对照方法，用于展示并行流式方法（GDPE/OSPE/GIPE）相对于现有交替流式的改进。开源实现：https://github.com/EIT-NLP/Speak-While-Watching。

涉及论文标题：
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models
