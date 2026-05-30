## Wait-K Streaming Policy（等待-K流式策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wait-K 是流式序列到序列任务中的经典调度策略，最早来自同声传译（simultaneous translation）领域。在 MLLM 流式视频理解中，Wait-K 策略规定：模型每接收一个视频帧后，必须生成恰好 K 个文本 token，然后才能接收下一帧。K 是超参数，控制感知和生成之间的交替节奏。K 越大→文本生成 batch 越大→吞吐可能更高→但感知延迟越大（新帧等待时间更长）；K 越小→响应更即时→但文本生成碎片化。Speak While Watching 论文基于 PE-Video 和 FunQA 的平均帧-文本 token 比例（≈3），设定 K=3。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Wait-K = 3 的流式推理调度（Interleaved baseline）
time  │  Action
──────┼─────────────────────────────────────────
t=0   │  receive frame_0 → prefill V_0 tokens
t=1   │  decode token A_0[0]
t=2   │  decode token A_0[1]
t=3   │  decode token A_0[2]  (K=3 tokens done)
t=4   │  receive frame_1 → prefill V_1 tokens
t=5   │  decode token A_1[0]
t=6   │  decode token A_1[1]
t=7   │  decode token A_1[2]
...   │  (continue for N frames)

# In Parallel Streaming (GDPE):
t=0   │  receive frame_0 → prefill V_0
t=1   │  decode A_0[0] | prefill V_1 (并行)
t=2   │  decode A_0[1] | prefill V_1 (继续)
t=3   │  decode A_0[2] | prefill V_1
t=4   │  decode A_1[0] | prefill V_2 (并行)
t=5   │  decode A_1[1] | prefill V_2
...   │  prefill 和 decode 持续重叠
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 Speak While Watching 论文中，Wait-K 通过以下方式实现：(1) 训练时——将完整视频和标注字幕按帧和 K 对齐，构造 interleaved 或 parallel streaming 训练序列；(2) 数据过滤——丢弃字幕 token 数 L < 视频时长×K（监督不足）或 L > 2×视频时长×K（退化为 offline）的样本；(3) 推理时——严格按 wait-K 调度（fixed）或随机扰动（Random，评估鲁棒性）。Wait-K 与位置编码策略正交：相同的 K 可配合 Interleave、OSPE、GDPE 或 GIPE 使用。K 的选取依赖任务特征——video description（短输出，视觉主导）和 video-CoT（长输出，均衡负载）可能有不同的最优 K。论文中 2fps × wait-K=3 → 平均 6 tokens/s 文本输出速率。

涉及论文标题：
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models
