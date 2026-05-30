## LMM-based Reward Assignment (大模型奖励评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LMM-based Reward Assignment 是 DIG 提出的帧-查询相关性评估方法，直接使用 LMM 本身对视频帧进行 relevance scoring，替代传统的 CLIPScore 或 object detection。核心流程：将候选 r-frame 和查询送入 LMM → CoT 推理 → 输出二维评分：(a) 帧对回答查询的直接有用性，(b) 帧是否暗示相邻帧包含补充信息 → reward ∈ [0, 100]。与传统 CLIPScore 的关键区别：(1) 语义深度——LMM 可理解复杂推理逻辑，非仅表面特征匹配；(2) 世界知识——利用预训练常识识别 CLIP 无法捕获的隐含关联；(3) 上下文感知——二维评分使 LMM 能评估帧的"指示价值"。DIG Table 2 证明 LMM reward (Qwen2.5-VL-7B/32B) 在所有 frame 配置下一致优于 CLIPScore，且更强的 LMM (32B vs 7B) 提供更好的 reward quality。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LMM Reward Assignment 流程
for each r_frame in r_frames:
    prompt = f"""
    Frame: <{r_frame}>; Query: <{Q}>
    1. Describe the frame, focusing on relevant elements.
    2. Assign reward 0-100 based on:
       (a) Direct usefulness for answering
       (b) Whether adjacent frames may supplement
    Output: {{"description": str, "reward": int}}
    """
    response = vLLM_inference(LMM_rewarder, r_frame, prompt)
    rewards.append(response["reward"])
```
与 CLIPScore 的对比（Table 2, 128 frames, LVB）：
- CLIPScore: 61.0% → LMM 7B: 63.1% → LMM 32B: 65.2%
- Gain 来自 LMM 的语义推理和二维评分设计

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMM Reward Assignment 使用 vLLM 后端加速，且 rewarder LMM 可与推理 LMM 解耦（如用 Qwen2.5-VL-32B 做 reward，Qwen2.5-VL-7B 做最终推理）。计算开销：reward assignment 是 DIG 中最耗时阶段（占总选择时间 ~70%）。局限性：(1) 额外推理成本；(2) 长视频 r-frames 多时耗时显著；(3) reward 准确性受限于 rewarder LMM 能力。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
