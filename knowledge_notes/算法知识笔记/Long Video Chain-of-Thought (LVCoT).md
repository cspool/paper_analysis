## Long Video Chain-of-Thought (LVCoT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Long Video Chain-of-Thought (LVCoT) 是 TDC 论文提出的训练无关（training-free）超长视频推理策略。当视频过长导致 LLM context window 无法容纳全部 token 时，LVCoT 将视频等分为 M 段（默认 M=3），每段独立进行 TDC 编码和推理，生成段级中间答案；所有段答案及时间戳拼接形成 "chain-of-thought" 推理链，最后基于全局视频和推理链生成最终答案。与 prior work 的区别：(1) Goldfish/StreamingLLM 通过 key frame selection 处理但破坏时序连续性；(2) VideoRecap 采用层级策略但限于 captioning；(3) VideoCoT 需训练数据且仅用于短视频。LVCoT 训练无关、任务无关，可应用于任意 MLLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LVCoT 推理流程 (training-free)
# 输入: long video V, question Q, M=3 segments
# 输出: final answer

segments = divide_equally(V, M)               # e.g., 720s → 3×240s
thoughts = []
for k, seg in enumerate(segments):
    F_TDC_seg = TDC_encode(seg)
    ans_seg = LLM(F_TDC_seg, Q)
    t_start, t_end = seg.time_range
    thoughts.append(f"From {t_start}s to {t_end}s: {ans_seg}")

# Global reasoning with accumulated CoT
F_TDC_full = TDC_encode(V)
chain = concat(thoughts)
final_answer = LLM(F_TDC_full, f"{Q}\n\nThought process:\n{chain}")
```
消融实验（Table 4e）：7B w/ LVCoT on MLVU: 63.9→64.1, VideoMME Long: 61.3→61.8 (+0.5)。增益随视频长度增加而增大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
核心开销：M+1 次 LLM forward pass（vs 1 次无 LVCoT）。适用场景：视频时长超过 LLM context window 可容纳 token 数时。局限性：有效性依赖 LLM 推理能力，模型未经 CoT 训练提升较小；论文指出未来方向包括训练模型更好利用此策略和建立更高效内存机制。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
