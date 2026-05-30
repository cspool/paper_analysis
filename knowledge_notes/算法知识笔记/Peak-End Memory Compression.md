## Peak-End Memory Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Peak-End Memory Compression（峰值-终点记忆压缩）是 LiveStar 中受认知心理学 "Peak-End Rule"（峰值-终点规则，由 Kahneman 等提出）启发的长视频记忆管理策略。人类在回顾经历时倾向于优先记住"峰值时刻"（最 intense 的体验）和"终点时刻"（最近的体验），而非均匀地记住所有时刻。LiveStar 将这一原理应用于在线视频推理：利用 SVeD 预计算的每帧 perplexity 作为"语义重要性"代理指标（低 PPL = 高重要性 = "峰值"），保留每个语义片段的终端字幕作为"终点"摘要，对超出窗口 W（默认 40 帧）的旧帧按概率剪枝，删除概率正比于 PPL_relative（该帧 PPL 与所在 clip 内最小 PPL 之比）和 elapsed_time。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在推理循环中的位置：
```
def peak_end_prune(frames, PPL_cache, window_W=40, current_time):
    kept = []
    for f in frames:
        if len(frames) - f.index <= window_W:
            kept.append(f)  # 窗口内帧全部保留
        else:
            # P_delete ∝ relative_PPL × elapsed_time
            ppl_rel = f.PPL / min(PPL_cache[f.clip])
            time_factor = (current_time - f.timestamp) / total_duration
            p_delete = ppl_rel * time_factor
            if random() >= p_delete:
                kept.append(f)
    return kept
```

效果（OmniStar-RNG）：Peak-End 压缩下 SemCor=3.19, TimDiff=1.91, FPS=3.82，优于 Uniform Dropout (SemCor 3.04) 和 FIFO Forgetting (SemCor 3.07, TimDiff 2.09)。关键原因：(1) Uniform 随机删除可能丢弃关键帧（-4.70% SemCor）；(2) FIFO 丢弃最早的历史事件字幕，损害 long-range temporal reasoning（TimDiff +9.42%）；(3) Peak-End 基于语义重要性选择性保留，同时保留终端字幕维护 narrative coherence。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现需要两个先决条件：(1) SVeD 在每个 decoding step 计算 PPL 并存储，作为帧重要性评分来源；(2) 语义片段边界已知（来自训练数据或在线检测）。剪枝操作发生在 KV cache 层面：被剪枝的帧对应的 KV cache 条目从 GPU HBM 中释放，配合 Streaming KV Cache 的双级缓存机制维持 cache 一致性。配置：W=40 frames (约 13.3s @3fps)，对 10+ 分钟视频持续推理时将 KV cache 大小维持在可控范围。

涉及论文标题：
- LiveStar__Live_Streaming_Assistant_for_Real-World_Online_Video_Understanding
