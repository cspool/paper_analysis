## ECO (Episodic COmpressor)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ECO (Episodic COmpressor) 是 HERMES 长视频理解框架的核心组件之一，受人类 episodic memory（情景记忆）认知机制启发。它是一种在线、training-free 的视频帧特征压缩算法：维护一个容量上限为 E 的 memory buffer M（存储 episode prototypes），按 window 顺序处理视频帧特征 $\mathcal{W}_k$。当新 window 到达时：(1) 若 buffer 有空间，直接追加；(2) 若 buffer 溢出，将 buffer 和新 window 临时拼接为 A，迭代执行：计算 A 中所有帧对之间的 cosine similarity，找最相似帧对 $(i^*, j^*)$，合并 $A_{i^*} = (A_{i^*} + A_{j^*})/2$（元素级平均），删除 $A_{j^*}$，直到 $\|A\| \le E$。ECO 的核心创新在于 **global similarity-based merge**：不同于 MA-LMM 仅合并相邻（temporally adjacent）帧，ECO 比较 memory buffer 内所有帧之间的全局相似度，使得无论两帧在视频中相距多远，只要内容相似即可聚合到同一 episode。位置编码（PE）在 ECO 前施加于帧特征以注入 temporal locality，防止跨时间段的乱序合并且保留时序连贯性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ECO 精确伪代码（论文 Algorithm 1）：
```
A = concat(M, W_k)                              # 临时拼接 buffer 和新 window
while ||A|| > E:                                # 超过最大 episode 数
    (i*, j*) = argmax_{i≠j} (A_i · A_j) / (||A_i|| · ||A_j||)  # 全局 cosine similarity
    A_i* = (A_i* + A_j*) / 2                    # 元素级平均合并
    A = A \ A_j*                                # 删除被合并的帧
M = A                                           # 更新 buffer
```
ECO 在 HERMES pipeline 中的位置：Video → ViT-G/14 Window Encoder → ECO（维护 episode memory）→ Episodic Q-Former（cross-attention to episodes）。默认参数：N=100 frames, window=10, E=20 episodes。PE 消融：移除 PE 后 MovieChat-1k accuracy 从 78.6 降至 77.3。ECO 隐式捕获事件频率：频繁出现的事件自然在更多帧中出现，因此更可能被合并/保留为强化原型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ECO 是 training-free 设计——在任意预训练 ViT 特征上直接操作，不需要额外训练。作为 plugin 模块插入现有 VLM：(1) 替换 MA-LMM 的 memory bank → accuracy +3.4%, latency -43%（Table 5）；(2) 插入 LongVA → latency -30%, GPU memory -46%（Table 3）；(3) 插入 LLaVA-OneVision → latency -35%, accuracy +0.67%（Table 4）。ECO 的全局 merge 策略使其相比 FIFO（77.1%）和 Random（76.9%）策略提升约 1.5%（Table 6: ECO=78.6%）。22 FPS on V100（接近实时），仅需 100 帧 vs MA-LMM 2048 帧。开源：https://joslefaure.github.io/assets/html/hermes.html。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding
