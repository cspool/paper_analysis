## Clear/Blurred/Discard Memory States

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clear/Blurred/Discard Memory States 是 CurveStream 中 HVMM 模块定义的三种层级化视觉记忆状态，对应帧的不同保留策略和分辨率。这是一个三级的 resolution-aware 记忆路由方案：(1) Clear Memory —— 最高优先级记忆，对应 CS_t ≥ g2 或触发查询的帧 t_q。帧以原始高分辨率（base MLLM 的动态高分辨率策略）保留，存储精确的空间细节以支持后续细粒度视觉推理（如 OCR、属性识别、小物体定位）。Clear Memory 帧在记忆队列中作为"语义锚点"。(2) Blurred Memory —— 中间优先级记忆，对应 g1 ≤ CS_t < g2。帧被降采样到固定的 224×224 分辨率以大幅压缩 token 开销（论文中 TRANSITION_SIZE=224），保留必要的时序因果关联和动作连贯性，同时以极低成本维持连续帧之间的平滑过渡。(3) Discard —— 最低优先级，对应 CS_t < g1，帧直接被丢弃，不占用任何 memory bank 空间。三种状态的配置设计使 CurveStream 能在严格 N_max=20 的常值内存约束下，同时保持关键语义细节的高保真（Clear Memory ~50%）、动作连贯性的低分辨率覆盖（Blurred Memory ~50%）和冗余静态背景的零成本丢弃（Discard ~50% 总帧数）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种状态的路由逻辑：
```
if CS_t >= g2 or t == t_q:
    state = Clear Memory
    resolution = High  # base model 原生动态高分辨率
elif g1 <= CS_t < g2:
    state = Blurred Memory
    resolution = Low   # 固定 224×224 (TRANSITION_SIZE)
else:  # CS_t < g1
    state = Discard
    resolution = None  # 不编码

# 仅 Clear 和 Blurred 状态存入 M_t
M_t.append(I_t encoded at resolution)
```
Clear Memory 保留比例的消融实验（图 3b）：
- 100% Clear (所有帧高分辨率): 耗尽 memory bank，触发 catastrophic forgetting → accuracy 下降
- 0% Clear (全 Blurred): 丢失关键空间细节 → accuracy 急剧下降
- ~50% Clear (自适应 hybrid): accuracy 最优，token 成本降低 ~40%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 CurveStream 中，Clear Memory 帧使用 base MLLM（如 Qwen2.5-VL-7B）的动态高分辨率编码策略——即根据图像内容自动调整分辨率和 token 数量。Blurred Memory 帧统一降采样到 224×224 后编码，每帧产生的 visual token 数量显著减少（约减少 60-80%）。当 |M_t| > N_max 时，无论帧处于何种状态，均按严格 FIFO 顺序驱逐最旧 token——这种简单的驱逐策略避免了为不同类型 token 设计复杂的 eviction priority。与其他方法的对比：(1) uniform sampling 对所有帧一视同仁（无状态区分）→ 关键帧可能被逐出；(2) HERMES 使用 KV cache 被动逐出 → 无主动信息评估；(3) FreshMem 使用频率/空间域混合记忆 → 无分辨率区分。CurveStream 的 Clear/Blurred/Discard 三层级是首个将"分辨率感知"和"语义感知"同时融入记忆状态设计的方法。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management
