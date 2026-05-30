## 3D Rotary Position Embedding（三维旋转位置编码 / 3D RoPE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
3D RoPE 是将 Rotary Position Embedding 从 1D 序列扩展到 3D 视频数据的位置编码方法。RoPE 通过旋转矩阵将相对位置编码到 attention score 中：Q_m^T x R(m-n) x K_n。3D RoPE 将视频的时空维度 (T=temporal, H=height, W=width) 分别编码：将 hidden channels 按比例分配给三个维度（EasyAnimate 采用 3/8 temporal, 3/8 height, 2/8 width），各维度独立计算 1D RoPE 后拼接。模型可区分"同一空间不同时间"和"同一时间不同空间"的 token，捕获时空关系。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_3d_rope(video_tokens, d_model, frames, height, width):
    d_F = int(d_model * 3/8)  # temporal channels
    d_H = int(d_model * 3/8)  # height channels
    d_W = d_model - d_F - d_H  # width channels (2/8)

    rope_F = apply_rotary(video_tokens[:,:,:,:d_F], freqs_F, positions=t_idx)
    rope_H = apply_rotary(video_tokens[:,:,:,d_F:d_F+d_H], freqs_H, positions=h_idx)
    rope_W = apply_rotary(video_tokens[:,:,:,d_F+d_H:], freqs_W, positions=w_idx)

    return torch.cat([rope_F, rope_H, rope_W], dim=-1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
3D RoPE 继承 1D RoPE 的高效性：旋转矩阵稀疏（对角块），可通过 element-wise 乘法实现，计算开销极小。相对位置编码特性使模型对序列长度有更好外推能力。各视频 DiT（CogVideoX, HunyuanVideo, EasyAnimate）的 3D RoPE 在 channel 分配策略上可能不同，但核心机制相同。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

**VideoRoPE 论文对 3D RoPE 设计的系统分析**：VideoRoPE 提出好的 3D RoPE 应满足四个关键属性：(1) **3D Structure**——保留 (t,x,y) 时空结构而非 flatten 为 1D；(2) **Frequency Allocation**——temporal 维度应分配低频（高维），因为空间分辨率有界、仅需高频覆盖，而时间可无限增长、需要低频避免远距离"hash collision"；(3) **Spatial Symmetry**——preceding text end → visual start 的距离 ≈ visual end → subsequent text start 的距离，简化学习并减少位置偏置；(4) **Temporal Index Scaling**——temporal spacing 应不同于 spatial spacing（δ≠1），体现不同粒度的维度编码。

**Qwen2.5-VL 的 3D 位置编码**：Qwen2.5-VL 对视觉 token 使用显式三维位置编码 (x, y, t)，分别对应空间宽度、空间高度和时间维度。文本 token 的三维坐标保持一致（t 维度固定或为零），使文本 token 在空间维度上无区分。该设计允许模型在统一嵌入空间内联合推理空间、时间和语义上下文。在流式推理中，每个新到达的视频帧的视觉 token 按 (x, y, t) 坐标分配三维位置，其中 t 维度随帧序号递增。

**流式推理中的位置连续性约束问题**：Qwen2.5-VL 原生要求全局位置连续——所有 token 共享同一递增位置索引空间。在流式场景中，由于下一帧视觉 token 的起始位置依赖当前文本生成长度（不可预知），prefill 和 decode 必须串行交替执行，无法真正并行。这是本论文 Speak While Watching 识别并解决的核心瓶颈。解决方案包括三种打破连续性的位置编码策略（GDPE/OSPE/GIPE），详见对应术语条目。
