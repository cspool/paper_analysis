## Adjustable Temporal Spacing (ATS, 可调时间间距)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adjustable Temporal Spacing (ATS) 是 VideoRoPE 提出的时间索引缩放机制，引入超参数 δ 控制视频帧间的 temporal index 间距。核心 insight：视频中相邻帧的时间间距与图像内相邻 patch 的空间间距本质不同——帧间变化通常更大且帧率可变。ATS 通过将 temporal index increment 设为 δ（默认 2），而 spatial/text index increment 保持 1，实现维度特定的编码粒度。公式：t = T_s + δ(τ-T_s)，使帧间 temporal jump = δ。消融实验（Table 6）：δ=0.5→56.34, δ=1.0→59.11, δ=2.0→60.92 (best), δ=3.0→59.18（三个 benchmark 平均）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ATS with δ=2
for τ in range(T_s + T_v + T_e):
    if τ < T_s:
        t_pos = τ
    elif τ < T_s + T_v:
        t_pos = T_s + δ * (τ - T_s)    # δ=2 temporal scaling
    else:
        t_pos = τ + (δ - 1) * T_v       # compensate offset
# Frame 0 to Frame 1: t jumps T_s→T_s+2 (jump=2)
# Text token 0 to token 1: t jumps 0→1 (jump=1)
```
Annotations: δ 过大（如 3.0）过度分散 temporal index 破坏与 spatial 维度的配合，δ 过小（如 0.5）压缩时间信息使相邻帧难以区分。最优 δ=2 表示相邻帧 temporal distance 为相邻 text token 距离的 2 倍，合理反映视频时间变化的粒度差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ATS 实现仅需在计算 3D position IDs 时引入 δ 因子，无需修改 Transformer 结构。ATS 与 LTA 和 DL 协同工作：LTA 决定 t 的频率范围，DL 决定 position 的布局方向，ATS 决定 t 的 index step size。δ 值应根据视频帧率和任务特性调节：高帧率可能需要更大 δ 保持时间区分度。VideoRoPE 推荐 δ=2 作为通用默认值。

涉及论文标题：
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding
