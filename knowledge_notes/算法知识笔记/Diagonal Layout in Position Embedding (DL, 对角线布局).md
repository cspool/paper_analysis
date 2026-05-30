## Diagonal Layout in Position Embedding (DL, 对角线布局)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diagonal Layout (DL) 是 VideoRoPE 提出的 3D 位置编码布局策略，将整个 multimodal 输入的 token 位置沿 3D 空间中的对角线排列。第 τ 帧视频的中心 patch 的 3D 坐标为 (τ, τ, τ)，其他 patch 按相对于中心的空间偏移排列——horizontal 偏移 ±(w-W/2)，vertical 偏移 ±(h-H/2)。text token 也沿同一对角线排列。核心目标是实现 spatial symmetry：T_v^start − T_pre = T_sub − T_v^end，使 visual token 从前后文本接收同等的上下文影响，减少模型对输入顺序的偏置。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Diagonal Layout position index calculation
for τ in range(T_s + T_v + T_e):
    if τ < T_s:  # preceding text
        t, x, y = τ, τ, τ
    elif τ < T_s + T_v:  # video
        center = T_s + δ * (τ - T_s)
        t, x, y = center, center + w - W/2, center + h - H/2
    else:  # subsequent text
        t = τ + (δ - 1) * T_v
        x, y = t, t

# Example (Ts=10, W=H=4, δ=2, 2 frames):
# Frame 0 center: (10, 10, 10) → patch(w=0,h=0)=(10, 8, 8)
# Frame 1 center: (12, 12, 12) → patch(w=2,h=2)=(12, 12, 12)
# → All patch centers near line y=x=t (diagonal)
```
Annotations: DL 保持相邻帧间对应位置 index 增量与相邻 text token 增量一致（对角线方向），保持了 Vanilla RoPE 的索引模式。MRoPE 的 non-diagonal layout 导致 frame 0 所有 patch 共享相同 t 值，且每帧最后 token 总在 (W-1,H-1) 处形成 corner stack。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DL 实现仅需在构建输入序列时按对角线公式计算 position IDs。与标准 RoPE fine-tuning 兼容。DL 对 spatial symmetry 的保证有理论优势：简化位置关系学习，减少输入顺序偏置。在 VideoHallucer 的 Object-Relation Hallucination 子任务上 +DL 提升 18.0 点 over MRoPE，体现了 DL 对空间关系理解的增强。需注意 DL 依赖 (δ, W, H) 参数，训练推理时保持一致。

涉及论文标题：
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding
