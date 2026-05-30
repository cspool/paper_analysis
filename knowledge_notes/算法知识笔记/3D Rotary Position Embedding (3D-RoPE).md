## 3D Rotary Position Embedding (3D-RoPE)

术语是什么？
3D-RoPE 将 Rotary Position Embedding 从 1D（语言序列）扩展到 3D（视频时空）。标准 RoPE 通过旋转矩阵对 Q/K 施加位置相关旋转变换，使 attention score 仅依赖相对位置。3D-RoPE 编码 (t, x, y)，对应时间维度和两个空间维度。OV-Encoder 使用 3D-RoPE 的核心原因是 Codec Patchification 产生的 token 布局高度不规则（不同 sample 选中的 patch 来自不同帧的不同位置），绝对位置编码无法在此类稀疏布局下保持一致性。3D-RoPE 的相对方案（Δp = (t1-t2, x1-x2, y1-y2)）天然适配。

从算法pipeline角度拆解术语：
频率分配 T:H:W=4:6:6（对应 16 attention heads）。三种 Δp 定义：
- Dense Video-Codec: Δp = (t_i-t_j, x_i-x_j, y_i-y_j)
- Chunk-wise: Δp = (c_i-c_j, x_i-x_j, y_i-y_j)
- Single-Image: Δp = (0, x_i-x_j, y_i-y_j)

```
# 在 ViT self-attention 前应用
def apply_3d_rope(q, k, Δp=(dt, dx, dy)):
    q[:, :d_t], k[:, :d_t] = rope_rotate(q, k, dt, freq_t)
    q[:, d_t:d_t+d_h], k[:, :] = rope_rotate(q, k, dx, freq_h)
    q[:, d_t+d_h:], k[:, :] = rope_rotate(q, k, dy, freq_w)
    return q, k
```

术语一般如何实现？如何使用？
与 Flash Attention 2 兼容（RoPE 是 pre-attention QK 变换）。三种输入共享同一 3D-RoPE 参数，推理时自动按输入类型选择 Δp 方案。使用场景：任何需要统一处理图像和视频的共享 ViT，特别是稀疏/不规则 token 布局。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence
