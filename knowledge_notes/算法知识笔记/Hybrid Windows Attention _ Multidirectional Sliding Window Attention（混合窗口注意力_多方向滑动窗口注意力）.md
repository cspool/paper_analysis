## Hybrid Windows Attention / Multidirectional Sliding Window Attention（混合窗口注意力/多方向滑动窗口注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Windows Attention 是 EasyAnimate 提出的视频 DiT 注意力机制，通过交替使用 3D full attention 和多方向滑动窗口注意力（Multidirectional Sliding Window Attention），在降低计算复杂度的同时维持视频生成质量。核心组件 Multidirectional Sliding Window Attention 将注意力头分为 6 组，每组沿不同的 3D 维度方向重排 token 序列后执行滑动窗口注意力：(1) fhw（frame->height->width，默认顺序），(2) fwh（frame->width->height），(3) hfw（height->frame->width），(4) hwf（height->width->frame），(5) wfh（width->frame->height），(6) whf（width->height->frame）。仅需一次 FlashAttention 调用（而非 spatial-temporal decoupled attention 的多次），计算复杂度从 O(N^2) 降至 O(N x W)，其中 W 为窗口大小。在 48 层 DiT 中，中间层（12-36）使用 window attention，浅层（1-12）和深层（36-48）使用 full attention，兼顾全局上下文和计算效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def multidirectional_sliding_window_attention(Q, K, V, num_heads, window_size):
    # Step 1: 头分组 — 6组，每组 num_heads/6 个注意力头
    head_groups_Q = chunk(Q, 6, dim='head')
    head_groups_K = chunk(K, 6, dim='head')
    head_groups_V = chunk(V, 6, dim='head')

    # Step 2: 各方向 token 重排
    dirs = ['fhw', 'fwh', 'hfw', 'hwf', 'wfh', 'whf']
    for i, direction in enumerate(dirs):
        head_groups_Q[i] = rearrange(head_groups_Q[i], direction)
        head_groups_K[i] = rearrange(head_groups_K[i], direction)
        head_groups_V[i] = rearrange(head_groups_V[i], direction)

    # Step 3: 合并后单次 FlashAttention 调用
    Q = concat(head_groups_Q, dim='head')
    K = concat(head_groups_K, dim='head')
    V = concat(head_groups_V, dim='head')
    output = FlashAttention(Q, K, V,
        window_size_left=window_size // 2,
        window_size_right=window_size // 2)

    # Step 4: 恢复原始 token 顺序
    head_groups_out = chunk(output, 6, dim='head')
    for i, direction in enumerate(dirs):
        head_groups_out[i] = rearrange(head_groups_out[i],
                                       inverse(direction))
    return concat(head_groups_out, dim='head')
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hybrid Windows Attention 基于 FlashAttention 的 sliding_window 参数实现。FlashAttention 原生支持 local/sliding window 模式，通过设置 window_size_left/right 自动限制每个 query token 只关注窗口内的 key token。多方向的关键技巧在于通过 token 重排（rearrange）而非修改 attention 计算本身来模拟不同方向的滑动窗口。窗口大小 ablation 显示 H x W（空间分辨率对应的 latent 尺寸）是最优平衡点（FVD=352.3，推理 21.32s/iter）。在 1024 分辨率下，Hybrid Windows Attention 训练加速 22.39%，推理加速 25.53%。该设计可推广到其他需要 3D 注意力且序列较长的场景（如 point cloud、medical 3D volumes）。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
