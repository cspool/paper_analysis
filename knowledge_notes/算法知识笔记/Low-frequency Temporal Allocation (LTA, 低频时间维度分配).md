## Low-frequency Temporal Allocation (LTA, 低频时间维度分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Low-frequency Temporal Allocation (LTA) 是 VideoRoPE 提出的 3D RoPE 频率分配策略，核心思想是将 Transformer head 的高维度（对应低频率 θ_n = β^{-2n/d}，即较宽的单调区间）分配给 temporal 维度，而将低维度（对应高频率，满足有限空间分辨率覆盖需求）分配给 spatial 维度（x 和 y）。这与 MRoPE 的分配策略相反——MRoPE 将低维（高频）分配给 temporal。LTA 的理论依据：(1) 空间维度受限于固定图像分辨率，高频足以覆盖所有空间位置的唯一编码；(2) 时间维度可无限增长（长视频），需要低频避免远距离位置产生"hash collision"——即 cos(θ_n·t) 在远距离上的周期性重复导致不同时间位置有相同 embedding。LTA 下 temporal 使用 θ_48..θ_63（d=128, β=10000），这些 θ 值极小（如 θ_63 ≈ 0.00011），在数千帧范围内几乎单调不减，确保不同时间的 temporal embedding 始终可区分。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# VideoRoPE LTA dimension allocation (d=128)
# MRoPE: dims[0:32]=t(高频), dims[32:80]=x, dims[80:128]=y
# VideoRoPE LTA: dims[0:48]=x,y interleaved(高频), dims[48:64]=t(低频)

# Frequency comparison (β=10000):
# θ_n = base ** (-2*n / d)
# MRoPE t: θ_0≈1.0, θ_15≈10000^(-30/128)≈0.115
#          → cos(θ_n·t) oscillates within 3000 frames
# VideoRoPE t: θ_48≈10000^(-96/128)≈0.001, θ_63≈0.00011
#          → cos(θ_n·t) nearly monotonic within 3000 frames

def compute_videorope_rotation(q, k, t, x, y):
    # dims 0-47: spatial (x,y interleaved), higher freq
    # dims 48-63: temporal t, LOW frequency (LTA)
    q_rot = rotate_spatial(q[:,:,:48], x, y, freqs=θ[:24])
    q_rot = rotate_temporal(q[:,:,48:64], t, freqs=θ[24:32])  # LTA
    return dot(q_rot, k_rot) / sqrt(d)
```
Annotations: LTA 的核心 insight 是 RoPE 中不同维度的频率决定了捕捉的依赖范围——低维（高频）捕捉局部相对距离，高维（低频）捕捉全局长程依赖。VideoRoPE 将低频分配给需要长程建模的 temporal 维度，将高频分配给范围固定的 spatial 维度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LTA 的实现仅需修改 RoPE position embedding 的 dimension allocation：在调用 RoPE rotation 前重新排布各维度组。无需修改 RoPE 旋转计算本身或 Transformer 结构。使用时注意：(1) x 和 y 的交错排列（interleaved）优于顺序排列，因交错保持 x/y 维度频率相似性，减少空间各向异性；(2) temporal 维度数量和频率 base β 影响对不同视频长度的外推能力；(3) LTA 与 FlashAttention 等优化兼容。VideoRoPE 论文的 V-NIAH-D 结果（+12.44 点 over MRoPE）验证了 LTA 的有效性。

涉及论文标题：
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding
