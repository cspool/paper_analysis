## Dynamic Tiling Vision Encoding (动态分块视觉编码)

术语解释
Dynamic Tiling Vision Encoding 是 DeepSeek-VL2 提出的高分辨率图像编码策略，将不同宽高比的高分辨率图像动态切分为多个固定大小（384×384）的 local tiles，配合一个全局缩略图 tile，通过共享的 SigLIP 视觉编码器处理所有 tile，在保持图像细节的同时控制视觉 token 总数。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Tiling 的核心替代了 DeepSeek-VL 的 hybrid encoder（SigLIP-384 粗粒度 + SAM-B-1024 细粒度，仅支持固定两种分辨率），通过以下步骤处理任意宽高比的图像：(1) 候选分辨率选择：从 C = {(m·384, n·384) | 1≤m,n≤9} 中选择最小 padding 面积的 (m*, n*) 作为目标分辨率；(2) 图像 resize + tile 切分：resize 到目标分辨率保持宽高比后 padding，切分为 m*×n* 个 384×384 local tiles + 1 个 384×384 global thumbnail tile；(3) 共享编码器：所有 tile 通过同一个 SigLIP-SO400M-384 编码（27×27=729 visual embeddings × 1152 dim/tile）；(4) Token 压缩：2×2 pixel shuffle 将每 tile 从 27×27 压缩到 14×14=196 tokens；(5) 序列构建：通过 <tile_newline> 和 <view_separator> special tokens 组织 global 和 local tiles 的 2D 空间结构。最大 tile 数 9×9+1=82，最大 visual tokens 约 82×196≈16,000（实际远小于此，因大多数图像不需要 9×9 tiling）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== Dynamic Tiling Algorithm ===
Input: image I (H, W), base_res=384, max_grid=9

// Step 1: Select best resolution minimizing padding
best_pad = inf
for m in 1..9, n in 1..9:
    scale = min(m*384/H, n*384/W)
    rH, rW = H*scale, W*scale
    pad = m*384 * n*384 - rH * rW
    if pad < best_pad:
        best_pad, m*, n* = pad, m, n

// Step 2: Resize and pad
I_resized = resize(I, (m*·384, n*·384), keep_aspect=True)

// Step 3: Tile generation
thumbnail = resize(I, (384, 384))  // global view
local_tiles = split_into_grid(I_resized, m*, n*)  // m*×n* tiles of 384×384

// Step 4: Encode each tile (shared SigLIP)
tiles = [thumbnail] + local_tiles  // 1 + m*·n* tiles
for tile in tiles:
    v = SigLIP(tile)               // output: 27×27×1152 = 729 embeddings
    v = PixelShuffle_2x2(v)        // 27×27 → 14×14, 196 tokens × 4608

// Step 5: Build visual token sequence
// Global thumbnail: append <tile_newline> per row → 14×15 = 210 tokens
// Local grid: (m*·14) × (n*·14) grid + n*·14 <tile_newline> row separators
// Full: [global_210] + <view_separator> + [local_grid]
// Multiple images (>2): disable dynamic tiling (use thumbnail only)
```
与 DeepSeek-VL 的 hybrid encoder 对比：dynamic tiling 统一使用单一 SigLIP 编码器（而非两个编码器融合），支持 1-81 个 tile 动态自适应（而非固定两种分辨率），视觉 token 数随分辨率线性增长（而非平方，因 SigLIP 使用 local attention）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic tiling 的实现思路广泛出现于高分辨率 VLM（如 LLaVA-NeXT, InternVL2, Qwen2-VL, NVLM）。共性：(1) 需要 vision encoder 支持 variable input chain（tile 数量可变）；(2) 训练时需做 image tile load balancing（不同图像 tile 数差异大，需在不同 data parallel rank 间均衡负载）；(3) 推理时需考虑 tile 数的性能影响（更多 tile=更多 visual tokens=更慢解码）。DeepSeek-VL2 多图场景（>2）禁用 dynamic tiling 正是出于 context length 和计算效率的考量。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
