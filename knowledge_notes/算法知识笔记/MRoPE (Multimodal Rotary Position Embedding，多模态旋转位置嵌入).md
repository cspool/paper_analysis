## MRoPE (Multimodal Rotary Position Embedding，多模态旋转位置嵌入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MRoPE (Multimodal Rotary Position Embedding) 是 Qwen2.5-VL 提出的多模态位置编码方案，将传统 1D RoPE 扩展为 3D 编码，同时处理文本（1D）、图像（2D）和视频（3D）的位置信息。核心设计：将 hidden dimension per head 拆分为三个 section —— 时间（T, temporal）、高度（H, height）、宽度（W, width），分别用不同的 position ID 进行旋转编码。在 Qwen2.5-VL 中，mrope_section 典型配置为 [16, 24, 24]（head dim=80）。对各模态：(1) Text：三个分量使用相同 position ID；(2) Image：T 分量恒为常数，H/W 分量使用 patch 的 2D 坐标；(3) Video：T 分量按帧递增（Qwen2.5-VL 进一步将 T 分量与绝对时间对齐，使用 second_per_grid_t 参数），H/W 同 image。注意：Qwen2.5-VL 的 ViT encoder 仅使用 2D RoPE（H+W），3D MRoPE 仅应用于 LLM backbone 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MRoPE 的 position ID 计算（以 video 为例，基于 Qwen2.5-VL 文档）：
```
# Qwen2.5-VL MRoPE position ID assignment
# Video: T frames, each frame H_patches × W_patches

second_per_grid_t = 1 / (fps * tokens_per_second)  # 默认定时分辨率
# tokens_per_second default=2

for frame_idx in range(T):
    # Temporal ID: 绝对时间对齐
    t_pos = int(frame_idx * second_per_grid_t * fps)
    for h in range(H_patches):
        for w in range(W_patches):
            pos_id = (t_pos, h, w)  # (T, H, W)
            # 分别用三组频率进行 RoPE rotation
            # dim[0:16] rotated by t_pos
            # dim[16:40] rotated by h
            # dim[40:64] rotated by w
```

在 TimeLens 论文中，MRoPE 作为 timestamp encoding 的 baseline 方案被评估。TimeLens 发现 MRoPE 在 VTG 任务上表现不理想（Charades-TimeLens mIoU 仅 36.6 vs Interleaved Textual 的 48.3），原因可能是：(1) 需要对 LLM 的 RoPE 机制进行底层修改，难以在大规模重训中实用化；(2) position embedding 方式的时间感知精度不如显式文本时间戳。TimeLens 通过将每帧作为独立 image 处理并复制两份，完全绕过 MRoPE 机制实现公平消融对比。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MRoPE 在 Qwen2.5-VL / Qwen3-VL 中直接在 LLM attention 层实现：在计算 query-key 点积之前，根据每个 token 的 (t_pos, h_pos, w_pos) 三组 position ID 分别应用不同频率的 RoPE rotation，然后拼接。HuggingFace Transformers 中的实现通过 `apply_multimodal_rotary_pos_emb` 函数完成。使用时只需配置 `mrope_section` 参数和 position ID tensor，模型自动处理。TimeLens 发现绕过 MRoPE、使用 interleaved textual timestamp encoding 既简单又有效，在 VTG 任务上无需修改 LLM 底层结构。

涉及论文标题：
- TimeLens__Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs
- VideoRoPE__What_Makes_for_Good_Video_Rotary_Position_Embedding

**VideoRoPE 论文对 MRoPE 缺陷的分析**：VideoRoPE 识别出 MRoPE 的三个关键局限：(1) **频率分配问题**：MRoPE 将高频率（低维度 dims 0-31）分配给 temporal 维度 t，而高频率对应的旋转角 θ_n = β^{-2n/d} 具有短单调区间，cos(θ_n·t) 在远距离上周期性重复。当帧号从 0 到 3000 时，低维的 cos(θ_n·t) 多次经过零点产生"hash collision"——距离很远的位置有几乎相同的 temporal embedding，导致 V-NIAH-D 中的 distractor 帧可在 temporal 维度伪装为 needle 帧。(2) **空间非对称性**：每帧 visual token 从 (0,0) 到 (W-1,H-1) 排列，每帧最后 token 总停在 (W-1,H-1) 角形成"corner stack"，text-video 边界距离非对称。(3) **缺乏时间缩放**：所有维度 index increment=1，无法区分帧间时间间距和图像内空间间距的不同粒度。VideoRoPE 通过 LTA（低频时间分配）、DL（对角线布局）和 ATS（可调时间缩放）系统性解决这三个问题。
