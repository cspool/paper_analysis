## DINOv2 for Video Frame Representation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DINOv2 是 Meta AI 的自监督视觉预训练模型（Oquab et al., 2023），基于 ViT 架构在 1.42 亿张无标签图像上通过自蒸馏训练。输出高质量的通用视觉特征（无需微调），提供 ViT-S/B/L/g 多种规模。在 DIG 的 CAFS 中，DINOv2 用于逐帧提取 global feature 计算相邻帧语义距离以检测场景边界。选择 DINOv2 的理由：(1) 自监督训练（无需标注）→ 泛化能力强；(2) 语义鲁棒性——对光照/视角不敏感，对内容变化敏感（适合场景切变检测）；(3) 计算高效——单帧特征仅需一次 ViT 前向。DIG 使用 DINOv2 ViT-B (768-d features) 的 [CLS] token 或 average pooled features 作为 frame-level 表示。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# DINOv2 in CAFS
for frame in video:
    tokens = DINOv2_ViT(frame)        # patch tokens: (N, 768)
    V_i = tokens.mean(dim=0)          # global feature, 768-d
    V_i = V_i / ||V_i||_2             # L2 normalize
d_i = 1 - dot(V_i, V_{i+1})           # cosine distance for scene boundary
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
加载：`dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` 或通过 HuggingFace。在 DIG 中冻结使用，不做微调。计算开销：对 2fps 视频每帧 1 次 ViT-B 前向，CAFS 总耗时 20-30 min (8×A100)。DINOv2 也被 CurveStream 用于 CAS 模块的 curvature 特征提取。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management
- EVA__Efficient_Reinforcement_Learning_for_End-to-End_Video_Agent
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
