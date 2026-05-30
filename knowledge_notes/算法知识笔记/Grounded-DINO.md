## Grounded-DINO

术语是什么？
Grounded-DINO是将DINO（DETR with Improved Denoising Anchor Boxes）transformer检测器与grounded pre-training结合的开放词汇目标检测器，发表于ECCV 2024。核心设计是**三阶段tight fusion**：(1) Feature Enhancer阶段做deep early fusion（deformable self-attention + image-to-text/text-to-image cross-attention）；(2) Language-Guided Query Selection阶段选择与文本最相关的top-Nq图像特征作为decoder queries；(3) Cross-Modality Decoder阶段每层做self-attention→image cross-attention→text cross-attention→FFN。支持任意文本类别名作为输入，输出检测bbox和对应类别标签。Grounding DINO 1.5 Pro使用ViT-L backbone在20M+ grounding images上训练达到SOTA零样本检测性能。

从算法pipeline角度拆解术语：
在FoundationMotion的Open-Vocabulary Object Detection阶段（Sec 3.2.1）：
```
# Step 1: Qwen2.5-VL-7B生成场景中的object categories
O = Qwen2.5-VL.scene_analysis(video_frame[0])  # → {o1, o2, ..., on}

# Step 2: Grounded-DINO逐类别检测（非拼接所有类别）
for each category o_i in O:
    bboxes_i = GroundedDINO(image=video_frame[0], text_prompt=o_i)
    # text_prompt单独query每个类别 → one-to-one box-label alignment

# 输出: B_obj = {(bbox, class_label)_i} 定位到具体像素坐标
```

论文的ablation关键发现：使用Grounded-DINO per-class query（而非concat所有类）可强制one-to-one bbox-label对齐，提升检测质量。

术语一般如何实现？如何使用？
通过HuggingFace Transformers或官方GitHub仓库使用。输入图像+文本prompt（如"a red car. a person."），输出detection bboxes和对应的text-matched类别。支持batch推理。官方：https://github.com/IDEA-Research/GroundingDINO；HuggingFace: `grounding-dino` model。也支持TensorRT部署（Grounding DINO 1.5 Edge在NVIDIA Orin NX上达75.2 FPS）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
