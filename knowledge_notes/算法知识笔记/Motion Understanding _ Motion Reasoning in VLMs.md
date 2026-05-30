## Motion Understanding / Motion Reasoning in VLMs

术语是什么？
Motion Understanding（运动理解）指视觉语言模型对视频中物体运动、空间关系变化和时间动态的理解能力，区别于传统VLM的"what"理解（物体识别、场景分类、事件检测），Motion Understanding聚焦于"how"——物体如何运动（方向、速度、轨迹）、运动之间的空间关系（相对位置变化、几何约束）和时间顺序（哪个动作先发生）。现有benchmarks如MotionBench涵盖6类motion task（action recognition, temporal ordering, motion attribute等），但缺乏spatial reasoning维度（如何交互、相对轨迹、几何约束）。FoundationMotion通过5类QA覆盖：Motion Recognition、Action Order、Motion-related Objects、Location-based Motion、Repetition Count。

从算法pipeline角度拆解术语：
Motion Understanding的训练pipeline（FoundationMotion方式）：
```
# 数据生成端
video → [detection + tracking] → bbox_trajectories_JSON 
       → GPT-4o-mini(frames, bbox_json, overlay) → motion_caption (7维度)
       → GPT-4o-mini(frames, caption) → 5-type QAs

# 模型训练端
VLM_base (NVILA/Qwen-VL) + 467K motion QAs → SFT fine-tuning
# 评估：模型在motion benchmarks上的QA准确率
# e.g. MotionBench: 45.7% → 46.7% (NVILA-Video-15B → +FT)
# e.g. AV-Car: 84.4% → 91.5% (NVILA-Video-15B → +FT, +7.1%)
```

关键发现：fine-tuning 46.7K videos (467K QAs)即可显著提升motion understanding，证明高质量motion数据比模型规模更重要（15B fine-tuned超越72B base和Gemini-2.5-Flash）。

术语一般如何实现？如何使用？
通过motion-centric数据fine-tuning实现。方法包括：(1)构建motion QA数据集（人工标注或自动pipeline）；(2)SFT fine-tuning开源VLM（使用llamafactory或官方training code）；(3)在MotionBench、VLM4D等benchmarks上评估。训练配置：cosine annealing LR schedule、Adam optimizer、无weight decay。评估指标：多选QA准确率（4选项随机分布）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos
