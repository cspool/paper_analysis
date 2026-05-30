## FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

- baseline方法是什么？
  现有VLM（Gemini、Qwen-VL、PerceptionLM、NVILA）使用通用视频-文本数据预训练，在motion understanding上表现不足。这些模型的训练数据主要回答"what is this motion"（如识别"pouring water"），但缺乏"how this motion happens"的细粒度运动数据（如"pouring water from a bottle into a glass"的具体轨迹和空间关系）。手工标注motion数据极其昂贵：一个标注员需要几分钟标注3秒视频，10人团队需约100天完成100K视频。此外，现有motion benchmarks（MotionBench、FAVOR-Bench）关注细粒度motion recognition但忽视spatial reasoning（运动交互、相对轨迹、几何约束）。
  
  全栈执行例子（Gemini-2.5 Flash on MotionBench, 给定视频clip）：
  - **模型推理算法层**：VLM接收video frames作为多模态输入，通过visual encoder提取帧级特征，经过cross-attention/projection注入LLM，LLM自回归生成QA答案。模型可识别视频中"a car is moving"但无法正确回答"which direction is the car turning"（Gemini在MotionBench上仅55.6%）。
  - **系统框架层**：VLM inference通过standard transformer serving pipeline执行——视觉编码器处理frames、LLM处理多模态tokens。无特殊motion-centric pre/post-processing。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
  
  Baseline缺陷：
  - (a) 训练数据缺乏fine-grained motion annotations（只有"what"没有"how"），导致VLM的motion reasoning能力弱
  - (b) 手工标注motion数据成本过高，无法规模化
  - (c) 现有自动标注pipeline未提供结构化spatial signals（bbox轨迹、tracking信息），LLM仅从raw video生成QA质量差
  - (d) 缺乏覆盖多领域（驾驶、机器人、日常手部运动）的"how" motion benchmarks

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FoundationMotion全自动数据标注pipeline + motion-centric VLM fine-tuning**：
  
  **解决缺陷(a)(b)——automated pipeline替代人工标注**：设计四阶段全自动pipeline：(1) Video Preprocessing（temporal cropping + 相机运动过滤）；(2) Object Detection & Tracking（Qwen2.5-VL+GroundingDINO做开放词汇检测，Cascade Mask R-CNN+ViTPose++Hands23做人体/手部检测，SAM2做两阶段时序tracking）；(3) Caption Generation（将tracking JSON+bbox overlay+frames输入GPT-4o-mini，按7维度生成motion caption）；(4) QA Generation（从caption生成5类QA）。该pipeline在46.7K InternVid视频上自动生成467K QA pairs。
  
  **解决缺陷(c)——structured spatial signals注入LLM**：关键创新是在LLM生成caption/QA时，不仅输入raw video frames，还输入归一化bounding box trajectory JSON（包含每帧每个object的bbox坐标、object_type、interactions关系），以及color-coded bbox visual overlay。Ablation（Table 2）证明video+bbox JSON vs video-only在Fine-grained Action Accuracy上+2.6，Motion Detail+2.6，Temporal Coherence+2.4，Overall QA Quality从6.3提升到8.6（GPT-4评分，0-10）。
  
  **解决缺陷(d)——自建zero-shot "how" motion benchmarks**：手动标注四个跨领域benchmarks——AV-Car（NuScenes car motion, 1968 QAs）、AV-Hand（NuScenes hand motion, 108 QAs）、Daily（100 Days of Hands, 832 QAs）、Robotics（YouTube robot videos, 102 QAs）。
  
  全栈执行例子（FoundationMotion fine-tuned NVILA-Video-15B on AV-Car benchmark）：
  - **模型推理算法层**：fine-tuned VLM接收"the car is turning right"类QA输入。模型在FoundationMotion 467K training pairs上学到了bbox级spatial reasoning（通过tracking JSON中归一化bbox轨迹学习motion方向/距离/速度模式），可在AV-Car上从84.4%提升到91.5%（+7.1%），超越Gemini-2.5 Flash（84.1%）和Qwen2.5-VL-72B（83.3%）。
  - **系统框架层**：使用llamafactory（Qwen）和NVILA official training code做SFT fine-tuning。Training在8x A100 GPUs上进行，lr=1e-5(Qwen)/1.5e-5(NVILA)，cosine annealing，Adam optimizer。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。
