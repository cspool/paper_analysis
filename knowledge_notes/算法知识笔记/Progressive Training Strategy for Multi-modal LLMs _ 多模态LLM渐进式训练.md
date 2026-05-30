## Progressive Training Strategy for Multi-modal LLMs / 多模态LLM渐进式训练

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Training Strategy 是一种多阶段递进训练方法，逐步将预训练语言模型转化为多模态长上下文模型。LongLLaVA 的三阶段：Stage I (Single-image Alignment) — 仅训练 projector，冻结 vision encoder 和 LLM，~600K captions 对齐视觉-文本空间；Stage II (Single-image Instruction Tuning) — 训练 projector + LLM，~932K QA pairs 赋予单图指令跟随；Stage III (Multi-image Instruction Tuning) — 全面多图训练 (~700K instances)，配合 Replay 机制（从前阶段采样数据混入）防止 catastrophic forgetting。相比 Mixed Training（所有数据混合训练），Progressive Training 在 multi-image 上提升显著 (Mile 46.5 vs 42.2)，单图保持持平。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Three-stage Progressive Training
# Stage I: Alignment
model.freeze(vision_encoder=True, LLM=True, projector=False)
model.train(ALLaVA-Caption+ShareGPT4V, 600K captions)

# Stage II: Single-image Instruction Tuning
model.freeze(vision_encoder=True, LLM=False, projector=False)
model.train(LLaVA-1.5+Mantis-Single, 932K QA)

# Stage III: Multi-image Instruction Tuning
# + Replay (200K single-image + 50K pure-text) from Stages I/II
model.train(Mantis+VideoChat2+ShareGPT4Video+Replay+SubImage, 700K)
# packed to 176K tokens/sequence, <eos> separated, 1 epoch

# All: cosine LR, warmup 0.03, peak lr=1e-5, 3x8 A800 GPUs
```

Annotations: Stage I 仅对齐不训练 LLM；Stage III Replay 关键（w/o → SQA↓ +18.5%）；Text replay 50K 饱和；Single-image replay 随数据量持续改善。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LLaVA (Li et al., 2024a) 提出 Stage I+II；LLaVA-1.5 扩展 Stage II 为 instruction tuning；LongLLaVA 增加 Stage III 专门针对 multi-image。Qwen2-VL 和 InternVL2 使用类似渐进策略（分辨率递增多阶段）。消融：Stage I+II+III (progressive) Mile 46.5 vs I&II+III 44.2 vs I&II&III 42.2 (mixed)。Replay 消融 (Appendix F) 验证了其在防止 forgetting 中的关键作用。

ReVisionLLM 的渐进式训练为两阶段：(1) Stage 1 短片段训练——先用 dense features 微调 LLM (LoRA) 学习精确边界预测，再冻结 LLM 微调 Hierarchical Adapter 生成 sparse features，引入 Contrastive Segments（不含目标事件的负样本）训练存在性判断以校准置信度。(2) Stage 2 长视频训练——冻结 Hierarchical Adapter，仅微调新 LoRA 模块，利用 sparse features 识别小时级视频中的感兴趣段。与 LongLLaVA 的 image-level 渐进策略形成互补：LongLLaVA 渐进扩展图像数量（单图→多图），ReVisionLLM 渐进扩展视频时长（短片段→小时级）。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- ReVisionLLM__Recursive_Vision-Language_Model_for_Temporal_Grounding_in_Hour-Long_Videos
