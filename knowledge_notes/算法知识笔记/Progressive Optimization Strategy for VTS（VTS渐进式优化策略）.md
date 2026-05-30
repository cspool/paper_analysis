## Progressive Optimization Strategy for VTS（VTS渐进式优化策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Optimization Strategy 是 GroundVTS 的三阶段训练范式，解决将 non-uniform visual token distribution 引入预训练 Vid-LLM 时的两个核心挑战：(1) 分布偏移 — 预训练 LLM 在均匀 token 分布上训练，直接输入非均匀 token 导致注意力不稳定；(2) 训练不稳定 — VTS 的 STE 离散选择和 LLM 的连续优化存在冲突。三阶段设计：Stage 1 (VTS Warm-up, lr=1e-5, 1 epoch) — 冻结 LLM + Projector，仅训练 VTS (W_v, W_q, MLP_vts) → Stage 2 (Joint LoRA Adaptation, lr=2e-4, 2 epochs) — LoRA (rank=8, α=16) 微调 LLM + VTS + Projector 联合训练，LLaVA-Video-178K 数据集 → Stage 3 (Grounding Fine-tuning, lr=1e-4, 3 epochs) — 同 Stage 2 配置，Grounding-FT 70K 专用 VTG 数据。消融证明每阶段必需：跳 Stage 1 (仅 2+3) R1@0.7=30.5 vs full=34.2，跳 Stage 2 (仅 1+3) R1@0.7=15.2。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === 三阶段训练 ===
# Stage 1: VTS Warm-up
freeze(LLM, Projector)
for epoch in 1..1:
    loss = CE(LLM(VTS(Projector(V)), Q), target)
    loss.backward()  # 仅更新 VTS (W_v, W_q, MLP_vts)

# Stage 2: Joint LoRA Adaptation
unfreeze(Projector); add LoRA(LLM, rank=8, α=16)
for epoch in 1..2:
    loss = CE(LLM(VTS(Projector(V)), Q), target)
    loss.backward()  # 更新 VTS + Projector + LoRA

# Stage 3: Grounding Fine-tuning
for epoch in 1..3:
    loss = CE(LLM(VTS(Projector(V)), Q), target)
    loss.backward()  # VTG-specific tuning
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
batch_size=2/GPU, gradient_accumulation=4, AdamW (β1=0.9, β2=0.999)。LoRA 作用于 LLM attention Q/V 投影矩阵。三阶段理论依据: Stage 1 预收敛 VTS 采样分布 → Stage 2 在通用视频数据上适应非均匀分布 → Stage 3 在 VTG 专用数据上精调。该策略的渐进性体现为分布稳定化 → 跨模态对齐 → 任务特化。training details 详见论文 Table 13/14。CVPR 2026。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
