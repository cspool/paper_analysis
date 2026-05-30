## Group-Decoupled Position Encoding (GDPE)（分组解耦位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GDPE 是 Speak While Watching 论文提出的三种打破位置连续性约束的并行流式策略中综合最优的一种。其核心思想是将 MLLM 流式推理中的视觉 token 和文本 token 分配为两个**独立的位置编码组**：视觉组从 pos_v=0 开始独立递增，文本组从 pos_a=0 开始独立递增，组间位置索引完全解耦。由于两组位置空间独立，视觉 prefill（新帧编码）和文本 decode（自回归生成）可以并行执行——新帧的位置索引仅依赖之前视觉 token 的数目（已知），不依赖当前文本生成的长度（未知）。训练时通过自定义 causal mask 确保：V_{i+1} 只 attend 到 V_{1..i}（视觉组内因果），A_i 只 attend 到 V_{1..i} 和 A_{1..i}（跨模态因果+文本组内因果）。仅需在 Qwen2.5-VL 上做少量 SFT（20K 样本），无需修改模型架构权重。GDPE 在鲁棒性和语义质量之间取得最佳平衡——BLEURT 51.53、流利度 4.56，且对调度扰动（Random schedule）表现最鲁棒（BLEURT 51.76 甚至优于 fixed 的 51.53）。

同类策略对比：
- **OSPE (Overlapped Streaming Position Encoding)**：视觉和文本从同一 max 位置共享起始索引，文本段内连续但跨段非连续 → 流利度 4.48 低于 GDPE/GIPE。
- **GIPE (Gap-Isolated Position Encoding)**：GDPE 基础上两组间加固定 gap Δ → 流利度最高 4.85（接近 Offline 4.84），但语义捕捉略低于 GDPE。
- **GDPE**：综合最优，是最推荐的默认并行流式配置。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GDPE 位置编码分配（推理时视觉和文本可并行执行）
pos_v = 0  # 视觉组独立位置计数器（t 维度）
pos_a = 0  # 文本组独立位置计数器（t 维度）

for i in 1..N:  # N 轮流式推理
    # === 视觉流（可与文本流并行） ===
    V_i_tokens = vision_encoder(frame_i)    # m_i 个视觉 token
    for j in 1..m_i:
        PE_3D(V_i_tokens[j], x, y, t=pos_v)  # Qwen2.5-VL 3D RoPE
        pos_v += 1

    # === 文本流（可与视觉流并行） ===
    A_i_tokens = autoregressive_decode(causal_mask)  # 生成 k_i 个文本 token
    for j in 1..k_i:
        PE_1D(A_i_tokens[j], pos=pos_a)      # 三维坐标一致
        pos_a += 1

# Causal Mask（训练时已知完整序列）:
# V_{i+1} attend to V_{1..i} only（视觉组内因果）
# A_i attend to V_{1..i} + A_{1..i} only（跨模态因果+文本组内因果）
# 关键：A_i 不受后续 V_{i+1} 插入的影响，文本序列连续性完整

# 理论加速:
# Interleave: T_i = m_i/R_v + k_i/R_t（串行）
# GDPE/Parallel: T_i = max(m_i/R_v, k_i/R_t)（并行）
# 加速比 S = (m_i/R_v + k_i/R_t) / max(m_i/R_v, k_i/R_t) ≤ 2×
# r = (m_i/R_v)/(k_i/R_t)：r≈1 时加速最大(≈2×)；r>>1 时接近无加速
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/EIT-NLP/Speak-While-Watching。通过环境变量 `QWEN2_5_VL_VARIANT=group` 选择 GDPE 模式。代码修改 Qwen2.5-VL 的位置 ID 分配逻辑和 causal mask，无需修改模型权重。训练 `scripts/sft.sh` → checkpoint output → `eval.sh` 评估。重要说明：代码仓库不实现物理多 GPU 并行——位置编码层面的并行逻辑需配合系统级实现（双 GPU prefill/decode 流水线）才能获得完整加速。wait-K=3 配置（每帧生成约 3 个 token，匹配 PE-Video/FunQA 平均帧-文本比）。2fps 采样，20K 训练样本。适用于 Video Description 和 Video QA 流式任务。

涉及论文标题：
- Speak_While_Watching__Unleashing_TRUE_Real-Time_Video_Understanding_Capability_of_Multimodal_Large_Language_Models
