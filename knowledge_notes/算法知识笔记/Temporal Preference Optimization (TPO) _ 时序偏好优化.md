## Temporal Preference Optimization (TPO) / 时序偏好优化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Preference Optimization (TPO) 是 Stanford 团队提出的 video-LMM 后训练框架，通过操纵视频输入自动构建时序偏好数据，使用 DPO 训练增强模型的时序定位（temporal grounding）能力。核心创新：不需要人工标注时序标签——仅通过改变视频输入的"可见证据量"（完整帧/部分帧/不相关帧），让同一 video-LMM 对同一问题产生质量有差异的回答，自动形成 preferred vs dis-preferred 偏好对。TPO 框架分三步：(1) Temporal Preference Modeling — 采样视频帧集合 F，用 CogVLM2 生成逐帧 caption → GPT-4o-mini 基于 caption 生成问题 Q → 使用 Q + 完整帧 F 生成 preferred response r⁺ → 使用 Q + 不相关帧或部分帧生成 dis-preferred response r⁻；(2) LLM-based Post-Filtering — GPT-4o-mini 过滤三类噪声：r⁻ 优于 r⁺、r⁺ 事实错误、问题模糊；(3) DPO + SFT 联合训练 — 使用偏好数据对 (V, Q, r⁺, r⁻) 进行 DPO 优化。两种 dis-preferred 生成策略：(a) Irrelevant Information — 完全排除相关帧；(b) Incomplete Information — 仅用部分相关帧。最优数据混合比例为 5:5。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TPO 完整 pipeline 伪代码：
```
输入: 视频集合, video-LMM π_θ, 参考模型 π_ref
输出: TPO 优化后的 video-LMM

# === Phase 1: 时序偏好数据生成 ===
For each video V:
    F = sample_frames(V)                   # 采样帧集合
    captions = CogVLM2.caption(each frame in F)
    Q_set = GPT-4o-mini(captions, task_prompts)

    For each q in Q_set:
        # Preferred: 使用完整相关帧
        r⁺ = video_LMM(V[F], q)

        # Dis-preferred (a): 不相关帧
        F_irr = sample(V \ F)
        r⁻_irr = video_LMM(V[F_irr], q)

        # Dis-preferred (b): 不完整帧
        F_inc = random_subset(F, ratio=0.5)
        r⁻_inc = video_LMM(V[F_inc], q)

        # Post-filtering (GPT-4o-mini, 3 条规则)
        for each (r⁺, r⁻) in [(r⁺, r⁻_irr), (r⁺, r⁻_inc)]:
            # Rule 1: r⁺ 是否优于 r⁻?
            # Rule 2: r⁺ 是否与 caption 矛盾?
            # Rule 3: r⁺ 基于 caption 是否正确的?
            if passes_all_rules:
                D.add((V, q, r⁺, r⁻))

# === Phase 2: DPO + SFT 训练 ===
For each batch (V, q, r⁺, r⁻) in D:
    # DPO loss (公式 2)
    L_DPO = -log σ(β · (log π_θ(r⁺|V,q)/π_ref(r⁺|V,q)
                      - log π_θ(r⁻|V,q)/π_ref(r⁻|V,q)))

    # SFT auxiliary loss (公式 3)
    L_SFT = -log π_θ(r⁺ | V, q)

    # Combined loss (公式 4)
    L = L_DPO + α · L_SFT

    θ ← θ - η · ∇_θ L  # full fine-tuning (LM+Projector)
```
超参数：LongVA-TPO: β=0.3, α=0.5, lr=4×10⁻⁶; LLaVA-Video-TPO: β=0.2, α=1, lr=3×10⁻⁷。8×A100 80GB, batch_size=64, 1 epoch, ~4h。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/ruili33/TPO。数据集和 checkpoint：https://huggingface.co/collections/ruili0/temporal-preference-optimization-67874b451f65db189fa35e10。使用流程：(1) 准备视频数据集（论文中手动 curator 200 关键词，爬取 8000 个互联网视频）；(2) 运行数据生成 pipeline（CogVLM2 captioning → GPT-4o-mini question generation → video-LMM response generation → post-filtering），生成 10K 偏好数据对；(3) 使用 DPO+SFT 联合损失训练（基于 TRL 或自定义脚本）；(4) 用 lmms-eval 评测 Video-MME/LongVideoBench/MLVU。消融发现：(a) TPO 随输入帧数增长性能持续提升（baseline 在 >64 帧退化），(b) 数据量 2K→10K 持续改善，(c) post-filtering 一致改善，(d) Incomplete:Irrelevant = 5:5 混合最优。关键优势：无需人工标注、可扩展（自动生成偏好数据）、即插即用（不修改推理架构）、可迁移到不同 video-LMM backbone。性能：LongVA-TPO 在 LongVideoBench +2.9%、MLVU +3.1%、Video-MME +2.5%。LLaVA-Video-TPO 成为 Video-MME 7B 模型 SOTA (71.5 w/ subs)。

涉及论文标题：
- Temporal Preference Optimization of Large Multimodal Models
