## Continual Pre-training (CPT) of LLMs (大语言模型持续预训练)

术语解释
Continual Pre-training (CPT) 是指在已有预训练模型基础上，用大规模新数据（>100B tokens）继续训练以扩展模型能力，而非从头重新训练。与 fine-tuning 的关键区别在于数据规模：fine-tuning 通常使用 MB~GB 级数据，CPT 使用 >100B tokens。CPT 已经在 dense LLM 中被证明可以有效替代 full re-training（Ibrahim et al., 2024），但在 MoE 中的行为此前未被系统研究。

术语是什么？
CPT 的核心挑战是灾难性遗忘（catastrophic forgetting）：模型在学习新分布时丢失旧分布上的能力。Ibrahim et al. (2024) 建立了 dense LLM CPT 的三项核心技术：
1. **LR Re-warming + Re-decaying**：从衰减 checkpoint 开始时，需重新 warm up LR 到 η_max 再 cosine decay 到 η_min
2. **Infinite LR Schedule (CosineInf)**：预训练时就使用不终止的 LR 方案，CPT 时从 constant phase 平滑过渡，无需 re-warming
3. **Replay**：CPT 时混合一定比例的旧数据以减缓遗忘

本文首次将这三项技术应用到 MoE，并证明了 MoE 在 CPT 中：1) 保持 sample efficiency 优势；2) 路由算法对分布偏移具有鲁棒性；3) 可以匹配 full re-training 性能（仅 ~1/3 成本）。

从算法pipeline角度拆解术语：
```
# CPT Pipeline (本文: FineWeb→German, 40% replay, CosineInf)
# Phase 1: Pre-training (FineWeb, 400B tokens)
model = init_moe_or_dense()
scheduler = CosineInf(total=192720, warmup=1%, const=10%, cooldown=70%)
for step in range(192720):
    batch = sample(FineWeb, bs=1024, seq=2048)
    loss = model(batch)
    optimizer.step()

save_checkpoint(model, phase="const", lr=1.65e-4)  # save at constant phase

# Phase 2: CPT (FineWeb→German, 200B tokens)
load_checkpoint(model, phase="const")
scheduler = CosineInf(total=95370, warmup=1%, const=80%)  # no cooldown yet
for step in range(95370):
    batch_replay = sample(FineWeb, bs=410, seq=2048)   # 40% replay
    batch_new = sample(GermanCC, bs=614, seq=2048)      # 60% new
    batch = concat([batch_replay, batch_new])
    loss = model(batch)
    optimizer.step()
```

术语一般如何实现？如何使用？
- **Compute-equivalent replay**：replay 比例增加时不增加总 token 预算，而是减少新数据量，保证不同 replay 比例的 compute 可比
- **Replay 比例**：本文 German CPT 用 40%，Stack CPT 用 30%（遵循 DeepSeek-CoderV2 的设定）
- **Overtraining regime**：本文 600B tokens 训练 570M/2B 模型，对应 dense 的 ~40× Chinchilla optimal，MoE 的 ~10×，代表真实应用场景
- **与 full re-training 的对比**：CPT 仅消耗 ~1/3 FLOPs（因为仅训练 200B 而非 600B tokens），但性能匹配或超越 full re-training

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
