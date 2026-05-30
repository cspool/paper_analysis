## Training with Token Length (TTL)（基于Token长度的训练策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Training with Token Length (TTL) 是 EasyAnimate 提出的一种视频 DiT 训练调度策略，核心思想是将具有相似 token 数的视频样本分组到同一训练迭代（iteration）中，以均衡多 GPU 集群中各 GPU 的计算负载。在视频 DiT 训练中，不同分辨率和帧数的视频经 VAE 压缩后产生不同的 token 序列长度（如 512^2 x 49 frames 和 768^2 x 21 frames 有相近 token 数）。Naive training 随机组合样本时，同一 batch 内各样本 token 数差异大，导致不同 GPU 处理不同量数据，部分 GPU 提前完成后空闲等待。TTL 通过在每次迭代前按 token_length = (H x W x F) / (patch_size^2 x temporal_compression) 分组，确保同一 batch 内所有样本 token 数接近，均衡 GPU 负载。EasyAnimate 实验显示 TTL 使每次迭代训练的 token 数从 6.17M 提升到 13.63M（+120.91%），batch_size=256，分辨率=1024x1024，帧数=49。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Training with Token Length (TTL) 调度流程
# 输入: 混合分辨率的视频池

def training_with_token_length(video_pool, batch_size, num_gpus):
    # Step 1: 计算每个样本的 token 数
    for sample in video_pool:
        sample.token_length = compute_token_length(sample)

    # Step 2: 按 token_length 分组
    video_pool.sort(key=lambda s: s.token_length)

    # Step 3: 构建均衡 batch
    batches = []
    for group in split_by_token_length_ranges(video_pool):
        # group_A: 512^2 x 49f (~12544 tokens)
        # group_B: 768^2 x 21f (~13056 tokens) — 相近可混合
        shuffle(group)
        for i in range(0, len(group), batch_size * num_gpus):
            batches.append(group[i : i + batch_size * num_gpus])

    # Step 4: 训练 — 所有 GPU 负载均衡
    for batch in batches:
        train_step(batch)

# Naive: 6.17M tokens/iter  ->  TTL: 13.63M tokens/iter (+120.91%)
```

```
flowchart TD
    A["输入: 混合分辨率视频池"] --> B["计算每个样本 token 数"]
    B --> C["按 token_length 排序/分组"]
    C --> D["同组内 shuffle, 构建 batch"]
    D --> E["多 GPU 分布式训练"]
    E --> F["所有 GPU 几乎同时完成<br/>最小化空闲等待"]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TTL 是一种 DataLoader/Batch Sampler 层面的优化策略，无需修改模型架构或训练框架。实现方式：(1) 预处理阶段统计所有训练样本的 token 数并记录到元数据；(2) DataLoader 的 batch sampler 按 token 数分组采样；(3) 在 Progressive Training 不同阶段动态调整 —— 随着分辨率/帧数变化重新计算 token 分组。该策略特别适合包含多种分辨率和帧数的视频训练，在大规模训练（34M+ 样本）中显著提升 GPU 利用率。类似策略在 LLM 训练中也存在（按序列长度分组减少 padding），但在视频 DiT 中尤为关键，因为视频 token 数差异远大于文本。

涉及论文标题：
- EasyAnimate__A_High-Performance_Long_Video_Generation_Method_based_on_Transformer_Architecture
