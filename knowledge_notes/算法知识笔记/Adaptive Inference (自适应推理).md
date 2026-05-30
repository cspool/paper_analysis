## Adaptive Inference (自适应推理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Adaptive Inference 是一种动态调整模型推理计算量的范式：根据当前资源约束（FLOP budget、延迟要求）、输入复杂度或期望准确率，动态选择推理配置（模型深度、token 数量、精度等），而非始终以最大计算量推理。Han et al. (TPAMI 2021) 将 Adaptive Inference 定义为"动态神经网络"的核心类别。AIM 将其引入多模态 LLM 领域：通过调节 Token Merging 的保留率 r_merge 和 Token Pruning 的 Scheduler 参数 (l₁, l₂)，实现从 2.5% 到 100% FLOPs 的连续可调范围，仅损失 <13% 准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**AIM Adaptive Inference 配置空间**：

```
# 自适应推理配置映射
configurations = {
    "extreme_efficiency": {"r_merge": 0.016, "l1": 14, "l2": 22},  # 2.51 FLOPs, 50.9 acc
    "high_efficiency":   {"r_merge": 0.031, "l1": 14, "l2": 22},  # 3.72 FLOPs, 52.3 acc
    "efficiency":        {"r_merge": 0.063, "l1": 14, "l2": 22},  # 6.17 FLOPs, 53.6 acc
    "balanced":          {"r_merge": 0.125, "l1": 14, "l2": 22},  # 11.14 FLOPs, 56.4 acc
    "default":           {"r_merge": 0.25,  "l1": 14, "l2": 22},  # 14.76 FLOPs, 58.2 acc
    "high_quality":      {"r_merge": 0.50,  "l1": None,"l2": None}, # 46.48 FLOPs, 58.5 acc
    "base_model":        {"r_merge": 1.0,   "l1": None,"l2": None}, # 99.63 FLOPs, 58.2 acc
}

def adaptive_inference(image_or_video, flop_budget):
    // 1. 根据 FLOP budget 选择最接近的配置
    config = select_config(flop_budget, configurations)
    
    // 2. Token Merging
    visual_tokens = merge_by_cosine_sim(visual_tokens, config.r_merge)
    
    // 3. Token Pruning with Scheduler
    for l in 1..L:
        visual_tokens, text_tokens = forward_layer(visual_tokens, text_tokens, l)
        if config.l1 and l >= config.l1:
            k = len(visual_tokens) * retention_ratio(l, config.l1, config.l2)
            visual_tokens = prune_by_pagerank(visual_tokens, k)
```

**自适应范围（Video, LLaVA-OV-7B）**：
- FLOPs span: 2.51 TB ~ 99.63 TB（40× 范围）
- Accuracy range: 50.9 ~ 58.5 VideoMME（<13% 降幅）
- Prefill time: 10.12 ms ~ 439.58 ms

术语一般如何实现？如何使用？

AIM 的自适应推理不需要修改模型权重，所有配置共享同一预训练模型。部署时根据目标设备（AR 眼镜、手机、PC、机器人）的计算资源选择配置。配置空间可预先采样并制表，运行时查表即可。代码开源：https://github.com/LaVi-Lab/AIM。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---
