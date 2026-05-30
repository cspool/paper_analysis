## Memory Budget Scheduler (内存预算调度器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Budget Scheduler 是 D2MoE 中管理端侧 MoE 推理 GPU 内存的运行时调度策略。它维护一个可配置的内存预算参数 M（表示 GPU 内存中可分配给 expert 权重的上限），动态决定哪些 expert 权重保留在 GPU 内存中、哪些应释放。

核心策略：
- 当加载新 layer 的 expert 权重时，检查 current_memory_usage > M
- 若超预算，优先释放**高 bit-width residual weights**（因为它们体积大且复用频率低）
- 若仍超预算，释放**低 bit-width base weights**（以复用率为代价）
- 低 bit-width 常驻权重（被高频率激活）倾向于保留在 GPU 内存中，避免重复加载

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
=== Memory Budget Scheduler (Algorithm 2 from D2MoE) ===
输入: 生成步数 n, layer 数 L, bit-width 种数 K, 内存预算 M

for i = 1 to n:           # 逐 token 生成
    for j = 1 to L:       # 逐 layer
        current_layer_mem = compute_memory(layer[j])
        
        if current_layer_mem > M:
            # Phase 1: 释放高 bit-width residual weights
            for k = K-1 down to 1:
                if layer[j-1] has bitwidth_k weights:
                    Free(layer[j-1][k])  # 释放上一 layer 的高 bit-width
                    Update(M)             # 回收内存
        
        if current_layer_mem > M:
            # Phase 2: 若仍不足，释放低 bit-width base weights
            Free(layer[j-1][0])  # 释放 INT2 base
        
        # Phase 3: 加载当前 layer
        Load_and_Store(layer[j])
        Update(M)
        
        # Phase 4: 执行 I/O-Compute Pipeline
        Execute_Bitwidth_Aware_Pipeline(layer[j])
```

**内存预算 M 的配置效果**（实验 Figure 10）：
- M=200MB：最低吞吐，频繁加载/释放 → 适合极端内存受限场景
- M=1600MB：接近 Hold-in-Memory-AWQ 的吞吐（89 vs 94 tokens/s）
- M 越大，更多低 bit-width 常驻权重 → 吞吐更高但内存占用更大

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
M 由用户根据可用 GPU 内存配置（如 RTX 3060 的 6GB）。D2MoE 框架的 Memory Budget Scheduler 运行在 PyTorch 层面，通过 Python-level 的内存管理跟踪 expert 权重在 GPU 内存中的布局。释放操作释放对应的 PyTorch tensor。在实际部署中，M 可动态调整以响应其他应用的内存需求（如多进程或多模型场景）。

Ablation study 显示 "+Budget" 贡献了 1.06×-1.21× 的额外吞吐提升（在 +Router +MWQ +HEBF 之上），因为它减少了因内存不足而频繁重新加载低 bit-width 权重的开销。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving
