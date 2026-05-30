## Expert Cache / Expert Prefetching

术语解释
Expert Cache是GPU显存中保留高频使用expert子集的缓存区域。Expert Prefetching是提前预测并异步加载即将需要的expert到GPU cache中的技术，以隐藏加载延迟。

术语是什么？
Expert Cache管理策略：
- **LRU策略**（Mixtral-Offloading、EdgeMoE等）：基于"最近使用的expert更可能再次被使用"的假设。在Mixtral-8x7B中，第i个token选中的expert有>10%的概率在第i+1个token仍被选中。
- **LFU策略**（MoE-Infinity）：基于请求级使用频率追踪，使用频率高的expert优先级高
- **静态重要性配置**（Fiddler）：使用静态数据集profile expert的活跃次数作为重要性度量
- **动态缓存更新**（SwapMoE）：基于定义的expert重要性分数动态更新缓存
- **自适应缓存大小**（AdapMoE）：不同层使用不同缓存大小（激活expert多的层分配更大cache）
- **混合精度缓存**（HOBBIT）：结合LRU + LFU + LHU（高精度优先）三维策略
- **Cache-aware Routing**（CacheMoE）：调整router logits使已缓存expert更可能被选中

Expert Prefetching策略：
- **跨层预测**：基于残差结构，用当前gate输入预测下层expert（准确率约90%）
- **预测表**（EdgeMoE）：校准数据集构建层间expert激活的统计相关表
- **学习型预测器**（ProMoE）：两层MLP预测器，滑窗预取（可预测第i+k层，k为窗口大小）
- **一次性全序列预测**（ExpertFlow/SiDA）：transformer/LSTM预测器一次性预测整个forward pass所需所有expert

从系统架构角度拆解术语。
以ProMoE的Expert Prefetching为例：
```
# 离线阶段：训练预测器
for iteration in range(num_iterations):
    # 正常MoE推理，收集训练数据
    for layer l:
        x_l = layer_input[l]
        experts_l = router(x_l).TopK(K)
        # 记录 (x_l, experts_{l+k}) 作为训练对
        training_data.append((x_l, experts_{l+k}))

# 训练预测器（两层MLP）
predictor = MLP([d_model, d_hidden, N_experts])
predictor.train(training_data)  # classification: predict expert active/idle

# 在线阶段：使用预测器
for layer l:
    # 预取第l+k层的expert
    predicted_experts = predictor(layer_input[l])
    async_prefetch(predicted_experts)  # 与当前层计算重叠
    # 正常计算第l层
    output_l = moe_layer[l](layer_input[l])
```

术语一般如何实现？如何使用？
- GPU显存中划分dedicated expert cache区域
- 异步预取需要非阻塞I/O支持（CUDA streams + CPU线程）
- 预测准确率直接影响端到端延迟（错误预测触发同步加载，成为瓶颈）
- 滑窗大小k需平衡预取提前量和预测准确率

**ExpertFlow的Routing Path Predictor (RPP)** 将预测提升到全局一次性层面：
- 使用T5-style encoder-decoder架构，在单次前向传播中预测所有token在所有MoE层的expert激活，输出形状 (B, S, L, E) 的概率矩阵
- 训练：离线收集30,000个(input, output, routing_path)三元组，使用binary cross-entropy多标签分类（loss: `(1/LE) Σ [r*log(p) + (1-r)*log(1-p)]`）
- 模型仅7.21 MB（FFN dim=2048, hidden size=32），跨域泛化仅下降5-10%准确率
- 与ProMoE逐层预测对比：RPP提前暴露完整routing plan，支持早期prefetch和全局调度

**ExpertFlow的Predictive Locality-aware Expert Caching (PLEC)** 将预测驱动自适应槽位分配引入缓存：
- 基于RPP预测自适应分配各层cache slot（如layer_1需求3 experts、layer_2需求2 experts、总容量4 → 分配3:1）
- 预取预测的最可能expert集合
- Runtime slot复用：early-layer expert完成后释放slot供后续层加载
- Real-time Correction：miss的expert通过异步加载与正在运行的expert compute重叠
- 在Switch-32上PLEC hit ratio 91.90%（CS=16, BS=4），比LRU高15-36%，在大batch下优势更大（CS=8, BS=16: PLEC 71.89% vs LRU 36.22%）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference

---
