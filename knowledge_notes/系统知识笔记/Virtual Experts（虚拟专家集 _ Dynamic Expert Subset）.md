## Virtual Experts（虚拟专家集 / Dynamic Expert Subset）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Virtual Experts (VE) 是 SwapMoE 提出的核心系统概念：在 MoE 推理中，不在 GPU memory 中保留全部 E 个 experts，而是仅维护一个动态更新的紧凑 expert 子集（大小 k << E），称为 Virtual Experts。VE 的本质是一个 runtime-managed sliding window over the full expert set：根据输入数据分布的变化，重要性高的 experts 被加载到 GPU memory 中（swap in），重要性低的 experts 被驱逐到外部存储中（swap out）。VE 的 size（每层保留多少 experts）和 update frequency（多久刷新一次）由 offline 阶段的 profiling-guided memory planning 决定，以满足给定的 memory budget 约束并最大化 accuracy。VE 使得推理计算量从 O(E) 降为 O(k)，memory footprint 按 k/E 比例降低，同时由于 activation locality，大部分需要的 experts 始终在 VE 中。

从系统架构角度拆解术语：
在 SwapMoE 系统架构中，VE 是连接 offline planning 和 online serving 的核心运行时数据结构：

```
Offline阶段（一次性 per hardware+budget）:
  Profiling: per-expert memory/latency/loading_time
  DNN training: config → accuracy prediction
  Genetic Search → config* = {frequency*, [k_1*, ..., k_L*]}
  输出: VE 的 size (k_l per layer) 和 update frequency

Online阶段（每个 sample, batch_size=1）:
  For each sample X:
    # 推理（仅使用 VE）
    For each MoE layer l:
      router_logits = W_r[l] @ x
      masked_logits = router_logits + mask_penalty  # 非 VE expert → -inf
      probs = softmax(masked_logits)
      expert_idx = argmax(probs)  # 必定在 VE[l] 内
      y = expert[expert_idx](x)
      importance[l][expert_idx] += ||x|| * |probs[idx]| * ||E_idx||
    
    # VE 更新（每 frequency 个 sample）
    If sample_count % frequency == 0:
      For each MoE layer l:
        sorted = sort_by_importance_desc(importance[l])
        VE_new[l] = sorted[:k_l]
        async_load(VE_new[l] - VE_old[l])   # CPU/SSD → GPU via async stream
        release(VE_old[l] - VE_new[l])       # free GPU memory
```

硬件内存布局示例（SwitchT-64, memory budget 4.7 GiB, Jetson AGX）：
- 全部 experts: 64 × 12 layers = 768, 合计 14.2 GiB
- VE: k_l 平均值 ~8 (genetic search output), 合计 ~4.7 GiB
- External storage (CPU/SSD): 其余 experts 权重
- Runtime IO overhead: peak 40 MiB/s, mean 20 MiB/s

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) VE 以 `Dict[layer_id, List[expert_id]]` 数据结构管理；(2) Expert 参数在 GPU memory 中以细粒度 buffer 组织，支持独立加载/释放；(3) Importance scores 以 `Dict[expert_id, float]` 跟踪，每个 forward pass 后原子更新，跨 sample 用 EMA 累积；(4) Async loading 通过额外 CUDA stream 或 I/O thread 执行，与 main computation stream 并行；(5) 在 HuggingFace 中通过 hook MoE layer 的 `self.experts` 访问路径实现——仅迭代 VE 中的 experts。适用场景：边缘设备 MoE serving（Jetson Nano/AGX ORIN）、移动端 LLM 推理、消费级 GPU 上运行大型 MoE 模型。

涉及论文标题：
- SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget
