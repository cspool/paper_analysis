## Task-aware Expert Loading in MoE (MoE 中的任务感知 Expert 加载)

术语解释
Task-aware Expert Loading 是一种根据任务类型对 token-to-expert routing accuracy 的敏感度来选择性加载 expert 的优化策略。发现某些任务（如 semantic classification / comparison）即使使用 inaccurate expert routing 仍保持 >90% 输出相似度，因此在 expert loading 时仅对 routing-sensitive 任务加载精确预测的 expert，对 insensitive 任务跳过预测加载以降低总体 expert loading latency。

术语是什么？
核心观察（eMoE §2.2.3）：不同 NLP 任务对 MoE 层 routing accuracy 的容忍度差异巨大。从靠近 output 的层开始 progressively apply inaccurate routing 的实验结果：
- Classification / Comparison 任务：即使全部层 routing 不准确，similarity 仍 >90%（高度 tolerant）
- Conversation / Summarization 任务：75% 层准确时 similarity 已 <80%（高度 sensitive）
- QA 任务：50% 层准确时 similarity >80%（中等敏感）

基于此，定义 per-task per-layer 的 binary sensitivity flag `s ∈ {0,1}`：
- 离线 profiling：对每个 task type，progressively apply random routing 从 input 侧开始，记录 accuracy drop。accuracy 高于 threshold（如 85%）的层标记为 insensitive（s=0）。
- 在线使用：在 expert loading 决策中，仅 sensitive 任务的预测 f_i 参与 `N_i` 的计算。

从系统架构角度拆解术语：

```
=== Task-aware Expert Loading Decision Flow ===

输入: scheduled requests with task types, expert prediction frequencies

Step 1 — Offline Profiling (per task type, per MoE layer):
  For each task type T:
    For each MoE layer L (from input → output):
      Apply random token-to-expert routing for layers [0..L]
      Measure output similarity with ground truth
      If similarity > threshold (85%):
        Mark task T as "insensitive" to layer L's routing accuracy (s=0)
      Else:
        Mark task T as "sensitive" to layer L's routing accuracy (s=1)

Step 2 — Online Expert Loading (每次 batch scheduling 时):
  For each MoE layer:
    For each task type T:
      s = sensitivity[T][layer]  # from profiling
      if s == 0: continue  # SKIP: 不参与 expert loading 计算
    
    For each expert i:
      # Eq. 2 from eMoE paper
      N_i[T] = (Σ_{j=0}^{T-1} W_j + T_current · W_o) · s · f_i[T]
      # where W_j = input tokens in running request j
      #       W_o = expected output tokens
      #       f_i = predicted routing frequency for expert i
      #       s   = binary sensitivity flag
    
    # Aggregate across all task types
    N_i = Σ_T N_i[T]
    
    # Sort experts by expected token count, pick top L
    sorted_experts = sort_descending(N_i)
    load_experts = sorted_experts[:L]  # L = memory budget
    
    # Load & evict
    new = load_experts ∖ currently_on_gpu
    evict = currently_on_gpu ∖ load_experts
    async_load(new); async_evict(evict)
```

术语一般如何实现？如何使用？
- Sensitivity profiling 是一次性离线操作，结果存储为 per-task per-layer sensitivity matrix
- 实现中 task type 通过关键词匹配识别（CPU 上运行，不干扰推理 pipeline）
- 与 Task-agnostic loading 对比：对 classification/comparison tasks accuracy 基本持平，但对 conversation/QA/summarization tasks accuracy 显著更高
- 适用场景：multi-task MoE serving with diverse task characteristics
- 与 Periodic Expert Invocation 正交：两者可叠加使用

涉及论文标题：
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference
