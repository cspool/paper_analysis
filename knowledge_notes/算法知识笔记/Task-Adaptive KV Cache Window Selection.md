## Task-Adaptive KV Cache Window Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Task-Adaptive KV Cache Window Selection 是 WindowKV（Zuo et al., 2025）提出的 KV cache 压缩方法，核心创新在于将逐 token 选择替换为以**连续语义窗口**为单位的保留/逐出决策，并根据任务类型自适应调整每个窗口内的 token 保留比例。

传统方法（H2O、PyramidKV、StreamingLLM）对每个 token 独立评分和选择——相邻 token 可能被不同步逐出，打断语义连贯性。WindowKV 观察到人类阅读以信息块（窗口）处理长文本（Rayner, 1998），因此将 context 切分为固定大小 ω 的 review windows，以 window 为粒度做 KV cache 保留。同时将下游任务分为 Information Localization（QA 类，p=ω，保留全窗口）和 Information Aggregation（摘要类，p<ω，仅保留 top-p 高分 token），训练 bert-base-cased 分类器自动判断。

算法流程：(1) Context 分为 observation window (最后 α tokens) 和 review context (切为 K 个 ω-token windows)；(2) 计算 observation window 对各 review token 的累积注意力 t_j = Σ A_ij；(3) 窗口级打分 s_k = (1/min(p,ω)) · sum(Top-p(W_k))；(4) 按 dynamic budget 选 top-n windows。

从算法pipeline角度拆解术语：

```
# WindowKV 完整 pipeline
# Input: n tokens, ω window size, α observation size, b_total KV budget

# === Phase 1: Task Classification ===
task_type = Classifier(input_context)  # bert-base-cased → localization/aggregation
p = ω if task_type=="localization" else p_small  # 决定窗口内保留比例

# === Phase 2: Per-Group Window Selection (仅 group-first layers) ===
for group_first_layer l_g in [0, γ, 2γ, ...]:
    Q, K = W_q @ h_lg, W_k @ h_lg           # [n, d_head]
    A = softmax(Q @ K^T / sqrt(d_k))        # [n, n]
    t_j = sum(A[n-α:n, j]) for j in [0,n-α) # observation → token scores
    windows = chunk(tokens[0:n-α], ω)       # partition review context
    scores = [(1/min(p,ω))*sum(top_p(w,p)) for w in windows]
    I_lg = indices_of(topk(scores, b_h/ω))  # retain top windows

    # Group sharing: for l in [l_g+1, l_g+γ-1]:
    #     I_l = I_lg  (直接复用首层 indices)
```

术语一般如何实现？如何使用？

开源：https://github.com/optim996/WindowKV。基于 HuggingFace Transformers，prefill 后执行 window selection。Group-first layer 执行 full attention + scoring，组内其余层复用 indices。Qwen2.5-1.5B: γ=7, ω=32; LLaMA3-8B: γ=8, ω=8/16。λ=14 控制金字塔形状。12% KV cache 下 LongBench 保持 41.35 vs FKV 41.51。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
