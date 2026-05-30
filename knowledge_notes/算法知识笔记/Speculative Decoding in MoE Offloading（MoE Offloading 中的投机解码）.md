## Speculative Decoding in MoE Offloading（MoE Offloading 中的投机解码）

术语是什么？通过联网搜索让回答具体和精准。
Speculative Decoding in MoE Offloading 是 SpecMoEOff 提出的首个将 speculative decoding 应用于 MoE 模型 CPU-GPU offloading 推理场景的系统方法。核心动机：MoE offloading 中，CPU-GPU 的 expert weight 传输（I/O bottleneck）和 MoE 的稀疏激活导致 GPU 利用率极低——batch=1 方案仅 0.76% GPU 利用率，throughput-oriented 方案也仅 3.13%。SpecMoEOff 通过 speculative decoding 增大每次 target model forward 处理的 token 数（从 1 token 变为 k+1 tokens），从而在相同的 expert loading 开销下完成更多计算，隐藏 offloading 延迟。

在该场景中，speculative decoding 使用的 draft model 为 EAGLE（利用 target model 的 hidden state 作为输入，仅含 1 层 attention + FFN，<2GB 参数）。关键适配：(1) draft model KV cache 也需要 offloading（在 large batch 下超过 GPU HBM 容量）；(2) verification 阶段的 chunked attention 在 CPU 执行（避免 CPU→GPU 传输 KV cache）；(3) hyperparameter 需要自动优化（draft length k 与 batch size/micro-batch size 交互影响性能）。

从算法pipeline角度拆解术语：
```
# SpecMoEOff: 单次 Speculative Decoding Iteration
输入: prefix tokens x_1:l, draft model M_d (EAGLE), target model M_t (Mixtral-8x7B)

# Step 1: Draft Phase (EAGLE)
# M_d 参数全在 GPU HBM (<2GB), KV cache: GPU Part + CPU Part
for i = 1 to k:  # k 由 Hyperparameter Optimizer 确定
    h = target_model_hidden_state  # 从 target model 获取 feature
    x_tilde_{l+i} = M_d.generate(h, prefix=x_1:l+i-1)
    # GPU Part: attention+FFN 均在 GPU
    # CPU Part: attention on CPU, FFN on GPU

# Step 2: Target Model Verification
# Q ∈ R^{k×d}, K,V ∈ R^{(l+k)×d} from CPU DRAM
extended_x = concat(x_1:l, x_tilde_{l+1:l+k})
# CPU Chunked Attention (Intel MKL):
scores = Q @ K^T / sqrt(d) + mask  # mask 仅存储 n×n draft 部分
attn_out = softmax(scores) @ V
# GPU MoE: expert weights CPU→GPU HBM → FFN
p_1:k = M_t.forward(extended_x)

# Step 3: Probabilistic Acceptance
n_accepted = verify_and_accept(p_1:k, x_tilde, q_1:k)
# n_accepted = a(k), acceptance rate function from profiling
```

术语一般如何实现？如何使用？
SpecMoEOff 基于 SGLang 框架实现，采纳 MoE-Lightning 的 FFN/expert cache 设计，增加 20,000+ 行 Python/C++/CUDA。Draft model 使用 EAGLE 框架（利用 target model hidden state，仅 1 层 attention + FFN）。在 MoE offloading 场景下与标准 speculative decoding 的关键区别：(1) target model verification 使用 CPU chunked attention（而非 GPU）；(2) draft model KV cache 也需要 offloading；(3) 需要 hyperparameter optimizer 自动确定最优 draft length k。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding
