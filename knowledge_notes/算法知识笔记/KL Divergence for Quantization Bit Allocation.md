## KL Divergence for Quantization Bit Allocation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KL Divergence (Kullback-Leibler Divergence) 在 SliM-LLM 的 SBA 中用作混合精度 bit-width 分配的优化目标函数，替代传统 MSE。形式为 D_KL(softmax(xW^T) || softmax(xŴ_sba^T))，即量化前后输出经 softmax 化为概率分布后的 KL 散度。动机：MSE 最小化权重重建误差但不保证输出分布对齐——相同 MSE 的量化方案可能产生不同 token 概率分布偏移。KL 散度从信息熵角度衡量输出分布的偏移，使 bit allocation 更偏向保护对输出分布影响大的 group。实验验证：2-bit 时 KL 优于 MSE（OPT-1.3B: KL=30.71 vs MSE=32.50; LLaMA-7B: KL=14.58 vs MSE=21.94），差异随模型增大更显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 在SBA双指针搜索中作为优化目标
for p in (1 to k//2):
    Ŵ_mixed = assign_mixed_precision(W, sorted_idx, p, N)
    out_q = x · Ŵ_mixed^T      # [t, n]
    out_fp = x · W^T           # [t, n]
    P = softmax(out_q)         # 概率分布
    Q = softmax(out_fp)
    kl = sum(P * log(P / Q)) / t  # 逐token平均KL
```
与 MSE 对比：MSE = mean((out_q - out_fp)²) 考虑逐元素差值；KL = D_KL(P||Q) 考虑分布形状。当 out_q 和 out_fp 各维度等比例偏移时 MSE > 0 但分布一致 KL ≈ 0；当 out_q 某维度剧烈偏差时 KL 惩罚远大于 MSE。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`F.kl_div(F.log_softmax(out_q, dim=-1), F.softmax(out_fp, dim=-1), reduction='batchmean')`。SBA 在 layer 级别计算 KL（非 global），逐层优化 bit allocation。计算比 MSE 多一次 softmax，但 SBA 搜索空间小（k ≤ 32），总体开销可忽略。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models
