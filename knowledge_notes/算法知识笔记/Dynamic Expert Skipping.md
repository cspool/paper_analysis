## Dynamic Expert Skipping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Expert Skipping 是一种在线推理加速技术，在 MoE LLM 的每个 MoE 层根据路由权重动态决定是否跳过次优专家（second-best expert），减少每个 token 实际执行的专家 FFN 数量。与 Expert Pruning 不同，Skipping 不删除任何专家参数，而是在推理时对每个 token 做在线决策：若 top-2 路由权重比 w_{e1}/w_{e0} < β（逐层阈值），则仅执行 top-1 专家计算，跳过 e1 的 FFN。阈值 β 通过校准数据确定：对每层收集所有 token 的 w_{e1}/w_{e0} 比值，取中位数（使跳过概率约 50%）。其理论基础（Appendix A.2）：在 top-k 设置下，动态跳过 i 个专家后的重构损失上界为 L ≤ (Σ_{m=i+1}^k w_m / Σ_{m=1}^k w_m)·D，其中 D 为不同专家输出的期望差异。在 top-2 特例下，跳过条件简化为 w_2 ≤ β·w_1，β=H/(D−H)，H 为允许的重构损失上限。Dynamic Skipping 与 Expert Pruning 正交——可同时使用：剪枝减少内存，跳过减少计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dynamic Expert Skipping pipeline：
```
# === 离线校准: 确定每层 β ===
for layer l in 1..L:
    ratios = []
    for token in calib_data:
        w = Softmax(Router(token))
        e0, e1 = Top2(w)
        ratios.append(w[e1] / w[e0])
    β[l] = median(ratios)         # 中位数 → 跳过概率≈50%

# === 在线推理: per-token 动态跳过 ===
for each token x in autoregressive generation:
    for layer l in 1..L:
        w = Softmax(Router(x))   # n 维路由权重
        e0, e1 = Top2(w)         # 取 top-2
        
        if w[e1] < β[l] * w[e0]:
            # 次优专家贡献小 → 跳过
            y = E_{e0}(x)         # 仅 top-1 专家
        else:
            # 两个专家都执行
            w̃[e0] = w[e0]/(w[e0]+w[e1])
            w̃[e1] = w[e1]/(w[e0]+w[e1])
            y = w̃[e0]·E_{e0}(x) + w̃[e1]·E_{e1}(x)
```
Mixtral 8x7B C4 经验 β 值（32层）：0.402, 0.494, 0.463, 0.484, 0.478, 0.491, 0.523, 0.521, 0.544, 0.570, 0.574, 0.489, 0.503, 0.618, 0.568, 0.535, 0.559, 0.519, 0.537, 0.487, 0.469, 0.461, 0.461, 0.469, 0.458, 0.418, 0.433, 0.418, 0.406, 0.433, 0.447, 0.535。层间差异显著（0.402-0.618），验证逐层独立 β 的必要性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于 HuggingFace Transformers 修改 MoE layer 的 forward 函数，核心改动 < 20 行。组合使用效果：r=6 剪枝 + 动态跳过 (Mixtral 8x7B Instruct) → LM-eval 66.04, speedup 1.27×；比 r=4 纯剪枝 (63.88) 精度更高且加速相当。MATH 校准 β 值：0.503-0.346（数值更小，跳过更保守），因数学任务对精度要求更高。局限性：跳过率约 50% 意味着平均每个 token 激活 1.5 个专家，加速上限约 1.33×（vs top-2 的 2 个专家），无法达到更大的加速比。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

---
