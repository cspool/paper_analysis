## Utilization-aware Speculation（利用率感知投机调度，Algorithm 1）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HybridSpec 的运行时投机参数自适应策略：硬件固定而工作负载动态，通过调节投机解码的 draft budget 与 tree width 来调制每轮任务的算术强度（每迭代处理 token 数），把 XPU 与 HB 栈的利用率推向各自 roofline 拐点，同时保证投机有效性。基于实验的两条原则：(1) draft budget 增大时接受 token 长度先增后饱和（图 10(a)），设上限 B 避免在拒绝 draft 上浪费计算；(2) 固定总预算下接受长度随 tree width 先增后减（图 10(b)，探索 vs 深度权衡，SVR 拟合），用查找表 T 记录各 budget 的最优 width 拐点 p（chain 视为 width=1 特例）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Algorithm 1 伪代码（一轮）：
```
while (prefill 后) and (未 EOS):
    p = T[draft_budget]                     # 最优 tree width
    while current_draft_tree_size <= draft_budget:
        AI_HB = HB_stack.process()
        if AI_HB < HB_stack.roofline:       # memory-bound → 加宽探索
            tree_width = min(p, tree_width + 1)
        else:
            tree_width = max(1, tree_width - 1)
        HB_stack.add_task(DecodeTask(tree_width))
    XPU.add_task(VerifyTask(draft_budget))
    AI_XPU = XPU.process()
    if AI_XPU < XPU.roofline:               # compute 未满 → 加大预算
        draft_budget = min(B, 2*draft_budget)
    else:
        draft_budget = max(1, draft_budget/2)
```
实测平均 draft_budget/tree_width 随请求率 1→4 从 (30.74,3.72) 降至 (9.25,1.58)（高负载利用率已高、无空闲可挖）；相对固定 budget 投机平均 1.81× 加速（高请求率下降、长序列更敏感）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SVR 对 (budget→最优 width) 离线拟合查找表 T，运行时按各单元实测算术强度（每轮 token 数）对照 roofline 粗调（±1 width、×2/÷2 budget，上下限 [1,p]/[1,B]）；B=32。用途：把"设计期 DSE"（静态离线推理的参数优化）迁移为"运行期自适应"，填补固定硬件与动态负载的利用率缺口（FIFO 55.63% → PFS 62.43% → CHK 66.17% 利用率递增）。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
