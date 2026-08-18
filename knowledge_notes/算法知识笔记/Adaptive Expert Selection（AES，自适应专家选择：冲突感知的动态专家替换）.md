## Adaptive Expert Selection（AES，自适应专家选择：冲突感知的动态专家替换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Expert Selection（AES）是 DIAMoND 提出的动态在线专家选择算法：MoE 路由（router = linear 层 + top-k）输出专家分数后，标准流程直接取 top-k 专家；AES 在此基础上做冲突感知替换——在 in-NAND 阵列上，被选专家可能发生两类冲突（共享 OU 输出端口；缺少可同时区分它们的 mask 模式），此时用"分数略低但无冲突"的专家替代，以换取 FFN 层单 read cycle 内并行完成全部 k 个专家（而非串行多 cycle）。调节旋钮 T（阈值）：仅当无冲突替代专家与原冲突专家的路由分数差 < T 时替换，否则保留原专家（接受额外 read cycle 保精度）。配套指标：pairwise difference = 专家对中至少一位与原始 top-k 不同的比例（衡量冲突解决程度）；expert similarity = Σ_{i∈E_T∩E_k} w_i / Σ_{i∈E_T} w_i（自适应所选专家集与原始 top-k 的重叠加权占比；GRIN-MoE 因 gate 把 top-k 外权重置零而不适用）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 单层 FFN、每 token 的专家选择（k 个专家、阈值 T）
s = router(h)                       # (N,) 专家分数，h 为 attention 输出
S = { argmax(s) }                   # 最高分专家必选
for e in sort_desc(s):              # 其余按分数降序尝试
    if e conflicts with S:          # 共享 OU 输出端口或缺兼容 mask
        e' = highest_score_free({e'' ∉ S ∪ {e} : e'' no conflict with S})
        if s_e - s_e' < T: S = S ∪ {e'}   # 分数差小 → 替换
        else: keep e                 # 保留冲突专家，FFN 多 1 个 read cycle
    elif |S| < k: S = S ∪ {e}
```
张量层面：每个专家 FFN = Up/Gate/Down 三投影（Mixtral：隐维 4096、专家中间维 14336），每个投影按 OU 切分为多个子矩阵做 in-NAND VMM；AES 保证 k 个专家的三投影在同一 read cycle 并行执行 → FFN 层恰好 3 cycles。例子（Fig.11c，8 选 4）：最高分 E6 先选；E4、E7 无冲突直接选；第四个候选 E5 与 E4/E6/E7 冲突 → 算法在剩余专家中找次高分无冲突的 E1 替代。硬件执行通路：Priority Queue（分数有序专家队列）→ Conflict FIFO（被推迟专家）→ Mask Pattern RAM（专家 ID → 兼容 mask 位向量，如 4'b1001）→ Pattern State Handler（4 寄存器跟踪各 Expert Group 可用 mask）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在线执行（每 token 每层选一次），由 DIAMoND 的 Dynamic Mask Selector ASIC 电路完成（面积 0.006mm²、0.76mW）；软件侧等价实现即上列伪代码。使用方式：任何"激活子集物理冲突"的稀疏/存内推理系统都可用（把硬件约束折进选择算法、以阈值 T 调节精度-并行权衡）。实测（DIAMoND）：T 敏感性——expert similarity > 0.9 时端到端精度（ARC-Challenge/PIQA/HellaSwag/WinoGrande）仅微小波动，pairwise difference 随 T 先快升后饱和（专家分数差有界）；AES 使解码加速至多 1.52×（与 mask 设计合计 1.95×），冲突率从 Mask-only 的 10.2%~93.5% 降超一个数量级（DeepSeek/Qwen 等专家数多的模型效果最显著）；DIAMoND-L+Mixtral（单专家粒度）与 DIAMoND-H+DeepSeek/Qwen（全专家可容纳）天然无冲突，无需 AES。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
