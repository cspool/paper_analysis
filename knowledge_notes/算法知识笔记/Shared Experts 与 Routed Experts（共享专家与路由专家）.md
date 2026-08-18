## Shared Experts 与 Routed Experts（共享专家与路由专家）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE 层的两类专家：Routed Experts 由 gating 每 token 动态选择、输出按 softmax 路由权重加权聚合；Shared Experts 对所有 token 恒激活、不参与路由，其输出以等权平均或加权和加入（STEP Eq.3：y_shared = Σ w_i^s·E_i^s(x) 或 ΣE_i^s(x)/j）。代表配置（STEP 表 I）：Mixtral-8x7B 无 shared（32 层、8 routed、top-2、routed expert (4096,14336)、激活 13B/总量 46.7B）；DeepSeek-V2-Lite-Chat 2 shared + 64 routed（top-6、routed expert (2048,1408)、激活 2.7B/总量 14.3B）；Qwen1.5-MoE-A2.7B 4 shared + 60 routed（top-4、shared (2048,5632)、routed (2048,1408)、激活 2.4B/总量 16B）。共享专家因恒激活可整体预加载、不进入 T_load 公式（STEP Eq.1 明确把 shared 排除）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一个含 shared 的 MoE 层前向（STEP 视角）
y_shared = mean/weighted_sum(FFN_shared_i(h) for i in 1..j)   # 全部 token 恒算
logits = gate(h)                                    # (N,)
topk_idx, w = topk(softmax(logits), k)              # routed 动态选择
y_routed = Σ_i w_i * FFN_routed_{idx_i}(h)          # 只算被选 routed
y = y_shared + y_routed                             # 合并（Eq.2）
```
Annotations：j=shared 数、k=routed 激活数、N=routed 总数。STEP 的关键算法改动：把"选中的高频 routed 专家"在一个 token 窗口内临时升格为 shared（结构从 j shared + k routed 变 j+c shared + k−c routed），使其整窗口常驻 GPU 且每步少动态加载 c 个专家——这是"临时共享专家"的语义（见该条目），不改模型权重、不改 gating 语义（仍算全部专家分数）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DeepSeek-MoE 论文首次系统提出 shared experts；HuggingFace Transformers 中 shared experts 以独立参数组存在（如 Qwen config 的 shared_expert_intermediate_size），MoE forward 先算 shared 再算 routed 最后合并。使用场景：shared 提供通用基础表示（防 routing 失误丢失能力）、routed 提供专长；对 offloading/预取系统，shared 恒激活 = 天然可预载、是"驻留集"的第一优先级（STEP 中 shared 与临时 shared 一起在计算开始前预取常驻），而 routed 的 88%（A100 上 Qwen3-30B-A3B INT8 profiling）执行时间花在专家取数上，正是预取要隐藏的对象。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
