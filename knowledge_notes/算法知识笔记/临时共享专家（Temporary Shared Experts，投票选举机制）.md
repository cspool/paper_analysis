## 临时共享专家（Temporary Shared Experts，投票选举机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STEP 提出的在线机制：利用专家选择的时间连续性，把最近解码窗口内高频被选中的 routed 专家临时"升格"为 shared 专家（temporary shared experts），使其在下个窗口内常驻 GPU、每步不再走动态加载，从而把每步动态加载的专家数从 k 降到 k−c（c=当选专家数）。选举方式：把输出序列切成 token 窗口，窗口内每个 decode step 记录 top-2k 专家（不只 top-k），每次出现记一票（反映频率与选择强度）；窗口结束按票数选 top-c 专家为下个窗口的临时 shared。当选后：有效 MoE 结构从 j shared + k routed 变为 j+c shared + k−c routed；临时 shared 总是被预取常驻 GPU，但仅被 gating 选中时才执行计算（与 routed 一致）；gating 仍对全部专家算分数以保证统计一致与后续选举。关键约束：临时 shared 不增加显存——STEP 在固定 cache 预算下用当选专家替换低使用率专家。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 窗口 W 结束时的选举（per layer）
votes = Counter()                       # 全零初始化
for step in W:
    top2k = topk(gate(h_step), 2*k)     # 跟踪 top-2k（>top-k 的候选视野）
    for e in top2k: votes[e] += 1       # 每次出现记一票
elected = votes.top(c)                  # 票数最高的 c 个专家
# 下一窗口：结构 j+c shared + k−c routed；elected 提前预取常驻 GPU
```
Annotations：W=窗口长度（token 数）、k=routed 激活数、c=当选数（表 I：Mixtral c=1、Qwen c=1、DeepSeek c=2）。为什么用 top-2k 投票而非直接 top-k：top-k 只反映"最终被算的专家"，top-2k 能捕捉"接近被选"的高频候选，提前把它们驻留可提高命中率（Table II-IV：CNN/DM 命中率 85.5–98.8%）。时间连续性的依据（Fig.4/5）：长序列生成（LongBench）中一小撮专家被连续 step 反复选中，且不同任务（Summary vs Translation）的连续选中长度分布差异大——这同时引出 token 感知自适应窗口（见该条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为运行时层嵌入 HuggingFace Transformers 推理路径，窗口级选举 + 独立 CUDA stream 异步预取（见 kernel 调度层"专家预取"条目）；与 expert parallelism 正交——每个 EP group 独立维护热专家本地缓存并运行自己的选举/预取。使用效果：decode 阶段受益最大（时间连续性仅存在于 decode），DeepSeek 因时间连续性比 Qwen 更强而预取收益更大（Table IV 高命中率 78.6–95.3%）。退化保护：当选专家整窗口固定，即使实际使用率下降也不频繁换出（避免 transient routing 波动引发抖动）；当窗口长度缩到 1 时停用实际预取、仅保留投票统计。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
