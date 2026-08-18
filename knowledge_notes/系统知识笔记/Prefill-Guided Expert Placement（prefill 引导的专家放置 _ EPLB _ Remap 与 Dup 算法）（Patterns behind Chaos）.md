## Prefill-Guided Expert Placement（prefill 引导的专家放置 / EPLB / Remap 与 Dup 算法）（Patterns behind Chaos）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill-Guided Expert Placement 是本文（ISCA 2026）Case Study 2 提出的 MoE serving 负载均衡策略：在 decode 阶段开始时（还没有足够 decode 历史数据时），用 prefill 阶段采集的专家选择信息预先确定专家在 GPU 上的放置，解决大 scale MoE serving（200+ GPU）的初始 decode 负载不均问题。动机：现有 EPLB（Expert Placement Load Balancing）通过周期性（每 3000+ step）调整专家放置来均衡负载，但初始 ~1000 个 decode token 没有 profiling 数据可用，短输出请求（EPLB 永远收集不到足够数据）尤其受影响；而本文 profiling 发现 prefill 与 decode 阶段的专家选择高度相似（Spearman ρ≥0.7 强相关；top-5/10/20 热门专家跨阶段重叠率约 60%/75%/90%），因此 prefill trace 可以作为 decode 初始放置的可靠预测（Insight 1）。提出两个放置算法（Algorithm 2）：(1) Remap-based——保持每 GPU 专家数不变（容量 E/G），按 roofline cost（Cost(f_{l,e})）降序排序专家、贪心分配给最轻负载 GPU，重排专家实现负载均衡；(2) Duplication-based——每 GPU 预留 R 个额外专家槽，从默认连续布局（0-15 在 GPU0、16-31 在 GPU1…）出发，迭代 R·G 次，每次选择使瓶颈负载 max_g load_g 下降最多的 (expert, GPU) 对复制热门专家；复制专家被 token 均分到所有副本。两者都用 roofline 成本模型估计每 GPU 负载。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Remap-based（每 GPU 容量不变的重排）
for layer l:
    f_{l,e} ← 从 prefill trace D 统计专家 e 的频率
    sort experts by Cost(f_{l,e}) 降序
    for e in sorted:
        g* ← 最轻负载 GPU（|S_{g*}| < E/G）
        把 e 放入 S_{g*}；L_{g*} += Cost(f_{l,e})

# Duplication-based（预留 R 槽复制热门专家）
for layer l:
    由 D 生成默认放置 S_g
    for i in 1..R·G:
        (e*,g*) ← argmin_{e,g: r_g>0, g∉hosts(e)} δ_{e,g}   # δ = 复制后 max_g L_g 的下降量
        复制 e* 到 S_{g*}；r_{g*} -= 1；更新受影响负载
```
Annotations：f_{l,e} 是逐层的（每层独立放置）；Cost 用 roofline 模型（专家计算量 × 频率）；Remap 不增加显存（适合显存紧张），Dup 用额外槽换更快的负载均衡（R=1 时 128+8=136 专家/层）；δ_{e,g} 是贪心选择"边际收益最大"的复制动作。系统架构中的位置：这是"初始放置（placement）+ 负载均衡"层的调度决策，与运行期的 per-request 调度正交——它决定专家权重装在哪个 GPU，运行期再由 SGLang/DeepEP 做 token 分发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Qwen3-235B（94 个 MoE 层、128 专家/层、top-8 路由）部署在 SGLang + 8×H100（NVLink）上；在 SGLang 中插入 cuda. Event timer 构建分布式 profiler，独立测量每 GPU 的 attention、top-k、all-to-all、MoE 各操作；通过 SGLang 的 init_expert_location 接口设置专家放置；MoE 后端用 DeepEP 且 ep_dispatch_algorithm="dynamic"（复制专家的 token 均分到各副本）。数据集 MMLU 与 Global-MMLU（按原顺序），batch 64-16384。指标：MoE 计算时间（三个专家线性层，不含 attention/all-to-all/top-k）。Baseline：Default（Qwen/SGLang 默认连续放置）、Best/Worst（oracle decode 选择的上下界）、Remap、Dup。结果：Remap +15.5%、Dup +12.5%（即最高 1.25x MoE 计算加速），均 >2x 于 Worst、与 Best 差距 <10%；EP8 规模下 max/min 执行时间比仅 ~1.3x，更大 EP 规模收益更明显。开源：https://github.com/zhongkaiyu/moe_exp_placement（DOI 10.5281/zenodo.19617695，Apache-2.0；需要 8×H100 80GB、CUDA 12.0+、~300GB 磁盘、PyTorch、修改版 SGLang fork、DeepEP、DeepGEMM；main_ae.py 复现 Figure 17，12-16 小时，±5% 波动）。EPLB 是相关 prior work：动态调整专家放置但每 3000+ step 才触发、依赖周期 profile。

涉及论文标题：
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
