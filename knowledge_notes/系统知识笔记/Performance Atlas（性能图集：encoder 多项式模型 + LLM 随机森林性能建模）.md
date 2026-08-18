## Performance Atlas（性能图集：encoder 多项式模型 + LLM 随机森林性能建模）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Performance Atlas 是 RESONATOR 的离线性能建模组件：一次 profiling 在目标 GPU 上扫代表工作点，拟合轻量预测器，供所有在线调度器查询（不采样高频硬件计数器）。两部分模型：①Encoder 性能模型——encoder 延迟取决于分辨率 r、TP 度 k、SM 配额 SM_enc，把分辨率映射为序列长 L_seq=⌈H(r)W(r)/P²⌉（P 为 ViT patch size），以 L_seq 为唯一复杂度参数用紧凑多项式拟合 T_enc(r,k,SM_enc)=T̂_enc^poly(L_seq,k,SM_enc)，可泛化到任意分辨率/宽高比；②LLM chunked-prefill 模型——chunk 特征 c=(n_p,n_d,L_c) 下延迟 T_llm(c,SM_llm)=RF_llm(n_p,n_d,L_c,SM_llm) 用随机森林拟合，并为每个 chunk 给 Tag∈{mem,comp}（按算术强度 I 与 roofline ridge point I*=PeakFLOPs/PeakHBM_BW 比较，I<η·I* 判 memory-bound）与最小 decode SM 配额 SM_dec_min。Atlas 还存合法 TP 集 K(r)（按显存容量过滤）、encoder 各 TP 延迟 T(R,k)、kernel profile 表 P（每 kernel 类型标 comp/mem 与典型 SM/HBM 用量）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Atlas 的构建与查询流程：
```
# Offline 构建（一次，Qwen2-VL-7B 约 6 小时）:
S = SM 分配层级集合（粒度 ΔSM 决定 |S|≤⌊SM_total/ΔSM⌋），K = TP 度集合
encoder: 以 ΔL 步长扫 L_seq × 各 TP 的显存合法点 → 拟合多项式系数
LLM: 从网格 G_n×G_d×G_c 采样代表 chunk（而非全笛卡尔积），KV cache 按 L_c bucket 分组摊销 setup → 训练 RF
# Online 查询:
T_enc = Atlas.GetLatency(R, k)     # PRISM 用 v(R,k)=1/T(R,k) 做调度价值
TAG(c), SM_dec_min(c) = Atlas.GetChunkHints(c)   # Intra-GPU 引擎用
P[kernel] = Atlas.GetKernelProfile(k)            # contending 场景选流
```
Annotations：profiling 空间不穷举（LLM 只采样代表 chunk 点、KV 按 L_c 分桶摊销），换来轻量；Qwen2-VL-7B 上 profiled 范围平均预测误差 4.7%、外推 8.1%；profiling 直接测 strided 执行路径，故 logical sharding 的布局开销已计入调度器用的延迟估计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为离线 Python profiler + 在线内存表/预测器：encoder 用紧凑多项式回归（对 L_seq 泛化），LLM chunk 用随机森林回归；kernel profile 表来自 offline 日志。使用：所有在线调度决策（Intra-GPU 共享模式选择、PRISM 的每计划延迟/吞吐查询、合法 TP 集过滤）只查 Atlas，使运行时决策开销降至元数据查找级。该设计把"运行时性能建模"与"调度决策"解耦，是把 roofline/预测模型作为在线调度目标的系统化用法。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
