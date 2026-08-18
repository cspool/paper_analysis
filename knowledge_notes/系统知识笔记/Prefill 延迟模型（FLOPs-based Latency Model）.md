## Prefill 延迟模型（FLOPs-based Latency Model）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill 延迟模型是 serving 调度器用解析式估计 prefill 计算延迟的性能模型。Tetris（ISCA'26）采用基于 FLOPs 结构的四系数多项式：对一个 chunk，令 C=历史 token 数、L=chunk 内 token 数、SP=s，则 T_s(R)=a_s + b_s·L + c_s·(C·L) + d_s·L²。四项分别建模：a_s=常数因子开销、b_s·L=全连接层（FFN/QKV 投影，随当前 token 数线性）、c_s·(C·L)=与历史 token 的注意力（KV 读取+计算随历史×当前线性）、d_s·L²=chunk 内 token 间注意力（平方）。系数 (a_s,b_s,c_s,d_s) 对每个目标 SP 大小用最小二乘拟合：离线收集不同 (C,L) 组合的 prefill 延迟数据拟合，在线 serving 复用直到 GPU/模型类型变化。论文实测：模型拟合误差 ≤7.64%/6.35%（8B/70B），模拟器（基于该模型）误差平均 6.9%/2.5%。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# 延迟模型在 CDSP 调度中的使用（Algorithm 2 单 chunk 调度）
for s in SP_candidates:
    instance_group = GetGroup(P, initial_group, s)      # 扩展实例组（节点内优先）
    T_queue = max{T_i | p_i in instance_group}          # 实例组最大排队延迟
    T_prefill = PerformanceModel(s, C, L)               # 式(1) 多项式
    TTFT = T_queue + T_prefill
    if TTFT < opt_TTFT * (1 - improvement_rate):        # 阈值防止过度 SP 扩张
        (opt_TTFT, opt_group) = (TTFT, instance_group)
# Algorithm 3 中用 T_budget = T_queue(next) - T_queue(current) 反解 chunk 长度：
#   把式(1) 视为 L 的多项式，牛顿法数值求解使 chunk prefill 延迟=预算的 L_chunk
```
Annotations: 模型把排队延迟（调度可观测）与计算延迟（模型估计）解耦求和得 TTFT；C 与 L 为调度器已知/可枚举变量；模型精度决定调度质量（论文用 16k 间隔采样拟合、全数据验证）。
该模型同时驱动 simulator 式 improvement rate profiler：按请求长度分布+Poisson 到达率模拟不同负载下各 improvement rate 的 TTFT，离线选出每到达率的最优 rate（0.05-0.75，0.5 req/s 步长），在线每 30s 按观测到达率刷新。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Tetris 中作为 CDSP scheduler 的性能评估核心（C++ 实现，与调度算法集成）；离线 profiling 收集 (C=0-256k step 8k)×(L=8k-256k step 8k) 全组合延迟（跳过 OOM 点）、16k 间隔子集拟合最小二乘。使用：chunk 计划求解（Algorithm 1/3）、improvement rate 离线模拟、scheduler 端到端开销极低（SP=128 时单次调度 ≤86.8µs）。类似工作：LoongServe/常见 serving 用 FLOPs 或 roofline 模型估计 prefill；本模型的价值在于把"与历史 token 的注意力"单独建模（c_s·C·L 项）以支持 chunk 内可变历史长度。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
