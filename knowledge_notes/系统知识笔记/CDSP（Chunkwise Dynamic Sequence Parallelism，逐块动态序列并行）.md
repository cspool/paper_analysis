## CDSP（Chunkwise Dynamic Sequence Parallelism，逐块动态序列并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CDSP 是 Tetris（ISCA'26，Peking University + ByteDance Seed）提出的细粒度序列并行调度策略：把每个请求的 prompt 拆成多个 chunk，为每个 chunk 分配不同的 SP 大小（前段小 SP、后段大 SP，SP 沿 chunk 递增）。动机：请求级 SP 分配（如 LoongServe ESP）把整请求统一 SP，在 ring attention 同步约束下，给长请求分配大 SP 会导致已占用实例上的空隙空闲（资源碎片），分配小 SP 又显著延长长请求 prefill 延迟。CDSP 像俄罗斯方块一样用前段小 SP 的 chunk 填实例排队空隙（部分执行提前开始），用后段大 SP 的 chunk 满足长序列计算需求，同时优化 TTFT 与资源利用率。配套约束：每个 chunk 的实例组必须包含前序 chunk 的所有实例（保证 KV cache 均衡重分布只需把部分 KV 前传，降低 cache balancing 开销）。核心收益（真实线上负载评估）：TTFT 最高降 4.35×、median TBT 降 40.1%、最大请求容量提升 45%、吞吐 1.24-3.38×。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# Tetris 中 CDSP 的运转流程（prefill-decoding 解耦集群）
请求到达 → C++ CDSP scheduler（Algorithm 1 递归）:
  1. Algorithm 2 单 chunk 调度：对每个候选 SP s，用延迟模型 T_s=a_s+b_s·L+c_s·(C·L)+d_s·L²
     估 TTFT=max(排队延迟)+T_s；仅当 TTFT 增益超过 improvement_rate 阈值才扩 SP
     （阈值按 30s 窗口观测到达率查离线 simulator 预计算的最优 rate 映射动态刷新）
  2. Algorithm 1 枚举 (s_current,s_next) SP 对，Algorithm 3 用两实例组排队延迟差设 chunk 延迟预算、
     数值求解当前 chunk 长度 → 递归求完整 chunk 计划 → 按估计 TTFT 选最优
  3. Ray 把 per-instance 计划转发 local managers → prefill 实例组按 chunk 顺序执行
     （Flash Attention zigzag ring + NVSHMEM；chunk 间 KV cache 均匀重分布并跨层重叠）
  4. KV cache 经 handshake 式 backend 分配流式传到 decoding 实例 → continuous batching decode
```
Annotations: L=未分配 token 数、A=前序 chunk 分配记录、S=SP 候选集（2 的幂）、P=实例池（含排队延迟 T_k）；improvement_rate 控制 SP 扩张幅度（低负载小 rate→大 SP 压 prefill，高负载大 rate→避免过度扩张让后到请求早执行）；递归终止于 S 只剩一个候选。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Tetris 服务系统（~17.5K 行 C++/Python）：FastAPI 前端 + vLLM 控制面（global/local managers，CDSP scheduler 用 C++ 消除调度延迟，initialize_schedule/update_schedule/cdsp_schedule 三接口）+ PyTorch/Triton-distributed 推理后端；调度开销 SP≤128 时 ≤86.8µs、端到端 ≤93.79µs（8B）。使用：在线长上下文 serving（TTFT/TBT SLO），支持前缀缓存（按 reuse ratio 枚举 Algorithm 1 选最小 TTFT 策略）；SP 大小变化时实例组扩展按"节点内优先、必要时跨节点"策略。论文未开源（arXiv 2511.06247 已撤回以符合机构政策）；可复用组件 vLLM、LoongServe、NCCL、NVSHMEM。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
