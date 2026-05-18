## Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

- baseline方法是什么？
  Baseline方法分为两类：(1) **Sarathi-Serve**：prefill-decode aggregation架构 + iteration-level chunked prefill + EDF调度。prefill被按固定token budget拆分为chunks，与decode iteration融合执行；为满足最严格TBT（50ms），每个instance限制prefill chunk size，导致prefill SLO violation显著。执行例子：请求到达→chunked prefill与decode融合batch→所有GPU SMs统一执行完整iteration（穿过所有模型layers）→下一iteration才能加入新请求或切换。全栈路径：算法层标准transformer→Serving框架层vLLM/Sarathi-Serve continuous batching + chunked prefill→编译框架层未修改→kernel调度层FlashAttention/FlashInfer融合attention kernel→硬件架构层A100 GPU，所有SMs统一执行。(2) **DistServe**：prefill-decode disaggregation架构，prefill和decode放到专用instance，但每个instance内部使用iteration-level scheduling + EDF。解决了Sarathi-Serve的prefill-decode资源争用，但无法在iteration内差异化执行multi-SLO请求。执行例子：请求到达→prefill instance处理（完整iteration）→KV cache迁移→decode instance批处理（完整iteration，统一TBT约束所有请求）。

  Baseline的缺陷：
  - **Prefill侧（Inflexible prefill chunking）**：chunk size是固定trade-off——增大chunk size（100→3200）使per-iteration latency增16×但per-token latency降45.4%。大chunk更高效但垄断compute导致head-of-line blocking（latency-critical请求被迫等待当前chunk完整iteration结束）；小chunk响应快但GPU utilization低。iteration-level无法在同一iteration内动态调整chunk大小或合并请求。
  - **Decode侧（Unified decode batching）**：不同SLO请求的TBT容忍度差异显著（relaxed请求可容忍7×更多并发），但iteration-level scheduler以统一TBT为目标配置batch size，所有请求被迫服从最严格SLO，严重限制了relaxed请求的batch容量，导致overall SLO violation增加。
  - **实例间调度**：DistServe的disaggregation改善了阶段隔离，但decode instance的batch sizing仍受multi-SLO constraint限制，无法充分利用relaxed请求的slack。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**layer-level scheduling**，将LLM serving调度粒度从完整的iteration细化到单个transformer layer，并配套设计Laser系统实现intra-instance layer-level scheduling与inter-instance SLO-aware dispatching的联合优化。

  核心设计及其与Baseline缺陷的映射：

  **(1) Layer-level chunked prefill（解决prefill chunking不灵活问题）**：prefill Scheduler在每个layer边界评估新请求的slack时间（TTFT target - estimated prefill latency）。若当前chunk剩余iteration time超过新请求slack，则在layer边界保存正在执行请求的intermediate state到GPU intermediate cache，然后决定：(a)把新请求推进到同一layer位置后与旧请求合并，提高batch size和GPU utilization；或(b)直接优先执行新请求，避免SLO violation。若无抢占且queue为空，动态合并新请求到当前chunk以提升效率。队列按EDF排序。这一机制同时解决了head-of-line blocking（可随时在layer边界抢占）和limited chunk-size execution（可在同一iteration内动态合并请求形成更大有效chunk）两个问题。

  **(2) Layer-level decode batching（解决统一decode batch问题）**：decode Planner为每个请求生成execution plan（L: 每iteration执行层数，O: 调度offset），利用不同请求的SLO slack差异：(a) strict请求（如50ms TBT）每iteration执行全部N层确保低延迟；(b) relaxed请求（如200ms TBT）每iteration只执行N/2或N/4层，多轮完成完整forward pass，释放batch slot给更多relaxed请求。Planner通过模块化latency model（stateless模块用分段线性函数，stateful attention用线性函数，Pearson系数>0.78）估计per-iteration latency，贪心策略优先减少最relaxed请求的层数并选offset平衡同group负载。更新仅在critical events触发且overlap执行，开销<3.8%。

  **(3) Inter-instance request dispatching（解决实例间SLO异构排布问题）**：prefill侧Global Controller选TTFT slack最大的实例，最大化SLO合并弹性。decode侧采用SLO-homogeneous group management——将decode instance按TBT target分组，新请求优先分到相近SLO group中TBT increment最小的instance，允许跨group分配以平衡负载。实例按arrival rate和SLO-compliant batch size的比值动态调整group大小。去中心化performance evaluation——每个decode instance本地运行ExecPlan评估接纳请求后的impact，仅向Controller返回aggregated TBT increment，避免中央调度瓶颈。

  **(4) Intermediate state management + latency model（支撑layer-level切换的系统机制）**：每个instance维护GPU intermediate cache（prefill 16384 tokens，decode 2048 tokens，Llama-70B上<256MB）存layer-level hidden states；fused CUDA kernel合并state caching/retrieval；KV cache migration按layer粒度异步overlap prefill computation。Latency model通过offline profiling（≤2秒）拟合参数，预测准确率94.6%-98.6%。

  论文方法全栈执行例子（以Qwen-14B + ShareGPT/HumanEval/LongBench mixed workload, 4×A100 80GB集群为例）：

  - **算法层**：标准transformer模型，论文未修改attention/FFN算子或模型结构。
  - **Serving框架层**：vLLM + Ray。Global Controller profiling→请求到达→prefill instance的Scheduler评估slack→layer boundary抢占/合并→EDF队列调度→prefill Executor执行layer-level chunked prefill→KV cache按layer粒度异步迁移→decode instance的Planner构建execution plan（L/O）→latency analysis估计per-iteration latency→贪心调整→Executor按plan在layer边界切换请求→intermediate cache管理intermediate states。
  - **编译框架层**：论文未修改编译框架。利用CUDA fused kernel优化state caching/retrieval。
  - **kernel调度层**：FlashInfer提供attention kernel。Laser新增的layer-level intermediate state switching通过fused CUDA kernel实现（合并caching和retrieval），switching overhead<1.5%。KV cache migration按layer粒度与prefill computation异步overlap。
  - **硬件架构层**：4台主机×4×A100 80GB，机内NVLink，机间100 Gbps LAN。1/2/4-way tensor parallelism分别对应14B/32B/70B模型。

  最终效果：Laser相比Sarathi-Serve和DistServe在Qwen-14B/32B和Llama-70B上分别提升goodput 43.4%/68.9%/56.6%。relaxed请求占比高时改进>86%；tight SLO (0.8×)下vs Sarathi-Serve goodput gain最高6.25×；99% attainment目标下throughput提升最高1.85×。prefill SLO violation rate最多降低21.6%，平均TTFT降低>10%；decode TBT violation rate降低最多17.2%（6.7% from layer-level batching + 10.5% from group-based assignment）。
