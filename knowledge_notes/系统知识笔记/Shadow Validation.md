## Shadow Validation

术语是什么？

Shadow Validation是SLINFER中的请求准入控制机制：在将新请求分配给某个model instance之前，虚拟添加该请求并仿真future compute timeline来检测是否会导致任何SLO violation。它基于headroom quantification的compute time estimation，在新的prefill/decode timeline上检查三类violation case。只有全部case通过验证，请求才被实际accept到目标instance；否则尝试其他instance或创建新instance。

从系统架构角度拆解术语：

Shadow Validation在SLINFER中的三种violation case检测：

**Case 1：新请求自身prefill超时**
新请求R在目标instance上的prefill iteration时间过长，导致R的headroom变为负（TTFT violation）。检测：计算R的estimated prefill time Tprefill(R)，模拟R排在当前instance等待队列末尾时，检查 headroom(R) − Tprefill(R) − aggregate_delay ≥ 0。

**Case 2：现有请求被新请求prefill延迟**
新请求的prefill执行期间，目标instance上现有in-flight requests的decode被暂停，导致已有请求的headroom变为负（TPOT violation）。检测：在新请求prefill期间暂停所有现有请求的decode iteration，计算每个r的headroom(r) − Tprefill(R) ≥ 0。

**Case 3：加入新请求后aggregate decode时间超标**
新请求加入后增大了target instance的batch size，导致所有请求的单次decode iteration时间增加，aggregate时间超过TPOT SLO。在共享node上，多个instance轮流执行decode iteration，若单个instance的decode过长，其他instance的请求等待时间超过TPOT SLO。检测：计算新batch size下的estimated decode time，验证对node内所有instance的每个请求：headroom(r) − (new_decode_time + other_instances_decode_time) ≥ 0。

Shadow validation在CPU instance上尤为重要，因为CPU更compute-bound、对batch size增长更敏感（13B CPU上32-batch的TPOT是1-batch的2×）。

术语一般如何实现？如何使用？

实现要点：
- **Simulation粒度**：shadow validation不实际执行推理，仅在headroom时间线上做arithmetic simulation
- **Overestimation margin**：考虑runtime波动和量化误差，所有iteration time overestimate 10%
- **Fallback**：当前instance验证失败→依次尝试其他instance（包括创建新instance）→所有都失败则queue请求
- **Performance**：SLINFER实现中shadow validation overhead随node数增加而轻微增长（更多candidate instance），但总体<1ms

涉及论文标题：
- Towards Resource-Efficient Serverless LLM Inference with SLINFER
