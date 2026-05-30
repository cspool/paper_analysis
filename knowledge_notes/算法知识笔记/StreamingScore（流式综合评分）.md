## StreamingScore（流式综合评分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
StreamingScore 是 StreamingEval 提出的综合评估指标，将流式视频理解中的四个核心维度（吞吐、准确率、延迟、资源消耗）整合为单一可调权重的标量分数。定义：

$$\text{StreamingScore}(\mathbf{w}) \triangleq \frac{\text{MaxFPS}^{w_f} \cdot \text{Acc}^{w_a}}{\text{TTFT}^{w_t} \cdot M^{w_r}}$$

其中 $M \triangleq \text{Mem} \cdot \ln(\text{Params})$（结合内存占用的资源项），权重约束为 $w_f, w_a, w_t, w_r \ge 0, w_f + w_a + w_t + w_r = 1$。更高的 StreamingScore 表示模型在更高吞吐下实现更好准确率、更低首 token 延迟和更低资源消耗。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# StreamingScore 计算
# 输入: 模型在统一流式协议下的四项指标
Acc = accuracy_on_benchmark(model, dataset)     # [0, 100]
MaxFPS = max_sustainable_fps(model)              # > 0
TTFT = time_to_first_token(model)               # seconds
Mem = memory_bank_budget                        # GB
Params = model_parameter_count                  # billions

# 资源项
M_resource = Mem * ln(Params)

# 默认统一权重 (w_f=w_a=w_t=w_r=0.25)
StreamingScore = (MaxFPS^0.25 * Acc^0.25) / (TTFT^0.25 * M_resource^0.25)

# 场景感知权重示例:
# "Best Answer" (w_a=0.4, w_f=w_t=w_r=0.2)
# A_score = (MaxFPS^0.2 * Acc^0.4) / (TTFT^0.2 * M_resource^0.2)

# "Interaction First" (w_t=0.4, others=0.2)
# I_score = (MaxFPS^0.2 * Acc^0.2) / (TTFT^0.4 * M_resource^0.2)

# "Edge Resource-Saving" (w_r=0.4, others=0.2)
# R_score = (MaxFPS^0.2 * Acc^0.2) / (TTFT^0.2 * M_resource^0.4)

# "Throughput First" (w_f=0.4, others=0.2)
# T_score = (MaxFPS^0.4 * Acc^0.2) / (TTFT^0.2 * M_resource^0.2)
```

StreamingEval 实验表明：不同权重下模型排名可互换（如 Qwen3-VL 在 Best Answer 排第 1，Flash-VStream 在其余三项排第 1），但整体趋势统计稳健（Spearman ρ ∈ [0.972, 0.993]），降低了模型仅通过倾斜权重 "刷榜" 的风险。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
StreamingScore 作为默认统一评估指标，使用均等权重（0.25, 0.25, 0.25, 0.25），同时支持四种部署场景偏好。在 StreamingEval 开源代码中通过 `compute_streaming_score(acc, maxfps, ttft, mem, params, weights)` 函数计算。局限性：(a) 权重选择是主观的——不同应用场景合理选择不同；(b) 单项指标异常值可能主导分数（如 MaxFPS=0.14 的 VideoChatOnline 在任意权重下 StreamingScore 都极低）；(c) StreamingScore 不替代单独指标分析，而是补充综合视角。

涉及论文标题：
- StreamingEval__A_Unified_Evaluation_Framework_for_Streaming_Video_Understanding
