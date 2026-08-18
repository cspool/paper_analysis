## Goodput（SLO 约束下的有效吞吐）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Goodput（有效吞吐）是在满足 SLO 约束（TTFT≤400ms、TBT≤100ms）的条件下系统每秒可成功完成的请求数（RPS），区别于不考虑延迟的裸吞吐：只统计"延迟达标"的请求，反映真正有用（SLO-compliant）的产出。本论文（Rearchitecting the Datacenter Lifecycle for AI）把 goodput 作为容量规划的核心中间量：roofline 模型预测 (硬件, 模型, 负载) 的 TTFT/TBT，不断增大负载直到任一延迟越过 SLO 阈值，取违反前的最大 RPS 为 goodput；再用需求 RPS / goodput 求最小 GPU 供给量与对应利用率，作为 IT provisioning 和 TCO 计算的基础。Goodput 让"FLOPS 增长 ≠ 实际效率"显性化——论文由此得出 A100 对 1B 模型每美元效率高于 H100、而对 70B 低约 3×10^10 的结论，直接支撑刷新策略（跳过边际代际、延长高性价比代际寿命）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Goodput 驱动的容量决策流程：
```
for each (model, GPU generation):
    for load R in [R_low, R_high]:              # 增压扫描
        (ttft, tbt) = roofline(model, GPU, R)   # 预测延迟
        if ttft > 400ms or tbt > 100ms:         # SLO 违反
            goodput = R - ΔR; break             # 违反前最大 RPS
    utilization = demand_RPS / goodput
    provisioned_GPUs = ceil(demand_RPS / goodput)
→ 供给量 + 利用率进入 TCO 模型 → 刷新策略蒙特卡洛搜索
```
与调度场景不同（在线调度器用 goodput 比较 serving 系统），本文的 goodput 是"跨 GPU 代际、跨 15 年"的容量指标——同一 SLO 下不同代 GPU 的 goodput 差距决定了"何时值得换新硬件"。论文验证：roofline 预测的 goodput 与 vLLM 实测（2K 序列、batch 8、TP1/TP4/TP8、T4/V100/A100/H100/H200）误差 <5%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：SLO-aware serving 系统（MuxWise/Bullet 等）以 SLO 约束下最大化 goodput 为目标在线调度；本文在 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）中把 goodput 计算内建于 performance 模块（roofline 模型 + SLO 阈值），随模拟器每季度调用，用户可通过 YAML 配置 SLO 阈值（TTFT/TBT）与负载增长曲线。论文中 SLO 取 400ms/100ms 源自 DynamoLLM [105] 的 Azure 生产推断。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
