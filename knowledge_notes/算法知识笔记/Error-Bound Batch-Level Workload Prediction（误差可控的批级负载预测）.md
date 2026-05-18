## Error-Bound Batch-Level Workload Prediction（误差可控的批级负载预测）

术语是什么？通过联网搜索让回答具体和精准。
Error-Bound Batch-Level Workload Prediction 是 PiLLM 提出的统计预测方法，将大数定律和中心极限定理 (CLT) 应用于 LLM serving 场景中请求输入/输出长度的批级预测。核心公式：对大小 |B| 的 batch，其平均输出长度上界（置信度 1-ε）为 μ_d + σ_d/√|B| · Φ⁻¹(1-ε)。其中 μ_d 和 σ_d 为滑动窗口内历史输出长度的均值和标准差；√|B| 项体现大数定律——batch 越大估计越精确；Φ⁻¹ 为标准正态逆CDF；ε 控制错误容忍度（如 ε=0.05 对应 95% 置信上限）。输入长度同理。预测值再通过离线校准的线性系数转换为 prefill FLOPs、decode FLOPs、prefill KV memory 和 decode KV memory 四维资源需求。

从算法pipeline角度拆解术语，预测计算公式：
```
Input: 滑动窗口历史 W = {(L_in_i, L_out_i)}, batch B,
       错误容忍参数 ε_p (prefill), ε_d (decode)
Output: Prefill_FLOPs, Decode_FLOPs, Prefill_KV_mem, Decode_KV_mem

// Step 1: 滑动窗口统计更新
μ_in = mean({L_in_i for all entries in W})
σ_in = std({L_in_i for all entries in W})
μ_out = mean({L_out_i for all entries in W})
σ_out = std({L_out_i for all entries in W})

// Step 2: 批级上界估计（中心极限定理）
z_p = Φ⁻¹(1 - ε_p)   // prefill 预测的 z-score
z_d = Φ⁻¹(1 - ε_d)   // decode 预测的 z-score
L_in_pred = μ_in + (σ_in / sqrt(|B|)) * z_p
L_out_pred = μ_out + (σ_out / sqrt(|B|)) * z_d

// Step 3: 资源转换（离线校准的线性系数）
Prefill_FLOPs = α_pf * L_in_pred + β_pf
Decode_FLOPs  = α_df * L_out_pred + β_df
Prefill_KV_mem = α_pk * L_in_pred + β_pk
Decode_KV_mem  = α_dk * L_out_pred + β_dk
```
校准系数通过 offline profiling 获得：用不同输入/输出长度运行模型，记录实际 FLOPs 和 KV cache 内存，拟合线性回归。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 滑动窗口长度需平衡响应速度和稳定性——PiLLM 按管理窗口设置（秒级）；(2) offline profiling 在部署前一次性完成；(3) ε_p 和 ε_d 可独立调节以适应 prefill（compute-bound）和 decode（memory-bound）的不同 SLO 敏感度；(4) CLT 近似在 |B|≥30 时良好，小 batch 时标准差项 σ/√|B| 较大使预测更保守；(5) 分布漂移时预测变差，PiLLM 以 spike reaction 兜底而非在线调参。该方法给出了从 LLM 服务统计量到 FLOPs/memory 的端到端可量化转换链路。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

