## Inference Latency Simulation Model (Random Forest-based, for MoE Parallel Strategy)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Inference Latency Simulation Model 是 HAP 中用于精确估计不同并行策略下 MoE 推理延迟的机器学习仿真模型。包含两个子模型：(1) **计算仿真模型**：基于 FLOPs 估计计算延迟，T_cal = (F_module / Max_FLOPs/s) × η，其中 F_module 为模块的浮点运算量，Max_FLOPs/s 为 GPU 理论峰值算力。η 是效率系数，由随机森林回归模型根据 batch size、hidden dim、sequence length 等参数拟合。(2) **通信仿真模型**：基于数据量和带宽估计通信延迟，T_comm = (V_data / Bandwidth) × ρ，其中 V_data 为集合通信数据量，Bandwidth 为节点内网络带宽（NVLink 或 PCIe）。ρ 是通信效率系数，由另一个随机森林回归模型仅以数据量和带宽为输入拟合。两个仿真模型训练数据来源于实测的算子延迟 profiling，计算误差 <10%，通信误差 <5%。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

仿真模型在 HAP 框架中作为 ILP 求解的前置步骤，提供延迟成本参数：

```
# HAP 仿真模型训练与使用流程

# Phase 1: Microbenchmark 数据采集
latency_data = []
for (b, s, h) in benchmark_configs:
    for parallel_strategy in [TP, EP, DP]:
        # 实测 Attention 模块延迟
        T_attn_measured = benchmark_attention(b, s, h, strategy)
        # 实测 Expert 模块延迟
        T_expert_measured = benchmark_expert(b, s, h, strategy, num_experts)
        # 实测集合通信延迟
        T_comm_measured = benchmark_collective(V_data, comm_type)
        latency_data.append({...})

# Phase 2: 特征工程 + 模型训练
# 计算仿真模型: η = f(b, s, h, ...)
# 多项式特征展开增强表示能力
poly_features = PolynomialFeatures(degree=2).fit_transform(
    [b, s, h, batch, seqlen, hidden]
)
rf_cal = RandomForestRegressor(n_estimators=100)
rf_cal.fit(poly_features, eta_targets)

# 通信仿真模型: ρ = g(V_data, Bandwidth)
rf_comm = RandomForestRegressor(n_estimators=100)
rf_comm.fit([[V_data, Bandwidth]], rho_targets)

# Phase 3: 推理时使用
def simulate_attention_latency(strategy, b, s, h):
    F = compute_flops_attention(b, s, h, strategy)
    eta = rf_cal.predict(poly_features(b, s, h))
    return (F / Max_FLOPs) * eta

def simulate_expert_latency(strategy, b, s, h, num_experts, top_k):
    F = compute_flops_expert(b, s, h, num_experts, top_k, strategy)
    eta = rf_cal.predict(poly_features(b, s, h))
    return (F / Max_FLOPs) * eta

def simulate_comm_latency(strategy, b, s, h):
    V = compute_comm_volume(strategy, b, s, h)
    rho = rf_comm.predict([[V, Bandwidth]])
    return (V / Bandwidth) * rho
```

计算仿真模型中 η < 1 反映实际 GPU 利用率（因内存带宽限制、kernel launch overhead 等无法达到理论峰值 FLOPs）。通信仿真模型中 ρ > 1 反映实际通信开销高于理论带宽（因协议开销、数据对齐、PCIe 链路效率等）。随机森林选择的理由是轻量级、低推理开销、对非线性关系拟合好，适合作为 ILP 的快速延迟评估器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HAP 使用 scikit-learn RandomForestRegressor 实现两个仿真模型。训练数据通过 DeepSpeed-FastGen 环境下的系统化 benchmarking 采集（覆盖不同 batch size、sequence length、并行策略组合）。多项式特征展开（degree=2）增强特征空间的表示能力，再输入随机森林。模型训练和推理均在 CPU 上执行（开销极小），总训练时间远小于一次推理。仿真结果作为 ILP 目标函数中各 T_attn、T_expert、T_comm 项的系数输入。

涉及论文标题：
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
