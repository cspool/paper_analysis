## Hyperparameter Optimizer for MoE Offloading Systems（MoE Offloading 系统的超参数优化器）

术语解释
Hyperparameter Optimizer 是 SpecMoEOff 中自动搜索最优 offloading + speculative decoding 超参数的组件。它将超参数优化形成为最大化 throughput 的搜索问题，采用 convex optimization（凸优化预决定部分参数）+ profiling-based estimator（基于 DAG 模拟的吞吐量估计）的混合方法。

术语是什么？
优化目标函数（Eq. 1）：
给定硬件 H、模型 M、请求特征 R，找到 P* = argmax_{P=(b,m,k,S)} T(H, M, R, P)，其中 b=batch size, m=micro-batch size, k=draft token 数量, S=memory+execution strategies。

三阶段求解流程：
1. **Convex Optimization 预决定**：b = max batch fitting in CPU memory；S_memory = maximize GPU cache for draft model（多余 GPU HBM 给 target KV cache）
2. **Profiling-based Estimator**：micro-benchmark 收集 GPU MoE/GPU Attention/CPU Chunked Attention/HtoD Transfer 性能数据 → linear model fitting → DAG simulator（节点=event，边=dependency，topological sort 估计 execution time）→ k → accepted tokens 映射
3. **Grid Search for k**：k 的范围有限，逐个评估 → 选 max throughput

从系统架构角度拆解术语：
```
# Hyperparameter Optimizer 工作流程
Offline Phase:
  1. 收集硬件配置 H（peak TFLOPS, bandwidth, memory size）
  2. 收集模型规格 M（expert count, hidden dim, layer count）
  3. 运行 micro-benchmarks 收集各算子的 performance profile
  4. 建立 acceptance rate mapping: k → a(k)（通过离线数据集评估）

Online Phase（部署前执行一次）:
  1. Convex optimization: 预决定 b, m, S_memory, S_execution
  2. For k in 1..K_max:
       DAG = build_execution_graph(b, m, k, profiled_ops)
       exec_time = topological_sort(DAG)
       n_tokens = acceptance_lookup(k)
       throughput[k] = n_tokens / exec_time
  3. k_opt = argmax(throughput)
  4. 生成 execution plan

Runtime: 动态调整 k（随 sequence length 变化）
  - Generation early: k high（prefix 短, KV cache 小 → more GPU mem）
  - Generation later: k reduced（seqlen 长, KV cache 大）
  - Request completion: k increased（free up GPU mem from finished reqs）
```

术语一般如何实现？如何使用？
SpecMoEOff 在 SGLang 基础上实现，profile-based estimator 的 error 在 10% 以内（Table 3: iteration time estimation error 9.7%）。动态调整 k 在输出长度相对于输入较短时 gain 有限（~2%），但在长输出场景下更显著。适用场景：需要自动适应不同硬件（A30/4090D）、不同模型（Mixtral）、不同 workload（APPS/CNN-DailyMail）的 MoE offloading 部署。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding
