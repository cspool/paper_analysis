## SLSQP-based Request-level KV Cache Optimizer

术语是什么？通过联网搜索让回答具体和精准。

SLSQP-based Request-level KV Cache Optimizer是eLLM系统request-level调度组件的核心优化器。它在每个调度轮次开始时运行，将batch size b和uncached token ratio r作为联合优化变量，在GPU显存容量和TPOT SLO双约束下，用SciPy的Sequential Least Squares Programming (SLSQP)求解器最大化throughput目标函数。SLSQP是一种基于序列二次规划的约束非线性优化算法，适合中小规模连续变量优化问题。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

优化器在eLLM系统架构中的运转流程：

**优化问题建模**（eLLM Eq. 5）：
```
maximize:  throughput = b / (compute_time(b, r) + overhead(b, r))
subject to:
  1. GPU_memory(b, r) ≤ M_GPU + M_saved(r) - M_overhead(b, r)
     // 显存约束：模型权重 + batch KV cache ≤ 总GPU显存
  2. TPOT_predicted(b, r) ≤ TPOT_SLO
     // 延迟约束：预估per-token output time满足SLO
  3. 0 ≤ r ≤ r_max, 1 ≤ b ≤ b_max
     // 变量边界
```

**计算过程**：
1. **输入收集**：scheduler收集current running requests的sequence length分布、waiting queue size、模型参数（hidden_size, num_heads, num_layers等）、GPU硬件参数（显存总量、SM数量、FLOPS峰值、PCIe带宽）。
2. **约束建模**：
   - GPU_memory(b,r) = model_weights + b × avg_seq_len × (1-r) × per_token_kv_size（cached KV占用）+ r × b × per_layer_kv_size（临时recompute workspace）
   - TPOT_predicted(b,r) = t_K1(b×r×avg_seq_len) + t_K2(b×(1-r)×avg_seq_len) + t_overlap(b,r)
3. **SLSQP求解**：调用`scipy.optimize.minimize(method='SLSQP')`，初始点取上一轮的最优解（warm start加速收敛）。论文报告90%求解<10ms overhead。
4. **输出下发**：最优(b*, r*)下发到layer-level scheduler，layer-level据此分配KV block和选择fused kernel配置。
5. **反馈闭环**：layer-level执行后计算实际额外显存Mo（kernel fusion和overlap的实际显存开销），反馈回optimizer。若Mo偏离预估，下一轮重新求解调整b和r。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
1. 使用SciPy库的`scipy.optimize.minimize`函数，method='SLSQP'，支持等式和不等式约束。
2. 目标函数和约束函数用Python实现，基于profiling数据（各操作的FLOPs和memory footprint）构建解析模型。
3. Warm start：每轮优化以上一轮最优解(b_prev, r_prev)为初始点，显著减少迭代次数。
4. 开销控制：限制SLSQP最大迭代次数，论文90%求解<10ms，不影响serving latency。
5. 论文在eLLM消融实验中禁用optimizer（固定b和r），显示throughput明显退化，验证了在线自适应的价值。

涉及论文标题：
- High Throughput and Low Latency LLM Serving via Adaptive KV Caching
