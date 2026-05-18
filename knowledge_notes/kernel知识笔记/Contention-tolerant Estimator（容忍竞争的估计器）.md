## Contention-tolerant Estimator（容忍竞争的估计器）

术语是什么？

Contention-tolerant Estimator是MuxWise中用于建模和预测PD Multiplexing下性能的核心组件。它量化了当Prefill和Decode在同一个GPU上通过SM空间分区并行执行时，由于HBM带宽竞争导致的性能退化程度。该估计器离线profile不同配置下的延迟/吞吐，在线进行插值预测，为SLO-aware Dispatcher提供决策依据。

Estimator由两个子组件构成：(1) **Solo-run Predictor**：基于离线profiling训练的回归模型，预测无竞争时的prefill和decode延时；(2) **Contention Guard**：通过网格采样profiling建立的最坏情况slowdown因子查表，作为保守保护叠加到solo-run预测上。

Solo-run Predictor的回归公式基于Table 2的compute complexity分析（d=hidden dimension，L=总token长度，r=reused context长度，n=L-r=new context长度）：

```
T_Prefill = θ1·Σ(ni²) + θ2·Σ(ni·ri) + θ3·Σ(ni) + θ4   (Equation 1)
T_Decode  = θ1·Σ(ri) + θ2·bs + θ3                        (Equation 2)
```

公式1中prefill延时由new context的二次项（attention中QK计算O(n²d)）、new×reused交叉项（O(L²d)含reused部分）和线性项构成。公式2中decode延时主要由reused context线性项（KV cache访问O(rd)）和batch size线性项构成。论文报告该模型最大偏差prefill 8.16%、decode 8.84%。

Contention Guard使用网格采样profiling覆盖5个变量（prefill new tokens、prefill reused tokens、decode batch size、decode total reused tokens、partition configuration），以16 SM为partition粒度（A100产生6种配置，H100产生7种），token长度按4的幂次采样（2K到128K），总计约7K个样本对/模型-机器对，profile耗时约12小时。Contention guard返回当前配置所在grid cell的最大slowdown factor，A100上最大slowdown≤20%，H100上≤30%。

从kernel调度角度拆解术语：

估计器的建模和计算流程（伪代码）：

```
// 离线Profile阶段
for SM_decode in range(8, SM_total, step=8):   // Arch 9.0最小8 SM
  for SM_prefill in range(0, SM_total - SM_decode, step=8):
    // 同时运行Decode和Prefill benchmark
    decode_latency[SM_decode][SM_prefill] = measure_decode_iteration_latency()
    prefill_throughput[SM_decode][SM_prefill] = measure_prefill_tokens_per_second()

// 在线估计阶段
function estimate_itl(SM_decode, SM_prefill, batch_size):
    // 基线：无竞争时的Decode延迟（仅Decode运行）
    base_decode_latency = profile_no_contention(SM_decode, batch_size)
    
    // 竞争系数：由SM_prefill带来的额外延迟比例
    contention_factor = lookup_or_interpolate(contention_table[SM_decode][SM_prefill])
    
    // 预测实际ITL
    predicted_itl = base_decode_latency * (1 + contention_factor)
    
    return predicted_itl

function find_min_sm_for_decode(batch_size, itl_slo):
    // 二分搜索满足ITL SLO的最小SM_decode
    for SM_decode in [8, 16, 24, ...]:
        predicted_itl = estimate_itl(SM_decode, SM_total - SM_decode, batch_size)
        if predicted_itl <= itl_slo:
            return SM_decode
    return SM_total  // fallback: 所有SM给Decode
```

关键建模挑战：
1. **DRAM访问模式的非线性竞争**：Decode的随机KV页访问与Prefill的连续权重读取在DRAM控制器层面产生复杂的交互，简单的线性模型不足。
2. **L2 Cache污染**：Prefill的连续访问容易刷掉Decode的KV Cache相关L2缓存行。
3. **Warp调度粒度**：不同SM分区上的warp调度器独立运行，但DRAM调度器全局共享。

术语一般如何实现？如何使用？

实际实现结合离线profile和在线校准：先在目标GPU型号上建立竞争矩阵（不同SM组合下的延迟/吞吐），部署时通过少量在线测量校准模型参数。估计器被SLO-aware Dispatcher周期性调用，用于SM分区决策。对于未在profile矩阵中的配置点，使用插值或小规模神经网络拟合。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

---

