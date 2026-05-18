## SM-scaling Roofline Model (SRM)（SM缩放Roofline模型）

术语是什么？

SM-scaling Roofline Model (SRM)是Bullet提出的性能预测模型，用于建模LLM推理中prefill和decode阶段在不同SM数量下的延迟上界。与传统roofline模型不同，SRM将SM数量作为自变量，同时建模compute（Tensor Core FLOPS）、memory（HBM bandwidth）和network（NVLink bandwidth，多GPU场景）三个维度的性能饱和边界随SM分配变化的动态关系。SRM仅需少量offline concurrent sample校准，声称profiling overhead低于1小时，在线预测和更新开销为微秒级。

从kernel调度角度拆解术语：

SRM的建模和计算流程：

```
// SRM核心公式：给定SM数量s，预测kernel执行延迟T(s)
// T(s) = max(T_compute(s), T_memory(s), T_network(s))

function predict_latency(sm_count, op_type, tensor_shapes):
    // 1. Compute bound: T_compute = FLOPs / (peak_FLOPS_per_SM * sm_count * efficiency)
    //    小SM数量下compute是瓶颈（prefill场景）
    roofline_compute = total_FLOPs / (PEAK_TFLOPS_PER_SM * sm_count * EFF_COMPUTE)

    // 2. Memory bound: T_memory = bytes / (HBM_bandwidth * contention_factor)
    //    大SM数量下memory bandwidth饱和成为瓶颈（decode场景）
    roofline_memory = total_bytes / (PEAK_HBM_BW * BANDWIDTH_EFF)

    // 3. Network bound（多GPU场景）: T_network = bytes / NVLink_bandwidth
    roofline_network = comm_bytes / (NVLink_BW * sm_count / total_SMs)

    return max(roofline_compute, roofline_memory, roofline_network)

// 校准：运行少量concurrent sample（prefill+decode并行的代表性配置）
// 用实际测量修正contention_factor和efficiency参数
function calibrate():
    for (sm_prefill, sm_decode) in sparse_samples:  // 稀疏采样，非全网格
        run_concurrent_prefill_decode(sm_prefill, sm_decode)
        // 采集实际延迟
        actual_decode_latency = measure_decode_step()
        actual_prefill_latency = measure_prefill_layer()
        // 修正模型参数（线性回归或简单拟合）
        update_model_params(actual_latency)

    // Bullet声称profiling overhead <1小时（对比MuxWise的Contention Guard约12小时）
```

SRM与MuxWise Contention-tolerant Estimator的关键区别：
- **Estimator**：基于全网格profiling（~7K样本对/模型-机器对，12小时），用solo-run predictor + contention guard查表预测
- **SRM**：基于roofline分析模型，用稀疏concurrent sample校准（<1小时），通过roofline边界预测

术语一般如何实现？如何使用？

SRM在Bullet的SLO-aware scheduler中作为performance estimator的核心组件运行：
1. 初始化阶段：对目标模型-硬件组合运行SRM校准（少量代表性SM配置下的concurrent prefill+decode sample）
2. 在线阶段：scheduler调用SRM预测不同SM分配方案下的TTFT和TPOT
3. 自适应修正：在线统计持续修正contention_factor等参数
4. 多维度建模：compute saturation（prefill对小SM数量敏感）、memory saturation（decode对HBM带宽敏感）、network saturation（多GPU tensor parallelism场景）

涉及论文标题：
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

