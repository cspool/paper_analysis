## Elastic GPU Instance Dispatch（弹性GPU实例调度）

术语是什么？通过联网搜索让回答具体和精准。
Elastic GPU Instance Dispatch 是 PiLLM 的跨GPU资源管理机制，根据预测的 prefill/decode 计算需求动态决定 GPU 实例数。区别于传统 utilization-based autoscaling（依赖 GPU utilization 等硬件指标间接反映负载——长请求时可能 utilization 尚未显著上升但 TTFT/TPOT 已恶化），PiLLM 直接以预测 FLOPs 为决策依据：N_prefill = ceil(Predicted_Prefill_FLOPs / (FLOPs_per_instance · TTFT_target))，N_decode 同理。Dispatcher 采用三级优先级分发：idle instance 优先 → 预计最早完成且满足 deadline 的 active instance → 无合适实例时触发 spike reaction（组成最大可行 batch 并可能激活新 instance）。

从系统架构角度拆解术语：
1. **需求计算**：Global scheduler 每管理窗口根据预测 FLOPs 和目标延迟计算所需 prefill/decode 实例数。
2. **Idle check**：检查 idle instance pool，若足够则直接分配请求。
3. **Active dispatch**：若无 idle，估算各 active instance 的预计完成时间（基于 running batch 剩余 FLOPs 和 per-instance throughput），选择最早完成且满足请求 deadline 的 instance。
4. **No fit → Spike reaction**：若无 instance 能在 deadline 内完成，请求进入 spike queue，触发 spike reaction。
5. **缩容**：当预测工作量下降且 idle instance 数超过预留 buffer 时释放多余 instance。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 LightLLM 的 disaggregated prefill/decode 部署实现，需维护独立的 prefill instance pool 和 decode instance pool。per-instance throughput 通过 offline profiling 在不同 batch size 下测量。请求分发采用集中式 dispatcher/controller——论文指出在数千 GPU 的超大集群中可能成为扩展瓶颈。与 KServe/Kubernetes HPA 的核心区别：PiLLM 直接估算 LLM FLOPs 而非靠硬件利用率滞后反馈，反应窗口为秒级（vs autoscaling 的分钟级）。

涉及论文标题：
- PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
