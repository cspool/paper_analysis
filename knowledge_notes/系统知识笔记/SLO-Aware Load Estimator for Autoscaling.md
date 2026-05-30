## SLO-Aware Load Estimator for Autoscaling

术语是什么？

SLO-Aware Load Estimator 是 ElasticMoE Coordinator 中的负载监测与缩放决策组件。它持续跟踪推理实例的 SLO 达标率（TTFT < α 且 TPOT < β 的请求比例），当 SLO 持续低于阈值（如 90%）时触发 scale-up，当 SLO 持续高于阈值且利用率低时触发 scale-down 回收资源。与 utilization-based autoscaling（基于 GPU utilization 等硬件指标间接反映负载，存在滞后）不同，SLO-aware 直接以端到端用户体验指标为决策依据，反应更快。

从系统架构角度拆解术语：

```
监控循环（每 T_monitor 窗口）:
  total = 窗口内完成的请求总数
  met = TTFT < α AND TPOT < β 的请求数
  slo_rate = met / total

判定逻辑:
  if slo_rate < 90% for N_consecutive_windows → trigger scale-up(DP+1, EP+(DP_increment))
  elif slo_rate > 99% and util < 50% for M_windows → trigger scale-down
  after scaling → cooldown_period → resume monitoring
```

术语一般如何实现？如何使用？

Coordinator (Python/ZMQ) 中实现，通过 vLLM request-level metrics 计算 SLO。论文中 SLO 阈值为 TTFT ≤ 1000ms + TPOT ≤ 1000ms，ElasticMoE 可维持 SLO≥90% 到 ~8.7 RPS（DeepSeek V2 Lite），显著优于 Vertical Cold Restart 和 Colocated baselines。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

---
