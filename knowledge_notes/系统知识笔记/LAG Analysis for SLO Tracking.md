## LAG Analysis for SLO Tracking

术语是什么？

LAG Analysis是AUM Runtime Controller中的SLO跟踪指标，量化每个serving request相对ideal schedule的位置。定义：对于request i在时间t，LAG_i = Σ_{token∈T_i(t)} (d_TPOT - e_token)，其中d_TPOT是TPOT SLO（单位token的deadline），e_token是token实际执行时间，T_i(t)是t时刻已完成的token集合。LAG=0表示exactly on schedule；LAG<0表示behind schedule需加速；LAG>0表示ahead schedule可释放资源。

从系统架构角度拆解术语：

LAG在AUM runtime decision中的作用：
1. **SLO setting**: decode phase SLO_L = d_TPOT + LAG_i——若LAG<0 (behind)→SLO_L更紧→需更多AU资源加速；若LAG≥0 (ahead)→SLO_L宽松→可收获资源
2. **Controller workflow** (Algorithm 1):
   ```
   SLO_H = d_TTFT - t_wait
   SLO_L = d_TPOT + LAG_i
   LAG_i(token, T_i(t)) = Σ_{token∈T_i(t)} (d_TPOT - e_token)
   ```
3. **Per-request tracking**: Runtime Controller维护每个active request的LAG值→每token完成时更新→LAG驱动resource allocation决策

术语一般如何实现？如何使用？

AUM实现中：prefill使用FCFS simple scheduling（LAG仅用于decode）；decode每token完成后更新LAG→Controller据此调整所有decode请求的AU资源配置。区别于GPU-based token pricing，CPU场景prefill token价格设为decode token的9×(α=1.8 vs β=0.2)，反映prefill密集AMX使用的高价值。

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving
