# **7 Conclusion**

Through our research and experiments, we have derived several key conclusions regarding fine-grained MoE models.

Compared to typical FFNs, MoE layers, despite having the same computational requirements, are more challenging to execute efficiently due to increased scheduling overhead and weaker batch-processing effects. However, expert parallelism offers potential for optimization. Expert skipping during inference improves throughput. Although the presence of

a shared backbone and attention mechanism constrains acceleration gains, small-batch and large-batch scenarios still exhibit significant improvements, whereas the effects remain less pronounced at intermediate concurrency levels. Encouragingly, the performance impact of expert skipping is minimal, and with appropriate skipping strategies, performance degradation can be further mitigated. Our best approach can increase throughput by at least 10% without any loss in performance on Deepseek-V3. On a global level, reducing the number of total experts before inference can yield a moderate increase in throughput, though the acceleration effect diminishes when the expert count is minimized. While reducing memory consumption lowers the deployment barrier, it also results in substantial performance loss, which limits its practical usability.

Overall, we believe that MoE optimization remains a promising research direction, both in terms of designing more efficient inference systems and exploring its potential from a language modeling perspective.

