# E. Cold Start and Algorithm Overhead

Cold start is a critical issue in serverless computing. As shown in Fig. 11, we compare the cold start times across different methods. While all approaches share the same container startup time due to a common base image, *Remoe* achieves the lowest cold start time, with a reduction of up to 57.14%. This improvement stems from its strategy of partitioning numerous experts into separate serverless functions, whose cold starts (labeled as REMOTE) can overlap with the main model's startup. Furthermore, Remoe's optimization logic (CALCULATE) is highly efficient; its overhead is negligible and introduces no additional waiting time.

## VI. RELATED WORK

Serverless LLM Inference. Research on serverless LLM inference has focused on several key optimizations. To mitigate the cold start problem, techniques such as pipeline parallelism [33] and multi-tiered local storage [6] have been explored to accelerate model loading. Another key focus is resource allocation, where efforts include using elastic hardware sharing to boost GPU utilization [34] and combining adaptive configuration with real-time monitoring for stable serving [35]. For cost optimization, Liu et al. [14] proposed a specific scheduling algorithm ODS for serverless MoE inference, although limited to a pure CPU environment. Despite these advances, cost-efficient serverless MoE inference, particularly on GPU-CPU hybrid architectures, remains largely underexplored.

<span id="page-8-1"></span>![](_page_8_Figure_9.jpeg)

Fig. 11: Time for cold start, predicting, and optimization

GPU Memory-Constrained MoE Inference. Prediction-based expert caching is the dominant approach for memory-efficient Mixture-of-Experts (MoE) inference. Strategies range from using historical data [7], [8] to more fine-grained, layer-level predictions, which have been successfully applied to memory-constrained devices with enhancements like mixed-precision loading [10] and graceful degradation [36]. Another line of work employs dedicated ML predictors to achieve higher caching accuracy [9], [37]. While effective, these token-level online prediction strategies are ill-suited for serverless environments that require resource pre-allocation, as the frequent adjustments would incur severe cold start overhead during execution.

## VII. CONCLUDING REMARKS

To minimize inference cost, we propose a heterogeneous system *Remoe*. We design algorithms for expert activation prediction, resource pre-allocation, and joint memory-replica optimization. Our implementation of *Remoe* on Kubernetes shows that it reduces inference cost and cold start latency significantly. Our current approach relies on idealized assumptions about the serverless environment. Consequently, our future work will focus on designing a highly fault-tolerant system to address real-world operational complexities such as unpredictable cold start times and network latency fluctuations.

#### APPENDIX

