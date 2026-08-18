# <span id="page-4-3"></span>D. Discussion: Bottleneck Shifting across Various Scenarios.

In practice, AFD systems may adopt various deployment configurations, which could shift the memory bottleneck. The deployment configurations include GPU model and quantity, context length, LLM parameter size and architecture, etc. An important contribution of our DRM-based methodology is its ability to analyze, for a given deployment configuration, how the memory configuration of each accelerator affects overall throughput. In this section, we further discuss how

![](_page_5_Figure_0.jpeg)

<span id="page-5-1"></span>Fig. 5. **DRM with GPU upgrade.**  $T_{DP}$  and  $T_{HP}$  denote the throughput of DIMM-PIM under its memory bandwidth constraint, and the throughput of HBM-PIM under its memory capacity constraint, respectively.

different deployment configurations could affect the memory bottleneck

**Different context lengths.** Fig. 3 and Fig. 4-a, b show how different context lengths affect the performance. First, we observe that across various context lengths, the more balanced configuration of DIMM-PIM consistently holds a performance and efficiency advantage, effectively improving throughput while reducing resource over-provisioning. It is because longer contexts exacerbate memory capacity and bandwidth bottlenecks simultaneously.

Second, we observe that with shorter contexts, the performances of various baselines become more comparable, because the marginal benefit of adding resources diminishes when the memory bottleneck is alleviated. We validate this analysis in §VII-B. The impact of context length can also be analyzed in conjunction with Fig. 2-b.

Influence of other configurations. Our DRM method can also analyze the impact of other configuration changes on accelerator selection. For example, with more advanced GPUs for executing FC, as shown in Fig. 5, the DRM reveals that this shifts the curve of FC throughput upward, enhancing HBM-PIM throughput while leaving DIMM-PIM throughput unchanged, thereby closing the performance gap of HBM-PIM relative to DIMM-PIM and even turning it into an advantage. Fig. 3 and Fig. 4-c, d show examples of this: 32× DIMM-PIM-L outperforms AttAcc [61] with A100 (Fig. 3), but underperforms AttAcc when paired with B200 (Fig. 4-d). Other scenarios, such as varying the number of GPUs used for FC, can also be analyzed following a similar methodology.

# <span id="page-4-3"></span>D. Discussion: Bottleneck Shifting across Various Scenarios.

In practice, AFD systems may adopt various deployment configurations, which could shift the memory bottleneck. The deployment configurations include GPU model and quantity, context length, LLM parameter size and architecture, etc. An important contribution of our DRM-based methodology is its ability to analyze, for a given deployment configuration, how the memory configuration of each accelerator affects overall throughput. In this section, we further discuss how

![](_page_5_Figure_0.jpeg)

<span id="page-5-1"></span>Fig. 5. **DRM with GPU upgrade.**  $T_{DP}$  and  $T_{HP}$  denote the throughput of DIMM-PIM under its memory bandwidth constraint, and the throughput of HBM-PIM under its memory capacity constraint, respectively.

different deployment configurations could affect the memory bottleneck

**Different context lengths.** Fig. 3 and Fig. 4-a, b show how different context lengths affect the performance. First, we observe that across various context lengths, the more balanced configuration of DIMM-PIM consistently holds a performance and efficiency advantage, effectively improving throughput while reducing resource over-provisioning. It is because longer contexts exacerbate memory capacity and bandwidth bottlenecks simultaneously.

Second, we observe that with shorter contexts, the performances of various baselines become more comparable, because the marginal benefit of adding resources diminishes when the memory bottleneck is alleviated. We validate this analysis in §VII-B. The impact of context length can also be analyzed in conjunction with Fig. 2-b.

Influence of other configurations. Our DRM method can also analyze the impact of other configuration changes on accelerator selection. For example, with more advanced GPUs for executing FC, as shown in Fig. 5, the DRM reveals that this shifts the curve of FC throughput upward, enhancing HBM-PIM throughput while leaving DIMM-PIM throughput unchanged, thereby closing the performance gap of HBM-PIM relative to DIMM-PIM and even turning it into an advantage. Fig. 3 and Fig. 4-c, d show examples of this: 32× DIMM-PIM-L outperforms AttAcc [61] with A100 (Fig. 3), but underperforms AttAcc when paired with B200 (Fig. 4-d). Other scenarios, such as varying the number of GPUs used for FC, can also be analyzed following a similar methodology.

