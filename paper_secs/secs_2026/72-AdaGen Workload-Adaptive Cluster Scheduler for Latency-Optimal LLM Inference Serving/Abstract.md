# Abstract

The inference workloads of Large Language Models (LLMs) pose significant latency and cost challenges due to increasing model sizes and demand for real-time responses. Existing cluster schedulers for multi-instance LLM serving primarily focus on load balancing to optimize memory usage, which is insufficient for workloads with diverse request characteristics. In such cases, the compute layout—the arrangement of tokens across iterations within each instance—plays a crucial role in determining latency. We propose AdaGen, a workload-adaptive cluster scheduler that minimizes latency and thus maximizes SLO attainment by optimizing compute layouts across instances. AdaGen employs a multistep scheduling strategy: it first classifies requests based on prefill and decode lengths, then balances load, and finally performs selective distributed execution across instances. Each step incrementally refines the scheduling based on the compute layouts derived from the decision of the previous step. To avoid the overhead of actual execution to generate the layouts, AdaGen introduces a novel simulation-based estimator. Extensive experiments using production workloads show that AdaGen achieves up to 3.6× higher SLO attainment and 2× better cost-efficiency compared to the existing systems, while ensuring scalability.

CCS Concepts: • Computer systems organization → Distributed architectures; • Computing methodologies → Distributed algorithms; Natural language processing.

<sup>†</sup>Currently at Meta, work done when the author was at HPE.

![](_page_0_Picture_14.jpeg)

[This work is licensed under a Creative Commons Attribution-](https://creativecommons.org/licenses/by-nc-nd/4.0)[NonCommercial-NoDerivatives 4.0 International License.](https://creativecommons.org/licenses/by-nc-nd/4.0)

EUROSYS '26, Edinburgh, Scotland Uk

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 <https://doi.org/10.1145/3767295.3769345>

Keywords: LLM, Inference, Infrastructure, Scheduling

#### ACM Reference Format:

Sudipta Saha Shubha, Ayush Goel, Diman Zad Tootaghaj, Khaled Diab, Hardik Soni, K. K. Ramakrishnan, Puneet Sharma, and Haiying Shen. 2026. AdaGen: Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving. In European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, [17](#page-16-0) pages. [https:](https://doi.org/10.1145/3767295.3769345) [//doi.org/10.1145/3767295.3769345](https://doi.org/10.1145/3767295.3769345)

