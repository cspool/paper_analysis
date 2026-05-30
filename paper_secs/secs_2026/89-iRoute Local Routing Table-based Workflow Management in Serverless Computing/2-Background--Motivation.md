# 2 Background & Motivation

Workflows orchestrate sequences of functions, typically modeled as DAGs, to manage complex processes and data flows.

<span id="page-1-2"></span><sup>&</sup>lt;sup>1</sup>https://github.com/tanksys/iRoute

With the advancement of serverless technology, *serverless workflows* have attracted significant attention from cloud providers for building latency-sensitive web services. For example, AWS has rearchitected web services like e-commerce [\[18](#page-13-14)] and airline booking [\[19\]](#page-13-15) using serverless. Microsoft also supports web applications based on Azure Functions[[20](#page-13-16)]. However, due to the high overhead of inter-function data transmission (e.g.,> 70% of the overall latency [\[10](#page-13-7)]), *serverless workflow*-based web services often struggle to meet strict latency requirements. As the adoption of*serverless workflows* continues to grow (e.g., > 31% [[4](#page-13-3)]), reducing workflow communication latency has become a critical challenge for *serverless workflow*-based web services.

In *serverless workflows*, communication between functions primarily consists of two steps: (1) *inter-function routing* and (2) *inter-instance communication*. Although Unum[[9](#page-13-6)] has reduced routing overhead by offloading the resolution process from the orchestrator to local instances, there is still significant overhead in locating instances and transmitting data between instances. We analyze the *posting* feature in the benchmark of *Social Network* [[8](#page-13-5)], and find that the computation times of its 10 functions range from 28 to 1.6 , with even 6 functions having latencies below 150 . However, the data transfer overhead between two functions can surpass 3.3 , more than 20× the computation time. Consequently, the substantial overhead incurred by data transfer poses a major challenge in adopting serverless computing for latency-sensitive services.

