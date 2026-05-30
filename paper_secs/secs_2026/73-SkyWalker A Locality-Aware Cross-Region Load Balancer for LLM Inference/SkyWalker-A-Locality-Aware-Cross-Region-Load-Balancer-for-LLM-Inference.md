# SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference

Tian Xia<sup>†</sup> Ziming Mao<sup>†</sup> Jamison Kerney<sup>†</sup> Ethan J. Jackson<sup>†</sup> Zhifei Li<sup>§</sup> Jiarong Xing<sup>†¶</sup> Scott Shenker<sup>†</sup> Ion Stoica<sup>†</sup> <sup>†</sup> UC Berkeley <sup>§</sup>Renmin University of China <sup>¶</sup>Rice University <sup>¢</sup>ICSI

### **Abstract**

Serving Large Language Models (LLMs) efficiently in multiregion setups remains a challenge. Due to cost and GPU availability concerns, providers typically deploy LLMs in multiple regions using instance with long-term commitments, like reserved instances or on-premise clusters, which are often underutilized due to their region-local traffic handling and diurnal traffic variance. In this paper, we introduce SkyWalker, a multi-region load balancer for LLM inference that aggregates regional diurnal patterns through cross-region traffic handling. By doing so, SkyWalker enables providers to reserve instances based on expected global demand, rather than peak demand in each individual region. Meanwhile, SkyWalker preserves KV-Cache locality and load balancing, ensuring cost efficiency without sacrificing performance. SkyWalker achieves this with a cache-aware cross-region traffic handler and a selective pushing based load balancing mechanism. Our evaluation on real-world workloads shows that it achieves 1.12-2.06× higher throughput and 1.74-6.30× lower latency compared to existing load balancers, while reducing total serving cost by 25%.

 $\begin{tabular}{ll} $CCS\ Concepts: \bullet Computing\ methodologies \to Distributed \\ artificial\ intelligence;\ Distributed\ computing\ methodologies. \\ \end{tabular}$ 

*Keywords:* Load Balancing, AI Serving, Multi-Region, Cloud Computing

#### **ACM Reference Format:**

Tian Xia, Ziming Mao, Jamison Kerney, Ethan J. Jackson, Zhifei Li, Jiarong Xing, Scott Shenker, Ion Stoica. 2026. SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference. In 21st European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3767295.3769353

![](_page_0_Picture_8.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

EUROSYS '26, Edinburgh, Scotland Uk
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2212-7/26/04
https://doi.org/10.1145/3767295.3769353

