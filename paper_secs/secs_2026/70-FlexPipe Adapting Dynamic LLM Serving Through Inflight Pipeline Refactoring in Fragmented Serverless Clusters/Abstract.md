# Abstract

Serving Large Language Models (LLMs) in production faces significant challenges from highly variable request patterns and severe resource fragmentation in serverless clusters. Current systems rely on static pipeline configurations that struggle to adapt to dynamic workload conditions, leading to substantial inefficiencies.

We present FlexPipe, a novel system that dynamically reconfigures pipeline architectures during runtime to address these fundamental limitations. FlexPipe decomposes models into fine-grained stages and intelligently adjusts pipeline granularity based on real-time request pattern analysis, implementing three key innovations: fine-grained model partitioning with preserved computational graph constraints, inflight pipeline refactoring with consistent cache transitions, and topology-aware resource allocation that navigates GPU fragmentation. Comprehensive evaluation on an 82- GPU cluster demonstrates that FlexPipe achieves up to 8.5× better resource efficiency while maintaining 38.3% lower latency compared to state-of-the-art systems, reducing GPU reservation requirements from 75% to 30% of peak capacity.

CCS Concepts: • Computer systems organization → Distributed architectures; Cloud computing; • Computing methodologies → Distributed artificial intelligence.

Keywords: Large Language Models, Serverless Computing, Resource Efficiency, Pipeline Parallelism, Dynamic Scaling

<sup>∗</sup>Corresponding Author

![](_page_0_Picture_12.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0/legalcode)[tional License.](https://creativecommons.org/licenses/by/4.0/legalcode)

EUROSYS '26, April 27–30, 2026, Edinburgh, Scotland Uk © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 <https://doi.org/10.1145/3767295.3769316>

#### ACM Reference Format:

Yanying Lin, Shijie Peng, Chengzhi Lu, Chengzhong Xu, and Kejiang Ye\* . 2026. FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters. In European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, [17](#page-16-0) pages. <https://doi.org/10.1145/3767295.3769316>

