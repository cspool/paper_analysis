# Matrix: Peer-to-Peer Multi-Agent Synthetic Data Generation Framework

Dong Wang1,† , Yang Li1,† , Ansong Ni1,† , Ching-Feng Yeh<sup>1</sup> , Youssef Emad<sup>1</sup> , Xinjie Lei<sup>1</sup> , Liam Robbins<sup>1</sup> , Karthik Padthe<sup>1</sup> , Hu Xu<sup>1</sup> , Xian Li<sup>1</sup> , Asli Celikyilmaz<sup>1</sup> , Ramya Raghavendra<sup>1</sup> , Lifei Huang<sup>1</sup> , Carole-Jean Wu1,† , Shang-Wen Li1,†

Synthetic data has become increasingly important for training large language models, especially when real data is scarce, expensive, or privacy-sensitive. Many such generation tasks require coordinated multi-agent workflows, where specialized agents collaborate to produce data that is higher quality, more diverse, and structurally richer. However, existing frameworks for multi-agent synthesis often depend on a centralized orchestrator, creating scalability bottlenecks, or are hardcoded for specific domains, limiting flexibility. We present Matrix, a decentralized framework that represents both control and data flow as serialized messages passed through distributed queues. This peer-to-peer design eliminates the central orchestrator. Each task progresses independently through lightweight agents, while compute-intensive operations, such as LLM inference or containerized environments, are handled by distributed services. Built on Ray, Matrix scales to tens of thousands of concurrent agentic workflows and provides a modular, configurable design that enables easy adaptation to a wide range of data generation workflows. We evaluate Matrix across diverse synthesis scenarios, such as multi-agent collaborative dialogue, web-based reasoning data extraction, and tool-use trajectory generation in customer service environments. In all cases, Matrix achieves 2–15× higher data generation throughput under identical hardware resources, without compromising output quality.

Date: April 21, 2026

Correspondence: Dong Wang [dwoanngg@gmail.com](mailto:dwoanngg@gmail.com), Shang-Wen Li [shangwel@meta.com](mailto:shangwel@meta.com)

Code: <https://github.com/facebookresearch/matrix>

