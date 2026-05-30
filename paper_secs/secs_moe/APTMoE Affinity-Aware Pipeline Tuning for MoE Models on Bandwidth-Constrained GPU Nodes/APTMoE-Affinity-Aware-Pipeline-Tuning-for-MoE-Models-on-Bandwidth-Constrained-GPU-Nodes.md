# APTMoE: Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

Yuanxin Wei *Sun Yat-sen University* Guangzhou, China weiyx25@mail2.sysu.edu.cn

Jiangsu Du\* *Sun Yat-sen University* Guangzhou, China dujiangsu@mail.sysu.edu.cn

Jiazhi Jiang *Sun Yat-sen University* Guangzhou, China jiangjzh6@mail2.sysu.edu.cn Xiao Shi *Sun Yat-sen University* Guangzhou, China shix36@mail2.sysu.edu.cn

Xianwei Zhang *Sun Yat-sen University* Guangzhou, China zhangxw79@mail.sysu.edu.cn

Dan Huang\* *Sun Yat-sen University* Guangzhou, China huangd79@mail.sysu.edu.cn

Nong Xiao *Sun Yat-sen University* Guangzhou, China xiaon6@mail.sysu.edu.cn

Yutong Lu *Sun Yat-sen University* Guangzhou, China luyutong@mail.sysu.edu.cn

*Abstract*—Recently, the sparsely-gated Mixture-Of-Experts (MoE) architecture has garnered significant attention. To benefit a wider audience, fine-tuning MoE models on more affordable clusters, which are typically a limited number of bandwidthconstrained GPU nodes, holds promise. However, it is non-trivial to apply existing cost-effective fine-tuning approaches to MoE models, due to the increased ratio of data to computation.

In this paper, we introduce APTMoE, which employs affinityaware pipeline parallelism for fine-tuning MoE models on bandwidth-constrained GPU nodes. We propose an affinity-aware offloading technique that enhances pipeline parallelism for both computational efficiency and model size, and it benefits from a hierarchical loading strategy and a demand-priority scheduling strategy. To improve the computation efficiency and reduce the data movement volume, the hierarchical loading strategy designs three loading phases and efficiently allocates computation across GPUs and CPUs during these phases, leveraging different levels of expert popularity and computation affinity. With the aim of alleviating the mutual interference among the three loading phases and maximizing the bandwidth utilization, the demand-priority scheduling strategy proactively and dynamically coordinates the loading execution order. Experiments demonstrate that APTMoE outperforms existing methods in most cases. Particularly, APT-MoE successfully fine-tunes a 61.2B MoE model on 4 Nvidia A800 GPUs(40GB) and achieves up to 33% throughput improvement compared to the SOTA method.

*Index Terms*—Large language models, Hardware acceleration, High performance computing

