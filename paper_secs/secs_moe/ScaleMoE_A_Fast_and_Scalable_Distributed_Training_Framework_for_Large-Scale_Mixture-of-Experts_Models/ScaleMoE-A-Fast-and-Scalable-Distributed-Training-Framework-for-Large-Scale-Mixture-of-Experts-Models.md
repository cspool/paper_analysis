# ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

Seohong Choi *Sungkyunkwan University* Suwon, South Korea lneil@g.skku.edu

Huize Hong *Sungkyunkwan University* Suwon, South Korea heatherhong@g.skku.edu

Tae Hee Han† *Sungkyunkwan University* Suwon, South Korea than@skku.edu

Joonsung Kim† *Sungkyunkwan University* Suwon, South Korea joonsungkim@skku.edu

*Abstract*—The size of pre-trained models has continuously increased to support growing demands for solving more complex problems. Especially, *mixture-of-experts* (MoE) model has become the most popular approach, enabling systems to easily train extremely large-scale models with relatively lower computational requirements. However, the current distributed training frameworks cannot achieve scalable performance for these large-scale MoE models due to substantial communication overheads.

In this paper, we propose ScaleMoE, a fast and scalable distributed training framework for large-scale MoE models. We first identify three problems in state-of-the-art distributed training frameworks: high all-to-all communication overheads, severe load imbalance in expert selection, and insufficient consideration of heterogeneous networks. We propose three novel optimizations to resolve these problems. First, to reduce communication volumes, we propose *adaptive all-to-all communication* that eliminates unnecessary zeros caused by zero padding. Second, to address the load imbalance in expert selection, we propose *dynamic expert clustering* that rebalances experts using a novel clustering methodology. Lastly, to further minimize communication overheads, we propose *topology-aware expert remapping* that carefully maps experts to GPU devices while considering heterogeneous network bandwidths. Our evaluations show that ScaleMoE achieves scalable performance, reducing all-to-all communication overheads by up to 81%. In general, ScaleMoE significantly improves system performance, achieving a speedup of up to 3.3× compared to the state-of-the-art framework.

