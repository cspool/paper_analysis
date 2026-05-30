# HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

Wenxiang Lin<sup>†</sup>, Xinglin Pan<sup>‡</sup>, Lin Zhang<sup>§</sup>, Shaohuai Shi<sup>†\*</sup>, Xuan Wang<sup>†</sup>, Xiaowen Chu<sup>‡§</sup>

<sup>†</sup>School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen

<sup>‡</sup>Data Science and Analytics Thrust, The Hong Kong University of Science and Technology (Guangzhou)

§Department of Computer Science and Engineering, The Hong Kong University of Science and Technology wenxianglin@stu.hit.edu.cn, xpan413@connect.hkust-gz.edu.cn, lzhangbv@connect.ust.hk

shaohuais@hit.edu.cn, wangxuan@cs.hitsz.edu.cn, xwchu@ust.hk

Abstract—The sparsely activated mixture-of-experts (MoE) transformer has become a common architecture for large language models (LLMs) due to its sparsity, which requires fewer computational demands while easily scaling the model size. In MoE models, each MoE layer requires to dynamically choose tokens to activate particular experts for computation while the activated experts may not be located in the same device or GPU as the token. However, this leads to substantial communication and load imbalances across all GPUs, which obstructs the scalability of distributed systems within a GPU cluster. To this end, we introduce HierMoE to accelerate the training of MoE models by two topology-aware techniques: 1) token deduplication to reduce the communication traffic, and 2) expert swap to balance the workloads among all GPUs. To enable the above two proposed approaches to be more general, we build theoretical models aimed at achieving the best token duplication and expert swap strategy under different model configurations and hardware environments. We implement our prototype HierMoE system atop Megatron-LM and conduct experiments on a 32-GPU cluster with DeepSeek-V3 and Qwen3-30B-A3B models. Experimental results show that our HierMoE achieves  $1.55\times$  to  $3.32\times$  faster communication and delivers  $1.18\times$  to  $1.27\times$  faster end-to-end training compared to state-of-the-art MoE training systems, Tutel-2DH, SmartMoE, and Megatron-LM.

Index Terms—Distributed Deep Learning; Mixture-of-Experts; Expert Parallelism; Token Deduplication; Expert Swap.

#### I. INTRODUCTION

The mixture-of-experts (MoE) architecture with sparse activation has gained significant research interest in large language models (LLMs) [1]-[7]. It provides an effective solution for model size scaling, where the computational requirement grows sub-linearly with increasing number of parameters. The MoE architecture incorporates the MoE layer, which comprises multiple feed-forward networks (FFNs), known as experts, substituting the dense feed-forward layer while activating only a subset of these experts for each input token [2,7]. A trainable routing function, generally a small neural network utilizing a softmax mechanism, is employed to dynamically select which experts should be trained for each input token [2]. This architecture allows the model size to expand to nearly Etimes (where E represents the number of experts per MoE layer) that of a standard dense model, yet the computational demand remains comparatively stable.

\*Corresponding author.

<span id="page-0-0"></span>![](_page_0_Picture_8.jpeg)

(a) Two-level hierarchy

(b) Four-level hierarchy

Fig. 1: Two commonly used hierarchical topologies.

However, training MoE LLMs typically requires expert parallelism (EP) [1.8] to place different experts on different GPUs since a single GPU has limited memory to hold all experts. Due to the dynamic nature of dispatching input tokens to experts that are located in different GPUs, EP introduces significant communications, which are implemented by the AlltoAll collective, easily limiting the scalability of the distributed training system. Recent research [2,7]-[12] suggests that communication overheads of the AlltoAll operation constitute 30-60% of the overall training time in GPU/TPU clusters. Some studies are trying to address the communication problem through 1) algorithmic optimization [13]-[17] by using better routing functions to balance the workload of experts, and 2) system-level optimization by designing more communication-efficient AlltoAll collective algorithms [8,18]— [20] and adaptive task scheduling to overlap communication tasks and computation tasks [8,10]-[12,21]-[31]. Since the process of algorithmic optimization can significantly impact model convergence, this study concentrates on system-level optimization that does not compromise model accuracy.

Specifically, AlltoAll requires each GPU to exchange data with all the other GPUs, so its performance is highly affected by the network topology of GPUs (i.e., the hierarchical connection between GPUs as shown in Fig. 1). That is, a low-bandwidth link may significantly slow down the overall communication performance. For example, in the four-level hierarchical topology as shown in Fig. 1b (Inter-node

through InfiniBand, Intra-node through NVLink, PCIe, and QPI), InfiniBand or QPI would easily limit the communication performance of AlltoAll. In larger-scale clusters, GPU nodes should be connected across switches, which introduces higher levels of the topology [32]–[35]. Existing related optimizations include 1) hierarchical AlltoAll algorithms like two dimensional hierarchical (2DH) AlltoAll in Tutel (Tutel-2DH) [8], PipeA2A in ScheMoE [11], and dedicated kernels for Nvidia Hopper architecture in DeepSeek-V3 [7] to better utilize Intra-node and Inter-node network bandwidths, and 2) expert placement algorithms to balance the communication workloads of different GPUs like SmartMoE [23]. These methods underestimate the impacts of the hierarchical structure of GPU connection and have not explored the full hierarchical structure to optimize AlltoAll communication, thus achieving suboptimal training performance.

To this end, we propose HierMoE to fully utilize the hierarchical structure to optimize token distribution and expert migration among GPUs, achieving minimal AlltoAll communication time in MoE model training. HierMoE incorporates three innovative strategies: 1) conducting theoretical research on the links between hierarchical dimensions and the redundant transfer challenge to design a hierarchical token deduplication AlltoAll algorithm aimed at decreasing data transfer redundancy among varying hierarchical dimensions, 2) designing a hierarchical expert swap mechanism to balance the communication workloads of different GPUs aimed at further improving the AlltoAll communication efficiency, and 3) devising theoretical frameworks that render the token deduplication and expert swap strategy broadly applicable and practical for varying models. We implement our HierMoE atop the widely-used LLM training system Megatron-LM<sup>1</sup>, and conduct extensive experiments on a 32-GPU cluster using representative real-world MoE models, including DeepSeek-V3 [7] and Qwen3-30B-A3B [4]. Experimental results show that HierMoE improves the AlltoAll communication efficiency by  $1.55\times$  to  $3.32\times$  and achieves  $1.18\times$  to  $1.27\times$  faster endto-end training over the state-of-the-art MoE training systems Tutel-2DH [8], SmartMoE [23] and Megatron-LM.

TABLE I: Notations.

<span id="page-1-1"></span>

| Notation       | tion Description                                          |  |  |  |  |
|----------------|-----------------------------------------------------------|--|--|--|--|
| $\overline{M}$ | Embedding dimension of a token                            |  |  |  |  |
| K              | Number of experts selected for each token                 |  |  |  |  |
| G              | Number of workers (or GPUs) in the cluster                |  |  |  |  |
| D              | Number of hierarchical dimensions                         |  |  |  |  |
| U[i]           | Number of experts group when                              |  |  |  |  |
|                | performing Inter-level-i AlltoAll                         |  |  |  |  |
| E              | Total number of experts                                   |  |  |  |  |
| $t_d$          | Time of d-dimensional deduplication hierarchical AlltoAll |  |  |  |  |

