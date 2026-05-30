## LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LSH-MoE框架，在MoE训练过程中使用Locality-Sensitive Hashing（LSH）对all-to-all通信前的token进行在线聚类压缩，仅传输聚类中心（centroid）替代完整token，从而减少通信量。核心包括两部分：(1) 基于cross-polytope hashing的高效LSH聚类算法，将token映射到桶中并计算聚类中心；(2) 基于残差的误差补偿方案（residual-based error compensation），记录每个token与其聚类中心的残差，在expert计算后将残差加回输出，弥补压缩带来的精度损失。
  - 实验比较：对比原始无压缩MoE训练与LSH-MoE（有/无error compensation）的收敛速度和下游任务精度。消融实验比较不同hash函数数量和类型（cross-polytope vs spherical-plane）对压缩率和模型质量的影响。

- 硬件平台是什么，配置是什么。
  - V100 Cluster: 2台服务器，每台8× NVIDIA V100 (32GB)，NVLink 2.0，跨机RDMA NIC 100 Gbps
  - A100 Cluster: 4台服务器，每台8× NVIDIA A100 (40GB)，NVLink 3.0，跨机双RDMA NIC 200 Gbps
  - 软件: Ubuntu 20.04, CUDA 11.3, cuDNN 8.2.0, NCCL 2.12.7, PyTorch 1.11

- 模型是什么。数据集和bench分别是什么。
  - 模型: RoBERTa-MoE (394M total, 16 experts), T5-MoE (~9.3B total, 16 experts), GPT-MoE 15B (16 experts, top-2 gating), GPT-MoE 52B (512 experts), Swin-MoE-L (946M, 32 experts)
  - 数据集: BooksCorpus (~800M words) + English Wikipedia (~2.5B words) for RoBERTa-MoE; 工业数据集 (~500M words) for T5-MoE span-masked LM pretraining; GLUE benchmark for GPT-MoE fine-tuning; ImageNet-1K for Swin-MoE fine-tuning

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文在补充材料中提交了代码，但T5-MoE因公司平台限制无法公开训练代码。论文明确指出方法是框架无关的，可应用于Hetu-MoE、DeepSpeed-MoE、Tutel等框架。
  - 算法pipeline执行流程（基于论文Algorithm 1）：
    1. Gate网络计算token到expert的映射：`ζ = G(X)`，将输入X分派到各expert的token集{X_i}
    2. 对每个expert i的token集X_i执行LSH聚类：
       - `IDX_i = LSH(X_i)` — 使用cross-polytope hashing将每个token映射到桶
       - `LSH(x) = argmax_{i∈{±1,...,±d}} |Rx|_i` — 随机旋转矩阵R将x映射到cross-polytope最近顶点
    3. 计算每个cluster j的聚类中心：`cluster_j = Mean(cluster_j)`
    4. 记录残差：`Δcluster_j = {x - cluster_j | x ∈ cluster_j}`
    5. 仅传输聚类中心C = {cluster_j}通过all-to-all通信（替代完整token）
    6. Expert对中心进行计算：`E(cluster_j)`
    7. 结果通过all-to-all传回
    8. 残差补偿还原输出：`Y_ij = {E(cluster_j) + ΔCluster_jk | k=1,...,N_j}`
  - 默认参数：6个hash函数，cross-polytope hashing，压缩率约20%时精度无损
