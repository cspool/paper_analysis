## LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

- baseline方法是什么？
  - Baseline是标准的expert parallelism MoE训练流程。在全栈的执行例子：
    - 算法层：每个MoE层，gate网络对每个token计算top-K expert选择（如top-2 gating），选中的token通过all-to-all通信发送到对应expert GPU，expert计算FFN输出，再通过all-to-all通信将结果传回原GPU。
    - 系统框架层：使用PyTorch + NCCL实现all-to-all通信，DeepSpeed-MoE/Tutel等框架管理expert分布和通信调度。
    - 编译框架层：论文未明确说明。
    - kernel调度层：论文未明确说明。使用标准NCCL all-to-all collective communication。
    - 硬件架构层：在A100/V100 GPU集群上运行，跨机通过RDMA NIC互联。All-to-all通信占训练总时间平均45%（GPT-MoE约30%，RoBERTa-MoE约40%，Swin-MoE约70%），且无法与计算重叠，成为训练瓶颈。
  - Baseline的缺陷：all-to-all通信量过大，因为每个token的完整hidden state（h维向量）都需要传输。随着模型规模（expert数、层数）和GPU数量增加，通信/计算比基本不变，通信瓶颈持续存在。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：LSH-MoE在all-to-all通信前插入LSH聚类压缩步骤，仅传输聚类中心而非全部token，在接收端通过残差补偿还原近似结果。
  - 全栈执行例子（对比baseline）：
    - 算法层：**核心创新**。在gate网络完成token-to-expert映射后、all-to-all通信前，对每个expert的token集合执行cross-polytope LSH聚类。具体地：将每个token x通过随机旋转矩阵R映射到cross-polytope顶点（`argmax |Rx|_i`），相同bucket的token归为一类，计算聚类中心`cluster_mean`作为传输单元。传输量从n×h降至m×h（m为cluster数，m<<n）。接收端expert对中心计算后，通过残差补偿`E(cluster) + Δx`还原每个token的近似输出。这直接减少了all-to-all通信量，压缩率可达11.7%-20%。
    - 系统框架层：方法框架无关，可插入PyTorch+NCCL的标准MoE训练pipeline。论文在PyTorch 1.11上实现LSH聚类模块和残差管理，通信层仍使用NCCL all-to-all，但传输数据量大幅减少。
    - 编译框架层：论文未明确说明。
    - kernel调度层：论文未明确说明。LSH聚类本身是GPU上的矩阵运算（旋转+argmax+mean），使用PyTorch原生算子。
    - 硬件架构层：与baseline相同硬件平台，但通信量减少直接转化为1.28×-2.2×端到端加速。Scalability分析表明加速比在更大模型和更多GPU下依然保持，因为通信/计算比保持恒定。
  - 设计思路总结：观察到MoE训练中all-to-all通信的token存在高相似度（PCA可视化呈现聚类现象，源于Zipf分布数据和Transformer attention的同质化效应），利用LSH将相似token在通信前聚类、仅传中心、传后残差补偿，以可控的精度损失换取大幅通信量减少。
