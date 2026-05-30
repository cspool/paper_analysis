# VII. RELATED WORK

MoE training. The superior performance of large-scale MoE models has driven extensive research efforts. MoE training systems such as DeepSpeed [24] and Tutel [52] have already shown remarkable results in optimizing distributed training. NetMoE [74] proposes dynamic expert placement to minimize inter-node communication. However, they do not consider the heterogeneous inter-node networks common in public clouds. DeepSeek-v3 [57] proposes expert replication within a single node. Differently, ScaleMoE provides remapping and replication between nodes.

Load imbalance. Previous research shows that the structured and systematic behavior of the router in assigning consecutive and specific tokens to the same expert often leads to load imbalance [10], which adversely affects training performance. To mitigate load imbalance, strategies include balance-aware auxiliary losses for even token distribution and limiting expert capacity to prevent overloading [16]. However, while these methods effectively address load imbalance, they compromise model quality [10], [54]–[56], [75]. In contrast, ScaleMoE provides the same training quality by preserving the original computation.

Heterogeneous network. In the context of network communication, the GPU communication in modern dense GPU systems utilizes various interconnect technologies such as InfiniBand, NVLink, and NVSwitch [76]. NCCL [77] and other communication libraries [78]–[80] provide efficient data transfer and collective communication operations. They are complementary to ScaleMoE and can enhance optimization, communication, and computation to enable efficiency in largescale distributed systems.

Sparsity/compression techniques. Sparsity/compression techniques minimize the amount of data to be transmitted between nodes, thereby improving system efficiency and reducing latency [81]–[84]. These operate at the tensor level, whereas ScaleMoE focuses on structured communication-level zero padding caused by load imbalance. Note that the concepts of these methods are complementary, and applying both could bring further improvements.

#### VIII. DISCUSSION

ScaleMoE offers a simple, effective way to reduce redundant communication and mitigate load imbalance, while remaining complementary to existing MoE models. As described in Section V, ScaleMoE preserves the original training process without requiring changes to model computation or router design. This orthogonal design ensures broad compatibility, enabling seamless integration with diverse architectures without model modifications. These include various routing mechanisms, shared-expert configurations, and models employing Multi-Head Latent Attention (MLA).

Limitation. Our end-to-end evaluation uses BERT/GPT-MoE for controlled analysis; however, the design is routerand model-agnostic. Based on our analysis of communication costs and load imbalance (Section III), we expect ScaleMoE to remain effective or even potentially more impactful in stateof-the-art MoE models (e.g., Mixtral, Llama-MoE).

Future work. We will evaluate ScaleMoE on state-of-theart MoE models to validate real-world effectiveness. Also, we will conduct more sensitivity analyses on diverse network setups for more comprehensive evaluations. Furthermore, our evaluation will cover its performance in MLA-based settings, under alternative gating mechanisms, and with shared-expert configurations. Finally, we will quantify scaling benefits at larger hidden dimensions.

