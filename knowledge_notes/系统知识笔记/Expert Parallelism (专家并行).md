## Expert Parallelism (专家并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Parallelism (EP) 是 MoE 模型分布式训练和推理的核心并行策略之一。与 Data Parallelism（每个 GPU 持有完整模型副本、处理不同数据）和 Tensor/Pipeline Parallelism（切分单个层的计算）不同，Expert Parallelism 将 MoE 模型的不同 expert 分配到不同的 GPU 上。对于每个输入 token，Gate 网络计算路由结果决定该 token 被派发到哪些 expert → 通过 AlltoAll 通信将 token hidden states 发送到对应 expert 所在的 GPU → 目标 GPU 执行 expert FFN 计算 → 结果通过 AlltoAll 通信返回原始 GPU。EP 的核心特征是不需要所有 GPU 存储所有 expert 参数（节省显存），但引入了跨 GPU 的 AlltoAll 通信开销。

从系统架构角度拆解术语：
Expert Parallelism 在 MoESys 中的训练流程（结合 Data Parallelism）：
1. **Dense 部分**（Attention 层）：使用 Data Parallelism——各 GPU 处理不同 batch 数据，backward 后通过 AllReduce 同步 dense 参数梯度。
2. **Sparse 部分**（MoE FFN 层）：使用 Expert Parallelism——Expert 1..8 分配到 GPU 0，Expert 9..16 分配到 GPU 1，以此类推。
3. **Forward pass**：每个 GPU 的 tokens 通过 Gate 网络选择 top-K experts → AlltoAll 派发 tokens 到目标 expert 的 GPU → 每个 GPU 对收到的 tokens 执行本地 expert FFN → AlltoAll 返回结果。
4. **Backward pass**：同理通过 AlltoAll 交换 expert 参数的梯度。
5. MoESys 在 EP 通信中引入 Hierarchical AlltoAll（利用 NVSwitch + 同 rank NIC 分组），优化了通信效率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- EP 通常与 Data Parallelism 混合使用：dense 参数用 DP（减少通信量，因 dense 参数少），sparse 参数用 EP（减少显存压力，因 sparse 参数多）。
- AlltoAll 是 EP 的核心通信原语，每层 MoE 需 2 次 AlltoAll（forward）和 2 次 AlltoAll（backward），共 4 次。例如 Switch Transformer 的设计。
- EP 的负载均衡问题是关键挑战：某些 expert 被频繁选中（hot expert），某些很少被选中（cold expert），导致 GPU 计算负载不均。解决方案包括 auxiliary loss（GShard）、随机 routing（Switch Transformer）、capacity factor 限制等。
- 在 MoESys 实验中，EP 结合 Hierarchical AlltoAll 在 80.7B model / 4 nodes 32 GPUs 下通信阶段加速 15.5%，端到端训练加速 10.3%。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing

### Shazeer et al. (2017) 的先驱性贡献

Shazeer et al. (2017) 首次提出了混合数据并行与模型并行（Mixed Data and Model Parallelism）的 MoE 分布式训练方案（第三节），这是 Expert Parallelism 的雏形：

- **核心思想**：同一组设备同时充当数据并行副本（处理标准层和 Gate 网络）和模型并行分片（各托管一部分 expert）。标准层在每个设备上全复制（DP），每个 expert 只在集群中保留一份共享副本（MP）。
- **组合 batch**：所有数据并行输入 batch 中的相关样本合并后送给每个 expert → expert batch size 约等于 kb·d/n（b=每设备 batch, d=设备数, n=expert 数），即 batch size 放大 d 倍，解决 shrinking batch problem。
- **卷积加速**：在 language model 中，等前一层所有时间步完成后再将 MoE 应用于所有时间步 → 将 seq_len 折叠进 batch → 进一步放大 expert batch。
- **Hierarchical MoE 与设备的协同设计**：第一级 Gate 的 branching factor = GPU 数量 → 主 Gate 选择 GPU (group) → 次级 Gate 在 GPU 本地选 expert → 次级 expert 间无跨设备通信。
- **与后续 EP 的关系**：该方法奠定了 Expert Parallelism 的基本设计原则——dense 部分用 DP，sparse 部分用 MP/EP。该论文尚未使用 AlltoAll 通信（因 Hierarchical MoE 避免了次级通信），后续 GShard (Lepikhin et al. 2020) 在 flat MoE 中引入 AlltoAll 通信，成为现代 EP 的标准通信原语。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

### Serverless MoE 的补充
该论文在 **serverless（AWS Lambda）平台**上采用 expert parallelism：每个 expert 部署为一个独立的 Lambda function（而非 GPU），通过三种 scatter-gather 通信方法（pipelined indirect via S3 / non-pipelined indirect / direct invocation）替代 GPU 集群上的 all-to-all 通信。Serverless 上的 EP 特点：
- 内存配置差异化：热门 expert 分配大内存（最高 3008MB，更多 vCPU），冷门 expert 分配小内存（最低 128MB），按 GBs 粒度计费。
- 专家副本（replication）：热门 expert 可部署多个 function 副本（最多 8），每个副本处理 1/g 的 token，解决单函数内存上限约束和 payload size 限制。
- 无 all-to-all 通信原语：serverless 平台不支持 GPU 集群的 NCCL all-to-all，因此通过 S3 中继或函数间直接调用实现 scatter-gather。
- 预测驱动：由于函数部署需要数分钟（cold start），必须在使用前预测 expert 负载，无法在推理中动态调整。

### MoETuner 的补充

MoETuner 通过 ILP 优化 Expert Parallelism 中的 expert-to-GPU 映射，解决默认 contiguous block placement 的两个缺陷：(1) Token 处理负载不均衡——高频/低频 expert 混合分配到同一 GPU cluster；(2) 跨 GPU 通信倾斜——利用跨层 expert 路由亲和性将频繁交互的 expert 对放在同一 GPU。ILP 1（Load-Balanced Expert Clustering）按层将 expert 聚类，min Σ|T_{c,l} - T̄_l|；ILP 2（Cluster-to-GPU Assignment）分配 cluster 到 GPU，min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2})。在 Megatron-LM 上修改 all-to-all 通信和 expert placement 模块实现自定义 mapping，Mixtral-8x7B 单节点（8×H100, 4EP-2TP）加速 9.3%，多节点（16×H200, 4EP-4TP）加速 17.5%。

### Orders in Chaos 补充

本论文从两个新角度扩展了 EP 的理解：
- **Wafer-scale single-GPU-like EP**：在 wafer-scale multi-chiplet GPU 上，EP 意味着 expert 分布在 wafer 的不同 die 上（而非不同 GPU 节点）。由于 single-GPU-like programming model 不暴露 die-level 控制，论文通过硬件架构扩展（Global/Local CP + ATU/PDU + Task Allocation Algorithm）在硬件层实现 expert-placement-aware 的任务分配。关键挑战：local vs remote HBM access 延迟差距达 10-15×，但传统 GPU CP 无视 dielocation 做 uniform 任务分配。
- **Prefill-guided EP placement**：利用 prefill 和 decode 阶段 expert selection 的高度相关性（Spearman's ρ ≥ 0.7 for most layers, top-5 prefill experts cover ~60% of top-5 decode experts），在 decode 开始前用 prefill traces 优化 expert placement。提出两种算法：(1) Remap-based——保持 expert 数不变重新分配；(2) Duplication-based——预留额外槽位复制热门 expert。在 Qwen3-235B / EP8 / 8×H100 上分别实现 15.5% 和 12.5% MoE computation 加速。在更大 EP scale 下预期效果更显著（EP8 下默认布局已相对均衡，max/min ratio ≈ 1.3×）。
