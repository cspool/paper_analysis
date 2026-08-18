## C3（Concurrent Computation and Communication，并发计算与通信重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- C3 指把通信 kernel 与计算 kernel 在同一设备上并发执行以隐藏通信延迟的技术，源自 HPC（LogP 模型时代用计算掩盖数据搬移），GPU 系统上表现为同一 GPU 同时调度两个并发 kernel（一个计算、一个通信）。分布式 LLM 训练中广泛用于把 AllGather（AG）、ReduceScatter（RS）、AllReduce（AR）等通信集合与 GEMM 重叠（FSDP 前向 AG 与输入投影 GEMM、反向 RS 与 MLP down/up 投影 GEMM 重叠），端到端平均 1.1×–1.6× 加速。C3 不是免费的：有限 GPU 资源被并发 kernel 瓜分，计算与通信互相干扰（共享计算单元与内存带宽），计算 kernel 运行时平均慢 18.9%、最高 40%（ConCCL 等报告）。
- Lit Silicon 论文（ISCA'26）的核心发现：C3 重叠不是均匀的——同一节点 8×MI300X 上同一 kernel 的重叠率跨 GPU 显著不同：straggler GPU（更热更慢）重叠率恒定最低（29.6%），leader GPU 重叠率动态增长（最高 52.7%，约为 straggler 的 1.8×），且重叠率与 kernel 时长强相关（Pearson 相关与余弦相似度均高）——C3 是节点级性能波动的主要贡献者。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FSDP 一层前向的 C3 调度（每个 GPU 独立执行，虚拟并发）
compute_stream  : GEMM_qkv_in  →  GEMM_attn_op  →  GEMM_mlp_* ...
comm_stream     :      AG(next_layer_params)  ── 与 GEMM_qkv_in 起重叠
# 反向
comm_stream     : RS(prev_layer_grads) ── 与 MLP down/up GEMM 重叠
```
Annotations：两条硬件队列（compute/comm stream）并发运行，GPU 抢占式/流式调度下共享 SM 与内存带宽。重叠率（overlap ratio）= 通信 kernel 与计算 kernel 并发执行的时间占比；lit silicon 用 Chopper 解析 trace 计算每 kernel 每 GPU 的重叠率与起始时间。leader 提前发 AG 但必须等 straggler 的 AG 完成（集合是同步点），等待期间计算流继续推进使 leader 的"重叠"变长、资源竞争加剧反而更慢。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(1) 软件——PyTorch/FSDP 用独立 CUDA stream（或 AMD 等价）把通信 kernel 放到非计算流，训练框架在层间自动插入（torch.distributed 的 async 集合）；RCCL/NCCL 集合本身支持异步；(2) 硬件——DMA 引擎（如 GPU 上的 copy engines）卸载通信减少计算干扰（ConCCL），或专用通信加速器。Lit Silicon 论文用 Chopper（作者自研 GPU 特性分析工具，arXiv:2512.08242）对 PyTorch trace 做 kernel 级分析，量化重叠率与 kernel 时长，用于检测 straggler（重叠率与时长作为 lead value 的输入）。使用场景：任何用集合通信同步的分布式训练/推理；注意 C3 重叠率差异本身就是节点级性能波动的放大器（与热致掉队耦合形成 Lit Silicon 负反馈）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
