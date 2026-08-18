## Model-Sharded Data Parallelism（MSDP / ZeRO / FSDP，模型分片数据并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSDP = 把数据并行训练中的模型状态（参数、梯度、优化器状态）按 worker 均匀切分（shard），使每 worker 只持有 1/N 状态的内存高效并行策略：ZeRO-1 只切优化器状态、ZeRO-2 再切梯度、ZeRO-3 全切（参数+梯度+优化器状态），PyTorch FSDP 即 ZeRO-3 思路（Web 证据：DeepSpeed ZeRO 文档）。逻辑链：DP 每 GPU 复制整个模型状态（7B 模型需 112GB 梯度+状态，80GB H100 装不下）→ 切分后容量随 worker 数线性放大 → 代价是参数/梯度按需跨 worker 移动，引入额外集合通信（ZeRO-3 每层前向/反向各 1×AllGather + 梯度 1×ReduceScatter）→ 每方向集合流量 3(N-1)S/N。
- 扩展变体：ZeRO-Offload 把优化器卸载到 CPU 内存；ZeRO-Infinity 再把模型状态卸载到 NVMe/SSD（8 机可训 175B）。DeepSeek/Llama 预训练广泛采用（Llama 3 用 128 度 MSDP + 8 度 TP + 16 度 PP）。
- 指标 MFU（Model FLOPS Utilization）= 实际吞吐相对模型理论 FLOPs 峰值的利用率：ZeRO-Infinity 在 8×1-GPU + 100Gbps 下训 175B 仅 15% MFU（集合与 GEMM 干扰），DisDP 达 3.98× 吞吐、276B 模型上 59% MFU。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 每训练迭代一层（ZeRO-3）：前向 AllGather 参数 → GEMM；反向 AllGather 参数 → GEMM → ReduceScatter 梯度 → 各 worker 只对自己分片做 Adam。CUDA 调用序列示例 AG1→GEMM1→AG2→RS1→GEMM2→RS2：GPU 非抢占调度下 GEMM1 占满 SMs 阻塞 AG2 启动，AG2 完成后 GEMM2 才能发射，65% 集合时间不可与 GEMM 重叠、迭代多 41%。
- DisDP 对 MSDP 的重构：worker 只跑前/反向 GEMM；集合改由 SmartNIC push/pull + SmartSwitch 聚合/广播（MiSDP 拓扑，流量 2S 收 + S 发）；优化器搬到单台 PS，worker 本地不再有任何模型状态存储。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 主流实现：DeepSpeed ZeRO 三阶段（ZeRO-Offload/Infinity，0.9.3）、PyTorch FSDP、ColossalAI。使用要点：ZeRO-3 通信密集，需 fast interconnect；小集群（8 机 100Gbps）MFU 低（15%），需要 DisDP 式算网存解耦或 SHARP/PAT 等集合优化。信息缺口：论文未给出 DisDP 在 TP/PP 三维并行下的真实实现，仅 ASTRA-sim 仿真（TP8+PP16+DP 至 256）。

Lit Silicon 补充视角（ISCA'26，FSDP 的 C3 行为与节点级性能波动）：FSDP 是"同构负载"的典型——每 GPU 以相同顺序执行相同维度算子，通信用 AllGather（AG）与 ReduceScatter（RS）集合：前向 AG 收集下一层参数分片并与下一层输入投影 GEMM（QKV）、输出投影 GEMM（attention op）重叠；反向 RS 归约上一层梯度并与 MLP down/up 投影 GEMM 重叠。但同构负载下 8×MI300X 同节点内仍出现强烈 kernel 级性能波动：同一 kernel 的重叠率（overlap ratio）与 kernel 时长跨 GPU 高度相关（Pearson 相关与余弦相似度均强），straggler 的 C3 模式几乎恒定（29.6% 重叠率），leader 动态增长（最高 52.7%，为 straggler 的 1.8×）。FSDP2 相对 FSDP 引入新的分布式张量格式以更好处理张量元数据；fp8 精度用 Transformer Engine kernel（E4M3 前向/E5M2 反向+动态缩放）时须回退 FSDPv1 兼容。论文工作量：Llama 3.1 8B、Mistral 7B v0.1（FSDP/FSDP2、bf16/fp8），MoE 对比用 DeepSeek V3 16B（torchtitan/Primus 8 路专家并行）。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
