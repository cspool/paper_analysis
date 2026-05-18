## FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters

- baseline方法是什么？
  Baseline涵盖两类系统：(a) serverless-based LLM serving：ServerlessLLM使用DeepSpeed parallelism做分布式推理，侧重低延迟serverless部署和快速资源provisioning；Tetris通过tensor sharing实现memory-efficient hosting，无专门pipeline parallelism。(b) offline-optimized pipeline系统：AlpaServe基于历史请求模式做pipeline architecture优化，使用statistical multiplexing with model parallelism；MuxServe使用spatial-temporal multiplexing做多租户serving。这些系统的共同特征是pipeline拓扑在部署前由离线优化决定，服务过程中基本保持不变——主要靠调整batch size或replica数应对负载变化，pipeline stage数和切分方式不会在运行时改变。

  Baseline全栈执行例子（以AlpaServe + OPT-66B + 4-stage pipeline为例）：
  - 算法层：OPT-66B标准Transformer decoder。AlpaServe离线分析历史请求分布，用整数规划确定pipeline并行度和stage placement。4-stage配置下per-stage compute约69.94ms，inter-stage communication约6.3ms，max batch 128。无运行时stage粒度调整能力。
  - 系统框架/Serving层：AlpaServe在OSDI 2023提出的statistical multiplexing框架，pipeline配置在模型加载时固定。收到请求后按FCFS或SLO-aware调度分发到固定stage sequence。当请求burst到来时，只能通过增加整个pipeline的data-parallel replicas扩容（pipeline-level scaling），每个新replica需要完整加载全部stage参数。
  - 编译框架层：论文未明确说明。Alpa依赖ML编译器（如XLA、TVM）做子图编译，但不涉及运行时pipeline重构。
  - kernel调度层：使用标准CUDA kernel + NCCL all-reduce/point-to-point通信。无定制kernel。
  - 硬件架构层：GPU集群。Baseline假设tensor parallelism依赖的NVLink/InfiniBand高速互联可在同机获得多卡；但在碎片化serverless环境中，同机4卡可用概率仅0.02%，78%的tensor parallelism请求被迫退化为pipeline parallelism。

  Baseline的根本缺陷：
  (1) 静态pipeline无法适应请求分布CV剧烈变化：请求CV在180s/3h/12h窗口间波动可达7×。4-stage pipeline在CV=1时性能良好，但当CV升至4时pipeline stall cycle ratio指数增长22×，goodput下降37%。静态pipeline只能为一种CV值优化。
  (2) 碎片化GPU资源无法被静态pipeline有效利用：GPU平均订阅率216%，仅8.7%概率获得单张>85%空闲显存的GPU，0.02%概率同机获得4卡。Tensor parallelism依赖的同机高速互联几乎不可用，而粗粒度pipeline在碎片化GPU上stage放置困难。
  (3) 弹性扩容冷启动慢且资源预留浪费：生产集群维持历史峰值75%的GPU为always-on，但平均SM utilization仅17%。弹性扩容需从持久存储重载参数，多秒级延迟违反交互式LLM应用的亚秒SLO。
  (4) 多模型multiplexing时bursty workload的CV²型干扰未被建模，静态sharing策略在资源波动时性能恶化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FlexPipe提出Inflight Pipeline Refactoring：将pipeline stage粒度从部署前的一次性决策转变为运行时连续控制变量。系统由三段协同设计构成：

  (1) Fine-Grained Model Partitioning：离线对LLM计算图做operator-level profiling（computation time, parameter size, activation size），用带约束的动态规划在GPU显存上限、inter-stage bandwidth、target computation-communication overlap cycle约束下求解最优切分，同时保留Transformer attention block层级结构边界以支持未来stage合并。多粒度候选granularity set覆盖从粗（4-stage）到细（32-stage）的pipeline配置。

  (2) Inflight Pipeline Refactoring：运行时持续监控CV、队列长度、吞吐/延迟profile。通过多目标优化公式 g*=argmax[α·T_k/T_max+(1-α)·L_min/L_k·exp(-|ν_t-ν_k|/σ)] 选择最优granularity。CV升高时执行Pipeline Expansion：申请新GPU、加载split stage、异步KV cache迁移（token-level validity mask C(t)=∪KV_i(t)⊗M_valid，优先RDMA、fallback sendfile）、更新gateway routing。CV回落时执行Stage Consolidation：layer-wise merge state、重建pipeline、释放多余instance。决策延迟<5ms。

  (3) Adaptive Pipeline Scaling with Topology-Aware Allocation：Hierarchical Resource Graph在server/rack/cluster三级协调资源。Memory-aware elastic scaling将参数保留在host memory，affinity-based scheduling优先选择最近承载过该模型的服务器，将cold start转warm start。GPU分配建模为约束优化：max throughput efficiency − γ(CV_i)·multiplexing penalty（penalty ∝ CV²）。

  对应解决Baseline缺陷的具体设计：
  - 对缺陷(1)：FlexPipe把pipeline深度变成CV的函数。高CV时自动切换到细粒度pipeline（如16-stage），单stage compute从69.94ms降至18.67ms，max batch从128扩至512，利用distributed buffering吸收burst；低CV时合并回粗粒度降低inter-stage communication。CV=4时16-stage pipeline的latency仅为4-stage的1/3。
  - 对缺陷(2)：FlexPipe放弃依赖同机高速互联的tensor parallelism，全面采用pipeline parallelism（point-to-point O(1) vs all-reduce O(n²)）。Fine-grained stage使单stage显存需求大幅降低（32-stage下参数加载仅5.43s vs 4-stage的47.14s），更容易在碎片化GPU中找到placement。
  - 对缺陷(3)：Host memory参数缓存+affinity scheduling将扩容初始化延迟平均降72%。Always-on GPU从峰值75%降至30%（降70%），剩余按需弹性分配，资源分配等待时间降85%。
  - 对缺陷(4)：Multiplexing penalty γ(CV_i)=γ_0·(1+α·CV_i²)建模bursty workload的并发资源尖峰。高CV时保持stage隔离避免干扰，低CV时允许GPU共享。实现43% utilization下维持12,000 req/s（vs static系统85% utilization仅1,543 req/s），资源效率最高8.5× better。

  论文方法全栈执行例子（以FlexPipe + OPT-66B在82-GPU Kubernetes集群为例）：
  - 算法层：OPT-66B标准Transformer，无算法修改。控制面新增CV-based pipeline granularity selection（公式4）、stage-level data parallelism M(g_k)计算（公式5）、multiplexing penalty modeling（公式9）、SLO约束scaling granularity validation（公式12）。
  - 系统框架/Serving层：FlexPipe自研~7K LoC serving runtime（非基于已有框架）。三组件协同——API Manager + Resource Monitor → Performance Controller（granularity adaptation + pipeline refactoring）→ Scheduler（HRG topology-aware allocation + memory-aware warm start）。核心机制：fine-grained stage切分→CV驱动的granularity selection→Pipeline Expansion/Stage Consolidation→KV cache一致性迁移（C(t)=∪KV_i(t)⊗M_valid）→gateway routing update。数据传输用RDMA/sendfile替代NCCL。
  - 编译框架层：论文未明确说明。离线partitioning阶段做operator-level profiling和计算图分析，但不涉及编译框架修改。
  - kernel调度层：无定制kernel。使用标准CUDA kernel。KV cache迁移使用RDMA或sendfile系统调用做kernel-space zero-copy传输。
  - 硬件架构层：Kubernetes集群42台服务器82张NVIDIA A100 GPU，100Gbps网络，部分支持RDMA。无定制硬件。
