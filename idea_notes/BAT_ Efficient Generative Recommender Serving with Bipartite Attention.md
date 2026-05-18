## BAT: Efficient Generative Recommender Serving with Bipartite Attention

- baseline方法是什么？
  现有GR serving系统采用User-as-prefix attention作为事实标准：将输入序列组织为[U, I1,...,I_N, Instr]，仅user profile token U的KV cache可跨同一用户的multi-turn请求复用。全栈执行例子：算法层标准causal self-attention对所有token执行QK^T→Serving框架层vLLM管理paged KV cache，prefix caching以LRU策略在CPU/GPU memory中存储user prefix cache→编译框架层未特殊修改→kernel调度层FlashInfer提供融合attention kernel→硬件架构层A100 GPU处理compute-bound prefill。Baseline三个核心缺陷：(1) **User cache hit rate极低（仅18%）**——大规模用户基（10^8）中大量用户不活跃，item token占总token 33%以上但无法被User-as-prefix共享；(2) **User cache存储不可行**——Qwen2-1.5B模型下单个user平均1000 tokens需约29MB KV cache，缓存10^8用户需超过2.9PB存储，远超单机/集群本地内存容量；(3) **热门item跨用户共享机会未被利用**——trace分析显示约90%的访问集中在top 10%的热门item，但这些item的KV cache在User-as-prefix中依赖前置user context而无法跨用户复用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Bat**，基于关键洞察"user和item语义在推荐prompt中排列不变"设计Bipartite Attention，并配套co-design disaggregated KV cache pool、HRCS item cache placement和hotness-aware prompt scheduling。

  核心设计及其与Baseline缺陷的映射：

  **(1) Bipartite Attention算法（解决user cache共享受限问题）**：利用排列不变性提出Item-as-prefix attention作为User-as-prefix的替代——将输入重组为[I1,...,I_N, U, Instr]，调整attention mask使item间无cross-attention（遵循HSTU原则），调整position encoding使所有item共享相同起始位置ID，使item KV cache独立于user context可跨用户自由共享。相比baseline仅能复用user cache，Bipartite Attention解锁了item cache复用——热门item可跨成千上万用户共享，cache hit rate从18%提升到最高58%。

  **(2) Disaggregated KV Cache Pool（解决user cache存储不可行问题）**：设计compute-storage分离架构——KV cache worker管理paged memory pool（CPU memory + GPU memory），cache meta service集中追踪索引和热度，inference worker专注forward computation。User-prefix cache和item-prefix cache作为独立组件被主动管理而非被动存储。Item cache仅需约287GB（1M items, Qwen2-1.5B）vs user cache的430TB（10M users），使本地内存存储成为可行。

  **(3) HRCS Item Cache Placement（解决item cache引入的额外内存开销和跨节点通信问题）**：利用item access分布的skewness——通过offline profiling网络带宽B和prefill computation time t确定max allowed communication ratio R_max→按item access frequency CDF计算hotspot replication ratio r→top-r% hot item在所有KV cache worker间完整复制（eliminate IO bottleneck for high-frequency access），其余cold item均匀分片（minimize memory usage per machine）。相比fully replicate节省内存（在100Gbps下Bat throughput比Bat-Replicate高16%），相比fully shard减少网络开销（在10Gbps下Bat-Hash throughput仅为Bat-Replicate的78%）。

  **(4) Hotness-aware Prompt Scheduling（解决prefix selection非trivial问题）**：基于sliding-window频率估计𝑓_u（empirically validated用户连续行为具有相似性），greedy policy决策prefix选择——当user token数≥item token数且user的estimated frequency > cache中最冷user page频率时选User-as-prefix（高频用户的user cache值得保留），否则fallback到Item-as-prefix（低频用户利用item cache避免compulsory miss）。相比cache-agnostic policy（naively选token数更多的做prefix），hotness-aware在user cache空间受限时（25GB）throughput和cache hit rate显著更高。

  论文方法全栈执行例子（以Qwen2-1.5B, Books dataset, 4-node A100集群为例）：
  - 算法层：Bipartite Attention根据scheduler决策选择User-as-prefix（输入=[U, I1,...,I_N, Instr], Attn(q_{I,Instr}, k_{I,Instr}∪k_U, v_{I,Instr}∪v_U)）或Item-as-prefix（输入=[I1,...,I_N, U, Instr], Attn(q_{U,Instr}, k_{U,Instr}∪k_I, v_{U,Instr}∪v_I)）。Attention mask屏蔽item间cross-attention，position encoding使所有item共享相同起始位置。
  - Serving框架层：vLLM + FlashInfer + 自定义Bat组件。Hotness-aware prompt scheduler接收retrieval stage的requests（user ID + 100 candidate item IDs）→查询cache meta service获取cache状态和hotness→根据greedy rule决策前缀类型→concatenate prompts成batch→load-balanced分发到inference workers→cache meta service协调KV cache worker执行物理KV cache传输（RDMA/DMA）。
  - 编译框架层：论文未修改编译框架。
  - kernel调度层：FlashInfer提供优化attention kernel，KV cache以PagedAttention兼容的fixed-size pages管理。HRCS通过offline polynomial regression model估计prefill time，online按frequency CDF动态决定replication ratio。GPUDirect RDMA用于跨节点KV cache传输。
  - 硬件架构层：4节点×A100-40GB GPU (PCIe 3.0x16) + 100Gbps网络互联。生产环境16节点×H20 GPU + 200Gbps网络。

  最终效果：Bat在多个数据集和模型上相比UP(User-as-prefix)提升throughput最多1.6×，相比RE(Recomputation)提升最多2.3×；cache hit rate最高达58%；减少total computation最多58%；在200ms P99 latency SLO下相比UP和RE分别sustain 1.47×和1.57×更高request rate。
