## BAT: Efficient Generative Recommender Serving with Bipartite Attention

- 属于Serving调度的实现是什么？实验比较什么？
  提出Bat，一个面向生成式推荐(GR)的高效Serving系统。核心实现包含四个模块：(1) Bipartite Attention算法层（见算法pipeline），自适应选择User-as-prefix或Item-as-prefix attention以最大化KV cache复用；(2) Compute-storage disaggregated KV cache pool：将user-prefix cache和item-prefix cache作为独立组件主动管理，通过KV cache worker池化多机内存，cache meta service集中管理索引和热度；(3) Hot-replicated Cold-sharded (HRCS) item cache placement：利用item访问分布的skewness——hot item（top 10%占90%访问）在所有worker间复制，long-tail item均匀分片，最小化跨节点通信开销并保持cache效率；(4) Hotness-aware prompt scheduling：基于sliding-window频率估计(𝑓_u)，greedy决策每个request的attention模式——当user token数≥item token数且𝑓_u大于cache中最冷user page的频率时选User-as-prefix，否则fallback到Item-as-prefix。实验比较系统QPS、cache hit rate、P99 latency，对比RE(Recomputation)、UP(User-as-prefix)、IP(Item-as-prefix)三种baseline。

- 硬件平台是什么，配置是什么。
  主测试平台：4节点集群（浙江大学），每节点Intel Xeon Silver 4214 CPU (2×24 threads @2.20GHz)、200GB内存、1×A100-40GB GPU (PCIe 3.0x16)、100Gbps网络。每节点部署1个inference worker和1个KV cache worker。生产测试平台：16节点集群，每节点1×NVIDIA H20 GPU、Intel Xeon Platinum 8469C CPU (2 socket×48 cores×2 threads)、500GB host memory、200Gbps网络。10Gbps和100Gbps两种网络带宽设置（用于评估HRCS cache placement）。

- 开源Serving框架是什么。修改了什么。
  基于vLLM + FlashInfer构建。主要修改：(1) inference engine基于vLLM，定制attention module集成Bipartite Attention机制，使用FlashInfer作为高性能backend优化执行；(2) 新增disaggregated KV cache pool架构——KV cache worker管理paged memory（CPU memory），以user/item粒度存储prefix KV cache，物理组织为兼容PagedAttention的fixed-size pages，内置transfer engine支持GPUDirect RDMA；(3) 新增cache meta service——逻辑集中式进程追踪各worker/tier上每个user/item KV entry的索引和热度，接收prompt scheduler的cache read/write/check请求并协调KV cache worker的物理传输；(4) 新增hotness-aware prompt scheduler——调度器定期查询批量user/item的KV cache状态，用sliding-window频率估计决策attention pattern，将tokens拼接成batch分发到inference worker做load-balanced执行；(5) max-batched-tokens limit通过offline profiling确定以保证latency SLA。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明代码开源地址。Bat系统使用流程：
  1. 离线初始化：分配pooled memory给item cache，剩余给user cache；运行HRCS算法(Algorithm 1)——profile网络带宽(B tokens/sec)→estimate prefill computation time via polynomial regression→根据超参数α计算max allowed communication ratio R_max→按item access frequency CDF确定hotspot replication ratio r→top-r% hot item在所有worker间复制，其余均匀分片。
  2. 在线Serving：hotness-aware prompt scheduler接收retrieval stage的ranking requests（含user ID和candidate item IDs）→查询cache meta service获取相关user/item的KV cache状态和hotness→根据decision rule选择prefix type：若𝜏_u≥𝜏_i且𝑓_u>min_{p∈C_u}𝑓_p则选User-as-prefix，否则选Item-as-prefix→concatenate tokens成batch→load-balanced分发到inference workers→触发cache read/write操作。
  3. Bat在Books dataset (Qwen2-1.5B)上相比UP提升throughput最多1.6×；在Industry dataset上相比RE提升2.3×；cache hit rate最高58%；降低total computation最多58%。

