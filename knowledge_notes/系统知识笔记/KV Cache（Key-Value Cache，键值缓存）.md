## KV Cache（Key-Value Cache，键值缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache 缓存每个已生成 token 在每层的 K/V 向量，decode 时复用避免每步重算历史 token 的 K/V（只算新 token 的 QKV 并与其 attention）。大小 ∝ 层数×头数×head 维×序列长度×精度，长上下文下容量与带宽成为推理系统的主要瓶颈：每步 decode 都要读全部 KV（带宽密集），可容纳请求数受容量限制（最大 batch）。别名：KV-cache、KVCache。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CHIME 视角：KV cache 存放在加速器侧（CHIME-PIM 2TB），容量限制最大 batch（DRM 的 Line-Cap：B_hp/B_dp/B_cpu），带宽决定 attention 吞吐（decode attention 各请求 KV 独立、不跨请求共享、不受益 batching）。GPT-175B：GPU/HBM-PIM 部署模型后仅 ~310GB 可用于 KV、HBM-PIM-EXT 同成本预算仅 320GB，CHIME 2TB 使平均 batch 达 GPU 基线的 6.6×（per-batch 延迟仅 2.2×），这是 5.15× 吞吐的主要来源。存储布局：CHIME 按 layer 粒度 interleaved 存放每请求 KV，使各 rankset 传输量相等、负载均衡；PIM 计算要求每个元素完整位于单一 chip（跨 chip 元素需先 re-layout）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PagedAttention 分页管理（vLLM/SGLang）、量化压缩、前缀共享（CachedAttention）、逐出策略；容量扩展靠卸载——CPU 主机内存（FlexGen、NEO、FastDecode）、SSD（InstAttention）、PIM（AttAcc、CHIME）。使用方式：带宽与容量分离决策——容量决定最大并发请求（GPU 利用率），带宽决定 attention 延迟；CHIME 的教训是两维需同时扩展（Liebig's Law），只扩一维收益递减。

Cassandra 补充视角（ISCA'26）：KV cache 在低 batch 解码中与 FFN 权重共同构成带宽瓶颈，且推理 LLM 的长输出使 KV 显著增大（推理模型序列长 2.99–5.20×）。Cassandra 把 KV cache 视作与权重同等的格式变换对象：(1) per-token 幅度剪枝（Mustafar，arXiv:2505.22913）把每个 token 的 K/V 按幅度 top-k 拆成 speculation/verification 两组（Key 有 channel-wise outlier、Value 剪枝等价 output-aware）；(2) 指数压缩——KV 指数 Shannon 熵约 2.7 bits，unary 编码无损压至约 2.85 bits（Cassandra-1）或 MX 共享指数（Cassandra-2）；(3) 4-bit 尾数截断；(4) 每 token 生成后经 Cassandra encoder 在线格式化（top-k 分组 + 指数编码）写回主存、decode 时经 decoder 解压（superblock 内存映射保证连续访存）。结果：显存占用低于 BF16 原始格式，固定内存预算下比 Eagle-3 多生成 1.81× token（RTX 4090，Llama3-8B）。

PLENA 补充视角（ISCA'26）：KV cache 被定位为两条内存墙的直接来源——带宽墙（每步 decode 全量读 KV + 新 K/V 写回）与容量墙（FP16 下 LLAMA-3-70B 128k 上下文单 batch KV ≈39 GB，限制 batch）。PLENA 的三层处置：(1) 量化——KV 用低精度 MX（如 MXINT4），新 K/V append 前先做选择性 Hadamard 旋转抑制 outlier、再量化为 MX 存 HBM；K/V 只被 attention GEMM 消费，读入 Matrix SRAM 后做逆 Hadamard 变换直接走 MX 通路（权重加载则绕过逆变换）；(2) 非对称精度——W/A/KV 独立配置（如 4/4/4），LLAMA-3.3-70B OSWorld-L（90k prefill/8k output、BS=8）KV 足迹从 239.26 GB 降到 59.81 GB（权重 129.46→32.36 GB，峰值带宽 8192→2048 GB/s）；(3) 容量→吞吐转换——同 HBM 容量下 batch 可更大（如 4→16），TPS 提升直接来自 KV 压缩释放的容量（评估协议：batch 取每硬件-负载组合下 HBM 容量可容最大值）。注意与 Cassandra 的异同：都做 KV 指数共享（MX）与在线格式化写回，PLENA 侧重"量化精度×硬件通路"的墙缓解，Cassandra 侧重无损压缩 + 投机解码的组合。

ConServe 补充视角（ISCA'26）：KV cache 在多轮对话下跨 turn 常驻并持续增长——ShareGPT 分析显示新 turn 高达 99% 的 token 来自历史上下文，现代系统只 prefill 新 token、复用历史 KV。ConServe 把瓶颈定位为"虚拟布局"而非容量本身：PagedAttention 的散页布局 + block table 双层翻译使地址翻译（TLB miss/页走查）成为多轮 prefill/decode 的关键路径（FlashInfer-paged 比 native 慢 12–24%，多轮累积至 1.75×）；解法是每个 conversation 一个连续 VA slice（layer-major 分段），所有 turn 的 KV 追加进同一连续范围、base+offset 算术寻址（VA(t,l)=base+seg_off[l]+t×B_layer）。由此把 KV cache 管理问题从"容量/碎片"拓展到"翻译局部性"维度——TTFT −64.1%~−74.4% (vLLM)、离线吞吐 +17.7%~35.1%。

DynoPipe 补充视角（ISCA'26）：KV cache 在边云流水边界迁移中被定位为 tier-1 "critical frontier state"（与 attention 权重并列，保持生成连续性所必需）。差分 KV cache 同步——迁移时只传差分而非全量（最坏 P99 传输 72ms），配合参数重叠缓存（两侧各留重叠参数集，sub-ms GPU-GPU 传输）与带宽感知状态分区（利用下行富余带宽异步流式）。单用户下 KV 随上下文线性增长、整驻单一域；多用户并发下 per-request KV 竞争边缘 GPU 内存，可触发 memory-pressure>90% 向云重配置移动边界，LRU 逐出冷 cache。带宽<1Gbps 时 adaptive recomputation fallback（计算 +15-25%、带宽 -90%）避免全量 KV 传输。迁移从 baseline 的秒级（数十秒 cache 重建）降到毫秒级（P99 85ms）。

Raptor 补充视角（ISCA'26）：KV cache 是 decode 容量与带宽双重瓶颈的直接来源——Llama-3.1-70B 类模型（80 层、GQA 8 KV 头、head dim 128、8-bit）每 token 每用户增长 ~0.16MB，8K/32K/128K 上下文分别达 1.25/5/20GB 每用户每会话，直接限制单卡可并发会话数；KV 读带宽下界 D_KV ≥ S·(2LH_KV·d_head·q)（读历史）+ (2LH_KV·d_head·q)（写新 token），长上下文下多 TB/s 带宽压力（Fig.2，排除权重/激活）。Raptor 的 3D-DRAM 用 stream-blocked 16KB tile 映射 KV：每 token 产生 2×8×128×2B=4KB KV 状态（FP16 例），4K 上下文 16MB 层缓存切成 1024 个 tile 摊到 16 channel；16KB 粒度匹配 paged-attention ≥4KB page，分配/逐出在 page 边界、不破坏 row-buffer 局部性；decode 每步把新 KV append 到 tile 尾部（带 stream-flipping metadata）。以 KV 为中心的 3D-DRAM 共设计（100TB/s/32GB/卡）使 Llama-70B 单卡 TP=1 部署，对比 SRAM 需 8 卡/TP=8（collective 大、网络敏感）与 HBM 单卡但带宽 <20TB/s——对 KV 流量带宽敏感的应用（decode、长上下文、MoE 大模型）尤为关键。

HybridSpec 补充视角（ISCA'26，KV cache 决定并发上限 + 异构放置）：KV cache 直接界定并发——Llama2-13B 服务 50 个并发 512-token 请求时 KV 需约 20GB（FP16），内存耗尽后新请求必须 stall。HybridSpec 的放置：target 的 KV cache 全部放 XPU 的 512GB LPDDR5X（容量支撑高并发、长序列），HB 栈只存 draft 模型的小 KV cache（draft 体积 <1/10 target，KV 亦小）——这让 HybridSpec 在高请求率/长上下文下不出现容量 stall（图 16），而 HB-ATTEN 把 attention 的 KV 放 HB DRAM（≤20GB/单层）导致容量耗尽、TTFT 剧增，HB-ATTEN-QT 用 4-bit KV 量化缓解。水印机制：内存利用率超阈值时挂起 prefill 限制 KV 增长（图 9）。

从系统架构角度拆解：容量即并发度——KV 占满则新请求挂起，因此"大容量内存放 target KV"是把容量约束与计算分布解耦的关键一步；XPU 内存替换实验（Fig.19：HBM2 40GB vs HBM2e 80GB vs LPDDR 512GB）实证请求率升高后容量成为主瓶颈、大容量 LPDDR 反超高带宽 HBM。

实现与使用：在线 serving 中 KV 管理仍是容量/带宽双瓶颈（PagedAttention 分页、量化、前缀共享等既有手段）；HybridSpec 的增量视角是"KV 按模型角色异构放置"——带宽敏感的 draft KV 放 HB 高带宽侧、容量敏感的目标 KV 放大容量 LPDDR 侧。

Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载的 KV 缩放与 Capacity Trap）：推理模型（OSL≫10k）使 KV cache 按输出序列长度线性增长且持久驻留（prefill 的 KV 瞬时、decode 的 KV 持久），单请求 10k reasoning token 即可占满小模型单卡 HBM。量化：Llama-405B（126 层 GQA）FP16 ≈1.05 MB/token，20M token 聚合输出需 >2 TB（8B 模型）；128 请求 × 10k token 的 batch 仅 KV 超 1.3 TB。系统层后果：DP 下各 replica 独立持权重复本（32B 权重 64GB/卡占掉 141GB HBM 近半），剩余 KV 空间不足→KV 占用冲 100%→调度器 preempt（Reasoning Cliff）；TP 分片权重（32B@TP=8 权重仅 8GB/卡，释放 ≈133GB 给 KV）是容量释放的主要机制。论文结论：KV 容量与带宽是推理的双重一阶约束，容量决定并发上限（Capacity Trap）、带宽决定每 token 延迟（TPOT）。
涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- Rearchitecting the Datacenter Lifecycle for AI
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

IroKnight 补充视角（ISCA'26，KV cache 的 Fully-State Encrypted 缓存）：IroKnight 论证 KV caching 与"全状态加密"完全兼容：自注意力首 MatMul 的 K^T 缓存与末 MatMul 的 V 缓存都以加密形式存储（片上与片外均密文），MatMul 的 tiled 执行规则性不变、pad 预计算照常成立，因此缓存 K^T/V 元素仍可在脉动阵列 ALU 内做 line-rate 瞬时解密；attention 输出加密中间态在向量引擎 SIMD 执行缩放与 softmax 也无需特殊处理。KV cache 的规模同时决定能量开销走势——增加输出 token 数时，首个 token 处理完整输入后、后续每个 token 主要复用 KV cache 而非重算整段上下文，于是 HBM 参数流量占比上升、加密/认证能量被摊销（Llama4-Scout 加密能量开销从 256 输出 token 的 24% 降到 4096 的 10%；认证 29%→15%）；而增加输入 token 时全输入上下文必须先处理完才出首个输出 token，能量开销上升（GPT-OSS-120B 加密 7%→21%）。结论：KV cache（容量/带宽双瓶颈对象）在 IroKnight 中额外承担"加密所有权边界"角色——它让历史上下文在片外也保持密文，且不改变 serving 的缓存管理语义。

MERIDIAN 补充视角（ISCA'26，文档 KV 预计算缓存 vs 运行时 KV cache 的去中心化驻留）：MERIDIAN 处理的文档 KV 预计算缓存（document KV store）与 serving 层 KV cache 不同——它是检索语料侧静态可复用的 KV（同一文档被多请求检索），规模可达 TB 级（500K 文档约 14 TB），远超设备显存（H100 80 GB），因此集中式系统必须驻留 host DRAM、query 时经 PCIe 搬上设备（通信占推理时间最高 93.40%）。MERIDIAN 把文档 KV 按 attention head 分片写入 CXL Type-3 PIM 设备（标准 CXL.mem load/store），"缓存驻留点=计算点"——每设备就地算文档注意力、只交换紧凑统计量；query/生成 token 的运行时 KV cache 由 CEC 主设备维护。文档更新/语料扩展走标准 KV 预计算流程写对应 shard，无需全局重排/重建索引。32 设备 16 TB 容量直接匹配 KV-precomputed RAG 的容量需求（对比 CHIME 2TB DIMM-PIM 面向运行时 KV cache 的容量扩展）。

Tetris 补充视角（ISCA'26，CDSP 下 KV cache 的分布与汇聚）：CDSP 把每请求的 KV cache 分散到多个 prefill 实例（各 chunk 的 KV 分布在对应实例组，ring attention 布局），这与非 SP 解耦集群"整请求 KV 在单实例"不同——带来两个新问题：(1) 跨 chunk 负载均衡：计算新 chunk 前需把前序 chunk 的 KV 均匀重分布到当前实例组（cache balancing，复用 zigzag 布局只前传后半段、与下一层 prefill 跨层重叠，开销 ≤1.8%）；(2) 汇聚传输：decoding 实例须收齐所有实例的全部 chunk 后才开始 decode，用 handshake 式 backend 分配仲裁有限传输 backend 防 starvation（详见本库 CDSP 缓存传输条目）。KV cache 容量还直接决定 SP 决策——实例排队延迟与可用 KV slot 共同构成 CDSP 调度的资源视图；decoding 调度器把正在传输 KV 的请求槽视为虚拟占用（Llumnix virtual usage）以精确路由。
