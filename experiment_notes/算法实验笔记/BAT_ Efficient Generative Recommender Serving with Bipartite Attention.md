## BAT: Efficient Generative Recommender Serving with Bipartite Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Bipartite Attention，一种用于生成式推荐(GR)的新注意力机制。核心设计：(1) 利用user和item语义在推荐prompt中是排列不变(permutation-invariant)的关键洞察——交换user和item token的顺序不影响上下文语义；(2) 基于此提出Item-as-prefix attention作为传统User-as-prefix attention的替代方案——输入序列组织为[I1,...,I_N, U, Instr]，item的KV cache可跨用户共享；(3) 调整attention mask使item间无cross-attention，并调整position encoding使所有item共享相同起始位置ID（在User-as-prefix中起始于user token长度，在Item-as-prefix中重置为0），保证每个item的token独立于其他item和后续user/instruction token，从而item KV cache可独立预计算和存储。实验比较：在三个Amazon数据集(Games, Beauty, Books)和一个工业合成数据集(Industry)上对比UP(User-as-prefix)和IP(Item-as-prefix)两种attention策略的Recall@k、MRR@k、NDCG@k指标（k∈[5,10]），结果表明IP在多数情况下保持与UP相当或更好的推荐质量。

- 硬件平台是什么，配置是什么。
  主测试平台：4节点集群（浙江大学），每节点Intel Xeon Silver 4214 CPU (2×24 threads @2.20GHz)、200GB内存、1×A100-40GB GPU (PCIe 3.0x16)、100Gbps网络。生产测试平台：16节点集群，每节点1×NVIDIA H20 GPU、Intel Xeon Platinum 8469C CPU (2 socket×48 cores×2 threads)、500GB host memory、200Gbps网络。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2-1.5B (L=28, H=2, D=128, KV cache/token=28672 Bytes)、Qwen2-7B (L=28, H=4, D=128, KV cache/token=57344 Bytes)、Llama3-1B (L=16, H=8, D=64, KV cache/token=32768 Bytes)，均使用FP16。数据集：Amazon公开推荐数据集Games (15K users, 8K items)、Beauty (22K users, 12K items)、Books (510K users, 280K items)，以及从真实电商广告workload生成的工业合成数据集Industry (10M users, 1M items)。评价指标：Recall@5/10、MRR@5/10、NDCG@5/10；系统指标：QPS、cache hit rate、P99 latency。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文未明确说明代码开源地址。Bipartite Attention算法pipeline：
  1. 离线阶段：预计算所有item的KV cache并加载到各KV cache worker的memory pool中。
  2. 在线推理：
     - User-as-prefix attention：输入=[U, I1,...,I_N, Instr]，user token U的KV cache预计算并缓存，实时仅计算item和instruction token。Attention: Attn(q_{I,Instr}, k_{I,Instr}∪k_U, v_{I,Instr}∪v_U)
     - Item-as-prefix attention：输入=[I1,...,I_N, U, Instr]，item token I的KV cache预计算并缓存，实时仅计算user和instruction token。Attention: Attn(q_{U,Instr}, k_{U,Instr}∪k_I, v_{U,Instr}∪v_I)
  3. 判别token为序列最后一个token，其hidden state投影到vocabulary logits，通过softmax计算每个candidate item的relevance score，输出top-k ranked list。
  4. Item-as-prefix三大优势：(a) item cache可跨成千上万用户共享；(b) 仅需本地内存存储所有item KV cache（如287GB for 1M items vs 430TB for 10M users）；(c) 对不活跃用户节省更多computation（>55%用户每小时仅访问一次，user cache面临compulsory miss）。
