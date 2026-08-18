## 边云协同推理与动态流水边界（Edge-Cloud Collaborative Inference & Dynamic Pipeline Boundary）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
边云协同推理指把 LLM 推理计算按隐私/算力/带宽三重约束在边缘设备与云端集群之间划分：边缘执行隐私敏感层（embedding 及早期层），云端执行算力密集层，二者经 WAN 传输中间激活（hidden state）。动态流水边界（dynamic pipeline boundary）是 DynoPipe（ISCA'26）的核心——split point 不离线固定，而是随实时资源条件（网络带宽/边缘负载/内存压力）在数分钟内动态移动。动机（论文 §2）：Cloud-only 在单 GPU 串行下排队爆炸（QPS=5 时排队占 62% 总延迟）、Edge-only 受内存墙限制（>8B 模型无法部署、>8 QPS 内存耗尽）、静态边云划分在资源波动下平均劣化 36–82%。论文把边云协同推理的收益定位为"pipeline 并行的吞吐倍增效应"而非资源扩充——SP=12 把容量从单 GPU ~5.6 rps 提到 ~8.9 rps。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DynoPipe 面对的三维不对称：计算 0.48×（H100 989 TFLOPS vs 边缘 Thor 2070 TOPS）、内存 65× 带宽（edge DRAM 51.2 GB/s vs cloud HBM 3.35 TB/s）、通信 80× 带宽 / 10,000× 延迟（10 Gbps/15ms vs 800 Gbps RDMA）。同构 pipeline 假设失效，边界 stage 须显式建模 T_boundary（激活序列化/反序列化 + 跨域传输 + 状态同步 + 数据格式转换，Eq.1 的 I_boundary(l_i)·T_boundary(l_i)）。运转流程（一个 LLaMA2-7B 请求、SP=12、QPS=5）：请求到达边缘 → RTX 3090 执行 embedding+layer 0-12（隐私部分留本地）→ 边界 stage 序列化 resolved hidden state → 经 10 Gbps uplink 传云端（优先 attention sparsity/quantization 压缩点）→ 云端 A40 执行 layer 13-32 → 返回 token；边缘（96ms）与云端（112ms）流水并行，排队占比从 cloud-only 的 62%（295/478ms）降到 26%（76/292ms）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：按 transformer block 边界切分（残差在块内解析、跨域只传 fully-resolved hidden state、无需额外 buffering 或重算）；配合 activation-minimal 边界（选压缩点）降低传输量；隐私敏感 embedding 留边缘。使用场景：边缘 LLM serving、隐私敏感行业（医疗/金融）、边云协同推理系统族（Neurosurgeon、IOON、FlexNN、EdgeShard、CE-CoLLM）。DynoPipe 结果：LLaMA2-7B 吞吐较 edge-only 10.1×、较 cloud-only 1.6×；TTFT 降 98.9%（68.53s→0.74s）；P99 延迟较 CloudOnly/FlexNN/EdgeShard 降 54%/60%/16%（MAF trace）。

涉及论文标题：
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
