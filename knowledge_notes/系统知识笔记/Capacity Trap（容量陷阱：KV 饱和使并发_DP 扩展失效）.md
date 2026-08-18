## Capacity Trap（容量陷阱：KV 饱和使并发/DP 扩展失效）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
本论文提出的推理系统失效模式：在 reasoning（长 CoT）负载下，增加并发或 DP 副本数一开始提升 GPU 占用与吞吐，但当累计 KV cache 足迹饱和 HBM 后，额外请求触发调度器抢占与 prefill 重算，吞吐增益崩塌、尾部延迟失控。根因是"DP 不池化内存"——每卡独立持有完整权重复本，各自面对相同的本地 KV 容量极限，一个副本撞墙即拖累整体。特征是计算利用率远未饱和（GPU 空转）时容量已满，打破"最大 batch 即最大吞吐"的标准启发式。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# 8×H200、DP=8、DeepSeek-8B、batch 从 500→5000（本论文 Fig.4 实测）
BS-500 → BS-5000：
  聚合吞吐 ↑（更多流并行）但 E2E 61s→165s（亚线性）
  每卡 625 请求，KV 占用冲 100% → 调度器抢占/限流（Fig.4d）
  HBM 带宽利用饱和却出现"锯齿"（40%–85% 振荡）——compute-bound prefill 与 memory-bound decode 被迫交错
# 容量构成（32B 例）：141GB HBM − 64GB 权重副本 = 77GB 给 KV（DP）
#   TP=8 后：141GB − 8GB 权重分片 = 133GB 给 KV（TP 释放 ≈1.7× KV 空间）
```
Annotations：Capacity Trap 的两个量化指标是 KV 占用率（容量维度）与 HBM 带宽利用的锯齿振荡（调度 thrashing 维度）；"stranded capacity"指 DP 副本间无法共享空闲内存（GPU0 抢占、GPU1 空闲）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Capacity Trap 是观测结论而非可安装组件，其治理手段即设计原则：(1) KV-aware 并发上限（不盲目拉 max_num_seqs）；(2) 内存池化——TP 分片权重、KV offload、memory-aware routing，提高单请求的有效容量裕度；(3) 在线 batch 调优（TTFT/TPOT/KV 占用/HBM 带宽做反馈）；(4) 对长输出负载在 admission 时预留 decode KV。论文用 8B/14B/32B/70B/405B/671B 六个模型 + 1K–10K 并发 + 500–5000 batch 系统量化该现象，并定位 32B 为 DP 崩溃→TP 接管（Parallelism Transition Point）的拐点。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
