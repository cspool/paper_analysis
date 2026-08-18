## NVIDIA H200（Hopper 高容量推理 GPU，141GB HBM3e）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA H200 是 Hopper 架构的推理旗舰 GPU（H100 的 HBM 升级版，SXM5 形态）：141 GB HBM3e 显存、峰值带宽 4.8 TB/s、FP16/BF16 峰值 1979 TFLOPS，配套第四代 NVLink + NVSwitch（每 GPU 双向 900 GB/s）。相比 H100（80GB HBM3、3.35 TB/s），H200 的核心卖点是近 1.8× 显存容量与更高带宽——对大模型权重驻留与长上下文 KV cache 容量尤其关键，成为 2025-2026 年 LLM 推理集群的主流卡（本论文 8×H200 节点即现代推理的"基本扩展单元"）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文把 H200 的硬件资源拆成三类相互竞争 HBM 的"账户"：(1) 权重——32B 模型 FP16 ≈64GB/卡（DP 复制）或 ≈8GB/卡（TP=8 分片）；(2) KV cache——容量决定可并发请求数（Capacity Trap），带宽决定每 token 解码延迟（TPOT）；(3) 激活/中间态。运转示例（8×H200 节点，DeepSeek-8B DP=8 推理）：每卡 141GB 中权重约占 16GB、剩余 ~125GB 分给 KV 块（PagedAttention block size 16）；10K 并发下 KV 占用数分钟冲 100% → 调度器抢占；HBM 带宽在 prefill（≈30% 利用，compute-bound）与 decode（≈85% 饱和，bandwidth-bound）间振荡——同一块 HBM3e 以 4.8TB/s 同时服务两种正交负载。论文结论：单节点 8×H200 的 NVLink 全互联（900GB/s/GPU）使 TP=8 的逐层 all-reduce 可行，TP 只在 NVLink 域内扩展（节点间仍靠 DP 复制）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
H200 已量产部署（SXM5/PCIe 形态，141GB HBM3e），主流 serving 引擎（vLLM/SGLang）原生支持；本论文的用法是 8×H200 单节点实测全部 DP/TP/PP/hybrid 组合。使用要点：H200 的大容量使单卡可装 70B 级模型（FP16 约 140GB 逼近极限），405B/671B frontier 模型必须多卡（TP/PP）；H200 的带宽-容量配比（4.8TB/s / 141GB ≈ 34/s）决定其 decode 每 token 的 KV 读取吞吐上界；与 H100（80GB）相比，H200 每卡多出 61GB 可投给 KV cache，直接提高单卡并发容量——这是"内存容量作为一等设计参数"的硬件前提。相关条目：HBM3e 物理组织见芯片设计层 HBM 条目；NVLink/NVSwitch 见芯片设计层 NVSwitch 条目。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
