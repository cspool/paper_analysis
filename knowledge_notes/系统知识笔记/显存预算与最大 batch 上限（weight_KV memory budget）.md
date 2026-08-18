## 显存预算与最大 batch 上限（weight/KV memory budget）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM serving 的显存账本：GPU 显存 = 模型权重（静态，与 batch 无关）+ KV-cache（随 batch × seqlen 线性增长）+ 激活/临时缓冲。decode 阶段内存受限（memory-bound），最大可行 batch 由"剩余显存能容纳多少请求的 KV-cache"决定；权重越小 → KV 预算越大 → batch 越大 → 吞吐越高。本论文的核心系统杠杆：把权重无损压缩（Qwen-14B 27.5→18.1 GB、Mixtral-176B 261.9→163.7 GB），KV 预算相应从 44.1→56.3 GB（Qwen-14B）/26.3→124.6 GB（Mixtral-176B）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
约束模型：
```
batch_max = floor((GPU_budget - weight_mem - overhead) / kv_per_token(seqlen))
throughput ≈ f(batch_max)          # decode 阶段随 batch 近似线性提升
```
论文实例（Table II）：Qwen-14B 80 GB 预算 seq 1024：batch 47→60（KV 44.1→56.3 GB）、吞吐 1131→1217 tok/s；Mixtral-176B 320 GB 预算 seq 1024：batch 20→95（4.8×）、吞吐 241→391 tok/s（1.6×）——压缩"把推理瓶颈从显存容量转移到计算吞吐"。注意权衡：TPOT 略增（解压开销），以延迟换吞吐；batch 上限的放大器效应在大模型（权重占比大）上更显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：scheduler 显存记账（SGLang/vLLM 的 mem-fraction-static、max-running-requests、KV 分页分配）。传统扩容手段：KV offload/逐出、CPU offload（KTransformer 把放不下的权重放 CPU 经 PCIe 流式加载——带宽瓶颈，本论文对比 prefill 7.7×/decode 18.1× 优势）、多卡 TP/EP（本论文 Mixtral-176B 用 EP 4 卡）。本论文的增量：压缩权重（无损）直接扩大 batch 上限，无需改调度、无需 offload，且与 KV-cache 压缩（未来工作方向）正交。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
