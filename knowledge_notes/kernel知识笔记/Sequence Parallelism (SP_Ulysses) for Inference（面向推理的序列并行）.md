## Sequence Parallelism (SP/Ulysses) for Inference（面向推理的序列并行）
**attention 之前（MLP、LN、softmax的不同token之間不存在計算，拆分獨立計算）：
  别让 GPU 只拿一段 token
  而是让 GPU 拿完整 token，但只拿部分 heads

attention 计算时（不同token之間需要一起計算）：
  每张 GPU 独立计算自己负责的 heads 的完整 attention

attention 之后：
  再转回按 sequence 切分，继续跑 MLP / LN / 后续层**

术语是什么？

Sequence Parallelism (SP)，亦称Ulysses SP，最初由DeepSpeed Ulysses（arXiv:2309.14509）提出用于长序列训练。核心机制是沿sequence维度切分输入数据到多个GPU，而不是沿model weight维度（如TP）或请求维度（如DP）。在attention计算前通过all-to-all通信将数据从sequence-parallel layout转换为head-parallel layout，attention完成后再通过all-to-all转回。与TP不同，SP的all-to-all通信量不随sequence length增长（Table 2），在长序列和大batch下提供接近DP的throughput。

Shift Parallelism将SP从训练场景改造为inference可用，补充了三个inference关键特性：(1) GQA支持——处理Q head数与KV head数不匹配时通过fused all-to-all中的KV cache replication完成；(2) 小batch load balancing——通过padding到SP degree倍数避免load imbalance（如batch=9, SP=8时效率仅50%，padding后100%）；(3) 任意(SP, TP)组合——支持mixed parallelism应对大模型（如Llama-17B-16E需TP=2才fit单GPU）。

从kernel调度角度拆解术语：

SP for inference的forward pass（Algorithm 1，以(SP, TP)组合为例）：

```
1: embed[n/SP, d] ← SP.slice(input_embeds[n, d])
2: for i = 1, ..., L do
3:   qkv_heads[n/SP, 3×h/TP] ← embed * layer_i.qkv[d, 3×h/TP]
4:   qkv_heads[n, 3×h/(SP×TP)] ← SP.all_to_all(qkv_heads)  // fused QKV通信
5:   attn_o[n, h/(SP×TP)] ← layer_i.attn(qkv_heads)          // head-parallel attn
6:   attn_o[n/SP, h/TP] ← SP.all_to_all(attn_o)              // 返回sequence layout
7:   embed[n/SP, d] ← attn_o * layer_i.o[h/TP, d]
8:   TP.all_reduce(embed)
9:   act[n/SP, d'/TP] ← embed * layer_i.mlp_up[d, d'/TP]
10:  embed[n/SP, d] ← act * layer_i.mlp_down[d'/TP, d]
11:  TP.all_reduce(embed)
12: end for
13: output_embeds[n, d] ← SP.all_gather(embed[n/SP, d])
```

关键点：(a) Line 3：QKV projection使用TP分片权重，每个GPU只处理`h/TP`个heads；(b) Line 4：SP all-to-all将sequence-partitioned的QKV重分布到head-parallel layout，使每个GPU获得完整sequence但仅部分heads的数据；(c) Line 5：attention在head-parallel layout下执行，无需跨GPU通信；(d) Line 6：第二个all-to-all将结果返回到sequence-parallel layout；(e) Lines 8,11：MLP路径使用TP all-reduce（TP沿weight维度切分，需同步partial results）。

术语一般如何实现？如何使用？

SP for inference在ArcticInference/vLLM中通过`--ulysses-sequence-parallel-size N`启用。SP可单独使用（SP=P）或与TP组合使用（SP×TP=P）。典型配置：8 GPU节点，(SP=4, TP=2)作为base config处理大batch；shift config使用(SP=1, TP=8) full TP处理小batch。SP的通信模式使用NCCL all-to-all collective（可fused为单次调用处理Q/K/V）。GQA扩展通过将QKV projection的head数从`3×h`替换为`h + 2×h_kv`实现（h_kv为KV head数），当h_kv < SP degree时通过all-to-all send/receive buffer复制KV head。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
- TetriServe: Efficiently Serving Mixed DiT Workloads

### DiT场景下的Sequence Parallelism

与LLM SP不同，DiT的SP有显著差异：(a) **目的不同**：DiT模型通常小到可fit单GPU（FLUX.1-dev仅12B/80GB H100），SP仅用于降低延迟而非解决显存容量；(b) **序列内容不同**：SP切分的不是text token序列而是image latent token序列（256×256=256 tokens, 2048×2048=16384 tokens）；(c) **通信开销与分辨率高度相关**：小分辨率（256×256）在SP=8时通信占比超30%导致scaling效率极低，大分辨率（2048×2048）通信占比<10%受益于更多GPU——这是固定SP度在异构workload下失效的kernel级原因；(d) **无KV cache**：DiT是stateless的（每步独立计算全部latent tokens），SP无需维护跨步KV cache一致性，但这也意味着无法像LLM那样通过切换SP度复用KV cache；(e) **两种实现方式**：Ulysses attention使用all-to-all collectives（适合NVLink高带宽），Ring attention使用P2P ring passing（overlap通信与计算）。TetriServe使用xDiT中的Ulysses SP实现，在8×H100 NVLink 4.0和4×A40 PCIe 4.0平台上运行。每步的scaling efficiency通过offline profiling获得，形成cost model lookup table供调度器查询。

