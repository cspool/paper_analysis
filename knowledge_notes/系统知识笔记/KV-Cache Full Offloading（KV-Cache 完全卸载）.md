## KV-Cache Full Offloading（KV-Cache 完全卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV-Cache Full Offloading 是将 LLM 推理过程中生成的所有 Key-Value Cache 全部存储在 host memory（CPU DRAM）而非 GPU 显存中的策略。在自回归解码中，每生成一个新 token，其对应的 K/V 投影需要追加到 KV-cache 中供后续 token 的 self-attention 使用。随着 context length 增长，KV-cache 内存占用线性增长：$KV_{MB} = 2 \times L \times H \times D \times C \times B / 1024^2$（L=层数，H=KV head 数，D=head dim，C=context len，B=bytes per element）。对于长 context 大模型，KV-cache 可轻易超过 GPU 显存。Full offloading 将 KV-cache 全部放在 host memory，每个 attention micro-batch 执行前通过 HtoD copy 将所需 token 的 KV-cache 传输到 GPU，执行后将新的 K/V 通过 DtoH copy 写回。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MoE-GEN 中 KV-cache full offloading 的运转流程：

```
解码阶段，处理 batch b_a 个 token（第 l 层 attention 的 self-attention 阶段）:
1. CPU 端: b_a 个 token 对应的历史 KV-cache 在 host memory 中
2. HtoD Engine: 将 b_a 个 token 所需的历史 KV-cache slice 
   copy 到 GPU KV-cache buffer（异步）
3. GPU 端: 等待 KV-cache copy 完成后，执行 self-attention:
   Q[b_a, D] @ K^T[D, C_history + b_a] → attention scores
   softmax(scores) @ V[C_history + b_a, D] → attention output
4. DtoH Engine: 将新生成的 K/V（b_a 个 token）异步写回 host memory
   追加到该 layer 的 KV-cache 尾部
```

Full vs Partial offloading 的权衡：
- **Partial offloading (vLLM)**：GPU 保留部分 KV-cache（如最近窗口），减少 HtoD copy 量，但 GPU memory 被 KV-cache 占用，降低 batch size 上限。
- **Full offloading (MoE-GEN)**：GPU memory 完全释放给 compute（更大的 batch、更大的 S_Expert buffer），代价是每个 attention micro-batch 都需要 HtoD KV-cache copy。MoE-GEN 的分析表明（图 4），大型 dataset 下 full offloading 可节省最高 20× 的 expert weight fetching 流量，因为更大的 batch size 使 expert weight 只需加载一次即可处理大量 token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 KV-cache offloading 的常见方法：
- **PagedAttention (vLLM)**：将 KV-cache 分块管理，block 可在 GPU/CPU 间 swap，类似 OS 虚拟内存的 page 管理。支持 partial offloading。
- **FlexGen**：将 KV-cache 作为可 offload 的 tensor 之一，通过线性规划决定在 GPU/CPU/disk 三级存储中的放置。
- **MoE-GEN**：KV-cache 在 host memory 中以连续 buffer 组织（按 layer、按 sequence 布局），CPU attention kernel 可直接访问 host memory 中的 KV-cache（无需 copy），GPU attention 通过 staging buffer 按需 copy 所需 slice。
- **适用条件**：host memory 需足够大以容纳完整 KV-cache + 模型参数。对于短 context 或小 dataset（GPU memory 足以容纳 KV-cache），partial offloading 可能更优。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Residency Aware Selection (ERAS) 是 MoE-ERAS 提出的在 MoE 推理时根据 expert 当前物理驻留位置（GPU HBM 还是 CPU DRAM）动态调整 gating 决策的调度策略。传统 Top-K gating 仅基于 router logits 选择 expert，完全无视 expert 的物理位置——若选中的 expert 在 CPU DRAM 中，解码阶段需等待 PCIe 传输完成（CPU read time 比 GPU read time 高数个数量级）。ERAS 在 gating 输出后、Top-K 选择前，根据 residency table（每层每个 expert 的 HBM/CPU 状态）修正 logits 或 probabilities，使路由器倾向选择已驻留 HBM 的 expert。核心权衡：牺牲少量 routing quality（选择"足够好"的 on-chip expert 而非"最优"的 off-chip expert），换取显著减少的 CPU↔GPU expert swap 和解码延迟。

从系统架构角度拆解术语：
ERAS 在 MoE serving 系统中的运转流程（以 Mixtral-8x7B, 3 experts offloaded/layer, batch_size=1 为例）：
1. **Profiling 阶段**：在代表性数据上运行模型，收集每层每个 expert 的激活频率 freq(E_i)（normalized activation frequency）。存储为 expert activation profile。
2. **Residency Table 维护**：serving 开始时，所有 expert 参数在 CPU DRAM。随解码进行，LRU caching 将部分 expert 加载到 HBM。每次 expert swap 后更新 residency table：`residency[layer][expert] ∈ {HBM, CPU}`。
3. **Per-layer Routing（解码每步）**：
   - Gating network 输出 Logits = H_i @ W_exp
   - Residency-Aware Adjustment：
     - Thresholding: Weights = Softmax(Logits); 对 HBM 中的 expert 加 α
     - Biasing: 对 CPU 中的 expert 减 β(1-freq(E_i))，然后 Softmax
   - Top-K selection from adjusted scores
   - 若选中 off-chip expert → 触发 CPU→GPU 传输 → 可能触发 LRU eviction
4. **效果**：α=0.15 时减少 10-13% 解码延迟；offload 越多、α 越大，效果越显著（最大 21.2%）。与 quantization、LRU caching、prefetching 正交，可叠加使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：在 `dvmazur/mixtral-offloading`（开源 MoE offloading 框架）的 gating→TopK 之间插入 residency-aware adjustment 模块。维护一个 `residency_table[layer][expert]` 布尔数组，每次 expert swap 后更新。
- 超参数选择：α=0.05~0.25（thresholding），β=1.0（biasing）。用户根据所需 speedup-quality trade-off 选择——α 越大 speedup 越大但 quality 下降越多。
- 适用场景：资源受限的 MoE 推理（batch_size=1），如边缘设备、消费级 GPU。硬件不对称性（GPU HBM bandwidth >> PCIe bandwidth）越大，ERAS 价值越大。
- 局限性：当前仅实现于 Mixtral-8x7B；inference-time routing 改动可能将 token 导向训练时较少见的 expert，增加幻觉风险。

涉及论文标题：
- MoE-ERAS: Expert Residency Aware Selection

---
