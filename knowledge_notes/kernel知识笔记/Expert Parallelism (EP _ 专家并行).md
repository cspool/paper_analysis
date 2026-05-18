## Expert Parallelism (EP / 专家并行)

术语是什么？
Expert Parallelism (EP) 是MoE模型推理中的分布式并行策略：将不同expert的完整权重分配到不同GPU，而非像Tensor Parallelism (TP)将每个expert权重切分到多个GPU。当gate network选择某expert时，token hidden state被发送到持有该expert的GPU执行计算，然后结果汇总。EP的核心优势是减少TP中每GPU都参与每token计算的通信开销（只有被选中expert所在的GPU接收token），但expert负载不均会导致GPU闲置。FineMoE在6×RTX 3090上使用EP，通过hash map和round-robin将expert分布到GPU，结合expert offloading进一步管理每GPU显存。

从kernel调度角度拆解术语：
```
# Expert Parallelism执行流程（Mixtral-8x7B, 2GPUs, 每GPU 4 experts）：
# GPU0: Experts E0-E3, GPU1: Experts E4-E7

# 对某token在第l层MoE block：
h = hidden_state[l]              # GPU0上的hidden state
gate_logits = h @ W_g            # Gate Network计算（每GPU复制W_g）
probs = Softmax(gate_logits)     # [E0:0.6, ..., E7:0.01]
selected = TopK(probs, 2)        # 假设选到E0(GPU0)和E5(GPU1)

# GPU0计算本地expert：
output_gpu0 = probs[E0] * FFN_E0(h)  # 本地计算，无跨GPU通信

# GPU1计算远程expert：
send(h, GPU0→GPU1)                   # NVLink传输hidden state
output_gpu1 = probs[E5] * FFN_E5(h)  # 远端计算
send(output_gpu1, GPU1→GPU0)         # 传回结果

output = output_gpu0 + output_gpu1   # 汇总
```
EP与Expert Offloading结合时（FineMoE）：GPU cache管理仅限于本地expert，每GPU独立执行prefetch/cache/evict决策。On-demand loading也按GPU本地expert执行，跨GPU的expert miss通过NVLink all-to-all通信处理。GPU侧task pool + 异步线程调度expert prefetching和on-demand loading。

术语一般如何实现？
TensorRT-LLM支持TP+EP混合并行（`moe_tp_size` + `moe_ep_size`）。Megatron-Core推荐Mixtral-8x7B在64GPUs上TP=1, EP=8, PP=4。FineMoE用MoE-Infinity的expert parallelism实现：hash map做expert→GPU映射，round-robin分配确保GPU间expert数均匀。跨GPU传输通过NVLink（RTX 3090 pairwise NVLinks），CPU→GPU传输通过PCIe 4.0（32GB/s）。EP主要挑战是load imbalance和通信开销。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

