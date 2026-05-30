## Speculative Expert Loading (Expert Prefetching in MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Expert Loading 是一种 MoE 推理的通信-计算重叠技术。传统 offloading 中，MoE 层必须等待 gate 计算完成后才知道需要哪些 expert——这意味着 expert 加载必须串行在 gate 之后，无法像 dense 模型那样预先加载下一层。论文发现可以利用 Transformer 残差连接的归纳偏置来预测下一层 expert：当前层的 hidden states 是下一层 hidden states 的合理近似（因残差连接逐层累加而非重算），因此将**下一层 MoE gate 函数应用于当前层 hidden states**可得到下一层 expert 选择的近似估计。系统在当前层计算期间异步预取预测的 expert，若预测正确则消除下一层加载延迟；若错误则仅浪费带宽不影响正确性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Speculative Expert Loading per token per layer
# 在处理 layer l 时预测 layer l+1 需要的 expert

# 当前层 hidden states (pre-MoE): h_l
# 当前层 gate (正常):
gate_l = softmax(W_gate[l] @ h_l)
top2_l = topk(gate_l, k=2)

# 投机预测下一层 gate (利用残差归纳偏置):
pred_gate_l1 = softmax(W_gate[l+1] @ h_l)    # W_gate[l+1] 应用到 h_l
pred_top1_l1 = argmax(pred_gate_l1)           # 最可能的 expert
pred_top2_l1 = argmax_second(pred_gate_l1)    # 次可能的 expert

# 异步预取 (独立 CUDA stream, 与当前层 expert 计算重叠):
async_stream.load_expert(l+1, pred_top1_l1)   # 后台 host→device copy
async_stream.load_expert(l+1, pred_top2_l1)   # 后台 host→device copy

# 继续当前层 expert 计算 (在 compute stream):
output_l = expert_compute(top2_l, h_l)

# 进入 layer l+1 时:
# - 若 pred_topk 正确: 即时命中，跳过加载延迟
# - 若 pred_topk 不正确: 重新加载正确 expert (仅浪费带宽)
```

论文评测了 1 层、2 层和 10 层 ahead 的预测 recall（图 2 right panel）。结果：1 层 ahead 时 recall 较高（因残差连接的归纳偏置最准确），2 层和 10 层 ahead 的 recall 显著下降——隐藏状态距离增加导致预测退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 投机预取在当前层所有 expert 加载完成后立即触发
- 预取 1-2 个最可能 expert，使用独立 CUDA stream 异步执行
- 预取的 expert 不替换当前层的 LRU cache，而是暂存于共享 device buffer
- 若预测正确，该 expert 后续替换目标层 LRU cache 中最久未使用的 expert
- 与 LRU cache 正交互补：LRU 减少平均加载时间，投机预取尝试消除剩余加载延迟
- **FloE 的扩展——双预测器架构**（Section 3.3）：FloE 在投机预取基础上引入两个专门的预测器：
  1. **Inter-Expert Sparsity Predictor（学习型）**：用一个小型 MLP（32K~2M 参数，随层深自适应）预测下一层激活的 expert 索引。输入为当前层 hidden state + 历史 expert 选择轨迹，平均 precision 0.88。该 MLP predictor 相比简单 gate reuse 的优势在于可以利用跨层的历史轨迹信息。
  2. **Intra-Expert Sparsity Predictor（复用型，参数免费）**：用当前层 hidden state 与下一层复用的 W_up 矩阵直接做矩阵乘法，近似估计 up projection 输出激活，预计算下一层的稀疏掩码。平均 recall 0.95，零额外内存开销（对比学习型方法如 PowerInfer 需 9GB 额外参数）。
  3. **关键洞察**：相邻 MoE 层的 hidden state 相似度 >0.95（FloE Figure 4），使得当前层 hidden state 可以准确预测下一层的 expert 选择和稀疏分布。双预测器配合 prefetching 实现了 DRAM→VRAM 传输与 GPU 计算的流水线重叠。

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

HOBBIT 的 Adaptive Expert Prefetching 扩展：
- **预测原理**：利用 MoE 层间 gating input 的高余弦相似度（因 Transformer 残差连接，相邻层 hidden state 高度相似）。Mixtral-8x7B 上相邻层 top-1 expert 预测准确率平均 96%，跳 2-3 层仍约 90%。
- **混合精度预取**：关键创新——即使预测错误，低精度 expert 的错误加载惩罚仅为高精度的 1/4（INT4 vs FP16），使预取在任何精度下都产生正向收益。对比：纯 FP16 预取在 Phi-MoE 上可能因误预测导致性能退化（<1.0× speedup）。
- **Stacking Computer**：将所有后续层的 gating 权重矩阵堆叠成 [N_layers_ahead, d_model, num_experts] 张量，与 hidden state 做一次矩阵乘，利用 GPU 并行性实现与单层 gating 几乎相同的计算速度。
- **预测深度**：从当前层开始逐层预测，若所有预测 expert 已在 cache 中则继续预测下一层，直到遇到 cache miss 或达到最大预测层数（建议 1-3 层）。预取的 expert 被 mask 保护不被 evict。
- 效果：prefill 阶段 latency 降低约 10%（因 prefill 激活所有 expert），decode 阶段约 5% speedup。

HarMoEny 的 Asynchronous Expert Prefetching（Section 4.3）采用了不同于基于预测的方法：
- **触发方式**：非预测驱动，而是 **rebalancing-driven**——token rebalancing 可能将 token 分配到当前不持有对应 expert 的 GPU，此时通过独立 CUDA stream 从 system memory 异步预取所需 expert 权重。
- **Overwrite-based loading**：直接覆写已完成 expert 的内存位置（无需先写回 system memory）。关键洞察——expert 权重仅需加载（推理中不变），overwrite 比 "write-back + load" 快 5.5×（11ms→2ms on V100）。
- **与计算重叠**：预取发生在当前 expert 计算期间（独立 CUDA stream），当 computation time > transfer time 时 transfer 完全被隐藏。由 token threshold q 保证（q > φ·d_type/(2β)，Section 4.4）。
- **适用条件**：要求至少 2 个 expert 可同时驻留 GPU memory（大多数 MoE serving 系统已满足）。
- 效果：在 token rebalancing 基础上进一步降低 layer latency 8.6%（Switch128）和 13.8%（Qwen）。

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- HarMoEny: Efficient Multi-GPU Inference of MoE Models
