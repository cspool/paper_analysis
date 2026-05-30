## Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

- 属于Serving调度的实现是什么？实验比较什么？
  实现：Star Attention 通过两阶段分布式推理调度实现多 host（多 GPU）的注意力计算。阶段一将 context 划分为 contiguous blocks 并分发到多个 context hosts 并行执行 blockwise-local attention（无 host 间通信）。阶段二将 query 广播到所有 hosts，各 host 独立计算 local attention 后，由 query-host 通过 gather 各 host 的 softmax 统计量（scalar s_h 和 vector A_h）聚合为 global attention。仅 query-host 更新 KV cache。实验比较 Star Attention vs Ring Attention（分布式 global attention 基线）、Vanilla 自回归生成（非分布式）的推理时间（time per sample, seconds）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU，bfloat16。8B 模型：16K-128K（8 GPU×4 workers）、256K-512K（16 GPU×8 workers）、1M（32 GPU×16 workers）。70B 模型：16K-32K（8 GPU×4 workers）、64K（16 GPU×4 workers）、128K（32 GPU×8 workers）。

- 开源Serving框架是什么。修改了什么。
  框架：HuggingFace Transformers（Wolf et al., 2020）和 NVIDIA TRT-LLM（NVIDIA, 2023）。修改内容：(1) 阶段一 context encoding 修改为 blockwise processing with anchor block —— 将输入的 long context 按 block size b 切块，每块 prefix anchor block c1，多 host 并行处理各自 block 并只保留非 anchor 部分的 KV cache；(2) 阶段二修改 attention 计算为 distributed softmax —— 各 host 独立对 query 做 local attention，query-host gather 所有 s_h 和 A_h 后通过 online softmax 聚合为 global attention；(3) 修改 KV cache 管理 —— 只有 query-host 在 decode 阶段更新 KV cache，context hosts 的 KV cache 保持冻结。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/NVIDIA/Star-Attention 。Serving 全过程如下：

  1. **输入**：用户提供 long-context prompt（context + query）。context 被切分为 n 个 contiguous blocks of size b。
  2. **阶段一启动**：n 个 augmented blocks（c'_1 = c1, c'_i = [c1, ci] for i>1）被分发到 H 个 hosts（GPU workers）。每个 host 获得 1 或更多 block。
  3. **阶段一执行（各 host 并行）**：每个 host 使用 Flash Attention 对 2b tokens 的 augmented block 做 self-attention → 生成 KV cache → 丢弃 anchor block 的 KV → 仅保留 ci 的 KV 到 kv_h。
  4. **阶段二启动**：query-host h_q 被指定，query tokens q 被广播到所有 hosts。
  5. **阶段二执行（per layer, per token）**：各 host 将 q 通过 Q, K, V 投影 → 使用 Flash Attention 对 local KV cache kv_h 计算 local attention A_h 和 softmax sum s_h → h_q 通过 all-gather 收集所有 (A_h, s_h) → h_q 执行 online softmax 聚合 A_global → standard transformer FFN → 生成 next token。
  6. **KV cache 更新**：仅 h_q 将新 token 的 K, V 追加到 kv_hq。context hosts 的 cache 不变。
  7. **输出**：逐 token 生成直到 EOS 或 max_new_tokens。

  **通信模式对比**：Ring Attention 在 prefill 阶段需要每个 host 顺序传递 KV cache block（ring communication），通信量 O(L×d)，延迟与 host 数成正比。Star Attention 在阶段一 zero communication（各 host 独立处理），阶段二仅每 token 传递 O(d) 数据（scalar + vector），通信量不随 context 长度增长。这使 Star Attention 对长序列有线性加速：32K 时 2.0×，64K 时 4.7×，128K 时 2.7×（8B），1M 时 16.9×（8B extended）。
