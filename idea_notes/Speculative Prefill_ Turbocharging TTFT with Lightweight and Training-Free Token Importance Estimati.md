## Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

- baseline方法是什么？
  Baseline 是标准 Llama 模型通过 vLLM 进行 prefill + decode 推理，以及对比方法 RAG-LLAMA（sentence-level RAG）、LLMLingua（文本级压缩）、MInference（sparse attention）。Baseline 全栈执行过程：

  **算法pipeline**：Transformer 标准 prefill 对所有 prompt token 执行完整的 attention + MLP 计算，每层计算 $A = \text{softmax}(QK^T/\sqrt{d_k})V$ 和 MLP 投影。RAG-LLAMA 基于 sentence embedding 相似度检索相关句子拼接后送入主模型。LLMLingua 使用小型模型做困惑度估计压缩 prompt 文本。MInference 使用离线搜索的稀疏 attention mask pattern 跳过部分 attention 计算，但不减少 MLP 计算量。

  **系统框架 (Serving)**：vLLM 0.6.3.post1，TP=8，enforce_eager=True。请求到达后直接进入 prefill phase，所有 prompt token 的 KV cache 被计算并写入 PagedAttention KV blocks，prefill 完成后再进行 decode phase。TP 组内各 GPU 通过 NCCL all-reduce 同步 MLP 和 attention 输出。

  **编译框架**：论文未明确说明（使用 vLLM 内置 CUDA kernels，无自定义编译优化）。

  **kernel调度**：论文未明确说明（使用 vLLM 默认 FlashAttention/PagedAttention kernel，TP=8 下自动并行调度）。

  **硬件架构**：论文未明确说明（标准 8×H200 服务器，各 GPU 通过 NVLink 478.1 GB/s + PCIe 5.0 x16 互联，无定制硬件加速器）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 SPECPREFILL 利用一个小型 speculator 模型（Llama-3.1-8B）在 prefill 前预测哪些 prompt token 是"局部重要的"，仅将筛选出的 token 子集送入主模型，从而跳过大量 attention + MLP 计算。核心设计对应 Baseline 缺陷：

  1. **Baseline 缺陷：Prefill 是 compute-bound，MLP 占比大且无法被 sparse attention 加速。**
     SPECPREFILL 解法：直接丢弃不重要 token，同时跳过 attention 和 MLP 计算（以及 TP all-reduce 通信），计算量正比于 token 保持率。对于 405B 模型 10% 保持率，理论 FLOPS 降至原来的 ~12.96%（speculator FLOPS 仅为主模型的 2.96%），实测 TTFT 加速达 7.66×，QPS 提升 7×。

  2. **Baseline 缺陷：MInference 等 sparse attention 方法在大 batch、短到中等长度 prompt 下 overhead 大。**
     SPECPREFILL 解法：跳过 attention + MLP 双重计算，在大 batch size 下优势更明显，相对 MInference 加速 2.54×-6.54×（同时保持 99.5% 质量）。

  3. **Baseline 缺陷：SwiftKV 需要轻量微调，GemFilter 需要两次完整 forward pass 开销大。**
     SPECPREFILL 解法：完全 training-free（利用同系列模型间的 token importance transferability），speculator overhead 随主模型增大而可忽略（405B 时仅 2.96% FLOPS），且可被 speculative decoding 复用摊销。

  **论文方法全栈执行过程**：

  **算法pipeline**：请求 prompt 先经过 speculator (8B) 的 N=8 步 look-ahead forward → 从各层各头提取 [N=8, L=32, S, H] 注意力张量 → max over H, L → mean over N → 得每 token 标量重要性分数 [S] → 1D avg pool 平滑 → 按 chunk 取平均 → Top-K chunks 选出 token 子集 → 保持各 token 原始 position IDs → 仅将 token 子集送入主模型 (70B/405B) 的完整 forward（包括该子集 token 的 attention + MLP 全流程）。

  **系统框架 (Serving)**：在 vLLM engine 初始化前 monkey patch 插入 speculator 加载与 token 选择逻辑。请求到达后：(1) 先由 speculator 处理（含 N 步 look-ahead decoding），利用 vLLM 的 slot mapping 机制追踪 query 数据；(2) N 步结束后 tp_gather_qk 收集 TP 组内 Q、K 分片；(3) 聚合注意力分数 → chunk selection → 筛选 token 子集；(4) 将筛选后的 token（含原始 position IDs）合并 decode 请求送入 base model forward。SPECPREFILL 与 speculative decoding 天然兼容：small model 同时服务于 prefill 阶段的 token 选择和 decode 阶段的 draft proposal，为"小型 speculator 完全辅助推理"范式铺路。

  **编译框架**：论文未明确说明（无自定义编译优化，沿用 vLLM 默认 CUDA graphs/kernels）。

  **kernel调度**：论文未明确说明（使用 vLLM 默认 FlashAttention + PagedAttention kernels，无额外 kernel 定制）。

  **硬件架构**：论文未明确说明（标准 8×H200/H100 GPU 节点，使用 NVLink + PCIe 5.0 互联，无定制硬件加速器）。
