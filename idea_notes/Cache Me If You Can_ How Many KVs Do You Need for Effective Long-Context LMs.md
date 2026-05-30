## Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

- baseline方法是什么？
  Baseline是现有的KV cache eviction方法，分为三类：(1) Post-fill eviction方法（PyramidKV, SnapKV, H2O）——在pre-filling全部完成后才基于attention scores evict KV，导致pre-filling期间的高peak memory和pre-fill stage几乎无KV footprint reduction；(2) Recency eviction方法（DuoAttention）——将attention heads分为retrieval heads和streaming heads，但依赖L2 reconstruction loss（而非next-token prediction loss）训练、continuous gating variable带来train-test gap、仅用synthetic passkey训练数据无力捕获复杂long-range dependencies；(3) Dynamic sparsity方法（NSA, MoBA, MInference）——仅减少inactive attention weights但不实际evict KV，无法降低KV memory。

  全栈执行例子（Baseline / DuoAttention on Llama-3.1-8B-Instruct, 128K context）：
  - 算法pipeline：训练时用synthetic passkey retrieval数据+L2 reconstruction loss重建hidden states，学习continuous gating z∈[0,1]；训练后按sparsity threshold rounding z→{0,1}产生train-test gap；inference时retrieval heads保留完整KV cache，streaming heads仅保留W=1024 local+S=128 sinks
  - 系统框架：基于PyTorch推理，支持chunked pre-filling（chunk_size=32K），无serving框架修改
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明（使用标准FlashAttention）
  - 硬件架构：通用GPU推理，论文未明确说明具体GPU型号

- 论文方法是什么？如何对应解决Baseline的缺陷？

  论文提出**(1) KV Footprint统一度量**和**(2) PruLong**以及**(3) Chunked Eviction**，分别对应baseline的三大缺陷：

  **1. KV Footprint → 解决fair comparison缺失（第2节）**：
  定义KV footprint = 所有timestep的un-evicted active+inactive KV entries的time-integrated sum，归一化至full causal attention。与KV cache size（instantaneous metric）不同，KV footprint同时捕获pre-filling和decoding两阶段的memory usage。引入critical KV footprint = 保留≥90% full attention性能的最小footprint，使不同方法在公平的utility-efficiency trade-off上可比。Appendix A展示了peak KV作为alternative metric，结论一致。

  **2. PruLong → 解决DuoAttention的三个核心缺陷（第4节）**：
  - **Next-token prediction loss替代L2 reconstruction**：直接优化语言模型的实际使用目标（token generation quality），而非proxy loss（hidden state reconstruction）；实验证明即使DuoAttention用4倍训练steps也无法在natural data上收敛，而PruLong轻松收敛（Recall 91.4 vs 38.6 at 70% sparsity）
  - **Hard concrete + Bernoulli masks替代continuous gating**：用hard concrete reparameterization [Louizos et al., 2018]将z建模为Bernoulli随机变量，训练时端到端采样离散mask并优化，消除train-test rounding gap；配合Lagrangian penalty实现精确target sparsity regularization，支持训练后任意sparsity extraction
  - **Natural long-context pre-training data替代synthetic passkey**：使用Gao et al. (2025)的continued pre-training mix（code repositories + books），包含多样化的long-range dependencies，使PruLong的head assignment能泛化到recall/RAG/re-ranking/ICL/summarization等多种task types

  **3. Chunked Eviction → 解决post-fill eviction的pre-fill高peak memory（第3节）**：
  将PyramidKV/SnapKV的eviction heuristic从"pre-fill后一次性执行"改为"chunked pre-filling的每个chunk后执行"：
  - Naive Chunked Eviction：每个chunk独立计算最后k个token的attention score → evict bottom KV
  - Patched Chunked Eviction：每个chunk末尾拼接prompt的最后k个token作为query → 用完整prompt的重要性信号指导eviction
  - 同时修复GQA下的KV replication issue：在KV group内mean-pool attention后再选择统一KV set，节省8×内存
  - Patched PyramidKV在RAG（<34% footprint）和LongQA（<35%）上取得所有方法中最优结果

  全栈执行例子（PruLong on Llama-3.1-8B-Instruct, 128K context, 70% streaming heads）：
  - 算法pipeline：1000 steps训练（batch 1M tokens, seq_len 131K, LR=1.0 for log α, LR=1.0 for λ1/λ2, model weights frozen）→ target sparsity warmup 0→0.7 over 800 steps → Lagrangian penalty驱动收敛 → 训练后按log α排序取top 30%为retrieval heads → inference时retrieval heads用full KV cache，其余仅保留W=1024+S=128 → KV footprint ~30%（critical KV footprint在Recall上46%，比DuoAttention低12 points）
  - 系统框架：基于PyTorch推理，支持chunked pre-filling（chunk_size=32K）；streaming heads的fixed-size KV cache使decoding阶段memory恒定；可应用于pre-SFT或post-SFT stage
  - 编译框架：论文未明确说明
  - kernel调度：标准FlashAttention（无自定义kernel），streaming heads的attention mask变化不影响FlashAttention tiling
  - 硬件架构：通用GPU推理；peak memory PruLong 26.3 GiB（Recall task at 70% sparsity）vs PyramidKV+P+C 33.7 GiB；throughput PruLong 10.8×10⁻² req/s vs DuoAttention 10.0×10⁻²

  **核心创新总结**：
  - KV footprint作为time-integrated memory metric是conceptual contribution，为KV eviction方法的fair comparison奠定基础
  - PruLong的三项设计（NTP loss + hard concrete masks + natural data）分别攻克DuoAttention的三项缺陷，在recall task上实现12 points critical footprint improvement
  - Chunked eviction让post-fill methods在chunked pre-filling时代重获竞争力——patched PyramidKV在RAG和ICL上最优
  - 关键发现：没有任何一种方法在所有task上最优（PruLong强于recall，PyramidKV+Patched强于RAG/ICL），揭示KV eviction没有one-size-fits-all解决方案
  - Pre-filling chunk size sensitivity是未预期的挑战——PruLong/DuoAttention在8K vs 32K chunk size下performance差异达20%，比PyramidKV更敏感
