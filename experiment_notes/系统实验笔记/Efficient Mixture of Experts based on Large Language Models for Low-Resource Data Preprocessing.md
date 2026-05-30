## Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

- 属于Serving调度的实现是什么？实验比较什么？
  MELD 在推理阶段基于 Punica + vLLM 构建多 LoRA query 系统，支持在单 GPU 上同时 serving 一个 base LLM model 和最多 200 个 LoRA weights（即 experts），动态为 incoming queries 生成和切换 expert 而无显著计算效率损失。实验比较了 MELD、JellyFish(13B) 和 Mixtral(8×7B) 在 4×3090 和 1×3090 配置下的推理吞吐量和模型处理时间。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 3090（24GB VRAM）。两种配置：(1) 4×3090 GPU (vLLM)；(2) 1×3090 GPU (vLLM)。单机 256GB RAM，Intel Xeon Gold 5320 CPU @2.20GHz。

- 开源Serving框架是什么。修改了什么。
  使用 vLLM (Kwon et al. 2023) 作为 serving 框架，结合 Punica (Chen et al. 2023) 的多 LoRA serving 能力。论文未修改框架本身，而是提出了 **动态 LoRA 切换（Dynamic LoRA Switch）** 技术：传统 MoE serving 需要将多个 LoRA merge 到 base model 中（耗时），MELD 避免 merge 操作，仅加载和 concatenate 多个 LoRA 权重，显著降低 I/O 开销。MELD 的 model process time 比 JellyFish 快 10×，比 Mixtral 快 30×。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  代码开源：https://github.com/authurlord/MELD。使用 Punica + vLLM 架构部署。

  **多 LoRA Serving 流程**（从 query 输入到 GPU 输出的全过程）：

  1. **Query 到达与序列化**：Incoming DP query q 经过 serializer 转换为统一 dict 格式（包含 tuple 内容、table title、column header 等元数据），并添加 task-specific prompt instruction。

  2. **Router 调度（CPU 端）**：Query embedding 通过 M_RAG（fine-tuned sentence-bert）编码为 emb_q。Router network N 计算 softmax(W_N · emb_q)，选择 top-k（默认 k=3）experts。

  3. **LoRA 加载（GPU 端 - Punica）**：vLLM 的 LLMEngine 已加载 base model（Mistral-7B, FP16）到 GPU 显存。Punica 的 multi-LoRA 机制为每个 query 动态加载对应 k 个 expert 的 LoRA 权重（每个 LoRA 约数 MB）。单 3090 GPU 可同时持有 base model + 最多 200 个 LoRA weights。MELD 的动态 LoRA switch 避免 traditional merge：不将 LoRA 权重写入 base model，而是 concatenate 后在 forward pass 中按需应用。每个 query 仅激活 k 个 LoRA adapter。

  4. **vLLM 推理执行（GPU 端）**：
     - **Prefill 阶段**：query tokens 批量送入 Mistral-7B backbone。在 MoE Router 层，Punica 根据 N(q_u) 选择的 expert ID 加载对应的 k 个 LoRA adapter。每个 expert 输出经 gating weight g_i 加权求和。
     - **Load Balancing 优化**：MELD + vLLM 将相似 query 聚集到同一 GPU，在 4×3090 配置下实现 data parallelism（而非 tensor parallelism）。因为每个 expert（7B + LoRA）足够小，单 3090 可容纳 16 个 experts。
     - **Decode 阶段**：autoregressive 生成 output tokens，PagedAttention 管理 KV cache。

  5. **性能结果**：
     - 4×3090：MELD 吞吐量为 JellyFish(13B) 的 3.7×（JellyFish 需 tensor parallelism 跨 GPU 通信），为 Mixtral(8×7B, 56B total) 的 5.6×。
     - 1×3090：MELD 可 full precision 运行；JellyFish 需 4-bit quantization 才能部署（导致 1.3× 吞吐优势但性能下降显著）；Mixtral 即使 4-bit 量化也 OOM 无法部署。
     - Model process time（LoRA 合并与模型准备）：MELD 比 JellyFish 快 10×，比 Mixtral 快 30×。
