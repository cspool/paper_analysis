## FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

- 属于Serving调度的实现是什么？实验比较什么？
  FastTree 作为 SGLang 的 plugin 实现，属于 Serving 调度层优化。核心实现：(i) 利用 SGLang 的 radix tree 管理全局 KV cache（已有机制），FastTree 读取该 radix tree 结构后生成 context-queries grouping plan；(ii) tree structure-adaptive runtime 在每次 radix tree 结构变化时（因新请求到来/旧请求完成）重新执行 greedy heuristic 搜索最优分组方案；(iii) 将 attention 计算从原有的 per-query 分离模式替换为 FastTree 的 tree-structured attention kernel，query 按共享前缀聚合后批量计算 attention；(iv) 预处理 overhead（CPU greedy search + grouping plan generation）被 SGLang 的多步连续 decoding 摊销，且可与 GPU 计算 overlap。
  实验比较：FastTree+SGLang vs SGLang-Triton vs SGLang-FlashInfer 在 4 种 tree-structured workload 上的端到端 throughput（tokens/s）：(A) multi-level system prompt（随机替换 Meta AI system prompt 中的 country/language）；(B) multiple few-shot learning（系统 prompt + 8 组 20-shot examples + 16 questions）；(C) multi-chain reasoning（每个问题 4 chains）；(D) multi-document QA（Llama-2 report 拆分为多文档前缀）。所有 benchmark batch=128, gen_len=256 tokens。额外进行 breakdown analysis（decoding latency、CPU preprocessing overhead、GPU kernel execution time）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU (80GB)，CUDA 12.2。CPU 端预处理轻量（BFS greedy search + virtual tree generation），在实验中 overhead 可忽略。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang v0.2.13（https://github.com/sgl-project/sglang），已集成 radix tree KV cache 管理。
  - FastTree 修改：作为 plugin 替换 SGLang 中 decoding 阶段的 attention backend。SGLang 原有两种 attention 实现（Triton kernels from LightLLM、FlashInfer CUDA kernels）。FastTree 新增第三种 backend，在 decode 阶段使用 tree-structured attention kernel，prefill 阶段沿用 FlashInfer。不修改 SGLang 的核心调度逻辑（continuous batching / radix tree 维护），而是在 attention 计算层插入优化。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/PanZaifeng/FastTree-Artifact（Apache-2.0）。Docker 环境含 SGLang 0.2.13、FlashInfer 0.1.6、Triton 3.0.0。

  使用例子（Llama-2-7B, GQA=1, benchmark B multiple few-shot learning, batch=128, gen=256）：
  1. **输入**：128 个并发请求到达 SGLang API server。这些请求共享 3-level tree prefix——系统 prompt (Meta AI, 3193 tokens) → 8 组 few-shot example 组合（20-shot each）→ 16 个独立 question per 组合。SGLang 将收到的请求组织为 radix tree（root=系统 prompt, L1=example combinations, L2=questions），KV cache 按 tree 结构非连续存储（paged KV cache blocks）。
  2. **Radix tree 维护**：SGLang 的 continuous batching 机制持续调度新请求进入 batch，已完成的请求移出。当 batch 成员变化（请求加入/完成），radix tree 结构更新，触发 FastTree runtime 重新搜索。
  3. **FastTree runtime**：读取当前 radix tree → BFS greedy heuristic 做 binary edge assignment → 生成 virtual tree → node-centric query aggregation → 输出 (context, {queries}) grouping plan。开销 < 1ms（被 decoding 循环摊销）。
  4. **Attention kernel 替换**：SGLang 原按 query 分别调用 FlashInfer attention kernel（每个 query 单独 load KV cache from HBM，GEMV 计算）。FastTree 替换为：按 grouping plan 将共享同一 context prefix 的 queries 聚合 → 单 kernel 处理 → Q 矩阵 tile 在 shared memory 中复用 KV tile → tensor core GEMM 替代 CUDA core GEMV。
  5. **Decoding loop**：SGLang 连续执行多步 decoding（amortize scheduling overhead）。每步的 attention 计算被 FastTree 加速（平均 1.9× over FlashInfer on Llama）。
  6. **输出**：生成的 tokens 返回客户端。Throughput = total output tokens / total time（含 scheduling + prefill + decode + communication）。FastTree 相比 SGLang-FlashInfer throughput 提升 up to 2.2×。

  - **作用**：弥补 SGLang 在 radix tree 内存层面的优化与其 computation 层面仍然执行 per-query 分离计算的 gap。具体而言，SGLang 的 radix tree 减少了 KV cache 内存占用（更多请求可同时服务），但 attention 计算仍重复加载共享 KV cache 且无法利用 tensor core（decode 阶段 GEMV）。FastTree 在 scheduling→computation 交界处优化，使内存布局（tree）直接指导计算聚合（grouping），实现 memory-aware computation optimization。
