## The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  基于 vLLM 推理引擎的 FlashAttention 层级拦截实现六种 training-free 稀疏注意力方法的统一评估框架。实现通过子类化 `AbstractAttention` 基类，在 vLLM 执行 attention 计算时拦截并替换为稀疏模式。框架将稀疏注意力解耦为预处理（importance estimation）、稀疏 attention 执行、后处理三个阶段，由 vLLM 管理底层内存和 KV cache。论文重点不在修改 vLLM 调度本身，而在于：(1) 建立统一的实验框架用于跨方法对比；(2) 使用硬件无关的计算成本指标——**prefilling 阶段**用 FLOPs 公式（含 attention/QKV投影/MLP/embedding/logits 及 sparse indexing 开销，公式见 Section B.1），**decoding 阶段**用 memory transfers 公式（含 weight loading + KV cache 加载，含 Quest indexing 开销），作为部署调度决策依据；(3) 对 7065 个 (方法 × 模型 × 序列长度 × 稀疏度 × 任务) 配置进行批量自动化评估。

  实验比较：(1) isoCost Pareto 前沿——不同模型大小+稀疏度配置在相同计算成本下的 accuracy 对比（Figure 1）；(2) 不同 batch size 下解码阶段 KV cache 占比——batch size 1 时 KV cache 仅占 7-35%，batch size 64 时达 80-97%，指导何时 sparse attention 有效（Figure 13）；(3) 序列长度驱动的 attention 占比变化——16K 时 attention 占总 FLOPs 40%，128K 时达 80%，指导 sparsification 收益范围（Figure 12）；(4) sliding-window 架构（Gemma 3）的 attention 占比分析——64K batch=8 时 Qwen 14B attention 76% vs Gemma 12B 42%（Figure 14）。

- 硬件平台是什么，配置是什么。
  4 节点 × 8 块 NVIDIA H100 GPU，全 bf16 精度，vLLM 推理引擎。运行 21 天。论文报告的是硬件无关成本（FLOPs 和 memory transfers）而非 wall-clock time，声称在优化实现下这些指标与 wall-clock time 高度相关。

- 开源Serving框架是什么。修改了什么。
  **开源框架**：vLLM (https://github.com/vllm-project/vllm)。**修改**：不修改 vLLM 核心调度逻辑，而是在 vLLM 的 attention 执行路径中插入 `AbstractAttention` 拦截层。每种稀疏注意力方法继承 `AbstractAttention` 并实现 `pre_process`（重要性估计）、`forward`（稀疏 attention 计算）、`post_process`（结果后处理/缓存更新）。框架自动继承 vLLM 的 PagedAttention、continuous batching、tensor parallelism 等优化。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  **开源**：https://github.com/PiotrNawrot/sparse-frontier（MIT 许可证）。框架以 Hydra YAML 配置驱动，一条命令即可启动评估。

  **vLLM Serving 全流程（以 Vertical-Slash prefill + Quest decode 组合为例）**：
  ```
  1. 输入: 客户端发送 HTTP 请求 {"prompt": "...", "max_tokens": 512}
  2. vLLM Scheduler: 接收请求 → tokenize → 分配 KV cache blocks (PagedAttention)
     → 将请求加入 running queue
  3. Prefill 阶段 (Vertical-Slash):
     a. vLLM ModelRunner 执行 prefill，调用 Vertical-Slash.forward(Q, K_cache, V_cache)
     b. 计算近端 query window (256/512 tokens) 的近似 attention scores
     c. 选择 top-(k_v vertials + k_s slashes) 的 QK 交互对
     d. FlashAttention 仅对所选 pairs 执行 block-sparse attention
     e. 计算好的 KV cache 写入 PagedAttention block table
  4. Decode 阶段 (Quest，每步):
     a. 对新生成的 1 个 token 计算 Q
     b. Quest.forward: 计算 query-page 近似相似度 (min/max key 近似)
        → 选择 top-k pages → 仅对所选 page 内 token 做精确 attention
     c. 完整 KV cache 保留不动（不 eviction），仅选择性加载
     d. 新 token 的 KV 追加到 cache → Logits → Sample 下一个 token
  5. vLLM 输出: detokenize → 流式返回 tokens 到客户端
  6. 成本监控: 后台统计 FLOPs (prefill) 和 memory bytes transferred (decode)
  ```

  框架通过统一接口允许组合任意 prefill/decode 方法对——例如 Vertical-Slash (prefill) + Quest (decode) 是论文推荐的默认配置。配置通过 `configs/attention/` 下的 YAML 文件指定方法和参数。



- 属于Serving调度的实现是什么？实验比较什么？
  在 vLLM (0.6.3.post1) 上通过 monkey patch 实现 SPECPREFILL，仅需少量代码行和一个配置文件即可启用。核心修改：在 vLLM 的 prefill 阶段之前，插入 speculator 模型的 token 重要性推测流程。具体为：(1) 将 vLLM engine 的混合请求按 prefill/decode 拆分；(2) 对 prefill 请求使用 speculator 模型执行 N 步 look-ahead decoding（store_q=True 以保留 query 用于后续注意力计算）；(3) 通过 tp_gather_qk 在 tensor parallel 组内收集 Q、K；(4) 计算并聚合注意力分数，执行 chunk selection 筛选 token 子集；(5) 恢复 position IDs 后合并 prefill 和 decode 请求，调用 base model forward。KV cache 在不需要 look-ahead 时可省略以节省内存，但需要显式存储解码 token 的 queries（通过 vLLM 的 slot mapping 机制追踪）。batch look-ahead 时通过检查 EOS token 判断 token 有效性。

  实验比较：(1) 端到端 QPS 实验：启动 vLLM server + OpenAI API client（https://github.com/openai/openai-python），以恒定 QPS 异步发送 LongBench 查询，测量 client 端 per-query 延迟（含 prefill + decoding steps），对比有无 SPECPREFILL、不同 token 保持率（10%-90%）、70B vs 405B 模型；(2) TTFT 合成数据实验：使用 vLLM 官方 latency benchmarking 脚本，设置 max decoding step=1，测量不同 batch size × sequence length 组合下的 TTFT 加速比，对比 vanilla dense model 和 MInference。

- 硬件平台是什么，配置是什么。
  8 × NVIDIA H200（Tensor Parallelism = 8），CUDA 12.7，总 GPU TFLOPS 428.2，总 RAM 1123.2 GB，单 GPU 内存带宽 4052.8 GB/s，NVLink 带宽 478.1 GB/s，PCIe 5.0 x16，Disk Bandwidth 4730 MB/s。部分 MInference 对比在 8 × NVIDIA H100 上进行。speculator 和 base model 使用相同 TP=8 度。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：vLLM 0.6.3.post1。
  修改内容：
  1. Monkey patch 在 vLLM engine 初始化前插入 speculator 模型加载和 token 选择逻辑。
  2. 暴露 API，仅需几行代码 + 配置文件即可启用。
  3. enforce_eager=True 且 chunked_prefill=False（避免 vLLM 默认优化产生意外行为）。
  4. 实现 tp_gather_qk 函数在 tensor parallel 组内收集 Q、K 分片。
  5. 实现 slot mapping 追踪机制以正确检索解码 token 的 query 数据。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源代码：https://github.com/anonymous/speculative_prefill（ICML 2025 发表时公开）。

  全栈执行流程（以 LongBench 端到端 QPS 实验为例）：
  1. **请求输入**：OpenAI API client 以恒定 QPS 异步向 vLLM server 发送 LongBench 查询（从各 category 随机采样并 shuffle），每条请求包含 prompt text。
  2. **请求拆分**：vLLM engine 将到达的混合请求拆分为 prefill 请求 B_p 和 decode 请求 B_d。
  3. **Speculator 推测**（针对 B_p）：speculator (Llama-3.1-8B) 执行 N=8 步 look-ahead forward，每步产生新的解码 token。store_q=True 确保 query 被保存到 speculator KV cache C_s。
  4. **Tensor Parallel 协同**：因 speculator 和 base model 均使用 TP=8，tp_gather_qk 在 NCCL group 内收集各 rank 的 Q、K 分片，归并完整注意力矩阵。
  5. **注意力聚合与 token 选择**：计算 [N=8, L=32, S=prompt_len, H=32] 注意力张量 → max over L,H → mean over N → 得 [S] 标量分数 → 1D avg pool 平滑 → chunk → Top-K → 筛选 token 子集 T（如保持率 10%）。
  6. **Position ID 恢复**：T 中各 token 使用原始 position ID（非连续），decoding 起始 position 设为原 context length。
  7. **主模型 Forward**：合并 T + B_d → 送入 base model M 的 model_forward → 经 TP=8 在各 H200 GPU 上并行执行 sliced MLP + attention 计算。
  8. **响应返回**：生成的 token 经 vLLM 的 PagedAttention kernel 写回 KV cache blocks，decode 阶段按 max_tokens budget 自回归生成，最终返回 completion text 给 OpenAI client。
  9. **延迟测量**：client 记录 per-query 端到端延迟（prefill + decode），按固定 timeout 判定超时。QPS 从低到高扫描，观察三阶段模式（constant → linear → timeout）以确定最大支持 QPS。
