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

## Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

- 属于Serving调度的实现是什么？实验比较什么？
  本论文将四种 KV cache 压缩算法（KIVI、GEAR、StreamingLLM、H2O）集成到 LMDeploy v6.0.1 serving 框架中，并在 LMDeploy（原生支持 PagedAttention + FlashAttention）上测量吞吐和端到端延迟。在此基础上设计了请求路由器（Request Router）：在 4 GPU 部署场景下，1 GPU 运行 FP16 baseline，3 GPU 运行压缩算法，通过 throughput predictor 和 length predictor 预测每条请求在不同 GPU 上的估计解码吞吐和响应长度，路由请求到估计端到端延迟最短的 GPU。

  实验比较：
  (a) FP16 baseline 在不同框架（TRL, TRL+FlashAttention, LMDeploy）下的 decoding 吞吐对比。
  (b) 压缩算法在 LMDeploy 上不同 batch size (1~32) 和 prompt length (512~8192) 下的 prefill/decoding 吞吐。
  (c) Tensor parallelism (TP=1/2/4) 对压缩算法吞吐加速比的影响（LLaMA-7B/13B/70B, Mistral-7B）。
  (d) 端到端延迟 CDF（ShareGPT 样本，batch=1）：比较不同压缩算法的尾部延迟。
  (e) 请求路由器四种路由策略对比：Baseline（load-balancing by GPU memory usage）、w/ Throughput（路由到最高吞吐 GPU）、w/ Length（路由到最短响应 GPU）、w/ Both（路由到最小端到端延迟 GPU），用平均端到端延迟（秒）评估。w/ Both 相比 FP16 baseline 加速 1.45-1.80×。

- 硬件平台是什么，配置是什么。
  主要：4× NVIDIA A6000 (48GB) 通过 NVLink 互联，Intel Xeon Gold 6326 CPU @ 2.90GHz。
  吞吐预测器 profiled on A6000。请求路由实验使用 Poisson 分布 with RPS=10。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：LMDeploy (https://github.com/InternLM/lmdeploy) v6.0.1，原生支持 PagedAttention + FlashAttention。
  修改内容：
  1. 在 LMDeploy 的 attention 模块中集成 KIVI、GEAR、StreamingLLM、H2O 四种压缩算法的 KV cache 读写路径。
  2. 实现吞吐预测器：基于 Vidur (https://github.com/microsoft/vidur) 的 LLM runtime predictor，对 attention operator 在不同 batch size × sequence length × stage (prefill/decode) 下的 runtime 进行离线 profiling，将 profiling 结果注入预测器。
  3. 实现长度预测器：使用 LongFormer (max_seq_len=4096) 作为 BERT-based classifier，输入 prompt，输出 response_length/prompt_length ratio，分类精度 >85%。
  4. 实现请求路由器：根据 throughput predictor 和 length predictor 的输出，计算每 GPU 的 estimated end-to-end latency = prefill_time + decode_throughput × estimated_response_length，选择延迟最小的 GPU 路由请求。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文开源：https://github.com/LLMkvsys/rethink-kv-compression

  **LMDeploy + KV Cache Compression 推理全流程（LLaMA-7B + KIVI-4bit，4× NVIDIA A6000）**：

  1. **请求到达**：HTTP request 携带 prompt text 到达 LMDeploy API server。
  2. **请求路由器调度**：
     - Throughput predictor 查询 offline-profiled attention runtime table，根据当前 GPU 上已有请求的 batch_size 和 pending prompt_length，预测该 GPU 的估计解码吞吐 T_est (tokens/s)。
     - Length predictor (LongFormer) 接收 prompt text，预测该 prompt 在给定压缩算法下的 response/prompt length ratio。
     - 计算 estimated end-to-end latency = len(prompt)/prefill_throughput + len(response_est)/T_est。
     - 路由到 estimated E2E latency 最小的 GPU。
  3. **Prefill 阶段（LMDeploy TurboMind engine）**：
     - Tokenizer 将 prompt 转换为 token IDs。
     - LMDeploy 分配 PagedAttention 管理的 KV cache page blocks（固定 page size，动态分配，避免预分配至 max_len）。
     - FlashAttention kernel 执行 attention：tiling + online softmax，单 pass 完成，不保存中间 attention scores。
     - KV cache 以 FP16 存储。
  4. **KIVI 量化（在 decode 每步触发）**：
     - 新 token 的 key 执行 per-channel quantization (group_size=32, INT4)。
     - 新 token 的 value 执行 per-token quantization (INT4)。
     - 保留最近 R=128 个 token 的 KV cache 为 FP16（window-based 设计）。
     - LMDeploy 的量化 kernel（比 vLLM 更高效的 4-bit kernel）执行量化/反量化。
  5. **Decoding 阶段**：
     - 每步从 KV cache 加载 quantized K/V → dequantize → 与 Q 执行 FlashAttention → 输出 attention output。
     - 因 PagedAttention 管理 fixed-type page blocks，而 window-based quantization 需要同时管理 FP16（window）和 INT4（历史）两类 tensor，引入非结构化计算模式。
  6. **响应返回**：解码完成（遇 EOS 或 max_tokens=1024），返回 response text。

  **为什么选择 LMDeploy 而非 vLLM**（论文附录 A.4）：
  - LMDeploy 的 4-bit 量化 kernel 比 vLLM 更高效（BentoML benchmark 验证）。
  - vLLM 对 KV cache 压缩算法的集成支持不成熟：KIVI 作者 2024 年 4 月即指出 integrate KIVI into vLLM 存在困难（GitHub issue #4），至论文成文时仍无实质进展。
  - LMDeploy 提供更友好的 KV cache 压缩算法开发接口。

## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- 属于Serving调度的实现是什么？实验比较什么？
  将 LessIsMore 的 CUSA 稀疏注意力机制集成到 SGLang（Zheng et al., 2024）serving 框架中，配合 FlashInfer attention kernel 库实现推理服务。修改 SGLang 的 attention 计算路径：对于 Full Attention Layers 保持标准 FlashInfer attention，对于 Sparse Attention Layers 替换为基于统一 token 索引 ρ 的稀疏 attention kernel（仅加载 ρ 中选中的 KV cache），减少 decode 阶段的 memory bandwidth 消耗。实验比较 SGLang + LessIsMore vs SGLang + TidalDecode vs SGLang + Quest vs SGLang + Full Attention（FlashInfer）在 DeepSeek-R1-Distilled-LLaMA-8B 上的端到端每 token 解码延迟，context lengths 16K/32K/64K，token budget 2K。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A5000 GPU（SGLang 集成端到端延迟测试，Table 1）；单张 NVIDIA A100 80GB GPU（端到端 TBT speedup 测试，Table 3/Figure 6a）。FlashInfer 作为 attention 后端。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：SGLang（https://github.com/sgl-project/sglang）+ FlashInfer（https://flashinfer.ai/）attention kernel 库。LessIsMore 开源：https://github.com/DerrickYLJ/LessIsMore。
  修改内容：
  1. **Attention 路径扩展**：在 SGLang 的 attention 层中增加三种 layer 类型的 routing——Full Attention Layers 保持标准 FlashInfer kernel，Token Selection Layer 执行完整 attention + 跨 head 统一 token 选择（CUSA），Sparse Attention Layers 使用基于 ρ 的稀疏 attention kernel。
  2. **Token 索引管理**：维护 per-sequence 的统一 token 索引集 ρ，在 token selection layer 更新，后续层复用。
  3. **KV Cache 加载优化**：Sparse Attention Layers 仅从 KV cache 加载 ρ 中的 token 的 K/V，减少 HBM 带宽消耗。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/DerrickYLJ/LessIsMore（论文仓库），集成到 SGLang + FlashInfer。

  **SGLang + LessIsMore 推理全流程（DeepSeek-R1-Distill-Llama-8B，单 GPU NVIDIA A5000）**：

  ```
  输入：用户 reasoning prompt → SGLang Tokenizer → token 序列
  ↓
  [1] Prefill Stage
    - 对所有层执行 Full Attention（FlashInfer kernel）
    - 生成完整 KV cache C（所有层，所有 token）
    - 首 token 生成（TTFT）
  ↓
  [2] Decoding Loop（逐 token 生成，最多 32K tokens）
    For each new token:
      a) QKV Projection: h → q, k, v（通过 W_qkv 矩阵乘法，cuBLAS）
      b) KV cache 更新: C.append(k, v)
      c) Per-layer Attention:
          Layer 0-1 (Full Attention Layers):
            - FlashInfer full attention kernel
            - 对所有 C[:] token 计算 O = softmax(qK^T/√d)V
          Layer 12 (Token Selection Layer for Qwen3-8B):
            - FlashInfer full attention kernel（计算 attention + token 重要性估计）
            - CUSA token selection:
              P = q @ C.K^T  # [32, 1, L_kv]
              各 head 独立 TopK: ρ_head = TopK(P, k=K·0.75)
              跨 head 统一: ρ_unified = UnionFlatten(ρ_head)
              ρ = ρ_unified[:K·0.75] ∪ Recent(K·0.25)
          Layer 2-11, 13-31 (Sparse Attention Layers):
            - 复用 token selection layer 的 ρ
            - Sparse FlashInfer kernel: 仅从 KV cache 加载 K[ρ], V[ρ]
            - 计算 O = softmax(qK[ρ]^T/√d)V[ρ]
      d) FFN: 标准 Feed-Forward Network
      e) lm_head → Sampling → next token
    Until EOS or max_tokens (32K)
  ↓
  输出：生成的 reasoning trace + 最终答案
  ```

  **端到端延迟**（Table 1, DeepSeek-R1-Distill-Llama-8B, A5000, budget=2K, ms/token）：

  | Method | 16K | 32K | 64K |
  |--------|-----|-----|-----|
  | LessIsMore | 23.0 | 23.4 | 24.1 |
  | TidalDecode | 24.3 | 24.7 | 25.4 |
  | Quest | 24.2 | 24.4 | 24.8 |
  | Full Attention | 25.3 | 28.4 | 34.4 |

  **端到端 TBT Speedup**（Table 3, A100, SGLang）：
  - LessIsMore-2K: 16K→1.11×, 32K→1.25×, 64K→1.51×
  - LessIsMore-4K: 16K→1.09×, 32K→1.22×, 64K→1.48×

## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- 属于Serving调度的实现是什么？实验比较什么？
  将 InfiniteHiP 的模块化层次化剪枝注意力机制集成到 SGLang LLM serving 框架中，实现单 GPU 上长上下文（最高 3M tokens）的高吞吐推理服务。核心 Serving 修改包括：(1) 在 SGLang 的 attention 计算路径中插入多阶段剪枝 kernel，以 block sparse attention 替代 full attention；(2) 基于 Nvidia UVM（Unified Virtual Memory）实现 KV cache offloading——维护 GPU key bank（cache）+ CPU 统一内存空间（完整 KV cache）+ page table（global-to-local index 映射），采用 LRU 驱逐策略替换 HiP Attention 的原始策略；(3) 稀疏注意力 mask 按 stage 独立缓存，通过可配置的 refresh interval（fast: 32/16/8, flash: 96/24/8）减少解码时 mask 重计算频率；(4) 将 KV cache offloaded attention 实现为 graph-capturable 操作，避免 CPU overhead。实验对比 SGLang Runtime（SRT with FlashInfer）的 end-to-end decoding throughput，在 RTX 4090 24GB 和 L40S 48GB 上测试单 batch 场景（预期单序列即超出显存，因此仅测 batch=1）。

- 硬件平台是什么，配置是什么。
  (1) NVIDIA RTX 4090 24GB（PCIe 4.0 x8）+ AMD Ryzen 7950X 16C/32T + 128GB DDR5 5600MHz + Ubuntu 22.04.4 LTS + GPU Driver 535.171.04；(2) NVIDIA L40S 48GB（AWS g6e.48xlarge 节点）。KV cache offloading 通过 PCIe 4.0 x8（31.5× 比 VRAM 访问更慢的延迟）访问 CPU 内存。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：SGLang（https://github.com/sgl-project/sglang），InfiniteHiP 的修改版开源在 https://github.com/DeepAuto-AI/sglang/。
  修改内容：
  (1) **Attention 层替换**：将 SGLang 的标准 FlashAttention2/FlashInfer attention kernel 替换为 InfiniteHiP 的多阶段剪枝 + Block Sparse Attention pipeline。
  (2) **KV Cache 管理层**：增加 UVM-based KV cache offloading 机制，包括 GPU key bank（两个独立的 bank：mask-selection 用和 BSA 用）、CPU unified memory space、GPU page table、LRU 驱逐策略。
  (3) **Mask 缓存与刷新调度**：每个 pruning stage 维护独立的稀疏 mask 缓存和 refresh counter，根据 configurable interval 周期性更新。
  (4) **Graph Capture 兼容**：将 offloaded attention 实现为 CUDA graph capturable 操作，消除 CPU kernel launch overhead。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：核心库 https://github.com/DeepAuto-AI/hip-attention/，SGLang 集成版 https://github.com/DeepAuto-AI/sglang/。

  **SGLang + InfiniteHiP 推理全流程（单请求 3M token 上下文，L40S 48GB，AWQ Llama 3.1 8B + FP8 KV cache）**：

  ```
  输入：用户 prompt（3M tokens）
  ↓
  [1] Tokenization & Prefill 入口
    - SGLang tokenizer 将输入文本切分为 token 序列
    - 创建请求对象，分配序列 ID
  ↓
  [2] Prefill Stage（逐 chunk 处理，chunk_size=32K）
    For each chunk of 32K tokens:
      a) QKV Projection: GPU 上对当前 chunk 执行 linear projection
      b) InfiniteHiP Context Pruning:
         - Stage 0: 从全部 KV cache（含 host memory 中已有 token）选 32K key
         - Stage 1: 从 32K chunk 中通过 SelectRep + max chunk score 选 8K key
         - Stage 2: 从 8K 中选 ~2-4K key（基于 preset）
         BSA: 仅对这 ~2-4K key 执行完整 block sparse attention
      c) FFN: 标准 FFN 计算
    End For
    首 token 生成（TTFT）
    KV cache 写入 UVM：新 key/value 写入 CPU unified memory + GPU key bank（cache miss 时 fetch）
  ↓
  [3] Decoding Loop（逐 token 生成）
    For each new token:
      a) QKV Projection（仅新 token）
      b) KV Cache 管理:
         - 检查 GPU key bank 是否有命中（通过 page table 查 global→local 映射）
         - Cache miss: 从 CPU UVM 通过 PCIe 加载缺失 key/value 到 GPU bank
         - LRU 驱逐: 若 GPU bank 满，驱逐最久未使用的 cold token 回 CPU
      c) InfiniteHiP Context Pruning（mask refresh 检查）:
         - c^(i) mod n_refresh^(i) == 0? 
           Yes: 重新运行第 i stage pruning，更新 mask I^(l,i)
           No: 复用缓存的 mask（temporal locality）
         - 默认 refresh: stage1 每 16 步、stage2 每 8 步、stage3 每 4 步
         - Flash 配置: (96, 24, 8) → 解码速度大幅提升
      d) Block Sparse Attention: 使用 I^(l,N) mask 执行稀疏 attention
      e) FFN + Sampling: 生成下一个 token
    Until EOS or max_tokens
  ↓
  输出：生成的 token 序列 → SGLang detokenizer → 文本
  ```

  **关键性能数据（RTX 4090 24GB, 3K-Fast Offload）**：
  - 64K context: 64.5 tok/s
  - 128K context: 55.9 tok/s
  - 256K context: 46.6 tok/s
  - 512K context: 31.8 tok/s
  - 1024K context: 17.3 tok/s

  **关键性能数据（L40S 48GB, 3K-Flash Offload, 带 mask 缓存加速）**：
  - 64K context: 56.6 tok/s
  - 256K context: 49.4 tok/s
  - 512K context: 43.7 tok/s
  - 1024K context: 35.2 tok/s
  - 2048K context: 28.0 tok/s
  - 3072K context: 23.8 tok/s（3M tokens!）
  - vs SRT Estimated 3M: 7.25× speedup（23.8 vs 3.3 tok/s）

  **Mask 缓存效果（Table 4, 256K decoding latency per token）**：
  - No cache（所有 stage 重算）: 9,803 µs
  - Stage 1 cached: 2,579 µs（3.8× faster）
  - Stage 1&2 cached: 779 µs（12.6× faster）
  - All stages cached: 110 µs（89.1× faster）
  - Mask hit ratio: Stage 1: 71.67% → Stage 1&2: 98.75% → All: ~100%

## KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

- 属于Serving调度的实现是什么？实验比较什么？
  将 KV-Compress 的 KV cache 压缩方法集成到 vLLM v0.6.0，修改 PagedAttention 的 block 管理机制支持 per-head per-layer 可变 KV cache 大小的 paged attention，并通过 GPU 端 block 管理器实现并行化的 block 分配与调度。核心改动：(1) PagedAttention Block Layout 扩展：将原 vLLM 中每 block 存储所有 layer×all heads 的 KVs 改为每 block 仅存储单个 KV head 的 KVs，block table 从 B×L_max/b 扩展到 B×l×H×L_max/b；(2) GPU 端 Block 管理器：将 block table 和 context lengths 移至 GPU device memory，避免 CPU 端调度在 block 数量变为 l×H 倍后的性能瓶颈，实现 block 计数、分配、preemption 的并行化；(3) Block-level Eviction 调度：压缩后释放被 evicted 的连续 blocks，block 管理器回收后可用于新序列的 prefill 或 decoding；(4) 压缩调度策略：prefill 后 + 当 preemption 即将发生时触发压缩，以最大化 batch 扩展与最小化 preemption。实验比较：throughput benchmark 上 KV-Compress 修改的 vLLM vs vanilla vLLM v0.6.0，在 Llama-3.1-8B on L4 和 Llama-3.1-70B-FP8 on H100 上测量不同压缩率（1x-64x）和不同输入长度（500-12000 tokens）下的总吞吐量（tokens/s）及最大 decoding batch size。

- 硬件平台是什么，配置是什么。
  Llama-3.1-8B-Instruct：单 NVIDIA L4 GPU（24GB），gpu_memory_utilization=0.9，max-model-length=19,000；Llama-3.1-70B-Instruct-FP8：单 NVIDIA H100 GPU（80GB），gpu_memory_utilization=0.96，max-model-length=33,000。Both 配置中 vRAM 受限于大模型参数（L4 上 8B ~16GB + KV cache，H100 上 70B-FP8 ~70GB + KV cache），是 throughput benchmark 的理想场景。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：vLLM v0.6.0（https://github.com/vllm-project/vllm）。KV-Compress 修改版开源在 https://github.com/IsaacRe/vllm-kvcompress/tree/main。核心修改：
  (1) **Block Table 扩展**：原 vLLM block table T ∈ R^{B×L_max/b} 共享索引跨所有 layers 和 heads。KV-Compress 扩展为 T ∈ R^{B×l×H×L_max/b}，每 (layer, head) 对有自己的 block table，block 中仅包含该 head 的 KVs ∈ R^{b×d}（原 block 为 l×H×b×d）。物理 cache 从 l 个 per-layer tensor K^{(m)} ∈ R^{N×H×b×d} 改为单一 unified cache K_u, V_u ∈ R^{N×b×d}。
  (2) **GPU 端 Block 管理器（On-device Allocation）**：原 vLLM 的 block 管理器在 CPU 端运行，scheduling runtime 随 block 数量线性增长。KV-Compress 中 block 数量为 l×H 倍（Llama-3.1-8B: 32×8=256 倍），CPU 端调度 loop 在某些情况下耗时超过 forward pass。因此将 block table、context lengths、free/allocated block tracking 全部移至 GPU，并行计算 block 分配与释放。Prefill 时从 token length 直接计算所需 blocks；decoding 时从 on-device context lengths tensor 并行计算额外 block 需求；preemption 时并行计算所有 layers 和 heads 的 freed blocks。
  (3) **压缩调度集成**：压缩步骤在每次 prefill 后和每次 preemption 即将发生时执行。使用 PyTorch sort API 进行 metric 排序和 block eviction 选择。GPU block 管理器回收 evicted blocks。
  (4) **Block Size**：b=16，所有实验使用 eager mode（无 CUDA graph）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/IsaacRe/vllm-kvcompress/tree/main

  **vLLM + KV-Compress 推理全流程（256 prompts, Llama-3.1-8B on L4, compression rate 32x）**：

  ```
  输入：256 input prompts，固定 output=500 tokens
  ↓
  [1] vLLM Scheduler 初始化
    - GPU block manager 分配 unified KV cache K_u, V_u ∈ R^{N×16×d}
    - Block tables 初始化 T ∈ R^{256×32×8×L_max/16}
    - Context lengths tensor C ∈ R^{256×32×8} on GPU
  ↓
  [2] Prefill Loop（逐 prompt chunk size 调度）
    For each schedulable prompt:
      a) Kernel: QKV projection (cuBLAS) → FlashAttention/PagedAttention kernel
         - 每层每 head 通过 T[seq, layer, head, :] 索引对应 blocks
         - 从 K_u, V_u 中按 block 加载 K,V 到 SRAM
         - Attention 计算（eager mode, no CUDA graph）
      b) Block Allocation: GPU block manager 计算并分配所需 blocks
         - Prefill: 每 head 初始分配相同数量 blocks = ceil(L_prompt/16)
      c) First token generation (TTFT)
      d) Metric Calculation（KVC-w, w=8, p=7）:
         - 对 observation window 内 queries 计算 Σ(A_hij)² 累积到 M ∈ R^{N×16}
         - GQA query-group aggregation
      e) KV-Compress compression iteration:
         - Sort M by (head, metric) → block-level eviction candidates
         - Sort blocks by max metric → select E_s blocks to evict
         - MoveCache: 重排物理 cache 使 evicted blocks 连续
         - Free E_s blocks → GPU block manager 回收
      f) Store logical indices P for this sequence's remaining KVs
  ↓
  [3] Decoding Loop（逐 token 生成）
    For each new token per sequence:
      a) QKV projection → PagedAttention（通过 block table 索引 compressed KV cache）
      b) New KV pair 写入 cache（分配新 block 或填入现有 block 空隙）
      c) Context lengths C updated on GPU
    ↓
    [3a] 压缩调度检查（每次 iteration 后）:
      if 有序列新完成 prefill:
          将该序列加入 compression batch
          执行步骤 [2e] 的压缩流程
      if preemption 即将发生（free blocks 不足）:
          选择 compression batch 中最早未压缩的序列
          执行压缩 → 释放 blocks
          若仍不足：preempt 最低 priority 的序列
    ↓
    [3b] Continual Compression:
      每 step 累积新 token 的 Σ(A_hij)² 到 M
      按需触发基于更新后 metric 的再次 eviction
  ↓
  输出：256×500 generated tokens → detokenize → 文本
  ```

  **GPU Block Manager 并行分配细节**：
  ```
  # Prefill 分配（token length → blocks）
  required_blocks = ceil(prompt_length / 16)  # 每 head 相同
  flat_free_tensor: 长度为 N 的 bool tensor（1=free, 0=allocated）
  allocated = cumsum_prefix_scan(flat_free_tensor)  # GPU parallel prefix scan
  for each sequence s, layer m, head h (parallel on GPU):
      T[s, m, h, 0:required_blocks] = allocated_indices[offset_s_m_h: ...]

  # Decoding 分配（对已有序列的 running heads）
  for each (s, m, h) in parallel:
      last_block_used = C[s, m, h] % 16
      if last_block_used == 0:  # 需要新 block
          allocate one block from free list
          T[s, m, h, C[s, m, h] // 16] = new_block_idx
  ```

  **关键性能数据（Llama-3.1-8B on L4, compression rate 32x）**：
  - L_c=500: 2.54x throughput over vanilla vLLM
  - L_c=2000: ~3x throughput
  - L_c=6000: 4.93x throughput
  - L_c=6000, compression rate 64x: 5.18x throughput
  - Max decoding batch size: 100+ (vs vanilla <20)，compression rate 16x+ 时 observed

  **Llama-3.1-70B-FP8 on H100**：
  - L_c=6000, compression rate 64x: 2.14x throughput
  - L_c=6000, compression rate 8x: 1.8x throughput

  **较大的 input context length 需较大 compression rate 才能观察到近似线性的 batch size 增长**：因为序列需在 prefill 后才能被压缩，即使 cache 空间足够装 10 个 compressed 序列，若无法装 1 个 uncompressed 序列也仍无法扩展 batch。

## MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

- 属于Serving调度的实现是什么？实验比较什么？
  MagicDec 将基于压缩 KV cache 的推测解码（Speculative Decoding）集成到 LLM serving 系统中，用于长上下文、大批量服务的吞吐与延迟双优化。核心服务端实现是将 self-speculation 或 small-draft speculation 的 draft-verify pipeline 嵌入到 decode 循环中，在 prefill 阶段生成压缩 KV cache 供 draft 使用，在 decode 阶段使用完整 KV cache 进行验证。对于静态 KV 压缩方法（StreamingLLM/SnapKV），压缩 KV cache 在 prefill 阶段一次性构建完成，decode 期间无需额外搜索开销。

  实验比较：
  1. 不同 draft 策略在 serving 场景下的 speedup：autoregressive decoding vs StreamingLLM self-speculation vs SnapKV self-speculation vs 小 draft model（Llama-3.2-1B + StreamingLLM KV）
  2. 不同 batch size（32-256）和 sequence length（1K-100K）下的 speedup 变化趋势
  3. MLC-LLM backend vs self-implemented backend 的性能对比
  4. 不同 GPU 平台（A100/H100/L40）上的 speedup

- 硬件平台是什么，配置是什么。
  NVIDIA 8×A100 80GB（8-way tensor parallelism）、NVIDIA 8×H100 80GB + 4×H100（tensor parallelism）、NVIDIA 8×L40（低成本 GPU，tensor parallelism）。bfloat16 精度。

- 开源Serving框架是什么。修改了什么。
  开源框架：MagicDec 实现了两种 serving backend：
  
  **Backend 1 — Self-implemented（GPT-Fast based）**：基于 PyTorch 官方 GPT-Fast（https://github.com/pytorch-labs/gpt-fast）构建，集成 FlashInfer（https://github.com/flashinfer-ai/flashinfer）加速 attention、torch.compile 编译模型、Triton-based matrix multiplication 加速 MLP 层、CUDA graphs 减少 CPU kernel launch overhead、tensor parallelism 用于 embedding layer 加速。这是论文主要结果使用的 backend。
  
  **Backend 2 — MLC-LLM**（https://github.com/mlc-ai/mlc-llm）：基于 MLC-LLM 实现的 speculative decoding，用于验证方法的跨框架泛化性。
  
  修改内容（Self-implemented backend）：
  1. **Speculative Decoding Pipeline**：在 decode 循环中插入 draft phase（使用压缩 KV cache 生成 γ 个候选 token）和 verify phase（target model 使用完整 KV cache 并行验证），按 greedy matching 确定最终接受的 token 数
  2. **压缩 KV Cache 管理**：在 prefill 阶段基于最后一层 attention scores（SnapKV）或 attention sink pattern（StreamingLLM）选择稀疏 KV 位置，存储压缩后的 K_draft/V_draft 供 draft phase 使用
  3. **Tensor Parallelism for Embedding**：对 embedding layer 实现 tensor parallelism 以加速 draft 阶段（draft 阶段 token-by-token 生成，embedding 延迟占比大）
  4. **CUDA Graph Optimization**：使用 PyTorch CUDA graphs 封装 decode step，减少 CPU-GPU kernel launch overhead
  5. **torch.compile + Triton MatMul**：编译模型并使用 Triton 实现矩阵乘法 kernel，加速 MLP 层计算

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/Infini-AI-Lab/MagicDec（ICLR 2025，MIT License）。

  **MagicDec Serving 全流程（Self-implemented backend，LLaMA-3.1-8B SnapKV self-speculation，8×H100，batch=128，S=32000）**：

  ```
  输入：用户 prompt 列表（128 个 32K-token 序列）
  ↓
  [1] Tokenization & Batching
    - Tokenize 128 prompts → token_ids [128, 32000]
    - 同质 batch（所有序列 same length 或 padded）
  ↓
  [2] Prefill Stage（dense attention，8-way TP）
    - FlashInfer 执行 batch prefill attention
    - 生成完整 KV cache C_full [128, 32000, 32 layers, 8 heads, 128 dim] ≈ 25.2 GB
    - 存储最后一层 attention weights for KV selection
    - 执行 SnapKV selection:
        attn = LastLayerAttentionWeights  # [128, 8, 1, 32000]
        pooled = AvgPool1d(attn, kernel=5)  # [128, 8, 1, 32000]
        obs_win = pooled[:, :, :, -32:]
        top_indices = TopK(pooled[:, :, :, :-32], K-32)
        draft_indices = sort(concat(obs_win_positions, top_indices))
      → K_draft, V_draft [128, 2049, 32 layers, 8 heads, 128 dim] ≈ 1.6 GB
    - 生成首 token → output_buffers
  ↓
  [3] Decode Loop（CUDA graph captured）:
    while not all sequences done:
      Step A: Draft Phase（γ=6，使用压缩 KV）
        for i in 1..γ:
          - Embed(token) → [128, d_model]
          - Sparse Attention: s = Q @ K_draft^T / sqrt(d_head) → [128, 8, 1, 2049+i]
          - Softmax + V_draft 聚合
          - FFN(LayerNorm(attn_output))
          - LM Head → next_token
          - 追加 (k_new, v_new) 到 K_draft, V_draft
          - 如果遇到 EOS，提前终止 draft
        → draft_tokens [128, γ']
      
      Step B: Verify Phase（完整 KV cache）
        - 对 [current_token] + draft_tokens 的 γ'+1 个位置并行 forward
        - 使用完整 KV cache C_full（FlashInfer attention）
        - 得到 verified logits → greedy match 比对
        - 接受 Ω(γ,α) ≈ 5.07 个 token（α≈0.85）
        → 追加 accepted tokens 到 output_buffers
        → 更新 C_full（追加新 KV）
      
      Step C: Batch Management
        - 检查 EOS → 标记完成序列
        - 若全部完成 → break
  ↓
  [4] Output
    - Detokenize output_buffers → 128 个 response 文本
    - 统计 metrics: TTFT, TPOT, throughput (tokens/s), speedup
  ```

  **性能指标（SnapKV self-speculation，8×H100）**：
  - Batch=128, S=32000, PG-19: AR=26.07ms/tok → SD=12.96ms/tok, speedup=2.01x
  - Batch=41, S=100000, cwe: AR=25.83ms/tok → SD=10.29ms/tok, speedup=2.51x
  - Batch=64, S=32000: AR=14.84ms/tok → SD=9.05ms/tok, speedup=1.64x (SnapKV)

  **与 MLC-LLM backend 对比（Table 4/5）**：
  - Self-implemented backend 显著优于 MLC-LLM（更低的 draft & verification overhead）
  - 但两者 trend 一致：speedup 随 batch size 增大而提升

## MagicPIG: LSH Sampling for Efficient LLM Generation

- 属于Serving调度的实现是什么？实验比较什么？
  提出GPU-CPU异构系统设计，将LLM解码分为三部分：(1) GPU执行所有线性投影(MLP, W_Q, W_K, W_V, W_O)和LSH随机投影哈希码计算；(2) CPU存储LSH哈希表并执行采样检索和稀疏注意力计算(o=Softmax(qK^T/√d)V)；(3) GPU上保留sink tokens和local tokens的KV cache（on-device cache），不经过LSH采样。系统通过recursive attention技术合并GPU和CPU的注意力输出。实验比较了不同硬件配置下的吞吐量和延迟：A100 (1.5× throughput提升)、L20 (5.0× throughput提升)、RTX 4090 (3.3× throughput提升，96K context单请求54ms解码延迟)，MagicPIG可以容纳比GPU全注意力baseline大12×以上的batch size。

- 硬件平台是什么，配置是什么。
  GPU: NVIDIA A100-80GB, L20-48GB, 模拟RTX 4090-24GB (L20限制显存)。CPU: Intel Platinum 8480+ (搭配A100), Intel 8563C (搭配L20)。CPU DRAM带宽100-200GB/s，约为GPU VRAM带宽的10-20%。

- 开源Serving框架是什么。修改了什么。
  论文未使用现成Serving框架(vLLM等)，而是自建PyTorch + FBGEMM系统。GPU部分使用原生PyTorch实现，CPU注意力计算使用FBGEMM (bfloat16精度)。修改/创新点：(1) 将KV cache完整offload到CPU DRAM，通过5-10×稀疏性弥补CPU带宽劣势；(2) 在GPU上新增LSH随机投影模块(内存开销400KB~825KB)；(3) CPU上新增L张哈希表存储和查询逻辑；(4) 引入on-device cache将sink tokens和local tokens的KV保留在GPU，避免全走CPU路径。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接: https://github.com/Infini-AI-Lab/MagicPIG。Serving框架全流程：
  输入：用户prompt → Tokenization → embedding
  GPU端执行顺序：
    1. 线性投影：计算q = W_Q·x, k = W_K·x, v = W_V·x，以及MLP层
    2. 随机投影：q_code = Sign(q @ W)，W∈R^{d×(K×L)}，产生K×L bit哈希码
    3. 传输q_code和新生成的k,v到CPU（通过PCIe）
  CPU端执行：
    4. 查询L张哈希表：S = Query(HT, q_code)，收集碰撞的key索引
    5. 稀疏注意力计算（FBGEMM bfloat16）：计算q·K_S^T → softmax → weighted sum of V_S
    6. 结果传回GPU
  GPU端收尾：
    7. Recursive attention合并：将CPU返回的ō_cpu与GPU上的ō_gpu(on-device sink+local tokens)合并
    8. 输出投影W_O → 下一个token
  作用：突破GPU显存限制，在24GB GPU上服务96K context、在48GB GPU上服务>12× baseline batch size，同时保持低延迟和高吞吐。

## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- 属于Serving调度的实现是什么？实验比较什么？
  ShadowKV 是一个面向长上下文 LLM 的高吞吐推理 serving 系统。其核心服务于调度层实现为：(a) **GPU 显存管理**：prefilling 阶段对每层 pre-RoPE key cache 做在线 SVD 低秩压缩（rank=160），仅保留低秩投影矩阵 A 和 B 在 GPU，value cache 全量 offload 至 CPU，仅保留检测到的 outlier chunk（0.3%）的 KV 对在 GPU，将 GPU KV cache 显存占用降低 >6×。(b) **请求调度与 batch 扩容**：由于 GPU KV cache 显存大幅减少，相同 GPU 可容纳更大 batch size（从 2-8 扩至 12-48），支持从 60K 到 488K 更长上下文的高吞吐服务。(c) **CPU-GPU 数据传输调度**：decoding 阶段使用 CUDA multi-stream 将 key cache 低秩重建（GPU 计算）与 value cache CPU 抓取（PCIe 传输）重叠，隐藏 PCIe 延迟，降低 sparse attention 的 decoding overhead。实验比较 Full Attention（GPU 显存内完整 KV cache）在不同 batch size 下的吞吐，以及 Quest、Loki、InfiniGen 在相同 sparse budget 下的效率，展示 ShadowKV 可支持 6× larger batch size，吞吐提升最高 3.04×。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU，GPU 内存带宽 2 TB/s，PCIe 带宽 31.5 GB/s。

- 开源Serving框架是什么。修改了什么。
  基于 PyTorch + HuggingFace Transformers 构建 serving 框架，集成了 FlashAttention（FlashAttention-2）、FlashInfer（fused kernels、layer norm）、vLLM（PagedAttention）中的高效 kernel，以及 CUTLASS。ShadowKV 在此基础上修改/新增了以下 serving 层组件：
  1. **Prefilling 阶段 SVD 计算**：对每层 pre-RoPE key cache 调用 SVD 分解，保留 rank-r 低秩投影
  2. **KV cache 存储管理**：替换原有 GPU-side KV cache 存储策略，GPU 端保留 A、B、landmarks L、outlier KV；CPU 端存储完整 value cache V_CPU
  3. **Decoding 阶段 KV 选择与重建调度**：用 landmarks 近似注意力选出 top-k chunks → 并发调度 key 重建（GPU）与 value 抓取（CPU→GPU）→ 多流重叠
  4. **Cache mechanism**：利用相邻 decoding step 间 KV 选择的高命中率（~60%），维护命中缓存减少重复计算和 PCIe 传输

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV（Apache 2.0）。Serving 框架全链路执行过程：
  
  **请求到达与 Batch 组建**：
  1. 用户提交推理请求（prompt + max_new_tokens），框架将多条请求按到达顺序组建成 batch
  2. 由于单个请求的 GPU KV cache 仅占原有的 ~1/7（低秩键 + landmark + outlier vs 完整 KV），batch size 可从 Full Attention 的 2-8 扩至 12-48
  
  **Prefilling 阶段**（以一条 prompt 128K tokens、batch=24 为例）：
  3. 输入 tokens 经过 embedding → 对每个 Transformer layer：
     a. QKV 投影：Q, K, V = W_Q·X, W_K·X, W_V·X
     b. Pre-RoPE K 在线 SVD：对 K_{pre-RoPE} ∈ R^{128K × 128} 做 truncated SVD，得 A ∈ R^{128K×160}、B ∈ R^{h_kv×160×128} 存 GPU
     c. Post-RoPE K 分 chunk（chunk_size=8，得 16K chunks），每个 chunk 算均值作为 landmark L ∈ R^{h_kv×16K×128} 存 GPU
     d. Cosine similarity 计算每 chunk 内各 token 与均值的相似度，选出 48 个 outlier chunk，其 KV 对存 GPU static cache
     e. 其余 value cache V offload 到 CPU（通过 PCIe），对应 landmark L 保留在 GPU
     f. FlashAttention 完成 prefill attention（本文保留完整 prefill attention）
  
  **Decoding 阶段**（autoregressive generation，每步生成一个 token）：
  4. 新 token 经 embedding + QKV 投影得 query Q ∈ R^{24 × h_q × 1 × 128}
  5. **Landmark-based KV 选择**：
     - P ← MatMul(Q, L^T) → Softmax(P/√d) → sum over query heads → max over kv_group
     - ArgTopK 选出 top-k=256 个 chunk indices I
  6. **缓存命中检查**：对比上一步选择的 chunk indices 与当前步，命中部分（~60%）跳过
  7. **并发执行（CUDA multi-stream）**：
     - Stream 1 (GPU compute)：K_sparse ← MatMul(Gather(A, I_miss), B)，RoPE(K_sparse)
     - Stream 2 (PCIe→GPU)：V_sparse ← Gather(V_CPU, I_miss) 从 CPU 内存通过 PCIe 读取
     - 两个操作时间接近，重叠后总延迟 ≈ max(PCIe 传输, 低秩重建) 而非二者之和
  8. **Attention 计算**：K ← [K_outlier; K_sparse; K_new]，V ← [V_outlier; V_sparse; V_new]，FlashAttention(Q, K, V)
  9. FFN → 输出投影 → next token
  10. 新 token 的 pre-RoPE key 投影到同一低秩空间（K_new × Ψ，其中 Ψ 为 prefilling SVD 的右奇异矩阵），追加到低秩状态
  
  **作用**：通过将 GPU KV cache 显存降低 6-7×，支持 6× larger batch size，吞吐从 Full Attn 的 80.78 tokens/s（batch=4, 122K context, Llama-3.1-8B）提升至 245.90 tokens/s（batch=24），加速 3.04×，甚至超过无限 GPU 显存假设下的吞吐（134.30 tokens/s, batch=Inf）。

## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现 ShadowKV 系统，通过 GPU-CPU 异构 KV cache 管理策略大幅提升长上下文 LLM 推理吞吐。核心调度优化：(1) **KV Cache 分层存储**：pre-filling 阶段对 pre-RoPE key cache 执行 SVD 后仅保留低秩投影（rank=160）在 GPU，value cache 全量下放 CPU，仅 outlier chunk 的完整 KV 对保留 GPU 作为 static cache；(2) **Sparse Attention 解码调度**：解码时用 landmarks + Q 近似注意力选择 top-k chunk，通过 CUDA multi-stream 将 CPU→GPU 的 value 取回与 GPU 上低秩 key 重建重叠执行，掩盖 PCIe 传输延迟；(3) **Temporal Locality Cache**：利用相邻解码步 KV 选择高重复率（>60%），通过 index scan 仅重建缺失 chunk 的 KV 对，减少 60% 计算和数据搬运；(4) **大 Batch 支持**：通过 GPU 内存节省（KV cache 占用降至 1/6），将最大 batch size 从 2-8 提升至 12-48，超越假设无限显存下的吞吐。

  实验比较：(1) 吞吐量实验：A100 上测量 Llama-3-8B-1M、Llama-3.1-8B、GLM-4-9B-1M、Yi-9B-200K 在 60K/122K/244K 上下文下的生成吞吐（tokens/s），对比 Full Attention baseline 最大 batch size 和 Infinite batch size（理论极限）；(2) Batch size 扩展性实验：Llama-3-8B-1M 在 60K/122K/244K/488K 上下文、batch size 2-48 下的吞吐矩阵；(3) 延迟分解实验；(4) 与 Quest 在 1M 上下文下的效率对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU (80GB)，GPU 内存带宽 2 TB/s，PCIe 4.0 x16 带宽 31.5 GB/s，搭配 CPU 大内存（用于存放 offloaded value cache）。CPU-GPU 通过 PCIe 连接。

- 开源Serving框架是什么。修改了什么。
  ShadowKV 基于 PyTorch 实现，与 vLLM 的 PagedAttention 机制兼容，集成 FlashInfer 的高效融合 kernel（如 layer norm）。使用 FlashAttention (v2) 作为注意力计算后端，利用 CUTLASS 编写自定义 CUDA kernel。修改包括：
  (1) KV cache 管理策略：不再使用 vLLM 标准的全量 GPU KV cache，而是在 pre-filling 后将 value 移至 CPU pinned memory，GPU 仅保留低秩 key 投影、landmarks 和 outliers；
  (2) Attention 计算流程：解码时不使用标准 FlashAttention 的完整 QK^T 计算，而是先通过 Q×L^T 的 landmark 近似 attention 选择 top-k chunk，再对选中的 chunk 执行精确 attention；
  (3) 数据搬运调度：使用 CUDA multiple streams 实现 key 重建（GPU 计算）与 value 取回（PCIe 传输）的 overlap，等效带宽可达 7.2 TB/s（理论分析）；
  (4) Temporal cache：维护 chunk index 的最近访问记录，检测相邻步的 chunk 重复，跳过已缓存 KV 对的取回和重建。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV 。框架输入到硬件执行全过程：

  **输入**：用户请求到达，包含 prompt tokens 和生成参数（max_tokens, temperature 等）。

  **Pre-filling 阶段**（每层重复）：
  1. 输入 tokens → PyTorch Embedding → Transformer Block
  2. QKV Projection：X @ W_qkv → Q, K, V（GPU 计算，Tensor Core）
  3. 对 pre-RoPE K 执行 SVD → 存储 A, B 在 GPU（cuSOLVER/cuBLAS）
  4. K 经 RoPE → K_RoPE，分 chunk 计算 landmark 和 outlier
  5. V（非 outlier 部分）→ CPU pinned memory（cudaMemcpy，PCIe）
  6. Landmarks L、outlier KV 对保留 GPU
  7. Q, K_RoPE(完整), V(完整) → FlashAttention 计算 prefill attention（所有 token 对其他 token）
  8. FFN 计算

  **Decoding 阶段**（每层每步重复）：
  1. 新生成 token → Q 向量
  2. Q × L^T → 近似 chunk attention scores（自定义 CUDA kernel，GPU 计算）
  3. Softmax + TopK → 选择 k 个 chunk indices
  4. 并行执行（CUDA multi-stream）：
     - Stream 1: Gather(A, I) × B → 重建 sparse K cache（GPU，Tensor Core）
     - Stream 2: cudaMemcpy(V_CPU[I], V_sparse, H2D)（PCIe 传输）
  5. Temporal cache index scan → 跳过已缓存 chunk
  6. 拼接 K = [K_outlier; RoPE(K_sparse); K_new], V = [V_outlier; V_sparse; V_new]
  7. Q, K, V → FlashAttention（仅对选中 chunk + outliers 计算）
  8. FFN → 输出 logits → 采样下一个 token

  **关键调度决策**：batch size 由可用 GPU 内存决定 — Full Attention 在 60K context 仅能容纳 batch=8，ShadowKV 可容纳 batch=48（6× 提升）。Sparse budget k=256 对应 1.56% 的 128K 序列。等效带宽分析公式：
  $$\widetilde{B} = \frac{2SB_{\mathrm{GPU}}}{S/C + 2(K+O)C + (1-\alpha)KCB_{\mathrm{GPU}}/B_{\mathrm{PCIe}}}$$
  对 S=128K, C=8, K=256, O=48, α=0.6 计算得等效带宽 7.2 TB/s。

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

## X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression

- 属于Serving调度的实现是什么？实验比较什么？
  评估 X-EcoMLA 转换后的 MLA 模型在推理部署时的系统级吞吐和内存表现。实现是将预训练 Llama 模型通过 SVD 初始化 + 知识蒸馏 + DPO upcycle 为 MLA 版本后，在 AMD MI300 GPU 上测量推理性能。论文未修改开源 Serving 框架的调度逻辑，而是在标准推理框架上比较 baseline（MHA/GQA）与 X-EcoMLA (MLA) 在不同 batch size 下的吞吐（sequences/sec）和峰值 GPU 显存（GB）。

  实验比较（Figure 2）：(1) **吞吐对比**：Llama3.1-8B baseline vs X-EcoMLA-8B（r_kv=128, 10.67× KV 压缩），在 8× AMD MI300 上 batch size 从 1 到 1024，X-EcoMLA 实现 1.7× 到 2× 的吞吐提升；(2) **峰值显存对比**：batch size=128 时 Llama3.1-8B 消耗 143 GB 显存且无法运行更大 batch，X-EcoMLA-8B 仅需 28 GB（5× 内存减少），可平滑扩展到 batch size 1024 而不 OOM。

- 硬件平台是什么，配置是什么。
  推理评估硬件：8× AMD MI300 GPU（系统级吞吐/内存测试），single AMD MI300（吞吐测试）。训练硬件：8× AMD MI300 GPU（训练耗时约 70-140 GPU hours）。

- 开源Serving框架是什么。修改了什么。
  论文未明确说明使用哪个特定 Serving 框架进行推理评估。系统级推理性能测试可能基于 HuggingFace Transformers / PyTorch 原生推理或 AMD ROCm 生态下的推理后端。论文未对 Serving 框架进行调度逻辑修改——其 Serving 层面的收益完全来自 MLA 架构对 KV cache 内存的压缩，KV cache 从 2·n_h·d_h·l 降至 (r_kv + d_r)·l，从而释放了大量 GPU 显存，使得在相同硬件上可以支持更大的 batch size 和更高的吞吐。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源：https://github.com/AMD-AGI/AMD-Hybrid-Models

  Serving 全流程（以 X-EcoMLA-8B on AMD MI300 推理为例）：
  ```
  1. 输入: 客户端提交 batch_size=128 的推理请求序列
  2. 模型加载: HuggingFace Transformers 加载 X-EcoMLA-8B 权重
     - 模型已通过 SVD 初始化 + 蒸馏 + DPO 转换为 MLA 架构
     - W_UK 已吸收进 W_Q, W_UV 已吸收进 W_O（推理时无显式 up-projection）
  3. Prefill 阶段 (prompt 处理):
     a. 对输入 prompt tokens 逐层计算 MLA:
        C_KV = H @ W_DKV (down-proj to r_kv=128)
        K_C, V_C = C_KV @ W_UK, C_KV @ W_UV (up-proj，可吸收)
        K_R = RoPE(H @ W_KR)  (共享 RoPE key, d_r=32 dims)
     b. KV Cache 写入: 仅存储 C_KV[r_kv=128] + K_R[d_r=32] = 160 dims/token
        vs baseline 存储 2·n_h·d_h = 2·32·128 = 8192 dims/token
        压缩比 = 8192/160 = 51.2× per-token KV（实际约 10.67× 总压缩）
  4. Decode 阶段 (逐 token 生成):
     a. 新 token 的 hidden state H_new → 计算 C_KV_new, K_R_new
     b. 追加到 KV cache（每个新 token 仅 160 dims vs baseline 8192 dims）
     c. 从 cache 读取全部历史 KV、重建 K_C, V_C、计算 attention
     d. 因 KV cache 大幅缩小，batch_size=128 时 peak memory 仅 28 GB (vs 143 GB)
  5. 输出: 生成文本返回客户端
  ```
  核心收益：MLA 通过低秩压缩将 KV cache 内存需求从 O(2·n_h·d_h·l) 降至 O((r_kv + d_r)·l)。在 batch_size=128、Llama3.1-8B 场景下，baseline 因 143 GB 显存耗尽无法运行更大 batch，而 X-EcoMLA 在 28 GB 下可扩展到 batch_size=1024，达成 1.7-2× 吞吐提升。
