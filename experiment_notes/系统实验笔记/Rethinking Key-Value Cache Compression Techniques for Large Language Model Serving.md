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
