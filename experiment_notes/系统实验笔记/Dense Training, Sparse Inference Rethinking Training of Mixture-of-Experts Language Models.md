## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  论文将 DS-MoE-6B 模型部署到 vLLM 开源 serving 框架中进行吞吐量和延迟 benchmark，与 Mistral-7B、DeepSeekMoE-16B、Qwen1.5-MoE-A2.7B 进行比较。实验比较在相同 GPU 硬件（A100-80GB / H100-80GB）和相同场景（1000 input tokens + 1000 output tokens）下的 requests/sec（Throughput）和 tokens/sec（TPS）。论文未修改 vLLM 框架本身，而是利用 vLLM 的现有能力进行标准 serving 部署和 benchmark。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80GB GPU 和 H100-80GB GPU，GPU 内存利用率设置为 0.9。

- 开源Serving框架是什么。修改了什么。
  使用 vLLM (Kwon et al. 2023) 作为部署框架，未修改框架本身。另外使用 HuggingFace Transformers (Wolf et al. 2020) 进行 latency 和 input token throughput 测量。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文代码未开源。vLLM 部署流程（基于论文 Table 7 描述）：
  
  1. **模型加载**：将 DS-MoE-6B（6.5B total params, FP16, ~12.6 GiB GPU memory）加载到 vLLM 的 LLMEngine。vLLM 使用 PagedAttention 管理 KV cache 内存，0.9 GPU memory utilization。
  
  2. **请求输入**：continuous batching 接收请求流，每个请求 1000 input + 1000 output tokens。
  
  3. **Prefill 阶段**：vLLM 将 1000 input tokens 作为 batch 送入模型 forward pass。Self-attention 层使用 dense inference（torch.nn；因 DS-MoE 的 attention 层 sparsity < 40%，密集计算更快）。MLP 层使用 sparse inference，通过 SimpleMoE 的 ParallelLinear 实现 top-K expert 选择。针对 batch 中不同 token 可能选择不同 expert 的情况，使用 Threshold-TopK 策略（先统计平均激活 expert 数，再统一选 K 个）。
  
  4. **Decode 阶段**：autoregressive 生成 1000 output tokens，每步仅处理新 token + KV cache。KV cache 用 vLLM PagedAttention 管理（block-based 分页），保证 memory fragmentation 最低。
  
  5. **性能指标**：Throughput = requests completed per second。TPS = total tokens processed (input + output) per second。测量结果：A100-80GB 上 DS-MoE-6B 吞吐 2.00 req/s, TPS 3992.8；H100-80GB 上 2.30 req/s, TPS 4603.9。分别相比 Mistral-7B 加速 1.86× (A100) / 1.64× (H100)，相比 Qwen1.5-MoE-A2.7B 加速 1.50× (A100) / 1.27× (H100)。

  另外在 HuggingFace Transformers 上测量 latency 和 input token throughput：
  - Latency：batch=64 sentences, 2000 tokens each, generate 20 tokens。DS-MoE-3B: 3.68s (vs Dense-3B 4.28s, 1.16× speedup); DS-MoE-6B: 5.75s (vs Dense-6B 8.58s, 1.49× speedup)。
  - Input TPS：seq_len=256, max batch size fit in GPU memory。DS-MoE-3B: 61515.9 (vs Dense-3B 40854.5, 1.51×); DS-MoE-6B: 35046.7 (vs Dense-6B 18354.2, 1.91×)。

  6. **扩展模型 Serving 分析**：为模拟更大 scale 下的性能，扩展模型到 10B/14B/19B 级别（Dense-10B/14B/19B, SMoE-17B/25B/34B (2× MLP params), DS-MoE-10B/14B/19B）。在 computation-bounded 场景（prefill, input throughput）和 I/O-bounded 场景（decode, output throughput）分别测量。DS-MoE 在两个场景均优于 SMoE（因 DS-MoE total params 更少，GPU memory 占用更低，batch size 可以更大），在 computation-bounded 场景显著优于 Dense（因 active params 少，计算量少）。
