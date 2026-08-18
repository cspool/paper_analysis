## vLLM（高吞吐 LLM 推理服务引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
vLLM 是 UC Berkeley 主导的开源 LLM 推理与服务引擎（Apache-2.0，github.com/vllm-project/vllm），datacenter 高吞吐在线推理的事实标准之一。核心创新 PagedAttention：把 KV cache 按固定大小 block（page）分页管理、按需分配，消除预分配连续 KV 空间带来的碎片化与显存浪费，显著提升可并发请求数；配合 continuous batching（iteration-level 调度，新请求无需等整批完成即可插入）与 prefix caching（共享前缀 KV 复用）进一步提升吞吐。另支持张量/流水线并行、量化（AWQ/GPTQ/FP8/INT8 W8A8）、投机解码（EAGLE/Medusa/ngram）、多 LoRA。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
request → AsyncLLMEngine → Scheduler（continuous batching：每 iteration 选请求组）
       → ModelRunner：prefill/decode 统一走 PagedAttention
       → KV cache 按 block 分配/释放，block table 映射逻辑块→物理块（虚拟内存思想）
       → INT8 路径：per-token 动态激活量化 + per-channel 权重 scale（W8A8）
```
本文用法：Cassandra 仅把 vLLM 的官方 INT8 量化实现作为 SmoothQuant W8A8 GPU baseline（论文 VI-C）：实测低 batch decode 下 W8A8 相对 BF16 仅约 1.3×——decode 阶段 GEMM 不是瓶颈，在线激活量化与 scale 乘加的开销无法被 GEMM 收益掩盖（与文献报道的 INT8/FP8 decode 1.25–1.42× 一致）；而 Cassandra 达 1.78–2.41×。即本文不修改任何 serving 框架，vLLM 是评估对照物而非实现载体。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
部署：pip install vllm 或 Docker 镜像，OpenAI 兼容 API 直接替换前端服务；生产使用于大规模在线推理（LMSYS Chatbot Arena 等）。量化实现：W8A8 用 per-token activation + per-channel weight 动态量化并把 quant/dequant kernel 融合进 GEMM；FP8 走 CUTLASS/MarLin kernel。局限：设计目标为高吞吐 datacenter 多请求场景，边缘低 batch 单用户不是其优化点；其量化路径在 decode 阶段的收益有限（如上）。

ConServe 补充视角（ISCA'26）：ConServe 以 vLLM 为 PagedAttention baseline——block=16 tokens（该环境 kernel 开销最小，与 vAttention 报告一致），指出其两层翻译（block table + 硬件页表）与散页 gather 在 multi-turn 下使翻译局部性成为关键瓶颈（vLLM 自报 PagedAttention 比非分页 FasterTransformer kernel 慢 20–26%）；ConServe 用 conversation 级连续 VA slice 替换之。结果：TTFT −64.1%~−74.4%、decode 吞吐 +19.4%~25.6%、离线端到端吞吐 +17.7%~35.1%、SLO attainment +11%~19%（增益随模型 KV footprint 与 batch 增大）。

  - SHyLA 补充：SHyLA 把 vLLM（连续 batching + PagedAttention）作为 GPU baseline 评估组件：8× NVIDIA H800（HBM2e 80GB/2000GB/s、756 FP16 TFLOPS）与 4× AMD MI300X（HBM3 192GB/5300GB/s、1307.4 FP16 TFLOPS）集群，batch 由 vLLM 运行时动态决定；SHyLA 的 DSA 结果与之对比（系统 token 吞吐 geomean 2.72× over H800、1.59× over MI300X）。论文指出 vLLM/DeepSpeed 面向单体内存 GPU 架构，无法直接移植到 SHyLA 的混合内存（roadmap 提及混合内存感知内存管理、LLM 数据分类、plane-aware 调度）。

- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，MLLM baseline）：vLLM 作为 text-only 强 baseline 对照——其把 vision encoder 直接与 LLM 共置、静态并行度/资源分配，在混合分辨率负载下 encoder 进入 prefill 关键路径导致 TTFT/TPOT 恶化、GPU 利用率低与 SLO 违例（Figure 3/9：vLLM 最差，Kimi-VL-16B@10RPS mean TTFT 59.7s、P99 104.8s，Qwen2-VL-7B 吞吐 257 tokens/s）。RESONATOR 对比显示 vLLM 因无 encoder-aware 调度与动态并行而在 MLLM 负载下显著落后（E2E 最高差 10.7×）。
Understanding Inference Scaling 补充视角（ISCA'26，vLLM v1 作为被测 Serving 系统）：论文用 vLLM v1 + PagedAttention（block size B=16）作为被测引擎，不改代码只调 max_num_batched_tokens 与 max_num_seqs 两个调度参数，量化 reasoning 负载下调度器的系统行为：(1) 并发-容量权衡——max_num_seqs 从 1K 扫到 10K，10K 下 KV 数分钟冲 100%、调度器进入 thrashing（Running→Waiting 抢占 + prefill 重算），吞吐增益崩塌（Capacity Trap）；E2E 延迟呈凸曲线、≈2K 并发为最优甜点；(2) DP=8 集群（8×H200）batch 500→5000 时 E2E 从 61s→165s 亚线性增长、KV 满载触发请求限流——DP 不池化内存、每卡仍是"孤岛"；(3) 5K batch 下激活 Chunked Prefill 进入 convoy 模式（Running 曲线平台化、请求串行准入），GPU 高占用但实际在"内存容量管理"上 stall。论文把 vLLM 调度器暴露为"内存流量整形"系统：HBM 容量约束可持续吞吐、带宽约束每 token 延迟。
涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

Tetris 补充视角（ISCA'26，vLLM 作为 CDSP 控制面/调度器扩展的宿主）：Tetris 把 vLLM 作为控制面与推理后端基础——global manager + 每实例 local manager（Python，Ray 通信），在其 scheduler 上扩展三个 CDSP 接口：initialize_schedule（记录元数据、初始化各 SP 延迟模型与初始 improvement rate）、update_schedule（由 FastAPI POST /update 触发，按观测到达率从离线 simulator 的最优 rate 映射刷新 improvement rate）、cdsp_schedule（对到达 prefill 请求调用 Algorithm 1 生成 CDSP 执行计划并构造 per-instance 元数据转发 local managers）；CDSP scheduler 本体用 C++ 写（消除调度延迟）。推理后端复用 vLLM 部分组件并叠加 PyTorch/Triton-distributed、Flash Attention zigzag 扩展、Flash Decoding、CUDAGraph、NVSHMEM/NCCL。这展示 vLLM 作为 serving 框架的可扩展宿主：调度策略与并行执行引擎均可插件式替换（同 LoongServe/Shift Parallelism 的做法）。注意 Tetris 论文未开源，vLLM 本身开源（github.com/vllm-project/vllm）。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
