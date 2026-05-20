## PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

- 属于Serving调度的实现是什么？实验比较什么？
  PAT作为vLLM v0.9.0的off-the-shelf plugin集成到LLM serving框架中，修改attention backend以利用跨请求共享prefix减少decode attention延迟。核心Serving层面实现：(1) pack scheduler将vLLM的block table（每行是一个query的KV block IDs）转为prefix tree，internal node表示多query共享的prefix段，用memory-centric profit model决定split/merge策略生成CTA partition，最小化全局KV cache重复加载；(2) lazy update机制使scheduler仅在block table变化时重新调度（如request arrive/depart或新KV block分配），否则复用上次调度结果，并与pre-attention tasks（metadata preparation、QKV projection）异步重叠执行；(3) Python侧通过pybind11暴露kernel API，约1.2k行Python代码集成为vLLM后端。实验比较end-to-end serving指标：mean TTFT、mean TPOT、P99 TPOT，在两模型（Llama-3-8B, Qwen3-8B）和两个真实trace（ToolAgent: tool/agent workload含系统prompt共享, Conversation: 三层prefix结构）上，对比RelayAttention++、FlashAttention、FlashInfer三种baseline。PAT减少mean TPOT 17.0-93.1%，减少TTFT 9.3-99.8%。进一步在Qwen2.5-72B-Instruct (4×A100 TP=2/PP=2)上减少TPOT 14.3-26.7%，Qwen3-30B-A3B (MoE, 单卡A100)上减少5.53-16.9%。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB单卡（端到端online serving主测试）。NVIDIA H100-SXM4-80GB（kernel benchmark扩展）。分布式实验：4×A100 (Qwen2.5-72B-Instruct, TP=2 PP=2)。软件：CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.9.0修改，将PAT作为自定义attention backend集成。主要修改：(1) 新增pack scheduler模块，在每次decode step前读取vLLM paged KV cache的block table，构建prefix tree并生成CTA partition；(2) 替换vLLM默认的FlashAttention backend为PAT的pack-forward-merge pipeline；(3) pack scheduler运行在asynchronous CPU thread上与pre-attention tasks重叠；(4) 复用vLLM的paged KV cache机制（KV entries以fixed-size blocks在block table中管理），pack scheduler仅操作logical block IDs，不重写KV paging实现；(5) 启用方式：设置环境变量`VLLM_ATTENTION_BACKEND=PAT`。PAT不改动vLLM的continuous batching语义或request-level scheduling逻辑。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/flashserve/PAT（MIT License）。vLLM集成使用流程：
  1. 环境准备：x86-64 Linux, ≥64GB RAM, 200GB disk, A100-80GB, CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0
  2. 克隆仓库：`git clone https://github.com/flashserve/PAT.git`
  3. 启动vLLM server：`VLLM_ATTENTION_BACKEND=PAT python -m vllm.entrypoints.openai.api_server --model Qwen3-8B --gpu-memory-utilization 0.95`
  4. Pack scheduler在每次decode step从vLLM获取block table → 构建prefix tree → profit model决策CTA partition → lazy update复用上次结果直到block table变化 → 异步重叠scheduler和QKV projection等pre-attention任务
  5. PAT的pack scheduler平均latency比pre-attention task latency低42.3%-49.6%，因此异步执行时不增加end-to-end延迟
  PAT的作用：在vLLM的continuous batching框架下，通过prefix-aware attention kernel利用跨请求共享prefix，减少decode attention的global memory KV cache加载次数。例如在Conversation trace (Qwen3-8B, 8 req/s)下，PAT相比FlashAttention减少mean TPOT 89.5%、TTFT 99.6%；相比FlashInfer减少TPOT 93.1%、TTFT 99.8%。

