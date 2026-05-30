## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 METRO 在 vLLM Serving 框架中的集成，核心修改是 EP (Expert Parallelism) dispatch 通信模式的替换：(1) 将传统 all-to-all dispatch 替换为 all-gather dispatch——每个 GPU 在 MoE expert 层计算前，先将本地 tokens all-gather 到所有 GPU，使每个 GPU 获得全局 token 集合；(2) 每个 GPU 基于全局 token 集合独立执行 top-k 计算，构建全局 T[1..N]（每个 expert 的总 token 数）；(3) 执行 METRO greedy routing（Algorithm 1）决定每个 expert 在哪个 GPU 上激活；(4) 仅计算分配给本 GPU 的 expert FFN；(5) all-to-all combine 将输出返回原 GPU。该修改的动机是：传统 all-to-all dispatch 只让各 GPU 知道本地 top-k 结果，无法获得全局 expert token 分布信息，而 METRO 的 MIN-EXP-ROUTING 算法需要全局 top-k knowledge (T[1..N]) 才能做 informed routing 决策。此外 METRO 集成 vLLM 的 CUDA Graph compilation framework，将 decode phase 的路由逻辑编译进 power-of-two batch sizes（up to 32 tokens per GPU）的 CUDA Graphs，消除额外 kernel launch overhead。

  实验比较：(a) METRO vs EPLB token routing 在 vLLM (8×A100) 上的 decode latency (TPOT) 和 total token throughput；(b) METRO vs EPLB 在 decode-heavy（InstructCoder, NuminaMath, Humaneval）和 prefill-heavy（GSM8K）workloads 下的吞吐影响差异；(c) METRO all-gather vs EPLB all-to-all 的通信时间对比；(d) 不同 replication ratio（1.0x, 1.125x, 1.25x, 1.5x）下 METRO 的性能增益变化；(e) decode throughput-latency Pareto 分析，变 batch size (64–1024) 和 parallelism (TP1-16 × EP1-16) 组合。

- 硬件平台是什么，配置是什么。
  真实系统：Google Cloud a2-highgpu-8g VM，8×NVIDIA A100 40GB GPU，600 GB/s NVLink（全部 GPU 在同一 NVLink domain）。batch size 限制：decode phase 最多 32 tokens/GPU，prefill phase 最多 32 prompts/GPU，context length 8K。模拟器：专有工业级 multi-GPU performance simulator，8×B200 192GB (Qwen3-235B) 和 16×B200 192GB (DeepSeek-V3)，900 GB/s NVLink。模拟器 configs: global decode batch size 1K, chunked prefill limited to 8K, sequence length 1K input + 2K output。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：**vLLM**（https://github.com/vllm-project/vllm）。论文在 vLLM 中做了以下修改/新增：
  (a) **EP dispatch 通信模式替换**：将 MoE expert layer 的 dispatch 阶段从 all-to-all 改为 all-gather，使得每个 GPU 在 top-k 前获得全局 token 集合；
  (b) **METRO routing kernel 集成**：实现 Algorithm 1 的 CUDA kernel，运行在单个 SM 上，使用 test-and-set lock 和 SM-local shared memory；
  (c) **CUDA Graph 集成**：利用 vLLM compilation framework 将 METRO routing 编译进 decode phase CUDA Graphs，预编译 power-of-two batch sizes 的图（up to 32 tokens/GPU），非 power-of-two batch 通过 padding 复用；
  (d) **EPLB placement/replication 保留**：METRO 仅替换 token routing 部分，不修改 EPLB 的 expert placement 和 replication 策略，避免干扰 prefill phase 性能；
  (e) **METRO 仅应用于 decode phase**：prefill phase 继续使用 EPLB token routing，因为 prefill 是 compute-bound 的。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文作者包含 NVIDIA 团队，**论文未明确提供 METRO 的独立开源代码仓库**。实现基于开源 vLLM 框架。以下是 METRO 在 vLLM 中的完整推理流程：

  ```
  === METRO + vLLM EP MoE Serving 全流程 ===

  Input: batch of requests with prompts, model distributed across G GPUs via EP

  === Prefill Phase (使用 EPLB token routing, compute-bound) ===
  For each transformer layer:
    Step 1 Attention: 数据并行 (DP)，每个 GPU 计算本地 tokens 的 attention
    Step 2 MoE Gating: 每个 GPU 独立对本地 tokens 计算 router top-k
    Step 3 EPLB Token Routing: 将每个 expert 的 token 均匀分配到其 replicas 上
    Step 4 All-to-all Dispatch: tokens 根据路由决策发送到目标 GPU
    Step 5 Expert FFN: 每个 GPU 计算分配给自己的 expert FFN
    Step 6 All-to-all Combine: expert 输出 embedding 返回原 GPU

  === Decode Phase (使用 METRO routing, memory-bound) ===
  For each transformer layer:
    Step 1 Attention: compute attention on local tokens (DP)
    Step 2 All-gather Tokens: 每个 GPU 将本地 tokens all-gather 到所有 GPU
        替换传统 all-to-all！
        通信量: 2MB/GPU (32 tokens * hidden_dim on 8 GPUs, fp16)
        NVLink 带宽开销: ~3us (on 600 GB/s)
        NCCL launch fixed cost: ~100us -> 带宽开销远低于固定开销
    Step 3 Global Top-K: 每个 GPU 在全局 (~256 tokens) 上计算 top-k
        冗余计算开销: <3us (<1% 层时间)
        -> 构建 T[1..N]: 每个 expert 在全局 batch 中的 token 数
    Step 4 METRO Routing (CUDA kernel, 单 SM):
        执行 Algorithm 1——greedy assign each expert to GPU with fewest activated experts
        开销: 最多 26us (1.5x replication)
    Step 5 Expert FFN: 每个 GPU 仅计算分配给自己的 expert FFN
        仅在激活的 expert replicas 上计算 -> 减少内存流量
        FFN 时间减少: 最多 81us (1.5x replication)
    Step 6 All-to-all Combine: expert 输出 embedding 返回原 GPU
    Step 7 Layer output: attention output + MoE output combined
  ```

  关键性能收益：(a) METRO 将 activated experts 数量减少 up to 42.3% vs EPLB routing；(b) decode latency 降低 11%-22%；(c) total token throughput 提升 3%-21%（co-deployed prefill+decode）；(d) 在 decode-heavy workloads 上增益更显著（up to 21%），prefill-heavy 上仍有 4.2% 提升；(e) 在固定 SLO 下 decode throughput 可达 EPLB 的 1.98x-4.11x。
