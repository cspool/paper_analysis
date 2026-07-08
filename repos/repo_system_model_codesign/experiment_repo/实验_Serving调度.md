# 实验_Serving调度

## Using Span Queries to Optimize Cache and Attention Locality

- 属于Serving调度的实现是什么？实验比较什么？
  实现是在 vLLM 中引入 Span Query 接口和优化引擎，仅需修改 **492 行代码**（跨 7 个文件），将 vLLM 从仅支持线性 token 序列（Chat Completion）扩展为支持树状 token 结构（Span Query）。核心修改包括：(1) **RoPE on read, not on write**：将 RoPE 位置编码的施加时机从 KV cache 写入时移到读取时，使同一物理 page 可被不同上下文的请求以不同位置编码复用；(2) **Prefix scan hash chaining 选择性禁用**：在 spanned region 的边界 block 上暂停 hash accumulation，使 span 内部的 KV cache 布局与位置顺序解耦（commutativity），跨 span 边界恢复 hash chaining 保持 prefix 依赖。Span Query 以 expression tree 形式提交（含 commutativity constraints），vLLM 端的 parser/planner/optimizer 自动优化 KV cache 布局和 attention 计算。
  实验比较：(1) Span Query vs Stock vLLM（无 span query 支持）的 TTFT（Time-to-First-Token）；(2) Span query stack cache hit vs cache miss 的 TTFT；(3) Attention-optimized span query（2B 模型）vs stock inference server（8B 模型）的 needle-in-haystack accuracy；(4) Span table implementation vs cropping implementation 的 accuracy（MSMARCO, HotpotQA）。

- 硬件平台是什么，配置是什么。
  论文未明确说明 GPU 型号和数量。实验使用 granite3.3 模型（2B 和 8B 参数）。论文提及 vLLM 作为基础推理框架（约 260,000 行 Python 源码），SPNL 库以 Rust 编写（93.8% Rust + Shell + Common Lisp）。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：vLLM（https://github.com/vllm-project/vllm）。修改仅 **492 行**，跨 7 个文件：
  | 文件 | 修改行数 |
  |------|---------|
  | `core/block_pool.py` | 68 |
  | `core/kv_cache_manager.py` | 89 |
  | `core/kv_cache_utils.py` | 73 |
  | `core/sched/output.py` | 4 |
  | `core/sched/scheduler.py` | 19 |
  | `core/single_type_kv_cache_manager.py` | 4 |
  | `worker/gpu_model_runner.py` | 232 |
  
  核心修改逻辑：
  1. **RoPE on read**：将 `worker/gpu_model_runner.py` 中 KV cache 写入时的 RoPE 应用移至读取时。允许同一物理 page 被不同 span 上下文以不同 position offset 读取，是实现 commutativity（page 可跨请求复用）的基础。
  2. **Prefix scan hash chaining 选择性暂停/恢复**：修改 `core/kv_cache_manager.py` 和 `core/block_pool.py` 中的 block hash 计算逻辑——若 block 以特殊 token `(`（子表达式开始）开头，暂停 hash accumulation；若以 `)`n`（子表达式结束+位置指针）开头，恢复 accumulation。这使得 span 内部的 KV cache 与位置顺序解耦，跨 span 边界保持 prefix 依赖。
  3. **Span-aware scheduler**：修改 `core/sched/scheduler.py`（19 行），使 scheduler 理解 span query 的 expression tree 结构，按树遍历顺序而非线性顺序调度 compute。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源代码：https://github.com/IBM/spnl（SPNL library），提供 Rust crate（crates.io）和 Python package（PyPI），CLI 工具 `spnl`（支持 Homebrew/Docker），Docker image 包装 Ollama 用于本地测试。
  
  Span Query Serving 全过程（以 RAG use case 为例）：
  
  1. **输入**：Client 构造 Span Query expression tree。RAG 场景：`R(← query, ++ docs)`——表示"query 与每个 document 配对，docs 之间相互独立（commutative）"。Span Query 是 expression tree of inference calls，linked with commutativity constraints。
  
  2. **Span Query 解析与规划**（vLLM 端 `spnl` parser）：解析 expression tree → 识别 commutative sub-trees（docs 之间 order 无关）→ 生成 span table（物理 page → 虚拟位置映射表）。
  
  3. **三层优化栈**：
     - **High-level optimizer**（Section 5.1）：固定点树重写（fixed-point tree rewriting），四条规则——desugaring C→G+⨝（chat 展开为 generator+join）、desugaring R→++F（RAG 展开为 concat+fold）、simplification（合并 commutativity hints 链）、"plus distribution"（将 commutative join 分布到 token generation 以解决 dual-output paradox——模型服务向 client 输出和向 KV cache 写入不同的内容）。
     - **Query tokenization**（Section 5.2）：将 span query 编码为 token 序列。使用特殊 token：□（padding）、`(`（子表达式开始）、`)(`（兄弟边界）、`)n`（子表达式结束+位置指针，n 指向该序列化子表达式开始位置）。`)n` 的设计避免了 vLLM 维护解析栈——vLLM 可直接从 token 序列中定位子表达式位置。
     - **Low-level optimizer**（Section 5.3）：(a) Block alignment——将特殊 token 对齐到 block 边界，仅每 block 首 token 需扫描；(b) Trailing partial blocks——crop inner generate 输出中会导致 prefix-scan cache miss 的尾部 token（vLLM 不缓存 partial blocks）。
  
  4. **KV Cache 管理**（vLLM 修改生效）：
     - 每个 span 内部 pages 按 `spnl` tokenization 写入 KV cache。
     - Block hash chaining 在 span 边界暂停（`(` 开头 block）→ span 内 pages 可任意顺序排列（commutativity）。
     - 跨 span 边界恢复 hash chaining（`)n` 开头 block）。
     - RoPE 在 KV cache 读取时应用（而非写入时），使同一 page 可被不同 span 以不同 position 复用。
  
  5. **CIDRA 算法**（Section 5.5.1，Concurrent In-place Duplicating ReROPE Algorithm）：
     - 当多个并发请求复用同一 block 但需要不同 position offset 时，构建 block repositioning 依赖图。
     - SCC 分析识别循环依赖和独立子图。
     - 出度 >1 的 block 被 duplicate（并发请求将同一 block reposition 到不同位置）。
     - GPU bin packing 并行执行；大循环回退到 CPU（罕见——循环通常 size 2）。
     - 小 batch 时 concatenate layers。
     - 最大吞吐：**500 tokens/ms**。
  
  6. **Attention 计算**：
     - Span query 使 token 仅在其所属 document 内做 attention（sparser attention pattern）。
     - 即使 cache miss，span queries 也比 stock vLLM 更快（因为 attention 更稀疏）。
  
  7. **输出**：vLLM 生成 token 序列 → detokenize → 返回 HTTP response。
  
  作用：将 inference serving 从"线性 token 序列"范式升级为"expression tree + commutativity"范式，使 KV cache 复用从 prefix-only 扩展到任意 commutative sub-trees。RAG 场景下 TTFT 加速 10-20×，Judge-Generator 场景下 fan-out 24 时 TTFT 加速 12-13×。仅需 492 行 vLLM 修改，证明空间换时间的 cache 复用策略可通过简单的 API 变更实现数量级收益。

## Sieve

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 Sieve Scheduler（Section 5）和 Sieve System（Section 6）两个组件构成的运行时框架，用于在配备 HBM-PIM 的多 GPU 系统上动态调度 MoE 模型的专家计算。Sieve Scheduler 在每次迭代中根据运行时 token-to-expert 分布计算每个 expert 的算术强度（arithmetic intensity），将低算术强度的 memory-bound unpopular experts 分配到 PIM 执行，高算术强度的 compute-bound popular experts 分配到 GPU 执行。Sieve System 通过依赖图（DAG）协调跨 GPU 和 HBM-PIM 的操作，重叠 GPU 计算、PIM 计算、inter-GPU 通信和 intra-device 数据传输。
  实验比较：Sieve vs NoExp（仅 attention 上 PIM，所有 expert 在 GPU）、AllExp（所有 expert 在 PIM）、PIMoE（静态 threshold 分配 expert）。评估指标为 throughput（tokens/s per GPU）和 interactivity（tokens/s per user），以 Pareto 曲线呈现。

- 硬件平台是什么，配置是什么。
  模拟的 NVIDIA B200 GPU：FP16 吞吐 2,250 TFLOPS，HBM-PIM 带宽 8.0 TB/s，8 个 HBM-PIM stack，共 96 GB（PIM PU 牺牲 50% 容量），NVLink 带宽 900 GB/s（单向），延迟 0.8 μs。HBM3E 时序参数：tRCD=28, tRP=28, tRAS=68, tRC=96, tCL=28, tWR=32, tCCD_S=2, tCCD_L=4, tRRD_S=6, tRRD_L=6, tFAW=12, tREFI=3,900 ns, tRFC=400 ns（@8.0 Gbps pin rate, tCK≈0.50 ns）。HBM-PIM 配置：pseudo-channels/stack=32, banks/pseudo-channel=24, page size=1 KB, compute density=1 op/byte。多 GPU 配置：GPT-OSS-120B 使用 4×B200 GPU，Qwen3.5-397B-A17B 使用 8×B200 GPU，Qwen3-30B-A3B 使用 1×B200 GPU，每个 GPU 均附带自己的 HBM-PIM stack。

- 开源Serving框架是什么。修改了什么。
  论文未基于现有开源 Serving 框架（如 vLLM、SGLang）修改。Sieve 是一个独立设计的运行时框架，从零构建调度器和执行协调逻辑。其核心修改理念为：
  1. **Sieve Scheduler（Section 5）**：替代现有框架中静态的 expert placement 策略。目标函数为 S* = argmin max(T_Comm, T_GPU(G), T_PIM(S))，使用贪心启发式算法——按 token count 降序排列所有 activated experts，从全部 assign 到 PIM 开始，逐步将最 popular expert 移到 GPU，直到 T_total 不再下降。overhead 约 20 μs（B200 GPU 上，未做 kernel 优化）。
  2. **Sieve System（Section 6）**：构建 MoE layer 的依赖图（DAG），协调 routing（gate）、AllGather token-to-expert mapping、metadata processing、dispatch（token 跨 GPU 分发）和 expert execution。PIM 侧使用 tensor parallelism 跨 PIM channels（而非 expert parallelism），GPU 侧使用 grouped GEMM 执行 popular experts。
  3. **PIM 命令控制**：通过自定义 GPU kernel 动态初始化 PIM 操作，使用 PIM_GWRITE 和 PIM_GEMV 命令（遵循 NeuPIMs 接口规范），将 skinny GEMM（multi-token expert）转换为串行 GEMV 操作序列。
  4. 不需要硬件修改：无新 PIM 命令，无架构改动。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  Sieve 自身代码未开源（论文未提供仓库链接）。模拟基础设施基于 Ramulator 2.0（开源：https://github.com/CMU-SAFARI/ramulator2）和 Duplex（用于 GPU 性能模型和 PIM 命令级时序仿真）。
  Sieve 的端到端 serving 全过程（以单次 MoE layer decode 为例）：
  1. **输入**：Attention 输出（token hidden states）+ 模型配置（expert 数量、维度、并行策略）+ 硬件配置（GPU/PIM 规格）。
  2. **Router（Gate）计算**（¹→²）：每个 GPU 上，gating network 对 local tokens 计算 token-to-expert 路由分数，选出 top-k experts per token。
  3. **AllGather 全局路由表**（³）：由于 expert parallelism 跨 GPU，各 GPU 通过 AllGather 交换 local routing maps，每个 GPU 获得全局 token-to-expert 映射，识别哪些 token 需要发往自己负责的 experts。
  4. **Metadata 预处理**（⁴）：每个 GPU 为本地 expert 准备固定大小的 tensor buffer（grouped GEMM 所需）。
  5. **Dispatch + Sieve Scheduling**（⁵ Dispatch 并行 ⁵ Sieve）：(a) tokens 跨 GPU dispatch——每个 token 发送到其 assigned expert 所在的 GPU；(b) 同时每个 GPU 执行 Sieve Scheduler 贪心算法，基于 runtime token counts 决定哪些 experts 在 GPU 执行、哪些在 PIM 执行。
  6. **GPU 路径**（popular experts）：HBM-PIM→GPU 加载 expert 参数（⁶），GPU 上用 grouped GEMM 批量执行（⁷），结果写入 GPU on-chip memory。
  7. **PIM 路径**（unpopular experts）：GPU 发送 token 到 HBM-PIM（⁶ GPU→HBM-PIM），PIM 执行 GEMV：broadcast token 向量到所有 PIM channels，每个 channel 用 adder tree 做 dot product（⁷ HBM-PIM），通过 tensor parallelism 均匀利用所有 PIM channels，结果回读 GPU on-chip memory（⁸）。
  8. **聚合**（⁹）：GPU 将 expert-grouped 结果重新排列为 token-grouped 结果，计算每个 token 的 expert 输出加权和（weighted sum per token），得到 MoE layer 最终输出。
  9. **Inter-GPU 通信**：dispatch token 和 combine result 的 all-to-all 通信开销由 T_Comm 建模，Sieve Scheduler 在分配决策时已将通信开销纳入目标函数。
  作用：在 PIM-enabled 多 GPU 系统上服务现代稀疏 MoE 模型（如 GPT-OSS、Qwen3.5），动态适应 bimodal expert distribution，避免静态 offloading 的 PIM 利用不足或 GPU 空闲问题。在 Qwen3.5 上 batch size 256 时比 PIMoE 提升 1.26× throughput 和 interactivity，在 GPT-OSS 上 B≥32 时提升 1.11×–1.17× throughput，Qwen3 单 GPU 场景提升 1.6×。Sieve 是唯一在满载高吞吐下仍严格满足 interactivity SLA 的方案。

## Inference Time Context Sparsity

- 属于Serving调度的实现是什么？实验比较什么？
  实现是将稀疏注意力（oracle top-k、vAttention 随机索引、Double Sparsity 索引器）集成到 vLLM Serving 框架中，通过 sparse-attention-hub 后端支持 per-query、per-head 的 token 级别稀疏 decode。在 SWE-Bench Django agentic 场景下，比较 dense（100%）、5× 稀疏（~22% density）和 50× 稀疏（~3.8% density）三种配置的 agent 任务完成率、平均 turn 数和 token 消耗。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 80GB HBM3，FP16 精度。vLLM 服务端（论文未明确说明 GPU 数量，但 kernel benchmark 在单 H100 上完成）。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架为 vLLM。修改为：通过 sparse-attention-hub 后端替换标准 dense attention 的 KV cache 访问路径。具体地，在 decode 阶段，每个 query head 不再读取完整 KV cache，而是通过稀疏索引器（如 Double Sparsity 的 8 通道 16-bit 精度索引器，或 oracle top-k）选择 k 个 token 参与 attention 计算。Sink token（前 128 个）+ local window（128 token）+ top-k 稀疏选择组成混合 mask。使用 mini-swe-agent v2.2.8 harness（step_limit=250, cost_limit=$3, 60s per-command timeout）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源代码：https://github.com/skylight-org/sparse-attention-hub（Apache 2.0）。Serving 全过程：
  1. **输入**：SWE-Bench Django 的 issue 文本 + 代码仓库 snapshot → 构造 agent prompt。
  2. **vLLM 接收请求**：每轮 agent 调用 LLM 生成下一步 tool invocation，历史对话持续累积为 context。
  3. **稀疏索引器选 token**：在 decode 每步，query 向量通过索引器（如 Double Sparsity 使用 8×16-bit 量化通道计算 query-key 近似分数）选出 top-k 个 KV cache token。
  4. **稀疏 attention 计算**：仅对选中的 k 个 token 执行 scaled dot-product attention（kernel 在 FlashInfer 之上实现，使用 paged KV-cache、NHD layout）。
  5. **生成输出**：LLM 生成 tool call → harness 执行 → 输出追加到下一轮 prompt → 循环直至 agent 提交 patch 或达到 250 turn 上限。
  6. **评估**：harness 判断 patch 是否通过 fail-to-pass 测试（resolved/unresolved/empty_patch/error）。

  作用：验证在真实 agentic 工作负载下，Serving 层集成的稀疏 decode 是否保持任务完成质量。结果显示在严格子集（n=58，三配置均产出有效 patch）上，dense 77.6%、5× 79.3%、50× 75.9%，稀疏匹配 dense 在 ~2pp 以内。稀疏 agent 消耗更少 turn（67→57→55）和更少 token（1.34M→1.14M→1.08M）。

## MAC-Attention: Match-Amend-Complete Attention for Efficient Long-Context Inference

- 属于Serving调度的实现是什么？实验比较什么？
  实现是通过 runtime hook 将 MAC-Attention 集成到 SGLang Serving 框架中，替换 decode 阶段的 attention 计算路径。SGLang 保持对模型执行、请求调度、paged KV 分配和 FlashInfer 后端的控制；MAC-Attention hook 负责 query state 保存、ring cache 维护、拦截 BF16 paged-KV decode 调用并启动 fused kernel。
  实验比较：(1) MAC-Attention vs Full Attention（SGLang + FlashInfer）在 64K–256K context 下的端到端 attention 延迟；(2) MAC-Attention vs Quest、RocketKV、Multipole 在相同 KV 访问比例下的质量和延迟。

- 硬件平台是什么，配置是什么。
  NVIDIA Hopper GPU（H100 级别），BF16 精度。CUDA 13.0。SGLang 服务端 + MAC-Attention wrapper。MAC_DISABLE_CUDA_GRAPH=1（benchmark 配置），radix cache 禁用，page size=1。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：SGLang（https://github.com/sgl-project/sglang.git）。**零源码修改**——MAC-Attention 使用 `mac_attention.integrations.sglang` 中的 runtime hook 注入，无需 patch SGLang 源码。具体 hook 点：
  (1) **query state preservation**：在 decode 前保存 pre-RoPE query；
  (2) **ring cache maintenance**：维护大小为 κ（默认 512）的滑动窗口 Q cache 和 A cache；
  (3) **decode call interception**：拦截 BF16 paged-KV decode 调用，启动 `mac_persistent_decode_bf16` fused kernel 替代 FlashInfer 路径；
  (4) **cache writeback**：将新计算的 A_n 和 Q_n 写入 ring cache。
  fallback 机制：当 MAC 禁用或请求超出支持配置时，回退到标准 SGLang/FlashInfer 路径。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源代码：https://github.com/YJHMITWEB/MAC-Attention.git（MLSys 2026，ACM AE Badge）。Serving 全过程：
  1. **环境配置**：克隆 SGLang 和 MAC-Attention 仓库，pip install -e 两者，安装 flashinfer-python。source env_mac_portable.sh 加载默认环境变量（MAC_THRESHOLD=0.45, MAC_LOOKBACK_TOKENS_LEFT=512, MAC_GEN_MIN_LIMIT=2048）。
  2. **启动服务**：通过 `run_sglang_mac_server.sh` 包装器启动 SGLang，内部调用 `mac_attention.integrations.sglang.launch_server`。参数：`--host 0.0.0.0`，FlashInfer 后端，CUDA graphs 禁用。
  3. **请求处理**：SGLang 接收 HTTP 请求 → tokenize → prefix caching（radix cache 禁用时跳过）→ 调度请求到 GPU。
  4. **Prefill 阶段**：标准 SGLang + FlashInfer prefill，MAC-Attention hook 保存 prefill 产生的 initial Q/A cache entries。
  5. **Decode 阶段（MAC hook 生效）**：每个 decode step：
     (a) SGLang 准备好当前 token 的 Q 向量；
     (b) MAC hook 保存 pre-RoPE Q；
     (c) MAC hook 启动 `mac_persistent_decode_bf16` fused kernel：kernel 内部执行 Match（L2 距离搜索）→ 命中的 head 走 Amend+Complete（rectification band + tail fusion），未命中的 head 走标准 full attention → 返回 attention output；
     (d) MAC hook 将 Q_n 和 A_n 写入 ring cache；
     (e) SGLang 继续后续 FFN 层。
  6. **输出**：生成 token 序列 → detokenize → 返回 HTTP response。
  7. **性能**：MAC-Attention 在 120K context 下 attention 延迟 62.9μs（1% KV）、64.0μs（5% KV），vs Full Attention 234.2μs；在 128K context 下 end-to-end token 生成延迟减少 60%+，attention-phase speedup 14.3× 以上（256K 达 ~46×）。

  作用：在不修改 SGLang 源码的前提下，将 MAC-Attention 的常数复杂度 decode 路径注入生产级 Serving 框架，实现长上下文 serving 的显著加速同时保持 full-attention 质量。

## MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings

- 属于Serving调度的实现是什么？实验比较什么？
  实现是将 MixLLM 的混合精度量化（W4A8/W4.4A8/W8A8）集成到 vLLM v0.9.0 Serving 框架中。通过 `vllm_v0.9.0_patch/` 目录下的补丁修改 vLLM 的量化权重加载路径和 GEMM dispatch 逻辑，使其支持 MixLLM 的 output channel-wise 混合精度格式（部分通道 INT8 + 部分通道 INT4）。补丁通过 `apply_vllm_patche.sh` 脚本应用到 vLLM 源码。
  实验比较：(1) MixLLM vLLM 集成 vs float16 vLLM baseline 的端到端吞吐量（不同 bit-width 配置）；(2) MixLLM vLLM 集成 vs TRT-LLM W4A16 的吞吐量对比；(3) GSM8K benchmark 上的推理精度验证（`vllm/run_gsm8k.sh`）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU，CUDA 12.1。Docker 部署（`--runtime nvidia --gpus all`），共享内存 `--ipc=host` 用于 LLM serving 的大共享内存需求。单 A100 ≤8B 模型；多 GPU（4×A100）用于 70B 模型。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：vLLM v0.9.0（https://github.com/vllm-project/vllm）。通过 `vllm_v0.9.0_patch/` 补丁修改：
  1. **量化权重格式支持**：vLLM 原生仅支持 uniform bit-width 量化（如全 INT4 或全 INT8），补丁增加对 per-channel mixed-precision 格式的解析——在模型加载时识别哪些 output channel 是 INT8（~10%）、哪些是 INT4（~90%）。
  2. **GEMM Dispatch 修改**：在推理时，根据当前 channel 的精度类型 dispatch 到对应的 CUDA kernel——INT8 通道走 INT8 Tensor Core MMA 路径、INT4 通道走 two-step dequant + INT8 Tensor Core 路径。
  3. **Epilogue Fusion**：将 INT4 和 INT8 两路 kernel 输出的 scatter 融合为单次 CUDA Graph 执行，消除中间 buffer 的 kernel launch 和 HBM 往返。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源代码：https://github.com/microsoft/MixLLM（MIT License），vLLM 集成子模块在 `vllm/` 目录。Serving 全过程：
  1. **环境搭建**：克隆仓库 `--recursive`（包含 vLLM 子模块 commit 5fbbfe9），`pip install -r requirements.txt && pip install -e .`。运行 `./apply_vllm_patche.sh` 将补丁应用到 vLLM v0.9.0，然后 `cd vllm && pip install -e .`（约 10 分钟）。或直接用 Dockerfile 构建容器。
  2. **模型量化**：先通过 `mixllm/evaluation/run.sh` 校准+量化 pipeline 生成 MixLLM 格式的量化权重（含 per-channel 精度标记、per-group scales/zeros）。输出权重文件包含 W4A8/W4.4A8/W8A8 等多种配置。
  3. **启动服务**：在 vLLM 中加载 MixLLM 量化模型——vLLM 的权重加载器通过补丁识别混合精度格式，建立 per-channel precision map。模型加载到 GPU 显存（INT4 packed 权重 + INT8 通道单独存储）。
  4. **请求处理**：vLLM 接收 HTTP 请求 → tokenize → scheduler 分配请求到 GPU → 每个 decode step：
     (a) vLLM 选出当前 batch 的 token；
     (b) Linear 层的 GEMM dispatch 根据 precision map 将 output channels 分为 INT8 和 INT4 两组；
     (c) CUDA Graph 并行 launch：INT8 路径直接做 `A_q_int8 × W_q_int8` int8 MMA → scale multiply；INT4 路径走 two-step dequant `(W_q_uint4 − z)` → int8 MMA → fast I2F → scale multiply；
     (d) Fused epilogue scatter 合并两路输出到最终结果矩阵；
     (e) 继续后续层和下一个 decode step。
  5. **基准测试**：`vllm/run_benchmark.sh` 评测 throughput（tokens/s），`vllm/run_gsm8k.sh` 评测精度。
  6. **性能**（Figure 5）：W4A8 vs float16 baseline 平均 1.90× speedup；W4A8 vs TRT-LLM W4A16 平均 1.26× speedup；W8A8 vs TRT-LLM W4A16 平均 1.78× speedup。

  作用：将 MixLLM 的算法级精度优势（W4.4A8 PPL 增量 <0.2 vs float16）转化为生产级 serving 的吞吐量提升——通过 vLLM 集成实现端到端的量化→部署→推理全链路。
