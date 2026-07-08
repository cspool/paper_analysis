## Agent-X: Full Pipeline Acceleration of On-device AI Agents

- baseline方法是什么？
  - Baseline 是未经加速的 TinyAgent [19]（基于 LLMCompiler [36] 的 plan-out agent 框架），运行在 Apple Mac mini M4 Pro 上，使用 TinyAgent-7B (WizardLM-2-7B 微调) 作为后端 LLM，基于 MLX-LM + MLX-engine 进行推理。
  
  - Baseline 全栈执行例子（以 "Schedule a meeting with John tomorrow at 5pm" 为例）：
    - **算法pipeline**: Planner prompt 由 ToolRAG 动态检索的工具描述、指南和 tool-use examples 拼装而成（~1,739 tokens），其中动态 token 出现在 prompt 的 1.6% 位置，导致 prefix caching 几乎无法生效（1,711 uncacheable tokens）。Planner 和 Arbiter 的 decode 阶段使用标准自回归生成（131ms/token），68.7% 的端到端延迟花在 decode 上。
    - **Serving框架**: 使用 Apple MLX-LM + MLX-engine 作为推理引擎。论文未明确说明是否涉及多请求调度——baseline 按单任务串行执行 Planner → Execution → Arbiter 的 agentic workflow。
    - **编译框架**: 使用 MLX v0.25.2 作为 Apple Silicon 的机器学习框架，自动将模型算子编译到 M4 Pro GPU 上执行。论文未修改编译框架。
    - **kernel调度**: MLX 自动将 LLM 推理的 attention、FFN 等算子调度到 M4 Pro 的 16 核 GPU 上执行。论文未涉及 custom kernel。
    - **硬件架构**: Apple M4 Pro SoC（38 INT8 TOPS 计算、546 GB/s 内存带宽 for M4 Max 级别，M4 Pro 数值论文未单独给出）。论文未涉及硬件修改。
    - **端到端延迟**: 35.4 秒（平均）/ 26.7 秒（简单任务 "Schedule a meeting..."），Planner prefill + decode 占 43.5%，Arbiter prefill + decode 占 46.9%，prefill 合计 21.7%，decode 合计 68.7%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **核心洞察**: 与服务端 LLM 不同，端侧 agent 由于 (1) agent 输入序列远长于输出，(2) 端侧硬件计算能力有限（最多 11% 的内存带宽和 ~2% 的计算吞吐 vs. H200），使得 prefill 和 decode 阶段均成为显著瓶颈。进一步 token 级分析发现：(a) prefill 阶段 prompt 虽含大量静态内容但因早期动态插入而无法利用 prefix caching；(b) decode 阶段 96%/87% 的输出 token 与输入 prompt 中的 few-shot examples 重叠，说明大量时间花在生成无需 LLM 推理能力的公式化序列上。
  
  - **PromptWeaver 解决 prefill 瓶颈（Baseline: prefix caching 命中率低）**:
    - **缺陷**: Baseline 中 ToolRAG 动态检索的工具描述在 prompt 前 1.6% 位置引入动态 token，使后续大量静态 token（32.7% 系统 prompt + 工具描述）无法被 prefix caching 复用。
    - **方法**: (1) 将动态的工具描述替换为包含系统所有工具的静态全量描述，扩大可缓存的静态前缀（以 1.4GB 额外 KV cache 存储为代价）；(2) 基于 tool co-activation locality 使用 NMF 将工具聚类（8 个 cluster，每 cluster 2-6 工具），对 cluster 进行 theme-based 固定排序，并用贪心算法选择最高 coverage 的 cluster combination KV cache 预计算存入 SSD（budget=15，5.87GB，74.4% coverage）；(3) 追加单工具 example + Top-1 动态 example 以弥补准确率（K=1 时准确率 0.841，略高于 baseline 0.836）。
    - **效果**: uncacheable tokens 从 1,711 降至 519（-70%），Planner prefill 加速 1.57×，Arbiter prefill 加速 4.35×，综合 prefill 加速 1.97×。
  
  - **ExSpec 解决 decode 瓶颈（Baseline: 标准 speculative decoding 在端侧不可行）**:
    - **缺陷**: 标准 speculative decoding 需要 draft LLM——小 draft 模型准确率太低（Llama-160M: 0.02 draft accuracy），大 draft 模型延迟太高且消耗内存（Llama-3.2-1B: 数百 MB），加上 MLX 框架的 multi-token tax（2-token 验证 244ms vs 1-token 131ms，1.86× 减速），最佳可实现的 speculative decoding 加速仅 1.20×。
    - **方法**: (1) 利用 96%/87% 输出 token 与输入 prompt 重叠的观察，从 few-shot examples + user query 构建 trigram (n=3) LUT 作为 draft model——无需训练、内存占用仅数 KB；(2) 提出 selective decoding：在生成首 draft token 前查 LUT，若当前 2-token 上下文不在 LUT 中则立即回退自回归生成——避免无效 draft 带来的 multi-token tax（Planner 平均 17 次/query 回退，Arbiter 平均 37 次/query 回退）；(3) LUT 构建从 few-shot examples + user query 而非全输入提取，避免无关 token 污染 LUT。
    - **效果**: draft token accuracy 0.25（selective），decode 加速 1.73×，端到端加速贡献 1.43×。

  - Agent-X 全栈执行例子（对比 Baseline）：
    - **算法pipeline**: PromptWeaver 在 Planner prompt 构造阶段动态重组 prompt——前置全部工具描述作为静态可缓存前缀 → 按固定 theme 顺序插入激活 cluster 的 tool-use examples（从 SSD 加载预计算 KV cache 复用）→ 末尾追加单工具 + Top-1 动态 example → 仅 ~519 tokens 需在线 prefill。ExSpec 在 decode 阶段从 few-shot examples + user query 构建 trigram LUT（KB 级内存），每个 decode step 先查 LUT 决策是否 speculative decode，命中则生成最多 4 个 draft tokens 并送出并行验证，未命中则回退标准自回归。
    - **Serving框架**: 论文未明确说明——Agent-X 作为纯软件方案集成到 MLX-LM + MLX-engine 中，修改了 prompt 构造逻辑（PromptWeaver）和 decode 逻辑（ExSpec）。未涉及多请求调度层面的修改。
    - **编译框架**: 论文未修改编译框架——继续使用 MLX 编译算子到 M4 Pro GPU。但 PromptWeaver 的 KV cache 预计算和 SSD 加载机制增加了推理引擎层面的存储-计算协同。
    - **kernel调度**: 论文未涉及 custom kernel 或 kernel 调度修改。论文未明确说明。
    - **硬件架构**: 纯软件方案，无需硬件修改。KV cache 存储在 SSD（6.26 GB total），按需加载到内存。
    - **端到端延迟**: 35.4s → ~22.0s（1.61× speedup），无准确率损失（Planner accuracy 0.841 vs. baseline 0.836）。

## Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference

- baseline方法是什么？
  - Baseline 是现有移动端 LLM 推理引擎的 GPU-only 或 NPU-only 执行模式。代表性的 GPU-only 框架包括 MLC-LLM [34]、MNN-LLM [55]、llama.cpp [11]；NPU-only 框架包括 llm.npu [56]、PowerInfer-2 [60]、Qualcomm-AI [42]、Onnxruntime [33]。这些框架均仅利用移动 SoC 中的单一加速器（GPU 或 NPU）进行计算，无法同时使用两种加速器。

  - Baseline 全栈执行例子（以 Llama-8B 在 Snapdragon 8 Gen 3 上的单请求推理为例）：
    - **算法pipeline**: 使用标准 Transformer decoder 结构（Llama 系列），W4A16 weight-only 量化（权重 INT4 存储、FP16 计算）。各算子（Attention Q/K/V/O、FFN Up/Gate/Down、RMSNorm、SwiGLU、RoPE）按模型图顺序串行执行。论文未明确说明 baseline 是否使用 KV cache 量化或 prefix caching。
    - **Serving框架**: MNN-LLM（GPU-only, OpenCL backend）将 LLM 模型图编译为一系列 GPU kernel，按层串行调度执行。MLC-LLM（GPU-only）使用 ML compilation (TVM Unity) 将模型编译到 Adreno GPU。llm.npu（NPU-only）将模型编译为 QNN 静态计算图并在 Hexagon NPU 上执行，通过 Chunked-prefill 处理长序列，使用 INT8/FP16 混合精度。PowerInfer-2（NPU-only）利用模型稀疏性在 NPU 上执行，使用 INT4/FP16 混合精度。所有 baseline 在 prefill 和 decoding 阶段均仅调用单一后端（GPU 或 NPU），无跨后端调度。
    - **编译框架**: MNN-LLM 使用自研的 MNN 推理引擎（https://github.com/alibaba/MNN）将模型算子编译为 OpenCL kernel。MLC-LLM 使用 TVM Unity（https://github.com/apache/tvm）进行 ML compilation 到 Adreno GPU。llm.npu 和 PowerInfer-2 使用 Qualcomm QNN SDK（https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk）将模型编译为 NPU 静态计算图。论文未修改这些编译框架。
    - **kernel调度**: Baseline 的 kernel 执行完全由各自后端的 runtime 管理。GPU-only 框架通过 OpenCL command queue 串行提交 GPU kernel，kernel 间通过 clFinish 同步（~400μs 开销）。NPU-only 框架通过 QNN runtime 执行预编译的静态计算图。各 framework 均无 GPU-NPU 间的 kernel 协同调度。
    - **硬件架构**: Snapdragon 8 Gen 3 SoC：Adreno 750 GPU（~1 TFLOPS FP16 实际）、Hexagon NPU（~10 TFLOPS FP16 实际，内含多个 32×32 systolic array，weight-stall 计算范式）、统一内存架构（UMA，理论带宽 68 GB/s）。Baseline 中单一处理器（GPU 或 NPU）仅能利用 ~40-45 GB/s 内存带宽（vs. 最大可达 ~61.9 GB/s），内存带宽未充分利用。
    - **端到端性能**（以 Llama-8B, seq_len=256 为例）：MNN-OpenCL prefill ~46 tokens/s, decoding ~9.3 tokens/s; MLC prefill ~42 tokens/s, decoding ~5.5 tokens/s; llama.cpp prefill ~9.5 tokens/s, decoding ~5.5 tokens/s; PowerInfer-2-FP16 prefill ~72 tokens/s, decoding ~10.6 tokens/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **核心洞察**: 通过深入分析移动 SoC 中 GPU 和 NPU 的硬件架构差异，识别出三个关键性能特征：(1) NPU 的 tensor-sensitive 性能——stage performance（32×32 systolic array 导致阶梯状延迟）、order-sensitive performance（weight-stall 范式下权重张量大小影响效率，[14336,4096]×[4096,K] vs [K,4096]×[4096,14336] 最高 6× 差距）、shape-sensitive performance（输入行列比对效率的影响）；(2) UMA 统一内存架构——CPU/GPU/NPU 共享地址空间，消除跨处理器数据拷贝；(3) 单处理器无法饱和 SoC 内存带宽——GPU-only ~40-45 GB/s vs. GPU+NPU 并发 ~60 GB/s。
  
  - **HeteroInfer 解决三个核心缺陷**：
    
    **缺陷 1: GPU 和 NPU 性能差异巨大且场景依赖，简单并行无法保证性能提升**
    - NPU 在某些张量形状下性能退化严重（NPU-1/2/3），仅按原始计算能力分配任务可能适得其反。
    - **方法 — Layer-level GPU-NPU Execution (§4.1)**:
      - 根据算子的计算亲和性在 layer 粒度分配后端：Matmul → NPU（systolic array 优势），RMSNorm/SwiGLU → GPU（灵活性优势）
      - 利用计算不变量 `[M,N] × [N,K] → [[K,N] × [N,M]]^T` 交换张量顺序，使较小的 K 维作为 weight 维度以适配 NPU weight-stall 范式
      - 效果：Llama-8B seq_len=256 时 prefill speedup 3.29× vs. PowerInfer-2-FP16, 5.85× vs. MNN-OpenCL
    
    **缺陷 2: 单处理器仅利用 ~40-45 GB/s 内存带宽（68 GB/s 理论），decoding 阶段受限于内存带宽**
    - GPU-only 或 NPU-only 执行无法饱和 SoC 的多通道内存，decoding 阶段（matrix-vector 操作）完全受限于内存带宽。
    - **方法 — Tensor-level GPU-NPU Parallelism (§4.2)**:
      - Weight-centric partition（静态形状）：沿权重行维度拆分，GPU 和 NPU 并行计算，由 solver 确定最优 partition ratio。在 decoding 阶段使内存带宽从 43.3 GB/s 提升至 59.5 GB/s（96% 最大可用带宽）
      - Activation-centric partition（动态形状）：处理 NPU 静态图不支持的动态序列长度。将激活沿序列长度维度拆分为标准形状子张量（NPU 预生成图执行）+ 动态形状子张量（GPU 执行）。避开了在线 NPU 图生成的高开销（seq_len=135 时 408.4ms，34.6% 总延迟）
      - Hybrid partition：结合两者，在处理 NPU shape-sensitive 退化时同时支持动态形状。如 [4096,14336] × [14336,257-384]，activation-centric 拆出动态部分 + weight-centric 优化 NPU 子任务的形状
      - 效果：Hetero-tensor 比 Hetero-layer 平均 prefill 提升 30.2%（最高 40.8%），decoding 提升 22.0%
    
    **缺陷 3: GPU-NPU 同步开销巨大（clFinish ~400μs），与 decoding kernel 执行时间（数百微秒）相当甚至更大**
    - 传统同步机制（clFinish、fence）未针对移动 SoC 优化，同步开销可能抵消并行带来的收益。
    - **方法 — Fast Synchronization (§4.3)**:
      - 利用 UMA 共享地址空间，维护专用 memory pool 分配 GPU/NPU 的输入输出张量（buffer 跨层复用、不被 driver 回收），消除数据拷贝
      - 利用 LLM 逐层相同操作的可预测性：CPU sync thread sleep(predicted_wait_time) → 轮询输出张量 flag bit（数微秒）→ 立即通知下一后端。替代固定的 clFinish ~400μs
      - Prefill 阶段（NPU-dominant）：GPU 执行被 NPU 覆盖，延迟 next GPU kernel 提交直到 NPU 完成（同步开销 ~数十微秒，可忽略）
      - Decoding 阶段（GPU-dominant）：NPU 执行被 GPU 覆盖，NPU 完成后 enqueue 下一个 GPU kernel，利用 GPU queue ordering 自动保证顺序（无额外提交开销）
      - 效果：decoding 阶段 Llama-8B 有 fast sync 时 4.01× speedup，prefill 阶段 Hetero-tensor 有 fast sync 时 24.3% speedup

  - HeteroInfer 全栈执行例子（对比 Baseline, Llama-8B seq_len=320 prefill + decoding）：
    - **算法pipeline**: 与 baseline 相同（W4A16 量化），但计算执行被拆分为 GPU-NPU 并行路径。利用计算不变量 `[M,N]×[N,K] → [[K,N]×[N,M]]^T` 将 Matmul 的操作数顺序改为适合 NPU weight-stall 范式（小张量作为 weight，大张量作为 activation 流入 systolic array）。
    - **Serving框架**: HeteroInfer 自建推理引擎。Offline 阶段：Profiler 在有限搜索空间（仅 LLM 权重形状 + 标准序列长度）内测量 GPU/NPU kernel 性能（< 20 分钟）→ Solver 为每种 (weight_shape, activation_shape) 组合求解最优 partition strategy 和 ratio → 预生成 NPU 标准形状计算图。Online 阶段：Control plane decider 根据当前 seq_len 查 solver 输出 → 为每个算子选择 GPU-only / NPU-only / GPU-NPU parallel → 执行 → fast sync → merge results。Baseline 中无此调度层，所有算子串行提交到单一后端。
    - **编译框架**: 论文未修改编译框架。GPU kernel 使用 OpenCL 直接开发（非编译生成）；NPU 算子通过 QNN SDK 编译为静态计算图。HeteroInfer 的创新在于绕过（而非修改）NPU 编译器限制——通过 activation-centric partition 将动态形状拆分为标准形状（直接用预编译图）+ 动态形状（卸载到 GPU），避免在线图编译开销（seq_len=135 时 Online-prepare 花 408.4ms 在图生成上）。Baseline 中的 NPU-only 框架要么接受图生成开销（Online-prepare），要么通过大量 padding 造成计算浪费（Padding, 平均 1.91× overhead）。
    - **kernel调度**: HeteroInfer 的核心创新层。对比 baseline：(a) 新增 GPU-NPU 间的 tensor partitioning——沿权重行维度（weight-centric）或激活序列长度维度（activation-centric）拆分张量，GPU 和 NPU 并行计算同一算子的不同子任务；(b) 新增 fast synchronization——替代 clFinish 的 CPU sync thread (sleep + polling)，利用 UMA 消除数据拷贝，利用 LLM 逐层可预测性优化等待时间；(c) 新增 offline solver + online decider——基于 profiling 数据的 min-max 优化选择最优 partition 策略。Baseline 中无任何 GPU-NPU kernel 协同。
    - **硬件架构**: 纯软件方案，无需硬件修改。利用 Snapdragon 8 Gen 3 的既有硬件特性：UMA（共享地址空间消除拷贝）、NPU systolic array（weight-stall 范式决定 tensor 形状偏好）、GPU 异步执行模型（queue ordering 保证 kernel 顺序）、小/中 CPU 核（低功耗轮询 flag bit）。论文在 §6 Discussion 中提出了对未来硬件设计的建议：统一 GPU-NPU 调度器、统一 API 和内存管理层、快速轻量同步库。
    - **端到端性能**: Hetero-tensor 在 Llama-8B 上 prefill 达 247.9 tokens/s（vs. MNN-OpenCL ~46 t/s），decoding 达 14.01 tokens/s（vs. MNN-OpenCL ~9.3 t/s）。InternLM-1.8B prefill 达 1092 tokens/s, decoding 达 51.12 tokens/s。端到端加速 1.34×~6.02× vs. SOTA，无模型准确率损失。能耗降低 55% vs. GPU-only。

## Fast On-device LLM Inference with NPUs

- baseline方法是什么？
  - Baseline 是现有移动端 LLM 推理引擎在 CPU 或 GPU 上的执行。代表性 CPU engine 包括 llama.cpp [59]（使用 K-Quant per-group INT8 量化）、MNN [49]；GPU engine 包括 TFLite [57]、MLC-LLM [75]；NPU baseline 为 PowerInfer-V2 [94]（也使用 NPU 做 prefill，但未解决 per-group 量化与 NPU 不兼容问题）。这些 baseline 均存在 prefill 延迟过高的问题，在典型移动端任务（如 UI 自动化、邮件回复）中 prefill 占 88.3%–98.8% 的端到端延迟。

  - Baseline 全栈执行例子（以 Gemma-2B 在 Snapdragon 8 Gen 3 CPU 上，prompt_len=1500 的自动邮件回复为例）：
    - **算法pipeline（量化推理层）**: llama.cpp 使用 K-Quant per-group INT8 量化——将权重和激活按组（如 group_size=128）分割为多个子张量，每组有独立量化尺度。Pre-fill 时按 token 顺序串行计算：对每个 token t，逐层执行 Q/K/V Linear（INT8 MatMul per-group → FP16 累加中间结果）、Attention Score（FP16）、O Linear（INT8 per-group）、FFN Gate/Up/Down（INT8 per-group）。per-group 方案虽在移动 CPU 上可接受（每组的子 MatMul 可向量化），但量化-反量化开销和组间 FP16 累加带来额外计算。同时 baseline 中 FP16 算子（LayerNorm、Attention softmax、SiLU）与 INT8 算子交替执行，无异构调度。
    - **Serving框架**: llama.cpp (CPU) 使用自研 C/C++ 推理引擎，将 LLM 模型权重加载到内存，按层串行调度执行。每次 prefill 请求中，每个 token 的 prompt 按序通过所有 decoder layer（变长序列无特殊的 chunk/batch 处理），无跨处理器调度能力。GPU baseline (MLC-LLM/TFLite) 将所有算子调度到 GPU 执行，但不使用 NPU。
    - **编译框架**: 论文未修改编译框架。CPU baseline (llama.cpp) 无编译环节——直接使用预编译的 C++ 算子 kernel。GPU baseline (MLC-LLM) 使用 TVM Unity 编译到 Adreno GPU。NPU baseline (PowerInfer-V2) 使用 QNN 编译到 Hexagon NPU。
    - **kernel调度**: CPU baseline 使用 llama.cpp 的 CPU SIMD kernel（ARM NEON），所有 kernel 在 CPU 大核上串行执行，无 GPU/NPU kernel 协同。GPU baseline 的 kernel 在 Adreno GPU 上通过 OpenCL/Vulkan 执行，无 NPU kernel 参与。每次 prefill 的变长 prompt 需要 QNN 重新编译生成 NPU 图（对 Gemma-2B 需 11.54s），NPU baseline 因此实际上比 CPU 更慢。
    - **硬件架构**: Qualcomm Snapdragon 8 Gen 3（Hexagon NPU 73 TOPS INT8, Adreno 750 GPU, ARM CPU）。CPU baseline 仅使用 8 个 CPU 核心（全核满载功耗最高）。GPU baseline 仅使用 Adreno GPU。NPU baseline 使用 Hexagon NPU。所有 baseline 均仅使用单一处理器，NPU/GPU 的并行计算能力和能效优势未能同时利用。
    - **端到端性能**: Gemma-2B, prompt_len~1500, CPU (llama.cpp) prefill ~26.43s + decode ~0.24s → end-to-end 26.7s; GPU (MLC-LLM) prefill ~78.03s + decode ~0.32s; GPU (TFLite) prefill ~2.40s + decode ~0.19s → end-to-end 2.6s（最优 baseline）。NPU (PowerInfer-V2) 论文未报告该模型数据。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **核心洞察**: 移动 NPU 在 INT8 矩阵乘法上具有显著的性能和能效优势（vs CPU INT8: 4.5–5.8×; vs GPU FP16: 1.8–3.5×），但直接用于 LLM prefill 面临三个根本性障碍：(1) 变长 prompt 要求 NPU 每次重新编译计算图（Gemma-2B 需 11.54s），效率反而低于 CPU；(2) 为保持精度的 per-group 量化与 NPU 的 per-tensor MatMul 硬件不兼容（性能损失 8.1–10.7×）；(3) LLM 推理中的 FP16 算子（LayerNorm、Attention）在 NPU 上执行极慢（FP16 MatMul in NPU 比 CPU INT8 慢 193–759×），但无法完全消除。llm.npu 通过三个层次的设计突破这些障碍。

  - **Chunk-sharing graph 解决缺陷 1（变长 prompt → 图编译开销）**:
    - **缺陷**: Baseline (CPU) 不存在图编译开销但计算慢；Baseline (NPU naive) 每次变长 prompt 重建图导致 11.54s 额外延迟；固定 max-length padding 方案浪费 NPU 算力且在短 prompt 时严重退化。
    - **方法**: (1) 利用 decoder-only LLM 的因果性——第 i 个 token 仅依赖前 i-1 个 token，因此处理长 prompt 等价于处理多个因果依赖的定长子 prompt（chunk, 256 tokens）；(2) 预构建固定大小的 chunk NPU 计算图（preparation stage 一次完成），推理时将变长 prompt 分割为多个 chunk 复用预构建图；(3) 进一步将 LLM 算子分为静态算子（Linear/LayerNorm，仅依赖 chunk 长度，跨 chunk 完全共享图）和动态算子（Attention，依赖 chunk 序列位置，按 chunk 独立构建），消除大量冗余图存储——120/144 图可共享，内存节省 75% (7.2GB)。
    - **效果**: per-chunk 图准备时间从 O(11s) 降至 0（one-time），chunk 内 padding 开销远小于 full-length padding（chunk_len=256 时最多 255 tokens padding vs. max_context_len padding）。

  - **Shadow outlier execution 解决缺陷 2（per-group 量化 vs. NPU per-tensor 硬件）**:
    - **缺陷**: Baseline (llama.cpp CPU) 使用 per-group 量化维持精度但 per-group 子 MatMul + FP16 累加在 NPU 上极低效（8.1–10.7× 性能损失）。Baseline (SmoothQuant per-tensor) 虽 NPU 友好但精度损失严重（3.9–8.4% drop）。两种方案无法兼得 NPU 效率与 LLM 精度。
    - **方法**: (1) 采用 NPU 友好的 per-tensor W8A8 量化作为主计算路径；(2) 运行时将激活中超出 INT8 范围 [-127,128] 的异常值通道（outlier channels, 仅 0.1–0.3% 通道）提取为紧凑子张量，在 CPU/GPU 上以 FP16 精度并行执行 MatMul，结果与 NPU 输出相加——等价于全精度计算但 NPU 承担 >99.9% 的 MatMul 计算量；(3) 利用异常值高度集中于少数通道的观察，仅将 3% "热通道" 的 FP16 权重副本保留在 CPU 内存，其余从磁盘按需加载（I/O 与 NPU 执行重叠）；(4) 离线分析逐层异常值重要性（最大异常值/量化尺度比），剪枝 85% 不重要层的异常值执行——这些层的 shadow 执行被完全跳过后 CPU-NPU 同步开销消失。该设计与任何 per-tensor 量化方法兼容。
    - **效果**: 精度 <1% loss vs FP16（优于 SmoothQuant 最高 32.9%、K-Quant 最高 70.9%），同时保持 NPU per-tensor 效率。Shadow 执行时间可被 NPU 完全隐藏（overlap）。

  - **Out-of-order subgraph execution 解决缺陷 3（FP16 算子无法消除 → CPU/GPU 执行造成 NPU 停顿）**:
    - **缺陷**: Baseline 中 LLM 推理的 INT8 和 FP16 算子交替执行——若将 INT8 分配到 NPU、FP16 分配到 CPU/GPU，则按原 chunk 顺序提交会导致 NPU 在等待 CPU 完成 FP16 算子时空闲（naive overlap 产生 37% bubble rate）。
    - **方法**: (1) 在 chunk + 子图两级划分后，子图间存在跨 chunk 依赖（Attention 依赖前序 chunk 的 KV）和块内依赖（LayerNorm/Linear 仅依赖同 chunk 前一个子图），但不是所有子图必须严格按 chunk 顺序执行；(2) 提出在线贪心调度算法——以 NPU 为关键路径，每次选择贡献值 C 最大的就绪子图：若子图分配到 CPU/GPU，则 C = 子图完成后释放出多少 NPU 工作量（越大越好）；若分配到 NPU，则 C = -释放出的 CPU/GPU 工作量（越小/越负越好）；(3) 该调度为微秒级在线决策，适应变长 prompt 带来的不同 chunk 数。
    - **效果**: bubble rate 从 37% 降至显著更低，prefill 延迟减少 18–44%（消融实验）。

  - llm.npu 全栈执行例子（对比 Baseline, Qwen1.5-1.8B prompt_len=1024 邮件回复, Redmi K70 Pro）：
    - **算法pipeline（量化推理层）**: 对比 baseline K-Quant per-group → 改为 llm.npu per-tensor W8A8 + shadow outlier。Pre-fill 时：对每个 chunk 的每层 MatMul，NPU 执行 `clamp(x/s, -127, 128) ⊙ W_INT8` 主路径（2ms），CPU 并行执行 `extract(outlier_channels) ⊙ W_FP16_hot` 影子路径（≪0.1ms）。只有未被剪枝的 15% 层（靠近输入/输出的层）需要影子路径，其余跳过。FP16 算子（LayerNorm, Attention, SiLU, RoPE, Residual）仍由 CPU/GPU 执行。
    - **Serving框架**: 对比 baseline llama.cpp 单 CPU 串行执行 → 改为 MLLM + llm.npu 的 NPU-CPU 协同。Preparation stage: 预构建 chunk-sharing graph（chunk_len=256），一次完成图优化（vs. baseline 每次推理都需重建）。Execution stage: prompt 分割为 4 个 chunk → OOE scheduler 在线选择最优子图提交到 NPU 或 CPU → NPU 执行 INT8 MatMul、CPU 执行 FP16 算子 + shadow 补偿 → unified memory shared buffer 同步。Chunk-sharing 消除每 prompt 的图编译开销（Gemma-2B: 11.54s → 0）。
    - **编译框架**: 基于 QNN 框架编译 NPU 子图（静态形状），但 llm.npu 的创新在于通过 chunk 方法绕过 QNN 对变长输入的编译限制——不修改 QNN 本身，而是将变长 prompt 分解为固定 chunk 使 QNN 可直接复用预编译图。论文未修改 QNN 编译框架。
    - **kernel调度**: NPU 上的 INT8 MatMul 使用 QNN 的 Hexagon NPU kernel（1024-bit SIMD），CPU 上的 FP16 算子使用 MLLM CPU backend kernel，shadow outlier 的 FP16 MatMul 使用优化的 CPU kernel。llm.npu 的 OOE scheduler 在此之上实现了跨 NPU/CPU 的子图级调度——非 kernel 级，但比传统 serving 的请求级调度更细粒度。另外 shape optimization（如 [1024×1×2048] → [32×32×2048] 获得 1.62× 加速）是一种张量级优化。
    - **硬件架构**: Qualcomm Snapdragon 8 Gen 3 SoC。llm.npu 充分利用 Hexagon NPU 的 INT8 SIMD 能力、统一内存架构（消除 CPU-NPU 数据拷贝）、以及 NPU 的低功耗特性（500-750 MHz → 能耗降低 1.9–59.5× vs CPU）。论文未修改硬件，但 §5 提出了三点硬件优化建议：动态形状感知优化、增大 NPU data cache 匹配 LLM 权重尺寸、混合精度计算单元支持（FP16×INT8 操作数）。
    - **端到端性能**: Qwen1.5-1.8B, prompt_len~1600, output~3 tokens → end-to-end 1.7s（vs. CPU 45.6s, 26.8×; vs. GPU-TFLite 2.6s, 1.5×）。Gemma-2B: 1.9s（vs. CPU 34.6s, 18.2×）。首次实现 >1000 tokens/sec prefill 在 COTS 移动设备（Mistral-7B, prompt_len=1024: 1038 tok/s）。

  - **三个技术协同关系**：Chunk-sharing graph 使 NPU 执行变长 prompt 成为可能（消除图编译瓶颈）→ Shadow outlier execution 使 NPU 高效的 per-tensor MatMul 维持可接受的精度（消除量化-精度矛盾）→ Out-of-order subgraph execution 使 CPU/GPU 的 FP16 计算不阻塞 NPU 的关键路径（消除异构停顿）。三者缺一不可——消融实验中 naive NPU offload 反而比 CPU 慢 2.55–2.68×，逐级叠加后最终达到 7.3–43.6× prefill 加速。

## LLM as a System Service on Mobile Devices

- baseline方法是什么？
  - Baseline 是现有移动端 app 各自独立拥有 LLM 或通过 Android Low-Memory Killer (LMK) 统一管理 app 内存与 LLM context 内存的方式。具体包括：(1) LMK——当系统内存不足时直接 kill 后台 app 及其 LLM context，context 被 kill 后再次调用需完全重计算（Llama2-7B 重计算 4K context 在 MI14 上需 22.92s, 94.57J）；(2) Swapping——将整个 LLM context 作为整体交换到磁盘，切换时整体换入（无压缩、无 chunk 粒度优化、I/O 开销大）；(3) VLLM-S——使用 chunk-wise KV cache 管理（类似 vLLM PagedAttention），但无压缩；(4) VLLM-SQ——chunk-wise + 统一 INT8 量化，所有 chunk 等量压缩。

  - Baseline 全栈执行例子（以 MI14 上 Llama2-7B，8 个 active contexts，Markov switching pattern 为例）：
    - **算法pipeline**: 使用标准 Transformer decoder（Llama2-7B），权重 GPTQ W4A16 量化，KV cache INT8 量化（SmoothQuant 类方法）。Baseline VLLM-SQ 对所有 chunk 统一压缩到 INT8，不区分 chunk 信息密度差异。Baseline LMK 无压缩——context 被 kill 后完全重计算。
    - **Serving框架**: Baseline LMK 依赖 Android 原生 memory manager——context 内存被视为 app 内存一部分，通过 oom_adj_score 决定 kill 优先级。LLM context 与 app 内存统一管理，不区分：(1) context 构建成本高（~23s 重计算 vs app 冷启动秒级）；(2) context 访问频率低（如 Gboard 仅在聊天时触发 smart reply）；(3) context 可压缩而 app 内存不可。VLLM-S/SQ 仅做 chunk-wise 内存管理，无 context 生命周期管理（eviction policy 为简单 LRU），无 swapping-recompute pipeline。
    - **编译框架**: 论文未涉及编译框架修改。LLM 推理通过 HuggingFace Transformers (PyTorch) 或 mllm 直接执行，算子调度由 PyTorch runtime / mllm 管理。
    - **kernel调度**: Baseline 无 custom kernel 调度。LLM 推理的 MatMul、Attention 等算子串行执行，CPU/GPU 在磁盘 I/O 期间空闲。
    - **硬件架构**: Qualcomm Snapdragon 8 Gen 3 (MI14)，8 GB RAM + UFS 4.0 存储。Baseline 中 context switching 的瓶颈在于：(1) LMK 路径——CPU/GPU 空转等待 context recompute（~23s 纯计算，无 I/O 重叠）；(2) Swapping 路径——UFS 4.0 顺序读 ~2 GB/s，但整 context swap-in 需 1-2s（2GB KV cache），I/O 期间 CPU/GPU 空转。
    - **端到端 context switching 延迟（8 contexts, Markov pattern）**: LMK ~22.92s（触发 kill → 全量 recompute），Swapping ~2.7s（2GB KV cache 从 UFS 4.0 读入），VLLM-S ~1.5s（chunk 粒度但无压缩），VLLM-SQ ~0.8s（chunk + INT8 统一压缩）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **核心洞察**: 将 LLM 从 per-app model 升级为 system service (LLMaaS) 后，产生了新的系统挑战——如何在多 app 共享 LLM 的场景下高效管理 stateful LLM context（主要是 KV cache）。传统移动内存管理（LMK/Swapping）不适用于 LLM context，因为：(1) LLM context 重建成本极高（时间+能耗），不能随意 kill；(2) LLM context 访问频率低但需持久化，轻量 app 拥有大 context 时容易被 LMK 误杀；(3) KV cache 是数据 chunk，天然可压缩、可分块、可重计算——这些特性在传统 app 内存管理中未被利用。

  - **LLMS 三个核心技术解决三个 Baseline 缺陷**：

    **缺陷 1: LLM context 内存占用过大（单 context 最高 2GB, Llama2-7B 4K），内存不足时 baseline 只能 kill 或全量 swap**
    - Baseline (LMK): 直接 kill → 重计算 ~23s。Baseline (Swapping): 整 context 换入 → ~2.7s I/O。Baseline (VLLM-SQ): 统一 INT8 → 节省 50% 内存，但未区分 chunk 重要性差异（低信息量 chunk 与高信息量 chunk 同等待遇）。
    - **方法 — Tolerance-Aware Compression (§3.2)**:
      - 利用 attention scores 量化每个 chunk 的"信息密度"——被更多其他 token 关注的 token 更 informative，压缩容忍度更低。
      - 公式化：D_i = 跨 head/layer/token 聚合的 attention score 列均值。
      - 在全局平均压缩比约束下求解优化问题：max Σ (1/ratio_w) × Σ D_i，为高密度 chunk 保留更多精度（INT8），低密度 chunk 激进压缩（INT4/INT2）。
      - 效果：ratio_global=50% 时，memory ≈ 全量 INT4（50% 内存），accuracy ≈ 全量 INT8（无明显损失）。vs 静态 INT4 全量压缩（59% accuracy loss），vs 静态 INT2（99% loss 不可用）。

    **缺陷 2: Context switching 时 I/O 是瓶颈，但 CPU/GPU 在磁盘 I/O 期间完全空闲**
    - Baseline (Swapping): 纯 I/O——2GB KV cache 从 UFS 4.0 读入 ~2.7s，CPU/GPU 空转。Baseline (VLLM-SQ): 纯 I/O——swap-in 压缩后的 ~1GB KV cache ~1.3s，CPU/GPU 空转。没有一个 baseline 利用 KV cache 可重计算的特性。
    - **方法 — Swapping-Recompute Pipeline (§3.3)**:
      - 核心观察：KV cache 本质是 LLM 中间激活，可从原始 prompt text 重计算恢复——重计算用 CPU/GPU，与磁盘 I/O 完全独立即可并行。
      - 技术挑战：chunk 可以不连续地被换出（如 chunk_3 和 chunk_7 在磁盘而其余在内存），标准 LLM 的连续 position encoding 和 causal mask 无法处理不连续 chunk 的重计算。
      - 解决方案：(1) 修改 position encoding——对重计算的 chunk 使用全局位置（如 chunk_3 → pos=48..63），而非相对位置；(2) 修改 causal mask——每个重计算 token 只 mask 掉其 position 之后的 token（跨 chunk 保留正确因果性）；(3) Pipeline 设计——I/O 线程加载下一层 chunk 到内存，计算线程在当前层 I/O 完成后重计算，两层流水线重叠。
      - 效果：context switching 的 wall-clock 时间从 max(T_IO, T_re)（无重叠）降至 ≈ max(T_IO_partial, T_re_partial)（pipeline 重叠），消融实验中该技术独立贡献 1.62s → 0.42s 的切换延迟降低。

    **缺陷 3: Baseline eviction policy（LMK 的 oom_adj_score, VLLM 的 LRU）未考虑 LLM context 的特殊性**
    - Baseline (LMK): evict 基于 app 类型（前台/后台/服务），完全忽略 context 自身的重建成本和访问模式。例如一个使用 LLM 聊天机器人的轻量 app 拥有大 context，系统内存不足时可能被优先 kill（因 app 本身轻量），但 context 重建成本极高。
    - Baseline (VLLM-S): 简单 LRU——最近最少使用的 chunk 优先 evict，不考虑 chunk 大小（压缩率）和 evict 后重载成本。
    - **方法 — Chunk Lifecycle Management (§3.4)**:
      - **Which to evict: LCTRU Queue（Least Compression-Tolerable and Recently-Used）**：
        - 按压缩率分组：Q_{8/8} → Q_{4/8} → Q_{2/8}，heavy chunk（低压缩率、占用更多内存）优先 evict。
        - 设计原理推导：(a) heavy chunk 占用更多内存，换出它们能释放更多空间；(b) 从 swapping-recompute pipeline 角度，给定需释放的内存大小 m，evict 更少的大 chunk 意味着重载时需处理的 chunk 数更少 → pipeline 延迟更低（Equation 4: T_re depends on chunk count x, not memory size m）。
        - 组内 LRU：利用 context 访问的时间局部性（类似传统缓存策略）。
      - **When to swap out: AoT (Ahead-of-Time) Swapping**：
        - 利用 LLM Service 内存修改模式简单（chunk 在 LLM 推理中顺序修改）的特点。
        - 在 callLLM() 返回阶段就主动将修改过的 chunk 写入磁盘（即使无内存压力），而非等到 Reclaim 被触发时才匆忙换出。
        - Reclaim 延迟被完全隐藏——context switching 时 reclaim 操作是 overhead-free 的。
      - **Working Set Lock**：callLLM() 执行期间锁定当前 context 的所有 chunk 不被 evict，避免 thrashing。
      - 效果：消融实验中该技术独立贡献 0.42s → 0.27s 的切换延迟降低（移除 lifecycle mgmt → switching latency 从 0.27s 升至 0.62s）。

  - **LLMS 全栈执行例子（对比 Baseline, MI14 上 Llama2-7B, 8 active contexts, Markov switching pattern）**：
    - **算法pipeline（量化推理层）**: 与 baseline 相同使用 GPTQ W4A16 + SmoothQuant INT8 KV cache。LLMS 的创新在于在 INT8 之上叠加 tolerance-aware chunk-wise 压缩（INT4/INT2），压缩决策由 attention score 驱动的信息密度指标引导，而非修改底层量化算法本身。算法层面的改动是压缩比分配策略（optimization formulation），不影响 LLM 推理的数学正确性。
    - **Serving框架（系统服务层）**: 对比 baseline 的 per-app LLM 或 LMK 统一管理 → LLMS 将 LLM 提升为 system service，解耦 app 内存与 context 内存管理。内存模型：context = memory-resident fragment (text) + swappable fragment (KV cache chunks)。四个原语：Claim → Reclaim → Load → Fault。LMK 完全不区分 app 内存与 context 内存，LLMS 区分并针对性优化。Swapping baseline 将 context 作为整体管理（粗粒度），LLMS 以 16-token chunk 为粒度（细粒度，平衡内存利用率和 I/O 带宽）。
    - **编译框架**: 论文未修改编译框架。LLMS 的 swapping-recompute pipeline 在 LLM 推理层之上实现（通过修改 position encoding + causal mask），不侵入底层编译器/框架。
    - **kernel调度（运行时计算层）**: Baseline 中 CPU/GPU 在磁盘 I/O 期间完全空闲。LLMS 通过 swapping-recompute pipeline 实现计算与 I/O 的重叠——I/O 线程（加载 chunk 到内存）与 Compute 线程（重计算 chunk）以流水线方式并行。这是 kernel 调度之上的系统级调度优化。Planning 阶段用线性规划求解最优 I/O-recompute 分配（基于 offline profiling 的 T_re 和 T_IO 函数）。
    - **硬件架构**: 纯软件方案，无需硬件修改。利用 COTS 设备的既有硬件特性：(a) UFS 4.0 高带宽（~2 GB/s 顺序读）支持快速 chunk swap；(b) CPU/GPU 重计算能力在 I/O 期间闲置，可被 pipeline 利用；(c) 设备 RAM 8GB——LLM 权重 ~3.5GB (4-bit)，其余 ~4.5GB 用于 contexts，LLMS 通过压缩和 swapping 扩展虚拟 context 容量。LLMS 在低磁盘带宽设备（TX2 SATA HDD）上仍有效但 switching latency 更高，受限于 I/O。
    - **端到端 context switching 延迟**: LMK ~22.92s → Swapping ~2.7s → VLLM-S ~1.5s → VLLM-SQ ~0.8s → LLMS ~0.27s。LLMS 相比最强 baseline (VLLM-SQ) 实现 9.7× 平均加速，相比 LMK 实现 2 个数量级加速。10ms latency constraint + 3GB budget 下 LLMS 支持 2.85× 更多 active contexts。

## Rethinking DVFS for Mobile LLMs: Unified Energy-Aware Scheduling with CORE

- baseline方法是什么？
  - Baseline 是 Android 移动设备上默认的三个独立 DVFS governor：(1) sched-pixel EAS（CPU governor）——基于任务 load 估计和 per-cluster load-to-frequency 查表，对 GPU-heavy 任务因 CPU 利用率低而倾向选择低频；(2) Quickstep governor（GPU governor）——基于厂商预定义的 utilization-to-frequency 表，利用率低于 target range 则降频；(3) Interactive governor（Memory governor）——基于内存总线利用率，高利用率时提升到峰值频率，低利用率时按公式逐步降低。三个 governor 各自独立运行，彼此不感知对方的频率调整。

  - Baseline 全栈执行例子（以 TinyLlama-1.1B 在 Pixel 7 上的 decode 阶段为例）：
    - **算法pipeline**: 使用标准 4-bit 量化 LLM 模型（TinyLlama-1.1B, StableLM-Zephyr-3B, Llama-2-7B 等），在 llama.cpp 框架 + CLBlast OpenCL 库上以 GPU 为主执行推理。decode 阶段每 token 执行一次 Transformer block 的完整前向（Attention Q/K/V/O → FFN Up/Gate/Down），batch_size=1（仅处理上一个 token），计算密度低于 prefill。论文未修改模型量化或结构。
    - **Serving框架**: llama.cpp（tag b2202）作为推理引擎，GPU 推理通过 OpenCL + CLBlast 支持。模型加载在 performance core（ARM Cortex-X1）上，profiling 进程固定在 LITTLE core（ARM Cortex-A55）。三个默认 governor 在 llama.cpp 推理过程中独立运行——CPU governor (EAS) 监测 CPU utilization 调整 CPU 频率；GPU governor (Quickstep) 监测 GPU utilization 调整 GPU 频率；Memory governor (Interactive) 监测内存总线利用率调整 MIF 频率。三者间无任何协调机制。
    - **编译框架**: 论文未修改编译框架。llama.cpp 使用预编译 C++ kernel，通过 CLBlast 将算子编译为 OpenCL kernel 在 Mali GPU 上执行。论文未涉及编译层面优化。
    - **kernel调度**: OpenCL runtime 在 CPU 上管理 command queue，将 GPU kernel（MatMul, Attention 等）逐个提交到 Mali-G710 GPU。由于 Mali GPU 仅支持 2 个 outstanding queue entries 的浅队列，CPU 需持续参与——每当 GPU 完成一个 kernel，CPU 上的 OpenCL runtime 立即提交下一个。论文未创建新的 kernel 或修改 kernel 调度算法。
    - **硬件架构**: Google Tensor G2 SoC（ARM Cortex-X1 + Cortex-A55 + Mali-G710 MP7 GPU），LPDDR5 内存（Pixel 7: 8GB, Pixel 7 Pro: 12GB），Android 13。CPU/GPU/Memory 各有独立的 DVFS 频率表（CPU: 18 档 500-2850 MHz; GPU: 12 档 151-848 MHz; Memory: 13 档 421-3172 MHz）。三个硬件组件的频率由各自的 governor 独立控制，存在跨组件的利用率耦合（CPU 频率影响 GPU kernel 提交延迟 → 影响 GPU utilization; GPU 频率影响 kernel 执行时间 → 影响 CPU 等待时间 → 影响 CPU utilization）。
    - **端到端性能（TinyLlama-1.1B, decode 32 tokens, 默认 governors）**: TPOT 215.1 ms, 402.7 mJ/token。GPU 有效频率 424.4 MHz（远低于最优 848 MHz），CPU 有效频率 1130.8 MHz（远低于最优 2252 MHz）。Antagonistic effect 可将 GPU 频率级联压低至 151 MHz（最低），CPU 频率至 500 MHz（最低），此时 TPOT 与最优 Pin 配置差距可达 41%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **核心洞察**: 移动端 LLM 推理同时使用 CPU、GPU 和 Memory 三个功耗密集型组件，但现代 Android 为三者设计的 DVFS governor 各自独立运行且互不感知。这种独立性在一般 workload 下有效，但在 LLM 推理（CPU 主要扮演 OpenCL runtime 角色而非计算角色，导致利用率极低）下产生了两个根本问题：(1) 单独运行时，GPU governor 和 CPU governor（EAS）均因利用率低而选择过于保守的低频率，导致长延迟和低能效；(2) 并发运行时，两者触发 antagonistic "downward spiral"——CPU 降频 → GPU kernel 提交变慢 → GPU utilization 降低 → GPU governor 降 GPU 频率 → kernel 执行变慢 → CPU 等待 GPU 时间增加 → CPU utilization 降低 → EAS 进一步降 CPU 频率 → 循环往复直至两者均降至最低频率。

  - **FUSE 的核心设计（Offline Profiling + Runtime Pinning，规避 governor 独立决策）**：

    **缺陷 1: GPU governor 在 decode 阶段因 GPU 利用率低（~50-70%）选择过低频率，导致 TPOT 比最优配置高 41%**
    - 根源：Quickstep governor 基于厂商 dvfs_table 的 target utilization range（Fig. 1），decode 阶段 batch_size=1 导致 GPU 利用率始终低于 target → governor 不断降频。
    - FUSE 方法 — 两阶段频率搜索（§6.1）：
      - Step 1 GPU 搜索：利用 GPU 频率与 energy-per-token 的 U 形曲线关系——从最高 GPU 频率递减，停在第一个满足 energy budget 的频率。同时保留前一个不满足 budget 的频率作为候选（因为搭配更高 CPU 频率可能满足 budget 并获得更低延迟）。对 G2（最小化能耗给定延迟目标），先定位 minimum-energy 频率（U 形底部），再由延迟目标决定最终频率。
      - Step 2 CPU 搜索：对至多两个 GPU 候选频率，从最高 CPU 频率递减搜索，停在第一个满足 energy budget（G1）或超过 latency target（G2）的 CPU 频率。选择延迟最低（G1）或能耗最低（G2）的 GPU+CPU 组合。
      - Memory 保持默认 governor——§5.2 发现 memory governor 已能实现近最优。
    - 效果：搜索次数从穷举 2808 降至平均 45 次推理/setting（62× reduction），搜索耗时 17.7-78.5 分钟 per model（取决于模型大小）。

    **缺陷 2: EAS（CPU governor）因 CPU 利用率极低（17-25%）选择过低 CPU 频率，导致 OpenCL runtime 响应变慢、GPU kernel 提交延迟增加**
    - 根源：EAS 的 task load 估计基于 CPU utilization + 频率缩放因子 + 历史指数衰减——LLM 推理中 CPU 大部分时间在等待 GPU（I/O wait），task load 持续衰减，EAS 因此选更低频率。EAS 呈现双峰分布——在 medium（~1426 MHz）和 low（~851 或 500 MHz）间频繁切换。
    - FUSE 方法：在 Step 2 的 CPU 搜索中，从最高频率开始而非从 EAS 选的低频开始——绕过了 EAS 的 load 估计逻辑，直接测试高 CPU 频率是否满足 energy budget。如对 TinyLlama，FUSE 选择 2252 MHz CPU（vs EAS 的 1130.8 MHz），TPOT 降低 13.2% 且同能耗。
    - 关键设计洞察：CPU 频率提升带来的额外功耗被 GPU 执行时间缩短所抵消（因为 CPU 更快提交 GPU kernel → GPU 利用率提升 → GPU 在更短时间内完成计算 → 总能量 = power × time 基本不变甚至降低）。

    **缺陷 3（最核心）: Antagonistic Effect — GPU governor 和 CPU governor 并发运行时触发双向"向下螺旋"，将 GPU 和 CPU 频率级联压低至硬件最低值**
    - 根源：每个 governor 独立尝试满足各自的利用率目标。CPU 降频 → OpenCL runtime 慢 → GPU task 提交延迟 ↑ → GPU utilization ↓ → GPU governor 降 GPU 频率 → kernel 执行变慢 → CPU 更多时间在等待 GPU → CPU utilization ↓ → EAS 降 CPU 频率 → 循环。
    - 量化证据：CPU 从 2850 降至 500 MHz 时 GPU utilization 从 70.9% 降至 52.9%；GPU 从 848 降至 151 MHz 时 CPU utilization 从 25.7% 降至 7.9%。
    - FUSE 方法 — 运行时频率 Pinning（§6 设计概述）：
      - Offline 阶段：频率搜索时将 GPU 固定（pin）在候选频率——这直接打破了 antagonistic effect 的反馈回路（GPU 频率不再由 GPU governor 的 utilization 反馈决定）。
      - Online 阶段：推理时通过 sysfs 将 scaling_min_freq 和 scaling_max_freq 设为同一值（等同于 pin 住频率），使 governor 失去调频能力。
      - 阶段感知切换：利用 llama.cpp 的 prefill/decode 阶段通知，在 prefill 和 decode 阶段使用不同的最优频率组合（因两阶段的计算特征不同——prefill 是 compute-bound 且 GPU utilization 高，decode 是 memory-bound 且 GPU utilization 低）。
    - 效果：在 ShareGPT trace 上，FUSE (G1) 对 TinyLlama 实现 TPOT -25.4%、E2E -22.1%（同能耗）；对 DeepSeek-R1-Distill-Qwen 实现 TPOT -36.8%、E2E -28.0%。

  - FUSE 全栈执行例子（对比 Baseline, TinyLlama-1.1B, Pixel 7, G1 同能耗目标）：
    - **算法pipeline**: 与 baseline 相同——4-bit 量化模型，标准 Transformer decoder 结构，llama.cpp + CLBlast + OpenCL。FUSE 不修改模型推理算法。论文未明确说明 baseline 是否使用其他算法级优化。
    - **Serving框架（governor 层）**: Baseline: llama.cpp 推理过程中，Android 三个默认 governor 独立运行——EAS 基于 CPU load 动态调 CPU 频率，Quickstep 基于 GPU utilization 动态调 GPU 频率，Interactive 基于 memory bus utilization 动态调 MIF 频率，三者无协调。FUSE: 在 llama.cpp 外部新增 governor 模块（~2K Python），通过 sysfs 接口直接控制频率。Offline 时 profiling 搜索最优 (f_cpu, f_gpu) 组合（6 settings × ~45 推理，17.7 min for TinyLlama）→ Online 时 llama.cpp 发送 prefill/decode 阶段通知 → FUSE 查表 → 写 sysfs pin 住频率 → 推理执行 → 阶段切换时重新 pin → 推理结束后恢复默认 governor。对比 baseline 的 utilization-feedback 闭环调频，FUSE 的 open-loop pinning 彻底消除了 antagonistic effect。
    - **编译框架**: 论文未修改编译框架。论文未明确说明。
    - **kernel调度**: 与 baseline 相同——OpenCL runtime 管理 Mali GPU command queue，CPU 提交 kernel 到 GPU 执行。FUSE 的创新不在于 kernel 调度本身，而在于通过 CPU/GPU 频率的协同设定优化了 OpenCL runtime 的 kernel 提交延迟和 GPU kernel 的执行时间，使二者在时间轴上更紧密衔接（减少 GPU idle bubble）。论文未创建新 kernel。
    - **硬件架构**: 纯软件方案，无需硬件修改。利用 Pixel 7/7 Pro 既有 DVFS 硬件能力（18×12×13 频率组合空间）和 sysfs 接口（scaling_min_freq/scaling_max_freq, min_freq/max_freq）。FUSE 的关键硬件洞察：现代移动 SoC 的 DVFS 硬件已支持细粒度频率控制，但 governor 软件层的独立设计导致频率选择劣化——FUSE 通过软件层的统一决策释放了硬件的全部能效潜力。论文在 Discussion 中未提出硬件修改建议，但研究结果暗示未来移动 SoC 设计应提供跨组件（CPU-GPU-Memory）的统一频率协调接口。
    - **端到端性能（ShareGPT 200 requests, avg prefill 232.4 + decode 70.0 tokens）**: Baseline (Gov) TTFT 10.56s, TPOT 210.7ms, E2E 25.2s, 总能耗 738.1 mAh → FUSE (G1) TTFT 9.04s (-14.4%), TPOT 157.2ms (-25.4%), E2E 19.6s (-22.1%), 总能耗 737.8 mAh（持平）。FUSE (G2): 同 TTFT 10.22s, TPOT 175.2ms (-16.8%), E2E 20.9s (-17.1%), 总能耗 672.3 mAh (-8.9%)。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- baseline方法是什么？
  - Baseline 是在移动设备上使用 GPU 或闭源 NPU SDK 进行 LLM 推理的标准方案。具体包括：(1) **llama.cpp OpenCL backend**（Adreno GPU, Q4_0 kernel）——GPU 在 batch=1 decoding 时效率可接受（GEMV 优化好），但 batch 扩展性差；(2) **QNN-based NPU 推理**（如 PowerServe）——使用 Qualcomm 闭源 QNN SDK，仅支持 per-tensor/per-channel 量化，精度严重损失（MATH500 仅 2.1% vs. AutoAWQ 15.9%），且静态计算图限制 batch size 不可变；(3) **其他 NPU 系统**（llm.npu 在 decoding 阶段不用 NPU，PowerInfer-2/HeteroLLM 未开源）。所有 baseline 均未针对 test-time scaling（并行多路径采样）优化，且未解决 NPU fine-grained group quantization 的硬件不兼容问题。

  - Baseline 全栈执行例子（以 Qwen2.5-1.5B, 单路径 decoding, Snapdragon 8 Gen 3 为例）：
    - **算法pipeline（量化层）**: Baseline 使用 QNN per-channel W4A16 量化或 AutoAWQ per-group W4A16 量化。QNN per-channel：将整个权重通道用单一 scale 量化，精度损失严重——Llama3.2-1B-Instruct 在 MATH500 上仅 2.1% accuracy（vs. AutoAWQ 15.9%）。AutoAWQ per-group：group_size=128，虽精度可接受但 NPU 矩阵单元不支持 fine-grained group quantization 的硬件加速，runtime dequantization 必须在向量单元上用 scatter 方式执行（极低效）。GPU baseline 使用 llama.cpp OpenCL backend 的 Q4_0 kernel——在 Adreno GPU 上 GEMV 优化相对良好，但无法利用 NPU 12 TFLOPS 算力。
    - **Serving框架**: llama.cpp CPU/GPU backend 或 QNN SDK。CPU/GPU backend：模型权重加载到内存/显存，按层串行调度执行。无 NPU 参与。QNN：通过 ONNX 中间表示→QNN 编译为静态计算图→Hexagon NPU 执行。弱点：(a) batch size 固定——无法适配 test-time scaling 的动态 generation budget；(b) 算子覆盖有限——仅支持标准 DNN 算子，LLM 特有的 RoPE、SiLU 等不在 NPU 上执行。
    - **编译框架**: QNN SDK（Qualcomm 闭源）将模型编译为 Hexagon NPU 静态计算图，开发者无法自定义底层 kernel。论文未修改编译框架——实际上论文通过 reverse engineering 和 LLVM toolchain 完全绕过了 QNN。
    - **kernel调度**: Baseline GPU: OpenCL command queue 串行提交 GPU kernel（MatMul, Attention 等），kernel 间通过 event/callback 同步。Baseline NPU (QNN): NPU runtime 管理预编译静态图的执行——HMX 执行 MatMul tile 操作，DMA 管理 DDR↔TCM 数据搬运。但 QNN 路径中：(a) decoding 阶段 GEMM→GEMV，HMX tile [1→32 padded, 32]×[32,32] 仅 1/32 有效行——31/32 HMX 算力浪费；(b) per-channel 量化虽 NPU 友好但精度不可接受。
    - **硬件架构**: Snapdragon 8 Gen 3 SoC（Adreno 750 GPU + Hexagon NPU）。NPU 含 HVX 向量单元（4-6 个，32×1024-bit 寄存器）和 HMX 矩阵单元（1-2 个，FP16 32×32 tile，~12 TFLOPS）。统一内存架构（UMA）。GPU baseline 仅用 GPU ~1 TFLOPS（浪费 NPU 12 TFLOPS）。NPU (QNN) baseline 在 decoding 阶段 HMX tile 利用率 ~3%（浪费 NPU 97% 算力）。
    - **端到端性能**: GPU (batch=1 decoding)：Qwen2.5-1.5B ~7 tok/s; NPU (QNN, batch=1)：论文未给具体数值但指出 llm.npu 在 decoding 阶段不用 NPU。QNN 的 accuracy 不满足 test-time scaling 最低要求（MATH500 2.1%→scaling 无法拯救）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **核心洞察**: 移动 NPU 在典型 LLM decoding（GEMM→GEMV）中存在大量闲置算力（HMX tile 仅 1/32 有效行）。test-time scaling 通过并行多路径采样自然增加 batch size → 填充 HMX tile 的空行 → 几乎零额外解码延迟地利用 NPU 算力。但实现这一洞察面临两个硬件挑战：(1) NPU 缺乏 fine-grained group quantization 的硬件支持（精度 vs. 效率不可兼得）；(2) NPU 向量单元通用计算和内存带宽远弱于矩阵单元（HVX 33 GFLOPS vs. HMX 12 TFLOPS，~365× 差距），Non-GEMM 算子成为瓶颈。

  - **硬件感知 Tile 量化方案 解决缺陷 1（NPU 不支持 fine-grained group quantization → 精度损失或效率损失）**:
    - **缺陷**: Baseline QNN per-channel 量化精度不可接受（MATH500: 2.1% vs. AutoAWQ 15.9%）。Baseline per-group 量化若不在 NPU tile layout 上排列 → runtime dequantization 需要 scatter 写入 TCM（非连续 memory access，HVX 上代价极高）。直接 transpose 也不解决——HMX 多级 tile layout 导致 scatter 依然存在（图 6）。
    - **方法**: (1) **Pre-quantization permutation**：将 FP16 权重在量化前重新排列为 HMX tile layout（外层 column-major tiles，内层每两行 permute），使量化后的权重在内存中已按 HMX 访问顺序排列 → 反量化结果可连续写入 TCM（消除 scatter）。(2) **Tile-group quantization**：在 permuted 布局上以 2×16 tile (32 elements) 为 group 执行量化——由于预训练权重近似零均值高斯分布，tile 内 reshuffle 不显著改变 group 内统计特性 → 量化误差与常规 group quantization 可比。(3) **Super-group coalesce**：将 8 个 group (256 INT4 = 128 bytes) 合并为一个 super-group → 恰好填满 1 个 128-byte HVX 向量寄存器 → 一次 vector load 处理 256 个量化值 → 最大化向量单元利用率。
    - **效果**: GEMM dequantization 加速 9.65–19.04× vs. baseline scatter 方法，仅比"no dequantization"上界慢 27%。精度无显著损失（Wiki PPL: tile 10.206 vs. common 10.190 vs. F16 9.798）。

  - **LUT-Based 计算 解决缺陷 2（向量单元算力弱 → Non-GEMM 算子成为瓶颈）**:
    - **缺陷**: HVX 向量单元 FP16 GEMM 仅 33 GFLOPS（vs. HMX 12 TFLOPS）。Softmax 中 exp 需对 Θ(N_q × N_kv) 个元素执行——传统 FP32 polynomial exp 需要大量 HVX 指令（Taylor 级数求值有序列依赖，限制 VLIW 指令级并行）。Dequantization 中传统 INT4→FP16 转换需要 mask→unpack→convert 多指令序列（V79 前还需 qfloat↔IEEE-754 转换）。
    - **方法**: (1) **LUT-Based exp (Softmax)**：在 64 KiB TCM 中预计算 32768-entry FP16 exp LUT。利用 safe softmax 确保所有 exp 输入 ≤0 → 忽略符号位 + 左移一位生成 vgather 字节偏移。vgather 一次收集 64 个 FP16 值 → 替代逐元素 polynomial 求值。LUT 初始化一次性开销，仅占 0.8% TCM。(2) **LUT-Based INT4→FP16 (Dequantization)**：使用 vlut16 指令（16-entry LUT，8-bit index → 16-bit output）将 4-bit 量化值直接转为 FP16。vlut16 同时用于 4 组 scales 的广播（替代标量广播+寄存器拼接）。LUT 方法直接在寄存器内完成转换——V79 前避免 qfloat↔IEEE-754 转换开销，V79 后利用硬件改进。LUT 内容可灵活更换以支持不同编码（FP4, NF4, IQ4_NL 等）。
    - **效果**: Softmax 加速 1.26–2.19× vs. FP32 exp，1.60× vs. FP16 poly exp。Dequantization 使用 vlut16 消除传统指令序列开销（具体加速体现在 GEMM 整体 9.65–19.04× 中）。

  - **Paper 方法全栈执行例子（对比 Baseline, Qwen2.5-1.5B Best-of-N N=8, Snapdragon 8 Gen 3）**：
    - **算法pipeline（量化+推理算法层）**: 对比 baseline QNN per-channel → 论文使用 hardware-aware tile Q4_0 量化（4.5 BPW）+ Q8_0 FFN down（8.5 BPW）。Pre-fill 和 decoding 中所有 MatMul 执行 on-the-fly dequantization：HVX LUT 反量化（INT4→FP16）→ 连续写 TCM（HMX layout）→ HMX FP16 tile MatMul。对比 baseline 单路径 decoding → 论文 Best-of-N（N=8）：8 条独立路径并行采样，每条路径共享 prompt KV cache。Batch=8 decoding 时 HMX tile [8→32 padded]×[32,32] = 25% 利用率（vs. baseline 3%）。Skywork-1.5B-PRM 作为 outcome-reward scorer 选出最优路径。
    - **Serving框架**: 对比 baseline llama.cpp GPU backend 或 QNN → 论文新增 Hexagon NPU backend（约 7K 行代码）集成到 llama.cpp。rpcmem shared memory 消除 CPU-NPU 数据拷贝。FastRPC 管理 remote NPU session。CPU-NPU 通信通过 shared memory polling + manual cache maintenance。算子级 CPU fallback（lm_head 因 NPU 32-bit 地址空间限制保留在 CPU）。无 QNN 依赖 → batch size 可动态变化（runtime 可变 generation budget），模型权重可灵活量化。
    - **编译框架**: 对比 baseline QNN（闭源，静态计算图，开发者无法自定义 kernel）→ 论文完全绕过 QNN。使用 Hexagon SDK 6.0.0.2 LLVM toolchain 直接编译 C/C++ + inline assembly 代码 → Hexagon DSP shared object。HMX 指令通过 reverse engineering 二进制库获取。论文未修改任何编译框架——而是绕过了需要编译框架的场景。
    - **kernel调度**: 对比 baseline OpenCL command queue（GPU kernel 串行提交）或 QNN runtime（预编译图自动执行）→ 论文直接实现底层 NPU kernel：
      - HMX GEMM：hmx_load_activation + hmx_load_weight + hmx_matmul_accumulate + hmx_store。Tile [B_q×32]×[32×32]，FP32 内部累加，FP16 I/O。
      - HVX Dequantization：DMA prefetch → vlut16 INT4→FP16 → vlut16 scales broadcast → vmpy → vstore（连续写入 TCM）。
      - HVX FlashAttention：HMX MatMul (QK^T) → HVX vgather LUT exp → HVX rowmax/rowsum/rescale → HMX MatMul (PV)。
      - DMA 管理：1D/2D async prefetch，DDR↔TCM。
      - CPU-NPU 同步：shared memory flag polling（微秒级，替代 FastRPC 默认 RPC 的高延迟）。
    - **硬件架构**: 纯软件方案，无需硬件修改。核心硬件洞察：(a) HMX tile [32×32] 的基本计算粒度意味着 batch<32 时必然有 tile 行浪费——test-time scaling 的多路径采样恰好填充这些空行；(b) HMX (~12 TFLOPS) 与 HVX (~33 GFLOPS) 之间存在 ~365× 算力鸿沟——将重计算推到 HMX（MatMul），轻计算用 LUT 在 HVX 上加速（exp, dequant）；(c) TCM 8 MiB 是稀缺资源——LUT (64 KiB) 仅占 0.8%，DMA 预取替代 cache 自动管理；(d) NPU 32-bit 地址空间限制 → lm_head 回退 CPU，是当前主要瓶颈（batch=16 时 CPU logits ≥50% 时间）。
    - **端到端性能**: 
      - Conventional (batch=1): NPU ~5 tok/s（GPU 更快 ~7 tok/s）
      - Best-of-N (batch=8): NPU ~22 tok/s（GPU 扩展性不如 NPU）→ 8 路径并行 Wall-clock 接近 1 路径
      - Test-time scaling Pareto: Qwen2.5-1.5B + Best-of-N ≥ Qwen2.5-3B baseline accuracy + 更低延迟
      - Qwen2.5-3B + Best-of-N ≥ Qwen2.5-7B baseline accuracy
      - 能耗 (batch=8, 1.5B): < 3B batch=1 能耗，同时精度可比
      - GEMM dequant: 9.65–19.04× vs. scatter baseline，仅比理论上界慢 27%
      - Softmax LUT: 1.26–2.19× vs. FP32 exp
