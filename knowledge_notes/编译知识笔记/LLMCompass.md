## LLMCompass

术语是什么？
LLMCompass 是 Princeton University 在 ISCA 2024 发表的 LLM 推理性能模拟框架（开源：https://github.com/PrincetonUniversity/LLMCompass）。它是一个 block-level（tile-by-tile）模拟器，利用 LLM 计算图的 dense operator（GEMM、softmax、LayerNorm 等）具有结构化和可预测的 compute/memory access pattern 的特性，在不损失精度的情况下实现比 cycle-accurate simulator 快数个数量级的模拟。LLMCompass 包含：(1) tile-by-tile 模拟引擎——分层建模 compute 和 memory hierarchy（Main memory → Global buffer → Local buffer → Lanes/PE array）；(2) Mapper——自动搜索最优 tiling scheme 和 schedule scheme；(3) Area & cost model——基于晶体管数和 die area 估算，对照 NVIDIA A100 GA100 和 AMD MI210 验证误差 ~5-8%。LLM inference 误差 ~4.1%，单次 GPT-3 175B 4-GPU 模拟仅需 16 分钟。

从编译框架角度拆解术语：
LLMCompass 作为 LLM 推理的 early-stage design space exploration 工具，模拟流程：
```
1. 输入: 模型架构（layers, hidden_dim, num_heads, seq_len, batch_size）
         + 硬件配置（compute FLOPs, memory BW, buffer sizes, interconnect BW）
2. Mapper 搜索: 对每个 operator (GEMM, Attention, etc.)
   - 枚举 tiling parameters (M_tile, K_tile, N_tile)
   - 枚举 schedule (loop order, double buffering)
   - 用 SCALE-Sim 模拟 systolic array 利用率
   - 选 runtime 最短的 mapping
3. Block-level 模拟: 按 Transformer layer 逐 block 模拟
   - Compute time = FLOPs / (utilization × peak FLOPs)
   - Memory time = data_movement / bandwidth（per hierarchy level）
   - Communication time = tokens × hidden_dim / interconnect BW
4. 输出: per-layer latency breakdown, area estimate, performance/power
```
MoE-GPS 增强 LLMCompass 以支持：(a) MoE + Expert Parallelism 的 EP-specific communication（All-to-All scatter/gather）和 FFN workload；(b) Mixtral 架构（GQA, SwiGLU activation, Sliding Window attention 4K）；(c) Prediction strategy modeling（tunable accuracy + overhead, exponential fit for accuracy-overhead curve, polynomial fit for accuracy-performance curve）。

术语一般如何实现？如何使用？
开源 Python 实现，依赖 SCALE-Sim（systolic array simulator）。使用方式：配置 YAML/JSON 描述模型架构和硬件参数 → Mapper 自动搜索最优 tile mapping → Simulator 输出 latency/area 报告。限制：(1) 不支持 FlashAttention（Attention latency 被保守高估）；(2) 面向 throughput-oriented 分析（assume at least one hardware unit saturated），小 batch/short seq 时可能在 prologue/epilogue 和 MMA loop 的 overlap 上欠准确；(3) 论文自身不直接做 GPU kernel 执行模拟，而是 higher-level structural modeling。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing
