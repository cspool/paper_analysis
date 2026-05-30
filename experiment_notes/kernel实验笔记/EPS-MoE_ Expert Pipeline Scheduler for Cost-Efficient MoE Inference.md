## EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  EPS-MoE 在 kernel 级别的核心实现包括：
  1. **Load-aware GEMM 动态切换**：根据输入 token 数 m 动态选择 GroupGemm（cutlass，小 m 时更优）或 DenseGemm（cublas，m≥4096 时更优）。通过 profiling 数据发现：m<2048 时 GroupGemm 效率高于 DenseGemm；m≥4096 时 DenseGemm 反超。
  2. **水平切分（Horizontal Split）输入张量**：对 MoE 输入按行切分，权重按专家切分。相比传统垂直切分（列切分），水平切分避免了参数矩阵的重复内存 I/O。当 pipeline 数 N=E（专家数）时，GroupGemm 退化为 DenseGemm，获得更高计算效率。
  3. **SM 数量控制优化**：控制 GEMM kernel 占用的 SM 数量（如 H800 上从 132 降至 116），留出 SM 资源给 all2all 通信 kernel（16 SM），实现计算与通信在 SM 级别的并行调度。通过 SM 控制，避免了 GEMM 独占所有 SM 导致通信 kernel 无法调度的问题。
  4. **Pipeline 重叠调度**：将 MoE FFN（Gate/Up/Down GEMM）与 all2all 通信以 pipeline 方式重叠，通过分 N 个 pipeline stage 顺序提交专家组计算，每轮计算时下一组通信已在并行执行。

  实验比较：
  - GroupGemm vs DenseGemm 在不同输入规模下的吞吐量对比（图5a-c）
  - 不同 SM 数量下的 GEMM + all2all 重叠延迟（表6，SM 从 92 到 132）
  - 不同 Pipeline Number（PN=1,5,20）的影响（表5、图11）
  - 不同输出维度（1536 vs 15360）下的性能差异
  - FP8 通信 + 重叠策略 vs 纯 FP8 vs 纯重叠（消融表5）
  - 计算 ε0（通信时间变化系数）和 ε1（计算时间变化系数）确定 EPS-MoE 有效工作区间（图10、表7）

- 后端平台是什么，配置是什么。
  - NVIDIA H800-80GB SXM GPU（132 SM，NVLink 互联），4x/8x 配置
  - NVIDIA A100-80GB SXM GPU（用于 profiling 和通信测试）
  - 矩阵计算库：cutlass（GroupGemm）、cublas（DenseGemm）
  - 通信原语：ncclAllReduce、ncclReduceScatter、ncclAllGather、all2all
  - 通信加速：Flux（kernel fusion 优化通信与 GEMM 的融合）

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 real model settings 的 GEMM profiling：
  1. **GEMM 效率 Profiling**（图5）：
     - 测试矩阵维度：Gate/Up 矩阵 [1536, 5120]（与 DeepSeekV2 一致），16 Experts
     - 变输入 m（batch token 数），测试 GroupGemm 和 DenseGemm 的吞吐量
     - 变 GroupGemm 的 group 数和 SM 数，测试相对吞吐量
  2. **Kernel 延迟 Profiling**（图6、7）：
     - 测试 silu_activation（memory-bound kernel）在不同 load 和 SM 数下的延迟
     - 测试 all2all 通信 kernel 在不同 load 和 SM 数下的延迟
  3. **消融实验脚本**（表5、6、7）：
     - 单层 Gate/Up/Down GEMM 的 MoE 层测试（输入 dim=5120, 输出 dim=1536 或 15360）
     - 变 m (256/1024/3072)、PN (1/5/20)、SM (92-132)、FP8 on/off、Overlapping on/off
     - 测试 DeepSeekV2 全模型 prefill throughput（表2）
     - 测试 Mixtral 8x7B / DBRX / Snowflake Arctic 的 TTFT（表3、4）
  4. **修改内容**：
     - 在 vLLM 框架中替换 MoE FFN 的 GEMM 调用路径，根据 load 动态选择 cutlass GroupGemm 或 cublas DenseGemm
     - 实现水平切分输入张量的逻辑，修改 weight partition 方式为按专家切分
     - 添加 SM 控制接口，限制 GEMM kernel 的 CUDA stream 优先级或 MPS 配置
     - 实现 all2all + GEMM pipeline 调度器，管理 token dispatch 和专家计算的流水线

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源代码。评估方法基于 cutlass/cublas 的 GEMM profiling 和 vLLM 集成。
  
  **评估原理和 Kernel 输入到性能输出全过程**：
  
  **阶段 1: GEMM Profiling（独立 kernel 评估）**
  - **输入**：MoE FFN 矩阵维度（如 Gate 矩阵 [1536, 5120]，16 experts），输入 token 数 m∈{256, 1024, 3072, ...}
  - **原理**：对每个 (m, dim, experts, groups) 配置，分别调用 cutlass GroupGemm 和 cublas DenseGemm（cublasGemmEx），测量 wall-clock 延迟。用 Nsight Systems/CUPTI 采集 SM 利用率、内存带宽、计算吞吐。
  - **输出**：各配置下的吞吐量（TFLOPS）、延迟（ms），绘制图5 的效率曲线，确定 GroupGemm vs DenseGemm 的交叉点（约 m=2048~4096）。
  
  **阶段 2: 通信 Profiling**
  - **输入**：all2all 通信数据量（取决于 batch token 数和每个 token 的 activation size）
  - **原理**：调用 ncclAllReduce/ncclReduceScatter/ncclAllGather/all2all，测量不同数据量和 SM 数下的延迟。测试 NVLink 带宽利用率。
  - **输出**：图7 的通信延迟曲线，确定通信 kernel 的 SM 需求（10-20 SM 足够，超过无显著改善）。
  
  **阶段 3: 单层 MoE Pipeline 评估**
  - **输入**：单层 MoE（Gate+Up+Down GEMM，dim [5120, 1536] 或 [5120, 15360]），配置 (PN, SM, FP8, overlapping 开关)
  - **流程**：
    1. 输入 tokens 按行切分为 N 组（N=PN）
    2. 每组 tokens 先进行 all2all dispatch 通信（如使用 FP8 量化则先量化）
    3. 同时启动下一组 all2all 和前一组 GEMM 计算（overlap）
    4. GEMM 根据 m/N 的大小选择 GroupGemm 或 DenseGemm
    5. GEMM SM 限制为 116（留 16 SM 给通信）
    6. 完成 all2all combine 聚合结果
  - **输出**：各配置下的单层延迟（ms），表5/6/7 的消融数据。
  
  **阶段 4: 全模型 Serving 评估**
  - **输入**：完整 MoE 模型（DeepSeekV2/Mixtral/DBRX），请求 batch（变 seqlen 和 batchsize）
  - **流程**：EPS-MoE 集成到 vLLM，端到端推理。对每层 MoE block 应用 expert pipeline scheduler。
  - **输出**：Prefill throughput（tokens/s）和 TTFT（s），表2/3/4。
