## MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现两大类自定义 CUDA kernel：(1) **高效 scatter/gather 算子**——基于 token routing 结果预计算 input→output 行映射表，直接通过 CUDA 执行数据传输，替代 Megatron-LM 使用的 torch.scatter_add / torch.gather；(2) **Intra-operator 通信-计算融合 kernel**——将通信操作以 tile 粒度嵌入 GEMM/GroupedGEMM 计算 kernel 中，使用 device memory barrier 实现 tile 级同步（无需 host CPU 干预）。
  - 四类 fused kernel：
    a. **A2A+GEMM**（SP attention Output Projection）：all-to-all 通信与 GEMM 同时启动，remote data tile 到达后通过 signal 通知 GEMM kernel 继续计算该 tile。使用 GPU copy engine 处理通信，SM 全部用于计算。
    b. **GEMM+A2A**（SP attention QKV Projection）：all-to-all 嵌入 GEMM kernel，每个 GEMM tile 计算完成后直接发起 remote data transfer 写入目标 rank。
    c. **AG+Scatter+GroupedGEMM**（FFN token dispatch）：对 token 按 routed expert index → source rank index 排序，使每个 computation tile 依赖尽可能少的 source rank。将 local scatter 融合进 kernel（按 index mapping 选择输入行），GroupedGEMM 按 tile 分块执行。
    d. **GroupedGEMM+Gather+RS**（FFN token combine）：类似 (c) 的逆过程。
  - 实验比较：ablation study 中逐步开启 intra-operator overlap（Table 5，+6% 吞吐）。六种模型下 fused vs non-fused 通信+计算总时间对比（Figure 16，1.2-4.7× 减少）。

- 后端平台是什么，配置是什么。
  - NVIDIA H800 SXM GPU（989 TFLOPS compute, 80 GB HBM, 3.4 TB/s memory BW, 400 GB/s NVLink），intra-node 8 GPU via NVLink。
  - 也测试 A100（312 TFLOPS, NVLink 600 GB/s）和 H20（148 TFLOPS, NVLink 900 GB/s）。
  - CUDA 编程模型，使用 device memory barrier + GPU copy engine。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 Megatron-LM（commit f1f03922）构建。MegaScale-MoE 将 MoE layer 的 attention 和 FFN 模块分解为独立的 GPU kernel 算子（而非 Megatron-LM 中依赖 torch.autograd 的 monolithic 执行），实现细粒度调度。
  - 修改包括：(1) 自定义 CUDA scatter/gather 替代 PyTorch 内置算子；(2) 实现四种 fused communication-computation kernel；(3) SM allocation tuning：为 A2A+GEMM 模式中的通信分配少量 SM（tuned to match computation latency via swizzling 重排 tile 顺序避免 NVLink contention）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未提供开源仓库链接；基于 Megatron-LM 开源（github.com/NVIDIA/Megatron-LM）。
  - Intra-operator overlap kernel 执行原理（以 AG+Scatter+GroupedGEMM 为例）：
    1. 输入：token hidden states [b*s, h] + routing decisions (每个 token 的 expert assignment)
    2. Token 排序：按 routed expert index 排序所有 token（使连续 token 属同一 expert），再按 source rank index 二次排序（使同一 expert 内同源 rank token 连续）
    3. 切分为 computation tiles：排序后的 token 序列按固定 tile size 切片
    4. 并行执行：每个 tile 启动 GroupedGEMM 前检查所需 source rank 的数据是否已到达（device memory barrier polling）。仅依赖本地/已到达 rank 的 tile 可立即开始计算
    5. Scatter 内联：通过预计算的 row index mapping 直接选取输入矩阵对应行，无额外 kernel launch
    6. 输出：Expert FFN 输出按 tile 组装回 token 维度，进入 gather + reduce-scatter 阶段
  - SM allocation 策略：通信分配少量 SM（数量 tuned 使 comm≈comp latency），其余 SM 用于 GEMM。Swizzling 重排 tile 顺序使各 rank 的 remote data 到达节奏与 computation tile 消费节奏对齐，降低 NVLink contention。
