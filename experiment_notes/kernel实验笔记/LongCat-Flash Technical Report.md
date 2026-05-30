## LongCat-Flash Technical Report

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - LongCat-Flash 在训练和推理侧均实现了大量定制 kernel：
    **训练侧 Kernel**：
    1. **Deterministic FlashAttention Gradient (FAG)**：默认 FAG 使用原子加法的非确定性归约。通过有限 extra workspace 以确定性顺序累积 tile，配合 double-buffer pipelining、tuned tiling schedules、load balancing 实现确定性和性能兼顾。性能达到原始确定性版本的 1.6x，非确定性版本的 0.95x。
    2. **Deterministic ScatterAdd**：默认实现因输入输出 operand count 不匹配，强制单 compute unit 串行执行导致 50x 减速。提出 hierarchical reduction algorithm，将所有可用 processors 并行化梯度聚合，性能与非确定性版本持平。
    3. **Optimized Grouped GEMM**：Grouped GEMM 计算量大但 compute density 低。三项优化：(a) Double-buffer pipelining 重叠计算、内存 I/O 和 epilogue；(b) Diagonal tiling 缓解 L2 cache 冲突；(c) HBM bandwidth control 通过限制 compute unit 使 Grouped GEMM 与 dispatch/combine 通信重叠。综合加速 5%-45%。
    4. **Fused GemmAdd**：dw 梯度累积时带宽受限。将 FP32 addition 融合到 GEMM epilogue 中，消除中间写回并在 tile GEMM pipeline 内完成加法。避免 BF16→HBM 转换精度损失，加速比 3.12x-3.86x。
    5. **IO-bound Kernel 重实现**：MoE layer permute/unpermute kernel 集成 drop-token 和 zero-computation experts 处理，保证确定性和性能。
    **推理侧 Kernel**：
    6. **MoE GEMM with SwapAB**：传统 MoE GEMM 以 token activations 为左矩阵（M×K）、expert weights 为右矩阵（K×N），M 维度需 64 元素最小对齐需 padding。SwapAB 反转映射——weights 为左矩阵、activations 为右矩阵——利用 N 维度的 8 元素粒度灵活填充，最大化 tensor core 利用率。
    7. **Custom Communication Kernels (NVLink Sharp)**：使用 PTX 内联汇编直接调用 NVLink Sharp 的 multimem.st（broadcast）和 multimem.ld_reduce（in-switch reduction），实现 reduce-scatter 和 all-gather。支持均匀和非均匀 token 分布，比 NCCL 和 MSCCL++ 更快（4KB-96MB message size 全范围），仅需 4 thread blocks。
  - 实验比较：
    - Deterministic FAG: 1.6x vs original deterministic, 0.95x vs non-deterministic
    - Deterministic ScatterAdd: 消除 50x 减速，达到与非确定性版本性能持平
    - Grouped GEMM: 5%-45% speedup over default
    - Fused GemmAdd: 3.12x-3.86x speedup
    - Communication Kernels: 比 NCCL 和 MSCCL++ 更快（4KB-96MB 全范围）
    - LongCat-Flash vs DeepSeek-V3 deployment performance (Table 6)

- 后端平台是什么，配置是什么。
  - **训练**：NVIDIA H800-80GB，tens of thousands accelerators，200Gb/s RDMA per accelerator，NVLink intra-node。
  - **推理**：NVIDIA H800-80GB (SXM5)，NVLink intra-node + RDMA inter-node (GPUDirect RDMA)。FlashMLA 可达 660 TFlops（H800 SXM5），DeepEP 带宽可达 40GB/s。FP8 量化推理支持。

- 评估性能的软件/脚本是什么。修改了什么。
  - 论文未明确提供具体的 kernel benchmark scripts 路径，但在架构部分详细描述了各 kernel 的设计和性能对比。
  - **训练侧**：kernel 集成在训练框架中。SDC 检测通过在 FlashAttention gradient backward 中嵌入 on-chip in-place recomputation 验证 bitwise 一致性。
  - **推理侧**：
    - DeepEP (https://github.com/deepseek-ai/DeepEP)：修改支持 zero-computation experts（zero-comp expert 输出无需通信）
    - EPLB：修改支持 zero-computation experts 的负载均衡
    - FlashMLA (https://github.com/deepseek-ai/FlashMLA)：用于 MLA kernel 性能参考
    - DeepGEMM (https://github.com/deepseek-ai/DeepGEMM)：用于 MoE GEMM kernel 性能参考
  - 修改/新增内容：
    1. SwapAB MoE GEMM kernel：权重/激活矩阵角色互换的 GEMM 实现
    2. NVLink Sharp PTX kernels：reduce-scatter 和 all-gather 的直接 PTX 实现
    3. TVD fused CUDA graph：将 Target forward + Verification + Draft forward 三个运算融合
    4. Multi-step overlapped scheduler 的 KV cache pre-allocation 逻辑

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：模型和部分代码在 GitHub (https://github.com/meituan-longcat) 开源。使用的开源组件：FlashMLA、DeepEP（修改版）、DeepGEMM、NCCL、MSCCL++。
  - **Deterministic FAG kernel 原理**：
    ```
    [输入] Q, K, V, dO 及其梯度（BF16 tensors）
    [Kernel 执行流程]
    1. 将 dQ, dK, dV 计算按 tile 维度划分为多个 tile
    2. 使用 extra workspace 按确定性顺序累积各 tile 的梯度
       - 替代默认的 atomicAdd（非确定性归约顺序）
       - Workspace 存储各 tile 的部分结果
    3. Double-buffer pipelining: 当前 tile 计算与上一 tile 结果写入重叠
    4. Tuned tiling: 按 H800 SM 数量和 shared memory 大小优化 tile 尺寸
    5. Load balancing: 在各 SM 间均匀分配 tile 计算量
    [输出] 确定性的 dQ, dK, dV（bitwise 一致的梯度）
    [性能] 1.6x vs naive deterministic, 0.95x vs non-deterministic
    ```
  - **MoE GEMM with SwapAB 原理**：
    ```
    [传统 MoE GEMM] C = A × B
      - A: activations [m=token_count, k=expert_dim] → m 需 padding 到 64 对齐
      - B: weights [k=expert_dim, n=intermediate_dim]
    
    [SwapAB MoE GEMM] C^T = B^T × A^T
      - B^T: weights transpose [n=intermediate_dim, k=expert_dim] → 作为左矩阵
      - A^T: activations transpose [k=expert_dim, m=token_count] → 作为右矩阵
      - n 维度具有 8 元素对齐粒度（vs m 维度的 64 元素），padding overhead 显著降低
    
    [输入到输出]:
    Input: activations [m, k] (BF16/FP8) + weights [k, n] (BF16/FP8)
    → Swap: 内存 reinterpretation 替代物理转置
    → Tensor Core GEMM: 波前级别的 tile 并行
    → Output: [m, n] (BF16/FP8)
    ```
  - **NVLink Sharp Communication Kernel 原理**：
    ```
    [All-Gather via multimem.st]
    1. 每个 GPU 持有部分数据
    2. inline PTX: multimem.st 指令 → NVSwitch 硬件广播各 GPU 数据到所有参与者
    3. 结果: 所有 GPU 获得完整数据副本，仅 4 thread blocks 驱动
    
    [Reduce-Scatter via multimem.ld_reduce]
    1. 每个 GPU 持有完整数据的不同部分
    2. inline PTX: multimem.ld_reduce 指令 → NVSwitch 在交换过程中执行 in-switch reduction
    3. 结果: 各 GPU 获得规约后的分片数据
    [性能] 4KB-96MB message size 全范围超越 NCCL 和 MSCCL++
    ```
