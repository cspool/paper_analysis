## Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  FlashOverlap 属于 kernel 调度/运行时计算层优化。实现包括：(i) 在 CUTLASS 模板 GEMM kernel 的 epilogue 中融合 signaling 机制——通过原子计数器追踪 tile 完成进度，当 wave group 内所有 tile 完成后发送信号触发通信；(ii) wave-based signaling timing——利用 GEMM 执行的 wave pattern（多个 tile 几乎同时完成），以 wave 而非单个 tile 为重叠单位提升带宽利用率；(iii) 可调 wave grouping——将多个 wave 组合为 group，在重叠机会和通信分段之间权衡；(iv) pre/post-communication reordering——pre-communication reordering 将非连续地址的 tile 重排为连续地址以直接调用 NCCL API，post-communication reordering 在通信后恢复数据顺序；(v) predictive search tuning——通过剪枝+延迟预测器实时搜索最优 wave group partition。
  实验比较：(i) 算子级：FlashOverlap vs non-overlap baseline、decomposition-based baseline（Async-TP、VanillaDecomposition）、fusion-based baseline（FLUX），覆盖 GEMM+AR、GEMM+RS、GEMM+A2A 三种通信原语，每种在 50+ GEMM sizes 下测试；(ii) 端到端：LLM 推理（Llama3-70B TP=8 on vLLM）、LLM 训练（Mixtral-8x7B EP=4/TP=2、Llama3-70B TP=8 on Megatron-LM）、T2V 生成（Step-Video-T2V TP=4 on xDiT）；(iii) 消融实验：固定大小 grouping vs 等大小 grouping vs FlashOverlap 搜索；(iv) 预测搜索准确性 vs 穷举搜索；(v) 华为 Ascend 910B NPU 跨平台验证。

- 后端平台是什么，配置是什么。
  - NVIDIA A800 GPU（NVLink 互联，pairwise 连接，1935GB/s HBM 带宽，312 TFLOPS FP16）——主要用于端到端评估
  - NVIDIA RTX 4090 GPU（PCIe 互联，跨 NUMA 节点，1008GB/s HBM 带宽，330 TFLOPS FP16）——用于算子级评估
  - 华为 Ascend 910B NPU——用于跨平台适配验证
  - 软件环境：CUDA 12.1、NCCL 2.19.3、PyTorch 2.5.1、CUTLASS 3.6.0
  - A800 服务器 pairwise NVLink；RTX 4090 服务器 PCIe 4.0 穿越 NUMA

- 评估性能的软件/脚本是什么。修改了什么。
  基于 CUTLASS 3.6.0 的模板 GEMM 实现。修改包括：(i) GEMM kernel epilogue 中融合 pre-communication reordering——将 tile 输出按执行顺序重排为连续地址（tile/subtile/subtoken 级粒度）；(ii) 添加 counting table（原子计数器，size=P 对应 P 个 group）追踪 tile 完成；(iii) signaling kernel——独立于 GEMM 在另一 CUDA stream 上运行，周期性查询 counting table，达到 group size 后触发 NCCL 通信；(iv) post-communication reordering 融合到后续 element-wise kernel（如 RMSNorm）中恢复数据顺序；(v) CUDA stream 管理——GEMM 在 stream A，signaling + 通信在 stream B，实现并发执行。
  评估脚本：Artifact repo 提供 e1_correctness.py（正确性验证）、e1_speedup.py（加速比测量）、e1_compare.py（与 SOTA 对比）、e2_predictive_search.py（预测搜索准确性）、e3_rmsnorm_overhead.py/e3_gemm_overhead.py（开销测量）。端到端评估通过替换 vLLM/Megatron-LM/xDiT 中的原始 linear layer + 通信为 FlashOverlap 实现完成。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/infinigence/FlashOverlap（ae 分支），Zenodo DOI: 10.5281/zenodo.17201530。

  评估原理与流程（以 GEMM+AllReduce on 4×RTX 4090 为例）：
  1. **Preparation**：运行 evaluation/preparation.py → profiler 对目标 GEMM size (M×N×K) 和 GPU 执行 CUTLASS profiler，获取最优 GEMM 配置（tile size、swizzling pattern、wave 数、duration）；通过多轮通信采样构建 (data_size, bandwidth) 曲线；确定通信原语占用的 SM 数 → 更新 wave 数 T。
  2. **Online Tuning**：对 GEMM size 生成 wave group partition 候选（剪枝后设计空间），对每个 candidate 运行 Alg.1 的 latency predictor：遍历每个 group G_i → 计算 computation latency t_p = GEMM_duration/T × |G_i| → 根据 data_size 插值 bandwidth curve 得到 communication latency t_m → 累加 t_m^acc = max(t_p^acc, t_m^acc) + t_m → 选择最小 t_m^acc 的 partition。
  3. **Execution**：创建两个 CUDA stream。Stream A 发射 GEMM kernel（含 fused pre-communication reordering 在 epilogue 中）。GEMM kernel 每个 tile 完成时通过 atomicAdd 更新 counting table。Stream B 发射 signaling kernel——以 busy-wait 周期性读取 counting table，当 group G_j 计数达到 |G_j| 时，调用 NCCL API（如 ncclAllReduce）对已重排的连续数据 buffer 执行通信。通信完成后，post-communication reordering kernel 根据 mapping table 将数据恢复为原始顺序。
  4. **Measurement**：CUDA event 记录 GEMM launch 到通信完成的 total latency。Speedup = non_overlap_latency / overlap_latency。Non-overlap baseline 为顺序执行 cuBLAS GEMM + NCCL 通信。
  5. **Post-communication reordering 硬件流**：对于 AllReduce，tile 在 wave 内任意顺序均可（只需所有 GPU 保持一致）；对于 ReduceScatter，tile 按 row 拆分为 subtile（每个 GPU 对应一个 subtile），通信后通过 local row exchange 纠正顺序；对于 All-to-All，tile 按 token(row) 拆分为 subtoken，各 destination GPU 有独立 memory pool，subtoken 在 pool 内按执行顺序重排。
  6. **Output**：terminal 输出 speedup table（最多 1.65× on RTX 4090, 1.30× on A800）。
