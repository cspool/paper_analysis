## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FloE 包含两个 kernel 调度/运行时计算层面的实现：
    1. **Efficient Sparse GEMV Kernel（Section 3.4.1）**：基于 Triton（参考 CATS kernel）实现的自定义稀疏 GEMV kernel。将 W_down 转置为列主序存储（W_down^T），与 gate projection 的列对齐；根据稀疏掩码选择性加载 W_gate 和 W_down^T 的列，减少内存访问次数。将 SiLU 激活和 element-wise 乘法融合到每个 block 计算中，节省中间结果 x' 的多次存储/加载，减少 kernel launch 次数。在 RTX 3090 上，90% 稀疏度时达接近 2× 加速。
    2. **Compact Asynchronous Transfer（Section 3.4.2）**：紧凑权重布局（co-locate gate 列 + down 行到连续 DRAM 区域，chunk 大小翻倍）；CPU 端 AVX-512 SIMD 指令 + 多线程打包压缩权重到 pinned memory；跨多 CUDA stream 异步发送传输请求，最大化 PCIe 带宽利用率。对比 PyTorch 原生实现加速 12.6×，达到 PCIe 4.0 峰值带宽的 88%。
  - 实验比较：单 expert sparse GEMV kernel 延迟 vs dense baseline（sparsity=0），在 H100/A100/A6000/RTX 3090 上对比不同稀疏度（50%/60%/70%/80%/90%）的加速比。传输效率测试：对比不同 chunk size（1~200）下 compact async transfer vs PyTorch 原生的传输延迟和带宽利用率。

- 后端平台是什么，配置是什么。
  - Single-expert kernel 测试：H100（计算吞吐高，但 kernel launch overhead 限制稀疏加速），A100，A6000，GeForce RTX 3090（consumer-grade GPU，稀疏加速最明显）。
  - 传输测试：RTX 3090 + 64核 CPU + 256GB DRAM + PCIe 4.0 ×16。

- 评估性能的软件/脚本是什么。修改了什么。
  - 使用 Triton（Tillet et al., 2019）实现 sparse GEMV kernel，修改自 CATS kernel。
  - 使用 PyTorch 自定义 C++ 扩展实现 compact async transfer（AVX-512 + pinned memory + multi-stream）。
  - 单 expert 延迟测试：C4 数据集 500 tokens，80 warmup + 200 timed iterations。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确给出 FloE 开源链接。sparse GEMV kernel 基于 Triton 构建，可参考 Triton（https://github.com/triton-lang/triton）和 CATS（https://openreview.net/forum?id=v3w2a7EInO）。
  - Sparse GEMV kernel 评估原理：
    1. **输入**：hidden state x (1×4096)，sparse threshold t_ij，expert weights {W_gate (4096×14336), W_down^T (4096×14336, 列主序转置), W_up (4096×14336)}。
    2. **Kernel 执行流程**：x 与 W_up 全精度 GEMV → 产生激活向量 v (1×14336) → |v| 与 t_ij 比较生成 binary mask → GPU 根据 mask 选择性从 global memory 加载 W_gate 和 W_down^T 被选中的列（列宽 = d_hidden=4096）→ Triton block 内融合执行 SiLU + element-wise multiply + sparse GEMV → 输出结果 y (1×14336)。
    3. **性能测量原理**：使用 CUDA event 计时，warmup 80 次后 200 次 timed iteration 取平均。对比 dense baseline（sparsity=0，全量加载全量计算）的 wall-clock 延迟（ms）。利用 Nsight Systems 验证 PCIe 带宽利用率和 kernel 时间线。
    4. **传输效率评估原理**：随机选取 20% expert 权重列，从 DRAM 通过不同 chunk size 传输到 VRAM。每个 chunk 对应一个 CPU 线程 + pinned memory 打包 + CUDA stream 异步拷贝。使用 PyTorch CUDA event 和 CPU timer 测量传输延迟，计算实际带宽 = transferred_bytes / latency，除以 PCIe 4.0 峰值理论带宽（~25GB/s 实测上限）得到利用率。
