## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- 属于Serving调度的实现是什么？实验比较什么？
  - FloE 实现了一个面向内存受限 GPU 的 on-the-fly MoE 推理 Serving 系统，核心调度优化包括：
    1. **Expert Hybrid Compression（Section 3.2）**：将 expert 三组投影矩阵在 DRAM 中压缩存储（gate/down 做上下文稀疏化，up 做 INT2 量化），减少 PCIe 传输量。每 expert 压缩比 9.3×（~300MB → ~32MB）。
    2. **Inter-Expert Sparsity Predictor（Section 3.3.1）**：学习型 MLP 预测器，在当前层 i 计算时预测下一层 i+1 激活的 expert，实现预取。参数规模随层深动态调整（单层 MLP 32K → 双层 MLP 2M）。平均 precision 0.88。
    3. **Intra-Expert Sparsity Predictor（Section 3.3.2）**：参数免费的复用型预测器，用当前层 hidden state 与下一层 W_up 做矩阵乘，预估 up projection 输出激活分布，预计算稀疏掩码。平均 recall 0.95。
    4. **Compact Asynchronous Transfer（Section 3.4.2）**：将 gate 投影的列和 down 投影的行在 DRAM 中紧凑排列（co-locate），提升 chunk 大小从 d_hidden×num_bytes 到 2×d_hidden×num_bytes；使用 AVX-512 SIMD 指令 + 多线程打包到 pinned memory，跨多 CUDA stream 异步传输。
  - 实验比较：DeepSpeed-MII（ZeRO-Infinity offloading，FP16）、Mixtral-Offloading（expert 预测+缓存+INT3 量化）、Fiddler（CPU-GPU 协同计算）、Mixtral-GPU（HQQ INT2 全量 GPU 驻留，作为延迟下界参考）。指标：端到端 TPS（tokens per second），单 expert 计算延迟，传输带宽利用率。

- 硬件平台是什么，配置是什么。
  - GeForce RTX 3090（24GB VRAM），64核 CPU @2.3GHz，256GB DRAM，PCIe 4.0 ×16（峰值带宽 ~32GB/s）。
  - 限制 VRAM 使用量从 12GB 到 24GB 进行消融实验。

- 开源Serving框架是什么。修改了什么。
  - 论文未明确给出 FloE 完整代码仓库。基于 PyTorch 构建，使用 Triton 实现 sparse kernel。核心修改点：
    - expert 权重在 DRAM 中的紧凑布局（co-locate gate 列和 down 行）。
    - 自定义的 pinned memory 管理 + AVX-512 SIMD 多线程异步传输模块。
    - 学习型 inter-expert 预测器（SGD 训练，<1min 收敛）。
    - Triton-based sparse GEMV kernel（替代 PyTorch dense GEMV）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文发表于 ICML 2025。未明确给出开源链接。
  - FloE 推理全流程（单 token 解码为例）：
    1. **输入**：token → embedding → hidden state x（1×4096），非 expert 权重常驻 VRAM。
    2. **第 i 层 MoE 计算**：
       a. Router/Attention 计算在 GPU 上完成，hidden state 传递到 MoE 层。
       b. Inter-expert predictor（在层 i-1 已运行）已预取层 i 的压缩 expert 权重到 VRAM cache。
       c. Intra-expert predictor（在层 i-1 已运行）已预计算 up projection 激活的稀疏掩码。
    3. **Sparse expert 计算**：GPU 执行 sparse GEMV kernel，仅加载选中通道的 gate 列和 down 列。
    4. **预取下一层**：当前层 hidden state 输入 inter-expert predictor（预测层 i+1 激活 expert）+ intra-expert predictor（用层 i+1 复用的 W_up 预估稀疏分布），触发 compact async transfer 从 DRAM 传输层 i+1 压缩 expert 权重。
    5. **输出**：logits → next token。整个过程 transfer 与 computation 流水化，PCIe 传输被 GPU 计算隐藏。
  - 端到端结果：在 RTX 3090 + 12GB VRAM 约束下，FloE 速度达到 Mixtral-GPU（全量 INT2 驻留 GPU）的 91%，对比 DeepSpeed-MII 加速 48.7×。
