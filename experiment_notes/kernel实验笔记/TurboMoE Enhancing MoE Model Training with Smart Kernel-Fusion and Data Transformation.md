## TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：开发 **Triton kernel** 高效实现 Expert Group Approximation 的计算逻辑。核心 kernel 为 "Router backward" kernel，用于计算传递给路由器的近似稠密梯度。此外还有用于构造专家分组近似和聚合的 Triton kernel。这些 kernel 负责在 GPU 上高效执行 $N^2$ 个专家组近似的构造、token 分组、以及梯度聚合操作。
  - 实验比较：(a) Throughput 对比——不同 hidden dim（1024/2048/4096）下 Expert Group Approx. vs TopK vs Sparsemixer 的 tokens/sec 和 overhead 百分比（Table 4）；(b) CUDA 时间开销 scaling——随 hidden size 增大，各 kernel 组件的 CUDA 时间占比变化（Figure 9），overhead 从 1024-dim 的 13.32% 降至 4096-dim 的 1.57%。

- 后端平台是什么，配置是什么。
  - NVIDIA GPU（具体型号论文未明确说明）。单 GPU 用于 throughput 测量和可复现性分析。

- 评估性能的软件/脚本是什么。修改了什么。
  - GPT-NeoX 训练框架 + Megablocks 稀疏训练库。修改内容：(1) 在 MoE 层的反向传播中插入 Expert Group Approximation 计算；(2) 用自定义 Triton kernel 替代原生 PyTorch 操作实现 token 分组、近似构造和梯度注入；(3) 添加跨数据并行 worker 的 all-reduce 通信以聚合近似梯度。
  - 评估指标：tokens/sec、TFLOPS、CUDA time breakdown。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源状态：论文声明开源但未提供具体链接（匿名投稿）。基于 GPT-NeoX + Megablocks 开源框架实现。
  - Kernel 输入到性能输出全过程：
    1. **输入**：每个 MoE 层的 token 嵌入 x（shape: [batch_size, seq_len, d_token]）、路由器权重 W、专家参数 E_1..E_N
    2. **Triton "Router backward" kernel**：对所有 $\binom{N}{K}$ 个路由决策分组，在每组内计算 $\hat{E}_i(x)$ 近似，聚合到稠密梯度向量 ∂y/∂π
    3. **All-reduce 通信**：跨数据并行 workers 聚合近似梯度，利用大批量样本估计稠密梯度
    4. **梯度注入**：通过 stop-gradient 机制将近似梯度注入计算图，更新路由器和专家权重
    5. **输出**：更新后的路由器权重 W 和专家参数 E_1..E_N，CUDA time profiling 记录各 kernel 耗时分布<br>
  - Overhead 分析原理：通过记录 MoE 层总的 CUDA 时间中 expert MLP matmul、Router backward kernel 等各自占比，展示随 hidden size 增大（1024→2048→4096），matmul 时间占比增大使得方法 overhead 占比下降。
