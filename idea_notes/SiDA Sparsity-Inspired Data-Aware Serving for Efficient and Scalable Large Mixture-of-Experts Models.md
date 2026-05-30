## SiDA Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models

- baseline方法是什么？
  Baseline方法：(1) **Standard MoE 推理**：直接使用 HuggingFace Transformers 中的 Switch Transformer 实现，所有 expert 参数常驻 GPU 内存，每次前向通过 router 函数（线性层 + SoftMax + top-k）在线确定激活的 expert 并调用。存在三个核心问题：①低效 GPU 内存利用——大部分 expert 参数在推理中闲置（Switch-base-256 上 MoE 参数占模型 99.07%，但仅 <20% 的 expert 被激活，短句场景下浪费高达 50GB GPU 内存）；②高 MoE 开销——expert 选择、expert 调用和通信开销占据高达 72% 的总推理时间（Switch-base-256），且随模型规模放大；③router 在线选择的延迟惩罚——在小 batch 场景下，调用 expert 的开销超过计算本身。(2) **DeepSpeed-MoE** 和 **Tutel**：优化了设备间通信、自适应并行和流水线调度，但均未利用数据感知（data-awareness）来进一步提升内存效率，所有 expert 仍在 GPU 上，无效 GPU 内存利用问题未解决。
  全栈执行例子（Baseline: Standard Switch-base-256 推理，单 A100 80GB，SST2 数据集）：
  - **算法Pipeline层**：token embedding → self-attention → router（W_r^T x → SoftMax → top-1 expert选择）→ 调用选中的 expert MLP → α 加权输出。所有 256 个 expert 的 MLP 参数（54.114 GB）全部提前加载到 GPU 内存。
  - **系统框架层**：HuggingFace Transformers 默认实现——每个 MoE 层调用所有 expert（即使无 token 分配到该 expert），以对齐高效计算的硬件要求。无任何 CPU-GPU 之间的参数 offloading。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明——默认使用 PyTorch GEMM kernel，无 expert-specific kernel 优化。
  - **硬件架构层**：A100 80GB GPU，所有数据流经 GPU HBM ↔ SM 的片上路径，无 CPU-GPU 数据交换，但 GPU 内存利用率极低（<5% for short sentences）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **SiDA-MoE** 提出 **数据感知（data-aware）的稀疏驱动推理系统**，核心机制是通过离线训练的 hash 函数预先预测 expert 激活模式，使推理系统能提前知晓哪些 expert 将在当前 batch 中被激活，从而在推理开始前完成 expert 的动态加载/卸载。解决 baseline 缺陷的方式如下：
  
  **(1) Hash-building 线程解决 MoE 开销瓶颈**：用离线训练的 LSTM+sparse attention hash 函数替代在线 router 选择。Hash 函数在独立 CPU 线程中运行，预测每批输入在所有 MoE 层的 expert 激活模式。由此将 expert 选择从推理关键路径中移除——吞吐量提升达 3.93×，延迟降至 28%。这是针对 baseline 中"expert 选择占据 72% 推理时间"的直接解决方案。
  
  **(2) 动态 offloading 解决低效 GPU 内存利用**：Inference 线程根据 hash table 仅将当前批激活的 expert 加载到 GPU，将未激活的 expert 卸载到 CPU 主内存（FIFO 驱逐策略）。利用现代服务器 CPU 可达 TB 级内存容量的特性，使 GPU 只存放当前有效的参数——GPU 内存节省达 80%（SST2），大模型（Switch-base-256）在长句场景（MultiRC）仍节省 20%+。
  
  **(3) 双线程管道并行实现零开销调度**：Inference 线程和 Hash-building 线程并行运行——推理线程处理 batch X_i 时，hash-building 线程已在预测 batch X_{i+1} 的 expert 模式。由于推理耗时远大于 hash 预测，hash table 始终在推理需要时已就绪，两条线程无空闲等待。此设计使 expert 选择、动态 offloading 和模型计算三者完全重叠并行。
  
  **(4) Sparse Attention + Truncated KD 确保预测精度**：Hash 函数虽轻量但需要高精度——Sparse Attention（SparseMax 激活）使预测器自动关注少数关键 token，匹配实验中发现的稀疏跨 embedding 依赖（c^i ≈ 1-4 个关键 token 影响 expert 激活）；Truncated KD（T=30）配合交叉熵损失使预测器在容量受限条件下仍达到 Top-3 准确率 >99%（SST2）。
  全栈执行例子（SiDA-MoE，Switch-base-256，单 A100 80GB，SST2 数据集）：
  - **算法Pipeline层**：Hash-building 线程（CPU）：token embedding → 2层LSTM → Self-Attention(SparseMax, 仅关注 c^i≈1-4 个关键token) → FC压缩 → Residual → FC → top-1 expert预测 + scaling factor α。Inference 线程（GPU）：跳过 router，直接根据 hash table 的 (expert_id, α) 调用对应 expert MLP，α 加权输出。
  - **系统框架层**：HuggingFace Transformers + SiDA-MoE Manager——维护双线程协调（shared queue 传递 hash table），管理 expert 设备置放（GPU HBM ⇄ CPU DDR4 主内存），FIFO 驱逐策略。每层 MoE 完成后流水线触发下一层的 expert 加载/卸载。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：CPU-GPU 之间的 expert 参数传输（PCIe），GPU 内使用默认 PyTorch kernel。论文未引入新的 CUDA kernel。
  - **硬件架构层**：A100 80GB GPU + CPU 主内存（TB 级 DDR4）。数据流：CPU 主存 → PCIe → GPU HBM（加载激活 expert），GPU HBM → PCIe → CPU 主存（卸载未激活 expert）。80% 的 expert 参数（~43GB for Switch-base-256）存在 CPU 侧，仅在激活时才传输到 GPU。
