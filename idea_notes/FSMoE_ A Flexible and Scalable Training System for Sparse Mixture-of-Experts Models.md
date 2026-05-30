## FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

- baseline方法是什么？
  - **Baseline 1: Tutel (w/ PipeMoE)**：Tutel 是专用的 MoE 训练系统，PipeMoE 为其优化版本，通过自适应调度确定流水线度（pipeline degree），将 AlltoAll 通信与专家计算重叠。但存在以下局限：
    1. 仅支持有限的 routing function，对新路由机制不灵活。
    2. 前向和反向使用相同的流水线度，未考虑二者计算量差异。
    3. 仅重叠 AlltoAll 与 expert computation，不探索 ESP-AllGather/ESP-ReduceScatter（节点内通信）与 AlltoAll（节点间通信）之间的重叠。
    4. Gradient-AllReduce 仅与 non-MoE 部分重叠，未与 MoE 层协同设计。
  - **Baseline 2: DeepSpeed-MoE**：专用 MoE 训练系统，支持 EP 和 ESP，但调度能力更弱（手动配置或不进行自适应调度）。
  - 全栈执行例子（以 Mixtral-7B、Testbed-A、48 GPU、EP=6, ESP=8, MP=8 为例）：
    - **模型推理算法层**：Mixtral-7B decoder-only Transformer，MoE 层使用 top-2 GShard gate routing，8 experts，每个 expert 为 Mixtral FFN（SwiGLU）。前向时 tokens 通过 gate 分派到 top-2 experts，反向时计算 expert 权重梯度和输入梯度。
    - **系统框架层**：Tutel/DeepSpeed-MoE 基于 PyTorch 实现 DP+MP+EP+ESP 混合并行。Tutel 使用统一流水线度 r 切分输入 token，在 CUDA stream 上重叠 AlltoAll Dispatch/Combine 与 expert GEMM 计算。Gradient-AllReduce 在 MoE 层完成后执行，仅与 attention 等非 MoE 部分重叠。节点内 ESP-AllGather/ESP-ReduceScatter 与节点间 AlltoAll 串行执行，无重叠。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：NCCL 2.12 提供 AlltoAll/AllGather/ReduceScatter/AllReduce 集合通信 kernel。PyTorch torch.matmul 提供 GEMM kernel。Tutel/PipeMoE 通过 PyTorch CUDA stream 机制调度通信 kernel 与计算 kernel 的 overlap。
    - **硬件架构层**：Testbed-A: 6 节点 × 8×NVIDIA RTX A6000 (48GB)，节点内 NVLink 112.5GB/s，节点间 200Gb/s InfiniBand。N_MP=N_ESP=8（对齐节点内 GPU 数），N_EP=6（等于节点数）。
  - **Baseline 痛点**：
    1. **路由函数不灵活**：Tutel 和 DeepSpeed-MoE 仅支持有限的 routing function，新增路由机制需要大量侵入式修改。
    2. **节点内/节点间通信无重叠**：节点内 ESP-AllGather/ESP-ReduceScatter（NVLink）与节点间 AlltoAll（InfiniBand）完全串行，浪费了 NVLink 的高带宽（900GB/s vs 100GB/s InfiniBand on DGX H100）。
    3. **前向/反向统一流水线度不最优**：912/1458 配置下前反向最优度不同（反向计算量约为前向 2 倍），统一度导致性能次优。
    4. **Gradient-AllReduce 未与 MoE 层协同设计**：Gradient-AllReduce 和 AlltoAll 均为节点间通信，不重叠时 Gradient-AllReduce 成为额外延迟开销。现有方案（PipeMoE、Lina）要么仅与非 MoE 部分重叠，要么使用固定 chunk size 无法适应多变配置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FSMoE 方法**：通过三大技术解决 baseline 的四个痛点：
    1. **MoE 模块化与统一抽象**（解决痛点 1）：将 MoE 层分解为 Gate、Order、I-Order、Dispatch、Combine、Expert 六个独立子模块，每个模块可被替换/扩展。预实现 4 种路由函数（GShard、Sigmoid、X-MoE、Expert Choice）和 4 种 AlltoAll 算法（NCCL-A2A、1DH-A2A、2DH-A2A）。通过 Hook 机制（BeforeMoeStartHook, BeforeDispatchHook 等）实现非侵入式扩展。
    2. **节点内/节点间通信协同调度 + 前向/反向分别优化**（解决痛点 2 和 3）：在 MP/ESP 对齐节点内 GPU 数时，节点内通信（ESP-AllGather/ESP-ReduceScatter）和节点间通信（AlltoAll）可通过流水线重叠。将调度场景分为 4 种 Case（Case1: 节点间通信主导；Case2: 计算主导；Case3: AlltoAll 主导；Case4: 节点内通信主导），通过线性性能模型建模各操作耗时，SLSQP 求解器分别优化前向和反向的最优流水线度 r_fwd 和 r_bwd。
    3. **自适应梯度分区**（解决痛点 4）：两阶段算法将 Gradient-AllReduce 的梯度分配到各 MoE 层的 overlappable parts 中。Step 1 贪心将梯度分配到各层的空闲时间段；Step 2 用差分进化算法优化剩余梯度的跨层分配，实现 Gradient-AllReduce 与 MoE 层通信/计算的最大重叠。

  - 全栈执行例子（与 baseline 同配置，FSMoE pipeline degree r=4）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-7B, top-2 gate），不改变模型结构或收敛性。额外支持 4 种路由函数的即插即用切换。
    - **系统框架层**：FSMoE 基于 PyTorch + C/C++/CUDA 扩展实现，替代 Tutel/DeepSpeed-MoE 的 MoE 层实现。在 DP+MP+EP+ESP 混合并行下，输入 token 按 r=4 切分，每个 chunk 依次经过：ESP-AllGather (intra-node) → AlltoAll Dispatch (inter-node) → ESP-ReduceScatter (intra-node) → Expert Compute → ESP-AllGather (intra-node) → AlltoAll Combine (inter-node) → ESP-ReduceScatter (intra-node)。**节点内通信与节点间通信在不同 chunk 上并行执行**（chunk i 的 ESP-AllGather 与 chunk i-1 的 AlltoAll 重叠）。反向额外将 Gradient-AllReduce 的梯度按 overlappable parts 自适应分配到各层，与最后一个 chunk 的 ESP-AllGather/ReduceScatter 和 expert 计算重叠。前向 r_fwd 和反向 r_bwd 独立优化。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：NCCL 2.12 集合通信 kernel（AlltoAll/AllGather/ReduceScatter/AllReduce）与 PyTorch GEMM kernel 在分离 CUDA stream 上调度。FSMoE 的在线 profiler 使用 nccl-tests 和 torch.matmul 微基准测量 α/β 参数（拟合 R² > 0.99），一次拟合 <10ms，SLSQP 求解平均 193ms。调度算法 O(1) 复杂度，训练前执行一次。
    - **硬件架构层**：与 baseline 相同（Testbed-A: 48×A6000）。对比 Tutel：1.18×–1.22× 加速（1458 配置层），对比 DeepSpeed-MoE：1.28×–3.01× 加速（真实模型）。在 Mixtral-7B 上，节点内/节点间通信重叠贡献约 5–6% 额外加速（FSMoE vs FSMoE-No-IIO），梯度分区贡献约 5–7%（FSMoE vs Tutel-Improved）。
