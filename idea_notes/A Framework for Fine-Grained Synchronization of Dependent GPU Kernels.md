## A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

- baseline方法是什么？
  **Stream Synchronization（CUDA Stream 同步）**：将两个有依赖关系的 CUDA kernel 发射到同一个 CUDA stream 上。CUDA runtime 保证同一 stream 上的操作按发射顺序执行，因此 consumer kernel 的所有 thread block 必须在 producer kernel 的**所有** thread block 完成后才能开始执行。

  全栈执行例子（以 MegatronLM GPT-3 MLP 的两个依赖 GeMM 为例，Batch=256，V100 80 SM）：
  - **模型推理算法层**：MLP 执行 XW₁ = GeLU(X × W₁)，然后 XW₁₂ = XW₁ × W₂。两个 GeMM 串行依赖。
  - **系统框架层**：PyTorch 调用 CUTLASS GeMM kernel，两个 kernel 在同一 CUDA stream 上发射。论文未明确说明上层 serving 框架修改。
  - **编译框架层**：论文未明确说明编译框架层修改，使用标准 nvcc 编译。
  - **kernel调度层**：Producer GeMM 的 grid=[1,48,4]（192 thread blocks），Consumer GeMM 的 grid=[1,96,2]（192 thread blocks）。两者均需 ceil(192/80)=3 wave。Stream 同步要求 producer 的 3 个 wave 全部完成后，consumer 的 3 个 wave 才能开始。最后每个 kernel 的 partial wave（第 3 波只执行 32 个 thread block）仅利用 40% SM，两个 kernel 共浪费 2×48=96 SM 时隙。
  - **硬件架构层**：80 个 SM 上，每个 thread block 占 1 个 SM。每 wave 中空闲的 SM 产生 bubbles。论文未明确说明硬件架构层自定义修改。

  Baseline 问题：GPU 利用率低（该例仅 60%），因为 partial wave 中 thread block 数不是 SM 数（× occupancy）的整数倍，且 stream 同步强制 kernel 间完全串行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **cuSync 细粒度 tile 级同步**：将依赖关系从 kernel 级别下推到 tile 级别。Producer 和 consumer kernel 发射到**不同** CUDA stream 上，通过 global memory semaphore 仅在依赖的 tile 之间同步。Independent tiles 可以并发执行。

  全栈执行例子（同样 MLP，Batch=256，V100 80 SM，使用 TileSync+WRT 策略）：
  - **模型推理算法层**：同上 MLP 计算逻辑不变。
  - **系统框架层**：cuSync header-only 库替代 stream 同步，cuSyncGen DSL 描述 tile 间依赖（如 consumer tile (x,y) 依赖所有 producer 同行 col tile），生成 policy 代码。论文未明确说明上层 serving 框架修改。
  - **编译框架层**：cuSyncGen 编译器将 DSL 依赖描述编译为 CUDA policy 代码（sem/value 方法 + tile order 函数）。自动生成 TileSync（每 tile 独立 semaphore）和 RowSync（同行 tile 共享 semaphore）两种 policy，以及优化的 tile 处理顺序。自动应用 reorder tile load（重叠 wait 与无关 tile load）等优化。
  - **kernel调度层**：Producer 和 consumer 同时发射到不同 stream。Producer wait-kernel 先占位确保 producer 先获得 SM。Producer tile 计算完后 `post()` 对 semaphore atomicAdd。Consumer tile 加载前 `wait()` busy-wait 对应 semaphore。由于 consumer tile E(x,y) 只依赖同行 producer tile C(x,0),...,C(x,N-1)，consumer 无需等 producer 全部完成。两个 kernel 的 independent tiles 可在同一 wave 中混合执行。从原来的 3+3=6 wave（StreamSync）降为约 2.4 wave（cuSync），消除 partial wave 浪费。
  - **硬件架构层**：80 SM 被 producer 和 consumer 的 independent thread block 混合填充，每 wave 的 SM 利用率接近 100%。论文未明确说明硬件架构层自定义修改。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: stream 同步强制 kernel 完全串行** → 方案：不同 stream + semaphore 实现 tile 级依赖，independent tile 可跨 kernel 并发。
  - **defect: partial wave SM 空闲** → 方案：tile 级混合调度，consumer tile 一旦依赖满足即可执行，填充原本空闲的 SM。
  - **defect: 通用性不足（Stream-K 仅支持 GeMM）** → 方案：cuSync 适用于所有 tile-based kernel（GeMM、Conv2D、Dropout、Softmax），仅需少量代码修改（0.5%-1%）。
