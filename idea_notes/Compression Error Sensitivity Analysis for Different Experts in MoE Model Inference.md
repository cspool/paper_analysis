## Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

- baseline方法是什么？
  **Baseline 为现有的 MoE 推理方案**：GPU 内存中驻留全部 expert 参数，或使用 offloading 策略将非激活 expert 存储在 CPU 主存中，通过 PCIe 总线按需传输到 GPU。当前 offloading 系统面临的关键瓶颈是 PCIe 带宽有限（PCIe 4.0 约 32 GB/s），远低于 GPU 内存带宽（约 300 GB/s），导致专家加载延迟。**现有压缩方案（量化）**虽然能减少传输参数大小，但低比特量化（1-4 bit）会导致不可控、不可预测的误差，严重损害生成性能（如 QMoE 的 20× 压缩伴随 6.7% 精度下降，CMoE 的 150× 压缩伴随 23.81% 精度下降）。此外，之前的 MoE 压缩工作缺乏对不同 expert、不同层在面临压缩误差时敏感性差异的系统性理解，导致无法设计针对性的分层压缩策略（如对误差敏感层分配更精确的压缩，对鲁棒层使用更高压缩比）。

  **Baseline 全栈执行例子（以 Moonlight-16B MoE 模型在 GPU 内存受限的单 GPU 上推理一个 GSM8K 数学题为例）**：

  - **算法层**：输入 token 序列 → Self-Attention（QKV 投影 + Attention 计算 + Output 投影）→ Router（top-6 gating）选择 6 个 expert → 选中的 expert 参数若不在 GPU 内存中则从 CPU 主存传输 → 各 expert FFN 计算（$W_{up}, W_{gate}, W_{down}$）→ 加权求和输出 → 下一层。非激活 expert 占约 70% 参数（~66.6 GB of 94 GB for Mixtral-8x7B）浪费 GPU 内存。
  - **系统框架层**：Offloading 框架（如 MoE-Infinity/Pre-gated MoE）管理 GPU resident store（驻留常访问 expert）+ CPU main memory（存储所有 expert 参数）+ GPU staging buffer（预取动态数据）。当 GPU 访问不在 resident store 的 expert 时，计算 stall 直到 PCIe 传输完成。
  - **编译框架层**：论文未明确说明（基于 PyTorch/HuggingFace Transformers 推理 pipeline）。
  - **Kernel/运行时调度层**：PCIe DMA 传输 expert 权重（FP16）→ GPU kernel 执行 GEMM。传统无压缩时每个 expert 满精度传输。量化方案（如 4-bit）减少传输量但引入不可控量化误差。论文未明确说明具体 kernel 实现。
  - **硬件架构层**：论文未明确说明 GPU 型号和 CPU 配置。分析依赖 PCIe 带宽（~32 GB/s for PCIe 4.0）vs GPU 内存带宽（~300 GB/s）的对比。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出使用 **error-bounded lossy compression（有界误差有损压缩）**，如 SZ3（CPU）和 CuSZp（GPU），替代传统量化方案来压缩 MoE 非激活 expert 参数，并对不同 expert 的压缩误差敏感性进行系统性分析。核心思路：
  
  - **有界误差保证**：与传统量化产生"不可控、不可预测"的误差不同，error-bounded 压缩算法保证每个重建参数与原始参数的绝对误差 ≤ 预设的 error bound ê。这使得压缩对推理精度的影响可控、可预测。
  - **高压缩比且低误差**：SZ3/CuSZp 等预测-量化-编码管线可以在给定 error bound 下实现比简单量化更高的压缩比，因为利用了参数值的空间相关性（如 Lorenzo predictor 利用相邻参数的梯度预测）。
  - **分层误差敏感性分析**：通过全面的 error injection 实验（正态分布 N(0, ê) 误差注入模拟压缩解压后的参数状态），系统性地揭示了三层分层敏感性：
    - **浅层（Shallow layers, L1-L10）**：主要负责 attention 机制和输入 token→向量表示的转换，对压缩误差高度鲁棒，引入误差后推理精度下降最小（如 Layer1 Top-6 experts 注入 80% error: ICA 0.74 vs baseline 0.85）。可以承受更高压缩比。
    - **中层（Middle layers, L13）**：负责核心模型推理/逻辑分析，对压缩误差极为敏感，误差会显著损害推理精度（如 Layer13 全部 64 个 expert 注入 80% error: ICA 暴跌至 0.38 vs baseline 0.86, PIA 降至 0.65）。需要最保守的压缩策略，或使用更小的 error bound。
    - **深层（Deep layers, L20-L26）**：负责指令跟随和输出整合，对误差有"反直觉"的鲁棒性甚至增益——注入 bounded error 有时会提升推理精度（如 Layer26 Top-6 experts 注入 80% error: ICA 升至 0.90 vs baseline 0.85），可能源于深层 expert 的隐性集成效应（implicit ensemble effect），引入噪声使模型生成更多样化的输出整合。

  - **路由机制的适应性保护**：实验发现当高激活频率 expert 参数被扰动时，路由机制会动态调整路由权重，将任务重新分配给其他功能完整的 expert，保护核心推理能力（PIA 保持稳定 ≥ 94%）。
  - **功能解耦发现**：ICA（指令合规精度）和 PIA（纯推理精度）对参数误差的解耦响应说明，semantic generation 和 instruction parsing 在 MoE 架构中是功能分离的，分别由不同层的 expert 负责。

  **论文方法全栈执行例子（以 Moonlight-16B 推理 GSM8K 为例，error-bounded compression + 感知分层的 offloading 流程）**：

  - **算法层**：Input token → Self-Attention → Router 选择 top-6 experts → **分层压缩策略**：浅层 expert 使用较大 error bound（高压缩比），中层 expert 使用极小 error bound（保守压缩），深层 expert 可使用中等 error bound（甚至利用噪声增益）→ SZ3/CuSZp 压缩后 expert 参数通过 PCIe 传输 → GPU 端解压重建（参数含 bounded error）→ Expert FFN → 加权输出。压缩后参数 = 原始参数 + 误差（误差 ∈ [-ê, ê] 由压缩算法保证）。
  - **系统框架层**：在现有 offloading 框架基础上，在 GPU staging buffer 和 CPU main memory 之间插入压缩/解压模块。CPU 端压缩 expert（SZ3）→ 压缩数据通过 PCIe 传输（数据量减少）→ GPU 端解压（CuSZp）→ 解压参数加载到 GPU resident store → Expert FFN 计算。pipeline 可设计为：压缩/解压与传输重叠，进一步减少延迟。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：CPU 端 SZ3 使用多线程 Lorenzo predictor + Huffman 编码压缩 expert 权重张量；GPU 端 CuSZp 使用 CUDA kernel 并行解压。传输数据量 = 原始 size × (1/压缩比)，压缩比由 error bound ê 决定。论文未明确说明具体 kernel 实现细节。
  - **硬件架构层**：论文未明确说明 GPU/CPU 具体型号。分析框架适用于支持 PCIe 通信的 GPU-CPU 异构系统。
