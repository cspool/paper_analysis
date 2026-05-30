## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- baseline方法是什么？
  Baseline 核心是**静态量化 + 按需加载**的 MoE 端侧推理：
  
  **(1) 固定 Bit-Width 量化**：EdgeMoE 通过离线 calibration 对每个 expert 分配固定 bit-width（如 INT2/3/4 混合），推理时不再改变。MC-MoE 根据 expert 激活频率和置信度设计固定分配。这种静态策略忽略 expert 重要性随输入 token 动态变化的特性（Observation #2），导致某些 token 分到不足以保精度的低 bit-width，另一些占用不必要的内存。
  
  **(2) 独立存储多版本权重**：若需支持动态 bit-width 选择，传统量化（GPTQ/AWQ）需独立存储 INT2/3/4 权重。LLaMA-MoE INT4 需 3.81GB，同时存 INT2/3/4 膨胀至 9.62GB。
  
  **(3) 按 Expert ID 顺序 I/O-Compute**：现有方法按 expert ID 升序加载和计算，无法重叠 I/O 与计算。Observation #3：LLaMA-MoE-3.5B 在 32 requests 时 I/O 2.6s + 计算 2.04s，但因 bubble 总延迟达 3.55s（增加 31%）。

  **Baseline 全栈执行例子（LLaMA-MoE-3.5B, RTX 3060, 单 request, 对比 D2MoE 的动态路由+MWQ）**：
  - **算法层**：输入 token → Embedding → Attention → MoE Gating（Top-2 选 expert 3 和 expert 7）→ static bit-width router（固定 INT4）。Expert 3 统计重要度低却被分配 INT4（过保守，内存浪费），Expert 7 统计重要度高也被分配 INT4，但无法利用更低 INT2/3 压缩。
  - **系统框架层**：Serving 按 expert ID 升序：[Expert 3 INT4 → Expert 7 INT4]。先加载 Expert 3 INT4（~320ms I/O），再计算（~3.1ms），此时 Expert 7 I/O 未开始（idle bubble）。计算完 Expert 3 后才加载 Expert 7（~320ms I/O），再计算（~3.1ms）。总时间 ≈ 646.2ms，bubble 3.1ms（Expert 3 计算时无并行 I/O）。
  - **Kernel层**：GPTQ-style dequantization（INT4→FP16 via scale+zero-point）→ cuBLAS GEMM。dequantization 占推理总时间 20%–70%。
  - **硬件层**：NVIDIA RTX 3060 (6GB) 无法容纳全量 INT4 权重（expert 参数占 ~89.9%），须逐 expert 从 NVMe SSD (3.5 GB/s) 加载。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  D2MoE 通过**算法-系统协同设计**的三个组件解决 baseline 缺陷：
  
  **(1) Token-Adaptive Bit-Width Selection → 解决静态 bit-width 无法适应 token 动态性**
  - 在每个 expert 前插入轻量化 bit-width router（额外开销 <0.5% 计算/0.5% 内存/1.7% 延迟），根据 token 表示动态选 bit-width
  - Quantized Expert Capacity：{c_k} 约束每个 bit-width 的 token 容量（如 D2MoE-V1 中 {0.3, 0.4, 0.3} 对应 INT2/3/4），超限 token 随机丢弃防止坍塌
  - Dynamic Bit-Width Selection Loss = CE(p, q) + (α/L) * Σ p_k^l * b_k，CE 保精度，正则项促选低 bit-width 以省内存
  - 解决 Observation #2：不同 token 下同一 expert 的重要度波动大，动态自适应获得更优准确率-内存权衡
  
  **(2) Matryoshka Weight Quantization (MWQ) → 解决多 bit-width 版本存储爆炸**
  - 嵌套量化：Asymmetric Quant 到最低 b_1（如 INT2）→ 对残差逐次 Binary Residual Quantization（+1/-1），每步增加 1 bit
  - b_K = b_1 + Σ_{k=2}^{K} 1-bit residual，高 bit-width 包含低 bit-width，如嵌套娃娃
  - 仅存一份 base INT2 + 若干 1-bit residual + 对应 scale factor，存储量接近 INT4 而非 INT2+INT3+INT4
  - 解决 Challenge #2（多版本高内存开销）
  
  **(3) Bit-Width-Aware I/O-Compute Pipeline + HEBF → 解决 I/O-Compute bubble**
  - MWQ 嵌套允许低 bit-width 权重被多个高 bit-width 请求复用。如 3 个 request 选 Expert 2（1×INT2+2×INT3），所有 3 个共享 INT2 base，仅 2 个 INT3 额外加载 1-bit residual
  - HEBF 按激活频率排序 I/O 队列（Hottest Expert Bit First），优先加载高频低 bit-width 权重
  - Memory Budget Scheduler：配置内存预算 M，保留高频低 bit-width 权重常驻 GPU
  - 解决 Observation #3（大 bubble）和 Challenge #4（轻量化调度）
  
  **D2MoE 全栈执行例子（同场景，LLaMA-MoE-3.5B, RTX 3060, M=1600MB）**：
  - **算法层**：token → Embedding → Attention → MoE Gating（Top-2 选 expert 3 和 expert 7）→ **bit-width router**（动态决策：Expert 3→INT2，Expert 7→INT3）
  - **系统框架层**：HEBF 按激活频率构建队列。假设 Expert 7(INT3) 频率 > Expert 3(INT2)：加载 Expert 7 INT2 base → 加载 Expert 7 +1-bit residual 的同时 Expert 7 INT2 dequant+GEMM 并行 → Expert 3 INT2 加载与 Expert 7 INT3 final 计算并行
  - **Kernel层**：Parallel Loading Dequantization Kernel 用 CUDA streams 重叠 disk→GMEM cudaMemcpyAsync 与 L2→CUDA cores dequantization。MWQ dequant = INT2 reconstruction(scale+zp) + Σ 1-bit residual * s_{b_k}（位操作代替传统 bit-transpose），单 kernel 减少 launch overhead
  - **硬件层**：NVIDIA RTX 3060。I/O 量从 INT4×2（~2.24GB→~640ms）降至 INT2 base + INT3 ≈ 448MB（~128ms），总 latency 从 ~646ms 降至 ~150ms（1.39× 吞吐提升），峰值内存降 33%–53%
