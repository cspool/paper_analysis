## Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

- baseline方法是什么？
  Baseline 方法是现有的 MoE 训练框架：(1) **Fairseq GShard**：使用 expert parallelism 将所有 expert 参数加载到 GPU 显存，通过 batched matrix multiplication + dispatch mask 执行 MoE 计算，随 expert 数量增加需要更多 GPU；(2) **Tutel**：MoE 特化框架，同样将所有 expert 驻留在 GPU 显存，使用 batched GEMM 和 zero-padding 对齐批量大小；(3) **ZeRO-Offload^E**（论文自行实现的 expert-wise offloading 版本）：将 expert 参数 offload 到 CPU 内存，但采用 naive 的层级别 pipelining 和 model-level CPU optimizer，无动态负载均衡。Baseline 的核心问题：(a) GPU 内存受限——expert 增加需要更多 GPU（MoE-L 8 experts 需 4×A100，128 experts 需 52×A100）；(b) dispatch mask 内存爆炸——batched matrix multiplication 需要 (tokens after padding)×(tokens) 的巨大映射表（如 MoE-L batch 32 需 48 GiB）；(c) zero-padding 浪费——expert 越多 token 负载越不均衡（32 experts 时 39% 零填充），GPU 利用率低；(d) CPU optimizer 慢——layer-wise CPU Adam 比 GPU Adam 慢 31×，造成 GPU 空转。

  全栈执行例子（Baseline: Fairseq GShard, MoE-L 32 experts, 4×A100 40GB）：
  ```
  # 算法层：标准 MoE，GPT-based decoder layer
  # - Gate: softmax(W_gate @ x) → Top-1 expert selection
  # - Expert FFN: gate_proj → SiLU ⊙ up_proj → down_proj
  # - 使用 batched matrix multiplication（所有 expert 同时在 GPU 上）
  
  # 系统框架层：Fairseq，expert parallelism + batched GEMM
  # - 所有 32 experts 的 params + optimizer states 常驻 GPU HBM
  # - 创建 dispatch mask [N_tokens_padded, N_tokens]
  # - All-to-All scatter tokens → experts → All-to-All gather
  # - 限制：MoE-L 32 experts + 4 GPUs → OOM（batch=1 都不够）
  
  # 编译框架层：PyTorch JIT / cuBLAS backend（论文未明确说明）
  
  # kernel调度层：
  # - cuBLAS batched GEMM 执行多个 expert 的 FFN 计算
  # - GPU-side Adam optimizer
  # - zero-padding 导致 39% 的无效计算
  # - 无 CPU-GPU 通信重叠
  
  # 硬件架构层：4×A100 40GB, PCIe 4.0, NVLink 600GB/s
  # - GPU HBM 容量成瓶颈，扩容专家需加 GPU
  # - PCIe 带宽未利用于 offloading
  ```
  Baseline 缺陷根因：(1) 所有 expert 必须同时驻留在 GPU，将 MoE 的"计算-参数量解耦"特性与系统内存解耦割裂；(2) batched GEMM 强制使用 dispatch mask 和 zero-padding；(3) layer-wise optimization 串行化 GPU 和 CPU 任务；(4) 静态 expert placement 无法适应 per-batch token 分布变化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**ES-MoE**——通过 expert 级别 offload + pipelined processing + dynamic placement 三项核心设计，使 MoE 训练规模不再受 GPU 显存限制。

  全栈执行例子（ES-MoE, MoE-L 32 experts, 4×A100 40GB）：
  ```
  # 算法层：标准 MoE，数学等价于 baseline（不做模型修改）
  # - Gate: softmax(W_gate @ x) → Top-1 expert selection（不变）
  # - Expert FFN: sequential processing（替代 batched GEMM）
  
  # 系统框架层：ES-MoE on Fairseq
  # - Expert params + optimizer states offloaded to CPU RAM / SSD
  # - GPU 仅保留：non-expert params + 当前处理 expert + activations
  # - 不使用 dispatch mask（零内存开销）
  # - Sequential expert processing → 支持 8× 更大 microbatch
  
  # kernel调度层（核心创新）：
  # Forward per MoE Block:
  # 1. Gate Network (GPU) → per-token expert selection
  # 2. Dynamic Expert Placement (CPU, <2.69μs):
  #    - Greedy scheduling: sort experts by (upload_time + compute_time)
  #    - Assign each expert → GPU with min accumulated load
  #    - 消除 zero-padding：expert 按实际 token 数分配到 GPU
  # 3. Token Permutation (GPU)：
  #    【同时：pipeline 上传第 1 个 expert, CPU→GPU via PCIe】
  # 4. Expert Processing Loop:
  #    for expert in assigned_experts:
  #      [Expert N 计算] || [Expert N+1 上传, CPU→GPU]
  #    — 计算与通信完全重叠
  # Backward per MoE Block:
  # 5. Expert FFN backward (GPU) → per-expert gradients
  # 6. Expert-wise CPU Optimizer：
  #    - Expert 0 backward done → 立即启动 CPU Adam
  #    - Expert 0 CPU optimizer || Expert 1 GPU backward
  #    - Expert N CPU optimizer || Layer(N+1) GPU forward
  #    — CPU optimizer 延迟被隐藏于 GPU 计算之后
  
  # 硬件架构层：4×A100 40GB, PCIe 4.0 (confirmed bandwidth ~25 GB/s), NVLink 600GB/s
  # - PCIe 持续用于 expert 上传/下载，与 GPU 计算重叠
  # - SSD offloading 模式下：expert CPU↔SSD 使用 LRU cache + prefetching
  # - DMA-able pinned memory 避免 page fault stall
  # - 3 种自适应模式：
  #   - GPU only: ≤32 experts（全在 GPU 内，仅消除 zero-padding）
  #   - CPU offload: 32-104 experts（offload 到 CPU RAM）
  #   - CPU+SSD offload: >104 experts（LRU cache on CPU, evict to SSD）
  ```

  解决 Baseline 缺陷的方式总结：
  1. **针对"GPU 内存受限"**：Expert offloading 将 expert 参数和 optimizer states 迁移到 CPU/SSD——GPU 仅保留 non-expert + 当前活跃 expert。MoE-L 64 experts (29.3B params) 仅需 4 GPUs，而 baseline OOM。支持 up to 67× 更多 experts（with SSD）和 63× 更大参数量。
  2. **针对"dispatch mask 内存爆炸"**：Sequential expert processing 替代 batched GEMM——无需 dispatch mask，按 gating 结果直接逐 expert 分配 token。节省 >48 GiB（MoE-L batch 32），允许 8× 更大的 microbatch。
  3. **针对"zero-padding 浪费"**：Dynamic expert placement——greedy scheduling 按 per-batch token 分布分配 expert 到 GPU，GPU 负载差异从 102%（Fairseq）降至 15%。同时消除 zero-padding 无效计算（39% → 0%）。
  4. **针对"CPU optimizer 慢"**：Expert-wise CPU optimization——每个 expert backward 完成后立即启动 CPU Adam，与后续 layers 的 GPU forward/backward 重叠。GPU 利用率提升 61.1%（vs 无 pipelined optimizer），总吞吐量提升 up to 63.0%。
  5. **自适应 offloading**：3 种模式自动切换——GPU-only 模式下仍因 sequential processing + 去 zero-padding 而优于 Tutel（1.7×-3.16× speedup）。Expert pinning（固定 top 25% 热门 expert 在 GPU）进一步提升 22.8% 吞吐量。
