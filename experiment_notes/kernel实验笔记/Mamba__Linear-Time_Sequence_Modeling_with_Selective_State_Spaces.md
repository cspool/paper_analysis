## Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Hardware-Aware Selective Scan——用于高效计算选择性SSM（S6）在GPU上的custom CUDA kernel。核心设计基于三种经典技术：kernel fusion、parallel scan和recomputation。由于selective SSM失去了与convolution的等价性（无法使用FFT），必须用recurrent模式计算，但直接materialize expanded state h∈R^{B×L×D×N}会导致IO瓶颈。硬件感知算法方案：(1) 不将scan输入(A_bar, B_bar)∈R^{B×L×D×N}写入HBM，而是直接从慢HBM加载(Δ, A, B, C)到快SRAM；(2) 在SRAM中完成discretization和recurrence/scan；(3) 仅将最终输出y∈R^{B×L×D}写回HBM；(4) 使用work-efficient parallel scan(Blelloch 1990)并行化recurrence；(5) backward pass不存储中间states，通过recomputation从HBM重新加载输入并重算。当seqlen超过SRAM容量时分chunk处理，保留intermediate scan states跨chunk延续。
  实验比较：(a) Scan Operation Speed(Figure 8 Left)：Mamba efficient scan vs standard PyTorch scan、FlashAttention-2(attention)、PyTorch convolution(FFT-based)，变seqlen 512-512K，D=1024, N=16, BF16, A100 80GB PCIe, batch=1——efficient scan 40× faster than standard scan, faster than FlashAttention-2 beyond seqlen 2K；(b) End-to-End Inference Throughput(Figure 8 Right)：Mamba 1.4B/6.9B vs Transformer 1.3B/6.7B，变batch size 1-128, prompt=2048, gen=128——Mamba 4-5× higher throughput(no KV cache, higher batch sizes)；(c) Memory Benchmark(Table 15)：125M model training memory, Mamba vs Transformer(w/ FlashAttention-2), seqlen=2048, batch=1-32——Mamba memory comparable to optimized Transformer(4.8GB vs 4.6GB at batch=1)。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU。Benchmark配置：batch size=1 for scan speed, 模型维度D=1024, state dimension N=16, BF16精度。Inference throughput: prompt length=2048, generation length=128, batch size 1-128, 测量3次取平均。Memory benchmark: 125M model, seqlen=2048, batch size 1-32 on 1 A100 80GB。

- 评估性能的软件/脚本是什么。修改了什么。
  自研fused selective scan CUDA kernel。Baseline对比：standard PyTorch parallel scan(无kernel fusion, 需materialize A_bar/B_bar/C in HBM)、PyTorch convolution(FFT-based, O(L log L))、FlashAttention-2(causal mask, Dao 2024)。修改内容：用fused kernel替代标准实现——将discretization、parallel scan和C乘法合并为一个CUDA kernel，在SRAM中完成全部计算，避免反复HBM↔SRAM的O(BLDN) IO传输。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  完全开源：https://github.com/state-spaces/mamba (Apache 2.0)，包含selective scan CUDA kernel实现。

  **Fused Selective Scan Kernel执行原理** (Section 3.3.2 + Appendix D):

  ```
  # 标准实现 (inefficient):
  # Step 1: 将Δ, A, B, C从HBM读到SRAM → discretize → A_bar, B_bar of size (B,L,D,N)
  # Step 2: 将A_bar, B_bar写回HBM
  # Step 3: 从HBM读A_bar, B_bar, C → parallel scan → 写结果回HBM
  # Step 4: 从HBM读scan结果 → 乘C → 写最终输出y回HBM
  # 总IO: O(BLDN) — 受state dim N (≈16) 放大

  # Fused kernel (efficient):
  # 1. 从慢HBM加载原始参数 (Δ, A, B, C) 到快SRAM
  #    — O(BLD + D*N) bytes, 不含N因子!
  # 2. 在SRAM中执行discretization:
  #    A_bar = exp(Δ ⊙ A)  → (B,L,D,N)
  #    B_bar = Δ ⊗ B       → (B,L,D,N)
  # 3. 在SRAM中执行parallel associative scan
  #    — 使用work-efficient Blelloch scan并行化
  #    — 中间states h ∈ R^{B×L×D×N} 仅存在于SRAM
  # 4. 在SRAM中乘C并累加: y_t = Σ C_t * h_t
  #    — 最终输出 y ∈ R^{B×L×D}
  # 5. 将最终输出y写回HBM
  #    — O(BLD) bytes, 无N因子
  # IO减少因子: O(N), N=16 → 16× less HBM traffic
  ```

  **Chunked scan处理长序列** (当seqlen超出SRAM容量):
  ```
  # 将序列分为chunks:
  For each chunk:
    在SRAM中fused scan该chunk
    保存最后一个time step的intermediate scan state
  # 将intermediate state传给下一chunk作为初始state
  # 保证全局scan的正确性
  ```

  **Recomputation策略** (backward pass):
  ```
  Forward: 不保存中间states h ∈ R^{B×L×D×N} (节省IO但需要backward重算)
  Backward: 从HBM重新加载输入(Δ, A, B, C)和output gradient(∈R^{B×L×D})
            → 在SRAM中重算intermediate states
            → 计算输入梯度(Δ_grad, A_grad, B_grad, C_grad)
            → 写回HBM
  # 总IO: O(BLD + D*N) input + O(BLD + D*N) gradient = 2(BLD + D*N)
  # vs 存储intermediates: O(BLDN) — recomputation wins when N > 2
  ```

  **Memory efficiency vs FlashAttention:**
  ```
  # Per-layer activation memory (BF16 training):
  # - FlashAttention: ~12 bytes/token
  # - MLP (Transformer): ~20 bytes/token
  # - Selective SSM (Mamba): ~16 bytes/token
  # 2×Mamba blocks ≈ 1×Attention + 1×MLP (both ~32 bytes/token)
  ```

  实测benchmark(Figure 8, A100 80GB):
  ```
  Scan Speed (seqlen=32K):
    Mamba efficient scan: ~0.15ms
    FlashAttention-2 (causal): ~1.0ms (6.7× slower)
    Standard PyTorch scan: ~6.0ms (40× slower)
    PyTorch convolution: ~0.3ms

  Inference Throughput (Mamba-1.4B vs Transformer-1.3B):
    Batch=64: Mamba ~3000 tokens/s, Transformer ~700 tokens/s (4.3×)
    Mamba-6.9B > Transformer-1.3B throughput despite 5× larger!
  ```
