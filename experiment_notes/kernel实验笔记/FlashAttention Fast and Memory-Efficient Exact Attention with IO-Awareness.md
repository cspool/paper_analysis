## FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现一个fused CUDA kernel，将attention的全部操作（QK^T矩阵乘 → softmax含masking和dropout → PV矩阵乘）融合为单个GPU kernel，避免中间$N \times N$ attention矩阵在HBM中的materialization。核心设计：(i) **Tiling with online softmax**——将Q/K/V分块加载到SRAM，沿KV block维度（外循环）和Q block维度（内循环）做block-wise计算，通过running max $m$和running sum $\ell$维护正确归一化；(ii) **Recomputation for backward**——前向仅保存输出O和softmax统计量$(m,\ell)$（$O(N)$内存），反向在SRAM中重计算S和P，比标准方法（从HBM读取$O(N^2)$中间值）更快（即使FLOPs增加）；(iii) **Block-sparse variant**——跳过预定义稀疏mask中零值block的全部计算（BMM1除外）。
  实验比较：(i) Forward+backward runtime vs standard attention（seq length 1024, head dim 64, 16 heads, batch 64, A100 GPU），测量GFLOPs、HBM R/W(GB)、Runtime(ms)；(ii) Block size消融——$B_c$从64到512下forward runtime变化，验证HBM accesses是runtime主导因素；(iii) Block-sparse FLASHATTENTION runtime vs sparsity比例（seq length 4K）；(iv) Full benchmark——forward+backward runtime和attention memory usage随sequence length（128-64K）变化，vs PyTorch exact attention和多种approximate/sparse attention（Linformer, Linear Attention, Performer, Reformer, Smyrf, Local Attention）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 GPU (Ampere架构)：40GB HBM，带宽1.5-2.0TB/s，108 SMs，每SM 192KB on-chip SRAM（带宽约19TB/s）
  - Benchmark配置：seq length 128-64K，head dim 64，16 heads，batch size 64，key-padding mask + dropout，单卡A100 40GB
  - CUDA环境：自编CUDA kernel，基于NVIDIA Apex FMHA代码（https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha）

- 评估性能的软件/脚本是什么。修改了什么。
  - 自编CUDA kernel实现FLASHATTENTION的forward和backward pass，以及block-sparse FLASHATTENTION variant
  - 修改：(i) 替代标准PyTorch attention实现（`torch.nn.functional.scaled_dot_product_attention`或等效的手写attention），将attention计算替换为单个fused CUDA kernel调用；(ii) Forward kernel中内循环结构——对每个KV block $j$，内循环遍历Q blocks $i$：计算$S_{ij}=Q_iK_j^T$（Tensor core MMA）→ online softmax（CUDA core: rowmax + EXP MUFU.EX2 + rowsum + rescale + combine）→ 累加$O_i$（Tensor core MMA: $\tilde{P}_{ij}V_j$ + rescale）→ write $O_i, \ell_i, m_i$ to HBM。中间$S_{ij}, \tilde{P}_{ij}$仅驻留SRAM；(iii) Backward kernel——利用保存的统计量$(m,\ell)$和输入Q/K/V/O重计算$S_{ij}, P_{ij}$ in SRAM，计算$dQ, dK, dV$梯度；(iv) Block-sparse kernel——仅在内循环的$M_{ij}=0$时跳过softmax+PV计算，其余逻辑相同。
  - Benchmark脚本：测量CUDA kernel wall-clock time（CUDA event timing）、HBM读写量（通过理论分析验证）、peak memory allocation（`torch.cuda.max_memory_allocated()`）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  已开源：https://github.com/HazyResearch/flash-attention（BSD许可证）。安装：`pip install flash-attn`。Python接口：`from flash_attn import flash_attn_func; output = flash_attn_func(q, k, v, causal=False, dropout_p=0.0)`。

  评估原理与流程（以单卡A100 forward+backward benchmark为例，N=1024, d=64, 16 heads, batch=64）：
  1. **Input准备**：在PyTorch中创建Q/K/V tensors in FP16/BF16：`[batch=64, seq_len=1024, nheads=16, head_dim=64]`，位于GPU HBM。应用key-padding mask和optional dropout mask。
  2. **Forward kernel launch**（单次CUDA kernel调用）：
     a. Compute block sizes: $B_c = \lceil 192\text{KB} / (4 \times 64 \times 2\text{B})\rceil = \lceil 12288 / 512\rceil \approx 384$，$B_r = \min(384, 64) = 64$（head_dim bound）。
     b. 划分：$T_c = \lceil 1024/384 \rceil = 3$个KV blocks（大小384），$T_r = \lceil 1024/64 \rceil = 16$个Q blocks（大小64）。
     c. 外循环（KV blocks, j=1..3）：从HBM加载$K_j(384\times64), V_j(384\times64)$到SRAM（约49KB per matrix）。
       内循环（Q blocks, i=1..16）：从HBM加载$Q_i(64\times64)$到SRAM（约8KB）。On-chip: $S_{ij}=Q_iK_j^T$（64×384, FP16 GEMM on Tensor core）→ rowmax（CUDA core reduction: 64 values per row）→ $\exp(S_{ij}-\tilde{m}_{ij})$（MUFU.EX2）→ rowsum → rescale running $m_i,\ell_i$ → $\tilde{P}_{ij}V_j$（Tensor core MMA）→ rescale and accumulate $O_i$ → write $O_i,\ell_i,m_i$ to HBM。
     d. 中间$S_{ij}$和$\tilde{P}_{ij}$（64×384 each）驻留在SRAM中，不写入HBM。
  3. **Backward kernel launch**（单次CUDA kernel调用）：
     a. 加载$O, dO, \ell, m$和Q/K/V from HBM。
     b. 对每个(i,j) block pair在SRAM中重计算$S_{ij}, P_{ij}$。
     c. 计算$\partial Q_i = dO \cdot V_j^T \cdot \text{diag}(P_{ij})$等梯度（由softmax反向公式展开）。
     d. Write $dQ, dK, dV$ to HBM。
  4. **Performance measurement**：CUDA events记录`flash_attn_func`的forward+backward总时间。FLASHATTENTION: GFLOPs=75.2（高于standard的66.6，因recomputation），HBM R/W=4.4GB（远低于standard的35.3GB），Runtime=11.7ms（vs standard 35.1ms，3× faster）。
  5. **Memory measurement**：Standard attention memory = $O(batch \times heads \times N^2)$ = 64×16×1024²×2B ≈ 2.1GB（仅S和P）。FLASHATTENTION memory = $O(N)$额外 = 64×16×1024×4B ≈ 4.2MB（仅$m,\ell$）。Memory减少约500×。
  6. **Scaling验证**（Figure 3）：runtime和memory随N从128到64K变化。FLASHATTENTION runtime grows quadratically（FLOPs $O(N^2)$），但constant factor远小于baseline；memory grows linearly。Block-sparse FLASHATTENTION runtime在64K时比所有approximate attention方法都更快。
  7. **Output**：Figure 2（left: standard vs FLASHATTENTION的runtime/HBM breakdown；middle: block size对runtime的影响——block越大HBM accesses越少，直到arithmetic瓶颈；right: block-sparse sparsity vs runtime）。
