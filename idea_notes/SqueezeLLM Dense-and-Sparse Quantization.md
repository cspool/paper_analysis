## SqueezeLLM Dense-and-Sparse Quantization

- baseline方法是什么？
  **Baseline是uniform weight-only post-training quantization（以GPTQ为代表）**，该方法对LLM权重采用逐通道或逐组的均匀量化。

  全栈执行例子（GPTQ 3-bit uniform quantization with activation ordering, LLaMA-7B, 单batch推理）：

  - **算法Pipeline**: 校准数据（128个C4样本）逐层前传→收集每层输入activation statistics→逐列OPTQ贪心量化：对每列权重，用Hessian信息（基于输入activation的outer product, H=2XX^T）做quantize-and-compensate——量化当前列→计算量化误差→把误差按H的逆传播到右侧未量化列→重复直到全矩阵量化完成。Uniform格式：每128个连续元素共享一个scale+zero（group-wise uniform scaling factor, g128）。这实际上是最小化层间输出activation扰动（layer-wise L2: ||WX - W_QX||^2），而非最终模型loss。

  - **系统框架（Serving）**: 论文未明确说明修改Serving框架。推理使用PyTorch + 自研/社区CUDA kernel加载quantized权重，以FP16执行dequant + matvec。

  - **编译框架**: 论文未明确说明。

  - **Kernel调度**: Uniform quant kernel：加载packed int3 weights→按group读取scale/zero→dequantize为FP16→与FP16 activation向量做matvec。**问题**：GPTQ with activation ordering引入permutation，使得同一channel的权重分布在不同的group中，需要不同scaling factor（通过group index间接访问）。在GPU上这种scattered memory access破坏内存合并（coalesced access）→导致latency从1.4s暴涨到13.7s（LLaMA-7B, A6000, 128 tokens）。

  - **硬件架构**: NVIDIA A6000/A100 GPU。GPU memory bandwidth是瓶颈（A6000: 768 GB/s内存带宽 vs 222 TFLOPS计算吞吐，带宽仅为算力的~0.3%）。LLM单batch推理是memory-bound——每个权重加载后仅参与一次乘加（arithmetic intensity极低）。

  Baseline的根本性缺陷：
  1. **Uniform quantization在非均匀权重分布下浪费量化分辨率**：LLM权重分布高度非均匀（99.9%的值集中在~10%的范围内），uniform bin allocation将大量bin浪费在稀疏分布区域，对密集区域的敏感权重分辨不足。
  2. **Layer-wise优化目标与end-to-end loss不一致**：GPTQ最小化||WX-W_QX||（层间输出扰动），而SqueezeLLM证明直接最小化final loss扰动（Hessian-weighted objective）显著更优（D.4消融实验：LLaMA-7B 3-bit PPL gap ~0.3）。
  3. **Outliers膨胀量化范围**：极少数outlier values使整个量化range扩大10x，严重降低quantization resolution。
  4. **Grouping不是outlier问题的直接解决方案**：GPTQ/AWQ用grouping (g128)间接隔离outliers→增加storage overhead（per-group scale+zero）→且在非均匀量化下overhead更严重（需per-group LUT）。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **SqueezeLLM = Sensitivity-based non-uniform quantization + Dense-and-Sparse decomposition**。两个技术逐一解决baseline缺陷：

  全栈执行例子（SqueezeLLM 3-bit + 0.45% sparsity, LLaMA-7B, 单batch推理）：

  - **算法Pipeline**:
    **缺陷①→方案**：将均匀量化替换为sensitivity-weighted k-means非均匀量化。优化目标从min||W-W_Q||^2改为min Σ F_ii(w_i-Q(w_i))^2，其中F_ii是Fisher信息矩阵对角线（≈Hessian对角线），通过calibration数据集（仅需10-100样本）的一次梯度前反向计算获得（7B: 0.3min on A100）。Weighted k-means自动将centroid向高敏感度权重聚拢（Fig. 3直观展示：均匀量化的8个bin均匀分布，而sensitivity-based的8个bin在敏感值区域更密集）→3-bit LLaMA-7B PPL从uniform的28.26降至7.75。
    **缺陷②→方案**：直接minimize final loss perturbation（Eq. 4-6），通过Taylor展开和Fisher近似将二阶Hessian信息融入k-means权重。相比layer-wise objective（如AWQ使用activation magnitude作为importance），final-loss-based方法在所有sparsity level下PPL优约0.3（D.4）。
    **缺陷③→方案**：Dense-and-Sparse decomposition——提取0.4% outlier (百分位阈值) + 0.05%最敏感值(按Fisher排名)作为稀疏矩阵S（CSR格式, FP16），剩下99.55%的dense矩阵D值域压缩约10x→非均匀量化的分辨率大幅提升→3-bit PPL从7.75再降至7.56。
    **缺陷④→方案**：直接用sparse component隔离outliers+sensitive values，而非用grouping间接处理。D.3消融实验证明：pure Dense-and-Sparse decomposition在所有model size下PPL优于grouping (g512/g1024)或grouping+sparsity hybrid方案。在非均匀量化下grouping需存per-group LUT（overhead巨大），而sparsity方案overhead可控（仅0.24 bit for 0.45% sparsity）。

  - **系统框架（Serving）**: 论文未明确说明修改Serving框架。推理时每个Linear层执行两个融合的kernel调用：LUT dequant matvec + balanced CSR SpMV。Dense和Sparse kernel在单次launch中融合，无额外result sum kernel开销。

  - **编译框架**: 论文未明确说明。

  - **Kernel调度**: 
    - **Dense kernel**: LUT-based非均匀dequant+matvec。压缩格式存3-bit indices→从per-channel LUT (8个FP16 centroid)查表获得真实FP16 weight→与activation做FP16内积。LUT overhead极小（延迟仅比uniform高~7%，即1.4→1.5s），但换来PPL 9.55→7.75的巨大提升。
    - **Sparse kernel**: Balanced CSR SpMV（10 nz/thread）。对比标准CSR kernel (thread-per-row)在处理skewed sparsity distribution（Fig. C.1: 少数channel含大量nonzeros）时的严重负载不均衡（3.9s vs 1.7s for 7B），balanced kernel通过固定per-thread nonzero数实现workload均衡（使用atomicAdd合并同一行的多线程结果）。
    - **整体**: 0.45% sparsity时延迟1.7s vs FP16的3.2s（1.9x加速），PTQ 9.55→7.56。相比之下GPTQ g128因permutation引入的scattered memory access降速到13.7s→实际上不可用。

  - **硬件架构**: NVIDIA A6000 GPU。Roofline model验证：LLM单batch生成推理是memory-bound问题（arithmetic intensity极低），因此LUT查表的overhead（少量额外计算）完全被memory bandwidth瓶颈掩盖→理论加速≈压缩比。实际speedup: 1.9x (3-bit) / 1.7x (4-bit) for 0.45% sparsity。

  效果总结（LLaMA-7B 3-bit, C4 perplexity）：
  | 方法 | PPL | Speedup |
  |------|-----|---------|
  | FP16 Baseline | 7.08 | 1.0x |
  | GPTQ uniform (no group) | 9.55 | 2.3x |
  | GPTQ uniform (g128, reorder) | 7.89 | 0.2x (unusable) |
  | AWQ (g128) | 7.90 | 2.0x |
  | **SqueezeLLM dense-only** | **7.75** | **2.1x** |
  | **SqueezeLLM 0.45% sparse** | **7.56** | **1.9x** |

  核心创新映射：
  - Sensitivity-based non-uniform quantization → 解决了uniform在非均匀分布+memory-bound场景下的次优性
  - Fisher-weighted k-means → 将final loss sensitivity融入量化，优于layer-wise perturbation minimization
  - Dense-and-Sparse decomposition → 直接解决outlier问题，比grouping更高效且与non-uniform天然兼容
  - Balanced sparse kernel → 使sparsity的latency overhead可控（<15%），实现practical speedup
