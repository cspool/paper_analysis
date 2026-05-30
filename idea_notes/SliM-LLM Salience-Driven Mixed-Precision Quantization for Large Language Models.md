## SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

- baseline方法是什么？
  Baseline是uniform-precision PTQ方案（以GPTQ为代表）：对LLM所有权重矩阵使用统一bit-width的group-wise量化（group_size=128），采用基于Hessian矩阵的逐列误差补偿（OBQ延续）。全栈执行例子（以LLaMA-7B 2-bit GPTQ在A800上推理为例）：
  - **算法Pipeline**：输入tokens(2048) → embedding → L层Transformer Block，每Block内：RMSNorm(FP16) → MHA(Q/K/V/O投影+RoPE+Softmax+Attention) → 残差 → RMSNorm(FP16) → FFN(Gate/Up/Down投影+SiLU) → 残差。所有Linear层的权重用INT2统一量化（group_size=128, per-channel scale/zero），Hessian近似H = (1/P) Σ x^T x，逐列OBQ误差补偿。
  - **系统框架**：AutoGPTQ推理（AutoGPTQ），对统一2-bit权重的每个128元素group做dequantize后与FP16 activation做矩阵乘法。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：AutoGPTQ CUDA kernel对统一2-bit权重按group做dequantize+向量点积，warp内32 threads处理128列group，data access pattern一致。
  - **硬件架构**：NVIDIA A800 GPU（Ampere架构），论文未涉及RTL或模拟器修改。
  
  Baseline的两大缺陷：
  1. **统一精度忽略权重重要性差异**：所有权重同等对待，但salient权重对输出loss影响远大于非salient权重（δ_{i,j}=w_{i,j}²/[H⁻¹]_{j,j}²）。尤其在2-bit场景，有限码本容量无法同时容纳重要信息，导致perplexity崩塌（如LLaMA-7B GPTQ 2-bit WikiText2 perplexity高达152.31）。
  2. **element-wise混合精度的硬件不友好性**：现有方法（SpQR、PB-LLM、LLM-MQ）使用非结构化element-wise混合精度，需要额外存储bitmaps或code indices，无法与AutoGPTQ的group-wise packing兼容，导致部署效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SliM-LLM，包含两个核心组件对应解决baseline的两大缺陷：

  **SBA (Salience-Determined Bit Allocation)** 解决缺陷1（统一精度忽略权重差异）：
  - 关键观察：salient权重在channel维度上呈现spatial clustering（因activation outlier channels始终出现在固定位置，由Theorem 1证明：x_{:,p}^* > x_{:,j} → H_{p,p} > H_{j,j} → δ_{:,p} > δ_{:,k}）
  - 依据group内平均salience排序，双指针搜索最优混合精度配置：高salience group给3-bit，等量低salience group给1-bit补偿，其余2-bit（|G₁|=|G₃|约束维持average 2-bit）
  - 优化目标从MSE改为KL divergence（D_KL(xW^T || xŴ_sba^T)），从信息熵角度对齐输出分布而非仅最小化权重差值
  - 优势：group-wise结构化混合精度可直接用AutoGPTQ的packing机制，无需额外bitmap，硬件友好
  
  全栈执行例子（LLaMA-7B 2-bit SliM-LLM对比GPTQ）：
  - **算法Pipeline**：每层Linear权重W → 先按128列分组计算average salience → SBA双指针搜索确定1/2/3-bit group分布 → SQC对每个group内1% salient权重做τ校准→ GPTQ的OBQ逐列误差补偿。关键差异：不同group用不同bit-width（1/2/3-bit），salient group精度更高。
  - **系统框架**：修改版AutoGPTQ，存储时额外记录每个group的bit-width（2-bit/group聚合为整数），weights按各group精度分别pack。推理时按group逐精度dequantize。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：修改版AutoGPTQ CUDA kernel，逐group读取bit-width确定解包方式→dequantize→与shared activation做向量点积。因group内部精度统一，warp内threads的code path和数据访问逻辑仍保持一致。
  - **硬件架构**：NVIDIA A800 GPU，论文未涉及RTL或模拟器修改。

  **SQC (Salience-Weighted Quantizer Calibration)** 进一步解决缺陷1在group内部的残余问题：
  - 即使SBA给高salience group高bit-width，group内部仍有个别稀疏salient元素（约1%）与非salient元素共享量化器参数
  - SQC通过3-σ规则选中这些salient权重（w_s），引入calibration参数τ对scale和zero point做区间搜索[1-λ, 1+λ]（λ=0.1，50个candidate）
  - 优化加权目标: argmin_τ (||w_s - τ·s·Q(w_s,τs,τz)||² + ||w_us - τ·s·Q(w_us,τs,τz)||²)，扩大量化器perception interval的同时w_s和w_us仍共享同一套(τs, τz)，无需额外存储
  - 效果：OPT-1.3B某channel绝对误差从0.0055降至0.0039，salient权重误差显著降低

  协同效果：SBA处理global（group间）salience差异 → 结构化group-wise混合精度 → 硬件友好；SQC处理local（group内）salience差异 → 保护离散的重要权重 → 性能提升。SliM-LLM 2-bit LLaMA-7B WikiText2 PPL=14.58（vs GPTQ 152.31），接近3-bit水平的16×压缩比（6×内存减少），且保持GPU推理可用速度（61.2 vs 83.9 token/s）。
