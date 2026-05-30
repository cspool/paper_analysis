## SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

- baseline方法是什么？
  Baseline是GPTQ（3-bit/4-bit group-wise PTQ）和RTN（round-to-nearest uniform quantization）。以LLaMA-7B GPTQ 4-bit在A100上推理为例：
  
  - **算法Pipeline**：输入tokens(2048) → embedding → L层Transformer Block。每Block内：RMSNorm(FP16) → MHA(Q/K/V/O投影+RoPE+Softmax+Attention) → 残差 → RMSNorm(FP16) → FFN(Gate/Up/Down投影+SiLU) → 残差。GPTQ对所有Linear层权重W做4-bit group-wise量化（group_size=128），基于Hessian矩阵 H=2XXᵀ 做逐列OBQ误差补偿：每量化一列，误差通过Cholesky分解的逆Hessian传播到右侧未量化列进行补偿。量化尺度和零点以16-bit存储。
  
  - **系统框架**：GPTQ量化后的PyTorch推理。每个128权重group共享一组scale/zero，dequantize ŵ = scale × (Q - zero)，与FP16 activation执行FP16 matmul。
  
  - **编译框架**：论文未明确说明。
  
  - **Kernel调度**：PyTorch默认矩阵乘法（cuBLAS），无自定义kernel。权重以INT4 packing存储，推理时解包为FP16后计算。Token-by-token生成是memory-bound：batch_size=1时，算术强度极低，瓶颈在DRAM带宽。
  
  - **硬件架构**：NVIDIA A100 GPU（Ampere架构），论文未涉及RTL或模拟器修改。
  
  Baseline的两大缺陷：
  1. **统一精度忽略权重敏感性差异**：GPTQ对所有权重同等处理（同一bit-width, 同一group内部统一scale/zero），但论文分析表明约1%的敏感权重贡献了75%以上的总量化误差。这些高敏感度权重呈现特定的结构模式（行异常值、列异常值、敏感attention heads、rotary embedding pattern、非结构化异常值），但GPTQ的group-wise量化只能以粗粒度group（128权重）补偿误差，无法精确处理离散的敏感权重。
  2. **量化统计量存储开销限制group size下限**：直觉上更小的group size可以提高精度（每个group的scale/zero更适配局部数据分布），但量化统计量（scale+zero）存储开销随group size减小而增大。例如对4-bit权重，group_size=16时统计量开销为2×16/16=2 bits/param，抵消了量化的内存收益，因此传统方法采用较大的group size（128）导致精度不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SpQR（Sparse-Quantized Representation），包含三个核心机制对应解决baseline缺陷：

  **双层量化（Bilevel Quantization）** 解决缺陷2（统计量存储开销限制group size）：
  - 关键思想：将极小group（β₁=8~16）的scale/zero这些"量化统计量"本身再量化为3-bit，然后对量化统计量的统计量做第二层量化（β₂=16），用16-bit存储最终统计量
  - 平均bits数：b̄ = b_w + (b_s+b_z)/β₁ + 64/(β₁β₂) + 32·r_o。例如b_w=3, b_s=b_z=3, β₁=16, β₂=32 → 统计量开销仅6/16+64/512=0.5 bits/param，使小group size的实际内存开销可控
  - MMQ量化器的min-max要求放宽：去除"max>0, min<0"约束，允许全正/全负group的非整数零点，进一步提升小group下的精度

  **非结构化异常值检测与高精度保留** 解决缺陷1（敏感权重识别并隔离）：
  - 基于Optimal Brain Surgeon框架推导敏感度：s_ij = (w_ij - quant(w_ij))² / (2[H⁻¹]_jj)。该公式捕捉了权重间的相关性——某个权重的高rounding error可被其他权重补偿（通过连续值优化补偿）
  - 异常值检测发生在GPTQ量化过程中（而非预处理）：这样检测的不仅是"初始敏感"权重，还包括量化过程中因误差累积而变得敏感的权重（即能补偿其他权重量化误差的权重）
  - 检测到的高敏感度权重（约1%，τ阈值由binary search确定）保留为16-bit，使得min-max scale计算排除outlier后大幅减小，进一步提升剩余权重的量化精度

  **CSR格式稀疏矩阵乘法GPU Kernel** 解决缺陷1引发的推理效率问题：
  - 虽然1%异常值以非结构化CSR存储，但设计专门的GPU kernel通过tile-based load balancing和row-wise越权内存访问，结合dense-quantized matmul实现了比FP16基线更快的推理速度（20-30%加速）

  全栈执行例子（LLaMA-65B 3-bit SpQR对比GPTQ 4-bit）：
  
  - **算法Pipeline**：校准数据X通过模型前向传播收集每层输入 → 计算Hessian H=2XXᵀ和Cholesky分解Hⁱᶜ。对每层W ∈ R^{d_out×d_in}，逐β₁列组处理：
    1. 在当前列组内检测outliers（leave-one-out error比对）→ 标记O
    2. 排除O对剩余权重拟合3-bit group-wise quantizer（双层：先fit 3-bit scales/zeros，再fit scales-of-scales/zeros-of-zeros）
    3. 量化非outlier权重为3-bit codes
    4. OBQ误差传播（GPTQ风格）到右侧未量化权重
    5. 收集outliers为CSR格式（row-first排序）
    → 输出：Q (3-bit packed), S_q/Z_q (3-bit first-level stats), S_s/Z_s/S_z/Z_z (16-bit second-level stats), W_sparse (CSR outliers)
  
  - **系统框架**：SpQR PyTorch推理代码 + 自研CUDA kernel。权重以custom SpQR格式存储（每256权重block内：256×3-bit codes + 16×3-bit scales/zeros + 4×16-bit statistics），CSR存储1% outliers
  
  - **编译框架**：论文未明确说明
  
  - **Kernel调度**：自研GPU kernel执行SpQR格式推理：
    - Dense部分：Thread block加载block statistics到SRAM→双层反量化→packed weights反量化→与activation做点积
    - Sparse部分：Tile-based load balancing→从CSR加载outlier slice到SRAM→逐row检测→加载列值→sparse dot product
    - Merge两种结果。在A100上batch_size=1时比FP16快20-30%（因为压缩率>3.4x，memory-bound场景下DRAM读取量大幅减少）
  
  - **硬件架构**：NVIDIA A100 GPU，论文未涉及RTL或模拟器修改
  
  效果：SpQR 3.94 avg bits LLaMA-65B WikiText2 PPL=3.68（vs FP16=3.53），GPTQ 4-bit=3.83。SpQR 4-bit将误差较GPTQ减半。在24GB GPU上可运行33B参数模型。
