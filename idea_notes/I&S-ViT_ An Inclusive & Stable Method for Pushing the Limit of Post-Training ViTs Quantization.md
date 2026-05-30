## I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

- baseline方法是什么？
  Baseline是标准PTQ方案：对post-Softmax激活使用log2量化器（LQ），对post-LayerNorm激活使用layer-wise均匀量化器，权重使用channel-wise均匀量化器，采用block-wise reconstruction优化目标。
  
  全栈执行例子（以DeiT-S W3A3在3090 GPU上推理为例）：
  - **算法Pipeline**：输入图像(224×224) → patch embedding → L个Transformer Block，每Block内：LayerNorm(FP32) → MHSA(QKV投影+Softmax+Attention) → 残差连接 → LayerNorm(FP32) → MLP(GELU+FC) → 残差连接。量化点：所有权重(MatMul输入)和激活值做INT3量化，log2量化器处理post-Softmax激活，layer-wise均匀量化器处理post-LayerNorm激活。
  - **系统框架**：PyTorch推理，无Serving框架修改。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：log2量化器的bit-shift操作在GPU上以标准整数运算执行，无自定义kernel。SULQ也通过bit-shifting执行。
  - **硬件架构**：NVIDIA 3090 GPU（Ampere架构），论文未涉及RTL或模拟器修改。
  
  Baseline的两个核心缺陷：
  1. **Quantization Inefficiency**：log2量化器的量化范围无法覆盖全部输入域，大量远离零的值被clamp到相同位置，造成大量化误差。例如3-bit时[8,26]段的值全部被clamp到7。
  2. **Rugged Loss Landscape**：channel-wise权重量化 + layer-wise post-LayerNorm激活量化的组合导致loss landscape粗糙且loss值放大，容易误导优化进入局部极小值。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出I&S-ViT，包含两个核心组件：
  
  **SULQ (Shift-Uniform-Log2 Quantizer)** 解决缺陷1：
  - 在log2变换前引入shift bias η: X_q = UQ(-log₂(X+η), b)
  - 反量化: X̄ = 2^{-round(D-UQ(X_q))} - η
  - 通过均匀量化器完整覆盖输入域（3-bit时8个整数均匀分布在[19,0]范围），同时保持对接近零区域的细粒度bit分配，匹配post-Softmax长尾分布。仅增加一次round和两次加法，可由bit-shifting高效执行。
  
  **SOS (Smooth Optimization Strategy)** 解决缺陷2：
  - Stage 1：全精度权重 + channel-wise量化的post-LayerNorm激活 → loss landscape平滑且loss值低，优化更稳定
  - Stage 2：通过scale reparameterization无损地将channel-wise转为layer-wise量化器（调整LayerNorm的affine参数和下一层权重）
  - Stage 3：量化权重并在全量化状态下微调恢复性能
  
  全栈执行例子（I&S-ViT对比Baseline）：
  - **算法Pipeline**：相同Transformer结构，但post-Softmax激活改用SULQ（shift+log2+uniform），post-LayerNorm激活在Stage 1用channel-wise量化获得平滑landscape，Stage 2通过scale reparameterization无损转layer-wise，Stage 3全量化微调。DeiT-S W3A3从Baseline的3.36%提升至55.78%（+52.42%）。
  - **系统框架**：同Baseline，PyTorch推理。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：SULQ的bit-shifting操作与标准log2量化器相同硬件效率，无额外kernel修改。
  - **硬件架构**：同Baseline，单张3090 GPU，论文未涉及RTL或模拟器修改。
