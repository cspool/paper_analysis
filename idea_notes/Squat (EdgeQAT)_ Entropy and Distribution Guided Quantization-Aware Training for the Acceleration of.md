## Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight SLMs on the Edge

- baseline方法是什么？
  Baseline是现有QAT方法的典型方案——具体对比了**NIPQ**（噪声注入伪量化）、**PACT**（参数化裁剪激活量化）和**LLM-QAT**（数据无关QAT+蒸馏），三者都采用粗粒度逐层量化（layer-wise quantization，即每层一个scale factor）。

  Baseline的核心缺陷有两个层面：

  **(1) 算法层面——量化自注意力模块信息失真**：
  现有粗粒度QAT（包括LLM-QAT等）直接将逐层量化应用于自注意力模块的query和key投影，导致：(i) query和key量化后分布方差显著偏离FP16（Figure 2），信息熵降低，等效于MOE/MAE准则下引入较大量化误差；(ii) 注意力图中初始token列的特有分布模式（distinct column pattern）消失（Figure 3），自注意力模块的表征能力退化。论文实验（Figure 1）表明：仅量化query和key导致的性能下降几乎等同于量化整个self-attention模块。

  **(2) 硬件层面——细粒度量化与SIMD硬件不兼容**：
  SOTA QAT方法（LLM-QAT、EfficientQAT、TSLD等）采用channel-wise或token-wise细粒度量化（同一矩阵内多个scaling factor），在GPU上可有效恢复精度。但移动端SIMD（ARM NEON）的GeMM kernel无法处理同一矩阵内有多个scaling factor的整数MAC操作，细粒度量化无法在移动设备硬件上高效部署。标准SIMD INT8 multiplier也不支持sub-8-bit混合精度MAC，4-bit数据被零扩展（zero-extend）到byte边界当8-bit处理，浪费计算能力。

  Baseline全栈执行例子（LLM-QAT, LLaMA-58M W4A4）：
  - 算法pipeline：加载FP16预训练LLaMA-58M → 插入逐层伪量化器（layer-wise symmetric quantization, per-matrix single scale）→ 用FP16教师模型蒸馏：L_distill = (1-γ)·L_CE + γ·τ²·L_KL → STE近似梯度反向传播 → 逐层更新权重 → 量化权重以INT4格式存储。没有熵/分布感知的针对性优化，也没有token粒度的自适应量化。W4A4 BLiMP All Avg=66.9%（FP16=69.7%），NIPQ=48.9%（4-bit权重崩塌），PACT=64.9%。
  - 系统框架：PyTorch + HuggingFace Transformers。GPU训练。移动端推理使用标准SIMD INT8 GEMM kernel（如gemmlowp/QNNPACK），4-bit权重/激活需零扩展至8-bit处理。
  - 编译框架：论文未明确说明（标准PyTorch eager mode，无编译器修改）。
  - kernel调度：移动端使用标准INT8 SIMD kernel（gemmlowp或QNNPACK），仅支持逐层单scale的INT8×INT8 GEMM，不支持sub-8-bit混合精度和token级自适应量化。
  - 硬件架构：商用ARM Cortex CPU（Snapdragon/BCM2712），SIMD最高支持8-bit粒度，无自定义硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**Squat (EdgeQAT)**框架，通过三个递进的创新设计分别解决baseline各层面缺陷：

  **(1) Entropy-Guided & Distribution-Aligned Optimization 解决量化自注意力信息失真（对应缺陷1）**：
  - **熵损失 L_E**：利用query/key近似高斯的特性（q~N, k~N），推导熵H(q)∝σ_q²和H(k)∝σ_k²。最大化熵等价于MOE准则下最小化量化误差（Messerschmitt, 1971）。L_E = -log(Σ_l Σ_h log(1+σ_q²·σ_k²))，对数缩放防止梯度爆炸。
  - **分布损失 L_D**：对每层每头计算量化注意力图attn_q与FP16注意力图attn_f的余弦相似度，恢复初始token列特征。L_D = log(Σ_l Σ_h cos_sim(attn_q, attn_f))。
  - 消融验证（Figure 7）：L_D比L_E更有效（单独使用提升更多），两者组合最优。

  **(2) Token Adaptive Quantization 解决token级冗余未利用问题（对应缺陷1延伸）**：
  基于注意力图中初始token列（attn[:,0]）评估每个token的重要性，TopK选择ρ比例重要token分配8-bit、其余4-bit。Token Control Logic Module (TCLM)用Heapsort高效执行分组+拼接+分别量化。混合策略（half 4-bit + half 8-bit）优于等价位宽的均匀量化（如6-bit uniform），因重要token获得更高精度、非重要token节省计算——在Raspberry Pi上混合W4A8额外加速超40%（vs pure W8A8）。

  **(3) SIMD-based MKMP Multiplier 解决移动端硬件不兼容（对应缺陷2）**：
  - **INT4 Concatenation**：将相邻两行4-bit权重拼接入16-bit寄存器，用ARM `mla`指令（32-bit目标寄存器）同时做乘加，4-bit GEMM的计算操作数减半（vs 传统零扩展到8-bit）。
  - **INT4 Multiplier**：基于现有INT8 multiplier构建，利用bit-shift + row-wise summation累加，节省50% INT8 multiplier资源。
  - **TCLM集成**：8-bit token组走INT8 multiplier、4-bit token组走INT4 multiplier，在GeMM kernel内无缝衔接。
  - **Compiler优化**：分配计算线程重叠内存读取，缓解LLM推理的memory-bound瓶颈。

  论文方法全栈执行例子（Squat, LLaMA-58M W4A8(1:1)混合精度, OnePlus 11推理）：
  - 算法pipeline：加载FP16 LLaMA-58M → 插入逐层对称量化器（W=INT4 per-matrix, A=mixed 4/8 per-token）→ **训练**：FP16教师蒸馏 + L_E(×0.5)最大化query/key熵 + L_D(×1.0)对齐注意力图 → 每步前向TCLM根据最新注意力图动态分配token位宽 → STE反向传播 → 收敛后输出量化权重+scale。W4A8(1:1) BLiMP All Avg=69.4%（仅↓0.3% vs FP16），优于W8A8 uniform（69.3%）。
  - 系统框架：PyTorch训练（GPU）。移动端推理：自定义MKMP multiplier（ARM NEON SIMD kernel）+ gemmlowp/QNNPACK INT8 kernel复用。
  - 编译框架：论文未明确说明（从编译器层面优化内存读取时间线程分配，但未修改编译框架本身）。
  - kernel调度：MKMP Multiplier → TCLM（Heapsort分组）→ INT8 Multiplier（8-bit token组，`vmlaq_s8()`）→ INT4 Multiplier（4-bit token组，concatenation + `mla` + 内部拆分 → bit-shift累加）→ 合并结果。OnePlus 11: W4A8(1:1)=2.23 ms/tok（vs FP16=4.54, 2.04×），GPT2-97M Raspberry Pi 5: W4A4=9.74 ms/tok（vs FP16=23.04, 2.37×）。
  - 硬件架构：商用ARM Cortex CPU（Snapdragon 8 Gen 2 / BCM2712），无自定义硬件。SIMD指令粒度≥8-bit，通过INT4 concatenation突破sub-8-bit效率瓶颈。

  关键设计动机映射：
  - Baseline粗粒度QAT无query/key针对性优化 → Squat引入熵损失（最大化信息熵=最小化量化误差）+ 分布损失（恢复注意力图结构）
  - Baseline均匀量化浪费token级冗余 → Token自适应量化：按attention score分配位宽，混合精度优于等价均匀精度
  - 细粒度QAT（channel/token-wise multi-scale）移动端无法部署 → Squat坚持逐层粗粒度量化（per-matrix single scale），per-token分组仅改变位宽不改变scale
  - 标准SIMD INT8 multiplier不支持sub-8-bit混合精度 → MKMP multiplier用INT4 concatenation实现50%资源节省，TCLM无缝衔接INT8/INT4两种multiplier
  - INT4精度下降大（W4A4 BLiMP ↓1.9%）→ W4A8混合策略用部分8-bit重要token弥补精度，同时获得4-bit加速（Raspberry上额外40%）
