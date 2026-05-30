## Modulated Diffusion (MoDiff): Accelerating Generative Modeling with Modulated Quantization

- baseline方法是什么？
  Baseline是两种独立的扩散模型加速方法：(1) Caching方法（如DeepCache）——利用扩散过程相邻时间步之间特征的相似性，每隔N步缓存一次high-level features并复用，跳过中间步的重计算。缺陷：重用历史计算结果引入approximation error，且该误差在迭代过程中不断累积（Figure 1a显示relative ℓ₂ distance在最终step达到40%），需要careful design of reuse schedules甚至retraining来弥补；(2) Post-Training Quantization (PTQ)方法（如Q-Diffusion、LCQ）——在训练无关的前提下估计scaling factor将网络参数量化到低位宽整数。缺陷：扩散模型中activation tensor范围在不同时间步之间变化显著，且每步内存在异常值（outliers with long-tailed distributions，Figure 1b），使得低bit activation量化时scaling factor难以同时最小化clipping error和rounding error——现有方法只能将activation quantize到8-bit，更低bit精度（<6-bit）质量急剧塌陷。

  全栈执行例子（以DDIM on CIFAR-10, T=100 denoising steps, Q-Diffusion W8A4 baseline为例）：
  - 算法层：DDIM采样过程的每一步中，U-Net的前向pass被量化为W8A4精度。每层线性operator A^{(l)}：量化激活 → Q(a_t^{(l)}) → 矩阵乘法 A^{(l)}(Q(a_t^{(l)})) → 输出 ô_t^{(l)}。不同时间步的计算相互独立，无法利用时序冗余。当activation降到4-bit时，activation range在时间步间大幅变化导致per-channel scale s无法覆盖全范围——一部分值被clip，另一部分值被round到过粗粒度。FID从W8A8的4.21快速恶化到W8A4的24.09（LCQ）甚至332.75（Q-Diff）。
  - 系统框架层：PyTorch + DeepSpeed inference（评估GBops而非实际加速）。Q-Diffusion通过MSE reconstruction loss校准量化参数。动态量化在每步运行时计算scale factor。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明——未实现实际GPU kernel加速，仅用DeepSpeed计算理论Bops。
  - 硬件架构层：论文未明确说明硬件平台——明确表示"Implementing acceleration on specialized hardware is beyond the scope of this work"。

  Baseline两大核心缺陷：
  1. **Caching methods' accumulated error**: 直接复用历史激活（approximate without error tracking）导致误差在时间维度上累积——每一步的偏差传到下一步后被放大，误差随时间步数线性以上增长。
  2. **PTQ methods' activation range diversity**: 不同时间步的activation分布不同（大的range variation + outliers），static/deep calibration得到的scaling factor无法在所有时间步都有效。低bit时clipping error或rounding error必定有一个占主导。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Modulated Diffusion (MoDiff)，通过**两个协同的数学设计**同时解决上述两个缺陷：

  **(1) Modulated Quantization**——解决"Activation range diversity"缺陷：
  重构扩散采样过程中每层线性算子的计算范式，从直接量化activation（基线：Q(a_t)）转为量化时序差分（MoDiff：Q(a_t - a_{t+1})）。时序差分 a_t - a_{t+1} 的数学性质：(a) 分布范围比原始activation小10×以上（Figure 1b：橙色vs蓝色violin plot高度对比）；(b) 分布更集中、异常值更少（Figure 1b：橙色violin plot宽度更窄、尾部更短）；(c) 在不同时间步之间范围更一致。根据Theorem 4.3，量化误差正比于输入范围的平方——范围缩小10× → 量化误差降低100× → 可用低3-4 bit达到同等精度。这是从数学上通过输入变换（subtraction of adjacent time steps）改变被量化量的统计特性，而非优化量化器本身——因此MoDiff与具体量化方法（Q-Diff/LCQ/LTQ）正交，可直接叠加。

  **(2) Error-Compensated Modulation**——解决"Caching/Modulation积累误差"缺陷：
  标准调制方法（Eq. 24-25）直接累加量化输出：ô_t = A(Q(a_t - a_{t+1})) + ô_{t+1}。但ô_{t+1}本身含有量化误差，叠加新的量化误差后误差传递方式为指数增长（Theorem 4.4, Eq. 30: Σ 2^{T-k-1} c ∥A∥² ∥a_k-a_{k+1}∥²）。MoDiff引入中间变量 â_t = Q(a_t - â_{t+1}) + â_{t+1}（Eq. 13），使当前步的量化误差 e_t = a_t - â_t 被显式追踪并通过输入反馈到下一时间步（因为下一时间步的差分基不再是 a_{t+1} 而是 â_{t+1} = a_{t+1} - e_{t+1}）。这使误差从"累积"变为"被吸收"——Theorem 4.4, Eq. 31证明误差以 (2c)^{T-k-1} 速率指数衰减（当 c<1/2 时），而非增长。直观理解：每一步的量化误差被显式记入误差追踪量ê，并在下一时间步通过输入被算子处理（补偿计算遗漏的A(ê_{t+1})）。

  全栈执行对比baseline（以DDIM on CIFAR-10, T=100 denoising steps, LCQ+MoDiff W8A4为例）：
  - 算法层：同U-Net架构，但每层linear operator的执行变为MoDiff范式：
    1. Step T（第一次迭代）：â_T = Q(a_T), ô_T = A(â_T)，使用全精度activation的量化计算（warm-up, 4-5步渐进收敛）
    2. Step T-1→1（后续迭代）：每层执行 â_t = Q(a_t - â_{t+1}) + â_{t+1} → ô_t = A(Q(a_t - â_{t+1})) + ô_{t+1}。时序差分 a_t - â_{t+1} 的range远小于 a_t 原始range → 4-bit量化误差大幅降低。误差 e_t = a_t - â_t 被自动追踪到 â_t 中，在t-1步作为输入被补偿。
    W8A4时LCQ+MoDiff FID保持4.38（vs LCQ baseline的24.09或33.97），甚至W8A3时FID=4.14（vs baseline的143.39/90.34）。10× computation savings (W8A3 154 GBops vs FP32 1636 GBops)。
  - 系统框架层：PyTorch + DeepSpeed Bops评估。Calibration phase: Q-Diff+MoDiff使用reconstructed calibration dataset（捕捉时序差分而非原始激活）。Bias removal（所有被MoDiff改造的层去除bias项保证线性）。Layer-wise reconstruction（逐步独立优化量化参数）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明——未实现实际GPU kernel加速。
  - 硬件架构层：论文未明确说明硬件平台。MoDiff引入的额外开销：(a) 1次matrix addition（a_t - â_{t+1}）；(b) 1次output addition（+ ô_{t+1}）；(c) 1次额外的dequantization of Q(a_t - â_{t+1})。但matrix multiplication占主导（dominant cost），这些额外操作与matrix multiplication相比overhead negligible。Memory overhead：需额外存储 â_t 和 ô_t 每层——单张图CIFAR-10上W8A4额外内存~4MB（Table 6: 36.4 MB vs baseline 35.09 MB W8A32）。

  设计思路核心：
  论文的核心洞察是：**乘法算子A的线性性质允许将"直接计算A(a_t)"转化为"增量计算A(a_t - a_{t+1}) + ô_{t+1}"，而时序差分a_t - a_{t+1}比原始激活a_t更易于量化的本质是其在统计上具有更小range和更少outliers**。这两个数学事实的结合催生了MoDiff——将扩散模型的加速从"优化量化器"（传统的PTQ做法）和"跳过计算"（传统的caching做法）两个独立方向，统一为"改变被量化量的统计特征来提高量化效率"的单一视角。而误差补偿则通过显式追踪和反馈量化误差，将caching的accumulated error问题转化为数学上被指数衰减控制的残余误差。Remark 4.1证明了caching方法是MoDiff在0-bit差分时的特例（当时序差分低于阈值时），使得MoDiff在概念上统一了caching和量化两个方向。该框架对具体量化方法（Q-Diff/LCQ/LTQ）、模型架构（U-Net/DiT/LDM）、采样器（DDIM/DDPM/DPM/PLMS）和数据分辨率（32×32到256×256到MS-COCO）都具有普适性。
