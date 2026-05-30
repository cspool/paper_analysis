## Binarized Diffusion Model for Image Super-Resolution

- baseline方法是什么？
  Baseline 是扩散模型图像超分辨率 (SR) 的二值化，直接套用现有二值化方法（BNN、DoReFa、XNOR、IRNet、ReActNet、BBCU）到 SR3 的 UNet 架构上。BBCU 是 SOTA 二值化 SR 方法，但仅针对端到端 CNN 设计，不涉及扩散模型的多步迭代。BBCU 的核心结构是 basic binary conv unit (BBCU)，移除了 BN 层以适配低层视觉任务。这些 baseline 方法在扩散模型上直接应用时面临三个核心缺陷：

  **(1) 维度不匹配（Dimension Mismatch）**：UNet 中 encoder 逐层下采样（H×W 减半、C 翻倍）、decoder 逐层上采样，特征维度不断变化。现有二值化方法依赖 identity shortcut 传递 FP 信息以补偿二值化信息损失（因 1-bit 表达严重受限），但维度变化使 shortcut 无法使用，切断了 FP 信息在全网络的传播路径。

  **(2) 融合困难（Fusion Difficulty）**：UNet 的核心结构 skip connection 需将 encoder 特征与 decoder 特征融合。常用方法 concatenation 导致输出维度翻倍（与 ResBlock 输入维度不匹配），addition 则因两种特征值域差异巨大（Fig. 3d 显示 encoder 和 decoder 特征的激活范围可达数倍差距）导致小值域特征被"遮盖"而无法有效参与融合。

  **(3) 多步激活分布变化（Multi-step Activation Distribution Shift）**：扩散模型需多步迭代去噪（T=2000），不同 timestep 下激活分布差异显著（Fig. 4 可视化：相邻 timestep 分布相似，但间隔大时分布剧烈变化）。二值化模块（Sign 函数和 1-bit 权重）对激活分布极度敏感——分布变化加剧了二值化后的信息损失（Sign 函数放大零附近值的差异 [38]），静态的 bias 和 RPReLU 无法适配多步的极端分布变化。

  Baseline 全栈执行例子（BBCU + UNet, ×2 SR, 4.82M Params）：
  - 算法pipeline：加载 SR3 的卷积 UNet → 所有卷积替换为 BBCU（二值化权重+激活，无 BN）→ 下采样/上采样用 stride 卷积 / PixelShuffle（维度变化，无 identity shortcut）→ skip connection 用 concatenation + 1×1 二值卷积调整维度 → 单组 bias+RPReLU 处理所有 timestep → 50 步 DDIM 推理。Manga109 PSNR=31.99dB, LPIPS=0.0326（全精度 SR3 PSNR=35.11dB, LPIPS=0.0161）。Baseline（无任何改进）PSNR 仅 27.66dB, LPIPS=0.0780。
  - 系统框架：PyTorch，A100-80G GPU 训练，无自定义 Serving/调度框架。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（标准 PyTorch 推理，未实现 XNOR/bit-count 定制 kernel）。
  - 硬件架构：论文未明确说明（NVIDIA A100 GPU，无自定义硬件）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 BI-DiffSR，首个专门为扩散模型 SR 设计的二值化方法，从结构（structure）和激活（activation）两个维度分别解决 baseline 的三个核心缺陷：

  **(1) CP-Down/CP-Up 解决维度不匹配（对应缺陷 1）**：
  Baseline 使用 stride 卷积或 PixelShuffle 改变维度，导致 ResBlock 输入输出维度不一致，identity shortcut 无法使用。BI-DiffSR 将所有维度变化操作统一封装到 CP-Down/CP-Up 中：CP-Down 先用双分支卷积（维度不变，可加 shortcut）处理输入，再用 Pixel-UnShuffle 降低分辨率+增加通道数；CP-Up 先用双分支卷积处理，再用 Pixel-Shuffle+Concat 提升分辨率。这样 ResBlock 内部始终保持维度一致，identity shortcut 全程可用，FP 信息流不被切断。消融验证：+Identity 使 PSNR 从 27.66→29.29 dB (+1.63)，+CP-Down&Up 进一步提升至 31.08 dB (+1.79)。

  **(2) CS-Fusion 解决融合困难（对应缺陷 2）**：
  Baseline 用 concatenation（维度翻倍，不匹配）或 addition（值域差异大，小值被覆盖）。CS-Fusion 利用 channel shuffle 思想：将两个输入特征 x1, x2 按奇偶通道索引交叉重组为 x1^sh（x1 奇数通道 + x2 偶数通道）和 x2^sh（x1 偶数通道 + x2 奇数通道）。shuffle 后两个特征的值域接近（Fig. 6 可视化验证），再分别经过二值卷积后相加融合。此设计维度不变（每个 shuffle 输出仍为 H×W×C），可加 identity shortcut，且 channel shuffle 为零开销索引操作。消融验证：CS-Fusion PSNR=31.99 dB vs Concat=31.08 dB (+0.91), Split=29.67 dB (+2.32), Add=18.89 dB (完全失效)。

  **(3) TaR/TaA 解决多步激活分布变化（对应缺陷 3）**：
  Baseline 使用单组静态 bias + RPReLU 处理所有 timestep。TaR/TaA 受 MoE 启发，设置 K=5 对 (bias, RPReLU)，将总 timestep T=2000 均分为 5 组，每组 timestep 激活专属的一对参数。这等价于将多步过程分割为 5 个子区间，每个区间内激活分布变化较小，bias/RPReLU 只需适配区间内分布，降低了学习难度。相邻 timestep 分布相似性使得固定分组策略有效。每个 timestep 仅激活 1 对参数，推理无额外计算开销。消融验证：+TaR&TaA PSNR 从 31.99→32.66 dB (+0.67)，LPIPS 从 0.0261→0.0200；需要同时使用 TaR 和 TaA（单独使用反而降低性能：仅 TaR=29.27dB, 仅 TaA=29.13dB vs 无=31.99dB），说明输入/输出端激活调整需协同工作。

  论文方法全栈执行例子（BI-DiffSR, ×2 SR, 4.58M Params）：
  - 算法pipeline：加载卷积 UNet（4 层 E-B-D, C=64）→ **结构层面**：所有 ResBlock 用 BI-Conv block（TaR→Sign→XNOR-Conv→TaA→+shortcut, 维度不变）→ CP-Down（双分支卷积+PixUnShuffle）→ CP-Up（双分支卷积+PixShuffle+Concat）→ skip connection 用 CS-Fusion（channel shuffle + 双分支二值卷积 + addition）→ **激活层面**：每个 BI-Conv block 的 TaR/TaA 有 K=5 对 (bias, RPReLU)，推理时按 floor(5*t/2000) 选对应组 → 50 步 DDIM 推理。Manga109 PSNR=33.99dB（BBCU=31.99dB, SR3 FP=35.11dB），LPIPS=0.0172（BBCU=0.0326, SR3=0.0161），达到全精度模型 93.6% 的感知质量。
  - 系统框架：PyTorch + 自定义 BI-Conv/CP-Down/CP-Up/CS-Fusion 模块 → 2× A100-80G 训练 → 无修改 Serving 框架。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。理论加速基于 XNOR+bit-count 替代浮点 MAC（32× 内存节省，64× 计算节省），但未实现定制 CUDA kernel。
  - 硬件架构：论文未明确说明（NVIDIA A100 GPU 训练+推理）。

  关键设计动机映射：
  - UNet 维度变化切断 FP shortcut → CP-Down/CP-Up 将维度变化隔离到独立模块，确保 ResBlock 维度一致、shortcut 全程可用
  - Skip connection 中 encoder/decoder 特征值域差异大无法融合 → CS-Fusion 通过 channel shuffle 平衡值域，使二值卷积能有效融合
  - 扩散模型多步迭代中激活分布随 timestep 剧烈变化 → TaR/TaA 用分组 (bias, RPReLU) 将 timestep 分段，降低单组参数适配范围
  - 分开使用 TaR 或 TaA 反降低性能 → 输入端（TaR）和输出端（TaA）激活调整需协同作用，因二值化前后均需适配当前 timestep 分布
