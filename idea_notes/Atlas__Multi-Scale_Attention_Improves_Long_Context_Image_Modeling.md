## Atlas__Multi-Scale_Attention_Improves_Long_Context_Image_Modeling

- baseline方法是什么？
  Baseline 方法是 Windowed Self-Attention (WA) / Window-ViT，即标准 ViT 中将 self-attention 限制在局部 k×k 窗口内的变体。WA 的 computational complexity 为 O(N·k²)（相对于全局 self-attention 的 O(N²)），但存在两个关键局限：(1) Limited Receptive Field —— 每个窗口独立处理信息，不同图像区域之间无法直接通信；(2) Boundary Effects —— 跨越多窗口的物体/特征无法在单次 attention 操作内建模。其他 baseline 包括：MambaVision（hybrid SSM+attention，线性复杂度但长上下文表现差）、FasterViT（Hierarchical Attention）、LongViT（Dilated Attention，也声称 O(log N) 通信但未利用局部性）、Swin（Shifted Window，两阶段窗口通信但每个 token 仅能看到相邻窗口）。

  Baseline（Window-ViT, Base-scale, 1024×1024 HR-IN100）全栈执行例子：
  - 算法层：输入 1024×1024×3 图像 → Conv Stem → patchify 为 64×64=4096 tokens（patch_size=16）→ 每个 MSA-like block：将 feature map 划分为 16×16=256 个 non-overlapping windows，window_size=16×16 → 每个 window 内执行标准 Multi-Head Self-Attention（QKV projection + Softmax(QK^T/√d) @ V）→ 所有 window 同时计算 → FFN → 下一个 block → 重复 N 层 → 最终对所有 token 取 mean/CLS token 做 readout → 输出 100-way classification logits。虽然 WA 每个 block 的复杂度仅为 O(4096×256)=O(1M)（远小于全局 attention 的 O(16.8M)），但不同窗口之间的 token 从未直接交互——跨窗口的信息融合仅在最终 readout 层发生。
  - 系统框架层：基于 PyTorch + timm library 训练，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention kernel（论文未明确说明是否使用 FlashAttention）
  - 硬件架构层：8×H100 GPU 节点，论文未进一步说明硬件细节

  Baseline 的缺陷：
  1. **有限的感受野（Limited Receptive Field）**：如图 3 所示，WA 的每个 token 在 attention 中仅能看到同窗口内的 K=256 个 token（而非全部 4096 个）——特征在深层之前无法跨越窗口通信，对于高分辨率图像（如 4096×4096→65536 tokens），同一物体跨越多窗口的概率显著增大，局部注意力丢失关键全局语义。
  2. **边界效应（Boundary Effects）**：物体跨越窗口边界时被分割为独立 token 组分别处理，其关系只能在最终 readout 层间接学习，导致高分辨率下有效特征交互不足。
  3. **SSM-based（MambaVision）的序列累积限制**：状态空间模型虽为线性复杂度，但其递归/卷积性质在极长序列上信息积累不足——信息在序列中单向/双向传播需 O(N) 步，导致 4096px 分辨率仅 23.36% 精度（Table 2）——远逊于注意力机制的任意 token 对直接交互。
  4. **Dilated Attention（LongViT）未利用局部性**：LongNet 的 dilated attention 通过指数间隔采样实现 O(log N) 通信，但忽略了图像的 2D 空间局部性——视觉任务中相邻像素高度相关，远距离 dilation 跳过邻近关键 token 导致信息丢失，Table 3 显示 MSA 比 Dilated Attention 快 2.39× 且准确率高 20.9%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Multi-Scale Attention (MSA) + Atlas 架构通过以下设计解决问题：
  (1) **Multi-Scale Hierarchical Representation**：使用 4×4 strided max-pool 从 finest scale 开始迭代生成 O(log_S N) 个空间尺度的粗粒度摘要表示。粗尺度 token 用极少的 token 数（如 scale-4: 仅 1 window×256 tokens = 整个 image 的全局摘要）概括大范围区域信息。
  (2) **Bi-directional Cross-Scale Communication**：Top-Down: 每个细粒度窗口内的 token 通过 cross-attention 读取所有更粗尺度的对应 child window tokens——scale-1 的每个 token 可同时 attend 到 scale-1/2/3/4 的所有 K 个 token，其中 scale-4 的 child tokens 已是经过所有中间尺度积累的全局上下文。Bottom-Up: 每个粗粒度 token 通过 cross-attention 从直接 parent window 恢复局部细节（在 strided max-pool 中丢失的 fine-grain 信息）。
  (3) **O(log N) 通信复杂度 + O(N·K·log N) Runtime**：每个 token 到任意其他 token 通过至多 O(log N) 个中间粗尺度 token 传播——通信步数远少于 WA（无法直接通信）和 SSM（O(N) 步），runtime 优于全局 attention（O(N²)）和 dilated attention（同样声称 O(log N) 但实际更慢 2.39×）。
  (4) **Progressive Scale-Dropping (Atlas Architecture)**：以 L 个 macro-stage 逐步放弃最精细尺度，聚焦计算资源于高层特征。如 D={2,2,2,6} 配置：前 2 个 block 积累所有尺度 cross-scale 信息 → 第 3-4 block 仅保留 scale-2/3/4 → 第 5-6 block 仅保留 scale-3/4 → 第 7-12 block 仅处理 scale-4（最粗全局尺度）。比传统 Conv downsampling 更快（38m vs 40m）且更准（70.09% vs 56.14%）。
  (5) **QKV Caching（Appendix C）**：缓存每个 scale 的 QKV 投影以复用跨 cross-attention 操作，避免多尺度场景下的重复计算。

  对比 baseline 的全栈执行例子（Atlas-B/16, D={d1,...,d_L}, 1024×1024 HR-IN100）：
  - 算法层：输入 1024×1024×3 图像 → Conv Stem (4×4 patches) → patchify 为 64×64=4096 tokens（X^(1)）→ 初始多尺度构建: S(stride=4,16): X^(1)→X^(2): 16×16=256 tokens, X^(2)→X^(3): 4×4=16 tokens → L=3 scales → Stage 1 (前 d1 个 MSA blocks): 每个 block —— Summarize: 更新 X^(2)+=MaxPool(X^(1)), X^(3)+=MaxPool(X^(2)) → Top-Down: X^(1) 每个 window(16×16 tokens) cross-attend to [window, X^(2) child(4×4), X^(3) child(1×1)] → X^(2) cross-attend to [window, X^(3) child] → X^(3) 做标准 window-attention (scale=L) → Bottom-Up: X^(2) cross-attend parent X^(1), X^(3) cross-attend parent X^(2) → Stage 2: scale-1 discarded, 仅处理 X^(2..3) → Stage 3: scale-2 discarded, 仅处理 X^(3) → X^(3) 经 readout 输出 100-class logits → 每个 token 的全局上下文路径: X^(1)_token_i → (Top-Down) 读取 X^(3) child token（该 token 已通过 3 层 bottom-up 聚合了 image 所有其他区域的 fine-grain 信息）→ 信息混合完成，至多 log_16(4096)≈3 步
  - 系统框架层：基于 PyTorch + timm library 训练循环，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention kernel（cross-attention 使用标准 MHA），QKV caching 减少跨 scale 重复投影；论文未说明使用 FlashAttention 但 MSA 的 windowed/cross-attention 模式与之兼容
  - 硬件架构层：8×H100 GPU 节点

  解决对应关系：
  | Baseline 缺陷 | Atlas/MSA 解决方案 |
  |---|---|
  | WA 有限感受野：token 仅见同窗口内 256 tokens | 多尺度表示：scale-4 的 256 个 token 是整个 image 的全局摘要，scale-1 的每个 token 通过 top-down cross-attention 直接读取该全局上下文 |
  | WA 边界效应：跨窗口物体无法直接建模 | Cross-scale communication 不限窗口：每个 token attend to 同 scale 窗口内的 token + 所有 coarser scale 对应 region 的 summary tokens → 跨窗口信息通过粗尺度 summary 间接传播 |
  | SSM (MambaVision) 长序列信息衰减：4096px 仅 23.36% | MSA bi-directional: 每个 token 在至多 O(log N) 步内与任意其他 token 交互（通过粗尺度中间 token）——4096px 达 55.84%（+32.48%） |
  | Dilated Attention (LongViT) 忽略局部性 | MSA 的 windowed cross-attention 仅在粗尺度执行跨区域通信、细尺度保持局部窗口 → 同时利用局部性（同一窗口）和全局性（coarse scale summary） |
