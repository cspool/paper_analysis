## Accelerating_Vision_Transformers_with_Adaptive_Patch_Sizes

- baseline方法是什么？
  Baseline 方法是 ViT 中使用的均匀固定 patch 划分（uniform patchification）：无论图像内容如何，每个 p×p 区域分配一个 token，产生固定数量 N = HW/p² 个 token。高分辨率图像下 token 数平方增长，而图像中大面积区域（天空、纯色背景、模糊虚化区域）信息冗余度高，与复杂区域（人脸、物体边缘、纹理丰富区域）被同等对待，造成大量注意力计算浪费。代表性 baseline 实现：(1) Vanilla ViT（timm library + FlashAttention）—— 图/像分为均匀 p×p patch，线性投影为 d_embed 维 token，全注意力计算 O(N²)；(2) 输入级 baseline（Random masking / Resizing-only）—— Random 按 APT 的压缩比随机丢弃 patch（FLIP 风格），Resizing 将大 patch resize 到 p×p 仅用单一路径编码（Quadformer 风格）；(3) 层级 token 合并 baseline（EViT, ToMe, PPT, DTEM）—— 在 ViT 各层之间合并相似 token，但固定合并比例且通常不兼容 FlashAttention。

  Baseline（Vanilla ViT，ViT-L/14@336×336）全栈执行例子：
  - 算法层：输入 336×336×3 图像 → 均匀划分为 24×24=576 个 14×14 patch → 每个 patch 经线性层 E 投影为 1024 维 token → 加上可学习位置编码 → 576+1 个 token 送入 24 层 Transformer（每层 self-attention 576² 次交互，共 ~8M FLOPs per layer）→ 取 CLS token 输出分类 logits
  - 系统框架层：论文基于 timm library 实现，使用 PyTorch 原生训练循环，未涉及 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：使用 FlashAttention-2 (Dao et al., 2022; Dao, 2024) 加速标准 scaled dot-product attention，xFormers (Lefaudeux et al., 2022) 处理 sequence packing 的 block-diagonal mask
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **固定 patch 大小不考虑内容冗余度**：均匀背景（如纯蓝天空）与复杂纹理（如鸟类羽毛）被等粒度编码，前者信息密度远低于后者但占用同等计算。高分辨率下，冗余 token 比例更高——336² 产生 576 token vs 224² 仅 196 token。
  2. **高分辨率/大模型下注意力计算瓶颈加剧**：ViT-L@448 的 GFLOPS 为 ViT-L@224 的 10.8 倍（645 vs 59.7），训练时间 31.4h vs 15.9h。
  3. **层级合并方法 token 减少有限且训练 Inference gap 大**：ToMe/EViT 在训练时可加速但在推理时由于不规则 shape 和 padding 而产生实际 wall-clock 加速远小于理论 FLOPs 减少；且多数不兼容 FlashAttention，用 weighted attention 导致更慢。
  4. **固定比例合并对图像复杂度不敏感**：纯白图像只合并 50% token 不足够，繁忙城市街景合并 50% token 可能丢弃关键信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Adaptive Patch Transformer (APT) 通过以下设计解决问题：
  (1) **多尺度熵驱动的自适应 patch 大小分配**：对图像以 quadtree 层级方式从粗到细计算像素熵 H(P) = -∑p_i log₂ p_i，低熵区域分配大 patch，高熵区域分配小 patch。低熵意味着高冗余度（如纯色背景），可用更少的 token 无损表达。
  (2) **双路 Patch Aggregation + Zero-initialized MLP**：大 patch 同时 (a) resize 到 p×p 经 E 嵌入 和 (b) 拆分为 p×p 子 patch、经 E 嵌入后 Conv2d 降采样聚合，两路通过 zero-initialized MLP 融合。ZeroMLP 初始输出为零，保证用于预训练 ViT 时初始性能不退化（无训练时等价于纯 Resizing），仅需 1 epoch 微调即可匹配原始性能。
  (3) **Sequence Packing + Block-diagonal Mask**：不同图像产生不同数量 token，拼接为单一序列 + attention mask，兼容 FlashAttention/xFormers 而不引入 padding 开销。
  (4) **位置编码插值**：大 patch 的位置编码从小 patch 网格双线性插值获得，无需学习新参数。

  对比 baseline 的全栈执行例子（APT，ViT-L/14@336×336, τ₁=5.75, τ₂=4.0）：
  - 算法层：输入 336×336×3 图像 → CPU 多核算熵：先以 56×56 patch 扫描（6×6 网格），每个 patch 计算 H(P)，H<4.0 则分配 56×56 patch；H≥4.0 则拆为 4 个 28×28 子区域，每个子区域计算 H(P')，H'<5.75 则分配 28×28 patch；否则再拆为 14×14 最终 patch → 得到变长 patch 列表（典型约 400~420 个 patch，相比 baseline 576 减少 ~28%） → 每个 patch 经双路 Aggregation（resize+子 patch Conv2d 聚合 + ZeroMLP）→ 得到 ~400+ 个 d_embed=1024 token → 位置编码从 24×24 网格插值到各 patch 对应位置 → batch 内多图 token 拼接 + block-diagonal mask → 送入 24 层 ViT（每层 attention 从 576² 降至 ~420²，GFLOPS 从 174.7 降至 76.8） → CLS token 分类 → 训练 wall-clock 从 15.9h 降至 9.9h（+61% speedup）
  - 系统框架层：论文基于 timm library 实现，使用 PyTorch 原生训练循环
  - 编译框架层：论文未明确说明
  - kernel调度层：FlashAttention-2 处理标准 self-attention，xFormers 处理 sequence packing 的 block-diagonal mask；熵计算在 CPU dataloader 上多核并行并与 GPU 前向计算重叠，无额外 GPU 开销
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（固定 patch 忽略内容冗余）→ 多尺度熵自适应 patch：纯蓝天空→64×64 patch（1 token），鸟类羽毛→16×16 patch（4 token），从 ~576 token 降至 ~400 token，GFLOPS 降 56%（174.7→76.8），且精度匹配（88.1 vs 88.2）
  - Baseline 缺陷 2（高分辨率/大模型瓶颈）→ 分辨率越高压缩越大：ViT-L@448 从 645 GFLOPS → 268 GFLOPS（-58%），speedup +86%；ViT-H@336 speedup +50%
  - Baseline 缺陷 3（层级合并不兼容 FlashAttention/训练推理 gap）→ APT 在模型前向之前完成 token 压缩，使用标准 FlashAttention，无 weighted attention 或动态 token shape
  - Baseline 缺陷 4（固定比例不感知复杂度）→ 每张图像根据内容自适应 token 数量分布（图 5：从接近上限到约 30% 下限），检测/分割任务可通过降低 τ 适应对细节的更高要求

- baseline方法是什么？
  Baseline 方法是现有的 fixed compression ratio 或 hand-crafted metric 的 visual token 压缩/剪枝方法，包括：(1) PDrop[7] —— 使用 LLM 深层（layer K之后）的第一个生成 token 对 visual token 的 cross-attention 作为重要性指标进行剪枝；(2) VisionZip[4] —— 使用视觉 token 间的相似度矩阵进行合并/剪枝；(3) VScan[5] —— 基于固定 cross-attention 指标进行 token 选择；(4) DivPrune[23]、CDPruner[20] —— 基于相似度/多样性的 token 选择。这些方法共同特点：使用固定的压缩比例和手工设计的 token 重要性度量。
  
  Baseline（PDrop 为代表的手工 cross-attention 方法，Qwen2.5-VL-7B）的全栈执行例子：
  - 算法层：用户上传高分辨率图像 + "What player is number 21?" → Vision Encoder 编码为 N_v 个 visual tokens → Projector 映射到 textual embedding space → 与 text tokens 拼接送入 LLM decoder → 完成全部 L 层 prefill → 生成第一个 response token → 取第一个 token 在深层（如 layer 28）的 cross-attention 作为 visual token 的重要性 → 基于此手工度量剪枝（若采用 step-by-step pruning） → 继续 autoregressive decode
  - 系统框架层：论文未明确说明部署 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明，但指出 VisionZip/VScan 的 dense similarity matrix 计算与 FlashAttention2 不兼容，导致 OOM
  - 硬件架构层：论文未明确说明

  Baseline 的缺陷：
  1. **手工度量不可靠（free-form generation 场景）**：如图 2 所示，当不要求简洁回答时，cross-attention 初始阶段聚焦于无关区域（而非答案相关视觉 token），导致 imprecise pruning 和 degraded output。Table 1 定量证明：PDrop 从 w/ brief 的 0.753 降至 w/o brief 的 0.406。
  2. **固定压缩比无法适应场景复杂度**：对所有输入使用相同压缩比，对小目标场景（如 DocVQA，目标区域仅占 6.8% 面积）可能剪枝不足，对大目标场景（如 VSR，目标区域占 41.6%）可能剪枝过度。
  3. **Per-step pruning 计算低效**：关键 cross-attention 仅在 LLM 深层（K层之后）出现，意味着必须保留前 K 层的全部 visual token KV cache，节省的计算和内存有限。
  4. **Dense similarity matrix 内存爆炸**：VisionZip 和 VScan 需要计算 visual token 间的稠密相似度矩阵，与 FlashAttention 的内存高效设计冲突，高分辨率输入下直接 OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：GlimpsePrune 通过以下设计解决问题：
  (1) **Glimpse Token + VIP 学习 Data-driven 剪枝度量**：在 prefill 阶段第 K 层，提取 glimpse token 对所有 visual token 的 cross-attention，与多层 hierarchical visual features 一起输入轻量 VIP 网络，输出每个 visual token 的重要性概率。该度量通过 20K GQA 样本训练得到（语言 loss + Dice/BCE 定位 loss，LVLM 参数冻结），能泛化到各种 VQA 场景。
  (2) **动态压缩比**：VIP 输出 per-token 概率分布而非固定比例，对不同图像自适应决定保留多少 token——对小目标场景（DocVQA）自动保留极低比例（3.6%），对大目标场景（VSR）保留较高比例（39.4% unrestricted）。
  (3) **One-shot 全深度 KV cache 剪枝**：仅在第 K 层执行一次剪枝，同步移除前 K 层和本层的无关 KV cache 条目——使剩余 L-K 层 prefill 和全部 decoding 阶段均在缩减序列上运行，最大化 decoding 阶段内存和 I/O 节省。
  (4) **线性复杂度 compatible with FlashAttention**：cross-attention from single query to all visual tokens 为 O(N_v × D) 线性复杂度，与 FlashAttention 兼容，不产生 OOM。

  对比 baseline 的全栈执行例子（GlimpsePrune, Qwen2.5-VL-7B, K=19）：
  - 算法层：用户上传高分辨率图像 + 问题 → Vision Encoder 编码 → Projector → 插入 glimpse token → LLM prefill 前 K=19 层（标准 forward，glimpse token 每层加上可学习嵌入） → 第 19 层提取 glimpse token cross-attention A(N_v × 28 heads) + 4层 hierarchical visual features V → VIP(M=4 self-attention blocks + 2D RoPE conditional attention) → importance map P(N_v,) → Top-K 选择保留 N_v' 个 visual token → 一次性移除不重要 token 在第 1~19 层 KV cache 中对应条目 → 丢弃 glimpse token → 剩余 9 层 prefill 在 N_v' + N_t 序列上执行 → Decoding 阶段 autoregressive 生成，每个 step attention 开销 O((N_v' + N_t) × D) ≈ O(202 × D) vs baseline O(5074 × D)
  - 系统框架层：论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明，但方法设计保证与 FlashAttention2 兼容
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline 缺陷 1（手工度量不可靠）→ Glimpse Token + Dice/BCE 定位 loss 训练：让模型学会数据驱动的剪枝度量，在 free-form generation 中稳定工作（PDrop w/o brief 0.406 → GlimpsePrune 0.939，Table 1）
  - Baseline 缺陷 2（固定压缩比）→ VIP 输出 per-token 概率 + Top-K 自适应选择：对 DocVQA 自动保留 3.6% token（仍保持 accuracy 0.962），对 VSR 自动保留 39.4%（accuracy 0.618 vs baseline 0.620）
  - Baseline 缺陷 3（per-step pruning 低效）→ One-shot pruning at layer K：prefill FLOPs 降至 69.1%，decoding 初始 KV cache 长度从 5074 降至 202.5 tokens，峰值内存降至 72.8%
  - Baseline 缺陷 4（dense similarity OOM）→ Linear-cost cross-attention + FlashAttention 兼容：高分辨率下不会 OOM，而 VisionZip/VScan 在同条件下 OOM

- baseline方法是什么？
  Baseline方法分为两类：(1) Vanilla —— 直接以固定分辨率（如448×448）将所有图像输入预训练LVLM（InternVL2-8B或Qwen2VL-7B），不对分辨率做任何特殊处理；(2) All-XN —— 将所有图像均等倍率上采样（最高pixel数放大N倍）后输入LVLM，以提升文本识别能力。两者均不区分图像重要性。
  
  全栈执行例子（All-X4，InternVL2-8B，MP-DocVQA场景）：
  - 算法层：用户上传M张文档图像+文本问题 → 所有M张图像统一被crop为16个sub-images（448×448） → ViT编码 → 与global image拼接 → 每张图像产生约1500+个visual tokens → M张图像共MnL_v个visual tokens → 与text tokens拼接送入LLM → 生成答案
  - 系统框架层：InternVL2动态分辨率处理策略(crop+resize)，论文未说明部署Serving框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  Baseline的缺陷：对不相关图像同样进行高分辨率编码，大量visual tokens与问题无关（论文指出仅12.5%的visual tokens与正确答案相关），导致GPU内存浪费和推理延迟增加，在token数受限场景下难以扩展。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：ADMIRE通过三个training-free模块动态调整多图像的分辨率：(1) TIE利用LLM第一层attention map，计算text token对所有visual token的注意力分数，聚合为每张图像的重要性分数并归一化；(2) KIE仅对Top-k张very important图像上采样，而非全部；(3) DVD对less important图像按attention保留50% token、对not important图像直接丢弃。

  对比baseline的全栈执行例子（ADMIRE-Top5-X4，Qwen2VL-7B，MP-DocVQA场景）：
  - 算法层：用户上传M张文档图像+文本问题 → TIE先用第一层LLM attention计算出每张图像的text-guided重要性分数S → Top5选为very important(p_kie)、低于0.5γ的为not important(p_Idvd)、介于0.5γ到1.5γ的为less important(p_Vdvd) → KIE对5张very important图像上采样4×(max pixels)后重新ViT编码，产生额外高分辨率visual tokens并插入原tokens之前 → DVD丢弃p_Idvd中图像的visual tokens、对p_Vdvd中图像只保留attention score最高的50% visual tokens → 最终visual tokens约1766个（vs Vanilla 1448 vs All-X4 2788） → 与text tokens拼接送入LLM → 生成答案
  - 系统框架层：论文未明确说明
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  解决对应关系：
  - Baseline缺陷1（token浪费）：DVD多级token压缩机制——对not important图像整张丢弃，对less important图像保留50%高注意力token，直接减少冗余visual tokens
  - Baseline缺陷2（无法聚焦关键信息）：TIE利用attention的文本引导评分，自动发现evidence candidate图像，使模型聚焦于包含答案的图像
  - Baseline缺陷3（全图增强导致token爆炸）：KIE仅对Top-k张图像增强分辨率，总计5*n*L_v + (M-5)*L_v个visual tokens，远小于All-XN的M*n*L_v，同时通过DVD压缩不相关图像以平衡总token数
