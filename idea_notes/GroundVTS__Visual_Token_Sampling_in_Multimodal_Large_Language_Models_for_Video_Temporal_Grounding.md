## GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding

- baseline方法是什么？
  Baseline 方法是现有的 Vid-LLM 使用 **uniform frame sampling** 的视频处理方法：给定视频后，以固定帧率（如 2 FPS）均匀采样帧，每帧经 Vision Encoder 编码为 visual tokens，经 Projector 映射后全部送入 LLM 进行时序推理和定位。这种方法将所有 visual token 平等对待，不区分其与查询的相关性。部分改进方法（如 KeyVideoLLM、VideoTree）在视频输入端引入 query-guided frame selection（基于 CLIP 等外部编码器计算跨模态相似度选帧），但仅是粗粒度帧级过滤且依赖外部编码器。

  Baseline（Qwen2.5VL-7B, uniform frame sampling @ 2 FPS, Charades-STA）全栈执行例子：
  - 算法层：输入一段视频 + 查询 "a person takes a book off a shelf" → 以固定 2 FPS 均匀采样所有帧 → Vision Encoder 逐帧编码为 visual tokens → Multimodal Projector 映射到 LLM embedding space → **所有 visual tokens 不加区分地** 拼接 text query embeddings → LLM 28 层 causal self-attention（每层对所有 tokens 做 full attention）→ 自回归生成时间边界预测 → "from 4.5s to 10.3s"（与 ground truth 6.2-12.0s 偏移）。关键帧（0-13s 书架上取书的动作）与非关键帧（13s 后的无关内容）获得相同的 token 配额和注意力计算。
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，论文未明确说明
  - 硬件架构层：GPU 推理（具体型号论文未明确说明）

  Baseline 的缺陷：
  1. **均匀采样稀释关键时刻信息**：Figure 1(a) 和 Figure 2 证明，uniform frame sampling 对每个时间段分配相同的 visual token 配额，当查询相关事件稀疏时，关键帧可能被大量无关帧稀释甚至漏掉。在 Charades-STA 上，frame rate 从 0.2 FPS 增至 2.4 FPS 时 mIoU 先升后降（峰值 47.8%），继续增至 3.0 FPS 时 mIoU 急剧下降——说明冗余 visual token 稀释了关键时序信号。
  2. **粗粒度帧级选择精度不足**：Figure 1(b) 中的方法（KeyVideoLLM、VideoTree）在视频输入端基于外部编码器做帧级筛选，但仅能粗粒度选帧而无法细粒度区分帧内哪些空间位置（token）与查询相关。此外依赖外部编码器增加了计算开销和解耦误差。
  3. **Visual token 冗余导致注意力分散**：LLM 的 self-attention 对所有 visual token 做全局计算，大量无关 token 不仅浪费计算，还可能在 attention 中引入噪声，干扰模型对关键时序边界的判断。
  4. **非均匀 token 分布下 LLM 无法有效适应**：即使强行用 query-guided 方式产生非均匀 visual token 分布，预训练 LLM 在均匀分布上训练的注意力机制难以直接适应这种分布偏移，导致训练不稳定和性能退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：GroundVTS 通过以下设计解决：

  (1) **Visual Token Sampling (VTS) 模块** —— 在 visual encoder + projector **之后**、LLM **之前**插入 query-guided token 采样，实现 **token 级别**的细粒度选择。Token Scoring: w = softmax(W_v V · W_q Pool(Q)^T / τ)，计算每个 visual token 与 query 的语义相关性。Differentiable Top-K Selection: 通过 Gumbel-Softmax + STE 实现可微分的 top-K 选择（forward 用 hard mask, backward 通过 soft relaxation 传播梯度），使 VTS 可端到端训练。

  (2) **非均匀 token 分布 + 位置编码保留** —— VTS 输出的 \tilde{V} 在时间维度上非均匀分布：高 query 相关性区域 token 密度高，低相关性区域 token 稀疏或为零，但保留 dense sampling 时的原始位置编码（mask 未选中 token 的位置），确保 selected tokens 的时间位置信息不变。

  (3) **三阶段渐进式优化** —— Stage 1 (VTS Warm-up): 冻结 LLM 训练 VTS，使其学习稳定的 token-query 相关性估计；Stage 2 (Joint LoRA Adaptation): LoRA 微调 LLM 使模型适应非均匀 token 分布（关键设计——直接用非均匀 token 微调 LLM 会因分布偏移导致训练不稳定）；Stage 3 (Grounding Fine-tuning): 在 Grounding-FT 数据集上精细微调 VTG 能力。

  对比 baseline 的全栈执行例子（GroundVTS-Q, ρ=0.5, 2 FPS, 同一视频 + 同一查询）：
  - 算法层：输入同一段视频 + 查询 "a person takes a book off a shelf" → Vision Encoder 编码全部帧 → Projector 映射 → **VTS 模块**: W_v 投影 visual tokens → W_q 投影 query → softmax 计算 每个 token 的 query-relevance w → Gumbel-Softmax + STE 选择 top 50% (ρ=0.5) 的 visual tokens → 重要区域（0-13s, 书架取书动作）token 密度高形成峰值，无关区域（13s 后）token 几乎全被抑制 → MLP 重编码 + 重归一化权重 → 保留原始位置编码 → LLM 在稀疏但高相关性的 token 序列上推理 → "from 6.0s to 12.0s"（与 ground truth 6.2-12.0s 高度吻合, mIoU 50.1 vs baseline QwenVL-G 31.7 = +18.4）
  - 系统框架层：基于 HuggingFace Transformers 推理，论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 FlashAttention，VTS 的 Gumbel-Softmax + STE 为纯 PyTorch 操作，无自定义 kernel
  - 硬件架构层：GPU 训练/推理（具体型号论文未明确说明）

  解决对应关系：
  | Baseline 缺陷 | GroundVTS 解决方案 |
  |---|---|
  | 均匀采样稀释关键信息 | VTS token-level query-guided 采样：仅在 query 相关的 spatio-temporal 区域保留高密度 token (ρ=0.5)，无关区域 token 被抑制。Figure 4: GroundVTS 在极低 token density (FPS×ρ=0.4) 时仍达 R1@0.7=29.2，远超 baseline QwenVL-G (10.2) |
  | 粗粒度帧级选择精度不足 | Token-level 细粒度选择：不是选帧而是选 token，可区分同一帧内不同空间位置的 query 相关性。Table 5: Token-Level VTS mIoU 50.1 vs Frame-Level 41.6 (+8.5 on Charades-STA) |
  | Visual token 冗余分散注意力 | ρ=0.5 即保留 50% 视觉 token，以一半的 token 预算超越全量 baseline (R1@0.7: 34.2 vs 30.5)。Figure 4(b) 展示 token efficiency 大幅领先 |
  | LLM 无法适应非均匀分布 | Stage 2 Joint LoRA Adaptation 在 LLaVA-Video-178K 上让 LLM 学习解释 query-guided 非均匀 token 序列。Table 4 消融：去掉 Stage 2 (仅 1+3) 后 R1@0.7 从 34.2 降至 15.2 |
  | 训练不稳定 | Stage 1 VTS Warm-up 先单独训练 VTS 学习稳定采样行为，Table 4: 跳过 Stage 1 (仅 2+3) 仍有 30.5 R1@0.7，但加 Stage 1 升至 34.2 |

  跨架构泛化验证（GroundVTS-I on InternVL3.5-8B）：
  - InternVL3.5 使用 fixed-number frame sampling（非 fixed-rate），VTS 仍持续提升：Charades-STA R1@0.7 +3.5, QVHighlights mAP +20.6, Hit@1 +48.6
  - 证明 VTS 的 token-level sampling 与底层 frame sampling 策略解耦，对不同 Vid-LLM 架构通用

  关键创新点总结：
  - **首次在 Vid-LLM pipeline 内部做 token-level query-guided sampling**（而非输入端 frame selection），实现细粒度 spatio-temporal 注意力分配
  - **Gumbel-Softmax + STE 使离散 token 选择可端到端训练**，无需外部 reward model 或 RL
  - **三阶段渐进式优化解决分布偏移问题**：warm-up → joint adaptation → task fine-tuning，消融实验证明每个阶段不可或缺
  - **Token-level > Frame-level**: 同一帧内可区分空间位置与 query 的相关性，比帧级选择精度高 8.5 mIoU
