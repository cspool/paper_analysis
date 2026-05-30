## HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models

- baseline方法是什么？
  现有 VLM visual token 剪枝方法主要基于两种思路：(1) LLM 内部剪枝——FastV 在 LLM decoder 前面几层丢弃低注意力 token，PyramidDrop 逐层金字塔式减少 token；(2) 静态度量选择——VisionZip 基于 token 相似度/diversity 选择保留 token，DivPrune 基于 diversity 选择。这些方法的共同缺陷是：未充分利用视觉编码器本身的内在注意力结构，部分方法依赖 CLS token（SigLIP 等无 CLS token 的编码器无法使用），且大多需要针对不同模型精心调参。

  Baseline 全栈执行例子：
  - 算法层：FastV → ViT 编码 576 tokens → projector → LLM 第 2 层后丢弃 attention score 低的 visual token → 后续解码层仅处理保留 token
  - 系统框架层：HuggingFace Transformers + LMMs-Eval 评估
  - 编译框架层：论文未明确说明
  - kernel调度层：标准 PyTorch attention（或 FlashAttention），token 丢弃发生在 LLM decoder 内部
  - 硬件架构层：NVIDIA A100 40GB，单卡推理

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 HiPrune，基于视觉编码器 ViT 的**分层注意力模式（Hierarchical Attention Pattern）**进行 training-free token 剪枝。核心发现：(1) 中间层注意力集中在图像 main object（IoU 验证：中间层 top-10% token 与 COCO segmentation mask 重叠度最高）；(2) 深层注意力均匀分布，编码全局信息。基于此设计三种 token 类型：
  - **Anchor Tokens**：从中间 object layer l 选 attention score 最高的 token，保留 object-centric 细节
  - **Buffer Tokens**：Anchor 的空间邻居（上下左右），抗注意力噪声
  - **Register Tokens**：从 ViT 输出层选高注意力 token，补充全局上下文信息
  - **HiPrune++ 可选**：额外保留与 text embedding 余弦相似度高的 token，增强指令跟随

  论文方法全栈执行例子：
  - 算法层：Image → ViT CLIP-L/14 (24 layers) → 在第 9 层提取 `mid_attn` (object layer attention) → topk 选 Anchor + spatial neighbor Buffer → 在第 24 层（输出层）提取 `deep_attn` → 补充 Register → 可选 text cosine similarity 补充 (HiPrune++) → 576 tokens → N' (如 192) tokens → projector → LLM → 生成
  - 系统框架层：HuggingFace Transformers + LMMs-Eval + calflops
  - 编译框架层：论文未明确说明
  - kernel调度层：HiPrune 在 ViT 输出后、projector 前执行，与 FlashAttention 完全兼容（不修改 attention 内部计算）
  - 硬件架构层：NVIDIA A100-PCIE 40GB，部分实验 RTX 5090

  解决对应关系：
  | Baseline 缺陷 | HiPrune 解决方案 |
  |---|---|
  | 未利用 ViT 内部注意力结构（仅用 LLM 内部 attention 或静态相似度指标） | 从 ViT 中间层（object-centric）和输出层（global）分层提取注意力信号，无需 training 或 external guidance。Table 1：中间层 top-10% token IoU 0.80×~1×（CLIP-L/SigLIP/DeiT 等均适用），证明模式跨编码器通用 |
  | 依赖 CLS token 的方法（SparseVLM 等）无法用于 SigLIP | 使用 global mean attention（Eq.3: a_i = mean_head sum_n A[h,n,i]），不依赖 CLS token，Qwen2.5-VL (SigLIP) 上 SOTA (Table 4: 11.1% tokens 保持 93.0%) |
  | 需要额外 training/merging（ToMe、PuMer） | 完全 training-free、plug-and-play：仅需设置 object layer l 和 α=0.1。LLaVA-1.5 上 l=9/LLaVA-NeXT 上 l=9/Qwen 上 l=16，通过 dispersion-based searching 自动确定 |
  | Token merging 与 FlashAttention 不兼容 | HiPrune 是纯 pruning（select tokens by index），不修改 attention 计算，与 FlashAttention 完全兼容 |
  | 低 token budget 下指令跟随能力严重退化 | HiPrune++ 通过 text cosine similarity 补充 β=0.1 token 缓解：LLaVA-1.5 64 tokens (11.1%), HiPrune 92.7% vs HiPrune++ 96.1%, POPE 73.0% vs 84.3%，证明 text guidance 在低 budget 下关键 |
  | 高分辨率场景（LLaVA-NeXT 2880 tokens）压缩效率差 | 保留 2/9 tokens (640) 保持 99.7% (HiPrune++), 甚至 5.6% tokens (160) 仍保持 94.4% (HiPrune++) |

  消融实验关键发现：
  - Token types: 去掉 Register → 性能下降最显著 (Table 6b: w/o Register Avg 97.9%)，证明全局信息最关键
