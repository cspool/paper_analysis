## VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

- baseline方法是什么？
  Baseline 是标准 Transformer-based VLM（以 LLaVA-1.5 为代表），使用 Vicuna LLM + CLIP 视觉编码器 + 投影层架构。视觉信息通过 cross-modal projector 将 CLIP 的视觉特征映射为与 LLM 同维度的 visual tokens，拼接到文本 token 序列中，通过 LLM 的 causal self-attention 实现跨模态融合。Baseline 的核心缺陷来自于 Transformer 自注意力机制的二次复杂度：(1) 推理时每生成一个 token 需要 O(N) attention 计算和 O(N) KV cache 内存增长，长序列（多轮对话、高分辨率图像、长文档）下推理延迟和内存开销快速膨胀，不适合边缘设备部署；(2) 即使使用 FlashAttention 优化，KV cache 仍随序列长度线性增长，在 24K+ tokens 时 GPU 内存成为瓶颈。

  Baseline 全栈执行例子（LLaVA-1.5 7B 推理一个包含 576 image tokens + 20 text tokens 的 VQA 请求）：
  - 算法层：输入图像 → CLIP-L(ViT-L/14, 336×336) → Z_v ∈ R^{576×1024} → Projector MLP → H_v ∈ R^{576×4096} → 拼接到 text tokens → Vicuna-7B 32 层 Decoder，每层 causal self-attention 在 (576+N_text+N_generated) 个 token 上计算 softmax(QK^T/√d)·V → 自回归逐 token 生成 → 输出 answer。每生成一个 token 需 O(576+N_text+N_gen) attention 计算，KV cache 线性增长。
  - 系统层：PyTorch + Transformers + DeepSpeed ZeRO。推理使用 FlashAttention 加速。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention kernel（IO-aware tiling），降低 HBM 访问但仍保持二次复杂度。
  - 硬件架构层：NVIDIA A100 GPU。

  Baseline 核心痛点：
  1. Transformer 自注意力使推理延迟和内存随序列长度线性以上增长（O(N²) 总计算量），限制了 VLM 在资源受限设备上的部署
  2. Linear RNN（如 RWKV）已经证明在纯文本 LLM 上可以匹配 Transformer 性能并实现 O(1) 推理，但在多模态 VLM 领域尚未被探索

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 VisualRWKV，首次将线性 RNN（RWKV）架构应用于多模态 VLM，通过三项创新设计使 RNN-based VLM 达到与 Transformer-based LLaVA-1.5 竞争的性能：

  **设计 1: Data-dependent Recurrence → 提升 RNN 模型容量**
  Baseline 中最初的 VisualRWKV-Base 使用了 RWKV-5 的数据独立 recurrence（固定 token shift μ + 固定 time decay w），模型容量有限。VisualRWKV 将 Token Shift 升级为 Data-dependent：`ddlerp(a,b) = a + (b-a) ⊙ lora(a + (b-a) ⊙ μ_x)`，其中 lora(x) = λ + tanh(xA)B 使用低秩矩阵引入输入依赖的偏移量。同时将 Time Decay 从固定 w 变为动态 `w_t = exp(-exp(lora_d(ddlerp_d(x_t, x_{t-1}))))`，使每个 token 在每个 channel 上根据内容决定遗忘速度。这一设计使模型能动态调整"记住什么、遗忘什么"，显著提升在 VQA/SQA/GQA 等 benchmark 上的表现（Table 1: VQA 51.08→65.82, SQA 41.94→46.55）。

  **设计 2: Sandwich Prompt → 解决 RNN 无法回溯的问题**
  Transformer 的 self-attention 允许模型在任何时刻从 KV cache 中检索任意历史 token，而 RNN 的序列特性意味着信息一旦被"读入"state 后，模型无法直接回溯原始 token。传统的 Image-First 或 Image-Last prompt 分别导致模型处理图像时不考虑问题（丢失上下文）或先读问题后读图像时忘记问题（RNN 信息被图像 token 覆盖）。Sandwich Prompt 将图像 token 插入 instruction token 中间（即 [Q_prefix]+[image]+[Q_suffix]），使模型：(a) 先读 prefix 激活正确的"检索意图"；(b) 带着意图处理图像信息，提取与问题相关的视觉特征；(c) 再读 suffix 完成最终回答。这种设计利用了 RNN 在短时间内（prefix 长度内）可以保持良好局部记忆的特性，同时避免了长序列中的遗忘问题。Table 3 证明 Sandwich Prompt 在所有三种 prompt 中表现最佳。

  **设计 3: 2D Image Scanning → 解决 RNN 单向性与图像多向性的矛盾**
  RWKV 本质上是为 1D 因果语言序列设计的，视觉序列（2D patch grid）无因果关系，直接进行单向扫描会丢失空间结构信息。VisualRWKV 将 RWKV blocks 的方向交替排列为 Forward → Backward → Forward → Backward（BiDir），或 Forward/Backward/Upward/Downward 四向交替（MultiDir）。每一层的扫描方向固定（训练和推理保持一致），不增加任何参数和计算量（仅改变 token 输入顺序）。这种交替设计使模型在不引入额外开销的情况下获得了 2D 空间感知能力。Table 4 证明 BiDir（VQA 65.62）和 MultiDir（66.04）均显著优于 UniDir（51.03）。

  论文方法全栈执行例子（VisualRWKV 7B 推理一个 VQA 请求）：
  - 算法层：输入图像 → CLIP-L（ViT-L/14, 336×336, 冻结）→ Z_v [576×1024] → Projector（2 层 MLP, 可训练）→ H_v [576×4096] → Sandwich Prompt 构建：[System|Q_prefix|576 image tokens|Q_suffix] → 32 层 RWKV-6 Block（每层: ddlerp Token Shift → dynamic decay w_t → WKV linear attention [matrix state S∈R^{64×64} per head, O(1) update] → SiLU gate → LayerNorm → output → channel mixing → residual）→ LM Head → logits → next token。逐 token 生成时，RWKV block 每步仅需更新 64×64 的矩阵 state（无 KV cache），计算 O(1)，内存恒定。
  - 系统层：NVIDIA PyTorch NGC Container (23.07-py3) + lightning 1.9.5 + DeepSpeed 0.12.6。开源代码：https://github.com/howard-hou/VisualRWKV。
  - 编译框架层：论文未明确说明。
  - kernel调度层：RWKV 的 WKV 计算使用 parallel scan 实现训练并行化，推理时切换为纯循环模式（O(1) per token）。RWKV-6 kernel 在 16K 序列时比 Flash Attention v2 快 4.2×（来自 RWKV-6 论文）。
  - 硬件架构层：训练 8×A100-80GB GPU；效率对比单张 L20-48GB。VisualRWKV 7B 比 LLaVA-1.5 7B 在 24K tokens 时快 3.98×，GPU 内存节省 54%。

  **效率优势的根本机制**：Transformer VLM 每生成一个 token 需要计算与所有历史 token 的 attention，随对话轮数/图像分辨率增加，延迟线性增长。VisualRWKV 通过 WKV 的递归形式 (`wkv_state_t = diag(w_t)·wkv_state_{t-1} + k_t^T·v_t`) 将所有历史信息压缩进固定大小的矩阵 state（per-head 64×64），推理时仅需 O(1) 更新和读取，且内存完全不受序列长度影响。这意味着在生成 24K tokens 的长序列时，VisualRWKV 不仅更快（3.98×），且内存使用保持在初始水平（LLaVA-1.5 此时已增长到初始的约 2.2×），使 RNN-based VLM 特别适合多轮对话、视频理解等长序列多模态场景。
