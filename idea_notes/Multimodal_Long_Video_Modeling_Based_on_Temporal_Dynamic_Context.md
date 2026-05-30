## Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context

- baseline方法是什么？
  - **Baseline 代表**：VideoLLaMA2、Video-LLaVA、LLaMA-VID 等现有视频 MLLM。典型方法：从视频中固定采样少量帧（如 8 或 16 帧），每帧独立通过 CLIP/SigLIP 编码为视觉 token，直接拼接到文本 token 送入 LLM。视觉和音频模态分开编码后简单拼接，缺乏模态间交互。
  - **全栈执行例子（VideoLLaMA2 baseline）**：
    - **算法层**：输入 120s 视频 → 采样 16 帧 → SigLIP encoder 每帧得到 72 visual tokens → BEATs encoder 每帧约 50 audio tokens → 拼接 visual + audio tokens (16 × (72+50) = 1952 tokens) → 与 question text tokens 拼接送入 LLM (Qwen2-7B) → 自回归生成答案。长视频场景下，16 帧均匀采样丢失大量时序细节；简单拼接各模态 token 导致 LLM 难以区分融合多模态信息。
    - **系统框架层**：PyTorch + HuggingFace Transformers。LLM 通过 causal self-attention 处理所有输入 token。无 token 压缩机制，token 数量随帧数线性增长。
    - **编译框架层**：论文未明确说明。使用标准 PyTorch eager mode 推理。
    - **kernel调度层**：标准 Transformer attention kernel（Flash Attention 或 PyTorch native SDPA）。无自定义 kernel。
    - **硬件架构层**：NVIDIA GPU（论文未明确型号）。Baseline 中每帧独立编码、简单拼接，无跨帧/跨模态融合计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **缺陷 1**：固定少量帧采样导致信息丢失 → TDC 以 1fps 密集采样全部帧，并通过语义场景分割保证时序一致性。
  - **缺陷 2**：长视频 token 数量随帧数线性增长 → TDC 将首帧作为静态参考（完整保留），后续帧通过 Q-Former 压缩为 K=16 个 context tokens，使每帧平均 token 数从 194 降至 16。
  - **缺陷 3**：视觉和音频模态独立编码、简单拼接 → TDC 将 visual + audio tokens 一起送入 Q-Former 做 cross-attention，在统一视频上下文中融合多模态信息；同时注入 instruction text (F_s) 使压缩过程自适应于用户问题。
  - **缺陷 4**：超长视频无法整体处理 → LVCoT 将视频等分为 M 段，逐段推理生成中间答案，再汇总为最终答案，利用分段推理链提升长视频理解深度。
  - **全栈执行例子（TDC 7B）**：
    - **算法层**：输入 T 秒视频 (T 可能 > 1000) → 1fps 密集采样 T 帧 → DINOv2 提取每帧 embedding，cosine similarity 找 S-1 个低相似度分割点 → 视频分割为 S≤24 个场景 → 每场景 sliding window 内首帧完整保留 (144 visual + 50 audio tokens) → AvgPool 从首帧 visual tokens 得 K=16 个 query tokens → 后续每帧 visual+audio tokens 送入 Q-Former (BERT initialized) 与 query tokens 做 cross-attention，同时注入 question text (F_s) → 输出压缩后的 context tokens(16/frame) → 场景表示 F_TDC = [static_tokens · <Sep> · context_tokens] → LLM (Qwen2-7B/LLaMA3.2-3B) 自回归生成。超长视频额外触发 LVCoT: 均分 3 段 → 每段独立 TDC 编码+推理 → 汇总段级答案作为 chain-of-thought → 最终全局推理。
    - **系统框架层**：PyTorch + HuggingFace Transformers。修改点：在 video encoding pipeline 中插入 scene segmentation module (DINOv2 similarity)、TDC Q-Former compressor（可训练）、LVCoT multi-pass 推理循环。三阶段训练：Stage 1 vision-language alignment (3.2M), Stage 2 video instruction tuning (2M/540K), Stage 3 audio-video instruction tuning (300K/120K + LoRA)。
    - **编译框架层**：论文未明确说明。使用标准 PyTorch eager 推理。
    - **kernel调度层**：Q-Former cross-attention 使用标准 Transformer attention kernel。无特殊 kernel 优化。场景分割的 cosine similarity 矩阵计算为标准矩阵乘法。
    - **硬件架构层**：论文未明确说明 GPU 型号。推理计算量主要来自 (1) per-frame encoder forward (SigLIP + BEATs), (2) Q-Former cross-attention (每帧 16 queries × ~194 key-value tokens), (3) LLM decoder self-attention。TDC 将每帧平均 token 数从 ~194 压缩至 16，LLM attention 成本降为原来的 ~(16/194)² ≈ 0.7%。LVCoT 每段独立推理引入额外 forward pass 但提升长视频准确性。
