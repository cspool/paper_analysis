## Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism (FlexMem)

- baseline方法是什么？
  Baseline 包含两类方法：
  **(a) MLLM 原生 uniform sparse sampling**：LLaVA-Video 7B 均匀采样 64 帧（13k tokens）或 LLaVA-OneVision 7B 均匀采样 32 帧（7k tokens），将所有帧一次性 concat 送入 LLM 做 full self-attention 并自回归生成答案。核心缺陷：(1) 输入帧数受限于 LLM 的 sequence length 上限（超过 200k tokens 即无法处理），均匀采样 64 帧相当于丢弃了 88%+ 的视频信息；(2) 对所有帧等权处理，缺乏对关键片段的聚焦能力；(3) 缺乏跨 clip 的信息传递和历史记忆，无法理解跨越多个采样的长期依赖。
  **(b) 现有高效长视频理解方法**：VideoRAG（如 AKS）通过相似度检索关键帧再输入 MLLM，但缺乏时序连续性理解，对需要全局/整体理解的任务表现差；视觉压缩方法（如 AdaRETAKE）逐 clip 压缩 KV cache 但最终仍需输入所有 compressed features，上下文长度随视频时长线性增长，存在计算瓶颈。

  Baseline（LLaVA-Video 7B, 64 frames uniform sampling）全栈执行例子：
  - 模型推理算法层：长视频 → 均匀采样 64 帧 → SigLIP Vision Encoder 逐帧编码为 ~182 tokens/frame → spatial pooling → 64×182≈11.6k visual tokens → Projector → 与 text prompt tokens 拼接 → Qwen2-7B 28 层 causal self-attention（对所有 11.6k tokens 做 full attention）→ 自回归解码生成答案。对所有视频等权均匀采样，无法区分关键帧和冗余帧。2h 视频仅用 64 帧相当于仅看 0.5fps，严重欠采样。
  - 系统框架层：PyTorch + HuggingFace Transformers，标准 Video-MLLM 推理 pipeline。无专用 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention（causal mask），无自定义 kernel。
  - 硬件架构层：单张 NVIDIA RTX 3090 GPU (24GB)。64 帧 uniform sampling 下可正常运行，但帧数扩展至 128+ 帧时视觉 token 数超 23k，24GB 显存紧张。

  核心缺陷总结：(1) **输入上限**：受 sequence length 和显存限制，无法处理 100+ 帧，大量视频信息被丢弃；(2) **缺乏记忆机制**：无跨 clip 的信息传递，前 30 分钟视频的信息在后 30 分钟完全丢失；(3) **缺乏聚焦能力**：对所有帧等权处理，无法针对问题聚焦关键片段；(4) **RAG 和压缩方法各有局限**：RAG 丢失时序连续性，压缩方法仍受限于线性增长的上下文。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FlexMem 通过**视觉记忆机制（Visual Memory Mechanism）**将长视频理解从"一次性全部输入"改为"迭代观看-形成记忆-召回相关片段"，解决了 baseline 的四大缺陷：

  **(a) 缺陷1：输入上限（sequence length + 显存限制）** → 迭代式记忆编码 + 双路径 KV Cache 压缩
  Baseline 必须一次性输入所有帧，64 帧即接近上限。FlexMem 将视频分片为 N 个 clips（每 clip 8 帧），每次仅处理当前 clip 的 KV cache + 前序 context memory，通过 Dual-Pathway Compression（DPC）将每 clip 压缩为极少的 context 和 local memory tokens。最终解码时仅召回最相关的 na 个 clip 的 memory，而非全量 compressed tokens。LLaVA-Video 解码时仅用 13k tokens（vs AdaRETAKE 的 40k）。理论上可处理无限长视频，实验已验证 1024 帧（16× baseline）。

  **(b) 缺陷2：缺乏跨 clip 信息传递和长期记忆** → Context Memory 链式传递 + Memory Bank
  Baseline 前 30 分钟和后 30 分钟视频信息完全独立。FlexMem 通过两种设计实现跨 clip 信息流：(i) Context Memory C 的链式传递——每步编码时 MLLM 接收前 ns 个 context memory {C_{i-ns}, ..., C_{i-1}}，将历史视频信息持续传递；(ii) Visual Memory Bank M_bank——所有 local memory Mi 被持久存储，并在需要时通过 memory recall 召回长期记忆 `<Ml>`。消融实验（Table 5 Block 2）证实 context + local 组合显著优于单独使用任一种。

  **(c) 缺陷3：缺乏对问题的聚焦能力** → Memory Recall（记忆召回）
  Baseline 对所有帧等权处理。FlexMem 在观看完全部视频后，通过 memory recall 从 M_bank 中召回最相关的记忆片段：(i) Encoding-based Reading 利用 MLLM 在 encoding 时的 cross-modal attention 计算各 clip 与问题 Tq 的 relevance score g_i，Top-K 选择最相关片段；(ii) MemIndex 通过线性回归学习 encoding-based reading 的 relevance 分布，用选定的 cache 层（K=3）和压缩视觉索引（k=5 tokens）做快速点积匹配，完全独立于 memory encoding，适合多问题和 streaming 场景。消融（Table 5 Block 3）验证 memory reading 远优于 indiscriminate loading of all memory。

  **(d) 缺陷4：RAG 和压缩方法各有局限** → 结合两者优势
  FlexMem 同时具备：(i) 压缩方法的全面理解——通过 context memory 链式传递保持时序连续性；(ii) RAG 方法的精确定位——通过 memory recall 从完整 M_bank 中精确召回相关片段。在单 3090 上，FlexMem 在五个 benchmark 上全面超越 AKS（RAG 代表）和 AdaRETAKE（压缩代表），且在 24GB 受限下仅损失 0.5% 性能（vs AKS 和 AdaRETAKE 的显著退化）。

  对比 baseline 的全栈执行例子（FlexMem + LLaVA-Video 7B, 512 frames, 单 RTX 3090）：
  - 模型推理算法层：
    (1) 视频 V → 均匀分 N=64 clips（每 clip 8 帧）→ 共 512 帧
    (2) First encoding: MLLM(V1, Tq) → MLLM 逐层计算 attention → DPC:
        Context path: s_j^l = Σ_{k∈C} a_{jk}^l + Σ_{h∈Vi} a_{hj}^l → top α_c 选 c_1^l
        Local path: ŝ_j^l = Σ_{k∈Vi} a_{kj}^l → top α_s 选 m_1^l
        → C1 = {c_1^1..c_1^L}(用于下一轮), M1 = {m_1^1..m_1^L}(→ M_bank)
    (3) Iterative (i=2..64): MLLM(C_{i-ns},...,C_{i-1}, Vi) → DPC → Ci, Mi → M_bank
    (4) Recall: 从 M_bank 计算 g_i → top na 连续 memory → MLLM(M_i..M_{i+na-1}, Tq) → Y
    关键差异：baseline 一次性输入 64 帧 full tokens 解码；FlexMem 迭代压缩 512 帧后仅用 13k tokens 解码，帧覆盖量 8× baseline。
  - 系统框架层：PyTorch + HuggingFace Transformers。Training-free 即插即用，无 fine-tuning，无框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 FlashAttention。DPC 的 attention score 计算完全在 MLLM 已有的 forward pass 中完成，无额外 kernel。
  - 硬件架构层：单张 NVIDIA RTX 3090 GPU (24GB)。FlexMem 在 24GB 下可处理 1024 帧（baseline 仅 64 帧），且性能仅损失 0.5%。
