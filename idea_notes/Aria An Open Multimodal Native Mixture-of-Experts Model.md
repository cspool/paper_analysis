## Aria An Open Multimodal Native Mixture-of-Experts Model

- baseline方法是什么？
  Baseline 主要对比两类模型：

  1. **Dense Multimodal Models**：Llama3.2-11B（11B dense，视觉编码器 + 语言解码器的标准 VLM 架构），同规模下全部参数参与每次推理，无法利用稀疏激活降低推理成本。
  
  2. **Modality-Specialized MoE Models**：Pixtral-12B（基于 Mixtral 的 multimodal MoE）、MoE-LLaVA（从 dense 模型 upcycling 的 multimodal MoE）、MoMa（modality-aware expert 架构）。这些 MoE 要么从 dense 模型 "upcycle" 而来（非原生的多模态稀疏训练），要么为不同模态设计专属 expert（增加架构复杂度和模块化成本）。
  
  **Baseline 全栈执行例子（以 Llama3.2-11B 推理一个图文混合请求为例）**：

  - **算法层**：输入 "Describe this image" + image → Visual Encoder (ViT) 将图像编码为 N 个 visual tokens → 与 text tokens 拼接为 [T_vis, T_txt] → 进入 11B dense Transformer decoder → 每层 self-attention (所有 token 相互 attend, O((N+M)²·d)) → 每层 FFN (d_model → d_ff → d_model, 全部 11B 参数参与计算) → auto-regressive 生成每个 output token。dense 模型中每个 token 都激活全部 11B 参数，无 expert 路由。

  - **系统框架层**：HuggingFace Transformers 标准推理。Pixtral-12B 使用 mistral-inference 框架。无特殊的多模态 serving 优化，visual encoder 独立前向计算后送入 LLM decoder。

  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution，使用 Flash-Attention 加速 attention 计算）。

  - **Kernel级**：标准 GEMM kernel 执行 attention projection 和 FFN。Pixtral-12B 的 MoE 层使用 token-choice routing（Top-2 experts），每个 token 的 expert 选择独立，batch 内不同 token 可能激活不同 expert 组合，需要 group_gemm 实现。

  - **硬件架构层**：NVIDIA A100/H100 GPU。dense 11B 模型 bf16 约 22GB 显存（不含 KV cache），对 consumer GPU（<24GB）压力大。

  **Baseline 的核心缺陷**：
  1. **Modality Performance Gap**：现有 open multimodal 模型在跨模态能力上不均衡——Pixtral-12B 在长视频理解（LongVideoBench 47.4 vs ARIA 65.3）严重落后，且缺乏对视觉/语言/代码的统一 high-quality 能力。dense 模型在扩展到 multimodal 后往往损害纯语言能力（knowledge forgetting）。
  2. **非原生多模态 MoE**：先前 multimodal MoE（MoE-LLaVA, MoMa）依赖从 dense checkpoint upcycling 初始化和/或设计 modality-specific expert，限制了 expert specialization 的自然涌现。
  3. **Context Window 受限**：多数 open multimodal 模型的 context window 有限（通常 <8K），无法处理长视频或多页文档等 long-context multimodal 输入。
  4. **训练效率 vs 模型能力的平衡**：dense 模型全部参数参与每次前向，推理 FLOPs 随参数线性增长；而 MoE 可用更少激活参数达到同等能力，但训练 non-trivial 的 load balancing 和 expert specialization。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ARIA 提出 **multimodal native fine-grained MoE** 配合 4-stage 渐进式预训练 pipeline：

  **方法核心**：
  1. **Fine-grained MoE with Modality-Generic Experts**：66 experts/layer (2 shared + 64 routed)，每个 expert FFN hidden dim 仅 1664（远小于 dense 模型的标准 FFN），每 token 激活 2 shared + 6 routed experts（3.5B activated / 24.9B total）。关键点：所有 expert 是 modality-generic 的（不预设某 expert 只处理视觉/文本），expert specialization 在训练中自然涌现（Section 4.2 可视化证明了 visual-specialized experts 的存在）。
  
  2. **Multimodal Native Pre-training from Scratch**：不从 dense checkpoint upcycle，而是从随机初始化开始，language 和 multimodal 数据混合训练。MoE decoder 同时在 text 和 visual tokens 上做 next-token prediction，无需 modality-specific 专家架构。
  
  3. **4-Stage Training Pipeline**：渐进式赋予模型不同能力——Stage 1 (语言基础) → Stage 2 (多模态理解) → Stage 3 (长上下文) → Stage 4 (指令遵循)。每阶段维护前阶段能力的同时增加新能力。
  
  4. **Lightweight Visual Encoder with Cross-Attention Projection**：438M 参数的 ViT + cross-attention projection module，将变长/变分辨率图像压缩为固定数量 visual tokens（128/256），降低 decoder 的计算负担。
  
  5. **Group-Level Load Balancing**：用 8-expert group 级别的 load balancing 替代 per-expert balancing，避免 fine-grained MoE（64 routed experts）场景下过强的 load balancing 约束压制 expert specialization。

  **论文方法全栈执行例子（以 ARIA 处理 "Describe this video" 请求为例）**：

  - **算法层**：输入 64K-length video (N frames) + text prompt → 每帧经 ViT-SO400M 编码为 patches → projection module (cross-attn with 128 learnable queries) 将每帧压缩为 128 visual tokens → 共 M 个 visual tokens 与 text tokens 拼接 → MoE decoder (28 layers, 每层 66 experts):
    Layer l: Self-Attention → RMSNorm → Router(W_router · x) → Top-6 expert selection → 2 shared experts (always active) + 6 routed experts 分别计算 SwiGLU_FFN(x) → 加权求和。训练时用 group-level load balancing loss (8 experts/group)，推理时 expert 激活稀疏化——仅 3.5B/25.3B 参数参与计算，比同能力 dense 模型 (如 InternVL2-40B) 推理效率高得多。

  - **系统框架层**：训练基于修改版 Megatron-LM（expert parallelism + ZeRO-1 data parallelism，无 tensor parallelism 以减少 all-reduce 通信），推理支持 HuggingFace Transformers（AutoModelForCausalLM + AutoProcessor）和 vLLM（RadixAttention prefix caching 加速）。单张 A100 80GB 即可 bf16 推理。

  - **编译框架层**：论文未明确说明。依赖 Flash-Attention 和 grouped_gemm 等性能库加速 MoE expert 调度和 attention 计算。

  - **Kernel级**：expert parallelism 下，66 experts 分布在多个 GPU 上，每个 GPU 持有 expert 子集 + 一份完整的 attention 参数和 shared expert 参数。Router 输出通过 all-to-all 通信将 token 路由到目标 expert 所在 GPU，expert FFN 计算完成后 all-to-all 回传。Group-level load balancing 减少了路由坍缩风险。ZeRO-1 将 optimizer states 分片到各 GPU。

  - **硬件架构层**：多节点 NVIDIA A100/H100 GPU 集群。expert parallelism 将 expert 参数分布在多个 GPU 上，batch 内 token 通过 all-to-all 路由到对应 expert。训练中视觉 encoder 参数复制到所有 GPU（数据并行），MoE decoder 的 attention 参数也使用 ZeRO-1 数据并行（仅分片 optimizer states），expert FFN 参数使用 expert parallelism 分片。

  **对应解决 baseline 缺陷**：
  - 缺陷1（跨模态能力不均衡）→ 4-stage 渐进训练 + 语言/多模态/代码混合数据，确保各能力同步增长
  - 缺陷2（非原生 MoE）→ 从 scratch multimodal native 训练 + modality-generic experts，让 expert specialization 自然涌现
  - 缺陷3（context window 受限）→ Stage 3 专门扩展 context 到 64K + RoPE theta 从 100K 提升到 5M
  - 缺陷4（训练/推理效率）→ fine-grained MoE (3.5B/25.3B) + expert parallelism + ZeRO-1，无需 tensor parallelism

- baseline方法是什么？
  Baseline 为两类：

  1. **Dense 模型**：与 MoE 模型 FLOP-aligned (FA) 或 Parameter-aligned (PA) 的标准 Llama3 架构 dense Transformer。FA Dense 与 MoE 有相同的 active parameters（即相同的每 token 推理 FLOPs），PA Dense 与 MoE 有相同的 total parameters（即相同的内存占用）。

  2. **标准稀疏 MoE 模型**：Llama3 架构 + MoE 层（8 total experts, 2 active per token, Token Choice routing），使用标准 load balancing loss (Fedus et al., 2022)。推理时需要 expert offloading 以在端侧设备上运行——未使用的 experts 从 GPU 卸载到 CPU，每个 token 生成可能触发 expert 切换并引发 offload 操作。

  **Baseline 全栈执行例子（以 Phone-sized MoE, 1.37B active / 3.75B total, 推理一个 token 为例）**：

  - **算法层**：输入 token x_t → Router: softmax(W_router · x_t) → Top-2 选择 expert (如 expert 3, 5) → Expert 3 FFN 和 Expert 5 FFN 分别计算 → 加权求和 → 输出 y_t。每个 token 独立路由，连续 token 间 expert 选择无关联约束。Offloading 推理时：若当前 token 选中的 expert 集合 S_t ≠ S_{t-1}（前一个 token 的选中集合），需将不再需要的 expert 从 GPU 卸载，将新需要的 expert 从 CPU 加载到 GPU。

  - **系统框架层**：HuggingFace Transformers 标准推理（无 Serving 框架修改）。Expert offloading 逻辑：维护 GPU 上的活跃 expert 集合（不超过 2 active experts），每个 token 生成后检查是否需要 offload/load。

  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution）。

  - **Kernel/运行时调度层**：标准 cuBLAS GEMM 执行 expert FFN（SwiGLU: gate_proj → SiLU → up_proj → × → down_proj）。Offload 时 CPU↔GPU 数据传输串行化在 token 生成之间，导致 4-20× 推理延迟增加。

  - **硬件架构层**：CPU + GPU 服务器环境。端侧设备 (Phone) 假设 <6GB RAM。GPU 仅保留 active experts 参数（≈ FA Dense），其余 experts 驻留 CPU memory。

  **Baseline 的核心缺陷**：
  1. **参数量膨胀（Memory）**：MoE 的 total parameters 远超 active parameters（Phone: 3.75B vs 1.37B, 2.7×），超出端侧设备内存限制，必须依赖 expert offloading。
  2. **Offloading 导致的延迟（Latency）**：标准 MoE 在 offloading 场景下 Expert Replacement Ratio 高达 43.82%，意味着几乎每 2-3 个 token 就需要一次 expert 加载/卸载操作。每次 offload 引入明显的 PCIe/内存带宽延迟，导致 4-20× 的推理减速。
  3. **Dense vs MoE 的不公平比较**：以往研究在比较 Dense 和 MoE 时存在混杂因素（不同训练数据、不同训练配方、不同架构），无法明确归因 MoE 组件本身的贡献。
  4. **Expert 参数冗余**：8 个 expert 中每个都有完整的 FFN 权重矩阵（d_model × d_ff），但每个 expert 实际只需"专门化"处理约 1/8 的 token，全秩矩阵可能存在参数低效。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：CoSMoEs 通过两个正交的算法创新解决端侧 MoE 的三个维度挑战（Quality, Memory, Latency）：

  1. **Weight-Decomposed (WD) Experts**：将 expert FFN 的权重矩阵替换为低秩分解（类似 LoRA 但用于预训练），减少总参数量同时通过鼓励"expert specialization"提升模型质量。
  2. **Block-wise Expert Selection (BlES) Loss**：在训练阶段引入 sequence-level 的辅助损失，鼓励连续 token 选择相同的 expert 集合，从而在推理时减少 expert offloading 次数，降低延迟。

  **Defect → Design 映射**：

  | Baseline 缺陷 | CoSMoEs 设计选择 | 解决机制 |
  |---|---|---|
  | 参数量膨胀 → memory 超出端侧限制 | WD Experts: 低秩分解 M → L×R (r=n/2) | 减少每个 expert 的参数量，在参数对齐比较下可堆叠更多层/注意力头（WD MoE: 26L/20H vs MoE: 24L/18H） |
  | Offloading 导致延迟 → 43.82% expert replacement | BlES Loss: H_norm × L_norm 惩罚连续 token 的 expert 切换 | 减少 6× expert replacement (43.82% → 6.55%)，1.54× 生成速度提升 |
  | Dense vs MoE 不公平比较 → 无法归因 MoE 贡献 | 严格控制混杂因素：相同训练数据(FW-edu)、相同训练配方、架构对齐 | MoE 比 FA Dense 平均提升 +2.35%，可明确归因为 MoE 架构贡献 |
  | Expert 参数冗余 → full-rank 矩阵低效 | WD: 利用"特殊化"直觉——每个 expert 只需处理 1/E 的 token | WD MoE 比标准 MoE 额外 +1.1%，且总参数更少 (3.65B vs 3.75B) |
  | Load balancing 跨层可被 exploit | Sequence-level load balancing（每层独立计算） | 防止模型通过跨层"分工"欺骗 BlES loss 和 load balancing loss |

  **CoSMoEs 全栈执行例子（以 Phone-sized WD MoE + BlES, 推理多个 token 为例）**：

  - **算法层**：
    1. Token t=1: x_1 → Router: Top-2 选择 expert 3, 5
    2. WD Expert FFN: expert 3 的 gate_proj 用 L_gate_3 [d_ff, r] × R_gate_3 [r, d_model] 替代直接矩阵乘法 → 同理 up_proj 和 down_proj → SwiGLU 前向
    3. Token t=2: x_2 → Router: Top-2 仍是 expert 3, 5（BlES 训练的效果——连续 token 倾向相同 expert）
    4. 无需 offloading（S_t = S_{t-1}）→ 无 GPU↔CPU 数据传输
    5. Token t=3: 仍需 expert 3, 5 → 继续无 offloading
    6. 平均每 15 tokens 才触发一次 expert 切换（vs baseline 每 2-3 tokens）
    7. Expert replacement ratio: 6.55%（hard selection 统计），生成速度: 23.10 tok/s

  - **系统框架层**：HuggingFace Transformers + gpt-fast 推理。Offloading 逻辑与 baseline 相同，但因 BlES 损失训练后 expert 切换频率大幅降低，offload 操作的实际发生频率降低 6.7×。

  - **编译框架层**：论文未明确说明。

  - **Kernel/运行时调度层**：WD Expert FFN 的计算路径：x → x@R_gate^T → @L_gate^T（两次小矩阵乘法替代一次大矩阵乘法）→ SiLU → 同理 up_proj → × → @R_down^T → @L_down^T。低秩分解增加了矩阵乘法次数（3→6 次），但总 FLOPs 减少（r ≪ n,m）。在 batch=1 的端侧推理场景下，小矩阵乘法更 cache-friendly。

  - **硬件架构层**：GPU 显存仅保留 active experts（2 个 WD experts ≈ FA Dense 大小）。CPU memory 保留全部 8 个 WD experts。由于 WD 减少了每个 expert 的参数量，offloading 时的数据传输量也相应减少。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (标准 MoE + Offloading):
  Pre-train MoE on FW-edu → Router 自由选择 → 推理: 每个 token 
  → Router → 可能切换 expert → CPU↔GPU offload (每 2-3 token 一次)
  → 高延迟 (15.02 tok/s, ExRep=43.82%)
  
  CoSMoEs (WD MoE + BlES + Offloading):
  Pre-train WD MoE on FW-edu → 同时施加 BlES loss (H_norm × L_norm) 
  + Sequence-level load balancing → 推理: 每个 token → Router → 
  倾向保持 expert 选择稳定 → CPU↔GPU offload (每 15 token 一次)
  → 低延迟 (23.10 tok/s, ExRep=6.55%)
  ```

  **关键创新总结**：CoSMoEs 的核心洞察是将端侧部署的三个约束（Quality, Memory, Latency）分别通过算法手段解决——WD 提升 Quality 并减少 Memory，BlES 降低 Offloading Latency。两个创新正交：WD 专注于"每个 expert 内部如何更高效"，BlES 专注于"expert 之间如何协作以降低切换频率"。与 inference-time 优化方法（如 MoE-Infinity, EdgeMoE）相比，CoSMoEs 的训练时优化是正交的——可直接叠加使用。
