## An_Empirical_Study_of_Mamba-based_Language_Models

- baseline方法是什么？
  Baseline是标准GPT3风格的8B参数Transformer模型（32层，hidden dim 4096，32 attention heads，128 KV-channels，4x MLP expansion，SwiGLU activation，LayerNorm，RoPE位置编码，untied embeddings，无bias，无Dropout）。训练使用1.1T/3.5T tokens数据（70% English + 15% non-English + 15% code）、BF16精度、Adam优化器（β1=0.9, β2=0.95, weight decay=0.1）、cosine LR schedule。实现基于NVIDIA Megatron-LM框架，支持tensor/sequence/pipeline parallelism在H100 GPU集群上训练。
  
  Baseline全栈执行例子（推理时生成一个token）：
  - 算法pipeline：输入token → token embedding lookup → 32层Transformer block（每层: RMSNorm → Multi-Head Attention(QKV投影, RoPE位置编码, softmax(QK^T/√d), attention over V) → residual → RMSNorm → SwiGLU MLP → residual）→ LM head projection → logits → softmax → 采样输出token。每生成一个token，attention需要计算与所有历史token的QK内积（O(n²)计算），且KV cache随序列长度线性增长。
  - 系统框架：Megatron-LM的tensor parallelism（每层1次all-reduce）将模型分片到多GPU，data parallelism将batch分布到多节点。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用Megatron-LM内置的cuBLAS/NCCL kernel）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是在8B参数规模下探索三类替代Transformer的架构：纯Mamba（56层，state dim 128，GELU，RMSNorm，无位置编码）、纯Mamba-2（56层，head dim 64，8 groups，expansion factor 2，conv window 4）以及Mamba-2-Hybrid（56层中24 Mamba-2 + 4 GQA Self-Attention + 28 MLP均匀分布，无RoPE）。核心创新在于通过大规模受控实验回答"SSM能否在>3B参数、>1T tokens规模匹敌Transformer"，并发现少量self-attention层（~7%）即可弥补SSM在in-context learning和copying任务上的短板，形成Hybrid设计。
  
  Baseline（Transformer）缺陷：
  1. 自注意力O(n²)计算复杂度和O(n) KV cache内存需求 → 长序列训练推理效率低
  2. 对极长上下文（如Phonebook）的KV cache存储压力大

  论文方法全栈执行例子（Mamba-2-Hybrid推理时生成一个token）：
  - 算法pipeline：输入token → embedding → 56层hybrid block（Mamba-2层: RMSNorm → input projection(expand 2x) → causal conv1d(window=4) → SiLU → selective SSM scan(O(1) per token via recurrent state) → SiLU gating → output projection → residual; Self-Attention层: RMSNorm → GQA(32Q/8KV, 无RoPE) → output projection → residual; MLP层: RMSNorm → GELU(4x expansion) → output projection → residual）→ LM head → logits。Mamba-2层仅需O(1)计算量和常量state memory（128维内部状态）生成每个token，无需KV cache。Self-attention层仅4/56=7.1%，其KV cache仅需存储4层的key-value（vs Transformer的32层），对长序列大幅减少内存。首层为Mamba层，天然学习位置信息，无需显式位置编码，因此模型可在训练序列长度之外泛化（Phonebook上128K模型在>150K tokens仍100%准确）。
  - 系统框架：Megatron-LM中Mamba-2每层仅需1次all-reduce（vs Mamba的2次），与Transformer的all-reduce量持平，MFU达29.9%（接近Transformer的30.7%）。推理时Hybrid模型在长上下文下生成速度预计达Transformer的8x（batch size 32）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。Mamba-2的selective scan使用硬件感知算法实现高效并行（比Mamba scan快8x）。
  - 硬件架构：NVIDIA H100 GPU，论文未涉及RTL/模拟器层面。

  关键设计选择映射到缺陷：
  - 缺陷1（O(n²)计算/O(n)内存）→ Mamba-2层的SSM scan提供O(1) per-token生成，消除attention的二次复杂度
  - 缺陷2（长序列KV cache压力）→ 仅4/56层需要KV cache（GQA with 8 KV groups），其余层用常量大小的SSM state
  - 纯SSM不足（in-context learning/copying弱）→ 混合7.1% self-attention层恢复信息路由和上下文复制能力，MMLU 5-shot提升到53.60（超Transformer的50.07）
  - 训练效率不足（原Mamba 3x慢于Mamba-2）→ 选用Mamba-2而非Mamba作为SSM骨干，将大state dim的scan开销降低8x
  - 位置编码导致长上下文泛化受限 → 取消RoPE，依赖首层Mamba学习位置编码，使128K模型可泛化到>150K tokens
