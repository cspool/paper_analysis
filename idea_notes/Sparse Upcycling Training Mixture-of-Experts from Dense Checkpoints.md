## Sparse Upcycling Training Mixture-of-Experts from Dense Checkpoints

- baseline方法是什么？
  Baseline 是将预训练 dense Transformer checkpoint 直接继续训练（"dense continuation"），即对已有 dense checkpoint 不做架构修改，用相同的超参数（batch size、inverse square root LR schedule、Adafactor optimizer）再训练额外 steps。另一个 baseline 是从头训练 MoE（MoE from scratch），即从随机初始化权重训练同样结构的 MoE 模型。
  全栈执行例子（Baseline: T5 Base Dense Continuation，TPU v4，64 chips）：
  - **算法pipeline层**：从已训练 1M steps 的 T5 Base dense checkpoint 继续训练，保持相同模型结构和超参数。输入 token sequence X ∈ R^{T×d} → Self-Attention → Dense FFN (W_in → GEGLU → W_out) → LayerNorm → output。每步 FLOPs 固定，参数量固定（248M），无 capacity 扩展。
  - **系统框架层**：TPU v4 集群上使用 T5X 框架（https://github.com/google-research/t5x），数据并行 + 模型分片进行分布式训练。dense continuation 仅需持续执行 forward/backward/optimizer step，无额外的路由计算或 expert 通信开销。
  - **编译框架/Kernel调度层**：论文未明确说明。
  - **硬件架构层**：TPU v4，使用 Adafactor optimizer 进行 mixed precision 训练，无 MoE expert 分片和路由相关通信。
  Baseline 的核心缺陷：dense continuation 受限于原始 dense 模型的参数容量，增加 compute 的边际收益递减——训练曲线已经饱和，额外 compute 带来的性能提升有限。MoE from scratch 则需要完全重新训练，浪费了已投入的 dense checkpoint 训练成本。
