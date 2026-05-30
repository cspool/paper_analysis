## Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

- baseline方法是什么？
  Baseline 为两种视角：(1) **未压缩 MoE 模型**：Mixtral-8×7B (47B total/13B activated, 87.7GB memory) 和 DeepSeek-MoE-16B (30.8GB memory)，包含全量 experts、全量 layers 和全量 blocks。(2) **现有压缩方法**：Expert Drop（Lu et al. 2024, Muzio et al. 2024）按重要性评分移除不重要 expert，减少参数量和内存但仍保留 MoE 层内的昂贵计算和 expert 间通信开销，speedup 不足 1%；Pruning（Wanda/SparseGPT）和 Quantization（GPTQ/AWQ）作为独立 Expert Slimming 技术存在但未与结构化 Expert Trimming 集成。Baseline 的核心痛点：(a) Expert Drop 移除 expert 后仍保留 MoE layer 内的 costly computation（expert FFN 的前向计算）和 communication overhead（分布式环境下的 All-to-All 通信），导致尽管参数减少但 inference speedup 微乎其微；(b) Expert Drop 破坏路由模式——部分 expert 被移除后，router 对某些输入可能选中"错误"的剩余 expert，导致性能大幅下降（如 MMLU 23% 下降 at 25% experts dropped）；(c) Expert Slimming 技术（pruning/quantization）仅关注单个 expert 内部压缩，未联合解决跨 expert 的结构冗余；(d) dense pruning 方法（Wanda/SparseGPT）在 MoE 上应用时，不考虑 MoE 的 inductive bias（如 shared expert vs routed expert 的不同冗余特性），对 shared expert 误剪枝导致额外性能损失（+3.6% for shared expert exclusion）。

  **Baseline 全栈执行例子（以 Mixtral-8×7B, Expert Drop 12.5%, 128 sequence × 2048 tokens batch 为例）**：
  - **算法层**: 加载 Mixtral-8×7B checkpoint。Expert Drop: 每层按 G(x) 平均路由分选择 7/8 expert 保留（丢弃 1 expert），更新 router weight G ← G_{i∈T'}。保留的 7 experts 仍做 MoE FFN（x → router Top-2 → active expert FFN → weighted sum）。FLOPs 不变（仍激活 2/7 experts，expert FFN 计算不变），内存减少约 1/8 expert 参数量，speedup < 1% 因 expert 内计算和通信未减少。
  - **系统框架层**: HuggingFace Transformers + AWQ/GPTQ 量化框架 + LM Evaluation Harness。未修改 serving 框架。Batch forward pass on input seq_len=2048, batch_size=1~8。论文未明确说明 serving framework。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 PyTorch FP16 matrix multiply kernels on NVIDIA GPUs。量化模型使用 INT4 GEMM kernels (AWQ/GPTQ 量化后)。论文未明确说明具体 kernel 实现。
  - **硬件架构层**: NVIDIA GPU (RTX 3090 作为部署目标提及)。Mixtral-8×7B FP16 需 87.7GB，量化后 24.4GB (AWQ 4-bit)，可在 24GB consumer GPU 部署。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出统一的 MoE 压缩框架，从两个互补视角系统性地解决 Baseline 缺陷：

  **(1) Layer Drop 解决 Expert Drop 的"保留专家内计算+通信"问题**：不是移除单个 expert，而是直接移除整个 MoE 层（含 Norm 模块）。通过 S^{(NM)} = cos_sim(x', x'+MoE(Norm(x'))) 评估每层的输入输出相似度来选择冗余层。移除整个 MoE 层后：(a) 消除了该层内所有 expert FFN 计算（彻底避免 cost computation within experts）；(b) 消除了该层的 All-to-All 通信（彻底避免 complex communication among experts）；(c) 减少参数量和内存（移除整层所有 expert）。实验：Mixtral-8×7B 用 Layer Drop 移除 8/32 MoE 层后 MMLU 仅降 1%，speedup 显著提升 vs Expert Drop 的 <1%。

  **(2) Block Drop 进一步解决 Layer Drop 中 Attention 计算保留问题**：Layer Drop 保留了 computation-costly attention layers。Block Drop 通过评估 block 级 S^{(NM)} = cos_sim(x^l, y^l) 移除整个 Transformer block（Attention + MoE + Norms）。移除 block 后：(a) 减少了 Attention 的 O(S^2·d) 矩阵乘法和 Softmax 计算；(b) 移除了对应层的 KV-Cache（如 batch=128, seq_len=2048 时节约 5GB KV-Cache）；(c) 移除了 FFN 计算。与 Expert Drop 的"精准但效果有限"相比，Layer/Block Drop 是"粗粒度但高效"的互补策略。实验：Mixtral-8×7B 移除 5/32 blocks 仍保持 >90% 性能，speedup 优于同压缩率的 Layer Drop。

  **(3) 集成 Expert Trimming + Expert Slimming 解决分别优化的"孤岛"问题**：将 Expert Slimming（AWQ 4-bit quantization）与 Expert Trimming（Layer/Block Drop）按"S+T"顺序组合——先对所有 expert 量化，再基于量化后模型计算相似度执行 Layer/Block Drop。量化减少每个 expert 的内存（→24.4GB from 87.7GB），Layer/Block Drop 进一步减少 FLOPs 和通信（→42.9T from 54.4T），两者互补达成 6.05× speedup + 77.1% 内存节省（20GB）。量化"保性能"（98%+ of original accuracy），Layer/Block Drop "增效率"（speedup 和 memory reduction）。

  **(4) 发现 MoE Layers 比 Dense 更冗余**：同深度 Mixtral-8×7B (MoE) vs Mistral-7B (Dense)，相同 Layer/Block Drop 下 MoE 模型性能衰减显著更小（Drop 8 layers: MoE -7.0 vs Dense -24.3 on MMLU）。这一发现验证了 MoE 架构中存在更高程度的结构冗余，Layer/Block Drop 特别适合 MoE 压缩。

  **(5) Post-Finetuning 解决压缩后的性能 gap**：在 Alpaca-GPT4 数据集上对压缩模型 full-finetune 3 epochs，性能 gap 从显著缩小（DeepSeek-MoE-16B Block Drop: 从 -5.5% 恢复到 -0.6%）。

  **(6) Expert Slimming 消融：Shared Expert 不可压缩性发现**：DeepSeek-MoE-16B 使用残差 MoE（2 shared + 64 routed），发现 shared expert 比 routed expert 更不可压缩——pruning 不含 shared expert 相比 pruning shared expert 提升平均精度 3.6%（Wanda）到 1.5%（SparseGPT）。

  **论文方法全栈执行例子（以 Mixtral-8×7B, AWQ + Block Drop B5/32, 128 sequence × 2048 tokens batch decode 为例）**：
  - **算法层**: 
    1. Expert Slimming: AWQ 4-bit 量化所有 32 层 × 8 experts 的 FFN 权重。W_i_quant = AWQ(W_i, 4-bit, group_size=128)。量化后模型总内存 24.4GB。
    2. 用 128 个 C4 样本在量化模型上计算每个 block 的 S^{(NM)}_l = mean(cos_sim(x^l, x^l+Block_l(Norm(x^l))))。
    3. 按 S^{(NM)} 降序排序 blocks，移除 Top-5 highest-similarity blocks（深层更冗余→多 drop 深层 blocks）。移除后 FLOPs 从 54.4T 降至 46.0T。
    4. Router: 保留 layers 的 router 不变（移除层无需 routing）。移除 block: block 的 attention + MoE + 2 Norms 全部移除，对应 KV-Cache 移除。
    5. Inference forward: tokens → embedding → 剩余 27 blocks（每 block: Attention + Norm + MoE FFN with 8 quantized experts, Top-2 routing）→ LM Head → next token。Speedup=5.94×, Memory=21.9GB。
  - **系统框架层**: HuggingFace Transformers 加载量化模型（AutoAWQ）+ PyTorch forward pass 测试 speedup/FLOPs（input seq_len=2048）+ EleutherAI LM Evaluation Harness 评估 zero-shot benchmarks。论文未修改 serving framework。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 论文未明确说明。AWQ 量化后使用 INT4 GEMM kernels；标准 PyTorch FP16 attention kernels。
  - **硬件架构层**: NVIDIA GPU（RTX 3090 24GB 为部署目标的提及）。量化后 Mixtral-8×7B 从 87.7GB → 20.0~24.4GB，满足 24GB consumer GPU 部署条件。Speedup 基于 forward pass on seq_len=2048 测量。
