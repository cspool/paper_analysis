## Trainable_Dynamic_Mask_Sparse_Attention

- baseline方法是什么？
  Baseline 是标准 **Multi-Head Attention (MHA)**，具有 O(n²d_h) 计算复杂度和 O(n²) 内存复杂度，所有 query-key 对参与完整 scaled dot-product attention。其他对比 baseline 包括：**SWA (Sliding Window Attention)**——固定局部窗口 w，O(nwd_h) 计算但无法捕获长程依赖；**MLA (Multi-Head Latent Attention)**——低秩分解压缩 KV，减少内存但丢失细粒度信息，且无法动态调整压缩策略；**NSA (Native Sparse Attention)**——硬件对齐的块稀疏但静态模式无法适应输入内容变化；**H2O/InfLLM/Quest/DAM**——内容感知的 KV cache 选择方法，但依赖启发式离散操作，不可微且与 FlashAttention 的连续内存访问不兼容。

  全栈执行例子（MHA dense attention 在 A100 GPU 上处理长序列）：
  **算法pipeline**：标准 Transformer self-attention，QKV 线性投影后执行完整 scaled dot-product attention。seq_len=n 时计算 QK^T (n×n 矩阵)，softmax 后乘以 V，复杂度 O(n²d_h)，内存 O(n²)。
  **Serving调度**：论文未明确说明（聚焦于 attention 机制层面，非 serving 框架修改）。
  **编译框架**：论文未明确说明。
  **kernel调度**：FlashAttention-2/3 CUDA kernel，完整 dense attention 前向传播与反向传播，所有 K/V tiles 均加载到 SRAM 参与矩阵乘。
  **硬件架构/芯片设计**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DMA 通过三个核心创新解决 baseline 缺陷：
  **(1) Content-Aware Dynamic Mask 解决静态稀疏的灵活性不足**：从 value 向量表示 δ=exp(τ(v·Δ)×A) 生成每 head 独立的动态 mask，top-w 选择保留最重要位置。相比 SWA 的固定局部窗口只能捕捉近邻依赖，DMA 的动态 mask 能自适应跳过无关 token 并直接关注远距离语义相关 token。相比 MLA 的全局低秩分解丢失细粒度信息，DMA 保留完整 KV cache 且稀疏化发生在 attention weight 计算阶段而非压缩阶段。
  **(2) Position-Aware Sparse Weights 解决内容感知方法的非可微和硬件低效**：mask 为 −∞ 的位置 attention weight≈0，kernel 在 block 级别判断——若整个 K block 的 mask 全零则直接跳过加载和 M×M 操作。相比 H2O/Quest 等的 token 级离散选择破坏内存访问连续性（与 FlashAttention 不兼容），DMA 的 block 级跳过多路复用 FlashAttention tiling，硬件友好且完全可微。
  **(3) Fully Differentiable End-to-End Training 解决训练-推理 gap**：DMA 的 mask 生成（线性变换+softplus+exp+top-w）和 sparse weight 计算全程可微，梯度在选中位置与 full attention 完全一致。训练和推理使用相同稀疏策略，消除 post-hoc pruning 导致的性能退化。复杂度从 O(n²) 降为 O(n·w)，可同时用于高效训练和推理。

  全栈执行例子（DMA 在 A100 GPU 上处理长序列）：
  **算法pipeline**：QKV 投影后，value 向量经 Δ 投影 + softplus + A 门控 + exp 得到 per-head 重要性分数 δ，top-w 选择后生成动态 mask m_t（非选中位置 −∞）。attention 计算 o_t = softmax(q_t K^T/√d_h ∘ m_t) V，仅有效 w 个 key-value 位置参与实际计算，复杂度 O(nwd_h)。
  **Serving调度**：论文未明确说明（DMA 训练和推理共用同一架构，但未涉及 serving 框架层面的调度修改）。
  **编译框架**：论文未明确说明。
  **kernel调度**：Flash DMA CUDA kernel，outer loop 中先加载 mask block 调用 Judge() 判断，active=0 则跳过整个 K/V block；active≠0 时用 FlashAttention online softmax 递推计算 S_ij=Q_i K_j^T/√d_h + M_j。backward 中 dM=dS，无需额外存储 mask 梯度，与 forward 共享 skip logic。block 级跳过多路复用保证硬件效率——Forward 在 32K seq 提速 21.5×，Decode 在 128K key 提速 92.7×。
  **硬件架构/芯片设计**：论文未明确说明。
