## Mixture of Diverse Size Experts

- baseline方法是什么？
  Baseline 是传统的 **Same-Size Expert MoE**（如 Switch Transformer、GShard、Mixtral 架构风格），即每个 MoE FFN 层内所有 N 个 expert 拥有完全相同的结构（相同的 hidden dimension h），仅通过 top-k gating 选择 k 个 expert 激活。以 300M×8（基于 Llama 2, dim=1536, n_layers=8, h=3840, N=8, k=2）在 NVIDIA A800 集群上的执行路径为例：
  - **算法层（MoE Routing + Expert FFN）**：Gating network 接收 self-attention 输出 x [B, S, 1536]，计算 logits = x·W_g → 加噪声 Softplus(RMSNorm(x·W_n)) → Softmax → KeepTopK(k=2)。每个 token 被路由到 top-2 experts。**每个 expert E_i 结构完全相同**：w1 [1536, 3840] → SiLU → w2 [3840, 1536]，参数量均为 2×1536×3840≈11.8M。8 个 expert 总参数量为 8×11.8M≈94.4M。**缺陷(1)**：所有 expert 能力相同，无法区分处理不同难度 token——容易预测的 token（如常见短语内部词）和困难 token（如跨领域知识推理）被路由到相同能力的 expert，导致"大材小用"或"小材大用"的资源浪费。困难 token 在相同尺寸 expert 中的预测准确度受限。
  - **系统框架层（ZeRO 分布式训练）**：基于 ZeRO 优化器状态分片，8 个 expert 均匀分布在多个 GPU 上。由于所有 expert 尺寸相同，天然负载均衡——每个 GPU 计算量相同。**缺陷(2)**：虽然 baseline 天然负载均衡，但这是通过牺牲 expert 的异构能力换来的（所有 expert 相同尺寸 = 无差异化预测能力）。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch 训练基础设施。
  - **kernel调度层**：论文未明确说明。标准 CUDA kernel 执行矩阵乘法（GEMM for expert FFN），每个 expert 的 GEMM 尺寸完全相同（1536×3840 和 3840×1536），无需特殊 kernel 调度。
  - **硬件架构层**：NVIDIA A800（80GB），NVLink + NVSwitch 节点内互联，ZeRO 模式下跨节点通信通过 InfiniBand（论文未明确说明 inter-node 互联方式）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoDSE（Mixture of Diverse Size Experts）**，核心是在同一 MoE layer 内设置不同 hidden dimension 的 expert，并通过 expert-pair allocation 策略保持 GPU 负载均衡。以 300M×8 模型在 A800 集群上的执行路径为例（expert 尺寸：[4.5×, 0.5×, 4.0×, 1.0×, 3.0×, 2.0×, 2.5×, 2.5×]，对应 h_i [6912, 768, 6144, 1536, 4608, 3072, 3840, 3840]）：
  - **算法层（Diverse Size Expert Routing + Expert Pair）**：
    1. **Diverse Size Experts**：Gating 逻辑不变（同 baseline top-k routing），但 expert 的 FFN hidden dimension 多样化——大专家 h_i > h（如 6912 vs baseline 3840），提供更强预测能力；小专家 h_i < h（如 768），处理简单 token。Experts 按对组织：每对 (i_k^1, i_k^2) 满足 h_{i1} + h_{i2} = 2h = 7680，总参数量与 baseline 一致（每个 pair: 2×1536×7680 = 2×1536×2×3840）。**解决缺陷(1)**：困难 token 可以被路由到大专家获得更强预测能力（Section 4.3 分析显示 CE>2.0 的 180 个高难度 token 中，6215 次选择大专家 vs 3085 次选择小专家，比例为 2:1），容易 token 路由到小专家节省计算。MoDSE 在 700M×8 所有 9 个 benchmark 上均超越 baseline（如 MMLU: 29.9 vs 26.5, SIQA: 60.9 vs 42.9）。
    2. **Auxiliary Load Balance Loss**：沿用 Switch Transformer 的辅助均衡损失 $L_a = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$，f_i 是路由到 expert i 的 token 比例，P_i 是 router 对该 expert 的平均概率。训练后期 token 分布趋于均匀（last epoch: max/min ratio 从早期 >3.0 降至 <3.0，多数在 1.5-3.0 之间）。
  - **系统框架层（Expert-Pair Allocation Strategy）**：每对 expert $(\hat{E}_{i_k^1}, \hat{E}_{i_k^2})$ 放置在同一 GPU 上。由于每对 expert 的参数量之和等于 baseline 两个 expert 的参数量（h_{i1}+h_{i2}=2h），每个 GPU 的总参数量和计算量保持均衡。在 8 expert 4 对×4 GPU 的配置下，每个 GPU 的参数量与 baseline 完全相同。**解决缺陷(2)**：在保持专家异构性的同时实现 GPU 负载均衡，MoDSE 的推理速度与 baseline 几乎相同（如 MMLU: 3min27s vs 3min26s, GSM8K: 20min43s vs 20min26s），无额外推理开销。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch 训练基础设施。
  - **kernel调度层**：论文未明确说明。不同 expert 的 GEMM 尺寸不同（6912 vs 768 宽度），但通过 expert-pair 分配在各 GPU 上对称分布，每个 GPU 处理相同的 expert 尺寸集合，无需特殊调度。
  - **硬件架构层**：同 baseline（NVIDIA A800, NVLink+NVSwitch）。MoDSE 的 expert-pair allocation 确保每个 GPU 的总参数和工作量一致，利用 NVLink 实现 GPU 内多 expert 并行计算，跨节点通过分布式数据并行（ZeRO）同步梯度。论文验证了即使 expert 尺寸多样化，训练和推理速度与 baseline 保持可比（Table 4）。

  核心设计直觉：预训练中 token 预测难度差异巨大——同一短语内 token 极易预测（靠局部上下文即可），跨领域知识推理 token 极难预测（需综合多种知识）。Same-size expert 抹平了这种差异，MoDSE 的 diverse-size expert 让不同能力的 expert "各司其职"：大专家专注高难度推理，小专家高效处理模式化预测，从而实现相同参数预算下更好的 loss 收敛和下游任务表现。
