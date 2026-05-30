## MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

- baseline方法是什么？
  Baseline 是**缺乏系统性 MoE 推理 benchmark 的状态**——即研究者各自使用不同的硬件、框架、模型和配置进行实验验证，导致：(1) MoE 推理性能结论无法横向比较；(2) 关键超参数（FFN dim、#experts、#active experts）对吞吐量的影响缺乏定量研究；(3) 多种优化技术（量化、剪枝、投机解码、Fused MoE、并行策略）在统一硬件和框架下的相对效果未知；(4) LLM 和 VLM 两类 MoE 模型的推理特性差异未被系统对比。以 Mixtral-8x7B 在 4×H100 + vLLM 上的执行路径为例：
  - **算法层**：Standard Top-K routing（每层 8 experts, k=2）。每个 token 经 self-attention → router logits → softmax → select Top-2 → 两个 expert FFN 计算 → weighted sum。Baseline 无任何优化（FP16, 无剪枝, 无 Fused MoE, 无投机解码）。**痛点(1)**：激活 expert 越多（k=2→k=8），每个 token 要经过更多 FFN 计算和参数读取，吞吐量下降 50-80%——但当前缺乏"k 值 vs FFN dim vs #total experts"的 joint scaling 指导。
  - **系统框架层**：vLLM 推理框架。支持 TP/PP/EP 并行策略配置和 PagedAttention KV-cache 管理。**痛点(2)**：vLLM 提供了多种并行策略（TP, PP, EP, Hybrid），但这些策略在 H100 上对 MoE 模型的 relative effectiveness 未知——研究者选 TP 还是 EP 缺乏实验依据；(3) vLLM 内置 Fused MoE kernel 可用但实际收益在不同 batch size/sequence length 下未量化。
  - **编译框架层**：论文未明确说明。使用标准 PyTorch + CUDA kernel。
  - **kernel调度层**：Fused MoE kernel（vLLM 内置）将 routing + FFN 融合，但未与其他优化（FP8、剪枝、投机解码）联合评估。
  - **硬件架构层**：NVIDIA H100 SXM5 80GB + NVLink。**痛点(4)**：H100 的 TP 扩展效率 vs EP 扩展效率缺乏实测数据；H100 vs Cerebras CS-3（wafer-scale）在 MoE 推理上的对比缺失。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE-Inference-Bench**——一个综合性 MoE 推理 benchmark 套件，不提出新算法，而是在统一硬件（H100）、统一框架（vLLM）下系统性地评估多种优化技术及其交互。以 Mixtral-8x7B 在 4×H100 + vLLM 上的全栈执行路径为例：
  - **算法层（系统评估多种优化技术）**：
    1. **FP8 Quantization**：将 MoE FFN 权重和激活从 FP16 量化为 FP8（H100 第四代 Tensor Core 原生支持 FP8），权重保留 FP8 master copy，计算时 dequantize → INT8 matmul on Tensor Core。吞吐量提升 20-30%（batch size=64 时差距最大），且在不同 sequence length 下保持稳定优势。**解决痛点(1)**：量化可在不增加 active experts 的前提下提升吞吐量，部分抵消多 expert activation 的性能代价。
    2. **MoE Pruning**：系统性评估 inter-expert pruning（移除整个 expert）和 intra-expert pruning（缩减 expert 内部 FFN dim）在 12.5%/25%/50% 三种比例下的效果。关键发现：(a) 50% 高比例剪枝反而显著改善吞吐量（因减少的总参数和计算量 > 负载不平衡损失），而 12.5%/25% 低比例剪枝因引入负载不均反而可能降低吞吐量；(b) OLMoE-1B-7B 对 intra-expert 剪枝容忍度高，Qwen1.5-MoE-A2.7B 更敏感。**解决痛点(1)**：提供剪枝比例与模型类型的选择指南。
    3. **Speculative Decoding**：Qwen3-30B-A3B (target) + Qwen3-1.7B (最优 draft) 的投机解码。Draft model 大小需要平衡——太小（0.6B）acceptance rate 低导致验证浪费，太大（8B）draft 开销抵消收益。最优 draft 大小（1.7B）在所有 input lengths 下吞吐量最高。**解决痛点(1)**：提供 draft model size 选择的实验指导。
  - **系统框架层（vLLM 并行策略 + Fused MoE）**：
    1. **并行策略对比**：TP-only 在 1→4 GPU 扩展时吞吐量 >2×（NVLink 高带宽掩盖 all-reduce 开销），TP+EP 扩展效率次之，PP+EP 几乎无加速，PP-only 持平。**解决痛点(2)**：明确建议 MoE 推理优先使用 TP，PP 和 EP 在单节点内收益有限。
    2. **Fused MoE 全面评估**：Fused MoE 在 batch size 增大时优势更明显（大 batch 时 15-20% throughput gain），因为更大 batch 下未经融合的 kernel launch 和中间显存传输开销更高。在不同 sequence length 下保持 12-18% 优势。**解决痛点(3)**：量化 Fused MoE 在不同 workload 下的收益。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Fused MoE kernel 将 token-to-expert dispatch → grouped GEMM → SiLU activation → weighted sum 融合为单 kernel。效果：减少中间 tensor 的 HBM 往返（每层节省约 3-4 次 HBM read/write），与 batch size 正相关。
  - **硬件架构层（H100 vs Cerebras CS-3 对比）**：
    - H100：在 context length >1024 时延迟急剧上升（受限于 HBM 带宽和 KV-cache 增长），吞吐量在长 context 下显著退化。
    - CS-3：WSE-3 的多数量级更高内存带宽 + 减少 inter-device 通信使其延迟增长平缓，在长 context 推理中优势明显。
    - **解决痛点(4)**：提供两类硬件的 quantitative 对比，指出 H100 适合短 context 高 batch 场景，CS-3 适合长 context 低延迟场景。
  
  **MoE 超参数 Scaling 的核心发现（指导部署设计）**：
  - FFN dim scaling：增大 FFN dim（1792→14336）导致吞吐量平均下降 50%，且大 FFN dim + 多 active experts 时降幅最大（~60%），原因是内存带宽饱和压倒计算并行优势。
  - Expert count scaling：小 FFN dim（1792/3584）时增加 expert 数量（8→64）可保持或略微提升吞吐量（5-15%），大 FFN dim 时增加 expert 数量反而受内存带宽限制无收益甚至 OOM。
  - Active expert scaling：active experts 从 1→8 吞吐量下降 50-80%，且下降幅度在 big FFN dim 时更大（60-80% vs 20-30%）。
  - **联合指导**：小 FFN（1792-3584）可灵活使用更多 active experts；大 FFN（7168-14336）必须使用保守的 activation 策略（1-2 active experts）以避免 OOM。

  **LLM vs VLM 的关键差异**：
  - VLM 的 latency 差距远大于 LLM（ITL 240% vs 100% gap，end-to-end 260% vs 120% gap），主要原因是视觉编码器的额外计算负载和多模态处理开销。
