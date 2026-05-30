## LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

- baseline方法是什么？
  - **固定 top-k MoE 推理 + Expert Pruning**：现有 MoE 模型在推理时所有层使用统一的 top-k（如 Mixtral 全部层 top-k=2，Qwen 全部层 top-k=4），无论每层的实际计算需求如何。Post-training 优化方法主要通过 expert pruning 减少模型参数：Inter-Expert Pruning (NAEE) 删除整层中不重要的 expert；Intra-Expert Pruning (MoE-I²) 缩减每个 expert 内部 FFN 的 intermediate 维度。然而，pruning 虽然减少显存占用，在 vLLM 等优化推理框架上的实际推理吞吐量提升有限甚至退化——原因是 token-to-expert 路由不变（仍需路由到固定 top-k 个 expert），剩余 expert 需处理更多 token，导致负载不均衡和 latency 增加。此外，pruning 方法依赖 calibration 数据集进行 expert 重要性评估，使得剪枝模型可能过拟合到 calibration 分布。
  - 全栈执行例子（Baseline Mixtral-8x7B, 4×H100, vLLM + FusedMoE）：
    - **推理算法层**：对所有 32 层，top-k=2 的固定路由。每个 token 经 router 选出 top-2 experts，Expert FFN (W_gate, W_up, W_down, dim=14336) 计算 → 加权求和。Inter-pruning (50%) 删除每层 4 个 expert 后，剩余 4 个 expert 需处理原 8 个 expert 的全部 token。
    - **系统框架层**：vLLM + FusedMoE。PagedAttention 管理 KV-cache。Tensor Parallelism 跨 4×H100。FusedMoE 将 expert 计算和路由融合以减少 kernel launch overhead。固定的 top-k 意味着固定的 all-reduce/broadcast 通信模式。
    - **编译框架层**：论文未明确说明（vLLM 使用 PyTorch eager mode + custom CUDA kernels）。
    - **kernel 调度层**：FusedMoE kernel 在 H100 Tensor Cores 上批量执行 expert GEMM。Token dispatch 按 expert 分组。Pruning 后 expert 负载不均衡：某些 expert 收到远超平均的 token 数 → 长尾 latency。
    - **硬件架构层**：4×H100 80GB，NVLink 互联，Tensor Cores。Expert 参数从 HBM 加载。Pruning 虽减少 HBM 占用但 bandwidth 节省有限（仍需加载所有未剪枝 expert）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LExI 方法**：核心洞察是"不同层对 expert 数量的敏感度不同"——有些层减少 active expert 几乎不影响输出（低敏感层），而有些层则需要更多 expert（高敏感层）。LExI 利用这一特性实现 layer-adaptive top-k 分配：
    1. **Data-Free Sensitivity Profiling**：仅用模型权重 + 随机 Gaussian 输入，通过 Monte Carlo 采样计算每层在不同 top-k 下的 Frobenius 范数输出偏差。解决了 pruning 方法对 calibration 数据的依赖问题。
    2. **Evolutionary Search**：以 sensitivity proxy 为引导，在总 active expert budget B 约束下搜索全局最优的逐层 top-k 分配。将"top-k 从{1,2,...,k_base}^L 的组合优化问题"转化为进化搜索问题，避免了梯度方法的巨大计算开销。
    3. **Static Per-Layer Top-k Assignment**：不同层使用不同数量的 active expert——低敏感层用更少的 expert 省计算，高敏感层保留更多 expert 保精度。
  - 对应解决 Baseline 缺陷：
    - **Pruning 不提升推理吞吐量（甚至退化）** → LExI 不删除任何 expert，而是通过减少低敏感层的 active expert 数量直接减少 FFN 计算量。每个 expert 仍保留完整参数，不存在负载不均衡问题（所有 expert 均可被路由）。
    - **固定 top-k 导致冗余计算** → LExI 的 layer-adaptive top-k 在低敏感层激活更少 expert，在高敏感层保持充分 expert 容量，实现"按需分配"的精细化计算。
    - **Pruning 依赖 calibration 数据** → LExI 的 sensitivity profiling 仅使用随机 Gaussian 输入和模型权重，完全 data-free。
    - **Pruning 不可逆/不可调节** → LExI 的 budget B 是可控参数：B 越小，计算越少（吞吐越高）；B 越大，越接近 baseline 精度。无需重新训练或重新 profiling 即可在不同 B 之间切换。
  - 全栈执行例子（LExI on Mixtral-8x7B, 4×H100, vLLM + FusedMoE）：
    - **推理算法层**：LExI 离线计算得到 32 层的 top-k 分配，如 [1, 2, 1, 2, 1, 1, 2, ..., 2]。总 budget B = Σ k_j = 50（vs baseline B = 32×2 = 64）。低敏感层用 k_j=1（仅激活 1 个 expert 而非 2 个）直接省去一个 expert 的 FFN 计算。
    - **系统框架层**：vLLM + FusedMoE 不变。仅修改 MoE 路由参数（set_topk），无需改变调度、内存管理或 kernel。推理时每层自动按各自的 k_j 激活对应数量的 expert。
    - **编译框架层**：论文未明确说明。与 baseline 相同。
    - **kernel 调度层**：FusedMoE kernel 不变，但减少了 total expert computation——每 token 每层减少 k_base - k_j 次 FFN forward。H100 Tensor Cores 计算负载降低。All-reduce/broadcast 通信量随 active expert 减少而降低。No load imbalance（所有 expert 保留完整权重）。
    - **硬件架构层**：同 baseline。减少的 computation 直接转化为更低的 latency 和更高的 throughput。
