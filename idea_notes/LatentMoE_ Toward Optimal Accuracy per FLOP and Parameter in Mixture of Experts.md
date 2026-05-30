## LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

- baseline方法是什么？
  - **Standard MoE**（以 Qwen3-235B-A22B / DeepSeek-v2-lite 为代表的 fine-grained MoE）：每个 MoE layer 包含 N=128 routed experts + S=2 shared experts，每个 expert 的 FFN 在原始 hidden dimension d=4096 中操作，中间维度 m=2688。Router 计算 softmax(W_r·x) 后 top-K=6 选择 expert，加权组合输出。所有 expert 共享相同的输入空间维度 d，expert 参数规模 3×d×m per expert。Standard MoE 的设计逻辑是基于高层次的稀疏性论证（仅激活少数 expert 减少 FLOPs），未充分考虑硬件实际瓶颈。
  - Baseline 核心缺陷：
    1. **Memory Bandwidth 瓶颈（低延迟场景）**：小 batch 推理中 MoE expert 计算是 memory-bound。GB200 系统 arithmetic intensity ≥1250 FLOPs/byte 才 compute-bound，Qwen3-235B 在 latency-critical 部署中 t_exp<1418，operating point 在 memory-bound 区域。
    2. **All-to-All 通信瓶颈（高吞吐场景）**：大 batch 下 MoE layer 通信/计算时间比 ~9:1（GB200 NVL72 + Qwen3-235B），all-to-all 是主要瓶颈。
    3. **专家组合空间有限**：N=128, K=6 仅 C(128,6) 种组合，限制 token 级组合稀疏性带来的表达能力。
    4. **d 过度预留给 routed experts**：虽然任务特征秩 r_eff << d，但所有 routed expert 的 input dim 保持为 d。
  - 全栈执行例子（Baseline Standard MoE, 95BT-8BA, EP on multi-GPU）：
    - **模型推理算法层**：Attention(DP/TP) → Router top-K=6 → 6 experts 在 d=4096 做 FFN(d×m, d×m, m×d) → gate-weighted combine → shared experts。Router/shared/attention 均在 d。
    - **系统框架层**：Expert Parallelism (EP), All-to-All dispatch/combine, vLLM serving。Data parallelism for attention。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL backend + vLLM CUDA kernels。
    - **kernel 调度层**：NCCL All-to-All(FP4+BF16) + CUTLASS GroupedGEMM + cuBLAS GEMM。通信 ∝ K·d=6×4096。Expert 权重加载 ∝ d·m。
    - **硬件架构层**：H100 GPUs / GB200 NVL72。All-to-All 占 ~90% 总时间(throughput)。Memory BW bound(latency)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LatentMoE 方法**：将 routed experts 输入从 d 压缩到 ℓ（latent dim），节省的 memory BW/communication 预算重新投资到 expert 数量和 top-K，实现 accuracy per FLOP/parameter Pareto 改进。核心创新：将"浪费"在过大 hidden dim 上的资源转为 expert diversity 提升。
    1. **Latent Space Projection (W_↓, W_↑)**：所有 routed experts 共享 W_↓∈R^{ℓ×d} 和 W_↑∈R^{d×ℓ}。Expert 权重从 3×d×m 降至 3×ℓ×m (α=4× reduction)。All-to-All 在 latent space ℓ 中进行。
    2. **Expert Count Scaling (N→N'=αN)**：用节省的参数预算将 experts 从 128 扩至 512。C(512,K) >> C(128,K)。
    3. **Top-K Scaling (K→K'=αK, ℓ-MoE_acc)**：top-K 从 6 扩至 24。通信量 K'·ℓ = 24×1024 = baseline K·d = 6×4096 (iso-communication)。C(αN,αK) ≥ C(N,K)^α，指数级扩大组合空间。
    4. **压缩比下界验证**：α sweep 实验验证 r_eff≈d/4，α=4 safe, α=8 精度塌缩。
    5. **Shared Experts 保持原始空间**：非瓶颈，保持 d 操作。
  - 对应解决 Baseline 缺陷：
    - Memory BW bottleneck → Expert 权重 ℓ×m << d×m (per expert ↓α×)，总 loading 在 ℓ-MoE_acc 不变、ℓ-MoE_eff ↓。
    - All-to-All bottleneck → 通信在 latent space (msg size ↓α× per token)，K'=αK 使 ℓ-MoE_acc 总通信不变、ℓ-MoE_eff ↓α×。
    - 组合空间有限 → N:128→512, K:6→24, C(512,24) >> C(128,6)。
    - d 过度预留 → 压缩到 ℓ=d/4，验证 r_eff 下界。
  - 全栈执行例子（LatentMoE ℓ-MoE_acc, 95BT-8BA, d=4096, ℓ=1024, α=4）：
    - **模型推理算法层**：Attention(d=4096) → Router(原始空间) → top-K'=24 → Shared W_↓[1024,4096] down-project → 24 experts 在 latent space ℓ=1024 做 FFN(W_g/W_up[2688,1024], W_FC2[1024,2688]) → combine → Shared W_↑[4096,1024] up-project → Shared Experts(d) → output。
    - **系统框架层**：EP, All-to-All in latent space ℓ。vLLM FP8。All-to-All msg size = ℓ (vs d baseline), token count = K' (vs K baseline), 总通信量相同(ℓ-MoE_acc)。
    - **编译框架层**：TensorRT-LLM v1.2.0+ 支持。提议优化: (1) 分离 CUDA streams for routed/shared experts, (2) CUTLASS small-inner-dim GEMM kernels for latent-space experts。
    - **kernel 调度层**：W_↓/W_↑ GEMM (modest, ~9% overhead at trillion scale) + Routed Expert GEMM(ℓ×m, 4× smaller) + Shared Expert GEMM(d×m) + All-to-All(FP4+BF16 in ℓ)。
    - **硬件架构层**：H100 GPUs (实测), GB200 NVL72 (投影)。High concurrency 下 throughput ↓≤6%。Trillion-scale: Kimi-K2-1T-LatentMoE 比 iso-accuracy Kimi-K2-1.35T 快 1.24×-3.46×。
  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | LatentMoE 设计 | 效果 |
    |-------------|---------------|------|
    | Memory BW bottleneck | ℓ·m << d·m per expert | Weight loading ↓α× per expert |
    | All-to-All bottleneck | Communication in latent space ℓ | Msg size per token ↓α× |
    | d 过度预留 | 压缩到 ℓ=d/α, α=4 | r_eff≈d/4, info loss negligible |
    | C(128,6) 有限 | N→512, K→24, C(512,24) | 指数级增长组合多样性 |
    | 仅优化 offline throughput | Accuracy per FLOP + per parameter | 两种 deployment 均受益 |
    | 压缩导致训练不稳定 | Expert scaling N→αN 补偿 | 恢复 baseline 稳定性 |
  - **核心设计洞察**：Standard MoE 的 hidden dim d 在 routed expert 层面过度配给——任务特征秩 r_eff<<d。N 和 K 受制于 d·m (memory BW) 和 K·d (communication) 无法扩展。LatentMoE 通过解耦 input space (d→ℓ) 获得三个自由度: (1) 降低 input dim → memory BW per expert ↓; (2) 增加 N → 更细粒度专业化; (3) 增加 K → 更大非线性预算 U_eff=K·m。在 iso-inference-cost 下将"浪费"在过大 d 上的资源转为 expert diversity 和 nonlinearity budget，系统性提升 accuracy-efficiency Pareto frontier。五项 Design Principles 构成 hardware-software co-design 框架使这一重新分配有据可依、有界可循。
