## BrownoutServe SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

- baseline方法是什么？
  **Baseline 为 vLLM**：一个高性能 LLM 推理引擎，支持 ContinuousBatching、FlashAttention、PagedAttention 等优化技术。vLLM 对 MoE 模型采用静态配置，所有 token 的 expert routing 为标准的 top-k gating——每个 token 经过 gate 后由 top-k 个原 experts 处理（zero-brownout），所有 experts 全部参与计算而不做任何降级。

  **Baseline 全栈执行例子（以 Qwen1.5-MoE-A2.7B-Chat, 60 experts/layer, batch_size=64, 4×A100-40GB 为例）**：

  - **算法层**: 输入 token x_t → Gate 计算 s_{i,t} = x_t^T · e_i → Top-2 Softmax routing → 60 个 experts 中被选中的 top-2 expert 处理 → h_t = x_t + Σ FFN_i(x_t)。所有 expert 按实际路由需求参与计算，不做降级。
  - **系统框架层**: vLLM Scheduler (FCFS) → ContinuousBatching 管理 batch → PagedAttention 管理 KV cache → Fused MoE kernel 执行 expert FFN。负载高峰时等待队列积压，请求延迟增大，可能出现 SLO 违规。静态配置下无自适应调节能力。
  - **编译框架层**: 论文未明确说明（vLLM 使用 PyTorch + CUDA/C++ kernel）。
  - **Kernel/运行时调度层**: CPU 端 block table → CPU→GPU 传输 → FlashAttention kernel → Gate kernel → Fused MoE kernel（多 expert FFN 合并为一次 sparse GEMM）。MoE 阶段成为瓶颈——prefill 阶段 MoE 占 transformer layer latency 81.23%，decoding 阶段占 93.89%（Fig. 1）。
  - **硬件架构层**: 4× A100-PCIE-40GB GPU，60 experts 分布在各 GPU 上。Cold experts（少数 token 路由到的 expert）无法充分利用 GPU 并行性——token 分布呈长尾模式，少量 hot expert 处理大量 token，多数 cold expert 利用率低。bursty workload 下 GPU 资源紧张，水平扩展（Kubernetes HPA/新实例）冷启动延迟 30-90s 模型加载 + 30-60s 实例初始化，无法及时响应。

  **Baseline 的核心缺陷**：
  1. **Expert 负载不均衡导致 GPU 利用率低**: 仅少量 hot experts 处理大量 token，cold experts 的 GPU SM 空闲，token 分布的 long-tail pattern 导致资源浪费。
  2. **Bursty workload 下静态配置无法自适应**: 突发流量时无降级机制，请求排队延迟急剧上升，SLO 违规严重。传统水平扩展冷启动延迟 1-2 分钟，不适用于短时突发。
  3. **MoE 模块是推理瓶颈**: prefill 阶段 MoE 占 81.23%，decoding 占 93.89%，但 vLLM 未针对 MoE 做专门的延迟优化。
  4. **全 expert 激活的巨大计算压力**: 大 batch 下 quasi-dense 激活，所有 expert 参与计算，通信带宽和计算资源压力大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: BrownoutServe = United Expert 知识蒸馏 + Brownout 路由算法 + SALC 动态阈值调节。核心思路是借鉴电力系统的 brownout（降级供电）概念，在推理时用精度换延迟：将冷门 experts 的 token 交由 united experts 处理，减少 expert 访问次数，通过 SALC 算法动态维持延迟在 SLO 范围内。

  **Defect → Design 映射**：

  | Baseline 缺陷 | BrownoutServe 设计选择 | 解决机制 |
  |---|---|---|
  | Expert 负载不均衡 → cold expert GPU 利用率低 | United Experts 合并多 expert 知识，partial-brownout 将 cold expert 的 token 聚合到 united expert | 减少 expert 访问次数（如 8 experts → 5 次访问），增大每个被访问 expert 的 batch size，提升 GPU 并行度 |
  | Bursty workload 下无法自适应 | SALC 算法：P90 latency < warning_line → threshold↑（提升精度）；P90 latency > SLO → threshold↓（降级保延迟） | 闭环反馈控制，threshold 在 [0,1] 动态调整，突发时自动触发 brownout，突发结束后恢复 |
  | MoE 模块是延迟瓶颈 | Brownout 减少 expert 访问次数，Triton 重写 MoE kernel，优化 PagedAttention block table | MoE latency 随 expert 访问次数线性下降；Triton kernel 进一步提升效率 |
  | 全 expert 激活的计算压力 | Brownout threshold 控制参与计算的专家比例 | threshold=0.4 时仅 37.5% experts 直接参与，其余由 united experts 替代 |

  **论文方法全栈执行例子（以 Qwen1.5-MoE-A2.7B-Chat, partial-brownout, way=8, threshold=0.4, bursty workload 为例）**：

  - **算法层**: x_t → Gate 计算 affinity → Top-K routing → 统计各 expert token 数 → 降序排列 → 前 40% token 由原 experts 处理（S1），后 60% token 按 way=8 分组由 united experts 处理（S2）→ h_t = x_t + ΣFFN^{(s)} + Σp_{i,t}·FFN^{(r)} + Σq_{i,t}·FFN^{(u)}。United experts 通过离线 MSE 蒸馏训练：L_MSE^j = (1/k) Σ ||H_u^j - H_o^{j·k+i}||²。
  - **系统框架层**: BrownoutServe Scheduler (FCFS + ContinuousBatching) → SALC 每 iteration 监控 P90 latency → 对比 SLO warning line 和 SLO → 动态更新 threshold。例如：突发前 threshold=0.8，delay<warning_line → threshold 逐渐增至 0.9；t=75s 突发 → latency 超 SLO → threshold × 0.8 降至 0.72 → 更多 token 走 united experts → latency 回落到 warning_line~SLO 之间 → 突发后逐步恢复 threshold。
  - **编译框架层**: 论文未明确说明（PyTorch + Triton kernel 编译为 GPU 代码）。
  - **Kernel/运行时调度层**: GPU 端 block table → FlashAttention → Gate kernel → Brownout 划分 kernel（GPU sort + partition）→ S1 Fused MoE kernel（原 experts）→ S2 Triton United Expert kernel（concat tokens → FFN）→ 输出。PagedAttention block table GPU 化消除 CPU→GPU 传输。
  - **硬件架构层**: 4× A100-PCIE-40GB，united experts 权重常驻 GPU 显存（与原 expert 同参数规模，总 united experts 数远少于 experts），无需 extra GPU 资源。way 切换时需通过 Experts Loader 重新加载不同的 united experts 权重（GPU memory ↔ CPU memory/disk），但 threshold 调整 zero-overhead。
