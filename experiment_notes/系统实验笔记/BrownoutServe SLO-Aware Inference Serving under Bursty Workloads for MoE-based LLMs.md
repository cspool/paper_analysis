## BrownoutServe SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是 **BrownoutServe** —— 一个面向 MoE LLM 的端到端 inference serving 框架，核心包含两大机制：(1) **United Experts**：通过知识蒸馏将多个 MoE expert 的知识合并到单个同参数规模的 united expert，减少推理时的 expert 访问次数；(2) **Dynamic Brownout Mechanism**（含 Brownout Approach 和 SLO-Aware Latency Control/SALC 算法）：在资源受限或突发流量时，动态将部分 token 路由到 united experts 处理，减少 expert 访问开销，同时通过 SALC 算法自适应调整 brownout threshold 以平衡延迟和精度。

  实验比较：
  - 吞吐量：BrownoutServe vs vLLM (non-fused) 和 vLLM (native/fused MoE)，在 ShareGPT 和 Alpaca 数据集上，不同 request rate 下持续 10 分钟
  - 精度：在不同 (way, threshold) 配置下的 accuracy loss，使用 PIQA、COPA、CEVAL、OBQA 四个 5-shot 任务
  - SLO 违规率：在突发流量场景下（250s trace，t=75s 时 RPS 翻倍），BrownoutServe vs vLLM 的 prefill/decoding 阶段 token-level SLO 违规率
  - 延迟 trace 分析：250s 内的 P90 prefill/decoding latency 变化轨迹
  - Threshold 自适应变化分析

- 硬件平台是什么，配置是什么。
  4× NVIDIA A100-PCIE-40GB GPU（每卡 40GB），Intel Xeon Gold 6238 CPU。

- 开源Serving框架是什么。修改了什么。
  BrownoutServe 是**自研的定制 Serving 框架**，并非直接修改 vLLM 源码，而是用约 5.5k 行 Python 从 PyTorch 构建，同时**集成了 vLLM 中的多项优化技术**：PagedAttention（并进行了优化——将 block table 移到 GPU，block table 操作实现为 GPU kernel）、FlashAttention、ContinuousBatching/iteration-level scheduling。MoE 模块引入 brownout approach，MoE 算子使用 Triton 重写。

  **控制平面修改**：
  - **Scheduler**: 使用 FCFS 调度，当 engine 达到最大 batch 容量时多余请求进入等待队列；支持 streaming I/O，允许动态插入新请求和提前移除已完成请求
  - **SLO Analyzer**: 持续监控 TTFT 和 TPOT，运行 SALC 算法动态调整 brownout threshold
  - **Experts Loader**: 负责加载/卸载 united experts，更新 united experts 的 way 配置

  **数据平面修改**：
  - 集成 BrowoutMoE 模块（含 fused MoE 和 brownout routing）
  - 优化 PagedAttention：block table 移至 GPU，block table 操作实现为 CUDA kernel
  - MoE 算子全部使用 Triton 重写

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  **开源**: https://github.com/beyondHJM/BrownoutServe（Apache-2.0 协议，约 32 commits），但预训练的 United Expert 权重需联系作者获取。

  **框架输入到硬件执行的全过程（以 Qwen1.5-MoE-A2.7B-Chat, partial-brownout, way=8, threshold=0.4 为例）**：

  1. **请求到达**: 用户请求通过 HTTP/gRPC 到达 Scheduler（FCFS），请求包含 prompt text 和 SLO 要求
  2. **Batch 组装**: Scheduler 从等待队列中取出请求组装 batch（max batch size=64），支持 ContinuousBatching——每 iteration 完成后从 batch 中移除已完成请求，加入新请求
  3. **Prefill 阶段**: 所有 prompt tokens → Embedding → Attention (FlashAttention + PagedAttention with GPU-side block table) → FFN → **BrownoutMoE**:
     - Gate 单元计算每个 token 对所有 60 个 experts 的 affinity score s_{i,t} = x_t^T · e_i
     - Top-K routing 选出每个 token 的 top experts
     - 统计每个 expert 的 token 数量，按降序排列
     - 根据 threshold=0.4，累计 token 数达到 40% 的 experts 进入 S1（由原 experts 处理），其余进入 S2（由 united experts 处理）
     - S1 tokens → 原 experts FFN（fused MoE kernel on GPU）→ 输出
     - S2 tokens → 按 way=8 分组（60 experts → 8 groups）→ 每组 tokens concat → 对应的 united expert FFN → 输出
     - 因部分 tokens 使用 united expert（参数固定在 GPU 显存中），减少 expert 访问次数 → 降低 latency
  4. **Decoding 阶段**: 逐 token 自回归生成，每个新 token 经过相同的 BrownoutMoE 流程
  5. **SLO Analyzer 反馈**: 每 iteration 后 SLO Analyzer 收集 P90 TTFT/TPOT latency。若 P90 latency > SLO → threshold × shrink_ratio (如 0.8) 降低更多 token 走 brownout；若 P90 latency < warning_line (SLO × warning_factor) → threshold + increment (如 +0.1) 提升精度
  6. **输出返回**: 生成的 token 流式返回给客户端
