## MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

- 属于Serving调度的实现是什么？实验比较什么？
  MegaScale-Infer 提出解耦式专家并行（Disaggregated Expert Parallelism），将 MoE 模型的 attention 模块与 FFN/Expert 模块分离部署到不同的 GPU 节点上，实现独立扩展和异构部署。核心包含三个机制：
  1. **Disaggregated Expert Parallelism**：将 attention 模块复制到多个 attention node（数据并行），FFN experts 分布在 expert node 上（expert 并行），每个 expert node 包含 1-8 个 GPU（节点内使用 tensor parallelism）。Attention node 聚合来自多个 replica 的请求，增大每个 expert 的有效 batch size，使 FFN 从 memory-intensive 转变为 compute-intensive。
  2. **Ping-Pong Pipeline Parallelism**：将请求 batch 拆分为 m 个 micro-batch，在 attention node 和 expert node 之间形成 ping-pong pipeline。Micro-batch 在 attention 和 expert 之间交替传递，前向计算覆盖通信开销，约束条件为 T_a ≈ T_e, T_c < T_f, m × T_f ≥ 2 × (T_f + T_c)。
  3. **Deployment Plan Search（Algorithm 1）**：枚举 tp_a, tp_e, n_a, m 组合，通过性能模型（基于 roofline 模型的 GEMM 时间估算 + profiling 获取的 k_i 系数 + network bandwidth utilization profiling）SIMULATE binary search 最大 global batch size 满足 SLO 约束。选择最大化 throughput per unit cost 的 deployment plan。
  4. **Heterogeneous Deployment**：attention node 使用高 per-cost 内存带宽/容量的 GPU（如 H20），expert node 使用高 per-cost 计算能力的 GPU（如 L40S）。
  实验比较了 MegaScale-Infer 与 vLLM（仅 tensor parallelism）和 TensorRT-LLM（tensor parallelism + expert parallelism）在不同模型（Mixtral 8x22B, DBRX, Scaled-MoE 317B）和硬件配置（同构 Ampere 集群、异构 H20+L40S 集群）下的 per-GPU decoding throughput、time between tokens (TBT)、end-to-end throughput（含 prefill）、per-cost throughput、per-unit-power throughput。

- 硬件平台是什么，配置是什么。
  同构集群：8 节点，每节点 8×NVIDIA 80GB Ampere GPU（如 A800），128 CPUs，2 TB host memory，8×200 Gbps InfiniBand NICs，节点内 400 GB/s NVLink。
  异构集群：NVIDIA H20（96 GB, 4096 GB/s bandwidth, 148 TFLOPS, 900 GB/s NVLink, 4×400 Gbps NICs）+ NVIDIA L40S（48 GB, 864 GB/s bandwidth, 362 TFLOPS, PCIe intra-node, 2×400 Gbps NICs）。
  bfloat16 用于 weights、activations 和 KV cache。

- 开源Serving框架是什么。修改了什么。
  MegaScale-Infer 是自研系统，与 vLLM 和 TensorRT-LLM 对比。核心修改：
  - 将 MoE 模型的 attention 和 FFN/expert 模块拆分为独立可部署单元。
  - 实现了 ping-pong pipeline 调度器，管理 micro-batch 在 attention node 和 expert node 之间的流水线执行。
  - 实现了 M2N 通信库（PyTorch extension，~4900 行 C/C++ + ~5000 行 Python）替代 NCCL 进行 attention-expert 间的 token dispatch/aggregation 通信。
  - 使用 Flux 实现 TP 通信与相邻 GEMM 的 kernel fusion（如 all-gather + GEMM 融合为单 kernel）。
  - 实现了 sequential memory-intensive operators 的融合（gating + top-k selection + token scatter），减少 kernel launch 和 memory access。
  - 基于 expert 流行度的 on-device redundancy load balancing：将 M 个 expert 分布到 N 个节点，minimize max C_j，使用 greedy approximation 解决。
  - 基座框架：论文未明确说明具体基座，但提到使用 Flux（ByteDance 的 kernel fusion 库）和自研 M2N 通信库。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未声明开源。代码为公司内部部署（已部署到近 10,000 GPU 的生产推理服务中，降低 serving cost 1.5-2.0×）。
  
  **MegaScale-Infer 推理全流程（以 Mixtral 8x22B decoding 为例，tp_a=2, tp_e=1, n_a=4, E=8, m=3）**：
  1. **请求接入与 micro-batch 划分**：用户请求到达 MegaScale-Infer runtime instance。请求被划分为 m=3 个 micro-batch，每个 micro-batch 大小 b_a = B/3。
  2. **Attention 计算（Attention Node，Layer ℓ）**：每个 attention node（包含 2 GPU tensor parallel）对 micro-batch i 执行 attention 计算（QKV projection + attention score + output projection），读取 KV cache（memory-intensive）。Attention node 1..n_a 各持有完整的 attention 参数副本和各自的 KV cache。
  3. **Gating + M2N dispatch**：Attention node 执行 gating network 计算（选择每个 token 的 top-K=2 experts）→ fused kernel（gating + top-k + scatter）准备 token embeddings → 通过 M2N Sender（CUDA event wait → CUDA stream block → Core Sender 通过 RDMA write with immediate 写数据到目标 expert GPU → poll completion queue → unblock stream）将 token embeddings 发送到对应 expert node。
  4. **Expert 计算（Expert Node，Layer ℓ）**：Expert node 通过 M2N Receiver（CUDA event wait → stream block → poll completion queue → GDRCopy flush → unblock stream）接收来自所有 attention node 的 tokens → 按 expert 聚合 tokens 为 batch → 执行 FFN Input GEMM + activation + FFN Output GEMM（compute-intensive，batch size = b_e = b_a × n_a × K/E）。
  5. **M2N aggregation**：Expert node 将 FFN 输出通过 M2N 反向发送回 attention node。
  6. **Ping-Pong Pipeline**：在 Layer ℓ 的 expert 计算期间，attention node 已开始 Layer ℓ 的下一个 micro-batch 或 Layer ℓ+1 的 attention 计算。m=3 时 pipeline 能完全覆盖通信时间（T_c < T_f 时），3 个 micro-batch 在 attention 和 expert 之间交替流动。
  7. **迭代完成**：所有 m 个 micro-batch 完成 L 层 MoE 的 forward pass 后，总 iteration latency T_total = (T_a + T_e + 2T_c) + T_f(mL − 1)。输出 tokens 返回用户。
