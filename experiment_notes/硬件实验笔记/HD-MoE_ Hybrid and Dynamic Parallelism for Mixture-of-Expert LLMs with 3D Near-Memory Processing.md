## HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

- 属于硬件架构的实现是什么？实验比较什么？
  - HD-MoE 在硬件架构层面的实现是针对 3D NMP（Near-Memory Processing）加速器的 **Hybrid and Dynamic Parallelism Mapping Framework**，包含：
    1. **Performance Analytical Model**：统一的性能分析框架，建模 3D NMP 上 MoE 推理的 computation overhead（t_comp = max_c{ Σ_i P_ic * f_i * B * 2h * IS / comp }）和 communication overhead。communication 部分采用 discrete-event simulator 精确建模 2D mesh NoC 中的不规则 all-to-all 通信（XY routing + 优先级队列事件调度 + 链路占用追踪），并提供线性近似模型 t̂_comm 用于 LP 优化。经验验证 R² > 0.9，并对比 ASTRA-sim 验证 ring all-reduce 延迟（分析模型 673µs vs 仿真 668µs，误差 <1%）。
    2. **Node-Link Balance Co-optimization**：两阶段优化——Stage 1 (Node Balance)：LP 求解器优化逻辑集群上的 expert placement，平衡计算负载并最小化通信量（min t_comp + 2γ*t̂_comm）；Stage 2 (Link Balance)：Bayesian Optimization 将逻辑集群映射到 2D mesh 物理节点，最小化链路拥塞。
    3. **Dynamic Placement Strategy**：运行时根据 expert activation 预测调整部署，包含 congestion-aware expert prediction、cost-optimal broadcasting（α-β 模型推导最优 chunk size）、communication-efficient token routing。
  - 实验比较：三种 baseline（TP、EP、Hybrid TP-EP with Compute-Balanced）在不同硬件配置（compute/bandwidth ratio）和不同 mesh 尺寸（4×4, 4×8, 8×8）下的 TBT latency 和 speedup。Ablation 分别评估 Node Balance、Link Balance、Dynamic Placement 各阶段的贡献。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 论文自建 Python-based discrete-event simulator 用于 2D mesh NoC 通信模拟（XY routing + priority queue scheduling + link occupancy tracking）。
  - 验证工具：ASTRA-sim（https://github.com/astra-sim/astra-sim）用于验证 ring all-reduce 分析模型的准确性。
  - 线性规划求解器（未指定具体工具，论文未明确说明使用何种 LP solver——论文提及 "LP" 但未给具体库名）。
  - Bayesian Optimization 框架（论文未明确说明使用何种 BO 库）。

- 模拟器模拟什么的性能，修改了什么。
  - 模拟器模拟 3D NMP 加速器上 MoE LLM 推理的端到端性能，具体包括：
    - **Computation latency**：基于各节点 expert 负载（tokens × FLOPs/token / comp_throughput），取 max across nodes 作为计算瓶颈。
    - **Communication latency**：离散事件模拟 2D mesh NoC 中 all-to-all dispatch + all-to-all combine 的通信时间线。通信任务按 chunk 切分，XY routing 计算最短路径（Manhattan distance），优先级队列调度传输，链路占用字典追踪可用带宽。
    - **Mesh topology**：支持可配置的 2D mesh 尺寸（4×4, 4×8, 8×8），计算吞吐（2.5/5/10 TFLOPS），NoC 链路带宽（25/50/75 GB/s）。
    - **Ring all-reduce 验证**：自建分析模型（t_comm = 4Bh/BW）与 ASTRA-sim 仿真结果对比，验证通信模型在不同延迟和带宽参数下的准确性。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 代码开源：https://github.com/angerybob/HD-MoE
  - 模拟器使用流程与评估原理：
    1. **输入配置**：模型参数（expert 数量 E, experts/token ē, layers, hidden_size h, intermediate_size IS）、硬件配置（D×D mesh 节点数, comp TFLOPS, BW GB/s per link）、batch size B、expert activation trace（f_i, f_g 从 MT Bench 统计）、placement matrix P_ic（LP + BO 优化结果）。
    2. **Computation 模拟原理**：对每层每个节点 c，计算分配给它的 tokens 数 = Σ_i P_ic * f_i * B。每个 token 在 expert FFN 中的 FLOPs = 2h * IS（gate_proj + up_proj + down_proj 的 GEMM）。节点 c 的计算延迟 = tokens_c * 2h * IS / comp。取 max_c 作为该层计算瓶颈。
    3. **Communication 模拟原理（离散事件）**：对每个 token 的 activated expert group g，生成通信任务（src=expert 物理节点, dst=随机聚合节点）。XY routing 计算 Manhattan 最短路径上的每一跳。所有通信任务按 chunk 切分，进入 priority queue，按时间戳顺序调度。每个链路维护占用时间表，新任务在链路空闲时才开始传输，传输时间 = chunk_size / BW。模拟器输出完整时间线，取最后一个通信任务完成时间作为 t_comm。
    4. **性能输出**：MoE Decomposed Latency = t_comp + t_comm。Normalized TBT = (t_comp + t_comm)_HD-MoE / (t_comp + t_comm)_TP。用于对比不同策略在不同硬件配置和 batch size 下的加速比。
