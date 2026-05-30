## MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：一个基于 Integer Linear Programming (ILP) 的 MoE 专家放置优化框架，包含三阶段：(1) **Token Routing Profiling**——在任务数据集的采样子集上运行推理，记录逐层的 token-to-expert 路由统计（每个 expert 的 token 处理量 P_{e,l}，以及跨层 expert 对间的 token 路由量 R_{e_1,e_2,l}）；(2) **ILP 优化**——分两步求解：ILP 1 按层将 experts 聚类到 G 个 cluster，最小化各 cluster token 处理负载与层内均值的偏差（min Σ|T_{c,l} - T̄_l|）；ILP 2 将 cluster 分配到 GPU，最小化跨 GPU 通信成本的层间最大值（min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2})），同时保证每个 GPU 的 expert 数量均衡（Σ x_{c,e,l}·y_{c,g,l} = E·L/G）；(3) **Custom Expert Parallelism Initialization**——将 ILP 输出的 expert-to-GPU 映射保存为 PyTorch tensor 文件，初始化模型时替换 Megatron-LM 的默认 contiguous block 放置。
  - 实验比较：(1) End-to-End Speedup：MoETuner vs Megatron-LM default expert placement，单节点（8×H100, 4EP-2TP）和多节点（16×H200, 4EP-4TP）下的推理加速比；(2) Token Processing Time：每层的 token 处理尾延迟（tail latency，最长 GPU 处理时间）和平均延迟；(3) All-to-All Time：每层的 all-to-all 通信尾延迟和平均延迟；(4) Token processing load distribution（箱线图）和 Token dispatching distribution（GPU pair 间 token 分发量箱线图）。

- 硬件平台是什么，配置是什么。
  - 单节点：8× NVIDIA H100 SXM5 80GB，NVLink Gen4 (900GB/s)，双路 Xeon Platinum 8462Y+，2048GB DDR5 4800MHz，NVIDIA ConnectX-7 IB (400Gbps)。
  - 多节点：2 节点共 16× NVIDIA H200 SXM5 142GB，NVLink Gen4 (900GB/s)，双路 Xeon Platinum 8562Y，2048GB DDR5 5600MHz，NVIDIA ConnectX-7 IB (800Gbps)。

- 开源Serving框架是什么。修改了什么。
  - 框架：Megatron-LM (https://github.com/NVIDIA/Megatron-LM) 作为 baseline expert parallelism 框架。
  - 修改内容：
    - **All-to-all 通信模块**：修改 Megatron-LM 的 all-to-all 通信和 expert placement 模块，支持自定义 expert-to-GPU 映射（允许每个 GPU 分配不同数量的 expert，打破默认 contiguous block 分配）。
    - **Custom Expert Parallelism 初始化**：加载 ILP 输出的 expert-to-GPU mapping tensor，按层替换默认 expert placement。
    - **Token Routing Profiling**：在 Megatron-LM 推理流程中插入 profiling hook，记录每个 token 在各层的路由路径，构建路由统计表。
  - 开源情况：论文未明确说明开源链接，web search 未发现公开 GitHub 仓库（论文 2025 年 2 月发布，代码可能尚未公开）。ILP 求解使用 Gurobi 12.0.0 (https://www.gurobi.com)。
  - 使用例子——MoETuner 对 Megatron-LM MoE 推理的优化全流程：
    1. **Profiling 阶段**：在目标 task 数据集（如 WikiText-103）的采样子集上运行 N 轮 Megatron-LM 推理 → 收集每层每个 token 路由到的 expert 对 → 构建路由统计表（P_{e,l}: expert e 在层 l 的处理 token 数；R_{e_1,e_2,l}: 层 l 到 l+1 间 expert e_1→e_2 的 token 路由数）。
    2. **ILP 1 求解**：输入 P_{e,l}、E（expert数）、L（层数）、G（GPU数）→ 决策变量 x_{c,e,l} ∈ {0,1}（expert e 是否分配到 cluster c）→ min Σ|T_{c,l} - T̄_l|，约束每个 cluster 至少一个 expert → Gurobi 求解至 tolerance 0.025 → 输出 x_{c,e,l}（每个 expert 的 cluster 归属）。
    3. **ILP 2 求解**：用 x_{c,e,l} 预计算 C_{c_1,c_2,l} = Σ R_{e_1,e_2,l} · x_{c_1,e_1,l} · x_{c_2,e_2,l} → 输入 y_{c,g,l} ∈ {0,1}（cluster c 是否分配给 GPU g），带宽 B_{g_1,g_2} → min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2} · y_{c_1,g_1,l} · y_{c_2,g_2,l+1})，约束每个 GPU 等量 expert、每个 cluster 一对一映射 → 输出 y_{c,g,l}（每个 cluster 对应哪个 GPU）。
    4. **部署阶段**：Megatron-LM 加载 MoE 模型 → 初始化时读取 ILP 输出的 expert-to-GPU mapping → 替换 contiguous block placement → 推理时按优化后的 layout 执行 all-to-all token dispatching → GPU 间通信量均衡且 token 处理负载均衡。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B（8 experts/layer, top-2 routing, HuggingFace Hub 预训练权重）。
  - 数据集：WikiText-103、MiniPile、LAMBADA、enwik8（均为语言建模数据集）。
  - 并行配置：单节点 8 GPU = 4EP-2TP（4 expert parallel × 2 tensor parallel）；多节点 16 GPU = 4EP-4TP。
