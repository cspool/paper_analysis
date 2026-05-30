## NetMoE: Accelerating MoE Training through Dynamic Sample Placement

- 属于Serving调度的实现是什么？实验比较什么？
  - 属于训练调度优化的实现：NetMoE 在 MoE 分布式训练中动态调整训练样本（training samples）在各 GPU 上的放置位置，将跨节点（inter-node）All-to-All 通信转化为节点内（intra-node）通信，从而加速训练。
  - 实验比较：NetMoE vs FastMoE（无 placement 调整的基线）、FasterMoE（动态 expert placement + 通信-计算重叠）、SmartMoE（expert placement + 负载均衡）。端到端加速比最高 1.67×（vs FastMoE）、1.37×（vs FasterMoE）、1.33×（vs SmartMoE）。还比较了 KM 算法 vs PuLP 的求解时间，以及 All-to-All 通信的理论加速 vs 实际加速。

- 硬件平台是什么，配置是什么。
  - 4 节点集群，每节点 8 张 NVIDIA A800-SXM4-40GB GPU（共 32 GPUs）。
  - 节点内：NVLink 互联，带宽 400 GB/s。
  - 节点间：InfiniBand 互联，带宽 100 GB/s。
  - 设备内（intra-device）：内存拷贝，约 2 TB/s（不计入通信建模）。
  - 默认每节点 8 GPUs，附录中也测试了每节点 2/4 GPUs 的配置。

- 开源Serving框架是什么。修改了什么。
  - 基于 **PyTorch** 实现，自定义操作用 C++ 和 CUDA 编写。与 FastMoE / FasterMoE / SmartMoE 对比（均为开源 MoE 训练框架：FastMoE https://github.com/laekov/fastmoe、FasterMoE https://github.com/laekov/FasterMoE、SmartMoE 论文未明确说明开源链接）。
  - 主要修改/新增：
    1. **动态样本放置求解器（Dynamic Sample Placement Solver）**：将 All-to-All 通信建模为 α-β 通信模型，formulate 为组合优化问题（整数规划），再拆分为两个阶段（Stage 1: 全局跨节点优化，Stage 2: 节点内优化），通过将 ILP 转化为加权二分图匹配问题，使用 Kuhn-Munkres (KM) 算法在多项式时间 O(I³) 内求解，其中 I 为全局 batch size。
    2. **Expert Residual Inlining**：将残差连接内联到 expert 计算中（在 scatter 之后、gather 之前执行加法），保证样本放置调整后计算正确性不变（详见论文 Appendix A.1，图 8）。
    3. **CPU Offloading**：KM 算法在 CPU 上执行（因 GPU 不适合并行化该算法），求解过程通过后台线程与 All-to-All scatter + expert 计算重叠，实现零额外开销。
    4. **下一层路由预测**：将当前层的输入传入下一层 router，提前获取下一层路由结果供求解器使用。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源**：论文未明确说明开源链接，在 ICLR 2025 页面和 web 搜索中未找到公开代码仓库。
  - **全流程使用例子（MoE-GPT-S, 4 nodes × 8 A800 GPUs, I=32 samples/iteration）**：
    1. **训练配置**：GPT-2 backbone MoE 模型，序列长度 S=1024，hidden dim H=768，E=2J=16 experts（每 GPU 1 expert），K=2（每个 token 选 top-2 experts），全局 batch I=32（每 GPU 1 sample）。
    2. **前向传播（MoE Layer l）**：
       - **Gating**：每个 GPU 上的 tokens 经过 gating network，得到 routing 结果 `route ∈ N^{I×L×K}`。
       - **计算通信矩阵 num**：统计每个 sample i 需要发给每个 expert e 的 token 数量 `num_{i,e}`（Eq. 2），在 GPU 上计算后传输到 CPU。
       - **Stage 1 求解（跨节点）**：CPU 后台线程构建二分图——左侧 P 为 I 个 samples，右侧 Q 为 N 个 nodes（每个 node 可容纳 B=I/N samples，通过复制 B 次使 |Q|=I）。边权重 `W_{i,n} = c_{i,n}^{(l,gather)} + c_{i,n}^{(l+1,scatter)}`（Eq. 8 计算跨节点通信量）。KM 算法求最小权完美匹配，得到每个 sample 的目标 node。
       - **Stage 2 求解（节点内）**：对每个 node n 独立构建二分图（左侧为该 node 分配到 I/N 个 samples，右侧为该 node 上 J/N 个 GPUs，每 GPU 复制 I/J 次），KM 算法求最小权匹配，得到每个 sample 的目标 GPU。
       - **All-to-All Scatter**：各 GPU 按 routing 结果将 tokens 发送到对应 expert 所在 GPU。此操作使用当前（优化前）的 sample placement。
       - **Expert Computation**：各 GPU 上的 experts 对收到的 tokens 执行 FFN 计算。同时 CPU 后台完成求解。
       - **Expert Residual Inlining**：output = input + expert_output（在 scatter 之后直接执行，而非等 gather 之后）。
       - **All-to-All Gather（使用优化后的 SmpDev）**：token 不再按原位置返回，而是按新求解的 sample placement 目标 GPU 发送——完成动态样本放置，无需额外通信。
    3. **效果**：以 2 nodes/16 GPUs 配置为例，MoE-GPT-S 的 inter-node 通信量从 191.07 MB 降至 116.37 MB（↓39.10%），约 43.66% 的 samples 跨节点交换，91.39% 的 samples 被调整位置。KM 求解时间（I/J=4 时 0.48ms）远小于 scatter+computation 时间（7.13ms），完全被隐藏。
