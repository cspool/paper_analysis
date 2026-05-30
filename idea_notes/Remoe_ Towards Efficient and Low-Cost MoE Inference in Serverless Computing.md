## Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing

- baseline方法是什么？
  - Baseline 有四类：(1) **CPU-only 部署**：整个 MoE 模型部署在 CPU 上，所有 expert 权重常驻 CPU 内存，无 expert offloading，但推理延迟极高；(2) **GPU-only 部署**：整个 MoE 模型部署在 GPU 上，所有 expert 权重常驻 GPU 显存，低频 expert 占用昂贵 GPU 内存造成成本浪费；(3) **Expert Offloading（Fetch/MIX）**：Fetch 假设理想 expert offloading（无 misprediction、无 offloading/reloading 时间），所有 expert 缓存于 CPU 内存并按需交换到 GPU；MIX 将 expert 模块部署在 CPU、非 expert 模块部署在 GPU，但 CPU/GPU 内存需足以缓存所有模块。两者均需持续分配大块 CPU 内存持有 inactive experts；(4) **Per-expert Serverless Function [14]**：将每个 expert 作为独立 serverless function，对 Deepseek-V3 (256 experts × 61 layers) 等现代 MoE 完全不可行。
  - 全栈执行例子（Baseline: MIX with GPU+CPU, Deepseek-v2-lite on A100, 500 token input）：
    - 算法层：无 expert 激活预测，所有 64 experts/层 权重常驻 CPU 内存
    - 系统框架层：Kubernetes 调度单 Pod（GPU + CPU），模型以单体 serverless function 部署
    - 编译框架层：论文未明确说明
    - kernel调度层：论文未明确说明
    - 硬件层：GPU 执行 Attention 和 Shared Experts（A100 Tensor Cores），token embedding 通过 PCIe 传输到 CPU，CPU 执行 top-6 experts 的 FFN → 结果回传 GPU → 重复 27 层 → 持续占用 CPU 内存持有所有 64×27=1728 个 expert 权重

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**Remoe** 异构 serverless MoE 推理系统，通过五个关键设计解决 baseline 缺陷：
    (1) **SPS 算法**解决 "expert 激活无法预知导致无法预分配资源" 缺陷：利用 prompt 语义相似度（SCS）+ 多叉聚类树实现 prompt 级 expert 激活预测，将预测提前到请求到达时完成，避免 token-by-token 在线预测在 serverless 环境下导致的冷启动开销；(2) **异构架构**解决 "所有 expert 常驻内存造成浪费" 缺陷：非 expert 模块在 GPU，expert 模块在 CPU，仅高频 "local experts" 与主模型同容器，低频 "remote experts" 部署为独立 serverless function（按需冷启动，pay-per-use），消除低频 expert 的持续内存占用；(3) **MMP 算法**解决 "serverless 冷启动时无法保证 SLO" 缺陷：基于霍夫丁不等式证明最坏情况 expert 负载上界（Theorem 1: per-expert token ≤ √(3n)/2 + n/K_l），在最坏情况下仍保证 TTFT/TPOT；(4) **Lagrangian 对偶优化**解决 "remote expert 内存规格选择是 NP-hard" 的缺陷：将离散内存规格连续化，构造指数衰减拟合函数 T̃=θ1·exp(-θ2·ỹ)+θ3，分析凹凸性（Theorem 2），利用 Slater 条件和 KKT 条件（Lemma 1, Theorem 3）找到全局最优解；(5) **LPT 多 replica 划分**解决 "多 replica 下 expert 子集划分导致负载不均" 缺陷：建模为 Multiway Number Partitioning 问题，LPT 算法 O(n log n) 复杂度，近似比 (4/3 - 1/3z_l)。
  - 全栈执行例子（Method: Remoe, Deepseek-v2-lite on A100, 500 token input, 200 output tokens）：
    - 算法层：请求到达 → SPS 预测 expert 激活矩阵 S_pred [27, 64] → utility score u_{l,k} 排序 → b=15% 的 experts 为 remote（约 10/64 per layer）
    - 系统框架层：MMP 用 Theorem 1 上界计算主模型最小内存 w_v → Kubernetes 调度主模型 Pod（GPU+CPU）→ Lagrangian 优化确定 remote experts 每层内存 ỹ_l → LPT 划分 remote expert 集到 z_l 个 replica → Kubernetes 创建 remote expert Pods（CPU-only）→ 冷启动并行（主模型启动 ∥ remote experts 冷启动 ∥ 优化计算）
    - 编译框架层：论文未明确说明
    - kernel调度层：C++ LibTorch 实现 expert FFN 计算，gRPC 传输 token embedding（token size 10KB < 6MB payload limit，无需中间存储）
    - 硬件层：GPU (A100) 执行 Attention + Shared Experts → token embedding 通过 gRPC/PCEe 发送到 remote expert Pods → CPU 并行执行 local experts (high-frequency) + remote experts (low-frequency, separate Pods) → 合并输出 → 27 层迭代 → 仅主模型 + 高频 experts 常驻内存，低频 experts 按需付费
