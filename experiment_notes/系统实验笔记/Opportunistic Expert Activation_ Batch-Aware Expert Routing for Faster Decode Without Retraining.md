## Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：在 SGLang 推理框架中集成 OEA 路由算法，修改 MoE 层的 decode 阶段路由逻辑：仅在 decode 阶段使用 OEA（不在 prefill 阶段使用），根据 batch 内 token 的 router score 动态决定每个 token 的激活专家集合，最小化 batch 内唯一激活专家数 T。
  - 实验比较：(1) 不同 k0 配置（3/4/5/6/7）下的 MoE 层平均延迟（微秒），对比 vanilla top-8 routing；(2) 平均激活专家数 vs. vanilla；(3) Qwen3-30B 和 Qwen3-235B 两个模型规模下的延迟降低比例（39% 和 15%）；(4) 发现并修复 SGLang 的 CUDA Graph padding 问题：padding token 会激活额外专家导致反向性能损失，解决方案是捕获 CUDA Graph 到 batch size 16（覆盖所有实际 batch size，消除 padding）。

- 硬件平台是什么，配置是什么。
  - Qwen3-30B-A3B：单卡 NVIDIA H100 80GB，bfloat16。
  - Qwen3-235B-A22B：8×H100 80GB，单节点 HGX H100，NVSwitch 互联，tensor parallelism degree=8。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang（Zheng et al., 2024）。
  - 修改内容：
    1. 在 MoE 层的 decode 调用路径中插入 OEA 路由逻辑（替换默认的 top-k 路由）。
    2. 仅 decode 阶段使用 OEA，prefill 阶段保持原始 top-k 路由。
    3. 使用 `--max-running-requests` 限制最大 batch size 为 16（因 KV cache 限制无法到 32）。
    4. 捕获 CUDA Graph 到 batch size 16，避免 SGLang 对不足 batch size 的 padding 行为引入额外专家激活。
  - 未修改：MoE 权重本身、Grouped GEMM kernel、KV cache 管理、prefill 路由。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供独立开源仓库。OEA 路由通过修改 SGLang 的 MoE 层 forward 实现。
  - 全过程示例（Qwen3-30B，batch size=16，k0=5）：
    1. **输入**：SGLang 调度器将 16 个请求的 decode step token 组成 batch，传入 Qwen3-30B 的第 l 层 MoE 模块。
    2. **Router 评分**：对每个 token x_i，路由器 R 输出 N=128 个归一化分数 R(x_i)，排序得到 e_{i,1..128}。
    3. **OEA Phase 1**：每 token 取 top-5 专家作为 baseline → S_i_base，得到 S_base = union_i S_i_base（约 35 个唯一专家）。
    4. **OEA Phase 2**：每 token 遍历其 6-8 位排名的专家，若在 S_base 中则附加（piggybacking），最终每 token 仍 ≤8 个专家，但 T = |S_base| ≈ 35（vs. vanilla 约 48）。
    5. **权重加载**：仅加载 T=35 个专家的权重从 HBM→SRAM（vanilla 需加载 ~48 个），节省约 27% 的 memory fetch 开销。
    6. **计算**：Grouped GEMM 对 35 个专家的权重和对应 tokens 做批量矩阵乘法，输出 shape=(B, D)。
    7. **输出**：路由加权的专家输出求和，送入下一层 transformer block。
    8. **延迟效果**：MoE 层平均延迟从 175.7μs（vanilla）降至 136.0μs（k0=5），降低 23%；k0=3 时降至 106.8μs，降低 39%。
