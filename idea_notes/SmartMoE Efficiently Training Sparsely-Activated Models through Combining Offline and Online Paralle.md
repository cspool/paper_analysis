## SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

- baseline方法是什么？
  Baseline方法：(1) **FasterMoE（纯Expert Parallelism + 运行时调度）**：仅支持 expert parallelism，通过 smart scheduling 和 expert shadowing 等运行时优化缓解负载不均衡。但不支持混合并行（DP/TP/PP 组合），通信开销大（全量 All-to-All），所有 expert 参数分布在所有 GPU 上。对高 capacity factor 场景（严重负载不均衡）调度效果有限。(2) **Alpa（纯离线自动并行化）**：基于 JAX 的通用自动并行化系统，在训练前用 ILP 一次性搜索最优混合并行执行计划（inter-op pipeline + intra-op data/tensor parallelism），不感知动态负载，生成静态执行计划后训练全程不变。搜索耗时 825s（对 16 expert 模型），远超单 iteration 时间。(3) **DeepSpeed-MoE/Tutel（混合并行但无自动搜索）**：支持 EP+DP+TP 混合并行，但需要专家手动调参，且不考虑 expert placement 策略影响。(4) **BaGuaLu**：在超算规模组合 EP+DP，但 expert placement 固定按串行顺序放置，不优化负载均衡。
  全栈执行例子（Baseline: FasterMoE 纯 EP，4 GPUs，16 experts，capacity factor=+∞）：
  - **算法Pipeline层**：Gate: x → W_gate → Top-K → expert indices；Expert FFN: SwiGLU (gate_proj → SiLU ⊙ up_proj → down_proj)
  - **系统框架层**：FastMoE 框架（PyTorch），纯 EP——16 experts 按索引顺序平均分配到 4 GPUs（GPU_0: E0-E3, GPU_1: E4-E7, GPU_2: E8-E11, GPU_3: E12-E15），无 expert placement 优化。All-to-All dispatch → Expert computation → All-to-All combine。FasterMoE 的 shadowing/scheduling 仅缓解计算端负载不均，不改变通信拓扑。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：All-to-All 通信在全部 4 GPU 间执行（全互联），跨节点带宽瓶颈时通信延迟高。GShard gate capacity factor=+∞ 时无负载上限，GPU_0 上的 {E0,E1,E2,E3} 可能因 token 路由不均而严重超载，其他 GPU 空闲。FasterMoE 通过 smart scheduling (token 级动态影子 expert 分配) 部分缓解但仅限计算端，通信量不变。
  - **硬件架构层**：GPU 集群（V100 PCIe, V100 SXM, A100 SXM），节点内 NVLink + 节点间 InfiniBand（50-200Gb/s）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **SmartMoE** 通过 "离线池构建 + 在线自适应并行化" 两阶段设计解决 baseline 缺陷：

  **(1) Enlarged Space of Hybrid Parallelism（扩大混合并行空间）→ 解决纯EP通信瓶颈**
  Baseline（FasterMoE）仅支持纯 EP，通信为全局 All-to-All。SmartMoE 通过 expert slot 抽象统一表达 DP/TP/PP/EP 的任意组合，支持混合并行（如节点内 TP+EP、节点间 DP）缩小通信范围至更少 GPU。同时引入 **Expert Placement** 新维度——expert 到 expert slot 的映射顺序从根本上影响负载均衡。如 4 GPU 16 expert 场景，按 token 负载交错映射 vs 按索引顺序映射，前者可使各 GPU 负载方差从 102% 降至 <1%。

  **(2) Workload-Aware Performance Modeling（负载感知性能建模）→ 解决无法离线预测MoE性能**
  Baseline（Alpa）假设 workload 均匀（data-insensitive），在 MoE 动态负载下预测不准。SmartMoE 利用 gating network 语义估算负载上界——对 GShard gate 用 capacity factor 计算 max_tokens_per_expert；对 topology-aware gate 按算法层次计算最大通信量。无需实际训练数据即可准确预测，R² > 0.5 for all configurations，支持离线阶段穷举搜索最优池。

  **(3) Two-Stage Auto-Parallelization（两阶段自动并行化）→ 解决离线搜索慢+在线需快速决策**
  Baseline（Alpa）的 ILP 搜索需 825s，无法在线执行。SmartMoE 分解为离线+在线两阶段：
  - **Offline**：构建"固定混合并行策略+可变 expert placement"的 pool，池内候选执行计划有相同 expert slot 配置（切换时无内存分配/释放，仅参数交换）
  - **Online**：三种轻量级算法——Greedy (O(NE), <1ms)、DP (O(N×4^E), 最优)、Hybrid (Greedy → virtual devices + DP → physical devices, 可调 M 权衡精度/速度)。利用 expert selection 的时间局部性每 10 iterations 搜索一次

  **(4) Adaptive Runtime Switching（自适应运行时切换）→ 解决固定执行计划效率退化**
  Baseline 使用训练前确定的静态执行计划，在动态负载下效率逐渐下降。SmartMoE 设置切换阈值过滤微小改进（避免通信开销超过计算收益），利用相邻 iteration 的 gating 分布局部性间歇性搜索。搜索在 CPU 侧执行（<1ms），与 GPU 计算并行不占关键路径。

  全栈执行例子（SmartMoE，4 GPUs×4 expert slots，GShard gate capacity=2.4）：
  - **算法Pipeline层**：同 baseline——Gate: x → W_gate → Top-K softmax → expert indices → Expert FFN。不同之处：expert 到 GPU 的 placement 由 SmartMoE 在训练过程中动态决定（ExpertPlacementHybrid(E=16, N=4, C[16])），不固定于模型初始化时。
  - **系统框架层**：SmartMoE 框架（基于 FastMoE+PyTorch），支持 expert slot 抽象下的 DP/TP/PP/EP 任意组合。Offline 阶段 pool search 遍历候选策略空间（如 DP=2×TP=2×(PP=1 inside node) × EP=16 across 4 GPUs），用 workload-aware 性能模型评估。Online 阶段每 10 iterations 触发 placement 搜索，搜索结果通过 NCCL All-to-All 交换 expert 参数（~20ms for 16 experts），然后各 GPU 按新 placement 执行 expert FFN。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：
    Offline: 穷举 + 性能模型 → 输出 pool（固定 DP/TP/PP，可变 expert placement）
    Online per 10 iters (CPU执行):
    输入: expert_token_counts C[16] from gate history
    Hybrid Placement: 
      Step1 Greedy to M virtual devices → 粗粒度均衡
      Step2 DP within each virtual device → 细粒度最优
    → 新 placement P': GPU_0={E5(520),E7(240),E12(245),E1(240)}
                       GPU_1={E0(512),E9(250),E15(248),E2(242)}
                       GPU_2={E3(508),E11(252),E6(249),E10(238)}
                       GPU_3={E4(505),E13(248),E14(247),E8(245)}
    Imbalance < 1%
    切换时 NCCL All-to-All 交换 expert 参数（仅被移动的 expert），Compute 在新 placement 下执行。
  - **硬件架构层**：3 种 GPU 集群——blinky (8×V100 PCIe/50Gb IB), pinky (4×V100 SXM/100Gb IB), inky (8×A100 SXM/200Gb IB)

  关键结果：vs FasterMoE baseline 最高 1.88× speedup（A100 cluster），平均 1.53×（A100）/1.17×（V100 SXM）/1.14×（V100 PCIe）
