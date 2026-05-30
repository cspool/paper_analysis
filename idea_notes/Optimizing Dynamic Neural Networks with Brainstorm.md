## Optimizing Dynamic Neural Networks with Brainstorm

- baseline方法是什么？
  Baseline 是现有的 DL 框架（主要 PyTorch eager mode）执行动态神经网络。PyTorch 的 tensor-centric 编程模型只能表达 tensor 级别的静态数据流图（DFG），无法理解 sub-tensor 级别的 dynamism（如 token、patch、pixel 的动态路由）。具体来说：
  - Router 的路由逻辑（如 MoE top-k gate、patch-based super-resolution branching）用 Python 原生 control-flow + 数据搬运算子（如 einsum）实现，与计算逻辑耦合。
  - 编译器无法追踪 sub-tensor 级别的数据流：不知道 "token" 是什么、如何在不同 expert branch 间分发、跨层 expert 之间如何关联。
  - 无运行时 profile 收集能力：不知道 branch 激活的统计分布、token 在 expert 间的不均匀分配模式。
  - 全栈执行例子（SwitchTransformer with MoE, batch=8, 128 tokens/sentence, 256 experts, single A100 GPU）：
    - **算法层**：SwitchTransformer MoE layer，每 token 通过 softmax gating 路由到 top-1 expert，每个 expert 有 capacity=64 tokens。
    - **系统框架层**：PyTorch eager mode。Router 用 Python 实现（linear gate + argmax），tokens 通过 einsum 操作重新排列并按 expert 分组，各 expert FFN 串行执行（for loop 遍历 256 个 expert），每个 expert 计算其收到的 token subset。
    - **编译框架层**：PyTorch 仅做基础的 vertical operator fusion（如 Conv+BN+ReLU），无 sub-tensor 级别分析或优化能力。
    - **Kernel层**：每个 expert 的 FFN 为一组 GEMM kernel launch，256 个 expert 产生 256×k 次 kernel launch。对于每个 expert 仅收到少量 token 的常见情况，GPU CU 利用率极低。
    - **硬件架构层**：单 A100 GPU，108 SMs，每个 SM 可并行执行多 warp。串行 expert 执行时，仅少数 SM 被使用，其余空闲。
  - Baseline 缺陷：
    1. **细粒度 dynamism 不可追踪**：tensor-centric DFG 无法表达 token/patch 级别的数据流，无法收集 branch 负载、激活频率等关键 profile 信息。
    2. **串行执行 branch 导致 GPU 利用率低**：每个 branch 独立 launch kernel，GPU SM 闲置，kernel launch overhead 累积。
    3. **无负载感知优化**：all-to-all 通信对 uneven token distribution 做大量 padding 冗余传输；无法基于 expert 相关性优化多 GPU 放置。
    4. **Router 计算开销不可隐藏**：Router 涉及 CPU-GPU 同步和 control-flow，在 MSDNet/DynamicRouting 中占 44%~65% 延迟，无法跳过或预取。
    5. **动态 branch 无法做 weight preloading**：现有 memory swapping 方案（如 SwapAdvisor, Capuchin）依赖静态执行顺序，动态网络无法提前知道下一个激活的 branch。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Brainstorm 框架，通过 Cell 和 Router 两大核心抽象统一动态网络的表达，使 sub-tensor 级别的 dynamism 可追踪，进而基于运行时 profile 实施四项动态优化。

  1. **Cell 抽象**：让开发者标注 tensor 中 dynamism 发生的粒度和维度（如 token=(0), patch=(0,1)），编译器通过符号执行推导 Cell 在静态算子间的传播关系（三种类型：保持、重排、混合）。
  2. **Router 抽象**：统一的路由接口（router_fn 定义规则 → Router 负责高效执行），解耦控制流与数据流，使编译器无需理解路由逻辑本身，只需收集 Routes 的统计分布。Router 的执行后端是高效的 GPU kernel（Cell rearrangement + sparse communication）。
  3. **四项动态优化**（基于 Router 统计 profile）：
     - **Dynamic Horizontal Fusion**：根据 branch Cell 负载分布编译多 shape tuned kernel，运行时按实际负载选择最优 kernel 并水平融合执行。
     - **Profile-Guided Model Placement**：分析跨层 expert co-activation 相关性，重新排列 expert 以减少 inter-GPU 通信。
     - **Speculative Routing**：预测高概率 branch 跳过 router_fn，错误时 unroll。
     - **Speculative Weight Preloading**：预测高概率 branch 权重提前加载，减少 GPU memory 占用 43.5%。

  - 对比 baseline 全栈执行例子（SwitchTransformer MoE, batch=8, 128 tokens/sentence, 256 experts, brain A100 GPU）：
    - **算法层**：未修改模型算法。仅通过 12 行代码改动接入 Brainstorm：用 `brt.annotate_cell` 标注 token 粒度，用 `Router(router_fn)` 包装 top-k gate。
    - **系统框架层**：Brainstorm 的 torch.fx 优化 Pass 将 256 个串行 expert 替换为一个 fused horizontal kernel，内含多个 tuned kernel variant（如 8/32/64/128-token shape）。Router 的 GPU kernel 高效完成 token→expert 的 scatter-gather 操作，无 einsum 开销。
    - **编译框架层**：AOT 静态 Cell-level 符号执行分析 Self-Attention → MoE → Self-Attention 的跨层 Cell 依赖关系，得出 Self-Attention 的 Cross-Cell mixing 约束（所有 token 需聚合到同一 GPU）。JIT Profiler 收集 256 个 expert 的 token 负载分布（发现 P50/P90/P100 分别对应不同 percentile 的负载），确定 tuned kernel 的 shape 集合。通过 TVM 对每个 shape auto-tune。
    - **Kernel层**：Fused kernel 一次 GPU launch 并发执行所有激活的 expert（而非 256 次 launch），根据每个 expert 实际收到的 token 数（如 4/8/27/64）选择 nearest tuned kernel 并 minimal padding。对于仅收到 0 token 的 expert，直接跳过。对比 Tutel（BatchMatmul 方式，需将所有 expert 的 token 数 pad 到 max），Brainstorm 因 token 分布不均（图 2a）而大幅减少冗余计算和显存占用——Tutel 在 256 expert 时甚至 OOM。
    - **硬件架构层**：单 A100 GPU。Fused kernel 使 108 个 SM 全部参与计算，CU utilization 显著提高。运行时 overhead 仅 12.3μs（branch 少时），可忽略。
  - **关键设计应对 Baseline 缺陷**：
    - 缺陷1（不可追踪）→ Cell/Router 抽象：开发者显式标注 dynamism 粒度，编译器通过符号执行 + JIT profiling 获得 sub-tensor 级数据流全貌。
    - 缺陷2（GPU 利用率低）→ Dynamic Horizontal Fusion：用 profile 决定 tuned kernel shapes，运行时选择最小 padding kernel，并发执行激活的 branch。SwitchTransformer 加速 3.63× vs PyTorch，3.33× vs Tutel。
    - 缺陷3（无负载感知优化）→ Sparse All-to-All + Profile-Guided Placement：TaskMoE 减少 42~87% inter-GPU 通信，SwinV2-MoE 加速最高 5.04× vs DeepSpeed。
    - 缺陷4（Router 开销大）→ Speculative Routing：预测 90~95% 准确，DynamicRouting 加速 1.7×，MSDNet 加速 8.44×（combined with horizontal fusion: 11.7×）。
    - 缺陷5（无预加载能力）→ Speculative Weight Preloading：DynamicRouting 加速 1.97×，GPU 内存减少 43.5%。
