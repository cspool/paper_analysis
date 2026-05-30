## Optimizing Dynamic Neural Networks with Brainstorm

- 属于编译框架的实现是什么？实验比较什么？
  实现是 Brainstorm 框架，在 PyTorch 之上引入 Cell（数据抽象，标注 dynamism 发生的子张量粒度，如 token/patch/pixel）和 Router（统一接口，表达 Cell 如何在多个分支间动态分发）两大核心抽象。Brainstorm 编译器执行：(1) 静态 Cell-level 数据流分析——通过符号执行（ahead-of-time compiling）推导 Cell 在静态算子（如 MatMul、Self-Attention）中的跨层传输关系，区分保持顺序、重排序、混合等三类数据流；(2) 动态 Cell-level 数据流分析——JIT Profiler 在运行时收集 Router 的路由决策统计分布；(3) 基于 torch.fx 的优化 Pass，利用 Router 统计 profile 对数据流图进行变换（融合、重排、插入 preload/unload 算子等），并通过 TVM 进行 kernel auto-tuning。提出四项动态优化：(a) Dynamic Horizontal Fusion——根据 branch 的 Cell 负载分布统计，将多个分支水平融合为一个 fused kernel，编译多个不同 shape 的 tuned kernel 以最小化 padding；(b) Profile-Guided Model Placement——基于跨 MoE layer 的 expert 激活相关性统计，重新排列 expert 在不同 GPU 上的放置以减少 all-to-all 通信量；(c) Speculative Routing——基于 Router 历史决策的偏向分布预测最可能激活的分支，跳过 router_fn 计算，若预测错误则 unroll 重执行；(d) Speculative Weight Preloading——基于 branch 激活分布预测最可能需要的权重并提前加载到 GPU，若预测错误则 fallback 到 on-demand loading。实验比较 PyTorch（eager mode，串行执行 branch）、Tutel（MoE 专用优化，BatchMatmul 并发执行 expert）、DeepSpeed-MoE 等 baseline，在 SwitchTransformer、TaskMoE、SwinV2-MoE、LiveSR、DynamicRouting、MSDNet 六种动态网络上的延迟/吞吐提升，以及 micro-benchmark 中各优化的独立效果。

- 硬件平台是什么，配置是什么。
  单 GPU 实验：AMD EPYC 7V13 CPU + 1× NVIDIA A100 80GB，CUDA 11.3 + cuDNN 8.6。多 GPU 实验：Intel Xeon E5-2690 v4 + 8× NVIDIA V100 32GB（NVLink），CUDA 11.3 + cuDNN 8.2。用 NCCL 作为通信后端。

- 开源编译框架是什么。修改了什么。
  开源框架：PyTorch（https://github.com/pytorch/pytorch）。Brainstorm 代码（13,000 LOC）在 https://github.com/Raphael-Hao/brainstorm（OSDI 2023 artifact: osdi2023ae 分支）。修改/扩展内容：
  1. **新增 Cell 抽象**（brt.annotate_cell(tensor, dims, shape)）：让开发者标注 tensor 中 dynamism 发生的维度和粒度（如 dims=(0), shape=(1,768) 表示第一维为 token 粒度）。
  2. **新增 Router 抽象**（class Router, router_fn）：统一接口接收 Cell-annotated tensor，返回 Routes（每个 Cell 应去往的 branch ID），收集到 JIT Profiler 用于统计分析。
  3. **编译器核心（~3,000 LOC Python）**：静态 Cell-level 符号执行 + 动态 JIT Profiler 收集 Router 决策分布。
  4. **优化 Pass（~3,000 LOC Python + 1,500 LOC torch.fx 自动变换）**：实现四项动态优化（Horizontal Fusion、Placement、Speculative Routing、Speculative Preloading）的图变换逻辑。
  5. **Custom GPU kernel（~3,000 LOC C++/CUDA）**：实现高效 Cell 重排列 kernel、sparse all-to-all 通信原语（点对点通信集合，避免 padding 冗余传输）、动态水平融合中多 shape tuned kernel 的调度。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  代码全开源在 GitHub（https://github.com/Raphael-Hao/brainstorm），OSDI 2023 artifact evaluation badge。提供 Docker 镜像预配置实验环境。

  **使用方式——以 SwitchTransformer 为例（仅 12 行代码修改）**：
  ```
  import brainstorm as brt
  # 1. 标注 Cell：token 位于第 0 维，每个 token 是 768 维向量
  x = brt.annotate_cell(input_tensor, dims=(0), shape=(1, 768))
  # 2. 定义 Router（top-k gating）
  class TopKRouter:
      def router_fn(self, x, k=2):
          scores = self.gate(x)  # (num_tokens, num_experts)
          _, indices = torch.topk(scores, k, dim=-1)
          return indices  # Routes: (num_tokens, k)
  router = Router(router_fn)
  # 3. 前向传播：Router 自动分发 Cell 到对应 expert branch
  outputs, routes = router(x)
  # routes 被 JIT Profiler 异步收集到 profile 文件
  ```

  **编译框架全流程（从输入到输出）**：

  1. **AOT 编译阶段**：Brainstorm 解析模型代码，为每个标注了 Cell 的 tensor 创建符号版本（每个 Cell 用不同符号标记）。通过算子张量表达式推导符号计算：例如 MatMul 中，若左矩阵每行是 Cell，右矩阵是常量权重，则输出每行仍保持原始 Cell 符号（第一类——保持顺序）；若右矩阵也是 Cell-annotated tensor，则输出每个 Cell 混合了输入所有 Cell（第三类——Cross-Cell mixing，如 Self-Attention）。记录静态 Cell-level 数据流约束：如 Self-Attention 中所有 token 互相依赖，要求所有 token 聚合到同一 GPU。

  2. **Profile 收集阶段（JIT）**：模型在训练集上推理，每个 Router 的 router_fn 决策（Routes tensor）被写入 buffer。独立线程异步将 buffer 流式写入 profile 文件。决策数据量极小（控制信号），开销 <1.0%。

  3. **动态优化阶段（offline）**：从 Router profile 提取每个 branch 的 Cell 负载分布（P50/P90/P100 percentile），对每种 shape 用 TVM auto-tune 最优 kernel，将所有 branch 的 kernel 融合为一个 fused kernel。对 Placement，构建 cross-layer expert 激活概率矩阵，求解最小化 inter-GPU 通信的放置方案。对 Speculative，若 branch 选择概率 > 阈值则编译为跳过 router_fn 直接启动该 branch 的图，同时保留 check+unroll 路径。

  4. **运行时执行**：torch.fx 追踪优化后的数据流图。对于 Horizontal Fusion：fused kernel 一次 GPU launch 并发执行所有激活分支，根据实际 Cell 数选择 nearest tuned kernel 并 padding。对于 Placement：Router 的稀疏通信原语仅传输实际需要跨 GPU 的 Cell，无 all-to-all padding。对于 Speculative：若预测命中则 Router 延迟被隐藏；若预测错误则 unroll 到正确 branch（开销与默认执行相当，is negligible per micro-benchmark）。

  **关键作用**：通过 Cell/Router 抽象让动态 NNs 的 sub-tensor 级别 dynamism 可追踪，从而将传统编译器 PGO（Profile-Guided Optimization）思想引入 DL 框架，开辟了基于运行时 dynamism 分布来特化模型执行的全新优化空间。平均加速 3.29×，最高 11.7×，或减少 GPU 内存 42%。
