## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- baseline方法是什么？
  在MoE推理优化领域，没有单一的baseline。但在综述的分类框架中，隐含的baseline是：
  **(1) 模型级baseline**：标准MoE架构使用固定top-K路由（如top-2 gating），所有expert以FP16全精度存储，无剪枝、无量化、无蒸馏。
  **(2) 系统级baseline**：
  - Expert Parallelism baseline：DeepSpeed-MoE和FasterMoE（标准all-to-all通信，无特殊调度优化）
  - Expert Offloading baseline：Mixtral-Offloading（按层加载expert，简单LRU缓存，无预取）
  **(3) 硬件级baseline**：传统GPU架构针对稠密计算优化，缺乏对MoE稀疏激活和动态expert调度的硬件支持。

  **Baseline全栈执行例子（以Mixtral-8x7B推理一个token为例）**：
  - 算法层：top-2固定gating，2/8 expert激活 → FP16 FFN计算
  - 系统框架层：无offloading时全部expert常驻GPU显存；或简单按层加载expert（如Mixtral-Offloading）→ LRU缓存，无预取
  - 编译框架层：论文未明确说明（使用标准PyTorch/Transformers执行）
  - Kernel调度层：标准cuBLAS GEMM kernel，无专为稀疏expert优化的kernel
  - 硬件架构层：标准NVIDIA GPU SM架构，无NDP/PIM/FPGA加速

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本论文是**综述**，本身不提出新方法，而是建立三层分类框架（模型级-系统级-硬件级），系统性地组织和对比现有方法，并识别关键挑战和未来方向。

  **综述对baseline缺陷的诊断与分类解决路径**：

  | Baseline缺陷 | 综述识别的方法方向 | 代表性工作 |
  |---|---|---|
  | 固定top-K浪费计算 | 动态门控（根据token复杂度自适应） | DynMoE、XMoE、AdapMoE |
  | 全精度expert占用过多显存 | 量化压缩（INT4/INT2/INT1） | QMoE、MC-MoE、MoQE |
  | 冗余expert浪费参数 | 剪枝/合并expert | TSEP、MoE-Pruner、MC-SMoE |
  | All-to-All通信瓶颈 | 分层通信、数据压缩、减少通信次数 | Tutel、ExFlow、Janus |
  | Expert加载延迟（offloading场景） | 预取+智能缓存+低精度加载 | HOBBIT、ProMoE、ExpertFlow |
  | 负载不均衡导致GPU闲置 | 性能建模+greedy搜索expert放置 | Prophet、FlexMoE、Lazarus |
  | GPU硬件对稀疏计算低效 | NDP/PIM/FPGA专用加速 | MoNDE、Duplex、FLAME |

  **综述方法论全栈执行例子（以优化后的MoE推理一个token为例）**：
  - **算法层**：动态门控根据输入复杂度自适应选择expert数量（非固定top-2）→ 量化后的INT4 expert权重进行低精度FFN计算
  - **系统框架层**：Expert cache中保留高频expert（LRU+LFU+LHU混合策略）→ 基于当前gate输出预取下层expert（跨层预测准确率~90%）→ CPU辅助处理低精度cold expert → GPU和通信流水线重叠
  - **编译框架层**：论文未明确说明
  - **Kernel调度层**：专用CUDA kernel处理量化权重的反量化+浮点计算（如MoE-CSP的4-bit kernel）→ FPGA上双缓冲expert权重加载（如FLAME）
  - **硬件架构层**：Hot expert在GPU执行 + Cold expert通过CXL发送到NDP核在LPDDR内执行（如MoNDE的Activation Movement模式）
