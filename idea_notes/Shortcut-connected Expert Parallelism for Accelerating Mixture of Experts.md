## Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts

- baseline方法是什么？
  Baseline方法：标准top-2 MoE专家并行。在分布式MoE训练/推理中，每个Transformer block的MLP被替换为top-2 gating MoE模块（每第二个block，即"Block-MoE"与"Block-MLP"交替放置）。执行流程严格串行：gate routing → input encode → All-to-All dispatch → expert computation → All-to-All combine → output decode。All-to-All通信可占MoE层总时间的约50%（多节点场景下因低带宽inter-node Ethernet甚至接近50%），成为主要瓶颈。现有优化：(1) Hierarchical All-to-All：利用层次化拓扑减少通信量；(2) Pipeline策略（如Tutel）：将tokens切分为fine-grained chunks，不同chunks的通信和计算在不同CUDA streams上交错执行实现部分重叠。但pipeline受限于首尾chunks（prologue/epilogue）的通信无法被计算隐藏——初始chunks仅通信无计算、末尾chunks仅计算无通信产生bubble。核心根本限制：**通信与计算存在顺序依赖**——expert computation必须在当前层representations就绪后才能启动，All-to-All通信必须在gate routing之后且expert computation之前。
  全栈执行例子（Baseline: Standard top-2 MoE + Pipelin, 8×A30-PCIe, training one iteration）：
  - **算法Pipeline层**：token x进入Block-MoE → gate network计算G(x) = Softmax(TopK(H(x), 2)) → 选择top-2 experts → 按公式MoE(x) = ΣG(x)_i E_i(x)计算 → 输出。MoE模块仅处理当前层representations，无跨层信息复用。
  - **系统框架层**：Tutel MoE framework，expert parallelism（每GPU分配不同expert），All-to-All token dispatch/combine基于NCCL实现，pipeline将tokens等分chunks交错通信与计算。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：pipeline strategy——输入tokens等分为M个micro-batches，每个micro-batch的dispatch和computation在不同CUDA streams上流水线执行。但第1个chunk的dispatch无法被任何computation隐藏（无前置计算），第M个chunk的combine同理。bubble time = T_disp/M + T_comb/M。算子调度严格遵循序列顺序：gate → encode → dispatch → compute → combine → decode，不可重排。
  - **硬件架构层**：8×A30-PCIe（PCIe互联，通信带宽低，All-to-All占MoE总时间~60%），8×A800-NVLink（NVLink高带宽，通信占~15%但仍有不可隐藏的prologue/epilogue）。多节点场景下inter-node Ethernet引入额外通信瓶颈。
  Baseline核心缺陷根因：MoE模块的输入仅来自当前层的attention输出，通信-计算依赖链条完全串行——必须先完成gate routing（依赖当前层表示）、再dispatch tokens、再expert computation。这种**顺序依赖**从根本上限制了任何overlap优化的天花板。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**ScMoE（Shortcut-connected MoE）**——通过引入shortcut连接将MoE模块的输入从仅当前层表示扩展为"前一层表示+当前层表示"，利用前一层表示进行gate-routed expert的选择和计算，从而将gating/通信/计算从当前层的串行链中解耦。结合自适应算子调度策略实现通信-计算最大重叠。
  核心设计对应：
  **(1) Shortcut-connected架构 → 解决"通信-计算顺序依赖"**：ScMoE使用top-1 MoE处理前一层（Block-MLP）的中间表示H_l^{MH}（通过shortcut），shared expert处理当前层（Block-MoE）的表示H_{l+1}^{MH}。由于前一层表示H_l^{MH}在Block-MLP的attention阶段就已确定，gating和token dispatch可以在Block-MLP的MLP计算和Block-MoE的attention计算期间提前执行——**gating不依赖当前层的任何计算**。这从根本上打破了baseline中"必须先等当前层attention完成才能开始gate routing"的顺序依赖。
  **(2) 自适应算子调度 → 解决"pipeline的prologue/epilogue bubble无法消除"**：在ScMoE解耦后，gate routing和encode可在MoE stream最早位置立即启动（不需要等待当前层计算），All-to-All dispatch可与shared expert stream的MultiHead attention + Shared Expert MLP重叠。Expert computation被插入shared expert stream的4个候选位置之一，通过最小化|T_comp_pre - T_disp| + |T_comp_post - T_comb|自适应选择。当通信时间≤重叠窗口（约50%总MoE时间）时实现100%通信隐藏——这是pipeline策略无法达到的，因为pipeline始终有prologue/epilogue bubble。
  **(3) 理论保证 → 解决"对模型质量的潜在担忧"**：Shortcut连接理论上保证梯度 ∂E/∂x_l = ∂E/∂x_L(1 + ∂/∂x_l ΣF_{W_i}(x_i))，加性分量确保信息直接反向传播至任意子层，避免梯度消失/爆炸。实验中表明相邻Transformer block的中间表示cosine相似度接近1.0（如Figure 10所示），因此用前一层表示替代当前层表示进行expert计算在模型质量上等价甚至更优。

  全栈执行例子（ScMoE Pos-2 + CG-1, 8×A30-PCIe, training one iteration）：
  - **算法Pipeline层**：同一对Block-MLP+Block-MoE中，Block-MLP先计算attention得到H_l^{MH} → 此时shortcut已将H_l^{MH}传给MoE stream → gate routing在Block-MLP MLP计算期间即可开始（因为H_l^{MH}已在Block-MLP attention后确定）→ 同时Block-MoE的attention以H_l^{MLP}为输入计算H_{l+1}^{MH} → shared expert直接处理H_{l+1}^{MH} → gate-routed expert处理H_l^{MH}（通过shortcut）→ 两路结果用CG-1系数组合：MoE(x) = Sigmoid(H_{l+1}^{MH}·W_CG) · SE(H_{l+1}^{MH}) + ΣG(H_l^{MH})_i E_i(H_l^{MH})。
  - **系统框架层**：基于Tutel+FaiRSeq实现。双CUDA stream架构——主stream执行Block-MoE attention + shared expert，MoE stream执行gating + All-to-All + expert computation。自适应调度器在CPU侧根据profiled T_disp/T_comb/T_comp选择最优expert computation位置K*。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：MoE stream中：gate + encode → Async All-to-All dispatch（与主stream的attention重叠） → expert computation（在自适应选择的K*位置，与主stream的shared expert重叠） → Async All-to-All combine（与主stream的后续计算重叠） → decode。8×A30-PCIe场景：重叠窗口 = T_Atten + T_SE + T_MLP ≈ 70% MoE时间，剩余30%可用pipeline augmentation进一步隐藏。8×A800-NVLink场景：通信仅占15%，完全被overlap_window覆盖 → 100%通信隐藏。
  - **硬件架构层**：同一A30-PCIe/A800-NVLink平台。关键变化：通信操作不再在硬件上串行暴露——因提前启动了gating和dispatch，通信时间被前置计算（Block-MLP MLP + Block-MoE attention）和后置计算（shared expert）双重吸收。对比baseline pipeline在A30-PCIe仍暴露显著通信bubble，ScMoE实现1.49×训练加速和1.82×推理加速。

  解决Baseline缺陷的方式总结：
  1. **针对"通信-计算顺序依赖"**：Shortcut连接将expert routing和dispatch的输入从"当前层表示"前移为"前一层表示"，打破串行依赖链——gating和通信可在当前层计算启动前就开始，从根本上扩大重叠窗口。
  2. **针对"pipeline的prologue/epilogue bubble"**：ScMoE的overlap window不依赖于数据切分（chunk），而是从时间线前端（Block-MLP计算期间）延伸到后端（Block-MoE shared expert期间），天然覆盖首尾，不存在pipeline的bubble限制。当通信≤overlap_window时实现100%隐藏。
  3. **理论+实验双重验证模型质量等价**：梯度传播理论保证训练稳定性；相邻层表示相似度分析（cos near 1.0）+ 多模型多任务实验证明ScMoE模型质量持平甚至超越baseline。为architecture-algorithm co-design提供可推广的范式。
