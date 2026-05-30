## Speculative Weight Preloading for Dynamic NNs (动态网络的投机权重复载)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Speculative Weight Preloading 是 Brainstorm 提出的 GPU 内存优化，针对动态网络中 branch 激活顺序不确定导致的 weight swapping 困难。传统 LLM 推理内存优化（如 SwapAdvisor, Capuchin）依赖静态执行顺序来 pipeline 预加载权重，但动态网络的 branch 激活在运行时才确定。Brainstorm 通过分析 Router profile 中 branch 的激活概率分布，投机性地预加载最可能被激活的 branch 权重到 GPU，预测成功则隐藏 weight loading 延迟；预测失败则 fallback 到 on-demand loading（开销与默认执行相当）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
全流程（以 DynamicRouting 为例，单 batch image inference）：
1. **Profile 分析**：从 Router profile 统计每个 Router 的 branch 选择概率。选择概率最高的 branch 标记为 speculative target。
2. **Memory 规划**：仅将 speculative target branch 的权重提前加载到 GPU（而非全部 branch），其余 branch 权重保持在 CPU/SSD。GPU memory 从 ~604.5MB（全加载）降至 ~340MB（speculative）或 ~298MB（on-demand）。
3. **执行流程**：
   - 前一层执行期间：异步 DMA 将 speculative target 权重从 CPU 传到 GPU
   - Router 执行：router_fn 决定实际 branch
   - 若命中：权重已在 GPU，直接计算（weight loading 延迟被隐藏）
   - 若未命中：从 CPU on-demand 加载正确 branch 权重（开销与 baseline 相同）
4. **GPU memory 效果**：Paper 报告 DynamicRouting GPU memory 减少 43.5%（vs 全加载的 604.5MB），同时加速推理 up to 1.97×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Brainstorm 通过 torch.fx 自动插入 weight preload/unload 算子到数据流图中。开发者无需手动管理 memory。适用条件：(1) Router 的 branch 选择有显著偏向性；(2) GPU memory 受限需要 weight swapping；(3) 单个 branch 的权重大小在可投机加载范围内。限制：分布漂移时预测准确率下降，需持续监控 profile。

涉及论文标题：
- Optimizing Dynamic Neural Networks with Brainstorm
