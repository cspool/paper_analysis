## LC framework（压缩组件搜索框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LC（Lossless Compression）framework（github.com/burtscher/LC-framework，Martin Burtscher 课题组开源）是一个跨平台数据压缩工具框架：提供多种通用压缩组件与预处理方法，支持把它们自由组合，并用"搜索"机制找出对给定数据最优的轻量压缩方案。其中 Reducer 是唯一用于缩短数据序列的组件类别，包含 HCLOG（分组位打包，16KB 块分 32 子块、按最小前导零数只存有效位）、RLE（游程编码）、RRE、RZE 等。ENEC 论文把它用作"搜索最优压缩算法"的基础：输入模型权重张量文件，独立调用改进的 LC 框架搜索最优配置（Observation 2：HCLOG 变体在多数模型上以 98%+ 胜出），并扩展了框架——加入支持不同子块数量的一组 HCLOG 压缩器变体（因为单 outlier 会迫使整子块位宽升高，可调子块数可缓解）。这属于"离线/在线算子优化"中的组件组合搜索工具，可类比编译器/框架层的自动调优（如 polyhedral/auto-tuning 搜索最优调度）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
ENEC 用 LC 框架做压缩方案搜索的流程：
```
# 输入：模型权重按文件切分（更细粒度分析）
for file in split(model_weights):
    best = LC_search(file)          # 框架内组合搜索：组件选择 + 参数（子块数等）
    record(best)                    # 汇总每文件最优配置（Table I：HCLOG 98%+ 胜出）
# 输出：确定"指数分组位打包"路线 → 启发 ENEC 的定长打包设计
```
Annotations：LC 搜索本质是"对数据特征做统计 → 选出轻量组件组合"，与编译器自动调优（对目标硬件搜最优 kernel 配置）同构——ENEC 论文借此做数据驱动的方法选型（Observation 1-5），而不是拍脑袋选压缩器。框架的 HCLOG 组件直接成为 ENEC 分组位打包的雏形。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：C++ 压缩组件库 + 搜索驱动；ENEC 在 Section II-C 扩展（可变子块 HCLOG 变体）后用于权重文件的最优组件搜索。使用：对特定数据类型（科学数据、模型权重等）自动找最佳轻量压缩组合，避免为每种数据手工调参；ENEC 用它得出"HCLOG 类分组位打包在模型权重上最优"的结论，进而设计出 NPU 友好的定长打包。局限：LC 组件的执行模型面向 CPU 通用向量，直接跑在 Ascend 上仍受分支/访存限制（ENEC 在硬件层做了再设计）。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
