## Neighbor Search Engine（邻居搜索引擎，NS-FPS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Neighbor Search Engine 是 NS-FPS（ISCA'26）ASIC 加速器的三大核心模块之一，负责把"迭代邻居搜索版 FPS"中每轮的距离更新流水化执行。由两个子模块组成：(1) **Morton Cube Searcher**——输入当前采样点 s_k(x_k,y_k,z_k) 与其搜索半径 d_k，计算搜索球对应的 Morton 码范围（含 cyclic 边界处理），周期性产生候选 Morton 码；(2) **Distance Update Module**——每收到一个 Morton 码，从 Morton 码页表系统取回该 cube 的点并更新其到采样集的距离。它是把算法层"Voronoi 部分更新 + Morton 球查询"（见算法pipeline库同名条目）落到硬件的关键数据通路，也是功耗/面积最大头（168.25mW 逻辑功耗的主体）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - Distance Update Module 四级流水（Fig.9）：(1) **Fetch Data**——按 Morton 码查页表系统取 cube 的点；cube 跨多个 Page Memory 项时沿 14-bit 链表追取全部点，取完立即处理下一个 Morton 码，空 cube 跳过避免 stall；(2) **Update Distance**——每 Page Memory 项内 16 个点并行计算到 s_k 的欧氏距离（fp32），只有更小值才更新 Distance Cache；(3) **Update Max-Value Cache1**——16-to-1 比较器取 16 点中最大更新距离写入 1K 深度 Max-Value Cache1 对应项；(4) **Record Address**——把被更新块的地址压入 Cache1 Change FIFO 供更高层缓存传播，1K-bit Record Table 保证每轮每地址只入队一次（去重）。流水线与访存重叠：32 项 Morton Cube Buffer 预取 cube，cube 派发进流水线后槽位立即释放预取下一个，隐藏 DRAM 延迟、保持连续吞吐。
  - 运转例子（120k 点帧一轮迭代）：Morton Cube Searcher 由 s_k、d_k 生成候选 cube 集 → 逐 cube 从 Page Memory 取 16 点 → 16 路并行算距更新距离缓存与 Cache1 局部最大 → 地址进 FIFO → Hierarchical Max Finder 消费 FIFO 完成全局 max。每轮只触及搜索球相关 cube，是"内存访问较 GPU 降 1700×"的直接来源。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Verilog RTL 描述，28nm Synopsys 综合（typical corner 25°C/0.9V），RTL 仿真验证 + DRAMsim3（DDR4-2400）建模片外行为。论文未开源 ASIC RTL（GitHub 仓库 hardware/ 目录无构建说明）；算法对应物开源为 CPU 版（C++17 + pybind11，https://github.com/satreeby/ns-fps/）。通用参考：并行邻居搜索加速器（QuickNN、ParallelNN、Tigris、CAMPER）多用并行距离计算器 + 排序/遍历单元，NS-FPS 特色是 Morton cube 流水化球查询 + 与 FPS 距离缓存/层次 max 缓存的融合。

涉及论文标题：
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds
