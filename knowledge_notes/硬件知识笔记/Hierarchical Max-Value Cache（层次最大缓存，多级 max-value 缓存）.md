## Hierarchical Max-Value Cache（层次最大缓存，多级 max-value 缓存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hierarchical Max-Value Cache 是 NS-FPS（ISCA'26）加速器中加速"找全局最远点（argmax）"的片上缓存结构。核心思想：FPS 每轮都要从距离缓存 T（R^N，按 Morton 序排列）中找出最大者；传统做法全量扫描 O(N)，NS-FPS 把 T 按每 16 个连续项一组块，每层缓存存下一层各块的局部最大值（16:1 压缩），形成多级 max 候选金字塔。算法版每级 16:1 压缩（Fig.6）；ASIC 实现为四级片上缓存，深度 1K/64/4/1。配合 Cache Change FIFO，每轮只刷新被距离更新触及的块，全局最大值自顶向下常数级比较得到——这是把"全量 argmax"降为"近对数/常数级 max 选择"的硬件机制，也是消融中 T2 技术（相对 T1 单独增益 2.5–4.2×）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
  - 硬件执行分两阶段（Fig.10）：(1) **Top-down Update（更新）**——Cache Change FIFO 中的块地址逐级触发更新：每块 16-to-1 比较器选出局部最大及其索引，向上传播到父块，直至 4 级顶端 Max-Value Cache4 得到全局最大；(2) **Bottom-up Lookup（查坐标）**——从 Cache4 的全局最大出发自顶向下逐级沿记录索引回溯，恢复出具体点下标，再从 Page Memory 取该点坐标加入采样集。1K 深度 Cache1 对应 1K 块（每块 16 点，覆盖 16K 点距离缓存）；16:1 压缩比与 Page Memory 16 点/块、16 路 PE 阵列一致，三者统一为 16 的粒度。
  - 关键收益：部分更新方案使只有被更新的块需要重算局部 max（缓存传播限定在更新块），避免全量扫描；相比 GPU 实现还需在全局内存上做层级 argmax 同步（跨 block 通信），ASIC 的片上层次缓存无此开销——这是 NS-FPS-ASIC 较 NS-FPS-GPU 内存流量再降 400× 的原因之一。120k 点帧最远点选择阶段耗时 1.65ms（占端到端 ~13%）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：全部层级在片上 SRAM（1K/64/4/1 深度），配合 Cache Change FIFO 与 1K-bit Record Table（去重）；Change FIFO 也用于 CPU 版的层次更新（Buffer_updated 逐层传播）。与 Segment Tree（线段树，区间 max 查询 O(log N)）思路同族，但固定 16:1 扇出更硬件友好。论文未开源 RTL；CPU 开源版（https://github.com/satreeby/ns-fps/，C++17）实现了 16-1 tree 结构的全局最远点查询。敏感性：块大小 8/16/32/64/128 中 16 点最优——匹配 DDR4 64B burst、配满 16 路 PE、保持统一 16:1 压缩比。

涉及论文标题：
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds
