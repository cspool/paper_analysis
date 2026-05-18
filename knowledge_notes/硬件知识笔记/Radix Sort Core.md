## Radix Sort Core

术语是什么？
Radix Sort Core 是 VAR-Turbo Accelerator 中专用于大 K TopK 选择的硬件排序核心。它将 TopK 问题从通用排序（如 Bitonic Sort + Merge Sort，在大 K 上需反复读写重排）转换为固定 4 阶段流水线：CountBin（按 radix digit 分桶计数）→PrefixSum（前缀和计算定位含第 K 大元素的 bin）→SelectBin（定位 candidate bin）→Filter（从 candidate bin 筛选最终 TopK 元素）。额外加入 Locality-aware Scheduling 利用 confidence map 的空间偏斜优先处理高置信区域。

从硬件架构角度拆解术语：
Radix Sort Core 在 VAR-Turbo accelerator 中的运转流程（以 PD 阶段 N=4096, K=1936 为例）：
1. 输入端 TP（Parallel-to-Sequential Converter）将并行 confidence vector 转为串行流
2. CountBin：按 radix digit 宽度将 4096 个 confidence 值分到 bins 并计数
3. PrefixSum：计算所有 bins 的前缀和，定位第 1936 大元素所在的 bin（candidate bin）
4. SelectBin：精确定位 candidate bin 内第 K 大的确切值
5. Filter：从 candidate bin 中筛选出 TopK 1936 个元素的 indices
6. Locality-aware Scheduling：维护 history table 标记已解码区域→PE 分组并行在不同空间区域独立执行 Radix Select→优先调度靠近已解码 token 的高置信区域
在 DB 阶段类似流程处理 per-token importance scores。Radix Sort Core 仅占 4.9% 面积和 6.3% 功耗，但解决了关键瓶颈——传统方案中 TopK 仅占 3.5% 操作数却占 20.9% 延迟。

术语一般如何实现？如何使用？
Radix Sort Core 以 SystemVerilog RTL 实现，集成在 VAR-Turbo accelerator 中（TSMC 28nm 综合）。Radix Select 相比 Bitonic Sort 优势在于利用 radix 分解消除全局比较和重排——TopK 不需要完全排序，只需定位第 K 大元素的边界并筛选。Locality-aware Scheduling 进一步利用图像 token 的 2D 空间结构（而非一维序列）加速处理。该设计方法可推广到其他需大 K TopK 的加速器。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

