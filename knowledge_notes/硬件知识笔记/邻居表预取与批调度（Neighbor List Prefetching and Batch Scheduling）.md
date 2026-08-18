## 邻居表预取与批调度（Neighbor List Prefetching and Batch Scheduling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
邻居表预取是 NASZIP 的硬件调度优化：ANNS 搜索在 hop 之间等待 CPU 合并全局优先队列存在空闲间隙，预取器在 hop 结束后即把每个查询当前最近节点的邻居表预取到 LNC，下一 hop 直接命中，隐藏 CPU 合并延迟。批调度把多个查询（batch）逐 hop 同步推进，hop 内距离计算顺序执行（轻量设计 FPU 有限），hop 间通过预取重叠。batch=16 为吞吐/延迟最佳折中（论文 Fig.22）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
预取+批调度运转流程（图 14，2 sub-channel、batch=2）：① 每 hop 结束，sub-channel 优先队列存各查询当前最近节点（如 q0→节点 2、q1→节点 5）；② 各 sub-channel 预取这些节点的邻居表到 LNC；③ 队列内容发往 CPU，CPU 合并排序进全局优先队列；④ CPU 返回下一 hop 的全局最近节点（如 2 和 5），sub-channel 在 LNC 中命中预取数据直接进入下一 hop。无预取基线在 CPU 合并期间空闲（图 14a 对比）。预取失败开销小：内容留在 LNC 仍可复用。预取命中率（Fig.21b）：上层命中率渐升、进入 base layer 时下降（上层邻居表与 base 层不同、缓存失效）；M（图密度）越大上层命中率越低、base 层越高；整体 >50%。batch 增大吞吐升（sub-channel 利用率与缓存复用更好）、延迟也升（batch 16→48 预取 miss 增多、缓存竞争加剧、Fig.22）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：预取引擎在 NDP 控制器中，hop 间隙把"每个查询当前最近点"的邻居表读入 LNC-D（地址由 NLT/LNC-T 提供）；调度由 CPU 与 NDP 协作——NDP 侧逐 hop 推进、CPU 侧合并全局队列。使用：批量 ANNS 查询场景，batch 大小 16 时预取最有效；与 LNC 缓存天然配合（预取即填充缓存）。效果：非距离侧延迟再降约 50%（相对 DaM+LNC 的 21.08%，Fig.25）。评估由 UniNDP 模拟器完成（prefetch_hit_rate.sh / cache_hit_rate.sh 脚本）。开源实现见 NasZip 仓库 simulate/。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
