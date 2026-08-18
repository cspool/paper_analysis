## Far-Node Pruning（远节点剪枝 / 远节点剔除）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- far-node pruning 是 GauTracer 提出的 BVH 遍历优化：利用"当前最大命中距离"作为几何屏障（barrier）裁剪后续节点访问。Gaussian ray tracing 中由于 Hit Gauss Buffer 容量有限，某些命中节点访问后被丢弃（buffer 已满/提前终止），产生冗余遍历。机制：引入 barrier flag，一次 traceRayEXT round 内 Hit Buffer 一旦满即置位，把当前最大命中距离（buffer 头 = 堆根）作为剪枝阈值，阈值随更近命中的发现持续更新；ray-box 命中距离超过该阈值的节点不入遍历栈，其子树不再被访问。相比 baseline treelet 遍历，PRUNE 使每射线访问节点削减 1.2~1.9×、遍历延迟降 1.4~3.0×、总内存流量降 1.7~2.7×，硬件开销可忽略（并入 AGHU+PRUNE 的 0.7% 面积增量）。
- 从硬件架构角度拆解术语，给出运转流程具体例子（GauTracer Fig. 11(b)）：射线沿 ray 推进 → treelet 遍历中节点出栈 → ray-box 求交得命中距离 t_box → 若 barrier flag 已置位且 t_box > 当前最大命中距离（剪枝阈值）→ 该节点（及其子树）不入栈直接跳过 → 否则正常入栈；命中更近的 Gaussian 时 AGHU 更新堆 → 根（最大 t_hit）可能变小 → 阈值收紧 → 更多远节点被剪。closest-first（PRUNE+SORT）策略（子节点按命中距离排序先遍历近者）可更早触发 barrier，但论文实测收益很小甚至降级（上层节点包围体高度交织重叠导致对"更近高斯集合"判断失误），故不采用。
- 术语一般如何实现？如何使用？：实现 = RTA 遍历控制器内维护 barrier flag + 阈值寄存器 + 比较逻辑（与 AGHU 共用 FSM，Vulkan-Sim 中实现）；剪枝有效性依赖 Hit Gauss Buffer 大小 K（K 越大 barrier 越难触发，剪枝效果减弱，见敏感性实验 Fig. 16）与场景结构（compact/dense 场景如 hotdog、materials、ship 遍历路径局部一致，剪枝收益相对小）。未来方向：改进 BVH 构建提升空间分布质量，或自适应 closest-first 遍历策略。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
