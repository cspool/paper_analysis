## In-pipeline 动态内存分配（buddy-tree allocator + slice 链表）

术语解释
- 把内存分配做成处理器流水线中的一个架构原语（第三流水级）：每执行 stage 一棵 buddy-tree 分配器以 256 B slice 为粒度单周期分配，slice 末字复用为链表指针做动态扩容与释放，使队列存储随执行同步伸缩。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 动机：队列尺寸动态变化，scratchpad 分配必须与 SIMD 执行同速，走 OS/runtime 分配会抹掉专用化收益。实现：8 棵 buddy tree（每 stage 一棵，管理自己的 8 KB sub-bank），深度 6、32 叶（8 KB / 256 B），由 32-to-5 priority encoder 输出第一个可用 slice 索引 → 确定性单周期分配；INIT_Q 指令到达第三级触发分配，8 棵树并行工作，各 stage 得到的 slice 索引可以不同。管理：分配器不追踪 slice 归属，改由各 stage Queue Manager 把 slice 末字复用为"下一 slice 指针"形成 FIFO 链表（避开 Q_ID→slice 列表大表）。释放：FREE_Q；slice 归还给 buddy tree。越界：队列超出片上额度时按 FIFO spill 到 off-chip（见 kernel调度层的条目）。参数：slice 尺寸是权衡内部碎片与分配效率的架构参数。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：`INIT_Q, q0, FALSE` 进入第三流水级 → 每 stage 的 buddy tree 并行查空叶、priority encoder 出 slice 索引（1 cycle）→ slice 索引随指令流经后续流水级 → 各 stage QM 把索引登记到 Q_ID=q0 的队表（head=tail=slice 起点）→ 后续 PUSH 填满 slice 后，QM 申请新 slice 并把旧 slice 末字写成新 slice 指针 → POP 越过 slice 边界时释放旧 slice 回 buddy tree。为什么选 buddy tree：结构仅一棵紧凑 OR 树 + 每 256 B slice 少量元数据，硬件开销可忽略（与 QM 合计 5% 能耗）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用 buddy 分配器（Web：Linux 内核风格实现，如 evanw/buddy-malloc；O(log N) 分配/释放、按 2 的幂向上取整、buddy 分裂/合并、内部碎片换取低外部碎片）是软件数据结构；本文是它的硬件化特例——固定 slice 粒度 + 每 stage 一棵 + 组合逻辑单周期。使用：任何需要硬件内动态尺寸缓冲的加速器（流式处理、动态稀疏结构）。论文未明确说明该硬件分配器是否开源。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
