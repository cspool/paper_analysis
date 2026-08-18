## Any-Gauss-Hit Unit（AGHU，任意高斯命中单元，max-heap Hit Gauss Buffer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AGHU 是 GauTracer 提出的硬件单元，把软件 any-hit shader 的"每射线最近 K 个命中维护"搬进 RTA。Hit Gauss Buffer 组织为 max-heap（完全二叉树，线性映射到数组，父子关系用位位移/增量导出），每个条目含命中距离 t_hit、alpha、primitive ID，按 t_hit 为 key 保证最远命中在根。支持两类行为：①未满时新命中追加到 N_hit 处并递归 shift-up 恢复堆性质（插入值保留在寄存器减少访存）；②满时新命中先与根（最大 t_hit）比较——更大直接丢弃，否则替换根并 shift-down。closest-hit 阶段按堆弹出（pop 根 + shift-down）得到 far-to-near 序列，配合 back-to-front 混合（式 7）避免再排序。RGIU 输出可缓冲异步由 AGHU 处理（Vulkan 不规定 intersection 任务顺序），与 BVH 遍历重叠隐藏堆操作延迟。面积/延迟：2~12 cycles、825.6 µm²，增量仅 baseline 运算单元组的 0.7%（简单 FSM + 寄存器）。
- 从硬件架构角度拆解术语，给出运转流程具体例子：RGIU 命中输出 → AGHU 插入流程：
  ```
  # 未满（N_hit < K=16）
  heap[N_hit] = new_hit; idx = N_hit; N_hit += 1
  while idx > 0 and heap[parent(idx)].t_hit < heap[idx].t_hit:
      swap(heap[parent(idx)], heap[idx]); idx = parent(idx)
  # 满：先比较根
  if new_hit.t_hit >= heap[0].t_hit: discard
  else: heap[0] = new_hit; shift_down()
  # closest-hit：循环 pop 根得到 far-to-near 序列
  ```
  对应 baseline 的软件 any-hit shader（Alg. 2 比较插入 + 全局 Closest Hit Buffer 访存），AGHU 使 shader 指令再削减 38.9×（避免全局 buffer 的 sort/insert 与访存），RGIU+AGHU 合计加速 3.1~3.3×。Hit Gauss Buffer 大小 K=16 权衡每射线性能与 SIMT 效率（更大 buffer 占寄存器、降每 SM 射线并发、削弱 treelet 预取与延迟隐藏；K≥32 假命中近线性增长、barrier 难触发）。
- 术语一般如何实现？如何使用？：AGHU 为论文新增单元；类比软件实现是 3DGRT [27] 的 k-buffer（全局内存每像素 K 深度缓冲）与 StopThePop 的射线级排序。硬件实现 = RTA 内 FSM + 寄存器堆 + 比较器（28nm 综合 825.6 µm²），通过 Vulkan-Sim 修改（Ray buffer 扩展 Hit Gauss Buffer + hit counter，每次 traceRay 前复位）评估。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
