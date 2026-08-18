## Off-chip Queue Spill 与 FIFO 预取（队列溢写调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Morpha Core 处理"队列超过片上 scratchpad 容量"的运行时数据调度机制：队列按 FIFO 顺序溢写到 off-chip 内存，片上仅保留 head slice，slice 末字复用为 off-chip 指针；利用队列访问的严格 FIFO 顺序性做自动 prefetch 与 double-buffering，从而不需要 cache 层次（tag 比较、替换、一致性等复杂逻辑）。公平性：Q_ID 指令中一个字段设置每队列最大片上 slice 数，超过即强制 spill。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 过程伪代码（队列读循环 + spill，按论文 §III-C 描述整理）：
    ```
    while queue not empty:
        if head slice on-chip:
            elem = sub_bank[head]        # head slice 驻留片上
            head += 1
        else:
            DMA_fetch_next_slices(queue, prefetch_depth=2)  # 按 FIFO 顺序预取下一批 slice
            # 双缓冲：当前 slice 被消费时，下一个 slice 已在途
        if tail slice full and onchip_slices >= max_onchip_slices:
            DMA_spill(oldest_slice, offchip_addr)            # 最老 slice 换出
            oldest_slice.last_word = offchip_ptr             # 末字改存 off-chip 指针
    ```
  - 评估（论文 Fig. 8c）：对输入/输出超过片上容量的通用向量 kernel，扫数据量 1.5×–20× 片上容量与算术强度 5/10/20 FLOP/B，per-element latency 归一化到"理想 cache + prefetch"。最坏情形（1.5× 容量、AI=5）溢写开销 12.7%，3× 容量降至 6.6%；AI=10 时仅 6.2%（memory-bound 时固定 spill 成本被摊薄）。对照：scratchpad 相比 cache 省 ~30% 面积与能耗（论文引 [106,107]），再加省掉的 cache 控制逻辑。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：溢写决策在 Queue Manager（片上 slice 计数 vs Q_ID 上限字段），数据面走 Data Movement Engine 的 DMA；预取依据是"队列必然按 FIFO 消费"这一确定性，等同于软件流式引擎（如 SDF/streaming accelerator）的确定性预取。使用场景：动态尺寸数据结构超过片上存储的加速器负载（大点云过滤、长 frontier 队列）；与通用 cache 相比牺牲随机访问友好性换取简单性与能效。论文未明确说明该机制的开源实现（Web 未找到）。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics
