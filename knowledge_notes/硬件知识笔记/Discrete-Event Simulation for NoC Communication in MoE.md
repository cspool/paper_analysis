## Discrete-Event Simulation for NoC Communication in MoE

术语解释
离散事件模拟是 HD-MoE 用于精确建模 3D NMP 2D mesh NoC 中不规则 all-to-all 通信延迟的技术。通过事件驱动的优先级队列调度、XY routing 路径计算和链路占用追踪，模拟器输出 all-to-all dispatch + all-to-all combine 的完整通信时间线。

术语是什么？
Discrete-Event Simulation (DES) 是一种通过追踪系统中离散事件的发生时刻和状态变化来模拟系统动态行为的仿真方法。在 HD-MoE 中，DES 用于模拟 MoE 推理中 all-to-all 通信在 2D mesh NoC 上的传输过程：每个通信任务（token hidden state 从 src node 到 dst node 的传输）被拆分为多跳 chunk 传输，每个 chunk 的传输是一个"事件"。模拟器维护 priority queue 按时间戳排序所有事件，维护 link schedule dictionary 追踪每条链路的忙/闲状态。当链路变为空闲时，队列中下一个等待该链路的事件被调度执行。HD-MoE 的 DES 模拟器经验证与 ASTRA-sim 高度一致（R² > 0.9 across scenarios），并为 LP 优化提供线性近似的校准系数 γ。

从硬件架构角度拆解术语
DES 模拟器在 HD-MoE 中的运转流程：
1. **输入**：placement matrix P_ic（expert i 在节点 c 上的分配比例）、expert activation trace（f_i, f_g from MT Bench）、batch size B、NoC 配置（mesh 尺寸 D×D、BW per link）。
2. **通信任务生成**：遍历 batch 中每个 token 的 activated expert group g → 对 group 中每个 expert i 查询其物理节点 src → 确定聚合节点 dst → 生成通信任务 task(src, dst, data_volume=Bh)。
3. **路径计算**：XY routing (src_x, src_y) → (dst_x, dst_y)，先沿 X 走 |src_x-dst_x| 步，再沿 Y 走 |src_y-dst_y| 步。缓存计算结果。
4. **事件调度循环**：while priority_queue not empty: pop 最早事件 → 检查路径上每条 link 的 schedule → 若当前时刻 t 所有 link 空闲，调度事件在 [t, t+chunk_size/BW] 占用所有 link → push 下一跳事件；否则延迟到最早可用时间。
5. **输出**：最后一个事件完成时间 = t_comm；每个 link 的利用率 heatmap。

术语一般如何实现？如何使用？
DES 技术广泛应用于 NoC/互连模拟（BookSim2、GARNET、ASTRA-sim）和分布式系统模拟。HD-MoE 使用自建 Python DES 模拟器，开源代码见 https://github.com/angerybob/HD-MoE。线性近似模型 t̂_comm = (4/BW)·max_c{ Σ_g (Π_{i∈g}⌈P_ic⌉)·f_g·B·h } 作为 DES 的替代用于 LP 优化，其中 t_comm = γ·t̂_comm（γ 经验校准）。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
