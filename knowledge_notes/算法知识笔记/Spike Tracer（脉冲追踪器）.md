## Spike Tracer（脉冲追踪器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Spike tracer（脉冲追踪器）是 ST-BIF 神经元内部的记忆单元，记为 S_t：它记录该神经元到当前时间步为止累计发射的脉冲代数和（S_t = S_{t-1} + y_t，y_t∈{-1,0,1}），并受上下界 S_max/S_min 约束。其作用是让 ST-BIF 的"激发决策"带记忆性：当膜电位 V̂_t≥V_thr 但 tracer 已到上界 S_max 时不再发 +1 脉冲（防止输出越界），当 V̂_t<0 但 tracer 已到下界 S_min 时不再发 −1 脉冲。正是这个追踪器使 ST-BIF 输出精确等于 Q-ReLU 的 clip(floor(...), S_min, S_max)，从而保证 ANN→SNN 无损转换。
- 在 ELSA 中，spike tracer 与膜电位 V 一样作为神经元的持久状态，在硬件里各占一块 SRAM（每 PE 4×102.4 KB tracer buffer），每次激发读/写一行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Spike tracer 位于神经元"激发-更新"步：
```
Θ(V̂_t, V_thr, S_t):                     # 决策函数，tracer 参与判定
    if V̂_t ≥ V_thr and S_t < S_max:  return +1
    elif V̂_t < 0 and S_t > S_min:    return -1
    else:                              return 0
S_t = S_{t-1} + Θ(...)                 # tracer 更新 = 累计发射
```
- 例（ELSA 论文）：某神经元 S_max=+4、S_min=−4；若它已连续发 4 个 +1（S=4），第 5 次膜电位再超阈值时 Θ 返回 0（饱和），膜电位照常 soft reset 但不再发射，等效 Q-ReLU 的 clip 上界；只有 V̂_t<0 时才会转向 −1。这使得输出脉冲计数与量化激活的整数值一一对应。
- Annotations：S_max/S_min 是量化位宽决定的 clip 界（4-bit 权重对应 Q-ReLU 输出界）；tracer 是 1 个整数计数器而非多 bit 历史窗。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF 定义 ST-BIF 时引入 tracer，转换后的 SNN 每个神经元携带一个 S 计数器，推理全程累积；它是"有损"的只有一种情况——超过 clip 界的累加被截断（这与 QANN 的量化截断完全一致）。硬件侧：ELSA 的 fire 组件以膜地址 x 读 spike tracer 行，与集成后的膜电位 V̂_t 一起送入决策逻辑；update 组件把 y_t 累加进 tracer 并回写 SRAM。ELSA 芯片中 tracer buffer 占每 PE 面积 17.49%、功耗 0.6%；Tab.IV 显示 tracer 存储是 PE 面积主导（93.97% 面积被 weight/membrane/tracer 三类 SRAM 占据）的原因之一。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
