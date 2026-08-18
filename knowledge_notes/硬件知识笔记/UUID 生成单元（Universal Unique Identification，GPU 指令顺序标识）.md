## UUID 生成单元（Universal Unique Identification，GPU 指令顺序标识）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UUID 生成单元是 sCROOGe 为 OoO 执行实现指令顺序跟踪的硬件：为每条被调度指令生成按 warp 递增的 UUID（唯一标识编号），嵌入指令数据直到 Commit 阶段，从而在乱序执行时仍能恢复程序序，且无需全相联 reorder buffer 的昂贵面积。逻辑链：乱序执行必须知道指令的原始顺序（用于仲裁最老指令、保证正确性），最直接的方案是全相联 ROB（面积大）；UUID 方案用轻量递增编号 + 循环复用实现"顺序比较"：UUID 位宽 N 时，把 MSB 为 '00' 的 UUID 定义为排在 MSB 为 '11' 之后（修改后的"小于(<)"算子），获得 2^N - 2 条指令的安全余量；超过则置越界标志并触发流水线 stall（Fig.3：检测 MSB 从 11→00、00→01 的过渡，若过渡处同流水级存在不同迭代的指令即越界）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：Schedule/发射时每个 warp 的 UUID 计数器递增 → UUID 附加进指令数据随流水传播（frontend：IsB 分配仲裁与"per-warp 最老"选择用 UUID 比较；backend：OC 分配仲裁检查 UUID 越界条件）→ 直到 Commit 阶段指令退役后 UUID 释放。例子（N=4 位，2^4=16 个编号空间，安全余量 14）：warp 连续发射指令获得 UUID 0,1,2,...,14，乱序执行中 Issue/Dispatch 仲裁用 UUID 判断谁是程序序中最老的待处理指令；当计数器绕回（11→00）时，若仍存在于途指令，越界检测单元发现同一流水级同时存在来自不同迭代的指令，stall 流水直到旧迭代清空。该机制替代了 ROB 的按程序序提交逻辑，是 sCROOGe 的关键轻量化设计（frontend/backend 两方案共用）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每 warp 一个小计数器 + 指令数据中扩展 UUID 字段 + 越界检测组合逻辑（比较 MSB 过渡）；面积开销远小于全相联 ROB。使用：与 IsB/CU 分配仲裁、"per-warp 最老"选择、frontend 的访存指令非独立判定（"非最老的访存指令不得视为独立"）配合；sCROOGe 论文将其列为五项设计优化之一（light-weight UUIDs for efficient instruction tracking）。评估中未单独报告 UUID 面积，但 frontend 方案整体面积开销峰值仅 7.5%，佐证其轻量性。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
