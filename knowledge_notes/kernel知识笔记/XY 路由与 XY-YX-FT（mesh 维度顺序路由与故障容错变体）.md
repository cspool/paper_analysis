## XY 路由与 XY-YX-FT（mesh 维度顺序路由与故障容错变体）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XY 路由（dimension-ordered routing 在 2D mesh 上的特例）是 mesh 上通用的最短路径确定性路由：数据包先沿 x 轴走到目标列、再沿 y 轴走到目标行（YX 为反向顺序，XY-YX 表示按维度顺序二选一/交替）。它简单、无死锁、路径最短，但对并发通信不感知：多个任务共享相同链路时产生严重带宽争用（Fig.7a 中任务 1/2 共用 (9,10)(10,11)(11,7) 形成橙色热点）。XY-YX-FT 是论文采用的 baseline 增强版——在 XY-YX 基础上加入回溯（backtracking）规则以覆盖更多故障情形：故障打破 mesh 对称性后，XY 固定顺序路由会失效或产生代价高昂的 detour，回溯规则允许在死路/故障时退回换向，从而容忍更多节点/链路故障，但路径长度增加、延迟劣化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XY 路由对单源单目标的路径计算：
```
# 从 (sx,sy) 到 (dx,dy)
if sx < dx: 先向东走 (dx-sx) 步
elif sx > dx: 先向西走 (sx-dx) 步
then: 再沿 y 轴方向走 |sy-dy| 步
```
失败例子（Fig.7b）：XY 下任务 2（9→11）走 9→10→11，与任务 1（8→7）走 8→9→10→11→7 在 (9,10)(10,11) 争用；XY-YX-FT 用回溯让任务改走 detour（如 9→5→6→7→11 类非最短路径），避开冲突但拉长路径、抬高延迟；而 BALD 通过负载感知的链路分配（见"BALD"条目）在最短路径内达成均衡，无需 detour。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NoC 路由器按确定性维序转发（广泛用于 BookSim2、gem5-Garnet、HD-MoE discrete-event simulator 等）；XY-YX-FT 需路由器支持回溯/换向（论文作为评估 baseline，未给具体规则实现细节）。使用：作为 BusyBarn/BALD 的对照 baseline（合成通信、映射敏感、端到端三组实验均与 XY/XY-YX-FT 对比）；工程上 Tesla Dojo、TSMC SoW 类 wafer-scale GPU 的 D2D 通信亦用 XY routing（见知识库_硬件知识笔记 Wafer-Scale Multi-Chiplet GPU 条目：1.7 TB/s D2D、200 ns/hop、XY routing）。局限：无全局争用感知（MultiTree 亦缺乏全局意识），故障下性能退化显著（BALD 相对其 1–2.55× 加速）。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
