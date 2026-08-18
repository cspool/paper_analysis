## 服务器编排式超分管线（Server-orchestrated SR，NEMO 风格）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
服务器编排式超分是"上采样策略由服务器决定、客户端只执行"的视频 SR 系统架构（代表：NEMO，MobiCom 2020）。流程：服务器离线分析视频（逐帧跑 SR 评估质量影响）→ 决定哪些帧/区域交给客户端 SR → 把计划以附加元数据（SR 帧标记、HR 运动矢量等）嵌入 bitstream 发给客户端 → 客户端只对选定帧做 SR，其余帧用"修改过的解码器"按宏块做类解码重构，以维持时间一致性。SLICE 论文将其作为主要对比 baseline（无 per-video 训练版本：固定 SR 模型、质量约束调到与 SLICE 相同 PSNR 以公平比较），并论证其三大缺陷：① 服务器离线分析成本高（论文实测 i7-14700K + RTX 3080 Ti 上每视频 24.95 分钟）；② 依赖服务器分析，无法用于直播、视频会议、设备本地视频（无网络/隐私约束）；③ 修改 codec 导致硬件解码器闲置、客户端被迫用低效 CPU 做宏块级重构。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
对比两种策略的"策略决策位置"：
```
Server-orchestrated（NEMO 风格）：                  SLICE（client-only）：
server: 离线分析视频(24.95min/视频) + per-video       client: 解码 bitstream
        微调模型 → 决定 SR 帧集合                         → 用码流自带 MV/残差做 patch 级决策
server: 把计划嵌入 bitstream（改 codec）             client: 硬件解码器标准解码（不改码流）
client: 解码 → 只 SR 选定帧 → 其余帧 CPU 宏块重构    client: reuse/SR/插值三路 patch 调度
```
SLICE 在论文评估中以"目标质量约束匹配 SLICE PSNR"配置 server-orchestrated，因此其 FPS 更低的原因不是质量差异，而是：未选帧的类解码重构在移动 CPU 上按宏块跑（慢）、固定模型需更大比例帧做全帧 SR 才能达标。SLICE 对应地全流程在 GPU 上执行且每帧 patch 级分发更新，帧率更高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NEMO 是公开研究项目（论文 [5]，H. Yeo et al., MobiCom 2020，GitHub: hyeonjaejeon/NEMO）。SLICE 论文实现了其"无 per-video 训练"变体作为对比：服务器用固定 SR 模型决策、客户端只在选定帧做 SR、其余帧解码器式重构，且只测量设备侧上采样成本（排除服务器分析成本）。论文还指出 NERVE（服务端分析经 side channel 发二进制 hint）、NeuroScaler（选择/增强 anchor 帧并显式信令）等同类系统都依赖服务器分析与非标准信息，而 SLICE 只依赖客户端可得的码流信号。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
