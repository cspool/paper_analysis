## PowerWeave Governor 与 DVFS Controller（OS 级电源管理控制平面：用户态策略 + 内核态执行分离）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PowerWeave 控制平面的两个角色分离设计：(1) Governor——用户态极小库（~250 行 Python），import 进现有 serving 框架（vLLM/SGLang），监控每频率域的请求率（RPS）与尾延迟 vs 配置的 SLO 目标，计算"应用还能容忍多少性能退化（performance slack）"，把 slack 发给 DVFS Controller；(2) DVFS Controller——位于 Interposer（Rust，~5500 行）内，接收 governor 指令与 Frequency-Latency Scaling 模型，为每独立频率域选择工作频率。分离的意义：把"应用级策略（面向具体 ML 用例，管理员可定制）"与"电源管理控制机制（可复用）"解耦。
- Governor 运行逻辑（Fig. 6）：Stage ①以最大频率运行建立每域参考 baseline → 由 baseline 与 SLO 算 slack s₁ → 交给 controller 选频率；Stage ②每个监控窗口检测 RPS/延迟短期偏离，更新 slack：s₂ = ((1−s₁)×l₁)/SLO（由当前延迟反推最大频率下的理论延迟，再除以 SLO 得新的可容忍退化）；Stage ③SLO 违反时快速纠正：立即把受影响域频率拉满，等延迟回到安全余量再重启适应循环。支持多 SLO（TTFT vs TPOT、不同输入长度）时取最保守 slack。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 系统架构运转流程（多租户 B200）：管理员为每个 workload 声明 SLO 与资源上限（TPC 分配，构成独立频率域）→ vLLM serving 请求到达（Azure trace / Poisson）→ Governor 每监控窗口经几行代码从框架取每域 RPS 与尾延迟 → 计算 slack 发给 DVFS Controller → Controller 查该域 frequency-latency 模型选 f_target → 经 NVML 设置该域频率 → 每 kernel 完成后 Interposer 对照预测延迟（偏离 >5% 触发重画像）→ SLO 逼近时 Governor 收紧/拉满频率。效果：TTFT 趋近 SLO 时只升 prefill 域、TPOT 有 slack 时 decode 域保持低频；热节流局部化；零 SLO 违反。
- 与 baseline（LithOS 单一 device-wide 频率、默认 GPU DVFS 被动降频）对比：LithOS 无每域 slack 概念、默认策略无应用语义，二者都"整卡一刀切"；PowerWeave 以"每域 slack + 每域模型 + 快速纠正"实现差异化服务与性能隔离。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Governor 是可直接 import 的库（获取 vLLM/SGLang 用户态指标仅需几行代码），SLO 目标、监控窗口时长、尾延迟百分位全部可配置（per-domain），同一核心机制可支持 latency-driven、per-tenant、throughput-balancing 策略而无需改代码；DVFS Controller/Interposer 在 GPU driver 之上透明拦截 CUDA API，频率经 NVML 下发，不修改应用/框架/driver。硬件前瞻（Discussion）：与 GPU 电源管理固件的 per-domain 请求队列接口，可把"反应式降频"升级为"前瞻式（proactive）按 kernel 序列预下发频率"，隐藏切换通信开销。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
