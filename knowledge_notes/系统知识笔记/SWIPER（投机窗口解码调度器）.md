## SWIPER（投机窗口解码调度器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SWIPER（Speculative Window dEcoding with Parallel Execution and Recovery，Viszlai, Chadwick, Joshi, Ravi, Li, Chong，ISCA 2025，arXiv:2412.05115）是并行窗口解码的投机调度方案：借鉴经典架构的 branch prediction，用轻量投机步骤预测相邻解码窗口间的数据依赖，从而同时解析多层解码任务，减少 T 门 teleportation 这类阻塞操作处因依赖等待造成的停顿。实现上窗口分 commit region 与 buffer region（约 d 轮），解码管线化；T 门（非 Clifford）是阻塞操作，测量到校正延迟由解码器反应时间决定。结果：相比先前并行窗口解码器平均应用运行时间降低约 40%。开源：github.com/jviszlai/swiper（含 SWIPER-SIM 模拟器、benchmark、artifact）。Triage 论文把它当作 SOTA 投机 baseline：复现其 successor-based 策略（乐观假设，10% misprediction rate、10% speculation time，投机解码模块不计入解码器占用）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SWIPER 作为 Triage 的 baseline 在调度仿真中的运转：
```
# Triage 框架内的 SWIPER 复现（每决策点）
1) 窗口按时间块排队；后继窗口的依赖被"投机"预测为可并行
2) 分派时乐观地把多个后继层同时派发给解码器池（misprediction 率 10%）
3) 预测正确 → 并行度极高，窗口边界开销最小（资源充裕时近乎最优）
4) 预测错误（10%）→ 需要恢复/重解，占用额外解码器与时间
# 与 Triage 对比（Bell4 上扫 M 与 τ_dec/τ_gen）：
- 资源充裕区：SWIPER 投机并行获得全局最低 idle 层
- 资源受限区：投机开销造成资源争用，性能显著下降；Triage 用启发式+紧急双模在该区最优
```
评估中 SWIPER 在跨 benchmark 的资源受限场景下性能与 FIFO 相当（论文因此省略其曲线）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：官方仓库 jviszlai/swiper 提供 SWIPER-SIM 模拟器与 Python artifact（3.10/3.11）；Triage 在其框架内复现 successor-based 策略做公平对比（speculation 不计入解码器占用是 Triage 侧对其的保守处理）。使用场景：作为并行窗口解码的 SOTA 基线评估新调度器；其弱点（资源受限区投机开销、依赖窗口粒度的依赖预测）正是 Triage 双模设计要解决的痛点。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
