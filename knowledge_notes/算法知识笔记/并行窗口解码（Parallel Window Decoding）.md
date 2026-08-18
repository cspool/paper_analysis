## 并行窗口解码（Parallel Window Decoding）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
并行窗口解码（Skoric, Browne, Barnes, Gillespie, Campbell, Nature Communications 2023, arXiv:2209.08552）是解决 FTQC 实时解码吞吐瓶颈的协议：把 syndrome 时间流切成固定大小窗口，时间上不相邻的窗口因果独立可并行解码（checkerboard 模式：所有偶窗口并行、再所有奇窗口并行），空间上不同逻辑比特的操作也可分区并行。串行 sliding window 需满足 τ_dec<τ_gen（否则指数级 syndrome 积压，Terhal 论证），且 τ_dec∝N 使任何解码器都存在 code distance 上界；并行窗口通过扩展窗口缓冲（包含邻居边界 syndrome 的 look-ahead 区域）让每个窗口自包含，以更多并行解码器换取吞吐——即使单个解码器慢（τ_dec≥τ_gen）也能维持系统吞吐，把指数退化降为多项式。Web：可与 union-find 或 MWPM 内层解码器结合，数值验证 surface code 无逻辑保真度损失。Triage 论文把其时间维度（time-parallel [24]）与空间维度（[27]）统一进 slice 约束图框架。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
并行窗口解码的输入到输出流程（Triage 视角）：
```
# 输入：连续 syndrome 流（每 d 轮一层）+ M 个解码器
for 时间块 k (checkerboard):                 # 时间维并行
    for 逻辑 patch p:                         # 空间维并行
        w = window(p, k)                       # 窗口 = 时间块内的 slice + 边界缓冲
        buffer(w) = 邻接窗口的边界 syndrome   # 缓冲大小 = 解码 volume，决定延迟
    decode(偶窗口集) in parallel on 解码器池   # 互斥约束内并行
    decode(奇窗口集) in parallel
    窗口边界人工 syndrome 由先解码的窗口产生
# 输出：每窗口的错误链 → 更新 Pauli frame；同步点前需因果锥全部解码
```
Triage 的改进：baseline time-parallel 只在时间维并行、不拆分多量子比特操作（lattice surgery 合并区仍是整块 → 高饱和层 floor）；Triage 以 slice 为原子单元同时并行时间与空间，把合并区切成可并行小窗口，并加资源感知调度（M-for-N 池 + 优先级）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：解码器（pymatching/UF）以窗口为单位跑 MWPM/UF，窗口缓冲由并行度参数与邻居重叠决定；调度器决定窗口→解码器的分配。Triage 用它做三件 baseline：serial sliding window（一次处理一个 lattice surgery 块）、time-parallel window（时间维并行，不拆多量子比特操作）、SWIPER（投机窗口解码，SOTA）。评估参数：解码器数 M 与相对速度 τ_dec/τ_gen 扫描，指标为插入 idle 层数与 LER；Triage 在慢解码器区（τ_dec>τ_gen）仍有效——论文声称"通过调度并行窗口可克服单解码器延迟限制"。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
