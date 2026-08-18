## GigaThread Engine（GPU 硬件 CTA 调度器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GigaThread Engine 是 NVIDIA GPU 的全局硬件工作分发单元（work distributor）：把 kernel launch 指定的 CTA grid 动态派发给各 SM，而不是一次性映射整个 grid。由于该硬件组件的精确行为未公开文档化，学界普遍从经验研究推断其默认策略为 round-robin（RR）式——先把 CTA 一轮轮地分给每个 SM（先保证每 SM 至少一个，资源允许再第二轮），直到 SM 因寄存器/SMEM/warp slot 等资源饱和，此后新 CTA 在旧 CTA 完成腾出资源时补位。PIPEWEAVE 把它作为"硬件调度范式"纳入 Scheduling Simulator：对常规 kernel（FA2、RMSNorm 等）显式模拟 RR，以得到真实的 task→SM 分布，从而捕捉跨 SM 负载不均——这是 Neusight/Habitat 静态 wave 假设所忽略的。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（PIPEWEAVE Scheduling Simulator 的 RR 模拟，输入 task 集合 T 与 SM 数 N_SM）：
```
# 硬件调度器范式（round-robin）：
assigned = [ [] for _ in range(N_SM) ]
round = 0
while T 非空:
    for sm in SMs:                      # 一轮内每个 SM 各拿一个 CTA
        if 资源允许(SM[sm], next_CTA) 且 T 非空:
            assigned[sm].append(T.pop())
            # 资源判定：寄存器/SMEM/warp slot 上限
    # 轮次推进；SM 饱和后不再参与；旧 CTA 完成释放资源 → 补位
# 输出 {T_1..T_N_SM}，每个 SM_j 的负载决定其 pipeline demand
```
与 persistent kernel 的对比：persistent 下 CTA 只 launch 一次、硬件分发退居其次，task 分配改由软件 tile scheduler 决定（见 Persistent Kernel 条目）。PIPEWEAVE 对两种范式都建模，才能解释 FA3（确定性软件调度）per-SM op 误差 0.45% vs FA2（动态硬件调度）6.34% 的差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：GPU 前端的 CWD（Command Work Distributor）单元维护 CTA 队列并按 SM 可用槽位派发，配合每个 SM 内的 SMSP 硬件调度器把 warp 分发到 warp scheduler（见 SMSP 条目）；对软件而言完全透明，程序员只指定 grid/block 配置。性能建模中通常用 RR 近似（经验文献 [Hong & Kim ISCA'09]、[Jog ISCA'16] 等），PIPEWEAVE 也按此实现。使用意义：硬件调度是决定 tail effect、wave quantization 与 per-SM 负载均衡的第一因素，任何 tile 级性能模型（含 PIPEWEAVE 的 task 分布与 Max SM 特征）都必须先建模它。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
