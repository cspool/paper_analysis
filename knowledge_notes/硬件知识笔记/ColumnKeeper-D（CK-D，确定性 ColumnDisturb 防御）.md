## ColumnKeeper-D（CK-D，确定性 ColumnDisturb 防御）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 内存控制器内的确定性 ColumnDisturb 防御（不改 DRAM 芯片）。组件：每 subarray 两张计数器表 CT-E（偶列激活数 c_k^even）与 CT-O（奇列激活数 c_k^odd），外加共享的 RPT（见 RPT 条目）。规则（§3.2）：对 subarray k 发 ACT 时，k 自身的全部列 bitline 被 hammer → CT-E[k] 与 CT-O[k] 各 +1；由于开放位线架构，k-1 只有偶列、k+1 只有奇列共享 SA → 只对 CT-E[k-1] 与 CT-O[k+1] 各 +1。当 max(CT-E, CT-O) 达到预防性刷新阈值 N_PR 时触发一次单行预防性刷新，并复位该 subarray 的两个计数器。N_PR = N_CD/S（S=每 subarray 行数；计算前先从 N_CD 减 2S 以抵扣 REF 命令引入的激活，§4.4）。安全性：对最坏访问模式用归纳法证明无翻转（§4.1）——任意窗口内 max(HC_odd, HC_even) ≤ r_k·N_PR，且 r_k ≥ S 时轮转刷新必已覆盖该行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- CK-D 挂在内存控制器命令路径旁（关键路径之外）。伪代码：
```
on ACT(subarray k):
    CT_E[k] += 1; CT_O[k] += 1
    CT_E[k-1] += 1; CT_O[k+1] += 1   # open-bitline: 邻居只打一半列（边界 subarray 只加内侧）
    if max(CT_E[k], CT_O[k]) >= N_PR:
        row = RPT[k]                 # probe: 取当前轮转指针
        issue ACT(row) + PRE         # 单行预防性刷新，分散在时间上
        RPT[k] = (RPT[k] + 1) % S    # 指针后移，round-robin
        CT_E[k] = 0; CT_O[k] = 0     # 触发刷新的 subarray 计数器复位
```
Annotations：N_PR = (N_CD−2S)/S；每个 ACT 最多更新 4 个计数器条目 + 最多触发 1 条刷新命令；预防性刷新以单行 ACT+PRE 形式执行，避免朴素方案一次刷 3K 行造成 3K·t_RC≈153μs 的 bank 阻塞；计数器复位只针对触发刷新的 subarray。效果（Ramulator 2.0，62 单核 + 60 四核）：N_CD=1M/128K 单核平均开销 0.15%/1.70%；16K 时 geomean IPC 0.84、能耗 +16.26%；与 Graphene/PRAC/Hydra 组合仅额外降 1.70%~2.15% IPC。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Verilog HDL 实现，OpenROAD 开源数字流程 + NanGate 45nm 库综合：面积 0.1 mm²、功耗 50 mW、存储 7.5KB（N_CD=1M、S=1K、双 rank：CT 条目 log2(N_CD/S) bit、RPT 条目 log2(S) bit）；计数/查表在 t_RRD（2.5ns）内完成且有 0.7ns 松弛。使用：作为 Ramulator 2.0 控制器插件运行（github.com/CMU-SAFARI/ColumnKeeper，Zenodo 10.5281/zenodo.19446517）；对照单计数器变体 CK-S（不区分奇偶列）在 16K 阈值下多刷 5.06% 的刷新、IPC 多降 0.55 个百分点，说明 open-bitline 感知的 double-counting 消除是性能关键。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
