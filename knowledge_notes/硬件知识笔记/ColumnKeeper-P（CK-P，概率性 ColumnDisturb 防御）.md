## ColumnKeeper-P（CK-P，概率性 ColumnDisturb 防御）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 无状态（无激活计数器）的概率性防御。每次对 subarray k 发 ACT 时以概率 P_PR"抛硬币"，命中则向 k-1、k、k+1 三个连续 subarray 各刷新一行（行由 RPT 轮转选择）。安全保证可配置（§4.2/§4.3）：设一年内至少一次成功攻击的概率上界为 P_Y；单次攻击成功概率 P_1 = I_q(N_CD−S, S)（q=1−P_PR，I_q 为正则化不完全 beta 函数，即 Binomial(N_CD−1, P_PR) 的 CDF 在 S−1 处的值）；每刷新窗口最多 A = t_REFW/((1+3S)·t_RC) + 1 次攻击；P_REFW = 1−(1−P_1)^A；一年 492.7×10^6 个 DDR4 刷新窗口 → P_Y = 1−(1−P_REFW)^(492.7M)。配置表（S=1K）：P_Y=1e-12 时 P_PR = 1.32e-3（N_CD=1M）/ 1.07e-2（128K）/ 9.50e-2（16K）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 触发逻辑退化为一个伪随机数生成器 + 比较器，硬件复杂度最低。伪代码：
```
on ACT(subarray k):
    if random() < P_PR:            # Bernoulli 试验，与任何其他事件独立
        issue refresh(RPT[k-1]); RPT[k-1] = (RPT[k-1]+1) % S
        issue refresh(RPT[k]);   RPT[k]   = (RPT[k]+1)   % S
        issue refresh(RPT[k+1]); RPT[k+1] = (RPT[k+1]+1) % S
```
Annotations：一次命中发出 3 条单行刷新（三 subarray 各 1 行）；P_PR 由式 P_1 = I_q(N_CD−S, S) 反解，Monte Carlo（1M 次实验）验证 HC_max 与 N_CD 的安全裕度（P_Y 从 1e-3 收紧到 1e-12 时裕度从 4.2% 升到 9.9%）；无计数状态 → 无 open-bitline 感知 → 邻居负载会"double-count"，这是 CK-P 在低阈值下比 CK-D 多刷 1.70-1.83× 的原因。效果：N_CD=1M/128K 单核开销 0.36%/2.73%；16K geomean IPC 0.78（CK-P12）、能耗 +24.69%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Verilog + OpenROAD + NanGate 45nm：面积 0.03 mm²、功耗 15 mW、存储 2.5KB；t_RRD 内完成、0.5ns 松弛。思想沿袭 PARA 的概率刷新（HiRA 的安全分析方法），但把"刷攻击行邻居"推广为"刷三个连续 subarray 各一行"。使用：与 CK-D 同一 Ramulator 2.0 集成（CK-P3 = P_Y 1e-3、CK-P12 = P_Y 1e-12）；适合面积/复杂度敏感、且可接受概率保证的系统。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
