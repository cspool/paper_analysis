## Row Pointer Table（RPT，轮转预防性刷新行指针表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ColumnKeeper 两变体共用的"刷新哪一行"组件：每 subarray 一个条目（log2(S) bit），保存指向"下一个待预防性刷新的行"的指针。触发机制对 subarray k 生效时，RPT 先返回当前指针 R_k 作为待刷新行，随后自增指向下一行（到 S−1 后回零），实现 subarray 内全部行的 round-robin 覆盖。核心不变式：RPT 当前指向的行 = 该 subarray 内最近最少被预防性刷新、因此 hammer 计数最高的行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- RPT 把"触发即刷新全部潜在受害行"的朴素方案改写为"每次触发只刷 1 行、按轮转覆盖全 subarray"。运转例子：subarray k 连续被锤，CK-D 每 N_PR 次激活触发一次刷新；第 1 次刷 RPT 指向的行 0，第 2 次刷行 1，…，第 S 次刷行 S−1 后指针回零——保证在 N_CD 次激活（对应 S·N_PR）内 subarray 中每一行都被刷过至少一次。CK-D 安全证明正是依赖 RPT 的轮转性质：r_k ≥ S 时最久未刷新的行 i 必然被再次覆盖。in-DRAM 版 CK-D 中 RPT 同样使用，只是每次 RFM 批量刷 7 行并 RPT+=7（CT 复位为 7）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为内存控制器中的小表（每条目 log2(S) bit；N_CD=1M、S=1K、双 rank 时 CK-D 的 CT+RPT 合计 7.5KB、CK-P 仅 RPT 2.5KB）；probe-then-increment 顺序保证并发安全（单入口逐条处理）。使用场景：任何"按 subarray 粒度做时间分散预防性刷新"的防御（RowHammer/ColumnDisturb 均可借用）；与 PRAC 的 RFM 批量刷新不同，RPT 是控制器侧细粒度（单行）摊派手段。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
