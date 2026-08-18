## PDOM 栈与 BTFNT 分支预测（SIMT 发散与重收敛跟踪）

术语解释
PDOM（Post-Dominator）栈：SIMT 发散/重收敛跟踪结构，每发散路径压入（next PC, reconvergence PC, active mask）三元组，路径完成时弹出恢复锁步；与 NVIDIA Fermi 的 divergence/token stack 同类。BTFNT（backward-taken forward-not-taken）是极低硬件开销的静态方向预测，DICE 用它在跨 p-graph 层次预取后继，使 CS/FDR 不必等 e-block 完成即可提前准备。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
后支配点：从分支点出发所有路径必经的汇合点即重收敛点（reconvergence point）；SIMT 硬件在发散时压栈记录路径起点、重收敛 PC 与 active mask，路径结束弹栈恢复未走路径的 mask 并最终回到重收敛点，嵌套分支多层压栈。DICE 每 CTA 一个 PDOM 栈（Fig.7c），可处理嵌套控制流与整块跳过（skipped computation）。BTFNT：向后分支（循环回边）预测 taken、向前分支预测 not-taken——硬件成本近零，在两条路径都有活跃线程时效果好（Fig.3 例中 BB1 与 BB3 都预测后继为 BB2）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DICE 运转流程：FDR 阶段 Branch Handler 读取 p-graph metadata 中的 BRANCH_* 字段 → 按 BTFNT 预测该 CTA 的下一 p-graph PC → 更新其 PDOM 栈 → CS/FDR 据此先行取出后继 e-block 的 metadata/位流（与执行重叠）；e-block 进入 DE 前必须等 active mask 解析（thread activeness resolved），误预测的 e-block 直接丢弃。barrier 处理因 CTA 级执行而简化：FDR 只需等待 RE 信号确认该 CTA 全部前序 e-block 访存完成，无需逐 warp 到达跟踪（GPGPU 需跟踪每个 warp 的 barrier 到达）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：per-CTA（Fermi 为 per-warp）栈，条目含 next PC / reconvergence PC / active mask（Fermi token stack 另含 entry type）；现代 GPU（Volta 后）用独立线程调度（ITS）+ BSSY/BSYNC barrier 寄存器替代隐式掩码栈。使用：处理不可谓词化的发散（长路径、嵌套、整块跳过），与谓词执行互补。Web sources：Analyzing GPU ISAs（Fermi token stack 语义）；ElTantawy et al., HPCA'14《A scalable multi-path microarchitecture for efficient GPU control flow》（SPS 栈局限与多路径重收敛）；NVIDIA Fermi/Volta 白皮书。

涉及论文标题：
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
