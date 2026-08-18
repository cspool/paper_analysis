## 泡利帧（Pauli Frame）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
泡利帧（Pauli frame，Riesebos et al. DAC 2017；Fowler & Gidney 2018）是 FTQC 经典控制层维护的经典数据结构：记录解码器推断出的、尚未物理纠正的累计 Pauli 误差。逻辑链：Clifford 门 C 把 Pauli 误差 E 共轭为另一个 Pauli E'=CEC†∈P_n，所以可以"在软件里"跟踪误差而不物理纠正（错误通过 Clifford 电路时只在帧上更新）；只有当非 Clifford 门（T 门）出现时，TXT†∉P_n 使误差不再能表示为 Pauli 帧，必须物理纠正。这样解码就可以异步进行（Clifford 门照常执行），把经典纠错负担延迟到非 Clifford 同步点。Web：lattice surgery 的 merge/split 奇偶结果也作为经典边信息吸收进 Pauli frame，即 Pauli-Based Computation（PBC）——所有 Clifford 门被推到末尾"软件执行"，运行时只剩 multi-qubit Pauli 测量序列。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Triage 把 Pauli frame 的同步要求形式化为调度器的 deadline/因果锥：
```
# T 门 gate teleportation 的同步（Triage 论文 Fig.4 逻辑）
|ψ⟩ 带累计误差 E_acc（存于 Pauli frame）
T 门 teleportation：准备 magic state |A⟩，CNOT + 测量，classically-controlled S 校正
S 校正不能 commuted 过 T 门吸收进帧 → 必须先物理施加 E_acc† 恢复 |ψ⟩
⇒ 同步点：E_acc 所在因果锥（该逻辑比特所有相关历史 slice）必须先解码完
⇒ slice 属性 deadline = 到最近关键同步点的层数；因果锥 = 必须解码的 slice 集合
```
调度器据此做优先级调度：Clifford 操作可异步（宽松），非 Clifford 同步点前必须保证因果锥已解码，否则插入 idle 层 → LER 上升。这解释了为什么 FTQC 解码是"带优先级的动态调度问题"而非纯吞吐问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：解码器每处理完一个窗口更新帧表（per-logical-qubit 的 Pauli 标记）；遇到非 Clifford 门时把所有相关帧的误差综合成需物理纠正的 E_acc，执行物理校正（同步）。使用场景：PBC/lattice surgery 的全栈协议中帧更新是经典控制层的中枢；Triage 的 T-gate 同步检查（每个关键操作执行前检查因果锥是否解码完）就是帧同步的调度器实现。局限：帧同步失败即 stall，且同步要求的紧迫性随 T 门密度上升（Triage 用 T-Den. 最高 49.61% 的 benchmark 验证）。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
