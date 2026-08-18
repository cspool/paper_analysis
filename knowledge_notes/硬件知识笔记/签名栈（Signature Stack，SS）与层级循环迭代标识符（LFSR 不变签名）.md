## 签名栈（Signature Stack，SS）与层级循环迭代标识符（LFSR 不变签名）

术语解释
SS 是 fetch 单元中一对寄存器（hpc + sig）加一个保存/恢复栈，用 32-bit Galois LFSR 增量维护"层级循环迭代标识符"的压缩形式（不变签名），作为 SBRB 的 key 成分。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 一个动态分支的多实例来自循环，因此动态实例可由层级循环迭代标识符唯一确定：{header block, iteration count} 栈，如 {loopA.iterA, loopB.iterB}；(2) 该标识符在 squashed path 与 resolved path 上不变（invariant）——无论两条路径之间插入/删除了多少分支实例或迭代——这是 SBRB 对齐的数学基础；(3) 字面栈实现需要每次访问 SBRB 时哈希整个栈（贵），因此改为"进行中的签名"：sig 存于 LFSR，进入循环时 sig = LFSR_step(sig ^ fold32(header PC))（LFSR 旋转 1 bit 兼作初始迭代计数 1），继续循环时 sig = LFSR_step(sig ^ 0)（旋转即迭代计数 +1），退出循环按 pop amount 弹栈恢复 {hpc, sig}；(4) hpc（8-bit fold8 压缩的当前循环 header PC）用于区分"继续当前循环"（新观察 header PC 与 hpc 相同）与"进入新循环"（不同）；(5) call 也 push 并用 call site PC 更新 sig、return pop 恢复——保证递归/同一循环内多次调用同一函数时签名唯一且不变；(6) key = fold32(PC) XOR sig。设计空间探索确认 32-bit LFSR 签名极少/从不碰撞，8 项栈、8-bit hpc 即可达最大收益（与 64 项栈、64-bit sig/hpc 几乎无差）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SS 与 LIT（循环信息表）协同工作：LIT-H 命中输出 hit_header(PC)、LIT-E 命中（且预测方向匹配 exit 方向）输出 hit_exit(pop_amount)、BTB 信号 hit_call/hit_return，SS 按 Table I 状态机更新：
```
hit_header(PC):
    if fold8(PC) != hpc:            # enter loop
        push({hpc, sig}); hpc = fold8(PC)
        sig = LFSR_step(sig ^ fold32(PC))
    else:                           # continue loop
        sig = LFSR_step(sig ^ 0)
hit_exit(pop_amount):               # exit loop
    while pop_amount: pop({hpc, sig})
hit_call(PC):
    push({hpc, sig}); hpc = 0
    sig = LFSR_step(sig ^ fold32(PC))
hit_return:
    pop({hpc, sig})
```
Annotations：LFSR_step 为 Galois LFSR 移位一步（带数据输入 d 的版本：sig_new = LFSR_step(sig ^ d)）；fold8/fold32 是逐半折叠异或压缩；pop_amount 由 loop descriptor 提供；栈仅 8 项、每项 40 bit（8+32）。恢复路径：64 个分支 checkpoint 每项追加 SS 副本（误预测恢复时精确回滚 sig 到分支之前）；BQ 每项 +6 bit 冗余维护"退休 SS"（load violation/异常到 ROB 头恢复时使用）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Galois LFSR：用本原多项式构造的反馈移位寄存器，n bit 可遍历 2^n − 1 个非零状态（maximal-length 序列），适合做紧凑、可逆、硬件便宜的"状态哈希"；本文在 LFSR 移位路径上加 XOR 数据输入口以吸收 header PC。签名不变性的成立依赖：squashed 与 resolved 两条路径在循环层级结构上一致（仅内容不同），进入/继续/退出循环的操作序列两边相同，因此恢复 checkpoint 后 sig 相同。使用前提是编译器把循环结构传给硬件（LIS/LIT）。局限：非循环控制流（non-loop cycles）得不到不变签名（如 401.bzip2 mainQSort3()，签名固定而行为迭代，SBRB 失效）；签名/折叠碰撞与 key 碰撞是残余的不精确来源（论文声称 near-perfect alignment）。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
