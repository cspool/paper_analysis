## pLRU（pseudo-LRU，伪 LRU 替换策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
pLRU 用一棵二叉决策树近似 LRU：n-way 组相联缓存每 set 维护 (n−1) 位，命中/插入时沿路径把所有节点位翻向"另一侧"，淘汰时从根沿节点位反向走到叶子的 way 即近似"最近最少使用"者。开销 O(log2 n) 位/set（每行 0 位），远低于真 LRU 需要的全序信息。Bumper 基线 L1I = 192KB 6-way pLRU、L1D = 128KB 4-way pLRU。Bumper 的 send_hint 位加在 L1I tag 上，与 pLRU 替换状态并行维护、互不影响。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程（6-way 例子）：树根位表示"上半组/下半组"，逐层二分到 way；L1I 命中 way k → 沿根到 k 的路径把所有节点位指向 k 的兄弟子树（表示 k 最近使用）；L1I 填新行 → 从根沿节点位反向找到 victim way 写入。Bumper 中 L1I 行被填时（fill）若 l2_vulnerable_fill=1 则置 send_hint=1，随后该行首条指令提交触发一次 L2C 提升——这与 pLRU 在 L1I 内的替换决策无关（L1I 内复用由 pLRU 管，L2C 提升由 commit 证据管）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每 set 一组触发器构成树；变体 tree-PLRU（Jiménez MICRO'46）在树节点上用"插入/提升"位而非纯指针位，可支持 RRPV 类似的插入策略。pLRU 常与随机/轮转结合用于 L1/L2 小容量缓存；大容量末级缓存多用 RRIP/DRRIP 类策略（Bumper 的 L2C 即 DRRIP）。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
