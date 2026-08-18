## 规范门交换性优化（Canonical Commutativity Optimization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量子线路中两比特门之间的交换性（commutativity）是指两个门在作用于有重叠 qubit 且顺序可交换时，不改变电路功能。传统 CX 编译器只利用"共享控制位或共享目标位的一对 CX 可交换"这一局限模式（需追踪 control/target 位置）。CANOPUS 通过规范表示给出一般化判据（Theorem 1）：共享一个 qubit 的规范门 Can(a,b,c) 与 Can(a',b',c') 可交换当且仅当 b=b'=c=c'=0，即两者都只含 XX 旋转（纯 XX）。该判据无需追踪 control/target，且能捕获 CX 表示下被掩盖的复杂交换模式（Fig.7(b) 的四种模式：如 C-1Q-门-C 序列的等价交换），这些模式在真实电路（算术、QFT、化学模拟）中常见，可用 TKET 变换得到。
- 作用：交换性扩大了路由时可重排的 2Q 门池，让路由器能发现更低成本的 SWAP 插入位置（把待交换门重新排序后做 SWAP absorption），并在维护 L 时记录 commutative pairs C 以正确更新 wire duration D（Algorithm 2）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架内运转流程：① 输入电路 rebase 为 {Can,U3}；② 路由中加新 2Q 门 g' 时（Algorithm 1），对其前驱 2Q 门 pred 用 Theorem 1 判定 isCommutativeCanonicalPair(g', pred)——两者共享一个 qubit 且规范系数满足 b=b'=c=c'=0；若可交换，把 (pred.q0,pred.q1)→(g'.q0,g'.q1) 记入 C，并从 L 弹出被"超越"的门，从而允许 g' 与 pred 在 DAG 中逻辑重排；③ 插入 SWAP 时（Algorithm 2），若 SWAP 的 qubit 对在 C 中，按匹配 qubit 调整 D，使深度计算反映重排后的真实时序；④ 消融数据：开启该优化后平均 Ccount 再降 2-11%、Cdepth 降 2-10%（峰值 37-48%，knn/swap_test 在 chain 上 31-37%）；规范表示下真实电路相邻 2Q 门对的可交换比例接近 100%（Fig.13），远高于 CX 表示。
- 深度计算细节：d = MAX(D[q0],D[q1]) + SYNTHCOST(g) 逐门累加；SWAP 吸收时 d = MAX(D[swap.q0],D[swap.q1]) + SYNTHCOST(can.MIRROR()) − SYNTHCOST(can)（净增合成成本）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：CANOPUS 中 C 数据结构的维护与 Algorithm 1/2 全部基于哈希与条件判断（O(1)）；交换性判定只需比较规范系数。论文用 pytket 得到可交换规范门的变换（"the transformation to commutative canonical gates can be readily obtained using TKET"）。对比：此前工作（[44]）仅支持 CX 对交换，且需追踪 control/target；CANOPUS 的判据是纯系数条件，统一覆盖全部共享 1 qubit 的 2Q 门。
- 适用场景：高密度重叠 2Q 门、非局部交互模式（knn、swap_test、算术电路）收益最大；本身局部连接（ising、wstate）或镜像门集（SQiSW_ 已低开销）收益小。该优化对深度改进尤其重要（Cdepth 增益略高于 Ccount）。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
