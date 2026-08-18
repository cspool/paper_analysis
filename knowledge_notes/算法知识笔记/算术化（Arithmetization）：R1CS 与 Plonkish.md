## 算术化（Arithmetization）：R1CS 与 Plonkish

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 算术化把 statement 的计算约简为有限域上的一组代数约束，是 ZKP 三层视图的第一层。两种常见形式：(1) R1CS（Rank-1 Constraint System）——把全部输入/输出/中间值展平为 witness 向量 w，每个约束（门）强制二次关系 ⟨a,w⟩·⟨b,w⟩=⟨c,w⟩；(2) Plonkish——把变量排成表格（trace），每行一个门含多条 wire（输入 wa,wb 与输出 wc），门约束 qL·wa+qR·wb+qO·wc+qM·(wa·wb)+qC=0（系数为 selector），行间 wire 一致性由置换（wiring）约束保证。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- R1CS 例子（a*b = c 门）：witness w=(1, a, b, c)，约束向量取 a-vec 在 a 位置为 1、b-vec 在 b 位置为 1、c-vec 在 c 位置为 1，则 ⟨a,w⟩=a、⟨b,w⟩=b、⟨c,w⟩=c，满足 a·b=c。Plonkish 例子（加法门）：一行 wa=x、wb=y、wc=z，selectors qL=qR=qO=1、qM=qC=0，约束 x+y−z=0。
- Annotations：R1CS 门数=约束数（Groth16 用）；Plonkish 表格支持自定义高次门（HyperPlonk/Plonky2 用）。稀疏性：Plonkish 的 control selector 天然二元（0/1），非算术操作（比较/位逻辑）把域元素拆成位占满 witness 单元——GenZA 利用这些稀疏性做 sumcheck 延迟绑定与稀疏 MSM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器（如 Circom/libsnark/Jsnark）把高层电路编译成 R1CS/Plonkish 约束与 witness 生成器；prover 侧把 witness 多项式化交给 PIOP/PCS。使用：任何 zk-SNARK 的第一步；选择 R1CS 或 Plonkish 决定后续 PIOP 形态（R1CS→NTT 域线性 PCP，Plonkish→MLE/sumcheck 或 NTT）。硬件影响：算术化决定 witness 的稀疏结构与 kernel 构成，GenZA 按此生成 mock circuits 评估（控制 selector 二元、witness 90% 稀疏）。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
