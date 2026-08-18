## 陪集集成解码（Coset Ensemble Decoding）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
陪集集成解码是本论文（ISCA 2026）提出的算法-硬件协同解码器：在 UF 式聚类之上，对逻辑等价陪集做"集成森林探索"近似——用 K 个独立随机优先级采样（keyed priority φ(v,e)=HashToUnit(seed,i,v,e)）为同一聚类结果生成 K 棵确定性优先级森林，每棵森林经逆序消元（ROE）剥除得到一个候选纠错 E_i 与逻辑错误 L_i，最后按逻辑结果多数投票（限定在最小 |E_i| 候选子集上）选出最频逻辑陪集。定位：介于 UF（快而糙）与 MWPM（准而慢）之间，解"聚类约束下的 sub-optimal coset ML"。K 是可调精度-资源旋钮。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Require: syndrome s; G=(V,E); K; seeds        # Algorithm 1
1: Ĝ ← CLUSTERING(G, s)                        # Phase I：UF 式聚类
2: E ← ∅, L ← ∅
3: for i = 1..K:                               # Phase II：集成森林探索
4:   for (v,e) ∈ Ĝ: φ(v,e) ← HashToUnit(seed,i,v,e)
5:   (parent, σ) ← PRIORITYFORESTS(Ĝ, φ)       # 按 φ 升序 BFS 建森林
6:   {E_i, L_i} ← ROE(parent, σ, s)            # 逆序剥除
7: Ê ← MAJORVOTE(E, L) on min-|E_i| 子集        # 逻辑结果投票
```
理论依据：Lemma 1——逻辑错误相同的候选互为退化错误、属同一逻辑等价陪集；Lemma 2——聚类把全局 coset ML 松弛为局部优化（B_c 位串空间）；投票频率 p̃(L_i|s)=n_{L_i}/K 估计陪集概率，K→∞ 在划分空间内收敛。效果：K=24、p=0.002、d∈{3..19}：LER 距 MWPM 1.0×（d=3）~2.1×（d=19，增大 K 可缩小）；repetition code 上 1.0–1.4× 与 BP+OSD 相当，远优于 UF；吞吐 1.88 M decodes/s（d=9）~29.8 M（d=3，p=0.001）。可调性：LER(K)=LER_∞+A·K^{−α}（α 从 d=3 的 1.98 降到 d=9 的 0.27），K*=2^{⌊(d+1)/2⌋} 捕获 ~70% 收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/IMSeonL/coset-ensemble-decoder（实测解析到 https://github.com/ihc-fan-lab/coset-ensemble-decoder；Python 实现 + cycle-accurate 硬件模拟器，README 注明 Verilog RTL 稍后在 hardware_code/ 发布）。硬件：两段式架构——7 级流水聚类引擎 + K=24 并行 EFE 实例 + Voting（见硬件架构层条目）。随机源鲁棒性：固定 base seed 的单 stateful PRNG 流即可，低质量 PRNG 下 LER 差异落在 95% 非显著带内。使用场景：需要亚微秒实时、精度高于 UF、且可按负载调节 K 的 FTQC 解码部署。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
