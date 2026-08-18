## Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：coset ensemble decoding（陪集集成解码）——在 UF 式聚类基础上引入"逻辑等价陪集"视角：K 次独立随机优先级采样生成 K 个优先级森林（Priority Forests），各森林经 Reverse-Order Elimination（ROE，逆序单趟 peeling）产出候选纠错与对应逻辑错误，再按逻辑结果多数投票（MAJORVOTE，限定在最小 |E_i| 候选子集上），以多项式时间近似陪集级最大似然 argmax_L p(L|s)。配套 lossless graph compression（聚类后仅保留根-根、根-边界边）降低 K 次探索的复杂度。属于"提出新算法模型 + 近似求解"类实现。
  - 实验比较：与 MWPM（PyMatching 软件实现，硬件对应 Micro-Blossom）、UF（自研 baseline UF 软件实现，硬件对应 Helios、QUEKUF）、BP+OSD（product-sum BP + OSD-CS order 15）比较：逻辑错误率 LER、解码延迟（均值/p95/p99 分布）、吞吐（decodes/s）、以及论文自定义的系统 infidelity 指标 Ĉ(R)（反馈解码场景下延迟对条件逻辑操作另一方保真度的影响）。
- 硬件平台是什么，配置是什么。
  - 算法级评估：Python-based hardware simulator（镜像最终微架构数据流、逐 cycle 计数、跟踪 multi-bank 布局下访存冲突、可逐项开关硬件优化做消融），与 RTL 交叉验证；硬件实现：Xilinx Virtex UltraScale+ VU19P FPGA，SystemVerilog HDL，Vivado 2024.2 综合，163 MHz，108k LUT / 43k FF / 252 BRAM（d=15）。
- 模型是什么。数据集和bench分别是什么。
  - 模型/码：surface code（rotated，periodic boundary conditions 周期边界，与 QUEKUF 同设定），码距 d∈{3,5,...,19}（精度）、d=3~11（延迟）、d=3~25（资源估算，d=3/9/15 为完整 Vivado 综合）；另有 repetition code（d∈{5,7}，p∈[0.04,0.08]）验证跨码族通用性。
  - 数据集/bench（噪声模型）：Stim 库生成 circuit-level depolarizing noise（Clifford 门后与轮间以 p 施加 depolarizing，测量错误为同概率 p 的经典比特翻转，reset 理想，q=p，T=d 轮）；biased/unbiased phenomenological noise（bias η=p_Z/p_X∈{0.5,1,10}）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文给出 https://github.com/IMSeonL/coset-ensemble-decoder（实测解析到 https://github.com/ihc-fan-lab/coset-ensemble-decoder，公开仓库：Python 解码器 uf_decoder.py、cycle-accurate 硬件模拟器、pipeline.py/find_subgraph.py 等与 4 档复现脚本；README 注明 Verilog RTL 稍后在 hardware_code/ 发布，当前硬件评估走软件模拟器）。依赖开源组件：Stim（噪声电路生成）、PyMatching（MWPM baseline）。
  - 算法 pipeline 执行例子（d 码距 surface code 一个 d 轮 syndrome 任务，syndrome parity s，候选数 K=24，见 Algorithm 1/2/3）：
    ① Phase I Clustering：Ĝ←CLUSTERING(G,s)，UF 式增长-合并把 syndrome graph 划分为若干非平凡子图（把 stabilizer 群划分为局部子空间 B_c={b∈F_2^m | b_g=0 for s_g∉G_c}，把全局 coset ML 松弛为聚类内局部优化，Lemma 2）；
    ② Phase II Ensemble Forest Exploration：for i=1..K：对每个 (v,e) 计算 keyed priority φ(v,e)=HashToUnit(seed,i,v,e)，按 φ 升序对顶点集与各邻接表排序；PRIORITYFORESTS（Algorithm 2）以 BFS 队列按优先级建森林，返回 parent[] 与发现序 σ；
    ③ ROE（Algorithm 3）：for t=|σ| down to 1：x=σ_t, r=parent[x]，若 p[x]=1 则 E_i←E_i∪{(x,r)} 并翻转 p[x],p[r]——单趟线性 peeling，复用森林遍历序，免全局叶子检测与度数重算；
    ④ L_i=DECODELOGICAL(E_i) 得到每个候选的逻辑错误；
    ⑤ MAJORVOTE(E,L)：对最小 |E_i| 子集按逻辑结果投票，采样频率 n_L/K 估计 p̃(L|s)，得最终纠错 Ê（Lemma 1 证明同 L 的候选互为退化错误、属同一逻辑等价陪集；K→∞ 时在聚类划分的候选空间内收敛）。
  - 效果：p=0.002 circuit-level noise、d∈{3,5,...,19}、K=24：LER 与 MWPM 之比从 d=3 的 1.0× 升至 d=19 的 ~2.1×（增 K 可继续缩小），显著优于 UF；repetition code 上 LER 距 MWPM 1.0–1.4×，与 BP+OSD（1.0–1.7×）相当，UF 落后 2.7–5.7×；X-biased 噪声下填补 UF 到 MWPM 差距的 ~94%。吞吐 1.88 M decodes/s（d=9）~29.8 M（d=3，p=0.001），为 Micro-Blossom 的 4–5×。可调性：LER(K)=LER_∞+A·K^{−α} 幂律收敛（α 从 d=3 的 1.98 降到 d=9 的 0.27），K*=2^{⌊(d+1)/2⌋} 捕获约 70% LER 收益。
