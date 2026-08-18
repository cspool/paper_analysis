## Bivariate Bicycle Codes（BB 码 / 双变量自行车码：gross 与 two-gross）

术语解释
IBM 于 2024 年提出的高码率 QLDPC 码族（Bravyi et al., Nature 627, 778），是自行车架构的编码底座；本论文的蒸馏工厂全部跑在 gross [[144,12,12]] 与 two-gross [[288,12,18]] 两个实例上。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 构造：取阿贝尔群 $\mathcal{M}=\{x^i y^j\mid i\in[0,\ell-1],j\in[0,m-1]\}$（指标模 ℓ、m）的置换矩阵表示；选 $A_1,A_2,A_3,B_1,B_2,B_3\in\mathcal{M}$，记 $A=A_1+A_2+A_3$、$B=B_1+B_2+B_3$，定义校验矩阵 $H_X=[A\mid B]$、$H_Z=[B^\top\mid A^\top]$（n=2ℓm 列 = 物理 qubit，行 = X/Z 稳定子）。由构造每个 check 支撑 6 个数据 qubit、每个数据 qubit 参与 6 个 check，物理实现为度 6 连通图 + 少量长程连接，天然适合超导芯片。gross 码 [[144,12,12]]（12 逻辑 qubit/144 物理，码率是 surface code 同码距的 ~12×）、two-gross [[288,12,18]]（距离更长、错误率更低）。ZX-duality：一个阶 2 物理置换 + Hadamard 层实现逻辑门，12 个逻辑 qubit 分成两个 6-qubit 块被互换并 Hadamard——这是双轨并行蒸馏的结构基础。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件运转流程（本论文单块工厂）：BB 码块 = 数据 qubit + X/Z check qubit 的环面晶格；逻辑操作经两类硬件原语执行：①shift automorphism 门——把数据 qubit 沿连接图做连续 swap 置换（与 syndrome extraction 共用同一稀疏连接，12 个生成元容错实现、τ=14 时间步、错误 10⁻⁶·⁴（gross）/10⁻¹⁴·⁵（two-gross）@10⁻³）；②LPU 原生测量——挂接 pivot $L_0$ 与对偶 $L_6$，测 $X_{L_0},Z_{L_0},X_{L_6},Z_{L_6}$ 的乘积（τ=120/216，错误 10⁻⁵·⁰/10⁻¹¹）。一个 exp(iπ/8·P) 旋转 = 注入 |T⟩ + 一段 automorphism 序列共轭到 LPU 可测形式 + 单次 LPU 测量。整块 378（gross+90-qubit LPU）/734（two-gross+158-qubit LPU）物理 qubit 内跑完 15-to-1/49-to-1 等协议。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- IBM 路线图：gross 码对应 Kookaburra 处理器（2026 目标）、two-gross 对应 Starling（2028）。解码用 BP+OSD（单一错误触发 >2 个 stabilizer 形成超边，MWPM 不能直接使用；arXiv:2604.07995 证明 BP 收敛可由 syndrome 缺陷数 mod 列权 w 一步预测，AUC 0.995）。逻辑门：牺牲 1 个逻辑 qubit、约 100 ancilla 实现全部 Clifford（Rall, arXiv:2407.18393）。2026 年 IBM 代数级联（quantum Reed-Solomon 外层码包 gross）达 teraquop 区、逻辑错误 4.10×10⁻¹⁵。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
