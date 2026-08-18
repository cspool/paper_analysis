## Triorthogonal Matrix（三正交矩阵 / Triorthogonal Codes）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bravyi-Haah 2012（arXiv:1209.2426，PRA 86, 052329）引入：二元矩阵 $G\in\{0,1\}^{m\times n}$ 称为 triorthogonal，当且仅当任意两行的 support 重叠为偶：$\sum_j G_{a,j}G_{b,j}=0\pmod 2$，且任意三行的 support 重叠为偶：$\sum_j G_{a,j}G_{b,j}G_{c,j}=0\pmod 2$。作用：任何含 k 个奇权重行的 triorthogonal 矩阵映射到一个有 k 个逻辑 qubit、允许 transversal π/8 旋转（T 门，可能配 Clifford）的稳定子码；偶权重行给出蒸馏协议中探测输入魔法态错误的稳定子。由 triorthogonal 码可构造 rate 1/3 的蒸馏协议，开销 $O(\log^\gamma(1/\epsilon))$、$\gamma=\log_2 3$。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文用 G 直接定义协议执行：列 c → 旋转 $e^{i\frac{\pi}{8}Z^{\otimes S_c}}$（$S_c=\{r:G_{rc}=1\}$ 为 Z 作用于的行集合），奇权重行 → k 个输出 |T⟩，偶权重行 → m−k 个 X 基 parity check 测量 + postselect。协议压缩（qubit recycling）即对 G 做保持 triorthogonality 的变换：列置换（重排对易旋转）、块内行置换（奇行之间/偶行之间各自换序）、$\mathbb{F}_2$ 行加法；对每行取首/末 1 列 $(f_i,\ell_i)$ 定义工作集 $W(j)=\{i:j\ge f_i\text{ 且 }(\text{偶行 }j\le\ell_i\text{ 或 奇行})\}$，峰值活动 qubit 数 $C(G)=\max_j|W(j)|$ 即压缩目标。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Bravyi-Haah 论文同时给出数值生成 triorthogonal 矩阵的方法。推广：generalized triorthogonal matrices（arXiv:1709.02832，Haah-Hastings? 实际为 "Codes and protocols for distilling T, controlled-S, and Toffoli gates"）支持蒸馏 |T⟩、|CS⟩、|CCZ⟩——本论文的 51-to-3CS、64-to-2CCZ 即属此类。与 CSS-T 码、自对偶码有理论联系（arXiv:2408.09685 等）。本论文压缩算法的最优解 NP-hard（k=0、偶行权重 2 时化简为 cutwidth 问题），实际用贪心聚类行起止 + 定向行加法，<5 s 编译。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
