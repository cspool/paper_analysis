# 知识库_算法pipeline

## 表面码（Surface Code）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
表面码是当前最主流（尤其超导 qubit 平台）的量子纠错码：把数据 qubit 排列在二维格点上，与两类 ancilla qubit（X 型、Z 型）交错纠缠。每个 ancilla 与相邻数据 qubit 做稳定子（stabilizer）奇偶校验，测量结果组成 syndrome——X/Z 错误改变其邻域 ancilla 的测量值，因此 ancilla 检测的是"错误的边界"而非错误本身：单个 X 或 Z 数据错误触发 2 个相邻 ancilla 产生非零 syndrome，Y 错误触发 4 个；若一个 ancilla 耦合偶数个错误数据 qubit 则奇偶相消报告 0，形成错误链（error chain）。码距 d 是格点每条边的数据 qubit 数（rotated 记法 [[d²,1,d]]，即 d² 物理 qubit 编码 1 逻辑 qubit），距离 d 的码可纠正 ⌊(d−1)/2⌋ 个错误；阈值约 1%（code-capacity 级）。web：低于阈值时每增加 1 码距，逻辑错误率指数级下降（Google Willow 距离 7 码每级约 16× 错误抑制，arXiv:2408.13687）；表面码只需近邻连接，适配平面工艺布线。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
表面码是 syndrome 压缩 pipeline 的输入结构（本论文）。每测量轮对全部 ancilla 做稳定子测量得到一张 syndrome 位图（每位 = 1 个 ancilla 的奇偶结果），该位图按行主序编号即为压缩器的输入位流：
```
# 单轮 syndrome 采样（Stim 内建 surface_code:rotated_memory_z，d=21）
circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                distance=d, rounds=n_rounds,
                                after_clifford_depolarization=p)
syndrome = sampler.sample(circuit)   # shape: [shots, n_detectors]
# 每个检测器 detector = 一个 ancilla 的稳定子奇偶结果，0/1
```
本论文的压缩对象是这张位图中的非零位置（index）及其时空模式：空间上 X/Z 错误产生水平/垂直对、Y 产生 cross，时间上测量错误在相邻轮同位置成对出现。错误链使 ancilla 只在链两端报告非零（d=21 时每轮非零 syndrome 稀疏，p=10^-3 时约 0.1% 数量级非零）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实际实现是量子处理器上的稳定子测量电路：每轮把每个 ancilla 与邻域数据 qubit 用 CNOT 纠缠后测量（syndrome extraction），重复多轮构建三维解码图（2 空间 + 1 时间轴，测量错误成为时间轴上的边）。本论文用法：1000 逻辑 qubit、d=11–31、p=10^-4–10^-2，Stim 运行时生成 syndrome 数据集驱动 IcePack 压缩评估；解码仍在 300 K 全精度执行，压缩保证无损。局限：需近邻连通（qLDPC 类非平面码不适用，但 BB 码等周期结构可适配）。

补充（Coset Ensemble Decoder 论文）：该文用 rotated surface code 但取 periodic boundary conditions（周期边界，与 QUEKUF 相同设定）跑全部算法精度评估（码距 d∈{3,5,...,19}）；Micro-Blossom/Helios 的硬件资源数字取自 rotated 变体原始论文。解码图构建：T=d 轮 syndrome 提取，相邻轮 syndrome 输出 XOR 得到 detector（把测量错误与数据错误隔离），顶点=detector 事件、边=潜在错误，构成 3D 解码图 G(V,E)——它是 MWPM/UF/陪集集成解码的共同输入结构。

补充（Triage 论文）：Triage 用 rotated surface code 做蒙特卡洛 LER 评估——d=9、circuit-level depolarizing noise p=3×10⁻³、Stim 每点 ≥10⁵ runs，外推到 d=21；Triage 的 slice 抽象正是以 d×d surface code patch 为空间单位：slice S(t,p)=一个 d×d 逻辑 patch 在 t 时刻（一个 d 轮 syndrome 测量周期）内产生的 syndrome 数据块，解码按 window-based lattice surgery 分 slice 并行后逐层聚合 LER。解码延迟模型直接来自 pymatching 对 surface code 解码 volume 的实测拟合（t_dec=A·volume^α，α=1.17），单 slice 延迟由其窗口缓冲大小（约束图中未解析邻居数，即 degree）决定。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

补充（O3LS 论文）：O3LS 把表面码抽象为 patch/tile 模型用于晶格手术编译——每个 patch 是 distance-d 表面码（d² 物理数据 qubit 编码 1 逻辑 qubit），四边为 X/Z 型边界（虚线=Z 算子、实线=X 算子，抽象为 patch 上的 X-/Z-边缘）；编译目标是"用最少 tile 与时间步实现算法"（space-time volume）。代表性布局：compact（[34]，顺序放置、按需加列）、sparse（[25]，每数据 patch 与邻居间隔至少一个空 tile、X/Z 边缘均邻接路由空间）、standard（[25]，类 sparse 但放置不同）、以及 O3LS 自动生成的 squeezed 布局（评分函数 S(B)=C(B)×(N_x+N_z−α_e·N_e) 迭代搜索）。评估设定：d=9 表面码、p=10⁻³ 电路级去极化噪声、STIM 表征原子操作、PyMatching 2 解码、分层 LER 模型（p_total≈Σ_t p_layer，PPM/PR/idle 三类错误复合）。

## 错误症候与症候测量轮（Syndrome & Measurement Round）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Syndrome 是稳定子测量的输出模式，标记错误发生的位置但不直接给出错误类型；数据错误只在其发生轮产生非零 syndrome（后续轮保持一致），测量错误则在相邻两轮同位置产生成对的瞬态非零 syndrome——这是解码图上"横向边 = 数据错误、纵向边 = 测量错误"的来源（本论文 Fig. 3）。测量轮（measurement round）指完成一次全体稳定子测量的周期，超导系统典型间隔 ~1 μs，是 QEC 系统时序的基本单位：串行化上传 syndrome 必须在一个测量轮内完成，且从测量到解码到控制的反应时间不能超过 ~10 μs。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文的压缩 pipeline 以 syndrome 位流为输入，以"每轮 index 流"为输出，三类 syndrome 来源对应三类压缩规则：
```
输入: 每轮 syndrome 位图 B_t（ancilla 行主序索引 0..N-1）
输出: 每轮 (index, opcode) 流
# 数据错误: X/Z -> 水平/垂直对（2 个非零），Y -> cross（4 个非零）
#          -> 空间聚类：1 个 index + opcode∈{1,2,3}
# 测量错误: 孤立非零 syndrome，连续两轮同位置复现
#          -> 时间聚类：预测 + 命中丢弃/失败补发
# 错误链:   链两端孤立非零 syndrome -> 时间聚类误预测时补发 index
```
测量错误远比错误链常见（这是时间聚类预测成立的统计前提）；每轮 syndrome 位流中零占绝大多数（p=10^-3 时非零占比 <0.1%），是全零块过滤（PPU）与稀疏编码的直接依据。时序约束：1 μs 测量轮内串行化上限 → IcePack 目标 500 ns（压缩后 300× 数据量减少才能满足）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上每个 ancilla 一轮一 bit（1 次测量 = 1 syndrome bit）；本论文假设数字读出（如 Josephson photomultiplier）下每轮每 ancilla 恰好 1 bit。使用时以轮为粒度流水：round t 的 syndrome 驱动压缩硬件，同时 TCU 用上一轮（round t−1）的预测流做预测对比并生成 round t+1 的预测（存于 PTL 环形延迟线，延迟 = 一个测量轮时长）。300 K 端解码器按轮序重建完整 syndrome 历史。

补充（Coset Ensemble Decoder 论文）：该文把 d 个 syndrome 测量轮 XOR 相邻轮输出形成 detector（detector events 为解码图顶点），并强调实时约束：超导平台上解码器需在 <1 μs（一次 syndrome 提取轮时长）内完成一个 d 轮任务；其系统指标把"解码延迟 R 折算为提取轮数 R=L/l"来量化反馈解码场景的保真度损耗。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## 空间聚类压缩（Spatial Syndrome Clustering）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把单个数据错误引起的多 syndrome 激活模式（spatial cluster）编码为"1 个索引 + 2-bit opcode"：水平对（X/Z 错误，opcode=1）、垂直对（X/Z 错误，opcode=2）、cross（Y 错误，opcode=3），孤立 syndrome 记 opcode=0。本质是复用层次化解码器（Clique、Predecoder）的局部模式规则，但只用于压缩而非解码决策——因此局部视野不会带来精度损失（解压在 300 K 无损完成）。优先级 cross > vertical > horizontal（按 index 减少率 75% > 50% = 50%，本论文 Table I）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SCU（硬件滑窗实现）对行主序位流逐位扫描：
for i in stream:                      # i 为当前 ancilla 索引
    w = 5-ancilla 滑窗(i 及其右/下/对角邻域)  # 2D 邻域映射到时间偏移
    if w 匹配 cross:      emit(i, OP=3); 清除 4 个匹配位
    elif w 匹配 vertical: emit(i, OP=2); 清除 2 个匹配位
    elif w 匹配 horizontal: emit(i, OP=1);清除 2 个匹配位
    else:                 emit(i, OP=0)   # 孤立：留给时间聚类
```
格点二维邻域映射为时间偏移：右侧 ancilla = 下一拍，下方 ancilla = 2d−1 拍之后（d 为码距）——这是 SCU 用移位寄存器/PTL 行缓冲做滑窗的数学基础。效果：仅数据错误时 index 减少 57–61%；加入测量错误后降到 32–35%（测量错误不形成单轮内多 syndrome 簇）。与 AFS 对比：AFS 不处理非零 syndrome，index reduction = 0.0。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
本论文以 PU 的 SCU 单元硬件实现：5-ancilla 搜索窗（行缓冲 + 固定偏移抽头）+ 组合逻辑真值表（Fig. 11b）输出 (opcode, valid)；命中后清除行缓冲中匹配位。边界处可能出现跨边界的假阳性（false positive），在 300 K 解压时无损反转，不需压缩端处理、也不增加编码位数。软件复现（artifact）中由 icepack.py 的空间聚类函数实现同一规则。扩展：换码型只需替换 SCU 匹配的局部模式集合（如 color code）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

## 时间聚类压缩（Temporal Syndrome Clustering / Prediction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对空间聚类后的孤立 syndrome（opcode 0）做时间维压缩：由于测量错误远多于错误链，opcode 0 在下一轮同位置大概率复现（测量错误成对出现），因此对每个孤立 syndrome 预测"下一轮同位置也是 0"并只发 1 个 index；若预测失败（下一轮该位置无 syndrome），补发一个 (index, opcode=0) 条目，300 K 端"未收到 index 即默认该位置有 syndrome"的约定使该补发条目表示"此处无 syndrome"——无损。与层次化解码器（Clique/Predecoder）"等下一轮才能处理测量错误"不同，本方法不增加任何测量轮延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TCU 真值表（本论文 Fig. 11c），P_in = 上一轮同位置预测位：
if P_in == 0:
    if V_in == 1 and OP_in == 0:      # 本轮出现孤立 syndrome
        P_o = 1;  V_o = 1; OP_o = 0   # 发 index 并预测下一轮复现
elif P_in == 1:                       # 上轮预测这里会复现
    if V_in == 1 and OP_in == 0:      # 预测命中（测量错误对）
        V_o = 0                       # 丢弃 index，不发
    else:                             # 预测失败（错误链等）
        V_o = 1; OP_o = 0             # 补发 index，表示"此处无 syndrome"
# 多 syndrome 簇 OP_in ∈ [1,3] 不参与预测（防丢数据）
```
预测流按 index 对齐、延迟一个测量轮循环存储。效果：index 减少从纯空间聚类的 32–35% 提升到 41–55%（p=0.01%–0.1% 时 1.6× 提升），p=1% 时仅 1.3×（错误链多、误预测多）。circuit-level 噪声下调整策略：仅当所有相邻 syndrome 均不活跃时才记录 opcode 0 预测，避免误预测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现（TCU）：SCU 输出流与预测流 P_in（来自 PTL 环形延迟结构，延迟 = 1 测量轮）按位对比，更新 (V_o, OP_o) 并生成 P_o；运行 index 由共享计数器采样，有效条目（index, opcode）写入队列。300 K 解压端用升序 FIFO 存预测，算术比较校验。适用前提：测量错误发生率高于错误链，该条件在可行工作区间内始终成立（本论文 VI-A3 论证）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

## Rice-Golomb 编码（RGE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Golomb 码是面向几何分布整数的最优前缀变长码：对 gap 值 n，取参数 m，编码为两部分——商 q=⌊n/m⌋ 用 unary 表示（q 个 1 接 1 个 0），余数 r=n mod m 用 truncated binary 表示。Rice-Golomb（Rice 码）是其硬件友好特例：限制 m=2^k，商/余除法退化为移位（q = n >> k，r = n[k:0]），余数恰为 k bit。本论文用 RGE 编码压缩后保留 index 之间的 gap——空间/时间聚类去掉了时空相关性后，gap 分布近似独立同分布的 Bernoulli 过程的几何分布，恰是 Golomb 码的理论最优场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 编码（硬件：减法器 + 硬连线移位 + unary 计数器）
gap = idx_cur - idx_prev
q = gap >> k          # 商：unary，q 个 '1' + 1 个 '0'
r = gap & ((1<<k)-1)  # 余数：k bit 定长二进制
code = 1*'1'*q + '0' + bin(r)[2:].zfill(k)
# 例子（本论文）：m=4(k=2)，ID1=632 → ID2=643，gap=11
# q=2 -> 3'b110，r=3 -> 2'b11，码字 5'b11011（10-bit 绝对索引省一半）
# 解码：unary 计数（计数器）-> gap；累加器还原绝对索引
```
m 由错误率决定：最优 m ≈ 0.69×均值（gap 几何分布），RGE 中取最近的 2 的幂。k 是 IcePack 中唯一依赖错误率的参数；错误率漂移 10× 时，k 按最坏端调谐仍保持功能正确（本论文：p=10^-2 端点调 k，p=10^-3 处仍有 1.9× 压缩、只占 21% 带宽；保持最优 3.5× 需 k 可调 3 bit，用 barrel shifter 实现，每 bit 两个 MUX，ENC 共享于数千 qubit 开销极小）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
经典用途：FLAC 音频残差编码、JPEG-LS 预测误差编码、零游程/幅度编码（几何/指数分布数值）。本论文硬件实现（ENC 单元）：减法器求 gap → 移位取 q/r → 计数器 binary→unary；xSFQ 无时钟门实现、sub-GHz 速率匹配电缆；300 K 解压端 counter（unary→binary）+ accumulator（gap→绝对索引），Synopsys DC/Nangate 45nm 综合出 2.5 ns 解码延迟。p=10^-2 时 RGE 贡献最大（2.50×，Table II）。对非 IID qubit（Willow 分布）：按均值 p_mean 调 k，压缩率与理想偏差 <1%（gap 分布在距离平均下仍保持几何）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

## 稀疏表示（Sparse Representation / AFS 症候压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AFS（Accurate, Fast, and Scalable Error-Decoding，HPCA 2022，Das 等）提出的 syndrome 压缩表示：不发送完整 syndrome 位图，只发送非零 syndrome 的索引（以及可选的动态零压缩/几何压缩两种分块跳零方法）。其依据是 syndrome 位高度稀疏（绝大多数为 0）。web 佐证：AFS 的 Syndrome Compression 将 200–2000 Gbps 的解码带宽需求平均降 ~30×；AFS 同时提出 Conjoined-Decoder Architecture（Union-Find，3 级流水，平均解码延迟 42 ns，p=10^-3 时逻辑错误率 6×10^-10）。本论文将其作为主要对比 baseline——AFS 只压缩零 syndrome，对非零 syndrome 零压缩（index reduction=0.0），且未提供任何硬件实现细节。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 稀疏表示（baseline）：逐位扫描 syndrome 位图
for i, s in enumerate(syndrome_bitmap):
    if s == 1:
        emit(i)          # 每个非零 syndrome 固定 log2(N) bit 索引
# IcePack 在同一步骤上的增量：
#   emit(i) 之前先做空间聚类（2/4 个非零 → 1 个 index+opcode）
#   再做时间聚类（测量错误对 → 1 个 index + 预测）
#   最后对保留 index 的 gap 做 RGE 变长编码（替代固定 log2(N) bit）
```
对比基线（本论文）：IcePack 总 bit 数比 AFS 稀疏表示少 2.4–4×（d=21，p=10^-4/10^-3/10^-2 对应 2.79×/3.45×/4.03×，Table II），其中 clustering 贡献 1.61–1.99×、RGE 贡献 1.40–2.50×；比无压缩数字读出少 300×。几何压缩变体粒度粗、低错误率下不如稀疏表示，未被后续工作（Clique、Predecoder）采用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AFS 是 CMOS 解码器架构概念（无硬件实现细节），本论文为其构造了一个"严格更便宜的"流式 SFQ 稀疏表示 baseline 做热负载对比：裁掉 SCU/TCU/ENC/预测存储，仅保留块单元（BU）与优先级选择器；该 baseline 少 37.5–63.4% JJ，但热负载不降反升——电缆每 ancilla 0.1 mW 占主导（JJ 贡献 <2.5%），而稀疏表示每电缆少支持 3.4–4.0× ancilla。结论：压数据量（而非省硬件）才是热负载的关键。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

## Stim（稳定子电路模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stim 是 Google quantumlib 开源的稳定子电路（stabilizer circuit）快速模拟器（GitHub: quantumlib/Stim，Apache-2.0，Craig Gidney，arXiv:2103.02202 / Quantum 2021）：对 Clifford 电路做 Tableau 仿真，采样 syndrome 极快（mega-sampling 用 256-bit AVX SIMD 批量并行采样；2 万 qubit、8 百万门的 d=100 电路约 15 s 分析、~1 kHz 采样）。限制：无非 Clifford 门、仅 Pauli 噪声。带 detector/observable 标注的电路可生成 detector error model（解码图/Tanner 图），是 QEC 研究的标配工具。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Stim 是本论文 syndrome 数据生成器与评估前端：
```
# 生成 d=11..31、p∈{1e-4,1e-3,1e-2} 的表面码 syndrome（artifact icepack.py）
circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                 distance=d, rounds=R,
                                 after_clifford_depolarization=p)   # 现象学噪声
# circuit-level: after_reset_flip_probability / before_measure_flip_probability
sampler = circuit.compile_detector_sampler()
shots = sampler.sample(n_shots)     # 每 shot = R 轮 detector 位图
# -> 送入 IcePack 压缩 emulator（空间聚类→时间聚类→RGE）
# -> 输出 reduction_rge（与 reference/ CSV 对比，误差 <0.1）
```
评估配置（本论文）：每个 d–p 对 20000 次独立运行、跨多轮；1000 逻辑 qubit；artifact 以 Docker 跑 icepack.py + artifact.ipynb，产出论文图 5/7/8/15 的 CSV 与 PNG。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Python 接口（pip install stim）：stim.Circuit.generated 内建表面码电路、compile_detector_sampler 批量采样、detector_error_model 生成解码图。本论文用法：现象学与电路级两种噪声模型、非 IID qubit（Willow 检测概率分布生成 10 组 × 10 万 ancilla）、错误率漂移、burst errors 场景，全部以 Stim 采样为数据源；还以 Stim 生成的 syndrome index 分布驱动 10 万周期队列仿真求 99 分位延迟。采样随机性导致每次运行数值不同，但 reduction 比例与 reference 相差 <0.1（artifact 自检标准）。

补充（Coset Ensemble Decoder 论文）：该文用 Stim 生成 circuit-level depolarizing noise 表面码电路——depolarizing 以 p 施加于 Clifford 门之后的数据 qubit 与相邻轮之间，测量错误建模为同概率 p 的经典比特翻转，reset 理想，q=p、T=d 轮；biased/unbiased phenomenological noise（p_X/p_Z，bias η=p_Z/p_X∈{0.5,1,10}）亦经 Stim 生成。生成的 syndrome 数据驱动 Python 硬件模拟器评估 LER 与 cycle 计数，并与 RTL 交叉验证。

补充（TUSQ 论文）：该文用 Stim 内建 rotated surface code memory 电路做 DFTT+Caching 性能恢复分析——`stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=R, after_clifford_depolarization=p)` 生成 26/64/118 物理比特（d=3/5/7）、p∈{10^-2,10^-3,10^-4} 的电路，d 轮测量即 d 个 non-invertible 通道；每电路采样 1M 次，统计树遍历操作数（单比特矩阵向量乘=1、双比特=4、非幺正边前向=1、反向=0）求性能恢复 α(K)。结论：容量 3 的 LIFO 缓存恢复 60%-100% 的 DFTT 性能。TUSQ 还用它说明 FTQC 逻辑级模拟的配套角色：物理层 Clifford 电路用 Stim（多项式可扩展）、逻辑层非 Clifford 深电路用 TUSQ（DFTT+Caching 支持 MCM）。

补充（Triage 论文）：Triage 用 Stim 做窗口化 lattice surgery 的 LER 蒙特卡洛——d=9 rotated surface code、circuit-level depolarizing noise p=3×10⁻³、每点 ≥10⁵ runs（Memory Experiment 式逐层 syndrome 生成，同步失败时插入 idle 层再模拟该层 syndrome），先得到 d=9 的逐层 LER 再外推到 d=21 聚合总 LER；Stim 也是解码器延迟校准的数据源之一（pymatching 在 Stim 生成的 rotated surface-code 电路上按 shot 测延迟，15K shots/设置，拟合 log-normal 抖动参数）。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

## 噪声模型（Phenomenological / Circuit-level Noise）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QEC 模拟的三级噪声抽象（越来越真实）：(1) code-capacity——只假设数据 qubit 以概率 p 出错；(2) phenomenological——数据 qubit 错误 + syndrome 测量结果按概率翻错，但不模拟具体测量电路（本论文主分析用此模型，与 AFS、Predecoder 一致）；(3) circuit-level——症候提取电路每一处（门、idle、初始化、测量）都可能出错。web：现象学模型 MWPM 阈值 ~2%（新研究最高 ~6%），circuit-level 阈值 ~0.5–0.7%（新研究到 ~1.4%），因为电路级错误位置多出 ~5×；p=10^-2 高于 circuit-level 阈值，不是可工作点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
噪声模型决定 syndrome 的时空统计结构，进而决定压缩参数与策略（本论文）：
```
phenomenological:  数据错误 p + 测量翻转 p
  -> 非零 syndrome 模式干净：X/Z 对、Y cross、测量错误对（相邻轮）
circuit-level:     5p 测量噪声+2p reset / 2p 测量+1p reset（两配置）
  -> 中途出错产生虚假 opcode 0 -> 策略调整：
     "仅当所有相邻 syndrome 不活跃时才记录 opcode 0 预测"
  -> 空间+时间聚类仍去 34% index，加 RGE 达 2.1–3.1×（vs AFS）
```
VI-A 其余场景：非 IID qubit（Willow 检测概率分布）、错误率漂移 10×、multi-bit burst errors（16 ancilla 区域、约 3 亿轮一次）、leakage（经 LRC 后类 circuit-level 特征）——论文逐一论证 IcePack 在每种噪声特征下无损或可忽略损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 Stim 的噪声参数化：after_clifford_depolarization（现象学数据错误）、before_measure_flip_probability / after_reset_flip_probability（测量/复位噪声，组合即 circuit-level）。使用时注意 p 上界：现象学可到 10^-2，circuit-level 需排除 10^-2（超阈值）。噪声模型选择影响压缩评估的结论：IcePack 在 circuit-level 下相对 AFS 仍有 1.9–3.1×，证明压缩方法对噪声模型鲁棒。

补充（Coset Ensemble Decoder 论文）：该文在两种模型下同时评估解码器精度以证明通用性——circuit-level depolarizing（p=0.002 固定，d∈{3..19}）与 biased/unbiased phenomenological（repetition code，d∈{5,7}、p∈[0.04,0.08]，bias η=0.5/1/10，X-biased 下 vanilla UF 落后 MWPM 6.2× LER）。biased 噪声改变 syndrome 各向异性，是该文验证"陪集集成解码填补 UF-MWPM 差距"（~94%）的载体。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## ML 模型权重解压（decompress+execute，压缩权重执行流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对稀疏化 + 量化的 ML 模型，推理时在线把压缩存储的权重 tile 解压成稠密矩阵再交给矩阵单元计算（"decompress+execute"）。存储层用稀疏（结构化/非结构化）+ 量化（低位宽）压缩权重以省内存与带宽；计算层解压出的 tile 直接喂给核内矩阵单元（如 Intel AMX/TMUL）。ATX 论文把它作为第四个评测 kernel：DECA-like NCA 从内存读压缩 tile、解压后写回核寄存器，核立即用 AMX 对解压 tile 做 GeMM——这是核与加速器**双计算**、细粒度交错的代表用例（此前三个 kernel 都是加速器主算、核只做控制）。任务输入仅 512B–2KB，任务产出被下一环节（AMX 指令）直接消费。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
流水线（论文图 19 场景，按压缩因子 CF 扫描）：
```
for tile in model_weights:                  # 权重按 tile 分块
    task = {VAccId, compressed_tile_addr, tile_shape, CF}
    ATX V1T2(task) → NCA(DECA-like) 解压   # 输出进 1-2 个 1KB tile 寄存器
    AMX_TMUL(decompressed_tile, activations) # 核立即消费解压结果
```
执行节奏：NCA 解压任务与核 AMX 计算交错进行——一个 tile 解压的同时上一个 tile 在做 TMUL；解压任务小（512B–2KB 输入）意味着高频的核↔加速器往返，调用开销成为关键：论文测得 ATX NCA 较 core-only 4.0×、ICA 1.8×、L2 OCA 3.9×、LLC OCA 18×（18× 正是小任务下 OCA 串行调用 + fence 开销的放大）。软件基线是 libxsmm 的 decompress+execute kernel（AVX512 + AMX）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
生产形态：权重稀疏化（如 2:4 结构化稀疏）与量化（INT8/INT4）后，推理库（libxsmm、PyTorch 后端）在计算前做反量化/稠密化；专用硬件如 DECA（MICRO'25）做近核解压器，配合 TEPL ISA 扩展支持乱序调用隐藏通信延迟，并配 3D roofline（Roof-Surface）性能模型。ATX 的使用方式：解压任务经 ATX 指令与 UTE 流引擎调度，解压结果直接进 tile 寄存器供 AMX 消费，省去"解压写内存 → 再读回"的往返。适用条件：压缩权重模型推理、带宽敏感（HBM）平台、任务粒度小且与核计算紧耦合；若解压本身可完全批量离线完成，则不需要在线流式交错。GPU 变体（Approaching Shannon Bound 论文）：把该模式升级为"压缩权重执行原语"——rANS 熵编码 tile 常驻全局内存，解码 warp 按 GEMM tiling 序解压直接写 shared memory（不落全局内存）、GEMM warp 经 tensor core 立即消费，双缓冲流水重叠；基线对照为 NeuZip/DFloat11 的层粒度 decompress-store-compute（整层解压写回全局内存再 GEMM，无重叠、有层同步屏障）。

涉及论文标题：
- ATX: Accelerator Task Extensions
- Approaching Shannon Bound with Lossless LLM Weight Compression

## MoE（Mixture-of-Experts，专家混合架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE 是稀疏激活的 Transformer 变体：把稠密 FFN 层替换为 N 个并行的"专家"（小型 FFN）+ 一个 gate（路由）网络。每个 token 经 gate 打分选出 topk 个专家（DeepSeek-V3 topk=8，带 bias 项调节负载），只在这些专家上执行 FFN 计算（GEMM-1：hidden→moe_hidden；GEMM-2：moe_hidden→hidden），topk 个输出按 gate 权重加权求和，激活计算量约为稠密模型的 k/N，从而在参数规模大幅扩张的同时控制算力开销。代表模型：DeepSeek-V3（256 路由专家 + 1 shared，MoE hidden 2048）、Mixtral 8x7B、GPT-OSS-120B、Qwen3-235B。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每层 MoE 前向（token 粒度）
scores = gate(x)                       # (S, N) 路由打分
topk_idx, topk_w = topk(scores, k)     # 每 token 选 k 个专家与权重
y = 0
for (e, w) in zip(topk_idx, topk_w):   # 各 token 的专家子集动态不同
    x_e = dispatch(x, e)               # EP 下把 token 发往专家 e 所在 GPU
    h_e = W1[e] @ x_e                  # GEMM-1
    o_e = W2[e] @ act(h_e)             # GEMM-2（激活函数视模型而定）
    y += w * combine(o_e)              # 加权聚合回 token 所在 GPU
```
Annotations：S=序列长度，N=专家数，k=topk。Dispatch/Combine 在单卡是局部 gather/scatter，在专家并行（EP）下变成跨 GPU 通信算子（见 kernel 调度层条目）。本论文在算法层的唯一改动：把加权求和 w×o 提前到 GEMM-2 的 epilogue（输出写回前先乘 gate 权重），使 Combine 可以退化为交换机内的无权重归约，算法语义保持不变（最终 y = Σ_i w_i·o_i 等价）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练/推理框架（Megatron-Core、DeepSpeed-MoE、HuggingFace Transformers 等）实现 gate、专家 FFN 与负载均衡（aux loss 或 bias 调节）。分布式下专家按 EP 切分到多 GPU，通信库（DeepEP、Tutel、FasterMoE）实现 Dispatch/Combine。token 分布：训练近似正态（本论文取 std=0.032，源自 ByteDance 对典型训练任务的观测；COMET 同设置），推理近似 power-law（α≈1.5，本论文灵敏度分析取 0.5-2.5）。本论文评估配置：DeepSeek-V3 Large（hidden 7168、MoE hidden 2048、128 注意力头、256 专家、topk 8）+ Small/Medium 扩展 + GPT-OSS-120B / Qwen3-235B。

DIAMoND 的边缘部署补充视角（ISCA'26）：边缘跑 MoE 的三重挑战——(1) 全量专家权重超边缘 DRAM 容量（Mixtral-8x7B INT8 需 47GB，全模型 3.7~7.5× 于激活模型），须驻留 SSD（密度 20.27~29.18 Gb/mm²、1Tb 容量）；(2) 批量 1 解码带宽敏感：以 Apple A18+LPDDR5X+NVMe 为例，单专家单层从 DRAM 到 NPU 的加载延迟是单 token 解码计算的 1.55~3.84×，SSD→DRAM 再加 15×；(3) 路由动态选专家（前向传播中才确定），权重无法预载、按需加载加剧带宽瓶颈。路由器实现细节：router = linear 层对 self-attention 输出打分 + top-k 选择（Mixtral top-2/8 专家；专家 FFN = Up/Gate/Down 三投影，隐维 4096、专家中间维 14336）；GRIN-MoE 的 gate 结构特殊——把 top-k 之外多数专家的路由权重置零（因此 DIAMoND 的 expert similarity 指标不适用于 GRIN）。DIAMoND 对路由的算法级改动（冲突感知的专家替换）见 Adaptive Expert Selection 条目。

Lit Silicon 补充视角（ISCA'26，MoE 训练的通信特征）：MoE 用专家并行（expert parallelism）把不同专家分配到不同 GPU，引入 all-to-all 通信把 token 路由到对应专家。与 AG/RS 不同，MoE 的 all-to-all 通常不与计算重叠（每层同步），因此每次同步点都重置 lead、产生比 dense 训练更小的 lead 值，但 token 分布不均导致偶发高延迟通信 spike（极大 lead 值），更难以分类 leader/straggler。论文用 AMD 推荐平台 Primus + torchtitan 后端训练 DeepSeek V3 16B（8 路专家并行，padding GEMM 使 MoE 权重计算均衡），比较 dense（Llama 3 8B b2s4）与 MoE 训练的功率/频率特征——两者相似，功率重分配算法仍收敛到稳定功率分布并取得与 dense 相当的节电。

- M100 补充视角（ISCA'26，车规自动驾驶 MoE 推理）：MindVLA 是理想汽车（Li Auto）下一代端到端 AD 模型，其 LLM 组件采用 MoE 策略（8 个专家）以提升模型容量与推理效率，评估使用 431M 参数配置。在 M100 NPU（12/14 cluster）上 MindVLA LLM 组件 decode 0.1ms vs Thor-U 0.3ms（3×）、prefill 0.84ms vs 1.74ms（2.1×）——数据流架构以 tensor 级指令 + TPB 功能单元数据流执行 MoE 的专家前向与加权聚合（论文未详述 MoE 的 dispatch/combine 在 M100 上的 kernel 级映射）。
MoE-Hub 补充视角（ISCA'26，MoE 通信的硬件化视角）：MoE 算法层的五步结构（routing → All-to-All dispatch → 专家计算（两次 GEMM + 中间激活）→ All-to-All combine → 加权缩放）在 EP 下每步都可能成为瓶颈：routing 结果动态变化使 dispatch 的目标集/消息大小运行时才确定，与 GPU address-centric 通信模型冲突，迫使软件先做地址解析再通信。MoE-Hub 不改算法语义（Top-K 路由、专家 GEMM、加权聚合均不变），只把 dispatch/combine 的地址解析与数据流编排下沉到 GPU hub 硬件：算法层每层仍是五步，但 dispatch 变成"路由结果一出来就发 st.rowsp"，combine 时源信息作为 expert 输入张量的一列随 token 一起传输（.nop 优先级），专家 GEMM 末尾读源元数据用常规 store 发起 combine，缩放照常加权求和。评估用三个代表模型：Mixtral 8x7B（32 层、Hidden 4096、FFN Hidden 14336、TopK/Experts 2/8）、Qwen2-MoE-2.7B（24 层、2048、1408、4/64）、Phi-3.5-MoE（32 层、4096、6400、2/16），total token = SeqLength×NGPU（128-32768）、token 分布 std 0-0.05（典型训练 std≈0.032）；结果 MoE 层 1.40×-3.08×、端到端 1.21×-1.98× 相对 SOTA，达理想 MoE 层 96.8%，且与负载均衡/TP+EP 等计算侧优化正交可叠加。

专家选择模式补充视角（ISCA'26，Patterns behind Chaos，首个 200B-1000B 四模型数据移动 profiling）：对 DeepSeek-V3（671B）、Llama4-Maverick-128E（402B）、Qwen3-235B、Kimi K2（1000B）用 SGLang（8×H100 DGX + 8×H200 AWS）采集 >24,000 请求、150GB+ JSON 专家选择 trace（开源 https://huggingface.co/datasets/core12345/MoE_expert_selection_trace）。时序模式（temporal）：(1) layer 级——相邻层专家共激活相关（top 20% 下一层候选覆盖 50%/65%/77%/56% 条件概率质量，Llama4 最强 DeepSeek 最弱；Qwen 热图比 DeepSeek 亮，相关性更强）；(2) token 级——同层相邻 token 倾向于选同一专家（高层的对角线模式），top 20% 下一 token 候选覆盖 47%/62%/80%/53%；(3) prefill-decode 级——两阶段专家对热图与专家频率分布高度相似（Spearman ρ≥0.7 强相关，top-5/10/20 重叠 60%/75%/90%）。空间模式（spatial）：(4) 单专家激活偏斜——部分专家激活频率超均值 16 倍；(5) 专家对共激活 affinity——DeepSeek/Qwen 的 top 10% 专家对贡献 60-80% 激活（Llama 每层只选 1 专家无共激活；DeepSeek 因路由限制只路由到相邻节点，共激活形成方块簇）；(6) 任务/语言影响——57 个 MMLU 学科的热门专家横线重叠但差异显著，中英相同题目下热门专家仅 2 个重叠（Insight 6：任务感知 serving 可提前迁移/复制专家）。这些模式证明"专家选择看似随机实则可预测"：时序可预测性支撑单单元策略（预取/缓存/复制，Insight 1/2），空间可预测性支撑多单元策略（放置/去中心化/分离，Insight 3/4/5）。

RoCC 补充视角（ISCA'26，专家并行 AllToAll 的 ROP 卸载）：RoCC 论文把 MoE 专家并行的 AllToAll（token dispatch/combine）纳入 CC 评估（A2A workload：每个专家与所有其它专家交换 token 的压力测试），并把 AllToAll 分解为最简 primitive（send→recv）在 ROP 上执行。区别于前述"大模型需 tensor 并行频繁同步"的动机，MoE 的 AllToAll 通信量动态变化（每轮 token 路由不同）、通常无法与计算重叠；RoCC 因 ROP 近内存、SM 全容量算专家 GEMM 且 warp 级细粒度重叠，对 AllToAll 大消息 CC-only 延迟达 25% 加速。
STEP 补充视角（ISCA'26，空间/时间局部性驱动的静态-动态混合 MoE 推理优化）：STEP 对 MoE 的算法级理解是"专家选择同时具有空间不均与时间连续"——(1) 空间：层内 top-k 低排名专家（如 top-4 的第 3/4 名）平均路由权重常 ≤0.05，贡献极小却照常进入加载与计算（Insight 1，Fig.3）；(2) 时间：长序列生成中一小撮专家被连续 step 反复选中（Insight 2，Fig.4），且跨任务/层/序列的激活模式差异大（Insight 3，Fig.5）。据此把每层结构从固定 j shared + k routed 改造为"离线层内剪枝（k_l 按归一化路由权重阈值 θ 下降）+ 在线窗口投票选举 top-c 高频专家为临时 shared（j+c shared + k−c routed）"，并用 token 感知自适应窗口（th_s=75%/th_f=40%/τ=3-4）动态调投票与预取跨度。加载时间模型 T_load = S·Σ_l(k_l − p_l·R_l)·t_expert（Eq.1，S=解码步数、k_l=层 l 激活 routed 数、p_l=预取数、R_l=预取命中率、t_expert=单专家加载时间）。效果：Mixtral 8x7B 平均专家数 2→1.75 时 MMLU 77.3→77.0（几乎无损）；prefill/decode 相对 llama.cpp 最高 3.12×/2.22×；与 MoE-I2 压缩、APTMoE 卸载正交可叠加（Table V/VI）。

- STAGE 补充视角（ISCA'26）：STAGE 把 MoE 作为一类模块模板建模（Gshard/Switch Transformer 的 MoE、DeepSeek-MoE 的 MoE+Shared Experts），在符号张量图中用专家激活直方图描述每层各专家被 token 激活的概率分布（默认均匀，可覆盖为自定义统计）；EP 下 token 经 AllToAll 路由到专家所在设备（dispatch/combine 两轮）。验证覆盖 Mixtral 8x7、DeepSeek-MoE 8E/144E 等：算子时间误差 3.0%~15.0%、通信量误差 0.945%~2.755%；指出真实训练 micro-batch=1 时部分专家不激活、与默认"全专家激活"假设产生偏差。

Understanding Inference Scaling 补充视角（ISCA'26，MoE 的推理特性与并行选择）：DeepSeek-R1-671B 作为稀疏 MoE 推理代表——总参数 671B 但每 token 只激活 ≈37B（激活参数比 ≈1/18），配合 MLA 压缩 KV。论文的核心推理侧发现：(1) 低激活参数量使计算-通信比远低于 405B 密集模型，TP=8 下高频 all-reduce 同步成为瓶颈（GPU 两次同步间计算时间短、同步开销无法摊销），故纯 TP=8（2047s）劣于 PP=4+TP=2 混合（1663s）；(2) MoE 的 routing 与 pipeline 同步延迟使 HBM 带宽利用率仅 ≈50–60%（对比 8B 密集 ≈85%）——frontier MoE 受同步与路由延迟约束而非原始带宽；(3) MLA 压缩的 KV 让 PP 每 stage 可容纳更多 micro-batch、填满 pipeline bubble——稀疏架构与 PP 天然协同。
涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- Rearchitecting the Datacenter Lifecycle for AI

IroKnight 补充视角（ISCA'26，MoE 层的细粒度规则访问作为加密范式的基础）：IroKnight 用 MoE transformer 层作全文运行示例——一个 MoE 层含三个子块：(a) self-attention = 两个 MatMul（Q·K^T、softmax 输出·V）+ softmax；(b) expert router = MatMul + softmax + top-k 选择；(c) expert FFN = 同一批 tiled/vectored 算子（MatMul、sigmoid 等）。关键观察：这些算子全部是细粒度 tiled/vectored 计算、地址呈仿射 walk（SIMD stride=1 扫描、vector reduction 的 stride=SIMD 宽度），因此每个算子都能由编译器给出基址/stride/offset 并提前预计算加密 pad——这是 Fully-State Encrypted Acceleration（所有存储保持密文、明文只在 ALU 组合逻辑瞬态）成立的前提。评估中 MoE 模型 GPT-OSS-120B（16 NPU）与 Llama4-Scout（16 NPU）经 Microsoft DeepSpeed 的 tensor-parallel + expert-parallel 跨 NPU 分发（芯片间经 RSA 私密通道交换 AES-GCM key 交换加密中间数据）；加密变体端到端运行时 0.2%、认证变体 3.3%（图 6/7 输入输出 token 256→4096 扫描），token pruning 0-80% 下加密 0.3%-0.7%、认证 3.3%-3.7%。
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

## Vertex-centric 图处理模型（顶点中心 / think-like-a-vertex）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- "像顶点一样思考"的图计算编程模型（Google Pregel 首创，BSP 超步迭代；后续 Giraph/GraphX/PowerGraph、GPU 侧 Gunrock/Ligra 沿用）：每个顶点维护自身状态与邻接边，每轮超步并行执行同一顶点程序——顶点先处理出边产生发给邻居的更新消息，再读入上一轮收到的更新消息修正自身状态，直至无消息发出（投票停机）。论文语境：机器人图算法（BFS、单源最短路 SSSP，用于 MoveBot-PRM 的 Gunrock/Ligra）自然符合顶点中心模型——A*/BFS 维护 frontier 节点队列，本质是"顶点处理出边队列 → 邻居收更新"的两阶段迭代，两阶段都是队列操作，因此能直接映射到 Morpha Core 的 queue-centric SIMD（graph morpha 的 MIMD-over-SIMD：每核用自己的 SIMD 指令流处理分配的顶点子集，跨核顶点更新走共享队列 + REMOTE_STORE，顶点→核映射静态）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 通用 SSSP 顶点程序（Web：GraphScope 文档）：
  ```
  def VertexProgramForSSSP():
      msgs = ReceiveMessages()               # 上一超步的入边更新队列
      m = Reduce(msgs, MIN)
      if dist > m:
          dist = m                            # 更新自身状态
      for n in neighbors:
          if dist + w(edge) < n.dist:
              SendMessage(n, dist + w(edge))  # 发出边更新
  ```
- 论文给出的 Morphatron 指令版（processEdges 第一阶段）：
  ```
  INIT_Q q0, FALSE; INIT_Q q1, FALSE; INIT_Q q2, TRUE   # q2 为共享队列
  ADD edge_load_offset, offset_queue.PEEK(), edges_addr
  LD_Q q0, edge_load_offset, offset_queue.POP()          # 标量段装载出边
  SYNC code_block_end; SYNC exe_start, SIMD
  Q_LOOP_UNTIL_EMPTY q0, 3:
      POP_Q src_queue, src
      ADD new_val, src.val, q0.POP()
      PUSH_Q q1, (src.dst, new_val)                      # 更新打包入队
  Q_LOOP_UNTIL_EMPTY q1, 1:
      REMOTE_STORE graph, q1.POP(), q2                   # 写入邻居的共享入边队列
  SYNC exe_end
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现：分布式（Pregel/Giraph，消息传递 + 超步 barrier）或单机 GPU（Gunrock 用 frontier/advance 抽象、Ligra 用 push/pull 方向切换）。局限：每超步信息只传播 1 跳，power-law 图上收敛慢（Web 综述）。本文用法：把"顶点内计算"与"顶点间更新传播"都看作队列操作，落到 queue-centric SIMD 硬件；跨核经电路交换互连 + 共享队列完成，两阶段间全体核 standby 让互连充当更新写入的存储介质。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics

## RoWild（端到端机器人基准套件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RoWild（Robotics in the Wild）是 CMU RoboArch 组的开源、跨平台端到端机器人基准套件，源自 "Agents of Autonomy"（SIGMETRICS/PERFORMANCE 2024，DOI 10.1145/3652963.3655043）。它建模 29 个工业机器人实际软件管线，用有限集通用任务（场景理解、路径规划、状态估计、定位建图等）组合出六个端到端应用：DeliBot（粒子滤波定位 + Raycast）、PatrolBot（Kalman 滤波 + YOLOv10 目标检测）、MoveBot-PRM（Gunrock/Ligra 图处理 + nanoflann 最近邻 + PRM 运动规划 + CCCD 碰撞检测）、MoveBot-RRT（nanoflann + RRT + CCCD）、HomeBot（点云 SLAM）、FlyBot（Octree 碰撞检测 + OSQP MPC + A* 路径规划）。与 kernel 孤立的传统基准不同，它保留真实机器人管线的跨域结构（感知→定位→规划→控制循环），CPU/GPU 基线为平台优化的 state-of-the-art 实现。开源：C++、MIT 许可（https://github.com/cmu-roboarch/rowild ，项目页 https://cmu-roboarch.github.io/rowild/ ）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 FlyBot 的管线为例（论文 Table I）：碰撞检测（Octree 树遍历，data-dependent irregular 指针追逐）→ MPC（OSQP 求解：稠密矩阵乘 + 变长向量更新，fine/structured 并行混合）→ 路径规划（A*：frontier 队列，data-dependent semi-regular）。论文用 RoWild 统计四类访存模式与四类并行度的运行时占比（Fig. 1）：regular 线性代数在 6 个机器人中 4 个不占主导，semi-regular 与 data-dependent 在 3 个中占主导；无单一并行形态主导——这是"加速器多态"的动机证据。指标：端到端 latency（论文用其做加速比与 PPW 的归一化基准）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：跨平台编译（低端嵌入式 CPU 到高端服务器 GPU），面向系统/硬件研究（评估缓存、预取、向量化）；可模块化配置任务/算法/参数；在 RTRBench 基础上扩展。本文用法：作为 Morphatron 与 ARM/Xeon/Orin Nano/RTX 3090 对比的统一 workload，并把六个应用按算法域映射到五种 morphas。论文指出其局限：只覆盖机器人算法子集、规模中等、未充分代表学习式方法的增长。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics

## Shannon 熵与 Shannon 极限（源码编码定理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shannon 熵 H(X) = −Σᵢ pᵢ·log₂pᵢ 是离散随机变量 X 的平均信息量（自信息 −log₂pᵢ 按概率加权）。Shannon 源码编码定理（无噪编码定理，1948）确立无损压缩的比特数下界：对任意前缀码平均码长 L ≥ H(X)，且存在码使 L ≤ H(X)+1；对 n 个 i.i.d. 符号，n 足够大时可压到 n·H(X) 比特而几乎必然无损，少于 n·H(X) 则几乎必然有损——H(X) 即"无损编码的 Shannon 极限"，与具体编码算法、数值格式、硬件布局无关。符号码等长只在 p=2⁻ᵏ 形概率下才可能（Huffman 离熵界的差距即源于此）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LLM 权重压缩中的用法（论文附 B）——把每个权重张量视为离散信源、逐层估计熵：
```
for l in layers:
    symbols = weight[l] as discrete codes        # 按数值格式取符号
    p_i = histogram(symbols) / |W^(l)|           # 经验分布
    H^(l) = -sum_i p_i * log2(p_i)               # 逐层熵（bits/weight）
H_model = sum_l H^(l) * |W^(l)| / sum_l |W^(l)|  # 按参数量加权平均
```
Annotations：|W^(l)| 为层 l 参数量；H_model 即任意无损编码下的 bits/weight 下界。论文实测：bf16 名义 16 bits 但熵仅 10–12 bits（冗余 4–5 bits/weight，约 1.5× 空间）；int8 熵 4–5 bits；int4 熵仅 0.6–1.0 bits（熵比 6–10×）；sq8/awq4 仍有 1.1–1.3× 冗余——由此证明无损压缩理论上最多 10× 空间。论文同时以 H 为基准对照 ANS 码率：实测与熵界差 0.01–0.05 bits/weight。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：直方图统计 + 熵求和（numpy/GPU 均可）；关键工程点是逐 tensor（而非全模型）建直方图——LLM 权重分布跨层差异大，层共享 codebook 是统计覆盖与元数据开销的折中。注意经验熵受样本量与符号表大小影响：bf16 的 2¹⁶ 符号表使有限精度 ANS 表（b=12）产生约 0.1–0.2 bits 定标偏差。使用：评估量化/压缩方法剩余冗余、为无损编码选型提供下界（本论文据此选 ANS）；可作为"无损 vs 有损"正交性论证的信息论依据。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices


（补充：Shannon 熵在 RNG/TRNG 评估中的用法——μRNG 论文把 Shannon 熵与 min-entropy 作为熵估计弱点测试）在随机数发生器评估语境下，Shannon 熵度量"无先验知识的盲猜攻击者"猜中下一输出的平均不确定性（H = −Σ p(xi)·log₂p(xi)，对所有 m 个样本求和）；而 min-entropy = −log₂(max_i p(xi)) 度量"拥有 RNG 历史先验知识的最强攻击者"猜中最可能输出的最坏情况不确定性。对输出 n-bit 完美均匀随机数发生器，两者都达理论最大值 n。µRNG 用二者量化 RNG 输出的"非均匀性"（TRNG 语境即"操作噪声的下降"）：在环境 corner（温度/电压）变化时观察熵的下降以暴露熵源退化。SRAM 熵源实测：名义条件 4KB 上电态每 bit 熵 0.149；-68°C 数据保持使熵崩至 0.004；+85°C 快速爬坡熵升至 0.108（但被布局偏置的 Moran's I 条带化掩盖）。RO 熵源实测（图 6）：8-bit 块熵随采样时间增大后饱和（jitter 随振荡波形多次穿越累积），低温+高电压熵最高，高温+低电压熵最低。

## Asymmetric Numeral Systems（ANS / rANS / tANS / FSE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANS 是 Jarek Duda 提出的熵编码族（arXiv:1311.2540）：用单个自然数状态 x 编码整个符号流，兼有 Huffman 的编码速度与算术编码的压缩率（接近 Shannon 熵）。rANS（range variant）：编码 s 时 x' = ⌊x/fs⌋·R + (x mod fs) + cs（fs 为归一化频率、cs 为累积频率、R=2^b 为精度基数）；解码：σ = T[x mod R]（查表取符号），x = fs·⌊x/R⌋ + (x mod R) − cs。tANS/FSE 把全部转移预计算为查表自动机。整个编码行为由一张几 KB 的"熵编码自动机表"（256 符号）确定；状态可任意初始化而不损压缩率，因此天然支持多条独立子流并行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文用 rANS 对 LLM 权重做 tile 级无损压缩：
```
# 离线编码（每投影矩阵）
freq = histogram(layer_weights); fs, cs = normalize(freq, R=2^12)  # 层共享 codebook
for tile in split(W, tile_shape):            # 与 GEMM tile 几何对齐
    x = x0[tile]                             # 独立初始状态 → 自包含 substream
    for s in reversed(tile):                 # rANS 逆序编码
        x = (x // fs[s]) * R + (x % fs[s]) + cs[s]
    stream[tile] = x; offset[tile] = pos     # 4B/条 offset 表 → tile 随机访问
```
Annotations：任意初始状态不损压缩率是 ANS 相对算术编码的关键性质——每个 tile 是独立 substream，可跳过前面 tile 直接解码（tile 级随机访问）。选型依据（论文 Table I）：LZ77/Zstd 字典指针链不可随机访问；算术编码位串行仅几 GB/s；Huffman 整数码长离熵界 5–10%；rANS/tANS/FSE 熵效率 >99%、字节级流式、可并行——是唯一同时满足"近 Shannon 码率 + tile 随机访问 + GPU 并行解码"的编解码族。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工业实现：Zstd 的 FSE（tANS）、Apple LZFSE、Fabian Giesen 的 ryg_rans（byte-aligned 参考实现）、Meta DietGPU（GPU rANS，A100 解码 250–410 GB/s）。解码实现要点：符号查表（per-slot sym 表或 (fs,cs) 数组）+ 重归一化（状态低于阈值时读入 32/64-bit 块）。使用：近熵压缩 + 高吞吐场景——LLM 权重、HPC 数据搬运、GPU 集体通信压缩；本论文将其升级为 GPU 推理执行原语（解码直接写 shared memory 供 tensor core 消费）。

ENEC 补充视角（Ascend NPU 侧）：ENEC 论文把 ANS 移植到 Ascend 910B2 实测吞吐惨淡（Figure 1b），与 LZ77 类似，因此没有采用 ANS/变长熵编码，而是转向"块式定长编码 + 只压指数"路线。原因：ANS 的解码依赖符号查表（T[x mod R]）、重归一化分支与变长状态管理——需要条件分支、scatter/gather 和不规则变长访存，而 Ascend AIV 是无条件分支的 SIMD 向量单元，没有这些指令；且 Ascend 每个 AI core 是单一重线程、无 CUDA 式轻量线程间同步，ANS 惯用的"多条独立子流并行解码"无法高效落地。因此 ENEC 选择定长编码（每组 ≤m 或 n 位 + bit mask），把熵编码的不规则控制流替换成向量化位运算，这在压缩率上较 ANS 系略低（BF16 CR 1.35 vs DietGPU-Float 1.47 类），但换来 263–523 GB/s 的 NPU 端吞吐。ENEC 论文把 ANS 描述为"GPU 友好的熵编码族"、在 Ascend 上"从根本上不兼容"（与 DietGPU 的 ANS float codec 对比，见 kernel调度层 DietGPU 条目）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## LLM 权重重尾分布与逐层熵估计

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 权重呈重尾（heavy-tailed）统计分布：大部分值集中在 0 附近（尖峰）、存在长尾离群值（outlier），且分布跨层显著变化。量化文献以 kurtosis、α-stable 分布刻画（QuaRot/Q-Palette 等用旋转把分布"Gaussianize"、KurTail 用 kurtosis 度量尾部、SmoothQuant 平滑激活离群）。对无损压缩的含义：重尾 → 符号频率高度偏斜 → 经验熵远低于名义位宽 → 熵编码可获得大压缩率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文的逐层熵估计即重尾分布的信息论量化（见 Shannon 熵条目伪代码）：逐 tensor 建直方图（而非全模型单一码本），因为分布跨层差异大。实测结果（1.5B–405B 模型一致）：bf16 熵 10–12 bits、int8 4–5 bits、int4 0.6–1.0 bits；低比特格式符号表小且分布尖锐 → ANS 几乎达熵界；群量化（sq8/awq4）的 per-group scale 元数据引入 1.1–1.3× 额外冗余。层共享 codebook 的策略正是"分布重尾但跨层相似"与"逐层有差异"之间的折中（聚合统计覆盖 + 元数据摊销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
统计实现 trivial（histogram）；工程意义在于指导码本粒度（per-layer/per-tensor codebook）与量化设计：AWQ 保护 salient 通道（0.1–1% 权重）、SmoothQuant 平滑离群、旋转类方法 Gaussianize 都是为了驯服重尾；无损压缩则直接利用偏斜分布做熵编码，与量化正交叠加。注意：经验熵是估计值，符号表大、样本少时偏高；论文对 bf16 的 0.1–0.2 bits 偏差即有限精度表的来源。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression

## LLM 数值格式与群量化（bf16 / FP8-E5M2 / INT8 / FP4-E2M1 / INT4 / SmoothQuant sq8 / AWQ awq4）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 部署的权重数值格式谱系：bfloat16（8-bit 指数 + 7-bit 尾数，动态范围同 FP32）、FP8-E5M2（5 指数 2 尾数，Hopper 原生支持）、INT8、FP4-E2M1 与 INT4（4-bit 极低位宽）。群量化（group-wise quantization）：按组（如 g=128）共享 scale（与 zero-point）的权重量化；代表：AWQ（activation-aware weight quantization，per-channel 缩放保护 salient 权重、W4A16、MLSys'24 Best Paper）与 SmoothQuant（per-channel 平滑因子把激活离群难度转移到权重、W8A8；SmoothQuant+ 扩展 W4A16）。sq8/awq4 即 SmoothQuant 8-bit 与 AWQ 4-bit 格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文把这些格式作为无损压缩的信源符号表：符号即数值格式的离散码（FP8 码、INT4 值、per-channel 量化索引等）。符号表大小决定 ANS 行为：int8/sq8/fp4/int4 小符号表 + 尖锐分布 → ANS 与熵界差 0.01–0.05 bits；bf16 2¹⁶ 符号表 → 2¹² 精度 ANS 表定标偏差 0.1–0.2 bits（仍是现有无损方案里最接近熵界者）。群量化的 per-group scale 是结构化元数据，本身保留 1.1–1.3× 冗余，说明"量化之后仍可无损压缩"。正交性：先量化（有损）→ 再 rANS（无损），解压后与量化模型 bit-exact，不改变任何权重值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AWQ 实现：per-channel scale 网格搜索（α ∈ [0,1]）→ 权重×s / 激活÷s → INT4-g128 量化；SmoothQuant：s = max(|x|)^α / max(|w|)^(1−α)；部署格式 INT4 走 Marlin kernel（Ampere+）、FP8 W8A8 走 Hopper WGMMA。使用：模型压缩部署（TensorRT-LLM/vLLM/SGLang 均集成 AWQ/GPTQ）；本论文的用法是把这些格式作为"熵仍有冗余"的证据与压缩对象，证明无损压缩与量化正交、可叠加（int4 还能再压 4–6×）。

FlexQ-NDP 补充视角（ISCA'26）：该论文把群量化推进到"分组低比特浮点"形态——QGroup 粒度远细于 AWQ/SmoothQuant 的 g=128，如 (G_N,G_K)∈{(1,16),(1,32),(16,16),(32,32),(64,64),(128,128)}，每 16 个 FP4 元素配一个高精度 scale（FP8）。格式族为 microscaling：MXFP4/6/8（OCP 标准，32 个低比特元素共享一个 E8M0 2 的幂 scale，真实值 v_i = X·P_i）与 NVFP4（NVIDIA Blackwell 专有，16 元素块 + FP8 E4M3 scale + 每张量 FP32 二级 scale），以及 DeepSeek-FP8（FP8 块量化，块 scale 128×128）。反量化公式：partial^g = Σ_i (x_i^g · s_A^g · s_W^g)——每个 QGroup 的低比特乘加部分和须乘激活 scale 与权重 scale 恢复到高精度再累加；NVFP4 矩阵乘每处理 16 个 FP4 元素就要做一次双 scale 乘法，dequant 约占 NDP 上总执行延迟 35%。weight-only 量化（W4A16S8）不量化激活、计算前把权重反量化到激活精度，反量化开销与权重数据量相关；weight-activation 量化（W4A4S8）权重激活都量化、运行时需动态量化激活，反量化开销只与计算 bundle（点积次数）相关。该论文把量化配置抽象为 QConfig = {组尺寸, value 精度, scale 精度, W-only/W-A}，作为 NDP 编译策略选择（算子划分/缓冲分配/循环序）的输入变量——不同 QConfig 偏好不同编译策略，性能差距可达 70%。

Cassandra 补充视角（ISCA'26）：BF16 的 8-bit 指数占位宽 50%——在低 batch decode（memory-bound）下指数是剪枝/截断之后剩余的主要压缩率与带宽瓶颈，故 Cassandra 对指数单独压缩：Cassandra-1 用 unary 编码无损压缩（BF16 训练权重/KV 指数 Shannon 熵约 2.6/2.7 bits，实测平均约 2.85 bits）；Cassandra-2 用 MX 格式共享指数（每 32 元素块共享一个 E8M0 2 的幂 scale，OCP MX v1.0，v_i = X·P_i；草稿阶段即用 MXINT）。SmoothQuant W8A8 在该文语境：vLLM 官方 INT8 实现作 GPU baseline——低 batch decode 仅约 1.3×（在线激活量化 + scale 乘加开销在 GEMM 非瓶颈时无法隐藏，与文献报道的 INT8/FP8 decode 1.25–1.42× 一致）；且在推理 benchmark 上精度下降（Deepseek-R1-Distillated-Llama3-8B 上 SmoothQuant GPQA 46.0 vs BF16 49.0、AIME 23.3 vs 26.7）。该文结论：量化（有损）与 Cassandra（无损投机）正交且可组合——MXINT8 可直接融入 Cassandra 的草稿/验证格式（MXINT8 开销与其他 8-bit 量化相当且精度优于 MXFP8）。

PLENA 补充视角（ISCA'26）：把 MX 从"固定标准格式"扩展为可配置单级缩放格式——参数化 (M,E,S,B)（minifloat 元素）与 (M,S,B)（integer 元素），块内元素共享一个 E8M0 幂次 scale（8-bit 纯指数无符号位，范围 2⁻¹²⁷–2¹²⁷），PLENA 硬件原生支持 MXFP/MXINT 与可调块大小（BLEN∈[2,64] 纳入 DSE），权重默认 MXINT4、激活可选 MXINT/MXFP、KV cache 低精度 MX。关键经验：(1) MXFP 与 GPTQ/QuaRot 类 PTQ 不兼容——MXFP4 直接套用 4/4/4 量化在 LLaMA-3-8B 上 PPL 256.22（vs MXINT4+PLENA 方法 7.22），最小浮点格式表示区间窄，误差传播按整数格式设计时失效；(2) MXINT 是权重量化 de-facto 选择，但需搭配块级裁剪——裁剪参数 p∈[0.5,0.99] 收缩有效范围 [p·min_w, p·max_w]，配输出范数引导的逐行搜索 P_b*=argmin‖X_b(W_b−Q(W_b;P_b,τ))^T‖²（内层）与 GPTQ 式 Hessian 外循环（H_F=2X_FX_F^T）；(3) 硬件侧：PE 原生消费 MX 输入（元素与 E8M0 scale 分流）、INT 累加、写回前转激活精度；HBM 中数据块与 scale 分离存储以对齐内存边界（块+scale 拼接很少对齐 2 的幂边界）。与 NVFP4（16 元素块+FP8 E4M3 scale+张量级二级 scale）和 MicroScopiQ（两级缩放）相比，PLENA 选单级缩放以平衡硬件复杂度与精度；向量/非线性算子量化到 MiniFloat E6M5 相对 FP16 省 25% 内存且 perplexity 无损。

Shining Light 补充视角（ISCA'26，SiPh 加速器）：该论文把 AWQ 用作 LLM 部署到硅光子加速器时的权重侧量化——Qwen2.5-7B-instruct 先做 fp16→int4 的激活感知权重量化得到 Qwen2.5-7B-instruct-AWQ（Wikitext-2 困惑度 6.79，[82]），再把激活从 fp16 进一步量化为 int4-int8（per-tensor/per-feature/per-block affine 量化）以适配光域 MAC 的多电平编码。关键发现：AWQ 只解决权重侧动态范围，激活侧的 outlier（尤其 per-tensor 粒度）在 int5 以下使困惑度从 6.79 剧增到 182~120 万；且 SiPh 加速器无法像数字加速器那样用 FP8/24-bit 高精度累加（[94][96]）弥补激活量化丢失的动态范围，因为 ADC 量化位不可复用——说明"低比特激活 + 光域模拟计算"的组合对 LLM 尚不可行，需算法/器件级进一步改进。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- Shining Light on Silicon Photonic DNN Accelerators

XtraMAC 补充视角（ISCA'26，混合精度 MAC 硬件需求侧）：该论文从硬件视角梳理 LLM 量化方案产生的 MAC 数据类型组合（Table I）——权重仅量化（AWQ/GPTQ/SpQR）：投影层/FFN 为 INT×FP+FP→FP、注意力层 FP×FP+FP→FP；权重-激活量化（SmoothQuant/Atom）：投影层 INT×INT+INT→INT、注意力层 FP×FP+FP→FP；原生 LLM（GPT-oss-20b/120b）：MoE 块 MXFP4、其余 BF16（MXFP4/BF16×FP+FP→FP）。据此定义两类硬件需求：混合精度 MAC（乘数 A/B 异构格式，如 INT4×BF16）与运行时数据类型切换（同一硬件随模型组件交替数据类型，如 Qwen-3-8B-AWQ decode 期 68% MAC 为投影层 INT4×BF16、注意力层保留 BF16×BF16）。部署 profile（Table VI，HuggingFace 2025-10 下载量）：Qwen-3-8B-AWQ（222,126）、Llama-3.1-8B-W8A8（27,536）、Qwen-3-8B-FP8（429,968）、Llama-3.1-8B-FP8（168,122）、GPT-oss-20B（4,633,438）——实际部署的量化格式以 INT4/FP4/FP8 为主，要求 GEMV 引擎原生支持混合精度与运行时切换。FP4=E2M1、FP8=E4M3（E5M2 亦支持）；GPT-oss-20b 的 UE8M0×BF16 乘通过偏移 BF16 指数实现、不单独算 MAC。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- Shining Light on Silicon Photonic DNN Accelerators
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA

## FHE（Fully Homomorphic Encryption，全同态加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FHE 是一种加密范式：允许第三方在不解密的情况下对密文直接执行任意计算，解密结果与对明文做同样计算一致（Rivest 等 1978 提出概念，Gentry 2009 首次构造）。形式化 = 公钥加密方案 + 同态求值算法 Eval：对任意电路 C 与密文集合，Eval 输出 Enc(C(m)) 而不解密。安全基于 LWE/Ring-LWE 困难问题；同态性 = 加法与乘法都可在密文域完成并可组合成任意电路。
- 核心机制是噪声：每个密文携带误差 e，同态加法噪声近似相加，同态乘法增长更快，噪声超过阈值则解密失败——因此计算深度受限，需 bootstrapping（同态执行自身解密函数）刷新噪声实现任意深度。性能代价被称 "performance tax"：20 层网络隐私推理比明文推理慢约 10^4 倍；bootstrapping 占 FHE 成本主体（部分分析 >90%，应用运行时通常 50–80%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FHE 在算法 pipeline 中把明文应用的每个算子替换为同态算子。以本论文加密 NPU 隐私推理为例，一层网络的同态计算过程：
```
for each layer l in network:
    enc_out = HomMul(enc_act, plaintext_weight)   # 密文x明文权重
    enc_out = HomAdd(enc_out, plaintext_bias)
    enc_out = ProgrammableBootstrap(enc_out, f)   # 噪声刷新 + 同态激活查表
return enc_out
```
- Annotations：enc_act 是 TFHE 密文；HomMul/HomAdd 由密文算术 CPE 执行；ProgrammableBootstrap 是 FHE 特有开销（FFT/IFFT + external product），是 pipeline 主要延迟来源；bootstrapping 插入时机由 FHE 算法设计者决定（噪声预算耗尽前必须刷新）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 方案族：BGV/BFV（整数域、SIMD 打包）、CKKS（近似实数）、FHEW/TFHE（快速逐 bit 自举 + 可编程自举）。软件库：TFHE-rs（Zama）、OpenFHE、HElib、SEAL、concrete；GPU 库 cuFHE、nuFHE。硬件：Strix/MATCHA/PPGNN/Trinity/Poseidon 等专用加速器把 FFT/NTT、external product 做成硬件单元。使用场景：隐私推理（PI）、私密信息检索（PIR）、加密数据库查询、隐私图处理（本论文三个案例域）。在 FEnc² 中，FHE 还被赋予新的系统视角：应用级密文打包布局（数据布局）是降低 HE 工作负载的一级设计维度，与底层 NTT/keyswitch 加速器优化正交互补——FEnc² 通过减少旋转/keyswitch/NTT 的数量来重塑暴露给硬件的同态负载。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving

IroKnight 补充视角（ISCA'26，FHE 作为"加密所有权"的理论金标准 vs 工程不可行）：IroKnight 把 FHE NPU 当作所有权保有的理论上限与对比基线，用乐观建模（基于 SHARP [9] 设计：专用功能单元 + 180MB 片上 SRAM 存中间 FHE 密文；权重保持 FHE 明文多项式、不密文化，因密文化更贵；非线性算子 softmax/layernorm/ReLU 只能近似）。开销来源：每次同态乘加积累噪声、需反复 bootstrapping（IroKnight 估计占 FHE 延迟主体），以及权重明文多项式/激活密文多项式的大量片外搬移。8 个 LLM 上 FHE 运行时 713x-1793x（Llama3-70B 单 query 10.9 小时）、能量 871x-7396x（0.59 kWh）；小网络（BERT/ResNet-50/ViT）607x-1745x 延迟、11904x-35564x 能量。对比：IroKnight 同时加密激活与模型参数且防篡改，运行时仅 0.2%（加密）/3.3%（认证）、LLM 能量 <=14%/>=<18%。结论：等待实用 FHE 不可行，IroKnight 以"明文仅瞬态存在于 ALU、存储全加密"的新设计点逼近 FHE 所有权而避开其开销。

## TFHE（Fast Fully Homomorphic Encryption over the Torus，环面全同态加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TFHE（Chillotti, Gama, Georgieva, Izabachène, JoC 2020）是基于环面（torus）的 FHE 方案族，核心卖点：极快自举（单次 <0.1 s 量级）与可编程自举 PBS，能同态评估任意一元函数，是隐私推理与加密查找的主流选择（ZAMA 商业化部署）。密文结构三层：TLWE（环面 LWE，(a,b)，b=⟨a,s⟩+m+e，二元消息编码为 ±1/8 或 ±1/4）；TRLWE（环上多项式版本，模 X^N+1）；TRGSW（秘密钥的 gadget 加密，构成 bootstrapping key BK）。
- 本论文表 I 参数集：n（TLWE 维度 500–630）、N（TRLWE 多项式度 1024/2048）、L（分解层数 2–3）、k（TRGSW 层数 1），对应 80/110/128-bit 安全级别。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TFHE 一次 bootstrapping = 三步（本论文 CPE 模板对应的算法结构）：
```
ACC = X^(-b) * test_vector(f)              # TRLWE 累加器初始化
for i in 0..n-1:                           # 盲旋转：每个 TLWE 密钥位一次 CMux
    ACC = CMux(BK_i, X^(a_i)*ACC, ACC)     # external product，主开销
c' = SampleExtract_0(ACC)                  # TRLWE -> TLWE（取常数项）
c  = KeySwitch(c', KSK)                    # 切回原 n 维密钥域
```
- Annotations：CMux 每步一次 external product（TRGSW x TRLWE），FFT/IFFT 把多项式乘从 O(N^2) 降到 O(N log N)，占自举延迟约 80%；PBS = 把一元函数 f 编码进 test vector 系数，自举同时完成查表求值（同态激活 ReLU/sign、加密表查找都靠它）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 库：原版 C++ TFHE、TFHEpp（纯 C++）、TFHE-rs（Zama 生产级）、cuFHE/nuFHE（GPU）。加速器：Strix（eNPU）、MATCHA（整数近似 FFT + bootstrapping-key unrolling）、FPT（单 CMUX 流式 PE）、OFHE（光电 FFT）。本论文 AutoFHE 以 TFHE 为代表性方案：arithmetic/bootstrapping/key-switching/HMUX 四类 CPE 模板即 TFHE 算子集的硬件参数化；使用场景：DeepCNN 隐私推理、加密 ALR（算术/逻辑/关系）、index 加密查找。

MNEMOS 补充视角（ISCA'26，GPU 内存访问优化框架）：MNEMOS 系统性剖析 ZAMA 的 TFHE-rs/Concrete GPU 实现，确认 TFHE 相比 CKKS 更适合 GPU 加速的控制密集型/bit 级/逻辑重负载（隐私保护量化神经网络要求精确而非近似计算），但代价是更高的计算与内存需求。其关键新发现：(1) PBS 在真实 TFHE workload 中严重 memory-bound——stall_long_scoreboard 超 50% 执行时间（加 stall_MIO_throttle 超 60%），根源是庞大的 bootstrapping key（BSK）作为"热数据"被多 SM 同时访问、仅获得 L2 级复用而无法驻留 SM；(2) 首次研究把高精度 TFHE FFT 映射到 FP64 Tensor Core（WMMA 8×8×4），并指出 CKKS 方案的 Tensor Core NTT 映射（低精度 INT8/FP16）因 TFHE 的 FP64 精度要求不能直接照搬——精度分析表明 4-bit 明文正确性需 ≥30 小数位（常 >35），FP32（24 尾数位）/FP16（11 尾数位）不足；(3) 噪声公式 n·2^ω·ℓ·2^(2β)·N²·(k+1)（ω≈2·(64−53)−2.6，64-bit 密文空间）显示尾数位宽影响呈指数级，N 为二次、n/ℓ/k 线性，因此 ZAMA 参数集正是在 FP64 舍入误差模型下联合优化的。参数集（Table II）：Para-A~D 由 Concrete 编译器为 CNN 生成（N=512/1024、k=2/4、ℓ=1/2/11、n=532~728、128-bit），Para-E 来自 tfhe-rs benchmark、Para-F 来自 Morphling，另有 Para-I/II 用于跨平台对比（80/110-bit）。

CASCADE 补充视角（ISCA'26，跨 HMUX 流水线并行的 TFHE 加速器）：CASCADE 把 TFHE 的瓶颈定位为 BSP 中 n 次串行 HMUX（盲旋转）迭代——n 为加密参数，为保 128-bit 安全需大 n（参数集 III：n=592、N=2048、L=3、k=1），CPU 实测 n 次 HMUX 占 BSP 执行时间 79%、BSK 搬运占总数据搬运 80%，且 n-HMUX 算术强度远低于 A100 平衡点（440 GOPS/s）。先前加速器（MATCHA/Strix/Morphling）全部串行执行 n 次 HMUX，吞吐受 Thp_seq ≈ 1/(n·t_HMUX) 硬约束；CASCADE 首次利用跨 HMUX 流水线并行（理论 Thp_pipe ≈ 1/t_HMUX，最高 n× 提升），但流水线并行要求 n 个 HMUX 并发访问各自的 BSK（GGSW 高阶多项式），产生集中式 HBM 无法支撑的带宽需求（Morphling 单 HBM stack 约 30W ≈ 加速器 die 功耗 56%）——CASCADE 用分布式 SRAM（BSK-distributed，126 MB）驻留全部 BSK 解决。四参数集（I/II/III/IV，80/110/128/128-bit）见表 I。评估：参数集 I/II/III 吞吐 2,133,624/1,235,248/416,408 BSP/s，Speedup/Area 相对 MATCHA/Strix/Morphling 30.5×/15.6×/3.1×。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## Bootstrapping（自举，含可编程自举 PBS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bootstrapping 是 FHE 中同态执行自身解密电路的过程：对噪声接近阈值、无法再运算的密文 c，利用 bootstrapping key 中加密的秘密钥位做一次"密文域的解密-再加密"，输出噪声重置到低水平、内容不变的新密文。它是所有已知 FHE 方案实现任意深度计算的前提，也是性能大头（FHE 开销常占 50–90%）。TFHE 的可编程自举 PBS 更进一层：把一元函数 f 编码进 test vector，噪声刷新的同时同态求值 f——相当于免费 LUT 查表。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 自举主体 = n 次串行迭代的密文处理（n 为 TFHE 参数），每次迭代含：多项式缩放 → external product → FFT/IFFT → 分解。本论文 bootstrapping 模板的模块化参数即这一算法结构：PoV（多项式缩放向量宽度）、PoD（分解并行 lane）、BFU/IBFU（FFT/IFFT butterfly 数）、PoE（external product MAC 数）。
```
for i in 0..n-1:                          # n 次串行迭代
    decomposed = GadgetDecomp(BK_i)       # 分解
    prod = ExternalProduct(ACC, decomposed)   # FFT/IFFT 加速的多项式乘
    ACC = ACC + X^(a_i) * prod
```
- Annotations：n 次迭代决定自举延迟；FFT 次数 = n（bootstrapping unrolling 可降到 n/r）；放置策略——TFHE 协议规定 arithmetic 下游实例化 bootstrapping 硬件单元刷新噪声（AutoFHE 自动实例化；算法级"何时自举"由 FHE 算法设计者决定）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：TFHE-rs 的 programmable_bootstrapping API；GPU 用批量自举摊薄密钥带宽（BOLT-FHE 等）。硬件：MATCHA 的 38-bit DVQTF 整数近似 FFT（自举后噪声被刷新故可容忍近似）、FPT 的单 CMUX 流水 PE、Strix CPE 内 FFT/IFFT + vector MAC。加速方向：自举展开、多值自举（MOSFHET）、NTT 替代浮点 FFT。在 FEnc² 中，深层 CNN（SqueezeNet/ResNet18/MobileNet）推理需要自举，采用 GPU 优化的 NEXUS bootstrapping（每次消耗 14 个密文 level）；FEnc² 通过减少卷积层旋转来压缩自举之外的 HE 负载，即使自举开销不可避免（ResNet18/MobileNet 分别 +191s/+105s）仍显著加速。

FlashTFHE 补充视角（ISCA'26，multi-bit 参数下的 PBS 分解与执行顺序）：PBS 运行时间分解 = key-switching ~10%（第二大耗时）→ modulus-switching <1% → blind rotation ~90%（n 次串行外部乘积迭代）→ sample extraction <1%。执行顺序两种等价选择：key-switching-first（FlashTFHE 采用）先 KS 后盲旋转，使 KS 结果可在 fanout 结构（多个 LUT 作用同一 ciphertext）中跨多个盲旋转复用（KS-dedup，最多省 47.12% KS 操作）；blind-rotation-first 单次 PBS 计算量相同但无此复用。位宽越宽 PBS 频率越低：8-bit 相对 4-bit 模拟，DNN 类 workload 原生运算占 ~99.5%、PBS 数量级下降带来 6.8–8.1× 加速；非 DNN（XGBoost/DecisionTree/KNN）PBS 占比更高，8-bit LUT 需 ≥32 个 4-bit LUT 模拟（信息论下界），LUT 模拟成为瓶颈。硬件实现：FlashTFHE 单 ciphertext bootstrapping 延迟 6.16–34.67ms（高 bit-width 参数集）、CNN-20/50 单 batch 0.28/0.85ms。

MNEMOS 补充视角（ISCA'26，GPU 端 PBS 内存优化）：PBS 四阶段为 Modulus Switching → Blind Rotation → Sample Extraction → Key Switching（Algorithm 1）；MNEMOS 的剖析量化了各阶段的 GPU 开销——盲旋转（含 Decompose+FFT+MAC+IFFT 的 n 次迭代）主导执行时间，其中 MAC 阶段因需取 (k+1) 倍于 GLWE 体积的 BSK 而成为 memory-bound 瓶颈，stall_long_scoreboard >50% + stall_MIO_throttle >60%。优化后（BSK 分块复用 + Tensor Core FFT + 跨迭代融合）stall_long_scoreboard 降到约 20%，PBS 吞吐在 128-bit 参数下最高 3.01×（A100，Para-D）/2.86×（H100），应用端到端平均 1.96×（最高 2.23×，batch 4096 的 VGG-9 达 2.21×）。正确性前提：内部 FFT 需 FP64（≥30 小数位），与 FPT 的固定点位宽分析（[36]）互为印证。

CASCADE 补充视角（ISCA'26，自举的跨 HMUX 流水线化）：CASCADE 的自举（Algorithm 1）把 BSP 拆为 c←(2N/q)·c_in → ACC←X^(-b)·c_T → n 次 HMUX 迭代（Line 5：BSK_i←(X^(-a_i)−1)·BSK_i；Line 6：ACC_i←BSK_i⊡ACC_{i-1}，即外积）→ SampleExtract → key-switching（KSK 标量乘）。自举的 n 次 HMUX 是主要瓶颈：串行执行吞吐 ≈1/(n·t_HMUX)，流水线并行后稳态吞吐 ≈1/t_HMUX（最高 n× 提升），但每个 HMUX 需要访问唯一的 BSK（GGSW 矩阵），流水线化使并发 BSK 访问成为内存带宽瓶颈（算术强度低于 A100 平衡点 440 GOPS/s）。CASCADE 通过把全部 BSK 驻留在分布式 SRAM（BSK-distributed）消除片外 BSK 搬运，并把 n 个 HMUX 用 Interleaved-Fusion 策略融合/交错映射到 12 个 HMUX Chiplet（HC）执行；每个 HC 的流水线完成一个 HMUX 的时延 ≈ 最长流水级，BSP 时延参数集 I/II/III 为 0.01/0.02/0.04 ms。消融显示：Monolithic（单 chiplet+HBM3 串行）→ 细粒度流水架构 13.2× → 加 OIFS 调度再 4.1×，共 53.5×。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## Bootstrapping unrolling（自举展开优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bootstrapping unrolling 是 TFHE 自举的算法级优化（思想源自 Bourse 等 CRYPTO'18、MATCHA 的 bootstrapping-key unrolling 实践）：自举主循环的 n 次迭代本身串行，把 r 次相邻迭代合并为一次展开迭代，每轮同时处理 r 个密钥位/旋转量，迭代深度 n→n/r、FFT/IFFT 次数 n→n/r（计算量下降），代价是需要 (2^r−1) 大小的展开阵列缓存中间结果、且 r 倍并发密钥访问（内存带宽压力上升）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 原循环每迭代一次 CMux/external product；展开后：
```
for j in 0..(n/r)-1:
    # 一次迭代同时应用 r 个密钥位，需 (2^r - 1) 展开阵列
    ACC = UnrolledCMux([BK_{j*r}, ..., BK_{j*r+r-1}],
                       [a_{j*r}, ..., a_{j*r+r-1}], ACC)
```
- Annotations：r 增大 → FFT 次数下降（计算减）但每轮并发密钥访问上升（带宽增）。本论文实测最优 r 与带宽耦合：Strix 场景（300 GB/s）最优 r=2，MATCHA 场景（640 GB/s）最优 r=3，r 过大带宽压力导致性能回退——最优展开因子无法跨硬件场景手工迁移，必须自动搜索（AutoFHE DSE 的核心论据）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件：bootstrapping 模板内置 (2^r−1) 展开阵列；AutoFHE 把 r 作为 DSE 设计变量自动选择（禁用 unrolling 性能降 39.5%）。手工实践：MATCHA 手工选定 r=3；MOSFHET 的 blind rotation unfolding 属同类思想（多值自举，2× 加速）。带宽有限场景（如 300 GB/s）不宜盲目增大 r。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators

## Key switching（密钥切换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Key switching 是把密文从一把秘密钥的密钥域转换到另一把密钥域的 FHE 操作：给定旧钥 s 加密的密文 c，用 key-switching key（新钥对旧钥的加密）做一次密文域线性变换，输出新钥 s' 加密的同一消息。TFHE 自举收尾必用：盲旋转使密文落到大维度密钥域，须切回原 n 维 TLWE；在 BGV/BFV/CKKS 中其特例 relinearization 把乘法后的 (1,s,s²) 密文拉回 (1,s)，旋转（Galois 自同构）也需要它。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算形式 = 密文向量 x key-switching key 矩阵（向量-矩阵乘）：
```
# c = (a, b) 在密钥 s'（维度 kN+1），目标密钥 s（维度 n+1）
a_dec = GadgetDecomp(a)          # L 层分解
a_new = K_a * a_dec              # 矩阵乘，输出 n 维
b_new = b + K_b * a_dec
return (a_new, b_new)
```
- Annotations：本论文 key-switching 模板集成向量单元 + 累加器，PoK = 向量单元并行度；放置于 bootstrapping 下游（TFHE 协议要求自举后切换回原密钥域）；分解层数 L 越大噪声越小但计算与密钥存储越大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 库实现：TFHE-rs/OpenFHE 的 keyswitch 函数（Gadget 分解 + 矩阵乘）；硬件实现为向量 MAC 阵列（本论文模板）。优化：延迟 relinearization（加法无需先切回，编译器层面推迟）、fused key-switching（合并乘-旋转序列，Transformer 隐私推理场景）、HEIR 的 optimize-relinearization ILP pass。AutoFHE 把 key-switching 作为 CPE 模板之一，与 arithmetic+bootstrapping 组成统一 CPE lane。
- FlashTFHE 补充视角（ISCA'26）：PBS 中 key-switching 约占运行时间 10%，为第二大耗时步骤，主要作用是把 LWE 维数从长切短（如 ~30000 降到 ~1000），从而减少盲旋转的串行迭代数。硬件实现在 LPU（LWE Processing Unit）中：8 个独立可寻址、可时钟门控的 lane，每 lane 处理 32 个并行 64-bit 值（匹配 2^64 torus modulus），含向量加/乘单元、decomposer 与 rotator；lane partitioning 使小向量 KS 与 native LWE 运算不需要占满整条 8-lane 流水线。FlashTFHE 采用 key-switching-first 执行顺序，使 KS 结果可被多个后续盲旋转复用（KS-dedup，最多省 47.12% KS 操作）——这是 Boolean TFHE 的 blind-rotation-first 顺序做不到的。
- HE² 补充视角（ISCA'26，CKKS 方案）：CKKS 中 keyswitch 占据约 80% 计算量，是所有乘与旋转共用的核心原语。其数据流为交替的 ComOps/MemOps 序列 ModUp→IP→ModDown：密文在模数 Q 下分解为 dnum 组，经 ModUp 提升到 PQ·dnum 域、与 evk 多项式做内积（IP）、再经 ModDown 降回 Q，结果加回原密文。keyswitch 的中间密文（ModUp 输出与 IP 结果）尺寸大（单次传输最高 144 MB 量级），在异构加速器中若走 IRF 数据流（IP 放近存 xMU）则这些传输落在关键路径上——这是 HE² 的核心优化对象（HERO DFG 优化降通信频率 + 双级流水 xPU 隐藏通信延迟，见本库"xPU-xMU 异构架构""EVF/IRF 数据流"与编译框架层"HERO"条目）。

CASCADE 补充视角（ISCA'26）：CASCADE 把 key-switching 与 sample extraction、同态加法、标量乘等轻量操作交给集成在 HC0 的 VPU（Vector Processing Unit）执行——因为这些操作只占 BSP 计算的一小部分（相对 n 次 HMUX 的盲旋转），VPU 用并行乘法器/加法器 + 局部 buffer 实现、与 HMUX 流水并行工作，避免打断高吞吐的 HMUX 流水线。BSK 之外，HC0 额外 0.5 MB SRAM（12 MB vs 普通 HC 11.5 MB）存储 KSK。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## HMUX（Homomorphic Multiplexer，同态多路选择器 / CMux）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HMUX/CMux 是 TFHE 的基本选择原语：HMUX(a,b,s) = a·s + b·(1−s)，在密文选择位 s 控制下从两个密文 a、b 中选出其一，选择位本身加密（外部无法得知选中者）。实现上它是 bootstrapping 一次迭代的同构操作：CMux 用 GGSW（TRGSW）加密的选择位做一次 external product。TFHE 借此支持"index 加密查找"——用 HMUX 树对加密索引逐位选择表项，同时隐藏访问地址与输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 加密查找（本论文 PPGNN/查找引擎场景）：
```
# 表 T 有 2^d 项，加密索引 idx = (s_0..s_{d-1})
out = T[0]
for j in 0..d-1:
    out = CMux(s_j, out, SelectRotated(out, T, j))   # 每层一次 external product
# out = 加密的 T[idx]
```
- Annotations：d 层 HMUX 树 = d 次 external product；选择位为 GGSW 密文；与 PBS 盲旋转共享同一计算内核——本论文 HMUX 模板直接用 bootstrapping 模板实例化（"HMUX 算法流程类似一次自举迭代"）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 用法：加密图查找（PPGNN）、加密数据库索引（ArcEDB）、隐私推理激活函数（HMUX 编码 ReLU/sign，ZAMA 方案把整个激活单元替换为 HMUX CPE）、盲旋转内部逐位 CMux。软件对应：TFHE-rs 的 cmux 函数、WOPPS 的 cmux-tree 大位宽查表。AutoFHE 算子映射规则：Chisel 两输入 MUX → HMUX 模板。

CASCADE 补充视角（ISCA'26，n 次 HMUX 迭代作为自举性能瓶颈）：CASCADE 中 HMUX 是自举（盲旋转）的迭代单位：每个 HMUX_i = BSK 旋转（Line 5：BSK_i←(X^(-a_i)−1)·BSK_i）+ 外积（Line 6：ACC_i←BSK_i⊡ACC_{i-1}），外积是矩阵-向量乘（L×(k+1)×(k+1) 多项式矩阵 × (k+1) 向量），占计算成本主体，用 FFT/IFFT 把多项式乘从 O(N²) 降到 O(NlogN)。关键性质：n 个 HMUX 严格串行（每个依赖前一 ACC），且每个 HMUX 访问唯一的 BSK（BSK 不能跨 HMUX_i 复用，只能跨多个 BSP 复用）——这两个性质分别是串行吞吐瓶颈与并发 BSK 带宽瓶颈的根源。CASCADE 的 HMUX Chiplet（HC）把一次 HMUX 实现为 Rotation→Decomposition→FFT→VMA→IFFT 的系数粒度流水（一个 HMUX 时延≈最长流水级），用 Interleaved-Fusion 把连续 HMUX 融合成组（组内回馈本地执行、减少 D2D 通信）并跨 chiplet 交错以保持流水并行。

涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## k-mer 计数与 de Bruijn 图遍历（基因组组装 DirectAP pipeline）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
de novo 基因组组装的两步核心：k-mer 计数——把 reads 切成全部长度 k 子串并统计频数（用于错误校正与图中节点权重/过滤低深度 k-mer）；de Bruijn 图构建与简化遍历——节点=k-mer、边=相邻 k-mer 重叠 k−1 个碱基，简化掉 tips/bulges、把非分叉路径合并成 unitig，最终拼接出 contig。SPAdes（Bankevich 2012，web：https://github.com/ablab/spades）是多 k 迭代 + 错误校正（BayesHammer）的代表实现，BAAP 用它作多核 CPU 参考实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 基因组组装 pipeline（论文 §IV-E，BAAP 在 UPMEM DIMM 上）
for k in K:                            # 多 k 扫描（BAAP 单 k 演示）
  counts = {}
  for read in reads:                   # host 阶段：k-mer 计数
    for i in 0..len(read)-k:
      counts[read[i:i+k]] += 1
  # BAAP DirectAP：计数改为 bank 内穷举 CAM 匹配（ap_regex 序列）
  G = deBruijn(counts, k)              # 节点=k-mer、边=重叠 k-1
  G.remove_tips(); G.remove_bulges()
  unitigs = G.collapse_nonbranching()  # 合并非分叉路径
  traversal_order = AP-BFS(G, start)   # BAAP 算法 1：tag 编码前沿的图遍历
```
BAAP 映射（论文 §IV-E）：UPMEM 基线 k-mer 计数 = 线性扫 bank + 哈希表更新，遍历 = DPU 标量前沿管理；DirectAP 把两步都变成 WRAM 上的关联查询——计数 2–38×（随 k）、遍历 1.1–2.8×；k>21 时 2^k 搜索空间爆炸、中间图可占满整条 DIMM，回退 16 核 host。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工具链：SPAdes（多 k 迭代、--isolate/--careful 模式）、KMC（k-mer 计数）。数据集：BAAP 用真实 A. thaliana（SRR29124148）。适用：错误校正、宏基因组学、群体研究（小 k 值场景同样常见）。硬件映射要点：k-mer 匹配是"重复成员查询于大而稀疏状态"的典型 DirectAP 形态；跨 DPU 前沿交换必须经 host 中转，是 k 增大时的主要瓶颈。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing

## Prefill / Decode（预填充与逐 token 解码两阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 推理的两个阶段：prefill（提示词全序列一次性前向，矩阵乘为 GeMM 形态、计算密集、吞吐优先）与 decode（逐 token 自回归，每步 GeMV 形态、带宽/访存受限、延迟优先），两阶段共用 KV Cache。CompAir 的两阶段硬件敏感度分析：prefill 是 compute-bound → SRAM-PIM 收益大（0.5K 长度 3.29–5.46×，加解耦列译码器后 4.1–7.89×）；decode 收益依赖 batch——batch=1 无 SRAM 收益（无复用机会），batch=64 时 2.67–6.28×；序列长度增大时 decode 的相对优势稳定在 ~2.5×、而 Curry ALU（非线性在途）贡献上升——128K 长上下文下 GPT3-175B/Qwen-72B 达 2.13–2.73×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# prefill: X ∈ R^{seq×d}，一次前向
for layer in layers:
    Q,K,V = X @ W_qkv          # GeMM（计算受限）
    A = softmax(Q @ K^T) @ V   # attention
    X = FFN(RMSNorm(X + A))    # GeMM + 非线性
# decode 第 t 步: x_t ∈ R^d
Q_t,K_t,V_t = x_t @ W_qkv      # GeMV（带宽受限）
attn over cached K,V; FFN
```
CompAir 的映射：prefill 的 GeMM 与 batched decode 的 Q/K/V、FFN 交给 SRAM-PIM（权重驻留）；decode 的 QK^T/SV（GeMV、K/V 输入相关）与 attention 交给 DRAM-PIM；Softmax/RoPE 由 NoC 在途完成。batch=1 decode 是纯 GeMV，SRAM-PIM 无复用、收益为零。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：vLLM/SGLang 等 serving 框架以 continuous batching 混排 prefill 与 decode 请求；硬件侧按阶段切换映射（CompAir 的静态分派规则：GeMM→SRAM、GeMV→DRAM）。使用方式：prefill 优化看重算力与带宽平衡（解耦列译码器提升 bank 读带宽）、decode 优化看重 batch 复用（SRAM-PIM）与非线性开销（NoC 在途计算）；TP 提高对两阶段都削薄每 bank 的 batch/复用（TP≤8 最优）。CHIME 的 AFD 视角：两阶段按设备拆分——prefill attention 在 GPU、decoding attention 卸载到 DIMM-PIM（CHIME 的 sub-batch 调度让一个子批的 decode attention 与另一子批的 prefill+FC 并发）；decode 阶段是带宽密集且不随 batch 复用（各请求 KV 独立），正适合 PIM。

ConServe 补充视角（ISCA'26）：multi-turn 的 prefill 是增量 prefill——新 turn 只 prefill 新增 token（每轮 512），同时 attention 读全部历史 KV（ShareGPT 新 turn 99% token 来自历史）；因此 multi-turn prefill 的访存主体是历史 KV 的读，其虚拟布局（连续 vs 散页）决定翻译局部性：FlashInfer-paged 相对 native 的 prefill 时间随轮数从 1.2× 升至 1.75× 后饱和。decode 每 token 读全部历史 KV（内存受限），即便数据 L2 命中，VA 仍需翻译——TLB miss 的页走查在 L2 hit 时也加长每 token 关键路径，这是 ConServe 连续布局带来 decode 吞吐 +19.4%~25.6% 的来源。

EVA 补充视角（ISCA'26）：decode 的 GEMV 形态是"权重主导访存 + 计算单元低利用率"的双重低效（M=1 时 32×32 阵列仅 1 lane 活跃）。EVA 通过向量量化（VQ）+ 码本驱动 GEMM 重构把 decode 从 GEMV 重写为 GEMM（输入向量×码本 → 输出码本 → 冲突无关查找+累加），使 decode 阶段也能填满矩阵单元（M 维从 1 扩到 V=K/d>512），同时与 prefill 兼容（同一阵列 INT8/FP16 重配）。这为"decode 必然 memory-bound"提供了一条计算侧重构路径：权重压到 2-bit 且查表转 GEMM 后，decode 关键路径从"权重带宽"转移到"加法树吞吐"（VQ-GEMM 256 cycles vs EU 4096 cycles）。

Raptor 补充视角（ISCA'26）：两阶段计算-访存特性直接决定内存基板选型——prefill 是计算受限（大矩阵乘、强权重复用），decode 是带宽/容量受限（自回归逐 token、每步读写全部累积 KV cache、复用有限）。Raptor 把 XPU 逻辑（10 PFLOPS）固定、只换内存基板评估 decode 域：3D-DRAM（100TB/s/32GB）相对 HBM（18TB/s/192GB）与 SRAM（150TB/s/4GB）在 decode 吞吐与交互性上占优（4.71×/2.44× vs HBM/SRAM，9.96× 更低 TPOT vs HBM）；MoE（DeepSeek-V3/GPT-OSS/Kimi K2）在 batch 扫描下呈现 attention 内存受限→专家加载主导→专家全激活→attention 计算主导四段行为；speech 模型（Whisper/Canary，上下文仅 448 token）为纯带宽驱动，SRAM 反而最高、3D-DRAM 次之。即"decode 内存受限"在不同基板上表现不同：SRAM 卡容量（高 TP/PP、collective 大、网络敏感）、HBM 卡带宽（TPOT 高）、3D-DRAM 两维均衡。

HybridSpec 补充视角（ISCA'26，算术强度视角的两阶段异构分配）：decode 每 token 前向的算术强度随 batch/token 数缩放——GPU 计算-带宽比 100-300，需数百 token/前向才达计算饱和（Fig.4 实测：吞吐随 token 数上升后平台化，与模型规模/系列无关）。SD 使两阶段跨两个模型：prefill/verification 在 target（XPU，高强度、容量敏感）、decode 在 draft（HB 栈，低强度、带宽敏感）；draft prefill 单独执行（参数不同、开销可忽略），target prefill 与 verification 联合批。prefill 长度差异大（几十到几千 token）造成计算不均衡 → CHK 切块；TTFT（prefill 延迟）与 TPOT（decode 延迟）是评估主指标。

从算法pipeline角度拆解：一次请求 = target prefill（XPU，计算密集）→ draft decode 多轮（HB 栈，memory-bound，每轮 1-token×批）→ target verification（XPU，一次 batched 前向）→ 循环至 EOS；算术强度在两单元间被"极化+匹配"。

实现与使用：硬件选型按阶段算术强度定（计算密集→强算力 XPU、访存密集→高带宽 HB）；调度上把 prefill 与 verification 的竞争用 PFS/CHK 仲裁（见系统架构层）。

- M100 补充视角（ISCA'26，车规 NPU 上的两阶段）：LLaMA2-7B（输入 1,024 token）在 M100（12/14 cluster 激活）上 decode 用 W4A16（权重 4-bit INT/激活 FP16）：21.34ms vs Thor-U 20ms（0.94×）——两平台 DDR 带宽同为 273 GB/s，decode 为带宽受限故性能相当（Thor-U 优势来自 NVIDIA 对开源模型的高度优化）；prefill 用 W8A8：79ms vs 154ms（1.95×）——prefill 计算密集，M100 的 tensor 级并行（TCU 8×64 MAC）+ 数据流同步（计算与搬运重叠、低同步开销）优势显现。MindVLA LLM 组件 decode 0.1ms（3×）、prefill 0.84ms（2.1×）。
MoE serving 补充视角（ISCA'26，Patterns behind Chaos）：现代 serving 趋向细粒度分离——传统 LLM 把 prefill 与 decode 分离到不同机器（DistServe 及后续），MoE 更进一步（MegaScale-Infer 把 attention 与 MoE 操作拆到不同机器取最优 batch），因此本文以 decode 阶段 MoE 层吞吐作为评估指标。核心发现（Insight 1）：prefill 阶段与 decode 阶段的专家选择高度一致——跨层/跨 token 专家对热图（Figure 6a-d）与单专家频率分布（Figure 7a）在前后缀两阶段基本相似，Spearman ρ≥0.7（多数层强相关、少数中等）；top-5/10/20 热门专家跨阶段重叠约 60%/75%/90%（Figure 7b）。含义：prefill 收集的专家信息可用于指导 decode 初始阶段（前 ~1000 token）——这正是 prefill-guided expert placement 与 PD 分离部署下"prefill 机器把专家选择信息传给 decode 机器"的基础；decode 初期生成 token 少、历史上下文稀缺，prefill 信息是唯一可用的预测来源。

  - SHyLA 补充：LLM 自回归推理 = prefill 阶段（并行处理全部 prompt token，GEMM 为主，compute-bound）+ decode 阶段（逐 token 生成，GEMV 为主、memory-bound，依赖 KVCache）。SHyLA 从系统/内存角度给出容量模型：微批大小 b、系统批大小 = 微批 × pipeline 深度；系统吞吐（系统输出 token/时间窗）随更大的微批提升（增强 Weight 复用），而每用户吞吐（=1/TBT，Time Between Tokens 的倒数）随批变大下降（每迭代计算增加）——即系统吞吐与每用户吞吐在此处互相矛盾，是 SHyLA 两阶段 DSE（SLO 约束下）要权衡的核心。数据放置上 prefill/decode 可用 PD aggregation（prefill/decode 共享 chiplet）或 PD disaggregation（分离实例）；decode 阶段 GEMV 配对（fused ATTN）避免中间 QK^T 结果写 DRAM。
- STAGE 补充视角（ISCA'26）：STAGE 用 DeepSeek-R1 推理架构（prefill/decode 分离）作为真实应用案例（Table VIII）：144 GPU 分为 4×36 / 2×72 / 1×144 集群，MoE 层用专家并行（EP）、其余层用数据并行（DP），总 batch=2048。结论：prefill 处理长序列大 batch、compute-bound，偏好低 EP 度（减少 AllToAll 开销）；decode 短序列/步，偏好高 EP 度大集群以增大有效 batch 提升吞吐——例如 decode step time 227.5→163.7 ms（36→144 GPU）、吞吐 62.5→86.9 tokens/s/GPU；prefill 吞吐 7097→3911 tokens/s/GPU（低 EP 更高）。

Tetris 补充视角（ISCA'26，CDSP 下两阶段的并行度异构分配）：prefill 与 decode 的并行偏好相反——prefill 受益于小 TP（SP 分配更灵活，调整 SP 只重分 token、不需重分片权重）与可变 SP（长请求大 SP、短请求小 SP）；decode 受益于大 TP（压低计算延迟，TP=8 vs TP=1/2/4 的 decode 延迟低 1.93×-5.73×，LLaMA3-8B 实测），且 SP 对 decode 不如 TP 有效（(SP8,TP1)/(SP4,TP2)/(SP2,TP4) 相对 (SP1,TP8) decode 延迟高 1.83×/1.41×/1.15×，因 decode attention 计算量小、无法掩盖 ring 通信）。Tetris 据此采用 prefill-decoding 解耦：prefill 统一 SP 池（TP=1 for 8B）+ decoding 大 TP DP（TP=8），同一请求的 prefill 用 CDSP 多 chunk 不同 SP、decode 固定在大 TP 实例上 continuous batching。评估指标即两阶段各自 SLO：TTFT（prefill，含排队+计算）与 TBT（decode 每 token 延迟，P50/P99）。

Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载的 prefill/decode 资源发散量化）：推理模型（OSL≫10k、CoT 长轨迹）使系统 >99% 墙钟时间花在 decode 阶段。遥测（8B/405B/671B，batch 100–2000）显示两阶段如同"两台机器"：prefill compute-bound——SM 占用高、HBM 带宽仅 ≈30%（8B）/≈20%（405B、671B），算术强度高（GEMM 权重复用），KV footprint 低且瞬时；decode bandwidth-bound——HBM 带宽饱和 ≈85%（8B）/≈65%（405B），算术强度塌缩（每 token 需读全部权重+活跃 KV），KV 持续累积。该发散是论文主张"prefill/decode 架构解耦"（prefill 高 TFLOP 加速器 + decode 内存中心化层次）的物理依据。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

MERIDIAN 补充视角（ISCA'26，KV-precomputed RAG 的两阶段变形）：RAG 的 prefill 只编码短 query（平均 ~16 token），文档 KV 已预计算复用，因此 prefill 退化为低算术强度的 skinny GEMM（17×d_model，batch 复用有限）；decode 每步做 query KV 与整份文档 KV（数千到上万 token）的注意力 GEMV，加上小 batch 下 FFN 也退化为 GEMV——attention 与 FFN 全程 memory-bound（roofline 图 3），H100 类加速器算力严重闲置。MERIDIAN 的对应：(a) 算法层——文档注意力分解把文档侧注意力移到 PIM 就地执行，decode 每步只搬 query 向量、不回搬文档 KV；(b) 硬件层——PIM 基板为 skinny GEMM 用"buffer 级复制（共享算术单元）"实现权重复用（对比 PAPI 的 full-datapath 复制），decode 的 GEMV 与非线性（GeLU/Swish 用 LUT 分段线性、softmax 专用精度硬件）全在内存侧执行。实测：MERIDIAN 通信占比 ≤6.34%（baseline 最高 93.40%），吞吐相对 CPU-GPU baseline 5.36×/6.64×、相对 PIM baseline（CENT/PAPI）3.98×/3.32×。
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference

## RoPE（Rotary Position Embedding，旋转位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RoPE 按绝对位置对 Q/K 向量的相邻两维做二维旋转（角度 θ_i 与位置 m 成正比），使注意力分数只依赖相对位置，支持训练长度外的外推。计算等价于复数乘法：对每对相邻元素 (x_{2i}, x_{2i+1}) 乘旋转矩阵，即乘 cos/sin 系数 + 相邻元素交换与取反。向量化 SIMD 阵列（DRAM-PIM 行粒度操作）做这种标量级邻居交换很别扭：CompAir 指出传统 DRAM-PIM 的 RoPE 要把数据搬回 CXL 控制器的 CPU 做 swap 与奇位取反（行粒度搬运），长上下文下开销大——这是"NoC 在途数据重排"的直接动机之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的 RoPE 流程：① NoC_Exchange(R-, SrcRow, DstRow, 1, 2)——在 NoC 内完成相邻元素对调与奇数位取反（4 router 五阶段交换、ArgReg 作缓冲，swap 目标为 (x+Offset)%Group）；② DRAM-PIM 以 EWMUL（元素乘）乘预存的 cos/sin 系数。伪代码：
```
for pair (x_{2i}, x_{2i+1}):
    y_{2i}   = x_{2i} * cos(m*θ_i) - x_{2i+1} * sin(m*θ_i)
    y_{2i+1} = x_{2i} * sin(m*θ_i) + x_{2i+1} * cos(m*θ_i)
# NoC_Exchange 完成"对调+取反"（旋转的符号部分），EWMUL 完成系数乘
```
数据流：DRAM row → NoC Exchange（在途重排）→ EWMUL → 写回 DRAM row，全程不离开 PIM 设备。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：预计算 cos/sin 表、旋转只作用于 head 维（可选部分维度）；GPU 上以融合 kernel 实现（避免物化中间张量）；PIM 上需要专门的数据重排机制（NoC in-transit exchange）或专用 NLU/CPU 往返。使用方式：Llama/Qwen 系标准位置编码（与 GQA/MLA 组合）；硬件侧按"标量重排 vs 向量乘法"拆解——重排走 NoC、乘法走 bank 内 EWMUL。

P3-LLM 补充视角（ISCA'26，NPU-PIM 边缘 LLM 推理）：RoPE 决定 KV cache 量化的位置选择——profiling 发现 RoPE 旋转对 key cache 分布的影响取决于模型最大序列长度：Llama-2（4K 序列）旋转角大，post-RoPE key cache 的结构化 outlier 被打散（不利于按通道量化），故采用 pre-RoPE 量化 + 每轮 decode 在 NPU 在线对 key 做 RoPE（元素级操作、开销可忽略），此时量化 key 缺位置信息、Q·K^T 留在 NPU 高精度执行；Llama-3/Mistral（128K 序列）在典型 4K context 下旋转角极小、post-RoPE 分布几乎不变（保留结构化 outlier 便于动态平滑），故采用 post-RoPE 量化，量化 key 可直接与 query 相乘、Q·K^T 可 offload 到低精度 PIM PCU。

Tetris 补充视角（ISCA'26，RoPE scaling 作为上下文扩展手段）：Tetris 评估 LLaMA3-8B/70B 的 context-extended 变体（RoPE scaling，即位置插值类方法）以支持其工作负载中的 190K+ 上下文窗口；RoPE 在此处的作用是让预训练于较短上下文（8K）的模型在推理时外推到长上下文，使 serving 论文能在 A100 集群上以真实长上下文负载（Short/Medium/Long 三条 trace，最长 190K）评估 CDSP 的 SP 调度收益。注意与 serving 调度正交：RoPE scaling 只改位置编码参数，不改变 KV cache 布局或并行策略，因此可与 CDSP/ring attention 自由组合。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation

## GQA（Grouped Query Attention，分组查询注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GQA 让多组 Q head 共享同一组 K/V head（如 Llama2-70B 的 8 个 Q head 共享 1 个 K/V head），在接近 MHA 质量的同时大幅减小 KV Cache 与 K/V 投影计算量（Llama3 同样采用）。对 PIM 系统的意义（CompAir）：K^T/V 权重被多个 Q head 复用，等价于给 K^T/V 引入 batch 级复用——普通 MLA/MHA 下 K^T/V 输入相关、每次推理都变，只适合 DRAM-PIM；GQA 下 K/V 共享使 SRAM-PIM 加速 attention 成为可能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的 GQA 映射：TP 沿 seqlen 切 K^T/V → SRAM-PIM 的 batch 维 = 序列长度段、输出维 = GQA group size（Llama2-70B 为 8）、输入维 = hidden size（QK^T）或 seqlen（SV）。
```
# Llama2-70B GQA：8 Q heads 共享 1 组 K/V
for q_head in group:
    score = Q_q @ K_shared^T      # K_shared 被复用 8 次 → 有复用
    out_q  = softmax(score) @ V_shared
```
权衡：长序列必然带来更多 cross-die 传输与更高能耗；QK^T 是否用 SRAM-PIM 取决于 TP 与 seqlen 组合，SV 恒用 DRAM-PIM（能量优势）。结论规则：GQA 的共享结构把 K/V 从"每次推理变"变成"可复用"，据此决定硬件分派。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：模型参数 num_key_value_heads < num_attention_heads，KV Cache 按 KV head 数存储、attention 时 repeat_kv 展开。使用方式：降低长上下文 KV Cache 内存与 K/V 计算量（对比 MLA：MLA 用低秩压缩、GQA 用 head 共享）；硬件侧按"是否有 head 级复用"决定映射（CompAir：复用→SRAM-PIM 候选、无复用→DRAM-PIM）。CHIME 的 GQA-8 使用（QWEN-72B）：bank PU 的计算访存比按 group 放大（N_cmr=8 才能满带宽利用），跨 chip 传输放大 N_gqa 倍使 bubble-free 条件更紧，head 映射只能取 N_hc=1（每 head 单 chip）；调度器对 GQA 的 decoding 请求加批步长取 N=16（head 少、rank 负载均衡难），且 GQA 更难达到 attention 瓶颈（需更大容量请求）。

PLENA 补充视角（ISCA'26）：GQA 从硬件利用率角度产生"per-head fat GEMM"问题——head_dim 小（LLaMA-3-70B 为 128）且一个 K 头被多个 Q 头同时相乘（GQA 组内复用），在方形大脉动阵列上 per-head GEMM 的计算维度小、利用率低。PLENA 的解法：FlashAttention 阶段把扁平化脉动阵列切分成多个小 flattened core，每个 core 执行 (BLEN,HLEN)×(HLEN,BLEN) 的 per-head GEMM、并行覆盖 MLEN//HLEN 个 Q 头（head 预加载），使 attention 计算与有效 batch 解耦——decode 长上下文（有效 batch 小）下仍保持高利用率；FFN 阶段同一阵列以 (BLEN,MLEN) 形态跑 fat GEMM（BLEN≈batch）。即同一阵列通过 BLEN/头级分解同时适配 FFN 与 GQA attention 两类 GEMM 形状。

ConServe 补充视角（ISCA'26）：评估模型均为 GQA——Yi-6B-200K（Hq=32、Hkv=4、L=32）、Llama-3-8B-262K（Hq=32、Hkv=8、L=32）、Yi-34B-200K（Hq=56、Hkv=8、L=60）。KV footprint 按 KV 头数计算：B_tok=2·L_shard·H_shard·d_head·b（H_shard=TP 分片上的 KV 头数，Llama-3-8B BF16 约 4 KB/token/layer）；GQA 的小 KV 头数直接决定 slice 每层段尺寸（B_layer=2×H_shard×d_head×b）与 resize 触发频率，K/V 共享结构不影响 ConServe 的连续布局（K、V 都落进该层段）。

Raptor 补充视角（ISCA'26）：GQA 直接决定 KV cache 足迹与带宽需求——Llama-3.1-70B（80 层、8 KV 头、head dim 128）每 token 产生 2×8×128×q B 的 KV 状态（q=1B 时 8-bit 2KB、q=2B 时 FP16 4KB）；KV 读带宽下界 D_KV ≥ S·(2LH_KV·d_head·q)+2LH_KV·d_head·q 中的 H_KV 即 GQA 的 KV 头数（比 MHA 小一个"Q 头/KV 头"组比），因此 GQA 在容量与带宽两个维度同时降低 decode 压力。Raptor 以 KV 为中心设计 3D-DRAM：单层 4K 上下文 16MB KV cache 切成 1024 个 16KB stream-blocked tile 摊到 16 channel（每 channel 3 bank、128B flit），KV tile 粒度（16KB）与 paged-attention ≥4KB page 对齐；GQA 的小 H_KV 使 per-token KV 字节少，单卡 32GB 3D-DRAM 即可容纳 Llama-70B 权重+KV（TP=1），这是其"低并行度、低 collective、网络不敏感"部署的结构前提。

P3-LLM 补充视角（ISCA'26，NPU-PIM 边缘 LLM 推理）：GQA 使 KV cache 容量按 group 数 G 缩小、算术强度 >1（区别于早期 MHA 的算术强度=1），因此低 batch 解码下 PIM 不再天然占优——roofline 显示 HBM-PIM 在 batch≥4 或 GQA 场景相对 NPU 优势消失，这驱动 P3-LLM 设计吞吐增强 PCU（TEP：时间维输入复用使同一 KV/权重切片在 tCCD_S 窗口内服务两个输入）并保持 attention 全模块在 PIM 执行（4-bit KV + 8-bit 注意力分数）。Llama-3.1-8B（G=4）与 Llama-3.2-3B（G=3）在 batch 2-64 下 attention 仍占主导，P3-LLM 借高内部 PIM 带宽 + TEP 优于 Ecco。

QiMeng-Tensify 补充视角（ISCA'26）：GQA 作为图级编译优化 benchmark 子图（Table VII，Arch. 列标注 LLaMA3-70B）——代表"多算子 + 复杂数据依赖"的 attention 类子图（Q 投影、K/V 投影、QK^T、softmax、SV、输出投影的图）。QiMeng-Tensify 把它用于 LLM prior 消融（Fig.8b，GQA 与 GatedMLP 并列：LLM 先验比统计先验高 20%-30%）与可移植性分析（L0/L1/L2，GQA 为三个代表子图之一）。同一 benchmark 还含 QKNorm（Chameleon-7B 的 GQA+QK 归一化）与 SelfAtten 等 attention 类子图；QKNorm 上 QiMeng-Tensify 超出 FlashAttention 1.66×、TensorRT 1.40×——表明自动图优化能超越注意力专用手写优化。图级视角：GQA 的 K/V head 共享（8 Q head 共享 1 组 K/V）在编译层面表现为可被 compute_at/tiling 利用的数据复用结构，但本论文关注点在"多算子图调度"而非 PIM 映射。

  - SHyLA 补充：GQA 减少 KV head 数 → KVCache 写事务相对 Weight/KVCache 读进一步被抑制（图 4b 红虚线），KVCache 压力下降 → 微批可更大 → DSE 中 NVM 偏好从"容量"转向"带宽"（GQA 模型如 Mixtral 8×22B 偏好更高 NVM 带宽）。SHyLA 对 GQA 的映射：decode 请求按 attention-group/request 级并行（每组在一个 tile 内处理，消除组间 KVCache 共享的跨 tile 传输）；同一 group 内多个 Q head 复用同一 K/V 对，KVCache 重载次数由 tile weight buffer 容量决定（装得下则只载一次）；并行化策略取决于 attention group 数 g 与张量并行度 pt（g≥pt 按组分配，g<pt 用 sequence parallelism 子切分）。
Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载下 GQA 的 KV 足迹量化）：GQA 的 8 KV head 配置（Llama 系蒸馏模型）把 KV footprint 相比 MHA 降 3×–8×，但内存成本仍随层数线性增长——32B 模型（≈64 层）FP16 下 ≈262 KB/token，70B 模型（≈80 层）≈328 KB/token，Llama-3.1-405B（≈126 层）≈1.05 MB/token；128 请求 × 10k reasoning token 的 batch 仅 KV cache 就超 1.3 TB（远超单 H200 的 141 GB）。论文用该数字论证"KV 容量是推理第一瓶颈"：GQA 只缓解、不消除 decode 容量压力，容量墙出现于 prefill 与 decode 交界（batch 4K/5K 时 KV 在 prefill 阶段即耗尽）。
涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

## LLM 非线性算子（Softmax / SiLU / RMSNorm）与超越函数近似（Taylor / Newton 迭代）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transformer 每层的非线性算子：Softmax（attention 权重归一，含 exp 与跨序列归约，延迟随序列长度线性增长）、SiLU（FFN 激活，sigmoid 系）、RMSNorm（RMS 归一，含 sqrt）。这些算子没有直接硬件原语，数字电路通常用迭代法近似：exp 用 Taylor 展开截断（e^x ≈ 1+x+x²/2!+…+x^n/n!）、sqrt 用 Newton 迭代。CompAir 实测这些非线性不可忽略：4K 上下文时占 block 时间约 20%、长上下文时通信+计算可超 25% 总延迟——推翻"非线性可省略"的假设，成为 CompAir-NoC 的核心动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的近似 pipeline（Curry ALU、BF16）：
```
# exp 的 Taylor 截断（自最内层向外），ArgReg 作迭代计数器
ArgReg, IterArg, IterOp = 6, 1, '-'
acc = 1
while ArgReg > 0:
    acc = acc * X / ArgReg + 1   # *=X, /=IterRound, +=1
    ArgReg -= 1                  # IterTag 触发
# sqrt 同理 Newton 迭代；每通道 16 bank × 2 ALU = 32 路并发
```
精度验证（Table IV，Llama2-7B perplexity）：FP32 vs 原生 BF16 vs Taylor n=4/5/6/7，三档上下文（prefill 73/341/1139 + decode 15/65/270 tokens）：相对偏差 <0.3%（最显著 medium 档 n=5..7 相对 FP32 为 −0.251%），误差不随上下文增长累积。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LUT + 多项式、CORDIC、迭代法；GPU 上以 exp2f/__expf 等近似指令；PIM 上用流式迭代（Curry ALU，零额外流水级）或专用 NLU（CENT 的 NLU 为 7nm 4.4mm²，约 4× 一个 32MB bank）。使用要点：迭代轮数决定精度-延迟权衡（n=4..7 均已验证可接受）；sqrt 的 Newton 迭代需初值估计与除法支持（Curry ALU 每 ALU 含 1 个 divider）；归约与 exp 在途合流避免中间结果搬移。

QiMeng-Tensify 补充视角（ISCA'26）：SiLU 在图级编译中的优化——GatedMLP 的 SiLU 由 exp/add/div/mul 四个子算子组成，初始为独立 kernel + 中间 buffer；QiMeng-Tensify 的 LLM 识别 SiLU(x)=x/(1+e^(-x)) 可融合表达式后，用 AutoInline 把四个子算子折叠为单一 SiLU block，再 compute_at 把 SiLU 提升进 GEMM 的 reduction 循环（S1→S4→S5），完全消除中间 buffer 与 kernel launch 开销。这体现非线性算子在编译框架层的"融合价值"：不止是近似精度问题，还有"可折叠进 GEMM 循环体避免物化中间张量"的访存优化。同样，RMSNorm/LayerNorm 子图（RMSNorm-LLaMA2-7B、LayerNorm-Transformer、nTrans 的 Norm+residual 融合）被用作 benchmark（Table VII），QiMeng-Tensify 通过布局处理与融合策略优于 TVM/Triton。注：本论文非线性算子不做数值近似（与 CompAir 的 Taylor/Newton 不同），其优化是"算子融合与调度"层面的。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS

## FlashAttention 式 chunked tile attention（tile 化 attention + online softmax + kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashAttention 把 attention 沿序列维切成 chunk（tile）计算：每个 chunk 内算局部 score→局部 softmax，跨 chunk 用 online softmax 的 running 统计量（running max m、running sum l）增量修正合并，全程不物化完整的 S=QK^T 矩阵（O(n²) 显存降为 O(n)），并通过 kernel 融合减少中间读写。CHIME-PIM 借鉴该思想但动机不同：bank PU 的 result buffer 有限、rank PU 片上 SRAM 有限，chunk 化能约束中间 head 足迹并使 score（bank PU）与 softmax（rank PU）跨单元流水；其 chunk 定义 = 单 head 跨（可能多个）bank PU 并行产生的数据，per-chunk softmax 后做 streaming 跨 chunk 归一化得到全局正确 S，S 元素写回与 context（S×V）计算再流水。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
online softmax 的核心三量维护（chunk 粒度，等价 FlashAttention 算法）：
```
m = -inf; l = 0; O = 0              # running max / sum / 输出累加
for chunk in chunks:
    s_chunk = Q @ K_chunk^T         # 局部 score（CHIME：bank PU 算，写 result buffer）
    m_new = max(m, rowmax(s_chunk))
    l = l * exp(m - m_new) + rowsum(exp(s_chunk - m_new))   # 修正 running sum
    O = O * exp(m - m_new) + exp(s_chunk - m_new) @ V_chunk # 修正输出
    m = m_new
S = O / l                           # 最终归一化
```
CHIME 的流水映射：score（bank PU）→ rank PU 取回（外部总线，与下一 chunk 的 MAC 重叠）→ adder 累加 + per-chunk softmax（rank PU）→ 全部 token 后 streaming 归一化 → S 写回 DRAM 与 S×V 流水。该融合保证数学上精确（与逐 token softmax 等价），不损失精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 上 FlashAttention-1/2（tiling + softmax recomputation + warp 协作 + 寄存器缓存，后续 FlashAttention-3/FlashDecoding 扩展）；PIM 上把"chunk 流水"映射到 bank PU/rank PU 的异步执行（CHIME），中间量经 result buffer/adder/softmax 单元流式传递。使用方式：任何长序列 attention（训练与推理）减少峰值显存/缓冲与中间往返；在 CHIME 中是 bubble-free pipelining 与跨 chip 传输隐藏的算法前提（chunk 越小流水越细、head 足迹越小，但传输次数越多——由 T_comm ≤ T_comp 约束 head 映射 N_hc 平衡）。

PLENA 补充视角（ISCA'26）：把 FlashAttention 从"GPU kernel 融合"下沉为"加速器 ISA 原生支持"——论文归纳现有 systolic 加速器无法原生支持 FlashAttention 的四个缺口：(1) 无 tile 级 off-chip 预取与计算重叠；(2) 无 transpose-on-read 与高效跨步/分块流式内存布局；(3) 只暴露 GEMM 原语，缺 online softmax 所需的行内归约与非线性（max/sum/exp/div）；(4) ISA 固定调度、粗粒度 kernel 边界，阻碍 tile-by-tile 融合。PLENA 对应机制：(1) H_LOAD_M/H_LOAD_V 指令控制的 SRAM 硬件预取引擎；(2) 转置可读 Matrix SRAM（见硬件架构库条目）；(3) vector/scalar 单元实现归约与元素操作，VLEN 可配对齐 tile，softmax 计算精度可配（常用高精度如 FP12）；(4) 47 条自定义 ISA 的 tile 级持久调度，把 QK^T→online softmax→PV 逐 tile 编排。收益：大中间激活留片内（Vector SRAM）不回写 off-chip，显著减少内存流量——这是长上下文内存墙下的关键收益；同时 aggressive 预取重叠隐藏 HBM 延迟。

MERIDIAN 补充视角（ISCA'26，跨设备 online-softmax 融合）：MERIDIAN 的文档注意力分解把 online softmax 的 running 统计量从"chunk/GPU SM 粒度"推广到"PIM 设备/分支粒度"——文档分支与上下文分支各自算局部 (o_d,m_d,l_d) 与 (o_c,m_c,l_c)，全局融合用共享基线 m=max(m_d,m_c) 后 l=e^{m_d-m}l_d+e^{m_c-m}l_c、o=(e^{m_d-m}o_d+e^{m_c-m}o_c)/l，与 FlashAttention 的 running max/sum 修正合并同构（等价 LogSumExp 结合律的两步归约，见 Tree Attention 跨 GPU LSE 归约）。关键差异：两个"chunk"分处不同物理设备，只有紧凑统计量跨设备交换（每设备只传 d_model/N 输出切片 + 2 个标量），无需像集中式那样聚拢全部 attention logits；softmax 本体用专用精度硬件（NMU 内 softmax 单元）保证数值稳定，而宽容算子（GeLU/Swish）用 LUT 分段线性。对比 M-non 消融（softmax 留在 GPU 需聚拢完整 logits、通信随文档长线性增长）仅 1.69×，MERIDIAN 全量 5.36×——证明跨设备 online-softmax 融合是去中心化 RAG 的通信关键。

QiMeng-Tensify 补充视角（ISCA'26）：FlashAttention 作为"手工专家优化"baseline（v2.7.3）与 benchmark 子图——(1) 子图级：SelfAtten 等 attention 子图上对比时使用 FlashAttention-V2（"FlashAttention-V2 used in FlashAttention, Triton, and QiMeng-Tensify"，Fig.9 注），QiMeng-Tensify 平均快 1.27×（FP16），但 NSA 等新型稀疏注意力上专家 FlashAttention 仍领先（论文承认 slightly behind expert-coded FlashAttention）；(2) 定位：FlashAttention 系列（FA1/FA2/FA3）是"手工专家优化"路线的代表（labor-intensive、难以随模型创新扩展），其 chunked tile + online softmax + kernel 融合是编译器/自动搜索（如 QiMeng-Tensify 的 MDP 图重写）试图自动复现的对象——对 QKNorm 子图 QiMeng-Tensify 甚至超过 FlashAttention 1.66×，说明自动图级优化可覆盖部分专家设计空间。

- SMOOTH 用法（ISCA'26）：FlashAttention 作为三种常见算子融合之一（QKV 投影融合、FlashAttention、FFN 融合）被建模为编译器的静态优化——融合虽提升数据复用与 locality，但强制 Q/K/V 激活同 kernel 同时存活、拉长中间 buffer lifetime，与 QKV/FFN 融合一起造成严重片上碎片；SMOOTH 的 block 级虚拟化 + early reclamation 正是为缓解这类融合带来的碎片与长 lifetime 而设计（融合版本下才能激进预取，明显降延迟；无融合时各策略收益都受限）。
涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference

## Speculative Decoding（投机解码/推测解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
投机解码是加速 LLM 自回归 decode 的无损推理范式：用轻量草稿模型（draft model）快速自回归预测 γ 个候选 token，再由目标模型（target model，即原始 LLM）对这 γ 个 token 做一次 batched 前向并行验证，按接受规则保留正确前缀。因为 γ 不大且 decode 阶段本就 memory-bound，γ-token 验证前向耗时近似单 token 前向：总时间 ≈ γ·t_draft + t_target，t_draft ≪ t_target 时收益显著。接受率（acceptance rate）= 草稿平均每轮被接受的 token 数，是加速比的决定因素；接受判定贪婪解码下要求草稿与 target 输出完全一致，采样解码下用概率接受（见 Speculative Sampling）。别名：speculative execution for LLM、draft-then-verify。代表工作：Leviathan et al.（arXiv:2211.17192）、Chen et al.（arXiv:2302.01318）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
x = prompt
while 未生成 EOS:
    drafts = []
    for i in 1..γ:                          # 草稿阶段：轻量、自回归
        q_i = draft(x + drafts)
        drafts.append(sample(q_i))
    p_1..p_γ+1 = target(x + drafts)         # 验证阶段：一次 batched 前向
    n = 接受判定(p, q)                       # greedy: 首个不一致前；采样: rejection sampling
    x += drafts[:n] + target_sample(x + drafts[:n])
```
本文（Cassandra）的用法：草稿不是独立小模型，而是目标模型权重/KV 的严格比特子集（speculation data，zero-padding 重建后标准 FP GEMM）；验证加载 speculation+verification 全量数据完全重建原始模型。γ 在 3–5 内取最优（Cassandra-1 γ=5、Cassandra-2 γ=3）。对比基线：EAGLE-3（训练型草稿，4×A100 约两天）、Draft&Verify（层跳过）、MagicDec（KV 稀疏检索）、Lookahead Decoding（n-gram）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：vLLM/SGLang/TensorRT-LLM 内置（支持 Medusa、EAGLE、ngram、独立草稿模型等），llama.cpp 支持草稿模型，HuggingFace transformers 的 assistant model 模式。硬件系统方向：PIM 加速草稿模型、FPGA（DFVG）。代价与适用边界：训练型草稿有训练成本与额外显存；低 batch 下部分方法（MagicDec 的 KV 剪枝）甚至慢于 baseline；训练无关变体见 Self-Speculative Decoding。

Raptor 补充视角（ISCA'26）：投机解码被定位为 Raptor 与 HBM-class GPU 的异构配对（与 AFD 相反）——draft 阶段是 K 步背靠背自回归、内存受限，放 Raptor（~100TB/s 压低串行 draft 关键路径延迟）；verify 阶段把 target 权重摊到 K 个 token 上、计算受限，放 GPU tensor core。Raptor 前代 Corsair 已在该配对下有规模化端到端加速的生产部署先例（Gimlet Labs 博客），NVIDIA Vera Rubin 平台（Rubin GPU + Groq 3 LPX）也把投机解码列为目标用例。这体现"draft=内存受限、verify=计算受限"的不对称性可被高带宽内存基板放大的系统级收益：draft 模型跑 Raptor 时每次推测步的 KV/权重带宽成本远低于 HBM 基板。

HybridSpec 补充视角（ISCA'26，SD 的"内存需求极化"与硬件设计杠杆）：SD 不止是延迟优化算法，其 draft/target 分裂天然极化内存需求——draft（体积 <1/10 target）逐 token 自回归、算术强度低、内存受限 → 需高带宽但仅小容量；target 一次验证多个 draft token、算术强度高 → 容忍低带宽但需大容量（权重 + 增长中的 KV cache）。据此 HybridSpec 把 draft 放 HB 栈（4TB/s 高带宽）、target 放 XPU+LPDDR5X（512GB 大容量），模型级映射使通信只在 draft-verification 边界。论文覆盖 SD 全谱系：chain 式（draft 逐 token 自回归）、tree 式（多候选 + masked attention 并行验证，见 Tree-based Speculation 条目）、hidden-state 式（在 target 内嵌投机头，用最后 token + 末层表征预测，图 3(c)）；成熟实现（vLLM/SGLang/llama.cpp 内置）接受率可达 ~80%[39]。

从算法pipeline角度拆解（极化后的一次 draft-verify 周期）：draft 在 HB 栈按当前 tree width 生成候选树（memory-bound 迭代）→ 达 draft budget 后 token 列表传 XPU → target 一次 batched 前向并行验证（rejection sampling）→ accepted 前缀回传、清误推测 KV。budget/tree width 由 Utilization-aware Speculation 按两侧 roofline 动态调（见系统架构层条目）。

实现与使用：算法侧与通用 SD 相同（draft-then-verify + 拒绝采样保证 lossless）；系统侧的新意是"为 draft/target 各自选择内存基板"——这是 SD 从纯软件优化升级为"软硬件协同异构设计"的用法，与 SpecPIM（PIM 侧 DSE）等并存。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving

IroKnight 补充视角（ISCA'26，Fully-State Encrypted 投机解码）：IroKnight 论证投机解码不改变 LLM 的细粒度 tiled/vectored 执行——draft 与 target 模型本身及其算子不变，只新增"候选 token 验证"的比较操作，而该比较等价于对 token 索引向量的规则等值检查（仿射访问模式），因此 pad 预计算与同周期 in-ALU 加解密照常成立，可实现"全状态加密"的投机解码（模型参数、用户 query、draft token 等在所有存储中保持密文，明文仅瞬态于 ALU）。评估：draft Llama3-1B + target Llama3-70B（conventional）与 draft Llama3-8B + target Llama4-Scout（mixed-vocabulary）两组，drafter 3/5 token、接受率 alpha 0-100% 扫描：加密变体延迟开销 0.1%-0.5%（低接受率略高，因 rollback 需重灌 PadGen 流水线），认证变体 3.1%-3.3%；能量加密 9.4%-29.3%、认证 13.6%-33.9%（接受率越高 target 生成 token 越少、HBM 流量越低，能量开销反而升高）。结论：SD 这类"粗粒度动态性"与 IroKnight 的"细粒度规则加密"正交兼容。

## Speculative Sampling（投机采样 / Rejection Sampling 接受机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
采样解码下保证投机解码输出分布与目标模型逐 token 数学等价的概率接受机制（Chen et al. arXiv:2302.01318 的投机采样；Leviathan et al. arXiv:2211.17192 的拒绝采样证明）。对第 i 个草稿 token：抽 r_i ~ U(0,1)，若 r_i ≤ p_i(x)/q_i(x) 接受（p=target 概率、q=draft 概率），否则拒绝并在该位置从 max(0, p−q) 归一化残差分布重采样一个 token 后终止本轮；全部接受时从 target 分布补采 bonus token。该构造使整个系统的输出分布与直接对 target 采样不可区分——这是投机解码"无损"的数学基础；贪婪解码可视为其退化特例。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
n ← min({ i−1 | 1 ≤ i ≤ γ, r_i > p_i(x)/q_i(x) } ∪ {γ})      # 论文式 (1)
# r_i ~ U(0,1)；p_i(x)、q_i(x) 为 target/draft 在第 i 个候选处的 logits 概率
# 若 n < γ: 在位置 n+1 从 max(0, p_{n+1} − q_{n+1}) 归一化分布重采样一个 token
# 若 n = γ: 用 p_{γ+1} 采样 bonus token
```
本文用法：Cassandra 在采样模式下用该机制接受草稿 token，保证与 BF16 目标模型输出分布一致（Cassandra-1 完全无损、精度表 III 与 BF16 逐项相同）；实测接受率 Cassandra-1(γ=5) 0.74–0.88、Cassandra-2(γ=3) 0.74–0.91（按模型/benchmark）。接受率高低由 draft 与 target 的分布接近度决定——这正是 Cassandra 用细粒度剪枝+截断而非粗粒度层跳过构造草稿的原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
几乎所有投机解码系统默认实现（vLLM/SGLang/TensorRT-LLM/llama.cpp 均内置）；实现要点：draft 与 target 共享同一次验证前向的 logits，对 p_i/q_i 向量化比较，拒绝位置只重采样一次。局限：低接受率时每轮只能推进 1 个 token（比贪婪更糟），故实际系统常混合 greedy 验证。

HybridSpec 补充视角（ISCA'26）：论文以"speculation efficiency = 接受 token 数 / draft budget"（非链式接受率，因树形不满足链式 Markov 假设）扫 draft 精度对延迟的敏感性（Fig.21）：效率上升延迟下降、超过某水平后边际趋缓——draft 精度达到一定程度后其对延迟的影响有限；chain 式方法按 (1-α^B)/((1-α)B) 折算（α=接受率）落点在这些参考线左侧，tree 式在同预算下效率更高、落在右侧区域。接受/拒绝由 target 验证前向的 logits 与 draft 概率比较决定。

从算法pipeline角度拆解：验证前向同时得到 target 对各候选位置的概率 p，与 draft 概率 q 比较（贪婪：取首个不一致前；采样：r≤p/q 接受否则重采样），树形下沿树取最长接受前缀、分支被拒绝部分的 KV 清除。

实现与使用：rejection sampling 是 SD 无损性的基础（vLLM/SGLang 内置）；HybridSpec 用它做 draft 精度敏感性分析（speculation efficiency 扫描）与 KV 回滚边界确定。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

## Self-Speculative Decoding（自投机解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
训练无关的投机解码变体：草稿模型直接由目标模型自身派生，无需额外训练、不引入独立草稿权重。派生手段三类：(1) 层跳过——Draft&Verify（贝叶斯优化选跳层）、Swift；(2) KV cache 简化——MagicDec（KV 稀疏检索）、QuantSpec（KV 低精度量化）；(3) 参数位级子集/压缩——Cassandra。Cassandra 的构造：权重经 Wanda 非结构化剪枝、KV 经 per-token 幅度剪枝、再叠加 4-bit 尾数截断与指数压缩（unary/MX），把每个张量拆成 speculation data + verification data；草稿模型 = 严格比特子集（零额外显存，显存甚至低于 BF16 原模型），验证 = 全量数据重建原始模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
spec_W, ver_W = split(W, mask)            # mask = Wanda importance top-k
spec_kv, ver_kv = split(KV, mask)         # per-token 幅度 top-k
draft_logits = LM(spec_W, spec_kv, x)     # zero-padding 重建、标准 FP GEMM
drafts = [sample(draft_logits) ...]
p = LM(spec_W∪ver_W, spec_kv∪ver_kv, x + drafts)   # 全量并行验证（拒绝采样）
```
为何细粒度优于粗粒度：低 batch、中等序列长度下 decode 瓶颈从 attention 转移到 FFN 权重加载——层跳过方法不压缩 FFN 权重（Draft&Verify 对 32 层模型只跳 9 个 FFN 层、草稿仍须加载 70.7% 参数），KV-only 方法（MagicDec）低 batch 下有时慢于 baseline；Cassandra 对权重与 KV 都做位级压缩，直击 FFN 带宽瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表实现：Draft&Verify、MagicDec、Lookahead Decoding、Swift、QuantSpec、Cassandra。适用场景：边缘低 batch（单/少数并发用户，batching 不可行）、资源受限（无训练算力、显存紧张）。局限：接受率与跨任务泛化一般弱于训练型草稿（EAGLE-3 在 AIME2025/GPQA 更优但依赖训练数据分布、长序列任务收益骤降）；MagicDec 依赖 KV 剪枝在低 batch 失效。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

## Wanda（Activation-aware Weight Pruning，激活感知权重剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（ICLR 2024，Sun et al.）是免重训练的 one-shot 非结构化权重剪枝方法：重要性分数 S_ij = |W_ij| · ‖X_j‖₂，其中 ‖X_j‖₂ 是用校准集算出的第 j 个输入通道激活的 L2 范数，与权重逐元素相乘后按输出行 top-k 保留（如 50% 稀疏度）。直觉：LLM 存在 emergent large-magnitude features——激活在固定通道持续大值，这些通道对应的权重更关键，单纯幅度剪枝（如 SparseGPT 无修正）会误剪。校准成本极低：约 128 个样本、单次前向，无梯度、无重训练（区别于 SparseGPT 的权重更新）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
X = forward_calib(model, calib_set)        # 约 128 样本，单次前向
for layer in linear_layers:
    a_j = sqrt(sum(X_j^2))                 # 输入通道激活 L2 范数（per-channel）
    S_ij = |W_ij| * a_j                    # 逐元素重要性
    mask = topk_per_row(S_ij, k=(1-p)*N)   # 每输出行保留 k 个
    W_pruned = W ⊙ mask
```
本文用法：Cassandra 用 Wanda 选择权重的 speculation 组（默认 40% 剪枝）；被剪掉的权重不丢弃，进入 verification 组供 target 前向使用——把"损失压缩"变成"无损投机"的关键。对照实验：Wanda 单独作损失压缩在推理 LLM 上精度崩塌（Deepseek-R1-Distillated-Llama3-8B：GPQA 16.0、Math-500 33.0、AIME2025 0.0 vs BF16 49.0/87.0/26.7），而 Cassandra-1（Wanda 草稿 + 全量验证）与 BF16 逐项相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：locality0/wanda（PyTorch 实现，支持 Llama/Vicuna/Bloom）；HuggingFace 有社区 Wanda 剪枝权重。使用：LLM 压缩（50–85% 稀疏）、量化前置（SqueezeLLM 结合）；Cassandra 式用法（草稿构造）与 vLLM 2:4/Wanda 稀疏集成。局限：非结构化稀疏在稠密 GEMM 上无直接延迟收益（需 2:4 结构化或专用稀疏 kernel/硬件）；离线校准存在 domain shift（μ-MoE 等指出）。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

## Per-token Magnitude Pruning（KV cache 每 token 幅度剪枝 / Mustafar）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对 KV cache 按 token 维度做非结构化幅度剪枝：每个 token 的 K/V 向量内仅保留幅度最大的 k 个元素（其余置 0 或存 bitmap 稀疏格式）。依据（Mustafar，NeurIPS 2025，arXiv:2505.22913）：Key cache 存在显著的 channel-wise outlier（特定头维通道持续大值），per-token 剪枝天然保留这些 outlier 通道；Value cache 分布均匀、无通道 outlier，但 attention 中同一 token 的所有 value 元素乘同一个 attention score，按幅度剪枝在功能上等价于 output-aware 剪枝。相比 ThinK 等结构化（整 token/通道）剪枝，非结构化 per-token 可在 70% 稀疏下无精度损失，且压缩后可达 45% 的稠密内存占用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for t in tokens:
    K_t' = K_t ⊙ topk(|K_t|, k)           # 每 token 保留幅度 top-k 元素
    V_t' = V_t ⊙ topk(|V_t|, k)
# 稀疏存储：1×64 列 tile + 64-bit bitmap + tile offset
# decode attention = 压缩 KV 上的 SpMV（共享内存反压缩、compute-as-dense）+ 最近 32 token 局部 dense MV + online softmax
```
本文用法：Cassandra 对 KV cache 用 per-token 幅度剪枝生成 speculation 组、被剪元素进 verification 组；再叠加 4-bit 尾数截断与指数压缩（unary/MX）进一步降低 KV 带宽；KV 侧优化使 Cassandra 在长序列 benchmark（LongBench-QMSum）上增益最大，接受率 0.78–0.91。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Mustafar 开源：github.com/dhjoo98/mustafar（自定义 CUDA SpMV kernel：load-as-compressed、compute-as-dense、共享内存 tile 反压缩）。使用：长上下文 decode 加速（Llama-3-8B 上 batch 6→8、最高 2.23× tokens/sec）；与 KIVI 量化、H2O token 逐出正交叠加。局限：需要专用稀疏 attention kernel；不保留最近窗口会伤精度（保留最近 32 token 的 dense 窗口）。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

## Mantissa Truncation（尾数截断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
直接丢弃浮点数低位尾数比特以降低表示位宽的压缩手段，区别于量化：不改变数值表示（仍为浮点）、不做 round/scale 变换，仅保留高 w_t 位 mantissa、低位清零。两个关键性质：(1) 重建开销极低——zero-padding 即可恢复为完整浮点布局；(2) 截断后的草稿模型是目标模型的严格比特子集（比特级包含关系），使 self-speculative decoding 中"验证数据补齐即完全恢复原模型"成立，从而无需存储任何独立草稿参数、显存低于原始 BF16 格式。BF16 = 1 符号 + 8 指数 + 7 尾数；截断 4 位后有效尾数剩 3 位（指数另行压缩）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
W_draft_mant = (W_mant >> t) << t           # 保留高 t 位（默认 t=4），低位清零
# BF16 mantissa 7-bit → 截断后 3-bit 有效；低位在 decoder 拼接时恒为 0
# draft 前向：mantissa concatenator 把高/低位拼回 BF16 布局，dynamic shifter 对齐
```
本文用法：默认配置权重 40% 剪枝 + 4-bit 截断、KV cache 4-bit 截断（可直接迁移到其他模型）；设计空间探索（图 7）显示剪枝+截断联合使用比单独使用任一者的接受率-压缩率曲线更鲁棒（Deepseek-R1-Distillated-Llama-8B，γ=5）；超参由目标函数 J = α / (S_w(1−w_p)(B−w_t) + S_kv(1−kv_p)(B−kv_t)) 的 grid search 确定（剪枝 30–60%、截断 0–5 bit，8 样本 dev set，A100 约 5 分钟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现即移位与掩码操作（PyTorch bitwise）；硬件上由 Cassandra decoder 的 mantissa concatenator + dynamic shifter 完成拼接与指数对齐。与量化的对比：量化改变数值表示、可能引入反量化 scale 乘法（低 batch decode 下该开销不可忽略，W8A8 仅 1.3×）；截断无此开销但单用压缩率有限，本文将其与剪枝+指数压缩组合以弥补。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

## Unary Coding（一元编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
一种变长无损熵编码：符号按出现频率降序排列，频率第 i 高的符号编码为 i 个 0 后跟 1（1、01、001、…）。核心性质：码字边界由结尾的 '1' 显式标记，解码只需数连续 0 的个数，可用纯组合逻辑全并行实现——无需 Huffman 式的 LUT（2^N 项，LLM 指数 N 可达 32）或分层 codebook、无需顺序位解析。压缩效率略低于最优前缀码 Huffman，但对 LLM BF16 权重/KV 指数（Shannon 熵约 2.6/2.7 bits）unary 实际达平均约 2.85 bits/指数，配合可并行硬件解码，端到端收益反而高于 Huffman（Huffman 解码开销可能吞掉压缩收益）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
freq = histogram(exponents)                 # 统计 8-bit 指数频率
codebook = { e: '0' * rank(e) + '1' }       # 高频 → 短码，边界 = '1'
stream = concat(codebook[e] for e in spec_exponents)
# 硬件并行解码（Algorithm 1）：
#   chunks = stream 按 8-bit 分块
#   for chunk: 并行数连续 0；遇 '1' 输出 cnt 并清零；跨块把前一 chunk 末尾
#              连续 0 计数进位到下一 chunk（reorganized 位 + sum 累加）
#   Exp[idx] = UNARY_CODEBOOK(cnt)
```
本文用法：Cassandra-1 对权重与 KV cache 的 8-bit 指数做 unary 无损压缩——BF16 指数占位宽 50%，是剪枝+截断之后剩余的压缩率瓶颈；对应硬件为 parallel zero counter（8-bit 分块 + 跨块进位 + LUT 码本 + zero eliminator queue）。Huffman 因 LUT 规模（2^N，N≈32）与层次 codebook 的复杂解码被否决；Cassandra-2 则用 MX 共享指数（有损、压缩率更高）作为替代配置。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：前缀码表 + 移位解析（变长读取）；硬件：Cassandra decoder 的并行组合逻辑实现。同族编码：Rice/Golomb（unary quotient + binary remainder，适合指数分布整数），与 unary 共享"按频定长"思想。使用场景：低熵小符号集（浮点指数、残差、游程长度）的无损压缩，尤其需要低延迟全并行解码的片上数据通路。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

## 选择性旋转（Selective Rotation）与 QuaRot 式旋转量化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QuaRot（arXiv:2404.00456）利用计算不变性 Y=Wx=(WQ^T)(Qx)，用正交（Hadamard）旋转同时变换权重与激活以压制激活离群值，实现 W/A/KV 全 4-bit 量化；Hadamard 矩阵 H_2n=1/√2·[H_n H_n; H_n −H_n] 的相干性达到 Welch 界 μ=1/√n。QuaRot 定义四类旋转（R1-R4）：Q/K/V 投影与首个 MLP 投影的输入旋转（离线并入权重）、attention 输出投影旋转（离线）、attention Q/K 在线旋转（服务 KV 量化）、末层 MLP 投影输入在线旋转。PLENA 的"选择性旋转"是其 MX 格式适配变体：旋转只施加到增益为正的层子集 S（按每层 perplexity 增量 Δppl 搜索），权重不旋转（MX 小块共享指数已捕获权重离群，旋转反而增 PPL：MXINT4 权重 6.83→6.98），激活/KV 旋转在线执行并配 PLENA 硬件原生乘法支持（运行时乘 H^{-1}）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
l_rot*(X) = Q(X·H) · H^{-1} · Q(W)          # 只旋转激活，权重不旋转
S* = argmin_{s⊆M} Σ_{s∈M} Δppl(l_rot*)      # 按层搜索旋转子集（以 PPL 增量为准）
```
- KV 路径（PLENA）：新 K/V append 前做 Hadamard 旋转 → 量化为 MX 存 HBM → 读入 Matrix SRAM 后做逆 Hadamard 变换 → 进 attention GEMM；权重加载绕过逆变换（旋转/逆旋转可按张量选择性施加）。激活路径：XH 后量化、运行时乘 H^{-1} 复原（硬件 vector 单元提供旋转指令）。
- 消融结论（LLaMA-3-8B，Table VI）：激活/KV 量化中旋转有效——MXINT4 7.24→7.05、MXFP4 29.75→14.50（但 MXFP4 仍差于 MXINT4）；full-system 中 Erry 裁剪 7.60 + 选择性旋转 → 7.22。权重侧旋转则普遍有害。
- 与 QuaRot 的关键差异：QuaRot 把旋转并入权重（离线 R1-R4），PLENA 权重不旋转、激活/KV 旋转在线执行——因为 MX 小块共享指数的权重量化已能吸收权重离群，旋转的必要性只存在于动态范围宽的激活与 KV。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Hadamard 矩阵用块对角/Kronecker 结构（I⊗H_dh）低开销应用；旋转子集用逐层 PPL 增量启发式搜索；硬件侧把旋转做成向量单元指令、逆变换在数据加载路径上执行。使用：W/A/KV 全低比特量化时压制激活/KV 离群；注意与 RoPE 的兼容性（RotateKV 处理 RoPE 例外、用 outlier-aware 自适应旋转）。工具链：AMD Quark（ONNX/PyTorch）提供 QuaRot R1-R4 配置化实现、昇腾 MindStudio msModelSlim 将其作为离群值抑制算法；后继工作 SpinQuant（学习旋转矩阵）、ButterflyQuant（可学习蝶形变换，2-bit LLaMA-2-7B 上 PPL 15.4 vs QuaRot 22.1）、GSR（sequency 排序 Walsh 旋转）扩展了旋转族。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference

## 简并性与逻辑等价陪集（Degeneracy & Logical-Equivalent Coset）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
简并性（degeneracy）是量子稳定子码的关键属性：多个不同的物理错误模式产生完全相同的 syndrome 测量结果，且这些错误在物理上不可区分。任何错误 E 可唯一分解 E = s(E)·t(s)·l(E)：s(E)∈S 为稳定子分量，t(s) 为纯错误分量（仅由 syndrome s 决定，t(s)=∏_g T_g^{(1−s_g)/2}），l(E) 为逻辑分量。若 E1 = E2·S（S∈S），则 E1|ψ⟩ = E2S|ψ⟩ = E2|ψ⟩——两个错误对码字作用完全相同，称为退化错误。逻辑等价陪集（logical-equivalent coset）即固定 syndrome s 与逻辑错误 L 下所有 {E | E = S_g·t(s)·L, ∀S_g∈S} 构成的集合：同一陪集内的错误共享 syndrome 与逻辑效果，仅在稳定子变形上不同。核心结论：最可能的物理错误未必对应最可能的逻辑错误——最优解码应最大化逻辑后验 p(L|s) ∝ Σ_{E:l(E)=L} p(E) = Σ_{S∈S} p(E=St(s)L)，即对指数大小的稳定子群求和（coset ML），而传统 MWPM 求解的是物理 ML argmax p(E|s)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文（ISCA 2026）的例子（Fig.2）：syndrome {1,2,3,4} 可匹配为 {1,4}∪{2,3} 或 {1,2}∪{3,4}，两者各 6 条边等权等概率，但第二种含 9 种等价组合 → 其逻辑错误的总概率更高，两匹配分属不同逻辑陪集。这是把解码目标从"物理单链"改为"逻辑陪集"的直接动机：
```
E = s(E)·t(s)·l(E)              # 任意错误三分量分解
p(L|s) ∝ Σ_{E:l(E)=L} p(E)      # 陪集后验 = 对所有稳定子变形求和
# 精确求解指数复杂（|S| 随稳定子数指数增长，且解码 NP-hard）
# 本论文近似：聚类划分稳定子群(B_c) → K 次优先级采样 → 投票
p̃(L_i|s) = n_{L_i}/K;  Ê = argmax_{L_i} n_{L_i}/K   # Lemma 1/2, Eq.12
```
Lemma 1 证明：K 个候选中逻辑错误相同的 E_i 互为退化错误、属同一逻辑等价陪集；Lemma 2 证明聚类把全局 coset ML 松弛为聚类内局部最优（B_c 位串空间，仅簇内 nontrivial syndrome 可激活稳定子变形）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
文献中简并性在稀疏/qLDPC 码中讨论最多（Fuentes et al., IEEE Access 2021 系统研究其解码影响）；处理方式有：Tensor-Network 收缩精确解 coset ML（精度高但收缩复杂度高）、BP 类消息传递 + 后处理、以及本论文的采样-投票近似（多项式时间，介于 UF 与 MWPM 之间）。使用时注意：简并性使"物理错误率"与"逻辑错误率"两个指标分离，评估解码器应以 LER/系统保真度为准而非物理匹配最优性。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## 最小权完美匹配解码器（MWPM / Blossom / PyMatching）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MWPM（Minimum-Weight Perfect Matching）是表面码最经典、精度最高的解码器：把 syndrome 解码化为解码图（detector 图）上的最小权完美匹配问题——每个非平凡 syndrome（奇数奇偶顶点）需与另一顶点配对，边的权重取错误概率的负对数（权重随空间/时间距离增长），用 Blossom 算法（Edmonds 花算法，工业实现 Kolmogorov Blossom V）求全局最优匹配，配对路径即推断的错误链。它是"物理 ML"解码：解 argmax p(E|s)，在 LP 框架下迭代维护 blossoms。代价：算法与实现复杂度高、串行性强、延迟大；但精度是 baseline 金标准。加速变体：Fusion Blossom（Wu & Zhong, arXiv:2305.08307，Flower 图融合，GPU/FPGA）、Sparse Blossom（Higgott & Gidney，Quantum 9:1600，稀疏技巧 1M errors/core/s）、PyMatching（Higgott 的 Python/C++ 软件实现，detector error model 输入）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 解码图 G(V,E)，V=detector 事件，E=潜在错误(权 w_e=-log p_e)
O = {v ∈ V : v 为奇数奇偶(非平凡 syndrome)}    # 需配对的顶点
M = MinWeightPerfectMatching(G, O)           # Blossom 算法，保证最优
Ê = 由 M 的边组成的最可能错误链
# 缺陷(本文)：只优化物理单链 argmax p(E|s)，忽略简并性/陪集
# 本文用法：软件精度用 PyMatching 实现作为 MWPM 基准
```
本论文中 MWPM 是精度上限参照：p=0.002 circuit-level、d∈{3..19} 时本文 K=24 陪集集成解码 LER 与 MWPM 之比从 d=3 的 1.0× 渐增至 d=19 的 ~2.1×；硬件对照 Micro-Blossom（ASPLOS 2025，d=15 867k LUT @43 MHz，延迟随 d 陡增，d≥5 超单轮提取时限致系统 infidelity 受罚）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：Blossom V（C++ 库）、PyMatching（pip 包，输入 stim.DetectorErrorModel）、Fusion Blossom（带 Python 绑定）。硬件：Micro-Blossom（FPGA/加速）、Astrea（ISCA 2023）、Promatch（ASPLOS 2024，预解码扩展）。使用场景：追求最低 LER 的离线/内存实验解码，或作为其他解码器的精度上限基线；实时场景因延迟常需近似替代（UF、本文方法）。

补充（Triage 论文）：Triage 把 pymatching（Sparse Blossom，v2 的 C++ 核心）当作解码器延迟与抖动分布的事实来源而非评估对象——①延迟建模：profiling pymatching 在不同解码 volume 下的单次解码延迟，幂律拟合 t_dec=A·volume^α（α=1.17），用作调度仿真里每个 slice 的延迟（volume 由窗口缓冲大小/约束图 degree 决定）；②抖动校准：在 Stim 生成的 rotated surface-code 电路上逐 shot 测 pymatching 延迟（每设置 15K shots，warmup 后），拟合平均保持的 log-normal 抖动模型 t_actual=t_est·exp(−σ²/2+σz)，σ(d,p)=clamp(σ_base+α_d·log₂(d/5)+α_p·(p−p_ref), σ_min, σ_max)，得到 σ_base=0.3447、α_d=0.0041、α_p=15.03、p_ref=10⁻³、σ∈[0.30,0.70]，LOO 验证 MAE 0.064、尾部分位数 ~15% 相对误差。Triage 假设"延迟随 volume 单调增长对任何实用解码器成立"，因此相对性能趋势可推广到其他解码器。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

## Union-Find 解码器（UF Decoder）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Union-Find 解码器（Delfosse & Nickerson, Quantum 5:595, 2021）是 MWPM 的近似替代，几乎线性时间：① 聚类/增长阶段——每个非平凡 syndrome 初始化一个簇，奇数奇偶的簇半径增长并与相邻簇融合，直到所有簇均为偶数奇偶（并查集 disjoint-set 数据结构 + 加权合并 + 路径压缩，O(N·α(N))，α 为 Ackermann 反函数；Phys. Rev. Research 6:013154 (2024) 进一步证明规模上可线性）；② peeling 阶段——簇内生成生成树，从叶到根逐层剥除（吸收 syndrome），O(N)。精度低于 MWPM，但并行度高、易硬件化（顶点/簇可映射到分布式 PE）。本文将其定位为"低延迟但精度 suboptimal"的 baseline。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
UF(G, s):
  for v in V: if s[v]==-1: 新建簇 C(v)      # 非平凡 syndrome
  while 存在奇数奇偶簇:
      增长所有奇数簇半径; 相邻簇相遇则合并   # union-find
  for 每个簇: T=生成树(cluster);  peeling(T)  # 叶到根剥除
```
本论文用法：① 陪集集成解码的 Phase I Clustering 是 UF-equivalent 聚类（把 stabilizer 群划分为局部子空间）；② 精度 baseline 用自研 UF 软件实现（避免边界条件处理差异混淆），硬件 baseline 为 Helios（QCE 2023，d=17 889k LUT @75 MHz，per-vertex PE 空间并行、sublinear 延迟但有 per-iteration 地板）与 QUEKUF（TRETS 2025，d=8 309k LUT / 548 BRAM @238 MHz）。对比结果：UF 类 LER 落后 MWPM 2.7–5.7×（biased X 噪声下 6.2×），本文方法填补该差距（repetition code 上距 MWPM 仅 1.0–1.4×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：Qsurface（Python，Delfosse 算法实现）、AFS（Das et al., HPCA 2022：UF + 3 级流水硬件 + syndrome 压缩，平均 42 ns）。硬件：Liyanage et al. TQE 2024（FPGA 分布式 UF）、QUEKUF、Helios。使用场景：需要亚微秒实时解码的超导平台；精度敏感场景可叠加本文的陪集集成/投票增强。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

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

## 逆序消元（Reverse-Order Elimination, ROE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ROE 是 peeling（剥除）的流水友好变体：传统 UF/MWPM 后处理需要反复全局寻找叶节点并重算度数；ROE 则利用建森林阶段的遍历结果——BFS 建森林时已按根→叶访问一遍，记录发现序 σ 后，逆序弹出顶点：对奇偶 p[x]=1 的顶点收集边 (x, parent[x]) 并翻转 x 与其父的奇偶。单趟、线性时间，免去第二次叶子发现遍历与度数重算，直接把解码延迟砍掉一趟全图扫描。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Require: parent[]; 发现序 σ; 奇偶 s            # Algorithm 3
E_i ← ∅; p ← s
for t = |σ| down to 1:
    x ← σ_t; r ← parent[x]
    if r ≠ NIL and p[x] == 1:
        E_i ← E_i ∪ {(x, r)}
        p[x] ← p[x] ⊕ 1; p[r] ← p[r] ⊕ 1   # 奇偶吸收到父
L_i = DECODELOGICAL(E_i)
```
关键观察：Algorithm 2（PriorityForests）在建森林时已经完成根→叶遍历，ROE 复用该顺序逆序剥除，等价于传统生成树 peeling 但省一趟。在陪集集成解码中，K 个候选各自执行一次 ROE，故其单趟性与无全局探测特性直接放大为 K 倍收益；硬件上 EFE 实例内的遍历状态不可时分复用（会覆盖在途邻接数据），故复制 K 份并行执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现随开源仓库发布（Python）；硬件在每个 EFE 实例中以状态机执行（邻接表构建与聚类重叠进行）。使用方式：任何"先建森林/树、后剥除"的解码流程都可替换为 ROE；前提是建树阶段能顺带记录发现序（BFS/DFS 均满足）。本论文配套的 lossless graph compression 进一步缩小 σ 与 parent 的规模。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## 无损图压缩（Lossless Graph Compression）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
聚类完成后对解码图的结构保持型规约：完整图保留簇内全部顶点与边（Fig.4 左，size 21），压缩图仅保留"根-根边"与"根-边界边"（Fig.4 右，size 8），并允许非曼哈顿方向的跨格点边（任意方向连接簇根与边界）。因为 Algorithm 1 的复杂度线性于输入图规模，且簇内结构信息可由簇根表示，该规约在逻辑上无损——候选纠错的陪集归属不变——同时把 K 次集成探索的成本按图压缩比例降低。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
聚类后: 4 个簇(由 4 个根顶点生长) + 边界
完整图: 所有簇内顶点与边(含绿色边) —— size 21
压缩图: 仅保留 {根-根边} ∪ {根-边界边} —— size 8
# EFE 在压缩图上建森林+ROE：复杂度 ∝ 压缩图规模
```
消融（硬件模拟器，d=11、p=0.0015）：Graph Compression 单独贡献 1.18× 加速，与 Multi-bank Hashing（2.30×）、Hierarchical ID Mapping（1.03×）协同，全开合计 3.24×。压缩图的边表示超越轴对齐曼哈顿连接，是"保留核心结构、去除冗余边"的关键——恰好与陪集解码只需簇间/边界信息的需求对齐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
属于解码图规约技术：先 UF 式聚类，再在簇粒度上重建稀疏图。实现时以根为簇代理节点，merge 期间记录根-根/根-边界邻接即可（本文在硬件聚类流水线中与邻接表构建重叠完成）。使用前提：下游算法只依赖簇间关系（如本文的森林探索与投票），若下游需要簇内细粒度结构则不可用。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## 逻辑错误率与系统不保真度指标（LER & System Infidelity Ĉ(R)）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LER（Logical Error Rate）：一个完整 QEC 周期（d 轮 syndrome 提取）内逻辑错误未被纠正的概率，是解码精度的主指标，阈值定理下随码距指数下降。系统不保真度 Ĉ(R) 是本论文自定义的系统级指标：量化 feedback decoding 场景中逻辑 patch B 的解码延迟对逻辑 patch A 保真度的损耗——从经验式 E(n)=½(1−(1−2ε)^n) 出发，用 d 轮 LER E(d) 重参数化（ε 不可直接测量，FTQC 的基本单位是整个 QEC 周期）：有效错误率 Ê(m)=½(1−(1−2E(d))^m)、保真度 F̂(m)=(1−2E(d))^m；B 延迟 R（以提取轮计）使 A 的保真度乘以 (1−2E(d))^{R/d}，反转为不保真度 Ĉ(R)=1−(1−2E(d))^{max(1,R)/d}∈[0,1)。max(1,R) 掩码含义：解码在一轮内完成即无 backlog、不影响 LER。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
E(n) = ½(1-(1-2ε)^n)                      # 文献[26] 经验式
(1-2ε)^d = 1-2E(d)                        # 重参数化：基本量=E(d)
F̂(m+R/d) = (1-2E(d))^{R/d} · F̂(m)         # B 的延迟 R 折算进 A 的保真度
Ĉ(R) = 1-(1-2E(d))^{max(1,R)/d}           # R=L/l, l=单轮提取时长
```
使用效果（本文）：Micro-Blossom 的 LER 最低，但 d≥5 时延迟超过一轮提取时限使 Ĉ(R)>0 受罚，最终系统 infidelity 反超 UF 类；本文在 d=11 时较 Micro-Blossom 降低 74.3%、较 Helios 降低 51.7%——把"精度×延迟"折算为单一可比量，证明低延迟对 feedback decoding（非 Clifford 门条件操作）的真实价值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Pauli-frame 解码（内存实验）只用 LER + 反应时间即可；feedback 解码必须用 Ĉ(R) 类联合指标。实现仅需各解码器的 LER(E(d)) 与延迟(L) 两个实测量；物理怠机误差、Dynamic Decoupling、Pauli Twirling 的影响全部封装于 E(d)。后续工作可直接复用该公式对比不同解码器在条件逻辑操作负载下的系统影响。

补充（Triage 论文）：Triage 用"插入 idle layer 数 + 总执行层数→LER"的间接度量——同步失败时插入 idle layer，idle 期间 qubit 经历额外纠错轮直接抬高 LER，因此 LER 是总执行层数（含 idle）的单调函数：先模拟 window-based lattice surgery（d=9、p=3×10⁻³、Stim ≥10⁵ runs/点）得到逐层 LER，再按每应用总层数聚合出整体 LER（外推 d=21）；每层时间 T_layer=d×T_meas（d=21 时超导约 21μs、离子阱/中性原子 2.1-21ms），T_total=N_total_layers×T_layer 把 idle 层减少直接折算为墙钟时间节省。结果：Triage 相比标准时间并行 baseline 平均 LER 降低 52.6%；慢解码器区（τ_dec>τ_gen）仍维持低 LER。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

## BP+OSD 解码器（Belief Propagation + Ordered Statistics Decoding）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BP+OSD 是置信传播与有序统计解码的组合解码器，qLDPC 码事实标准（Panteleev & Kalachev 提出）：在 Tanner 图上运行 BP 消息传递，迭代估计与 syndrome 一致的最可能错误；当 BP 因量子简并/短环不收敛时，触发 OSD 后处理——按软信息可靠度（LLR）对错误位置排序、重排校验矩阵列、对可靠子集做矩阵求逆解线性方程确定错误。优点：对 qLDPC/表面码通用、精度接近最优；缺点：OSD 矩阵求逆昂贵，实时硬件化难（首款并行 BP+OSD FPGA/ASIC 2025 年才出现：EPJ Quantum Technology，d≤9 单 VCU129 @200 MHz 134 μs；bicycle 码 d≤12 @244 MHz 84 μs）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
1: 在 Tanner 图(校验节点=稳定子, 变量节点=qubit/错误)上迭代 BP
   消息 = 软信息(对数似然)传播，估计每边错误概率
2: if 硬判决满足全部 syndrome: return 估计错误     # BP 收敛
3: OSD: 按可靠度降序排列列 -> 最可靠列求逆(高斯消元)
   -> 解方程得错误集合; 必要时 OSD-0/CS/w 组合搜索
```
本论文用法：作为精度参照——product-sum BP + OSD-CS（order 15），同一 Tanner 图、repetition code、phenomenological noise、d∈{5,7}、p∈[0.04,0.08]：BP+OSD LER 距 MWPM 1.0–1.7×，本文陪集集成解码 1.0–1.4× 与之相当。作用是在"UF（快差）—MWPM（准慢）"谱系中给本文方法一个独立标定（本文不声称超越 MWPM/BP+OSD 精度，只求近似并保持低延迟）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件：开源 ldpc/BPOSD 实现（Roffe 的 ldpc 库）、qecsim、GPU 加速（NVIDIA CUDA-Q QEC 2026）。硬件：EPJ QT 2025 FPGA/ASIC 设计空间探索、进化式 EBP+OSD（arXiv:2512.18273，差分进化调权、更少 OSD 激活）。使用场景：qLDPC（BB 码）与表面码离线解码、其他解码器的精度上界参照；实时部署需控制 OSD 激活次数（Astra+OSD 可减少 >2000× OSD 调用）。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

## LLM 推理层的访存特征划分（memory-intensive GEMV/attention 层 vs compute-intensive GEMM/FFN 层）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM（GPT-3、LLaMA-2 等）推理按层的算术强度可二分：compute-intensive 层是全连接 FFN 的 GEMM（batch×seqlen 大时权重复用高、受 GPU 算力主导）；memory-intensive 层是 attention 的 QKV 生成与 projection 的 GEMV 形态（QK^T、SV：权重=每请求新产生的 K/V 或激活，复用低、每元素约 2 FLOP，受外部带宽主导）。CPU/GPU 处理器为中心的系统跑 memory-intensive kernel 时被 off-chip 数据搬移瓶颈卡住——这正是 PIM offload 的机会窗口。DCC 的 kernel 选择：GEMV/RED/ATTN/VA/RELU（AttAcc 额外有 softmax/accumulator 单元支持整段 attention 在 PIM 侧）；GEMV/ATTN 输入尺寸 128 是 LLM 最常见的 per-head 维度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GPT3-13B/LLaMA2-33B 一层的执行拆解（DCC + AttAcc 的划分）：
```
x = input_embeddings                       # [batch, seqlen, hidden]
Q, K, V = split(x @ W_qkv + b_qkv)         # QKV 生成：GEMV 形态 → PIM（DCC 使能后）
score = softmax(Q @ K^T / sqrt(d))         # attention：GEMV+softmax → PIM（AttAcc 原生支持）
ctx = score @ V                            # context projection：GEMV → PIM
h = x + W_o @ ctx                          # output projection：GEMV → PIM（DCC 使能后）
h = FFN(h)                                 # FFN：GEMM，主体留 GPU
```
AttAcc 默认实现只把 attention 放 PIM（QKV 生成与 projection 留 GPU，固定 tiling 下这两层放 PIM 反而落后 GPU 1.25×）；DCC 联合搜索数据分区与计算调度后，把 QKV 生成与 projection 也移上 PIM（分别 2.58×/2.91× 对 GPU），使 AttAcc_Full+DCC 端到端平均 4.52× 对 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：按"层访存强度 + 后端硬件（PIM 有无专用单元）"决定落点——有 softmax 单元的 AttAcc 可整段 attention 上 PIM，HBM-PIM 只有 16-way FP16 FPU 故跑 GEMV/RED/VA/RELU；DCC 在 LLM 中按张量尺寸动态选 draft（输入/输出 token 数、batch 变化触发在线生成）。对 batch 增大时 QKV/projection 变成 batched GEMM 形态，GPU 相对优势上升——DCC 在大 batch（MT-NLG-310B、batch≤64、8×A100）下仍较 AttAcc Base/Full 提速 1.59×/1.67×。参考划分原则（CompAir 等一致）：GeMM→compute 单元、GeMV→memory 单元。

EVA 补充视角（ISCA'26）：decode 阶段 GEMV 的低效有两层——(1) 计算侧：M=1 时 GEMM 单元窄条活跃、算术强度低（大多数 PE lane 空闲）；(2) 访存侧：权重矩阵每 token 全量重取、无复用。EVA 用向量量化（VQ）+ 码本驱动 GEMM 重构同时解两层：权重压缩为索引+码本（2-bit，memory 侧），并把解码从 GEMV 重写为"输入×码本"的 GEMM（计算量 K×N → K×2^n，约 16×，M 维扩到 V=K/d>512），使 decode 阶段成为可填满矩阵单元的 GEMM 形态。这与"GeMM→compute 单元、GeMV→memory 单元"的传统划分不同——EVA 在算法层就把 GeMV 变成 GeMM，计算形态不再固定受限于 batch=1。

- SMOOTH 视角（ISCA'26，移动 SoC）：移动 NPU（2–8MB SRAM、LPDDR5 13–34GB/s、batch=1）上 decode 期 OI 特征在单层内交替——线性投影（QKV、W0）是低 OI 的 GEMV、带宽饱和（I/O-bound），softmax/GELU 等非线性是高 OI、带宽空闲（compute-bound），导致突发性（bursty）访存与带宽浪费。三类平台上非线性运算占端到端时间 10–20%（Jetson/TinyLLaMA 20.4%、S24 17.0%、EdgeTPU 14.1%）。动机是：高 OI 运算的空闲带宽窗口可被预取利用，但静态编译器看不到运行期进度；静态 tile size 因序列长度与带宽波动失效（延迟最多恶化 2.9×）。
涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference

## Epitopological Sparse Sampling（ESS，表观拓扑稀疏采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ESS 是 DESSCam 提出的像素级注意力引导稀疏采样机制：用训练集事件帧构造 Sample Matrix（M×N，M=H×W 像素、N 帧），计算 M×M Pearson 相关矩阵（每对像素的全局相关性），逐行求和得 M×1 Feature Importance Matrix（每个像素与其余像素的相关性和，代表其对该任务的重要性），再用稀疏阈值 TH 二值化得到 binary Mask Matrix，写入像素阵列的 1-bit SCtrl SRAM 使能约 2% 的像素做事件化（50× 下采样）。其灵感来自 brain-inspired 的 epitopological learning（ICLR 2024，Zhang/Cannistraci 等，"Epitopological learning and Cannistraci-Hebb network shape intelligence"）：该方法把全局相关矩阵/连接预测用于稀疏化全连接层（ESML + CH3-L3 Cannistraci-Hebb 规则，约 1% 连接保留时在 VGG16/ResNet 上超过全连接网络）。DESSCam 把"全局相关矩阵稀疏化"的思想从网络层迁移到像素阵列，用相关性高的像素保留全局数据结构，抑制冗余/热像素。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ESS 在眼动追踪 pipeline 中位于最前端（离线掩码生成 + 传感器内在线应用）：
```
# 离线（A100 约 2 分钟，22/27 subject）：
Sample = stack(frames)                # M×N, M=H×W
rho(i,j) = cov(i,j)/(sigma_i*sigma_j) # M×M Correlation Matrix
importance(i) = sum_j rho(i,j)        # M×1 Feature Importance
mask(i) = 1 if importance(i) > TH else 0
# 在线（传感器内）：
for pixel i: if mask(i): enable eventification   # 仅 2% 像素使能
events -> 16×16 patch -> count>2 激活 -> ViT 推理
```
TH 可调控制压缩率（50× 压缩率 = 仅 2% 像素采样）；掩码离线生成、跨 subject 复用、无需用户特定重校准。效果：50× 压缩率下 AE 0.5°，同压缩率下 PAC-only（无 ESS）为 4.7°；迁移到 ini-30 瞳孔追踪（50× 压缩）pixel error 2.76±0.15，优于无稀疏的 Retina（3.24±0.79）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对比三种稀疏采样实现：① BlissCam 随机稀疏采样——每帧刷新 10-bit in-pixel SRAM 随机数，只控制总体稀疏度、不指定有效像素位置；② MESA 像素级动态注意力——实时逐像素自适应遗忘因子，硬件代价大；③ ESS——预计算全局注意力系数 + 1-bit SRAM 掩码，硬件友好且保留全局相关性。ESS 还可以替代事件相机的去噪预处理（背景活动噪声过滤、热像素抑制）与 ROI 分割（后两者每 inference 需数千万 MAC）。使用前提：掩码由特定硬件设置采集的训练数据初始化；论文指出未在无约束环境验证。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

## 传感器内 Token Pruning（PAC，Patch Activation，patch 激活机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PAC 是 DESSCam 的传感器内 token 剪枝机制：把像素阵列按 N×N（16×16）分组为 patch，每个 patch 配一个 PAC 电路用列级加法器 + 加法树累计事件数，只有当累计事件数超过配置阈值（DESSCam 取 2）时，patch 才被激活并经握手读出——只有高事件密度的 patch 成为送入 ViT 的 token。其本质是 token pruning 的前移：把"对稀疏 token 序列做剪枝"从 host NPU 前移到像素阵列内，避免空间孤立冗余事件产生无效 token 和无效计算（最多减少 61% host 端算法 MAC），同时降低接口传输数据量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每个 event frame：
for patch p in 22×17 grid:
    cnt(p) = adder_tree(event_flags of p)   # 16×16 事件计数
    if cnt(p) > TH:                          # TH=2
        p.active = True; handshake(p)        # ReqX/ReqY
        emit AER packet(p)                   # addrX/addrY + 512bit 事件 + 时间戳
tokens = [p for p.active]                    # 稀疏 token 序列
if len(tokens) >= 12: run ViT inference      # 12 patch 触发一次 gaze 估计
```
PAC 激活频率实测 162.76–64,170.78 Hz，12 patch 触发一次推理 → 等效帧率 13.56–5,347.56 Hz（延迟 0.19–73.75 ms 自适应）。与 Event Transformer（ICIP 2022）等 off-sensor token 剪枝对比：PAC 把 token 稀疏性计算放在像素阵列内，off-sensor 算法只需处理已剪枝的 patch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：每 16×16 像素共享一个 PAC 握手控制单元与加法树（patch 级握手替代标准 DVS per-pixel 握手，省复杂仲裁逻辑、输出延迟从 120 ns 降至数 ns）；算法等效：对 patch 序列施加稀疏掩码。与 BlissCam 片内 ROI 预测 NPU 对比：PAC 用简单计数电路实现 token 稀疏性，不需要片内 NPU，数字功耗更低。使用场景：事件相机输出天然稀疏、空间孤立噪声事件多的任务（眼动追踪），阈值可按噪声水平配置。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

## Robust ViT（CNN-Transformer 混合眼动追踪模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Robust ViT 是 DESSCam 为高稀疏事件数据设计的轻量 CNN-Transformer 混合模型：conv stem（两层 depthwise-separable 卷积，输出 128 维，引入局部归纳偏置）+ conv enhancement（两层 3×3 卷积替代标准 ViT 的位置嵌入，在 token 序列形成前做跨 patch 交互，增强局部空间信息）+ 3 个 transformer encoder（每个含 8 head、128 维多头自注意力）+ 平均池化 + 检测头（两层全连接 + sigmoid 输出 gaze 坐标）。设计动机：标准 ViT 非重叠 tokenization 会丢失邻域信息、形成孤立 patch 表示；在高稀疏输入（50× 下采样）下，前置 CNN 层增强局部纹理并恢复被稀疏采样破坏的上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
x = conv_stem(event_frame)        # 两层 depthwise-separable conv -> 128 维
x = conv_enhancement(x)           # 两层 3×3 conv 替代位置嵌入
tokens = flatten(x)[ESS_mask]     # 应用同一 ESS 掩码生成稀疏 token
for _ in range(3):                # 3 个 transformer block
    tokens = MHA(tokens)          # 8 head、128 维
z = avg_pool(tokens)
(x_pred, y_pred) = sigmoid(fc(fc(z)))
AE = arccos(v_pred·v_gt / (|v_pred||v_gt|)),  v=(x,y,L0)
```
计算量分布：早期卷积层在全分辨率操作、承担大部分 MAC；transformer 只处理 ESS 掩码后的稀疏 token，计算量大幅下降——这一设计同时保证精度与部署友好（卷积层可上 NPU，transformer 计算量小）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 MobileViT、CvT、Conformer 同属 CNN-Transformer 混合族（卷积提供局部归纳偏置、transformer 提供全局相关性）；训练 batch 64、500 epochs（EVBEYE）。部署：LSQ 量化到 INT8、ONNX 导出、STM32Cube.AI 异构部署（卷积/线性在 Neural-ART NPU、LayerNorm/Softmax 等在 Cortex-M55）；由于 Neural-ART NPU 对 transformer 块无原生加速，计算量集中于卷积层是关键部署优势。效果：50× 压缩率 AE 0.5°，压缩率 1×–50× 全程 AE < 2°。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

## LSQ（Learned Step Size Quantization，学习步长量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LSQ（Esser 等，ICLR 2020，IBM）是一种把量化步长（step size）作为可学习参数训练的量化方法：每个量化层维护一个浮点步长 s，量化操作 v̄ = clip(round(v/s), -QN, QP)（QN/QP 为量化整数范围），通过直通估计器（STE）让梯度穿过 round/clip 回传到 s，训练中联合学习权重与步长，从而找到比固定范围（如 min/max 标定）更优的量化间隔。DESSCam 用它把 Robust ViT 量化到 INT8 后部署到 STM32N6。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 前向：v_bar = clip(round(v/s), -QN, QP)      # s 可学习
# 反向（STE）：dL/dv_bar = dL/dv_bar；v_bar 对 v 的梯度在量化区间内视为 1
# 步长梯度：dL/ds = sum(-v/s + round(v/s))     # 区间内
#                    -QN 或 QP                  # 越界部分
# 更新：s <- s - lr * dL/ds（与权重同训练）
```
相比 QAT 固定范围、LSQ 的范围端点随 s 连续可调，INT8 下精度损失更小；DESSCam 的量化 pipeline：LSQ INT8 → ONNX（QDQ 格式）→ STM32Cube.AI 部署。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 PyTorch/TensorFlow 量化训练流程（LSQ 官方提供 STE 自定义算子），也可作为 QAT 一部分插入任意网络；使用场景：边缘 NPU 仅支持 INT8 的部署（Neural-ART NPU、NPU 类硬件）。注意：量化后的卷积/线性层可映射到 NPU，而 LayerNorm/Softmax 等非线性算子通常需回落 CPU 或保持高精度，DESSCam 即采用该异构切分。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

## Gaze-Tracked Foveated Rendering（TFR，注视追踪注视点渲染）与 Motion-to-Photon Latency（MPL）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TFR 是 AR/VR 中利用眼动追踪的 gaze 方向数据驱动 GPU 只对注视点（foveal）区域做高分辨率渲染、周边区域低分辨率渲染的渲染负载削减技术（Tobii 在 Pico 头显实测渲染开销最多降 72%、平均降 60%）。MPL 是"从虚拟动作到其视觉反馈被感知"的端到端延迟 = 眼动追踪延迟 + TFR 与显示延迟；研究证明 MPL 需 <5 ms 才不引起视觉不适/晕动，而商用 HMD（如 Vive Pro Eye）MPL 高达 79 ms。眼动追踪延迟在商用系统占 MPL 的 63.3%、在 SOTA 研究中占 77.7%，是 TFR 能否落地的主要瓶颈；提高帧率降延迟会带来功耗激增（1 kHz 追踪频率需 96 W），形成延迟-功耗权衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
gaze 采样 -> gaze 估计 (x,y) -> 划分 fovea/周边区域 -> 分级分辨率渲染 -> 显示
MPL = t(眼动追踪) + t(TFR 渲染) + t(显示)   # 要求 < 5 ms
```
眼动追踪是 TFR 管线的前置级：其延迟决定注视点渲染跟随眼球运动的实时性（延迟过大则渲染区域落后于实际注视点、产生视觉伪影），其功耗决定 HMD 整机功耗预算（商用眼动追踪 >2W，接近 VR 系统功耗预算一半）。DESSCam 以亚 1 ms 眼动追踪 + 数 mW 功耗直接服务 TFR 部署。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用实现：HTC Vive Pro Eye、Tobii XR（帧式传感器，延迟 >50 ms）；研究实现：BlissCam（in-sensor 稀疏采样，8.2× 省电但 9.4 ms 延迟）、TinyTracker（IMX500 近传感器计算）、DESSCam（DVS 事件驱动 + ESS 稀疏采样，15.2× 延迟降低）。使用场景：HMD 渲染负载削减、超轻量智能眼镜（如 Meta Ray-Ban 154 mAh 电池、49.6 mW 全天功率预算）中把眼动追踪压到数 mW 级。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

## 事件相机眼动追踪数据集（EVBEYE / ini-30）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVBEYE（Angelopoulos 等，TVCG 2021）是首个近眼事件相机 gaze 数据集：27 个 subject、DAVIS346 传感器（346×260）同步采集左右眼事件流（events.aerdat）与约 25 fps 灰度帧，两个实验范式对应两类眼动（random saccades 与 smooth pursuits），刺激显示在 40 英寸 1920×1080 屏、40 cm 阅读距离；开源于 https://github.com/aangelopoulos/event_based_gaze_tracking。ini-30（Bonazzi 等，Retina 工作，CVPRW 2024）是首个在传感器上标注 pupil 中心的事件相机眼动数据集：两台 DVXplorer（640×480）镜架式采集，受试者自由观看收集自然眼动。两者分别支撑 gaze 估计与 pupil 追踪任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# EVBEYE 用法（DESSCam）：
清理标签（剔除 stop/pause、每次 saccade 前 15 个标签；左右眼不区分）
随机 22/27 subject -> 生成 ESS 掩码（A100 约 2 分钟）
全部 subject -> 训练 Robust ViT（batch 64、500 epochs）
全部 subject -> 测试，AE 逐 inference 平均
# ini-30 用法（泛化验证）：
5-fold 交叉验证 ESS + ViT，50× 压缩率，pixel error 2.76±0.15 vs Retina 3.24±0.79
```
注意点：掩码生成 subject 与训练测试数据分离（unseen 验证泛化性）；AE 定义为预测/真值 3D gaze 向量夹角 arccos(v_pred·v_gt/(|v_pred||v_gt|))，v=(x,y,L0)、L0 为受试者到屏幕距离。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
EVBEYE 仓库提供 conda 环境（ebv-eye.yml）、setup.sh 下载脚本与 visualize.py 可视化；被 Papers with Code 收录、广泛用作事件相机 gaze 基准（gaze 精度 0.45°–1.75°@45°–98° FOV）。ini-30 随 Retina 工作发布（Speck 芯片上 5 mW 运行）。使用场景：事件相机眼动追踪算法评测、稀疏采样/去噪/剪枝算法消融（DESSCam 以二者分别作为主评估与跨任务泛化评估）。

涉及论文标题：
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

## Adaptive Expert Selection（AES，自适应专家选择：冲突感知的动态专家替换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Expert Selection（AES）是 DIAMoND 提出的动态在线专家选择算法：MoE 路由（router = linear 层 + top-k）输出专家分数后，标准流程直接取 top-k 专家；AES 在此基础上做冲突感知替换——在 in-NAND 阵列上，被选专家可能发生两类冲突（共享 OU 输出端口；缺少可同时区分它们的 mask 模式），此时用"分数略低但无冲突"的专家替代，以换取 FFN 层单 read cycle 内并行完成全部 k 个专家（而非串行多 cycle）。调节旋钮 T（阈值）：仅当无冲突替代专家与原冲突专家的路由分数差 < T 时替换，否则保留原专家（接受额外 read cycle 保精度）。配套指标：pairwise difference = 专家对中至少一位与原始 top-k 不同的比例（衡量冲突解决程度）；expert similarity = Σ_{i∈E_T∩E_k} w_i / Σ_{i∈E_T} w_i（自适应所选专家集与原始 top-k 的重叠加权占比；GRIN-MoE 因 gate 把 top-k 外权重置零而不适用）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 单层 FFN、每 token 的专家选择（k 个专家、阈值 T）
s = router(h)                       # (N,) 专家分数，h 为 attention 输出
S = { argmax(s) }                   # 最高分专家必选
for e in sort_desc(s):              # 其余按分数降序尝试
    if e conflicts with S:          # 共享 OU 输出端口或缺兼容 mask
        e' = highest_score_free({e'' ∉ S ∪ {e} : e'' no conflict with S})
        if s_e - s_e' < T: S = S ∪ {e'}   # 分数差小 → 替换
        else: keep e                 # 保留冲突专家，FFN 多 1 个 read cycle
    elif |S| < k: S = S ∪ {e}
```
张量层面：每个专家 FFN = Up/Gate/Down 三投影（Mixtral：隐维 4096、专家中间维 14336），每个投影按 OU 切分为多个子矩阵做 in-NAND VMM；AES 保证 k 个专家的三投影在同一 read cycle 并行执行 → FFN 层恰好 3 cycles。例子（Fig.11c，8 选 4）：最高分 E6 先选；E4、E7 无冲突直接选；第四个候选 E5 与 E4/E6/E7 冲突 → 算法在剩余专家中找次高分无冲突的 E1 替代。硬件执行通路：Priority Queue（分数有序专家队列）→ Conflict FIFO（被推迟专家）→ Mask Pattern RAM（专家 ID → 兼容 mask 位向量，如 4'b1001）→ Pattern State Handler（4 寄存器跟踪各 Expert Group 可用 mask）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在线执行（每 token 每层选一次），由 DIAMoND 的 Dynamic Mask Selector ASIC 电路完成（面积 0.006mm²、0.76mW）；软件侧等价实现即上列伪代码。使用方式：任何"激活子集物理冲突"的稀疏/存内推理系统都可用（把硬件约束折进选择算法、以阈值 T 调节精度-并行权衡）。实测（DIAMoND）：T 敏感性——expert similarity > 0.9 时端到端精度（ARC-Challenge/PIQA/HellaSwag/WinoGrande）仅微小波动，pairwise difference 随 T 先快升后饱和（专家分数差有界）；AES 使解码加速至多 1.52×（与 mask 设计合计 1.95×），冲突率从 Mask-only 的 10.2%~93.5% 降超一个数量级（DeepSeek/Qwen 等专家数多的模型效果最显著）；DIAMoND-L+Mixtral（单专家粒度）与 DIAMoND-H+DeepSeek/Qwen（全专家可容纳）天然无冲突，无需 AES。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture

## QC-LDPC 前向纠错（FEC，Quasi-Cyclic Low-Density Parity-Check）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LDPC 码是一种线性分组纠错码，由稀疏奇偶校验矩阵 H ∈ {0,1}^{m×n} 定义，合法码字满足 H·cᵀ ≡ 0 (mod 2)；码率 R = k/n = 1 − m/n。QC-LDPC 是其硬件友好的子类：H 由基矩阵 B（元素为 -1 或移位值 s）经 Z×Z 循环置换子块扩展而来（扩展因子 Z 控制并行度），编码只需移位寄存器/旋转即可实现。FEC（前向纠错）与 CRC 检测+重传的区别：FEC 在接收端主动纠正比特错误，避免重传。DICE 在 flit 级（128-bit）做 QC-LDPC 编码：R≈0.88（+16 奇偶位，Z=8），每 flit 独立编解码。该码广泛用于 SSD（替换 BCH）、5G NR 与高速互联（Web 证据：QC-LDPC 用于 NAND Flash 控制器、5G NR 的 LDPC 实现文献）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DICE 中 flit 级 QC-LDPC 编码的伪代码（Z=16 的示例，128-bit flit 分成 8 个 16-bit 块 {u0..u7}）：
```
# 编码：计算 16-bit 奇偶块 p
p = P(0)@u0 XOR P(3)@u1 XOR P(7)@u2 XOR P(11)@u3 XOR P(2)@u4 XOR P(9)@u5 XOR P(14)@u6 XOR P(5)@u7
codeword c = [u0||u1||...||u7||p]
# 发送：c 经 PAM4 调制 → AWGN 信道 → LLR 解调 → 解码器
```
Annotations：P(s) 是 16×16 循环右移 s 位的单位阵；由于 H 系数为常数，乘法退化为 XOR 树——综合结果 7 个 16-bit XOR 门、175 cells，满足 2.0 GHz（Yosys+OpenSTA，TSMC 40nm）；而 packet 级（768-bit）编码需 2320 cells 且不满足时序，这正是选 flit 级粒度的算法-硬件联合依据。码率敏感性（Fig.5）：2B 奇偶/flit 是甜点（R≈0.88）——更高 R 奇偶不足、post-FEC FER 上升，更低 R 带宽浪费且纠错收益递减；SNR 降至 22.5 dB 时 2B 不够，需 4B 奇偶或更强解码预算或退回重传。解码结果统计：DICE 的 FEC 平均纠正 97.8% 的错误，仅 2.2% 需重传。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：QC-LDPC 编码器为 shift-register + XOR 网络（块循环结构避免非结构 LDPC 的复杂布线）；解码器为迭代消息传递（min-sum 或分层 min-sum）。使用方式：DICE 以 1-cycle 编码延迟把编码器嵌入 PHY 路由器流水线（不在关键路径）；奇偶字节可与 UCIe 68B flit 格式的未用字节兼容注入。注意：LDPC 无 BCH/Hamming 的有界距离保证，码率/迭代预算需按 SNR 工作点经敏感性实验标定。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation

## LLR 软判决解调与 layered min-sum 迭代解码

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLR（对数似然比）是软判决接收的核心量：对收到的每个比特 b_k，L_ch(b_k)=log[P(y|b_k=0)/P(y|b_k=1)]，符号给极性（>0 判 0、<0 判 1）、幅值给置信度（如 |L|=3.0 高置信、0.2 低置信）。PAM4 Gray 映射下，接收符号 y 相对符号子集 X_k^(0)/X_k^(1) 计算 LLR，常用 min 近似 L_ch(b_k)≈(1/2σ²)(min_{x∈X_k^(1)}(y−x)² − min_{x∈X_k^(0)}(y−x)²)。layered min-sum 是 LDPC 解码的低延迟调度：按 H 的行（层/check node）逐层更新——check-node 更新用"exclude-self"规则（符号取邻居符号积、幅值取邻居 LLR 绝对值最小），variable-node 做增量累加 L(v_i)←L(v_i)+m_{cn→v}，后续层立刻复用本层更新的 LLR（比 flooding 收敛更快）；每轮后做硬判决 ĉ 与 syndrome 检查 H·ĉᵀ≡0，为 0 则终止，否则继续迭代直至预算 N 或触发重传。解码是 NP-hard，迭代式解码是该困难性的工程出路（Web 证据：layered min-sum 广泛用于 5G NR LDPC、NAND flash LDPC，早期终止标准基于 syndrome/LLR 可靠性）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DICE 接收端解码流水线伪代码：
```
# 初始化
for v_j: L(v_j) = L_ch(v_j)            # 信道 LLR 输入
# 迭代（每轮 sweep 所有层）
for t in 1..N:
    for layer i in 0..m-1:
        for edge (cn_i, v_j):
            m_cn_to_v = prod(sign(L_other)) * min(|L_other|)   # check-node，exclude-self
            L(v_j) += m_cn_to_v                                # variable-node，增量累加
    hard_decision: ĉ_j = 0 if L(v_j)>=0 else 1
    if H @ ĉ == 0 (mod 2): return SUCCESS                    # syndrome 检查
return NACK_RETRANSMIT                                          # 预算 N 用完仍未收敛
```
Annotations：DICE 标定 N=4（35 dB 下所有码率 ≤2 迭代收敛，Fig.10）；每迭代 1 cycle（含全层 LLR 更新）、syndrome 1 cycle，总延迟 = (N+1)·L_syn + N·L_iter = 2N+1 cycles（≤9 cycles）。示例（论文）：y=[-45,-171,+137,+158]mV → L_ch=[+22.8,-27.8,+122.5,+35.9,-88.1,+18.7,-109.4,+29.4] → 一轮三层 sweep 后 L=[69.3,-80.0,170.6,58.7,-117.5,69.3,-132.2,80.0] → 硬判决 [0,1,0,0,1,0,1,0] 通过 syndrome。迭代预算与 SNR 的耦合：15 dB 噪声下 2B 奇偶只能纠少量错（2 迭代内可纠的都纠了）、更多奇偶需更多迭代——迭代预算-码率-噪声三者构成解码成本环。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上 check-node 单元做符号乘积 + 最小值树，LLR 缓存 + 分层调度器；DICE 在 Verilog 实现后经 Yosys/OpenSTA 合成标定 1 cycle/iteration 时序。使用方式：作为接收端 flit 恢复的前置（S2P 之后），成功则 ACK 释放发送缓冲、失败则 NACK 仅重传该 flit；gem5 开销大头来自该迭代解码（占总开销主导，DICE 平均开销 9.2%），论文提出 memoization 缓存符号→LLR 模式作为未来优化。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation

## Ising 模型与实值扩展哈密顿量（H_Ising → H_DS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ising 模型源自统计物理（Ising 1924，研究铁磁性），描述全局相互作用的二值自旋系统，能量函数（哈密顿量）为 $$H_{Ising} = -\sum_{i \neq j}^{N} J_{ij}\sigma_i\sigma_j - \sum_{i}^{N} h_i\sigma_i$$：J_ij 是自旋 i、j 间耦合强度，h_i 是作用在自旋 i 上的外场（偏置）。物理实现的 Ising 机求该哈密顿量的最低能态，对应映射到模型上的组合优化问题之解（实现路线：量子退火、光学系统、耦合振荡器、CMOS）。DS-ISA 论文的核心算法扩展是把线性自相互作用项换成二次项，得到实值节点哈密顿量 $$H_{DS} = -\sum_{i \neq j} J_{ij}\sigma_i\sigma_j + \frac{1}{2}\sum_i h_i\sigma_i^2$$：二次项 h_iσ_i²/2 作为能量调节器（self-coupling），阻止能量发散、让连续实值节点稳定在有效平衡点而非饱和到边界，从而把二值优化机器推广为可承载实值 ML/科学计算的动力系统处理器。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DSU 上求解 pipeline（以实值 H_DS 的一次推断为例，对应 DS-ISA 的 A1 模式）：
```
# 映射：变量 → 节点，交互 → 耦合
for (i,j) in model_edges:            # 模型权重编码为耦合电导
    J_ij = weight[i][j]              # C_LOAD 写入耦合组
for i in input_nodes:                # 输入特征编码为节点电压
    sigma_i = feature[i]; lock(i)    # N_LOAD + N_LOCK 锁定边界
h_i = self_coupling[i]               # 自耦合防止发散
# 演化：梯度下降动力学，能量单调下降 dH/dt ≤ 0
while not converged(time_limit):     # N_EVOLVE 按指定时长触发
    for i in free_nodes:
        d(sigma_i)/dt ∝ sum_j((J_ij+J_ji)*sigma_j) - h_i*sigma_i
    # 硬件上：耦合电导电流 I_in^i = Σ J_ij σ_j 对节点电容充放电
# 输出：平衡点 sigma* = 自然给出的解（N_STORE 读回）
```
二值 Ising 版（优化，B1 模式）为同一 pipeline 取 σ∈{−1,+1}：H_Ising 最低能态即 Max-Cut/SAT 等组合优化解；实值扩展后能量景观变成连续二次型，梯度流保证 dH/dt ≤ 0 收敛，可用于 GNN 层前向（DS-GL）、LLM 层映射（DS-LLM）、微分方程对齐求解（DS-TIDE）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BRIM 约定下节点状态 = 电容电压、耦合参数 = 可编程电导、h_i 为对角自耦合（图 2 对角线）；实值扩展工作有 DS-GL（ISCA'24，图学习）、DS-TPU（ISCA'25，Chebyshev 多项式非线性节点交互 + 片上训练）、DS-TIDE（MICRO'25，时不变 PDE）、DS-LLM（ICLR'24，LLM 训练/推理）、InstaTrain（ICLR'25）。使用方式：任何可写成"变量交互 + 边界条件"的能量最小化/梯度动力学问题都可映射（优化、图学习、DE、EBM/Hopfield 类双向网络）；训练侧把节点锁真值、让耦合电导在电流误差反馈下演化（见 Electric-Current Loss 条目）。注意 J_ij+J_ji 对称项出现在双向演化中；节点被钳制时反向项 J_ji 可省略（A1 单向模式）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units

## Electric-Current Loss（EC-Loss，电流误差损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EC-Loss 是 DSU 片上训练（耦合演化模式）的损失机制：训练时输出节点被钳制（locked）到真值数据，输出节点内部电流 I_R^i = h_iσ_i 随之保持恒定，而输入节点经耦合流入的电流 I_in^i = Σ J_ijσ_j 由数据与当前耦合决定，两者之差 I_loss^i = I_in^i − I_R^i 直接就是该节点的误差/失配电流。反馈回路用 I_loss 调整可编程电导（耦合参数），使 |I_loss| 最小化，等效于最小化对应训练损失——把损失函数的梯度下降转化为模拟电流误差驱动的电导自适应，不需要数字反向传播。该机制由 [36]（DS-TPU）引入，并被扩展到多层网络（DE 对齐 [15] DS-TIDE、LLM 训练 [29] DS-LLM）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
片上训练 pipeline（对应 DS-ISA 的 C1 耦合演化模式 + Evolve-Load 循环）：
```
# 映射：输入变量→输入节点，输出变量→输出节点，可训练参数→两集合间耦合
C_LOAD J_ij                          # 初始化耦合电导
for iteration in 1..T:               # DS-ISA 评估用 T=100，每轮演化 10ns
    N_LOAD 钳制输入节点 = batch 输入
    N_LOCK 钳制输出节点 = batch 真值        # 节点全锁
    C_EVOLVE [GM=耦合子集, Time=10ns]      # 标签-触发：仅耦合演化
        for (i,j) in trainable_couplings:
            I_in^i = sum_j J_ij * sigma_j      # 输入侧电流
            I_R^i  = h_i * sigma_i             # 真值钳制下的恒定内部电流
            I_loss^i = I_in^i - I_R^i          # 逐节点误差电流
            J_ij -= eta * f(I_loss^i)          # 反馈回路调电导，min |I_loss|
C_STORE J_ij                          # 保存训练后的权重
```
与推断的对称关系：推断 = 锁耦合（权重恒定）演化节点；训练 = 锁节点（数据恒定）演化耦合。微调（C2 部分耦合演化）只需 CLM 掩码只解锁待训练耦合子集，其余锁定——同一机制直接支持 fine-tuning。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：模拟域反馈回路（I_loss 电流直接驱动电导编程，无数字梯度计算）；在 DS-ISA 下由 C_LOCK（CLM 掩码选择演化耦合子集）+ C_EVOLVE（GM + Time 触发）实现，节点侧由 N_LOCK 全部钳制。使用方式：DSU 原生训练/终身学习（lifelong learning：耦合常驻、持续观察新数据演化，绕过每次耦合重载的 O(N²) 开销）；评估中每迭代 10ns、训练 100 轮，C_EVOLVE 是训练 workload 的主要成分（Fig.12）。局限性：论文以推理/训练/优化/DE 四类控制负载为评估对象，EC-Loss 的收敛精度对照数字训练未在本文评估（沿用 [14][36] 的物理时间尺度假设）。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units

## DiT（Diffusion Transformer，扩散 Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DiT 是用 Transformer 取代 U-Net 卷积骨干的扩散模型（Peebles & Xie, ICCV 2023, arXiv:2212.09748）。与传统 U-Net 扩散模型相比，DiT 把潜空间表示切成 patch 序列当作 token，用多个 attention + FFN block 堆叠建模，并通过 adaLN（adaptive LayerNorm）或 in-context 方式把 timestep 条件注入每一层。本论文语境下 DiT 是 VLA 系统中的动作规划器：给定视觉/语言/动作多模态 token 与噪声动作，迭代去噪输出 7-DoF 动作。与图像生成 DiT 的关键差异（论文 TABLE II）：动作规划 DiT 的计算层级为 trajectory-iteration-model 三级（图像 DiT 只有 iteration-model 两级）、模态多出 action/state 等、输入 token 长度仅 10^1–10^2（图像 DiT 为 10^2–10^3），这些差异使图像 DiT 的优化方法（如 Δ-DiT、BlockDance 式背景/轮廓跳过）无法直接迁移。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
动作规划 DiT 的推理 pipeline：一个任务轨迹 = 数百次动作推理（LIBERO-Long 平均 376 次）；每次动作推理 = 10–50 个去噪步（Dita 约 50 步、π0.5 约 10 步、GR00T N1.5 约 4 步）；每个去噪步 = 多个 attention + FFN block。伪代码（单动作生成）：
```
x_T ~ N(0, I)                          # 随机噪声动作
for t in [T, T-1, ..., 1]:
    h = concat(vision_tokens, language_tokens, action_tokens, x_t)
    for block in blocks:
        q,k,v = h @ Wq, h @ Wk, h @ Wv        # QKV 投影
        h = softmax(q @ k^T / sqrt(d)) @ v    # 自注意力
        h = FFN(h)                            # 两个 Linear + GELU
    eps = noise_prediction_head(h)
    x_{t-1} = (x_t - (1-alpha_t)/sqrt(1-bar_alpha_t) * eps) / sqrt(alpha_t) + sigma_t * z
action = x_0                              # 7 DoF: 平移3 + 旋转3 + 夹爪1
```
该 pipeline 的三个冗余来源（本论文核心观测）：轨迹级相邻动作高一致（55.2% 旋转变化 <2°、97.2% 平移 <1cm）、迭代级相邻步 attention/FFN 特征 >98% 相似且每步重复加载权重（60.1% 重复外部访存）、模型级 91.7% 多模态输入每步不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：本论文在 PyTorch 中基于 Dita 模型（arXiv:2503.19757）的 DiT action planner 实现，部署基线为 NVIDIA A40 GPU（INT8/PTQ4DiT），约 300M 参数 + 50 步的配置在 A40 上仅 2.6Hz。使用：作为 VLA 的动作生成端（vision-language 语义推理端可卸载云端、动作端要求 50–200Hz 实时频率），评估于 LIBERO-Long（20Hz 控制频率、最多 520 环境步）、CALVIN、SimplerEnv。Web 补充：DiT 原文 arXiv:2212.09748。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## VLA（Vision-Language-Action）模型与 DiT 动作规划器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLA 是把视觉、语言、动作三模态统一建模的具身智能模型（代表：RT-2、OpenVLA、Physical Intelligence π0.5、NVIDIA GR00T N1.5、Figure Helix）。主流 VLA 系统由两部分组成：vision-language model 做语义推理（对时延不敏感、可卸载云端），DiT 做动作规划（要求高动作频率，服务机器人 ≥50Hz、工业 ≥200Hz）。DiT 动作规划器的算法源头是 Diffusion Policy（Chi et al., RSS 2023 / IJRR 2025, arXiv:2303.04137）：把 visuomotor policy 表示为条件去噪扩散过程，动作序列经 receding-horizon 执行。本论文的动作输出为 7 DoF：平移向量 (ΔX, ΔY, ΔZ)（笛卡尔偏移）、旋转向量 (ΔΦ, ΔΘ, ΔΨ)（各轴朝向偏移）、夹爪状态 g（开/合）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VLA 闭环控制 pipeline（本论文 Fig.1）：观测（视觉帧 + 语言指令 + 当前状态）→ VLM 语义推理 → 多模态 token 输入 DiT 动作规划器 → 迭代去噪输出下一动作 → 机器人执行 → 环境新观测 → 循环。动作频率 = 1 / 单动作推理时延；任务执行时间 ≈ 动作总数 × 单动作时延（本论文评估中 actuation 时间 <0.3% 被忽略）。LIBERO 默认控制频率 20Hz、每个任务最多 520 环境步，因此 GPU 上动作频率只有几 Hz 时单个任务需数分钟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：常用开源 VLA（OpenVLA 7B、π0.5、GR00T N1.5）均在 GPU 上以自回归 VLM + 扩散动作头运行。本论文对 Dita（DiT 约 100M）、π0.5（DiT 约 300M、10 去噪步）、GR00T N1.5（约 500M、4 去噪步）三个模型评估 DiTPA 框架的通用性。使用场景：机械臂操作（抓取、开关、桌面整理）、导航 VLA（如 NaVILA）。Web 补充：Diffusion Policy arXiv:2303.04137；LIBERO benchmark（Liu et al., NeurIPS 2023）为 130 任务、5 套件、20Hz 控制频率。

- M100 补充视角（ISCA'26，车规 VLA 推理）：M100 SoC/NPU 面向自动驾驶（AD）、LLM 与智能人机交互三大域，VLA 是端到端 AD 的前沿（视觉感知、环境理解、动作规划），是 M100 设计的重要驱动之一。MindVLA 为理想自研下一代 AD 模型，集成 LLM 组件 + MoE（8 专家）transformer；UniAD 作为端到端 AD 基准（感知+预测+规划），其大量 query token（如 TrackFormer 900 query）提供充足并行机会，契合数据流架构。论文未涉及 DiT 动作规划器或机器人 VLA 的部署细节。
涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing

## 朝向条件动作预测（Orientation-conditioned Action Prediction，动作级冗余利用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
轨迹级动作冗余利用策略：机器人相邻动作具有"局部集中"特性——10 任务 × 50 初始化环境下平均朝向变化仅 2°、距离变化仅 0.5cm（55.2% 旋转变化 <2°、97.2% 平移 <1cm、最大距离变化 1.1cm）。据此，当相邻动作绝对朝向差小于阈值时直接复用当前动作、跳过下一次完整 DiT 推理。为防止"轨迹段边界误判"（相邻动作朝向差小但后续大偏差）导致误差累积，用 Skip_flag 交替强制完整推理与预测。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（论文 Algorithm 1）：
```
输入 Act_cur，输出 Act_nxt
if Skip_flag == False:
    Act_nxt = FullDiTInference()
else:
    Rad_rel = rotation_extract(Act_cur)      # 步骤1: 取旋转向量得相对弧度
    Rad_acc += Rad_rel                       #        累积为绝对弧度
    Deg_abs = 180/pi * Rad_acc
    Deg_sim = dot(Deg_abs_i, Deg_abs_{i-1}) / (|Deg_abs_i| * |Deg_abs_{i-1}|)  # 步骤2: 朝向相似度
    Deg_diff = 180/pi * arccos(Deg_sim)      # 步骤3: 朝向差
    if Deg_diff <= th:                       # 步骤4: 阈值比较
        Act_nxt = Act_cur                    # 复用当前动作
    else:
        Act_nxt = FullDiTInference()
Skip_flag = !Skip_flag                       # 交替预测与全推理
return Act_nxt
```
预测误差边界分析：轨迹初始化与物体交互阶段才出现突变朝向，且突变可分解为多关节协同运动；跳过复用不改变整体轨迹（论文 TABLE V：轨迹长度仅 +0.62%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件端为 Algorithm 1；硬件端化简为无除法/三角函数的单次比较（Eq.1–3，见硬件架构层"动作预测器"条目），4 个周期完成、0.05% 功耗/0.23% 面积开销。使用：朝向阈值由自动化 Pareto 搜索确定（LIBERO-Long 场景取 2°），消除 42.28% 的动作推理（2° 阈值）；阈值 >3° 时成功率骤降（宽松约束破坏抓取等精细操作的空间定位），≤2° 时速度与成功率兼得；阈值收紧到 0° 即退化为全推理模式。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## 交替去噪特征复用（Alternating Denoising with Feature Reuse，去噪级冗余利用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
迭代级去噪冗余利用策略：反向扩散相邻步的 attention/FFN 输出特征相似度 >98%（最低仍 >70%，且随去噪进程逐步下降），同时每步重复加载相同权重造成 60.1% 重复外部访存。该策略以粗粒度"步"为单位跳过与上一步近乎相同的计算：被跳过步省略完整 attention/FFN 计算与权重重载，仅保留低代价残差噪声更新。与 Cambricon-D/Ditto 的差分计算、EXION 的细粒度稀疏不同，它在 attention 和 FFN 两个块同时消除计算与访存，且规避了细粒度控制开销与对 GELU 等非线性算子的不兼容。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
跳过步选择（离线）：计算相邻步 attention/FFN 特征相似度 → 阈值初筛候选跳过步 → 评估跳过每个候选步引入的 MSE loss（早期步对 loss 不敏感）→ 确定最终跳过集。执行流伪代码：
```
full denoise at step t                       # 先做完整计算
for t' in schedule:
    if t' in skip_set:
        feat_attn[t'], feat_ffn[t'] = cached(feat_attn[t'-1], feat_ffn[t'-1])  # 特征复用
        x_{t'-1} = residual_noise_update(x_t')      # 仅低代价残差噪声更新
        # 省略 QKV/FFN GEMM 与外部权重访问
    else:
        x_{t'-1} = full_attention_ffn(x_t')          # 完整计算
```
关键性质：动作规划 DiT 的相邻步相似度全矩阵都高（图像 DiT 只在对角附近高、非对角迅速降为 0），因此适合"整步粗粒度跳过"而非逐区域细粒度复用；注意力与 FFN 相似度趋势一致，可统一按步优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件框架 S2 策略，跳过集离线按相似度 + MSE 判据生成；硬件由 multimodal scheduler 的迭代索引表存储跳过配置。使用：约 40% 去噪迭代被消除；与"每 20 个跳过迭代重插完整去噪"配合重置累积误差；消融中在动作冗余之上再贡献 2.90× 总加速（1.74× → 2.90×）；特征复用带来的轻微"动量效应"使成功率甚至略高于 baseline。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## 校准多模态近似计算（Calibrated Multimodal Approximate Computing，模型级冗余利用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
模型级多模态冗余利用策略：多模态输入按更新频率（lifespan）分层——language token 整任务不变、vision token 跨多个去噪步不变、action token 每步都变（lifespan 最短），平均 91.7% 的输入每步不变却参与全部计算。直接跳过不变 token 的计算会引入两类错误：(1) attention-shift 误差：SoftMax 依赖全部 token 的全局比较，去掉冗余 token 后 Q/K 维度缩短、归一化分母变小，注意力分布整体漂移；(2) 迭代累积误差：DiT 多步迭代使单步近似误差累积到不可接受。因此采用"校准"：缓存冗余模态的 K 特征补全 SoftMax 分母、缓存 V 特征保持聚合对齐，并周期性重插完整去噪重置误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
计算流（论文 Fig.9–10）：
```
# 按 lifespan 识别不变 token
unchanged = vision_tokens_unchanged ∪ language_tokens_unchanged
K_cached = K[unchanged]            # 缓存冗余模态 K（跨步不变）
V_cached = V[unchanged]            # 缓存冗余模态 V
for each step:
    Q,K = project(updated_tokens)  # 仅更新 token 做投影
    score = Q @ [K; K_cached]^T    # K_cached 校准 SoftMax 输入：分母保持全 token 贡献
    attn = softmax(score / sqrt(d))
    out = attn @ [V; V_cached]     # V_cached 保持数据对齐
    # FFN 同样跳过不变 token 行
# action 模态列稀疏（约 52% 列近零）：
    zero_cols = where(all(|score[:, c]| < eps))
    skip softmax 输出零列与 V 对应行；相应 V 投影行旁路
# 每 20 个跳过迭代插入一次完整去噪，重置累积误差
```
张量视角：设 token 总数为 N、不变 token 为 M（≈91.7%），跳过后每步 GEMM 规模从 N 降至 N−M；attention 矩阵只对更新 token 行 × 全部 token 列计算，未更新行沿用上步结果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：软件框架 S3 策略 + 硬件 multimodal scheduler（lifespan 序列生成器管理校准数据地址）。使用：消除 91.74% 的冗余 token 计算；消融中贡献总加速 32.60× 的主体（配合数据管理硬件避免 GPU 上 35.4% 的数据操作时延）；vision token 从 64 增至 512 仍保持 115.48Hz（跳过冗余历史视觉帧），language token 增长几乎无影响（指令整任务不变）。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## PTQ4DiT（DiT 后训练量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PTQ4DiT（Wu et al., NeurIPS 2024, arXiv:2405.16005）是首个面向 Diffusion Transformer 的有效后训练量化方法，解决 DiT 量化两大难题：显著通道（salient channel）的激活/权重量化误差大（极值通道在均匀量化下误差显著，需截断）、显著激活跨去噪时间步剧烈变化（静态量化参数失效）。方法：Channel-wise Salience Balancing（CSB，用激活/权重分布统计出的 salience balancing matrix 做通道级变换，利用权重与激活显著通道的互补性——二者不会同时取极值）；Spearman ρ-guided Salience Calibration（SSC，沿时间维扩展通道显著性，加权到 CSB 收益最大的时间步）；重参数化把平衡矩阵离线吸收进相邻层（推理零额外开销）。效果：W8A8 接近全精度、W4A8 保持高质量生成。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文中的角色：所有硬件基线（A40 GPU、EXION、Ditto）统一按 PTQ4DiT 量化到 INT8 精度，其中激活 tensor-wise、权重 channel-wise（论文 A-Evaluation-Methodology 原文表述）。量化流程：校准集上统计激活/权重分布 → 求 salience balancing matrix → 变换激活与权重使显著通道误差减小 → 离线吸收进相邻层参数 → 推理时纯 INT8 GEMM。归一化基线：所有加速器归一化到 A40 峰值 37.4 TFLOPS，保证架构对比公平（排除硬件规模差异）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源实现见论文官方仓库（论文 Web 证据：arXiv:2405.16005，https://ar5iv.labs.arxiv.org/html/2405.16005 ）。使用：DiT 图像生成与动作规划的 INT8 部署基线；对动作规划 DiT 而言，INT8 量化与三层冗余消除正交——DiTPA 在 PTQ4DiT 之上进一步利用动作/去噪/多模态冗余获得 386.93×/13.22×/9.54× 加速。注意：论文只报告了采用 PTQ4DiT 的 INT8 基线设置，未给出量化感知的消融数据。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## 场景感知自动化阈值搜索（Scenario-aware Pareto 多目标阈值搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
为 DiTPA 的三层冗余消除策略（朝向阈值 th、迭代跳过比例）自动选取阈值的离线搜索机制：用场景特定校准任务集（占总任务 3%）采样候选阈值组合，对成功率与速度做多目标 Pareto 优化得到前沿，再按当前场景需求沿前沿选点。场景分类（对齐 VLA 加速框架惯例，如 Sp-VLA）：latency-sensitive（如分拣机器人：动作频率 ≥200Hz、容忍 ≤2% 成功率损失，错误可由 re-planning 补救）与 mission-critical（如救援机器人：不可重规划、要求近零精度损失）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
搜索流程伪代码：
```
calib = 场景特定校准任务集（总任务数的 3%）
candidates = grid(orientation_th ∈ {0°,1°,2°,3°,4°}, skip_ratio ∈ {...})
for (ot, sr) in candidates:
    (success_rate, action_freq) = evaluate(calib, ot, sr)   # 模拟器 rollout
front = pareto_front(candidates, maximize=success_rate, maximize=action_freq)
th = pick_on_front(front, requirement)   # latency-sensitive: ≥200Hz 且损失 ≤2%；mission-critical: 损失≈0
# 部署期：同场景任务共享动作空间动力学与轨迹模式，固定朝向阈值
# 未来工作：按相邻视觉帧平均光流幅值自适应调整阈值（高动态环境）
```
结果：LIBERO-Long 属 latency-sensitive，搜索得朝向阈值 2°、迭代跳过 40%，并每 20 个跳过迭代插入完整去噪；收紧朝向阈值可得 226.68× 加速且零成功率损失，收紧迭代跳过阈值得 245.69× 且零损失（均归一化到 GPU）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线 Python 搜索 + LIBERO 模拟 rollout 评估（论文未开源该搜索脚本细节，论文未明确说明具体实现工具）。使用：不同机器人场景部署前的参数整定；与消融/敏感性分析配合（论文 Fig.20/21 展示阈值 0°–4° 的成功率-加速权衡曲线：>3° 成功率骤降、>2° 加速收益饱和——因为新增跳过动作多为重复抓取等无效操作）。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## Adam 优化器（AdamW、fp32 主副本与 CPU Adam 卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Adam = 一阶自适应矩估计优化器：维护梯度一阶矩 m 与二阶矩 v 的指数滑动平均（β1≈0.9、β2≈0.999），偏差校正后按 m̂/(√v̂+ε) 方向更新权重。混合精度训练（bf16/fp16）中：前/反向用低精度权重与梯度，Adam 状态与主权重副本保持 fp32——每参数共 2B 梯度 + 12B 状态（m 4B + v 4B + 主权重 4B）+ 2B 参数副本（Web 证据：HF model memory anatomy）。
- CPU Adam（ZeRO-Offload 风格）：SIMD + 循环展开 + 多线程，每参数 17 次浮点运算；DisDP 以此估算 PS 算力需求（100Gbps 聚合梯度需 99 GFLOPS）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 每参数更新（x 为权重、g 为梯度、lr 学习率）：
```
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
m_hat = m / (1-β1^t);  v_hat = v / (1-β2^t)
x = x - lr * m_hat / (sqrt(v_hat) + ε)
```
- DisDP 中 Adam 全部在 PS 的 CPU 上执行（worker GPU 不碰优化器），是 step-centric 流水中的第 3 步（读 2B 梯度 + 12B 状态、写 12B 更新状态 + 2B 参数副本）；收敛与 GPU 侧 Adam 一致（OPT-66B rm-static 微调 loss 曲线与 ZeRO-Infinity 重合）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch FusedAdam、DeepSpeed CPUAdam/ZenFlowCPUAdam；fp32 状态放 CPU/SSD（ZeRO-Offload/Infinity）或单台 PS（DisDP）。使用：bf16 训练需考虑动态范围（论文用 bf16 + 激活检查点）；卸载场景需与参数服务流水重叠以隐藏优化器时延。信息缺口：论文未给出 β1/β2/ε 具体值。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training

## Magic State Distillation（MSD，魔法态蒸馏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSD 是一种容错协议：消耗 n 份低保真（误差 $p_{\rm in}$）的噪声魔法态，输出 k 份更高保真的魔法态，用"数量换质量"。逻辑链：①非 Clifford 门（如 $T=\mathrm{diag}(1,e^{i\pi/4})$）受 Eastin-Knill 定理限制，在大多数码上无法 transversal 实现，通常经 gate teleportation 消费魔法态实现；②直接注入的魔法态误差 $p_{\rm inj}$ 远高于算法需要的逻辑误差 $p_L$；③蒸馏码（如 [[15,1,3]] Reed-Muller 码，Bravyi-Kitaev）带 transversal T 门，把 n 份输入映射到 k 份输出，单轮输出误差 $p_{\rm out}\approx c\,p_{\rm inj}^t+O(p_L)$（t=O(d) 为抑制阶数，c 为协议常数；15-to-1 协议 c=35、t=3）；④多轮串联 $p_{\rm out}^{(r)}\sim \tilde c\,p_{\rm inj}^{t^r}$，两轮即够大多数应用。Bravyi-Haah triorthogonal 构造使开销随精度多对数增长 $O(\log^\gamma(1/\epsilon))$、$\gamma=\log_2 3\approx 1.6$。在 surface-code 架构中蒸馏工厂占物理 qubit 的 90% 以上（"T-gate 瓶颈"）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文的 Pauli-measurement-based 形式（triorthogonal 矩阵 $G\in\{0,1\}^{m\times n}$，前 k 个奇权重行=输出、m−k 个偶权重行=parity check，列 c 对应旋转 $e^{i\pi/8}Z^{\otimes S_c}$，$S_c=\{r:G_{rc}=1\}$）：
  ```
  init: 全部物理 qubit 置 |+⟩ + 1 轮 syndrome extraction → m 个逻辑 |+⟩
  for c in 1..n:                # n=15 个 π/8 旋转
      取 1 个噪声 |T⟩（p_in），经注入 gadget 实现 exp(iπ/8·Z^⊗S_c)
      （测量结果=1 时补 exp(iπ/4·Z^⊗S_c) 条件校正）
  measure 偶行 qubit 于 X 基；postselect 全 |+⟩
  成功 → 前 k 行输出蒸馏 |T⟩^⊗k；失败 → 丢弃本轮
  ```
  相比 Bravyi-Haah 原构造（n-qubit 稳定子态制备 + 逐 qubit T 门 + Clifford unencoding），只用 m 个逻辑 qubit、更少 Clifford 门。本论文把 15-to-1、20-to-4、8-to-CCZ、49-to-1、51-to-3CS、64-to-2CCZ 全部放到单个 BB 码块内执行。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式分代：①surface-code lattice surgery 工厂（Litinski）——多 patch + surgery 路由，15-to-1 需 4620 qubit/256 时间步；②QLDPC 单块工厂（本论文）——378（gross）/734（two-gross）qubit，15-to-1 two-gross 输出 1.0×10⁻⁸、49-to-1 达 2.0×10⁻¹¹（$p_{\rm phys}=10^{-3}$）；③作为 MSC 的二级协议突破其 10⁻⁹ 天花板（$35\cdot(10^{-9})^3=3.5\times10^{-26}$）。工业实现：Microsoft QDK 的 RoundBasedFactory（"15-to-1 RM prep"+"15-to-1 space efficient" 级联）。Web 参考：Bravyi-Kitaev quant-ph/0403025（PRA 71, 022316）、Bravyi-Haah arXiv:1209.2426（PRA 86, 052329）、Litinski 1708.07197 系列。
- 补充（O3LS 论文）：O3LS 的评估用 15-to-1 蒸馏协议的 magic state factory（Bravyi-Kitaev），工厂置于自动生成的 squeezed 数据布局之外、与数据区之间保证至少一条路由路径连接；π/4 与 π/8 Pauli-product measurement 用标准 gate teleportation 协议（Litinski [34] Fig.7/11(b)）。MSD 是 O3LS 优化目标的外在约束：布局与调度只优化数据区（架构因素），magic state 供给成本不在其调度范围内，但工厂放置（布局外+连通性）纳入布局设计评分（C(B) 要求 ancilla 与工厂路由连通）。
- 补充（TACO 论文）：TACO 用 15-to-1 蒸馏作为默认协议并给出资源模型：11 个逻辑 qubit tile、每 11 cycles 产 1 个 magic state、错误抑制 35p³（p=10⁻³ 时 3.5×10⁻⁸）。TACO 把 MSD 当作架构一级的"可调资源"而非固定开销：magic state 吞吐（块数）由空间-时间体积最小化决定——18 比特 QFT 从 1 到 10 magic states/cycle 扫描，最优 4 个/cycle（体积比单 block 降 57%）；20 比特 QFT 下 3 个 distillation block × 11 units = 363 tiles 支撑 760,901 cycles（对比 PBC Fast 11 blocks/121 tiles/2,382,355 cycles）。随 T 门成本下降（MSC），TACO 的 magic-state 开销占比几乎恒定而数据体积显著下降——这是与 PBC 的本质区别（PBC 中 T 优化收益递减，TACO 中收益更大）。
- 补充（Triage 论文）：Triage 把 T 门（非 Clifford）当作整个解码调度问题的"绝对同步点"来用——T 门经 gate teleportation 实现，末端的 classically-controlled S-gate 校正不能穿过 T 门吸收进 Pauli frame，执行前必须物理纠正 E_acc（恢复 |ψ⟩），因此解码器必须先解码完相关因果锥完成 Pauli frame 同步，否则逻辑操作 stall、插入 idle 层、LER 上升。这与"Clifford 门可异步"形成 dichotomy：正是魔术态/非 Clifford 操作把解码从吞吐问题变成带优先级的实时调度问题（Triage 的 deadline/causal cone 属性、紧急模式全部围绕同步点设计）。Triage 评估用 15-to-1 MSD 协议与 T-gate 密度不同的 benchmark（T-Den. 7.69%-49.61%，如 rotation_C+T 11.80%、MSD15to1 45.83%），验证调度器在 T 门密集应用上的收益。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

## Magic State Cultivation（MSC，魔法态培养）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSC 是 Gidney 等人 2024 年提出（"Magic state cultivation: growing T states as cheap as CNOT gates"，arXiv:2409.17595）的魔法态制备替代方案：把物理 |T⟩ 注入 d=3 三角 color code，利用 2D 三角 color code 上 H_XY 门的 transversal 性"长"到 d=5 surface code，再经"escape"逃逸进大片 surface code；全程靠多轮 post-selection 抑制不保真度（早期阶段全 postselect——开销随码距指数增长所以只在小题距做；escape 阶段只 postselect 特定 detector + decoder "gap" 判据）。本论文作为 baseline：454 qubit、$p_{\rm out}=2\times10^{-9}$（$p_{\rm phys}=10^{-3}$，d=5，时间步 2167）。缺点：post-selection 指数扩展、不 asymptotic，够不到大规模 FTQC 需要的 ≤10⁻¹²。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文中的两级工厂 pipeline（Cultivation + (15-to-1)Two-gross）：
  ```
  ① surface code patch 上注入物理 |T⟩ → 多轮 post-selection + 物理非 Clifford 门
     → 输出 p_in=10⁻⁶ 级 |T⟩（454 qubit）
  ② 经 adapter（universal surgery ancilla）的 inter-module 测量注入 BB 块
  ③ BB 块内 15-to-1 蒸馏（734 qubit）→ p_out ≈ 35·(10⁻⁶)³ ≈ 3.5×10⁻¹⁷ 量级
  总：454+734 qubit、τ=11080、二级体积 8.1×10⁶、p_out≈4.1×10⁻¹²（10⁻³）/≤10⁻¹⁷（10⁻⁴）
  ```
  MSC 供误差"够低但不达标的输入"，MSD 再压 t 次方——两者互补而非互斥。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 后续优化：Vaknin et al.（arXiv:2502.01743）用 surface code 非局域连接 transversal 操作免 grafting、大幅降低 post-selection 率；in-patch multiplexing（arXiv:2605.03616）四站点复用使 discard 从 ~83% 降到 ~49%（d₁=5, p=10⁻³）；fold-transversal surface code cultivation（arXiv:2509.05212）保持 surface code 家族内、时空开销最低。实验验证：Google/NASA（arXiv:2512.13908）在超导处理器上实现 ~40× 不保真度改善、post-selection 后保留 ~8% 数据。
- 补充（TACO 论文）：TACO 把 MSC（magic state cultivation）作为比 MSD 更高效的 magic state 供给方案用于架构对比：MSC 使 magic state 体积比蒸馏降一个数量级（[34]），TACO 最优架构含 4 个 compute & distillation block 时 QEC cycles 从 760,901（MSD）降到 595,604（MSC），code distance 仍为 19，总 qubit-cycle 体积 3.8×10^10（vs PBC Compact/Fast 的 7.9×10^11/1.1×10^11，降 95%/63%）。MSC 的低成本也改变了架构权衡：当 T 门成本接近 CNOT（图 4 中自 2012 年下降 100×+）时，Clifford 开销占 58-65% 成为主导，TACO 的 Clifford 消除收益放大——TACO 与 MSC 互补（MSC 降 T 成本，TACO 降 Clifford 成本，共同压低体积）。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing

## Triorthogonal Matrix（三正交矩阵 / Triorthogonal Codes）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bravyi-Haah 2012（arXiv:1209.2426，PRA 86, 052329）引入：二元矩阵 $G\in\{0,1\}^{m\times n}$ 称为 triorthogonal，当且仅当任意两行的 support 重叠为偶：$\sum_j G_{a,j}G_{b,j}=0\pmod 2$，且任意三行的 support 重叠为偶：$\sum_j G_{a,j}G_{b,j}G_{c,j}=0\pmod 2$。作用：任何含 k 个奇权重行的 triorthogonal 矩阵映射到一个有 k 个逻辑 qubit、允许 transversal π/8 旋转（T 门，可能配 Clifford）的稳定子码；偶权重行给出蒸馏协议中探测输入魔法态错误的稳定子。由 triorthogonal 码可构造 rate 1/3 的蒸馏协议，开销 $O(\log^\gamma(1/\epsilon))$、$\gamma=\log_2 3$。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文用 G 直接定义协议执行：列 c → 旋转 $e^{i\frac{\pi}{8}Z^{\otimes S_c}}$（$S_c=\{r:G_{rc}=1\}$ 为 Z 作用于的行集合），奇权重行 → k 个输出 |T⟩，偶权重行 → m−k 个 X 基 parity check 测量 + postselect。协议压缩（qubit recycling）即对 G 做保持 triorthogonality 的变换：列置换（重排对易旋转）、块内行置换（奇行之间/偶行之间各自换序）、$\mathbb{F}_2$ 行加法；对每行取首/末 1 列 $(f_i,\ell_i)$ 定义工作集 $W(j)=\{i:j\ge f_i\text{ 且 }(\text{偶行 }j\le\ell_i\text{ 或 奇行})\}$，峰值活动 qubit 数 $C(G)=\max_j|W(j)|$ 即压缩目标。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Bravyi-Haah 论文同时给出数值生成 triorthogonal 矩阵的方法。推广：generalized triorthogonal matrices（arXiv:1709.02832，Haah-Hastings? 实际为 "Codes and protocols for distilling T, controlled-S, and Toffoli gates"）支持蒸馏 |T⟩、|CS⟩、|CCZ⟩——本论文的 51-to-3CS、64-to-2CCZ 即属此类。与 CSS-T 码、自对偶码有理论联系（arXiv:2408.09685 等）。本论文压缩算法的最优解 NP-hard（k=0、偶行权重 2 时化简为 cutwidth 问题），实际用贪心聚类行起止 + 定向行加法，<5 s 编译。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture

## Gate Teleportation（门隐形传态）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 用预先制备的魔法态 + 只含 Clifford 门的电路实现非 Clifford 门（如 T 门）的标准机制。逻辑链：①准备 ancilla 魔法态 $|T\rangle=(|0\rangle+e^{i\pi/4}|1\rangle)/\sqrt 2$（= T|+⟩）；②目标 qubit 与 ancilla 之间做 transversal CNOT 纠缠；③测量 ancilla；④按测量结果施加条件 Clifford 校正（对 T 门是 $S^\alpha$ 门）。净效果 = 对目标施加 T 门，全程只用了可容错实现的 Clifford 操作——绕过 Eastin-Knill 定理。代价：每个 T 门消费 1 个魔法态，因此魔法态制备（蒸馏/培养）成为 FTQC 成本主体，运行时 teleportation 相对便宜。变体：1-bit teleportation 与 Knill teleportation（qubit 数、连接性、分布式场景有取舍）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文把 teleportation 实例化为 exp(iπ/8·P) 的三种注入方案（图 4）：标准注入随机产生 exp(±iπ/8·P)，需条件校正 exp(iπ/4·P)。①Direct injection + factory correction（a）：|T⟩ 经 inter-module 测量直接 teleport 到全部目标 qubit，校正用 measurement-to-rotation 电路显式实现；②Pivot injection（b，默认）：|T⟩ 先 teleport 到 pivot L0 再经 in-module 测量传到目标，校正吸收进 pivot 的条件 X/Y 测量——把噪声注入限制在单 qubit、inter-module 噪声集中在源-pivot 界面；③Direct injection + source correction（c）：源模块支持高保真 Y 测量时由源自行校正，pivot 完全不参与。错误传播细节：注入错误中 Z 分量等价于 Z 制备错误（多引入 π/2 旋转）、X 分量可被最终 X 基测量吸收，故魔法态 qubit 上的错误危害较低。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- surface-code 语境：lattice surgery 工厂蒸馏出的 |T⟩ 经 surgery 路由到消费点 teleport（Litinski 系列）；本论文在 QLDPC 语境经 adapter（universal surgery ancilla）做 inter-module 测量注入。工程上还需 byproduct Pauli 跟踪：所有条件校正都是 Clifford/Pauli，可经典跟踪延迟处理。Web 参考：PennyLane magic states 教程、Qualtran TGate 文档、arXiv:2502.16939（rotated surface code 上 T 态消耗的扩展稳定子仿真）。
- 补充（TACO 论文）：TACO 把 gate teleportation 作为 T/Rx(π/4) 门的标准实现并给出其硬件成本模型：消费已备好的 magic state 需 2.5d+4 QEC cycles（d=19 时约 51.5 cycles），S 门经 |Y⟩ 态 teleportation 共 1.5d+3 cycles；magic-state injection 符号概率性（±），若符号相反用 Clifford correction 恢复——比再做一次非 Clifford 旋转便宜得多，故可接受。teleportation 的"每门一个 magic state"属性使 TACO 把 magic state 吞吐当作架构参数（compute block 每 cycle 一个 π/4 旋转，需 4 个 magic states/cycle 配置），并由高 π/4 旋转局部性保证 target qubit 驻留 compute block 避免移动开销——teleportation 的消费端成为架构 locality 优化的直接对象。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing

## Protocol Compression / Qubit Recycling（协议压缩·逻辑量子比特回收）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 把蒸馏协议（triorthogonal 矩阵）改造为更少"峰值并发活动逻辑 qubit"的等价协议，而不改变蒸馏性能（错误抑制阶数 t 与输出数 k 不变）。核心观察：triorthogonal 矩阵常含大片全零子块——偶权重行在首个 1 之前无需初始化（$f_i$ 前 idle）、在末个 1 之后（$\ell_i$ 后）可测量释放；奇权重行编码输出、一旦初始化不可释放。已释放的偶行 qubit 可"回收"给稍后初始化的行使用。衡量指标 $C(G)=\max_j |W(j)|$。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- ```
  输入: triorthogonal G ∈ {0,1}^{m×n}（前 k 行奇权重=输出，余 m−k 行偶权重=check）
  每行 i: f_i=首个 1 的列（无则 +∞），ℓ_i=末个 1 的列（无则 −∞）
  working(i,j) = (j≥f_i) 且 (偶行: j≤ℓ_i；奇行: 恒真)
  C(G) = max_{j∈[n]} |{i: working(i,j)}|
  允许变换（保持 triorthogonality）:
    列置换（重排对易的 π/8 旋转）/ 块内行置换（奇行间、偶行间）/ F_2 行加法
  目标: min C(G')，s.t. G' 与 G 等价（同蒸馏性质）
  ```
  效果：49-to-1 从 13→7 个逻辑 qubit、51-to-3CS 从 18→9、64-to-2CCZ 从 17→10，使这些协议可装进单个 gross/two-gross 块（pivot 注入方案容量上限 11 逻辑 qubit）。最优解 NP-hard（k=0、偶行权重 2 时化简为 cutwidth），实际用贪心聚类行起止 + 定向行加法（编译 <5 s）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 该方法是"协议级"优化：不依赖具体电路实现细节，对任何 triorthogonal 生成的协议（|T⟩/|CS⟩/|CCZ⟩）通用。相关并行工作（arXiv:2606.07734 "Exploring the landscape of compact magic-state distillation factories"）用经典纠错码 + SAT 求解器把 49T-to-1T 压到仅 5 个活动 qubit（mid-circuit measurement + reinitialization 回收），说明压缩方向仍是活跃前沿；其"压缩 + 检测"二分法（压缩：含多 T 门的电路完成简单任务；检测：错误产生可观测症状）是对本技术的互补视角。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture

## Sound Spatialization（声音空间化 / SS）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 声音空间化（Sound Spatialization, SS）是把场景中的源音频转换成听者双耳（binaural）信号的过程，显式建模声音在环境中如何传播、如何受听者位置与朝向影响，是 VR 沉浸感与空间感知的关键（区别于普通音频渲染的解码/混音/panning/简单混响）。ECHO 论文把 SS 流水线分解为三阶段（对应其 Fig.4）：①声传播（Sound Propagation）——用镜像源法（ISM, Image Source Method）模拟声音在室内经墙面/障碍物反射、衍射、混响后到达听者的传输路径，基于听者位姿、源位置、场景几何与材质计算房间冲激响应（RIR）；②BRIR 生成（Binaural Room Impulse Response Generation）——用听者特定或通用的头相关传递函数（HRTF, Head-Related Transfer Function）把 RIR 转换成左右耳各自的双耳房间冲激响应（BRIR），建模头、躯干、耳朵的滤波；③可听化（Auralization）——把源音频与 BRIR 做卷积生成空间化信号。三个源在不同位置时三阶段需独立执行，因此渲染成本随源数线性上升。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - SS 在 VR 中是一个感知-估计-渲染闭环流水线：传感（S^M 单目 SLAM 图像 + S^I IMU）→ 位姿估计（PE）→ 声传播+BRIR 生成（R^1）→ 可听化（R^2）→ DAC 输出。ECHO 论文按此建模 motion-to-sound 延迟 T_m-s = T_IN + T_S + T_P + T_R1 + T_R2 + T_O。具体计算过程（一个声源）：设源在房间位置 x_s、听者位姿 (R,t)，ISM 把源对每面墙镜像得到镜像源 x_s'，对每个镜像源/直达声计算到达听者的路径、延迟 τ 与幅度 a（含墙面吸收），叠加所有路径得 RIR h(t)=Σ_j a_j δ(t-τ_j)（+扩散混响尾）；BRIR 用 HRTF 卷积：h_L(t)=h(t)*hrtf_L(θ,φ)、h_R(t)=h(t)*hrtf_R(θ,φ)，其中 (θ,φ) 是源相对听者的方位/仰角；可听化输出 y_L(t)=s(t)*h_L(t)、y_R(t)=s(t)*h_R(t)。多源时每源重复上述三阶段再求和。ECHO 评估中 R^1 用 Pyroomacoustics（CPU）与 gpuRIR（GPU）的 ISM 实现，R^2 为 BRIR 卷积。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：房间声学模拟库（Pyroomacoustics：Python 房间模拟 + 阵列处理，ISM 生成 RIR/BRIR；gpuRIR：GPU 加速 RIR 模拟）；HRTF 数据集（KEMAR dummy-head 测量的远场 HRTF，或近场 HRTF 数据库）；可听化可用分区卷积（partitioned convolution）做实时长滤波器卷积。在 VR 系统中把最新 head pose 传给渲染器，对每音频块（5-20ms）用最新位姿执行传播/BRIR 生成，再用最近 BRIR 卷积当前块，两阶段异步流水以提升稳态吞吐（ECHO Fig.6b）。实时性约束：motion-to-sound 延迟须 <50-60ms 保沉浸感。优化方向（ECHO）：按头朝向做声学注视（acoustic foveation）减少活跃源数、用 GPU 并行（gpuRIR）加速传播、算法-硬件协同。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## Acoustic Foveation（声学注视）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 声学注视（Acoustic Foveation）类比视觉注视渲染（foveated rendering）：人类听觉空间分辨率并非均匀，朝向中央方位（frontal）时最灵敏、偏离中央方位（peripheral）时显著下降，因此处于低感知重要性区域的声源可被分组并合并为单一源（把它们的源音频求和），从而降低 SS 渲染计算量，同时在最关键区域保持空间精度。ECHO 论文指出先验工作（IEEE VR 2025 "Perceptually-Guided Acoustic Foveation"）基于该感知特性；论文将其扩展为"鲁棒声学注视"（robust acoustic foveation），即考虑位姿估计误差的注视聚类。核心心理物理依据：最小可听角度（MAA）随方位角单调递增（正前方 ~3°、侧向 ~90° 处接近 40°）；距离感知不是精确值而是区间，距离判断标准差超过源距离的 20%，故相距小于该阈值的源在感知深度上不可区分。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - ECHO 的鲁棒声学注视算法流程（对应 Fig.8）：①把 3D 房间沿高度切分为若干水平层，把问题降为若干 2D 子问题；②每层内按听者朝向（Ori.）计算每个源的方位角 θ，方位角差 < MAA 阈值的源并入角向组；③角向组内按径向细化：距听者距离差 < 最远源距离的 20% 的源视为感知等价并入簇；④位姿误差鲁棒化：设角向跟踪误差 Δθ^r 与平移误差 Δt，源相对听者的角偏差下界为 θ_eff = θ - Δθ^r - Δθ^t（Δθ^t≈||Δt||/r，r 为源距离），MAA 在 θ_eff 处取值，从而实施更严格的聚类阈值，保证即使位姿有误差也不破坏感知有效性；⑤每个簇用位于原源质心的单一虚拟源替代，进入后续 BRIR 生成与可听化。伪代码示意：
  ```
  def acoustic_foveation(sources, pose, layers, MAA_fn, dist_thresh=0.2):
      for layer in layers:
          for src in sources_in(layer):
              src.theta = azimuth(src, pose.orientation)
              src.theta_eff = src.theta - pose.err_rot - pose.err_trans/src.dist
          angular_groups = group_by_MAA(sources_in(layer), MAA_fn(src.theta_eff))
          for g in angular_groups:
              cluster(g) by radial distance within dist_thresh of farthest src
      return [virtual_source(centroid(c)) for c in clusters]
  ```
  ECHO 把 MAA 实现为对 [77] 感知曲线的分段方位函数；用户研究（17 人，544+288 trial，2IFC）显示注视渲染与全渲染在感知上无显著差异（p≈0.25/0.82），证明注视保留空间音频保真度。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：聚类在渲染前完成，源数从 N 降到簇数 K（K≪N），T_R1（每簇一次 ISM+BRIR）与 T_R2（每簇一次卷积）随之下降。与声源空间分布强相关：Poisson Cluster Process（PD，空间聚簇）比均匀随机分布（UD）产生更少簇（256 源时 70 vs 129），注视收益更大（PD 在 256 源仍 <50ms，UD 接近 70ms）。依赖高质量 head pose（聚类以位姿为条件），位姿误差会破坏 MAA——ECHO 用 θ_eff 保守化解决；其 RRE 更低（1.014° vs ORB-SLAM3 1.194°）也直接支撑注视有效性。使用场景：多声源 VR 场景（8-256 源）、大型展厅等；注意远场 HRTF 假设（源距 ≥1m）。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## Minimum Audible Angle（MAA，最小可听角度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MAA 是心理声学中最小可听角度：听者能分辨两个同类型声源方位差异的最小角度。经典研究（Mills 1958）表明 MAA 依赖频率与方位：正前方（0°）最灵敏，200-4000Hz 下可小于 4°（500Hz 纯音可到 ~1°），随方位偏转急剧劣化，侧向（~90°）约 9-10°。ECHO 论文引用的心理物理数据（[77]，IEEE VR 2025）给出单调递增关系：正前方 θ≈0° 时 MAA 低至 3°，侧向 θ≈90° 时升至接近 40°；ECHO 将 MAA 实现为分段方位函数拟合该曲线。MAA 是声学注视聚类的核心判据：方位差小于 MAA 的两个源听者无法分辨，可合并。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在 ECHO 声学注视 pipeline 中 MAA 是聚类阈值函数：对每个源按 θ_eff（考虑位姿误差的保守下界方位）查 MAA(θ_eff)，角向分组即判断 |θ_i - θ_j| < MAA(θ_eff)。由于 MAA 随 θ 单调增，越靠侧向越容易合并源（感知分辨率低）；正前方保留最多细节（MAA 小）。ECHO 特别指出 RRE（相对旋转误差）会直接影响 MAA——方位误差扩大时实际 MAA 比名义小，故用 θ_eff 收紧阈值。计算例子：听者朝正北，源 A 方位 10°、源 B 方位 15°，MAA(10°)≈3°+ε，|15-10|=5°>MAA 则不分组合并；源 C 方位 70°、源 D 方位 100°，MAA(70°)≈30°，|100-70|=30°≤MAA(70°) 则并入一组。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：①心理物理测量——2IFC（two-interval forced choice）任务让听者判断两个音序列中哪一个源的方位与参考不同，二分逼近求阈值（IEEE VR 2025 方法，双耳耳机渲染）；②应用——作为 foveation 的聚类阈值、双耳渲染算法质量评估（用 MAA 测渲染引入的方位 JND，PKU 工作显示 Ambisonics 阶数越高 MAA 阈值越低）。ECHO 将其固化为分段函数并保守化（θ_eff 处取值），在 17 人用户研究中验证 47-50% 偏好率（与全渲染不可区分）。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## ORB-SLAM3（视觉-惯性 SLAM 框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ORB-SLAM3 是 University of Zaragoza 的 Carlos Campos 等人提出的开源视觉-惯性 SLAM 框架（IEEE T-RO 2021），支持单目/双目/RGB-D/惯性（IMU）配置，被广泛用于机器人与 AR/VR 的 marker-free inside-out 6DoF 位姿跟踪。它由三线程组成：①Tracking（跟踪）线程——对每帧做 ORB 特征检测（图像金字塔多尺度）、描述子匹配、PnP 局部地图跟踪（LM tracking）、IMU 预积分、位姿估计与关键帧决策，逐帧执行，其每帧延迟决定跟踪频率上限；②Local Mapping（局部建图）线程——关键帧插入时处理共视/生成地图点、局部 Bundle Adjustment（惯性模式下滑动窗口）；③Loop Closing（回环）线程——BoW2 词袋重定位检测回环、Sim(3)/SE(3) 几何验证、位姿图优化。后两者异步、仅在特定条件触发，不阻塞实时跟踪。ECHO 论文用 ORB-SLAM3 作为商业 VR 跟踪管线的代表做延迟剖析与优化对象。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在 ECHO 中 ORB-SLAM3 是位姿估计（T_P）主路径，每帧流程：输入 480×640 单目灰度帧（AEA 数据集 10Hz）+ IMU（1000Hz）→ ①构建 8 级图像金字塔（迭代下采样，尺度不变性）；②每层按 35×35 像素 cell 划分，FAST 角点检测器逐像素扫描，比较中心像素强度与 16 邻域像素，n≥12 个连续邻域显著更亮/更暗即角点，再用强度矩为关键点分配主方向（旋转不变性）——ORB 提取约占 tracking 时间 23.92ms；③BRIEF 描述子匹配建立当前帧 2D 关键点与 3D 地图点对应，RANSAC 剔除离群点；④IMU 预积分 + 位姿预测初始化位姿，LM tracking 以最小化重投影误差做 40 次 Gauss-Newton 迭代优化（每迭代投影全部对应并算 Jacobian）——约 24.09ms；⑤关键帧决策。ORB 提取与 LM tracking 合计占 >95% tracking 时间（总 ~50ms），是 ECHO 优化与硬件加速的两个靶点。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：开源仓库 https://github.com/UZ-SLAMLab/ORB_SLAM3（C++，g2o 后端优化，DBoW2 重定位）；可编译运行于桌面 CPU 与嵌入式平台（ECHO 用 Nvidia Jetson Orin NX）。用法：提供相机标定（含鱼眼 Kannala-Brandt 模型参数）、IMU 标定与数据集（如 AEA、TUM VI）即可跑通；评估位姿精度常用 evo toolkit 算 ATE（绝对平移误差，m）与 RRE（相对旋转误差，°）。ECHO 在其上叠加低精度量化、点过滤、RNN 高频位姿与硬件加速器，将每帧 tracking 延迟降到 11.0ms（TUM VI），且 ATE/RRE 与全精度相当。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## Local Map Tracking（LM 跟踪）与重投影误差位姿优化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LM tracking 是 ORB-SLAM3 跟踪线程中把当前帧特征与 3D 地图点匹配、并优化相机（头）位姿的阶段。流程：IMU 预积分与位姿预测给出初始位姿 → 选邻近关键帧建立 2D-3D 对应（BRIEF 描述子匹配）→ RANSAC 剔除离群 → 通过最小化重投影误差迭代优化 6DoF 位姿（旋转 R∈R^{3×3}、平移 t∈R^3）。目标函数为 min_{R,t} Σ_i ||u_i - π(R·x_i + t)||²，其中 x_i 是第 i 个世界系 3D 地图点、u_i 是其 2D 关键点、π(·) 是相机投影函数。ECHO 剖析显示该阶段约占 ORB-SLAM3 tracking 时间的一半（24.09ms），是两大瓶颈之一。VR 中 SLAM 相机多用鱼眼镜头（强径向畸变），π 采用 Kannala-Brandt（KB）非线性模型。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 优化过程（ECHO 论文给出细节）：KB 投影对相机系点 x^c=[x,y,z]^T 定义 ρ=√(x²+y²)、θ=arctan(ρ/z)、φ=arctan2(y,x)，径向畸变 d(θ)=θ+k1θ³+k2θ⁵+k3θ⁷+k4θ⁹，投影 π(x^c)=[fx·d(θ)cosφ+cx, fy·d(θ)sinφ+cy]^T，含三角函数与 9 阶多项式、开销高；位姿优化用 Gauss-Newton（论文引用 [104]）在流形上迭代 ~40 次，每次迭代对每个 3D 点执行坐标变换 x_i^c=R·x_i+t（大批量矩阵-向量乘，FP64）并计算投影/残差与 Jacobian。伪代码：
  ```
  R, t = init_from_imu_preintegration()
  for iter in range(40):
      J, r = [], []
      for (x_i, u_i) in correspondences:          # 2D-3D 对应
          x_c = R @ x_i + t                        # 坐标变换（主要瓶颈）
          r_i = u_i - project_KB(x_c)              # 重投影残差
          J_i = jacobian_KB(x_c)                   # 鱼眼投影 Jacobian
          J.append(J_i); r.append(r_i)
      delta = solve_gauss_newton(J, r)             # 法方程
      R, t = exp_map(R, t, delta)                  # 流形更新
  ```
  ECHO 的加速手段：①低精度——R 量化为 INT4（缩放 8）、x_i 用 FP8 E4M3，x_i^c=Q_INT4(R)·Q_FP8(x_i)/8+t 在混合精度下计算（除 8 靠指数位调整），投影/Jacobian 保持 FP32；②点过滤——按 FP8 量化误差 E1>α 与量化重投影误差 E2^q>β 剔除低质量/不稳定对应，位姿已准（拒绝率<r1）时随机丢弃 r2 比例（默认 α=0.1、β=120、r1=5%、r2=40%），平均削 75% 点；③硬件——计算引擎 8×8 脉动阵列算坐标变换、PJ 模块单遍算投影+Jacobian，T_P^MI 平均降 3.4×。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：LM tracking 是通用 VIO/SLAM 前端模式（VINS-Fusion、OKVIS、HybVIO 也有类似角点+重投影管线，ECHO 的优化可迁移）。落地方式：在 CPU 上跑优化循环（ORB-SLAM3 C++），或把坐标变换/投影/Jacobian 这类可并行、可复用的计算卸载到硬件（ECHO 加速器）；精度评估用 evo 工具算 ATE/RRE（RMSE），数据集用 AEA/TUM VI。使用注意：鱼眼模型的 trig/高阶多项式对数值敏感，量化时投影与 Jacobian 必须留 FP32 防精度劣化（ECHO 消融显示 INT4/FP8 量化相对 FP32/FP16 几乎无损：ATE 0.030 vs 0.030/0.029，RRE 1.153 vs 1.133/1.133）。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## 低精度位姿估计（INT4/FP8 混合精度量化 + 量化感知点过滤 + 选择性采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ECHO 提出的位姿估计低精度化方案，目标是削减 LM tracking 中坐标变换（x_i^c=R·x_i+t）的算术开销：该步是大批量矩阵-向量乘法，baseline 在 CPU 上以 FP64 执行。低精度方案：①旋转矩阵 R 元素 ∈[-1,1] 且正交，乘缩放因子 8 后四舍五入成 4-bit 有符号整数（Q_INT4(r)=clamp[round(8r), -8, 7]），选 8 而非常见 7 是因为除 8 可通过浮点指数位调整实现（硬件友好，免完整乘法）；②3D 地图点 x_i 用 FP8 E4M3（Q_FP8，范围 -448~+448，跨度 896m，覆盖室内场景）；③坐标变换变为 x_i^c=Q_INT4(R)·Q_FP8(x_i)/8+t；④因鱼眼投影非线性高、对数值敏感，π 与其导数全程 FP32。配套点过滤（量化感知）：按 FP8 量化误差 E1=||x_i-Q_FP8(x_i)||>α 剔点，一次性稳定性检查 E2^q=||u_i-π(Q_INT4(R)·Q_FP8(x_i)/8+t)||²>β 丢弃不稳定对应，选择性采样（被拒比例<r1 时随机丢 r2 剩余对应）避免近最优位姿时的多余优化。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 计算过程（量化 + 过滤 + 采样三步）：
  ```
  def low_precision_lm_track(R, t, correspondences):
      R4 = Q_INT4(R)                       # 缩放 8 → [-8,7] 整数（指数位调整）
      pts = [x for (x,u) in correspondences if ||x - Q_FP8(x)|| <= alpha]   # E1 过滤
      stable = []
      for x, u in pts:
          x_c = R4 @ Q_FP8(x) / 8 + t      # 混合精度坐标变换
          if ||u - pi_FP32(x_c)||^2 <= beta: stable.append((x,u))            # E2^q 检查
      if rejected_ratio(stable, pts) < r1:  # 位姿已准
          stable = random_sample(stable, keep=1-r2)                          # 选择性采样
      for iter in range(40):                # Gauss-Newton，复用 R4/Q_FP8
          ... # 每迭代在 INT4×FP8 脉动阵列上算 x_c，投影/Jacobian 用 FP32 PJ 模块
  ```
  消融证据：FP16/FP32 替换低精度模块只带来微小 RRE 增益（1.133 vs 1.153°），ATE 无差别（0.030 vs 0.030/0.029）→ 低精度几乎无损；去掉点过滤（No F）导致跟踪发散（无有效位姿输出），只留 E1（QAF）则 ATE/RRE 劣化到 0.042/1.264 → 稳定性检查+选择性采样必要。超参敏感性：α=0.1、β=120、r1=5%、r2=40% 为最优（α 过小 0.01 跟踪失败、过大 1 精度降；r2=80% 跟踪失败）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：软件侧在 CPU 上做量化和过滤（缩放 8 的 INT4 量化、FP8 转换），迭代优化中坐标变换与投影/Jacobian 卸载到 ECHO 加速器计算引擎（INT4 量化单元在线量化动态更新的 R——位姿在迭代中变化不能离线量化；FP8 单元转换 3D 点与 RNN 激活）。除 8 的除法在硬件上通过 FP 指数位 -3 实现。使用收益：平均丢弃 ~75% 点（point filtering 单独贡献 1.26× 延迟降低）、低精度额外 1.10×（相对 FP32 加速器），能量较无过滤方案省 2.39×。迁移性：该方法针对通用 VIO/SLAM 前端的坐标变换+重投影内核，可迁移到 VINS-Fusion/OKVIS/HybVIO 等。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## IMU 高频位姿估计（RNN + 量化感知训练 QAT）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ECHO 提出的 IO（Inertial-Only）模式位姿估计：因 SLAM 相机仅 10-30Hz，帧间捕获间隔 T_IN 大、运动检测滞后，而 IMU 可达 1000Hz。ECHO 用轻量循环神经网络（RNN）在两次 SLAM 位姿更新之间做高频插值：以当前 IMU 数据 + 最新 SLAM 优化位姿/速度/传感器偏置为输入，输出 100Hz 的 7D 位姿（3D 平移 + 4D 四元数）。为降低推理开销，权重做 per-channel INT4 量化、激活用 FP8，并用量化感知训练（QAT）保持精度。RNN 只做短期预测（被后续 SLAM 优化输出周期性纠正，不作为独立跟踪器），跨数据集泛化研究显示其训练集选择几乎不影响精度（Combined/AEA-only/TUM-only 训练 ATE 0.0332/0.0338/0.0337 vs 默认 0.0326）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 运行流程（IO 模式，Fig.11b）：①ECHO 加速器直接读 IMU 测量（Step 1）；②计算引擎跑量化 RNN：输入串联最新 IMU 读数（多步窗口）与 SLAM 提供的位姿/速度/偏置，RNN 单元（INT4 权重 × FP8 激活）在 8×8 weight-stationary 脉动阵列上执行矩阵乘，SFU 提供 ReLU/Tanh 非线性，输出 100Hz 7D 位姿（Step 2）；③位姿交 CPU 做声学注视聚类（Step 3）、GPU 渲染（Step 4）、DAC 输出（Step 5）。RNN 在 MI 帧之间以更短间隔产出位姿，使 T_IN 从相机周期（10Hz→100ms）降到 100Hz 的 10ms 期望（ECHO 建模取 T_IN=5ms）。训练：每数据集分别 QAT 训练（AEA 用 4 测试 + 10 训练/验证序列；TUM VI 每序列 20% 测试/80% 训练验证）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现/使用：QAT——训练时在前向传播中插入 INT4/FP8 量化器（直通估计器）让模型感知量化噪声，推理时用纯整数/混合精度计算；per-channel 权重量化把缩放因子按输出通道分组，比 per-tensor 精度高。硬件落地：权重离线量化后驻留片上 buffer，运行时载入脉动阵列；输入激活经 FP8 量化单元实时转换。使用收益：压缩 T_IN（motion-to-sound 延迟公式中的采样延迟），与 MI 模式交错（Hybrid mode）使整体 T_m-s=max(T_m-s^MI, T_m-s^IO)，且 RRE 比 ORB-SLAM3 更低（1.014° vs 1.194°，混合模式平均），方位误差小直接有利于声学注视聚类。注意点：RNN 位姿是短期插值，长期漂移靠 SLAM 纠正，不能独立长时跟踪。


涉及论文标题：
- ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

## 弹性推理（Elastic Inference）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 弹性推理是脉冲神经网络（SNN）独有的一种时序特性：由于 SNN 按离散时间步逐次积分-激发，输出（分类概率/检测框）随时间步推进逐步涌现、逐步收敛，因此对于"显著输入"（salient inputs），正确输出可比完整推理（跑满全部时间步）更早出现；给定足够推理时间，最终预测会收敛到与完整执行一致的结果。它对应生物神经系统的"显著刺激触发更快神经响应"现象。数学上，若神经元输出随累积输入单调收敛（如 ST-BIF 与量化 ReLU 等价），则早停输出是完整输出在时间上的前缀近似。ELSA 论文（ISCA 2026）把该特性视为 SNN 加速器此前未开发的关键机会：Fig.1 显示自主驾驶场景下首正确响应（FCR）可比稳定态输出早 82%。
- 别名/相关：early response、first-correct-response（FCR）、progressive inference、early exit（早停/提前退出是其在系统层的实现手段）。Web 证据显示同类工作包括 SIREN（熵基早退）、Elastic Spiking Transformers（NESTformer，运行时自适应弹性）、NEURAL（单时间步 + 早退的弹性神经形态架构）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 SNN 推理 pipeline 中，弹性推理 = 逐时间步评估 + 每步后的置信度检查。执行例子（ELSA 论文，ResNet50，T=32）：
```
for t = 1..T:
    V_t = V_{t-1} + Σ_i x_{i,t}·w_i          # 积分（spike × weight 加法）
    y_t = Θ(V_t, V_thr, S_t)                 # 激发（ST-BIF 三元 {-1,0,1}）
    V_t = V_t - y_t·V_thr;  S_t = S_{t-1}+y_t  # 膜更新 + tracer
    p_max = max(class_probs(t))              # 分类头置信度
    if p_max ≥ threshold: break              # 提前终止，输出当前预测
```
- 关键点：早停的粒度可以是整网（层间必须同步的 TBT/LBL 架构只能整网早停），也可以是 spine/token 粒度（ELSA 的细粒度流水使每个 spine/token 可独立早停）。ELSA 测量：置信度阈值 0.55 时平均延迟减 21.9%、精度损失 <0.2%；COCO2017 检测用 objectness score，sweet point 0.2 时 match 率 94.9%、延迟减 45.4%。
- Annotations：T 是最大时间步（ELSA 全部 benchmark 用 32）；V_thr 激发阈值；S_t 为 spike tracer（见独立术语）；threshold 是提前终止置信度阈值（分类取最大类概率、检测取 objectness）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：模型需在训练/转换时保证"时间累积输出收敛"，即用 ST-BIF 类神经元（与量化 ReLU 精确等价）构建 SNN，使每多跑一个时间步只是让量化输出更精细；推理时在分类头/检测头附加置信度函数（最大类概率 / objectness score）与阈值比较，超阈即停。ELSA 的 artifact（GitHub Intelligent-Computing-Research-Group/ELSA，ELSA_Algorithm 目录，PyTorch 2.4.1）直接输出逐时间步的 SNN 精度曲线与 FCR 延迟。系统/硬件侧：早停收益取决于执行模式——LBL 架构因层间全同步无法提前，TBT 只能整网早停，只有 spine/token 级细粒度流水（ELSA）能把早停粒度压到单个 spine/token，从而获得 Fig.20 中 2.0×（ViT-S）/2.4×（ResNet50）的同精度延迟缩短。Web 证据（SIREN）显示同类实现常用熵而非最大概率作为置信度，并加 patience 参数防止抖动误停。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## ST-BIF 神经元（Spiking-Trace Bipolar Integrate-and-Fire）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ST-BIF（带脉冲追踪的双极积分-激发）神经元是 SpikeZIP-TF 提出、ELSA 采纳的 SNN 神经元模型：与只发 {0,1} 二值脉冲的 IF 神经元不同，ST-BIF 发三元脉冲 {-1,0,1}（双极），并在神经元内维护一个"脉冲追踪器"（spike tracer）S_t 记录历史累计脉冲数。其关键在于：在特定条件下 ST-BIF 与量化 ReLU（Q-ReLU）数学等价，因此 ANN（量化）→SNN 转换无损，SNN 精度与 QANN 完全一致。ELSA 论文明确指出：转换损失是 IF 式 SNN 相对 ANN 精度下降的主因（conversion errors），ST-BIF 消除该损失。
- 三步动力学（ELSA 论文 Eq.1-3）：① 积分 V̂_t = V_{t-1} + Σ x_{i,t}·w_i（x 为 {-1,0,1} 预突触脉冲，w 为突触权重）；② 激发 y_t=Θ(V̂_t,V_thr,S_t)：V̂_t≥V_thr 且 S_t<S_max → +1；V̂_t<0 且 S_t>S_min → −1；否则 0；③ 更新 V_t=V̂_t−y_t·V_thr（soft reset）、S_t=S_{t-1}+y_t。S_max/S_min 是 tracer 上下界，对应 Q-ReLU 的 clip 上/下界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 作为 SNN 推理 pipeline 的最小计算单元，ST-BIF 把"乘法-激活"（QANN 的 MAC + ReLU）拆成"加法-脉冲-累积"：
```
# 一个输出神经元的单时间步（输入脉冲 x_i ∈{-1,0,1}，权重 w_i 4-bit）
acc = Σ_{i=1..N} x_i * w_i          # 硬件里 = 加法树（x=0 跳过，x=-1 权重取二补码）
V = V + acc
if V ≥ V_thr and S < S_max: y = +1
elif V < 0 and S > S_min: y = -1
else: y = 0
V = V - y*V_thr;  S = S + y          # 输出 y 成为下一层输入脉冲
```
- 例（ELSA Fig.3/Fig.10c）：输入 spike batch (0,1),(0,3) 触发读取权重矩阵第 2 行 [2,2,3,3] 与第 4 行 [1,3,1,1]，加法树累加得膜电位行 [3,5,4,4]，fire 组件结合 spike tracer 判定激发并回写膜与 tracer。
- Annotations：N 是每个输入 spine 的突触数；V_thr 阈值（如 8-bit 整数）；S_max/S_min 对应 Q-ReLU 的量化上下界；负脉冲 y=-1 使权重在硬件中按二补码取反后累加。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF（arXiv 2406.03470，GitHub Intelligent-Computing-Research-Group/SpikeZIP_transformer）在转换时把 QANN 的量化 ReLU 层一一替换为 ST-BIF 神经元，输出 S_T = clip(floor((Σ_t V_t^in + V_0)/V_thr), S_min, S_max)，实现 ANN↔SNN 精确等价；SpikingJelly 已集成 ann2snn.SpikeZIPTFQANNRecipe 路径。硬件侧：ELSA 把它实现为 ST-BIF 神经元电路 = 16 输入加法树 + fire 组件（读 spike tracer 与膜电位判激发）+ update 组件（回写膜与 tracer）；每 PE 128 个该电路、每周期 1024 次加法；路由器内的 SSoftmax/SLayerNorm 单元也复用少量 ST-BIF 电路。ELSA 全部 benchmark（VGG16/ResNet18-101/ViT-S/YOLOv2，4-bit 权重、T=32）精度与 QANN 一致（如 ResNet50 ImageNet 75.60%）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## Spike Tracer（脉冲追踪器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Spike tracer（脉冲追踪器）是 ST-BIF 神经元内部的记忆单元，记为 S_t：它记录该神经元到当前时间步为止累计发射的脉冲代数和（S_t = S_{t-1} + y_t，y_t∈{-1,0,1}），并受上下界 S_max/S_min 约束。其作用是让 ST-BIF 的"激发决策"带记忆性：当膜电位 V̂_t≥V_thr 但 tracer 已到上界 S_max 时不再发 +1 脉冲（防止输出越界），当 V̂_t<0 但 tracer 已到下界 S_min 时不再发 −1 脉冲。正是这个追踪器使 ST-BIF 输出精确等于 Q-ReLU 的 clip(floor(...), S_min, S_max)，从而保证 ANN→SNN 无损转换。
- 在 ELSA 中，spike tracer 与膜电位 V 一样作为神经元的持久状态，在硬件里各占一块 SRAM（每 PE 4×102.4 KB tracer buffer），每次激发读/写一行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Spike tracer 位于神经元"激发-更新"步：
```
Θ(V̂_t, V_thr, S_t):                     # 决策函数，tracer 参与判定
    if V̂_t ≥ V_thr and S_t < S_max:  return +1
    elif V̂_t < 0 and S_t > S_min:    return -1
    else:                              return 0
S_t = S_{t-1} + Θ(...)                 # tracer 更新 = 累计发射
```
- 例（ELSA 论文）：某神经元 S_max=+4、S_min=−4；若它已连续发 4 个 +1（S=4），第 5 次膜电位再超阈值时 Θ 返回 0（饱和），膜电位照常 soft reset 但不再发射，等效 Q-ReLU 的 clip 上界；只有 V̂_t<0 时才会转向 −1。这使得输出脉冲计数与量化激活的整数值一一对应。
- Annotations：S_max/S_min 是量化位宽决定的 clip 界（4-bit 权重对应 Q-ReLU 输出界）；tracer 是 1 个整数计数器而非多 bit 历史窗。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF 定义 ST-BIF 时引入 tracer，转换后的 SNN 每个神经元携带一个 S 计数器，推理全程累积；它是"有损"的只有一种情况——超过 clip 界的累加被截断（这与 QANN 的量化截断完全一致）。硬件侧：ELSA 的 fire 组件以膜地址 x 读 spike tracer 行，与集成后的膜电位 V̂_t 一起送入决策逻辑；update 组件把 y_t 累加进 tracer 并回写 SRAM。ELSA 芯片中 tracer buffer 占每 PE 面积 17.49%、功耗 0.6%；Tab.IV 显示 tracer 存储是 PE 面积主导（93.97% 面积被 weight/membrane/tracer 三类 SRAM 占据）的原因之一。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## ANN-to-SNN 转换（ANN-to-SNN Conversion / SpikeZIP-TF）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ANN-to-SNN 转换（conversion）是把已训练的 ANN（或 QANN）权重/结构改写成等价的 SNN，使 SNN 无需直接训练即可获得高精度，是 SNN 落地的两条路线之一（另一条是 surrogate-gradient 直接训练）。传统转换（IF 神经元 + 发放率编码）存在转换误差（conversion error），SNN 精度低于 ANN。SpikeZIP-TF（arXiv 2406.03470）提出用 ST-BIF 神经元（与 Q-ReLU 数学等价）实现"无损转换"：转换后 SNN 与 QANN 精度完全一致（ELSA 论文 Tab.VII：QANN 与 SNN 精度逐项相等）。ELSA 的全部 benchmark SNN 均按 SpikeZIP-TF 生成。
- 关键思想：QANN 的每个算子（量化卷积/线性、量化 ReLU、softmax、LayerNorm、残差加）都有对应的 spike 版本（MM-sc/MM-ss、ST-BIF、ssoftmax、slayernorm、残差加），把算子逐一替换即得 SNN，且因 ST-BIF=Q-ReLU，数值等价。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 转换 pipeline（QANN → SNN，ELSA 采用 SpikeZIP-TF）：
```
1) 训练 QANN（4-bit 权重/激活，量化 ReLU 与 S_min/S_max 对齐）
2) 逐层替换算子：
   QConv/QLinear  → MM-sc（spike-continuous 矩阵乘：输入为脉冲、权重为连续值）
   QAttention(QK^T, AV) → MM-ss（spike-spike：把 spike tracer 当连续操作数，
                          由两个 MM-sc 实现，SpikeZIP-TF 的做法）
   Quantized ReLU  → ST-BIF 神经元（含 spike tracer，clip 界 = Q-ReLU 界）
   Softmax        → ssoftmax（整数版，输出仍为脉冲）
   LayerNorm      → slayernorm（整数版）
   im2col / residual add → router 侧广播实现
3) 推理：时间步 t=1..T 内按脉冲形式执行，最终输出 = clip(floor(Σ V_t/V_thr), S_min, S_max)
```
- 例（ELSA Tab.VII）：ResNet50 QANN=75.60% → SNN=75.60%（无损）；ViT-S QANN=79.07% → SNN=79.07%。检测侧 YOLOv2（ResNet34 backbone）同理。
- Annotations：MM-sc 是输入脉冲 × 权重（连续值）矩阵乘；MM-ss 是脉冲 × 脉冲（需把 tracer 视作连续数，故实现为两次 MM-sc）；T 为时间步（ELSA 用 32）；tracer 的 clip 界 S_min/S_max 必须与 QANN 量化器一致才能无损。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF 官方实现 GitHub Intelligent-Computing-Research-Group/SpikeZIP_transformer，SpikingJelly 已集成 ann2snn.SpikeZIPTFQANNRecipe；转换后 SNN 可直接在 SNN 框架/加速器上运行。ELSA 的 artifact（ELSA_Algorithm 目录，PyTorch 2.4.1）包含转换与逐时间步精度评估脚本。硬件侧：转换决定加速器必须支持的算子集——ELSA 的 Tab.I 列出 MM-sc（CNN/Transformer）、MM-ss（Transformer 注意力）、ssoftmax/slayernorm/残差加/im2col；这些算子分别映射到 PE（MM）与 router（ssoftmax/slayernorm/im2col/残差加广播）。转换还决定了"弹性推理可用"：因为 ST-BIF 输出逐时间步向 Q-ReLU 结果收敛，早停不会造成本质精度损失（ELSA 早停平均 21.9% 延迟缩减、<0.2% 精度损失）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## LBL / TBT 执行模式与时间步（Layer-by-Layer / Time-step-by-Time-step / Time-step）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SNN 推理有三个固有维度：时间步 T（synaptic transmission 发生、神经元积分激发一次的单位区间）、层 L、层内 spine/token 数 N。既有 SNN 加速器按遍历这三个维度的顺序分为两类执行模式：LBL（layer-by-layer）——先跑完某层全部 T×N 计算再进下一层，输出只在整网完成后出现，与弹性推理天然不兼容（代表：C-DNN、SpinalFlow、SASAP、Prosperity、Phi）；TBT（time-step-by-time-step）——每个时间步评估所有层，输出逐时间步涌现，可支持弹性推理（代表：TrueNorth、Darwin、MorphIC、PAICORE）。TBT 的问题在于其层间流水是粗粒度层级的：必须等一层全部 N 个 spine/token 缓冲并同步后才前传，完成的个体不能立即转发，因此首响应延迟仍被层内同步拖到 O(L×N)。
- ELSA 论文把时间步定义为"synaptic transmissions occur and neurons integrate inputs and generate spikes once"的离散区间；把 spine（CNN 的 Z^{1×1×C}）与 token（Transformer 的 Z^{1×D}）定义为流水粒度单位（Fig.4）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 两种执行模式的遍历顺序（T=时间步、L=层、N=spine/token）：
```
LBL:  for layer l in 1..L:            # 层外循环
        for t in 1..T:                 # 层内先跑满所有时间步
          for n in 1..N: compute(l, t, n)
      输出只在整网完成后可用（无弹性）
TBT:  for t in 1..T:                  # 时间步外循环
        for layer l in 1..L:
          for n in 1..N: compute(l, t, n)
          barrier: 等本层 N 个 spine 全部完成才进下一层   # 粗粒度层同步
      输出逐时间步涌现，但首响应被层内 barrier 延迟
ELSA: for t in 1..T:
        for l in 1..L:
          for n in 1..N:
            compute(l, t, n)  # spine/token 完成后立即前传下一层，无 barrier
```
- 例（ELSA Fig.5）：L=74、N=197 的 Spikeformer 中，TBT 的层内 barrier 使每个时间步都要等 197 个 token 集齐；ELSA 的 spine/token 级流水把"整层同步"替换为"每 token 完成后立即流入下一层"，首响应从 O(L×N) 降为 O(L)。Tab.XI 对比：Loihi/SpiNNaker/PAICORE 是 spike 级计算+层同步，ELSA 是 spine/token 级计算+PE 级时间推进。
- Annotations：barrier 指层内全 spine 完成才前传的同步点；ELSA 在每 PE 内独立推进时间步（PE-level time advance），是粒度最细的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- LBL 实现：加速器跑满每层所有时间步，膜电位可在层间丢弃（C-DNN 因 LBL 而避免存膜，能效高但无弹性）。TBT 实现：每核存全部权重+膜+tracer（SRAM-only，如 TrueNorth 4096 核 + 1kHz 全局 tick、PAICORE 1024 核），层间用全局同步。ELSA 实现：6×6 神经核 + 2D-mesh NoC，每核 4 PE×128 ST-BIF 电路，spine/token 完成即由 Output Scheduler 调度前传，FIFO Queue 作核间 pipeline register。指标影响：Tab.IV 中 ELSA 以 spine/token 调度取得 4135.4 GOPS / 25.55 TOPS/W，相对 PAICORE 的 1.65× 速度增益即来自流水粒度（Fig.16/图 22-B 消融显示 spine/token 流水在 ResNet50 提速 6.7×、ViT-S 提速 15.2×）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## 置信度早停（Confidence-based Early Termination）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 置信度早停是弹性推理在系统层的实现机制：在推理的每个时间步（或每个 spine/token 完成后）计算一个置信度分数，一旦超过预设阈值就提前终止剩余时间步，用当前输出代替完整推理输出。ELSA 的早停策略：分类任务用最大类概率（max class probability）作置信度，检测任务用检测器的 objectness score（如 YOLO 的输出）作置信度；阈值可选"保守"（保持精度）或"激进"（更大延迟缩减、轻微精度损失）。
- 它依赖一个前提：SNN 输出随时间逐步收敛（ST-BIF 等价 Q-ReLU），因此早期输出是完整输出的高质量前缀，可无损/低损提前返回。ELSA 是首个把早停粒度做到 spine/token 级的加速器——配合细粒度流水，每个 spine/token 可独立早停（Fig.20），而非像粗粒度 TBT 那样整网早停。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 早停控制流（ELSA 论文实验设置）：
```
# 分类（confidence = max class probability）
for t in 1..T:
    logits_t = 前向推理至时间步 t（累计膜/脉冲）
    p_max = max(softmax(logits_t))
    if p_max >= τ: return 输出_t          # τ=0.55：平均减 21.9% 延迟、<0.2% 精度损失
# 检测（confidence = objectness score）
for t in 1..T:
    obj = detector.objectness(t)           # YOLO 的框目标度
    if obj >= τ: return 检测结果_t         # τ=0.2：match 率 94.9%、延迟减 45.4%（1.83×）
```
- 例（ELSA Fig.1/Fig.18）：COCO2017 YOLOv2 上，FCR 最早 1.19 ms、相对完整推理 2.76×；显著目标（框面积比 0.05→0.85 on VOC2007）延迟从 2.38ms 降到 1.88ms——显著输入更快响应正是弹性推理的价值。
- Annotations：τ 是置信度阈值；分类 max class prob、检测 objectness；mismatch 定义为早停检测与最终检测类相同且 IoU>0.5；阈值越高延迟越长、mismatch 越低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：ELSA_Algorithm（PyTorch）逐时间步记录精度与置信度，产出 Fig.1b 的 accuracy-vs-latency 曲线与早停缩减表（Tab.VII）。硬件侧：ELSA 的 spine/token 流水使每个 spine/token 可独立退出（置信度高即停止该 spine 的后续时间步），ELSA Output Scheduler 调度退出事件；Fig.21 显示即使在 NoC 拥塞（注入率>0.04）下，早停带来的 cycle 缩减仍稳定 >19%。指标：Tab.VII 中 ResNet18/34/50、ViT-S 早停延迟缩减 16.6%~26.1%（保守）/ 19.3%~39.1%（激进），精度损失 <0.2%/<3.3%。Web 证据（SIREN）显示同类工作可用熵或 patience 机制替代最大概率、防抖动误停。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## MM-sc / MM-ss（spike-continuous / spike-spike 矩阵乘法）与 SNN 非线性算子（ssoftmax / slayernorm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SNN 的矩阵乘法与 ANN 不同：MM-sc（spike-continuous）是一个操作数为脉冲（{-1,0,1}）、另一个为连续值（权重/膜）的矩阵乘，用于脉冲卷积与线性层；MM-ss（spike-spike）是两个操作数都是脉冲的矩阵乘，用于脉冲注意力（QK^T 与 AV 的 spike 版本）。由于两个脉冲操作数直接相乘大多是 0（稀疏），ELSA/SpikeZIP-TF 按 SpikeZIP-TF 的做法把 MM-ss 用两个 MM-sc 实现：把 spike tracer 当作连续操作数参与计算。除 MM 外，SNN 还需要杂项算子：ssoftmax（spiking softmax）、slayernorm（spiking layer normalization）、残差加、im2col，均来自 SpikeZIP-TF 的整数实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 算子在推理 pipeline 中的计算：
```
MM-sc:  out[x,:] += Σ_{spike (x,y,q)} (q? -1:1) * W[y,:]   # 每 spike 触发一行权重累加
MM-ss:  用 tracer 当连续数：先算 Q_trace · K^T 得注意力分数（MM-sc），
        再对分数做 ssoftmax，最后 score · V_trace（MM-sc）
ssoftmax: 整数指数/求和近似 softmax，输出仍为脉冲
slayernorm: 整数均值/方差归一化，输出脉冲
im2col: 卷积输入按核窗口展开（router 侧广播）
```
- 例（ELSA Fig.10c）：MM-sc 中 spike batch (0,1),(0,3) → 读 W 第 2、4 行 → 加法树累加出膜行 [3,5,4,4]；MM-ss 在 ViT-S 的注意力中占主导（Tab.II 中 ViT-S 的 #Sops 90.74G 远大于 #Ops 8.50G，即 spike-spike 计算量的放大）。
- Annotations：q 为脉冲极性位（q=1 负、q=0 正）；负 spike 时权重行取二补码；MM-ss 的"tracer 作连续数"是 SpikeZIP-TF 的关键技巧（论文脚注 4）；ssoftmax 中生成单个 token 需要全部 query/key token 就绪，故 ELSA 在该处停顿流水（token-wise pipeline 的 stall 点）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF（arXiv 2406.03470）定义这些算子并保证数值等价于 QANN 的对应算子；SpikingJelly 的 SpikeZIPTFQANNRecipe 提供实现。硬件侧：ELSA 的 Tab.I 列出支持的算子集——PE 执行 MM-sc（mini-batch Gustavson 数据流）；路由器内置 SSoftmax Unit 与 SLayerNorm Unit（各含少量 ST-BIF 电路与 tracer/膜存储，占 ELSA 面积 6.72%）执行 ssoftmax/slayernorm；im2col 与残差加在路由器侧以广播实现。Tab.III 显示 SSoftmax/SLayerNorm 单元占 ELSA 面积 3.45%/3.27%。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## 无损浮点压缩与指数-尾数分离（Lossless Floating-Point Compression / Exponent-Mantissa Separation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
对 IEEE 浮点数（FP32/FP16/BF16）做无损压缩：任何位模式都可无损还原。通用浮点压缩分两类：一类利用空间冗余（LZ 系：LZW/LZ78/LZ77、Zstd/nvCOMP），另一类利用符号频率（熵编码：Huffman、算术编码、ANS）。针对 AI 模型权重（ZipNN、DFloat11、Huff-LLM、DietGPU、ENEC）的关键观察：浮点权重的符号位与尾数（mantissa/fraction）近似均匀分布（高熵、不可压），而指数（exponent）高度偏斜（低熵、可压）——BF16 用 1 符号位 + 8 指数位 + 7 尾数位，分析显示符号/尾数熵约 7.97 bits、指数熵仅约 2.58 bits。因此"指数-尾数分离"成为模型权重无损压缩的通用范式：把权重拆成 {指数 E, 符号 S, 尾数 M}，S/M 直接存储（或不压），只对 E 做统计/熵编码，实现整体 ~1.3-1.5× 的压缩比（BF16）而保证 bit-identical 重建。ENEC 论文还给出两条补充数据观察支撑该范式：①指数取值高度受限、集中在一个窄连续区间（Observation 3）；②指数值与其频率排名呈负线性关系（Observation 5，可拟合 Y=-1.00X+123.00），这是线性映射能替代查表的统计基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ENEC 的指数-尾数分离 pipeline（BF16，8192 元素块）：
```
# 压缩
{E, S, M} = Split(W)                  # BF16: 1bit 符号 + 8bit 指数 + 7bit 尾数
S, M → 直接写入压缩流                    # 不可压部分原样存储
E' = T_freq[E] 或 y = (2^n - x + b) % 2^n   # 频率映射 / 线性变换（ENEC 用后者）
分组位宽阈值 + 分层对半打包 → 压缩流       # 定长编码，见"位宽量化与分层对半位打包"条目
# 解压
还原 E' → 逆变换 x = (y + b - 2^n) % 2^n → E
W = Combine(E, S, M)                    # 位级重组，bit-identical
```
Annotations：S/M 占比大（BF16 中 8/16=50%）但不压，因为均匀分布压不动；指数只占一半但贡献全部压缩率。DietGPU 的 Diet_Float、ZipNN、DFloat11 都走该范式但指数编码用变长（ANS/Huffman），ENEC 改用定长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：①直接按位拆分（ENEC/DietGPU-Float）；②ZipNN 的 byte grouping 把尾数分流进一步找模式；③Huff-LLM 做端到端编码。使用：LLM 权重存储/传输/推理部署（减少存储、网络与 CPU-NPU 搬运）；ENEC 在 Ascend 上以此消除权重传输瓶颈（端到端 TTFT 最高 6.3× 提速）；配合压缩权重执行流（decompress+execute）使用。局限：FP16 只有 5 位指数、压缩空间小（ENEC FP16 CR≈1.09-1.12）；尾数占比高时整体压缩比天花板受限。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## 位宽量化与分层对半位打包（Bit-Width Quantization with Hierarchical Halving Bit-Packing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ENEC 提出的无损定长位打包技术，把"每组按最大值计算可变位宽 + 乘除运算"替换为"两级位宽量化 + 纯位运算的 lane folding"。核心思想：(1) 位宽量化——数据块按组长度 L 分组（组内交错 scheme），若组内最大值所需位宽 ≤ 阈值 m，整组用 m 位存储；否则整组用 n 位（n 为表示所有出现指数所需的最小位数），用 1-bit bit mask 区分两种组。这用高效 bitwise OR 替代了计算昂贵的 reduction max 和乘法/除法（Ascend AIV 整数算术指令受限）。(2) 分层对半打包（hierarchical halving bit-packing，Algorithm 2）——N 元素 a-bit 数据（N=2^k, 0<a≤8）：迭代把数据块"对半折叠"，下半元素 data[i] 与上半元素 data[i+length] 左移 width 后 OR 合并进同一 lane，width 翻倍；当有效位宽超过 8 位字节边界时触发 byte 归一化——低 8 位拆出成可存字节，溢出位收集成新子块递归处理；最后补齐使总长偶对齐（16-bit aligned），再经一次折叠拼成输出流。效果：把变长打包需要的乘法/除法/规约全换成 OR、移位，压缩吞吐较基础版 +30%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 伪代码（N=8, a=3 举例，先演示 lane folding）：
```
data = [v0..v7], width=3, length=8, total=0
# 迭代1: length=4, data[i] |= data[i+4] << 3, width=6  # 两元素并到一 lane，6bit
# 迭代2: length=2, data[i] |= data[i+2] << 6, width=12 # 4元素并到一 lane，12bit>8 → 字节归一化
#   temp_bytes[j] = data[j] & 0xFF  (低8bit→字节)
#   data[j] >>= 8                    (溢出位留待下一轮，width=12-8=4)
# 迭代3: 对剩余溢出位继续 folding → 输出更多字节
# 末尾: total_length 补齐偶数 → 折叠拼接成 16-bit 对齐输出流
```
Annotations：位宽量化环节先保证每元素只需 ≤n 位；打包环节用 OR+shift 做"位平面压缩"（把多个窄值并进宽 lane），一次处理 2 的幂个元素天然对齐 SIMD。解压是精确逆过程（逆 gather + OR 还原）。组内超过 m 位的元素其高 (n-m) 位单独收集到 32KB buffer，满后同样打包，解压时逆 gather 放回原位。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AscendC 向量指令（元素级 OR、移位）在 AIV 上执行；GPU 移植版用 shuffle/移位指令优化 lane folding。参数 (m, L) 由离线联合搜索确定：B_exp = 1/L + n + (m-n)·p(m)^L 最小化（p(m) 为值可用 ≤m 位表示的概率，1/L 为组 mask 均摊开销；L≥16 因 Ascend 数据搬运 32 字节对齐，论文实测 L=16 最优）。使用：作为 ENEC 压缩/解压 kernel 的核心打包原语；也适用于任何"窄整数数组定长打包"场景。特点：定长 → 解压无需变长解析，天然适合无分支 SIMD。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## 向量化无分支整数变换（Vectorized Branch-Free Integer Transformation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ENEC 用于替代"指数频率映射 gather 查表"的线性变换：依据指数值-频率排名负线性关系（Observation 5），用 f(x) = b - x 把指数映射到更小的数值（高频指数→小值→更少位数）。由于 b-x 可能为负，利用二补数性质：额外 1 位符号位（n+1 位）区分正负，映射后取值范围 [0, 2^(n+1)-1]（正数 [0,2^n-1] 高位为 0，负数 [2^n,2^(n+1)-1] 高位为 1）。实现分三阶段：①向量加法单元同时减去参数 b（分布中心移到 0 附近）；②向量乘法单元乘 -1 得到 b-x；③移位实现模 2^(n+1)，把负数 -c 环绕成 2^(n+1)-c（如 n=5 时 -2 → 62）。全部是加、乘、移位的向量逐元素运算，无分支、无查表、无双射破坏（injective），把不规则访存（gather 占压缩 35%/解压 45% 开销）换成纯算术，吞吐近翻倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 参数搜索 (V-E 节)：对候选 b 计算 n = max(⌊log2(b-l)⌋+1, ⌈log2(h-b)⌉)+1（l,h 为指数最小/最大值）
# 变换：y = (2^n - x + b) % 2^n   （即 (b-x) mod 2^n，n 位 + 符号位共 n+1 位）
# 示例 (b=123, n=5)：x=125 → b-x=-2 → y=2^6-2=62；x=122 → b-x=1 → y=1
# 逆变换：x = (y + b - 2^n) % 2^n
```
Annotations：b 参数接近指数众数（BF16 模型 b≈121-123），使映射后高频值集中在 0 附近、位宽小；n 由指数范围跨度决定（论文中 BF16/FP32 都收敛到 n=6）。与查表映射 T_freq[E] 的区别：查表是 gather（内存随机访问），线性变换是算术（向量单元流水化）。这正是"以计算换访存"的典型硬件友好设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Ascend AIV 的向量加法/乘法/移位指令，C++（AscendC）逐元素表达；GPU 版同样用向量指令。参数 b 在离线三阶段搜索中按式 D = Σ_x p(x)·y 最小化选取（概率加权变换值之和最小）。使用：任何"符号集频率随值单调分布"的熵压缩预处理——把非均匀符号表变成可定长打包的小整数；ENEC 用它替换频率映射表，压缩端和解压端都受益（解压逆变换同样无分支）。局限：依赖指数分布的强线性（Observation 5），若分布偏离线性则压缩率略降（V2 相比 V0 压缩比小幅下降即是此代价）。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## HCLOG（分组位打包压缩组件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LC framework（github.com/burtscher/LC-framework）中的压缩组件之一，属于"分组位打包"类：把 16 KB 数据块划分为固定数量的 32 个子块（sub-chunk），对每个子块计算其最小值前导零个数作为元数据，只存储各元素的有效位（低 bit），从而利用数据中高位全 0 的冗余。LC 框架是跨平台数据压缩工具库，提供多种压缩组件与预处理方法，其中 Reducer 是唯一用于缩短数据序列的组件（含 HCLOG、RLE、RRE、RZE 等）。ENEC 论文用改进的 LC 框架对模型权重做组件组合搜索（Observation 2：HCLOG 变体在多数模型上取得最高压缩比 98%+ 的胜出率），并扩展了 LC 框架——支持不同子块数量的一组 HCLOG 压缩器变体（因为单个 outlier 会迫使整个子块采用更高位宽，可调子块数可缓解）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# HCLOG 分组位打包（LC 框架内）
for sub_chunk in split(block_16KB, 32):
    leading_zeros = min_lz(sub_chunk)        # 子块内元素最小前导零数 = 元数据
    for v in sub_chunk:
        store_low_bits(v, width=16 - leading_zeros)  # 只存有效低位
```
Annotations：该思路与 ENEC 的分组位宽打包同源——都是"按组算一个公共位宽、只存有效位"；ENEC 进一步改成阈值量化（≤m 用 m、否则 n）+ 纯 OR/shift 的 lane folding，去掉乘除/规约。LC 搜索的价值：用少量代表性模型（deepseek-llm-7b-base、Llama-3.1-8B）离线确定哪种轻量压缩组件组合在模型权重上最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LC 框架的 C++ 压缩组件库；ENEC 论文在 Section II-C 对其扩展（可变子块数 HCLOG 变体）并用它做最优组件搜索。使用：作为压缩组件组合搜索的候选集之一——论文观察到 HCLOG 在模型权重上几乎总是最优（98%+ 情况），这直接启发了 ENEC 采用分组位宽打包路线。局限：经典 HCLOG 的 reduction max 在 Ascend 向量单元上开销大（ENEC 分析占 40%），且单 outlier 拖累整子块位宽——ENEC 用两级阈值量化规避。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## ZipNN（AI 模型无损压缩库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IBM Research（与 BU/Dartmouth/MIT/Tel Aviv 合作）开源的面向神经网络模型的无损压缩库（CLOUD 2025，arXiv:2411.05239，github.com/zipnn/zipnn）。关键发现：浮点权重的符号位与分数位看似随机，但指数高度偏斜——256 个可能指数值中仅约 12 个出现 99.9% 的时间。做法：分离指数并用 Huffman（zstd 内置）熵编码，另加 "byte grouping"（字节分组）把分数位分流找更多模式；基于 Zstd v1.5.6，约 2000 行 C + 4000 行 Python。效果：BF16 模型（Llama/Granite/Mistral）约 33% 体积缩减（比 Zstd 好 11%），部分"干净"模型超 50%；解压吞吐约 80 GB/s、压缩约 13 GB/s（多线程）；完全无损、可集成 Hugging Face Transformers（zipnn_hf()），支持 delta 压缩与检查点版本管理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ZipNN pipeline（BF16 权重文件）
streams = split_bytes(w):  # 按字节位平面拆分：符号位流 + 指数字节流 + 尾数高位/低位流
exponent_stream → Huffman 熵编码          # 指数偏斜 → 大压缩率
fraction_streams → byte grouping 找模式后编码
拼接各流头部(元数据) → 压缩文件
```
Annotations：与 ENEC 的指数-尾数分离思路一致（同为"指数可压、尾数难压"观察），区别在编码器——ZipNN 用 Huffman/变长（CPU 友好）、ENEC 用定长+线性变换（Ascend SIMD 友好）。ENEC 论文把 ZipNN 列为 CPU 侧主要 baseline：BF16 压缩比 1.50-1.51（高于 ENEC 的 1.35-1.37，因变长编码更接近熵），但 CPU 吞吐仅 0.4 GB/s 级，ENEC 在 NPU 上压缩吞吐为其 987×（BF16）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 Zstd 库的 Huffman + 字节分组，chunk 级并行；提供 pip install zipnn、zipnn_hf() 无缝接入 HuggingFace；支持 torch 权重文件与 checkpoint delta。使用：模型存储/下载/分发压缩（HuggingFace 存储后端已采用其 byte grouping 技术，报告约 20% 存储节省）、训练检查点管理；ENEC 将其作为 CPU 端压缩比/吞吐对比基线。局限：CPU 吞吐远低于 GPU/NPU 专用实现，且变长解码需要分支，不适合 Ascend 这类无分支 SIMD 加速器直接落地。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## HANS（Ascend NPU 无损压缩算子）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
华为为 Ascend NPU 开发的无损压缩算法/算子（ENEC 论文称之为 HANS，闭源、仅提供 Python API 压缩张量；gitee.com/ascend/op-plugin 有相关 PR）。联网证据显示 CANN ops-math 仓库含 hans_encode 算子（aclnnHansEncode 接口）：对张量做指数字节的 PDF（概率分布）统计，按 PDF 分布做无损压缩，结果存 device 内存或卸载到 host 侧；输入支持 FLOAT16/BFLOAT16/FLOAT32（ND 格式，元素数需为 64 的倍数且 ≥32768），输出 PDF 分布 (1,256) INT32、尾数部分、定长压缩部分（fixed）与变长压缩部分（var）；支持 Atlas A2/A3 系列。ENEC 论文的评价：HANS 在压缩比与吞吐上都有限，且闭源（ENEC 是首个在 Ascend 上开源且达到 SOTA GPU 压缩器性能的无损压缩器）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# HANS encode（据 CANN ops-math 文档与论文描述）
pdf = PDF_stats(exponent_byte_of(tensor))     # 指数字节概率分布 (1,256)
fixed, var, mantissa = encode(tensor, pdf)    # 指数分定长+变长两段压缩，尾数单独输出
```
Annotations：HANS 与 ENEC 同为"按指数统计做无损压缩"路线，但 HANS 采用定长+变长混合结构（含变长部分 → 需要不规则访存/分支，可能限制其吞吐）；ENEC 全定长。ENEC 论文中的 NPU 基线对比（910B2）：ENEC 压缩吞吐为 HANS 的 1.36×（BF16）到 2.47×（FP32），解压 2.11×；压缩比两者接近（BF16 1.35-1.37 vs 1.33-1.35）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CANN 算子（hans_encode/hans_decode），aclnn 接口调用，PyTorch 侧经 op-plugin 暴露 Python API（ENEC 论文用它做张量级压缩测试）。使用：Ascend 上模型权重无损压缩；由于只有 Python API，论文用 msprof 在 kernel 级测其性能。局限：闭源、不能修改/定制；定长+变长混合在 Ascend 上吞吐受限；ENEC 定位为它的开源替代（性能相当或更优）。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## Vector Quantization（VQ，向量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VQ 是非解析量化（non-analytic quantization）的一种：把权重矩阵 W∈R^{K×N} 沿 K 维每 d 个连续元素分为一组 d 维向量，用 k-means 等聚类把所有权重向量映射到共享码本（weight codebook, WC）B∈R^{d×2^n} 的 2^n 个 centroid（n 为索引位宽），权重矩阵被替换为低精度索引矩阵（weight index, WI）I∈[0,2^n)^{K/d×N}。与解析量化（AWQ/GPTQ 的线性缩放+取整，可闭式表达）不同，VQ 的量化函数没有算术闭式，重建靠 1-to-1 查表。EVA 采用 d=8、n=8（码本 2^8=256 条目），单码本平均每元素 n/d=1 bit；用 C 个码本叠加（加法 VQ）达到有效精度 q=C·n/d bits（C=2/3/4 → 2/3/4-bit）。VQ 在 2-bit 级仍保持高保真（解析法在此崩坏），但 decode 时 1-to-many 查表导致不规则访存与 bank 冲突，且比 FP16 推理更慢（VQ-LLM 观察）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线量化（k-means / AQLM）：W ∈ R^{K×N} → 每 d 元素一组 → 聚类得 B ∈ R^{d×2^n}、I ∈ [0,2^n)^{K/d×N}
# 在线 decode（常规 VQ）：y = x @ W_hat，W_hat[i,j] = B[:, I[i,j]]（逐元素查表重建后 GEMV）
# 压缩比：FP16 权重 K×N×2B → 索引 (K/d)×N×(n/8)B + 码本 d×2^n×2B
```
EVA 例子：LLaMA-2-7B FC 层 W∈R^{4096×4096}，d=8、n=8 → WI∈[0,256)^{512×4096}（1 字节/索引）+ WC∈R^{8×256}（FP16），单码本压缩 16×；AQLM-2×8（2 码本）平均 2-bit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AQLM（EVA 采用的算法，https://github.com/Vahe1994/AQLM）、GPTVQ、QuiP# 等 PyTorch 后训练量化框架，离线 k-means/残差量化学习码本；EVA 仓库（https://github.com/dbw6/Eva.git，MIT）的 algorithm/ 提供 eval_ppl.py/lmeval.py 复现精度（依赖 aqlm[gpu,cpu]>=1.1.6）。使用方式：作为 weight-only 压缩手段部署到内存受限场景；GPU 上常规实现因查表不规则常比 FP16 慢（VQ-LLM 用 hot/cold profiling 缓解）；EVA 用"码本×输入 GEMM + 输出码本查找"（VQ-GEMM）把查表变成规则访存，专门适配硬件加速器。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

## Weight-only Quantization（仅权重量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
weight-only 量化只压缩模型权重、保留激活高精度（FP16），是面向 LLM decode 阶段（内存受限、权重主导访存）的主流压缩范式（AWQ、GPTQ、SqueezeLLM）。由于 decode 每 token 只做小 GEMV、激活轻量而权重张量占主导内存流量，权重位宽降低几乎线性加速 decode（AWQ 论文观察）。解析型 weight-only 量化（INT4/INT8，线性缩放+取整）可原位逐元素反量化、无额外依赖；EVA 把这一趋势推到 2-bit 级：采用 VQ 型（非解析）weight-only 量化，精度保持优于解析法，但查表反量化引入访存冲突（EVA 的动机之一）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
decode 每步 y = xW：x∈R^{1×K}（FP16），W 以低比特存储。量化后内存流量 ∝ 权重比特数；AWQ/GPTQ 4-bit 时 decode 近线性提速，VQ 2-bit 再砍一半。EVA 精度对比（Table V，WikiText-2 ppl，L-2 7B）：EVA-A16W2（激活 FP16、权重 2-bit VQ）6.69 保持竞争性，而解析法 2-bit（FIGNA-INT2 AWQ）崩到 2.2e5——说明 2-bit 级只有非解析 VQ 能保精度；4-bit 时 EVA-A16W4（AQLM 4×8）5.43 优于 AWQ 4-bit 5.60。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AWQ（激活感知，按激活幅度选保护通道）、GPTQ（二阶 Hessian 逐层量化）、SqueezeLLM 等框架；EVA 采用 AQLM/GPTVQ 的 VQ 型 weight-only 方案。使用方式：配合 serving/硬件加速器使用；EVA 硬件为 FP16 激活 + 2/3/4-bit VQ 权重设计（A16W2/W3/W4），在 32×8 FP16 重配阵列上执行，同时保持 INT8 prefill 兼容（同一阵列重配）。

- M100 补充视角（ISCA'26，车规 LLM 推理的量化选择）：LLaMA2-7B decode 阶段采用 W4A16（4-bit 权重 + FP16 激活）以压缩权重访存（decode 为带宽受限、权重主导访存）：M100 21.34ms vs Thor-U 20ms（两平台 DDR 带宽同为 273 GB/s）；prefill 阶段改用 W8A8（权重+激活 8-bit INT）以利用整数张量计算（compute-bound）：M100 79ms vs 154ms（1.95×）。选择逻辑与常规认知一致：带宽受限阶段用 weight-only，计算密集阶段用 weight-activation 量化。
涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing

## Additive Vector Quantization（AQLM，加法向量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AQLM（Additive Quantization of Language Models，https://github.com/Vahe1994/AQLM）把单个权重向量表示为 C 个码本条目的和（每个码本贡献一个 centroid），即 additive/残差式多层量化，在 2-bit 级达到 SOTA 精度-压缩权衡。EVA 以它作为默认 VQ 算法：C 个码本、每码本 2^n 条目、向量维 d，有效平均精度 q=C·n/d bits（EVA 支持 C=2/3/4 → 2/3/4-bit；Table III：AQLM 2×8=2bit、3×8=3bit、4×8=4bit）。AQLM 的"多码本求和"正好对应 EVA 硬件里 EU 的对角累加（across C0-C3 输出级并行归约）。对比：AQLM-1×16（n=16、65536 条目、单码本）精度与效率都差于 AQLM-4×8（Table III：norm latency 22.86× vs 1.98×）——大码本引入 spurious 乘法且利用率低，验证 n=8 最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# AQLM 重构：w_hat = Σ_{c=1..C} B_c[:, i_c]   # 每个权重向量 = C 个 centroid 之和
# EVA 侧：对每个码本算 O_c = X·B_c（输出码本），最终输出 y = Σ_c Lookup(O_c, I_c)（EU 对角累加）
```
例子（AQLM-2×8，d=8,n=8,C=2,q=2bit）：W∈R^{4096×4096} → 2 组 (B_c∈R^{8×256}, I_c∈[0,256)^{512×4096})；decode 时 x∈R^{1×4096} reshape 为 X∈R^{512×8}，O_c=X·B_c∈R^{512×256}，y=Lookup(O_1,I_1)+Lookup(O_2,I_2)∈R^{1×4096}。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AQLM 官方 PyTorch/CUDA 库，离线学习码本（可配合 PV-Tuning 微调码本提升 2-bit 精度）；EVA 算法评估直接用 AQLM 预训练 checkpoint（Hugging Face dbw6/eva collection）跑 perplexity 与下游任务。使用方式：2-bit 级部署选择；MoE 模型上 AQLM-2×8 只掉 5.3pp（Mixtral-8x7B 下游平均），远优于 GPTQ（32.7pp）；EVA 架构与 AQLM 解耦，可换用 GPTVQ 等其他 VQ 算法（GPT-W2* 配置仍优于 FIGLUT 基线 1.15×/2.31×）。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

## LUT 查找表量化（非解析量化 / Codebook-based Lookup）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
非解析（non-analytic）量化去掉闭式量化函数，直接用 k-means 等最小化重建误差（MSE）学习映射，典型代表是查找表（LUT）量化：权重→码本索引，decode 时 1-to-1 查表重建（硬件加速器 GOBO、FIGLUT、LUT Tensor Core、LUT-DLA；软件框架 AQLM/QuiP#）。优点：表示灵活、2-bit 级精度仍高（Table VI：EVA 2-bit 下游平均比 LLM.265 VB 高 19pp）；缺点：不规则、非合并的查表访存导致 bank 冲突、并行化困难，且硬件需复制（duplication）或广播（broadcast）码本到多 PE——FIGLUT 广播 16×32×(8×16bit) 带宽、LUT-DLA 复制 256×(16×16bit) 寄存器、GOBO 复制 768×(8×16bit)，码本有效规模被限制在 ≤16 条目（Table I）。EVA 定位为第一个架构级 VQ/查表 LLM 推理加速器，核心洞察是把"查权重码本"变成"查输出码本"（见 VQ-GEMM 条目），从根上消除冲突且无需复制/广播。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 常规 LUT decode（FIGLUT 式）：激活作输入、权重模式作索引取预计算部分和
for each tile: partsum = LUT[act_pattern][weight_idx]   # 广播/复制到多 PE，同 bank 冲突时串行
# EVA 式：先 O = X·B（GEMM 产出全部"输入×centroid"点积的输出码本），再 y = Lookup(O, I)
```
查找次数减少、全部规则化、跨 bank 无冲突、每次访问带宽从 d 个 FP16 降为 1 个 FP16。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GOBO（寄存器复制）、LUT-DLA（复制）、FIGLUT（BCQ 二元编码 + 4-input LUT 广播）、LUT Tensor Core；软件侧 AQLM/QuiP#（PyTorch）。使用方式：用于低比特权重推理，尤其 2-bit 级；EVA 的对比基线 FIGLUT 在 32×32 阵列上 decode 利用率仅 4.34%、吞吐 44.49 GOPs（2.82× SA），而 EVA 以"输出码本查找"达 498.49 GOPs（31.64× SA）——查表对象（查 WC vs 查 OC）是硬件效率的分水岭。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

## VQ-GEMM 与 Output Codebook（码本驱动 GEMM 重构，GEMV→GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVA 的核心算法贡献：把 VQ decode 的 GEMV 重构为 GEMM。由于每个权重向量都来自码本，无需现场重建权重——直接把输入向量与权重码本做点积：输入 x∈R^{1×K} reshape 为 X∈R^{(K/d)×d}，与码本 B∈R^{d×2^n} 相乘得到输出码本（Output Codebook, OC）O∈R^{(K/d)×2^n}（每元素 = 一个输入向量与一个 centroid 的点积，跨输出通道 N 复用）；再用索引矩阵 I 从 OC 查找并累加得最终输出 y=Lookup(O,I)。收益：①计算量从 K×N 降到 K×2^n（N=4096、2^n=256 时约 16× 少）；②M 维从 1 扩到 V=K/d>512，填满矩阵单元（GEMV→GEMM）；③访存规则化/合并化；④带宽每访问从 d 个 FP16 降到 1 个；⑤查 OC 无 bank 冲突（OC 行与 WI 行共享高度 V=K/d，每行独立 bank，同列不同索引自动落不同 bank）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 常规 VQ decode：重建权重后 GEMV  y = x @ W_hat（K×N MAC，memory-bound）
# EVA VQ-GEMM decode：
X = x.reshape(K/d, d)                # x∈R^{1×K}
O = X @ B                            # O∈R^{K/d × 2^n}，输出码本（GEMM，K×2^n MAC）
y = Lookup(O, I).sum(codebooks)      # I∈[0,2^n)^{K/d×N}，冲突无关查找 + 加法树归约
```
LLaMA-2-7B FC 例子（d=8,n=8,C=2）：x∈R^{1×4096}→X∈R^{512×8}；O_c=X·B_c∈R^{512×256}（512×256×8≈1.05M MAC vs 常规 4096×4096≈16.8M）；y=Lookup(O_1,I_1)+Lookup(O_2,I_2)∈R^{1×4096}。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：需要硬件支持（EVA 的 32×8 FP16 重配阵列跑 VQ-GEMM + Epilogue Unit 跑查找/归约）；算法上依赖 AQLM/GPTVQ 等 VQ 码本。使用方式：decode 阶段替代 GEMV；PE:EU 计算比 = 2^n:N（Table III）——2^n<N 时 GEMM 非瓶颈、EU（加法树）为关键路径，可加 EU 数扩展（4 EU 匹配 64GB/s 带宽饱和，再增仅增能耗）；2^n>N 时出现 spurious 乘法（centroid 无输出通道引用，利用率下降）——因此 n=8/256 条目是 EVA 的实用折中。EVA-A16W2 单 batch decode 对 SA/ANT/FIGNA/FIGLUT 分别 31.56×/32.53×/33.50×/11.17× 加速（Fig. 10）。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

## CKKS（Cheon-Kim-Kim-Song 近似数同态加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CKKS（Cheon, Kim, Kim, Song，ASIACRYPT 2017）是面向近似实数的 Leveled 全同态加密方案，专为加密数值计算（尤其是神经网络推理）设计：它支持固定点数/浮点数编码、SIMD 风格的向量化并行、以及加/乘/旋转等近似运算，解密结果与明文计算相比仅含可容忍的近似误差。其核心机制：明文消息 m 被编码为次数 N 的多项式（环 R=Z[X]/(X^N+1)），密文是环上的两个多项式 (c_0, c_1)（外加一个 a），满足 c_0 + c_1·s ≈ m（s 为秘密钥）；实际数值被放入多项式的"槽（slot）"中，N 次多项式可承载 N/2 个复数槽，全部槽上的运算并行执行（SIMD）。
- 关键参数与符号（本论文 Table I）：N=多项式次数（coefficient 数），N/2=可用槽数；Δ（scale factor，本论文用 Δ=2^40 即 40 位）用于把实数定标为整数编码并控制精度；模数链 {q_0,...,q_L} 构成 level 预算。乘法后噪声与规模增长，需 Rescale（除以 2^Δ、截断模数）管理噪声、消耗一个 level；level 耗尽前必须 Bootstrapping 刷新噪声（深层 CNN 如 SqueezeNet/ResNet18/MobileNet 需要）。安全基于 Ring-LWE 困难问题，本论文所有模型保证 λ≥128 bits。
- 本论文中 CKKS 是 FEnc² 的底层方案：RNS-CKKS 实现（GPU 端用 Liberate-FHE），输入图像在客户端编码为多项式并加密，服务端在密文域执行全部 CNN 电路。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 CKKS 编码-加密-运算-解密 pipeline（以单张图像的一个卷积输出为例）：
```
1) encode:  m = [x_0, x_1, ..., x_{N/2-1}]   # N/2 个实数槽（本论文打包 4D 块元素）
            m_poly = CanonicalEmbedding(m) * Δ   # 定标到整数
2) encrypt: ct = (c_0, c_1) = (a·s + m_poly + e, -a)   # a 随机多项式，e 小噪声
3) compute: ct_out = PMult(ct, plaintext_weight)  # 密文×明文权重（卷积核）
            ct_out = Add(ct_out, ...)             # 累加
            ct_out = Rot(ct_out, k)               # 槽循环移位（对齐聚合）
            ct_out = Rescale(ct_out, Δ)           # 每次乘法后截位、耗 1 level
4) decrypt: m_out = (c_0 + c_1·s) / Δ             # 近似恢复结果
```
- Annotations：第 1 步的槽布局（哪些标量放哪个槽）即"密文打包"问题，决定后续旋转次数与槽利用率；Rot 是本论文主要优化目标（4.8ms vs PMult 0.15ms，Fig.1(a)）；Rescale 消耗 level，深层网络需在耗尽前插入 Bootstrapping（NEXUS GPU 自举每次耗 14 个 level）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件库：SEAL（微软，CKKS 原生实现）、HElib、OpenFHE、TenSEAL；GPU 库：Liberate-FHE（本论文 GPU 后端，纯 Python+CUDA 多 GPU，RNS-CKKS，BSD-3-Clause-Clear，现已弃用、继任 DESILO FHE https://fhe.desilo.dev/）、HEonGPU、TensorFHE、Cheddar。使用流程：设置参数（logN、logQ、scale、安全级别）→ 生成密钥（sk/pk/evk 旋转与重线性化密钥）→ 客户端编码加密 → 服务端同态计算 → 客户端解密。典型场景：加密 CNN/Transformer 推理、加密矩阵乘法、隐私梯度聚合。
- HE² 补充视角（ISCA'26，面向加速器的算子分类）：HE² 按算术强度（AI，ops/byte，SHARP 参数下）把 CKKS 算子分为两类：ComOps（计算密集型）= NTT（0.89）、BConv（1.60）、ModUp（3.38）、ModDown（2.92），复杂计算模式、由定制 ASIC 模块（xPU）加速；MemOps（内存密集型）= IP（0.12）、PMul（0.09）、CAdd（0.07）、Rescale（0.11），内存足迹大、由近存模块（xMU）加速。该分类是"ASIC-NMP 异构加速"的硬件分工依据（参数 N=2^16、L=35、L_eff=8、k=12、α=12、dnum=3、λ=128-bit，见本库"xPU-xMU 异构架构"与"ModUp/ModDown"条目）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption

## 密文打包（Ciphertext Packing）与密文槽（Slot / Slot Utilization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 密文打包是把多个明文标量（图像像素、通道值、batch 元素等）塞进单个 CKKS 密文的 N/2 个槽（slot）中，使一次同态运算（Add/PMult/Rot）同时作用于多个数据（SIMD），摊薄 HE 原语（NTT、旋转、keyswitch）的高额成本。槽利用率（slot utilization）= 有效数据槽数 / 总槽数，反映 SIMD 效率：利用率 50% 意味着密文数翻倍、内存与下游计算膨胀。
- 打包的核心矛盾（本论文核心动机）：卷积在密文域引入两类数据依赖，都靠旋转解决——(1) 空间依赖（intra-channel，同通道相邻像素）、(2) 通道依赖（cross-channel，多输入通道聚合）。不同打包布局决定：(a) 旋转次数（内旋转/外旋转数量）、(b) 槽利用率、以及 (c) 密文数量与内存占用。HE-CNN 中旋转贡献约 70% 端到端延迟（Fig.1(b)），因此打包布局是性能一级因素。
- 本论文指出现有打包方案的通用缺陷：静态启发式、逐层无关；跨层碎片化（1×1 缩减层后槽利用率掉到 12.5%-31.25%）；旋转优化不完整（只减内或只减外）。FEnc² 的 Conv-aware Encoding + AAC 正是对这三点的统一求解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文 Algorithm 1 的槽映射公式（Conv-aware Encoding 的核心，BS×C×H×W 输入按 S×S 块打包）：
```
输入 X ∈ R^{BS×C×H×W}，块大小 S，M=max(pad(H),pad(W))，m=M/S
对块内坐标 (u,v)，0≤u,v<S：
    X_{uv} ← 收集所有 m×m 块中的 (u,v) 元素 → R^{C×BS×m²}
    展平并映射：X_{ijk}^{(u,v)} → slot l
      其中 i = ⌈l/(BS·m²)⌉（通道）、j = l mod BS（batch）、
            k = ⌈l/BS⌉ mod m²（块内坐标）
加密得满装密文 ct_{uv}（共 S² 个密文）
```
- Annotations：S=1,BS=1 退化为 row-major/Orion 式布局（无外旋转、内旋转最大）；S=M 退化为 CryptoNets 像素式布局（无内旋转、外旋转最大）；中间 S 平衡两类旋转。打包密度决定后续层密文数：利用率 100% vs 25% 时同一层密文数差 4 倍（Fig.5 瓶颈 8→2→8 例子）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：客户端按框架返回的"索引-槽映射"做 CKKS 编码加密（本论文 FEnc² 只凭公开元数据 H,W,C,BS + 模型结构自动生成映射，无需运行期 profiling）；服务端按该布局执行同态电路。已有方案族：LoLa/CHET（row-major）、Gazelle/Fast-HEAR/Multiplexed/Orion/Hyena+（interleaving+BSGS 多通道）、HELayers（block tiling+多图打包）、Fhelipe（层后合并稀疏槽）、CryptoNets（pixel-wise）、FEnc²（统一 fragment 编码，包含前两者为特例）。用途：加密 CNN/Transformer 推理的输入编码与层间布局管理；衡量指标：旋转数、槽利用率、密文数、内存。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding

## Slot Rotation（槽旋转）与 Key-Switching（密钥切换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 槽旋转 Rot(ct,k) 是 CKKS 的 SIMD 原语：把密文 N/2 个槽的内容循环移位 k 位（等价于对消息多项式做自同态 X→X^k），用于对齐数据做聚合（如卷积的通道/空间累加）。旋转实现 = 自同态（automorphism） + 密钥切换（key-switching）两步：自同态把系数映射 X^i→X^{ik}，得到的是用"旋转后密钥"加密的密文，必须用旋转求值密钥 evk_rot^k 做 key-switching 换回原密钥才能继续运算；keyswitch 占据旋转延迟主体（本论文式 (1)：Rot(ct,k)=(c(X^{ik}),0)+P^{-1}(a(X^{ik})·evk_rot^k)）。
- 性能：旋转 + CMult 远贵于 PMult/Add（Fig.1(a)：4.8ms vs 0.15ms），每次旋转含多次 NTT/iNTT 与大规模向量 shuffle，应用级端到端延迟约 70% 来自旋转（Fig.1(b)）。因此"减少旋转次数"是 HE-CNN 打包优化的首要目标。
- 两类旋转（本论文定义）：内旋转（inner rotation）= 空间聚合，每输入密文需 (K²−1) 次生成 K×K 卷积的移位副本；外旋转（outer rotation）= 通道聚合，每个输出密文需 (α−1) 次（α=每密文打包通道数）对齐通道。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次多通道卷积的旋转流水（本论文 Fig.2，4 输入/输出通道、3×3 核、BSGS）：
```
# 内旋转：生成 K²=9 个移位副本做空间 MAC
for i,j in 0..K-1:
    X_shifted = Rot(ct_X, offset(i,j))        # 内旋转 ×(K²-1)
    acc += PMult(X_shifted, plaintext_w[i,j]) # 与明文核相乘
# 外旋转：跨 α 个通道对齐并累加
for c in 1..α-1:
    acc += Rot(acc_c, channel_offset(c))      # 外旋转 ×(α-1)
```
- Annotations：旋转次数被 FEnc² 的块大小 S 控制：内旋转复杂度在 K>S 时为 (⌈K/S⌉²−1)/密文、K≤S<M 时为 4(S−1)/S²；外旋转为 N_out/α×(αS²/BS−1)；最优 S* = ⌈(K²N_in/(αN_out))^(1/4)⌉（Theorem 1，式 (8)）。Conv-aware Encoding 使旋转复杂度从 O(K²)（乃至 Hyena+ 的 O(K⁴)）降到 O(K)（Table III）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：任何 CKKS 库（SEAL/OpenFHE/Liberate-FHE/TenSEAL）的 rotate 接口 + 预先生成的旋转密钥（每偏移一个 evk）。硬件/系统级：keyswitch 依赖 NTT/iNTT 在分解基间变换（本论文指出旋转是 NTT 单元的最大消耗者，减旋转即减 NTT 压力）；GPU 实现把旋转映射为 ciphertext 系数 shuffle + 多项式乘 kernel。使用场景：卷积/矩阵乘/FFT 在密文域的对齐聚合，以及密文内数据重排（如 FEnc² 的 rot-mask-add 重打包）。优化方向：预旋转副本复用、BSGS 拆分、减少旋转次数（本论文）、keyswitch 密钥复用（ARK）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding

## BSGS（Baby-Step-Giant-Step）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSGS（Baby-Step-Giant-Step）原是数论中解离散对数的时空权衡算法，在 HE 推理中被借用来分解多通道卷积的嵌套循环、降低旋转复杂度：把卷积按核/通道维度拆成"小步（baby step）"与"大步（giant step）"两组，用预旋转副本 + 分组聚合替代逐元素旋转，是 SOTA HE 推理系统（Orion、Multiplexed、HEAR 等）普遍采用的技巧（本论文 Sec. III-A 默认 HE 多通道卷积使用 BSGS）。
- 本质：把需要 K² 次内旋转的空间卷积，与需要 (α−1) 次外旋转的通道聚合解耦，避免把 K²×α 的旋转全部串行化；通过预先旋转并缓存输入密文的若干副本（giant step），后续只需较少组合（baby step 增量旋转）即可覆盖所有窗口/通道对齐，以内存换旋转次数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一个 3×3 核、α 通道打包的卷积 BSGS 分解示意：
```
# Giant step：预旋转输入副本（一次性代价）
for k in 0..g-1:  X_giant[k] = Rot(X, giant_offset(k))     # g 个大步旋转
# Baby step：对每个输出位置做小步增量
for b in 0..b-1:
    Y_b = Σ_k  PMult(X_giant[k], W_k)      # 组合大步副本
    Y_b = Σ_b'  Rot(Y_b, baby_offset(b'))   # 小步聚合
```
- Annotations：总旋转从 K²×α 量级降为 g+b 量级（g·b≈K²）；FEnc² 的 Conv-aware Encoding 与 BSGS 正交可叠加——块分解（fragment）进一步降低每密文需预旋转的副本数，二者共同把旋转复杂度压到 O(K)；本论文表 III 中 Orion/HELayers 的复杂度即含 BSGS 的基线，FEnc² 优于它们。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 HE 推理框架的卷积内核中，先做输入密文的多偏移预旋转并缓存（消耗内存），再按核权重做明文乘与累加，最后用少量旋转对齐输出。使用场景：任意 HE 多通道卷积/全连接层（对角线矩阵乘也可用 BSGS）；本论文 FC 层即"对角线矩阵乘法 + BSGS"。局限：预旋转副本占用内存、且打包布局差时收益有限——这正是 FEnc² 强调"布局决定 BSGS 效率"的原因。
- HE² 补充视角（ISCA'26，BSGS 与 hoisting 的硬件权衡）：在 EVF 单体 ASIC（SHARP）中 bs 与 gs 相等（均 8）时计算代价最低，但 baby-step 密文超片上容量，故取 bs=4；在 IRF 异构 + hoisting 场景下，bs 与 gs 差距更大反而暴露更多 keyswitch 并行、减少计算与通信，但增大 evk 存储需求（可能超 HBM 容量，Fig. 7）。BSGS 把一个并行度 D、入/出度为 1 的 PKB 拆成两个串行 PKB（PKB1 并行度 bs、PKB2 并行度 gs=D/bs），降低 keyswitch 并行度、提高子图出入度。HE² 的 HERO 框架按内存约束选择 BSGS 配置：8 GB HBM 足够时禁用 BSGS（C2S/S2C），内存受限时偏好 bs 与 gs 差距大的配置；BERT 首 FFT 阶段因高层级仍保留 bs=2/gs=32（见"PKB 与 PKB 融合"条目）。
- HyperDrive 补充视角（ISCA'26，BSGS PCMM 与密文复用 PMAC）：在 CKKS bootstrapping 的 CtS/StC 相位，BSGS 把明文-密文矩阵向量乘（PCMM）的旋转数从 O(N) 降到 O(√N)（bs×gs 个非零对角、bs+gs 量级旋转），配合 hoisting（复用 BS 相位的 ModUp 输出）与 double hoisting [7]（PMAC 中间量全程留在 R_{PQ_ℓ}，省掉大部分 ModDown）。BS 相位只做一次预旋转（ModUp）并在 BS-Rots 间复用，故 (NTT2-IP) 协同优化无处可用；GS 相位对密文 b(X) 做 ModDown 回 R_{Q_ℓ} 后接标准 KeySwitch。HyperDrive 的 CRPMAC 把全部 GS-Rots 推迟到末尾、按 GS 方向批处理 PMAC——单个 baby-step 密文乘以一批明文（一次 GMEM 读复用 gs 次），对比 [22] 按 BS 方向批处理的内存足迹更小；另对 bootstrapping 做操作重排：GS 相位 ModDown 紧跟 ModUp 时把 EWSub 提前，使 ModDown INTT 与 ModUp INTT 可批处理并合并进 (BConv2-NTT1) kernel。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## Conv-aware Encoding（卷积感知 Fragment 编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Conv-aware Encoding 是 FEnc² 的第一个组件：一种卷积感知的 fragment（块）编码布局，把 4D 特征张量（batch、channel、H、W）按 S×S 子块分解，每个块编码进独立密文，从而同时解耦相邻像素（空间依赖）与跨通道（通道依赖）两类卷积数据依赖。通过解析旋转代价模型选择最优块大小 S*，使内/外旋转总代价最小，无需运行期 profiling（Algorithm 1 + Theorem 1）。
- 关键性质：一般性（generality）——S=1,BS=1 时退化为 row-major/Orion 编码，S=M 时退化为 CryptoNets 像素式编码，故 FEnc² 统一包含先验方案为非最优特例；最优性（optimality）——内旋转项随 1/S² 递减、外旋转项随 S²α 递增（式 (3)-(5)），两者相等时总旋转最小（Cauchy-Schwarz），得 S* = ⌈(K²N_in/(αN_out))^(1/4)⌉（式 (8)），S 还需满足上界 S ≤ √(BS·N_in/α) 防止槽浪费（式 (7)）。
- 大 batch 特例：batch 足够大时不同样本可单独装满密文，外旋转完全消除（式 (6) Rot_amortized 只剩内旋转），与实验（Fig.7 大 batch 用大块收益更明显）一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 1×16×4×4 输入、卷积 (16,16,3,1)、BS=1、16 slots 为例（本论文 Fig.4）：
```
M=max(4,4)=4；选 S=2 → m=2（2×2 个 2×2 块）
对每个块内坐标 (u,v)∈{(0,0),(0,1),(1,0),(1,1)}：
    收集 4 个块中 (u,v) 位置的元素 → X_{uv}∈R^{16×1×4}
    按 Algorithm 1 槽映射展平 → 16 槽满装 → 加密 ct_{uv}
卷积时：K=3>S=2 → 每密文内旋转 (⌈3/2⌉²−1)=3 次
外旋转 N_out/α×(αS²/BS−1) 由 α 与 S 共同控制
S=1：内旋转最大、无外旋转（Orion/CHET 式）；S=4：无内旋转、外旋转最大（CryptoNets 式）
S=2：内/外平衡 → 总旋转最小（最优）
```
- Annotations：S 决定每密文装多少同通道相关像素（1/S²）与多少通道（αS²/BS）；S* 解析可算、无需搜索或 profiling；stride≥2 卷积/平均池化直接丢弃多余密文实现（免去先验方案的后处理重排）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：客户端初始化时只发非敏感元数据（H,W,C,BS,模型 id），FEnc² 依据张量形状与模型结构自动算出 S* 与层间块大小调整方案（rot-mask-add 重打包，开销仅 0.42%-3.7%），返回索引-槽映射给客户端编码加密；服务端照布局执行。与硬件无关（任何 CPU/GPU/FPGA/ASIC HE 执行平台通用）。效果：旋转数相对 HELayers 降 67%-94%，keyswitch 降 80%-93%，NTT/iNTT 降 89%-94%，密文数与同态乘法降 78%-94%。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding

## Arch-aware Ct Compression（AAC，架构感知密文压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AAC（Architecture-aware Ciphertext Compression）是 FEnc² 的第二个组件：一种跨层槽利用率优化机制，在通道/特征缩减层（如 MobileNet/SqueezeNet/ResNet 的 1×1 卷积把通道 N_in 压到 N_DS）之后，用轻量 rot-mask-add 序列把稀疏填充的密文压实成满装密文，恢复密文密度、减少后续层密文数，且不改变打包格式、自动适配所有中间形状。
- 动机：CKKS 对所有槽做 SIMD 运算，N_DS<α 时稀疏密文浪费计算并迫使后续层用更多密文覆盖 N_out 输出通道（Fig.5 瓶颈 8→2→8：无 AAC 需 4 个 25% 利用率的稀疏密文、4× HE 计算；有 AAC 只需 1 个满装密文）。Fhelipe 等虽做层后槽合并，但忽略下一层计算模式，布局次优。
- 关键设计：AAC 不增加乘法深度——其 0/1 明文掩码不要求比卷积权重更高的明文 scale（标准 CKKS 每 PMult 用统一 scale Δ，权重与掩码分别编码在 Δ₁、Δ₂ 且 Δ₁·Δ₂=Δ），两个连续乘法后只做一次 rescale，不额外消耗 level。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 瓶颈块 8→2→8 通道的 AAC 压实流程（本论文 Fig.5）：
```
Step1: 1×1 缩减层 8→2 通道 → 密文含 8 槽但仅 2 槽有效（25% 利用率）
Step2: 施加 0/1 掩码的 rot-mask-add：
       ct_dense = Σ_{valid} Rot(PMult(ct_reduced, mask_c), offset_c)
       即对每个有效通道，掩码提取（PMult）+ 旋转对齐（Rot）+ 累加（Add）
       —— 2 个有效通道压实到同一密文的相邻槽
Step3: 扩展层 8 输出通道只需 1 个满装密文（无 AAC 需 4 个）
       scale：权重 PMult 用 Δ₁、掩码 PMult 用 Δ₂，Δ₁·Δ₂=Δ → 只做一次 Rescale
```
- Annotations：rot-mask-add 是 FHE 中的"数据复制/重排"标准模式（Fhelipe/Pantheon/Coeus 也用于复制或通信），AAC 把它专门用于跨层保密度；步骤中旋转数受 Conv-aware Encoding 的旋转上界约束（AAC 不破坏布局的旋转保证）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在推理图编译阶段，FEnc² 识别所有通道缩减-扩张模式（fire-module、residual-shortcut 等），对每个缩减点自动插入 AAC 重打包；运行期执行 mask-rotate-add 的密文操作。效果（Table X）：fire-module 各层加速 1.47×-4.68×（合计 2.09×），residual-shortcut 各层 1.016×-1.75×（合计 1.64×），slot 利用率从 0.02-0.5 恢复到 1.0；端到端重打包开销仅占总延迟 0.42%-3.7%。

## blocked-Cuckoo 哈希（Blocked Cuckoo Hashing，SSD 原生 KV 存储）

术语解释
- blocked-Cuckoo 哈希是把 Cuckoo 哈希与"桶=块"结合的哈希结构：每个 key 映射到两个候选桶（各对应一个 SSD 块，块内多个槽位），桶满时用重定位（relocation）而非丢弃处理溢出；论文用它构建完全无 DRAM 驻留索引/元数据的 SSD 原生持久 KV 存储。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 标准 Cuckoo 哈希（Pagh & Rodler 2004）：每个 key 有两个候选位置，插入冲突时把旧 key "踢"到其另一候选位，形成位移链；无 stash 时负载因子实用上限约 50%，加入多槽桶（blocked）或 stash 后可到 ~90-95%（网络来源：EMOMA 用 stash+CBBF 达 ~95% 负载、每次查找至多一次片外访问）。论文的 KV 设计：key 映射到两个 SSD 驻留候选桶，每桶=一个 SSD 块（l_blk=512B on Storage-Next、4KB on Normal SSD），桶大小 B=l_blk/l_KV（l_KV≈64B），每次查找需 1-2 次 SSD 块读（平均 1.5）；负载因子需低于临界值 α_critical（B≥4 时通常 >0.95），插入位移链期望长度 E[L]=α^(2B)/(1−α^B)，运行在远低于临界处使 E[L]≪1、插入延迟近常数。DRAM 全部用于缓存热 KV 对（个体粒度），SSD 驻留 WAL 合并更新后批量提交回桶块。
- 从算法pipeline角度拆解术语：一次 GET 的 pipeline：主机算两个桶哈希 → 查 DRAM 缓存（命中即返回）→ 未命中发 1-2 次 SSD 块读（平均 1.5）→ SSD 返回后返回客户端；一次 PUT（insert 或 update）的 pipeline：更新先追加到 WAL 合并同桶更新 → WAL 超阈值 → 提交合并更新进 blocked-Cuckoo 块 → 回收日志空间；更新负载下每次 WAL flush 把分散的 KV 更新聚合成块级读改写。相比 CacheLib（桶溢出即丢弃条目）与内存 KV（DRAM 索引随 key 基数线性增长），本设计把索引完全放到 SSD 块内，DRAM 成本与 key 数无关。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：桶哈希函数（两个独立哈希）、桶内线性扫描、插入时沿候选链重定位、负载因子控制与 stash/重哈希兜底；SSD 版本额外要求块对齐、WAL 持久化与更新合并。论文评估：5TB KV、800 亿 64B 条目、负载因子 0.7、GET:PUT 100:0/90:10/70:30/50:50、lognormal 强弱局部性（σ=1.2/0.4），GPU+SN 在读重混合下达 100+ Mops/s（FASTER 内存级水平），CPU+SN 则受 host IOPS 限制。论文为模型驱动评估（分析框架+MQSim-Next 模拟），无开源实现。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy

## HNSW（Hierarchical Navigable Small World）

术语解释
- HNSW 是多层可导航小世界图 ANN 索引：层 0 含全部节点、上层节点指数级稀疏、层间长程连接，查询从顶层粗粒度贪心下降到底层精化，搜索复杂度 O(log n)；论文在 SSD-resident ANN 案例中用它并配合"图链接元数据与节点同驻 SSD、高层热节点驻 DRAM"的放置策略。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HNSW（Malkov & Yashunin 2018）把向量组织成多层邻近图：层 0 最密（含所有节点），越往上节点越少、连接越长程，节点所在最高层由指数衰减概率决定（mL 参数）；查询从顶层入口贪婪导航逐层下降，每层局部贪心搜索候选集，复杂度 O(log n)（网络来源：Milvus/Pinecone/Weaviate/FAISS/cuVS 等向量库的骨干）。论文利用 HNSW 的层次访问模式：高层节点少、访问间隔短（DRAM 友好），低层节点多、访问间隔长（SSD 友好），把图链接元数据与节点 co-locate 在 SSD、DRAM 只缓存高层节点；用校准的 layer-aware 合成 trace 模拟其 coarse-to-fine 流水。
- 从算法pipeline角度拆解术语：一次查询的 pipeline：从顶层入口开始 → 逐层贪婪下降（每层访问若干节点，比较距离）→ 在层 0 找到 k 近邻候选。SSD-resident 版本：每访问一个节点需从 SSD 读其向量与邻居链接（小块随机读）；高层节点命中 DRAM，低层节点读 SSD。论文叠加两阶段渐进（见"两阶段渐进式 ANN"条目）：先用 512B reduced 向量粗筛、再对 5%-20% promoted 候选取 full 向量精排，使大部分访问落在高 IOPS 的小块读上。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现参数（网络来源）：M（每节点连接数，默认 16）、m_max0（层 0 上限 2M）、ef_construction（建图候选数 200）、ef_search（搜索候选数 50）、mL（层分配衰减）；开源实现包括 FAISS、hnswlib、Milvus 等。论文使用方式：80 亿 embedding 语料、reduced 固定 512B、full 2/4/6/8KB（promotion 5%/10%/15%/20%），GPU+SN 达 13-17 KQPS（512GB DRAM），相对 Normal SSD 一致 2-3×；对照 DiskANN（约 5 KQPS 量级）。论文为模型驱动评估，无开源实现。
- NasZip 补充视角（ISCA'26，HNSW 作为 NDP 加速对象）：NASZIP 聚焦 HNSW 搜索阶段（反复执行、主导系统性能），在其上叠加 FEE-sPCA 早退（PCA 变换后按 burst 估计距离提前剪枝）、Dfloat 位级压缩、DaM 邻居表映射与 LNC 缓存/预取，并给出 NDP 执行模型：候选优先队列维护 threshold（队列最远点距离）、逐 hop BFS 由 NDP 的 VPE 并行算距、邻居表查找卸载到 NDP 并按数据感知映射避免跨 sub-channel 通信。efSearch 增大提升 recall 但降 QPS（Fig.19），batch=16 为吞吐/延迟折中。HNSW 索引用 NVIDIA cuVS 构建（预建索引保证可复现，自建因随机性略有偏差）。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## 两阶段渐进式 SSD-resident ANN 搜索

术语解释
- 两阶段渐进式（two-stage progressive）SSD-resident ANN：每个 embedding 在 SSD 上同时存 reduced-dimension（如 512B）与 full-dimension（如 2-8KB）两种形式，查询先取 reduced 向量粗筛淘汰大部分候选，再仅对少量 promoted 候选取 full 向量精排，把 IOPS-bound 的小块读与带宽-bound 的大块读分层。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 动机：SSD 驻留 ANN 中，多数距离比较只是"确认拒绝"（Gao et al. 报告 >90% 比较用于淘汰候选），全维度求值往往不必要；先降维淘汰、再全维精排可大幅减少大块读。reduced 向量来源：(1) 线性变换（PCA、随机投影）、(2) 双模型 embedding pipeline、(3) MRL（原生支持多分辨率向量）。论文在 MRL 生成的 MS MARCO、20 Newsgroups、DBpedia 语料上验证 recall>98%。Storage-Next 直接受益：绝大多数访问命中 512B reduced 向量（小块随机读 → 极高 IOPS），promoted 子集（5%-20%）带宽-bound 但被大拒绝率摊薄。
- 从算法pipeline角度拆解术语：一次查询的 pipeline：①取查询的 reduced 向量与候选集的 reduced 向量计算距离 → 淘汰 >90% 候选（小块随机读、IOPS-bound）→ ②对 promoted 子集取 full 向量重排（大块读、带宽-bound）→ 返回 top-k。论文的量化评估（Fig. 10）：512B→2KB（95%/5%）、512B→4KB（90%/10%）时 GPU+SN 保持 SSD-IOPS 受限（7-11→13-17 KQPS 随 DRAM 512GB）；512B→6KB（85%/15%）400GB 后 GDDR 带宽封顶（8.3 KQPS）；512B→8KB（80%/20%）300GB 即带宽受限。promotion 率越高、DRAM 流量越大、带宽天花板越早出现。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：embedding 双份存储（reduced+full）、查询端两阶段执行、promotion 判定（距离阈值或 top-k 动态）；reduced 版本可由 MRL 直接截断得到（无需额外训练）。论文用途：展示"flash 作为主动层"如何催生新算法——低成本的 TB/PB 级 embedding 表留驻 flash，GPU+SN 把吞吐推到几十 KQPS（DiskANN ~5 KQPS 量级）且保持 HNSW 级 recall。论文为模型驱动评估（分析框架+MQSim-Next 模拟），无开源实现。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy

## MRL（Matryoshka Representation Learning，套娃表示学习）

术语解释
- MRL 是一种表示学习方法：用一个模型同时训练多个嵌套维度（如 768→512→256→128→64 维前缀均可用），使 embedding 的不同前缀保留不同粒度的语义信息，推理时可动态截断维度适配计算/带宽约束，无需额外推理成本。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MRL（Kusupati et al., NeurIPS 2022）在损失中加入嵌套维度的 softmax 损失，使向量前缀自相似地编码信息，可在 25-50% 维度处保持检索质量、全库检索提速达 14×（网络来源）；截断非 MRL 模型会无警告地掉 recall（"recall cliff"），MRL 训练则把下降点推迟到模型特定 knee 点之后。论文把 MRL 作为两阶段渐进式 ANN 的 reduced 向量来源之一（原生支持多分辨率向量），在 MRL 生成的 MS MARCO、20 Newsgroups、DBpedia 语料上验证 recall>98%。
- 从算法pipeline角度拆解术语：MRL 在训练 pipeline 中一次性产出多分辨率表示——embedding 的 512B 前缀作为 reduced 向量（粗筛用）、4KB 全长作为 full 向量（精排用），两者来自同一模型、无需额外推理；推理 pipeline 中粗筛阶段只取前缀、精排阶段取全长，构成论文 ANN 案例的向量供给侧。与 PCA/随机投影（需额外线性层）相比，MRL 无需变换层。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：训练时把同一 batch 的 logits 在多个嵌套维度分别计算损失并加权求和（MatryoshkaLoss，Sentence-Transformers 等提供实现）；推理时按需截断向量前缀。论文用途：在 SSD-resident 场景中 MRL 使"一份存储、两档精度"成为可能，与 Storage-Next 的 512B 高 IOPS 配合。论文仅在案例研究中使用 MRL 生成语料并报告 recall，未展开 MRL 训练细节。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy

## WAL（Write-Ahead Log，预写日志）

术语解释
- WAL 是持久化存储系统的经典机制：写操作先追加到日志（顺序写、落盘）再更新主数据结构，崩溃后可重放日志恢复一致性。论文的 SSD 原生 KV 用它做持久化与写摊销——合并同桶更新、超阈值后批量提交进 blocked-Cuckoo 块并回收日志空间。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- WAL 的核心是把"随机小写"转化为"顺序追加"，并以日志作为崩溃一致性的权威来源（主流存储引擎如 RocksDB/WiredTiger/SQLite 均采用）。论文把 WAL 放在 SSD 上（SSD-resident WAL）：PUT 更新先追加到 WAL 合并目标同桶的更新，WAL 超阈值时把合并后的更新提交（commit）进 blocked-Cuckoo 哈希块、再回收被重用的日志空间。这样既保证持久性（不丢条目，区别于丢弃溢出的 CacheLib），又通过合并把多个 KV 更新聚合成块级读改写、摊销写成本。
- 从算法pipeline角度拆解术语：一次 update 的 pipeline：主机收到 PUT → 追加到 SSD WAL（顺序写）→ WAL 达到阈值 → 读目标桶块（read-modify-write）→ 合并更新写回桶块 → 回收日志。GET:PUT 混合下，写比例越高、read-modify-write 越多、I/O 流量越大、吞吐越低（论文 Fig. 8：读写比从 100:0 到 50:50 吞吐显著下降）；强局部性（σ=1.2）使同桶更新更集中、每次 WAL flush 的 read-modify-write 更少。DRAM 只缓存热 KV 对，索引/日志全在 SSD。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：WAL 区域管理（追加指针、阈值触发、回收）、崩溃恢复（重放）、与主结构的批量提交事务；论文未给出 WAL 具体实现细节（块大小、刷盘策略等），仅描述其合并-提交-回收流程。论文用途：作为"flash 主动层"下持久 KV 的写路径设计示范——把 DRAM 从索引/日志中完全解放。信息缺口：WAL 的具体刷盘/组提交机制论文未明确说明。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding

## Multi-bit TFHE（多位 TFHE：整数密文扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Multi-bit TFHE 是 TFHE（Fast FHE over the Torus）的整数加密扩展（Chillotti-Joye-Paillier ASIACRYPT'21 等）：不再每个 ciphertext 只加密 1 bit，而是把多个 bit 打包进单个 LWE ciphertext 加密一个整数（如 3–10 bit），从而把可编程自举（PBS）的成本摊薄到大量廉价线性运算上。Boolean TFHE 每个 bit 一次操作都要一次 PBS（每个 gate 一次 bootstrapping，几百万 gate）；multi-bit TFHE 只有两类原语：线性运算（加法、明文标量乘法，LWE-native、无需自举、极快）与 LUT（查任意非线性函数，每次需一次 PBS）。因此程序执行画像从"百万次 PBS 主导"变为"海量廉价线性运算 + 少量昂贵 PBS"，且位宽越宽，两次 PBS 之间可连续执行的本征线性运算链越长，PBS 频率按数量级下降（FlashTFHE 论文 profiled 七个真实 workload 验证：DNN 类 ~99.5% 运算是 LWE-native，PBS 仅占 ~0.5%）。
- 代价：更宽位宽需要更大的密码参数集（LWE 维数 n、GLWE 度 N、gadget 分解深度 l_b 都要随位宽增大才能维持 128-bit 安全与 p_err<2^-14 的噪声界）。例如 10-bit ciphertext 需要 N=2^16、n≈1000+，BSK key 达数 GB（GPT-2 decoder layer 的 key 有 4.7GB），远超 Boolean TFHE 的几 MB。参数变大的后果正是 FlashTFHE 的动机：现有空间加速器（Morphling 只支持 6-bit/N≤4096、Strix N≤16384）被带宽、利用率、面积三重瓶颈卡住，FlashTFHE 用时间域密钥复用把支持推到 10-bit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一个 8-bit 整数加法的两种实现对比（论文 Figure 5）：
```
# Boolean TFHE：6-bit 加法 = 逐 gate 拆解，每 gate 一次 PBS
for each bit b in 0..5:
    out[b] = Gate(FA(...))   # 每个 full-adder gate 都含一次 PBS
# 253 ms（每 gate 11ms，加法需 23 个 gate）
```
```
# Multi-bit 8-bit TFHE：整个整数在一个 ciphertext 里
acc = c1 + c2                # LWE-native 向量加，无 PBS
# 0.008 ms；5-bit 分段版因 carry 需 bivariate LUT 反而要 1 次 PBS（47ms）
```
- 程序级 pipeline（以 GPT-2 decoder layer，7-bit 量化/6-bit rounding）：量化权重与激活 → 逐层 matmul/逐元素加（LWE-native，零 PBS）→ 非线性（GELU/softmax 等）编码进 LUT → 每 LUT 一次 PBS 刷新噪声 → 继续下一层。PBS 频率 = 非线性激活/函数个数而非算子个数，这是 multi-bit 相对 Boolean 端到端加速（论文实测 8-bit 相对 4-bit 模拟加速 6.8–8.1×，DNN 类）的根本来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：Zama 的 Concrete/Concrete-ML/TFHE-rs 库原生支持多 bit 整数密文与参数搜索（Concrete Optimizer），Concrete-ML 对 PyTorch/scikit-learn 模型做 PTQ 后自动生成 multi-bit TFHE 程序；FlashTFHE 论文所有 workload（CNN-20/50、KNN、XGBoost、Decision Tree、GPT-2）均由 Concrete-ML v1.6.1 + Concrete Compiler v2.7.0 生成。硬件实现：FlashTFHE 加速器（BRU 做外部乘积/盲旋转、LPU 做 key-switching 与 LWE-native 运算）。使用注意：位宽选择要在参数代价（N/n/l_b 增大、key 变大）与 PBS 频率降低之间权衡，需 Lattice Estimator/Concrete Optimizer 联合搜索安全参数。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption

## LWE / GLWE / GGSW 密文与外部乘积（External Product）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 TFHE 的三种内部密文类型与核心计算原语。Torus（环面）T 概念上是 [0,1) 上的实数，实现为 w-bit（通常 32/64）离散定点小数。(1) LWE 密文：加密客户端消息的最小密文，参数化 LWE 维数 n（通常 500–1000），含 n 个 mask 元素 + 1 个 body 元素；(2) GLWE 密文：把 LWE 的每个 torus 标量换成 degree-N 多项式（N 为 2 的幂），一个 GLWE 含 k+1 个多项式（k 为 GLWE 维数），用于编码 LUT 与存 PBS 中间结果；(3) GGSW 密文：构成 bootstrapping key（BSK）的密文类型，每个 BSK 含 n 个 GGSW，每个 GGSW 是 (1+k)^2×l_b 的多项式矩阵（l_b 为 gadget 分解深度），支持外部乘积。外部乘积（External Product，GGSW □ GLWE → GLWE）是自举（盲旋转）的核心运算：本质上是"向量-矩阵乘"，每个元素都是 degree-N 多项式乘法（用 FFT/IFFT 加速）。
- 三者的角色：LWE 在"明文侧"（用户消息），GLWE 在"函数侧"（LUT/test vector），GGSW 在"密钥侧"（BSK）。一次 PBS 中三者反复交互（见盲旋转条目）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 外部乘积的计算过程（论文 Figure 4）：对 GGSW 密文做 gadget 分解得到 l_b 层整数向量，与 GLWE 多项式的各次幂做多项式乘加，最后累加。硬件里每个多项式乘法用 FFT/IFFT：FFT(GGLWE) 与 FFT(分解后的 GGSW chunk) 逐点乘 → IFFT。FlashTFHE 的数据通路（Figure 10/11 伪代码）：
```
for bsk_chunk in BSK:                 # 外层：载入一个 BSK chunk（≤0.8MB）片内复用
    for i, decomp_glwe in rr_ctxts:   # 内层：round-robin 遍历所有在飞 ciphertext
        fft_out = FFT(decomp_glwe)    # FFT-A/FFT-B 每周期产 chunk
        acc += VecMAC(fft_out, bsk_chunk)   # 与 BSK subchunk 做 tiled 乘累加
    # 满 (k+1)*l_b 次累计后 I-FFT → sample extraction
```
- 关键量：一次盲旋转要做 n 次外部乘积迭代；每迭代含 (k+1) 个多项式点积（k=1 时仅 2× 复用）与 l_b 层分解。BSK 总量 O(n·N·l_b)，10-bit 时数 GB——每个 BSK 系数在一次自举中恰好用一次，算术强度极低，是"流式大 key"问题的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：TFHE-rs/Concrete 的 external product 用浮点或近似 FFT/NTT 实现；硬件：所有 TFHE 加速器（MATCHA、Morphling、Strix、FlashTFHE、Trinity/UFC）的核心 MAC/FFT 流水线都围绕外部乘积设计。FlashTFHE 用 VecMAC（512 coefficients/cycle/core）+ 48-bit 定点双实数 FFT 集群实现，支持 N 至 2^16。实现选择要点：FFT 位宽（32-bit 会损失正确性，FlashTFHE 用 48-bit 保证 TFHE-rs 与 Concrete 全参数集正确）、分解深度 l_b（越大噪声越小但 BSK 越大）、k（multi-bit 实用参数强制 k=1 以控制 O(k²) 自举成本）。

MNEMOS 补充视角（ISCA'26，GPU 上外部乘积的 BSK 分块复用）：在 GPU 上执行一次外部乘积（MAC）时，对单个 GLWE 做 MAC 需取 (k+1) 倍于 GLWE 体积的 BSK 数据（BSK 形状 (k+1)ℓ×(k+1)，含 (k+1)ℓ×(k+1)×N×n 个元素），且 BSK 预计算后跨一批 PBS 复用。朴素"整 BSK 缓存进共享内存"不可行（部分参数集 BSK 超 A100 每 SM 192KB 合并 L1/SPM 上限，且过度分配共享内存会蚕食 L1 容量）。MNEMOS 利用 BSK 与傅里叶系数之间是逐元素 Hadamard 积（非一般矩阵乘）的性质做分块（tiling）：单个线程块只需处理一块 TBSK 对一块 TGLWE，同一 BSK 分块被一批中多个 PBS 实例（同卷积层共享参数）并发复用，把复用层级从 L2 提升到 SM 级；分块几何取 8 个连续复数 FP64 元素（128B）对齐内存事务粒度保证合并访问。参数 k 增大时（安全级别依赖 kN，Concrete 常用大 k）BSK 足迹占比上升，该复用收益随之增大（消融：+MAC 单独 1.10×~1.77×，k 大时最显著）。

CASCADE 补充视角（ISCA'26，BSK 的 GGSW 结构与外积的硬件数据流）：CASCADE 中 BSK 是 GGSW 密文（L×(k+1)×(k+1) 多项式矩阵，每个元素是 N 阶多项式），中间密文 ACC 是 RLWE（(k+1) 向量，N 阶多项式），外积 = 矩阵-向量乘。硬件上外积映射到 HC 的 VMA（Vector Multiplication-Add）单元：FFT 域里 BSK 多项式与 ACC 多项式变成逐系数相乘，VMA 由向量乘法单元（逐系数乘）+ 累加器（逐系数加）组成；FFT 单元做 log2N 级 butterfly（BU 个并行 butterfly 单元，2·BU 系数/级，总约 log2N·N/(2·BU) cycle），IFFT 单元因 Decomposition 单元使 FFT 侧多项式数更多而分配更少资源。BSK 总量 126 MB（参数集 III 需 112 MB、IV 需 90 MB）全部驻留分布式 SRAM——对比 CKKS 需 GB 级密钥，TFHE 的 10s-100s MB 量级正是"可全部片上驻留"的前提（见"Bootstrapping Key（BSK，自举密钥）"条目）。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## 盲旋转（Blind Rotation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 盲旋转是 TFHE 可编程自举（PBS）中最核心、最耗时的步骤（约占 PBS 运行时间 90%）：根据输入 LWE 密文的整数化分量（modulus-switching 到 Z 后），把 GLWE 密文（编码 test vector/LUT）整体旋转对应步数，从而"盲"地（在密文域、不泄露明文）完成一次多项式位移，等价于对 LWE 的每个分量做一次多项式幂乘 X^{a_i}·prod 并累加。它由 n 次串行迭代的外部乘积组成（n 为 LWE 维数，key-switching 后通常从 ~30000 降到 ~1000），每次迭代 = gadget 分解 + 多项式乘（FFT/IFFT）+ 累加。旋转完成后，test vector 中由明文决定的项被移到常数项位置，再经 sample extraction 取回 LWE 密文，同时噪声被刷新到低水平——这就是 PBS 既能求 LUT 又能刷新噪声的原因。
- 在 multi-bit TFHE 中盲旋转的输入/输出都是更大参数（N 至 2^16、l_b 8–10），因此它主导了整个加速器设计：外部乘积吞吐、FFT 单元、BSK 流式访存都围绕它优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- PBS 四步（论文 Figure 3，key-switching-first 顺序）：
```
PBS(c_lwe, f):                      # c_lwe: n 维 LWE，f: LUT
  (A) c' = KeySwitch(c_lwe)         # KSK 把 n 从长切短（~30000→~1000），~10% 时间
  (B) mu = ModSwitch(c')            # torus → 整数，<1%
  (C) ACC = BlindRotation(mu, test_vector_f)
      for i in 0..n'-1:             # n' 次串行外部乘积迭代
          ACC = X^{mu[i]} * (GadgetDecomp(BSK_i) □ ACC)   # FFT/IFFT 多项式乘
      # 占 ~90% 时间
  (D) out = SampleExtract(ACC)      # 取 GLWE 常数项回 LWE，<1%
```
- Annotations：盲旋转的迭代次数 = key-switching 后的维数 n'；每迭代的算力 = (k+1)·l_b 个多项式乘；FlashTFHE 把 C 步骤的数据通路设计成外层循环扫 BSK chunk、内层 round-robin 扫 ciphertext（时间域复用），与空间架构"BSK 单遍流经 PE 阵列"相反。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：TFHE-rs/Concrete 的 blind rotation 用 FFT 加速多项式乘；硬件：加速器把盲旋转映射到 FFT 流水线 + VecMAC。FlashTFHE 的 BRU 内：FFT-A（256-point）/FFT-B（128-point）混合基双实数 FFT 集群 + VecMAC（512 coef/cycle）+ 9.2MB 累加 buffer + 共享 I-FFT（FFT:I-FFT 操作数比 l_b:1，两个 BRU 共用一个 I-FFT）。执行顺序选择：key-switching-first（FlashTFHE 采用）允许 key-switching 结果跨多个后续盲旋转复用（KS-dedup），而 blind-rotation-first 单次 PBS 计算量相同但无此复用机会。

MNEMOS 补充视角（ISCA'26，GPU 端盲旋转的迭代结构与跨迭代融合）：MNEMOS 的 Algorithm 1 逐迭代结构为 Rotation（ACC_rotate = X^{ã_i}·ACC − ACC）→ Decompose（按基 β 位切片成 ℓ 组）→ FFT（Tangent FFT，N/2 点复数 FFT）→ MAC（ACC_fourier ⊙ BSK）→ IFFT + 累加，共 n 次迭代。GPU 剖析显示 BSK 访问导致的 stall_long_scoreboard 是盲旋转最大瓶颈（超 50%）。MNEMOS 两个针对性设计：(1) MAC 阶段 BSK 分块复用（见"LWE / GLWE / GGSW 密文与外部乘积"条目补充）；(2) 跨迭代 kernel 融合——由于 FFT 与 IFFT 使用同一组 twiddle factors / precomputation factors 的共轭版本，把迭代 i 的尾部与迭代 i+1 的头部（即 IFFT 后接下一次 FFT）融合为单个 kernel，两套系数在片上跨迭代复用、消除主循环内对这些系数的冗余全局载入，收益随分解层数 ℓ 增大（ℓ 越大融合窗口越宽）。

CASCADE 补充视角（ISCA'26，盲旋转 = n 次 HMUX 的流水线化）：CASCADE 论文把盲旋转明确表述为 Algorithm 1 第 4-6 行的 n 次 HMUX 迭代（与 FlashTFHE/MNEMOS 的"n 次外部乘积迭代"是同一结构）。CASCADE 的洞察是：这 n 次迭代并非不可流水——把 HMUX_i 的 ACC 输出在算完多项式系数后立即流向 HMUX_{i+1}（inter-HC 系数粒度流水），稳态吞吐可从 1/(n·t_HMUX) 提升到 1/t_HMUX。但盲旋转流水化引出两个此前被忽视的硬件挑战：(1) n 个 HMUX 并发访问各自 BSK 造成极端带宽压力（集中式 HBM 无法支撑）；(2) 每个 HMUX 依赖前一 HMUX 输出的中间密文（ICT），跨 chiplet 的 ICT 传输造成 D2D 通信瓶颈（D2D 时延 > HMUX 计算时间时 HC 严重欠利用）。CASCADE 分别用 BSK-distributed 分布式 SRAM 与 Interleaved-Fusion（融合+交错）策略 + OIFS 调度解决。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## Lattice Estimator（格估计器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Lattice Estimator 是 Albrecht、Player、Scott 等维护的开源密码学工具（https://github.com/malb/lattice-estimator，Sage 实现），用于估计基于格的问题（LWE、NTRU、SIS 等）在给定参数下的实际破解成本（比特安全强度），把主流攻击算法（BKZ 格基约化、primal/dual 攻击、Meißner 等）的复杂度估计出来。它由论文 "On the concrete hardness of Learning with Errors"（J. Math. Cryptology, 2015）提出，是 FHE 社区选安全参数的事实标准。
- 在 TFHE 参数选择中，Lattice Estimator 用于验证：(1) 给定 LWE 维数 n、GLWE 度 N、模数等，估计的安全级是否 ≥ 目标（如 128-bit）；(2) 结合噪声传播分析（保证 PBS 错误概率 p_err<2^-14），在"安全-噪声-位宽"三维空间找可行参数点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FlashTFHE 论文用 Lattice Estimator 映射"安全级别 vs 常用参数"（Figure 7）：对每个消息位宽（2–10 bit），搜索满足 128-bit 安全与 p_err≤2^-14 的 (n, N, l_b) 组合，绘制 log2(N) 与 n 随位宽的增长曲线，得到结论"位宽增大 → n 与 N 必须同步增大"（10-bit 需要 N=2^16、n>1000），并据此标注 Morphling 支持的 6-bit 上限与 FlashTFHE 的 10-bit 目标。流程：候选参数 → Lattice Estimator 估安全级 → 若 ≥128-bit 且噪声估计 ≤2^-14 则接受 → 否则增大 n/N/l_b 重试。
- 具体到一次评估：给定 (n=1070, N=65536, k=1, l_b=8, 位宽9)，Lattice Estimator 估计该 LWE 实例的 BKZ 成本对应的安全 bit 数；同时用 gadget 分解与多项式乘的噪声增长公式估计自举后噪声，两者都达标才进入 Table II 的可用参数集。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Sage 库 + 攻击复杂度模型（BKZ 成本用 sieve/enumeration 模型），命令行或 Python API 调用，输出每攻击路径的估计成本与最小安全位。使用：FHE 库（如 Concrete Optimizer、OpenFHE、TFHE-rs 参数搜索）内嵌 Lattice Estimator 自动选参；研究者用它批量扫描参数空间生成曲线（本论文 Figure 7 即一例）。注意点：它是"估计"工具，不同版本/攻击模型假设会改变结果，论文通常固定版本与攻击集以保证可复现。

涉及论文标题：
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption

## 二值化神经网络（Binarized Neural Network, BNN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 二值化神经网络（BNN）是把权重与激活都约束为 1-bit（{+1,−1}，通常用 sign 函数）的神经网络。它把浮点乘加（MAC/FMA）替换为按位 XNOR + popcount 运算，实现约 32× 存储压缩与约 58× 计算量下降，适合超低功耗边缘/微架构硬件。代表作：BinaryNet（Courbariaux & Bengio 2016，权重与激活都二值化，用 STE 训练）与 XNOR-Net（Rastegari 等 2016，引入 BWN 仅二值权重+缩放因子 α、XNOR-Network 权重激活都二值化 → 卷积退化为 XNOR+bitcount）。逻辑链：1-bit 量化 → 乘法变位逻辑 → 面积/功耗数量级下降 → 但精度受损 → 靠先进训练技巧（STE、混合精度潜在权重、缩放因子）弥补。Moirai 论文把它用于 L1D 预取器，把 TCN 变成 CaPNet：权重与激活 1-bit，前向只用 XNOR/popcount，780 Bytes 存储、1178 μm² 面积、8.5mW 功耗（ASAP7 7nm @4GHz），精度仅比 INT8/full-precision 下降 <2%（Figure 19c）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- BNN 在 Moirai 中的计算 pipeline（一次前向）：
  1. 二值化：W_bin = sign(W_raw)（W_raw 为混合精度潜在权重，首层 7-bit、其余 4-bit）；激活 A_bin = sign(A)；
  2. 二值卷积：Ac^k = bitcount(A_bin ⊙ W_bin^k)（式 1，XNOR 逐位比较 + popcount 计数，等价于二值向量的点积：popcount(x_b XNOR w_b) = 匹配位数，64 位字打包后一条 XNOR+POPCNT 指令算 64 个乘法）；
  3. 结果经 sign 传给下一层，通道结构 [8,4,2] 逐层下采样；
  4. 反向（训练）：梯度经 STE 绕过 sign 的非可导点更新 W_raw（式 2：ΔW_raw^k = G_{i+1} * Ac_i^k）。
- 与浮点卷积对比：常规 y = Σ x_i·w_i（64 次乘法+累加）→ BNN 变成 popcount(x_b XNOR w_b)（1 次位运算簇）。论文合成评估：Moirai 二值化相对 INT8/full-precision 仅 <2% 精度损失，换取 >8×/>32× 存储缩减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练侧实现：保留全精度潜在权重（latent/raw weights），前向时 sign 二值化，反向用 STE 更新潜在权重（BinaryConnect/BinaryNet 范式）；实践中常保留首层（特征保真度敏感）与末层为更高精度——Moirai 即给首层 7-bit、其余层 4-bit 的混合精度 W_raw。硬件侧实现：XNOR gate 阵列 + popcount 加法树（Moirai 的 FCC），反向用 BCC（条件符号翻转器 + 浅加法树）。使用场景：边缘 NPU（STM32N6 类）、FPGA、ASIC、内存内计算；Moirai 开创了把它用于 L1D 预取器的新场景（BTCP 是 L1 之外用 B-TCN 的先前工作，4.5KB/134-cycle 只能放 L2）。开源参考实现：BinaryNet.pytorch、XNOR-Net-PyTorch。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework

## 时序卷积网络（Temporal Convolutional Network, TCN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TCN 是 Bai、Kolter、Koltun（2018，arXiv:1803.01271）提出的用于序列建模的卷积网络，核心是因果卷积 + 空洞卷积 + 残差连接：每个时间步 t 的输出只依赖 t 及之前的输入（因果、无未来信息泄漏），空洞卷积让感受野随深度指数增长，残差块保证深网络可训练。与 RNN/LSTM 相比，TCN 可并行处理整个输入序列（无串行状态更新）、感受野可调、梯度稳定、训练内存低。在 Moirai 中，设计空间探索（Figure 3）显示同参数预算（≈380 参数）下 TCN 预测准确率最高：RNN/LSTM 的理论无限时序感受野在实际 L1D 场景下被串行状态更新的延迟瓶颈抵消——推理时间过长使预测"迟到"，完全抵消长程模式覆盖收益；TCN 的卷积并行 + 空洞感受野更适合低延迟硬件实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Moirai 中 CaPNet 的 TCN 前向（3 层，通道 [8,4,2]，输入 10 个历史 delta）：
  ```
  # 输入：A_0 = [delta_{t-9}, ..., delta_t]  (10 个 delta)
  for layer i in 1..3:                      # 通道 8 → 4 → 2
      for channel k in 1..C_i:              # 每层 C_i 个并行卷积滤波器
          Ac_i^k = sign( bitcount(A_{i-1} ⊙ W_bin^k) )   # 空洞因果卷积(二值化)
      A_i = 拼接所有通道输出
  D_pred = A_3                               # 预测下一个 delta
  ```
  整个输入窗口广播到各通道，每个通道独立学习一个空间-时间模式（通道指并行滤波器，非硬件布线路径）。
- 关键点：CaPNet 是 TCN 的 BNN 化（权重/激活 1-bit），空洞感受野让 3 层网络捕获 delta 序列中的长程依赖；序列式 TCN 能在模式起始点就前瞻识别（predictive lookahead），这是其 92.37% 及时性的来源之一。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：1D 全卷积（FCN），每层保持输入长度，空洞系数随层指数增长（d=1,2,4,...），残差块含两层空洞因果卷积 + weight norm + ReLU + dropout；PyTorch 有官方 TCN 教程与社区实现。Moirai 的硬件实现是 FCC（前向卷积单元，3 输入对应 kernel size 3）+ BCC（反向，10 输入对应对整个时序序列长度，跨所有时间步累计共享权重梯度）。使用场景：语言建模、时间序列预测（客流/盾构预测等）、以及本论文的硬件预取器；BTCP 是先前把 B-TCN 用于预取（drop-in 设计，PC 流跟踪 + 无片上反向传播 + 无污染控制，4.5KB/134-cycle，只能放 L2）。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework

## 空洞卷积（Dilated Convolution）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空洞卷积（dilated/atrous convolution）是在卷积核元素间插入空洞（跳过步长 dilation d）的卷积，感受野从普通卷积的线性增长（每层 +k−1）变为指数增长（第 L 层感受野 ≈ 1 + Σ(k−1)·d_L，d 逐层翻倍时指数扩展），而不增加参数与计算量。源自 WaveNet（音频生成），是 TCN 的核心组件。在 Moirai 中，空洞卷积让仅 3 层、≈380 参数的浅 TCN 也能覆盖 delta 序列中的长程依赖——这是"极简硬件预算下保持泛化能力"的关键。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一维空洞因果卷积（Moirai 的 CaPNet 中）：
  ```
  # 卷积核大小 k=3，第 i 层空洞系数 d_i，输入 A（长度 10 的 delta 序列）
  for t in range(L):                                  # L=10，逐时间步
      acc = 0
      for j in range(k):                              # j=0,1,2
          idx = t - j*d_i                             # 空洞：只取过去 d_i 步前的点
          if idx >= 0:
              acc += bitcount(A_bin[idx] ⊙ W_bin[j])  # 二值化乘加
      Ac[t] = sign(acc)
  ```
  例：d=1（第 1 层）覆盖 t,t-1,t-2；d=2（第 2 层）覆盖 t,t-2,t-4；d=4（第 3 层）覆盖 t,t-4,t-8——3 层即可触及 8 步前的 delta，感受野随层指数增长，参数仍只有 3×通道数。
- 因果性：只允许看向过去（idx = t - j·d ≥ 0），保证预测只用历史，符合预取"预测未来"语义。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：深度学习框架中直接指定 dilation 参数（PyTorch `nn.Conv1d(dilation=d)`）；硬件上空洞只是改变数据访问索引/移位量，不增加乘加单元——Moirai 的 FCC 输入为 3（kernel size），硬件把空洞实现为跨周期的寄存器延迟线取数。使用场景：序列建模（TCN）、音频（WaveNet）、语义分割（DeepLab）；在 Moirai 中作为 CaPNet 的低延迟长程建模手段。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework

## 直通估计器（Straight-Through Estimator, STE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- STE（Bengio 等 2013，arXiv:1308.3432）是训练含不可导操作（sign/round 量化、二值化）网络的梯度近似技术：前向用真实不可导操作（如二值化 sign），反向把"零梯度"替换为代理梯度（常用恒等/直通，即 ∂L/∂x ≈ ∂L/∂x̂，或饱和 STE 1_{|x|≤1}），让梯度"直通"不可导点到达潜在权重。Yin 等（ICLR 2019，arXiv:1903.05662）从理论上把 STE 梯度形式化为"粗梯度"（coarse gradient），证明恰当选取的 STE 的期望粗梯度与真实梯度正相关、其负方向是下降方向。Moirai 用它解决 BNN 在线训练的核心难题：sign 函数几乎处处零梯度，STE 让梯度绕过整个网络的二值化器，成功更新所有层的潜在权重 W_raw（含 7-bit 与 4-bit 层），使 L1D 预取器能实时在线学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Moirai 的 BCC 反向计算（带 STE 的梯度流）：
  ```
  # 前向：A_bin = sign(A);  W_bin = sign(W_raw)
  # 反向（STE：把 sign 的导数近似为恒等直通）：
  dL/dW_raw = dL/dW_bin * 1      # STE：d(sign(x))/dx ≈ 1
             = ΔW_raw^k          # 式 2：ΔW_raw^k = G_{i+1} * Ac_i^k
  W_raw <- W_raw - lr * ΔW_raw   # 7-bit(首层)/4-bit(其余层) 潜在权重更新
  ```
  STE 让梯度 dL/dW_bin 直通到 W_raw，因此二值化不阻断学习；配合混合精度潜在权重（首层 7-bit 保特征保真度、其余 4-bit 省面积）保证梯度累积精度。
- 关键点：STE 是"近似梯度"，代理导数选择（恒等 vs 饱和 vs sigmoid 导数）影响训练稳定性；Moirai 与 LSQ 等量化方法都用 STE 让梯度穿过 round/clip 或 sign。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 中自定义 autograd Function（forward 用 sign/round，backward 返回恒等梯度）或 `torch.where`/直通写法；LSQ 等 QAT 库内置 STE 自定义算子。硬件实现：Moirai 的"Gradient Computation"块因前向权重已二值化，完全避开复杂乘法，综合为条件符号翻转器阵列 + 浅加法树（跨 K 通道累加）。使用场景：所有二值/量化网络训练（BNN、QAT、LSQ）、量化感知训练部署到 INT8 NPU；Moirai 首次把它放进 L1D 预取器的片上在线训练路径。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework

## XNOR-Popcount 位运算（二值化乘加）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- XNOR-Popcount 是把二值向量点积替换为按位 XNOR + 逐位计数（popcount）的运算范式：对二值向量 x_b,w_b ∈ {+1,−1}^n，有 x_b·w_b = 2·popcount(x_b XNOR w_b) − n（匹配位数×2 减 n）。它把 n 次浮点/整数乘法换成 1 簇位逻辑：64-bit 字打包后一条 XNOR + POPCNT 指令（x86/ARM）同时算 64 个"乘法"。这是 BNN（BinaryNet、XNOR-Net）硬件加速的基石：乘加阵列 → 位逻辑阵列，面积/功耗数量级下降。Moirai 的 CaPNet 依赖它把 L1D 场景下不可行的浮点 MAC 变成可综合的位逻辑：FCC 前向 y = bitcount(A ⊙ W_bin)（式 1），面积仅 1178 μm²。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 具体计算（Moirai 式 1，一次二值卷积输出）：
  ```
  # A_bin = [1,-1,1,1,-1,...], W_bin^k = [-1,1,-1,1,1,...]（±1 编码为 1/0 位）
  # XNOR：同号 → 1，异号 → 0
  xnor = ~(A_bin XOR W_bin^k)          # 逐位
  Ac_i^k = bitcount(xnor)              # 统计 1 的个数 = 匹配位数
  # 点积 = 2*Ac - n（n=向量长度；可直接用 Ac 作激活强度）
  ```
  例：A=[1,1,-1,-1,1,1,-1,-1]，W=[1,-1,1,-1,1,-1,1,-1] → XNOR=[1,0,0,1,1,0,0,1] → popcount=4 → 点积=2·4−8=0。硬件上 64 位字并行算 64 路，popcount 用加法树或 LFSR 计数器（Ishiura 等 SASIMI 2021 的紧凑 FPGA 实现，论文 [27] 引用）。
- 反向（式 2）：ΔW_raw^k = G_{i+1} * Ac_i^k 仍是普通乘（梯度为实值），故训练侧保留高精度（7/4-bit），只有前向被二值化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件侧用位打包 + SIMD（AVX-512/NEON 的 XNOR/POPCNT 指令）；硬件侧用 XNOR gate 阵列 + popcount 加法树（Moirai 的 FCC），FPGA 侧有专用 BNN 架构增强（Kim 等 FPT 2018，论文 [32]）；ASIC 侧可配内存内计算（近存/存内 XNOR）。使用场景：TinyML 边缘推理（XNOR-Net 类模型）、FPGA/ASIC BNN 加速器、以及 Moirai 的 L1D 预取器 CaPNet（前向 1-3 周期完成，2.5-4GHz）。局限：仅适用于权重/激活都二值化的层；精度依赖训练技巧（STE/缩放因子）。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework

## 地址 delta 序列特征（Delta-based Feature Representation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 地址 delta 特征是把连续访存地址的差（delta = addr_t − addr_{t-1}）作为建模对象，而不是原始 64-bit 地址。理由：原始地址缺乏平移不变性（同一模式在不同地址要重复学习）、特征空间巨大稀疏；delta 表示天然平移不变（stride +64 在不同基址都是 +64）且高度结构化。这在预取研究中历史悠久（Global History Buffer、BOP、MLOP、Berti 都用 delta）。Moirai 基于 SPEC 分析提出"delta 稀疏性"（delta sparsity）：按频率排序的 unique delta 中，top 5% 的"频繁 delta"覆盖 62.3% 的访存（cam4_s 达 82%，Figure 2），这让轻量网络只需把有限资源集中在少数关键模式上即可有效泛化；不规则 workload（pr、bc）分布平坦则触发控制单元暂停预取防污染。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- delta 序列 pipeline（Moirai）：
  ```
  # 原始流：addrs = [A0, A1, A2, ...]
  naive_deltas = [A1-A0, A2-A1, ...]        # 朴素 delta（受交替页访问/大跳变污染）
  # Window-based Extended Delta（Algorithm 1，ws=5，8KB 空间约束）：
  for i in 1..ws-1:
      if same_page_8KB(aw[0], aw[i]):       # (aw[0]>>13)⊙(aw[i]>>13)==1
          Δ = aw[i] - aw[0]; return Δ        # 只算低位的"真实局部 delta"
  return abnormal                            # 窗口内无同页 → 丢弃
  features = 滑动窗口取 10 个连续 Δ           # 输入 CaPNet
  ```
  该算法一次解决两个问题：交替页访问（多数据结构交错跳远页产生的振荡假 delta）与 delta 稀疏（异常控制流产生的大离群 delta 噪声）；spec 中交替页访问约占 60% 访存。8KB 空间约束是权衡：单页限制会截断跨页数据结构、放宽则膨胀 delta 词表并交叉污染独立流。
- 效果：干净的 delta 流让 TCN 聚焦频繁 delta 的转移学习；正则化交错形成可学习的新模式、不规则交错被卷积感受野平滑（表式预取器则单点噪声即断链）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件上只需位运算+减法（与运算判断同页、减法算 delta），无 PC 表存储（对比 PC-based stream splitting 需几百字节到 KB 的每-PC 状态表）。使用：作为 BNN/TCN 预取器的输入特征（Moirai 的 Input Processing Unit）；Berti 等表式预取器也按 IP 定位统计本地 delta 分布。局限：可能错过大的跨页 stride 模式（论文承认的权衡）；全局流（非 PC 分流）下的偶发"假 delta"靠 TCN 泛化能力兜底。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework

## Genome Graph 与 de Bruijn 图（DBG）查询（节点中心 vs 边中心、compacted DBG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Genome graph（基因组图）用图 walk 而非独立线性序列表示基因组数据库：把数据库条目间共享的子序列折叠成单一节点、相邻子序列用边连接，节点元数据（metadata）标明哪些条目包含该序列，可把序列集压缩上千倍，表达力强（编码进化历史与多样性、降低 bias、提高分析准确率），并利用冗余避免对共享序列的重复计算。与一般图（社交/网页/路网，幂律度分布、需显式存边、可节点重排）不同：基因组图节点度最多 4（A/C/G/T 固定字母表决定）、节点或边隐式获得、且"查询本身是生物序列"带来共享子串 k-mer 映射到索引附近区域的结构耦合局部性。de Bruijn 图（DBG）是重叠图的一种，近年大规模基因组图分析的主流：节点=唯一的 k-mer（长度 k 子串），节点 u→v 有向边当且仅当 u 的 (k-1) 后缀等于 v 的 (k-1) 前缀；compact DBG 把最大非分支路径（unitig）合并成单一节点以缩小规模。两种表示：节点中心（node-centric，如 Fulgor：存全部 k-mer 节点、边隐式定义）与边中心（edge-centric，如 MetaGraph：存 (k-1)-mer 节点、只存表示观察到的 k-mer 的边）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 查询 pipeline（k-mer set lookup，GRAINS 图 7 流程）：给定 read 提取 k-mer → 对 k-mer g 计算 minimizer（哈希最小的 m-mer）→ 用 minimizer 的最小完美哈希值 h 索引 Sizes → 由 Sizes[h+1]−Sizes[h] 得 Offsets 区间 → 读 Offsets 得 Strings 中页内偏移 → 在 Strings 的 k−m+1 窗口内找到 minimizer 并校验其余 k-mer → 命中则取 unitig 的颜色（元数据）。read mapping 分 alignment-free（全部 k-mer 匹配→汇总元数据→分类 read，大样本研究常用）与 alignment-based（k-mer 命中定位候选区域后再做近似字符串匹配/动态规划精化，可集成 SeGraM 对齐加速器）。GRAINS 的 DBG 数据布局：unitig 按预定顺序存连续字符串（Strings），Offsets/Sizes 辅助定位，颜色按 unitig 排序后用 Color Bitmap 标记每色起点。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 工具：Fulgor（节点中心，SSHash 字典，github.com/jermp/fulgor）、MetaGraph 框架（边中心，github.com/metagraph-rs/metagraph）、GRAINS（用 SSHash 做字典）。图构建自 MetaSUB Consortium 全球样本子集（G_MetaSUB，含颜色 Fulgor 659 GB / MetaGraph 822 GB）与 SRA 公共代表子集（G_SRArep，161/231 GB）。用途：物种/病原体鉴定（k-mer 集合查找）、个性化医疗、宏基因组与废水监测、群体规模病原体监测。DBG 是无损编码，各工具（Fulgor/MetaGraph/GRAINS/IdealAccMem）查询精度一致；变体图工具（VG、minigraph）在遗传多样性大时不缩放，不在大数据库场景使用。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design

## k-mer 与 Minimizer（基因组序列查询）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- k-mer 是 DNA/RNA 序列中长度为 k 的连续子串（k 常见 21–63），测序产生的 read 被切成全部 k-mer 作为与大规模数据库比对/查找的基本单元。Minimizer 是给定窗口内哈希值最小的 m-mer（m<k）：对每个 k-mer 取 minimizer 可得到位置确定的代表性子串，使"共享 minimizer 的 k-mer 在字典/索引中聚在一起"，作为路由键大幅减少索引比较量。k-mer minimizer 具有稀疏且偏斜分布的统计特性，是 SSHash 等紧凑 k-mer 字典（空间-时间折中）的设计基础。GRAINS 利用该特性做主机侧查询重排：按 Sizes 排序后连续 k-mer 共享同一 minimizer，传输时只存一次 minimizer、其余只留差分，将 host→SSD 传输数据量平均压缩 2.3×。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 查询/压缩流程（GRAINS 图 7/9）：① 对每个 read 提取全部 k-mer（2-bit 编码，每 k-mer 一个整数）；② 对 k-mer g 求 minimizer：在 g 的 k−m+1 个 m-mer 中取哈希值最小者（如 m=15 于 k=31）；③ 用 minimizer 的最小完美哈希 h 索引 Sizes[h]；④ 排序/分批后，批次内连续 k-mer 共享 minimizer 时只传 minimizer+差分（例：同一 minimizer 下 5 个 k-mer 只传 1 个 minimizer + 4 个差分子串）；⑤ die 内 IFP comparison 把 k-mer 与 Strings 窗口做位级逐位匹配。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：k-mer 提取把 read 每个长度 k 窗口编码为 2-bit 整数；minimizer 用滚动哈希或最小哈希（如 MinHash/ntHash 类）对窗口内 m-mer 取最小。工具：SSHash（jermp/sshash，C++，2-bit 编码、默认 k≤31 可编译到 63）、Fulgor（Seeding 用 minimizer，类似 MinSeed）。使用场景：k-mer 集合查找、read mapping seeding、序列去重、宏基因组分类。GRAINS 强调 k-mer 提取在 host 完成（SSD 内频繁写会缩短 NAND 寿命），压缩/排序也是 host 软件步骤。
- Lembas 补充视角（ISCA'26，Minimap2 播种阶段）：Minimap2 在滑动窗口内取**字典序最小 k-mer 作为 minimizer 种子**，用**内存哈希表**做随机查找发现匹配 anchor——哈希访问不可预测随机、表必须整体驻留内存，是 seed 阶段内存容量瓶颈（Minimap2 靠 memory chunking 限内存但跨 chunk 不做匹配检查、降低输出质量）。Lembas 的播种加速器**完全移除哈希表**：minimizer parse 产出 16 B 〈minimizer, index〉 元组流，经 PCIe 溢出到 NVMe，用外部内存 columnsort 按 minimizer 字典序全局排序，reference/query 两有序流做**流式 zip 匹配**（顺序扫描）得 anchors → 内存需求恒定 ~8 GB（7× 降低）。代价：刻意不做 Minimap2 的启发式 anchor 过滤（需随机访存），下游工作量放大（人类基因组 7.06× 更多 chains）。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration

## SSHash 与最小完美哈希函数（MPHF）（k-mer 字典）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 最小完美哈希函数（MPHF）是双射哈希：把 n 个键（这里为 n 个不同 k-mer）无碰撞地映射到 [0,n)，无空桶，空间接近理论下界（~1.44 bits/key，PTHash 等实现）。SSHash（Sparse and Skew Hashing of K-Mers，Pibiri 2022，Bioinformatics i185–i194）是基于 MPHF 的"压缩、关联、精确、带权"k-mer 字典：利用 k-mer minimizer 的稀疏与偏斜分布 + 最小完美哈希 + 紧凑编码，得到比此前序列字典明显更好的空间-时间折中，存储 unitig 为连续字符串并支持精确成员查询（每个字符串关联 [0,n) 唯一整数 ID）。GRAINS 采用 SSHash 作为 DBG 骨干字典（与 Fulgor 相同选择），并强调其技术依赖一般 DBG 性质、也可用于其他 k-mer 字典骨干。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- GRAINS 的查询利用 SSHash 的 Sizes 数组远小于 Offsets/Strings 的特性（Sizes 仅占大图 <4% 空间）：主机侧先查 Sizes 拿到 Offsets 索引、排序分批后再发 SSD，把对 Offsets 的访问变成顺序流——这是 Genome-Graph-Aware Query Reordering 的物理前提。查询流程伪代码：`h = mphf(minimizer(g))` → `lo = Sizes[h], hi = Sizes[h+1]` → 对 `i in [lo, hi)` 读 `Offsets[i]` → 在 `Strings` 的 `Offsets[i]` 处窗口 `k−m+1` 内比对 minimizer 与 k-mer → 命中则 unitig ID 即为该 k-mer 的关联整数 ID，取其颜色。注意：虽然整图太大无法进 host（§3 动机分析），Sizes 只占小部分，可在 host 安全驻留处理。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MPHF 由 PTHash 库（jermp/pthash，C++）构建；SSHash 用 2-bit 核苷酸编码、CMake 构建（`git clone --recursive https://github.com/jermp/sshash.git`，`-DSSHASH_USE_MAX_KMER_LENGTH_63=On` 支持 k≤63，`-DSSHASH_USE_ARCH_NATIVE` 提性能）。使用：构建对 k-mer 集的 MPHF 与 Strings/Offsets/Sizes 索引 → 查询时按 minimizer→h→Sizes→Offsets→Strings 定位。GRAINS 把 SSHash 结构存进 SSD（Strings/Offsets 低复用大数据放 NAND、Sizes 可放 host），并为其设计存储友好布局与调度。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design

## Cross-Read K-Mer Batching 与 Genome-Graph-Aware Query Reordering（GRAINS 主机侧查询优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GRAINS 的两项主机侧算法优化：(1) Cross-Read K-Mer Batching——不再逐条 read 查询，而是把同一 read set 中不同 read 的 k-mer 合并批量查询，用轻量数据结构维护 k-mer→所属 read 的映射，图查询返回匹配 k-mer 的颜色后按 read 汇总，从而成量减少对图节点的随机访问次数（利用"不同 query read 的共享子串 k-mer 映射到索引附近区域"的基因组图特性）；(2) Genome-Graph-Aware Query Reordering——利用最小完美哈希 k-mer 字典（SSHash）中 Sizes 数组远小于 Offsets/Strings 的特性，先在 host 用 Sizes 完成 k-mer 查找、拿到 Offsets 索引，据此把 k-mer 排序并切成等长 disjoint 批次，使 SSD 侧对 Offsets 的访问变成顺序流；排序时利用"按 Sizes 排序后连续 k-mer 共享同一 minimizer"只存一次 minimizer+差分，把 host→SSD 传输量平均压缩 2.3×。二者与数据传输/查询构成流水线：一批排序与上一批传输和 Offsets 查询重叠。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 执行流程（图 9）：① host 提取每 read 的 k-mer；② host 查 Sizes（用 minimizer 的 MPHF h）得到 Offsets 索引；③ 按 Sizes 值把 k-mer 切成等长 disjoint 批次、排序（该批次排序与上一批传输/查询流水线重叠）；④ 批次内同 minimizer 只传一次+差分压缩；⑤ 批次经标准 NVMe 数据路径送 SSD 内部 DRAM（不写 flash）；⑥ SSD 顺序访问 Offsets、经 GST 调度查 Strings、ISP 扫 Color Bitmap；⑦ 结果回 host，按 k-mer→read 映射把颜色汇总到每个 read 完成分类。k-mer 提取与排序在 host 完成（SSD 内频繁写会缩短 NAND 寿命）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 GRAINS 查询准备阶段的 host 软件（AMD EPYC 7742 + 1.5 TB DRAM 实测），排序/压缩开销由流水线隐藏（论文称需保证排序与传输不引入显著开销）。消融验证：GRN-B（仅 batching）平均 1.5×/2.2× 超 FG/MG（改善 Offsets 访问）；GRN-B-S（batching+scheduling）再 2.3×（Strings/Colors 存储友好）；GRN-B-S-SCC 完整版再 2.0×。GRN-Ext（优化在 SSD 外、PCIe 16 GB/s）也因存储友好执行流获得 3.4×/5.0× 平均加速，证明优化本身（不依赖 ISP/IFP）即有价值。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design

## 3D Gaussian Splatting（3DGS，三维高斯泼溅）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 3DGS（Kerbl et al., SIGGRAPH 2023）把场景表示为大量各向异性三维高斯椭球原语的集合，替代传统显式网格或隐式神经场（NeRF）。每个原语由中心 μ、3D 协方差矩阵 Σ（分解为缩放矩阵 S∈R^{3×3} 与旋转矩阵 R∈SO(3)，存紧凑形式：四元数 q∈R^4 + 缩放向量 s∈R^3）与不透明度 o 参数化，高斯响应为 G(x) = o·exp(-½(x-μ)^T Σ^{-1} (x-μ))（式 1），Σ = RSS^T R^T（式 2）。渲染时把 3D 高斯经 world-to-camera 变换后投影到像平面（局部仿射近似，丢弃变换后协方差第 3 行第 3 列得到 2D 泼溅），按深度从前到后做体积 alpha 混合（式 3：C = Σ T_i α_i c_i）。参数通过光度损失训练拟合 ground-truth 图像。GauTracer 论文中 3DGS 是 ray-oriented 渲染（3DGRT）的算法基础：3D 高斯在单位球空间中的"射线-高斯交点"定义为高斯响应沿射线最大点（原点在射线上的投影）。
- 从算法pipeline角度拆解术语，给出伪代码或具体计算过程例子：3DGS 推理 pipeline = 高斯参数加载 →（每像素/tile）把高斯按 view matrix 投影 → 计算 2D 协方差 → 计算不透明度贡献（2D 高斯泼溅在像素上的积分）→ 深度排序 → front-to-back alpha 混合累加颜色与透射率 → 提前终止（透射率低于阈值）。伪代码示意：
  ```
  for tile in image:
      gaussians = tile 内投影覆盖的高斯（由 3D AABB 剔除）
      sort(gaussians, by depth)
      T, C = 1.0, 0
      for g in gaussians:
          alpha = g.opacity * splat(g, pixel)   # 2D 高斯积分值
          C += T * alpha * g.color
          T *= 1 - alpha
          if T < threshold: break
  ```
  训练侧：photometric loss（L1 + SSIM）反向传播更新 μ, q, s, o 与颜色（SH 系数），并做自适应密度控制（clone/split）。GauTracer 论文按式 1-3 给出响应与混合定义，并指出 3DGS 是 tile-based 光栅化管线，天生受限无法直接支持 ray-oriented 渲染（畸变相机、二次光线）。
- 术语一般如何实现？如何使用？：主流开源实现为原始 3DGS（https://github.com/graphdeco-inria/gaussian-splatting，diff-gaussian-rasterization CUDA 光栅化 kernel），2DGS（https://github.com/hbb1/2d-gaussian-splatting），以及 ray-oriented 的 3DGRT/3DGUT（https://github.com/nv-tlabs/3dgrut，OptiX+Slang）。GauTracer 论文用 3DGS 的 ray-tracing 变体（3DGRT [27]）作为算法 baseline，BVH 用 Intel Embree 构建（分支因子 6），NeRF-Synthetic 数据集 8 场景 30,000 次训练迭代的点云做质量评估。硬件侧（GScore [29]、GCC [33]、GausPU [36] 等）研究 rasterization 的软硬件协同，GauTracer 则把 3DGS 作为一等原语做 RTA 硬件扩展。

NeRArch-Sim 补充视角（ISCA'26，3DGS 作为 primitive-based pipeline 的模拟对象）：NeRArch-Sim 把 3DGS pipeline 按分类学分解为 Field Sampler（视锥剔除）→（无 Encoding，原语自带属性）→ Field Computation（SH 颜色）→ Blending（排序 + alpha 混合），作为 splatfacto（Nerfstudio 3DGS 模型）算子图被插桩提取；复现的 3DGS 加速器包括 GSCore（tile 化排序-光栅化重叠，FPS 190→182.2，误差 4.1%）、GBU（180→172）、Uni-Render（65→63，3DGS pipeline 评估 PSNR 33.0/33.0）、GS Processor（373→343，与流片芯片对比延迟误差 8.0%）。内存侧（表 VIII）：GSCore 每 tile 从 DRAM 流式读 78.8MB 高斯特征（Gaussian In FIFO），排序缓冲 12.4MB SRAM 读写，Pixel Out Buffer 写 3.1MB；bank conflict 开销仅 0.01%（顺序 tile 处理几乎无争用）。DSE 案例即在 GSCore 上用模拟退火调 (Culling Conversion Units, Quick Sorting Units, Bitonic Sorting Units, VRCs, Buffer sizes) 五参数，最优 (16,8,4,32,4) 相对原配置 (4,8,4,64,8) 取得 1.3× energy-delay product 与 1.6× 面积下降。

3DGS 加速器补充视角（ISCA'26，算法-硬件 co-design 的光栅化/排序优化）：论文 profiling（Jetson Orin Nano + MipNeRF-360）显示 projection/sorting/rasterization 各占 14.2%/25.3%/60.5% 延迟；α-computation 每像素/Gaussian 需 8 MUL+4 ADD+1 EXP（α=o·exp(-½(p-μ)ᵀΣ⁻¹(p-μ))，Σ⁻¹ 圆锥矩阵用 a,b,c 参数化），α-blending 需 5 MUL+4 ADD。两项优化：(1) axis-shared rasterization——把 α 指数分解为 X/Y 二次项+交叉项，X-PE/Y-PE 线预计算共享项广播给 16×16 PE 阵列，α-computation 摊销降至 2.31 MUL+2.13 ADD/PE（-63%），总 MAC 减 38%；(2) MLP-based OIT——2 层 10 参数 MLP（输入深度 d_i+视角 (x,y,z)，推理仅 6 MAC，指数输出激活）直接预测透射率 F(d_i) 替代显式排序，渲染式改顺序无关 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i，PSNR 26.90 vs 排序 baseline 27.21（-0.3）、优于 weight-sum[18]（25.43），训练约 30 分钟/场景（预训练 checkpoint 初始化，MLP lr=0.005、GS lr×0.01，关闭 cloning/splitting）。统一可重构 PE 阵列加速器 3.85mm²/1.64W（28nm、1 GHz、256 PE、96KB SRAM、DDR5-4800），相对 GSCore/MetaSapiens/GBU 1.33~1.88×、相对 edge GPU 端到端 4.0~5.5× 加速与 16.2~31.9× 能耗节省（150+ FPS）；GPU 实现基于 gsplat（https://github.com/nerfstudio-project/gsplat），代码开源 https://github.com/WangZhican/ISCA26_3DGS_Acc。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

## 2D Gaussian Splatting（2DGS，二维高斯泼溅）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 2DGS（Huang et al., SIGGRAPH 2024）由 3DGS 退化而来：把 3D 椭球的 z 方向缩放分量 s_z 置零，3D 椭球退化为平面椭圆（surface-aligned 的 2D 原语），天然适合表达表面几何与法线估计，视图一致性好，常用于几何敏感任务。与传统在质心近似深度不同，2DGS 显式计算像素 (x,y) 视线与 2D 椭圆平面的精确交点 (u,v)（式 4：u = (h_u[2]h_v[4]-h_u[4]h_v[2])/(h_u[1]h_v[2]-h_u[2]h_v[1])，其中 h_u=[-1,0,0,x]·P、h_v=[0,-1,0,y]·P，P∈R^{4×4} 为椭圆空间到世界空间的变换矩阵），保证视图一致的深度评估。GauTracer 论文把 2DGS 用于 ray tracing（2DGRT/IRGS [28]）：射线与 2D 椭圆的交点是显式平面交点（变换后 z 轴垂直于 2D 平面，t_hit = 变换后射线原点的 z 分量 / 变换后射线方向的 z 分量）。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：2DGS 推理 pipeline = 高斯参数加载 →（每像素）变换到椭圆局部空间 → 求视线与 z=0 平面的交点 (u,v)（式 4 的射线-椭圆求交）→ 在该 (u,v) 处评估 2D 高斯响应（比 3D 投影-积分更精确）→ 得到 alpha → 深度排序 alpha 混合。伪代码示意：
  ```
  # 每条像素射线与 2D 椭圆平面求交
  t_hit = -O.z / D.z          # 变换空间中，z=0 平面交点
  P_uv  = O + t_hit * D       # 交点坐标
  u, v  = P_uv.x, P_uv.y      # 椭圆局部 2D 坐标
  alpha = opacity * exp(-0.5*(u^2+v^2))   # 在 (u,v) 处的高斯响应
  ```
  GauTracer 的 RGIU 在 2DGS 模式下就是执行这一求交（z 分量除法 + 沿射线前进取 (u,v)），与 3DGS 模式（原点投影）共享点积/MAC 单元，仅由模式开关切换。
- 术语一般如何实现？如何使用？：开源实现为 2DGS（https://github.com/hbb1/2d-gaussian-splatting）与 2DGRT/IRGS（C. Gu et al., CVPR 2025；相关仓库 fudan-zvg/gtracer 与 fudan-zvg/gaussian-raytracing 为 OptiX 实现）。GauTracer 论文把 2DGRT 作为第二种评估模式：RGIU 无需结构性改动即可支持 2DGS（单 bit 模式开关 + 复用计算单元），2DGRT 模式下平均加速 7.3×（硬件 shader 贡献 5.6×）。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation

## Volumetric Alpha Blending（体积 alpha 混合 / 体积泼溅合成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 体积 alpha 混合是 Gaussian splatting 的最终合成步骤：把沿视线/射线方向排好序的高斯颜色按"不透明度 alpha + 累积透射率 T"从前到后（front-to-back）累加，得到像素最终颜色。标准公式（式 3）：C = Σ_{i=1..N} T_i α_i c_i，T_i = Π_{j<i} (1-α_j)。T_i 是到达第 i 个高斯前的剩余透射率（前面高斯的遮挡积累），α_i c_i 是第 i 个高斯对光线的贡献，本质是离散化的体渲染（ray marching）积分。GauTracer 论文用该混合更新射线颜色与透射率，并在 closest-hit shader 中做 front-to-back 混合、以透射率低于阈值作为 ray-gen 循环的提前终止条件。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：ray-gen 循环（GauTracer Alg. 1/3）：每条射线迭代 traceRayEXT，每次收集 K 个命中高斯，closest-hit shader 把 K 个命中按深度排序后 front-to-back 混合：
  ```
  # front-to-back（baseline，式 3）
  T, C = 1.0, 0
  for entry in ClosestHit[rayID][0..N_hit]:
      alpha = entry.alpha; color = GaussParam[entry.GID].color
      C += T * alpha * color
      T *= 1 - alpha
  ```
  GauTracer 的 AGHU 输出 far-to-near 序列（max-heap 弹出根=最远），为避免再排序采用 back-to-front 混合（式 7）：C^p_{i+1} = α_i c_i + (1-α_i) C^p_i，C^p_0=0，再以射线透射率缩放合入像素：C += T·C^p；透射率按 round 级更新 T *= Π(1-α_i)，实现 round 级（而非逐高斯）提前终止。该近似只影响透射率权重低于 0.001 的高斯，PSNR 几乎无损（33.40→33.32 dB）。
- 术语一般如何实现？如何使用？：软件上在 closest-hit shader（Vulkan）或 OptiX 内核（3DGRT）中循环混合；光栅化侧在 fragment shader 或专用光栅化 kernel 中按 tile 深度排序混合。硬件上，GauTracer 把"排序+混合"拆给 AGHU（排序）与 closest-hit shader（混合）；相关硬件工作（Gaussian Blending Unit [30]、GScore [29]）也在硬件中做高斯混合。混合正确性依赖命中排序，这正是 OIT（order-independent transparency）类近似（Local-GS [71]）在 ray tracing 中不可行的原因——ray tracing 需要精确命中距离决定二次光线生成位置。

涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 加速器补充视角（ISCA'26，OIT 用 MLP 预测透射率替代排序混合）：本论文重新审视混合公式 C=ΣT_iα_ic_i、T_i=Π_{j<i}(1-α_j)，指出排序的唯一目的是算正确的累积透射率 T_i（深度递增则 T 递减的衰减因子），并把 α-blending 与图像合成（"over" 算子，3DGS 前到后混合 == 图像合成后到前）类比，引入顺序无关透射率（OIT）渲染式 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i（式 5），F(d_i) 由 2 层 10 参数 MLP（输入深度+视角方向，指数输出激活，推理 6 MAC）预测；质量 PSNR 26.90 vs 27.21、SSIM 0.8263 vs 0.8309、LPIPS 0.1739 vs 0.2017（略优），且优于 handcrafted depth-function 的 weight-sum[18]（25.43）。硬件上 α-blending 在可重构 PE 中实现：M-3 乘 F(d_i)α_i、A-3 累加分母，M-4-{1~3} 乘 RGB、A-4-{1~3} 累加 RGB 分子，除法阵列归一化；相对 32 并行 bitonic 排序网络 21.1~32.4× 加速。与 GauTracer 的结论（OIT 类近似在 ray tracing 中因需精确命中距离而不可行）形成对照：本论文证明 tile 光栅化场景下 OIT 可行且质量损失小。

## 零知识证明（ZKP）与 zk-SNARK / zk-STARK

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ZKP 是密码学协议：prover 向 verifier 证明某个关于私有 witness 的 statement 为真，而不泄露 witness 本身（零知识性）。现代应用使用 succinct 形式：zk-SNARK（Succinct Non-interactive Arguments of Knowledge，证明大小 polylog/常数、验证毫秒级、可非交互）与 zk-STARK（Scalable Transparent Arguments of Knowledge，透明设置、抗量子，但证明对数级更大、验证更慢）。证明生成（prover）计算量大，是硬件加速的目标。GenZA 采用标准三层视图描述现代 ZKP 证明生成：算术化（Arithmetization）→ 多项式交互式预言机证明（PIOP）→ 多项式承诺方案（PCS）；PCS 经 Fiat-Shamir 变换后协议变为非交互。
- 本论文角色：GenZA 面向三类代表性协议 Groth16/HyperPlonk/Plonky2 的证明生成阶段，按三层视图识别出 dominant kernels（NTT、MSM、sumcheck、Merkle tree、多项式运算、hash），并设计统一可重构硬件覆盖这些 kernel。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 证明生成 pipeline（以一个电路 statement 为例）：
```
1) 算术化：把程序/电路展平成代数约束（R1CS 或 Plonkish 表格），生成 witness 向量 w
2) PIOP：把约束编码成多项式，prover 对 witness 多项式求值/承诺，
   与 verifier 交互若干轮（sumcheck/NTT 域切换）验证低度、置换、积/零性质
3) PCS：绑定多项式并打开选定点（KZG 用 MSM+配对，FRI 用 Merkle 树+哈希）
4) Fiat-Shamir：用哈希挑战替代交互，输出非交互证明
```
- Annotations：各协议只替换某层——Groth16 用 R1CS+NTT+MSM(KZG)，HyperPlonk 用 Plonkish+sumcheck+MSM(KZG)，Plonky2 用 Plonkish+NTT+Merkle/FRI。证明生成的时间占比（Table II）：MSM/Merkle 各占 59–70%，NTT/sumcheck/多项式各占 1–33%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 上用 libsnark/Jsnark（Groth16）、官方 HyperPlonk/Plonky2 库；GPU 用 GZKP/cuZK/plonky2-gpu 等；ASIC 用 PipeZK/SZKP/zkSpeed/UniZK/LegoZK/GenZA 等加速器。使用场景：隐私区块链/rollup（Zcash、zkEVM）、可验证云计算、ZKML、匿名投票、递归证明组合（内层 Plonky2 快协议 + 外层 Groth16 恒定大小验证）。硬件侧关键点：不同协议场/bitwidth/kernel 差异大（64-bit Goldilocks 到 768-bit EC 域），单一专用单元无法覆盖全部，需统一可重构架构。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

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

## 多项式交互式预言机证明（PIOP，Polynomial Interactive Oracle Proof）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PIOP 是 ZKP 三层视图的第二层：prover 把 witness 承诺为多项式集合，verifier 查询少量随机点上的求值，以高健全性检查低度关系、置换约束与积/零性质。prover 侧多项式计算两种方式：(1) 用数论变换 NTT 在系数域/求值域间切换，把多项式乘法变成逐元素操作；(2) 用布尔超立方上的多线性扩展（MLE）结合 sumcheck 协议，避免大 NTT。代表性 PIOP：Groth16 用线性 PCP（NTT 域）、HyperPlonk 用 MLE+sumcheck、Plonky2 用 PLONK 风格（NTT 域）+FRI。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- PIOP prover 计算例子（多项式乘法域切换）：系数域多项式 a(x)、b(x) → NTT 转求值域 â、b̂ → 逐元素乘 ĉ=â⊙b̂ → 逆 NTT 回系数域得 c(x)=a·b。MLE+sumcheck 路径：把约束写成多线性多项式 g(x1..xn)，prover 对 g 在超立方上的和 ∑g 用 sumcheck 协议逐轮约简（见 sumcheck 条目）。
- Annotations：NTT 路径是计算密集型（logN 级蝴蝶）；MLE/sumcheck 路径是访存密集型（向量逐元素操作）。GenZA 中 NTT 与 sumcheck 都映射到同一批 PE 的不同模式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Groth16（libsnark）、HyperPlonk（EspressoSystems/hyperplonk）、Plonky2（mir-protocol/plonky2）各自实现其 PIOP；硬件加速器把 PIOP 翻译成计算图（GenZA 手动翻译，可交给 ZKP 编译器）再调度到 PE。使用：决定 prover 的 kernel 构成（NTT vs sumcheck vs 多项式操作）与访存/计算特征，是加速器 kernel 映射设计的主要输入。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## 多项式承诺方案（PCS）：KZG 与 FRI

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PCS 是 ZKP 三层视图的第三层：把多项式绑定成承诺，之后用简洁证明打开选定的求值。两种主流实例：(1) 配对型 KZG（Kate 等）——基于椭圆曲线与配对，需可信设置，常数大小证明、极快验证，典型用 MSM kernel；(2) 哈希型 FRI——基于 Merkle 树承诺与低度测试，透明（无需可信设置），但证明对数级更大、验证哈希重。PCS 结合 Fiat-Shamir 使协议非交互。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- KZG 承诺例子：对多项式 f(x)=Σ f_i·x^i，承诺 C = Σ f_i·[τ^i]_1（τ 为 SRS 秘密，[]_1 为群元素）——这是对系数向量的 MSM；打开 f(α) 时验证配对等式 e(C−f(α)G, H)=e(w, τH−αH)。FRI 例子：把 f 的 Merkle 根作承诺，逐轮把多项式折叠并抽样承诺，最后做低度测试。
- Annotations：KZG 的 MSM 是 Groth16/HyperPlonk 的最大 kernel（占 prover 时间 59–70%）；FRI 的 Merkle 树+哈希是 Plonky2 的最大 kernel（68.84%）。GenZA 对 MSM 做动态 window sizing 与 window-major 映射，对 Merkle/哈希用向量 PE+树分片。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：KZG 用 libsnark/Bellman（zkcrypto）、MSM 加速用 PipeZK/SZKP/GenZA；FRI 用 Plonky2/starky、Merkle 哈希用 Poseidon（Plonky2）。使用场景：按应用权衡选择——加密货币/rollup 偏爱 KZG 小证明快验证（接受可信设置），可验证云计算/ZKML 偏爱 FRI 透明设置快证明。硬件：PCS 阶段 kernel 决定加速器的主要单元需求（EC 运算 vs 哈希/向量）。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## Fiat-Shamir 变换

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Fiat-Shamir 把交互式证明变成非交互式：用对"当前协议状态（前几轮消息+statement）的哈希"作为 verifier 随机挑战，替代真实 verifier 的随机数。在 ZKP 中，PIOP/PCS 的每轮挑战（如 sumcheck 的 r_i、KZG 打开点）都由哈希（如 SHA-256、Poseidon）从已通信消息派生，使证明可在单条消息中发出（非交互），是 zk-SNARK/STARK 可落地的关键。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- sumcheck 中的 Fiat-Shamir 例子：prover 算 g1(x1) → 挑战 r1 = H(statement || g1) → 算 g2(x2) → r2 = H(statement || g1 || g2) → ... 取代"verifier 随机发 r_i"。硬件影响：哈希挑战形成严格串行点——后续 kernel 依赖该挑战，故 GenZA 的调度器把 Fiat-Shamir 变换（hash 计算状态生成下一挑战）识别为一类 cut 点，kernel 流水线必须在此处序列化。
- Annotations：GenZA 为 HyperPlonk 内置 SHA3 core（<1% 时间）做 Fiat-Shamir，Poseidon 哈希也可用于此；挑战的串行依赖限制了跨 kernel 流水线的自由度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：通用哈希（SHA-256/SHA-3）或 ZK 友好哈希（Poseidon）；在 libsnark/HyperPlonk/Plonky2 中以"transcript"对象累计消息并派生挑战。使用：所有非交互 ZKP 的标准组件；加速器上以专用小 hash core 或复用向量 PE 实现（GenZA 用 SHA3 core，面积 0.01mm² 级）。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## Sumcheck 协议

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Sumcheck 是证明布尔超立方上多线性多项式之和 ∑_{x∈{0,1}^n} g(x) = C 的交互协议（Lund-Fortnow-Karloff-Nisan 1992），是 MLE 类 PIOP（HyperPlonk、Spartan）的 prover 核心。协议 n 轮：第 1 轮 prover 算 g1(x1)=∑_{x2..xn} g(x1,...,xn)，verifier 检查 g1(0)+g1(1)=C0=C 并发挑战 r1 固定 x1；第 i 轮算 gi(xi)=∑_{x_{i+1}..xn} g(r1,...,r_{i-1},xi,...)，检查 gi(0)+gi(1)=C_{i-1} 并发 ri；Fiat-Shamir 后非交互。prover 计算本质是向量逐元素求和（访存密集型，算术强度低）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- prover 一轮的向量计算例子（g = qL·wa + qR·wb + ... 因子多项式之和）：逐轮对剩余变量求和，每轮把因子多项式逐元素相乘相加。GenZA 用两个优化压缩访存：(1) 延迟绑定（delayed binding）——稀疏 0/1 系数与稠密挑战分开存储，逐轮 on-the-fly 绑定（只在向量长度足够短后才物化稠密 gi），2^23 实例流量 2.9→0.7 GB；(2) 等号多项式空间压缩（eq-poly space reduction）——e~q(w,X) 用 O(√N) 工作存储 on-the-fly 求值替代 O(N) 物化，再省 1.3× 流量。
- Annotations：zkSpeed 的固定功能 sumcheck 单元把所有系数当稠密场元素处理、忽略稀疏性；NoCap 的 64-bit 向量单元使延迟绑定收益天然小（收益正比于场元素 bitwidth）。GenZA 首次在专用加速器中实现这些算法优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 库（HyperPlonk 官方实现、Speeding Up Sum-Check Proving [4]）；硬件按向量 PE 映射（逐元素加/乘、树式求和、分段并行 segmented-parallel 处理串行归约链）。使用：MLE 类 PIOP（HyperPlonk/Spartan）的 prover 主计算；访存受限，硬件需减少 off-chip 流量与提高 PE 利用率。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## Poseidon 哈希函数

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Poseidon 是面向 ZKP 的 ARX 类哈希（Grassi-Khovratovich-Rechberger-Schofnegger, USENIX Security 2021）：把消息映射到有限域元素并用域算术（S-box + 线性层）做压缩，比 SHA-2 在电路规模/证明成本上低数量级，广泛用于 Merkle 树、Fiat-Shamir 挑战、zkEVM。GenZA 遵循 Plonky2 设置：64-bit Goldilocks 域、状态宽 t=12、x^7 S-box，含若干 full 与 partial rounds，底层原语为 S-box 与稠密/稀疏 MDS 矩阵-向量乘。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一轮计算例子：S-box 层把每个状态元素求 x^7（4 个乘法器的快速幂链，x^7=x^4·x^2·x）；稠密 MDS 矩阵-向量乘 t×t·state 拆成 t=12 个独立点积，每个点积长 12 由 4 个乘法器+归约链 3 轮完成，多点积流水；稀疏 MDS 乘拆成若干点积与逐元素乘。GenZA 把每 PE 分成 8 个 1D Poseidon 单元（各 4 个 64-bit 乘法器匹配 Goldilocks），对比 UniZK 的 2D systolic 在 full round 损失 25% 利用率，1D 向量更紧凑。
- Annotations：Poseidon 的 MDS 常数预装进 PE scratchpad；S-box 的乘法链充分使用 4 乘法器避免串行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Plonky2/starky 内置 Poseidon 参考实现；硬件用向量/systolic 阵列（UniZK 2D、GenZA 1D 向量）。使用：Plonky2/FRI 类协议的 Merkle 树哈希与 Fiat-Shamir（GenZA 为忠实匹配 HyperPlonk 另留 SHA3 core，但 PE 原生支持 Poseidon）；hash 类 kernel 算术强度高（191 modmul/元素）但需大量并行向量单元。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## Merkle 树

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Merkle 树是哈希树：叶为数据块的哈希，内部节点为子节点哈希之哈希，根承诺整棵树。FRI 型 PCS（Plonky2）用 Merkle 树承诺多项式求值表，打开某点需给出叶子到根的兄弟路径哈希。Plonky2 中 Merkle 树是最大 kernel（占 prover 时间 68.84%），算术强度高（191 modmul/元素）但树结构导致并行度在近根处下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 构建/打开例子（N 个叶子，哈希 h）：
```
build: for level in 1..log2(N): 每对兄弟 h(左||右) → 父节点（并行）
open: 对叶子 i，输出路径 (i, 兄弟哈希序列) → verifier 逐层重算到根
```
- Annotations：GenZA 把树式 workload 分成子树 fit 片上，每层节点并行处理；打开路径的哈希是串行链（沿路径逐层），用向量 PE 并行处理多条路径。树的每层并行度按 2 递减，近根层是资源利用率瓶颈——GenZA 的混合时空映射（见硬件架构库）正好缓解此类低并行度场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Plonky2 用 Poseidon 哈希建树；硬件按"分片子树+层内并行"映射（GenZA Section VI-E）。使用：FRI 承诺与打开、区块链轻客户端验证等；内存/带宽友好型（逐层流式），对哈希吞吐要求高。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## Groth16 / HyperPlonk / Plonky2（代表性 ZKP 协议）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 三协议覆盖现代 ZKP 的主要权衡维度（可信设置、证明大小、证明/验证时间），是 GenZA 的目标协议：(1) Groth16（EUROCRYPT 2016）——最优化 zk-SNARK，R1CS+线性 PCP+NTT+KZG，每电路可信设置，证明 3 个群元素、3 个配对验证，256/384/768-bit EC 域；(2) HyperPlonk（ePrint 2022/1355）——Plonkish+MLE/sumcheck+KZG，通用（电路无关）可信设置，近线性时间 prover，256/384-bit EC 域，无大 NTT；(3) Plonky2（Polygon Zero 2022）——Plonkish+PLONK 风格+FRI，透明设置，64-bit Goldilocks 域（p=2^64−2^32+1），面向快速递归，大证明但生成快。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 三协议 kernel 构成对比（Table II，2^20 门 mock circuit，80 线程 CPU 实测）：Groth16——MSM 69.73%、NTT 29.32%、多项式 0.96%；HyperPlonk——MSM 59.34%、sumcheck 33.49%、多项式 7.14%；Plonky2——Merkle 树 68.84%、多项式 14.17%、NTT 0.15%。位宽/场：Groth16/HyperPlonk 用 BN128/BLS12-381/MNT4-753（256–768 bit），Plonky2 用 Goldilocks（64 bit）。
- Annotations：三协议揭示"多样性"——硬件必须同时支持 64 到 768-bit 多 bitwidth、通用与特殊模、计算密集（MSM）与访存密集（NTT/sumcheck/多项式）kernel，这是 GenZA 统一架构设计动机的直接来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Groth16 用 libsnark/Jsnark（CPU）、GZKP（GPU）、PipeZK/SZKP/LegoZK（ASIC）；HyperPlonk 用 EspressoSystems/hyperplonk、zkSpeed（ASIC）；Plonky2 用 mir-protocol/plonky2、plonky2-gpu（GPU）、UniZK（ASIC）。GenZA 在一套 16×8 PE 阵列上运行全部三协议。使用场景：隐私币/rollup（Groth16 小证明）、可验证云计算/ZKML（HyperPlonk/Plonky2 快证明）、递归组合（内层 Plonky2+外层 Groth16）；云端需同时服务多协议客户，正是 GenZA 通用性价值所在。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## GAS（Gather-Apply-Scatter，图处理编程模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GAS 是图处理的高层编程抽象（源自 PowerGraph）：每次迭代分为 Gather（从邻居聚合消息）→ Apply（用聚合结果更新顶点状态）→ Scatter（把更新传播到邻居），迭代至收敛。有 vertex-centric 与 edge-centric 两个变体；edge-centric 变体以流式顺序访存适合 HBM/FPGA 加速（ACTS、GraphLily、ForeGraph、Swift 等广泛采用，Web 佐证：Swift 的 decoupled-asynchronous GAS 在 8-FPGA 上 12.8× 优于 ForeGraph）。Graph.hls 论文把 GAS 作为 DSL 的 baseline 表达力基准：GAS 是 Graph.hls Frontend 的特例（Scatter≈iteration_input+map 边流、Gather≈reduce(可交换可结合 lambda)、Apply≈归约后 map 读 self 属性），且 DSL 是其超集。
- 关键局限（Graph.hls 动机）：GAS 的 undifferentiated Gather 无法表达"排除目标邻居"的选择性聚合——Belief Propagation 需聚合所有入边邻居"除目标邻居外"的消息，GAS 无法表达；Graph.hls DSL 用 filter 先排除目标边再 reduce 解决。

从算法pipeline角度拆解术语，比如术语如何在算法pipeline中发挥作用，给出术语在算法pipeline中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 算法pipeline 运转流程（PageRank 的 GAS 映射，Graph.hls 论文 Figure 5a）：
```
# 每轮迭代（一次 GAS pass）
iteration_input(G.EDGES)                       # 生成边流（结构数据）
map:   val = e.src.rank / self.out_deg          # Gather：沿出边读源顶点属性
reduce(e.dst, val, lambda a,b: a+b)             # Gather：按键聚合邻居贡献
map(self): self.rank = 0.15 + 0.85 * reduced    # Apply：更新顶点状态
# Scatter 由下一轮 iteration_input 承载（主机迭代调用直至收敛）
```
- pipeline 特征：GAS 天然是"边流 → 聚合 → 状态更新"的三段数据流水；edge-centric 变体下内存访问流式、可乱序处理边（Graph.hls 假设流无序以最大化带宽）；Apply 依赖 reduce 结果构成迭代间依赖，故硬件 kernel 只实现一个 pass、主机负责跨迭代循环与收敛判断（ε 阈值属 L1 参数）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件框架（PowerGraph、Ligra、GraphIt）与 FPGA 加速器（ACTS/GraphLily/ForeGraph/ThunderGP/ReGraph/Swift）均围绕 GAS 构建；FPGA 侧 edge-centric 变体以流式流水 + 片上聚合缓冲（URAM reduce buffer）实现。Graph.hls DSL 把 GAS 编译为空间数据流 DAG（iteration_input/map/filter/reduce/return 五类节点）→ GH-Architect 生成 HLS 硬件。
- 使用：作为"描述任意图迭代算法"的通用模板——需可交换可结合的聚合（GAS 的 reduce 假设）与单 pass 表达；Dijkstra 等顺序依赖算法需改写为 Bellman-Ford 式并发松弛才可高效空间并行。跨论文复用：把 GAS 作为图加速器 DSL 的表达力基准（GAS 可表达 ⊆ DSL 可表达），并据"GAS 无法表达什么"（选择性排除、不规则数据流）定位 DSL 扩展点。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design

## Hoisting（提升优化，FHE keyswitch 中间结果复用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hoisting（提升）是 CKKS 程序级的算法优化（Bossuat et al. EUROCRYPT'21 提出，HE² 采用）：利用"模数可交换性质"（EWO、Autom 等 Commutative Operators 与 ModUp/ModDown 可交换执行顺序），把多个并行 keyswitch 中的冗余 ModUp/ModDown 提取合并到 PKB 的输入/输出端，使同一输入密文的 ModUp 只做一次、多个 IP 聚合结果只做一次 ModDown。本质是用"中间结果复用"换取"evk 复用"：虽然各 keyswitch 的 evk 不同，但 ModUp 结果可跨多个 IP 复用，聚合后的 IP 输出只需一次 ModDown。
- 关键权衡：hoisting 减少 ComOps 数量（ModUp/ModDown），但把 MemOps（PMul、CAdd 等）的计算顺序交换、模数域从 Q 升到 PQ 或 PQ·dnum，MemOps 计算量增加——总收益取决于削减的 ModUp/ModDown 是否超过 MemOps 增额。其收益上限受 keyswitch 并行度约束：并行 keyswitch 越多、PKB 入/出度越低，可合并的 ModUp/ModDown 越多。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 PKB（n 条并行 keyswitch，同一输入 ct）为例（HE² 论文 Fig. 2(c) 与式 (1)(2)）：
```
# hoisting 前：每条 keyswitch 独立做 ModUp/IP/ModDown
for i in 0..n-1:
    c_up[i] = ModUp(ct)            # n 次 ModUp（同输入冗余）
    ip[i]   = IP(c_up[i], evk_i)   # n 次 IP
    out[i]  = ModDown(ip[i])       # n 次 ModDown
# hoisting 后：ModUp 提到前端共享，ModDown 聚合到后端
c_up = ModUp(ct)                   # 1 次共享 ModUp
for i in 0..n-1:
    ip[i] = IP(c_up, evk_i)        # n 次 IP 复用 c_up
out = ModDown(Σ_i ip[i])           # 1 次共享 ModDown（线性组合后）
```
- Annotations：共享前提是各 keyswitch 的输入密文相同（同一 PKB 内）；ModDown 可合并到输出端线性组合（PMul/CAdd）之后；代价是 IP 结果在 PQ·dnum 域累加、MemOps 域变大（见"ModUp/ModDown"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：软件库层面在 CKKS 程序变换中做（Anaheim 对原始程序直接应用 hoisting、FAST 在低密文层应用）；HE² 的 HERO 框架把 hoisting 作为 DFG 优化的末端步骤——先识别/扩展/融合 PKB（提高并行度、压低出入度），再在 PKB 输入输出端应用 hoisting，使 ModUp/ModDown 削减最大化（相比直接 hoisting 再多削减 2.25× 计算与 2.42× 通信）。硬件影响：hoisting 后 IP/PMul 域变大、可整块卸到近存（IRF），但 evk 复用率下降——EVF 单体 ASIC 直接应用 hoisting 反而因 off-chip evk 访问 stall 性能下降（SHARP+hoisting 仅 39.4% 加速需 2.89× 片上内存），IRF 异构架构（中间结果在 xMU 侧复用）才是 hoisting 的受益场景。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption

## PKB（Parallel Keyswitch Block，并行密钥切换块）与 PKB 融合

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PKB（Parallel Keyswitch Block）是 HE² 对 CKKS 数据流图（DFG）的核心抽象：一组并行的 keyswitch（同一输入密文 ct 经不同旋转步长 s_i 后各自做 keyswitch），在明文矩阵×密文向量乘法与 bootstrapping 的 C2S/S2C 阶段占据主导。状态最先进实现用并行 keyswitch（不同旋转步长）+ PMul/CAdd 线性组合完成；旋转步长在 PKB 内通常构成算术级数。HE² 的 HERO 框架先遍历 DFG、按路径顺序给 keyswitch 分层并把同层分组为 PKB（PKB identifying），再用交换律算子贪心扩展 PKB 压低入/出度（degree-minimized PKB expanding），为 hoisting 创造最大削减空间。
- PKB 融合（PKB Fusing，HE² 首次提出）：利用旋转可加性 Rot(Rot(ct,s),t)=Rot(ct,s+t) 与 EWO 后移（Rot(PMul(ct,pt))=PMul(Rot(ct),Autom(pt))），把两个串行 PKB（n1 与 n2 条旋转路径）融合为 O(n1·n2) 条并行旋转的大 PKB（逆 BSGS 变换），从而把 CKKS 程序中大量低并行（<10）的碎片 PKB 合并成高并行（>30）PKB，让 hoisting 的 ModUp/ModDown 共享潜力被完全释放。代价：evk 数量（按非重复旋转步长子集计）、IP 数与中间 MemOps 计算量上升。融合收益由 Fusion evaluator 的 FuseScore 量化（融合后 evk 超存储容量判无效），全局最优融合方案由 DP 递推式 DP[i][j]=max_{j'}DP[i][j']+DP[j'+1][j]+FuseScore(j',j'+1) 求出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 两串行 PKB 融合（HE² 论文式 (4)，n1=n2=2 简化例）：
```
# 融合前：PKB1（n1=2 并行旋转）→ EWO F → PKB2（n2=2 并行旋转）
PKB1: { Rot(ct, s1), Rot(ct, s2) }
F:    求和/线性组合（EWO）
PKB2: { Rot(·, s1'), Rot(·, s2') }
# 融合：EWO 沿各路径后移 + 旋转步长相加
Fused: { F_i'( Rot(ct, s1+s1'), Rot(ct, s1+s2'),
                 Rot(ct, s2+s1'), Rot(ct, s2+s2') ) }   # 4 条并行旋转（3 条不同步长）
# 融合后 hoisting：4 条并行 ModUp → 1 次共享 ModUp；输出端线性组合后 1 次 ModDown
```
- Annotations：s_j+s_i' 出现重复步长时可去重减少 evk 数；Fusion evaluator 依据相对 IP 数、中间结果尺寸与所需 evk 数评估"省下的 ModUp/ModDown 通信 vs 增加的 evk 存储与 MemOp 计算"，在 8 GB HBM 存储约束下用 DP 选全局最优；案例 ConvBN DFG（3 个 9/8/8 并行 PKB）：原始 25 个 ModUp/ModDown，直接 hoisting 只优化 PKB1，融合后 PKB2+PKB3 的 ModUp/ModDown 可提取到 8 条并行路径首尾（Fig. 9）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为离线程序变换在 FHE 编译器（EVA/CHET/ResiBM 生成 DFG）之后、映射到硬件之前执行；HE² 的 HERO 框架完整流程 = PKB 识别 → 度数最小化扩展 → DP 融合评估 → BSGS 配置选定 → 按 PKB 并行度映射 IRF/EVF 数据流。作用：把 hoisting 的通信削减从"程序天然并行"中挖出（相比直接 hoisting 再多削 2.25× 计算/2.42× 通信），并让融合后的 MemOps（IP/PMul）整块卸到近存 xMU，省掉 EVF 所需的巨大片上 evk 存储（SHARP 180+18 MB → HE² 44/84 MB）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption

## ModUp / ModDown（模提升 / 模下移，RNS 基转换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ModUp / ModDown 是 CKKS keyswitch 的核心基转换算子，在 RNS（Residue Number System）表示下把密文多项式在不同模数基之间转换。keyswitch 中密文在模数 Q 下分解为 dnum 个 group，ModUp 把每组提升（lift）到 PQ·dnum 域（在原始模数基之上附加特殊模数 P 的基），使密文与 evk 能在足够大的域中做内积；ModDown 把 IP 结果从 PQ·dnum 域降回 Q 域并完成模归约，与原始密文相加完成 keyswitch。二者都是计算密集型算子（算术强度 3.38/2.92 ops/byte，Table I），计算模式复杂（BConv 依赖：ModUp = 原始基下常数乘 + 目标基下常数乘与归约，复杂度 O(l1·l2·N)）。
- 模数可交换性质：EWO/Autom 可与 ModUp/ModDown 交换顺序（ModUp(PMul(ct,pt))=PMul(ModUp(ct),PModUp(pt))；ModUp(CAdd(ct,ct'))=CAdd(ModUp(ct),ModUp(ct'))），这是 hoisting 能把 ModUp/ModDown 合并提取到 PKB 首尾的数学基础——代价是交换后 MemOps 的模域从 Q 升到 PQ 或 PQ·dnum、计算量增大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 keyswitch 的 ModUp-IP-ModDown 数据流（HE² 论文 Sec. II-B1）：
```
# 输入：ct（模 Q）、分解组数 dnum、特殊模 P、evk（PQ·dnum 域）
for g in 0..dnum-1:
    ct_g = ModUp(ct, g)            # 提升到 PQ·dnum 域（BConv 模式）
    ip_g = IP(ct_g, evk_g)         # 与 evk 内积（MemOps，内存密集）
    acc  += ip_g                   # 累加
out = ModDown(acc) + ct            # 降回 Q 域并加回原密文
```
- Annotations：ModUp 输出与 IP 结果是异构加速器（xPU 做 ComOps、xMU 做 MemOps）中最大的中间结果传输（单次最高 144 MB 量级），且落在 keyswitch 关键路径上——这是 HE² 通信优化的直接对象；hoisting 后 ModUp 由 n 次共享为 1 次、ModDown 聚合为 1 次（见"Hoisting"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库（SEAL/OpenFHE/Liberate-FHE 等）以 BConv + 常数乘实现；硬件上 ModUp/ModDown 各走 INTT→BConv→NTT 流水（见"BConv"与"NTTU/BConvU"条目）。使用：每个 keyswitch（乘与旋转都依赖）必经 ModUp→IP→ModDown；HE² 中 xPU 主要承担 ModUp/ModDown（含 INTT-Resident 流水：把 INTT→BConv→NTT 拆成并行 BConv→NTT 与 NTT 两路提升并行），MemOps 卸到 xMU。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption

## NTT / INTT（数论变换 / 逆变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NTT（Number-Theoretic Transform，数论变换）是 FFT 在有限域（模素数）上的对应，把次数 N 的多项式从系数域变换到求值（槽/NTT）域，复杂度 O(N log N) 次蝶形运算；INTT 是逆变换。CKKS 中多项式乘法（密文乘法、BConv、IP 等）利用 NTT 把系数卷积变成逐点乘法：先 NTT 到求值域逐点乘，再 INTT 回系数域。NTT/INTT 是计算密集型算子（算术强度 0.89 ops/byte），其可并行性与数据访问模式决定硬件 NTT 单元（NTTU）的微架构。
- 在 keyswitch 的 ModUp/ModDown 中，每个密文 group 沿 INTT→BConv→NTT 流水执行：INTT 把系数域多项式转到 BConv 需要的求值形式、BConv 完成基转换、NTT 转回系数域。NTT 域（求值域）密文利于多项式乘（PMul/CMul），INTT 域密文仅出现在 BConv 之前——这一不平衡是 HE² 的 INTT-Resident/NTT-Resident 自适应密文格式管理策略的出发点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次多项式乘（p = a·b mod (X^N+1)）的 NTT 计算过程：
```
A = NTT(a);  B = NTT(b)       # 系数域 → 求值域（各 O(N log N) 蝶形）
C = A ⊙ B                     # 逐点相乘（N 次模乘）
c = INTT(C)                   # 求值域 → 系数域（O(N log N)）
```
- Annotations：NTT/INTT 蝶形访问同一多项式内的不同系数（高多项式内并行），而 BConv 同时处理来自多个多项式的系数（高多项式间并行）——两类单元的并行模式错配导致难以重叠，HE² 用可配置迭代 radix-2 NTTU 与 tree-based BConvU 做吞吐匹配解决（见"NTTU/BConvU"与"双级流水 xPU"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库用迭代 radix-2/radix-4 蝶形 + twiddle 因子表；硬件 NTTU 用蝶形阵列 + 冲突无关访存（HE² 采用 Mu et al. 的可配置迭代 radix-2 NTTU，NTT/INTT 动态共享）。使用：CKKS 每次多项式乘（PMul/CMul/keyswitch 的 IP/BConv）都含 NTT/INTT；HE² 中 NTTU 按 dnum 组均分到 BConv 所需 limb 上保证并行供数，NTTU allocator 在 INTT-Resident 流水两条并行路径间动态平衡负载（HE² xPU 的 NTTU 平均吞吐 768 w/ns，对比 SHARP 1024 w/ns）。
- HyperDrive 补充视角（ISCA'26，GPU TCU 上的分层 NTT 分解）：采用 Bailey 4-step NTT 递归分解（Alg. 1）把 N=2^13~2^16 的多项式写成 N=N1·N2、N1=8·8·N13、N2=8·8·N23，递归直到 radix-64 基例（匹配 FP64 TCU 的 8×4×8 MMA 维度，即 PTX m8n8k4），复杂度保持 O(N log N)（Inner-NTT 开销 C_NTT=Nk(t+1)、Hadamard C_HP=t·O(N)，N=k^(t+1)）；(N13,N23) 按 (2,1)/(4,1)/(2,4)/(4,4) 对应 N=2^13/2^14/2^15/2^16。NTT 被拆成 Inner-NTT（radix-64 基例，全片上执行）与 Outer-NTT（EWMult、Residual NTT、转置、GMEM/片上搬运），并把 32-bit 字长在 FP64 TCU 上用轻量 MPA 处理（单次 32-bit 乘法仅 2 次 FP64 乘法，对比 INT8 方案的 16 次）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## BConv（Basis Conversion，基转换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BConv（Basis Conversion，基转换）是 RNS-CKKS 中把一个 RNS 基（l1 个模数）下的多项式转换到另一 RNS 基（l2 个目标模数）下的算子：先在原始基下与常数相乘，再在目标基下与常数相乘并做模归约，复杂度 O(l1·l2·N)。BConv 是 ModUp/ModDown 的内部核心（模提升/降回本质就是基转换），计算密集型（算术强度 1.60 ops/byte）。
- 在 keyswitch 的 ModUp/ModDown 流水（INTT→BConv→NTT）中，BConv 同时处理来自多个多项式的系数（对分解组内所有 limb 各收一个系数做流水树归约），与 NTT 的高多项式内并行形成互补的并行模式——这一差异是 HE² 设计 tree-based BConvU 并做 NTTU/BConvU 吞吐匹配的核心动机（BConvU 每分解组通常需要不到 15 个 limb）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次基转换（原始基 B1 的 l1 个模数 → 目标基 B2 的 l2 个模数）的计算过程：
```
# 输入：a 在基 B1 下的表示 a = (a mod q_1, ..., a mod q_{l1})
# 第一步：原始基下常数乘（解基数重建系数的缩放）
for i in 1..l1:
    s_i = a_i * c_i mod q_i          # 常数乘，得到"缩放后的剩余"
# 第二步：目标基下常数乘 + 归约（逐目标模数重建）
for j in 1..l2:
    a'_j = ( Σ_i s_i * b_{j,i} ) mod p_j   # 常数乘 + 模归约
return a' = (a'_1, ..., a'_{l2})          # 基 B2 下的表示
```
- Annotations：两步常数乘 + 归约正是复杂度 O(l1·l2·N) 的来源；ModUp = BConv（Q→PQ·dnum 域）、ModDown = BConv + 模归约（回 Q 域）；hoisting 交换模域后 BConv 的目标模数变多、计算量上升（见"Hoisting"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：软件库中作为 keyswitch/relinearization 的内部步骤（SEAL/OpenFHE 的 BaseConv）；硬件上 HE² 用 tree-based BConvU——每单元每周期从分解组所有 limb 各收 1 个系数、做流水化树归约，吞吐 672 w/ns（对比 SHARP 16384 w/ns 但配合 NTTU 吞吐匹配仍追平 IRF 关键路径性能）。使用：任何需要跨 RNS 基变换的场合（ModUp/ModDown、乘后降模）；HE² 中 BConv 全部由 xPU 执行（ComOps），与 xMU 的 MemOps 通过 1 TB/s HBM 交换中间结果。
- HyperDrive 补充视角（ISCA'26，GPU 上 BConv 的两阶段分解与 NTT 融合）：采用 fast BConv [6]，把基转换拆成两阶段——BConv1（EWMult，与前置 INTT2 融合为 INTT2-BConv1 kernel）与 BConv2（矩阵乘法，与后置 NTT1 融合为 BConv2-NTT1 kernel）；BConv2 与 BConv1 不直接融合（BConv2 的矩阵乘法结构会使不同 block 基于同一共享输入重复计算）。融合前提是 Row-Major NTT 消除了 NTT 的多 pad 约束：BConv2-NTT1 kernel 中每 thread block 处理单个 limb i、BConv 约减沿 α' 维（GMEM→Reg→SMEM），SMEM 中间系数直接喂 NTT Stage-1（SMEM→Reg→GMEM），避免把 L+α-α' 维中间数据物化到 GMEM（Alg. 2）。BConv 的 stall long scoreboard 占比 60.6%（off-chip 访存）是融合的主要收益点（图 5）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## SparseGPT 与 STR（LLM/CNN 权重剪枝方法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SparseGPT（Frantar & Alistarh, 2023）是面向生成式 LLM 的一次性（one-shot）权重剪枝方法：把大规模权重剪枝建模为逐层稀疏回归问题，用近似 Hessian 逆（基于层的输入激活二阶信息）在剪掉权重的同时最小化重建误差，无需重新训练即可把 100B+ 参数模型剪到 50% 稀疏度且保持困惑度；默认支持 2:4/4:8 等半结构化与任意非结构化稀疏。STR（Soft Threshold Reparameterization，Kusupati et al.）是 CNN 的结构化幅度剪枝方法：把阈值作为可学习参数，通过软阈值函数 S(x)=sign(x)·max(|x|−t,0) 对权重做可微重参数化，与网络联合端到端训练，训练后按阈值 t 得到真正的稀疏权重。Harmonia 用它们生成评估负载：对生成式 LLM（LLaMA-7B、OPT-1.3B，序列长 1024）应用 SparseGPT 得到整体密度 0.2/0.4/0.6 的权重；对视觉模型用 STR 把 ResNet-50 剪到平均权重密度 0.1/0.2、用幅度剪枝（magnitude-based pruning）把 VGG-16 剪到 0.1/0.32——这些剪枝权重正是 Harmonia 验证"端到端稀疏推理负载"的输入（attention/MLP 投影呈现严重的 token 级稀疏偏斜与动态变化）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SparseGPT 逐层剪枝（对每层）：采样该层输入激活 X，算海森矩阵 H=XᵀX，对每行权重做"稀疏回归"——用 Cholesky 分解近似 H 的逆，迭代：选当前最大幅值权重保留、其余剪掉，用 d = H⁻¹·(w−w_pruned) 调整剩余权重补偿误差（OBS 式更新）；伪代码骨架：
```
for layer in model.layers:
    H = X^T X + lambda I            # X 为该层校准激活
    L = cholesky(H)                 # 近似 H^-1
    for row in W_layer.rows:
        mask = keep_topk_by_magnitude(row, k)
        row[mask==0] = 0
        delta = solve(L, row)       # 用 H^-1 补偿剩余权重
        row -= mask * delta
```
STR 训练式剪枝（每轮）：权重 W 过软阈值 S(W)=sign(W)·max(|W|−t,0)（t 可学习、随训练更新），前向用阈值后的 W_hat=S(W)，反向经 STE 直通回传梯度给 W 与 t；训练结束按 t 生成最终稀疏权重。两者都输出"保持准确率的稀疏权重矩阵"，后续推理用稀疏 kernel（如 Harmonia 的 SpMSpM 数据流）执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：SparseGPT 官方实现开源（github.com/IST-DASLab/sparsegpt，PyTorch，支持 OPT/LLaMA 等，`python opt.py facebook/opt-1.3b c4 --sparsity 0.5` 一类命令行）；STR 有官方与社区实现（常用于 ResNet/ImageNet 稀疏训练）。Harmonia 中的用法：把剪枝后的权重矩阵作为 SpMSpM/稀疏 GEMM 的输入，与 SuiteSparse 矩阵一起构成 16 个评估 workload 的 DNN 子集，验证分层调度在真实剪枝网络（LLaMA-0.2/0.4/0.6、OPT-0.2/0.4/0.6、ResNet-0.1/0.2、VGG-0.1/0.32）上的端到端加速（平均 1.87×）与鲁棒性。注意：Harmonia 论文只用其生成稀疏权重，未修改剪枝算法本身。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

HiT 补充视角（ISCA'26，用幅值剪枝生成稀疏 LLM 评估负载）：HiT 对 Llama2-7B 的三个投影层（1024×11008、1024×4096、1024×11008，序列长 1024）的权重矩阵应用 magnitude-based pruning（幅值剪枝——按 |W| 从小到大置零到目标稀疏度），剪枝水平取 0.2/0.4/0.6，与近期 GPT 稀疏化研究（SparseGPT [49]、Wanda 类 [50]）一致；激活保持稠密（密度 1），故这些 workload 属于 MS×D（中稀疏权重 × 稠密激活）矩阵乘。它证明了幅值剪枝作为"生成可复现稀疏 DNN 评估负载"的简单手段的价值：无需重训练、无 Hessian/校准开销，直接以目标密度裁剪即可让硬件评估覆盖中稀疏度段——与 Harmonia 用 SparseGPT/STR 生成 LLM/CNN 稀疏权重是同一工作流，区别仅是剪枝方法更朴素、权重稀疏度更高（0.2-0.6 且投影层尺度 4096/11008）。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

## Pointwise Maximal Leakage（PML，逐点最大泄漏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PML 是 Saeidian、Cervia、Oechtering、Skoglund（KTH）在 ISIT 2022（arXiv 2205.04935，期刊版 IEEE TIT 2023, DOI 10.1109/TIT.2023.3304378）提出的信息论泄漏度量。它把"单个观测"的泄漏定义为：观测到 y 后攻击者猜秘密 x 的后验/先验最大乘性增益，$\ell_{P_{XY}}(X \to y) = \log \max_{x: P_X(x)>0} \frac{P_{X|Y=y}(x)}{P_X(x)}$。与平均类指标（互信息、最大泄漏）不同，PML 把泄漏视为随机变量（其分布由观测分布 P_Y 诱导），从而支持把隐私保证表达为泄漏分布的统计性质。Helium（ISCA 2026，Stanford）是第一个把 PML 应用于硬件侧信道泄漏量化并给出可计算方法的工作（论文引 [79] 即 PML 原文献）。
- 逻辑链：PML 的定义需要"观测"级别的后验 P_{X|Y=y}(x)——这要求知道程序级观测分布；Helium 用 µobs functions 建模指令级可观测执行、用 Tracer 计算 µtrace 概率分布，从而把 PML 落到可计算处；在确定性信道下（Helium 的默认威胁模型，泄漏函数是操作数的确定函数），PML 简化为 $\ell(y) = -\log P_Y(y)$——即观测概率越低、泄漏越大。例：32-bit 均匀秘密下优化 2 的 y₁（x=0）PML=log(2³²)=32（泄漏全部 32 bit），y₂（x≠0）PML=log(2³²/(2³²−1))≈3.36×10⁻¹⁰。
- Web 证据：https://arxiv.org/abs/2205.04935（ISIT 2022 论文）；https://dl.acm.org/doi/abs/10.1109/TIT.2023.3304378（TIT 期刊版）。PML 定义、动机（最大泄漏是平均保证、无法区分"个别观测完全泄露秘密"的通道）与隐私保证（把泄漏看作随机变量的统计性质）均与 Helium 论文描述一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 Helium 的泄漏量化 pipeline 中，PML 是"度量-建模-分析"三部分的最顶层输出标准。pipeline 计算流程（确定性信道）：
```
输入：程序 P、秘密输入分布 X、µobs functions F
1. Tracer 计算每个程序级观测（µtrace）y 的概率 P_Y(y)
   （TracerSym：符号执行+模型计数得精确概率；
     TracerSim：Monte Carlo 频率估计+Clopper-Pearson 保守界）
2. 每条 µtrace 的 PML：ℓ(y) = -log P_Y(y)        # 确定性信道简化式
3. 构造 tail-bound 保证：选"可容忍划分"中概率最低的 µtrace 为 ε-µtrace，
   ε = ℓ(ε-µtrace)，1-δ = 可容忍集概率和
输出：P_Y[ℓ(Y) ≤ ε] ≥ 1-δ
```
- 具体例子（论文 §VII-A Poly1305，zero-skip 乘法）：128-bit 均匀 key 下 TracerSym 得到 8 条 µtrace；概率最低（最不可容忍）的 µtrace 是"所有秘密相关乘法都非零"的情形，其概率 1−9.39×10⁻¹⁰，PML=−log(1−9.39×10⁻¹⁰)≈1.35×10⁻⁹ bit；其余更高泄漏的 µtrace 总概率 ≤9.39×10⁻¹⁰ ⇒ 保证 P[ℓ≤1.35×10⁻⁹]≥1−9.39×10⁻¹⁰。对照：digit-serial 乘法下高泄漏 µtrace 概率和达 0.49，得 P[ℓ≤0.97]≥0.51——同一程序、不同硬件优化，PML 分布完全不同。
- Annotations：ℓ(y) 对观测 y 逐点计算而非平均；ε 是"可容忍"泄漏上界（用户可接受的最高逐点泄漏），δ 是高泄漏观测总概率上界；ε-µtrace 的选择构造出"泄漏≤ε 的概率≥1−δ"这一最可解释的保证形式。确定性信道假设使 PML 只依赖观测概率，无需显式后验。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现上需要"观测概率分布"，这正是 Helium Tracer 的产出：TracerSym 用 Angr 符号执行把秘密符号化、逐 transponder 与 µobs 约束合取、Ganak 模型计数得精确概率；TracerSim 用 Intel Pin 动态插桩逐 trial 记录 µtrace、频率/N 估计概率（支持任意输入分布，如非均匀秘密）。Helium 用它输出 tail-bound 隐私保证（§VI-B），并给出 Clopper-Pearson 95% 置信与 Rule of Three（单 µtrace 时未观测事件概率<3/N）的保守统计版本。
- 使用场景：硬件/软件设计者权衡"接受多少小概率泄漏以换取多少性能"——如论文 Case Study IV 中，Chacha20-Poly1305 接受 P[ℓ≤0.0004]≥0.9997 即可省去 cio 缓解的 mul64 2.31×、cs32 3.37× 开销；Ed25519 按函数分解（表 VII）选择性缓解。限制：TracerSym 受符号执行可扩展性限制（路径爆炸、密码哈希不可符号化）；TracerSim 无法达到密码级（≤2⁻⁸⁰）保证——Helium 明确不提供密码学证明，面向"愿牺牲绝对安全换性能"的设计空间。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees

## Tail-bound guarantee（尾部界隐私保证，ε/δ 概率隐私界）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tail-bound guarantee 是 PML 原文献（Saeidian et al., ISIT 2022, arXiv 2205.04935）提出的隐私保证形式：由于 PML 是观测 y 的函数、y 的分布是 P_Y，PML 可视为随机变量，隐私保证即对 PML 分布施加统计约束。tail-bound 要求"泄漏超过 ε 的观测出现概率低于 δ"：$P_Y[\ell(Y) \le \epsilon] \ge 1 - \delta$，即把观测划分为"good"（低 PML）与"bad"（高 PML）两类，bad 类总概率受 δ 约束。Helium 是第一个把 tail-bound 应用于硬件侧信道泄漏量化并给出具体 (ε,δ) 构造方法的工作。
- 逻辑链：平均类指标（互信息/最大泄漏）给单值平均、掩盖低概率高泄漏事件；tail-bound 把保证表述为"高泄漏以低概率发生"——更贴合安全实践者"程序必须以极高概率泄漏极少"的直觉。Helium 的核心贡献之一是给出了"可容忍划分"构造：当少数 µtrace 承载大部分概率质量（低泄漏）而其余罕见 µtrace 高泄漏时，选概率最低的容忍 µtrace 为 ε-µtrace，ε 取其 PML，1−δ 为可容忍集概率和，得到 ε 与 δ 都较小的可解释保证。
- Web 证据：PML 论文（https://arxiv.org/abs/2205.04935）第 IV-D 节定义 tail-bound；Helium 论文 §IV-D 与 §VI-B 使用该定义并给出构造与 TracerSim 统计版本（Clopper-Pearson 95% 置信、Rule of Three）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 Helium pipeline 中 tail-bound 的计算过程（§VI-B）：
```
1. 用 Tracer 得到 µtrace 概率分布 {P_Y(y)}
2. 对每条 µtrace 计算 PML ℓ(y) = -log P_Y(y)
3. 若存在"可容忍划分"：
   把 µtrace 按 PML 排序，找到可容忍（低泄漏）集合
   ε-µtrace = 可容忍集合中概率最低的 µtrace
   ε = ℓ(ε-µtrace)          # 所有可容忍 µtrace 的 PML ≤ ε
   1-δ = Σ_{y ∈ 可容忍集} P_Y(y)   # 高泄漏(bad)概率 ≤ δ
4. 输出保证 P_Y[ℓ(Y) ≤ ε] ≥ 1-δ
   （TracerSim 下：ε 用 ε-µtrace 概率的 Clopper-Pearson 下界
    的 -log；1-δ 用 bad 概率上界对应的下界，95% 置信；
    仅单 µtrace 时用 Rule of Three 3/N）
```
- 具体例子（论文 §VII-D，Chacha20-Poly1305 mul64，TracerSim N=10,000/轮）：两轮各 10,000 trials 均只观察到单一 µtrace ⇒ Rule of Three 得未观测 µtrace 概率 <3/10000=0.0003 ⇒ 保守估计该 µtrace 概率 ≥1−0.0003 ⇒ ε=0.0004 bits（−log(0.9997)）、1−δ=0.9997 ⇒ 保证 P[ℓ≤0.0004]≥0.9997。对照 cs64 类别：多 µtrace、高泄漏，得弱保证 P[ℓ≤2.1461]≥0.9552——提示程序员 cs64 仍需缓解。
- Annotations：ε（PML 阈值，bits）与 δ（超过 ε 的观测总概率）成对出现；点 (ε,δ) 构成可行保证集合（论文图 7 的 Poly1305 曲线，x 轴候选 ε、y 轴累计高泄漏概率）；"可容忍划分"不总存在——Ed25519/Argon2id 多数类别所有 µtrace 都高泄漏，此时无意义保证、须缓解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TracerSym 精确给出 µtrace 概率（需秘密输入均匀分布假设，符合密码学 key 场景）；TracerSim 用两轮 Monte Carlo（N₁ 定 ε 防选择偏差、N₂ 定 δ）+ Clopper-Pearson 二项置信区间产生保守 (ε,δ)，N₁/N₂ 可调提供运行时-精度权衡，trials 可并行。
- 使用场景：程序员把 (ε,δ) 作为"安全-性能权衡"的决策输入——若可容忍泄漏风险（如 Chacha20-Poly1305 的 mul64/cs32），免去 cio 缓解的 2.31×/3.37× 开销；若泄漏不可容忍（Ed25519/Argon2id），保留缓解或只缓解泄漏重的函数（表 VII 的 sc25519_reduce、ge25519_scalarmult_base）。限制：Helium 不提供密码学证明（TracerSim 到密码级 N 不现实），tail-bound 针对单一固定公共输入计算，未覆盖攻击者自适应选择公共输入的场景（§VIII 列为未来工作）。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees

## Constant-time（CT）编程（恒定时间编程）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Constant-time 编程是侧信道防御的黄金标准：编写代码时避免把秘密值传给"不安全指令"的"不安全操作数"——历史上不安全操作数限于内存地址、分支操作数与少数可变时算术指令（如除法），因为这些指令在秘密相关时会产生数据相关的硬件资源占用（执行时间、cache 行为等）可被攻击者观测。Helium 论文（§I）指出：现代微架构采用越来越多的数据相关优化（计算简化 zero-skip、流水线/寄存器文件压缩、silent stores、计算复用、值预测、数据内存相关预取等），威胁使所有指令操作数都可能不安全，CT 编程变得不可能（Opening Pandora's Box, ISCA'21 的论点）。
- 逻辑链：CT 保证"零泄漏"（秘密永不经不安全操作数）→ 需要安全指令集合仍足够大以变换秘密值；随数据相关优化扩散，安全指令集合萎缩，CT 成本上升甚至不可行 → 催生替代策略：硬件 ISA 扩展（Arm DIT、Intel DOIT、RISC-V Zkt/Zkvt，粗粒度关闭优化）与软件细粒度变换（如 cio 的二进制码变换）；Helium 则提供"有界泄漏"（bounding leakage）路径——不追求零泄漏，而是量化泄漏概率供设计者权衡。
- Web 证据：常数时间编程为公开通用概念（Almeida et al., "Verifying Constant-Time Implementations", USENIX Security 16，论文引 [5]）；Intel/OpenSSL 均有官方 CT 指南。论文 §I 引 [37]（cio）报告软件侧 CT 变换开销可达 28×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在秘密处理程序的算法 pipeline 中，CT 编程是对"秘密数据流向不安全指令操作数"这一数据依赖的约束。伪代码层面的 CT 模式：
```
# 非 CT（泄漏）：用秘密作分支条件
if secret_bit == 1:
    y = table_a[x]
else:
    y = table_b[x]

# CT 变换：消除秘密依赖的地址/分支（位掩码选择）
mask = 0 - secret_bit        # secret_bit∈{0,1} ⇒ mask=0 或 0xFF..F
y = (table_a[x] & mask) | (table_b[x] & ~mask)   # 两条路径都执行
```
- Helium 论文的动机例子：zero-skip 优化使算术指令（历史上"安全"、CT 代码可放心传秘密）也变成 intrinsic transmitter——乘法器在操作数含 0 时走快速 µobs、否则慢速 µobs（图 1）；因此即使传统 CT 代码也可能在新微架构上泄漏。cio（论文 §VII-D 的 baseline）把不安全操作数变换为永不在不安全值集合取值（如 32-bit 减法：两操作数零扩展、第 33 位置 1、相减、取低 32 位），使所有指令恒走同一 µobs，但开销 2.31×–15.71×。
- Annotations：CT 的核心约束是"指令的时序/资源行为与秘密值统计独立"；位掩码选择示例消除地址/分支依赖但增加指令数与执行路径；zero-skip 例子说明"安全指令集合"随微架构演化而缩小——Helium 正是为评估这类新泄漏而设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现途径：① 手工编码纪律（crypto 库如 Libsodium 的既定做法）+ 工具验证（ctgrind、dudect、SideChannelMarvel 等）；② 编译器/语言级（FACT 语言、SynthCT 可移植 CT 代码合成）；③ 二进制码变换（cio，ASPLOS'24，github.com/counter-optimization，对 x86_64 与 libsodium 实现，处理寄存器溢出、地址计算与复杂指令微操作，发现首个微架构缓解组合安全问题的实例）；④ 硬件模式（DIT/DOIT/Zkt/Zkvt）。
- 使用场景：CT 用于需要绝对零泄漏的安全关键代码（密码学、密钥处理）；其成本随微架构数据相关优化增多而上升（Intel 警告 DOIT 未来处理器性能影响可能"显著更高"）；Helium 提供的替代路线是量化为 (ε,δ) 的有界泄漏——当程序员可接受极小概率的高泄漏（如 P[ℓ≤0.0004]≥0.9997）时可免去高开销缓解。局限（§III-A）：Helium 只覆盖非推测性泄漏（非 Spectre 类）与 intrinsic transmitter。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees

## 稀疏矩阵乘法（Sparse Matrix Multiplication，SpGEMM / SpMSpM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
稀疏矩阵乘法 C=A×B 是只对两个稀疏矩阵的非零元素做乘累加的线性代数运算（SpGEMM 泛指任意稀疏×稀疏，SpMSpM 特指 sparse-matrix × sparse-matrix，SpMM 为 sparse × dense）。它是图分析（三角计数、PageRank、图神经网络）、科学计算（有限元、电路仿真）、稀疏 CNN 与剪枝 LLM 推理的核心 kernel。相比稠密 GEMM，稀疏乘法的难点是：(1) 数据访问不规则——非零位置由矩阵结构决定，无法按稠密网格预测；(2) 数据复用低、计算强度低（M/K/N 循环中大量迭代无有效乘法）；(3) 负载不均衡——不同行/列的非零数差异大；(4) 输出也不确定——C 的稀疏结构在计算前未知，需要合并（merge）各中间部分和。Web sources（Gamma ASPLOS'21、Flexagon、SpecBoost）确认内积/外积/Gustavson 三类数据流以不同方式权衡这些难点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
算法级 SpGEMM 的执行骨架（外积视角，HiT 采用）：
```
输入: A (MxK, 稀疏), B (KxN, 稀疏), 输出 C (MxN, 稀疏)
# 外积：对每个 k，A 的第 k 列 x B 的第 k 行 生成一个 rank-1 片
C = {}
for k in nonzeros(K):
    for (m, a) in A.col[k]:        # A 列 k 的每个非零 (行 m, 值 a)
        for (n, b) in B.row[k]:    # B 行 k 的每个非零 (列 n, 值 b)
            C[m][n] += a * b       # 累加进输出位置 (m,n) —— 跨多个 k 的合并
# 输出合并：同一 (m,n) 被多个 k 贡献，需按 (m,n) 聚合
```
关键计算特性：每个 (m,n) 输出需要"列-行索引匹配"（A 非零的列索引 == B 非零的行索引，即 k 相同）才产生有效乘法；交叠率（有效匹配/总比较数）在高度稀疏时极低（HiT 实测 HS×HS geomean 仅 0.12%），决定 MAC 利用率。HiT 的算法变体：按列索引再行索引连续排列的 COO-like 片上格式使非零按外积顺序流式读取；HS×HS 输出同样稀疏，用压缩格式累积 psum（见 DMAccum）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用分三层：(1) 软件——SuiteSparse 等库提供 SpGEMM，支持 CSR/CSC/COO 格式与并行调度；HiT 用 27 个 SuiteSparse 真实矩阵（p2p-Gnutella24 密度 9.3e-5、cage12 1.2e-4、poisson3Da 1.9e-3、opt1 8.1e-3 等覆盖 1e-5~1e-2 密度段）做 HS workload，HS×HS 计算 M×M^T；(2) 硬件——专用稀疏加速器（HiT、Trapezoid、SpArch、OuterSPACE、Sigma、Flexagon、Spada）用专用数据流/相交单元/psum 合并网络加速，避免对零做无效 MAC；(3) 数据生成——MS/D 段用剪枝 DNN（ResNet50/VGG16 40% 密度非结构化稀疏 + im2col 转矩阵乘、Llama2-7B 幅值剪枝 0.2/0.4/0.6）构造。HiT 的意义：首次用统一架构把 HS/MS/D 三段都跑到高吞吐（全谱 performance/area 比 Trapezoid 高 1.93×）。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication

SegFold 补充视角（ISCA'26，动态数据流下的 SpGEMM）：SegFold 针对双端稀疏（dual-side sparsity，A 与 B 都稀疏）的 SpGEMM，核心是把 SpGEMM 数据流从"静态循环序"扩展为"细粒度动态"：
- 静态数据流的公共缺陷：inner product 只复用 C（行-列点积交点数随非零位置变化）、outer product 复用 A/B 但每次迭代生成整个 T_{M,N,k} 部分和矩阵（输出约简距离最远达 M×N）、Gustavson 牺牲 B 复用且中间输出行大小不定；没有任何单一静态调度能同时最大化 A/B/C 三操作数复用，且静态循环对非均匀非零分布产生负载/计算失衡。
- SegFold 的动态扩展：SELECTA 利用 K 维约简的结合律，在 active window（默认 32 个 k）内逐周期贪心重排 (m,k) 顺序（优先共享 k 以复用 B 行、避免同 m 冲突）；SEGMENTBC 在虚拟坐标空间 V 中即时定位/创建 C 部分和，让 C 元素在 PE 间动态迁移以平衡约简负载——同时拿到 element-wise A 复用、row-wise B 复用与 tensor-wise C 复用。
- 实验意义：15 个 SuiteSparse 矩阵上 geomean 1.95× over Spada（runtime-adaptive baseline）、5.3× over 最佳 Flexagon 静态配置，证明"动态"是数据流设计空间中静态调度无法覆盖的维度。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow

## im2col（Image to Column，卷积到矩阵乘变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
im2col 是把卷积运算等价重写为矩阵乘法（GEMM）的数据重排变换：把每个卷积窗口（filter 覆盖的输入 patch）展平成一列（或一行），把输入特征图重排成"展开矩阵"，使所有卷积窗口的计算变成一次标准 GEMM。它牺牲存储（展开后数据冗余、内存膨胀，经典地约放大 k×k×C_in 倍）换取"GEMM 高度优化"的好处——GEMM 在 CPU/GPU/TPU 上有成熟的高性能实现与硬件支持。对卷积神经网络训练/推理的框架（Caffe、早期 cuDNN、TensorFlow）与稀疏加速器评估都常用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
im2col 的展开过程（单输入图 I，C_in 通道，H×W；卷积核 K 个 filter，每 filter c×k×k；输出 C_out=K 通道）：
```
输入: I[C_in][H][W], Filter[K][C_in][k][k], 步长 s, padding p
# 1. 构建展开输入矩阵 Col[C_in*k*k][H_out*W_out]
#    每输出位置 (oh, ow) 一列：按 (c_in, kh, kw) 顺序展平该窗口的像素
for oh in range(H_out):
    for ow in range(W_out):
        col = []
        for c_in in range(C_in):
            for kh in range(k):
                for kw in range(k):
                    col.append(I[c_in][oh*s+kh][ow*s+kw])
        Col[:, oh*W_out+ow] = col
# 2. 权重矩阵 FilterMat[K][C_in*k*k]：每个 filter 展平成一行
# 3. 一次 GEMM：Out[K][H_out*W_out] = FilterMat @ Col
# 4. 结果按 (oh, ow) 重排回输出特征图 Out[K][H_out][W_out]
```
HiT 的用法：把 ResNet50/VGG16 的三个卷积层（如 3×3、512 通道层）离线经 im2col 转换成矩阵乘法，得到 MS×MS workload（激活与权重均为 40% 密度非结构化稀疏），再用稀疏 GEMM 加速器执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：主流深度学习框架内置 im2col（PyTorch 的 unfold / Caffe 的 im2col 层；cuDNN 用隐式 GEMM 避免显式展开内存开销）；HiT 论文中 im2col 是离线数据预处理（把卷积层转成标准矩阵乘，遵循 [48]），作为 MS 评估 workload 的构造步骤——转换后输出仍保持卷积层语义，但可以复用矩阵乘加速器与稀疏数据流。注意：im2col 使非零布局从"卷积结构"变为"GEMM 矩阵结构"，稀疏加速器（HiT 等）据此按行列索引匹配执行，而无需感知卷积窗口拓扑。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication

## Tree-based Speculation（树形投机解码：draft budget / tree width / masked attention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
投机解码的一类扩展：draft 模型每步提出多个候选 token 形成树（候选树），target 用带掩码的注意力（masked attention）一次前向并行验证整棵树的候选，提高接受概率与单轮产出（EAGLE、Medusa、Lookahead 等 [3][38][48][64]）。两个控制参数：draft budget（每轮候选树的总 token 预算）与 tree width（每层分支数/每步候选数）；chain 式是 width=1 的特例。HybridSpec 用它做运行时调制的杠杆：(1) budget 增大接受长度先增后饱和（图 10(a)，超出阈值收益递减，冗余计算在拒绝候选上浪费）→ 设上限 B；(2) 固定预算下接受长度随 width 先增后减（图 10(b)——宽树覆盖更多候选但单支变浅，探索 vs 深度权衡）→ 用 SVR 拟合 (budget→最优 width) 查找表。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每轮（width=p、预算 B_t 的候选树）
HB 栈（draft）: 按 tree width 并行/自回归扩展候选树（逐 token，masked 位置）
    current_size += tree_width; 若 current_size <= draft_budget 继续扩展
XPU（target）: VerifyTask(draft_budget) —— masked attention 一次前向验证整树
接受: 沿树保留被接受的最长前缀；拒绝分支的 KV cache 清除
```
参数调制（Algorithm 1）：HB 栈算术强度低于 roofline 时 tree_width+1（≤p，多探索吃满带宽）、否则 -1；XPU 未满时 draft_budget ×2（≤B）、否则 ÷2。实测预算/宽度随请求率从 (30.74,3.72) 降到 (9.25,1.58)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：候选树以树/序列集合表示，验证用自定义 attention mask 指示各候选的位置前缀；开源实现见 EAGLE-3/Medusa/Lookahead（vLLM/SGLang 支持）；HybridSpec 的 Fig.10 数据来自 [64] 的开源实现测量。使用要点：width 与 budget 存在"探索-深度"权衡，需按目标模型接受特性离线拟合、运行时按利用率在线调制。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding

## KeySwitch（密钥切换 / Key Switching，含 ModUp-ModDown）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KeySwitch（密钥切换）是 FHE 中把密文从一把密钥下转换到另一把密钥下的过程（不泄露明文）：HMult 后 relinearization 把三元密文压回两元，Rot（automorphism）后把旋转密钥换回原密钥。RNS-CKKS 采用 hybrid KeySwitch [20]：先把密文末分量在 R_{Q_ℓ} 中分解为 β 个 digit（β=⌈ℓ/α⌉，α 为 special prime 模数个数、dnum 为分解数），ModUp（NTT + BConv）把 digits 提升到扩展环 R_{PQ_ℓ}，与求值密钥 evk（R_{PQ_ℓ}^{2×β} 中）做 inner product（IP），再经 ModDown（NTT + BConv）回到 R_{Q_ℓ}，最后加到原密文其余分量上。
- KeySwitch 是 CKKS 操作级（HMult/Rot）的主要性能瓶颈：它频繁执行 NTT、BConv、IP 三个高开销子操作，且每次调用伴随大量跨多项式、跨 limb 的高维数据搬运。论文通过 Nsight Compute 剖析发现 IP kernel 的 stall long scoreboard 占比达 74.6%（off-chip 访存），BConv 为 60.6%，是 COOP 融合优化的直接目标。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- hybrid KeySwitch 的完整流水（β 个 digit，输入密文 ct=(c0,c1)∈R_{Q_ℓ}²）：
```
# 分解：把 c1 按特殊素数结构拆成 β 个 digit
for d in 0..β-1:  digit_d = Decompose(c1, d)          # 各 digit 在 α'_d 个 limb 上
# ModUp：digit 从 R_{Q_ℓ} 提升到 R_{PQ_ℓ}
for each digit_d:  NTT(digit_d);  BConv(Q→PQ, digit_d) # 变换到扩展环（hoisting 可复用）
# IP：与求值密钥点积累加（输出两分量）
IP = Σ_d  digit_d ⊙ evk_d      # evk 尺寸约为输入 2 倍，GMEM 访存密集
# ModDown：结果从 R_{PQ_ℓ} 降回 R_{Q_ℓ}
NTT(IP);  BConv(PQ→Q, IP);  ModDown 归约
# 合并：加到 c0 上
ct_out = (c0 + IP_downtoQ, 0)
```
- Annotations：ModUp/ModDown 内部即 NTT+BConv 交替（BConv 必须在非 NTT 域执行）；double hoisting [7] 把 PMAC 中间量全程保留在 R_{PQ_ℓ}，省掉大部分 ModDown；hyperdrive 的 COOP 把 KeySwitch 的两处 NTT-跨多项式边界融合为 (BConv2-NTT1) 与 (NTT2-IP)（Alg. 2/3），并把 IP 的 evk 在 NTT 计算期间预取到片上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件库（SEAL/OpenFHE/TenSEAL）中作为 relinearization/rotation 的内部过程，evk 在密钥生成阶段预计算；GPU 库（TensorFHE/WarpDrive/Neo/Cheddar/HyperDrive）把 KeySwitch 拆成多个 CUDA kernel（NTT/INTT、BConv、IP、EWAdd），并做 kernel fusion（[22] 的 intra/inter-operation fusion；HyperDrive 的 COOP）与操作重排（ModDown 后接 ModUp 时提前 EWSub）。使用场景：任何 HMult（relinearization）与 Rot（密钥基恢复）；论文实验 KeySwitch（N=2^16、L=32、β=2）中 COOP 相对 NTT+ 提速 1.24×，(BConv2-NTT1) 相对分开执行 1.36×、(NTT2-IP) 1.32×；H100 上 KeySwitch 延迟 414 μs（Set-E，A100 为 710 μs）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## CRPMAC（Ciphertext-Reused PMAC，密文复用明文乘加）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PMAC（Plaintext-Multiply-and-Add）组合 PMult（明文乘密文）与 HAdd（密文加），计算一批密文与明文系数的线性组合 Σ_i pt_i·ct_i，是 CKKS bootstrapping 的 CtS/StC 相位中 PCMM（明文-密文矩阵向量乘）的核心原语，在高 level（ℓ 大）下 PMAC 可能主导成本。CRPMAC（Ciphertext-Reused PMAC）是 HyperDrive 提出的 PMAC 变体：在 BSGS PCMM 中把所有 GS-Rots 推迟到末尾、按 GS 方向批处理，使单个 baby-step 密文被一批明文复用（一次 GMEM 读、多次乘加），消除 [22] 批处理 PMAC 每轮重复加载同一 baby-step 密文的冗余 GMEM 读。
- 背景：prior work [22] 只把 PMult 与 HAdd 融合进每轮 GS 密文生成，且按 BS 方向批处理（图 7）；HyperDrive 的 CRPMAC 按 GS 方向批处理（图 14），内存足迹更小，并对所有 EWArith kernel 施加向量化访存缓解带宽压力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- BSGS PCMM 中 CRPMAC 的批处理计算过程（bs 个 baby-step 密文 ct_i、gs 个 giant-step 明文对角 pt_{i,j}，i=1..bs、j=1..gs）：
```
# BS 相位：只做一次预旋转（ModUp），得到 bs 个 baby-step 密文
for i in 1..bs:  ct_i = BS_Rot(ct, offset_i)     # 复用同一 ModUp 输出（hoisting）
# CRPMAC：推迟全部 GS-Rots，按 GS 方向批处理（一次读 ct_i 复用 gs 次）
for i in 1..bs:                                  # 外层遍历 baby-step 密文
    ct_i ← GMEM-Load 一次                        # 关键：每 i 只读一次，跨 j 复用
    for j in 1..gs:  acc_j = acc_j + PMult(ct_i, pt_{i,j})   # 批量乘加
# 末尾统一做 GS-Rots，得到 ct'_j 输入给后续
for j in 1..gs:  ct'_j = GS_Rot(acc_j, offset'_j)
```
- Annotations：对照 [22] 的 BS 方向批处理（外层 j、内层 i）每轮 j 都要重读同一批 ct_i，GMEM 读冗余；CRPMAC 把密文读从 O(bs·gs) 降为 O(bs)。与 hoisting 场景结合时在 GS 方向批处理使内存足迹更小。论文消融：CRPMAC 使 CtS/StC 的 EWArith 相对 batched PMAC 提速 1.34×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 bootstrapping 的 CtS/StC 相位把 PCMM 的 PMAC 阶段重排为"按 baby-step 密文外层循环 + 按 GS 方向内层批处理"的 CUDA kernel，配合向量化（vectorized）EWArith 访存；与 hyperdrive 的 COOP kernel（BConv2-NTT1/NTT2-IP）集成进 bootstrapping 流程（论文 §V）。使用场景：BSGS/double-hoisting 的 CKKS bootstrapping；论文在 Set-E 上 Bootstrap 39.49 ms，HyperDrive-CORE 相对 BASE 的 bootstrap 提速 1.85×（含 NTT+ 1.49×、COOP 1.21×、CRPMAC 对 CtS/StC 的 1.34× 贡献）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## AES-XTS（XEX-based Tweaked CodeBook with Ciphertext Stealing 加密模式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AES-XTS 是 NIST SP 800-38E 标准化的可微调块密码模式（tweakable block cipher mode），广泛用于存储与内存加密（AMD SME、Intel TDX、Microsoft BitLocker 等）：用两个 AES 密钥（Key1 加密 tweak、Key2 加密数据块），tweak 取数据地址/扇区号，使相同明文在不同位置产生不同密文。MANATEE 用它加密 NVM 页（64B 页 = 4 个 16B XTS 块）。
- MANATEE 选 AES-XTS 而非 AES-CTR 的原因：AES-CTR 依赖 counter 新鲜度保证保密性，需完整性树（integrity tree，如 Bonsai Merkle Tree）防止 replay 攻击，而完整性树在 EHS 上能量开销巨大（论文指出 CME + 完整性验证相对 MANATEE 约 50× 慢）；AES-XTS 天然免 replay（每个 tweak/位置独立）——论文强调它"无需完整性树即保证数据保密性"，且比 AES-CTR 更强的抗单比特翻转等攻击（虽然不能防篡改，因无完整性验证）。
- MC-ORAM 语境（TEE 内存加密）：Intel TDX/AMD SEV-SNP/ARM CCA 用 TME（Total Memory Encryption）引擎对 DRAM 做确定性 AES-XTS 加密，C = AES_XTS(addr_128, D_128)，tweak=物理地址、无任何 nonce 元数据——高效但确定性：同一物理地址写同一明文产生同一密文，形成**密文侧信道**（攻击者经内存总线嗅探 DRAM 密文可检测"值重复或变化"）。该泄漏破坏 ORAM 的不可区分性（树/暂存内容不变→密文不变，暴露暂存占用与 dummy 位置，区分优势可达 1/4）。MC-ORAM 用 112 位随机掩码+16 位计数器保证同一物理地址两次访问的加密前 128 位值不同（同掩码周期概率 1、跨周期 1−2^−112），从而让确定性 AES-XTS 密文每次访问都变化。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（加密一个 16B 块 P_i，位于页偏移 i，tweak = 页地址派生）：
```
T_i = AES_Key1(tweak) ・ alpha^i        # GF(2^128) 中乘 alpha 派生 tweak
C_i = AES_Key2(P_i XOR T_i) XOR T_i     # XEX 结构
# 64B 页 = 4 个块：连续加密 4 个 16B 块，SPM 内缓冲凑齐后一次原子写 NVM
```
- 例子：页 P3 被驱逐/断电，page manager 取页基地址作 tweak，逐块 AES_Key2 加密，4 块凑 64B 原子 flush；读回时同 tweak 解密。MSP430FR5994 提供 AES 加速器/库，MANATEE 用其实现 AES-XTS。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：利用 MSP430FR5994 的 AES 加速器/库（先于本文的方案 [55,57] 同样使用）；页粒度加密（64B）摊销加解密开销，使加解密频率远低于字级全加密。性能分解显示 CRC32 这类写密集负载加解密占 ~74% 执行时间。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## Token Pruning（动态 token 剪枝，LazyLLM 式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Token pruning（token 剪枝/动态 token 丢弃）是长上下文 LLM 推理的加速优化：在每个 transformer 层，根据注意力输出动态选出相关性最高的 token 子集（例如按注意力分数取 top-k），只对这些 token 做后续层（自注意力、FFN/MoE）计算，而跳过被剪掉的 token，从而把随序列长度线性/平方增长的计算量降下来。代表工作 LazyLLM（arXiv:2407.14057）在 prefill 阶段逐层剪枝、被剪 token 的 KV 不写入 cache。它与 KV cache 量化的区别：剪枝减少"参与计算的 token 数"（计算量），量化减少"每个 KV 的位数"（容量/带宽）。它是近似/有损优化（丢弃 token 可能影响精度，属"精度保持的算法改动"谱系，通常按比例控制）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 算法流程（以 LazyLLM/IroKnight 描述为例）：每个 transformer 层先正常计算 attention（MatMul Q·K^T → 缩放 → softmax → MatMul softmax·V）；然后对注意力分数做 Top-K 选择——顺序扫描分数数组、与运行阈值比较、保留分数最高的 k 个 token 索引（k 由剪枝比例决定，如保留 20%-100%）；被保留 token 才继续进入下一层（其 KV 写入 cache），被剪 token 的后续计算跳过。这个 Top-K 扫描是规则、顺序的仿射访问（步长 1 遍历数组），与 MatMul/softmax 的 tiled/vectored 执行同属细粒度规则访问。IroKnight 的视角：正因为剪枝只新增 Top-K 这种规则扫描算子、不改变算子级细粒度仿射访问，全状态加密（见 Pad/PadGen）照常成立。
  - 伪代码：
```
for layer l in model:
    scores = softmax(Q_l @ K_l^T / sqrt(d_k))   # 注意力分数
    keep = topk_by_scan(scores, ratio)          # 顺序扫描+阈值比较，规则仿射访问
    x = layer_l(x[keep])                        # 只计算保留 token
    # KV cache：只写保留 token 的 K/V
```
  - Annotations：topk_by_scan 是剪枝引入的唯一新算子；扫描是步长 1 的顺序数组访问，pad 可预计算；剪枝比例 0% 即不剪（全部保留）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：软件 serving 层（vLLM/SGLang 的 token 级调度、LazyLLM 的 per-layer 剪枝器）；评估指标为剪枝比例（0-80%）下的延迟/能量与精度权衡。IroKnight 的评估（LLM，1024 in/out token、batch 1、剪枝 0-80%）：Llama4-Scout 加密变体延迟开销 0.3%-0.4%、GPT-OSS-120B 0.5%-0.7%，加认证 3.3%-3.4% / 3.6%-3.7%——因剪枝不破坏细粒度规则执行；能量随剪枝比例升高而下降（Llama4-Scout 加密 13.6%→9.7%、GPT-OSS-120B 9.6%→6.4%；认证 17.1%→14.2% / 13.3%→11.1%），因为计算量下降而权重 HBM 流量不变、加密成本被摊销。作用：长上下文 serving 里减少每层计算量、降低延迟与能耗，且与加密/完整性保护正交兼容。

涉及论文标题：
- IroKnight: Ownership-Preserving Neural Acceleration for Inference Serving

## DLRM 数据预处理算子（MapId Transform / MergeBucketizedDense / Batch Event Truncate）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据预处理算子把原始特征转换为模型可输入表示，训练时在分布式 worker 执行（Zhao 2022），推理时嵌入模型 module 同步执行、延迟直接计入端到端响应。广告推荐模型消费四类特征：dense（float32 连续属性）、sparse（list<int64> 分类 ID，变长需 jagged tensor）、weighted sparse（list<pair<int64,float32>>）、event-based（用户历史事件时间序列）。预处理变换三类：dense 归一化（BoxCox、Logit、one-hot + 线性缩放）、sparse 处理（top-k 截断可选排序、cryptographic hashing 映射 ID 到 embedding 索引、int64→int32 downcast）、特征派生（bucketize 连续值到分类 bin、多稀疏表集合操作、n-gram hashing），共 200+ 算子分支。三个代表性算子：(1) MapId Transform——把高基数稀疏 ID 重映射为密集连续整数（1-indexed 位置，未知映射为 0）：idx=bucketize(v,M)；clamp；若 M[idx]==v 输出 idx+1 否则 0。(2) MergeBucketizedDense（MBDT）——连续特征按每特征 border 列表批量 bucketization：Y_{f,i}=min{k | X_{f,i}<B_f[k]}，border 展平加 inf 哨兵、offsets 全局索引（例：feature0 值[0.1,0.4,0.8]→bin[0,1,2]，feature1 值[0.2,0.5,0.9]→[3,4,5]）。(3) Batch Event Truncate——嵌套 jagged tensor（outer_lengths 每用户事件数、inner_lengths 每事件属性数、values 展平属性数据）按 N 事件截断，跨多特征协调三层索引算术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline 中位置：raw features → 预处理算子链（dense 归一化 + sparse 截断/哈希 + 特征派生）→ embedding lookup（hash 后 ID 查表）→ NN 阶段。MapId 具体计算（伪代码）：
```
# V: [B] 输入 ID；M: 排序 mapping 表
idx = torch.bucketize(V, M)                    # 二分找插入位置
idx = torch.clamp(idx, max=M.numel()-1)
mapped = torch.gather(M, 0, idx)               # 查映射值
out = torch.where(mapped == V, idx+1, 0)       # 匹配→1-indexed，未知→0
# 例：V=[100,300,500,200,999], M=[100,200,300,400,500] → out=[1,3,5,2,0]
```
- MBDT 计算：每特征 border 展平（inf 哨兵分隔）+ offsets 记录起点，并行时每值在自身 border 区间二分（或向量化计数，对 3-10 元素 border 数组 O(n) 优于 O(log n)），输出加 offsets 得全局唯一 bin 索引。Batch Event Truncate：保留每用户前 N 个事件的所有属性（跨所有 feature 同步截断），被丢弃属性从 values 移除、outer/inner_lengths 重算（例：User0 有 [1,0,2] 属性事件 3 个、截断到 N=2 时丢弃第 3 事件 2 个属性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：PyTorch 参考用 torch.bucketize/clamp/gather/where、torch.compile 编译；MTIA 上这些 ATen 算子部分缺失（v2i：clamp.out/gather.out/sort.values_stable/all.all_out/_unique2 等；v3 仍缺 clamp.out/sort.values_stable/_unique2/unique_consecutive），导致 CPU 回退与分离式部署。KernelEvolve 生成 fused Triton kernel：MapId 把 4 算子融合为单 kernel（in-register 20 步编译期 unroll 二分、tl.where 无分支更新、coalesced block-parallel 布局），MTIA v2i 最高 4.07×、v3 最高 1.36×；MBDT 融合全流程（向量化计数、自适应 block size 64/128/256、寄存器驻留），v2i 2.94-9.25×、v3 2.31-3.09×；Batch Event Truncate 把逐 feature 串行循环改成单 launch 多 feature 并行 batched kernel，最高 14.5×、生产端到端 2×。这些算子低算术强度但决定部署架构，是"kernel 覆盖优先于 GEMM 优化"论点的依据。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

## Hamiltonian Simulation（哈密顿量模拟）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hamiltonian simulation 指按照目标量子系统的哈密顿量（Hamiltonian，系统的总能量算子 H）去演化一组量子比特：给定初始态 |ψ(0)>，求 t 时刻的态 |ψ(t)> = e^{iHt}|ψ(0)>（含 iH 约定），其中 e^{iHt} 是时间演化算子。它是量子计算的核心价值场景之一（Feynman 原始动机），广泛应用于材料科学、量子化学（分子能谱）、核物理与高能物理（格点场论）、凝聚态（Fermi-Hubbard、Heisenberg、Ising 模型）等经典方法难以处理的系统。困难在于：直接实现 e^{iHt} 需要把 H 分解为可执行的量子门序列，而对一般 H 无闭式分解，必须用近似方法（product formula / Trotterization、LCU、qubitization 等）。本论文目标即为哈密顿量模拟的编译优化：在达到给定精度（L2 范数 >99.5%）下最小化门数与电路深度。
- 关键组成部分：H 的 Pauli 串分解（H=Σ_i w_i P_i）、时间演化算子的近似展开（Trotter 乘积公式）、把近似展开编译为基本门线路（unitary synthesis），以及误差度量（L2 范数差矩阵、保真度 ≈1−(L2 norm)²）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 常规算法 pipeline：输入物理模型 → 二次量子化（费米子哈密顿量）→ 费米子到量子比特映射（Jordan-Wigner / Bravyi-Kitaev，见 Bravyi-Kitaev 条目）→ 把 H 分解为加权 Pauli 串（如 H=JΣ⟨i,j⟩Z_iZ_j+hΣ_iX_i 的 Ising 模型）→ Trotter 化展开 e^{iHt}≈(Π_k e^{iH_k t/N})^N → 每个 Pauli 指数 e^{iP t} 用 CNOT ladder + RZ 门实现 → 门级优化（重排、同时对角化、synthesis）→ 输出线路在量子硬件/模拟器上执行并测量。
- 本论文示例（1D Ising，图 3）：输入为部分 Trotter 化的 unitary 集合 {e^{i(ΣH_i)t}}，共同组成一个 Trotter step；编译输出为可执行线路。仿真验证用 8-10 qubit 的 LiH/HF 分子、Ising/Heisenberg/Fermi-Hubbard 自旋模型，扩展性到 28-220 qubit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：软件生态包括 Qiskit（PauliEvolutionGate，含 Rustiq/Paulihedral 集成）、Cirq、OpenFermion（分子哈密顿量生成与 fermion-to-qubit 映射）、BQSkit（unitary synthesis）。物理实现依赖量子硬件（超导、离子阱等）或经典 statevector 数值仿真（本论文在 A100 GPU + EPYC CPU 上做数值验证，未用真实量子硬件）。使用时：用户给定 H 与时间 t，编译器输出 e^{iHt} 的近似线路；精度由 L2 范数差矩阵度量，目标保真度决定所需 Trotter 步数与线路规模。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

## Trotterization（Trotter 乘积公式：Lie-Trotter / Trotter-Suzuki）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Trotterization 是把哈密顿量时间演化 e^{t(H_i+H_j)} 用"每个项单独指数化"的乘积来近似的一类数值方法，理论基础是 Lie-Trotter 乘积公式（product formula）：e^{t(H_i+H_j)} ≈ (e^{(t/N)H_i}e^{(t/N)H_j})^N，误差界 ||e^{t(H_i+H_j)}−(e^{(t/N)H_i}e^{(t/N)H_j})^N|| ≤ (t²/2N)||[H_i,H_j]|| + O(t³/N²)，其中 [H_i,H_j] 是两项的对易子（commutator），N 是 Trotter 步数。误差由"非对易项之间的 commutator"主导，随步数 N 线性抑制（一阶 Lie-Trotter）；更高阶用 Trotter-Suzuki 公式（二阶对称展开 e^{t(H_i+H_j)}≈(e^{(t/2N)H_i}e^{(t/N)H_j}e^{(t/2N)H_i})^N，误差 O(t³/N²)），消除奇阶误差项。本论文 Evaluation 明确：一阶用 Lie-Trotter 公式、二阶用 Trotter-Suzuki 公式，仅通过步数参数控制近似误差。
- 关键点：Trotter 误差直接依赖项间的非对易性（commutator），因此"减少非对易性影响"（本论文 Partial Trotterization 的出发点）能在不增加步数的前提下降低误差，或等价地在相同误差下减少所需步数与门数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（一阶，N 步）：
```
输入: H = H_1 + H_2 + ... + H_m（加权 Pauli 串）, 时间 t, 步数 N
dt = t / N
for step in 1..N:
    for k in 1..m:            # 每项单独指数化
        U_k = exp(i*dt*H_k)   # CNOT ladder + RZ 实现
    circuit.append(U_1 U_2 ... U_m)
误差: O( Σ_{i<j} |[H_i,H_j]| * dt^2 ) 每步，总 O( Σ|[H_i,H_j]| * t^2 / N )
```
- 本论文例子（Eq.1 与 Eq.3）：4 项互不对易的 H=H_i+H_j+H_k+H_l（H_i=X_1Y_2Z_3 等 3-qubit），全 Trotter 化误差 ∝ [H_i,H_j]+[H_i,H_k]+[H_i,H_l]+[H_j,H_k]+[H_j,H_l]+[H_k,H_l]（6 个 commutator 全保留）；要达到 <1% 误差需要把 N 提到很高，导致线路极长（自旋/费米子哈密顿量高保真模拟的公认瓶颈）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：几乎所有量子 SDK 的标准方法——Qiskit PauliEvolutionGate 默认一阶 Trotter、Cirq/OpenFermion 提供 trotter 模块；用户设步数（或按误差界自动推算）。使用时先分解 H 为 Pauli 串，再逐项实现 e^{iP dt}（Pauli 指数线路：基底变换 + CNOT 链 + RZ），重复 N 次。局限：每项必须单独分解、无法利用项间结构，误差随非对易对数增长（一阶 Δt²/N、二阶 Δt³/N²），本论文正以此为目标做 partial Trotterization 改进。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

## Partial Trotterization（部分 Trotter 化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Partial Trotterization 是本论文提出的新哈密顿量编译范式：不再对每个哈密顿量项单独指数化（全 Trotterization），而是把非对易项分组进 partition（本论文选每组至多 n=3 个不同 qubit，即 8×8 unitary），把"组内多项之和"的指数 e^{itΣH_i} 作为一个整体直接编译。例：全 Trotter 把 e^{i(H_1+H_2+H_3)t} 展开为 e^{iH_1t}e^{iH_2t}e^{iH_3t}，partial Trotterization 则可展开为 e^{i(H_1+H_2)t}e^{iH_3t}。核心收益：BCH 公式下误差主导项由组内 commutator 构成，分组后这些组内 commutator 随组的指数化一起消失——误差从全 Trotter 的 sum_{i<j}|[H_i,H_j]|Δt²/2 降为仅跨组项 sum_{A≤B}|[H_A,H_B]|Δt²/2（Eq.7-9），达到相同精度所需 Trotter 步数大幅下降。
- 理论结果：误差缩减随 group 大小 ~n_A² 组合增长（组内 commutator 数以 n_A² 量级被消除）；一阶/高阶 Trotter 下电路深度随 group 大小呈二次缩减（quadratic reduction）。实测（图 8）：一阶 Trotter 10 步下误差随每 unitary 的 qubit 数（1→3）急剧下降；非对易对占比不随 lattice 增大（图 9，n=3 与 n=5），保证常数 partition 大小的可扩展性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pipeline 第一步（Algorithm 1 Greedy Partitioning，Table 1 示例）：
```
输入: [X_3, X_1X_2, Z_3Z_4, Z_1]  # 4-qubit 自旋哈密顿量项
排序: 按最高 qubit 索引，平局按权重 -> [Z_1, X_1X_2, X_3, Z_3Z_4]
贪心分组: 每组并集 qubit 数 <= 3
  -> partition1 = {Z_1, X_1X_2}, partition2 = {X_3, Z_3Z_4}
输出: U_1 = e^{i(t/N)(Z_1+X_1X_2)}, U_2 = e^{i(t/N)(X_3+Z_3Z_4)}  # 8x8 unitary
```
- 误差对比例子（Eq.3 vs 4）：H=H_i+H_j+H_k+H_l 四互不对易项，全 Trotter 误差 ∝ 6 个 commutator；partial Trotterization 分组 {(H_i,H_j),(H_k,H_l)} 后误差只余 4 个跨组 commutator（2 个组内 commutator 被消除）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现需要解决两个挑战（论文 Section 3）：(1) 如何有效划分项（高密度 partition）——用"高层算子表示"做划分（high-level circuit partitioning），排序+贪心，比门级电路划分更稠密；(2) 如何高效编译分组指数 e^{itΣH_i}——用 MCTS unitary 分解（见编译框架库 MCTS 条目），因为逐项实现会退回 vanilla Trotterization 失去误差收益，通用 unitary 分解（QSD 等）门数又太高。使用时：输入 H、时间 t、步数 N、partition 大小 n（论文取 n=3），输出优化线路；n 越大误差收益越大（组内 commutator 以 ~n_A² 消除）但 unitary 分解难度上升。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

## Pauli String（Pauli 串）与哈密顿量分解

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pauli string 是 n-qubit 系统中长度为 n 的张量积算子 P = ⊗_{i=1}^n σ_i，其中每个 σ_i ∈ {X, Y, Z, I}（单比特 Pauli 算子或恒等）。所有 Pauli 串构成 n-qubit Hermitian 算子线性空间的一组完备基，因此任何哈密顿量（Hermitian 算子）都可分解为加权 Pauli 串之和：H = Σ_i w_i P_i（w_i ∈ R，本论文把权重吸收进项记 H=Σ_i H_i）。单个 Pauli 串的指数 e^{iPt} 可用 Pauli 门 + CNOT 链 + Z 旋转门精确实现（Pauli 指数线路），但"多个 Pauli 串之和的指数" e^{itΣP_i} 一般无闭式分解，必须近似——这正是 Trotterization 与 partial Trotterization 的用武之地。
- 本论文中 Pauli 串承载全部编译输入：partitioning 排序按"最高 qubit 索引 + 权重"、conflict graph 的对易性判定（[t_i,t_j]≠0 即加边）都直接作用于 Pauli 串；term 权重与 locality 决定分组密度与误差行为。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 分解与指数化流程：
```
输入: 哈密顿量 H（如 4-qubit 自旋链）
1) 分解为 Pauli 串: H = J1*X_3 + J2*X_1X_2 + J3*Z_3Z_4 + J4*Z_1   (权重并入项)
2) 单个 Pauli 串指数: e^{iθ X_1X_2} = H_1(CNOT_{1,2} RZ(2θ) CNOT_{1,2}) H_1   (CNOT ladder + RZ)
3) 多串和指数: 用 Trotter / partial Trotter 近似
```
- 对易性判定示例：X_1X_2 与 Z_1 共享 qubit 1 且算子不同（X vs Z）→ 不对易，在 conflict graph 中连边，被分组进同一 partition（Table 1 中 {Z_1, X_1X_2}）；而 Z_1 与 Z_3Z_4 无共享 qubit → 对易。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Pauli 串在 Qiskit（SparsePauliOp/Pauli）、OpenFermion、Cirq 等 SDK 中是标准数据类型；哈密顿量分解可用 fermion-to-qubit 映射自动生成（如分子哈密顿量经 OpenFermion → Qiskit）。使用时按本论文流水线：排序（最高索引、权重）→ 贪心分组（≤3 qubit）→ 冲突图分组与重排 → MCTS 重写。局限：分解到 Pauli 串后 term 数可能很大（尤其费米子/分子哈密顿量），但 locality 与权重信息帮助排序把非对易项聚拢。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

## Bravyi-Kitaev 映射（Bravyi-Kitaev Transformation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bravyi-Kitaev（BK）映射是把费米子（fermion）哈密顿量编码到量子比特的方法之一，与 Jordan-Wigner（JW）、parity 变换并列。JW 把每个费米子轨道占用数局域存在单个 qubit（代价是产生/湮灭算子的 Pauli 权重 O(n)，随系统线性增长）；BK 用递归编码（Update/Flip/Parity/Remainder 四类 qubit 集合，通过递归矩阵 β_n/β_n^{-1} 在占用数基与 BK 基之间变换）同时非局域存储占用数与 parity，使单个费米子算子的 Pauli 权重降到 O(log n)（单产生/湮灭从 O(n)→O(log n) qubit 操作），适合大系统电子结构模拟。本论文所有费米子模型（Fermi-Hubbard、LiH、HF、PD-1 蛋白）都用 BK 映射转成自旋哈密顿量后输入编译流水线。
- 相关变体：Bravyi-Kitaev Superfast（BKSF，Setia & Whitfield 2018）、BK-tree、symmetry-conserving 变体，OpenFermion 提供 bravyi_kitaev() 等实现；Qiskit Nature 提供 BravyiKitaevSuperFastMapper。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pipeline 位置：物理系统 → 二次量子化（费米子算子）→ BK 映射 → 自旋/Pauli 哈密顿量 → 本论文的 Kernpiler 编译（输入即 BK 映射后的 Pauli 串集合）。
- 编码逻辑（BK）：对每个轨道 i 维护 qubit 集合 U(i)（占用数变化时被翻转）、F(i)（parity 对应子集）、P(i)（低索引轨道 parity）、R(i)（F(i) 在 P(i) 内的补集）；产生算子 a_i^† 的 Pauli 表示为 (X_{U(i)} ⊗ ...) 的产物，权重 O(log n)。论文在可扩展性分析中特别指出：BK 映射下 Pauli 项权重随系统对数增长，因此大系统时权重项可能超出 partition 大小（n=3/5）装不下，导致非对易对占比呈对数曲线（图 9）——可用 constant-weight 映射（Derby et al. 2021）缓解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：OpenFermion 的 opconversions 模块（bravyi_kitaev / bravyi_kitaev_fast / bravyi_kitaev_tree）、PennyLane qml.bravyi_kitaev、Qiskit Nature 的 mapper 类。使用时：给定费米子哈密顿量（如 Fermi-Hubbard：qubit#=2×site，本论文 8-128 qubits；分子：LiH/HF 10 qubits；PD-1 蛋白 28-222 qubits），BK 映射输出自旋哈密顿量 Pauli 串集合，作为 Kernpiler 的输入。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

## Randomized Compilation（随机化编译）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Randomized compilation（随机化编译）是把量子线路中的相干误差（coherent error，系统性、可累积）转化为随机/随机化误差（stochastic error，随运行平均抵消）的技术，源于 randomized compiling 与 Pauli twirling 思想（Wallman & Emerson 2016；早期理论框架 [4,48,46,47,17,18]）。在 product formula 中，随机化编译通过"每步 shuffle Trotter 项的顺序"实现：让快速变化的演化项平均掉错误项，得到更好的误差缩放（Campbell 2019 randomized compiler；Childs et al. 2019 randomization）。本论文在 Trotter step 内的 group 间随机 shuffle 项顺序（图 3 Step 3），把"每个 Trotter step 都重复相同的近似误差"（coherent noise）变为"每个 step 不同的随机误差"（stochastic noise），从而在平均意义下降低误差。
- 关键区别：shuffle 只在 group 内部进行（不跨 group），保证 group 结构（跨 step 合并、边缘 group 位置）不被破坏——这是本论文在随机化与编译优化之间保持兼容的设计选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文用法（图 3 Step 3 + Algorithm 2）：
```
对每个 Trotter step k:
  groups_k = GreedyCommutingGroups(conflict_graph(H_k))  # 互交换 group
  对每个 group: randomize(顺序)                            # 组内随机 shuffle
  # 保持 group 边界: 不 shuffle group 之间的顺序
相邻 step 重排 + 跨 step 合并对易项 -> 输出 modified Trotter steps
```
- 效果：每个 step 的乘积顺序不同 → 近似误差的高阶项随机化，避免相干累积；与"把两个最大 group 放 step 两侧、相邻 step 翻转"（使相同 group 相邻可合并）协同工作，zero 额外近似误差地降低 unitary 数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：通用 randomized compiling 在运行时对每层应用随机 Pauli 伴算（gate twirling），在编译期/运行期都可做；本论文把它限制在"Trotter step 内 group 内顺序 shuffle"这一更轻量的形式，作为 Algorithm 2（REORDERTROTTERSTEPS）的一步。使用时与分组/重排/合并流水线集成，无需额外硬件或测量开销。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

## Data Structuring（DS，数据整理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Data Structuring（DS）是 point-based 点云网络（PCN）每个基本 Building Block 中的第一步，作用是把空间稀疏、无序的点云整理成"规则"的输入特征图供后续特征计算（FC）使用。一个点云表示为 x = {(p^n, f^n)}，p^n=(x,y,z) 是 3D 坐标，f^n 是特征向量。DS 先选取若干中心点（central points），再对每个中心点做邻居收集（KNN 或 Ball Query）形成 K 个点的 point subset，随后把这些点的特征向量从内存取出来组成输入特征图。DS 是 PCN 独有的操作，无法直接由商用 DLA（如 NPU）加速，未加速时是 PCN 的主要瓶颈；域专用 PCN 加速器主要就是定制 DS 单元（准确型如 PointACC/HgPCN，近似型如 EdgePC/Crescent）。L-PCN 中 DS 由 Data Structuring Unit（DSU）执行，含 Sampling Module（选中心点）、Neighbor Search Module（邻居收集）、Pruning Module（剪枝预构建的 Input Octree 得 Sampled Octree 与 Hub Octrees）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文（PointNet++ Set Abstraction，K=32，1024 点输入）中 DS 的 pipeline 伪代码：
```
# DS pipeline（PointNet++/DGCNN 的 Building Block 第一步）
Input: 点云 X = {(p^n, f^n)}, n=1..N; 预构建 Input Octree
# 1. Sampling：选中心点（如 FPS 从 1024 点选 512 中心点）
C = Sample(X)                      # 中心点集合，|C|≈N/2
# 2. Neighbor Gathering：对每个中心点 c∈C 收集 K 个最近邻
for c in C:
    subset[c] = KNN_or_BallQuery(X, c, K)   # 形成 32 点 subset
# 3. Feature Fetch：从内存取每个 subset 中点的特征向量 f
#    -> 相邻 subset 间共享的重叠点被重复取（冗余访存根源）
Fmap[c] = FetchFeatures(subset[c]) # (K, d_in) 输入特征图
```
  - 关键特征：相邻 point subsets 之间共享大量重叠点（论文基准测量可达 87.5%–93.75% 的重叠率），重叠点的特征被重复从内存取、重复进入后续 MLP——这是 L-PCN 要消除的冗余来源。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：GPU/CPU 上用 KNN（如 k-d tree / 暴力搜索）或 Ball Query（固定半径搜索）的 gather kernel；域专用加速器用定制硬件单元，如 PointACC 用 16 个并行距离计算器 + 32-way bitonic sorter 做硬件排名核，HgPCN 用 Octree 缩小搜索空间后排名收集，EdgePC 用 Morton code 索引法近似收集，Crescent 用 KD-tree 近似搜索。L-PCN 假设 Input Octree 由现有方法预构建，DSU 通过 Octree 搜索与剪枝（Pruning Module）得到 Sampled Octree 与 Hub Octrees，供后续岛化与重叠检测复用。论文未提供 DS 的软件实现代码；开源参考：PointNet++ 官方实现（https://github.com/charlesq34/pointnet2）的 farthest_point_sample 与 ball_query 算子。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

## Feature Computation（FC，特征计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Feature Computation（FC）是 point-based 点云网络（PCN）Building Block 中的第二步：对 DS 形成的 point subsets（输入特征图）执行 MLP 卷积层与 Pooling 层，把每个 subset 的 K 个点的特征聚合到中心点。FC 与传统 DNN 的卷积/全连接层功能相同，可直接由商用 DLA（NPU）或脉动阵列加速。例如 PointNet++ 中一个 subset 的特征维度在 MLP 中从 (32,6) 变成 (32,128)，最后 max pooling 把 32 点聚合为中心点的 128 维特征。论文指出：当 DS 被专用单元加速后，FC 成为当前 PCN 加速器的主要延迟来源（占比可超 85%），而 FC 中 MLP 占 FC 计算量的 98%+——因此削减 MLP 输入量是加速关键。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中 FC 的 pipeline 伪代码（K=32，MLP 6→128）：
```
# FC pipeline（Building Block 第二步）
Input: 输入特征图 Fmap[c] = (32, d_in)
for c in C:
    # 每个 subset 的 32 个点依次过共享权重 MLP
    H[c] = MLP(Fmap[c])          # (32,6) -> (32,128)，同一权重作用于每点
    Out[c] = MaxPool(H[c])       # 沿 32 点维池化 -> (1,128)，聚合到中心点
# 冗余：相邻 subset 共享的重叠点重复进入 MLP（如 subset A 与 G 共享的 D,E,F）
#   -> 重复的 MLP 计算量 = 重叠点数/32 * 总计算量（论文测量可达 ~90%）
```
  - L-PCN 的优化：Islandization Unit 阻止重叠点重复进入 FCU 的 MLP，只对非重叠点计算，重叠点直接复用 Hub Cache 中的 MLP 结果（经结果增量补偿），从而在算法层面把 MLP 输入量本质减少。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：GPU 上用 batch GEMM 把 K 点 × MLP 权重批量矩阵乘，再 max-pool；加速器上用脉动阵列（如 L-PCN 的 16×16 systolic array FCU）或商用 NPU 执行。L-PCN 的 FCU 由现有 AI 加速器 + Dataflow Controller 组成，Dataflow Controller 区分两条数据流：Hub point subset 全量计算并缓存结果，non-Hub point subset 只把非重叠点送入 MLP、重叠点从 Hub Cache 取缓存结果直接进 Pooling。论文未明确说明 FCU 的软件框架；参考实现为 PointNet++ 官方 PyTorch 代码的 shared MLP 层。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

## Octree-based Islandization（Octree 岛化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Octree-based Islandization 是 L-PCN 提出的点云划分方法：利用 Octree（八叉树，一种把 3D 空间递归细分为立方体 voxel 的空间数据结构）的邻接搜索，把 DS 步骤收集到的相邻 point subsets 聚类成"L-PCN Islands"——组内 point subsets 空间强相关（共享大量重叠点）。关键性质：一个 point subset 只能属于一个 Island。该方法分四步：(1) 从 Sampled Point Cloud 选若干中心点为 Hub points；(2) 对每个 Hub point 用 Octree 搜索逐轮收集相邻中心点成 Hub List（重复收集到的节点只保留给最近的 Hub point）；(3) 回原 Input Point Cloud 按 Hub List 把 point subsets 聚成 Islands；(4) 用 Island List 表示每个 Island，供后续 Hub-based Scheduling 使用。该方法与 GCN 加速器 I-GCN（MICRO'21）的 islandization 思想同源（把强内部连接、只与 hub 相连的节点簇聚类以提升片上局部性），但 L-PCN 面向点云的空间稀疏性与无显式邻居索引，改用 Octree 邻接收集。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中 Octree-based Islandization 的伪代码（基于 Sampled Octree 与 Sampled Point Cloud）：
```
# Octree-based Islandization（在 DSU 之后、FC 之前执行）
Input: Sampled Point Cloud（中心点集）; Sampled Octree（剪枝后的八叉树）
# Step1: 随机选固定数量中心点作为 Hub points
Hubs = RandomPick(CentralPoints)
# Step2: 对每个 Hub 用 Octree 搜索逐轮收集邻接中心点
for h in Hubs:
    hubList[h] = [h]
    while 存在未入任何 hubList 的中心点:
        # 每轮沿 Octree 向外扩展一圈相邻 voxel 节点
        nodes = OctreeSearch_Adjacent(SampledOctree, h, round++)
        hubList[h] += nodes 内的中心点
        # 若某 Octree 节点被多个 Hub List 重复收集，中心点只保留给最近 Hub（早轮收集视为更近）
# Step3: 回原 Input Point Cloud 形成 Islands（中心点同属一个 Hub List 的 point subsets 归为一个 Island）
# Step4: 每个 Island 用 Island List 表示（Hub point subset 在首行）
```
  - 时间复杂度受益于 Octree 搜索（树遍历，避免暴力搜索）；与 FPS 造成的"相邻迭代子集空间相距远"问题互补：岛化重排处理粒度，把空间相邻的高重叠子集聚在一起处理。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：L-PCN 在 Islandization Unit 的 Partitioning Module 中硬件实现，两个 Octree-Search Engine（OSE）基于 Morton code + linked-list traversal 在 Sampled Octree 上并行执行邻接节点收集，Sampled Octree 存于分层 BRAM 的 Octree Buffer（双端口）。使用：作为 DSU 与 FCU 之间的插拔模块，兼容准确型与近似型现有 PCN 加速器；论文未提供开源实现。一般参考：Octree 由现有方法预构建（如 ParallelNN HPCA'23 [6]）；I-GCN 的 islandization 开源参考（https://github.com/panmn/I-GCN，MICRO'21）。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

## Ball Query（BQ）与 KNN（K 近邻）邻居收集

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Ball Query（球查询）与 KNN（K-Nearest Neighbors）是点云网络 DS 步骤中为每个中心点收集邻居点形成 point subset 的两种标准算法。KNN 取与中心点欧氏距离最近的 K 个点；Ball Query 取以中心点为球心、固定半径 r 的球内所有点（最多 K 个，不足则 padding）。Ball Query 保证固定空间尺度（所有点都在恒定半径内），使学习的局部特征对不同点密度更可泛化；KNN 可能拉到距离差异很大的点。PointNet++ 用 Ball Query，DGCNN 用 KNN。L-PCN 论文以 K=32 为例：每个中心点收集 32 个最近邻形成 point subset，相邻 subset 之间共享大量重叠点（论文实测相邻 subset 重叠可达 87.5%–93.75%）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - PointNet++/DGCNN 中邻居收集伪代码：
```
# Ball Query（PointNet++）
for c in C:
    subset[c] = { p ∈ X : ||p - c|| <= r }   # 球内点，最多 K 个
# KNN（DGCNN 的 k-NN graph）
for c in C:
    subset[c] = argsort_k(||p - c||_2 for p in X)[:K]  # 最近 K 点
```
  - 在 L-PCN 中，邻居收集由 DSU 的 Neighbor Search Module 执行（准确型如 PointACC 的硬件排名核、近似型如 EdgePC 的 Morton 索引法/Crescent 的 KD-tree 搜索）；收集结果送入 Islandization Unit 做重叠检测与复用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：GPU 上用 PyTorch 的 ball_query / knn_point 算子（PointNet++ 官方仓库 https://github.com/charlesq34/pointnet2）；加速器用并行距离计算器 + 排序器（PointACC 的 16 并行 distance calculator + 32-way bitonic sorter）。L-PCN 的 Islandization 与 Hub-based Scheduling 依赖邻居收集的结果做空间聚类，论文未提供自定义邻居收集的代码。
  - **NS-FPS 补充（ISCA'26）**——NS-FPS 把邻居搜索（neighbor search）从"给每个中心点收集邻域点"升级为 FPS 自身的核心计算原语：利用 Voronoi 图证明距离缓存更新等价于"找落在新采样点搜索球内的邻居"（Eq.5 等价关系），从而把 FPS 重述为迭代邻居搜索。论文按复杂度把邻居搜索分为三类：蛮力搜索（每查询 O(N)，GPU 可并行但线性复杂度不实用）、空间划分（grid/voxel，每 cell O(N/g)，对偏斜分布退化）、层次树（k-d/octree，O(log N) 但不规则访存与遍历开销、动态点云建树代价高）。NS-FPS 的平衡方案：Morton cube 划分 + 自适应半径球查询——每轮只枚举与搜索球相交的 cube（用索引表取点），半径 d_k 随采样推进自适应收缩（120k 帧从覆盖大量 cube 快速降到 <1m），显著减少每轮球查询范围。相关邻居搜索加速器：QuickNN、KD Bonsai（ISA 扩展压缩 k-d 树）、ParallelNN（并行 octree）、Tigris、CAMPER，NS-FPS 借鉴 octree 思路但用 Morton 码查找替代显式建树。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

## Farthest Point Sampling（FPS，最远点采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Farthest Point Sampling（FPS）是点云下采样的经典算法：迭代地从未选点中选出"到已选集最远距离最大"的点加入选集，直到选够目标数量。相比随机采样，FPS 保证采样点在空间上均匀覆盖整个点云（避免只采到局部区域）。PointNet++ 的 Sampling Layer 用 FPS 从 N 个点选 N' 个中心点（如 1024→512）。FPS 也是 L-PCN 论文指出的一个 PCN 特有挑战的来源：由于 FPS 倾向选相距最远的点作为中心点，标准 PCN 流程中相邻迭代处理的 point subsets 往往空间相距很远（破坏空间相邻执行顺序），这使"相邻子集重叠"在原始执行顺序下不可见，需要 Octree-based Islandization 重排处理粒度才能利用空间局部性。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - FPS 伪代码（PointNet++ Sampling Layer）：
```
def FPS(X, M):                      # X: N 点云, M: 目标中心点数
    C = [random_point(X)]           # 随机起点
    dist = ||p - C[0]||_2 for p in X
    while len(C) < M:
        c = argmax_p(dist[p])       # 到已选集最远距离最大的点
        C.append(c)
        dist[p] = min(dist[p], ||p - c||_2)   # 更新最近距离
    return C                        # M 个均匀覆盖的中心点
```
  - L-PCN 中 FPS 由 DSU 的 Sampling Module 执行，输出 Sampled Point Cloud（中心点集）；后续 Octree-based Islandization 基于这些中心点选 Hub points 并做邻接聚类。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：PyTorch 的 farthest_point_sample 算子（GPU 并行版），复杂度 O(M·N)；加速器用硬件采样模块。L-PCN 论文未给出 FPS 的自定义硬件实现细节（沿用现有 PCN 采样方法），强调 FPS 与岛化/调度互补。参考开源：PointNet++ 官方仓库 https://github.com/charlesq34/pointnet2。
  - **NS-FPS 补充（ISCA'26，硬件-软件协同设计）**——FPS 是内存受限而非计算受限：RTX 3090 上 120k 点 25% 采样需 >900ms、95% 时间耗在内存事务，profile 显示 600M 次请求、164.65GB L1 + 74.81GB L2 请求，缓存吞吐 95.62% vs SM 指令吞吐 27.17%；FPS 在 16k 输入的点云网络中占 30–70% 总运行时间。维护 N 长距离缓存 T（Eq.2 递推 t_p^m=min(t_p^{m-1},d(p,s_m))）把每轮复杂度降到 O(N)，但仍有 O(MN) 全量扫描。NS-FPS 把 FPS 重述为邻居搜索：用 Voronoi 图证明距离更新只发生在以最新采样点 s_k 为球心、半径 d_k=min_{s_i∈S_{k-1}}||s_k−s_i||² 的球内（该球严格包含真实 Voronoi cell），每轮更新点数从 O(N) 降到 O(N/k)，总复杂度 O(N log N)，且采样结果与传统 FPS 完全一致（lossless）。CPU 版在 16k–120k 点相对 vanilla FPS 加速 100.1×/130.3×/106.2×/191.7×，相对 QuickFPS-CPU 在 64k/120k 上 1.22×/1.80×；ASIC 版相对 GPU 加速 17.2×–81.6×、内存访问降 1700×。
  - NS-FPS 算法 pipeline 伪代码（Algorithm 1）：初始化：Morton 码桶排序建索引、T←∞、随机选 s_0；每轮 k：(a) d_k=T[s_{k−1}]；(b) 枚举与球 B(s_{k−1},d_k) 相交的 Morton cube，对 cube 内点 p 更新 T[p]=min(T[p],||p−s_{k−1}||²)；(c) 对受更新块刷新 16:1 层次 max 缓存；(d) s_k=argmax_p T[p]，加入采样集。搜索半径自适应收缩是剪枝关键：120k 帧前 100 轮半径覆盖很多 cube、占 27.3% 总迭代时间，随后半径快速降到 1m 以下。
  - NS-FPS CPU 实现开源：https://github.com/satreeby/ns-fps/（C++17 + pybind11，`pip install -e .`，`yf.fps(points, n_samples=..., range=SpaceRange)` 返回采样索引；README 声称 CPU 上较 naive FPS 最高 191×、较 QuickFPS-CPU 1.72×、较 naive GPU FPS 4.2×）。GPU 侧 baseline 为 OpenPCDet 的 CUDA FPS 实现，profiling 用 NVIDIA Nsight Systems/Compute。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

## PointNet++（含 Set Abstraction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PointNet++ 是 Qi 等人（NeurIPS 2017）提出的分层点云特征学习网络，解决原始 PointNet 缺乏局部结构捕获能力的缺陷。它由多个 Set Abstraction（SA）层堆叠而成，每层：Sampling Layer（FPS 选中心点）→ Grouping Layer（Ball Query 收集局部邻域）→ PointNet Layer（共享 MLP + max pooling 把邻域聚合为中心点特征）。随着层加深，点数减少、感受野半径与特征维度增大，形成类似 CNN 的分层表示。SA 层即 L-PCN 论文所称 PCN Building Block 的实例。论文用 PointNet++ 三个变体做 benchmark：PointNet++(c)（分类，ModelNet40）、PointNet++(ps)（部件分割，ShapeNet）、PointNet++(s)（语义分割，S3DIS）；其前两个 Set Abstraction 占整体运行时间 90%+（论文 Figure 4 基准）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - PointNet++ SA 层 pipeline（L-PCN 论文即以此为 Building Block 实例）：
```
# Set Abstraction 层 = DS（采样+分组）+ FC（PointNet MLP + 池化）
# DS:  FPS 选 N' 中心点 -> Ball Query 每组 K=32 邻居 -> 取特征
#      （相邻组共享 ~90% 重叠点 -> 冗余访存）
# FC:  共享 MLP 对每组 32 点计算 (32,6)->(32,128) -> max pool -> (1,128) 中心点特征
#      （重叠点重复进 MLP -> 冗余计算）
# 归一化: 非中心点 XYZ 减去中心点 XYZ（相对坐标），
#         -> 这是 L-PCN 需要 Result Delta Compensation 的原因
```
  - L-PCN 对 SA 的加速点：在 DS 与 FC 之间插入 Islandization Unit，把高重叠相邻子集聚类成 Island 并按 Hub-based Scheduling 复用缓存结果，理论 feature fetching 减 55.2%–90.2%、feature computation 减 45.4%–73.1%（PointNet++/DGCNN 基准）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：官方 PyTorch 实现（https://github.com/charlesq34/pointnet2 与 yanx27/Pointnet_Pointnet2_pytorch），含 farthest_point_sample、query_ball_point 与 shared MLP；加速器实现：PointACC、HgPCN、L-PCN 等。L-PCN 用 PointNet++(c) 作为主要原型（DSU 采用 PointACC 的 Mapping Unit、FCU 16×16 脉动阵列）做资源与 cycle-accurate 延迟评估。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

## DGCNN（含 EdgeConv）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DGCNN（Dynamic Graph CNN，Wang et al., TOG 2019）是点云分类/分割网络，核心是 EdgeConv 层：对每个点用 KNN 在特征空间构造动态 k-NN 图，对每条边计算边特征（中心点特征与邻居点特征之差拼接），再经共享 MLP 与 max 聚合更新点特征。由于图在每个 EdgeConv 层动态重建，DGCNN 能捕获局部几何结构。EdgeConv 层即 L-PCN 论文所称 PCN Building Block 的另一实例（用 KNN 而非 Ball Query 收集邻居）。论文用 DGCNN(c)（分类，ModelNet40）与 DGCNN(s)（语义分割，ScanNet）作为 benchmark，并在 DGCNN(c) 上观察到"激活只在本 Building Block 末尾应用时 CONV(A−B)=CONV(A)−CONV(B) 严格成立，可完全补偿结果增量"的特殊情形。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - DGCNN EdgeConv 层 pipeline（Building Block 实例）：
```
# EdgeConv = DS（KNN 动态构图）+ FC（边特征 MLP + max 聚合）
# DS:  对每点 p，KNN 取 K 个邻居构成边 (p, q_i)
#      （相邻点的邻域共享大量重叠邻居 -> 冗余访存与计算）
# FC:  edge_feat_i = MLP([p_feat || (q_i_feat - p_feat)])   # 边特征
#      p_new = max_i(edge_feat_i)                          # 聚合
#      （重叠邻居点重复参与 MLP 与聚合 -> 冗余）
```
  - L-PCN 对 DGCNN 的加速：与 PointNet++ 相同，用 Islandization Unit 聚类 + Hub Cache 复用消除重叠点冗余；理论上 feature fetching 与 feature computation 的削减同样适用。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：官方 PyTorch 实现（https://github.com/WangYueFt/dgcnn），用 knn 构建动态图；加速器实现：DGCNN 是常见 PCN 加速器 benchmark（如 PointACC、EdgePC、Mesorasi、L-PCN）。L-PCN 论文未提供 DGCNN 自定义实现，沿用公开模型定义。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

## Morton code（Z-order 曲线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Morton code（莫顿码，也称 Z-order curve / Z 曲线）是一种空间填充曲线编码：把多维坐标（2D/3D）的各位按维度交错（bit interleaving）合并成一个 1D 整数，使空间上相近的点映射到数值上相近的编码，从而把空间局部性转化为一维线性局部性。Morton code 广泛用于四叉树/八叉树索引、GPU 并行建树（按 Morton 排序点即可递归细分）、点云压缩与数据库空间索引。L-PCN 中，Octree-Search Engine（OSE）基于 Morton code 在 Octree 上执行查询（配合 linked-list traversal 遍历模式），两个 OSE 并行处理两条 Octree-search 查询；EdgePC 也用 Morton code 对点结构化后做近似邻居收集。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 3D Morton code 编码伪代码（每坐标 b bit，总码长 3b bit）：
```
def morton3(x, y, z):              # 逐位交错
    code = 0
    for i in range(b):             # 从低位到高位
        code |= ((x >> i) & 1) << (3*i)
        code |= ((y >> i) & 1) << (3*i+1)
        code |= ((z >> i) & 1) << (3*i+2)
    return code                    # 空间邻近点 -> 码值邻近（Z 形扫描序）
# L-PCN 用法：OSE 用 Morton code 定位 Octree 节点/点，
#   配合 linked-list traversal 在 Sampled Octree / Hub Octree 上搜索
```
  - 在 L-PCN 中的作用：Octree 搜索的索引原语——按 Morton 码对点排序可线性化空间位置，使八叉树节点的子节点地址可由 Morton 码直接推导（常数时间定位），两个 OSE 才能高效并行执行邻接节点收集与重叠检测。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：软件库如 Go 的 github.com/habedi/morton、Rust 的 space crate（支持 BMI2 pdep/pext 加速）；硬件里用按位交叉逻辑或 LUT。使用：GPU 并行建 octree 时先对点算 Morton 码并排序；L-PCN 的 OSE 用 linked-list traversal 模式（Madeira et al. GPU Octrees and optimized search [33]）遍历。论文未提供 OSE 的公开 RTL；一般参考 ParallelNN（HPCA'23）的并行 Octree 搜索加速器。
  - **NS-FPS 补充（ISCA'26）**——NS-FPS 以 Morton 码作为 FPS 邻居搜索的空间索引原语，替代 k-d 树/八叉树建树：坐标量化成 15/15/11-bit 整数后取 7/7/3 个 MSB 按位交织成 17-bit Morton 码（默认配置；敏感性分析比较 (5,5,1)/(6,6,2)/(7,7,3)，32k 点用 (6,6,2) 最优、120k 用 (7,7,3) 最优），每个码隐式定义一个 3D cube。重排用**桶排序**（线性时间，免比较排序/免建树，显著低于 k-d 树的预处理延迟）。邻居查询时枚举与搜索球 B(s_k,d_k) 相交的 cube、用索引表取 cube 内点，滤掉不可能更新的远处点。这与 L-PCN 用 Morton 码做八叉树查询的用法互补：NS-FPS 直接以 Morton 码分组存点、无显式树结构。
  - NS-FPS 中 Morton 编码伪代码（硬件 4 级流水）：p(x,y,z)→量化 (x_q,y_q,z_q)（15/15/11-bit）→取 MSB (x_h,y_h,z_h)（7/7/3-bit）→交织得 17-bit Morton code→查 Occupancy Table（新 cube 建 Page Table 项，否则复用）→分配/追加 Page Memory。GPU 版 NS-FPS 因 Morton 遍历在 GPU 通用内存层级上开销大，小规模劣于 QuickFPS-GPU，故需 ASIC 释放潜力。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

## Result Delta Compensation（结果增量补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Result Delta Compensation 是 L-PCN 使"跨 point subset 复用缓存 MLP 结果"成为可能的关键算法技术。现代 PCN（如 PointNet++）在 MLP 前把非中心点的 XYZ 坐标按中心点归一化（减去中心点坐标），因此两个不同中心点子集的共享点（如 subset A 与 G 共享的 D,E,F）在各自子集中的输入不同，缓存的 MLP 结果不能直接复用。L-PCN 利用 MLP 的线性部分：w·(P−P_G) = w·(P−P_A) + w·Δ(A−G)，即用缓存的 w·(P−P_A) 加上由中心点差 Δ(A−G) 计算的增量 w·Δ(A−G) 补偿出实际需要的值。由于 MLP 含非线性激活，MLP(A−B) ≈ MLP(A)−MLP(B) 仅近似成立（Mesorasi 可因此损失至多 0.9% 精度）；L-PCN 只对重叠点做补偿（选择性近似），对非重叠点保持精确计算，因此精度损失更小；当激活只在 Building Block 末尾应用时（DGCNN(c)、PointVector-L），CONV(A−B)=CONV(A)−CONV(B) 严格成立，可完全补偿、零精度损失。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - L-PCN 论文中的补偿公式与流程（Figure 8 示例，N=3 重叠点）：
```
# 处理 Point-subset G 时，复用 subset A 缓存的 N 个重叠点结果
for d in overlapping(D,E,F):
    cached = HubCache[d]                      # w·(P_d - P_A) 已缓存
    w_delta = MLPlinear(w, Delta(A-G))        # 增量 w·Δ(A-G)（送入 FCU 计算）
    result[d] = cached + w_delta              # Eq.1 补偿后的实际复用值
# Eq.1:  w·(P-P_G) = w·((P-P_A) + Δ(A-G)) = w·(P-P_A) + w·Δ(A-G)
# 补偿带来一次性额外计算开销 -> feature computation 节省略低于访存节省
```
  - 与 Mesorasi 的 Delayed-Aggregation 对比：Mesorasi 对全部 MLP 结果做近似（MLP(A−B)≈MLP(A)−MLP(B)），L-PCN 只对重叠点近似——非重叠点（多为点云边界点，max pooling 常保留其高值）精确计算，故精度更优、跨域鲁棒性更好。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：L-PCN 中把 Δ(A−G) 输入 Feature Computation Unit 计算 w·Δ(A−G)，与 Hub Cache 取出的缓存结果相加完成补偿；这是 Islandization Unit 内 Hub Cache 读路径上的附加加法器逻辑。论文未提供补偿单元的独立 RTL；GDPCA 的 Geometry-aware Differential Update 与 Mesorasi 的 Delayed-Aggregation 是相关先例（GDPCA 用 Bit-Pragmatic 加速器 PRA 利用低位宽差分输入）。

涉及论文标题：
- L-PCN: A Point Cloud Accelerator Exploiting Spatial Locality through Octree-based Islandization

## Minimap2（seed-chain-extend 长读基因组比对工具）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Minimap2（Li, H.，Bioinformatics 34(18):3094-3100, 2018，GitHub: lh3/minimap2）是通用核酸序列两两比对工具，按 **seed（播种）→ chain（链化）→ extend（扩展）** 三阶段流程工作，是长读（PacBio/ONT）比对与 de novo 组装的事实标准（NextDenovo、Canu、MetaFlye、Shasta 等内部用它做 all-to-all 比对）。三阶段：(1) Seeding——把 reference/query 切成滑动窗口、每窗口取字典序最小 k-mer（minimizer）作种子，匹配 minimizer 对（anchor）经内存哈希表随机查询发现；(2) Chaining——对 anchor 列表做 1D 动态规划：每 anchor 回看至多 N 个前驱、以 max reduction 计算 chaining score（奖励重叠、惩罚 gap），连出长链；(3) Extend——对每 chain 做 banded Smith-Waterman-Gotoh（affine gap，默认 20 kbp band）+ 逐单元 traceback，输出 SAM。de novo 组装中 Minimap2 承担 ~76% 运行时间与数量级更高的内存（NextDenovo 人类基因组实测）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 reference-based 比对流程：`minimap2 -ax map-ont ref.fa reads.fq > out.sam`（参考比对）或 `minimap2 -x ava-ont reads.fq reads.fq > overlaps.paf`（all-to-all，供 de novo 组装）。内部：① 索引——对 reference 所有 minimizer 建哈希表（可选 HPC minimizer 压缩同聚物跑）；② seeding——对 query 每窗口取 minimizer、查表得 anchor（高频 minimizer 可过滤）；③ chaining——1D DP 递推 chaining score、按 0.8 阈值保留主/次比对；④ extend——双向扩展 + ksw2 SIMD 加速的 banded SWG 精化比对。Lembas（ISCA'26）将其作为算法语义"金标准"：不引入任何 trade accuracy 的近似/启发式过滤，结果必须包含 Minimap2 相同配置下的全部结果；三阶段分别被外部内存 columnsort 播种加速器、流式 chaining 加速器（复用 Guo et al. FCCM'19）、tiled SWG 扩展加速器替换。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C 语言单文件可执行程序，SIMD（SSE/AVX/AVX-512）向量化 DP；`-k`/`-w` 控 minimizer 长度与窗口、`-A/-B/-O/-E` 控替换/缺口罚分、`-z` 控 chain 拆分阈值；内存 chunking 参数可调（降低内存但跨 chunk 不做匹配检查、输出质量下降——Lembas 论文实测其 7× 内存与质量差异）。使用场景：reference-based read mapping、all-to-all 重叠检测（de novo 组装）、全基因组比对（WGA）；是 BWA-MEM 之外长读领域的对标基准。后续改进见 Li 2021（Bioinformatics 37:4572，"New strategies to improve minimap2 alignment accuracy"）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration

## Smith-Waterman-Gotoh（SWG，仿射 gap 罚分的动态规划序列比对）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SWG 是序列局部比对的动态规划算法：Smith-Waterman（1981）定义局部最优比对分数；Gotoh（1982, J. Mol. Biol. 162(3):705-708）引入**仿射 gap 罚分**辅助数组把复杂度从 O(mn²) 降到 O(mn)。仿射罚分 $g(k)=\alpha+\beta k$（$\alpha$=gap 打开罚分、$\beta$=gap 延伸罚分）。递推三值（Lembas 论文的 S/E/F 即此 H/E/F）：$E[i,j]=\max\{E[i,j-1], S[i,j-1]-\alpha\}-\beta$（水平 gap）、$F[i,j]=\max\{F[i-1,j], S[i-1,j]-\alpha\}-\beta$（垂直 gap）、$S[i,j]=\max\{S[i-1,j-1]+Z[A_i,B_j], E[i,j], F[i,j], 0\}$（0 截断保证局部性）；计算顺序严格按 $i,j$ 递推（每 cell 依赖左/上邻居）。**banded SW** 忽略远离对角线的区域降计算量（Minimap2 默认 20 kbp band，研究常用 W=1024 折中精度/性能）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 正向分数矩阵的**反斜对角线（wavefront）依赖结构**与脉动阵列/GPU 天然契合：同一反斜对角线上的 cell 可并行计算。Lembas extend 加速器（ISCA'26）：16 kernel × 16 PE 的 1D systolic array，每 PE 算矩阵一行、交错推进，每 cycle 用一个 cell 的四个输入（$S[i-1][j-1]$、$E[i-1][j]$、$F[i][j-1]$、$b^j$）算 S/E/F 三值——$F$ 随 PE 从左向右移动而缓存在 PE 内（无需跨 PE 传递）、$E/b$ 走 E,b 寄存器链（差 1 cycle）、$S$ 走 2 元素 FIFO（对角线差 2 cycle）→ 全流水。反向 traceback 每步依赖前一步、串行紧依赖，FPGA 低时钟（数百 MHz）下低效——Lembas 用 8×8 tile 位并行 traceback（见"tiled bit-parallel traceback"条目）解决。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU 上 ksw2（Minimap2 内嵌）以 SIMD 条带化（striped SW，Farrar 2007）向量化；GPU/FPGA 上以 systolic/wavefront 阵列映射（AGATHA、Logan、PipeBSW 等）。Lembas 实测 extend 48 GCUPS/FPGA（双 FPGA 96 GCUPS，mm64 为 27.21 GCUPS，≈4×）；W=2048 时 traceback 开销比次优设计（PipeBSW/Li21）低 1.77×；W=512–25K 全范围总延迟最低（图 16/17，对比 Cheng24/Li21/Liao18/Turakhia18/Teng23）。使用场景：任何需要最优局部比对的序列比对/组装（read mapping 精化、重叠检测、数据库搜索）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration

## 外部内存 columnsort（External-Memory ColumnSort，排序替代哈希的播种算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Columnsort（Leighton, "Tight bounds on the complexity of parallel sorting"，1984/85）是面向"高瘦" r×c 网格的并行排序算法：把全局排序拆成反复"列内排序 + 网格转置/移位"——列内排序各列互不通信、天然可并行；转置后同一批值落入不同列，多轮后全局收敛。约束 $r \ge 2c^2$（网格必须高瘦）。经典 8 步：排序列 → 转置 → 排序列 → 逆转置 → 排序列 → 前移 ⌊r/2⌋ → 排序列 → 回移。**外部内存变体**（Lembas 新颖点，ISCA'26）：数据集超出 FPGA 片上/HBM 容量时把 256 MB 列存于 NVMe、经 PCIe 来回搬运（排序 + 转置），设计取舍随之外移（用简单 16-to-1 单发射 merger 即可达 PCIe 上限，无需复杂宽发射 merger）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Lembas 用它把 Minimap2 播种的"内存哈希表随机查找"替换为"全量外部排序 + 流式 zip"：① minimizer parse 产出 16 B 〈minimizer, index〉 元组流 → 溢出存 NVMe；② columnsort 加速器：数据组织成 r×c 网格（$r \ge 2c^2$，256 MB HBM bank 上限 → 可排序 ≤512 GB，超出按 512 GB 分块后软件合并），16 个 16-to-1 单发射 merge-sort kernel 各独占一对 HBM pseudo-channel（1 数据 + 1 scratchpad），250 MHz/4 GB/s 每 kernel，6 次 sweep 排序 256 MB 列；③ 4 轮列排序 + 3 次转置回传（host 侧多 KB 大块 memcpy 重组转置列）→ 8 GB/s 双工 PCIe 上限 → 有效端到端排序吞吐 ~2 GB/s；④ 两有序 minimizer 流流式 zip 匹配得 anchors → 按 idxR 二次 columnsort（供 chaining 用）。16-to-1 是带宽/芯片面积最优：更大 fan-in 减 sweep 但装不下、更小 fan-in 需更多 pass；16 kernel 恰好饱和 U50 的 ~8 GB/s 双工 PCIe。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 一般实现：FPGA 上既有 Hypersort（ICFPT'22，HBM-FPGA 内存内实现）；CPU/分布式系统上用 out-of-core 变体（Dartmouth "Stupid Columnsort Tricks"：4-pass、s 不必整除 r、$r\ge 4s^{3/2}$ 可松弛）。Lembas 的用法：seeding 加速器（minimizer 解析与 anchor 匹配简单、与 columnsort 共享同一 bitfile），资源占用 Seed 361,624 LUT (41.53%)/517 BRAM (38.47%)（表 IV）。效果：seed 内存恒定 ~8 GB（7× 降低、无 Minimap2 chunking 质量损失）、seed 性能比 mm64 快 70%（比 G³SA 慢 15%）。使用场景：任何"哈希随机访问成为内存容量瓶颈、可转化为排序+流式扫描"的数据密集型比对/去重。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration

## tiled bit-parallel traceback（分块位并行回溯）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SWG 比对有前向（分数矩阵）与反向（traceback）两趟：traceback 从最优分数单元出发、按"上一步决策决定下一步"逐单元回溯出比对路径——**每步依赖前一步**、串行紧依赖，需 GHz 级高频才高效；FPGA 时钟仅数百 MHz，逐单元 traceback 极慢。此前工作或省略 traceback（只加速分数矩阵，端到端收益有限）、或用 2-bit 编码路径（Cheng24 [8]）、块单元级间并行（G³SA [25]、Nawaz [61]）。Lembas（ISCA'26）的 **tiled bit-parallel traceback**：把矩阵切成 8×8 tile，前向时在每个 tile 边缘单元（灰格）**完整编码"从相邻 tile 外起点到本 tile 各边缘单元的最优路径"**——每步 2-bit（x/y 偏移），8×8 tile 内最长 16 步、15 个边缘单元 → 每 tile 32×15=480 bit，正好对齐 512-bit HBM 接口；反向时读编码、**popcount x/y 位**即可算出下一 tile 入口单元，**每 cycle 前进一整 tile** 而非单单元。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 实现细节（论文 VI-B，图 7b/8）：每 PE 带**历史寄存器（Hist Reg）**——维护最近两个访问 cell 的滑动窗口最佳历史；PE 算出当前 cell 分数并决定下一步时，从三个输入历史寄存器按方向选一个、把新确定的路径步（2-bit）追加进去；前向算分与历史编码同流水完成。反向流程：读 tile 边缘单元编码（例 4×4 tile 的 `11111010`，每 2 bit 一步 x/y 偏移）→ popcount x 位得横向位移、popcount y 位得纵向位移 → 定位下一 tile 入口 cell（一步跳过整个 tile）→ 重复直到出 band。8×8 是 timing 约束下的最大 tile（更大则历史寄存器访问不满足时序）；每 tile 480 bit 与 512-bit HBM 接口匹配；每次加载 tile 预取 4 个缓解 HBM 延迟。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 在 PE 微架构内（16×16 PE 脉动阵列 + 历史寄存器），扩展阶段资源占用 Extend 431,957 LUT (49.61%)/105 BRAM (7.81%)（表 IV，tile 寄存器面积占比小）。效果：traceback 从"每 cycle 1 单元"变"每 cycle 1 tile"（每 cycle 前进 8 单元）；W=2048 时 traceback 开销比次优设计低 1.77×；窄 band（W=1024 研究常用）下收益显著；extend 总计 48 GCUPS/FPGA。使用场景：任何"低时钟硬件 + 需高质量 traceback"的 DP 比对加速器（FPGA、PIM、近存）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration

## 相位多项式（Phase Polynomial）与 sum-over-paths 表示

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 相位多项式是量子线路中一类 {CNOT, Rz} 子电路（phase polynomial circuit）的紧凑代数中间表示（IR），由 Amy、Maslov 与 Mosca 在 2013 年以 sum-over-paths（路径求和）形式引入（"Polynomial-Time T-Depth Optimization of Clifford+T Circuits via Matroid Partitioning" 等系列工作，T-par 工具）。其核心形式（本文 Eq.1/2）：对任意计算基态 |x⟩（x∈F₂ⁿ），电路作用可写为 U|x⟩ = e^{i·p(x)} |g(x)⟩，其中 p(x) = Σ_i θ_i·(x₁y_{i1} ⊕ … ⊕ xₙy_{in}) 是布尔 parity 的带权（旋转角 θ_i）线性组合（相位函数），g(x) 是 CNOT 网络实现的 GF(2) 线性可逆变换（输出基变换）。换言之：CNOT 计算输入变量的 XOR parity（奇偶性），Rz(θ) 在这些 parity 上施加相位旋转。一个 phase polynomial block 是通用线路中只含 {CNOT, Rz} 的极大连续子电路；H 等换基门会终止该 region。
- 该表示的精妙处：把"门序列"抽象为"一组 (parity 向量, 旋转角) 对 + 一个线性变换矩阵"，从而把线路优化转成矩阵/线性代数问题（parity 项合并、CNOT 网络合成），并可精确用于等价性验证（两个 {CNOT,Rz} 电路相位多项式相同则酉等价）与硬件感知合成。Rz 包含 Clifford 旋转（Z、S）与非 Clifford 旋转（T），因此 {CNOT,Rz} 同时覆盖 Clifford 与非 Clifford 门，是 FT 与非 FT 编译的共同目标。
- 论文动机数据：在 MCX、Grover、Shor、QAOA、Hamiltonian 模拟等基准中，>75% 的门属于 {CNOT,Rz} 区域（部分 >90%），且 FT 视角下 CNOT 与 T 占比相当——因此相位多项式优化直接命中线路主导成本结构。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 算法 pipeline（构造相位多项式）伪代码：输入 {CNOT,Rz} 线路 → ① 初始化输出基变换 g = 单位阵（n×n GF(2)），相位项表 P = {}；② 逐门处理：CNOT(c,t) 使 parity 传播——对每个已记录相位项，若其 parity 含 qubit c 则把 qubit c 替换为 c⊕t（等价于更新 phase-parity 矩阵列）；同时 g 的第 t 行 ← 第 t 行 ⊕ 第 c 行；③ Rz(θ) on q：把当前 q 上的 parity 向量 y 加入 P（角度累加，θ₁y + θ₂y → (θ₁+θ₂)y）；④ 输出 P（相位项集合）与 g（线性变换矩阵）。张量计算例子（本文 Fig.2）：3-qubit 电路 p(q₀,q₁,q₂) = (π/4)q₀ + (π/2)(q₀⊕q₁) + (π/4)(q₁⊕q₂) + (π/4)q₀ = (π/2)(q₀⊕q₁) + (π/4)(q₁⊕q₂)（两个同名 parity 合并），g(q) = (q₀, q₀⊕q₂, q₀⊕q₁⊕q₂)；原电路 5 CNOT + 3 T，等价电路 4 CNOT + 1 T。
- 作用：把线路优化从"逐门重写"提升为"代数化简 + 网络合成"两层问题——先合并相位项（rotation merging），再最小化实现这些 parity 与输出基变换的 CNOT 网络（NP-hard，需启发式）。PhasePoly 论文在此基础上新增：phase 与 output parity 联合优化（耦合矩阵）与跨 block 表示。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：经典侧 Python/C++ 库解析 OpenQASM 线路构造相位多项式。代表性工具/算法：T-par（Amy 2013，T-count 最小化）、Gray-Synth（Amy 2018，CNOT 网络贪心 Gray 码合成）、Rotations/Quilc 与 TKET 的 rotation merging、QUESO 用多项式恒等式过滤生成等价电路类。开源：T-par/Gray-Synth 见 Amy 主页与 arXiv:1804.06022；QUESO（PLDI'23）仓库 qqq-wisc/queso；PhasePoly 开源 https://github.com/ruadapt/PhasePoly。使用场景：量子编译流水线中的逻辑优化 pass、等价性检查（phase polynomial 作为规范形）、硬件感知 CNOT 网络合成。
- 与本文关系：本文把"相位多项式仅作局部重写辅助"提升为"一等编译阶段"，并扩展到跨 block（见 Cross-block IR 条目）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization

## 相位奇偶性网络与输出奇偶性网络（Phase-parity Network / Output-parity Network）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 在 sum-over-paths 表示 U|x⟩ = e^{i·p(x)}|g(x)⟩ 中，相位多项式优化天然分为两个子网络：(1) phase-parity 网络——用 CNOT 构造各个相位项所需的输入 XOR parity（phase-parity，如 q₀⊕q₁），并在对应 qubit 线上施加 Rz(θ) 旋转；每个相位项对应一个 parity 列向量（如 (110)ᵀ 表 q₀⊕q₁），全部相位项构成非方阵的 phase-parity 矩阵（列=parity 项，行=qubit）。(2) output-parity 网络——实现输出基变换 g(x) 的 CNOT 网络，g 是 GF(2) 上的 n×n 可逆方阵，每行/每列编码一个输出 parity（如 g(q)=(q₀, q₀⊕q₂, q₀⊕q₁⊕q₂)）。
- 关键区别：phase-parity 矩阵不可用高斯消元规约到单位阵（非方阵，列代表"要实现的旋转条件"而非"输出映射"）；output-parity 矩阵可经高斯消元规约到单位阵（CNOT 网络合成经典问题）。二者共享同一组 CNOT 行操作，因此是耦合的。
- 论文核心洞察：先前工作把两个网络分开优化（phase 网络用贪心/phase-only，output 网络事后高斯消元），但同一 phase 实现可诱导不同 output 代价——本文 Fig.3 给出两个均以 2 CNOT 最小实现同一 phase 函数的电路，其 g 函数代价分别为 2 与 3 CNOT。因此分开处理会错过联合最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 联合优化张量计算（本文 Fig.7）：把两者统一为耦合 parity 矩阵 [phase-parity | output-parity]。例：3-qubit 电路，phase 项 (110)ᵀ、(011)ᵀ，输出 g(q)=(q₀,q₀⊕q₂,q₀⊕q₁⊕q₂) 转置后列向量。初始 M=[[1,0,1,1,1],[1,1,0,0,1],[0,1,0,1,1]]（前 2 列 phase，后 3 列 output）。CNOT(q₁,q₀) 令 row₀ ← row₀ ⊕ row₁ → [[1,0,1,1,1],[0,1,1,1,0],[0,1,0,1,1]]，phase 列 (110)ᵀ 变 (100)ᵀ（Hamming weight 1）→ 该 Rz 可发射、删列；继续行操作清空 phase 列后，剩余 output 矩阵用高斯消元合成。伪代码：while 存在 phase 列：选 active row pair (i,j)（能降低某列 Hamming weight）做 row_i ← row_i ⊕ row_j；若某 phase 列变单位向量 → 发射 Rz(θ) on 对应 qubit 并删列；最终对 output 矩阵做高斯消元。A* 搜索以 f=g+h₁+h₂（已用 CNOT 数 + phase 矩阵总 Hamming weight + output 高斯消元估计）引导选择。
- 作用：把"先 phase 后 output 的两段式贪心"改为"同一搜索空间内联合最小化"，找到整体 CNOT 最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：以 GF(2) 矩阵（numpy/位运算）表示 parity；CNOT 即行 XOR。基线实现：Single-block Greedy（phase 用贪心、output 用高斯消元，本文复现平均总门减 26.93%/两比特门减 8.14%）；Gray-Synth 只优化 phase 部分（平均 CNOT 减 17.62%）。PhasePoly 的 row_heap 合成器用 space-bounded A* + 多解池实现联合优化（平均总门减 34.70%、CNOT 减 26.83%）。开源：PhasePoly（https://github.com/ruadapt/PhasePoly）内置 row_heap / single_block_greedy 等 6 种合成方法可对比。
- 使用场景：任何以 {CNOT,Rz} 为主的线路（算术、MCX、QAOA、Hamiltonian 模拟、FT 基准）的逻辑优化；也用于硬件感知 CNOT 合成（Parity 网络可映射到特定拓扑）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization

## CNOT 网络合成与 GF(2) 线性可逆电路合成（CNOT Network Synthesis / Gaussian Elimination over GF(2)）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 线性可逆电路合成：给定 GF(2) 上可逆方阵 A，求最小 CNOT 序列实现该线性变换（每个 CNOT 对应一次初等行操作 row_j ← row_j ⊕ row_i）。经典算法：高斯消元——把 A 逐步归约为单位阵，每步记录对应 CNOT，逆序即合成网络；复杂度多项式，但门数非最小（CNOT 网络合成对一般矩阵是 NP-hard，Patel-Markov-Hayes 2008 证明任意 n×n 线性变换最坏需 Θ(n²/log n) 个 CNOT，最优合成 NP-hard）。常见实用方法：高斯消元（O(n³)）、Patel 算法（下三角/上三角分解，最坏 O(n²/log n)）、贪心 Gray 码合成（Gray-Synth 用于相位项排序）。
- 在相位多项式优化中它承担"实现 output-parity 网络"与"实现 phase-parity 网络"两个角色：CNOT 既是构造 phase parity 的机制（把 parity 传播/对齐到可发射旋转的 qubit），也是实现输出基变换 g 的机制。
- 论文要点：单个 CNOT 对应 GF(2) 行操作，CNOT 线路 ⟺ 从单位阵出发的矩阵更新序列；合成 CNOT 网络 ⟺ 把矩阵归约回单位阵（Fig.6）。phase-parity 矩阵因非方阵不能直接高斯消元，需与 output 矩阵耦合处理（见耦合矩阵条目）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 高斯消元合成伪代码（output-parity）：输入 g（n×n 可逆矩阵）→ for col=0..n-1：找到 pivot 行（第 col 列为 1 的行 r）→ 对每行 k≠r 且 g[k][col]=1：CNOT(r, k)（row_k ← row_k ⊕ row_r）→ 直到矩阵变单位阵 → 记录的全部 CNOT（逆序）即合成网络。张量例子（Fig.6）：G₁ → CNOT(q₀,q₁) 使 row(q₁) = [1,1,0,0]；合成即逆向归约。
- phase-parity 侧计算过程（本文耦合矩阵）：CNOT(i,j) 的约定是"更新 phase-parity 项而非量子态本身"（沿用 Amy 2018）：row_i ← row_i ⊕ row_j 同时作用于 phase 与 output 两 block；当 phase 列 Hamming weight 降到 1 时，对应 parity 只依赖单 qubit，可立即发射 Rz 并删列（Fig.7 例：CNOT(q₁,q₀) 使 (110)ᵀ→(100)ᵀ）。重复直至 phase 列清空，剩余 output 矩阵高斯消元。
- NP-hard 的应对：PhasePoly 用 space-bounded A* 搜索（priority queue 上限 + 多解池 k，f=g+h₁+h₂）而非精确最优；搜索空间按"active row pair"（能降低 active column set 中某列 Hamming weight 的行对）剪枝避免 livelock。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：经典侧线性代数库或位运算实现 GF(2) 矩阵操作。开源参考：PhasePoly（https://github.com/ruadapt/PhasePoly，row_heap 合成器）、T-par/Gray-Synth（Amy 系列，arXiv:1804.06022）、Qiskit 的 LinearFunction 合成（Patel 算法）、BQSkit（Berkeley Quantum Synthesis Toolkit）等。使用场景：量子线路逻辑优化（output 基变换合成）、Clifford 电路化简、量子编译器中线性映射到拓扑的预优化。
- 与本文关系：PhasePoly 把"output 网络事后高斯消元"升级为"与 phase 网络联合的 A* 搜索"，并证明在耦合视角下整体 CNOT 更优（优于分开处理 9.21 个百分点 vs Gray-Synth）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization

## Clifford+T 门集与 magic-state distillation / cultivation（容错通用门集）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Clifford+T 是容错量子计算（FTQC）的标准通用门集：Clifford 群门（H、S、CNOT，由 {CNOT,H,S} 生成）+ 非 Clifford 的 T 门（π/4 Z 旋转）。Clifford 门可由稳定子码透明地容错执行，T 门不能直接容错，需 magic-state 辅助：预制备 |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 类非 Clifford 态（经 distillation/cultivation 提升保真度），再经 teleportation 注入实现 T。
- 传统 FT 成本模型：T 门昂贵（magic-state distillation 需要大量物理资源与空间时间体积），因此 FT 编译长期聚焦最小化 T 门数/深度（T-count/T-depth 优化，如 T-par 的 matroid partitioning、phase polynomial 的 T 优化）。论文引用的新进展：magic-state cultivation（2024-2025，低开销直接培育高保真 magic state）与更新资源模型表明 T 与 CNOT 的成本日益可比——CNOT 不再是"免费"门，需与 T 联合优化。
- 与相位多项式关系：Clifford+T 门集中，{CNOT,Rz} 区域自然涌现——CNOT 算 parity，T/S/Z（Rz 的子集）累积相位；Fig.1 显示 FT 基准中 CNOT 与 T 数量相当，二者共同主导。因此"联合减 CNOT 与 Rz/T"（PhasePoly 的目标）直接降低 FT 资源成本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FT 编译 pipeline 计算过程：逻辑线路 → 逻辑优化（PhasePoly 减 CNOT/Rz）→ 任意 Rz 分解到 Clifford+T（GridSynth/Ross-Selinger：Rz(θ) → H、S、T 序列，T-count 在 O(log(1/ε)) 内近最优）→ 容错编码（surface code）→ 资源估计（Azure Resource Estimator：按 magic state 制备成本、CNOT 空间时间成本、测量/路由开销估算 wall-clock 与物理 qubit）。论文 Q5 流程：(A) GridSynth→PhasePoly vs (B) PhasePoly→GridSynth 两种顺序；结论 B 更优——先 PhasePoly 化简大 {CNOT,Rz} 区域，再 GridSynth 引入额外 H 门（会把 phase polynomial block 再切碎、限制后续 rotation merging 机会）。
- 资源模型例子：surface-code nearest-neighbour 架构下，PhasePoly 减 CNOT 44.62% FT wall-clock（vs Quartz 11.99%、QUESO 31.80%，Fig.17）；T-count 变化温和（主要影响 HWPA 等结构化电路，深度 ~10% 减）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Rz→Clifford+T 用 GridSynth（Ross & Selinger arXiv:1403.2975；实现：Qiskit gridsynth_rz、Quantinuum/grid_synthesis Rust 版、Haskell gridsynth）；FT 资源估计用 Azure Quantum Resource Estimator（微软云服务，surface-code 假设）；T 优化工具 T-par、PyZX 等。magic-state cultivation 见 2024-2025 文献（如 arXiv:2409.17543 等，论文引 [25-27]）。
- 与本文关系：PhasePoly 在 Clifford+T 合成前运行收益最大，且其联合减 CNOT/Rz 在"CNOT 成本可比 T"的新资源模型下价值上升。
- 补充（O3LS 论文）：O3LS 在 PBC 框架下使用 Clifford+T 门集——任意旋转经 GridSynth（qiskit-gridsynth-plugin，基于 Ross-Selinger arXiv:1403.2975，合成误差容限 10⁻⁵）分解到 Clifford+T；随后转译为 Pauli product rotations（S=Z_{π/4}、T=Z_{π/8}、H=Z_{π/4}X_{π/4}Z_{π/4}、CNOT=(Z⊗X)_{π/4}(I⊗X)_{−π/4}(Z⊗I)_{−π/4}），Clifford 门按 Pauli 映射规则吸收进最终测量。T 门成本（magic state 消费）仍是开销来源，但 O3LS 关注点从"最小化 T-count"转向"布局/调度/合成联合优化时间步与空间"；Y-synthesis 正是针对 PBC 转译后 Y 算子分解的合成优化（对应本条目"相位多项式"外的另一条合成路径）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## FP8 直接低精度训练（DLP，Direct Low Precision）与 LRM 低精度挑战

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FP8（8-bit 浮点，E4M3 前向/E5M2 反向）是 GPU 低精度算术的主力格式：B200 的 FP8 稠密 FLOPs 是 A100 TF32 的 29×，远超 TF32 的 7×。低精度训练按"量化时机与训练关系"分四类：QAT（量化感知训练，训练中模拟量化但权重保持全精度，提精度不提训练速度）、PTQ（训练后量化，简单但精度损失大）、PQT（PTQ+微调恢复精度）、DLP（Direct Low Precision，直接低精度——全训练与推理都用原生低精度，吞吐收益最大但技术挑战最大）。LoKA 聚焦 DLP，需要原生低精度贯穿整个训练与推理。
- LRM（大型推荐模型）与 LLM 在低精度上的根本差异：①质量约束极紧（0.02% relative log loss 即显著退化），几乎没有近似空间；②架构异构（宽 ensemble、深层次堆叠、专用交互模块，各自数值敏感度不同）；③算术强度低——由大量小 GEMM 紧跟归一化层组成，量化/反量化开销可吞掉低精度收益。直接应用 TorchAO 对 Wukong 全线性层做 FP8（64 H100，tensorwise）实测 1.3× 变慢 + 2.5% relative log loss 退化。生产现状：top-500 Ads 模型 95% TF32 训练、99% FP16 推理、FP8 训练 0%（推理仅 1% PTQ）——数值稳定、小 GEMM 量化开销、通信密集是三大阻碍。
- 关键推论：这些挑战不能用"更好的 FP8 kernel"解决，需系统-模型协同设计（分布感知 profiling 找安全位点 + 模型组件与硬件协同改模扩大安全位点 + 跨 kernel 库逐算子编排最大化收益）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DLP 训练一个 Wukong 线性层的 pipeline：输入 x∈R^{M×N}、权重 W∈R^{N×K} → 量化（按缩放策略把 x、W 从 BF16/FP32 转到 FP8，scales 由统计量推导）→ FP8 GEMM（FP32 快速累加）→ 反量化输出 → 归一化（LoKA 用 BlockNorm 融合进 epilogue）→ 激活（Hard Swish）→ 损失 → 反向传播输入梯度同样走 FP8。相比 QAT（前向模拟量化、权重/梯度全精度），DLP 的前向与反向全部原生 FP8。
- 误差度量链：对每层用学习分布采样合成输入/权重 → 跑 FP8 kernel vs TF32 参考 → 按 MERE=Σ_mΣ_n|(out−ref)/ref| 量化每层误差 → 判定该层是否可安全低精度。MERE 几何均值在真实 LRM 分布下比标准正态输入高 15%，证明随机基准漏检真实量化误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：低精度库（NVIDIA Transformer Engine、DeepGEMM、FBGEMM、TorchAO、AMD Quark）提供 FP8 kernel 与缩放 recipe（tensorwise/rowwise/blockwise）；训练框架（PyTorch/TorchRec）经自定义 autograd 包装层接入。LoKA 的落地方式：LoKA Probe 在线学每层分布 → 离线 MERE+吞吐评估 → LoKA Mods 改模型（No Bias/BlockNorm/Hard Swish）→ LoKA Dispatch 逐算子选最快满足精度约束的 kernel。效果：Wukong/Interformer/ELFM 上 FP8 全轨迹 loss 与高精度基线持平，训练最高 1.19×/推理 1.4×，生产 5–20% 训练 / 10–17% 推理加速。
- 别名/关联：与知识库"LLM 数值格式与群量化（bf16 / FP8-E5M2 / ...）"条目互补——该条目讲格式本身，本条目讲 DLP 训练方法论与 LRM 特有挑战。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## MERE（Mean Element-wise Relative Error，逐元素平均相对误差）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MERE 是 LoKA 用于量化"低精度 kernel 相对高精度参考的输出误差"的统计指标：MERE(out, ref)=Σ_mΣ_n |(out_{m,n}−ref_{m,n})/ref_{m,n}|，逐元素相对误差求和（论文公式中未除元素数，可理解为对 M×N 输出逐元素相对误差累加）。它度量低精度执行相对 TF32 全精度结果的平均元素级偏差，是判断"该层能否安全用 FP8"的核心依据。
- 关键性质：MERE 对输入分布极度敏感——标准正态输入的 MERE 会系统性低估真实误差（FBGEMM/TorchAO/DeepGEMM 在 LRM 学习分布下 MERE 几何均值比正态输入高 15%，数值如 BF16 0.03/0.04、TorchAO RW 0.47/0.53、DeepGEMM BW 0.49/0.56、FBGEMM RW 0.48/0.52）。因此 MERE 必须配合"真实分布采样"使用才有意义。
- 附带价值：LoKA Probe 用学习分布测 MERE 时发现 FBGEMM 生产 benchmark 的 faulty test code——随机输入下正确/错误实现的 MERE 几乎相同（0.42 vs 0.42），而用 Probe 输入时相差 47×（17.04 vs 0.37），促使与开发者修复。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算流程：① 对每层从学习分布采样 50–100 对输入-权重（激活 T'=1_Bμᵀ+ZL_Σᵀ，权重 W'=M+L_UZL_Vᵀ）→ ② 分别跑低精度 kernel 与 TF32 参考得到 out 与 ref → ③ 按公式累加逐元素相对误差 → ④ 对多层求几何均值得整体 MERE → ⑤ 与阈值比较（LoKA Dispatch 用 MERE<0.2 作为入选候选 kernel 的精度门槛，配合 speedup>1.05×）。
- 张量计算例子：设 ref 为 (2,2) 张量 [[1.0,2.0],[4.0,8.0]]，FP8 out 为 [[1.02,1.96],[4.10,7.80]]，则 MERE=0.02+0.02+0.025+0.025=0.09。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为量化误差度量在低精度库测试（DeepGEMM/TorchAO/FBGEMM 自带数值测试对比精度）、以及 LoKA Probe 离线 benchmark 中使用。使用场景：跨库跨 recipe 的低精度 kernel 精度筛选、量化方案选择、检测 kernel 实现缺陷。局限：论文明确指出基于误差的 probing（含 MERE）无法推理误差在网络中的传播——各算子误差可能端到端抵消，MERE 会保守禁用本可低精度的层，错过机会。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## LoKA Probe（分布感知低精度 Profiling：在线分布学习 + 离线误差量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LoKA Probe 是 LoKA 三大组件之一，实现"分布感知 profiling"原则：在线学习每层激活与权重的统计分布（不存原始张量，避免存储爆炸与过拟合），再离线从学习分布采样合成输入/权重做统计显著的 MERE 与吞吐评估，定位"哪些层能安全又高效地用 FP8"。核心洞察：标准库用随机（正态）张量做基准会系统性低估真实量化误差——真实 LRM 激活重尾、相关、非平稳，随机基准漏检。
- 统计建模：激活按 batch 维独立，建模为多元高斯 T~G(μ,Σ)，把协方差存储从 O(M²N²) 降到 O(N²)（推荐模型避免跨 batch 算子如 BatchNorm 防信息泄漏，使 batch 独立成立）；用批量 Welford tracker 流式更新均值/散度（合并公式见下）。权重无维独立假设，建模为矩阵正态 W~MN(M,U,V)（vec(W)~N(vec(M),V⊗U)），Kronecker 积分解耦行列协方差，存储 O(M²+N²)，用 flip-flop 式 EMA + Cholesky 线性求解在线更新，trace 重归一化防尺度漂移。为控制开销：每 100 训练迭代激活统计、每 10000 迭代异步保存参数，总开销 ≤1%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在线合并（Welford，激活）：当前批 X∈R^{B×K}，批均值 μ_b、批散度 S_b=(X−1_Bμ_bᵀ)ᵀ(X−1_Bμ_bᵀ)；合并历史 (n_old,μ_old,Σ_old)：n_new=n_old+B、δ=μ_b−μ_old、μ_new=μ_old+(B/n_new)δ、Σ_new=Σ_old+S_b+(n_old·B/n_new)δδᵀ；样本协方差 Σ=Σ_new/(n_new−1)（FP32 累积）。
- 在线更新（权重，矩阵正态）：每 minibatch 解 L_VL_Vᵀ=V+εI 得 U'=(1/N)(W_c L_V⁻ᵀ)(W_c L_V⁻ᵀ)ᵀ；解 L_UL_Uᵀ=U+εI 得 V'=(1/M)(L_U⁻¹W_c)ᵀ(L_U⁻¹W_c)；EMA 平滑 U''=mU+(1−m)U' 后对称化+εI 正则；尺度重归一化 s=trace(U)/M、U←U/s、V←sV（m∈[0.9,0.99]，ε≈10⁻⁶×trace(U)/M）。
- 离线采样与评估：激活 T'=1_Bμᵀ+ZL_Σᵀ（Z~N(0,I_K)，L_Σ 为 Σ+εI 的 Cholesky）；权重 W'=M+L_UZL_Vᵀ。对每层采样 100 对，跑 FP8 vs TF32 算 MERE + 计时，MERE 高或加速比低的层标记为低精度不安全。
- Probe 关键发现（Wukong 分析）：bias 项发散（部分 bias 范数不收敛、≥0.1，级联导致越界/量化湮没小值）；归一化开销与 mean-cancellation 误差（LayerNorm 需高精度反量化/重量化往返）；sigmoid 型 Swish 指数运算放大离群值、量化损失剧增。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为训练钩子（hook）接入 PyTorch 线性层，流式维护每层统计量；离线 benchmark 模块采样合成张量驱动 TorchAO/DeepGEMM/FBGEMM kernel 评测。使用方式：训练早期启用收集分布 → 离线跑 MERE+吞吐矩阵 → 输出每层 (库, recipe) 的精度/性能表 → 供 LoKA Dispatch 过滤与选择。作用：把"哪里低精度安全"从拍脑袋变为统计可判，并发现标准基准漏检的误差（含 FBGEMM faulty test code，MERE 差 47×）。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## BlockNorm（块级 RMS 归一化，Grouped RMSNorm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BlockNorm 是 LoKA Mods 的核心组件：把 GEMM 后的归一化改为沿特征维按固定块（如 256 元素）独立做 RMS 归一化，公式 RMSNorm((Wx+b).view(-1,BlockN)).view(B,N)，数学上等价于无参数 Grouped RMSNorm。设计动机：把归一化直接融合进 GEMM epilogue（输出 tile 还在片上 L1/L2/寄存器时完成），避免 HBM 往返；但标准归一化沿特征维求全局统计，与 GEMM 输出的物理数据布局错位，无法片上完成。
- 为什么用 RMS 而非 LayerNorm：RMSNorm 只按激活 L2 范数归一化、不做均值相减，避免低精度下相近数值相减的灾难性相消（mean cancellation）误差；BlockNorm 把全局单统计量拆成块内独立统计量，一个离群值不再压制全部特征，解耦特征子空间、增加表示自由度（类比 GroupNorm 分通道归一化）。
- 两种形态：Case 1 大 batch 小输出维——整行特征可放单个 thread block，统计全本地计算，行为与标准归一化一致，激活/量化/反量化可一并融合；Case 2 小 batch 大输出维——单 block 装不下整行，RMS 统计需跨 block 同步，抵消融合收益；缩小 batch tile 又引入 SM wave quantization 与 W 矩阵 L2 命中率下降。最终取舍：放松数学等价性，用固定块（train/test 一致）规避全局同步，鲁棒适配任意 shape。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算流程（一次 GEMM+归一化+激活融合）：FP8 GEMM 输出 y∈R^{B×N}（tile 在片上）→ y.view(B, N/BlockN, BlockN) → 每块独立算 RMS：rms_b = sqrt(mean(block²)+ε)，out_b = y_b / rms_b（可选乘缩放参数，无参数版即纯归一化）→ 紧跟 Hard Swish out·ReLU6(out+3)/6 与反量化 → 写回 HBM。块内全部在寄存器/SMEM 完成，无全局同步。
- 与标准 RMSNorm 对比：全局 RMSNorm 用单一统计量（一个离群值压低所有特征）；BlockNorm 每 256 元素独立归一化。论文实验（Wukong 生产模型，BlockNorm 256 vs RMSNorm）收敛到相同 loss，说明块内归一化保持模型质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：融合进 GEMM kernel 的 epilogue（类似 epilogue fusion，但论文强调应用在低精度上下文）；PyTorch 侧以 reshape+分组 RMS 实现（.view(-1,BlockN) 后逐块归一化再 view 回 (B,N)）。使用要点：块大小 train/test 严格一致以保证一致性；块足够大（如 256）时收敛对块大小不敏感；与 Hard Swish、量化/反量化同 kernel 融合最大化效率。关联：设计上对齐 MX（Microscaling）硬件标准的块共享缩放思路，避免全局同步开销；参考 GroupNorm、pRMSNorm（用 6.25% 神经元估 RMS）等先例。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## Hard Swish（硬 Swish / h-swish 激活函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hard Swish 是 Swish 的分段线性近似：h-swish(x)=x·ReLU6(x+3)/6，其中 ReLU6(x)=min(max(x,0),6)。Swish（SiLU）x·σ(x) 依赖指数运算，在低精度下指数放大离群元素、压缩小元素，量化损失剧增且计算开销大。Hard Swish 消除指数运算，分段线性天然适合低精度（范围有界、可精确表示），表示能力对推荐模型典型输入范围与 Swish 相当。
- LoKA 用它替换 LRM 中重度使用的 sigmoid 型激活（Swish x·σ(x)、SwishNorm x·σ(Norm(x))）——LoKA Probe 识别出 sigmoid 不稳定是三类低精度隐患之一。Hard Swish 与 BlockNorm 天然可融合进同一 kernel（归一化+激活+量化全在 epilogue）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程：输入 x → ReLU6(x+3)=clamp(x+3,0,6) → 乘 x → 除 6。分段表达：x≤−3 时输出 0；−3<x<3 时 x(x+3)/6（二次）；x≥3 时输出 x。全程无指数、无分支发散（可用 clamp+乘加实现），FP8 低精度下数值稳定。
- 示例（张量计算）：x=[-4, -1, 2, 5] → ReLU6(x+3)=[0, 2, 5, 6] → h-swish=[0, -0.333, 1.667, 5]。对比 Swish：σ(-4)≈0.018、σ(-1)≈0.269、σ(2)≈0.881、σ(5)≈0.993 → swish=[-0.072,-0.269,1.762,4.966]，在典型输入范围内近似。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为模型激活替换（把 nn.SiLU 换为 hard-swish 自定义 autograd 或融合进 GEMM epilogue kernel）；训练/推理同定义（无需特殊处理）。使用场景：低精度（FP8/INT）训练与推理中替代指数型激活，与块归一化、量化融合减少 kernel 数与 HBM 流量。注意：Hard Swish 首次广泛用于 MobileNetV3（效率优先），LoKA 将其用于低精度 LRM 稳定性；tradeoff 是略简化激活动力学换取低精度稳定性。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## No Bias（无偏置模型设计）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- No Bias 是 LoKA Mods 的组件：从 Wukong 模块中移除所有 bias 项（仅最终预测层保留，因不同预测任务 bias 有益）。动机来自 LoKA Probe 发现"bias 项发散"——训练中显著比例 bias 的 L2 范数不收敛、部分达 ≥0.1，级联到后续模块造成越界，在 clamp+量化时令小值完全湮没，是 FP8 训练不稳定的来源之一。
- 借鉴 LLM 趋势：DeepSeek 从所有 FFN 与归一化层移除 bias；PaLM/Falcon 在 FFN 层移除、归一化层保留。LoKA 把该实践引入 LRM 并给出额外系统收益：FSDP per-parameter padding 下，小于 world size 的 bias 张量会引入显著 padding 通信开销，去 bias 同时降低通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（有无 bias 对比）：原始线性层 out=xW+b；No Bias 版 out=xW。对 (2048,256)@(256,768) 的 GEMM，bias 为 (768,) 向量，在 FSDP 分片/填充下 b 的通信与填充开销不再存在；训练中 b 的梯度更新与发散路径也整体消失。LoKA 消融显示 No Bias 是单独贡献最大的延迟降低来源（消除参数开销、简化计算路径），且减少早期训练不稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：模型定义中移除 linear/norm 层的 bias 参数（nn.Linear(bias=False) 等），仅预测头保留；低精度 kernel 侧 epilogue 不再有 bias 加项。使用场景：低精度（FP8）LRM 训练/推理的稳定性改进；与 BlockNorm、Hard Swish 合并（三者单独都不足以稳定，合并后 FP8 全轨迹 loss 与高精度基线持平）。注意：不是所有任务都适用——论文明确预测层保留 bias。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## Chebyshev 多项式（Chebyshev Polynomials）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Chebyshev 多项式（第一类，T_n(x)）是 [-1,1] 上一族正交多项式，由递推 T_0(x)=1、T_1(x)=x、T_n(x)=2x·T_(n−1)(x)−T_(n−2)(x) 定义，等价于 T_n(x)=cos(n·arccos(x))。相比 Taylor 级数：在整个区间上极小化最大误差（minimax 性质）、收敛更快、达到同样 sup-norm 误差所需次数更低、避免 Runge 振荡、数值条件更好（B.-Chebyshev-Polynomials 论文依据）。正交性使最小二乘系数求解稳定。用于函数逼近时，f(x)≈Σc_i·T_i(x)，系数 c_i 由采样点最小二乘解得。
- 在 LoRA（ISCA'26）中的作用：作为分段逼近的基函数——对用户给定输入范围 [a,b] 的非线性函数，先做区间变换 x'=(2x−(b+a))/(b−a) 映射到 [-1,1]（Chebyshev 自然定义域，正交性在 [-1,1] 成立），构造 Chebyshev 矩阵 V（第 j 列=T_j(x_i')），用最小二乘 min Σ(f(x_i')−Σc_j·T_j(x_i'))² 解系数 c，再用递推把 Σc_i·T_i(x') 展开回标准多项式 Σp_i·x'^i、逆区间变换得到 Σp_i·x^i，供 XCore 硬件以 LNS 高效执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（单段 6 项逼近，n=5）：
  1. 采样 m 点 {x_i, f(x_i)}（LoRA 用 curvature-based：均匀采样后数值微分估曲率、高曲率区插点）；
  2. 区间变换 x_i'=(2x_i−(b+a))/(b−a)∈[−1,1]；
  3. 构造 V=[T_0(x_i')...T_5(x_i')]（m×6 矩阵），解最小二乘 V·[c_0..c_5]^T≈[f(x_i')]^T；
  4. 用递推把 Σc_i·T_i(x') 展开为 Σp_i'·x'^i，逆变换得 Σp_i·x^i；
  5. 利用奇偶性：奇函数 V 偶数列置 0（如 [0,T1,0,T3,0,T5]），同 6 项可支持 9 次多项式（x^{1,3,5,7,9}），更高次数→更高精度与数值稳定性（cos 考虑奇偶性使 MAE 降 148×）。
- 示例（cos(x) 逼近，[−π/2,π/2]）：变换到 [-1,1] 采样 → 因 cos 是偶函数只用偶阶项（T_0,T_2,T_4...）→ 最小二乘得系数 → 硬件 XCore 在 LNS 中算 Σp_i·x^i（每项 c_i·x^(k_i)=2^(log2(c_i)+k_i·log2(x))）。结果：XCore-A/C 在 [−π/2,π/2] 上 AAE≈1.03e−6、MSE≈2.18e−12，与软件双精度逼近同数量级。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 的 PiecewiseChebFitter（Python，开源于 COFFA 仓库 LoRA-ISCA-AE 分支）实现完整流程，`python3 run_func.py <gelu|sigmoid|softplus|swish|tanh>` 对每函数（config 文件夹配参数）输出 breakpoints/系数/次数到 fig/ 与 result/；遗传算法负责各段次数分配、三种分段策略确定 breakpoints（见分段逼近条目）。
- 使用场景：误差容忍的硬件非线性函数实现——AI 激活函数（Sigmoid/Tanh/GELU/Swish/Softplus/Swiglu）、DSP（sin、sqrt、arcsinh）、LLM（Softmax、GELU）；与 PACE（同为 Chebyshev 分段逼近，3-term）对比，LoRA 利用奇偶性+equal-error 分段精度更高（EfficientNet/MobileNetV3 误差 0.002%/0.006% < PACE 最小 0.01%）。其它用法：图神经网络（Chebyshev 卷积近似，NeurIPS 2022）、DS-TPU 用 Chebyshev 多项式建模非线性节点交互（另见 vault 中 DS-ISA、Oracle-MoE 论文对 Chebyshev 不等式/逼近的引用）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

## 分段逼近（Piecewise Approximation，分段多项式逼近）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 分段逼近是把目标函数定义域 [a,b] 分成若干子区间，每段用低阶多项式（LoRA 中 ≤5 次）近似；相比单一高次多项式，能覆盖大输入范围并降低单段多项式次数。它可视为 LUT 与多项式方法的结合：只需存储各段 breakpoints 与多项式系数（比 LUT 省内存），且通过调整系数即可重构用于不同函数。LoRA 中每段用 Chebyshev 最小二乘求系数，段数与每段次数由算法联合优化（见 Chebyshev 多项式条目）。
- 在 LoRA 中的作用：让通用 CGRA 以"低阶多项式+少段数（6-7 段）"支持任意用户定义输入范围的函数；三种分段策略（uniform/curvature-based/equal-error）决定 breakpoints，遗传算法决定各段多项式次数，二者联合最小化平均 MSE。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 三种分段策略（Fig.3/Fig.4）：
  1. Uniform：等宽分段，不区分函数行为；
  2. Curvature-based：均匀采样后估每点曲率 κ(x_i)，累计曲率 W_k=Σ_{i≤k}κ(x_i)Δx（W_m 为 [a,b] 总曲率），找 breakpoints 使每段累计曲率=W_m/N——高曲率（变化快/曲率大）区段更密，误差更小；
  3. Equal-error：从 curvature 分段出发迭代优化，在 [x_(s−1), x_(s+1)] 内用 Brent 法解 MAE(x_(s−1),x)−MAE(x,x_(s+1))<ξ（默认 ξ=1.5e−5）使左右段 MAE 接近，直到各段 MAE 方差低于阈值——误差分布更均匀、接近最优精度 [68]。
- 伪代码（LoRA 分段逼近）：
  ```
  # 输入: f(x), [a,b], 最大段数 N, 每段最大项数
  for seg = 2..N:
    breakpoints = uniform | curvature(W_m/N 均分) | equal_error(Brent 迭代)
    次数分配 = 遗传算法(k_seg1..k_segN, #gen=10, #pop=16)   # 防高次过拟合
    对每段: 最小二乘解 Chebyshev 系数(含定点溢出约束 |p_i||x^i|_max<Q_max)
    记录平均 MSE
  选平均 MSE 最小的 (段数, breakpoints, 系数, 次数)
  ```
- 示例（sigmoid，[−8,8]）：7 段（XCore-A/B）或 6 段（XCore-C），每段 6 项多项式；XCore-A 在 [−8,8] 上 sigmoid AAE=3.73e−6、MSE=2.36e−11，优于先前工作 [76]（AAE 1.70e−3）、[32]（AAE 3.40e−4）、[4]（sq-AAE 6.5e−9）。6 段 vs 7 段：段数越多精度越高但硬件（LUT 大小）开销越大（Fig.8 ADPP 权衡）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 中软件（PiecewiseChebFitter，Python）离线生成配置，硬件 XCore 的 Pre-Process 级把输入 x 与 breakpoints 比较查 LUT 取该段参数（log2(c_i)/k_i/bias），运行时无分支开销；用户只需提供函数与输入范围。最大段数受 XCore LUT 容量限制（论文评估 6-7 段），超过 6 项多项式可由多个 XCore 计算。
- 使用场景：误差容忍的 AI/DSP 非线性函数硬件实现（与 Chebyshev 条目同场景）；相关既有工作多为固定数据格式（定点/浮点）与一次/二次多项式（Flex-SFU 分段二次 [4]、ReAFM [76]、等误差分区 [79]），LoRA 支持多格式+奇偶性+三策略，算法级用 ξ=1.5e−5、#gen=10、#pop=16（论文经验值）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

## Taylor 展开（Taylor Expansion / Taylor 级数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Taylor 展开把函数在展开点附近表示成多项式：f(x)=Σ (f^(n)(a)/n!)(x−a)^n + 余项。用于硬件非线性函数逼近时（如指数 e^x、log、sin、cos），用有限阶多项式近似，精度只在展开点附近高、远离时急剧下降，且高阶需要多次乘加（MAD）运算（6 阶 exp 至少 6 个 MAD [4]）。在 LoRA 论文中是基线方案（PICACHU [56] 与 NX-CGRA 等）。
- LoRA 中 Taylor 的作用与对比：作为 baseline 方法——PICACHU 用 Taylor 展开逼近 exp/log/sin/cos，配合 FP2FX（浮点转定点）模块与算子融合（把 Taylor 的 MAD 融合进单个 PE）；DCT 端到端精度对比显示需 ≥4 阶 Taylor 才能匹敌 LoRA（Table VIII：PICACHU-3rd 的 MSE/PSNR 明显更差，4th/5th 才持平），故后续评估取 4 阶起。Taylor 的痛点（LoRA 动机）：高阶→更多 MAD→更多 PE 占用、限制 loop unrolling、只支持有限函数子集、复合函数更难处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（PICACHU 的 exp 逼近）：输入 x → FP2FX 把浮点 x 转定点 → 泰勒多项式 Σ x^n/n!（如 4 阶：1+x+x²/2+x³/6+x⁴/24）→ 每个 MAD 一个 PE 节点 → 结果。6 阶 exp 需 ≥6 个 MAD；精度提升→阶数提高→MAD 数线性增加→PE 占用增大。
- 与 Chebyshev 对比：Taylor 在展开点附近精度高、远离差（输入范围受限）；Chebyshev 在整个区间上极小化最大误差、同 sup-norm 误差下次数更低、避免 Runge 振荡、数值条件更好（B.-Chebyshev-Polynomials）。例：PICACHU-4th 在 DCT 上与 LoRA 持平，而 LoRA 用 6 项 Chebyshev 多项式即可。
- 其它用途（vault 中广泛出现）：量化误差建模（用二阶 Taylor 展开损失函数扰动：L(W_Q)≈L(W)−g^T(W−W_Q)+½(W−W_Q)^T H(W−W_Q)，SqueezeLLM/APHQ-ViT/GuidedQuant 等）、重要性/敏感度估计（一阶 Taylor：Saliency=gradient·weight）、softmax 线性化（ViTALiTy 一阶 Taylor）、指数硬件实现（Taylor series [44]）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件上把 Taylor 系数（阶乘倒数）固化为常数，用 Horner 规则（嵌套乘加，Horner's rule [33]）减少乘加数；PICACHU 用专用 MAD 融合模块在一个 PE 内算完；LoRA 的 XCore 则用 LNS 把每项 c_ix^(k_i) 变成 2^(log2 c_i + k_i·log2 x)，6 项多项式只需一个可编程 30b 乘法器（5×30b×6b），硬件开销远低于 6 个 MAD。
- 使用场景：作为非线性函数硬件的经典基线（CORDIC/LUT/Taylor 三类的多项式类），LoRA 论文在单元级与端到端（DCT/DNN/LLM）都把它作为对比；在模型量化/剪枝领域作为误差分析的数学工具广泛应用。局限：展开点局部性、阶数-硬件开销线性关系、输入范围受限（Taylor series accuracy is high only near the expansion point）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

## CORDIC（Coordinate Rotation Digital Computer，坐标旋转数字计算机）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CORDIC 是 1959 年 Volder 提出的迭代算法：基于所选坐标系（圆/线性/双曲）与旋转/向量模式，用简单的移位+加法迭代逼近三角函数、双曲函数、对数、指数、开方等。优点：硬件开销低（只需 add/shift）。缺点：精度靠迭代次数，迭代越多延迟越大；支持输入范围有限，某些版本需旋转前后额外处理（LoRA 论文 II-B 背景）。
- 在 LoRA 中的作用：作为非线性函数硬件实现的基线之一——huicore [10] 是 CORDIC 通用复杂函数加速器（28nm、153k µm²、≥20 cycle/次迭代数，支持 2GHz）；XCore 与其对比：更低延迟（4/7 cycle）与硬件开销（40nm、71.7–78.4k µm²），支持可编程定点格式，且能单步逼近复合函数（CORDIC 级联方式做不到）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（旋转模式算 sin/cos）：初始化角度累加器与 (x,y)，每轮 i 做 x'=x−σ_i·2^(−i)·y、y'=y+σ_i·2^(−i)·x、z'=z−σ_i·arctan(2^(−i))，σ_i 按 z 的符号选（±1）；迭代 n 轮后 (x,y) 收敛到 (cos(z0), sin(z0))·K（K 为增益常数）。精度↑⇔迭代数↑⇔延迟↑。
- LoRA 对比点：huicore（CORDIC 级联）逼近复合函数需串联多次旋转/向量操作，而 XCore 把复合函数（tanh(x)+1、sin(x)+cos(x)、ln(sin(x))）作为一个多项式直接逼近，一个 XCore 节点完成；且 CORDIC 只支持有限输入范围，XCore 支持用户定义任意范围。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件为移位器+加法器迭代阵列（可用展开流水线提高吞吐）；软件库/FPGA IP 核常用（Xilinx CORDIC IP）；嵌入式 DSP（STM32 CMSIS-DSP、FFT 多旋转 CORDIC）广泛使用。
- 使用场景：三角函数/双曲/对数/开方的低成本硬件实现；LoRA 论文把它归类为"迭代式（iterative-based）"三类非线性硬件方案之一（另两类：LUT 式、多项式式），与 Flex-SFU、PACE 等多项式式方案对比时作为精度参考（XCore 逼近精度与 CORDIC 目标相当甚至更好：arcsinh [−19.4,19.4] XCore-A AAE=1.67e−5 vs CORDIC [10] 8.91e−6，sin [±π/2] XCore 与 [10] 同量级）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

## 遗传算法（Genetic Algorithm，用于分段多项式次数分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 遗传算法（GA）是受自然选择启发的元启发式优化：种群（一组候选解/个体）经选择（按适应度）、交叉、变异迭代演化，收敛到近似最优。适合大规模/组合/不可微的搜索空间；随机性使其不一定最优，但迭代充分时逼近最优。在 LoRA 中用于分段逼近的次数分配：个体=各段多项式次数分配 (k_seg1,...,k_#seg)，breakpoints 由三种分段策略确定，适应度=整函数平均 MSE，选平均 MSE 最小的个体。
- 为什么需要：高次多项式在每段都可能导致过拟合，且各段最优次数不同；穷举 6 段×6 项=6^6=46656 种次数分配一周内不可行（论文 VIII-A），GA 在迭代充足时更合适。经验设置：#gen=10、#pop=16、equal-error 容差 ξ=1.5e−5。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（LoRA 算法级）：
  ```
  初始化种群 P = 随机次数分配 {k_seg1..k_segN}×#pop
  for gen in 1..#gen:
    对每个个体: breakpoints = 分段策略(seg 数); 逐段最小二乘求系数; 计算平均 MSE
    适应度 = 1/平均MSE; 按适应度选择父代; 交叉/变异生成新种群
  返回平均 MSE 最小的个体
  ```
- 算法级评估结论（Fig.7）：ξ 越小误差分布越均匀、越接近近优但 runtime 越大；#gen 越多越好但收益递减；GA 优于穷举（时间可行性）。
- 其它用法（vault 中广泛出现）：ScaleMoE 用 GA 做专家重映射（coverage×bandwidth 矩阵下最小化通信）、Cocco 用 GA 做硬件映射-内存配置协同探索（比贪心/DP 更稳）、AutoFHE 用多目标 GA 做 FHE 加速器设计空间探索（NSGA-II）、TileFlow 用 GA 做 tiling 调度、GAMMA/Magma 用 GA 自动映射 DNN 到加速器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 的 PiecewiseChebFitter（Python）内置 GA；标准实现需定义编码（个体表示）、适应度函数、选择/交叉/变异算子与终止条件；可用库如 DEAP、PyGAD。超参（#gen/#pop/交叉率/变异率）需按问题调。
- 使用场景：与分段策略/最小二乘联合构成"算法-硬件协同"的离线配置生成（LoRA）；以及分布式训练专家放置、硬件映射、调度等组合优化问题。局限：随机性导致结果不稳定（Cocco 论文指出 GA 比 DP 不稳定，需多次运行取优）；论文明确穷举不可行时才用启发式。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

## UniAD（端到端自动驾驶模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- UniAD（Planning-Oriented Autonomous Driving，CVPR 2023 最佳论文，Hu et al.）是端到端自动驾驶算法：统一框架集成自动驾驶两大核心任务——感知（object detection/tracking：BEVFormer、TrackFormer、MapFormer）与预测（motion forecast/occupancy prediction：MotionFormer、OccFormer），各模块均为 transformer 架构，经大量 query token 连接（如 TrackFormer 900 个 query），提供丰富并行机会。M100 基准版本用 RegNet 替换 ResNet-101 主干，以更贴近理想汽车实际部署的 AD 算法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（M100 评估的 MAC 分布，Table II）：CNN 主干 RegNet+FPN（30M 参数、2381.6 GFLOPs，占大部分算力，来自高分辨率图像密集卷积）→ BEVFormer（85.6M、1492.9）→ TempFusion（0.2M、49.0）→ TrackFormer（8.5M、97.17）→ MapFormer（6M、105.94）→ MotionFormer（22.6M、266.55）→ OccFormer（46.2M、687.62）→ Planner（3.5M、220.75）。感知模块（BEVFormer/TrackFormer/MapFormer）通常以高于预测模块的帧率运行，计算需求更大，因此分析聚焦 CNN 主干与感知 transformer。感知推理一帧 ≈ 并行执行上述模块链，query token 之间的并行是数据流架构的主要收益来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 开源实现（OpenDriveLab/UniAD，CVPR 2023），图像+LiDAR 输入。使用：作为 AD 推理 benchmark 对比 M100（8/14 cluster）与 NVIDIA Thor-U——RegNet 13.1ms vs 57.4ms（4.4×）、FPN 4.23 vs 5.1（1.2×）、BEVFormer 7.92 vs 32.83（4.1×）、TempFusion 4.47 vs 17（3.8×）、TrackFormer 1.27 vs 7.95（6.3×）、MapFormer 1.46 vs 6.14（4.2×）；perception 30 FPS vs 7.9 FPS（3.8×，同功率预算，满足高速自动驾驶实时要求而 Thor-U 未达）。剩余 6 cluster 保留给座舱功能，验证多域隔离。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing

## Oblivious RAM（ORAM，不经意随机访问内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ORAM 是隐藏内存访问模式的安全原语（Goldreich & Ostrovsky, J.ACM 1996）：通过每次逻辑访问都伪装成对一组"看似随机"位置的访问、并把被访问块重映射到新位置，使服务器侧观察者无法区分任意两条等长访问序列（computationally indistinguishable）。本论文采用经典两方模型：*Trusted Client*（持有全部隐私关键逻辑：位置图、stash、解密/重加密、随机重映射）与 *Untrusted Server*（只按请求在结构化存储（如二叉树）中存取整条根到叶路径的密文块）。每次访问带宽被放大（PathORAM 为 log(N) 块/次），因此传统 ORAM 部署在 WAN 下带宽负担大，且客户端必须驻留本地/可信第三方。
- TEE+ORAM 变体：把 ORAM 客户端放进 VM-based TEE（Intel TDX/AMD SEV-SNP），与服务器同机部署，WAN 流量削减到只传目标块；此时 ORAM 树/暂存直接放 TEE 的 TME 加密 DRAM（确定性 AES-XTS），省去客户端额外重加密——但确定性加密产生密文侧信道（见密文侧信道条目），破坏 ORAM 不可区分性。MC-ORAM 在 TEE 内用掩码+计数器恢复密文非确定性且不修改底层 ORAM 访问序列。
- 威胁模型（论文 II）：CPU 包/缓存/TEE 特性可信；敌手可观察 DRAM 访问模式与密文内容（stash、位置图、ORAM 树）；可做内存总线嗅探。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一次逻辑访问（PathORAM 语义）：
P = PosM[d]                       # 1) 位置图查目标块叶子
read_path(P) -> stash             # 2) 服务器返回整条根到叶路径密文，解密入 stash
process(d)                        # 3) 客户端对目标块计算
PosM[d] = Rand()                  # 4) 分配新随机叶子并更新位置图
evict(stash, P)                   # 5) 贪婪回填驱逐写回服务器
# 关键：5 步的流量形状与 d 无关（恒为整路径+全暂存扫描），故不可区分
```
- 例子：N=2^14、L=14、Z=4、B=256B 的 PathORAM，每次逻辑访问移动 14 节点×4 块路径并全扫描暂存（90 槽）；MC-ORAM 在其中叠加 112 位掩码写+16 位计数器递增，访问延迟约 0.87ms（vs 64 位计数器 baseline 1.48ms）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：树形构造（PathORAM/RingORAM）或分层/电路构造；客户端侧位置图+stash+驱逐逻辑；服务器侧只存密文桶。开源参考：PathORAMSimulator（https://github.com/renling/PathORAMSimulator）、oram_simulator（https://github.com/wangxiao1254/oram_simulator，含 PathORAM/Circuit ORAM 等）。本论文基于这两个实现开发 TDX 内版本（每实现 <1000 行，<200 行 MC-ORAM 特有）。
- 使用场景：AES 密钥恢复防护、推荐模型嵌入表（LAORAM）、LLM 嵌入表、数据库安全存储（Menhir）等所有"访问模式敏感"负载；本论文用于 TEE 内安全嵌入表管理器（DLRM/Qwen-8B 评估）。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## PathORAM（路径 ORAM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PathORAM 是最广泛使用的树形 ORAM 构造（Stefanov et al., CCS'13/J.ACM）：把 N 个数据块（块大小 B）组织成服务器侧高度 L=log(N) 的二叉树，每节点是一个桶，最多存 Z 个 ORAM 块；为防止桶占用率泄露，每桶用 dummy 块补齐到恒为 Z 块。客户端维护位置图（块→叶子映射）与 stash。每次访问：查位置图→读整条根到叶路径入 stash→处理目标块→分配新随机叶子更新位置图→按驱逐规则贪婪回填写回。带宽放大 log(N) 倍。协议简单、客户端状态小，是 ORAM 事实标准。
- 本论文以 PathORAM 为 baseline 与承载协议之一：baseline 采用 64 位交错计数器（每 64 位数据配 64 位计数器，Obelix 风格）实现非确定性，MC-ORAM 在其上加 112+16 位掩码计数器布局。泄露分析（V-B）：确定性加密下第二次重复读可能观察不到树密文变化，攻击者可区分 ⟨Read 0, Read 0⟩ 与 ⟨Read 0, Read 3⟩，区分优势 1/4。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PathORAM 一次访问（含 MC-ORAM 掩码/计数器）：
P = PosM[d]
for node in 路径P:                 # 读路径：每节点 Z 块
    for i in 1..Z:
        wrMask = 计算条件写掩码(暂存空槽)
        TreeToStash(stash, node[i], wrMask)   # 掩码写 + 全暂存计数器+1
process(d); PosM[d] = Rand()
evict(P): for node in P:
    for i in 1..Z:
        StashToTree(stash, node[i], wrMask, found)  # 反向 + 树/暂存计数器+1
# 计数器溢出(2^16-1) → Refresh(node/stash)：新掩码+清零
```
- 例子：N=2^14、Z=4、stash=90，PathORAM 访问延迟 1.48–39.92ms（B=cacheline~2048B）；MC-ORAM 版 0.87–22.56ms，最高 1.82× 加速；暂存优化版（PathORAM+，Oblix 式：只把目标块入 stash、每 3 次访问额外驱逐、stash=10）0.19–8.66ms vs MC-ORAM+ 0.11–5.00ms。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：服务器侧二叉树（每桶 Z 块含 dummy）+ 客户端位置图/stash/驱逐；开源参考 PathORAMSimulator（https://github.com/renling/PathORAMSimulator）。本论文在 Intel TDX VM 内实现（Ubuntu 22.04.5、双路 Xeon 6548Y+、512GB DDR5），并用 Intel PIN 采集 SPEC CPU2017 轨迹确定 ORAM 高度。
- 使用：作为 TEE 内安全嵌入表/安全存储的原语；参数 Z=4、stash naive 90/优化 10；评估 N=2^14/2^23、B=512b(cacheline)/256B/2048B(embedding)。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## RingORAM（环形 ORAM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RingORAM 是与 PathORAM 同树的 ORAM 构造（Ren et al., USENIX Security'15）：每桶除 Z 个真实块外加 S 个永久 dummy 块（桶共 Z+S 块，空间更大），从而每次逻辑访问**每桶只读 1 块**（目标块或随机 dummy），而非 PathORAM 的整桶 Z 块，显著降低带宽。驱逐不是每次访问都执行，而是每 A 次访问一次（论文 A=4），驱逐路径按 reverse-lexicographic（逆字典序）固定调度顺序选择、与访问模式无关；驱逐时桶内 (Z+S) 块按 rotation schedule 洗牌。因每桶只读一块，块以 1/(Z+S) 概率保留原内容，确定性加密下仍可泄露（论文 V-B）。
- 本论文把 MC-ORAM 集成到 RingORAM：掩码/计数器/刷新机制不变，差异为单块访问上应用掩码+计数器、节点含 (Z+S) 块、每 A 次访问驱逐、驱逐时洗牌在掩码域内完成（无需额外密码机制）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# RingORAM 单次访问（MC-ORAM 集成版）：
P = PosM[d]
for node in 路径P:
    i = Rand() % (Z+S)            # 每节点只读 1 块（目标或 dummy）
    读 node[i]，TreeToStash 合并入 stash（全暂存扫描+计数器+1）
process(d); PosM[d] = Rand()
每 A 次访问触发 eviction（reverse-lexicographic 路径）：
    整条驱逐路径逐节点读入 stash（每节点全 (Z+S) 块）→ 掩码写回 + 桶内洗牌
# 任一节点计数器溢出 → Refresh 整节点
```
- 例子：S=3、A=4、Z=4、stash=90，RingORAM 访问延迟 0.78–33.08ms；MC-ORAM 版 0.42–19.05ms（最高 1.85×）；RingORAM+（stash=10）0.16–5.83ms vs MC-ORAM+ 0.10–3.44ms（最高 1.60×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：树形桶 (Z+S) 块 + 周期驱逐 + reverse-lexicographic 路径调度 + 桶内洗牌；开源参考 oram_simulator（https://github.com/wangxiao1254/oram_simulator）。论文 RingORAM 计数器在节点内可能不同（单块访问），任一溢出即刷整节点。
- 使用：带宽敏感场景（每桶单块读取降低路径流量）；本论文在 TDX 内评估 N=2^14/2^23 与多块大小，并用于 DLRM/Qwen-8B 安全嵌入（RingORAM+ 作为 ML 端到端 baseline：Qwen-8B 36.2ms→MC-ORAM+ 25.8ms 1.41×、DLRM 6→3.61ms 1.66×）。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## Stash（暂存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Stash 是 ORAM 客户端侧的小型可信缓冲区：服务器读回的整条路径先解密落入 stash，客户端在此完成目标块计算；驱逐时无法立即回填的块（桶容量已满）暂留 stash 等待后续驱逐。stash 占用率高度依赖访问模式（重复访问占用小、非重复占用大），因此是访问模式泄露的敏感结构。
- TEE 内要求：每次逻辑访问必须对 stash **oblivious 全扫描**（读与驱逐各一次，共 2ZL 次线性槽更新），使每次 enclave 访问触达相同足迹，防微架构侧信道。因 stash 相对整树很小，线性扫描成本可摊薄。MC-ORAM 中 stash 持有共享 112 位掩码 stash.mask，所有槽计数器随每次逻辑访问整体递增（PathORAM 中保持相等），溢出即 Refresh(stash)。
- 泄露示例（V-C）：stash 条目内容不变→AES-XTS 密文不变，攻击者从密文推断槽是否被覆盖/换块，暴露 stash 占用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TreeToStash（Algorithm 3）：stash 槽 j 的 128 位 AES 块 k
if stash.ctr == 2^16-1: Refresh(stash)
dst = wrMask[j] ? (node[i][j] XOR node.mask XOR stash.mask)   # 掩码域转换
               : stash[j][k].data                             # 不写也保留
stash[j][k].bits = (dst || ctr+1)   # 计数器无条件 +1 → 密文必变
```
- 例子：PathORAM stash=90 槽（naive）/10 槽（+，Oblix 优化）；N=2^14、Z=4 时 stash 每 2^16/(2ZL)≈585 次访问刷新一次掩码，刷新开销 <1% 运行时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：客户端内存中固定大小数组，槽含 112 位掩码数据+16 位计数器；每次访问全扫描（读+驱逐各一遍）。stash 大小静态分配（90/10）。+ 变体（Oblix 思路）只把目标块放入 stash 降低占用，但需每 3 次访问额外驱逐防溢出。
- 使用：ORAM 客户端（TEE 内）的路径数据中转与驱逐缓冲；防侧信道的关键是"无论是否命中都扫描全部槽"。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## Position Map（位置图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 位置图是 ORAM 客户端维护的映射表：记录每个逻辑块地址 d 当前在树中的位置（PathORAM 为叶子路径标签 P(l)）。每次逻辑访问前查位置图得叶子，访问后把块分配到新随机叶子并更新位置图。它是客户端唯一持有真实映射的敏感元数据，泄露它即泄露真实访问模式。因位置图访问与输入索引强相关，本身也是访问模式泄露源，需递归 ORAM 保护（见递归 ORAM 条目）。
- 本论文把位置图查询纳入每次访问延迟统计（secure position map lookup → readPath → eviction），根位置图 2^11 条存 TEE 内存、每条目配 64 位计数器保证重复访问非确定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一次访问中位置图的作用：
P = PosM[d]        # 查位置图：逻辑块 d → 叶子路径 P(l)
... 处理 ...
PosM[d] = Rand()   # 访问后随机重映射：d → 新随机叶子（概率均匀）
```
- 例子：N=2^23 的位置图用六级递归 ORAM（N=2^21/2^19/2^17/2^15/2^13/2^11，B=16 每块 4 条目）保护，每层 log(N) 减 2，根 2^11 条线性扫描存 TEE 内存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：哈希表/数组存块→叶子映射；根位置图放 TEE 内存（配 64 位计数器），上层用递归 ORAM（与主 ORAM 相同的非确定性机制）。位置图查询占总访问延迟的比例随 B、L 增大而下降。
- 使用：所有树形 ORAM（PathORAM/RingORAM）的标准客户端组件；递归深度与每层 B 的配置（表 III：N=2^14 单级 B=32；N=2^23 六级 B=16）影响总带宽。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## Dummy ORAM block（哑 ORAM 块）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Dummy 块是 ORAM 桶填充机制：为防桶占用率泄露，PathORAM 每桶用 dummy 块补齐到恒为 Z 块（RingORAM 为 Z 真实+S 永久 dummy 块）。服务器/攻击者无法区分真实块与 dummy 块，故桶大小恒定、路径流量形状恒定。dummy 块内容可保持为掩码（MC-ORAM 初始化：所有 112 位数据初始化为 node.mask）。
- 泄漏风险：若 dummy（及未变真实块）在确定性加密下密文不变，攻击者可检测"哪个位置没变"从而定位 dummy 与真实块。MC-ORAM 让所有块（含 dummy）计数器随访问递增、溢出刷新掩码，消除该定位信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PathORAM 桶结构（N=4, L=3, Z=4 示例）：
node = [block0, block1, dummy, dummy]   # 真实块+补齐 dummy，恒 Z 块
# MC-ORAM 初始化（Algorithm 1）：所有 112 位数据= node.mask（dummy 与真实同格式）
# RingORAM：桶 = Z 真实 + S 永久 dummy（Z+S 块），驱逐时洗牌
```
- 例子：RingORAM Z=4、S=3，每桶 7 块；访问时每桶只读 1 块（目标或随机 dummy），dummy 与真实块在掩码/计数器/刷新上同等对待。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：初始化时把桶填满 dummy（内容可为掩码值），驱逐/洗牌时与真实块同规则移动；MC-ORAM 要求 dummy 块与真实块在掩码、计数器递增、节点刷新上完全一致处理，防占位差异泄露。
- 使用：PathORAM/RingORAM 的桶填充与 RingORAM 单块读取（读 dummy 掩盖目标位置）依赖 dummy 块。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## 随机掩码（Masking，112 位共享掩码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Masking 是让密文非确定性的手段：数据与随机掩码（one-time pad）异或后再加密，使同地址同明文两次写入的加密前值不同。纯 masking 每次访问 128 位块都要重新生成随机掩码，需要持续高熵随机流与昂贵 AES 运算，代价过高（论文实测纯 masking baseline N=2^14/B=256B 平均 38.4ms/访问，比 64 位计数器方案慢 13.5×）。
- MC-ORAM 变体：**同一树节点/暂存内所有 112 位 AES 块共享同一个 112 位随机掩码**（node.mask/stash.mask），数据以 D[i]⊕mask 存储；掩码不随每次访问变化，只在计数器溢出刷新时更新。共享掩码足够的原因：非确定性只需覆盖"同物理位置重复写"场景，不同物理位置由 AES-XTS 的地址 tweak 天然区分。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 初始化（Algorithm 1）与掩码写（Algorithm 3）的字节级流程：
node.mask = Rand(); stash.mask = Rand()        # 每节点/暂存一个 112 位掩码
node[r][i].data = D[i] XOR node.mask           # 落树前掩码化
dst = wrMask[j] ? (node[i][j] XOR node.mask XOR stash.mask)  # 树→暂存掩码域转换
                : stash[j][k].data
stash[j][k].bits = (dst || ctr+1)              # 128 位块 = 112 位掩码数据 || 16 位计数器
```
- 安全论证（VII-A）：同掩码周期内两次访问计数器必不同（概率 1）；跨周期掩码独立均匀 → 加密前值不同概率 1−2^−112 → AES-XTS 密文不同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每节点/暂存维护一个掩码寄存器；写块时 XOR 掩码；掩码随 Refresh 更新（换新随机掩码并重 XOR 全部块）。掩码只存在于 TEE 内部（客户端可信侧），服务器只见密文。
- 使用：与 16 位计数器组合构成 MC-ORAM 的核心；带宽仅 baseline 的 1.125×（对比 64 位计数器 2×），存储减少 43.75%。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## 交错计数器（Interleaved Counter，64 位 vs 16 位）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 交错计数器是强制密文非确定性的手段：数据与单调递增计数器交错存储，每次访问计数器 +1，使同一物理地址的加密前值必变。Obelix [43] 采用**64 位数据∥64 位计数器**（每 64 位数据配 64 位计数器），是本论文所有 baseline（PathORAM/RingORAM/+/+）的非确定性方案。
- 缺点：存储足迹与 DRAM 流量翻倍（ORAM 本就比非 oblivious 多 6–8× 存储），端到端比 TME 无计数器方案慢约 1.99×。MC-ORAM 用 **16 位计数器**替代：每 112 位数据配 16 位计数器，元数据占比从 1:1 降到 16/128=12.5%，带宽从 2× 降到 1.125×；代价是计数器会溢出，需掩码刷新（见掩码刷新算法条目）。计数器位宽消融（VIII-C）：16 位最优——64/32 位百万次访问不溢出但流量大，4/8 位刷新过频，8 位在某些配置下略慢于 16 位。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 64 位交错计数器（Obelix 风格 baseline）：
word_128 = (data_64 || ctr_64)
读取: ctr = word.ctr
写入: word.ctr = ctr + 1     # 每 64 位数据带 64 位计数器读-更新-写 → 流量 2x

# MC-ORAM 16 位计数器：
block_128 = (data_112 XOR mask || ctr_16)
写入: block.ctr = block.ctr + 1          # 每次访问 +1（PathORAM 暂存/树节点同步）
if block.ctr == 2^16-1: Refresh(node/stash)   # 溢出换掩码+清零
```
- 例子：N=2^14、B=256B、Z=4 时 stash 每 585 次访问刷新一次（2^16/(2ZL)）；树节点刷新期望 3.05×10^−5 次/访问；两者摊销开销 <1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：计数器与数据同 128 位 AES 块共存（低 16 位），读-改-写；PathORAM 中同节点/暂存计数器同步递增保持相等，RingORAM 单块访问使节点内计数器可不同、任一溢出即刷整节点。
- 使用：Obelix 是 64 位交错计数器的代表系统（编译级加固）；MC-ORAM 以 16 位计数器+掩码作为低开销替代，用于 TDX/SNP 式 TEE 的 PathORAM/RingORAM。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## 掩码刷新算法（Mask Refresh Algorithm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MC-ORAM 的掩码刷新（Algorithm 6）：当某节点/暂存内任一 16 位计数器达到 2^16−1 即将溢出时，对该节点/暂存的全部 128 位 AES 块重新生成一个随机掩码并清零全部计数器：
```
Refresh(node):
  new_mask = Rand()
  for i in 1..|node|_bits/128:
      node[i].bits = (node[i].data XOR node.mask XOR new_mask) || 0
  node.mask = new_mask
```
- 关键性质：刷新频率只由公开 ORAM 参数（Z、L）与访问次数决定、与输入访问模式无关（VII-B）——节点在 level ℓ 被触达概率 1/2^(L−ℓ)，节点刷新期望 Σ 1/2^(16+L−ℓ) ≤ 2/2^16 ≈ 3.05×10^−5 次/访问；暂存每 2^16/(2ZL) 次逻辑访问刷新一次（N=2^14 时 585 次、N=2^20 时 409 次）。因此刷新不引入新泄漏（access oblivious）且摊销开销 <1%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 触发点（Algorithm 3/5 入口）：
if stash.ctr == 2^16-1: Refresh(stash)
if node[i].ctr == 2^16-1: Refresh(node)
# 刷新期间对全部块重 XOR（旧掩码→新掩码），计数器置 0；
# PathORAM 节点计数器同步递增可同刻溢出；RingORAM 任一溢出即刷整节点。
```
- 例子：N=2^14、Z=4、L=14 时 stash 刷新周期 2^13/14≈585 次访问；每次刷新只重写一个节点/暂存（约一个 AES 块组），成本 <1% 总访问时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TEE 内软件流程，触发于 TreeToStash/StashToTree 入口；生成新掩码（Rand()）、遍历节点全部 128 位块重 XOR、清零计数器。只操作 TEE 内部表示，写回时经 TME AES-XTS 加密。
- 使用：作为"16 位计数器+共享掩码"方案的溢出处理机制，使掩码从"每次访问"降为"每掩码周期"粒度，实现 1.125× 带宽下的密文非确定性。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## 递归 ORAM（Recursive ORAM，位置图递归）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 递归 ORAM 是保护位置图的经典方法：位置图随逻辑地址被访问、访问模式与输入索引强相关，因此把位置图再存进一层 ORAM；如此递归，直到根位置图小到可放客户端/TEE 内存并线性扫描。带宽成本随递归层数增加，因此用更小块 B 的递归层降低每层流量（带宽 ∝ log(N)×B）。
- 本论文配置（表 III）：N=2^14 时单级（level1 N=2^11、B=32）；N=2^23 时六级递归（N=2^21/2^19/2^17/2^15/2^13/2^11，B=16），每块 4 条目使每层 log(N) 减 2；根位置图 2^11 条存 TEE 内存、每条目配 64 位计数器保证重复访问非确定性；深层递归的非确定性用与主 ORAM 相同的掩码+计数器机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# N=2^23 的递归位置图（表 III）：
Level1: N=2^21, B=16   ← 存主 ORAM 位置图（每块 4 条目）
Level2: N=2^19, B=16   ← 存 Level1 位置图
... 逐级减 2 级 log(N) ...
Level6: N=2^11, B=16   ← 根位置图，线性扫描存 TEE 内存（每条目+64 位计数器）
# 每次主 ORAM 访问 → 顺带逐级查询/更新递归位置图
```
- 例子：位置图查询占总访问延迟的比例随 B、L 增大而下降；深层递归用小 B 使总流量低，论文实测/理论上该配置最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：位置图条目按块打包（每块 4 条目），每层一个 ORAM 实例；根位置图存 TEE 内（配 64 位计数器）。MC-ORAM 中递归层与主 ORAM 用同一掩码+计数器机制。
- 使用：任何需要保护位置图规模的 ORAM 部署（N≥2^14 时位置图无法整放客户端）；本论文用于 N=2^14/2^23 评估。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## Oblivious Stash Scan（全暂存扫描）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Oblivious stash scan 是 TEE 内 ORAM 的防侧信道手段：每次逻辑访问对暂存**全部槽**执行相同操作的线性扫描（读路径与驱逐各一次，共 2ZL 次线性槽更新），无论目标块是否已找到、是否被写入。MC-ORAM 中每个被扫描槽的计数器都 +1（即使 wrMask 为假不写入），保证每次访问暂存每条目的密文都变化。先例：Oblix [22]、ZeroTrace [34]、OBLIVIATE 等。论文 VIII-B 用四种访问模式（LS/均匀随机/高斯/重复访问 RA）验证延迟几乎一致，证明性能只依赖配置参数。
- 必要性：stash 占用率高度依赖访问模式（重复访问小、非重复大）；若处理只在找到目标时停止或只更新被写槽，指令足迹/DRAM 流量随模式变化，TEE 内可被观察。全扫描使每次 enclave 访问触达相同足迹。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 2/3 (readPath/TreeToStash) 的 oblivious 结构：
for j in 1..|stash|:
    wrMask[j] = !found && stash[j].isEmpty; found = found || wrMask[j]
for j in 1..|stash|:                       # 无条件遍历全部槽
    for k in 1..|stash[j]|_bits/128:
        stash[j][k].bits = (wrMask[j] ? node[i][j] XOR node.mask XOR stash.mask
                                      : stash[j][k].data) || (ctr+1)   # 无条件递增
```
- 例子：PathORAM 每次访问产生 2Z·L 次线性 stash 更新（读+驱逐各 Z·L 块处理）；stash=90 槽时全扫描是延迟主导项之一；优化变体（+）只把目标块入 stash 减少占用，但仍保持全扫描。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：两层循环（槽×槽内 AES 块）+ wrMask 条件选择 + 计数器无条件递增；驱逐（StashToTree）反向同理并同时更新树节点。stash 需小（相对整树）以摊薄扫描成本。
- 使用：TEE 内 ORAM 客户端（TDX/SNP）每次访问必做；与掩码/计数器/刷新机制正交，MC-ORAM 保留之。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

## 文档注意力分解（Document Attention Decomposition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MERIDIAN（ISCA'26，HUST）提出的去中心化 RAG 推理机制：把标准 softmax 注意力模块按数据来源拆成两个独立分支——DocumentAttention 分支（对预计算的文档侧 K/V 做注意力）与 QueryResponseAttention 分支（对用户 query 与已生成 token 的 KV 做注意力），各自只产出紧凑局部摘要（未归一化输出 o、局部最大值 m、归一化因子 l），再经数值稳定的 online-softmax 全局融合（共享基线 m=max(m_d,m_c)）合并。与一般"注意力矩阵按 token 类别结构化拆分"（如 VLM KV Cache 剪枝里的 Intra/Inter-modality attention decomposition，见知识库笔记）不同，MERIDIAN 的分解是执行范式的改变：文档侧 K/V 按 attention head 分片静止在 PIM 内存设备上，每个设备对本地 shard 就地算注意力，只交换紧凑统计量，无需把整份文档 KV 搬到计算设备。数学上等价于标准 softmax（精确无损），下游 LN/FFN/残差全部不变。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MERIDIAN 算法 1（In-Layer Document Attention Decomposition）逐 token 流程（单设备持文档 KV (K_d,V_d)、另一设备持上下文 KV (K_c,V_c)）：
```
# 1) QKV 投影
(q, k, v) = QKVProjection(x)
# 2) 文档分支（PIM 设备本地执行）与上下文分支并行：
s_d = q @ K_d^T ;  s_c = q @ K_c^T          # 局部 logits（GEMV）
m_d = max(s_d) ;   m_c = max(s_c)           # 局部 max 基线
o_d = Σ_j exp(s_d[j]-m_d)·V_d[j] ;  l_d = Σ_j exp(s_d[j]-m_d)
o_c = Σ_j exp(s_c[j]-m_c)·V_c[j] ;  l_c = Σ_j exp(s_c[j]-m_c)
# 3) 全局融合（共享基线 m = max(m_d, m_c)，数值稳定）：
l = exp(m_d-m)·l_d + exp(m_c-m)·l_c
o = ( exp(m_d-m)·o_d + exp(m_c-m)·o_c ) / l
# 4) 下游不变：x ← LN1(x+o)；f ← FFN(x)；y ← LN2(x+f)
```
通信量对比（FP16）：集中式 V_ce = #Doc tokens×2×d_model×2 bytes；MERIDIAN V_de ≈ (#Query+#Response tokens)×2×d_model×2 bytes——文档平均比 query+response 长 ~380×（2Wiki/HQA/NQ/TQA 实测 doc 857–14749 token vs query+response ~20 token），通信降两个数量级以上；按 head 分片到 N 设备后每设备只传 d_model/N 输出切片，跨设备总流量近似恒定。跨设备融合与 Tree Attention 的 LSE/max 树归约同源（LogSumExp 结合律），但 MERIDIAN 在 PIM 设备粒度做两步（局部 → 全局）归约。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：与 PIM 加速器协同设计——文档 K/V 离线预计算后经 CXL.mem load/store 写入 head-sharded PIM 位置，DAC（Document Attention Cluster）执行文档分支、CEC（Context Execution Cluster）执行上下文分支与融合及全部其余算子；融合在 CEC 或 BOOMv2 RISC-V 核上完成（softmax 用专用精度硬件保证模型保真）。使用方式：KV-precomputed RAG 推理场景（TurboRAG/BlockAttention 微调过的模型），文档更新走标准 KV 预计算流程写对应 shard，无需全局重排。效果：MERIDIAN 通信占比 ≤6.34%（baseline 最高 93.40%），吞吐 5.36×/6.64×（vs TurboRAG/BlockAttention），准确率差 <0.4pp（LUT 近似仅用于数值宽容算子，softmax 专用精度）。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

## 文档 KV 预计算与复用（Document KV Precomputation & Reuse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV-precomputed RAG 的核心优化：对检索语料中每篇文档，离线预计算其 Key-Value（KV）缓存并存库（如 TurboRAG [44]、BlockAttention [46]），推理时不再重复编码长文档，直接加载缓存 KV 与 query 实时生成的 KV 拼接进注意力，配合轻量微调维持生成质量，可将 TTFT 降低最高 98%。这是"KV 复用"思想在 RAG 语料侧的扩展（区别于 serving 层跨请求共享 KV Cache/prefix caching）：文档 KV 是静态可重用的（同一文档被多请求检索），但规模可达 TB 级（500K 文档约 14 TB），远超设备显存（H100 80 GB），因此必须驻留 host DRAM 并在 query 时经 PCIe/CXL 搬运到设备——这正是集中式 KV-reuse 范式（centralized KV-reuse paradigm）的通信瓶颈来源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线（检索语料构建一次）：
for doc in corpus:
    K_doc, V_doc = encode(doc)          # 文档侧 KV 预计算，轻量微调对齐
    store(K_doc, V_doc)                 # 存 host DRAM（集中式）或 shard 到 PIM（MERIDIAN）
# 在线推理（每 query）：
(q,k,v) = QKVProjection(query)          # 仅 query 需实时编码（~16 token）
K_c, V_c = cache[doc_ids]               # 拉取文档 KV（集中式：整份搬上设备）
attn = softmax(q@[K_c;K_c]^T) @ [V_c;V_c]  # 拼接后集中注意力
```
通信量：集中式每次 query 搬 #Doc tokens×2×d_model×2 bytes（FP16）；MERIDIAN 用文档注意力分解把文档 K/V 分片驻留 PIM，只传 query 向量并回收局部摘要，使"预计算"的红利不被跨设备搬运抵消。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现代表：TurboRAG（chunked 文本预计算 KV cache）、BlockAttention（块级 KV 缓存，prefill 只算 query token）；选择性重算变体：CacheBlend（高偏差 token 重算拼接）、EPIC（chunk 首 token 重算），与预计算正交。MERIDIAN 采用该复用范式但重构 KV 驻留与执行：文档 KV 按 head shard 写进 CXL Type-3 PIM 设备（标准 CXL.mem load/store），文档更新/语料扩展直接写对应 shard、无需系统级重排或重建索引。使用场景：企业 RAG 服务、个性化/隐私敏感部署（小 batch、低延迟 SLO）、长文档 QA。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

## RAG（Retrieval-Augmented Generation，检索增强生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RAG 把外部知识检索与 LLM 生成结合：用检索到的文档支撑生成，缓解幻觉与知识过时，支持低成本知识更新。两阶段：(1) 检索阶段——离线用 embedding 模型把知识项编码建索引（通常用 ANNS 如 FAISS/HNSW，也可用 BM25 关键词检索），服务时把 query 编码后检索相似文档；(2) 生成阶段——query 与检索文档拼接成增广上下文输入 decoder-only transformer（每层 self-attention + FFN + 残差 + LN）。与普通 LLM 推理的关键差异：输入序列因拼接长文档而大幅变长（每请求可达上万 token），prefill 长上下文占计算主导并抬升 TTFT；文档静态可复用催生 KV 预计算优化。检索（ANNS/BM25 等）与生成解耦、可独立优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 检索阶段
index = build_index(embed(corpus))        # 离线：embedding + ANNS 索引（HNSW/FAISS）
docs  = ANNS_search(index, embed(query))  # 在线：query 编码 → top-k 检索
# 生成阶段（RAG 推理，MERIDIAN 聚焦此段）
ctx = concat(query, docs)                 # 增广上下文（doc 可达 ~14749 token，query ~10-20 token）
for layer in decoder_layers:              # 自回归生成
    attn(qkv(ctx)); ffn(...)
```
RAG 推理（generation stage）通常主导端到端延迟：HeterRAG 对比中 generation 占端到端延迟 88.84%+；因此 MERIDIAN 聚焦加速 generation，检索优化（IKS/DReX/Pyramid/ANSMET 等）作为正交补充。实测四数据集（2Wiki/HQA/NQ/TQA）doc 平均 856.76–14748.69 token、query 仅 10.28–20.41 token、response 3–5 token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：主流框架 LlamaIndex、LangChain、Amazon Bedrock Knowledge Bases 等；检索用 FAISS（开源，github.com/facebookresearch/faiss）、HNSW、BM25；LLM 用任意 decoder-only 模型。RAG serving 优化方向：文档 KV 预计算复用（TurboRAG/BlockAttention/MERIDIAN）、缓存 KV 融合（CacheBlend）、检索加速（PIM 近存 ANNS）。MERIDIAN 在其 KV-precomputed 设定上做去中心化 PIM 推理，32 PIM 设备（16 DAC+16 CEC）共 16 TB 容量以容纳 TB 级文档 KV 库。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

## 结构化稀疏（Structured Sparsity）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 结构化稀疏是把稀疏模式约束到规则、可预测结构（块/带状/分块对角/蝴蝶/窗口/2:4 等）上的稀疏化方法：相比非结构化稀疏（逐元素任意位置置零），它牺牲一定压缩率换取"可预测性"——稀疏模式由固定块形状、固定变换层级或确定性混合路径决定，而不是任意不规则非零分布。LLM/Transformer 语境下它同时提供算法压缩与架构规整性：暴露可复用的数据流、有界依赖与可特化的执行调度（MLX 论文 II-A 对结构化稀疏的定义）。典型形式包括块对角矩阵分解（butterfly factorization）、2:4 半结构化稀疏（NVIDIA Sparse Tensor Core 支持，每 4 个连续元素恰好 2 个非零）、block-wise N:M、分块对角因果掩码、以及 FFT/滑动窗口等固定混合模式。MLX 论文的核心观察是：结构化算子的数据流图具有"前向分层、有界局部性"的公共执行形态（closed-set locality），可折叠到紧凑空间阵列执行。
- 与半结构化/非结构化稀疏的关系（本地知识库旁证）：知识库已有 N:M 半结构化稀疏（N_M Semi-structured Sparsity）、2:4 半结构化稀疏（2_4 Semi-structured Sparsity）与 BBC（Bitmap-Bitmap-CSR）等条目——2:4 要求固定 50% 稀疏率且对齐 tensor core 4×4×4 粒度；Mustafar 等 KV cache 剪枝工作指出非结构化稀疏可到 70% 稀疏度但需要专用 kernel/硬件处理不规则索引，而结构化稀疏牺牲比例换硬件友好性。MLX 的蝴蝶/FFT 结构化稀疏则走向另一极：稀疏模式完全确定（无需索引），代价是分解本身引入近似误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MLX 混合化 Transformer block 中的结构化稀疏 pipeline（伪代码，s=压缩率、B=蝴蝶块大小）：
```
# 沿序列维 N：结构化 = 固定长度 L 的 chunk 内 FFT + 低频截断（FFT-CMP）
for c in range(N//L):
    F = FFT_L(Q[cL:(c+1)L, :])     # 每 chunk 一个 L 点 FFT（固定混合模式）
    F_trunc = F[:sL, :]            # 截断高半频，保留 sL 个低频系数（确定性）
    Qs[c] = IFFT_{sL}(F_trunc)     # 缩短 token 序列 → N 变 sN
# 沿隐藏维 D：结构化 = B×B 块内蝴蝶分解（hierarchical BSMM）
#   W → (D/B)×(D/B) 个 B×B tile，每 tile W_b = ∏_{k=1}^{log2 B} B_B^(k)（块对角蝴蝶因子）
#   Y = X @ W  ≈ 逐 tile 的蝴蝶稀疏矩阵乘，复杂度 O((D²/B)·log B)
```
对比非结构化稀疏（如 Wanda 逐元素剪枝）需索引数组/位图与 gather 访存；2:4 半结构化对齐 tensor core 但固定 50% 稀疏率；MLX 的结构化稀疏把稀疏模式变成"分层交换 + 截断"的确定算子，非零位置编译期已知——这使数据流可在片上静态路由（蝴蝶 stride ±2/±4/±8 映射到 skip-hop 网格），这是非结构化稀疏做不到的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用分层：(1) 算法层——分解/截断得到结构化矩阵（蝴蝶因子 B_B^(k)、FFT 截断比 s、块大小 B），精度-效率由 (s, B) 双旋钮调节（s=0.75/0.5、B=16/32/64，B=32 最优）；LLM 上配合 LoRA 微调压缩层恢复精度（Llama2-7B/InternLM2-7B 超 60% 层应用后 QKV+Attention 计算削减 57%-72%、精度降 <1.45%）。(2) kernel/硬件层——GPU 上蝴蝶/FFT kernel 落 CUDA core（TensorCore 支持 2:4 类规则稀疏但不支持蝴蝶，导致执行单元不匹配、速度增益远小于 FLOP 削减）；MLX 空间阵列上用 CDC + tagged block 把稀疏依赖折叠成跨层流水，roofline 利用率 52%-84%。(3) 使用例子（ViT 从头训练验证）："bd.*" 块分解替代稠密投影削减 45%-55% FLOP 仅轻微精度损失，2D-FFT token mixing（FNet）同 FLOP 削减但 2-3% 精度损失，FFT-CMP（s=0.5）65% FLOP 削减仅 1.6% 精度下降。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

## 蝴蝶分解与蝴蝶稀疏矩阵乘（Butterfly Factorization / BSMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 蝴蝶分解是把稠密矩阵 W∈R^(n×n) 近似为 log2 n 个块对角稀疏因子矩阵的乘积 W≈B_n^(1)·B_n^(2)···B_n^(log n) 的矩阵分解：第 k 个因子 B_n^(k) 对固定距离 2^k 的索引对做结构化两两混合（2 参数 2×2 混合参数化下每因子 2n 参数、总参数 2n log2 n，相对稠密 n×n 的压缩比 2 log2 n / n）。乘法（BSMM，butterfly-sparse matrix multiplication）只需 O(n² log n) 复杂度，比稠密投影 O(n³) 低一个数量级。数学根源是 Cooley-Tukey FFT 的蝴蝶图——DFT 矩阵可分解为 log n 层稀疏因子，每层做固定距离的加/减乘混合（radix-2 蝴蝶 a±b·ω）；把该结构推广到一般线性变换即为 butterfly factorization（Tri Dao 等 ICML 2019 / Monarch ICML 2022 系列）。本地知识库旁证：FWHT/Fast Hadamard Transform 条目同样用 in-place butterfly 结构（log₂n 层、每层 n/2 对 (a+b, a-b)）；Block-Sparse Attention 条目用固定 butterfly sparsity pattern 逼近任意稀疏矩阵。
- MLX 论文对先验蝴蝶稀疏的批评：全局蝴蝶分解应用于整个投影矩阵，大 d 时分解问题复杂度高、收敛难、近似误差大；且 GPU 上蝴蝶 kernel 运算强度低（bandwidth-bound）却远低于 CUDA 带宽 roofline（多级 strided/shuffle 重排破坏局部性 + stage-wise 依赖与批量同步/tile 规则执行错配，执行单元不匹配——只能跑 CUDA core 而非 TensorCore）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BSMM 的蝴蝶层级计算（n=8 的 3 层分解，radix-2 双路混合）：
```
# 输入向量 x ∈ R^8，因子层 k=0,1,2（stride = 2^k）
for k in 0..2:                              # 每层 O(n)
    stride = 2**k
    for i in 0..n-1 step 2*stride:
        for j in i..i+stride-1:
            a, b = x[j], x[j+stride]
            x[j], x[j+stride] = a + b, a - b   # 蝴蝶对混合（2 参数 2×2 混合）
# 总复杂度 O(n log n)；作为权重 W 的分解时，逐因子乘输入 = BSMM
```
MLX 把 BSMM 表达为三层嵌套循环映射到空间阵列：最内层 i2 在 4×4 网格上全展开（64 输出元素并发）、中层 i1 在 PE 内本地执行、外层 i0 作为数据流图迭代由片上序列器驱动；蝴蝶层的确定性 stride（±2,±4,±8,...）直接映射为 skip-hop 网格的跳距，PE_x 把部分和路由给消费 PE_{x+s}，多个 BSMM 层并发执行形成严格分层片上流水（Fig.10）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：(1) 算法：把 QKV/FFN 投影权重离线分解成蝴蝶因子（全局版 Monarch 式 O(D log D)，或 MLX 分层版 O((D²/B)·log B)），推理时输入逐因子乘；(2) 软件：PyTorch 层用矩阵分块 + 分层交换实现 BSMM kernel（H100 上 prefill 结合 FFT-CMP 2.72× vs eager / 1.64× vs FlashAttention2）；(3) 硬件：MLX 空间阵列用 CDC/tagged-block/skip-hop 路由执行蝴蝶层折叠流水，相对先验稀疏加速器（SpAtten/DOTA/Sanger/ViTALiTy/BitVert）最多 5.8× 加速，与 FABNet（FPGA 蝴蝶加速器）重实现对比 1.19-1.30× 端到端加速、1.14× LUT 开销。局限：蝴蝶分解是近似（有精度损失）、需离线分解开销、GPU 上受执行单元限制收益打折。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

## 分层蝴蝶分解（Hierarchical Butterfly Decomposition / Hierarchical BSMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 分层蝴蝶分解是 MLX 对全局蝴蝶分解的改进：把权重矩阵 W 划分为 (D/B)×(D/B) 个 B×B 局部 tile，只在每个 tile 内应用蝴蝶因子（而非对整个 D×D 矩阵做全局分解）。总蝴蝶参数计算量从全局 O(D log D) 降到 (D/B)²·O(B log B)=O((D²/B)·log B)；复杂度比从 O(log D / D) 变为 O(log B / B)。B 是第二个可调精度-效率旋钮：B 越大结构化稀疏越强（复杂度比 O(log B/B) 更小、算得更省）但近似误差越大。论文在 B∈{16,32,64} 上做敏感度评估：更大 B 线性层 FLOP 削减更多但精度损失更大，长上下文设置下 B=32 最佳；B 还可与 FFT 压缩率 s 联合调节。
- 结构意义：该分解天然形成两级数据流——tile 间（inter-tile）按粗粒度 blocked-GEMM 数据流执行，tile 内（intra-tile）BSMM 实现细粒度结构化蝴蝶数据流；与语义感知傅里叶压缩（序列维 N）正交，在隐藏维 D 上暴露并行性，二者共同构成"混合化蝴蝶 kernel"（Table I：FFT-CMP 用于 Attn./KV Cache，hierarchical BSMM 用于 QKV/FFN）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
分层蝴蝶投影 pipeline（D=4096、B=32、tile 数 (D/B)²=16384）：
```
# 离线：对每个 (i,j) tile 做蝴蝶分解
for i in 0..D/B-1, j in 0..D/B-1:
    W[iB:(i+1)B, jB:(j+1)B]  ≈  ∏_{k=1}^{log2 B} B_B^(k)   # 32 点蝴蝶，5 层，2×32×5 参数/tile
# 推理：Y = X @ W
for i in 0..D/B-1:                             # tile 间：coarse blocked-GEMM 数据流
    for j in 0..D/B-1:
        Y[iB:(i+1)B, j] += X[:, jB:(j+1)B] @ B_tile(i,j)   # tile 内：蝴蝶稀疏数据流
```
复杂度对比：全局分解 O(D log D)=O(4096·12)≈49k 参数单位 vs 分层 O((D²/B)log B)=O((4096²/32)·5)≈2.6M——分层参数更多但分解收敛更容易、误差更小（论文核心论点：块结构把蝴蝶稀疏局部化到小子矩阵，使分解更易收敛、精度损失更小）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：在 Llama2-7B/InternLM2-7B 的 QKV/FFN 投影中替换稠密权重为 B×B 块内蝴蝶因子，配合 FFT-CMP（s=0.75/0.5）在 >60% 层上应用并 LoRA 微调，QKV+Attention 计算削减 57%-72%、整体精度降 <1.45%（Winogrande-xl/Wikitext-2/103/Ada-LEval 评估）；H100 decode 阶段结合块 BSMM 1.4-1.9× 端到端加速（减少 KV-cache 流量）。在 MLX 硬件上，B×B tile 内蝴蝶 = 闭环 CDC（n/B 个不相交 closed set），配合闭集局部性重排（I/O shuffle 把长 stride 交换转成紧凑本地数据流 + 有界次数的 stage 间交换）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

## 语义感知傅里叶压缩（Semantic-Aware Fourier Compression，FFT-CMP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FFT-CMP 是 MLX 提出的序列维压缩方法：利用 LLM 层沿序列维 N 的语义频率局部性，把 Q/K/V 的 FFT 频谱中能量较低的高频分量截掉，再把保留的低频系数逆变换回一个更短的 token 表示。关键洞察：浅层 transformer 关注局部细粒度 token 细节（能量在高频），深层编码更广上下文（能量偏向低频）——论文对 Llama2-7B 各层 Q/K/V 做 FFT 验证（Fig.5/6：layer 1 高频主导、layer 16 低频主导）。对每层定义语义 chunk 长度 L=N/f_H（f_H 为能量超过相对阈值的最谱峰，Pow2Round 到 2 的幂做硬件友好对齐），把 Q,K,V∈R^(N×D) 重塑为 N/L 个 chunk，每 chunk 沿序列维做 L 点 FFT、保留前 sL 个低频系数、sL 点 iFFT 生成缩短表示。prefill 代价从 O(N²D) 降为 O(s²N²D)，附加 chunked-FFT 开销仅 O(ND log L)。s 是可调压缩率（评估 s=0.5/0.75）。该方法的优点：保留信息丰富的低频分量（对照 FNet 式全局 2D-FFT 的精度损失）、按层自适应、天然兼容 decode 的增量更新（chunk-granular 压缩 KV cache）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FFT-CMP 的 prefill 与 decode pipeline：
```
# 离线/每层：确定 L = Pow2Round(N / f_H)（f_H = 能量超阈值的高频谱峰）
# prefill（Q 示例，K/V 同）：
for c in 0..N//L-1:
    F_c = FFT_L(Q[cL:(c+1)L, :])          # 每 chunk L 点 FFT（每特征维）
    Qs[c] = IFFT_{sL}(F_c[:sL, :])        # 截断到 sL 低频系数 → 缩短 token 表示
# 注意力在缩短序列上执行：Attn = softmax(Qs·Ks^T/√d)·Vs（注意力矩阵 sN×sN）
# decode（append-only chunk 压缩 KV cache）：
#   新 token 累积到 L 才触发一次 FFT 压缩 → append 新压缩块；已完成 chunk 复用缓存压缩块
#   固定 L，不重变换整个 prefix；FFT 开销在 L 个 token 上摊销
```
张量计算例子（Llama2-7B、N=2048、D=4096、L=256、s=0.5）：Q 重塑为 8 个 256×4096 chunk，每 chunk 256 点 FFT、保留 128 个低频系数、128 点 iFFT → 8×128×4096=1024×4096 缩短 Q；注意力矩阵 2048×2048→1024×1024（4× 缩减），KV 流量与缓冲压力同步下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：(1) 算法：QKV 投影后按层 FFT 压缩（BERT 上可逐层应用 k 层：替换全部 12 层达 69% FLOP 削减、仅 1.75% EM/1.3% F1 损失；ViT 上 s=0.5 达 65% FLOP 削减、1.6% 精度降，优于 FNet 2-3% 损失）；(2) 软件：PyTorch 层实现（torch.fft），H100 prefill 长序列 2.72× vs eager / 1.64× vs FlashAttention2（未融合 FA、TensorCore 不支持蝴蝶故受限），decode 减少 KV-cache 流量贡献 1.4-1.9×；(3) 硬件：MLX 上 FFT-CMP 与 BSMM 在 SIMD-striped scratchpad 上对齐（列向 SIMD lane 对齐序列轴 N 做 BSMM、行向流式隐藏轴 D 做 chunk FFT，避免全阵列转置），L 点 segment 形成闭集依赖。局限：s 过小精度下降、短序列收益有限（H100 上短序列无明显加速）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

## FFT-based Token Mixing（FNet 式傅里叶 token 混合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FFT-based token mixing 是用傅里叶变换替换自注意力的 token 交互机制：FNet（Lee-Thorp et al., NAACL 2021）用 2D-FFT 同时沿 token 维与隐藏维做全局混合（固定傅里叶基），移除注意力基于内容（content-dependent）的 pairwise 权重。复杂度 sub-quadratic（O(ND log N) 量级），大幅降低 quadratic attention 的 FLOP 与数据流量。它是 MLX 论文的 baseline 之一（Fig.1(c) 方向："用稀疏注意力或傅里叶变换替换 token mixing"），也是 FABNet（FPGA 蝴蝶加速器）的注意力实现方式。
- 缺陷（MLX 论文指出）：(1) 完全去除内容相关的 token-to-token 交互会伤害精度——2D-FFT 无法适配输入特定的局部或语义依赖；(2) 与 prefill/decode 流水不兼容——cache 增量更新（KV-cache）困难；(3) 无法在标准 LLM pipeline 直接部署。论文实验佐证：FNet 式 2D-FFT（"fnet.fft"）在 ViT 上同 FLOP 削减下比稠密 baseline 损失 2-3% 精度，而 MLX 的 FFT-CMP（保留低频、按层自适应）65% FLOP 削减仅 1.6% 精度下降。MLX 保留傅里叶思想但改成"语义感知 chunked FFT + 低频截断"以保留 informative 分量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FNet 2D-FFT 注意力替代 pipeline（N=序列长、D=隐藏维）：
```
# 对每个 transformer block 的输入 X ∈ R^(N×D)：
F = FFT2D(X)                    # 先沿 token 维再沿隐藏维的 2D FFT（或反之）
X' = real(F)                    # 取实部（FNet 取实部丢弃虚部）
# 下游：X' → 前馈网络；无 Q/K/V 投影、无注意力矩阵、无 KV cache
# 复杂度 O(ND log N log D) ≈ O(ND log N) 量级（对比注意力 O(N²D)）
```
对照 MLX 的 FFT-CMP：按层 chunk（L=N/f_H）内做 1D FFT + 截断 sL + iFFT 得到缩短序列，保留内容信息（低频语义分量）且 decode 可用 append-only 压缩 KV cache——是"傅里叶混合 + 内容保持 + cache 兼容"的折中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：FNet 直接把 BERT 的 self-attention 替换为 2D-FFT + 取实部（无需训练改动、收敛更快）；FABNet 硬件上用 2D-FFT 做注意力加速（专用复数蝴蝶单元）。MLX 用它作算法验证与硬件对比基线：算法上对比精度-计算折中（fnet.fft vs bd.* vs FFT-CMP），硬件上 FABNet 重实现对比（MLX 2D-FFT attention 部分 1.11-1.23× 加速、BSMM-FFN 1.21-1.31×）。局限：无内容自适应交互、精度损失（2-3%）、KV-cache 不兼容、GPU 上 FFT kernel 带宽受限（OI 低且低于 roofline）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

## Tangent FFT（切向 FFT）与负循环卷积（Negacyclic Convolution）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tangent FFT（Bernstein 2007）是把环 $\mathcal{R}_q=\mathbb{Z}_q[X]/(X^N+1)$ 上的多项式乘（即负循环卷积 negacyclic convolution）归约为一个 N/2 点标准复数 FFT 的专用变换族：正变换 $\mathrm{TFFT}[\mathbf{a}]=\mathrm{FFT}_{N/2}[\mathbf{b}]$，其中 $b_j=(a_j-i\cdot a_{j+N/2})\cdot\omega^j$、$\omega=e^{-i\pi/N}$ 为 2N 次本原单位根；逆变换 $\mathbf{b}=(\mathrm{IFFT}_{N/2}[\mathbf{c}])^*$、$a_j=\operatorname{Re}(b_j\cdot\omega^j)$、$a_{j+N/2}=\operatorname{Im}(b_j\cdot\omega^j)$。相比把多项式长度翻倍到 2N 的常规做法，Tangent FFT 的辅助运算（前后处理）更简单且完全可并行，天然适合 GPU。负循环卷积是格基 FHE 中多项式乘的核心：因环模为 $X^N+1$（而非 $X^N-1$），标准 FFT 的循环卷积（模 $X^N-1$）不能直接使用，必须用带符号折叠的专用变换。两多项式 u、v 的负循环积 $c=\mathrm{ITFFT}[\mathrm{TFFT}[\mathbf{u}]\circ\mathrm{TFFT}[\mathbf{v}]]$（∘ 为逐元素乘），把复杂度从 $O(N^2)$ 降到 $O(N\log N)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 TFHE 盲旋转中，负循环卷积出现于每次外部乘积的多项式乘（ACC 与 BSK 分量的乘积），是 PBS 每迭代的核心计算。伪代码（N=512 的 Tangent FFT，即 N/2=256 点 FFT）：
```
# 正变换（Forward Tangent FFT）
for j in 0..N/2-1:
    b[j] = (a[j] - i*a[j + N/2]) * omega^j     # omega = e^{-i*pi/N}，预计算
c = FFT_256(b)                                  # 256 点复数 FFT

# 负循环卷积：c = ITFFT[ TFFT[u] ∘ TFFT[v] ]
cu = TFFT(u); cv = TFFT(v)                      # 两个正变换
c_pt = cu ∘ cv                                  # 逐元素乘（Fourier 域）
c = ITFFT(c_pt)                                 # 逆变换，含共轭与 omega^j 复原
```
- Annotations：`omega^j` 是旋转因子（precomputation factor）；正/逆变换分别使用 `omega^j` 与其共轭版本（MNEMOS 跨迭代融合即利用这一共轭对称性）；MNEMOS 在 GPU 上把 N/2 点复数 FFT 映射到 FP64 Tensor Core（WMMA 8×8×4）执行，且精度分析表明该 FFT 需 FP64（≥30 小数位）才能保证 PBS 解密正确。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Bernstein 原版 C++、TFHE-rs/Concrete（Zama 生产库，CPU/GPU）、Lattigo（Go）、cuFHE 等 FHE 库内置；加速器（MATCHA/Morphling/Strix/FlashTFHE/CASCADE）的 FFT 单元即实现该变换（或等价的双实数 FFT/负循环卷积归约）。MNEMOS 在 GPU 上以 CUDA kernel 实现：b 向量构造与 omega^j 乘加在寄存器完成，256/512/1024 点 FFT 用四步 FFT 递归分解到 8 点 WMMA 基例，Fourier 矩阵运行时片上生成。使用场景：所有环为 $\mathbb{Z}[X]/(X^N+1)$ 的 TFHE/类 TFHE 方案的多项式乘加速；N/2 点 FFT 也显著减小变换规模（相对 2N 做法省一半点数）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## Modulus Switching 与 Sample Extraction（PBS 的缩放取整与降维提取阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Modulus Switching（模数切换）是 TFHE PBS 的第一阶段：把输入 LWE 密文 c=(a_0,…,a_{n-1},b) 的每个分量按模 2N 缩放取整，即 $\tilde{a}_i=\lfloor 2N a_i\rceil_{2N}$、$\tilde{b}=\lfloor 2N b\rceil_{2N}$，把连续 torus 值映射到整数域，从而让盲旋转能用整数旋转量决定测试多项式的位移步数。Sample Extraction（样本提取）是盲旋转之后的阶段：从包含 (k+1) 个 N 次多项式的 GLWE 累加器 ACC_n 中取出第 0 个明文分量，把 GLWE 密文（形状 (k+1)×N）还原为 LWE 密文（维度 kN+1），本质是一系列按式 $SE^i((A_0,A_1,\dots,A_{n-1},B))=((a_{0,0},\dots,a_{0,i},-a_{0,N-1},\dots,-a_{0,N-i-1}),\dots,(b_i))$ 的系数置换。两者都是开销极小的"管道"阶段（FlashTFHE 剖析各占 PBS 时间 <1%），但却是 PBS 算法链中不可省略的正确性环节。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 位置与数据流（Algorithm 1）：MS 在最前（LWE→整数化）→ 盲旋转 n 次迭代 → SE 在盲旋转后（GLWE→LWE）→ Key Switching 收尾（kN+1→n+1）。伪代码：
```
# Modulus Switching：把 torus 密文整数化到模 2N
for i in 0..n:
    a_tilde[i] = round_to_nearest(2N * c[i]) mod 2N   # 含 b 分量
# Blind Rotation 使用 a_tilde 作为旋转量（X^{a_tilde[i]}）

# Sample Extraction：取 GLWE 累加器 ACC 的常数项回 LWE
out = []
for poly in ACC[0..k]:                     # k+1 个多项式，各 N 系数
    out += [poly[0], poly[1], ..., poly[i],        # 正序前 i+1 项
            -poly[N-1], -poly[N-2], ..., -poly[N-i]] # 负号折叠后 N-i-1 项
out += [ACC_body[0]]                        # 提取的 body 项 b'
# 得到维度 kN+1 的 LWE 密文，交由 Key Switching 切回 n+1 维
```
- Annotations：`a_tilde` 的量化步长 1/2N 决定旋转精度（更大 N 更准但更贵）；SE 的输出维度 kN+1 直接决定 Key Switching 的矩阵规模（k、N 越大 KS 越重）；MNEMOS 中 MS/SE/KS 在 GPU 上与盲旋转同流水执行，SE 的置换在 kernel 内完成、不引入额外全局内存往返。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TFHE-rs/Concrete 的 PBS 流水内嵌实现（MS 为逐元素缩放取整，SE 为寄存器/共享内存内置换）；硬件加速器在专用单元中实现（如 FlashTFHE 的 LPU 处理 SE 与 KS，盲旋转在 BRU）。使用要点：MS 必须在盲旋转前完成（旋转量需为整数）；SE 后密文维度膨胀到 kN+1，必须紧跟 Key Switching 恢复 n+1 维；位宽管理（bit-removal rounding，ZAMA）也通过 PBS 实现、同样走这套 MS→BR→SE→KS 流程。MNEMOS 在 A100/H100 上整套流水全程 GPU 执行（修改 Concrete 后端把 PBS 独占 offload 到 GPU）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## DLRM 训练（Deep Learning Recommendation Model，训练视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DLRM（Deep Learning Recommendation Model）是 Meta 的推荐模型架构：dense 特征（如用户年龄）过 MLP，sparse 特征（如 post ID）查 embedding 表（table-batched embedding，TBE），经 dense interaction 层连接后进最终 MLP 输出预测（点击率/参与度）。训练视角（MTIA 300，ISCA'26）：DLRM 训练与 GenAI 训练不同——FLOPS 需求中等（单样本 ~3 GFLOPs）但 HBM 容量/带宽与网络带宽需求大、collective 通信频繁，常致加速器利用率低。MTIA 300 的生产 DLRM 训练模型约 150B 参数（99% 在稀疏侧，embedding 表常超单卡容量故需混合并行分片），用 TorchRec 实现 + TorchInductor 全图编译 + 分布式 Shampoo 优化器 + 分布式数据并行。性能：40 卡 local batch 6144 时通信超 H100 3.9×、端到端 Perf/TCO 1.42×（Table IV）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 DLRM 训练迭代的算法 pipeline（MTIA 300）：
```python
# 前向
for u in batch:                                    # 批量用户
    dense_vecs = MLP(dense_features[u])            # dense 特征 → MLP（PE 的 DPE）
    emb = TBE_forward(sparse_idx[u], tables)       # 稀疏特征 → 查 embedding 表
    out = interaction(dense_vecs, emb)             # dense interaction 层
    yhat = final_MLP(out)
loss = CE(yhat, label); loss.backward()            # AOTAutograd 生成反向图
# 反向: embedding 索引 radix-sort 重排 + TBE_backward + dense 梯度
# 优化: 分布式 Shampoo（AllGather 阶段）+ AllReduce 梯度同步 + AllToAllv 特征交换
# 通信画像（40 卡）: AllReduce 1.6 GB / AllGather 2.1 GB / 35×AllToAllv(1KB-1GB)
```
MTIA 300 侧 co-design：关闭 row-wise FP8 量化通信（+4.4%）、Shampoo 特征分解 offload host CPU（1:1）、local batch 10240（24 卡、+2%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TorchRec（PyTorch 推荐域库）建模 + TorchInductor 编译（MTIA 与 H100 同栈对比）；嵌入表按 table-wise/row-wise 混合并行分片；Shampoo 分布式优化。使用场景：Meta 广告/短视频/好友流推荐训练；DLRM 与 GenAI（LLM）训练的系统需求差异是 MTIA 300 硬件设计（内置 NIC/ME/NMC、高 HBM bytes-to-FLOPS）的直接动机。演进：GenAI 影响下 DLRM 采用更大 dense 组件与 Transformer 结构（MTIA 400 提高 FLOPS 应对）。信息缺口：论文未给出具体 DLRM 网络结构（层数/维度）。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines

## 分布式 Shampoo 优化器（Distributed Shampoo Optimizer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shampoo 是预条件随机张量优化器（Gupta et al., ICML 2018）：对每个参数张量按各维计算 Kronecker 因子（G_t = G_t + g g^T 的逐维累积），用矩阵逆根（Kronecker-factored preconditioner）做预条件更新，比 Adam 收敛更快但需存储矩阵平方根/逆根（内存与计算开销大）。分布式 Shampoo（Shi et al., arXiv:2309.06497）是 PyTorch 的分布式数据并行实现：优化阶段需 AllGather 交换预条件器/统计量。MTIA 300（ISCA'26）的生产 DLRM 训练用分布式 Shampoo 做 dense 组件优化（稀疏侧用 TBE/稀疏优化器），通信画像中 AllGather 入站 2.1 GB 即来自 Shampoo 优化阶段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 Shampoo 更新步骤的算法流程（MTIA 300 训练）：
```python
# 每参数张量 W ∈ R^{m×n}（如 dense 层权重）:
# 1. 统计累积: L_t = L_{t-1} + G W^T;  R_t = R_{t-1} + G^T W   (G 为梯度)
# 2. 预条件: L^{-1/4}, R^{-1/4}（矩阵特征分解/逆根）
#    → MTIA 300: 特征分解 offload 到 host CPU（1:1 架构，保数值精度;
#       若 1:8 或片上实现会损失 7.8%）
# 3. 更新: W_{t+1} = W_t - η · L^{-1/4} G R^{-1/4}
# 4. 分布式: 每步需 AllGather 预条件器/参数分片（2.1 GB 入站, 40 卡）
```
MTIA 300 上 Shampoo 特征分解是 host CPU offload 的代表算子（论文 3 项 co-design 之一）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：分布式 Shampoo 开源（PyTorch 参考实现，https://arxiv.org/abs/2309.06497）；MTIA 300 上特征分解 offload host CPU（1:1 host:加速器支撑）；H100 用 cuSOLVER 在 GPU 上算（8:1 时无碍）。使用场景：DLRM dense 组件训练（与稀疏侧 TBE 优化器并存）；通信开销（AllGather 2.1 GB）被 MTIA 300 的 ME/NMC 卸载消化。信息缺口：论文未给出 Shampoo 的预条件器更新频率与精度策略细节。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines

## MXFP / MX 格式（Microscaling Floating Point，OCP 微缩放浮点格式）

术语解释
MXFP（OCP Microscaling Formats，MX）是 AMD/Intel/Microsoft/NVIDIA/Qualcomm 在 Open Compute Project 下定义的块共享指数（block-wise shared exponent）缩放浮点格式：把一组 FP 值（block，如 32 元素）共用一个 8-bit 指数（E8M0）做归一化，block 内每个元素用更少的 exponent/mantissa 位表示（MXFP4 E2M1、MXFP6 E2M3/E3M2、MXFP8 E4M3/E5M2），以少量元数据换取大幅扩展的动态范围。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：极低位宽浮点（如 FP4，S1E2M1）单个张量只用一个缩放因子 S，离群值抬高 S 导致正常元素丢失精度（精度差）；→ 块共享指数把整张量切成 block，每个 block 独立 8-bit 共享指数（E_b^shared，OCP 规范中为 E8M0），先按 block 归一化再量化，等效按 block 调整动态范围、缓解离群值影响（精度提升）；→ block 结构天然匹配 Tensor Core 的块状数据（如 NVIDIA Blackwell 原生支持 MXFP4/FP4 低比特执行），元数据（8B/block）被 block 内元素分摊（硬件/内存高效）。数学形式（MXFP4，S1E2M1）：X_{b,fp4}^q = Quant(X_b^{fp16}/2^{E_b^{shared}})，E_b^{shared} = |log2(max_b(|X_b^{fp16}|))| − E_element^MAX，其中 E_element^MAX 是元素格式最大指数（E2M1 的 bias=2^(2−1)−1=1，E_element^MAX=11_2−bias=2）。Web 证据：OCP Microscaling Formats (MX) Specification v1.0（https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf）；NVIDIA NVFP4 博客（https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在量化推理 pipeline 中，MX 格式按"块内共享指数 + 固定元素配置"组织张量：例如 4-bit MXFP4，block=32：
```
对每个 block b:
  E_b^shared = |log2(max_b|X_b^{fp16}|)| - E_element^MAX   # E_element^MAX=2 (E2M1)
  for x_i in X_b: x̂_i = quant(x_i / 2^{E_b^shared})        # 4-bit E2M1
```
Tensor Core 执行时先按 block 取回 8-bit 共享指数与量化元素，点积后按指数缩放累加进高精度 accumulator。局限：MXFP 对所有 block 用单一固定配置（如 MXFP4 一律 E2M1），无法适应块间/块内值多样性（见"块间/块内值多样性"条目），正是 MXFFP 论文的切入点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：OCP 规范定义 E8M0 共享指数与 E4M3/E5M2/E3M2/E2M3/E2M1 等元素格式；参考实现与生态：OpenXLA/StableHLO 加入 MX 类型（f4E2M1FN、f6E2M3FN、f6E3M2FN、f8E8M0FNU，https://github.com/openxla/stablehlo/pull/2582）、ggml/llama.cpp 的 MXFP 实现（https://github.com/ggml-org/llama.cpp PR #20609）、NVIDIA CUTLASS。使用：权重静态离线转 MX 格式、激活运行时转换（OCP-compliant conversion rule）；NVIDIA Blackwell 硬件原生支持 FP4/MXFP 低比特执行，XLA 生态通过新增 MX primitive type 打通到硬件。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration

## 低位宽浮点位配置（exponent-mantissa bit configuration，range-resolution 权衡）

术语解释
同一比特宽度下浮点数可分配不同的 exponent/mantissa 位数（如 4-bit 可配 E0M3、E1M2、E2M1、E3M0，S1ExMy 记法：1 符号位 + x 指数位 + y 尾数位），指数位多则动态范围大、分辨率低，尾数位多则分辨率高、范围窄——构成"范围-分辨率"（range-resolution）权衡。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：浮点数由 sign/exponent/mantissa 三段组成；固定总位宽（如 4-bit）下三段位宽互相挤占；指数位决定可表示的最大/最小指数（范围），尾数位决定相邻可表示值的间距（分辨率）；因此 E0M3（全尾数）只适合值域窄但需高精度的数据，E3M0（全指数）适合动态范围大但精度要求低的数据，E1M2/E2M1 是中间权衡（Fig.3 显示四种 4-bit 配置的可表示值分布明显不同）；没有单一配置对所有数据分布最优——这是 MXFFP 的核心动机。论文实测：oracle（逐 block 选最优配置）下 E1Mx 与 E2Mx 两类配置覆盖 97.2% 的最低 MSE 选择，E0/E3 等极端配置仅 2.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在量化 pipeline 中，位配置直接决定 block 内每个元素的量化误差。以 MXFFP4 的相对指数比较为例（E1M2 vs E2M1，表 I）：元素相对指数 E_i^r=E_i−E_b^MAX（≤0）为 0 时，E1M2 用 2 位尾数（2^1×1.xx₂）分辨率高于 E2M1（2^1×1.x₂）；E_i^r∈{−2,−3} 时 E2M1 凭借额外指数位（2^0×1.x₂）比 E1M2（2^1×0.01₂）表示得更细；E_i^r=−1 或极小值两者等价。选择规则（Algorithm 1）：count_E1=|{E_i^r=0}|（偏好 E1M2）、count_E2=|{E_i^r∈{−2,−3}}|（偏好 E2M1），若 count_E1²>count_E2 选 E1M2（E_element^MAX=1）否则选 E2M1（E_element^MAX=2），共享指数 E_b^shared=E_b^MAX−E_element^MAX。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上每种配置是数据通路中的一段位映射（exponent 位域长度、bias、mantissa 位域长度不同）；MXFFP 用每 block 1-bit 配置字段在 E1Mx/E2Mx 两配置间选择（8 个配置位合成 1B configuration set 保持字节寻址）；Tensor Core 用 bit mapper（多路选择器阵列）按配置位把操作数重排进统一算术核（FP4 E2M1 核加宽为内部 E2M2）。若需更多配置可换 preset 对（E0/E2、E2/E3）或加宽选择器。使用场景：低位宽（4/6/8-bit）post-training quantization 中按数据分布动态选择位分配，提升表示精度。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration

## 块间/块内值多样性（Inter-block / Intra-block Value Diversity）

术语解释
块间值多样性指不同 block（权重/激活矩阵的共享指数分组）之间值分布特征（所需指数范围）不同，单一固定位配置无法兼顾；块内值多样性指同一 block 内元素的值分布差异随 block 增大而放大，迫使更多元素共享同一指数与位配置而丢失细粒度表示。二者是 MXFP 在"极低比特 + 大 block"趋势下精度退化的根源。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：MXFP 用 block 共享指数缓解整张量范围失配，但极低比特下共享指数归一化后 block 间仍残留差异（inter-block diversity）——实验（Llama3-8B layer9 query projection 激活热力图）显示归一化后下层 block（如 Block 25）仍需要更大指数值（配 E2M1 保范围），上层 block（如 Block 679）多数元素只需小指数（可省指数位给尾数用 E1M2 提分辨率）；同时为摊薄 8-bit 指数元数据（Llama3-405B 从 block 32 的 11.7GB 降到 block 256 的 1.46GB）而增大 block size，会让更多元素共享同一指数/配置，放大 block 内值分布差异（intra-block diversity）。两者都导致固定配置（MXFP4=E2M1）表示失配、perplexity 随 block 增大单调上升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化 pipeline 中的体现：oracle 格式（逐 block 允许任意配置、选 MSE 最小者）在 Llama3-8B 上测得的配置分布（Fig.5）表明激活与权重对指数位数的偏好不同且随位宽变化，单一固定配置（如 MXFP8 的 E5M2/E4M3）与实测偏好错位导致更高 MSE；oracle 在 4-bit 下 perplexity 达 Llama3 8.3/OPT 15.1，而 MXFP4 严重退化（Llama3 30.98/OPT 88.81，WikiText-2）。Oracle-SB（oracle+sub-blocking）在 block 64/256 时 perplexity 8.5/9.0 vs oracle 15.2/44.2，证明 intra-block diversity 由 sub-blocking 解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/应对：MXFFP 用"1-bit 配置字段（块级，解 inter-block diversity）+ sub-block 结构（解 intra-block diversity）+ 基于相对指数统计的运行时配置选择（Algorithm 1）"同时处理两类多样性；权重离线按双配置量化选 MSE 小者（=oracle），激活运行时用计数规则近似。使用场景：4-bit 低比特推理与 256 大 block 部署（元数据降 4× 仍保精度），也扩展到 ViT 等非 LLM 负载（ViT-base/large 4-bit Top-1 从 MXFP 的 76.46%/79.54% 恢复到 MXFFP 的 79.36%/81.36%）。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration

## 子块化（Sub-blocking / sub-block）

术语解释
把大 block（如 256 元素）内部划分为更小的 sub-block（如 32 元素），每个 sub-block 拥有独立的 1-bit 位配置字段（8 个 sub-block 配置位合成 1B configuration set 保持字节寻址），但整个大 block 共享单一 8-bit 指数——以"配置粒度细、缩放粒度粗"的解耦结构同时获得低元数据与高精度。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：大 block 降低共享指数元数据开销但放大块内值多样性、损害精度（oracle 在 block 256 时 perplexity 44.2 vs block 32 的 15.2）；→ 把 block 拆成 32 元素 sub-block，每个 sub-block 独立选 E1Mx/E2Mx 配置，就能在 block 内继续捕捉值分布差异（intra-block optimization）；→ 但多个 sub-block 若各自存指数会放大元数据，故所有 sub-block 共享一个按 E2Mx 假设计算的指数，E1Mx 的 sub-block 使用时减 1 补偿偏置差——"配置位 per sub-block（1 bit）+ 指数 per block（8 bit）"解耦，元数据仅 1 bit/sub-block。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化 pipeline 中的流程（block 256 = 8 sub-block × 32）：
```
对每个大 block:  E_b^shared = 以 E2Mx 假设计算的统一指数
对每个 sub-block s (32 元素):
  相对指数 E_i^r = E_i - E_b^MAX(s)
  count_E1/count_E2 统计 → cfg_s ∈ {E1M2, E2M1}
  实际缩放: cfg_s=E1Mx 时用 E_b^shared - 1, E2Mx 时用 E_b^shared
```
结果：Llama3-8B 4-bit 下 MXFFP block 256（sub-block 32）perplexity 24.3，仍低于 MXFP block 32 的水平（≈30.98），且 sub-block 4 时平均 perplexity 退化仅 0.98、比 MXFP8/MXFP6 内存需求低 47.8%/29.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存布局上 configuration set（8 个 sub-block 配置位 → 1B）与共享指数并排存放；硬件上每个 threadgroup 处理一个 sub-block（映射到 4×8 子矩阵、4 step 执行），转换阶段先跨 threadgroup 同步全局 E_b^MAX（Max 单元），再逐 sub-block 用 Config Selector 选配置，Normalization & Round 按统一指数输出。使用：需要大 block（256）摊薄指数元数据、同时保持 4-bit 精度的推理部署。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration

## 运行时量化转换与 oracle 配置选择（Runtime Conversion / Oracle Format，Algorithm 1）

术语解释
把 FP16/BF16 张量转换为低比特格式的过程。MXFFP 中权重为静态数据走离线 oracle 转换（两种配置都量化、选 MSE 小者，零运行时开销）；激活是运行时数据，用"相对指数统计计数"的轻量规则（Algorithm 1）近似 oracle 的每 block 最优配置选择，把逐 block 配置决策降到几次计数 + 1 次比较的硬件代价。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：MXFFP 支持多配置，但激活在推理运行时才出现、逐配置试量化会引入显著转换开销；→ 观察相对指数 E_i^r=E_i−E_b^MAX（≤0）直接决定哪种配置表示更精细（表 I：E1M2 偏好 E_i^r=0、E2M1 偏好 E_i^r∈{−2,−3}）；→ 因此只需统计 block 内 E_i^r=0 的个数 count_E1 与 E_i^r∈{−2,−3} 的个数 count_E2，用二次加权规则 count_E1²>count_E2 决策（大值元素主导数值保真度）；→ 选完配置再算共享指数 E_b^shared=E_b^MAX−E_element^MAX，逐元素量化输出。该规则使激活配置选择与 oracle 高度吻合（Fig.20a），最终输出 MSE 几乎等于 oracle（Fig.20b），端到端零额外延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（MXFFP4 运行时转换，输入 block x={x_1..x_N}）：
```
E_i ← exponent(x_i);  E_b^MAX ← max_i E_i
E_i^r ← E_i − E_b^MAX
count_E1 ← |{E_i^r = 0}|;  count_E2 ← |{E_i^r ∈ {−2,−3}}|
if count_E1² > count_E2:  cfg ← E1M2, E_element^MAX ← 1
else:                     cfg ← E2M1, E_element^MAX ← 2
E_b^shared ← E_b^MAX − E_element^MAX
x̂_i ← quant(x_i / 2^{E_b^shared})
```
大 block 时 sub-block 各自决策配置，但统一指数按 E2Mx 计算、E1Mx sub-block 使用时减 1，复用同一硬件路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件上对应 MXFFP converter 流水线——Max 单元取 E_b^MAX、Subtractor 算相对指数、Counter 统计 E_i^r 分布、Config Selector 按二次加权规则决策、Normalization & Round 输出量化 block（Max 与 Normalization&Round 在 baseline MXFP 已存在，只需小幅扩展）；软件上权重离线走 oracle（双配置量化选 MSE）。使用：LLM 推理中所有 MMA 相关张量转 MX 格式的转换路径，覆盖 4/6/8-bit，也推广到 ViT 等非 LLM 负载。

涉及论文标题：
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration

## 自回归解码（Autoregressive Decoding）与递归数据依赖

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
自回归解码是因果语言模型（LLM）的生成方式：逐 token 生成，每个新 token 的预测以之前所有已生成 token 为条件（P(t_1..t_n)=Π P(t_i|t_<i)），因此每次迭代都等待上一次输出作为当前输入，形成递归数据依赖。这带来两种典型执行形态：prefill（并行处理整段输入序列，计算密集）与 decode（逐 token 迭代，访存/延迟敏感）。对部署系统（尤其 wafer-scale）的直接影响：解码的"每迭代依赖前输出"使 pipeline 化的层间数据流形成闭环——最后一个流水段（输出）必须把结果送回第一个流水段（输入），若两者物理距离远（如 ZigZag 映射下近直径路径），每次 decode 迭代都要承担长距离通信，成为端到端延迟的主要成分（BusyBarn 的 Fig.1 黄箭头）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
decode 迭代的计算-通信依赖（pipeline 视角）：
```
for step in 1..T:                        # T 个新 token
    h = embed(x_{step-1})                # 上一 token 输出作为输入（递归依赖）
    for layer_group g in 1..G:           # PP：die 组间串行
        h = Attn(g, h); h = FFN(g, h)    # 组内 TP/SP/CP 并行计算
        send(h, next_group)              # 组间传输激活（D2D 链路）
    x_step = sample(head(h))             # 最后一个 die 组产出 token
    send(x_step, group 1)                # ★ 回环：最后组→第一组（距离决定延迟）
```
自回归递归依赖使"最后一个 die 组与第一个 die 组"之间的通信距离成为关键指标：ZigZag 映射下该距离近 mesh 直径（Fig.5a/5c 虚线箭头），BusyBarn 的 Hamiltonian Loop 映射（Fig.5b/5d）把 die 组排成环使最后↔第一相邻，每次 decode 迭代的回环通信降到一跳邻居距离——这是 inter-die 映射直接由解码算法特性驱动的设计。评估中序列长度 512/2048/8192 覆盖 prefill 与 decode 两种形态（两者计算与通信模式差异显著）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：所有自回归 LLM 推理框架/硬件均按此逐 token 迭代执行（vLLM 等 serving 框架用 continuous batching 同时推进多条请求的 decode，见"Prefill/Decode"与"Continuous Batching"条目）；并行系统用 PP 把层切到多设备并承担回环通信。使用：BusyBarn 以 decode 的递归依赖为映射优化目标（Hamiltonian Loop）而非仅考虑 DNN 前馈数据流；其数据流执行允许 TP 部分和在单个 tile 完成后立即 reduce-scatter（vs bulk-synchronous 等整矩阵乘完），进一步隐藏通信（ablation 显示通信优化对端到端延迟的贡献大于单独映射改进）。局限：自回归本质限制批量并行度（decode 每步只推进一个 token），是 KV cache 与批处理优化（PagedAttention 等）的动机。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference

## Voronoi Diagram（VD，Voronoi 图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Voronoi 图（Voronoi Diagram，以数学家 Voronoi 命名）是计算几何的经典空间划分结构：给定点集 S={s_1,...,s_m}（称为 sites/种子点），把空间划分为 m 个 Voronoi cell V(s_i)={p∈R^3 | ∀j≠i, d(p,s_i)≤d(p,s_j)}，每个 cell 包含"离该 site 比离任何其他 site 更近"的所有点；cell 边界由相邻 site 的垂直平分面构成。Voronoi 图与 Delaunay 三角剖分互为对偶，广泛用于最近邻查询、路径规划、插值与覆盖分析。NS-FPS（ISCA'26）首次把 Voronoi 图与 FPS 联系起来：FPS 维护的"每点到已采样集最近距离"缓存 T 隐含地就是一个 Voronoi 图——每个未采样点被分配给离它最近的已采样点（其 Voronoi cell 中心），T 的值即该点到 cell 中心的距离。当新一轮采样出 s_{m+1} 时，只有落在新 cell V(s_{m+1})（即离 s_{m+1} 比离旧采样集更近）内的点需要更新距离缓存。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - NS-FPS 的 Voronoi 部分更新理论（Section III-A）：设第 k 轮采样集 S_k，最新点 s_k 的搜索半径 d_k=min_{s_i∈S_{k-1}}||s_k−s_i||²（s_k 到旧采样集的最近距离）。证明：任何需要更新距离的点 p 必满足 ||p−s_k||² < min_{s_i∈S_{k-1}}||p−s_i||²；若 p 在球 B(s_k,d_k) 外则 ||p−s_k||²>d_k，而由 s_k 是第 k-1 轮最远点知 min_{s_i}||s_k−s_i||² ≥ min_{s_i}||p−s_i||²，矛盾。因此更新区域被界定为球 B(s_k,d_k)（该球安全包住真实 Voronoi cell，Fig.4 黄球包绿 cell），复杂度：
```
# vanilla FPS 每轮: 更新全部 N 个点 + 全量找 max  -> O(N)
# NS-FPS 每轮:      只更新球内 O(N/k) 个点 + 层次 max -> O(N/k)
# 总复杂度:          O(Σ_{m=1..M} N/m) = O(N log M) ≈ O(N log N)
# 伪代码（核心替换）:
for k in 1..M-1:
    d_k = T[s_{k-1}]
    for p in ball_query(s_{k-1}, d_k):   # 只有 Voronoi cell 邻域内的点
        T[p] = min(T[p], dist2(p, s_{k-1}))
    s_k = hierarchical_argmax(T)          # 只刷新受影响块
```
  - 关键性质：部分更新机制同时缩小"求全局最大"的范围——传统 FPS 每轮要全量扫描 T 找最远点，NS-FPS 把 max 搜索限制在受影响的 Voronoi cell 内，配合层次缓存进一步降低开销；且该重述与原始 FPS 采样结果逐点一致（lossless）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 通用实现：Voronoi 图构造算法（Fortune 扫描线 O(n log n)、增量法、Bowyer-Watson 经 Delaunay）多在 CPU/GPU 计算几何库实现；但在 NS-FPS 中**不显式构造** Voronoi 图——精确构造 VD 边界计算代价过高（论文明确说明），而是用"可证明充分"的球形搜索区域 B(s_k,d_k) 松弛替代，配 Morton cube 划分快速枚举球内点。该思路把 VD 的几何洞察转化为可硬件化的部分更新 + 球查询原语；后续 k-NN/ball query 等下游点云操作可直接复用 Morton 重排布局。

涉及论文标题：
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

## ANNS（Approximate Nearest Neighbor Search，近似最近邻搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANNS 是最近邻（NN）搜索的松弛版本：给定 N 个 D 维数据库向量与查询向量 q，返回与 q 距离最接近的 k 个向量的近似结果，以可接受的精度损失换取亚线性搜索时间。核心距离度量包括 L2 范数与内积（IP）。精度用 recall@k = |P'∩P|/|P|（ANNS 返回集与真 kNN 集的重合比例）衡量，效率用 QPS（每秒查询数）衡量。ANNS 是向量数据库、RAG（检索增强生成）、推荐与信息检索的核心算子；本论文将其置于 LLM RAG 场景，检索阶段的内存带宽瓶颈直接决定整体推理性能。索引方法分为哈希式、树式、量化式与图式四类，其中图式（graph-based）在商用数据库（Milvus、Weaviate 等）与 RAG 系统中被广泛采用，可提供数量级的吞吐提升同时保持高 recall（vault 笔记：/data3/paper_analysis/knowledge_notes/算法知识笔记/Approximate Nearest Neighbor (ANN) Search（近似最近邻搜索）.md）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 ANNS 查询 pipeline：① 索引构建（一次性离线）——把语料向量组织成可搜索结构（如 HNSW 多层图、IVF 聚类+量化码）；② 查询搜索（在线、反复执行，决定系统性能）——从入口出发遍历索引，对候选向量计算与 q 的距离并维护 top-k 候选。对图式 ANNS（HNSW），查询 pipeline 为逐层 BFS：初始化候选优先队列（含入口点）→ 每 hop 取出队列最近点 → 取其邻居表 → 计算邻居与 q 的距离（全 D 维）→ 距离小于 threshold（队列最远点距离）则插入队列 → 重复直到队列耗尽。伪代码（HNSW 搜索，每 hop）：
```
cand ← {entry}; visited ← ∅; threshold ← +∞
while cand 非空:
  x ← cand 中距 q 最近且未访问的点; visited ← visited ∪ {x}
  for n in neighbors(x):                       # 邻居表查找
    if n ∈ visited: continue
    d ← distance(q, n)                          # 全 D 维距离计算（内存受限热点）
    if d < threshold:
      更新候选队列并弹出最远点; threshold ← 队列最远距离
```
核心特征：距离计算算术强度极低（每元素仅一次减/乘/加），性能完全被数据访问带宽限制（roofline 分析见硬件架构条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现库：FAISS（Meta）、ScaNN（Google，SCANN 为本论文 CPU SOTA baseline）、hnswlib、Milvus、cuVS（NVIDIA，本论文用其构建 HNSW 索引）。本论文：以 HNSW 图式 ANNS 为对象（Dfloat 数据布局 + FEE-sPCA 早退 + NDP 加速），baseline 含 CPU 的 HNSW/SCANN、GPU 的 CAGRA、NDP 的 ANSMET 等，在 SIFT/GIST/BigANN/GloVe/Wiki/MS_MARCO 六数据集上以 recall@k≥90% 比较 QPS。开源：https://github.com/Intelligent-Computing-Research-Group/NasZip。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## Early Exiting（早退，EE / FEE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Early Exiting 是在逐维累加的距离计算中，一旦部分距离超过阈值即提前终止剩余维计算的一种剪枝技术：给定 D 维向量 x 与查询 q，部分距离 d_part^k(x,q) 是前 k 维的累加距离（L2 下严格小于全维距离 d_all），当 d_part^k 超过当前候选队列最远距离 threshold 时，该向量不可能成为更近的候选，继续计算是浪费，因此触发退出。它把平均访问维度数从 D 降到 k<D，直接削减内存受限的 ANNS 中的 DRAM 访问量。本论文指出既有 EE 的局限：部分距离收敛到 threshold 的速度太慢，导致实际只省约 20% 计算（Fig.3-5 动机分析）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EE 在距离计算 pipeline 中的位置（每候选向量）：
```
d_part ← 0
for k = 1 to D（按 DRAM burst 步进，如每次 2 维）:
  d_part ← d_part + Σ_{i∈burst(k)} (x_i − q_i)²      # 部分距离累加
  if d_part ≥ threshold: return REJECT              # 早退：丢弃该候选
return ACCEPT（d_all = d_part < threshold → 入候选队列）
```
例子（论文 Fig.6a）：候选队列阈值 2.5，邻居 s1/s2/s3 分别在第 2、4、… 维触发早退被拒绝，仅 s0 全维计算后入队。对比朴素 EE：若一个"应被拒绝"的向量要到第 109 维才满足 d_part≥threshold，而使用估计距离可在第 4 维就触发（见 FEE-sPCA 条目），早退效率的差别即在此。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上 EE 需要：部分距离累加器、threshold（来自候选优先队列最远点）、逐 burst 比较逻辑。硬件上由 NDP 的 FEE 模块实现（每次累加器更新即比较）；软件 ANNS 库中常用"维度重排 + 阈值剪枝"近似实现。局限：仅用 d_part 收敛慢（论文动机），需结合 PCA 估计（FEE-sPCA）与位级压缩（Dfloat）才显著。论文按 burst 粒度触发（每 2 维/步），与 DRAM 突发访问天然对齐。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## FEE-sPCA（Statistics-based PCA-guided Feature-Level Early Exiting，统计PCA引导的特征级早退）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FEE-sPCA 是 NASZIP 提出的算法级压缩技术，解决朴素 EE 部分距离收敛慢的问题：先对向量库做 PCA 变换使前 k 维集中最富信息的分量，再以"估计全维距离"d_est^k = α_k·d_part^k/β_k（α 放大、β 校正）与 threshold 比较，使 d_est^k≥d_part^k 从而更早触发早退；β 由统计方法（Chebyshev 不等式）保证估计不低估真实距离，避免误杀本应入队的候选、维持 recall。它把平均特征计算量削减约 50%（高维数据集更多：GIST 960 维中 80% 早退发生在第 193 维内）。Offline 阶段（一次）完成 PCA 与 α/β 计算，Online 搜索阶段仅查表缩放。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Offline 预处理（图 6 上半部分）：(1) 对数据库 P 做 PCA → 变换库 VD 与特征值 {λ_i}；(2) 由期望性质 E(‖v_1:d‖²/‖v‖²)=Σ_{i≤d}λ_i/Σ_{i≤D}λ_i 得 α_k = Σ_{i=1..D}λ_i / Σ_{i=1..k}λ_i；(3) 用 Chebyshev 不等式 P(|α_k·d_part^k/d_all − 1| ≤ ε_k) ≥ 1 − Var_k/ε_k²，取 1+ε_k=β_k 使 P(α_k·d_part^k/β_k < d_all) ≥ 1−Var_k/(2ε_k²)（论文取 ≥90%），Var_k 在索引构建时统计。Online 搜索（每候选向量，逐 burst 步进）：
```
d_part ← 0
for k = 1 to D（每 burst 读入 b 个特征）:
  d_part ← d_part + Σ_{i∈burst} (x_i − q_i)²
  d_est ← α_k · d_part / β_k                    # 估计全维距离
  if d_est ≥ threshold: return REJECT          # 更早触发早退
return ACCEPT
```
例子（论文 Fig.6b，SIFT 场景）：阈值 2.5，s2 在前 2 维算得 d_part²，d_est²=α₂·d_part²/β₂<threshold 继续；第 4 维后 d_est⁴≥threshold 即早退。对比朴素 EE（图 7）：朴素要算到第 109 维才触发，FEE-sPCA 第 4 维触发；对 d_all<threshold 的"应接受"向量，β 校正（黄色虚线）防止估计过高误触发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线 PCA 在 A100 GPU 上执行（BigANN 1B 约 430s、SIFT 1M 约 6.5s），α/β 表随索引存储；在线查询也需一次 PCA 变换（表 IV：0.1-0.8ms，占搜索延迟 0.1%-3.8%）。硬件上由 VPE 的 FEE 模块实现（按 burst 更新部分和并用 α/β 缩放比较，见硬件架构 VPE 条目）。开源（https://github.com/Intelligent-Computing-Research-Group/NasZip）中对应 preprocess_idx/ 的 PCA 预处理与 simulate/ 的搜索；离线过程仅在数据库更新达约 30% 时重跑。适用：高维向量检索（GIST/Wiki 收益最大），与 Dfloat 位级压缩正交组合。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## Dfloat（Dynamic Floating-Point Representation，动态浮点表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dfloat 是 NASZIP 的位级压缩技术：每个特征用可配置位宽 1+n_exp+n_man（符号位+自适应指数位+尾数位，∈[12,32] bit）的动态浮点表示，不同特征段采用不同位宽，使每个 DRAM burst（DDR5 每 device 128 bit）装入更多特征、减少访问向量所需的 burst 数。与传统 BF16/FP16/FP8 均匀量化的关键区别：FEE-sPCA 变换后各维贡献不均（低维承载更多信息），Dfloat 按段适配位宽、对敏感的低维段保留更多位（如 SIFT 128 维三段 18/14/16 bit），保持 recall 的同时最大化压缩。通用背景：动态浮点/自适应浮点（自适应指数/尾数位宽）在神经网络训练/量化中有成熟先例（Schrödinger's FP MLSys'24、AFP ICCV'21、"Be Like Water" ICML'22 等，Web 证据），本论文把该思路与 NDP 突发访问对齐并做设计空间搜索。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dfloat 配置搜索（Algorithm 1）：对可能的每向量 burst 数 N_burst ∈ [N_min=d/(B_burst/32), N_max=d/(B_burst/12)] 做二分 + 枚举验证，目标是 min N_burst 且 recall@k≥R_target；验证用 host CPU 位掩码模拟各配置精度损失（无需重建索引）。约束：同一 burst 内特征同格式；位宽随特征索引增大而递减；N_burst 须是每 sub-channel device 数的倍数（device 同步工作）；B_burst 依 DDR 代际（DDR5=128bit、DDR4=64bit）。伪代码：
```
N_burst_min ← d/(B_burst/32); N_burst_max ← d/(B_burst/12)
while N_burst_min < N_burst_max:
  N_burst ← ⌊(N_burst_min+N_burst_max)/2⌋
  C ← cfg-validate(N_burst)          # 枚举所有满足约束的 {n_exp,n_man} 分段配置
  for C_i in C: if R(C_i) ≥ R_target and R(C_i) > R(C_opt): C_opt ← C_i; N_burst_min ← N_burst
  else: N_burst_max ← N_burst
return C_opt
```
张量示例（SIFT 128 维，图 11）：段 1~42 / 43~74 / 75~128 分别 18/14/16 bit → 每段 6/4/6 个 burst，4 个 device 交叉并行，每个 burst 一次取 4 device×128bit。Dfloat 值进 FPU 前零填充回 FP32，因此不改变计算单元（可移植性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Dfloat 打包在离线预处理阶段完成，与具体浮点格式无关、可套在现有 FP 表示上；硬件解码由 VPE 的 Dfloat 处理模块完成（16-to-1 MUX 逐 cycle 装 burst 进 128-bit 寄存器 → barrel shifter 按偏移寄存器抽 n-bit 元素 → 零填充 FP32）。ECC 兼容性：不改 DDR5 die 结构，on-die ECC 与 side-band ECC、内存控制器 ECC 均不受影响。开源仓库提供 FEE-sPCA 与 Dfloat 算法源码及配置搜索脚本。适用：需在固定 DRAM 带宽下检索海量向量的场景（BigANN 1B 等），与 FEE-sPCA 正交叠加（Dfloat 额外 1.79× 距离延迟削减）。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## PQ（Product Quantization，乘积量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PQ 是把 D 维向量切分为 M 个子向量、对每个子空间分别做 K-means 聚类并用聚类中心索引（codeword）编码的向量压缩方法：每个向量用 M 个 code（每 code 约 log2 K bit）表示，距离用查找表近似（ADC：子空间距离查表求和），可把向量存储压缩到 1/8~1/64。它是 ANNS 的经典压缩 baseline，也是本论文内存流量对比的压缩基线之一（Fig.20）。局限：压缩引入明显精度损失，为保持高 recall 只能降低压缩比，导致比 RabitQ/NASZIP 高约 2× 的内存流量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PQ 的 pipeline（vault 笔记 ANNS 条目亦概述 IVF+量化 pipeline）：(1) 索引构建：把每个向量 x∈R^D 切为 M 个子向量 x^(j)∈R^(D/M)，对每个子空间独立跑 K-means 得 K 个中心，x 编码为 M 个 codeword；(2) 查询：对查询 q 计算每个子空间到 K 个中心的距离表，候选向量的近似距离 = Σ_j Table[j][code_j(x)]。伪代码：
```
# 离线：per-subspace 聚类
for j in 1..M: C_j ← KMeans({x_i^(j)})   # K 中心
    code(x) ← [argmin_k ||x^(1)−C_1[k]||, ..., argmin_k ||x^(M)−C_M[k]||]
# 在线：近似距离
d_est(x,q) ← Σ_j T_j[code_j(x)],  T_j[k] ← ||q^(j) − C_j[k]||²
```
论文用法：作为压缩内存流量 baseline（HNSW on PQ 编码），recall@10≥90% 时流量约为 RabitQ/NASZIP 的 2 倍。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FAISS（IndexPQ/IVFPQ）、ScaNN 等主流库均内置；参数为 M（子空间数）与 K（每子空间中心数）。使用：先训练码本（采样向量聚类）→ 编码向量 → 查询用查表近似距离，必要时精排（re-rank 用原向量）。局限（论文指出）：PQ 主打压缩、精度损失大，高 recall 下需弱压缩，内存流量反而更高；且不直接匹配 NDP 的 burst 访问模式。论文以其为对照，说明 FEE-sPCA+Dfloat 在同等 recall 下流量更低。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## RabitQ（Quantizing High-Dimensional Vectors with a Theoretical Error Bound）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RabitQ 是一种带理论误差界的向量量化方法，用于 ANN 候选过滤：对高维向量做有理论保证的紧凑量化，使量化后的距离估计有可证明的误差上界，从而在候选过滤（filter）阶段安全地淘汰大部分非候选、且不损失精确性。它是论文内存流量对比的另一个压缩 baseline（vault 笔记 ANN 条目：/data3/paper_analysis/knowledge_notes/算法知识笔记/Approximate Nearest Neighbor (ANN) Search（近似最近邻搜索）.md 引用该论文）。局限（论文 Fig.20 分析）：RabitQ 加速候选过滤，但幸存候选仍需全维精确距离做 re-rank，因此内存流量仍高于 FEE-sPCA+Dfloat（后者通过特征级早退直接砍掉访问维度数、Dfloat 再压位宽）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RabitQ 的 pipeline：离线对向量量化（带误差界），在线查询时对每个候选先算量化近似距离 d_q(x,q)（紧凑读入）→ 结合误差界判定：d_q − bound ≥ threshold 则安全淘汰（无需读原向量）→ 幸存少数候选读原向量做精确距离与精排。伪代码（过滤阶段）：
```
for x in candidates:
  d_q ← dist_quantized(x, q)            # 仅读量化码（紧凑）
  if d_q − err_bound(x) ≥ threshold: skip(x)   # 理论保证可安全淘汰
  else: d_all ← dist_exact(x, q)       # 幸存者读原向量精排
```
对比：RabitQ 的淘汰决策基于"量化码+误差界"但仍需为幸存者读全维原向量；FEE-sPCA 用 PCA 估计距离在部分维度内就完成淘汰、Dfloat 进一步减少每次读取的位数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RabitQ 以其理论误差界为核心设计量化器（基于随机量化的低偏差估计），可用于 FAISS 风格索引；论文将其配置到 HNSW 上做 re-ranking 前的过滤。使用场景：需要保证 recall 的候选过滤、与图/IVF 索引结合。局限（论文角度）：过滤后仍需全维精排 → 流量高；本论文的 FEE-sPCA 在 NDP 上与之兼容且流量更低（Fig.20，与 PQ 归一化对比）。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

## 神经渲染（Neural Rendering / Neural Radiance Field，神经辐射场）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 神经渲染指用神经网络/可学习表示替代传统显式几何（网格、点云、体素）来做 3D 场景表示与图像合成的技术族（参考 Tewari et al., "Advances in neural rendering", CGF 2022）。按 NeRArch-Sim 论文的分类，主流神经渲染 pipeline 分三类：(1) MLP-based（如 NeRF [Mildenhall 2020]）——用 MLP 隐式建模辐射场，低内存占用；(2) Grid-based（如 Instant-NGP 多分辨率哈希编码、voxel grid）——用离散化空间结构存预计算场景特征，渲染质量高；(3) Primitive-based（如 3D Gaussian Splatting）——用显式几何原语（三角形/3D 高斯）走光栅化，渲染速度快，可视为"零层网络"。此外还有 hybrid（如 grid+3DGS 混合）管线。NeRArch-Sim 论文把神经渲染 pipeline 统一分解为四个阶段：Field Sampler（沿相机射线采样 3D 点/做视锥剔除）→ Encoding（位置编码，RFF/哈希编码）→ Field Computation（MLP/球谐算颜色密度）→ Blending（体积渲染/排序+alpha 混合）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 NeRF 推理 pipeline 为例（MLP-based）：`for pixel: 发射相机射线 r → 沿射线采样 N 个 3D 点 {x_i} → RFF(x_i) 位置编码 → MLP 输出 (σ_i, c_i)（密度+颜色）→ 体积渲染积分 C(r) = Σ T_i·α_i·c_i → 得像素颜色`。NeRArch-Sim 中同一 pipeline 用统一算子接口搭建：`g = OperatorGraph(); s = UniformSampler(dim, graph=g); e = HashEncoding(dim, num_levels=16, graph=g); m = MLP(dim, in_dim=e.out_dim, num_layers=4, graph=g)`，即把采样/编码/MLP 映射为分类学算子，再由插桩框架自动提取成算子图供硬件模拟。MLP/grid/primitive 三种 pipeline 在 NeRArch-Sim 中被建模成不同算子图（vanilla-nerf、instant-ngp、splatfacto 三种 Nerfstudio 模型），是"跨 pipeline 公平基准"的算法侧基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现层面：算法框架用 Nerfstudio（NeRF 开发框架）、GauStudio、Kaolin Wisp 等；NeRArch-Sim 利用与这些框架分类学相似性，通过运行时钩子插桩 Nerfstudio（checkout 指定 commit，注入 tracing.py/eval.py）用 `ns-eval --enable-trace` 渲染一帧输出 execution_dag.pkl 算子图。加速器实现方面，NeRArch-Sim 复现的 ICARUS（NeRF 专用）、NeuRex（Instant-NGP 类）、CICERO、SRender、GSCore、GS Processor、GBU、Uni-Render 等即各类 pipeline 的硬件实现，用 SystemC/Catapult HLS 按同一分类学模块化建模。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators

## 体积渲染（Volume Rendering，体绘制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 体积渲染是沿视线方向对采样点按光学模型（吸收/发射）累积颜色与透射率的渲染技术（经典参考 Max, "Optical models for direct volume rendering" 2002）。NeRF/Instant-NGP 等 MLP/grid-based 神经渲染用其把每点预测的密度 σ 与颜色 c 合成像素颜色：C(r) = Σ_{i} T_i (1−exp(−σ_i δ_i)) c_i，T_i = exp(−Σ_{j<i} σ_j δ_j)，δ_i 为采样间距。NeRArch-Sim 论文把 Blending 阶段定义为"聚合场景属性产生最终像素颜色"，其中 MLP-/grid-based 管线用体积渲染，primitive-based（3DGS）用排序后 alpha 混合。3DGS 论文说明其 α-blending 即 NeRF 式体积模型（复用经典图形学 alpha 术语）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 体积渲染在 NeRArch-Sim 中是 Blending 分类学阶段的核心算子，硬件库有 3 个 VRU（Volume Rendering Unit）变体（分别对应 ICARUS/CICERO/GSCore 的实现），表 VI 给出 ICARUS 的 VRU 延迟 192/192 cycle、面积 4755/4960 µm²、功率 1917/2110 µW（NeRArch-Sim vs 全 ASIC flow）。伪代码：
```
C, T = 0, 1
for i in sorted_samples:            # 沿射线从前到后
    alpha_i = 1 - exp(-sigma_i * delta_i)
    C += T * alpha_i * c_i          # 累加颜色
    T *= 1 - alpha_i                # 更新透射率
    if T < eps: break               # 提前终止（early termination）
```
- 内存侧（NeRArch-Sim 表 VIII）：ICARUS/NeuRex 在 Field Sampler 阶段做 ray marching，采样坐标从 DRAM 经 Input FIFO 流入；ICARUS 每帧 1.4GB 采样坐标、权重 1.9MB 只加载一次复用约 10 万次、激活值片上 ping-pong。GSCore 不做 ray marching，每 tile 从 DRAM 流式读 79MB 高斯特征，排序与光栅化中间量全片上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：Nerfstudio 的 nerfstudio field 组件逐点预测后做体积渲染积分；硬件实现：专用 VRU（如 ICARUS 的 volume rendering unit、CICERO 的 NRU、GSCore 的 VRU），NeRArch-Sim 硬件库含 VRU_v1/v2/v3 变体并支持配置累计方式（accumulation_type）。算子级优化上，"per-ray early termination（基于累计不透明度）"是体积渲染特有的 element-level skip 优化（NeRArch-Sim 调度优化库）；相关硬件研究如 VR-Pipe（HPCA 2025）专门流水化硬件图形管线做体积渲染。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators

## 射线行进（Ray Marching，射线步进 / 射线采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Ray marching 是沿相机发出的射线按一定步长/采样策略离散地探测 3D 场景（采样点、求交）的渲染技术，是 NeRF/Instant-NGP 等隐式神经渲染的核心采样步骤，区别于传统显式 ray tracing 的 BVH 求交。NeRArch-Sim 论文把 Field Sampler 定义为"沿相机射线采样物体（3D 点）定义感兴趣区域"，MLP-/grid-based 管线用 uniform 或 PDF-based 采样，primitive-based（3DGS）则用视锥剔除（frustum culling）丢弃目标 2D 区域外的原语。论文指出通用 NN 加速器模拟器（Timeloop/SCALE-Sim 类）不支持 ray marching、spatial sampling 等图形学专用算子，是构建神经渲染专用模拟器的动机之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 NeRArch-Sim 中 ray marching 是 Field Sampler 阶段算子，且与硬件调度直接耦合：表 VIII 显示 ICARUS 与 NeuRex 都在 Field Sampler 阶段做 ray marching——每帧把射线采样坐标从 DRAM 流式读入（ICARUS 1.4GB、NeuRex 469MB position streaming），体现"采样→编码→MLP"的流水。伪代码：
```
for pixel (u,v):
    r = camera.generate_ray(u, v)         # 生成射线
    for i in range(num_samples):           # ray marching
        x_i = r.sample(depth_i, strategy)  # uniform/PDF 采样 3D 点
        features.append(encode(x_i))       # 交给 Encoding 阶段
```
- 采样策略差异：NeRArch-Sim 分类学里 Sampling 阶段参数含 num_samples 与 sampling_strategy（uniform / PDF / frustum culling）。算子级优化中，"restricted hashing（在 subgrid 内处理射线）"（NeuRex）是 region-level reuse 优化、与 ray marching 的采样组织相关。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：Nerfstudio/Instant-NGP 的 ray sampler（uniform、PDF 采样器）；硬件：专用 sampling unit、Culling & Conversion Unit（GSCore）等。NeRArch-Sim 硬件库 Sampling 类含 Culling conversion unit、Skipping controller、Sampling unit。GauTracer 论文则是把 ray tracing 硬件（RTA）扩展用于高斯表示——ray-gen shader 沿像素发射光线、BVH 遍历（与之对应的是显式 ray tracing 而非 ray marching）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators

## 哈希编码（Hash Encoding，多分辨率哈希编码 / Instant-NGP 式哈希网格）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hash encoding（Müller et al., "Instant neural graphics primitives with a multiresolution hash encoding", SIGGRAPH 2022）用多分辨率哈希表存场景特征：每个采样 3D 点按 L 层分辨率量化到网格顶点，用空间哈希函数查表得到各层特征向量并线性插值，拼接到 MLP 输入。相比 RFF 或 dense voxel grid，它用固定大小哈希表覆盖无限细节、训练快（秒级）内存小，是 Instant-NGP 类 grid-based pipeline 的核心。NeRArch-Sim 论文把 Encoding 阶段定义为"把采样位置转成特征向量"，hash encoding 与 RFF 是 MLP-/grid-based 管线的代表编码；其在 NeRArch-Sim 中作为分类学算子（HashEncoding，参数 num_levels、hash_table_size、feature_dim），并配套专用硬件（hash 地址生成、查找单元）。注意与本仓库中"HashEncode for LLM Attention（HATA）"、"Hash Encoding of Operator Fusion Schemes（STOF）"是不同上下文里的同名概念，勿混淆。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- NeRArch-Sim 用 NeuRex pipeline 作例子：`e = HashEncoding(dim, num_levels=16, graph=g)`，随后 MLP 输入维度 = e.out_dim。伪代码：
```
# 对采样点 x（3D）
feat = []
for l in 1..L:                       # 16 层分辨率
    v = quantize(x, res_l)           # 按层分辨率量化到整数网格
    hash_idx = hash(v) % table_size  # 空间哈希查表
    feat_l = trilinear_interp(table_l, v)   # 表内插值
    feat.append(feat_l)
h = concat(feat)                     # 拼成 MLP 输入特征
```
- 硬件/内存视角（NeRArch-Sim 表 VIII）：NeuRex 的 DRAM 流量由 position streaming（469MB）与 hash subtable loading（16MB，细分辨率层的不规则查找）主导；Grid Cache（64KB，DRAM Rd 9.8GB）与 Subgrid Buffer（128KB，DRAM Rd 16MB）服务哈希查找；NeuRex 的 Subgrid Buffer 因细分辨率层近似随机哈希查找产生最高 bank conflict 开销（表 IX：conflicts 7.7M、stalls 384K、overhead 2.25%）。算子级优化"restricted hashing（在 subgrid 边界内处理射线）"是 NeuRex 的 region-level reuse 优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：tiny-cuda-nn/Instant-NGP 的 `MultiResHashEncoding`（NeRArch-Sim 环境依赖 tiny-cuda-nn）；Nerfstudio 的 instant-ngp 模型即此编码。硬件：NeRArch-Sim 的 Encoding 硬件库含 Address generator、Tree reducer、Index generation/computation unit 等（对应 NeuRex/CICERO/SRender 的哈希相关单元）；SRender 复用 Hash Index Generators、新增 Point Rearrangement/Distance Compute/Comparison units。哈希表大小、分辨率层数、特征维度均可配置（hash_table_size、num_levels、feature_dim）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators

## 随机傅里叶特征（Random Fourier Features，RFF / 位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Random Fourier Features（Rahimi & Recht 2007）把输入映射到随机傅里叶特征空间以近似核函数；NeRF（Mildenhall 2020）采用其思想作为位置编码：对 3D 采样点 x 按不同频率的正弦/余弦（高频分量）升维，使 MLP 能表示高频细节。NeRArch-Sim 论文明确 MLP-based 方法（NeRF 类）"sample 3D points along camera rays, encode them via Random Fourier Features (RFF)，再 query MLP 预测密度与颜色"，并指出 RFF 与 hash encoding 是 MLP-/grid-based 管线的代表 Encoding 方案（primitive-based 如 3DGS 跳过此阶段，原语自带场景属性）。注意本仓库中"Random Feature Attention（RFA，Mamba 线性注意力用 RFF 近似 softmax 核）"是同一数学工具在注意力上的不同应用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- RFF 在 NeRArch-Sim 中属于 Encoding 分类学阶段，与 Position encoding unit（ICARUS 用）等价。伪代码：
```
def rff(x):                            # x ∈ R^3
    return [sin(2^0 π B x), cos(2^0 π B x),
            sin(2^1 π B x), cos(2^1 π B x), ...]   # 多频段
# 之后: MLP(concat(x, rff(x))) → (σ, c)
```
- 硬件侧（NeRArch-Sim 表 VI）：ICARUS 的 Pos Encoding Unit（PEU）延迟 130/130 cycle、面积 6714/5200 µm²、功率 305/330 µW（NeRArch-Sim vs 全 ASIC flow）；图 12 显示 ICARUS 中 PEU 利用率持续高企、MLP 是主导瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：Nerfstudio 的 vanilla-nerf 模型（`ns-eval` 可跑）与 Nerfstudio 的 `PositionEncoding`；实现通常用预计算正弦/余弦表或即时三角函数。硬件：ICARUS 的 Position Encoding Unit、NeRArch-Sim Encoding 硬件库中的 Position encoding unit，支持 CORDIC/分段线性等 exp/三角实现选择（表 IV：Implementation = CORDIC, piecewise linear）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators

## 球谐函数（Spherical Harmonics，SH）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Spherical Harmonics 是球面上的正交基函数族，图形学里用它紧凑编码方向相关量（如随视角变化的颜色/光照）。NeRArch-Sim 论文指出 Field Computation 阶段（"基于编码特征计算采样物体的场景属性（颜色/密度）"）"typically using MLPs or Spherical Harmonics"——3DGS 等 primitive-based 管线用 SH 系数存高斯的方向相关颜色。3DGS 论文参数化中：位置 3 + 旋转四元数 4 + 不透明度 1 + SH 系数 16×3=48 参数（三阶 SH），即每个 3D 高斯的外观由 SH 系数决定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 3DGS 渲染中，像素颜色按视角方向 d 从 SH 基重构：`color(d) = Σ_{l=0}^{L-1} Σ_{m=-l}^{l} c_{lm} · Y_{lm}(d)`，其中 Y_{lm} 为 SH 基、c_{lm} 为学习到的系数（3DGS 默认 3 阶 16 组 × RGB 3 通道 = 48 参数）。NeRArch-Sim 中此计算是 Field Computation/Blending 阶段的颜色相关算子，其分类学参数含 encoding_type 等；硬件侧 GSCore/GS Processor 的光栅化/混合路径需访问高斯外观（GauTracer 论文把 SH/颜色存为纹理数据经 Gaussian ID 访问）。NeRArch-Sim 表 XII 显示 GS Processor 有 Feature Computing Unit（计算高斯特征），即此类方向相关颜色计算硬件。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：3DGS（graphdeco-inria/gaussian-splatting）用 SH 系数 + CUDA 光栅化 kernel 计算视角相关颜色；Nerfstudio 的 splatfacto 模型即 3DGS 类（NeRArch-Sim 支持其 trace）。硬件：NeRArch-Sim 硬件库 Field Comp/Blending 类含 MLP 引擎与加法树等，可建模 SH 求值；SRender 的 interpolation units 等处理方向相关插值。SH 阶数（degree）是可配置精度参数，与"low bit 优化"（SRender 的 sensitivity-aware 动态精度）相关。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 加速器补充视角（ISCA'26，SH 作为 GS 特征存储与加速器访存）：本论文把投影后 RGB（SH 求值结果）作为 9 参数 GS 特征之一存入 GS-feature cache（每 cache line 28-bit GS ID tag + 4-bit 记录该 GS 相交 tile 数，32-bit 对齐），光栅化时经广播寄存器广播 16 次供同线 PE 的 α-blending 使用；59 参数中 SH 占 48（16×3），是参数占比最大的部分。SH 求值在 GPU 侧 projection 阶段完成，加速器不重复计算——体现"投影留 GPU、光栅化/混合上专用阵列"的分工。

## Pauli 乘积旋转（Pauli Product Rotation，P_θ）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pauli 乘积旋转是 Pauli-Based Computation（PBC）的基本计算原语（Litinski "A Game of Surface Codes", Quantum 2019）：$P_\theta = \exp(-iP\theta)$，其中 P 是多 qubit Pauli 算子（X/Y/Z 的 tensor 积），θ 为旋转角。Clifford+T 门集的元素可表示为 P_θ 的特例：S=Z_{π/4}、T=Z_{π/8}，标准分解 H=Z_{π/4}X_{π/4}Z_{π/4}、CNOT=(Z⊗X)_{π/4}(I⊗X)_{−π/4}(Z⊗I)_{−π/4}。化简规则：若 Pauli 算子 P 与 P' 交换（PP'−P'P=0），P_{π/4} 可越过 P'_θ；若反对易，P'_θ 变成 (iPP')_θ。Clifford 门把 Pauli 映射到 Pauli，可吸收进最终测量。执行方式：非 Clifford 旋转（如 π/8）经 magic state teleportation 消费 |T⟩ 态实现；π/4 与 π/8 Pauli 乘积测量用标准 gate teleportation 协议（Litinski [34]）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- P_θ 在 O3LS 编译流水线中的位置：QASM → ①Clifford+T 分解（GridSynth，容限 10⁻⁵）→ ②Pauli-based transpilation 生成 PPR 序列（P_θ 及其化简）→ ③表面码映射与调度。执行 P_θ 的具体过程（以 π/8 旋转为例）：目标 Pauli 串（如 Z_0Z_1Z_2）→ 初始化 ancilla patch → 与 magic state |T⟩ 做 gate teleportation（π/4 与 π/8 PPM 按 [34] 协议）→ 若测量结果为 1 则补条件校正 exp(iπ/4·Z_0Z_1Z_2)。Y 算子（Y^{⊗N} 旋转）因 X/Z 不能同时访问需先经 Y-synthesis 分解为 X/Z 组合（偶数个 Y 时二分分组、吸收抵消）。示例：$(Y^{\otimes N})_{\pi/8}$ 分解为 $[(Z^{\otimes n})_{\pi/4}\otimes(Z^{\otimes N-n})_{\pi/4}](X^{\otimes N})_{-\pi/8}[(Z^{\otimes n})_{-\pi/4}\otimes(Z^{\otimes N-n})_{-\pi/4}]$（n 与 N−n 均为奇数），选可抵消的分组。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器产物（PPR 序列）与物理协议两层：编译侧用 Pauli DAG（O3LS-IR）表达依赖与并行、用交换/反对易规则化简；物理侧以 lattice surgery PPM + magic state teleportation 执行，旋转角决定是否消费 magic state。评估：STIM（d=9、p=10⁻³）表征 PPM/PR/measurement 错误率，分层 LER 模型累加。工具：Qiskit LitinskiTransformation、PennyLane Pauli Product Rotations 编译插件（cite 论文）实现 PBC 到 PPR/PPM 的转译。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## 非均匀量化与学习码本（NU-WAQ / K-Means 码本量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
非均匀量化（non-uniform quantization，NU-WAQ 指权重-激活双侧非均匀量化）用不等间隔的量化电平（centroids/质心）拟合数据真实分布，与均匀量化（固定 scale+zero-point 的等间隔整数映射）相对。学习码本（learned-codebook）方法通过聚类/训练优化质心，代表即 K-Means 量化（论文公式1）：x̃_i = C_{idx_i}，idx_i = argmin_k ‖x_i − C_k‖²——即用 n-bit 索引矩阵 + 2^n 个 FP 质心码本表示数据，重建靠查表。因为质心可贴合 LLM 权重/激活的重尾+离群分布，NU-WAQ 在低比特下精度显著优于均匀方案（SqueezeLLM 3-bit LLaMA-7B PPL 6.32 vs GPTQ 7.55）。OASIS（§III-A）具体化：权重 4-bit 采用输出通道级量化（整矩阵共享质心 + 每输出通道独立缩放因子，无 outlier 保护）；激活 3/4-bit 采用 token 级量化（每 token 独立质心与缩放因子），激活质心用 C4 数据集 16 个校准样本、经 Fisher 信息矩阵加权的 K-Means 离线学习，在线只做聚类分配（offline/online 质心 RMSE 仅 0.01，图5，验证离线学习可行性）。其余参考：低精度浮点格式（MXFP4、NVFP4）也属非均匀量化但非学习码本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线量化（OASIS §III-A）
W_c, W_idx = kmeans(W, k=2^4)          # 4-bit 权重：16 质心 + 索引矩阵
A_c = fisher_weighted_kmeans(calib_acts(C4, 16 samples), k=2^4)  # 激活质心
# 在线推理（每 token）
idx = argmin_k ||x - A_c[k]||^2         # 聚类分配（OASIS 用 Clustering Unit 硬件）
x̃ = A_c[idx]                            # 重建（查码本）
```
压缩比：n-bit 时索引矩阵 K×N×n bit + 码本 K×2^n×16 bit（权重）或每 token 码本；token 级激活量化把量化参数动态化——这是其精度高但需在线聚类开销的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表开源实现：SqueezeLLM（github.com/SqueezeAILab/SqueezeLLM，3/4-bit K-Means + dense-sparse 分解）、Any-Precision LLM（多精度分裂质心）、Bitsandbytes（NF4）、SpQR（稀疏+非均匀混合）。OASIS 本身无公开代码（arXiv:2507.23035 无 Code 链接）。使用场景：超低位宽（≤4-bit）权重/激活压缩；注意 NU-WAQ 索引格式与现有 INT 低精度计算单元不兼容——传统执行需反量化为 FP16 再 GEMM，OASIS 用 LUT-GEMM 直接计算（见 WAQ LUT-GEMM 条目）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration

## 权重-激活量化（WAQ）与权重-only 量化（WOQ）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WOQ（weight-only quantization）只量化权重、激活保持 FP16（W4A16 等），推理时需把权重反量化回 FP16 再做 GEMM——反量化开销可占 GEMM 时间的 20-90%（OASIS §I 引用 [25][30]），且激活仍占内存与带宽、无法利用低精度计算单元。WAQ（weight-activation quantization，双测量化）同时量化权重与激活（W4A4/W8A8）：可全低精度 GEMM（INT4×INT4）、压缩权重与 KV-cache 内存、消除混合格式计算。WAQ 内部两条路线：INT-WAQ（整数等距量化，可被现有低精度硬件直接执行，但表示能力有限、低比特精度差）；NU-WAQ（非均匀码本量化，精度高但索引格式与现有计算单元不兼容，传统执行需反量化回 FP16 再 GEMM，计算优势被抵消）。OASIS 定位即解决"高效低精度 INT-WAQ vs 高精度低效率 NU-WAQ"的两难（图1）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# WOQ (W4A16)：推理每层
W_deq = dequant(W_idx, W_c, per_channel_scales)  # 反量化（开销 20-90%）
Y = W_deq @ X_fp16                               # FP16 GEMM
# INT-WAQ (W4A4)：直接用低精度单元
Y = INT4_GEMM(W_int4, X_int4)                    # Tensor Core / 加速器
# NU-WAQ 传统执行：查码本反量化 + FP16 GEMM
X_deq = C_A[X_idx]; W_deq = C_W[W_idx]
Y = X_deq @ W_deq                                # 反量化抵消量化收益
# OASIS（NU-WAQ 高效版）：WAQ LUT-GEMM 直接算（见下条）
Y = LUT_GEMM(X_idx, W_idx)
```
关键点：WOQ 精度好但 dequant 主导耗时；INT-WAQ 高效但精度差；NU-WAQ 精度好但计算效率差——OASIS 用预计算 Cartesian Product LUT 打破最后一条。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WOQ 代表：GPTQ/AWQ/SpQR（W4A16）；INT-WAQ 代表：SmoothQuant（W8A8）、QuaRot、Atom（group-128 W4A4）；NU-WAQ 代表：SqueezeLLM、K-Means 码本。部署：TensorRT-LLM、MLC-LLM 支持 W4A4 推理；GPU INT4 Tensor Core 支持有限（INT8 更成熟），实际加速低于理论。OASIS 在算法精度上对比 INT-WAQ baselines（RTN/SmoothQuant/QuaRot/Atom），W4A4 下平均 accuracy drop 仅 1.94%、比 Atom 低 6.34%（论文表 III/IV，A100-80GB 上 Transformers+PyTorch+lm-eval-harness 评测，模型 OPT/LLaMA/LLaMA-2/LLaMA-3/Mistral 共 11 个）。

- SMOOTH 模型配置（ISCA'26）：评估的 8 个模型（TinyLLaMA 1.1B、GPT-Neo/GPT-3 XL 1.3B、Gemma-2 2.0B、GPT-3 2.7B、LLaMA2 7.0B、Bloom 7.1B、GPT-3 13B）全部采用 w4a8/int8 权重-激活量化格式，批量 1，对齐移动端部署；SMOOTH 不做任何模型级改动（无精度损失），其 block 级内存管理可正交叠加在 w4a8/int8 等量化方案之上进一步加速。
涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference

## WAQ LUT-GEMM（笛卡尔积 LUT 查表矩阵乘法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-GEMM 用查找表直接完成量化 GEMM 而非逐元素 MAC。现有 WOQ LUT-GEMM（FIGLUT [42]、LUT Tensor Core [37]、LUT-GEMM [43]）把"权重索引 → 组内 inner-product 结果"存入 LUT，μ 个权重 bit 作索引，运行时查表得部分和再跨组累加；缺陷：(1) LUT 依赖流式激活必须 on-the-fly 生成；(2) inner-product LUT 大小 2^μ·(K/μ)，K 大（LLaMA-7B q_proj K=4096）时爆炸，只能小 group（μ=4）抑制；(3) 跨 group partial-sum 增加 FLOPs、并行度受限。OASIS 的 WAQ LUT-GEMM（§III-B）利用 WAQ 三大机会：权重与激活质心均离线学习 → Cartesian Product LUT 可离线预计算（消除 on-the-fly 生成）；双操作数量化后可能乘积仅 2^(nW+nA) 项（W4A4 为 256）→ 存 Cartesian Product 而非 inner product，LUT 相对 4096×4096 层小 64×；Cartesian Product LUT 与归约长度 K 无关 → 归约粒度可达整层 K（group size 相对 μ=4 提高 1024×），归约 FLOPs 降 16×。与知识库现有"LUT 查找表量化（非解析量化 / Codebook-based Lookup）"条目的区别：后者（EVA 视角）是 weight-only 的查码本解码/输出码本 GEMM，本条目是双测量化下以拼接索引计数的 Cartesian Product 查表计算方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WAQ LUT-GEMM 计算过程（M=1, K=6, N=4, nW=nA=1 示例，论文 Fig.6）：
```
# 离线：LUT[j] = C_W[j>>nA] * C_A[j & (2^nA-1)]，共 2^(nW+nA) 项
# 在线：
# (1) 拼接：concat_idx[k,n] = (idx_A[k] << nA) | idx_W[k,n]
# (2) 计数：count[j] = Σ_k 1{concat_idx[k,n] == j}
# (3) 加权和：Y[n] = Σ_j count[j] * LUT[j]
```
FP16 加法次数从 K 降到 2^(nW+nA)（K=4096 → 256）；与 WOQ LUT-GEMM 的对比见论文 Table I（LUT 大小 2^(nW+nA) vs 2^μ·K/μ；归约 FLOPs 2^(nW+nA)·N vs K/μ·nW·N）。消融（论文 Fig.16，q_proj 层）：OASIS-A4 相对 FIGLUT/LUT Tensor Core 平均降 LUT 大小 62.1×、归约 FLOPs 497.1×；相对 LUT-GEMM 降 994.2×/248.6×；模型越大（K 从 4096 到 26728）WOQ 方案 LUT 爆炸而 OASIS 恒定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WOQ 侧 LUT-GEMM 开源：github.com/naver-aics/lut-gemm（BCQ 格式 GPU kernel）、FLUTE（MIT CUDA 库，LUT 向量化+跨 bank 复制消冲突）、T-MAC（CPU in-register LUT）、FIGLUT（HPCA 2025，RAC 单元硬件）。OASIS 无公开代码；其硬件实现见知识库硬件架构的 Concat Unit / Index Counter / MAC Tree 条目（OASIS 加速器 2KB Cartesian Product LUT、16 PE Line 流水执行）。使用场景：让 NU-WAQ 无需反量化直接高效 GEMM，尤其 decode（batch=1）等 memory-bound 与 prefill 等 compute-intensive 场景（Concat Unit 极简面积设计兼顾两者）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration

## 激活 Outlier 与 Outlier-aware 量化（动态 top-k 保留 + 误差补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
激活 outlier 指 LLM 中间激活中少数元素/通道幅度系统性远超其余（某些通道大 20-100×、集中在 ~0.1% 通道）的现象，它扩大量化范围、降低 inlier 的有效位分辨率，是激活量化的主要误差源。处理策略谱系：(1) 混合精度保留（Atom/LLM.int8()：outlier 保持 FP16/INT8）；(2) 等价变换迁移（SmoothQuant 缩放到权重）；(3) 旋转消除（QuaRot Hadamard）；(4) 动态 top-k 检测（KVQuant、OASIS）。OASIS（§II-C/§III-A）选动态识别每 token top-0.5% 最大 + bottom-0.5% 最小激活保留 FP16：因为 offline/online 的 upper outlier 阈值 RMSE 高达 0.32-0.38（图3），静态 outlier channel 识别不准确；动态检测精度更高（KVQuant 结论）。OASIS-S 变体复用离线校准阈值（省 Orizuru 硬件但精度略低，W4A4 下 PPL 高 0.05）。outlier 数量敏感性（论文 Fig.15）：0.5%→1% 吞吐几乎无损（主分支主导），1%→10% outlier 分支成新瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# OASIS outlier-aware 量化流程（每 token 激活 x∈R^N）
mask = topk(|x|, k=0.005N) ∪ bottomk(|x|, k=0.005N)   # Orizuru 检测
x_in = quantize(x[¬mask], A_c)                        # NU4 量化 inlier
x_out = x[mask]                                       # FP16 保留 outlier
# 计算：Y = LUT_GEMM(x_in, W) + Σ_out (x_out - C_A[idx]) * W_deq   # 主分支+补偿分支
```
对比常规动态检测（图4a）：先扫描全向量分 inlier/outlier 再分别 GEMM，检测在关键路径；OASIS 的 look-ahead 双分支把检测并行化（见下条）。tie 处理：FP16 激活存在相等值（约 2% token），确定性选左孩子保证每 token 恰输出 k 个 max + k 个 min。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现即 top-k 排序/选择（PyTorch topk、分块扫描）；硬件实现 OASIS 用 Orizuru 双折叠二叉树引擎（见知识库硬件架构条目，1.5N+2k·log2(N) 次比较 vs SpAtten 引擎 6N）。部署要点：outlier 比例是精度-吞吐旋钮（论文在 0.5%-10% 扫描）；outlier 通道需在权重侧取对应通道做反量化补偿，每 cycle 只取一个通道（论文 §III-C）避免稀疏 GEMM 与多 MAC 开销。评测（论文表 III/IV）：OASIS W4A4 相对 FP16 平均 accuracy drop 2.05%（PPL）/1.94%（zero-shot avg），优于 Atom/QuaRot/SmoothQuant。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration

## Look-ahead 计算与误差补偿（outlier 关键路径隐藏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Look-ahead 计算与误差补偿是 OASIS 把动态 outlier 检测从 GEMM 关键路径上移除的方案（§III-C，图4b/图7）：主分支（main branch）先对"整个激活向量（含 outlier）"做聚类量化并执行 WAQ LUT-GEMM，暂时忽略 outlier 的量化误差（look-ahead）；outlier 分支（outlier branch）并行地由 Orizuru 检测 outlier、按通道索引取权重反量化、计算残差 (x_out − C_A[idx]) 并乘加生成误差补偿项；最终 Y = Y*（look-ahead）+ Y'（补偿），与"先检测再分 inlier/outlier 各做 GEMM"的常规动态检测数学等价，但检测延迟被并行隐藏。论文消融：相对常规设计（OASIS-C）吞吐高 16%（W4A4）/18%（W4A3）于 LLaMA-2-7B。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 主分支（并行）：全激活聚类 + look-ahead LUT-GEMM
A_idx = cluster(x)                 # 全部激活聚类（含 outlier，暂时容忍其误差）
Y* = LUT_GEMM(A_idx, W_idx)        # WAQ LUT-GEMM
# outlier 分支（并行）：
for i in 1..k:
    (v_i, ch_i) = Orizuru_pop(x)   # 每 cycle 顺序输出一个 outlier 及其通道索引
    w_i = dequant(idx_W[ch_i], C_W)   # 取该通道权重索引反量化
    r_i = v_i - C_A[idx_A[ch_i]]      # 残差 = 原始 FP16 - 量化值
    Y' += r_i * w_i                   # 误差补偿（每 cycle 1 个 MAC 通道）
Y = Y* + Y'
```
硬件关键点：outlier 分支每 cycle 只处理一个通道 → 无需稀疏 GEMM 表示、MAC 单元数少（每 PE Line 8 个 FP16 MAC）；Memory Controller 对双分支流水调度（论文 Fig.14：1-4096-4096 W4A4 1% outlier 各步骤 cycle 数，outlier 分支约快 33%，先完成并写 Output Buffer 等主分支）。Memory/energy 分解（Fig.18）：Weight Index Buffer 占内存流量 76.0%、LUT 占 19.2%；能耗主要来自归约 33.1% 与分支合并 22.1%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
算法层面即"先算近似值、再并行修正"的分解计算模式，与 FlashAttention 在线 softmax 修正、GPTQ 误差补偿同属一类"先 look-ahead 后补偿"思想，但 OASIS 把它用在量化 outlier 上并做成硬件双分支。实现依赖：(1) Orizuru 实时检测引擎（见硬件架构条目）；(2) 主/outlier 分支延迟匹配（论文按 1% outlier 调硬件配置使双分支延迟相当）；(3) outlier 比例作为旋钮——≤1% 时不构成瓶颈，>1% 时 outlier 分支主导端到端延迟。该模式对"动态稀疏/异常值检测开销大"的推理加速器设计有普适参考价值。OASIS 无公开代码。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration

## BCQ（Binary-Coding Quantization，二值编码量化）与 BC-UQ（Binary-Coding Uniform Quantization，二值编码均匀量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BCQ（Binary-Coding Quantization）是一种非均匀量化方法（源自 [70] Xu et al. 2018 的 alternating multi-bit quantization，以及 ARB-LLM 等二值化工作）：把全精度向量 W 分解为高精度缩放因子向量 a 与二值编码矩阵 B∈{-1,+1}，重建为 W_i ≈ Σ_{j=1}^k B_{ij}·a_j（k 个二值基、B_ij 为二值系数、a_j 为缩放因子）。它是"非均匀"的——量化等级由数据分布自适应分配，对含 outlier 的数据比均匀量化误差更小。Omni-LUT（ISCA 2026）为兼容 LUT-based GEMM 加速器，把 BCQ 扩展为带 offset 与结构化 power-of-2 scaler 的变体：不学习缩放因子而是用固定 power-of-2 基 {α_u·2^{-1}, α_u·2^0, ..., α_u·2^{b-2}} 整体乘以 uniform scaler α_u，并加 zero-point z_bcq=z_u−(2^b−1)/2——即 BC-UQ（Binary-Coding Uniform Quantization），把均匀量化表达成 binary-coding 格式。BCQ 本身则从校准数据用交替优化（Algorithm 1）学最优缩放因子：GREEDY_INIT 初始化 → 每轮 LEAST_SQUARES（固定 B 解最小二乘 α）+ BST（固定 α 用二分搜索树求最优 B），共 R 轮，低位宽更准但离线校准更重；Key 校准一次每模型，LLaMA2-13B 校准 <10 分钟（H200 GPU）。bit-plane 表示上每平面 B_i∈{-1,+1} 存 1 bit/元素（packed binary），数学本质 Ŵ=Σ α_i B_i（B_i 是"方向"、α_i 是"幅度"），LUT-GEMM 直接对 {±1} 平面做加减 + 查表，无需非均匀量化 centroid index 的 bit-transpose。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - BCQ 离线校准（Omni-LUT Algorithm 1，Key cache）：输入 Key 校准数据 K_cal、位宽 B、轮数 R；GREEDY_INIT(K_cal,B) 得初始 α,B；for r=1..R：α ← LEAST_SQUARES(B,K_cal,α)；B ← BST(K_cal,α)；输出 α*,B*。BC-UQ 在线编码（BEA 贪心残差，Value/Key 通用）：r^(0)=x−zp；for i=1..q：B_i=sign(r^(i-1))；r^(i)=r^(i-1)−B_i·α_i；得 x≈zp+Σ_{i=1}^q α_i⊙B_i。张量计算例子（q=4，d=128 head）：设 token 的 Value 向量 x∈R^128，TSE 求 x_min/x_max → zp_v、δ_v → α={4δ_v,2δ_v,δ_v,0.5δ_v} → BEA 贪心得 4 个 ±1 bit-plane → 每个 bit-plane 由 LUT PE 按 4 激活组查表累加（与 32 个量化权重并行 RAC）→ 各 plane 结果乘 α_i 累加得最终点积。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：BCQ 在算法侧用交替优化/PyTorch 校准；硬件侧由 Omni-LUT 的 BQU（BEA 贪心编码器）在线实现、LGU/PE 消费 bit-plane。已开源的同类参考：NAVER LUT-GEMM（github.com/naver-aics/lut-gemm，支持 BCQ+uniform）、AnyBCQ（BCQ 的多精度扩展，支持直接 bit-plane 运算与按需加载前 p 个平面）。用途：把量化权重/激活表达成硬件友好的二值平面，使 mpGEMM 变成查表+加减；多精度推理可按需只加载所需平面，p=2 比 p=4 少 50% 数据。在 Omni-LUT 中 BCQ（Key per-channel 离线）+ BC-UQ（Value per-token 在线）共同构成 KV cache 量化，实现 KV4 平均 PPL 仅增 0.17、KV3 增 0.75。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization

## AS-Bit（Attention-aware Sensitivity-based Bit Allocation，注意力感知敏感度位分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AS-Bit 是 Omni-LUT（ISCA 2026）提出的 Key cache 自适应位分配算法。动机：BC-UQ/BCQ 对所有通道用固定位宽，但并非所有 Key 通道对 attention score 计算同等重要——每通道分布不同、对最终 QK^T 点积贡献不同，固定低位宽预算均匀惩罚所有通道，尤其伤害少数高敏感关键通道（小模型上更严重）。AS-Bit 的核心思想：给对量化误差更敏感的通道分配更高位宽，其余通道用低位宽。敏感度定义：attention A=QK^T 中 Key 的量化误差 ΔK 被对应 Query 项幅值放大，故有效敏感度 = 通道固有量化误差 × 对应 Query 通道能量。量化指标：(1) per-channel Query 能量 E[Q²]_d=(1/T_cal)Σ_{t=1}^{T_cal}(Q_{t,d})²（T_cal 校准 token 数）；(2) Key 在位宽 b 下的 per-channel 量化误差 MSE_b[d]=(1/T_cal)Σ_t(K_{t,d}−K_{q,b}[t,d])²；(3) 边际增益 ΔJ_d=E[Q²]_d·(MSE_{bℓ}[d]−MSE_{bh}[d])——从低位宽 bℓ 升到高位宽 bh 的 Key 误差减少量，用 Query 能量加权。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 离线校准流程：用校准数据对 Key 做双路径量化（dual-path quantization）分别算 MSE_{bℓ}[d] 与 MSE_{bh}[d]，并统计 Query 能量 E[Q²]_d → 对每通道算 ΔJ_d → 取 ΔJ_d 最大的 top k%（论文用 25%）通道分配高位宽 b_h，其余用低位宽 b_ℓ → 生成 per-channel 位分配表（如 b_h=4、b_ℓ=3，则 Key 有效位宽 = 0.25×4+0.75×3 = 3.25 bit；论文在 KV4 配置下给出 25% 高位 → 有效 4.25 bit）。在线：BEA 按位分配表逐通道用对应位宽编码 Key。效果（Fig.4）：仅 10-30% 通道用高位宽即可接近全高位精度；比只考虑 Key MSE 的分配收敛更快；跨模型（OPT/LLaMA2 等）趋势一致。直觉：ΔJ_d 中 E[Q²]_d 捕获"该通道误差被 Query 放大多少"、MSE 差捕获"加位宽能减多少误差"，两者乘积是加位宽的真实收益。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：PyTorch 离线校准（一次每模型），硬件侧由 BQU 的 Key Path 消费 per-channel 位分配结果（每个 Key 通道按分配位宽用 BEA 编码）；配合 LUT-based GEMM 加速器可处理可变位宽的灵活性。用途：任何 per-channel KV cache 量化 + 可变位宽 LUT 加速器的组合；相比 KIVI/KVQuant/Oaken 依赖 sparsity-based outlier 保留（有效 KV 位宽 4.8-5.0 bit），AS-Bit 不加任何额外位（Value 不加位）就达到更高有效位宽效率。论文未明确说明 bℓ/bh 的具体取值表与 top-k 超参搜索过程（仅给 25% 与 b=3/4 配置）。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization

## KV Cache 量化（hardware-aware 变体：Key per-channel + Value per-token 二值编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KV Cache 量化是把 LLM 推理中存储的 Key/Value 张量从 FP16/BF16 压缩到低比特表示（INT4/INT2 等）以降低内存占用与访存流量的技术。与权重量化的区别：(1) KV 是流式结构——新 token 的 K/V 实时追加，无法用需要离线全局统计的方法；(2) 数值分布随序列动态变化；(3) 量化误差跨层 residual 累积。主流的分布发现（KIVI [46]/KVQuant [29] 同期独立结论，Omni-LUT Fig.3 验证）：Key cache 有强 per-channel 特征——某些通道持续是大幅值 outlier，适合 per-channel 量化；Value cache 分布高度 token 相关、无稳定 per-channel 结构，适合 per-token 量化。Omni-LUT 的 hardware-aware 变体：Key 用离线校准 per-channel BCQ + AS-Bit 位分配，Value 用在线 per-token BC-UQ（TSE 在线求 min/max），量化结果直接是 LUT 可消费的 binary-coding bit-planes，不依赖 outlier 高精度隔离与 dequant。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 流水（一次 decode 步骤）：新 token → QKV 投影得 q,k,v → Key 走 Key Path：查离线校准的 per-channel zp_{k,c}/α_{i,c}（含 AS-Bit 位分配），BEA 贪心编码为 bit-planes；Value 走 Value Path：TSE 求 x_min/x_max → zp_v=(x_max+x_min)/2、δ_v=(x_max−x_min)/(2^b−1) → α_i=δ_v×power-of-2 basis → BEA 编码 → 量化 KV 追加到 cache（KV4=4 bit-plane、KV3=3 bit-plane）→ attention 的 QK^T 与 Attn×V 在 LUT datapath 上按 bit-plane 查表执行（bit-slicing：计算量∝bit-plane 数）。效果：KV3 相对 KV4 的 AA-GEMM 计算少 25%、KV 流量更小；3/4-bit KV vs FPE/FIGLUT 的 16-bit、Tender 的 8-bit，DRAM 能量优势显著（8192 input tokens 下总能量比 FPE/Tender/FIGLUT 低 50%/32%/38%）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：算法侧 PyTorch + HuggingFace transformers，在每层 attention 内模拟 KV 量化；硬件侧 BQU（TSE+BEA，宽 128 匹配 head dim，不增 cycle）。同类公开实现：KIVI（github.com/jy-yuan/KIVI，per-channel Key + per-token Value + 全精度滑动窗口）、KVQuant（per-channel+per-token 非均匀 + outlier 隔离）、QuaRot（head-wise Hadamard 旋转消除 outlier 后简单 per-head asymmetric INT4）。Omni-LUT 对比结果（Table II/III）：KV4-BCQ 平均 PPL 增 0.17、KV3 增 0.75，与 SOTA 相当；AS-Bit 使 Key 有效位宽 4.25 bit，低于 KIVI/KVQuant/Oaken 的 4.8-5.0 bit。论文未开源（联网 2026-08 未找到仓库）。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization

## AW-GEMM 与 AA-GEMM（activation-weight GEMM 与 activation-activation GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 按 GEMM 两个操作数来源分类的 LLM 推理 GEMM 类型：(1) AW-GEMM（activation-weight GEMM）——一个操作数是模型权重（离线已知）、另一个是运行时激活，出现在 linear 层（QKV 投影、attention output 投影、FFN）；权重量化后成为 FP-INT 混合精度 GEMM（mpGEMM）。现有 LUT-based GEMM 加速器（FIGLUT [52]、LUT Tensor Core [50]）主要针对 AW-GEMM。(2) AA-GEMM（activation-activation GEMM）——两个操作数都是运行时激活，出现在 attention 的 QK^T 与 Attn×V（操作数是缓存的 Key/Value）。复杂度随上下文长度不同：prefill 中 AW-GEMM 线性层 O(T)、AA-GEMM 因 token 对 QK^T 为 O(T²)；decode 中每个新 token 的 AW-GEMM 不随上下文增长，而 attention 要读全量 KV cache 做 QK^T 与 Attn×V，AA-GEMM 计算与 KV 流量都随上下文增长。因此长上下文下 AA-GEMM 是计算与能量主导项，高效 LUT 执行必须覆盖 AA-GEMM。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 一次长上下文 decode 的 GEMM 序列：token embedding → AW-GEMM（QKV 投影，W4A16）→ AA-GEMM（QK^T：Query 与缓存 Key 的点积，W=0 权重参与）→ softmax → AA-GEMM（Attn×V：attention score 与缓存 Value 加权）→ AW-GEMM（output 投影）→ FFN 的 AW-GEMM。量化形式：AW-GEMM 中权重可离线量化（低 bit 权重×FP16 激活）；AA-GEMM 中 Key/Value 是运行时激活，必须 KV cache 量化（Omni-LUT：Key per-channel BCQ + Value per-token BC-UQ，都转成 binary-coding bit-planes）。执行例子（Omni-LUT LUT datapath）：QK^T 的 K 以 4 个 bit-plane 存储，PE 每 cycle 按 4 激活组查表、32 个量化 K 值并行 RAC，查表次数∝bit-plane 数（KV3 比 KV4 少 25% 计算）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：商业加速器（GPU/TPU）不原生支持 FP-INT mpGEMM，通常先 dequant 权重到高精度再做 GEMM（低效）；LUT-based 加速器把"高精度激活×低比特权重"的部分积预计算进 LUT，查表+累加替代乘法。FIGLUT 只对 AW-GEMM 用 LUT、AA-GEMM 回退 FP systolic 后端（面积 0.39→1.03mm²）；Omni-LUT 通过 scale-aware LGU（row-wise 缩放内嵌查表）+ BQU（KV 在线量化）让 AA-GEMM 也留在 LUT datapath，等峰值吞吐下能效比 FIGLUT 高 1.25×-1.91×。对比指标（Table VI，OPT-6.7B 8192/512）：有效 GEMM TOPS Omni-LUT-KV4 1.78 vs FPE 0.76/Tender 0.75/FIGLUT 0.96。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization

## LUT-based GEMM（查表矩阵乘法 / LUT 加速的混合精度 GEMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LUT-based GEMM（查表矩阵乘法）是把量化（尤其低比特权重/二值 bit-plane）GEMM 中"高精度激活 × 低比特量化值"的点积预计算并存入查找表，运行期用量化值作为索引直接查表、把乘法换成查表+累加的加速方法。它避免两种低效：(1) 商业加速器不支持 FP-INT mpGEMM 而需先 dequant 权重；(2) 逐元素浮点乘法的高能耗。运作（LUT-GEMM [53]，Fig.2）：把激活分组（通常 4 个一组），该组激活与一组低比特量化值的全部组合点积预计算成 2^4=16 个高精度表项；每 cycle 该组按量化矩阵一列的 binary 值查表，查表次数∝量化位宽（2-bit 权重需 2 次查表）；LUT 在激活 tile × 量化矩阵列间复用，partial sum 累加得结果；新激活组重新生成 LUT。Omni-LUT 在此之上：LGU 把 row-wise 缩放与 zero-point 补偿内嵌进表生成（scale-aware），使 AA-GEMM（Key/Value 量化）也能走 LUT datapath；PE 每 4 激活组共享一个 LUT、32 个 binary weight 并行 Read-and-Accumulate。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 张量计算例子（μ=4 分组，A∈R^{1×K} 激活、B∈{-1,+1}^{K×N} 量化矩阵）：对每 4 元素组 g 构建 LUT：for pattern in 0..15：LUT[pattern]=Σ_{i=0..3} sign_i·A[g+i]（sign 由 pattern 的 bit 决定）；计算时 for n：w_bits=B[g:g+4,n]（4-bit 模式）；output[n]+=LUT[w_bits]。在 Omni-LUT 中进一步按 bit-plane 展开：每个量化操作数（权重 W4 或 KV3/KV4）有 q 个 bit-plane，对每个 plane 做上述查表+累加，最后 Σ_i α_i·partial_i 得点积；LGU 生成表时先对组内 4 个激活各乘 row-wise scale、并在首 plane 内嵌 zero-point 补偿（把激活组×zero-point 向量的点积加进每个表项），从而支持精度最优的 row-wise 量化方向。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现方式：软件侧 NAVER LUT-GEMM（github.com/naver-aics/lut-gemm，首次在 GPU 实现 BCQ 格式 LUT 计算）、FLUTE（LUT 向量化 + 跨 bank 复制消 conflict，2-4×）、T-MAC（CPU in-register LUT，ARM TBL/x86 PSHUF）；硬件侧 FIGLUT（HPCA 2025，custom Read-Accumulate 单元替代 MAC）、LUT Tensor Core（ISCA 2025）、Omni-LUT（ISCA 2026，AW+AA 全覆盖）。硬件实现要点：每 PE 存 LUT（4 激活组 = 16 项 FP16）、32 个量化权重并行查表、half-LUT 符号对称省一半表项（T-MAC 思想）、LUT 生成器与 PE 并行避免 stall。用途：低比特 LLM 推理（W4A16/KV4A16 等 mpGEMM），decode（batch=1、memory-bound）与 prefill 都受益；Omni-LUT 等峰值吞吐下能效比 FIGLUT 高 1.25×-1.91×。

涉及论文标题：
- Omni-LUT: Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization

## Order-Independent Transmittance（OIT，顺序无关透射率）/ MLP-based OIT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OIT 在本论文中指一种替代显式深度排序的算法：直接计算/预测每个 Gaussian 的透射率（衰减因子）F(d_i)，使最终颜色可按与顺序无关的方式合成，从而省掉排序环节。动机链：3DGS 的 α-blending 本质是图像合成（"over" 算子），而 "over" 非交换 → 传统需要按深度排序；计算机图形学为绕开排序发展了 order-independent transparency（OIT）技术（A-buffer、stochastic transparency、weighted OIT 等），其中 weighted OIT 用深度单调递减权重 F(d_i) 做加权合成 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i，质量损失可忽略。本论文把这一思想引入 3DGS：观察到排序的唯一目的是算正确的累积透射率 T_i（随深度递减的衰减因子），于是提出直接用轻量 MLP 预测 F(d_i)。关键扩展：3DGS 是视角相关渲染，同一深度在不同视角下贡献不同，因此输入为 (深度 d_i, 归一化视角方向 (x,y,z))——视角信息在推理时对同一相机位姿恒定、可折入 MLP bias；MLP 为 2 层 10 参数（Leaky ReLU(1/8) + 指数输出激活），推理仅 6 MAC，训练约 30 分钟/场景。最终渲染式 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i（式 5），与 baseline 排序 3DGS 相比 PSNR 仅降 0.3（26.90 vs 27.21）、SSIM 几乎不变、LPIPS 略优，且优于 handcrafted depth-function 的 sort-free weight-sum 渲染[18]（25.43 PSNR，LC-WSR 最优变体）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
训练与推理 pipeline（本论文 IV-B 章 + Fig.7）：
```
# 训练（RTX 3090，每场景约 30 min）
1) 用原始排序 3DGS 预训练 7000 epochs 得 checkpoint（初始化，加速收敛）
2) 对每个相机位姿 projection 得深度 d_i，构造样本 (d_i, 视角(x,y,z))
3) 前向：MLP → F(d_i)；按 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i 渲染，与 GT 算 loss（原 3DGS 设置）
4) 反向：MLP lr=0.005（大步长快速收敛），Gaussian lr×0.01（小步长慢精修）；
   关闭 cloning/splitting 保持 Gaussian 数恒定、训练稳定
# 推理（加速器）
1) 相机位姿给定 → 视角 (x,y,z) 恒定，预计算并融合进 bias（b_i=b'_i+c_i·view）
2) 对每 tile 内 Gaussian 深度 d_i 做 6 MAC 前向得 F(d_i)（可重构 PE 阵列 MLP 模式，
   每周期处理 256 个深度值，写回 depth buffer）
3) α-blending 用广播寄存器中的 F(d_i) 累加分子分母 → 除法阵列归一化输出像素
```
动态场景扩展：Neu3D + 4DGS baseline，每 30 帧更新一次 10 参数 MLP（300 帧序列共 10 组），PSNR 仅降 0.45。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU 实现（本论文 VI-E 章）：用 cuBLAS 以 GEMM 形式做 MLP 推理替换 Gsplat 的 Radix sort，因 MLP 算术强度低（1 深度参数仅 6 MAC vs 光栅化每 GS 256×6 MAC，~30 倍差）而 memory-bound，几何均值延迟为 baseline 排序的 1.59×（更慢）——说明 GPU 上 MLP-OIT 不划算，需专用硬件。加速器实现：复用光栅化 PE 阵列的 MAC/EXP 单元做 MLP 推理（可重构仅 +5% 面积/+6% 功耗），相对 32 并行 bitonic 排序网络 21.1~32.4× 加速，配合 fine-grained interleaved pipeline 隐藏 memory-bound 延迟（见硬件架构层对应条目）。对比：weight-sum[18] 是手工深度函数（无视角信息），硬件上需额外除法单元（+0.363mm²/+341mW），而本 MLP 复用现有除法阵列，仅 +0.147mm²/+88mW。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

## 空间数据结构（Spatial Data Structure：Kd-Tree / Octo-Tree / R-Tree / BVH）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间数据结构是组织几何数据、避免线性遍历的基本抽象，用于点云处理、光线追踪、碰撞检测。共同特征：每个非叶节点代表一个物理空间，子节点空间包含于父节点空间，叶节点对应点/图元。三种代表结构按空间划分方式区分：Kd-Tree（Bentley 1975）沿不同维度用平面交替二分，逐层平分点分布（split-domain 顶向下）；Octo-Tree 每个节点 8 个子节点，对应三维边长各半分的 8 个子立方（split-domain）；R-Tree（Guttman 1984）自底向上构建，父节点是全部子节点包围盒（bounding-box）。BVH 是光线追踪的包围体层次结构（同属 bounding-box 类）。NNS 复杂度从暴力 O(NM) 降到 O(N log M)。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：Kd-Tree NNS（Listing 1）三步通用结构：(i) 叶节点处理——算查询点 Pos 与叶内点的距离并更新结果表 ResList；(ii) 扩展规则（结构相关）——比较 Pos[Node.Axis] 与分裂阈值 Node.Thresh 决定先走左/右子节点；(iii) 递归+剪枝（通用）——`if NeedExpand(Pos, Node, ResList): KdTreeNNS(Pos, inf_child, ResList)`，NeedExpand 用 `d*d < ResList.MaxDis()*Alpha` 判定是否回溯展开另一分支。关键区别：与 B+ 树/skip list 等 1D 索引不同，空间数据结构"包含 ≠ 邻近"（Containment ≠ proximity），DFS 回溯不可避免（Fig. 4 反例：坐标小于阈值却仍可能需访问另一分支），这是它难以被哈希/非回溯加速器加速的根本原因。Octo-Tree 的距离计算用 Iter 原语对 8 分支流水化复用计算单元；R-Tree 用优先队列按包围盒距离决定访问顺序。
- 术语一般如何实现？如何使用？：软件实现常用 PCL（Point Cloud Library，RoboCortex 用它构建三种结构）；搜索算法通常带 k-d tree kNN 的 FLANN、CGAL、Embree（BVH 构建）。硬件/系统使用：RoboCortex 用 RSU 数据流（Stack 用于 Kd/Octo-Tree、Priority Queue 用于 R-Tree）执行搜索，不同数据结构共享同一套原语——把搜索拆成叶距离/扩展/递归三部分即可适配。数据集：自主驾驶（KITTI-360，点分散）vs 物体重建（EPFL Statues，点紧凑）；实验结果：Kd-Tree 最受益于物理局部性+预取，Octo-Tree 最受益于 RSU 硬件本身（其地址连续、L1/L2 hit 本就最高），R-Tree 在紧凑点云收益最大。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

## 最近邻搜索（NNS，Nearest Neighbor Search / KNN）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 最近邻搜索是为查询点找到数据集中距离最近的 k 个点（KNN）的基础操作，是点云配准（ICP）、SLAM、碰撞检测的核心开销。点云场景（RoboCortex）：对源点云 P 的每个点 p，在目标点云 Q（N,M>10³）中搜 k 近邻；暴力法 O(NM)，用 Kd-Tree 等空间数据结构可降到 O(N log M)。并行策略：把不同源点绑定不同线程（Listing 2 ConcurrentNNS 的 parallel_for）。注意与 ANNS（近似最近邻，如 Faiss/量化/哈希）区分：RoboCortex 是精确 NNS（无精度损失，这是相对 Tartan 近似算法的卖点）。与 L-PCN/PointNet++ 的 Ball Query/KNN 邻居收集不同：那是为 MLP 特征计算收集点集，RoboCortex 是为 ICP 配准找几何最近邻。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：Kd-Tree KNN 伪代码（RoboCortex Listing 1）：
```
def KdTreeNNS(Pos, Node, ResList):
    if Node.IsLeaf():                       # Part 1 叶节点
        ComputeDisForLeaf(Pos, Node)
    if Pos[Node.Axis] < Node.Thresh:        # Part 2 扩展规则
        sup_child, inf_child = Node.Left, Node.Right
    else:
        sup_child, inf_child = Node.Right, Node.Left
    KdTreeNNS(Pos, sup_child, ResList)      # Part 3 递归
    if NeedExpand(Pos, Node, ResList):
        KdTreeNNS(Pos, inf_child, ResList)
def NeedExpand(Pos, Node, ResList):
    if ResList.size() < k: return True
    d = Pos[Node.Axis] - Node.Thresh
    return d*d < ResList.MaxDis()*Alpha
```
执行链：根节点 → 按维度坐标比较下探 → 到叶算距离更新 ResList（MaxD 用 Reg 维护）→ 回溯时 NeedExpand 判定是否展开另一分支 → 输出 k 近邻。NNS 占 ICP 一次配准延迟的 53.25%（point-to-plane）到 71.66%（point-to-point），是核心瓶颈。
- 术语一般如何实现？如何使用？：软件：PCL kdtree/knn、FLANN、scikit-learn KDTree；硬件/系统：RoboCortex 用 near-cache RSU 数据流加速（显式栈支持递归回溯）+ Path Buffer 物理局部性复用 + RSU 引导预取，NNS 相对 CPU 加速 2.74-13.07×（自主驾驶）/12.73-77.94×（物体重建）。GPU 方案（RTX、RTNN ray tracing）受分支发散与 CPU-GPU 数据搬运（占 16.24-42.57%）限制加速 <2×。精确 NNS 与近似（Tartan）对比：高精度需求下近似方案性能收敛于 baseline，RoboCortex 不牺牲精度。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

## ICP（Iterative Closest Point，迭代最近点配准）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ICP 是点云配准（registration）的标准迭代算法：给定两帧点云（如自动驾驶车辆运动前后 LiDAR 扫描的 Q 与 P），通过迭代"找最近邻对应点 → 估计刚体变换（旋转 R/平移 t）→ 应用变换"最小化两帧点云的几何错位。RoboCortex 以 ICP 为端到端评估 workload：每次迭代的核心开销是 NNS（找 P 中每点在 Q 中的 k 近邻），加 Jacobi SVD 分解估计变换。变体按最近邻数与矩阵迭代算法区分：point-to-plane（k=10）、point-to-line（k=5）、point-to-point（k=1，纯最近邻）。k 越小，NNS 优化带来的端到端收益越大。NNS 占单次配准延迟 53.25%-71.66%。
- 从算法pipeline角度拆解术语，给出具体计算过程例子：ICP 一次迭代（Initial Registration）流水 = Build（初始空间数据结构构建，仅首帧）→ NNS（对 P 每点在 Q 中搜 k 近邻）→ SVD（Jacobi SVD 分解估计 R/t）→ 应用变换；Following Registration 只含 NNS+SVD。端到端例子（Fig. 17）：ICP point-to-plane 在 KITTI-360 上，RoboCortex 的 NNS 优化使并行部分 2.27× 提升、端到端 18%-28% 改善；随真实配准进行，收益从 Initial Registration 转移到 Following Registration（Build 一次性开销摊薄）。三种模式对比（Fig. 18，去除数据加载时间）：k 越小 NNS 优化收益越显著。
- 术语一般如何实现？如何使用？：软件实现参考 https://github.com/FeeZhu/ICP（论文引用 [13]）；点云库 PCL 有 pcl::IterativeClosestPoint。硬件/系统使用：RoboCortex 在 zsim 上执行 ICP（对标 Jetson AGX Orin），RSU 加速 NNS、Path Buffer 挖掘物理局部性、语义预取降低缓存 miss；对比 baseline 包括 base CPU、Tartan（机器人 CPU，近似 NNS+预取，收益有限且牺牲精度）、stream 预取器版 RoboCortex。数据集：自主驾驶（KITTI-360）与物体重建（EPFL Statues）。缩放实验（Fig. 20）：1/8/16 核下 RoboCortex 均显著优于 baseline 与 Tartan。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）

## W4A8KV4P8 混合精度量化（操作数依赖的混合数值格式 hybrid numerical formats）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4A8KV4P8 是 P3-LLM（ISCA 2026，Cornell/KU Leuven/Stanford）为 edge LLM inference 设计的混合精度量化方案，含义为：权重（W）4-bit、激活（A）8-bit、KV-cache 4-bit、注意力分数（P，attention-scores）8-bit。其核心思想是 operand-dependent quantization——不为所有操作数套用同一数值格式，而是根据每个 LLM 操作数的数值分布特征（动态范围、outlier 模式、符号性、数值范围）分配专属数值格式，同时兼顾"内存压缩率-模型精度-硬件计算效率"三者的平衡。四个操作数各自采用不同格式：权重用 4-bit BitMoD（FP4 负零编码重映射特殊值 {±5,±8}，group 128）；KV-cache 用 4-bit INT4-Asym（非对称整数，per-head 量化，128 元素共享 16-bit scale + 4-bit zero-point，有效精度 4.16 bit）配合动态输入感知平滑抑制 key cache outlier；激活用 FP8-E4M3（per-token，宽动态范围容纳 outlier，无需 Hadamard 变换或 SmoothQuant 式平滑）；注意力分数用无符号 FP8-S0E4M4（softmax 后值域 [0,1]，无需符号位，4-bit 指数 bias −15 覆盖 [−14,−1]，4-bit 尾数提供数值保真）。动机依据（Fig.3 分析）：权重和 KV-cache 占解码阶段内存主导、对量化不敏感（4-bit 可接受），激活与注意力分数占内存小但对量化敏感（保持 8-bit），且 8-bit 注意力分数使 P·V 能完全跑在低精度 PIM 上（否则 value cache 需搬回 NPU 用 FP16 计算）。通用量化公式（Eq.1）：Δ=|X|_max/Q_max，X_Q=Round(X/Δ,Q)，X̃=X_Q·Δ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 decode 迭代中 W4A8KV4P8 的计算流（以 Llama-3.1-8B post-RoPE 为例）：
```
# 预填充阶段（离线/在线各一次）：
for c in 0..H-1:  s_c = Max(|K_prefill[:,c]|)        # 动态平滑因子（无校准数据集）
for c in 0..H-1:  K_S[:,c] = K[:,c] / s_c             # 平滑后 key cache
# 权重离线：BitMoD 4-bit，group=128，搜索最优特殊值替换负零
# 解码阶段（每 token）：
h = layer_input(token)
q = Q_linear(h); q = RoPE(q)                          # NPU 高精度
q = FP8_E4M3(q * s_K)                                 # SSF 融合进 query 缩放
for head:  K_new = K_linear(h); K_new = RoPE(K_new)
           Kq = INT4_Asym(K_new / s_c)                # per-head 量化
           P = PCU_GEMV(q, K_S^T)                     # Q·K^T 在 PIM
           P = softmax(P)                             # NPU
           P8 = FP8_S0E4M4(P * S^V / S^V_max)         # 融合 value 缩放
           O = PCU_GEMV(P8, V_INT4) * S^V_max         # P·V 在 PIM
out = O_linear(O)                                     # 线性层 GEMM，dequant 后置
```
效果：平均 perplexity loss 仅 0.25（Wikitext-2）/0.31（C4），比 QuaRot（0.30/0.48）与 QoQ（0.30/0.38）更低；MMLU/ARC-C/GSM8K 平均准确率比 QuaRot/QoQ 高 2.57%/3.05%；decode 内存相对 FP16 降 3.7×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/yc2367/P3-LLM（MIT，wkvaq_quant/ 代码库 + AWQ、LM-Eval 子模块）。使用流程：`run_awq.sh`（`wq_dtype=int|bitmod`，4-bit group 128）做权重量化 → `run_awq_save_4b_model.sh` 保存 fake-quant 模型 → `test_ppl_template.sh`（flag：`--kv_quant_method KTVT/KCVT`、`--apply_k_scale`、`--k_quant_post_rope`、`--p_bits`）测 Wikitext-2/C4 perplexity（仅支持 Llama/Mistral）。在硬件侧，P3-LLM 把这套格式与低精度 PCU 协同设计：6-bit 定点乘法器（5-bit 尾数含隐藏位 + 符号位）同时解码 BitMoD 权重与 INT4-Asym KV，4-bit 指数移位替代浮点乘法中的指数对齐，从而使 PIM 在等面积下获得 4× 计算吞吐（详见本库 PIM/DRAM-PIM 与 Throughput-Enhanced PCU 条目）。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

## FP8-S0E4M4（无符号 8-bit 浮点格式，注意力分数量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP8-S0E4M4 是 P3-LLM 提出的无符号 8-bit 浮点格式：0 个符号位 + 4-bit 指数 + 4-bit 尾数，指数 bias 为 −15。用于量化 softmax 之后的注意力分数（attention-scores）。设计依据两条观察：(1) softmax 输出恒在 [0,1]，无需符号位；(2) FP16 有 5-bit 指数（bias −15，指数范围 [−14,15]），而注意力分数恒 <1，正指数完全用不上，有效指数范围仅 [−14,−1]（14 个值），4-bit 指数足够覆盖，省下的 1-bit 给尾数提升数值保真。相比 INT8 与 FP8-E4M3（表 II 实验）：INT8 量化注意力分数带来明显 perplexity 退化，FP8-S0E4M4 达到 near-lossless（Llama-2-7B Wikitext-2：FP16 5.15 → FP8-S0E4M4 5.15，INT8 5.19，FP8-E4M3 5.16）。注意 FP8-S0E4M4 与工业标准 FP8-E4M3/E5M2（NVIDIA/AMD 支持）是不同格式：FP8-S0E4M4 无符号且 4+4 划分专为 [0,1] 分布设计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
注意力分数量化与融合流程（解码阶段）：
```
# softmax 后得到 P ∈ [0,1]^T（FP16）
P_scaled = P * (S^V / S^V_max)      # 融合 per-value-head 缩放（二级缩放防越界）
P8 = round_to_fp8_s0e4m4(P_scaled)  # 直接保留 FP16 高 4 位 mantissa + 指数重映射（无缩放因子）
O = PCU_GEMV(P8, V_INT4) * S^V_max  # P·V 在低精度 PCU 执行，结果乘回 S^V_max
```
要点：FP8-S0E4M4 不需要量化缩放因子（格式本身覆盖所需数值范围），因此"直接截位"即可量化，省去 scale 存储与乘法；S^V_max 的二级缩放保证融合后 P_scaled 仍在 [0,1]（不破坏无符号假设），P·V 完成后乘回。硬件侧（见本库硬件架构 PCU 条目）：该格式的 8-bit 尾数（5-bit 含隐藏位）+ 符号位恰好匹配 PCU 的 6-bit 定点乘法器输入宽度，使 P·V 不必回退到 FP16 单元。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 PyTorch 中作为 fake-quant（论文开源仓库 https://github.com/yc2367/P3-LLM 的 `--p_bits 8` 路径）——对 FP16 注意力分数直接取高 4 位 mantissa 并重映射指数到 bias −15 的无符号 4-bit 指数表示；硬件上用 6-bit 定点乘法器消费（尾数 5-bit 含隐藏位 + 1 符号位），4-bit 指数仅用于移位乘积。适用场景：任何需要对 [0,1] 有界张量做 8-bit 量化的低精度 MAC 硬件（PIM、NPU 低精度单元）；该格式表明"为操作数分布定制指数位分配"比通用 FP8 更优。开源状态：算法代码已开源，硬件 RTL 未开源。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

## BitMoD（FP4 负零编码重映射的特殊值权重量化格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BitMoD 是 P3-LLM 采用的 4-bit 权重量化格式（论文引用自文献 [6]），核心思想：FP4（4-bit 浮点）的基本量化值集合为 {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}，其中 ±0（尤其负零）的编码是冗余的——同一数值 0 对应多个编码。BitMoD 把这个冗余的负零编码重映射为 4 个预定义的特殊值 {±5, ±8}，并按权重分组搜索每组最优的一个特殊值来替换负零，从而比非对称整数量化（INT4）更小地降低量化误差，硬件开销极低（仅需解码器将特殊值映射回）。相比 MANT 的自适应数值类型（需把乘法分解成两个高精度部分和，增加面积/能耗）与 Ecco 的 k-means codebook + Huffman 编码（需在线解压回 FP16），BitMoD 的 6-bit 定点表示（4-bit 值 + 组级缩放相关位）能被低精度乘法器直接消费。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BitMoD 权重离线量化与在线解码：
```
# 离线（每 group 128 个权重）：
Q_base = {-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6,SPECIAL}
for v in {±5, ±8}:          # 候选特殊值
    Q = Q_base with 负零 -> v
    err(v) = Σ |w_i - nearest(Q, w_i)|²
v* = argmin err(v)           # 选最优特殊值
Wq[i] = index of nearest(Q(v*), w_i)   # 4-bit 索引 + 组级 scale + v* 元数据
# 在线 MAC（PIM PCU，见硬件架构条目）：
product = w_mantissa(6-bit 定点，含 v* 解码) * x_mantissa(6-bit)
product <<= x_exponent(4-bit)          # 指数移位
acc += product                          # 32-bit 定点累加
# 线性层：GEMM 完成后统一乘组级 dequant scale（fusion 后置）
```
效果：消融实验（Table VI）显示 INT4 权重量化使 Llama-2-7B Wikitext-2 PPL 增 0.13，换用 BitMoD 后降至 0.01 增量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：算法侧在 AWQ 流程之上实现（开源仓库 https://github.com/yc2367/P3-LLM 的 `wq_dtype=bitmod`，group size 128）；硬件侧 PCU 为权重与 KV-cache 共享同一 6-bit 定点乘法器——由于权重（BitMoD）与 KV-cache（INT4-Asym）映射到 MAC 硬件的同一操作数位置，PE 内需要一个小解码器同时支持两种格式（BitMoD 特殊值 6-bit、INT4-Asym 5-bit 含 zero-point）。与仅权重量化（weight-only）不同，BitMoD 在 P3-LLM 中是 W4A8KV4P8 的一部分：权重 4-bit 压缩内存、激活保持 8-bit 降低精度损失，从而在低精度 PIM 上同时获得带宽与计算收益。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

## 动态输入感知 KV cache 平滑（Dynamic Input-Aware Smoothing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
动态输入感知平滑是 P3-LLM 抑制 key cache outlier 通道的量化预处理技术。LLM 的 key cache 在固定通道上呈现明显 outlier（Fig.5(b)(f) 显示 pre-RoPE key cache 有结构化 outlier 通道，而 value cache 无 outlier），直接 4-bit 量化误差大。P3-LLM 对每个 key 通道除以该通道的绝对最大值（平滑因子），把数值压到 [−1,1]：K_S[:,c]=K[:,c]/Max(|K[:,c]|)。与既有方案的两个关键区别：(1) 无需校准数据集——Oaken 靠离线校准定 outlier 阈值、QoQ/SmoothQuant 靠校准定平滑因子，都会对新数据集过拟合（Fig.8 显示 QoQ 用 Pile 校准在 Wikitext-2/C4 上量化误差最高）；P3-LLM 的平滑因子直接在 prefilling 阶段从当前输入计算，动态感知输入；(2) 同时研究 pre-RoPE 与 post-RoPE 两种量化位置——通过 profiling 发现 RoPE 旋转对 key cache 分布的影响取决于最大序列长度：Llama-2（4K 序列）post-RoPE key cache 结构化 outlier 被打散（不利量化），故用 pre-RoPE 量化；Llama-3（128K 序列）在典型 4K context 下旋转角很小、post-RoPE 分布几乎不变（保留结构化 outlier 利于量化），故用 post-RoPE 量化。开销分析：每个通道一个平滑因子（所有 token 共享），额外内存 <1%（与 context 长度成反比）；平滑因子计算仅需 prefilling 上下文，A6000 上 Llama-3.1-8B 全层 <5ms（即使 32K context），相对 250ms TTFT SLO 仍 <2%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Prefilling（每层每 head）：
for c in range(H):                       # H = head dimension
    s_c = max(abs(K_prefill[:, c]))      # 每通道绝对最大值
    K_S[:, c] = K_prefill[:, c] / s_c    # 平滑到 [-1,1]
save s_c                                 # decode 阶段复用（内存 <1%）
# Decoding（每 token）：
K_new = RoPE(K_linear(h))
Kq_new = INT4_Asym(K_new / s_c)          # 用同一平滑因子后量化
# 硬件融合：Q·K^T 时把 s_c (SSF) 元素乘进 query，再 FP8 量化 query
q8 = FP8_E4M3(q * s_c)
score = PCU_GEMV(q8, Kq^T)               # 无需在线对 K 解量化
```
效果：消融（Table VI）显示动态平滑把 Llama-2-7B/Llama-3.1-8B 的 Wikitext-2 PPL 降低 0.10/0.17；相对 Oaken（有效 4.8 bit）在更低有效精度（4.16 bit）下 perplexity 更好，且避免校准过拟合（Fig.8 在 C4 上误差最低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：算法侧在 PyTorch 中实现（开源仓库 https://github.com/yc2367/P3-LLM 的 `--apply_k_scale` 与 `--k_quant_post_rope` flag 组合），每个 head 内按通道求 max 并保存 scale 张量，decode 阶段对新 key 做除法后 INT4-Asym 量化（per-head，128 元素共享 16-bit scale + 4-bit zero-point，有效精度 4.16 bit）；硬件侧把平滑因子融合进 query 的 FP8 量化缩放（SSF fusion），使 Q·K^T 在 PIM 上直接消费量化 key 而无在线解量化。适用场景：任何 KV-cache 4-bit 量化的 LLM 推理（尤其 key cache 含结构化 outlier 通道、且希望避免校准数据集依赖的部署）；pre/post-RoPE 的选择需按模型最大序列长度 profiling 决定（短序列 pre-RoPE、长序列 post-RoPE）。


涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

## FP8 量化（W8A8）与 Scaled Matrix Multiplication（Scaled MM，缩放矩阵乘）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP8 量化是 W8A8（8-bit 权重 + 8-bit 激活）低精度推理范式：权重与激活都量化到 FP8（E4M3/E5M2），推理时用 FP8 数据做矩阵乘，输出前按每 token/每 channel 的缩放因子反缩放（dequantize）。Scaled MM（Scaled Matrix Multiplication，vLLM 的 FP8 后端 kernel，源自 CUTLASS/DeepGEMM）正是 W8A8 推理的核心算子：`C = (A_fp8 * scale_A) @ (B_fp8 * scale_B)`，缩放因子随张量/块携带，避免整型量化需要的在线反量化开销——FP8 的浮点格式使 MMA 可直接在 Tensor Core 上以 FP8 精度执行、以更高精度累加，再乘回 scale。在 PIPEWEAVE 中它是 6 类被建模 kernel 之一（vLLM、CUDA C++、FP8、Tensor pipeline、HW/SW 双调度范式），是 FP8 推理场景性能预测的验证对象。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
W8A8 FP8 推理的量化-计算 pipeline（以 GEMM 为例）：
```
# 离线量化（post-training）：
A_fp8 = quantize_to_fp8(A_fp32, scale_A)   # 每 token 或每 tensor 一个 scale
B_fp8 = quantize_to_fp8(B_fp32, scale_B)   # 每 channel 或每 block 一个 scale
# 在线推理（Scaled MM kernel）：
#   Tensor Core 执行 FP8 MMA：D = A_fp8 @ B_fp8 (FP32 累加)
#   epilogue 阶段：C = D * (scale_A * scale_B)   # 反缩放，fuse 进 epilogue
# PIPEWEAVE 对 Scaled MM 的建模维度：M∈[2,131072], N∈[384,8192], K∈[256,8192]
#   Tensor ops = α·tile_M·tile_N·tile_K（α=2），FP8 使每 SM Tensor 吞吐翻倍
```
FP8 在 Hopper 上每 SM 的 Tensor 吞吐是 BF16 的 2 倍，Scaled MM 因此成为 decode 阶段（带宽受限）之外 prefill/GEMM 密集阶段的主要加速手段。PIPEWEAVE 评估：在 seen GPU（H20/H800）上 Scaled MM 的 MAPE 1.9%/4.1%，unseen（H100/H200）4.2%/5.2%，比 Roofline/Linear/Habitat/Neusight 精度提升 10.8×/9.5×/5.5×/7.8×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：vLLM 的 FP8 路径用 CUTLASS scaled-GEMM 模板（scaled_mm 支持 E4M3/E5M2 与 per-tensor/per-channel scale），DeepGEMM 提供 Hopper/Ada 的高性能 FP8 GEMM（WGMMA + TMA），Transformer Engine 提供 FP8 训练推理栈。使用方式：模型权重离线量化到 FP8（可配合 activation 校准得到 scale）→ 推理框架（vLLM）在 FP8 支持的 GPU 上调用 Scaled MM kernel 替代 BF16 GEMM → PIPEWEAVE 等性能模型可据此预测 FP8 kernel 的延迟（其 Tensor pipeline demand 按 FP8 吞吐计算）。注意：FP8 精度敏感，需要 scale 校准与 outlier 处理（常与 SmoothQuant 式激活缩放结合）。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

## BERTScore（基于 BERT 上下文的文本生成评估指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BERTScore（Zhang et al., arXiv:1904.09675）是用预训练 BERT 的上下文嵌入评估生成文本与参考答案语义相似度的指标，范围 [0,1]（越高越相似）。流程：候选文本 y 与参考文本 x 各 token 化并经 BERT 提取上下文向量 → 计算两文本 token 对间的余弦相似度矩阵 → 对每个候选 token 找参考中最相似 token（Precision）、对每个参考 token 找候选中最相似 token（Recall），F1 为调和平均。公式（Web 证据确认）：R_BERT = (1/|x|)·Σ_{x_i∈x} max_{y_j∈y} x_iᵀy_j，P_BERT = (1/|y|)·Σ_{y_j∈y} max_{x_i∈x} x_iᵀy_j，F_BERT = 2·P·R/(P+R)。相比 BLEU/ROUGE 的精确词匹配，BERTScore 语义感知（同义/改写仍高分），适合 QA、摘要、RAG 等生成任务。PRowhammer（ISCA'26）用它量化 LLM 攻击：Llama-2-7B/Mistral-7B/Falcon-7B（4-bit 量化，llama.cpp/GGML）在 Google Natural Questions 100 问上的平均 F1 从 pristine 0.58–0.62 跌到 corrupted 0.25–0.30——论文还验证：即使对任意常量/无关字符串，BERTScore 也落在 0.25–0.30 区间（说明该区间近似"无语义"下限）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BERTScore 计算（对单个 QA 样本）
def bertscore(x_tokens, y_tokens, model):        # x=参考答案, y=模型生成
    Ex = model.embed(x_tokens)                    # 参考 token 上下文嵌入
    Ey = model.embed(y_tokens)                    # 候选 token 上下文嵌入
    sim = cosine_similarity(Ex, Ey)               # |x|×|y| 相似度矩阵
    P = mean( max_j sim[i][j] for i in cand )     # 候选→参考 对齐
    R = mean( max_i sim[i][j] for j in ref  )     # 参考→候选 对齐
    return 2*P*R/(P+R)                            # F1
```
PRowhammer 的评估管线：对 100 个 NQ 问题，人工标注参考答案 → 用 pristine 模型与 corrupted 模型各生成回答 → 分别算 BERTScore F1 → 报告 100 问平均（表 VI：A6000/4090/5060 上三模型 pristine 0.58–0.62 → corrupt 0.25–0.30）。攻击效果分级：灾难性（输出 # 串或跨语言乱码，BERTScore≈0.25–0.30）与"连贯但错误"（如 "Spike"→"Momo"，表面流畅而事实错误，更难检测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bert_score Python 库（预训练 BERT/RoBERTa 模型权重）；可调 token 加权（IDF）、层选择（约第 9 层嵌入最佳）、线性重缩放。使用：LLM 生成任务评估（QA/摘要/翻译/RAG）。在 PRowhammer 中作为攻击效果度量而非优化目标；论文用它证明单 bit-flip 足以让 7B 级 LLM 生成无意义或事实错误文本，同时指出 0.25–0.30 是常量/无关字符串的 BERTScore 下限，需配合人工检查区分"乱码"与"连贯但错误"两类失败模式。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU

## 可利用位翻转定位（exploitable bit-flip）与递归分段剪枝

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
可利用位翻转（exploitable bit-flip）指在 GPU 共享库压缩 nv_fatbin 中、翻转后使 kernel 输出与预期不同（而非崩溃或不受影响）的 bit 位置。PRowhammer（ISCA'26）提出无需逆向闭源压缩算法即可定位它们的算法管线：(1) 可行性验证——自编译 CustomLib（默认压缩，nv_fatbin 21KB）：随机单 bit-flip+GPU 执行（100ms/trial，10000 trials），崩溃率 8.13–11.16%、可利用率 0.21–0.25%（21–26 个可利用位）；(2) 剪枝策略——应对大库（cuBLASLt 压缩 255MB、逐 bit 全查需约 11805 天）：把 nv_fatbin 均分 n=2 段，对每段翻转全部 bit 并执行目标库 kernel，输出正确则丢弃该段，崩溃或输出改变则标记为有用段并递归二分，直到阈值 T=1KB；再从有用段随机抽 10000 bit 逐个 flip+执行（500–700ms/次），cuBLASLt 得 3–83 个、GGML 得 41–99 个可利用位，最大库 ≤90 分钟；(3) 单 bit-flip 多指令改义——压缩码中单 bit-flip 解压后平均改 2–5 个（最多 25 个）合法 SASS 指令，多数组合崩溃、相当数量只改输出。效果：50000 次随机 trial 找到 MNIST/FMNIST/CIFAR-10 218 个、ImageNet 93 个可利用位（不同输出维度调不同 kernel）；同一翻转位跨模型/数据集转移（RPL>80% 转移位 92–169 个）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 剪枝定位（Sec. IV-A，n=2, T=1KB）
def find_useful(seg):
    if len(seg) <= T: return [seg]              # 1KB 阈值
    out = exec_kernel(flip_all_bits(seg))       # 翻转段内全部 bit 后执行目标 kernel
    if out == golden: return []                  # 输出正确 → 丢弃
    return find_useful(seg[:n//2]) + find_useful(seg[n//2:])   # 崩溃/改输出 → 二分递归
# 主流程：seg=整个 nv_fatbin → useful 1KB 段集合 → 随机抽 10000 bit 逐个 flip+执行
```
artifact 五阶段 profiling 管线（脚本名）：kernel_locater_<lib>.sh（定位应用调用的 kernel → regions 文件）→ choose_target_region.sh（选大而连续、可 Rowhammer 的候选区域）→ run_flipper_watchdog_<lib>.sh（候选区域精确 bit-flip 实验）→ segregate.sh（整理原始结果）→ extract_useful_flips.sh（过滤汇总成 bitflip_data.csv）；用 cuobjdump+diff 验证翻转后 SASS 合法性。预计算关键位：mnist/fmnist/cifar10 用 cuBLASLt 偏移 0x95c787a 的 bit 4；imagenet 用偏移 0xc56745c 的 bit 8。black-box profiling 模型：单线性层（随机权重、输出维度=目标类别数）定位实际调用 kernel（3508 个中只调 1–2 个），在 profiling 模型上最致命的 bit 在目标模型上也最致命。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线 profiling 脚本（Zenodo artifact 10.5281/zenodo.19326669，prowhammer-artifact.tar.gz；Ubuntu 24.04、CUDA 12.8、Python 3.12/Anaconda；GPU RTX A6000/4090/5060、≥8 核 CPU、≥16GB DRAM、~70GB 磁盘；准备 2–3 小时、实验 6–9 小时）。使用：`cd profiling_bit_flip_location && bash get_golden_lib.sh prowhammer` → `bash run_profile_custom.sh / run_profile_cublas.sh / run_profile_ggml.sh` → bitflip_data.csv → 攻击阶段用 Rowhammer 在 hDRAM 中翻转选定 bit → 受害者使用被篡改库。局限：结果依赖 (库版本, GPU 架构) 对（kernel 选择随 autotune 变化），需逐组合重复 profiling；剪枝不保证找到全部可利用位，只保证足够子集；profiling 需 GPU 散热良好（过热导致驱动崩溃、中断实验）。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU

## Top-K Sparse Matrix-Vector Multiplication（Top-K SpMV，稀疏矩阵向量乘 + Top-K 选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-K SpMV 是稀疏 embedding 相似度检索的核心算子：给定稀疏矩阵 A∈R^(m×n)（每行一个数据库候选向量）与稠密查询向量 v∈R^n，计算 y=A·v 后返回最大 K 个 y 值的索引集合 TopK(y,K)={i_1,...,i_K}（y_ij≥y_r 对所有 j 且 r∉TopK）。在检索系统中 A 的行是稀疏 embedding（RAG、推荐、知识图谱中的实体/文档/商品向量），v 是查询，内积即相似度。问题同时含两部分：稀疏矩阵-向量乘（SpMV）与 Top-K 选择。工作量三大特征（Fig.2）：(1) 不规则访存——稀疏矩阵访问随机、缓存命中率低、预取无效，CPU 上内存延迟占执行时间 60–70%；(2) 计算负载不均——各行非零数分布不均导致并行核间失衡；(3) 稀疏输出——K≪m，需堆或预过滤结构在线维护 Top-K。近似精度用 Recall@K=|TopK_approx∩TopK_exact|/K 衡量（TopK_exact 为全精度穷尽 Top-K SpMV 结果）。论文指出处理约 20% 候选向量即可达 ~80% Recall（Pareto 规律），启发"选择性计算"范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 Top-K SpMV 检索的算法 pipeline（ParetoES 的选择性计算版本，区别于全计算）：
```
# 离线：聚类索引构建（A 行按簇重排、量化、剪枝、编码）
clusters = SphericalKmeansPPRefine(A, K=nlist)      # 聚类 + 动态精化
A_q = INT6_quantize(A)                               # 对称 6-bit 量化
A_q = ReSparse_prune(A_q)                            # 非零粒度剪枝
submat = partition_by_cluster(A_q)                   # 按簇 ID 重排为子矩阵
# 在线：单查询 v
v_q = INT6_quantize(v)
nprobe = lookup_min_nprobe(recall_target=0.8)        # 满足 Recall@100≥0.8 的最少簇数
top_probe = topk({<mu_c, v_q> for c in clusters}, nprobe)   # 质心相似度选簇
y = 0
for c in top_probe:
    for (i, j, val) in submat[c]:  y[i] += val * v_q[j]    # 簇内稀疏 SpMV
result = topk(y, K=16) per core -> aggregate top-512      # 核内 Top-16 聚合
```
张量计算例子：对选中簇子矩阵，y_i=⟨A_i,v⟩=Σ_{j∈nnz(A_i)} A_i[j]·v[j]，仅在非零元素上做乘加（INT6×INT6），跳过全零列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPU/GPU 通用实现：MKL 的 mkl_sparse_?_mv + sort、sparse_dot_topn（ing-bank，Python 稀疏矩阵乘 + top-n 选择，精确全计算基线）、cuSPARSE（SpMV kernel）+ Thrust（sort-select pipeline）、Faiss（IndexIVFFlat 倒排索引 + nprobe 可调，支持选择性计算但基于 K-means 且非稀疏无关）。FPGA 实现：FPGA32（BS-CSR 块流式）、AccelES（Ultra-CSR/Random-CSR + 低比特 + ReSparse）、ParetoES（ACPE 多核 + DMSU Bitonic-16 微排序）。ParetoES 实验：Recall@100∈[0.8,1.0] 下 QPS 最高 4761.9（Sp.Baidu），比 CPU/GPU baseline 高至 540×/79×，平均 2.27× vs AccelES。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## Spherical K-means++ Refine（球面 K-means++ 精化聚类）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Spherical K-means 是 K-means 的方向敏感变体（Dhillon 2001，Hornik 2012 的 R 实现）：把向量与质心归一化到单位范数、用余弦相似度（内积）替代 L2 距离度量方向对齐，用于高维稀疏文本/词向量聚类。传统 K-means 在稀疏高维空间的缺陷：(1) 距离度量失配——L2 范数被零元素主导、掩盖判别信息，把仅方向不同的语义相关向量错误聚类；(2) 质心漂移——稀疏向量的算术均值引入噪声、降低簇代表性。ParetoES 的 Spherical K-means++ Refine 在其上叠加三处：(a) K-means++ 初始化（Arthur & Vassilvitskii 2006）——按与已选质心最小余弦距离成比例的 p(x_i)=(1−max_cj x_iᵀc_j)/Σ(1−max x_kᵀc_j) 采样，从 m=min(0.01n,10000) 随机子集采质心，把单次迭代复杂度从 O(nd) 降到 O(md)、总复杂度 O(mKd)；(b) 质心更新取簇内最接近均值归一化向量的成员（argmax_{x∈c_i} x·normalize(Σx)），缓解质心漂移；(c) 动态精化（Post Refine）——merge/split 策略：质心间 cos>θ_merge=0.9 的近重复簇对合并为新簇（新质心取并集中最接近均值者），簇内平均 cohesion=mean(xᵀμ_k)<θ_split=0.6 时用 2-means 分裂并迭代重分配直至两子簇 cohesion≥0.6 或达 max_refine_iter。实验：稀疏空间下聚类内距低 75.8%、紧凑性差 82.6%，精化后（+ReSparse）在相同 Recall 下扫描更少向量（Fig.9 优于 K-means/Spectral/Hierarchical）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ParetoES Algorithm 1 的伪代码（精简）：
```
Input: 稀疏向量 {x_i}, 初始簇数 K, θ_merge=0.9, θ_split=0.6, max_refine_iter
# 初始化：K-means++ 采样 K 个质心
centroids = []
while len(centroids) < K:
    p(x_i) = (1 - max_{c in centroids} x_i^T c) / sum_k (1 - max_{c} x_k^T c)
    sample x_i ~ p(x_i)      # 从 m=min(0.01n,10000) 子集
# 迭代：归一化 + 分配 + 质心更新
repeat until convergence:
    x_i = x_i / ||x_i||_2
    c_i = argmax_k (x_i . mu_k)                    # 余弦分配
    mu_k = argmax_{x in c_k} (x . normalize(sum(x))) # 最接近均值向量
# 动态精化
while max_{i≠j} cos(mu_i, mu_j) > 0.9:             # merge
    (i*,j*) = argmax cos; mu_new = argmax_{x in c_i* ∪ c_j*} (x . normalize(sum))
    merge c_i*, c_j* -> c_new
for each c_k with mean(x^T mu_k) < 0.6:            # split
    2-means -> c_k1, c_k2; repeat max_refine_iter: reassign + update
    if cohesion(c_k1)>=0.6 and cohesion(c_k2)>=0.6: replace c_k
```
pipeline 角色：作为离线索引构建第一步，产出 K=nlist=⌊√(m/2)⌋ 个簇与质心，供在线检索做质心相似度簇筛选（Top-nprobe），决定"扫哪些子矩阵"——聚类质量直接决定同 Recall 下的扫描簇数（论文：稀疏 vs 稠密需多扫 1.83× 簇，精化后减少冗余激活）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现：Faiss 的 IndexIVFFlat 用 K-means（L2）做量化器、scikit-learn KMeans/SphericalKMeans、Hornik 的 R skmeans 包。ParetoES 在 NVIDIA A100 GPU 上实现全部聚类（nlist=√(m/2)、迭代上限 1000、收敛阈值 10⁻⁴ 按质心位移 L2 范数），全精度浮点执行（与检索端 INT6 混合精度）。对比基准用 Faiss v1.7.2 的 IndexIVFFlat（METRIC_INNER_PRODUCT + IndexFlatIP quantizer）、Spectral Clustering、Hierarchical Clustering。效果：精化版（Spherical Refine + ReSparse）在 Recall@100=0.8 约束下扫描向量更少（Fig.9），同 nprobe 下 Recall 更高。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## ReSparse（非零粒度非结构化剪枝，Enhanced ReSparse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReSparse 是 AccelES（HPCA 2025，同组先前工作）提出的稀疏 embedding 检索非结构化剪枝算法：在非零元素（non-zero）粒度上，把幅度小于阈值的小非零元素置零，以减少冗余计算与访存。其核心直觉：稀疏向量中小幅度非零对相似度排序影响有限（Retrieval 场景下剪掉它们几乎不损失 Recall）。ParetoES 首次把非结构化剪枝应用于"选择性计算"Top-K SpMV 范式（此前仅用于全计算范式），并做两处结构性增强：(1) 剪枝阈值计算改为"仅对非零元素取均值"——原始 ReSparse 用全局均值作阈值，被大量零元素拉低，导致高稀疏矩阵下大部分非零被保留、剪枝几乎失效（Sp.10M 密度 0.72% 时剪枝率仅 2.34%）；改为非零均值后剪枝率恢复（Sp.10M 达 11.48%）；(2) 剪枝比例计算与 Spherical K-means++ Refine 迭代集成，并在 Recall@100≥80% 约束下确定每数据集最小 nprobe 以保证精度。效果：增强版 ReSparse 峰值剪枝率 61.25%、平均 37.41%（原始 23.93%），相对 AccelES 平均再减非零 18.09%（最高 39.14%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
剪枝伪代码（增强版，以簇为单位离线执行）：
```
for each cluster c:
    nz = nonzeros(submat[c])                       # 簇内非零元素集合
    thr_c = alpha * mean(|v| for v in nz)          # 仅非零均值，alpha 为比例系数
    for (i, j, v) in nz:
        if |v| < thr_c:  submat[c][i,j] = 0        # 幅度低于阈值 -> 剪掉
    prune_ratio[c] = pruned_nnz / total_nnz
# 与 Refine 集成：在 Recall@100>=0.8 约束下迭代选 nprobe，
# 使剪枝后的簇子矩阵仍满足目标召回
```
张量计算例子：某簇含 1000 个非零、均值为 3.2，阈值取 0.5×3.2=1.6，则幅度 <1.6 的非零（如 0.3、1.1）置零，后续 SpMV 跳过这些位置。对 FPGA 而言剪掉的非零直接减少 HBM 流式读取字节与 DSP 乘加次数（sparse-agnostic 架构下剪枝收益与稀疏度成正比）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AccelES 在 Ultra-CSR/Random-CSR 编码前离线剪枝，配合低比特量化（INT6）实现 73.5% 平均访存减少与 2.7× 计算并行度（HPCA 2025 数据）；ParetoES 把它集成进聚类-量化-剪枝-编码预处理流水（A100 GPU 上执行，全精度聚类后、INT6 量化后剪枝），剪枝后矩阵按簇重排为子矩阵、编码 Ultra-CSR 载入 FPGA HBM。使用注意：剪枝率需按数据集/Recall 目标标定，过度剪枝会掉 Recall（论文在 Recall@100≥80% 约束下选最小 nprobe）；对非 sparse-agnostic 的 CPU/GPU 平台，剪枝收益难以兑现（Faiss+ReSparse 在 CPU 仅 +14%、GPU 反而 -7%）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## INT6 对称均匀量化（Low-bit Uniform Integer Quantization，6-bit 定点检索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT6 是对称均匀线性量化：把浮点值 x 以动态缩放因子 α=max(|x|) 线性映射到 6-bit 有符号整数区间 [-31,31]，Q_x=round(31·x/α)（S=31，无零点和无裁剪）。ParetoES 用它做稀疏 embedding 检索的低比特格式，与 Ultra-CSR 编码结合使有效内存带宽较 FP32 提升 6×（每非零从 32-bit 降到 6-bit）。位宽权衡：INT6 保持 Recall@100 不变，而 5-bit/4-bit 分别掉 Recall 最多 10%/32.5%。与 AccelES 的关键差异：ParetoES 不裁剪（no clipping）——裁剪虽可保 Recall 但会压制高范数分量、扭曲方向相似度（余弦/内积语义），不裁剪保留角度缩放是更保几何的选择。混合精度策略：聚类全精度浮点执行，聚类后向量/质心/查询一次性量化到 6-bit，使在线检索全程整数内积。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化的张量计算：
```
# 离线：对每个稀疏向量 x（或质心/查询），动态缩放
alpha = max(|x_j| for j in nnz(x))          # 每向量一个缩放因子
Q_x[j] = round(31 * x[j] / alpha),  for j in nnz(x)   # -> [-31, 31]
# 在线：内积在量化域近似
<A_i, v> ≈ (alpha_i * alpha_v / 961) * sum_j (Q_A[i,j] * Q_v[j])
# 因排序只需相对大小，实际检索直接比较 sum_j Q_A[i,j]*Q_v[j]（尺度因子单调共享）
```
pipeline 位置：离线端（聚类之后、ReSparse 之后/同时）量化全部矩阵元素并连同缩放因子编码进 Ultra-CSR packet；在线端查询向量同样量化后下发 FPGA。INT6 使每 512-bit HBM packet 容纳 30 个非零（FP32 下仅 ~11），DSP 单周期可做 3 个 6-bit 乘法（相对 FP32 并行度 2.7×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：对称量化在 FPGA DSP 上以 6-bit 整数乘法直接执行（Alveo U280 DSP48 支持小位宽乘法复用）；缩放因子 α 存每行/每向量头（Ultra-CSR 元数据），检索排序只依赖量化内积的相对大小。使用场景：任何内存受限的稀疏检索/SpMV 加速器（AccelES-INT6 同方案，FPGA32 为 FP32 对比）。注意事项：无裁剪依赖 α=max 的动态缩放捕获全局动态范围，若个别向量有极端 outlier 会导致该向量其余元素量化分辨率下降（论文未展开讨论此 trade-off）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## 选择性计算与簇探测（Selective Computation / Cluster Probing with nprobe）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
选择性计算（selective computation）是 ParetoES 的核心范式：不做全库穷尽 Top-K SpMV，而是先按聚类质心相似度选出 Top-nprobe 个最相关簇，只在这些簇对应的子矩阵上计算相似度，用少量扫描换取 Recall@100∈[0.8,1.0] 的可接受召回。动机（Pareto 规律）：处理约 20% 候选向量即可达 ~80% Recall；生产 RAG/推荐系统在 80–90% Recall 下几乎无损（RAG 掉 Recall 到 80% 仅损 0.3–3.6% 准确率，推荐系统 GMV 从 92.5%→85.5% Recall 仅 -1.2%），而 1s 延迟上升可致收入 -7~10%。对比：Faiss 的 nprobe 也是"扫前 nprobe 个倒排桶"，但基于 K-means（L2）聚类质量差、且 SIMD/warp 架构对稀疏不规则访存低效；FPGA 全计算加速器（AccelES/FPGA32）固定延迟、无法运行时用 Recall 换吞吐。ParetoES 用硬件原语实现选择性计算：ACPE 的 Bitonic-16 质心筛选 + Mem Map 调度器按 sub_nprobe 动态取簇块，nprobe 参数通过软件接口（sub_nprobe=⌈nprobe/32⌉/核）运行时配置、无需重综合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
选择性计算检索流程（单查询）：
```
# 簇筛选（硬件：Bitonic-16 质心排序）
scores_c = [<mu_c, v_q> for c in 0..C-1]          # 与全部质心内积（质心在 HBM 通道头部）
top_probe = topk(scores_c, nprobe)                  # 选出 nprobe 个簇（软件配置）
# 簇内评估（仅扫描选中簇子矩阵）
for core in 0..31:                                  # 32 ACPE 并行
    for t in 0..sub_nprobe-1:                       # 每核承担 sub_nprobe=ceil(nprobe/32) 个簇
        addr = LUT[core][t]                          # Mem Map 查表定位簇块地址
        stream_cluster_block(addr) -> decode -> INT6 MAC -> aggregate
    local_top16 = bitonic16(topk, K=16)
global_top512 = merge(all cores' top16)             # host 聚合
Recall = |top512 ∩ exact_topK| / K
```
张量计算：只在选中簇的稀疏子矩阵上算 y_i=⟨A_i,v⟩（跳过未选中簇的整块矩阵，消除 68%/44%/4% 访存+计算对应 Recall 0.8/0.9/1.0）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现：Faiss IndexIVFFlat 的 nprobe 参数即此概念（倒排桶探测）。ParetoES 硬件实现：ACPE 把簇筛选（Bitonic-16 排序质心分数）与簇内 Top-K（DMSU 局部 Top-16）做成固定流水，sub_nprobe 由 host 在初始化时计算下发（软件-硬件协同设计，避免参数调优触发重综合）；Recall 目标通过查表映射到 nprobe（论文在 Recall@100≥80% 下求每数据集最小 nprobe）。使用时在 Recall 与吞吐间沿 Pareto 前沿调节：Recall@100≈0.8/0.9/1.0 时 Sp.Baidu QPS=4761.9/2857.1/1851.9（nprobe=128/224/384，nlist=398），相比 AccelES 固定 1818.2 QPS（无选择性计算）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## MBQC（Measurement-Based Quantum Computation，基于测量的量子计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MBQC（测量基量子计算，又称 one-way quantum computation 单向量子计算）是与传统的门模型（gate-based）量子计算并列的一种量子计算模型：计算不通过执行量子门序列进行，而是先制备一个高度纠缠的多体量子态——图态（graph state），然后对图态上的量子比特执行按顺序的、依赖于先前测量结果的单比特测量（adaptive single-qubit measurements），测量结果即计算输出。关键事实：图态对 MBQC 是普适的（universal）——只要图态被制备出来，任何量子计算都只需单比特测量即可完成（Raussendorf-Briegel 2001）。核心逻辑链：(1) 程序被表示为图态；(2) 图态由小的资源态通过 fusion 操作拼接而成；(3) 测量图态上的量子比特完成计算——测量把纠缠"消耗"为计算结果；(4) 测量基的选择可以是 adaptively 的（feed-forward），把前面测量的结果作为后续测量的条件。在论文（MemTree）中，MBQC 是光子量子计算机（PQC）的计算模型：与超导、中性原子等门模型硬件不同，光子硬件天然适配 MBQC——光子是飞行比特，图态在发射过程中边生成边测量，无需在空间上驻留。PQC 编译的核心挑战因此变为"如何稳健高效地生成目标图态"。论文将 MBQC 作为背景提出：graph state 的定义 |G⟩=∏CZ_{(i,j)}|+⟩^⊗V，fusion 是拼接小图态生成大图态的关键操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MBQC 计算 pipeline（以论文中的 QAOA 程序为例，图态生成 → 测量执行）：
```
# 阶段1：图态生成（本论文的研究重点）
target_graph = compile_program_to_graph(QAOA_circuit)   # 程序 -> 图态 G=(V,E)
caterpillars = emit_from_spin_memory(target_graph)       # 光子源发射 caterpillar 态
big_graph = fuse(caterpillars)                            # Type-II 融合拼接成大图态
# 阶段2：单比特测量执行计算
for t in measurement_schedule:
    outcome_t = single_qubit_measure(q_t, basis_t)       # 按图态结构逐比特测量
    basis_next = feed_forward(outcome_t)                  # 测量基自适应更新
# 输出 = 测量结果序列（经典后处理得到程序结果）
```
张量/量子态计算：图态 |G⟩ = ∏_{e(i,j)∈E} CZ_{(i,j)} |+⟩^{⊗V}——每个顶点初始化为 X 本征态 |+⟩，每条边施加 CZ 门。一次 X 基测量把该量子比特从图态中移除并断开其所有纠缠边；一对相邻 X 测量把两个量子比特移除并在它们各自的邻居之间建立直接连接（即"测量即计算"——纠缠在测量下被传输和消耗）。论文在 spin memory 架构上实现此 pipeline：caterpillar 态是 MBQC 的资源态，融合完成后按时间方向测量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件侧——光子量子计算机（PQC）天然实现 MBQC：PsiQuantum（全光子，线性光学 + SPDC 源）、Quandela（自旋内存架构，半导体量子点发射器 + 线性光学融合）、以及 emitter-based（量子发射器，理论确定性生成）三大架构都以"生成图态 → 融合拼接 → 测量"为执行模型；(2) 软件/编译侧——MBQC 编译器把门模型程序（如 Qiskit 电路）转译为图态生成 + 测量调度：OneQ（ISCA'23）、OnePerc（ASPLOS'24）、OneAdapt（MICRO'25）、RLGS（ISCA'25）、以及本论文的 MemTree。使用场景：光子平台（室温运行、退相干时间长、天然适合量子网络集成）；"图态生成"环节决定编译器的核心优化空间（融合次数、光子利用率、错误容错）。论文未开源；其真实硬件实验基于 Quandela 云平台 + Perceval 工具包。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Graph State（图态）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
图态（graph state）是一类特殊的多体纠缠态，其纠缠结构完全由一张图 G=(V,E) 决定：V 的每个顶点对应一个量子比特，E 的每条边对应一对量子比特之间的 CZ（受控-Z）纠缠门。形式定义 |G⟩ = ∏_{(i,j)∈E} CZ_{(i,j)} |+⟩^{⊗V}：所有量子比特初始化为 X 本征态 |+⟩，对每条边施加 CZ 门。图态在量子计算中的核心地位来自 Raussendorf-Briegel 定理：图态对 MBQC 是普适的——图态一旦生成，仅靠单比特测量即可实现任意量子计算。图态的另一个重要性质是 stabilizer 结构：每个顶点 i 对应一个 stabilizer 生成元 X_i∏_{j∈E(i)} Z_j（顶点 i 的 X 算符乘以其所有邻居的 Z 算符），这一性质是"间接 Z 测量"容错方案（见对应条目）的数学基础。在论文中，图态是量子程序在 PQC 上的执行载体：程序 → 目标图态 → 由 caterpillar 态经 fusion 拼接生成 → 测量执行。图态上的测量规则：Z 基测量移除目标量子比特并断开其所有纠缠边；一对相邻 X 基测量移除两个量子比特并在其邻居之间直接相连。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
图态在论文 pipeline 中的角色与计算过程：目标图态 g_prog 是 MemTree 编译器的输入/中间对象——(1) 编译器把量子程序（如 36-qubit VQE）映射为图态 g_prog（顶点=qubit，边=CZ 纠缠）；(2) MIP 模型把 g_prog 划分为线性子图集合 G^l（约束 deg(v)≤2）；(3) 每个线性子图由 caterpillar 态通过树编码逻辑量子比特组装；(4) 线性子图经 Type-II 融合拼接还原为 g_prog。图态测量计算的张量例子（3 顶点线性图态 |G⟩=CZ_{12}CZ_{23}|+++⟩）：
```
# 图态制备（张量积形式，3-qubit 线性链）
|G> = CZ(1,2) * CZ(2,3) * (|+>⊗|+>⊗|+>)
# 图态测量（MBQC 计算）
X_measure(q1)  # 移除 q1，q2 与其邻居断开
X_measure(q3)  # 移除 q3
# 剩余: q2 单独 -> 完成一次单比特逻辑计算（如 teleport 式传输）
```
在融合语境中：Type-II fusion 把两个小图态的顶点合并成更大的图态（两个输入顶点都被移除，一个顶点的邻居连接到/断开另一顶点的邻居）。图态的"已知/未知"结构是融合失败与融合擦除的本质区别：失败 = 已知图变换（可恢复），擦除 = 未知图结果（必须额外保护）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 光子硬件侧——图态由光子源发射的纠缠光子经线性光学（HWP 半波片 + PBS 偏振分束器）融合生成：all-photonic 架构用 SPDC 贝尔对拼接任意拓扑图态；spin memory 架构用 InGaAs 量子点发射器生成 caterpillar 图态（分支链结构）；emitter-based 用相互作用的量子发射器理论确定性生成。(2) 软件侧——图态是 MBQC 编译器的核心 IR：OneAdapt 生成资源状态层（RSL）并 normalization 成 2D 图态层；MemTree 用 MIP 图划分 + 平衡二叉树生成。使用场景：图态是"以纠缠换时间"的模型——制备图态的开销（融合次数、光子数）与图的结构（度、线性化难度）直接相关，因此"把目标图划分为可高效生成的结构"是编译优化的核心。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Type-II Fusion（Type-II 融合操作）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Type-II fusion 是光子量子计算中把两个较小的图态合并为更大图态的线性光学操作，是图态生成（graph state generation）中最关键的操作，因为它使图态的资源高效并行生成成为可能。与 Type-I fusion（两个输入顶点合并为一个，继承双方边）不同，Type-II fusion 更复杂：两个输入顶点都被移除，一个顶点的邻居连接到另一个顶点的邻居（若先前未连接）或与之断开（若先前已连接）。Type-I 和 Type-II fusion 都可通过线性光学概率性实现（HWP 半波片 + PBS 偏振分束器）。论文聚焦 Type-II fusion，因为它支持 heralded photon loss（光子丢失可被 heralded/预警），这是设计擦除容错的前提。Type-II fusion 有三个结果（如图 1 所示）：两个融合量子比特被探测器捕获在不同侧 → 融合成功；被捕获在同一侧 → 融合失败（failed qubits 被有效测量在 Z 基并断开，图结构对编译器已知）；其中一个量子比特未被捕获（光子丢失）→ 融合擦除（输出图态结构未知）。融合成功率理论上限 50%（可通过额外光学硬件提升至 75% 或更高）。论文的融合错误模型：1-p_fail=0.75（无擦除时的成功率，需额外干涉装置 [18][22][49]），σ_fus=(1+V_HOM)/2=99.75%（HOM 可见度 99.5%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Type-II fusion 在论文 pipeline 中的具体计算过程（树编码融合的一个分支）：
```
# 输入：图态 A 的叶子量子比特 q_i^c(A)、图态 B 的叶子量子比特 q_i^c(B)
# 线性光学：HWP + PBS，双光子 Hong-Ou-Mandel 干涉
def type2_fusion(qA, qB):
    # 两个光子模式经过 PBS：水平/垂直偏振分束
    det1, det2 = PBS_interference(qA, qB)
    if det1 != det2:            # 两个光子在不同探测器侧
        return SUCCESS           # 融合成功：邻居互联，图结构确定
    elif det1 == det2:           # 两个光子在同一侧
        return FAILURE           # 融合失败：两 qubit 被 Z 基测量移除，图结构已知
    else:                        # 某个光子未被捕获（丢失）
        return ERASURE           # 融合擦除：图输出不确定，需间接测量保护
```
图论效果：成功时——两个输入顶点 v1、v2 被移除，v1 的邻居集合 N(v1) 与 v2 的邻居集合 N(v2) 直接相连（或断开重叠边）；失败时——v1、v2 被移除且无新连接（等同于 Z 测量）；擦除时——结果未知。在树编码中，融合失败时 q_i^c 被测量掉而 q_i^a/q_i^b 保留（Z 测量移除 q_i^b 留备份 q_i^a）；擦除时用间接 Z 测量恢复。论文真实硬件实验中用 Perceval 搭建 fusion 电路：双轨编码（dual-rail）下融合电路 = 两 qubit 光子模式的置换 + 相移 + 两个分束器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：线性光学元件（HWP、PBS、探测器）构成干涉仪。文献中的实现细节存在差异（是否在第一个 PBS 前插入 HWP 导致成功时的局部酉校正与失败时有效测量不同），论文采用 [25] 的方案。成功率增强：(1) 冗余编码（m 次融合尝试，P_fail=p_fail^m）；(2) repeat-until-success（成功后终止，需 ancilla 光子）；(3) 本论文的树编码（b 个分支 + 间接 Z 测量，S_tree=1-(1-(1-p_eras)^2+p_fail)^b）。额外干涉测量装置（ancilla-assisted Bell measurement）可将成功率提升至 75%（Grice 2011；Ewert & van Loock 2014）。使用场景：Type-II fusion 是三类 PQC 架构（all-photonic、emitter-based、spin memory）共同的图态拼接原语；论文在 Quandela 云平台用 Perceval 实现真实 fusion 电路（实测 HOM 不可区分度 92.0%、透射率 5.16%）。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Fusion Failure（融合失败）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
融合失败（fusion failure）是 Type-II fusion 的两种主导错误源之一：当两个融合量子比特被探测器捕获在同一侧时（同侧探测器同时触发），表明期望的纠缠未被建立，融合失败。关键性质：融合失败是 heralded 的（可预警的）且产生已知的图变换——失败的量子比特被有效测量在 Z 基并从图中断开，因此尽管融合尝试未成功，编译器仍然确切知道结果图的结构。正是这个"结果已知"的性质，使 OneAdapt、OnePerc 等先前的 PQC 编译器能够仅通过 normalization/重试策略处理融合失败。数学上，融合失败概率记为 p_fail；融合成功概率 1-p_fail=0.75（无擦除时，采用 OneAdapt 论文 Sec 5.1 的错误模型，通过额外干涉测量装置实现）。在 boosted fusion 设计中，融合失败与融合擦除紧密耦合：增加融合尝试次数（抑制失败）会暴露更多量子比特给光子丢失（增加擦除），因此"只抗失败不抗擦除"的编译器在真实光子丢失条件下是不够的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
融合失败在三种融合方案 pipeline 中的处理（对比逻辑错误率）：
```
# 冗余编码（m 次尝试）：P_fail = p_fail^m，但 P_eras = 1-(1-p_eras)^(2m) 指数恶化
# RUS（重复直至成功）：P_fail = p_fail^m, P_eras = Σ_{i=0}^{m-1} p_fail^i * 2p_eras
# 树编码（本论文，b 分支）：每分支失败不影响其它分支
for branch i in 1..b:
    if type2_fusion(...) == FAILURE:
        # q_i^c 被测量掉；q_i^a/q_i^b 留在树中
        Z_measure(q_i^b)          # 移除 q_i^b，留 q_i^a 作备份（供全分支失败时重试）
        # 分支 i 不再参与，但其它分支继续独立尝试
```
图论结果：失败 = 已知图变换——两个融合量子比特被 Z 基测量移除，无新连接产生，编译器知道确切图结构。例：p_fail=0.25、p_eras=0 时，单次融合成功率 75%；m=5 冗余编码把失败率压到 (0.25)^5≈0.1%，但若 p_eras>0，逻辑擦除率 1-(1-p_eras)^10 迅速占据主导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/处理方式：失败由探测器同侧触发判定（heralded），编译器收到失败信号后：(1) OneAdapt——通过 normalization 把资源状态层重排成有效 2D 图态层，容忍失败；(2) 冗余编码——m 次尝试任一次成功即成功；(3) RUS——成功后立即终止，但消耗更多 ancilla 和时间；(4) 本论文树编码——分支独立失败 + 备份量子比特重试。使用场景：任何基于概率性融合的 PQC 图态生成都必须处理融合失败；论文把融合失败率设为 p_fail=0.25（1-p_fail=0.75）作为统一对比条件。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Fusion Erasure（融合擦除）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
融合擦除（fusion erasure）是 Type-II fusion 的另一种主导错误源，由光子丢失（photon loss）引起：融合过程中一个融合量子比特无法被探测器捕获（光子丢失），因此融合操作的结果保持未知。融合擦除比融合失败更恶劣，因为：(1) 被擦除的量子比特不再可用于计算，且其影响无法用直接 Z 测量移除；(2) 更关键的是，融合输出图态变得不确定——无法知道纠缠是否被建立，而 MBQC 中后续测量依赖精确的图态结构，因此被破坏的融合输出必须被丢弃（除非施加额外保护）。在真实 PQC 硬件上观测到的擦除率 p_eras≈10%，1% 的擦除率已足以让 OneAdapt 的 84×84 资源状态层（需 >10^5 次融合）几乎必然经历擦除。数学上，冗余编码方案对擦除的暴露为 P_eras=1-(1-p_eras)^(2m)（m 个物理 qubit 的每个逻辑 qubit 独立暴露），RUS 为 P_eras=Σ p_fail^i·2p_eras——两者都随编码参数增长而恶化。本论文的核心贡献——树编码融合——专门设计"间接 Z 测量"来容错融合擦除。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
融合擦除在树编码 pipeline 中的处理（论文 Fig.4(d) 方案 3）：
```
# 分支 i 的叶子 q_i^c 经历光子丢失（擦除）
if type2_fusion(...) == ERASURE:
    X_measure(q_i^b)      # 对相邻量子比特做 X 测量
    Z_measure(q_i^a)      # 对其余相邻量子比特做 Z 测量
    # 效果：基于 stabilizer X_i∏_{j∈E(i)}Z_j，确定性地揭示
    # 若直接对 q_i^c 做 Z 测量会得到的结果 —— 无损消除 q_i^c
    # 其它分支与根量子比特 q_root 完全不受影响
```
对比基线（论文 Fig.4(f) 模拟，10^3 次融合试验）：固定融合失败率，扫描擦除率 p_eras——树编码 S_tree=1-(1-(1-p_eras)^2+p_fail)^b 在 p_eras 升高时远优于冗余编码（P_eras=1-(1-p_eras)^(2m) 指数恶化）与 RUS。例子：p_eras=10%、p_fail=25%、b=m=4：冗余编码 P_eras≈1-(0.9)^8≈57%；树编码每分支 (1-p_eras)^2=0.81、分支失败概率 0.19+0.25=0.44、总失败率 0.44^4≈3.7%——擦除从主导错误变为可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：擦除由探测器"缺失"触发判定（光子未被任何探测器捕获）。处理策略：(1) 本论文——树编码 + 间接 Z 测量（X 测量相邻 qubit + Z 测量其它相邻 qubit，利用 stabilizer 关系无损揭示被擦除 qubit 的 Z 测量结果），这是首个在 PQC 编译器中系统处理融合擦除的方案；(2) 论文自设计的 OneAdapt-ET——在 OneAdapt 中集成间接 Z 测量（对 normalization 路径外的相邻自由 qubit 做 X 测量 + 对其余相邻 qubit 做 Z 测量）；(3) 先前的编译器（OneAdapt、OnePerc、OneQ、RLGS）完全忽视融合擦除。使用场景：任何存在光子丢失的光子量子计算平台（真实硬件 p_eras≈10%，仿真 0%~10%）；擦除率 1% 时 OneAdapt 已不可用，论文在 0%~5% 区间评估。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Tree-Encoded Fusion（树编码融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
树编码融合（tree-encoded fusion）是本论文提出的核心新方案：一种同时容错融合失败与融合擦除的逻辑量子比特编码与融合方案，受 QEC 树码（tree cluster-state code）和冗余编码融合双重启发。方案要点：参与 Type-II fusion 的逻辑量子比特 A、B 被编码为树结构——根量子比特 q_root 连接 b 个分支，每个分支是 3 个量子比特的线性图 {q_i^a, q_i^b, q_i^c}；叶子 q_i^c 用于融合测量，q_i^a/q_i^b 是间接测量辅助量子比特。按融合结果执行不同测量模式：(1) 成功——对 q_i^a、q_i^b 做一对 X 测量，把成功融合的纠缠直接连到 q_root；(2) 失败——q_i^c 被测量掉，q_i^a/q_i^b 留在树中，对 q_i^b 做 Z 测量移除它，留 q_i^a 作备份；(3) 擦除——对 q_i^b 做 X 测量、q_i^a 做 Z 测量，实现 q_i^c 的间接 Z 测量（基于 stabilizer X_i∏Z_j），无损消除被擦除 qubit 的影响；(4) 全部分支失败/擦除的极端情况——用 (2) 留下的备份 q_i^a 再试一次。逻辑成功率 S_tree = 1-(1-(1-p_eras)^2+p_fail)^b。树结构可从 caterpillar 态高效组装（主路径 q_root + 叶 qubit + 经 Z 测量分离的 4-qubit 线性图融合），与 spin memory 架构天然契合。参数选择 b=4、b_prep=6（30-qubit caterpillar 限制下 photon 源与执行时间的 trade-off）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
树编码融合的完整算法流程（伪代码，含间接 Z 测量恢复）：
```
# 逻辑融合：逻辑量子比特 A、B（各编码为 b 分支树）
for branch i in 1..b:
    outcome = type2_fusion(q_i^c(A), q_i^c(B))
    if outcome == SUCCESS:
        X_measure(q_i^a); X_measure(q_i^b)   # 成功纠缠直接连到 q_root
    elif outcome == FAILURE:
        Z_measure(q_i^b)                      # 移除 q_i^b，留 q_i^a 作备份
    elif outcome == ERASURE:
        X_measure(q_i^b); Z_measure(q_i^a)    # 间接 Z 测量 q_i^c：stabilizer X_i∏_{j∈E(i)}Z_j
# 备份机制：若全部分支失败/擦除且存在备份
if no_branch_succeeded and backup_exists:
    fusion_retry_with(q_i^a)
# 制备侧（caterpillar 组装，b_prep 次并行尝试）：
for attempt in 1..b_prep:
    branch = prepare_branch_from_caterpillar()   # 4-qubit 线性图融合到叶 qubit
    if branch_fusion_failed: discard (自动测量掉)
    if branch_fusion_erased: indirect_Z_measure({q_i^b, q_i^e})   # 恢复
# 若成功分支 < b，下一 timestep 重试；参数 b=4, b_prep=6
```
成功率公式对比：S_redun=(1-p_fail^m)(1-p_eras)^(2m)、S_rus=1-Σp_fail^i·2p_eras-p_fail^m、S_tree=1-(1-(1-p_eras)^2+p_fail)^b。例：p_fail=0.25、p_eras=0.1、b=4：S_tree≈1-(0.44)^4≈96.3%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 逻辑量子比特制备——在 caterpillar 态上组装：caterpillar 主路径提供 q_root 与 b 个叶 qubit；另生成 b 个 4-qubit 线性图（从长线性图经 Z 测量分离），用融合拼到叶子上；制备参数 b_prep=6（b_prep>b），同 timestep 并行尝试，失败分支自动测量掉、擦除分支用间接测量恢复；(2) 真实硬件——Quandela 云平台 + Perceval 构建双轨编码光学电路（光子模式置换 + 相移 + 分束器），融合结果触发 FFCircuitProvider 的条件前馈（对 q_i^a/q_i^b 施加 X 或 Z 测量）。实验验证：83.3% 单 timestep 制备成功率、97.1% 两 timestep 内；相对 redundantly-encoded 执行时间 1.9×10^-3×、相对 RUS 1.7×10^-2×（光子源多 2.55×/1.63×，用空间换时间）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Indirect Z Measurement（间接 Z 测量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
间接 Z 测量（indirect Z measurement）是图态测量的一种容错规则，源自 QEC 树码（Varnava-Browne-Rudolph 2006 的 loss tolerance 协议）与图态 stabilizer 结构：若目标量子比特 j_0 因光子丢失（擦除）无法直接测量，可以选一个与 j_0 相邻的量子比特 i，对它做 X 测量，再对连接到 i 的其它所有量子比特 j_1, j_2,... 做 Z 测量；基于图态 stabilizer 生成元 X_i∏_{j∈E(i)}Z_j，这组测量确定性地揭示"若对 j_0 直接做 Z 测量会得到的结果"。也就是说，被擦除的量子比特的 Z 测量结果可以从其邻居的测量中"间接读出"，从而无损地从图态中消除被擦除量子比特的影响——这是融合擦除容错的核心原语。关键前提：图态必须包含足够的辅助结构（邻居 + stabilizer 关系），这正是树编码（为每个叶子配备 q_i^a/q_i^b 辅助量子比特）与 OneAdapt-ET（利用 normalization 路径外的自由邻居）所构造的。本论文在融合擦除场景中的用法：q_i^c 经历擦除 → 对 q_i^b 做 X 测量 + 对 q_i^a 做 Z 测量 → 等价于间接得到 q_i^c 的 Z 测量结果，q_i^c 被无损消除，其余量子比特不受影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
间接 Z 测量的张量/stabilizer 计算过程（论文 Fig.4(c)，目标 qubit j_0 被擦除）：
```
# 图态 stabilizer：对每个顶点 i，S_i = X_i ∏_{j∈E(i)} Z_j 是 |G> 的保真生成元
# 目标：读出被擦除的 j_0 的 Z 测量结果
# 步骤：
# 1) 选相邻 qubit i，执行 X 测量 -> 得结果 x_i
# 2) 对连接到 i 的其它所有 qubit j_1, j_2, ... 执行 Z 测量 -> 得结果 z_{j1}, z_{j2}, ...
# 3) 由 stabilizer 约束：x_i * ∏ z_j = s_i  (s_i = stabilizer 特征值, 已知)
#    -> 读出 z_{j0} = s_i * x_i * ∏_{j≠j0} z_j
#    即：被擦除的 j_0 的 Z 测量结果被确定性揭示
```
树编码中的应用（分支 i 融合擦除）：q_i^c 被擦除 → X_measure(q_i^b) 得 x + Z_measure(q_i^a) 得 z → 由 stabilizer X_{q_i^b}·Z_{q_i^a}·Z_{q_i^c} 读出 q_i^c 的 Z 结果 → q_i^c 无损移除，q_root 与其余分支不受影响。这使"每分支擦除损失"从全逻辑量子比特崩溃降级为单分支失效，是 S_tree 相对 baseline 指数优势的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：间接 Z 测量在测量调度层面实现——编译器/控制器把"对 q_i^b 做 X、对 q_i^a 做 Z"的测量模式作为擦除结果的前馈动作下发（论文用 Perceval FFCircuitProvider 实现该条件前馈逻辑，feed-forward 延迟 <5 ns）。使用场景：(1) 本论文树编码融合——融合擦除恢复；(2) OneAdapt-ET——对 OneAdapt 中经历擦除的 qubit，在 normalization 路径外找相邻自由 qubit 做 X 测量 + 对其余相邻 qubit 做 Z 测量；(3) QEC 树码（tree cluster-state code，误差修正动物园收录）——一般性的光子丢失容错。注意：间接 Z 测量需要图态中预先存在相邻辅助 qubit（树编码的 q_i^a/q_i^b 就是为此设计的），这是编码开销的来源（光子源多 2.55×/1.63×）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Caterpillar State（毛毛虫态）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
毛毛虫态（caterpillar state）是量子自旋内存架构（spin memory PQC）生成的一种特殊图态：具有分支链（branched-chain）结构——一条线性纠缠的量子比特链作为主路径（main path），每个主路径量子比特额外连接若干叶量子比特（leaf qubits，每条叶子链只连一个主路径顶点，因形似毛毛虫而得名）。定义见 Pettersson-Sørensen-Paesani（PRX Quantum 6, 010305, 2025）；物理制备过程见 Huet 等（Nature Communications 16, 4337, 2025）。制备机制：在半导体量子点（QD，如 InGaAs）腔体上迭代施加纵向声学（LA）激发脉冲，可发射线性纠缠的光子图态；在激发脉冲间插入光学自旋旋转脉冲（OSRP），则发射出 caterpillar 结构——主路径链 + 叶量子比特。关键性质：(1) caterpillar 态已在实验中小规模演示（[29]），是 near-term 可实现的结构；(2) 其结构具有灵活性——主路径 + 叶子可直接作为树编码逻辑量子比特的骨架（q_root 在主路径上，叶 qubit 由 caterpillar 提供），还能通过 Z 测量从长线性图中分离 4-qubit 线性图再融合到叶子上组装树结构；(3) near-term 限制——单个 caterpillar 态最多 30 qubit（[30]），初始化 12 ns + 每 qubit 发射 0.6 ns。论文用 caterpillar 态作为图态生成的基本资源态：MemTree 编译器把目标图划分为可被 caterpillar 覆盖的线性子图。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
caterpillar 态在论文 pipeline 中的生成与使用（组装树编码逻辑量子比特，Fig.4(e)）：
```
# 阶段1：caterpillar 发射（spin memory 硬件）
for each qubit on main_path:
    LA(pi/2)          # 纵向声学激发脉冲 -> 发射与 QD 纠缠的光子
    OSRP(pi)          # 光学自旋旋转脉冲 -> 定义叶分支结构（间隔插入）
# 输出：caterpillar 态 = 主路径链 {m1-m2-...-mk} + 每主路径顶点挂 b 个叶 qubit
# 阶段2：组装树编码逻辑量子比特
q_root = main_path_qubit                        # 根：主路径上的量子比特
leaf_qubits = caterpillar_leaves(q_root)        # 叶子：caterpillar 提供的叶 qubit
for each branch i:
    linear4 = Z_measure_separate(long_linear)   # 从长线性图经 Z 测量分离 4-qubit 线性图
    fusion(linear4, leaf_qubits[i])             # 融合拼到叶子上 -> 形成 {q_i^a,q_i^b,q_i^c}
# 阶段3：融合拼接成目标图态（BBT 分层 + Type-II fusion）
```
关键参数：单个 caterpillar ≤30 qubit（near-term 硬件限制）、b=4 分支、b_prep=6 制备参数（30-qubit 限制下 b_prep>6 时光子源急剧增长）。caterpillar 态同时是"程序无关"的——其结构由目标图态按需确定，光子源利用率 ~10%（vs OneAdapt 0.03%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件——硅基量子自旋内存（semiconductor quantum dot emitters）：InGaAs QD 腔体 + LA 激发脉冲 + OSRP 脉冲序列，实验演示见 Huet 等（Nat. Commun. 16, 4337, 2025；确定性、可重构图态生成）；(2) 模拟——论文自研 spin memory 模拟器按上述硬件配置（12 ns 初始化 + 0.6 ns/qubit、30-qubit 上限）模拟 caterpillar 发射；(3) 编译——MemTree 的 MIP-1 把目标图态划分为线性子图（后处理按 30-qubit caterpillar 上限细分）。使用场景：任何以量子点发射器为光源的 PQC；论文真实硬件实验在 Quandela 云平台（其平台基于自旋内存/QD 技术）验证。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## QEC Tree Code（QEC 树码 / 树簇态码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QEC 树码（tree cluster-state code，误差修正动物园收录于 https://errorcorrectionzoo.org/c/tree_cluster）是一类基于树形图态的量子纠错码，用于基于测量的量子计算（MBQC）中的光子丢失（loss）容错。核心思想：用树形纠缠结构冗余编码量子信息，使单个/少数光子丢失可以通过 stabilizer 测量模式被探测和纠正，而不破坏整个逻辑量子比特。树码的关键协议是 Varnava-Browne-Rudolph（PRL 97, 120501, 2006）提出的 loss tolerance：通过"间接 Z 测量"模式——对被擦除量子比特的相邻量子比特做 X 测量、对其余相邻量子比特做 Z 测量——确定性地读出被擦除量子比特的 Z 测量结果（基于 stabilizer X_i∏_{j∈E(i)}Z_j），从而无损移除被擦除的量子比特。Bell 等（PRX Quantum 4, 020328, 2023）系统优化了用于测量基丢失容错的图码。本论文的树编码融合直接受树码启发（引用 [1][7][68]），但把树码从"量子比特制备阶段的丢失容错"推广到"融合操作本身的失败 + 擦除容错"：每个融合分支配备 q_i^a/q_i^b 辅助量子比特以实现擦除时的间接 Z 测量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
树码的容错测量模式（论文 Fig.4(a)-(c) 借用的三条图态测量规则）：
```
# (a) 直接 Z 测量规则：Z 基测量移除目标 qubit 并断开其所有纠缠边
Z_measure(j)  ->  G' = G - {j}  （j 与所有邻居的边断裂）
# (b) 一对 X 测量规则：两个相邻 X 基测量移除 qubits 并在其邻居间建立直接连接
X_measure(j1); X_measure(j2)  ->  N(j1) 与 N(j2) 直接相连
# (c) 间接 Z 测量规则（loss tolerance 核心）：
#     目标 qubit j0 丢失 -> 选邻居 i：X_measure(i)；对 E(i) 中其它 qubit 做 Z_measure
#     stabilizer S_i = X_i ∏_{j∈E(i)} Z_j 确定性揭示 z_{j0}
```
树码在编码参数上给出"分支数 vs 丢失容错"的 trade-off；本论文把 b 分支树嵌入 caterpillar 态（b=4、b_prep=6），把树码的丢失容错思想转化为融合级擦除容错，成功率 S_tree=1-(1-(1-p_eras)^2+p_fail)^b。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：树码以"测量模式调度"实现——不需要额外的量子门，只需在测量阶段按 stabilizer 关系安排 X/Z 测量基（论文用 Perceval FFCircuitProvider 做前馈实现）。参考实现/工具：(1) 误差修正动物园收录树簇态码条目（errorcorrectionzoo.org/c/tree_cluster）；(2) Bell 等的图码优化（PRX Quantum 4, 020328）；(3) Varnava-Browne-Rudolph 的 loss tolerance 协议（PRL 97, 120501）。使用场景：光子 MBQC 中任何存在光子丢失的平台；本论文把它用于 spin memory PQC 的融合擦除容错，并指出同样的 loss-tolerant 逻辑融合思想可推广到 all-photonic 架构（把原融合单元替换为树编码逻辑量子比特即可）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Repeat-Until-Success Fusion（RUS 融合，重复直至成功）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Repeat-Until-Success（RUS）融合是增强融合成功率的一种 boosted fusion 方案（Lim, PRL 95, 030505, 2005；并在 Gliniasty 等的 spin-optical 架构 [21] 与 Thomas 等的融合实验 [66] 中沿用）：使用辅助光子（ancillary photons）在两个光子源之间反复施加融合操作，一旦某次融合成功即终止，从而在 caterpillar 态之间建立纠缠。其思路与冗余编码类似（多次尝试），但"成功即停"避免冗余编码的固定 m 次尝试浪费。逻辑错误率：P_fail=p_fail^m（失败率被压缩），P_eras=Σ_{i=0}^{m-1} p_fail^i·2p_eras——每次尝试暴露 2 个新量子比特给擦除，失败尝试还会累积擦除暴露。相比冗余编码，RUS 略优（擦除暴露随失败次数累积而非固定 2m），但消耗更多 ancilla 资源、耗时更长，且与冗余编码同样对擦除不容错。论文中 RUS 作为树编码融合的 baseline 之一：代码尺寸 m_RUS=6（按 [12][25] 最优容错性能），在真实硬件实验中 RUS+photonic 是 MemTree 的对比对象（PST 2.68×/IST 3.23× 落后于 MemTree）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RUS 融合的计算过程（伪代码）：
```
attempt = 0
while attempt < m:
    attempt += 1
    outcome = type2_fusion_with_ancilla(source_A, source_B)
    if outcome == SUCCESS:
        entanglement_established()      # 成功即终止
        break
    elif outcome == FAILURE:
        continue                        # 消耗 ancilla，重试
    elif outcome == ERASURE:
        # 无容错：纠缠建立与否未知，整体结果必须丢弃或依赖重试
        record_erasure()
# 逻辑错误率：
# P_fail = p_fail^m
# P_eras = Σ_{i=0}^{m-1} p_fail^i * 2p_eras   （每次尝试 2 个新 qubit 暴露给擦除）
```
例：p_fail=0.25、p_eras=0.1、m=6：P_fail≈2.4×10^-4，但 P_eras≈2×0.1×(1-0.25^6)/0.75≈0.266——擦除率显著高于失败率，成为主导错误。RUS 在论文 Fig.4(f) 模拟中相对树编码明显退化（擦除率升高时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 线性光学——需要 ancilla 光子注入与重复融合测量硬件；(2) 编译集成——论文按 [12][25] 的最新协议实现 RUS 并集成进 MemTree 编译框架作对比（m_RUS=6）；(3) 真实硬件——论文在 Quandela 云平台用 Perceval 实现 RUS+photonic 作为真实硬件实验 baseline。使用场景：需要提升概率性融合成功率的 PQC 图态生成；与树编码相比，RUS 在擦除率 >0 时性能快速退化、且 ancilla 开销高，是论文 ablation 的关键证据——MemTree 换用 RUS 后除 Grover 外全面劣于 OneAdapt-ET，证明树编码才是性能来源。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

## Redundantly-Encoded Fusion（冗余编码融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
冗余编码融合（redundantly-encoded fusion，Hilaire 等, Quantum 7, 992, 2023）是第一个面向自旋内存架构的融合失败容错方案：利用 caterpillar 态的特性，把逻辑线性图态的每个节点编码为 m 个叶量子比特组成的逻辑量子比特；两个逻辑量子比特融合时，对两方每对叶量子比特各施加一次融合操作——即对两个逻辑量子比特执行 m 次融合尝试，任意一次成功即逻辑融合成功。这压缩了逻辑失败率：P_fail=p_fail^m。然而代价是逻辑擦除率 P_eras=1-(1-p_eras)^(2m)：每个逻辑量子比特的 m 个物理量子比特各自独立暴露于擦除（2m 指数），因此 m 越大，失败率越低但擦除率越高——在真实光子丢失（p_eras≈10%）下，擦除很快成为主导错误，这正是该方案与 RUS 的共同致命缺陷。论文中 redundantly-encoded 作为树编码融合的 baseline：代码尺寸 m_Redun=5（按 [12][25] 最优容错性能）；对比结果：树编码执行时间为其 1.9×10^-3×，但光子源多 2.55×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
冗余编码融合的计算过程（伪代码）：
```
# 逻辑量子比特 L 编码：m 个叶 qubit {l_1,...,l_m}（来自 caterpillar 的叶分支）
# 逻辑融合 A-B：对每对叶 qubit 施加融合
for k in 1..m:
    outcome_k = type2_fusion(l_k(A), l_k(B))
    if outcome_k == SUCCESS:
        logical_success = True        # 任一成功即逻辑成功
        break
# 逻辑错误率：
# P_fail = p_fail^m
# P_eras = 1 - (1-p_eras)^(2m)      # 2m 个物理 qubit 独立暴露于擦除
```
例：p_fail=0.25、p_eras=0.1、m=5：P_fail≈9.8×10^-4，但 P_eras=1-(0.9)^10≈65%——擦除完全主导。对比树编码 b=4 时 S_tree≈96%（见树编码条目），冗余编码在 p_eras 升高时不可用（Fig.4(f)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件——利用 caterpillar 态每个主路径顶点挂多个叶 qubit 的结构，天然提供 m 个叶量子比特；(2) 编译集成——论文按 [12][25] 的最新协议实现 redundantly-encoded 并集成进 MemTree 框架作对比（m_Redun=5）；(3) 模拟——论文 Fig.4(f) 以 10^3 次融合试验统计其成功率。使用场景：作为 boosted fusion 的早期方案，可用于融合失败为主、擦除可忽略的假设场景；在真实光子丢失条件下被树编码取代。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion


## 湿球温度预报的两级 ML 集成回归（Wet-Bulb Temperature Forecasting Ensemble）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 湿球温度（WBT）是蒸发冷却能力的极限温度：冷却塔通过蒸发水分把热量排入环境，蒸发潜力越高（WBT 越低）冷却能力越强；气候模拟（CMIP6）直接提供干球温度（DBT）与相对湿度（RH），但因为数据时间错位、网格粗，无法用标准公式直接算出 WBT。Prometheus（Google, ISCA 2026）用"两级 ML 集成回归"解决：第一阶段两个基回归器——随机森林（RF：100 棵树、max_depth=5、每分裂节点最少 2 样本、输出为树均值，防过拟合的浅树）与支持向量机（SVM：RBF 核，把日 min/mean/max DBT 与 RH 共 6 个特征映射到高维空间使非线性关系可分）——各自独立预测日最大 WBT；第二阶段神经网络（NN：两个隐层 16/8 神经元、ReLU 激活、L2 正则 0.5、输出层单神经元无激活）以两个基模型预测 + 原始 6 特征为输入，输出单一稳健的 WBT 预报。核心动机：多种互补模型集成降低方差、提高极端值（99.5 百分位）预报精度——Table I 显示集成 RMSE 0.67°C vs 最佳单一 baseline 1.71°C（-43%）、99.5 百分位正误差 3.5 vs 9.7（-60%+），因为数据中心设计恰恰依赖分布尾部的极端温度。
- 该术语从算法pipeline角度拆解：这是"特征→两个独立基学习器→元学习器融合→偏差校正→极值分布拟合"的完整预报 pipeline。伪代码：
  ```
  # 输入：0.25° 网格上 CMIP6 投影的日最低/平均/最高 DBT 与 RH（X ∈ R^(6)）
  y_rf  = RF(n_estimators=100, max_depth=5, min_samples_split=2).predict(X)   # 每棵树叶子均值，再平均
  y_svm = SVR(kernel='rbf').predict(X)                                        # K(x,xi)=exp(-γ||x-xi||²)
  y_nn  = NN(hidden=[16,8], activation=ReLU, L2=0.5).predict(concat(X, y_rf, y_svm))  # 单输出无激活
  y     = y_nn - bias(site)      # 地理偏差校正（5 年历史均值差）
  # 之后：Gumbel 拟合 → N 年回返温度（见独立条目）
  ```
  张量计算：RF 沿特征阈值分裂、输出 = 落入叶子的训练样本均值；SVM 核函数 K(x,x_i)=exp(-γ‖x-x_i‖²)；NN 前向 h1=ReLU(W1x+b1)（16 维）→ h2=ReLU(W2h1+b2)（8 维）→ ŷ=w3·h2+b3（标量）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：SVM/RF/NN 均为标准监督回归模型，可用 scikit-learn（SVR/RandomForestRegressor）与任意深度学习框架实现；输入来自公开 CMIP6 数据（经 Google Data Commons 访问，0.25° 网格、日 DBT/RH 到 2100 年）；6 个 CMIP6 模型构成覆盖高/中/低平衡气候敏感度（ECS>4K / 2.87-4K / <2.87K）的集合以包含不同未来情景。训练每站点仅需数小时，推理成本可忽略（决策在年度/十年尺度，不在关键路径）。
  - 使用：对每个数据中心站点，用 25 年历史观测 + 20 年 CMIP6 前向投影拟合模型，输出该站点未来日最大 WBT 分布，再交给 Gumbel 拟合得到 50 年回返温度，驱动冷却容量设计。论文未提供代码仓库链接，联网检索（2026-08）未发现公开实现，无法确认开源。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure

## Gumbel 极值分布与 N 年回返温度（Gumbel Distribution and N-Year Return Period Temperature）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Gumbel 分布（I 型极值分布，也称 Fisher–Tippett I 型）是极值理论中用于建模"每年最大温度/降水"等块最大值的概率分布，适合数据中心"设计温度"这种低频极端事件风险量化。其累积分布 F(x;µ,β)=exp[-exp(-(x-µ)/β)]，位置参数 µ 与尺度参数 β 由均值 T_max-mean=µ+βγ（γ=欧拉常数 0.5772）与标准差 T_max-std=πβ/√6 反推。N 年回返温度 T_N 指"平均每 N 年才被超过一次的温度"，对应 Gumbel 分布的 1-1/N 百分位：T_N = T_max-mean - (√6/π)[0.5772 + ln(ln(N/(N-1)))]·T_max-std。ASHRAE 只用量化前 30 年气象站历史数据拟合，忽略气候变暖；Prometheus 用"25 年历史观测 + 20 年 CMIP6 多模型投影"的混合数据拟合 µ/β，把未来气候情景的方差嵌入分布参数，从而把 London 2022 的 40.2°C 从 ASHRAE 的"1-in-200 年"修正为"1-in-50 年"，DBT/WBT 50 年回返温度相对 ASHRAE 平均高 4.4°C/1.4°C。
- 该术语从算法pipeline角度拆解：它是 ML 预报输出的概率后处理层，把"逐日 WBT 预报序列"转成"可决策的年超温概率"。pipeline 伪代码：
  ```
  # 输入：站点 s 的历史观测 + CMIP6 投影的年最大 WBT 序列 {T_max_year}
  T_mean, T_std = mean({T_max_year}), std({T_max_year})
  beta = T_std * sqrt(6)/pi                 # 尺度参数
  mu   = T_mean - beta * 0.5772             # 位置参数（γ=欧拉常数）
  # N 年回返温度（对应 1-1/N 百分位）：
  T_N  = T_mean - (sqrt(6)/pi) * (0.5772 + ln(ln(N/(N-1)))) * T_std
  # 年超温概率：Pr[annual max T > T_design] = 1 - F(T_design; mu, beta) = exp(-exp(-(T_design-mu)/beta)) 取补
  # 例：London 2044, SSP5-8.5: T_50(DBT)=41.2°C vs ASHRAE 2021: 37.7°C
  ```
  该概率直接喂给升级判据式 5：Pr[Annual max T > 冷却设计温度]_Gumbel > 2% 时触发冷却升级（对应 1-in-50 年事件风险容限）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Gumbel 参数可用极大似然估计或矩估计（上式即矩估计）；Python 中 scipy.stats.gumbel_r / gumbel_l 提供现成分布与拟合函数。论文未提供代码，联网检索未发现公开实现，无法确认开源。
  - 使用：Gumbel 分布是工程上（水文、建筑、气候）估计设计回返期的标准工具（IEEE/ASHRAE 设计工况、大坝防洪等均用类似极值方法）。在 Prometheus 中它把"气候模型投影的不确定性"转化为"冷却容量该配多少"的定量依据，也是与 ASHRAE 后向方法对比的桥梁（同样输出 50 年回返温度，但输入数据不同）。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure

## 地理偏差校正（Geo-Specific Bias Correction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 偏差校正是气候降尺度/预报的后处理技术：全球/区域气候模型（CMIP6）输出在特定站点存在系统性偏差（模型网格点与站点海拔、局地微气候不符），直接把模型输出当站点真值会导致持续偏移。Prometheus 采用站点特异性偏差校正：把某站点模型预报与同站 5 年历史观测对比，计算两者均值差 bias = mean(y_nn_hist - y_obs)，再从集成输出中减去该偏差 y_corr = y_nn - bias，从而消除系统性误差、使站点级预报与历史记录一致。这是大空间数据集（如 NEX-GDDP-CMIP6 降尺度数据）中"bias-corrected"产品的标准做法（论文引用 [7]）。
- 该术语从算法pipeline角度拆解：它是 ML 集成之后的轻量校准层，输入是"集成输出的日最大 WBT 序列"，输出是"与该站点气候一致的校正序列"，之后才进入 Gumbel 拟合。伪代码：
  ```
  # 站点 s：用最近 5 年历史观测校准
  bias_s = mean_over_years( ensemble_forecast_s[t] - observed_wbt_s[t] )
  y_corrected_s = y_nn_s - bias_s        # 逐站点偏差可正可负（站点高于/低于模型网格）
  # 校正后序列进入 Gumbel 拟合（µ, β），再算 T_50
  ```
  作用：0.25° 网格（约 27km）仍比单站点粗，偏差校正把"区域平均预报"归一到"站点实际气候"，是支撑"站点特异性概率风险评估"的关键一环（论文称减少系统性误差）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：本质是简单均值偏移校正（delta method 的一种简化）；更一般的气候学中还有分位数映射（quantile mapping，校正整个分布而非仅均值）。论文未提供代码，联网检索未发现公开实现，无法确认开源。
  - 使用：Prometheus 对 30 个生产数据中心逐站点应用该校正，使预报与各站历史气候一致；配合 ML 集成与 Gumbel 拟合，产出站点级 50 年回返温度（Table II 中 Dublin/London/Phoenix/Council Bluffs/Dalles 的具体数值即校正后结果）。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure

## GatedMLP（门控 MLP 子图，Falcon-7B）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GatedMLP（门控多层感知机 / Gated Linear Unit 型 FFN）是 LLM FFN 层的常见形态：两个并行线性投影后做门控逐元素运算。QiMeng-Tensify（ISCA'26）中 GatedMLP 定义为 O = SiLU(X·W1) ⊗ (X·W2)（· 为矩阵乘、⊗ 为逐元素乘），取自 Falcon-7B，作为最重要的图级优化 benchmark 与工作示例：它含 3 个 GEMM（含 2 个共享输入 X 的 GEMM）+ SiLU（exp/add/div/mul）+ elementwise mul + 动态门控，是非规则数据流与条件执行的代表性子图，传统编译器（TVM 切成两个子图）与模板编译器（Mirage block 级融合）都无法全局最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GatedMLP 的计算过程：
```
# X: (B, S, H) 输入；W1, W2: (H, 4H)/(4H, H) 权重
O1 = X @ W1            # GEMM1，输出 (B,S,4H)
O2 = SiLU(O1)          # SiLU(x) = x * sigmoid(x) = x/(1+e^(-x))，逐元素
O3 = X @ W2            # GEMM2，共享输入 X，输出 (B,S,4H)
O  = O2 ⊗ O3           # 逐元素乘
```
QiMeng-Tensify 的全融合版本把四步合成单个 loop nest：GEMM1 与 GEMM2 在共享 (i0,j0,k0) tiling loop 下并行 tile（复用 X 的 global→shared 加载），SiLU 与 MUL 被 compute_at 提升进 GEMM 的 reduction 循环逐 block 计算，无中间 buffer（O1/O2/O3 全部消除），端到端只读写 X 与 O。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 benchmark 子图（Table VII，Arch. 列标注 Falcon-7B），FP32 在 CUDA core、FP16 在 TensorCore 评估；FP16 下 QiMeng-Tensify 比 TVM MetaSchedule 快 2.80×、比 Mirage 快 1.47×（案例研究 G 节）。使用方式：GatedMLP 是观察"变换空间受限"（Fig.1：TVM 空间 1e10 但错过全融合、Mirage 空间 1024 但 block 级受限）与 LLM 先验价值的核心例子（Fig.8 消融、Fig.12 搜索收敛、Fig.13 搜索时间分解——compute-intensive 算子如 GatedMLP 的 Parameter Specification 阶段占 >85% 编译时间）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS

## LoRA（Low-Rank Adaptation，低秩适配子图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA（低秩适配）是参数高效微调方法：冻结原权重 W，训练低秩增量 ΔW = BA（B、A 为低秩分解矩阵），推理时输出 h = xW + xBA。QiMeng-Tensify（ISCA'26）把 LoRA 作为图级 benchmark 子图（Arch. 列标注 LLaMA3-lora）：LoRA 层含 3 个矩阵乘（xW、xB、A，其中 xB@A 沿 low-rank 维串行），计算密集 + 数据依赖链复杂，是"传统 autoscheduler 固定策略失效"的典型多算子图（论文 Background B 节明确举例"3 matrix multiplications in LoRA"）。端到端评估里 GPT-3-7B-LoRA 也是四个网络级 benchmark 之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LoRA 子图计算过程：
```
h = X @ W0              # 冻结权重投影 (B,S,H) @ (H,H)
h = h + (X @ A) @ B     # 低秩增量：A:(H,r) 下投影、B:(r,H) 上投影，r << H
# 或融合写法：h = X @ W0 + X @ A @ B，可整体看作 (B,S,H) 的线性映射
```
优化机会：把 X@A 与 (X@A)@B 的中间激活（B,S,r）融合/消除、与 X@W0 共享 X 的加载与并行 tile；QiMeng-Tensify 在该子图上编译时间 1.92h（A100，Table VIII）、FP16 子图平均加速比 PyTorch 达 6.49× 量级（含 LoRA 等不规则子图贡献，论文称 LoRA/GatedMLP 上最高 2.3× over PyTorch）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 benchmark 子图（Table VII），与 GQA/GatedMLP 等一起构成 9 子图集合；FP32/FP16 双精度对比 8 个 baseline。使用方式：代表"不规则数据流 + 多 GEMM 依赖链"（相对单算子），验证 QiMeng-Tensify 从算子级推广到子图级的一般化能力；GPT-3-7B-LoRA 用于端到端网络级评估（A100/H100、batch 1/8、seq 4096，vs PyTorch/TensorRT-LLM/Mirage）。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS

## NSA（Native Sparse Attention，原生稀疏注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NSA（Native Sparse Attention，DeepSeek 2025）是"硬件对齐且原生可训练"的稀疏注意力：把注意力分解为压缩（compress，聚合粗粒度 token 块）、选择（select，按重要性选细粒度块）、滑动窗口（sliding window，局部窗口）三条并行分支，压缩/选择分支的块稀疏结构可与硬件对齐（如块粒度稀疏、压缩 block 紧凑布局），保持训练效率的同时降低长上下文注意力成本。QiMeng-Tensify（ISCA'26）把 NSA 列为最新 benchmark 子图（Table VII，Arch. 列 Transformer）：它是稀疏、非均匀负载的算子，验证框架对新算子的泛化能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NSA 子图的稀疏注意力计算骨架：
```
for q_block in query_blocks:
    # 三条并行分支：
    o_c = attention(q, K_compressed, V_compressed)   # 压缩分支（粗粒度）
    o_s = attention(q, K_selected,   V_selected)     # 选择分支（细粒度，块稀疏）
    o_w = attention(q, K_window,     V_window)       # 滑动窗口分支（局部）
    o   = fuse(o_c, o_s, o_w)                        # 输出融合
```
在 QiMeng-Tensify 中该子图作为输入 TensorIR 被图重写/MCTS 优化：结果相对 Triton 快 1.51×、相对 Reasoning Compiler 快 1.18×（FP16，A100），略低于专家手写 FlashAttention 但证明对稀疏非均匀新算子可泛化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 benchmark 子图（Table VII），与 FlashAttention 同为"最新算子"代表（论文 E 节：benchmark 含 Mirage 的子图集 + LayerNorm + 最新算子 FlashAttention/NSA）；FP16 TensorCore 评估对比 Triton/Reasoning Compiler 等。使用方式：验证 QiMeng-Tensify 对"尚未有成熟手写实现/模板"的新算子的自动优化能力（专家仅 FlashAttention 领先，其余自动方法均落后），说明其一般化范式可覆盖稀疏非均匀负载。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS

## LZ 序列压缩（LZ77/LZW/LZMA Lempel-Ziv 字典压缩家族）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LZ（Lempel-Ziv）是一族基于"序列（symbol sequence）替换"的无损压缩算法：把重复出现的变长符号序列（通常 3–258 个连续字节）用较短的"序列标识符"代替，从而获得高压缩率。核心机制（本论文背景部分）：先用更早出现的序列作为字典压缩更晚的数据。LZW：把首次遇到的唯一序列按出现顺序加入字典（如 "XY"→1*、"YZ"→2*），之后遇到字典中已有的序列就用其索引替换；为区分字典索引与字面量，LZW 输出用 9-bit 符号（8-bit 只能编码 256 个字面值，9-bit 可额外编码字典索引）。LZ77：用 ⟨offset|length⟩ 对作为序列标识符，offset 指向滑动窗口中该序列首次出现的位置、length 为其长度，从而让输入早期出现的重复长序列也能被压缩（克服 LZW 需先选短序列再选长序列的顺序限制）。LZMA 是 LZ77 的高压缩比变体（更大窗口+范围编码）。压缩率高的关键在"重复的变长序列"，内存数据中的重复指针、零初始化变量、模式化数据是典型冗余来源。
- 从算法pipeline角度拆解术语（本论文 Fig.1 的 LZW 例子）：输入 "XYXYZXYZYX..." 这类数据时，压缩 pipeline 为：① 扫描输入，把首次遇到的新序列按序加入字典（"XY"→1*、"YZ"→2*、"ZX"→3*）；② 后续输入若匹配字典序列则输出其索引（如 1*、2*）；③ 字典本身不写入压缩输出——解压器按相同顺序从压缩数据动态重建同一字典；④ 动态重建要求"短序列先于长序列被选中"，导致长序列只能压缩更晚的数据（LZ77 的 ⟨offset|length⟩ 正是为此设计）。注意 LZW 的字典是"按出现顺序隐式构建"的，与 RST 的"按 utility 显式挑选存储"形成对比。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- LZ 家族是 Deflate（LZ77+Huffman）、Zstandard（Zstd）、LZMA 等主流算法的"序列压缩"阶段（通常配合第二阶段的符号压缩）。硬件实现：IBM Power9/z15 数据压缩加速器（Deflate，ISCA'20）、CDPU（通用 LZ 家族加速器，支持 Deflate/LZ4/Zstd，ISCA'23）、TMCC 的 ASIC Deflate（面向内存压缩，MICRO'22）、OCP Project Zipline（开源 streaming Deflate RTL，存储/网络 I/O）。软件实现：zlib（软件 Deflate）。本论文应用场景：页级 LZ 被 Hyperscale Tiered Memory Expander Specification 强制用于硬件内存压缩，4KB 页整体作为单一压缩单元（压缩率随粒度增大而提高）。
涉及论文标题：
- Random-Access Hardware Sequence Compression

## Deflate（LZ77 + Huffman 两阶段无损压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Deflate 是广泛使用的无损压缩格式（zlib/gzip 的基础），顺序执行两个阶段：第一阶段用 LZ77 对数据做序列压缩（用 ⟨offset|length⟩ 对替换重复序列），得到字面量（literal）与序列标识符（sequence identifier）的混合流；第二阶段用 Huffman 符号压缩对该混合流编码——频繁符号用更短码、罕见符号用更长码。两阶段配合使 Deflate 在多样化数据上获得高压缩率（本论文实测软件 zlib 在 88 benchmark 内存 dump 上几何平均 3.84×）。Huffman 阶段使用符号字典映射字面量到码字，字典存在压缩输出中（与 LZ 不同）。
- 从算法pipeline角度拆解术语：压缩 pipeline 为 ① LZ77 匹配：在滑动窗口中找最长匹配，输出 ⟨offset|length⟩ 或字面量；② Huffman 编码：统计字面量与序列标识符的出现频率构建 Huffman 树，把各符号映射为变长码；③ 输出"码字流 + Huffman 字典"。解压 pipeline 必须从页/流首开始：先 Huffman 解码出字面量与 ⟨offset|length⟩，再按序从滑动窗口复制重建数据——服务中间任意位置的一个 64B 块需要解压其之前的所有数据（本论文的核心痛点）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件实现（本论文的 baseline）：TMCC [MICRO'22] 的 ASIC Deflate 面向内存压缩，用截断 Huffman（truncated Huffman）保证快速解压，16 符号字典、固定 hash 表、受限 lookahead，硬件设计约束使压缩率比软件低 ~10–12%。本论文对比：TMCC ASIC Deflate 半页解压延迟 140ns（2.5GHz、7nm 综合），RST 每 64B 块 18ns；压缩率 3.3×（Deflate）vs 3.4×（RST）。其他硬件：IBM Power9/z15 Deflate 加速器（~1µs）、OCP Project Zipline（open-source RTL，~2µs）、CDPU（~1µs）。软件：zlib/znzlib。
涉及论文标题：
- Random-Access Hardware Sequence Compression

## Huffman 符号压缩与静态符号字典（含 reuse codes）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Huffman 符号压缩是"按符号出现频率分配码长"的熵编码：给频繁符号更短码、罕见符号更长码，从而降低平均码长。经典实现为每个数据集动态构建 Huffman 树并把符号字典随数据一起存储（如 Deflate 的第二阶段）。本论文 RST 的关键改动是把符号字典改为"静态"——用一个压缩/解压双方事先已知的固定字典，把每页符号字典开销降到 0B。静态字典设计要点：① 字典索引的码长利用 utility 选择已捕获的频率（靠前的索引对应更高 utility、更频繁出现的序列，用更短码）；② 内存数据值中"零"最重要，用最短码；③ 观察到重复字符倾向于在附近重现，引入 reuse symbols（reuse symbol i = "重复 i 个位置前看到的符号"，i∈{1..8}）提供无需每页字典开销的动态适配。Table 1 给出字典树与数据树各符号类别的码长（如字面量 0 在数据树 6 位、reuse 1/2 各 5 位等）。
- 从算法pipeline角度拆解术语：RST 压缩最后阶段 pipeline 为 ① 序列压缩后得到"字面量+字典索引"流；② 用静态符号字典单趟编码（每序列在插入序列字典前先编码，精确追踪剩余容量）；③ 输出静态码流。解压 pipeline：先用静态符号字典解码（压缩数据 ~8 cycle、序列字典 ~16 cycle），再进入序列展开。鲁棒性：静态树的 1-bit 字面量标志位把最坏开销封顶在 12.5%（全字面量、无字典项）；不可压页由硬件压缩系统经每页翻译项原样存储（如 TMCC/DyLeCT 的 CTE）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 常规动态 Huffman 实现：统计频率→建 Huffman 树→生成变长码→字典随输出存储（解码需先读字典）。RST 的静态实现：固定码长表（Table 1：字面量 0:255、字典索引 0:62、reuse 1:8 三类符号的 Dict/Data 两套码长），解码器无需每页字典即可解码；reuse codes 提供数据依赖的短程适配（类似把"重复前 i 位置的符号"当作一个预测符号），这在内存数据（重复指针、零填充）上尤其有效。
涉及论文标题：
- Random-Access Hardware Sequence Compression

## 压缩粒度与压缩率（Compression Granularity & Compression Ratio）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 压缩粒度（granularity）指压缩器一次性处理的上下文（context）大小；压缩率（compression ratio）= 未压缩大小 / 压缩后大小。两者强相关：更大的粒度暴露更多重复模式，压缩率通常更高（本论文 Fig.2：88 benchmark 几何平均，Deflate 压缩率随粒度显著上升）。内存块级压缩（64B CPACK/BDI、128B BPC）压缩率低（块独立压缩，无跨块上下文）；页级序列压缩（4KB LZ）压缩率高，因此被行业规范强制用于硬件内存压缩。
- 从算法pipeline角度拆解术语：压缩率决定硬件内存压缩的容量增益——压缩率 c 意味着每物理字节可多存 c−1 字节逻辑数据（3.3× = 每物理字节多 2.3B；1.75× = 只多 0.75B，容量增益损失约 3×）。粒度与随机访问的矛盾：页级 LZ 用整页做字典获得高压缩率，但随机解压单个 64B 块必须取+解压该块之前的所有数据（平均 704B）；若把字典限制到 128B（朴素方案），压缩率掉到 1.75×。RST 的解法：128B 总字典空间/页（A 序列字典 + B 8B 位置元数据 + C 0B 静态符号字典），达到 3.4× 压缩率并支持随机解压。128B 为性价比点（消融：256B 字典仅小幅提升，但 per-access 取数从 192B 涨到 320B，+67% 流量）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 度量方法（本论文 §VI-A）：在 4KB 页粒度、同一 88 benchmark（7 类：数据库 Redis/TPC/SPECjbb2015、GraphBig、PARSEC-3.0、SPEC CPU2017、Spark Bench、DaCapo、Renaissance）、同一方法论（忽略内存 dump 全零页）下测压缩率，每类内几何平均再对 7 类取几何平均：RST 3.4× vs ASIC Deflate 3.3× vs 软件 zlib 3.84×。粒度对硬件设计的影响：页级压缩需位置元数据定位块、支持随机访问需小字典+精确的位置跟踪。
涉及论文标题：
- Random-Access Hardware Sequence Compression

## RST top-utility 序列选择（Random-access Sequence Compression with Top-utility Selection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RST 是本论文提出的核心算法：在每 4KB 页内全局搜索 utility 最高的序列，迭代选入显式 128B 序列字典，使压缩率不降（3.4× vs Deflate 3.3×）的同时支持任意 64B 块随机解压。utility 定义为"该序列若入选字典可带来的总空间节省 / 该序列消耗的字典空间"：序列 "XY" 的字典开销 D=2·9+L（2 个 9-bit 符号 + L 位长度字段），潜在节省 S=4·2·9−(4·9+D)（4 次出现压缩为 4 个索引），utility=S/D。与 LZ 家族的区别（Table 2）：序列选择是"全局 utility 最大化"而非"局部贪心最长匹配"；字典是"显式存储"（128B/页）而非"隐式全页字典"；序列长度上限 5 符号（跨迭代可捕获长重复："VWXYZ"×5→1*×5→2* 两轮压成单个索引）；解压是"逐块独立"而非"从页首串行"。
- 从算法pipeline角度拆解术语（Algorithm 1 伪代码）：
  ```
  Input: page（4096B）
  U = COUNTSEQS2TO5(page)            // 统计所有 2~5 符号唯一序列出现次数进 utility 表
  D = {}                              // 空序列字典
  page' = page
  while HASSPACE(D) and HASPOSITIVEUTILITYSEQUENCE(U):
      s* = FINDTOPUTILITYSEQUENCE(U)        // 每长度子表取最高 count 算 utility，跨长度取最大
      dict_idx = AddToDictionary(D, s*)       // 显式存入字典
      substitution_sites[] = SUBSTITUTION(page', s*, dict_idx)   // 替换所有出现为索引
      UTILITYUPDATE(U, page', substitution_sites)   // 重算受影响序列的 count
  return (D, page')
  ```
  关键点：选中一个序列会改变所有与它重叠的未选序列的 utility（每个符号只能被一个选中序列压缩），必须做"替换步骤 + utility 更新步骤"迭代；朴素全表重算需 8×10^6×64≈5 亿次操作/页，两个优化把操作降 ~1000×：① 只更新与最新选中序列重叠/含新索引的条目；② 序列长度上限 5 符号（使长度字段 L=2 位）。即便如此每页仍需 >3×10^5 次操作，串行 >100µs，必须硬件并行。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为三个并行硬件模块（见硬件架构层条目）：substitution module（CAM 匹配+overlap filter+替换+compaction）、update generator（splice-and-cancel 防双计数）、table update module（32-bank 组相联 SRAM+sorting network+FIFO）。开源：GitHub https://github.com/HEAP-Lab-VT/rst（BSD 3-Clause Clear），Zenodo artifact https://doi.org/10.5281/zenodo.19449274（C++ 参考实现 + SystemVerilog RTL + QEMU VM 镜像）；复现压缩率用 `bash regenerate_figures.sh`（~50 分钟生成 Fig.18/19）。应用：硬件内存压缩（内存控制器集成与 CXL 内存扩展场景），128B 字典已是每次压缩块访问取数的 2/3（128B 字典+64B 块=192B）。
涉及论文标题：
- Random-Access Hardware Sequence Compression

## Range Identifier（RID，范围标识符）与 RangeMap 范围映射

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RID 是 RangeGuard（ISCA 2026，SKKU IRIS Lab，https://iris.skku.edu/publication/c97_isca_2026/）提出的元数据中心纠错框架的核心概念：把每个数值的数值域划分为少量预定义"范围"（range），每个值映射到其所在范围的索引（如 4-bit RID 表达 16 个范围、8-bit RID 表达 256 个范围）。RID 不存储原始数据——写路径只把 RID 的 ECC 冗余存下来、原始数据照常存；读路径从取回数据重新生成候选 RID。逻辑链：DNN/LLM 的实际崩溃来自极少数 exponent 位翻转制造的天文数字 outlier（BF16 e[7] 0→1 翻转 ×2^128≈3×10^38），而不是 mantissa 低阶位的微小扰动 → 与其保护"原始比特"（恢复一个 32-bit 值至少需 64 bit 冗余），不如保护"值落在哪个范围"这个语义元数据（恢复 4-bit RID 只需 8 bit 冗余）→ 把稀缺的 16-bit parity 预算花在"改变范围"（inter-range）的错误上，范围内（intra-range）扰动直接放行。RID 概念上类似量化 level，但关键区别是：范围只作为"恢复目标"，无错值仍以全精度存储和使用，只有被纠错的损坏值才"吸附"到范围代表值——这是"有界近似纠错"（bounded approximate correction）的核心。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RangeMap 构造（简单映射，论文 §V-B）的算法流程（BF16，K 个范围）：
```
输入：值分布（零均值高斯，标准差 σ）、范围数 K、exponent 域 E={0..255}
1. 求 exponent 概率质量：P(e=k) = 2[Φ(2^(k+1-127)/σ) − Φ(2^(k-127)/σ)]   // Φ=标准正态 CDF
2. 定义 scale 函数：f(e) = 2^(e-127)
3. 把 E 划分为 K 个连续区间 {[l_k, r_k]}，每区间赋代表 exponent ê_k
4. 最小化 MAE：L = Σ_k Σ_{e=l_k..r_k} P(e)·|f(e) − f(ê_k)|     // 全局最优表
```
例子（σ=4 的 4-entry 表，Table II）：区间 [0,127]→代表值 0.5、{128}→2、{129}→4、[130,255]→8；σ=4 使 ±3σ≈±12 覆盖多数 LLM 激活。理想映射用 L1 最优标量量化（Lloyd-Max 类）：归一化高斯阈值 (−0.8217, 0, 0.8217)、代表值 (−1.2657, −0.3778, 0.3778, 1.2657)，可缩放复用到 FP32/BF16/INT8。执行例子（8 个 FP32 值、RG 4b DSC）：写时每值查 RangeMap 得 4-bit RID → 8 个 RID 进 RS(12,8) 编码生成 16-bit parity；读时重新生成候选 RID → RS 解码纠正 ≤2 个错误 RID → 被纠值替换为该范围代表值 → 误差上界 = 范围宽度。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件侧为 flip-flop 实现的 RangeMap（每 32B 访问把 16 个 16-bit 值的 exponent 与 16 个 8-bit 表项并行比较，256 个 8-bit comparator，1 cycle 完成 RID 提取）；多格式支持用多个子映射（32/16/8-bit 值分别用 4/2/1-bit RID，同一 32-bit 区域的 RID 打包进一个 ECC 符号）。使用要点：范围宽则浪费 RID 空间、范围窄则 MAE 收益边际；σ 过大/过小都使保护失效（敏感性实验：最优 σ 下 Llama-3.2 在 BER=10^-6 保持精度，偏离则骤降）；全局映射（σ=4、4-bit RID）已足够，tensor 级 Lloyd-Max 映射精度增益有限且面积开销大。效果：16-bit parity 预算下每 256-bit block 容忍 64+ 个翻转数据 bit（8× bit 级方案）。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs

## 浮点位翻转敏感性分析（sign/mantissa/exponent 位翻转误差模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 RangeGuard 的动机分析（§III-B/C）：系统刻画 DRAM 存储错误（bit flip）如何映射为数值错误、再传播为 DNN 端到端准确率损失的数值模型。对 BF16（1 符号位 s + 8 指数位 e + 7 尾数位 m，bias=127），x = (−1)^s × 2^(e−bias) × (1.m)：(1) 符号位翻转 → x'=−x，误差 = 2|x|，有界且正比于原值；(2) 尾数位翻转 → |x'−x| ≤ ½|x|（k=0 最重），全部尾数位翻转累积仍 <|x|，指数衰减；(3) 指数位翻转 → x' = 2^(±2^p)·x，scale factor SF=2^(±2^p) 随位指数 p 双重指数增长（p=7 时 0→1 翻转 ×2^128≈3×10^38，1→0 翻转 ×2^-128 趋零）。核心结论：exponent 位（尤其高位）是唯一能制造"超出原值量级"的灾难性错误的位，且 0→1（放大）比 1→0（衰减）危险得多；尾数/符号位错误相对良性。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
端到端验证流程（§III-C，PyTorch 推理 + 100 次 Monte Carlo trials）：对每个目标 BER 在权重与中间激活上均匀注入随机 bit flips → 跑完整推理 → 统计准确率。位级结果（Llama-3.2-1B）：sign/mantissa 位错误到 BER=10^-5 才明显，exponent 位错误在 BER=10^-12 就足以让 1/100 trial 崩溃（平均仅 0.15 flipped bit/trial）；e[7] 位在 BER=10^-11（约 1.5 flipped bit/trial）时 10/100 trial 跌向随机。模型对比：ResNet-50 在 BER=10^-8 开始退化、10^-7 崩溃；Llama-3.1-8B 在 10^-10 退化、10^-9 崩溃——LLM 比 CNN 脆弱 2~3 个数量级，源于 transformer 的 attention 放大 + residual 跨层保留 + LayerNorm 稳定化使单个 outlier 长期存在并级联。这就是"保护 exponent 类范围变化"（RID 语义保护）的直接依据。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 Hugging Face 模型（PyTorch）+ lm-evaluation-harness（ARC-Easy、ImageNet-1k）的软件错误注入框架，方法学与 PyTorchFI/MRFI/ReaLM/FIdelity 等 fault-injection 框架一致；论文未开源注入代码。使用价值：① 评估任何 ECC/容错方案前先定位"哪些位真正致命"（本文证明 exponent 高位主导、低阶位浪费预算）；② 为 RID/range 类语义保护提供"哪些范围变化需要保护"的量化依据；③ 灵敏度模型（SF=2^(±2^p)）可直接用于推导误差上界与 RangeMap 构造。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs

## Weight Nulling 与 VAPI（基于权重值分布的容错 ECC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
两条利用神经网络权重分布特性的轻量容错 baseline 路线（RangeGuard 论文 §IV-B 对比对象）：(1) Weight Nulling（Qin et al. arXiv:1709.06173）——每个权重借用最低有效位（LSB）放 1 bit parity，读到 parity 不匹配时把损坏权重直接置零而非纠错；缺点：parity 只检奇数位错、对大幅值权重置零会丢失重要信息。(2) VAPI（Value-Aware Parity Insertion，Lee & Yang DATE 2022）——针对 8-bit 量化 CNN 权重：观察多数权重靠近零，采用 sign-magnitude 表示使高序位（如 b6、b5）很少使用，把这些"不重要位"覆写为 parity；用 DEC(64,50) 码每 64-bit 权重块纠 2 个位错、无需重训练；缺点：只针对 8-bit 量化权重的特定值分布，只保护存储的权重。共同点：仍在保护原始数据比特，只是按值分布"挑选值得保护的位"，因此对 FP 类 DNN 的 exponent 位故障（位数不固定、位置随值分布变化）覆盖不足。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VAPI 执行流程（8-bit 权重）：① 离线统计权重分布，确定哪些高序位几乎恒为零 → 标记为 parity 位；② 存储时把 LSB/低价值位替换为 DEC(64,50) 编码生成的 parity 覆盖 64-bit 块；③ 读时解码，纠 ≤2 bit 错、否则 DUE；④ 纠错后恢复原值。Weight Nulling 执行流程：① 每权重存 1 bit LSB parity（奇偶）；② 读时校验，奇数个 bit 错 → 检错 → 该权重置 0（不纠值）；③ 偶数个 bit 错漏检（SDC）。RangeGuard 的对比结果（Table III）：两种方案对 32E（32-bit 簇错）几乎无纠正能力（Weight Nulling 32E SDC≈75%、VAPI 32E DUE≈99.994%），而 RangeGuard 4b DSC 对单/双故障场景 CE/BE 覆盖显著更强——证明"按位挑选保护"不如"按范围语义保护"。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：都是内存写入/读取路径上的轻量编解码逻辑（parity 位与权重位复用的 bit 重排 + 线性码编码/解码），无需重训练、无需模型架构改动，适合量化模型权重存储保护。使用场景与限制：VAPI 只适用于值分布稳定的 8-bit 量化权重（高序位确实冗余），Weight Nulling 适用于奇偶错误为主的场景但无纠错能力；两者都难以覆盖 HBM 的 SWL/SWD 类 16–32-bit 簇错与 FP 模型的 exponent 灾难性错误——这是 RangeGuard 论文将其作为"bit-centric"代表进行对比的原因。
涉及论文标题：
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs

## State Space Model（SSM / Mamba，状态空间模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
State Space Model（状态空间模型）是用线性递归状态 $h_t = \bar{A} h_{t-1} + \bar{B} x_t$、$y_t = C h_t$ 编码序列历史的模型族。S4（Gu et al., 2022）引入结构化参数化（HiPPO 初始化 + 对角 A 矩阵），实现 O(N) 复杂度长序列建模；Mamba（Gu & Dao, 2023）进一步做选择性 SSM——离散化步长 Δ 与输入投影 B、C 由输入动态生成（$B_t=W_B x_t$, $C_t=W_C x_t$, $\Delta_t=\mathrm{Softplus}(W_\Delta x_t)$），用 selective scan 替代卷积，保留 RNN 式递推但具备输入依赖的"选择/遗忘"能力；Mamba-2（Dao & Gu, 2024）用 SSD（Structured State Space Duality）统一 SSM 与线性注意力，支持 chunk 并行。与 Transformer attention（O(N²) 计算、O(N) KV cache）相比，SSM 是 O(N) 计算、O(1) 固定大小 state（per-layer hidden state），推理时无需 KV cache，内存占用与序列长度无关。本论文（Rearchitecting the Datacenter Lifecycle for AI）用 Mamba-2.8B vs Llama3-3B 的跨代 GPU 实验论证"模型架构决定硬件兼容性"：2K 序列 TP1 下 Llama3 在 V100 上比 H200 慢 7.7×，而 Mamba 仅慢 3.6×——state-space 架构与旧/弱 GPU 更兼容，延长旧硬件生命周期。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba 单层的推理 pipeline（对照 attention）：
```
输入 x_t ∈ R^d（第 t 个 token 激活）
# 1. 输入投影
Δ_t = Softplus(W_Δ x_t);  B_t = W_B x_t;  C_t = W_C x_t
# 2. 选择性离散化（ZOH）
Ā_t = exp(Δ_t A);  B̄_t = (Δ_t A)^{-1}(exp(Δ_t A) − I)·Δ_t B_t
# 3. 递归状态更新（decode 逐 token）
h_t = Ā_t h_{t-1} + B̄_t x_t          # O(d_state × d) 矩阵-向量，state 固定大小
# 4. 输出
y_t = C_t h_t                          # + output projection 到词表
训练/长 prefill 用 selective scan 并行；推理 decode 只做 3 步的矩阵-向量
→ 无 KV cache 增长、无 attention 的 O(N) 内存；每步工作量恒定
```
论文的硬件观察：decode 本就是 memory-bound 低算术强度，SSM 因省去 KV cache 的加载与 attention 计算，在内存带宽更弱的旧 GPU（V100）上退化远小于 transformer——这正是"架构选择影响硬件寿命"的证据：若 fleet 以 SSM 为主，旧 GPU 的可服务年限显著延长，刷新节奏可放缓。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：Mamba 官方代码（https://github.com/state-spaces/mamba）、Mamba-2 与 Jamba（AI21，Mamba+Transformer 混合）、NVIDIA Mamba-2-Hybrid 系列（4 attention + 24 SSM + 28 MLP 层）；推理运行时（vLLM 等）已支持 Mamba 系模型，decode 用逐 token 递推（state 常驻寄存器/片上）、prefill 用并行 scan 或 chunk-scan kernel（TileLang 对比 Triton 的 chunk-scan 平均 1.77×、chunk-state 2.10× 加速）。论文用 Mamba-2.8B 在 T4/V100/A100/H100/H200 上跑 vLLM 测 TTFT/TBT（2K 序列、TP1、batch 8，按 H200 归一化），结论并入 TCO 框架的 workload 模型（模型架构参数决定 roofline 的算术强度与内存占用），供刷新策略判断"未来模型若转向 SSM，旧硬件仍具竞争力"。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI

## SAT（布尔可满足性）与 CNF（合取范式）/ kSAT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SAT（Boolean Satisfiability，布尔可满足性）是判断是否存在一组布尔变量赋值使给定命题公式为真的问题，是第一个被证明为 NP-complete 的问题（Cook-Levin 定理），也是组合优化问题（COP）的经典代表。标准表示是合取范式 CNF（Conjunctive Normal Form）：一组子句（clause）的合取，每个子句是若干文字（literal）的析取，文字是变量 x 或其否定 ¬x。一个 CNF 实例可满足当且仅当每个子句都能取真。kSAT 表示每个子句恰好含 k 个文字的 SAT 子类；任何 k>3 的 kSAT 子句可在多项式时间内转化为等价的 3SAT 子句集合（kSAT→3SAT 转换）。SAT 的实际应用覆盖软件/硬件验证、计算生物学、密码分析、金融建模与神经网络验证（SAT 已成为形式验证的标准模型）。在 SATIC 论文中，SAT 是待映射到 Ising 机器的输入问题：变量对应 Ising spin（或 QUBO 二值变量），子句对应 spin 间交互，满足赋值对应能量低态。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
从算法 pipeline 角度，一个 SAT 求解流程（以 3SAT 为例，DIMACS CNF 格式输入）：
```
# 输入：CNF，如 F = (x1∨x2∨x3) ∧ (¬x3∨x4∨x5)（DIMACS：p cnf 5 2）
# 输出：可满足赋值 或 UNSAT
assign = {}                              # 部分赋值
repeat:
    (unit, val) = find_unit_clause(CNF, assign)   # 单元传播：单文字子句强制赋值
    while unit: assign[x]=val; simplify(CNF)
    if empty_clause(CNF): backtrack      # 冲突 → 回溯（DPLL/CDCL）
    if all_vars_assigned and all_clauses_true: return SAT, assign
    x = pick_variable(CNF)               # 变量选择启发式（如 VSIDS）
    branch on x=0/1
```
在 SATIC 的 Ising 编译 pipeline 中，SAT 问题作为 CNF 输入后：构建 VIG（变量为节点、共现为边）→ 在 CNF 层做子问题形成（freeze 未选变量 + 单元传播化简）→ 公式化为 QUBO（引入 ancillary 变量）→ Ising 硬件退火 → 回收解更新全局解向量 → CheckSolution 验证全部分子句。示例：F=(x1∨x2∨x3)∧(¬x3∨x4∨x5)，冻结 x4=0、x5=0 后单元传播得 (x1∨x2∨x3)∧(¬x3)，公式化仅需 4 spin。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：经典求解器（DPLL/CDCL，如 MiniSat/CaDiCaL/WalkSAT）用回溯 + 冲突学习 + 启发式搜索；Ising/量子路线（本论文）把 SAT 编码为能量最小化问题交给 Ising 机/退火器（Lucas 证明 Karp 的 21 个 NP-complete 问题都可写成 Ising 哈密顿量）。使用方式：SAT 实例用 DIMACS CNF 格式存储（行 `p cnf <n> <m>` 声明变量与子句数，每行以 0 结尾列出一个子句的正负整数文字），SATIC 读取该格式后执行编译流程；benchmark 来源包括 SATLIB 与 QuICC-SAT-Datasets。SAT 在验证领域通常与 bounded model checking、equivalence checking 结合使用。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

## Transition Region（相变区 / 过渡区，clause-to-variable ratio）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transition region 是随机 SAT 问题的难度相变现象：随机均匀生成的 SAT 实例，其可满足性概率与难度由子句-变量比（clause-to-variable ratio，m/n）决定。当 m/n 较小时实例几乎都可满足（过约束不足、易解），m/n 较大时几乎都不可满足（过约束、也易证伪），而在某个临界比值附近（SAT/UNSAT 相变的过渡带），实例最难求解——解的存在概率在此从 1 急剧下降到 0，求解器所需搜索量在该区域达到峰值。对 3SAT，相变区约在 m/n ≈ 4.26（论文给出 ≈4）；对 4SAT 可到 ≈10；k 越大比值越高。随机 SAT 问题在相变区内被广泛用作 stressmark（压力测试基准），因为该区域对各类求解器（经典、量子、Ising）都极具挑战。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
从算法 pipeline 角度，相变区决定基准生成的难度控制：
```
# 生成随机 kSAT 基准：固定 n（变量数），扫描 m/n 比值
for ratio in [2.0 .. 6.0]:                    # 3SAT 相变区 ≈ 4.26
    instances = []
    for i in 1..N:
        F = random_kSAT(n=k*n_ratio, k=3)     # 均匀随机抽变量+极性构成子句
        instances.append(F)
    hardness[ratio] = measure_solve_time(instances)   # 峰值即 transition region
```
SATIC 论文的用法：seen 基准 Batch-4-100-1000 是 100 变量/1000 子句 4SAT，m/n=10 恰好落在 4SAT 相变区，作为压力测试；unseen 基准 SATLIB UF 系列（UF20~UF250）都是相变区随机 3SAT（如 UF250：250 变量/1065 子句，m/n≈4.26，near phase transition）。相变区实例"解稀疏"，因此对 Ising 编译中的能量景观失真（ancillary 固定导致错位）极敏感——这正是论文论证 ancillary-awareness/clause-completeness 关键性的场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：随机生成（uniform random kSAT generator）按固定 n、m 抽样变量与极性即可构造。使用：求解器/硬件研究用它制造最坏情况基准（stressmark），衡量鲁棒性与可扩展性；SATIC 用它验证 45-spin 芯片在 73× 容量压力下的表现（Batch-4-150-1570 等 seen 批次）、UF250 等 unseen 批次。相关理论：SAT 相变与随机图论、临界现象联系（与 k-COLORING、随机图连通性相变同族）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

## SATLIB 基准（UF 系列 / CRAFTED QuICC-SAT-Datasets）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SATLIB 是 SAT 研究领域公开的基准库（Hoos & Stützle，SAT 2000，http://www.satlib.org），提供大量随机与结构化的 SAT 实例，是 SAT 求解器评测的标准资源。其 UF 系列（Uniform Random 3-SAT）是位于相变区的随机 3SAT 实例（如 UF20：20 变量/91 子句、UF50：50/218、UF75：75/325、UF100：100/430、UF125：125/538、UF150：150/645、UF175：175/753、UF200：200/860、UF225：225/960、UF250：250/1065），naming 按变量数区分，公认难解。QuICC-SAT-Datasets（UMD-ARLIS，https://github.com/UMD-ARLIS/QuICC-SAT-Datasets）是另一组面向量子启发求解器评估的 SAT 基准（含 quiet planting 型 CRAFTED 与 AI planning 型 CRAFTED）。SATIC 论文将 SATLIB UF 系列作为 unseen（未见）测试集验证泛化性，将 CRAFTED 系列作为 seen（定制压力）测试集。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
基准使用流程（SATIC 评估 pipeline）：
```
# 批次（batch）= 同配置（k, n, m）的实例组；实例 = 单个 SAT 问题
for batch in [Batch-4-50-500, ..., UF20, UF50, ..., UF250]:   # Table III
    for inst in batch:                          # 每批 ≥50 实例
        for rep in 1..120:                      # repeats ≥100 保证统计
            (sol, ok) = SATIC_compile_and_run(inst, max_iter=50K)  # 迭代=硬件调用数
            if ok: repeats_success += 1
        solved[inst] = (repeats_success > 0)    # 任一次成功即算解出
    batch_solved = all(solved[inst] for inst)
```
SATLIB UF 系列被用作 unseen 测试（训练/调参时未见），CRAFTED seen 系列用于探测容量上限（Ratio 列 = QUBO 规模/45-spin 容量，如 Batch-4-150-1570 达 73.1）。评估结果：SATIC++ 在 UF175（23× 容量）之前全解，UF200/225/250 解 98/97/92。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SATLIB 提供 DIMACS CNF 格式文件（网页/镜像下载），UF 系列来自 uniform random generator 的固定种子生成；QuICC-SAT-Datasets 提供 GitHub 仓库（按 quiet planting、AI planning 等类别组织，含 generator 与实例）。使用：评测器按批次-实例-重复的三层结构运行（论文定义 batch/instance/repeats/iteration count 指标），用 TTS 与 solved 比例比较不同编译器/分解器（SATIC vs SATIC++ vs D-Wave EID）；也是很多经典与量子求解器论文的标准对照。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)

## Weight / KVCache / 中间激活（IA）：LLM 推理三类数据与异质性表征（数据量/事务/写强度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LLM 自回归推理涉及三类数据：Weight（静态权重矩阵）、KVCache（QKV Generation 层在 decode 每步动态写入的键值缓存，避免重算）、中间激活 IA（层间中间张量）。三者异质性显著：数据量上 Weight 与 KVCache 主导（$D_{Weight}=12d^2L/(p_p p_t)$，$D_{KVCache}=2dsbL/p_t$，d=隐藏维、L=block 数、b=微批、s=序列长、p_p/p_t=流水/张量并行），IA 峰值执行足迹小几个量级（Fig. 4a 跨 GQA/MoE 结构不变）；事务上 Weight/KVCache 读主导（体积大且反复重载）、KVCache 写最少（每 entry 只写一次）、prefill 主导时读下降而长 prompt 的 IA 流量大（Fig. 4b）；写强度 = 写事务/容量/时间，反映 cell 访问频率（Fig. 4c 上界=集中写、下界=理想磨损均衡）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 GPT3-175B（MHA+Dense，d=12288、L=96）微批 64 为例：prefill 处理 prompt token（GEMM，Weight 从 NVM 读、IA 在 DRAM 计算/写回），decode 逐 token 生成（GEMV，从 KVCache 读历史 KV、新 KV 写入 KVCache）；KVCache 随序列/批增长主导内存。三类数据按度量偏好映射到混合内存：Weight/KVCache = 容量+读密集、写稀疏 → NVM；IA = 写密集、容量小 → DRAM（KVCache 溢出放 DRAM）。GQA（Mixtral）KV head 少 → KVCache 写事务进一步被抑制；MoE 粗粒度专家激活 → Weight 大块连续读，保持 NVM 读带宽利用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在算法/系统层面，该表征指导静态数据放置（SHyLA 的 runtime 初始化时按类别放置）与混合内存设计（NVM 面积优先给 Weight 操作、DRAM 保留最小容量给 IA + 其余转带宽）。量化/压缩（INT8 Weight）与分类放置正交。三类数据的体积公式、事务分布与写强度图（Fig. 4）为后续研究提供了"workload–hardware 接口"的数据侧模型。SHyLA 数据（仿真按 Sec. VII-A 方法学，微批 64）论文未开源（联网未找到）。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

## 运动矢量（Motion Vector，H.264）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
运动矢量是视频编码中记录"当前块相对参考帧的坐标位移"的元数据。H.264/AVC 中编码器把帧划分为 16×16 宏块（可再分割为 8×8、4×4 等更小分区），对每个分区在参考帧中做运动估计（motion estimation，搜索最佳匹配块），将坐标偏移编码为运动矢量。运动矢量只建模线性平移，无法表达旋转、缩放、遮挡/去遮挡与非刚性运动——这些由残差捕获。解码端解析运动矢量并对参考帧做运动补偿（motion compensation）生成预测块。在 SLICE 中，运动矢量网格（4×4 块粒度，每元素一个 MV）被当作"该区域是否静态/运动剧烈程度"的代理信号：MV 均值=0 表示无运动（可跨帧复用）；MV 幅值大表示快速运动区域（帧内易模糊，SR 增益小）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 SLICE 的 Patch Analysis（Algorithm 2）中，运动矢量网格 G^mv 经平均池化聚合为每 patch 的统计量，参与两类决策：
```
mv_mean = AvgPool2D(G^mv, kernel=P/4, stride=P/4)   # G^mv 是 4×4 块粒度网格，P=16 → 核/步长 4
# ① 复用判定：mv_mean==0 且像素域残差均值==0 → M^reuse
# ② SR 打分：score = 0.9·hf_ratio + 0.1·(1 − clip(mv_mean/10, 0, 1))
#    运动项 clip(mv_mean/10)：MV 幅值大 → 该项接近 0 → 分低 → 不选做 SR
```
例子（270p 帧，P=16）：某静态背景 patch 内 4×4 块 MV 全为 0 → mv_mean=0 → 与残差共同判定复用；某快速运动物体 patch 的 mv_mean≈25 → 归一化 clip(25/10)=1 → 运动项 0 → 仅靠残差项得分，难以进入 TopK。论文实测 270p/540p 视频中 MV 幅值超过 10 的块仅占 3.6%/6.2%，故除以 10 是合理的归一化。Fig.7(c) 显示 patch 平均 MV 幅值越大，SR 相对插值的期望 PSNR 增益越小（时序错位与模糊所致），这是把 MV 作为负向信号的理论依据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
编码器侧由 x264/x265/FFmpeg 等实现运动估计（全搜索/菱形/HEPS 等算法），MV 以差分形式写入码流语法元素；解码侧硬件/软件解码器解析 MV 做运动补偿。SLICE 用扩展版 Compressed Video Reader（基于补丁化 FFmpeg，https://github.com/Yaojie-Shen/Compressed-Video-Reader）在 H.264 解码过程中导出每块 MV 网格，模拟 SoC 硬件解码器未暴露的码流侧信号；由于只读取不改写 bitstream，解码仍走标准硬件解码器。论文用 MV 与残差联合而不是单用 MV（MV-only baseline 质量更差，因为会优先选中静态背景而非高频区域）。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## 残差（Residual：像素域与频域，H.264）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
残差是运动补偿预测与原始帧的差值，捕获运动矢量无法表达的细节（旋转、缩放、遮挡/去遮挡、非刚性运动）。编码流程：运动估计形成预测 → 残差 = 原始帧 − 预测 → 残差经 DCT 类整数变换（H.264 的 4×4/8×8 整数变换，近似 DCT 且保证精确反变换）转为频域系数 → 量化熵编码；频域中能量集中在少量低频系数、高频系数多接近零便于压缩。解码端：反量化+反变换（IDCT 类）得到像素域残差，加回运动补偿预测重建帧。SLICE 同时使用两类残差：频域残差的高频能量占比指示"SR 增益最大的区域"（边缘/纹理）；像素域残差均值=0（且 MV=0）指示"可复用的静态区域"——复用判定用像素域残差而非变换系数，是因为 inter 帧可能含 intra 块（无 MV），像素域残差可防止这类块被错误地跨帧复用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 中残差网格的聚合与使用：
```
res_pixel_mean = AvgPool2D(|G^pix|, kernel=P, stride=P)      # 像素粒度残差，核/步长 P=16
hf_ratio       = AvgPool2D(G^hf, kernel=P/4) / AvgPool2D(G^t, kernel=P/4)  # 频域 4×4 块粒度
# 复用：R = (mv_mean==0) ∩ (res_pixel_mean==0) → M^reuse
# 打分：score = α·hf_ratio + β·(1 − clip(mv_mean/10,0,1))，α=0.9, β=0.1
```
例子：一块纹理丰富的 patch 的频域残差大部分能量落在高频带 → hf_ratio 高 → 得分高进入 SR TopK；一块纯色背景 patch 的频域残差几乎全低频 → hf_ratio≈0 → 被插值处理。Fig.7 给出定量规律：patch 中低频残差占比越高，SR 相对插值的期望 PSNR 增益越低（插值已接近最优）；高频残差占比越高，增益越大。Fig.18 参数消融显示 α=0.9/β=0.1 最优——残差直接反映预测误差、比 MV 更可靠，但仍给 MV 一个小权重补充信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
编码器实现：变换+量化+熵编码；解码器实现：熵解码+反量化+反变换得到像素域残差、加回运动补偿预测。SLICE 用扩展版 Compressed Video Reader 在 H.264 解码时解析频域残差网格，并对残差做 IDCT 类反变换得到像素域残差网格，供 patch 分析使用；整个 bitstream 不被修改，与硬件解码器兼容。残差的频带划分与变换系数区域一致（Fig.3：左上为低频、向右/向下为高频）。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## Patch 级选择性超分推理（Selective Patch-level SR Inference）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把视频帧划分为等尺寸非重叠 patch，逐 patch 决定上采样策略（SR 推理 / 跨帧复用 / 像素插值），只对信息量最大的区域执行昂贵的 SR 模型推理，从而在满足实时预算（30FPS/33ms）的前提下把 SR 计算量降到最低。核心动机是 SR 增益在空间上高度选择性：实测 44.1% 的 patch 上 SR 相对插值无增益甚至负增益（Fig.4），背景/平坦区域 SR 与插值视觉等价；patch 级高频残差占比与 PSNR 增益正相关、大 MV 区域增益小甚至为负（Fig.5/7）。SLICE 默认参数：patch 16×16、推理面积比 k=35%（按 score TopK 选出）、intra 帧做全帧 SR、inter 帧三路选择。效果：2.72× 帧率提升、62.57% 能量节省、PSNR 仅降 0.35dB（对比无复用变体 SLICE-noreuse 的 0.78dB，说明复用的质量贡献）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1 伪代码：
```
for each frame f in video:
    if INTRACODED(f):            # 每个 GOP 开头的 I 帧，占比小
        F^SR ← FULLFRAMEInference(f)
    else:                        # 占绝大多数的 inter 帧
        (M^reuse, M^SR) ← PATCHANALYSIS(f)        # codec 元数据驱动，全 GPU
        P^HR ← PATCHWISEUPSCALE(f, M^SR, M^reuse) # reuse 直拷 / SR 推理 / 插值
        F^SR ← MERGEPATCHES(P^HR)
```
例子（270p 帧，P=16 → 30×17 patch）：MV 与像素域残差均为 0 的静态 patch → 复用；score=0.9·hf_ratio+0.1·(1−clip(mv/10)) 前 35% 的 patch → EDSR 推理；其余 → bicubic 插值。patch 大小权衡（Fig.16）：32×32 接受野大、单 patch 质量高，但 patch 变大导致复用率下降、复用收益变小；16×16 只牺牲少量接受野而显著提升复用机会，综合质量最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 PyTorch GPU 管线：unfold 把 patch 网格转为紧凑张量，按 M^SR gather 出需推理的 patch 组成 batch 做一次或少数几次 EDSR(FP16) forward；复用 patch 从 GPU 常驻 HR cache 直拷；其余 patch GPU 插值；按行分带（row-wise banded）合并写 framebuffer。硬件平台为 NVIDIA Jetson AGX Orin，能量用 Tegrastats 测。与模型级高效 SR（APE 的 patch 级 early exit、轻量/量化 SR 模型）正交，可叠加。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## Patch 复用与 HR Cache（Patch Reuse & HR Cache）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
利用视频时域冗余：连续帧间静态背景等区域在运动矢量与残差均为零（预测完美、内容无变化），可直接把上一帧超分结果（HR cache）中对应位置的像素拷贝过来，完全跳过该 patch 的 SR 推理与插值。零 MV+零残差像素占比：Vimeo90K 30.1%、Kinetics-400 25.5%、K600 25.3%、K700 20.7%（Fig.6），表明真实视频中存在大量可复用区域。复用正确性由实验保证：Fig.8 显示任意复用比例下，把上一帧超分 patch 贴入当前帧造成的 ΔMSE 以 <0.2% 幅度集中在零附近。复用还提升质量：SR 增强的细节从早期帧传播到后续帧（SLICE 0.35dB vs SLICE-noreuse 0.78dB 质量损失）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 的复用判定 + 上采样阶段的使用：
```
# 判定（全 GPU）：R = (mv_mean==0) ∩ (res_pixel_mean==0) → M^reuse
# 使用：PATCHWISEUPSCALE 中 reuse patch 从 HR cache（上一帧 SR 结果常驻 GPU）按坐标直拷；
#       MERGEPATCHES 中把水平相邻复用 patch 合成连续带整段拷贝，减少拷贝开销
```
例子：某直播视频的固定演播室背景 patch 在连续几十帧中 MV 与残差均为 0 → 每帧都从 HR cache 直拷，SR 推理量随复用率进一步下降；I2 视频复用率 79.43% → SLICE 达 52.19 FPS。Fig.8(b) 显示加速比随复用率超线性增长：极端整帧可复用场景达 98.24×（运行时间约为全帧 SR 的 1.02%）。与 frame-skip 方案的区别：SLICE 每帧都分发 patch 级更新，而 frame-skip 整帧跳过会导致新增高频区域得不到 SR 更新（Hybrid baseline 因此质量更低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现 = GPU 显存中的上一帧 HR 结果缓存 + 按 mask 的 banded 拷贝；缓存与合并全部留在 GPU 侧，避免 CPU-GPU 往返（即使 Jetson 统一内存架构下也仍存在逻辑内存拷贝）。复用判定顺序优先于 SR 选择（先找可复用 patch 缩小候选集），以最大化复用收益。GOP 敏感：GOP 越长 inter 序列累积运动估计误差、残差变大、复用率下降（Fig.17），但更长 GOP 也减少 intra 帧的全帧 SR 次数。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## Patch Statistics Maps（PSM）与 Codec 引导的 Patch 分析（Codec-guided Patch Analysis）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSM（Patch Statistics Maps）是把解码器产出的码流元数据网格（MV 幅值、像素域残差、频域高频/总能量）经平均池化聚合为每 patch 的统计量图（mv_mean / res_pixel_mean / hf_ratio），作为"该 patch 是否值得 SR 推理"的运行时信号。Codec 引导 = 用标准 bitstream 中解码器本就要解析的元数据做 SR 调度决策，无需服务器辅助、无需修改 bitstream，因此硬件视频解码器可被完整使用，调度运行时开销可忽略。SLICE 的 patch 分析三步：先识别复用 patch（MV=0 且像素残差=0），再按 score 的 TopK 选 SR patch（预算 k=35%），其余插值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 全流程（P=16，全 GPU）：
```
# ① 生成 PSM（AvgPool2D 各一次完成聚合）
mv_mean      = AvgPool2D(G^mv,  kernel=P/4)        # MV 网格 4×4 块粒度 → 核/步长 4
res_pixel_mean = AvgPool2D(|G^pix|, kernel=P)      # 像素粒度 → 核/步长 16
hf_ratio     = AvgPool2D(G^hf, kernel=P/4) / AvgPool2D(G^t, kernel=P/4)  # 频域块粒度
# ② 识别复用 patch
R = (mv_mean==0) ∩ (res_pixel_mean==0);  M^reuse[R]=True
# ③ 打分并 TopK 选 SR patch
score = α·hf_ratio + β·(1 − clip(mv_mean/10, 0, 1))     # α=0.9, β=0.1
S = TopK(score, k=35%);  M^SR[S]=True                   # GPU TopK kernel
```
例子：270p 帧 30×17 个 patch 中，hf_ratio 高的纹理 patch 得分靠前入选 SR；hf_ratio≈0 的平坦 patch 被插值；复用 mask 先行剔除静态 patch，避免 SR 名额浪费。Fig.15 设计空间显示 35% 为吞吐/质量折中（40% 仅多 0.08dB）；Fig.18 显示默认权重最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PSM 聚合用 PyTorch 的 AvgPool2D（像素粒度核/步长 P、4×4 块粒度核/步长 P/4），一次池化出全帧每 patch 均值；TopK 用 GPU 排序/选择 kernel。元数据网格来源：扩展版 Compressed Video Reader（补丁化 FFmpeg）在 H.264 解码时导出 G^mv/G^pix/G^hf/G^t，仿真硬件解码器未暴露的码流侧信号；部署时若解码器开放相关接口可直接读取。复用判定特意用像素域残差（而非变换系数）以排除 inter 帧中的 intra 块被误复用。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## EDSR（Enhanced Deep Residual Networks）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EDSR 是 CVPRW 2017 的经典单图超分（SISR）模型（Lim et al.）：堆叠去除 BatchNorm 的残差块（常规 16 或 32 块），仅用卷积+ReLU，避免 BN 在小 batch 归一化伪影与对超分任务的负作用，提升训练稳定性与表现；设计上支持多尺度（同一主干共享、仅尾部上采样模块按倍数切换）。SLICE 以 EDSR 作为 4× 上采样基准 SR 模型，全部推理在 Jetson AGX Orin GPU 上用 FP16 执行。论文实测 EDSR 推理延迟是 bicubic 插值的 120.7×——这个巨大计算差距正是 SLICE 选择性推理的核心动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 SLICE 管线中 EDSR 只对选中的 patch 批做 forward：
```
# inter 帧：270p(480×270) → 30×17 个 16×16 patch → TopK 选 ~35%≈178 个
#   → unfold 聚合成 (178, 3, 16, 16) batch → 一次 EDSR(FP16) forward
#   → 输出 4× 的 (178, 3, 64, 64) patch → 合并成 1080p 帧
# intra 帧（GOP 开头）：全帧 forward
```
EDSR 的残差学习结构（残差块：Conv→ReLU→Conv + 跳跃连接）与 SLICE 的"patch 级选择性调用"正交：模型参数/权重共享，SLICE 只决定何时何地调用该模型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现众多（如 EDSR-PyTorch，基于 DIV2K 数据集训练，可下载预训练权重）；SLICE 以 PyTorch FP16 部署于 Jetson AGX Orin GPU。论文未明确说明其使用的具体仓库与权重来源（记为论文未明确说明）。与模型级高效 SR（APE 的 patch early-exit、量化/轻量 SR 模型）互补：模型效率降低单次推理成本，SLICE 降低推理次数。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## Operational Intensity（运算强度 OI，compute-to-memory 比率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- OI 定义为"计算量/访存量"的比率（FLOPs per byte moved），roofline 模型中决定一个算子落在 compute-bound（高 OI，受算力限制）还是 memory-bound（低 OI，受带宽限制）区域。SMOOTH（ISCA'26）用它解释移动 NPU 上 LLM decode 的突发带宽：prompt 期全序列 self-attention 是 GEMM（高 OI、compute-bound）；token 生成期输入从 d×l 矩阵缩为 d×1 向量、attention 矩阵从 l×l 缩为 l×1，反复执行低 OI 的 GEMV（QKV 投影、W0 等线性运算需搬动整块 d×d 权重只做少量计算、极度 I/O-bound），而 softmax/GELU 等非线性运算主要靠向量吞吐、高 OI、带宽严重空闲。两类算子交替执行 → 带宽一会儿饱和一会儿空闲的 bursty 流量。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 单层 decode 执行拆解（移动 NPU，batch=1）：QKV 投影 GEMV（OI=2·d·1/(4·d) ≈ 0.5 FLOP/byte 量级，I/O-bound，带宽饱和）→ FlashAttention 内 QK^T/softmax/SV（softmax 高 OI、带宽空闲）→ 输出投影 W0 GEMV（低 OI）→ FFN W1 GELU W2（W1/W2 低 OI、GELU 高 OI）。SMOOTH 的量化观测：非线性（高 OI）运算占端到端时间 10–20%（TinyLLaMA 在 Jetson 20.4%、S24 17.0%、EdgeTPU 14.1%；模拟器保守估计 9.4%/5.7%），这些高 OI 阶段正是可被预取利用的空闲带宽窗口。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：OI 是 roofline/架构分析的核心指标，SMOOTH 用它做动机分析（哪类算子带宽瓶颈、哪类有预取头寸），并用于判断静态编译器为何失效（tile size 的选择受运行期 OI 变化影响，序列长度与带宽波动使静态选择最多恶化延迟 2.9×）。也常见于 PIM/近存架构论文（如 Pimba、BAAP、InstAttention 用 OI 划分 GeMV→memory 单元、GeMM→compute 单元）。SMOOTH 不改变模型 OI 特性，而是用硬件内存管理把低 OI 阶段的带宽需求摊平到高 OI 阶段的空闲窗口。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference

## KV Cache（键值缓存，Key-Value Cache）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KV Cache 是自回归 LLM 推理（prefill/prompt 与 decode/token 生成两阶段）用于避免重复计算历史 token 的 Key/Value 激活缓存：prefill 阶段并行处理整个输入序列并计算各层 K/V 存下；decode 阶段每步只算新 token 的 Q，与缓存的历史 K/V 做注意力。代价是内存占用随序列长度线性增长（长上下文下成为主要片上/片外开销），且 K/V 访问量随序列增长而增大访存流量。SMOOTH（ISCA'26）在移动 NPU 场景把它作为运行期动态因素与内存压力源：① KV cache 大小随用户序列长度变化，编译期无法预知，静态 tile size 因此失效（延迟最多恶化 2.9×）；② decode 期 KV cache 数据不断涌入片上，其块（如 V_cache 单 block）在 S×V 计算进行中逐块消费——SMOOTH 的 block 级分配允许"V_cache 单块一空出就用于预取"，且长序列（32K）下 KV cache 内存开销显著，SMOOTH-ER 较 Gemmini 平均收益从 2K 的 50.1% 增至 32K 的 66.8%。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- decode 单步执行拆解（含 KV cache）：输入 token x_t → 与缓存 K_{1:t-1}、V_{1:t-1} 拼接做注意力：
```
Q_t = x_t @ W_Q            # 只算新 token
K_t = x_t @ W_K; V_t = x_t @ W_V
K_1:t = concat(K_1:t-1, K_t)   # KV cache 追加（随序列增长）
V_1:t = concat(V_1:t-1, V_t)
score_t = Q_t @ K_1:t^T / sqrt(d)   # 矩阵从 l×l 缩为 l×1
ctx_t = softmax(score_t) @ V_1:t
```
注意矩阵从 l×l（prefill）缩为 l×1（decode），配合权重矩阵的 d×d GEMV 形成低 OI 的 I/O-bound 执行。SMOOTH 的关注点：K/V 数据的片上放置与预取——V_cache 块在 attention 计算中被逐块消费后可 early reclaim，块级预取把 KV 流量摊平。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：KV cache 在移动 SoC 上由编译器分配进 SPM 或随权重混排；SMOOTH 的 block 表/bitmap 管理其片上驻留，use_cnt/end_cmd 驱动其消费后回收，N_preload 预取后续 KV 块。评估模型（TinyLLaMA 1.1B–GPT-3 13B，w4a8/int8）在 batch=1 下，KV cache 随输入序列（1K–32K）增长成为 generation 期延迟的主要来源；SRAM 占用/每 token 延迟实验中，无融合时 KV cache 增长使带宽饱和、各策略收益受限，融合后预取机会显现。论文未对 KV cache 做压缩/剪枝类算法改动（相关技术如 H2O、KV 量化属正交方向）。

涉及论文标题：
- SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference

## Fine-grained MoE（细粒度专家混合架构 / fine-grained expert segmentation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-grained MoE 是 DeepSeekMoE（Dai et al., 2024）提出的 MoE 架构变体：在总参数量不变的前提下，把传统 MoE 的少量大专家拆分为大量小专家（专家数量 N 增大、每个专家 FFN 中间维度缩小），并增大 top-k 使每个 token 激活更多专家，从而提升专家专业化（specialization）、允许更丰富的知识子域划分。核心数量关系：总参数量 ≈ N_experts × d_expert_size，细粒度通过增大 N、减小 d 保持参数量不变。与 vanilla MoE 的典型对比：Mixtral 式 E=8、top-2、d_ff=4h vs DeepSeekMoE 式 E=64、top-8、d_ff=h/4。DeepSeek-V2 扩展到 2 shared + 160 routed（top-6 routed），Qwen2-57B-A14B、XVERSE-MoE-A4.2B 均采用该设计。SMoE 论文正是以 fine-grained MoE 为研究对象：因为共享专家吸收通用知识 + 细粒度产生高度特化的非共享专家，激活专家中 gate score 高度不均（只有少数 top-score 专家显著影响输出），这构成了"专家替换"的算法前提。fine-grained MoE 的代价：top-k 更大使 dispatch/All-to-All 通信量随 top_k 线性增长（BigMac 表 1：top_k=8 时 All-to-All 占训练 91.8%、推理 90.6%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 同一 MoE 层，vanilla vs fine-grained 配置对比
# Vanilla（如 Mixtral）：E=8, top_k=2, d_ff=5632
# Fine-Grained（如 DeepSeekMoE / Qwen2-57B-A14B）：E=64, top_k=8, d_ff=704

# Fine-Grained MoE 前向（token 粒度）
x = input_token                 # [h]
logits = x @ W_gate             # [N] 全部专家打分
# SMoE 视角：logits 排序后高度不均——前几名（top-score）主导输出，
# 其余被激活专家（low-score）分数与未激活专家相当
topk_idx, topk_w = TopK(SoftMax(logits), k)
output = Σ_i topk_w[i] * Expert_i(x)     # 激活的 k 个专家 FFN（可选加 shared expert 项）
```
从系统角度，fine-grained MoE 使每个专家更小，单次加载一个专家的 PCIe 传输量更小、cache 可容纳更多专家（利于替换候选池），但每 token 激活专家数多导致未命中时加载次数多——这正是 SMoE 用专家替换把 low-score 专家加载量消掉的原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DeepSeekMoE 首次系统化提出（2 shared + 64 routed，top-6 routed）；DeepSeek-V2 扩展到 160 routed + 2 shared；Qwen2-57B-A14B（107GB，S3 设置）与 XVERSE-MoE-A4.2B 直接继承该设计。在 HuggingFace Transformers 中由 MoE 层 config（num_experts、num_experts_per_tok、expert intermediate size、shared_expert_intermediate_size）表达。相关系统工作：BigMac（DCCA 低维通信）、IFMoE（细粒度 MoE 推理框架）、X-MoE（HPC 上 DeepSeek 风格 expert-specialized MoE）、FasterMoE（fine-grained MoE 推理分析）、Scaling Laws for Fine-Grained MoE。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution

## Shared Expert（共享专家 / shared expert isolation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Expert 是 MoE 架构的一种设计：在 MoE 层中设置一组始终激活、不参与路由的专家，所有 token 都必须经过它们计算，与 router 选择性激活的 routed experts 并行；其输出与 routed 输出合并。动机：传统 MoE 中通用知识（common knowledge）被冗余存储于各专家，shared expert 把通用知识集中承载，让 routed experts 专注特化域，提升参数效率。典型配置：DeepSeekMoE 2 shared + 64 routed；DeepSeek-V2 2 shared + 160 routed；Qwen1.5-MoE 4 shared + 60 routed；Qwen2-57B-A14B、XVERSE-MoE-A4.2B 继承。SMoE 论文中 shared experts 有两个作用：(1) 常驻 GPU 的共享专家提供稳定的通用表示，使"用共享专家+缓存专家预测下一层 top-score 专家"的预取打分更准确；(2) 因 shared expert 吸收通用知识，非共享专家中 low-score 与 top-score 分化更明显，强化了专家替换的合理性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
h = input_hidden_state                     # 当前 token
# Shared Experts（始终激活，对所有 token，不经过 router）
shared_output = Σ_{i=1}^{n_shared} FFN_shared_i(h)
# Routed Experts（router 选择性激活）
logits = h @ W_gate
topk_vals, topk_idx = TopK(SoftMax(logits), K)
routed_output = Σ_i gate_weights[i] * FFN_routed_i(h)
output = shared_output + routed_output     # 合并
```
SMoE 的具体用法：解码每一层时 shared experts 与 attention、gate 同属常驻 GPU 的 common parameters；预取预测阶段用"GPU 中未共享专家（已缓存）+ 共享专家"生成 hidden state → 走下一层 attention（用下一层 KV cache）→ 计算下一层 gate 分数，从而在真实 router 运行前预测出下一层 top-score 专家并提前 PCIe 预取。命中率约 82%（95% 概率为 active）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在模型 config 中以独立参数组存在（如 Qwen 的 shared_expert_intermediate_size），HuggingFace Transformers 的 MoE forward 先算 shared experts 再算 routed experts 后合并。DeepSeekMoE 论文首次系统提出，Qwen/DeepSeek 系列采用；MoLE 中 shared expert 保持标准 FFN 计算、与 attention 一起常驻 VRAM 不参与 offload。研究还发现 shared experts 会降低局部路由一致性（bypass effect + 缩小 expert combination space），offloading 场景下需权衡（Not All Models Suit Expert Offloading）。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution

## Expert Substitution（专家替换）与 Expert-Cache Router（专家缓存路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Substitution 是 SMoE（NJU/Tsinghua/Honor，ISCA'26）提出的第三种 MoE 专家调度范式（前两种为预取类与剪枝类）：利用 router gate score 反映的专家重要性差异，把被激活但分数低（low-score）的专家替换为 GPU 显存中已缓存、gate score 与之相近的未激活专家，从而在不损失精度（α 阈值内）的前提下减少 CPU→GPU 的 PCIe 专家加载与 CPU 专家计算。核心观察：fine-grained MoE 中只有少数 top-score 专家显著影响输出，low-score 专家分数与未激活专家相当（routing noise + load balancing 使 tail expert 行为趋同），替换它们几乎不损失精度、甚至因抑制 noisy activation 而提升（论文表 VI 手动降分实验验证）。配套算法是 Expert-Cache Router（Algorithm 1）：按超参 α（substitution threshold）与第 k+1 高分 S_{k+1} 分档——score > (1+α)S_{k+1} 为 top-score 专家保留；(1−α)S_{k+1} ≤ score < S_{k+1} 且已在 GPU 或属 top-score 集的为可替换候选 E_s；S_{k+1} ≤ score < (1+α)S_{k+1} 为 low-score 专家 E_l——用 E_s 中最高分者替换 E_l 中不在 GPU 者，不足部分回退为 PCIe 加载/CPU 计算。优化目标为逐层 max |G∩E_a| + min(|E_l\G|, |E_s|)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Expert-Cache Router（Algorithm 1，单 token t，k=top-k，α 阈值）
S_t = gate_scores(t) sorted desc;  S_{k+1} = (k+1)-th score
T = (1+α)S_{k+1};  L = S_{k+1};  R = (1−α)S_{k+1}
for e in experts:
    if score(e) > T:        O[t] += e; C += e              # top-score，保留
for e in experts:
    if L ≤ score(e) < T:    B_t += e                        # low-score（E_l）
    elif R ≤ score(e) < L and (e in GPU or C): A_t += e     # 可替换候选（E_s）
if |A_t| ≥ |B_t|:  O[t] += top |B_t| of A_t                 # 全量替换
else:              O[t] += A_t; 剩余 |B_t|−|A_t| 个 low-score 走 PCIe/CPU
# 配合：score-aware eviction 保留高分专家扩大 E_s；top-score prefetch 保证 top 专家在 GPU
```
示例：Qwen2-57B-A14B（k 较大）中某层激活 5 个专家，原 GPU 命中 2/5；替换 low-score 专家 d,e 为 GPU 驻留的 f,g 后命中 4/5，预取量从 3 降到 1，GPU 命中率提升到 71%（S3 设置）。α 选取：min_α A(α) s.t. T(α) ≤ R（TPOT 预算），多项式拟合 T(α) 后一维搜索；S1/S2/S3 分别取 0.35/0.3/0.25。注意与 DIAMoND 的 Adaptive Expert Selection（冲突感知动态替换，用于边缘 MoE 推理的另一机制）名称相近但机制不同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SMoE 已开源：https://github.com/goingshr/SMoE（figshare: https://doi.org/10.6084/m9.figshare.31982136）。Python 3.13 free-threading（no-GIL）环境，Ubuntu dependency.sh 装 Rust 工具链并源码编译 tokenizers；运行入口 run.sh（环境变量传参）/ main.py，config JSON 字段：replaceScoreRatio（等价论文 --alpha，替换比例）、window_size（null=LRU）、if_prefetch、if_usecpu、if_replace。模型：deepseek-moe-16b / Qwen2-57B-A14B-Instruct / XVERSE-MoE-A4.2B；GPU：3080Ti 12GB / 4060Ti 16GB / A6000 48GB（PCIe 3.0/4.0）。效果：TPOT 相对最优 baseline 降 24%（batch=1）/35%（batch=3），S3 达 48%；GPU 命中率 >60%；α≤0.35 精度无损。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution

## Top-K Gating（门控路由 / Gating Network）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-K Gating 是 MoE 层的路由决策机制：gating network（通常 1-2 个前馈层 + softmax + top-k 选择）对每个输入 token 计算所有专家的路由分数，选出分数最高的 k 个专家参与计算，其余专家不激活。在 STEP 论文中，gating 打分公式为 s = W_gate·x + b_gate（Eq.4），选出 top-k 索引 l_1..l_k，路由权重经 softmax 归一化 w_i^r = e^{s_{l_i}} / Σ e^{s_{l_j}}（Eq.5），输出 y_routed = Σ w_i^r·E_{l_i}(x)（Eq.6）。关键洞察：不同层的 top-k 权重分布极不均——部分层的低排名专家（如 top-4 的第 3/4 名）平均路由权重 ≤0.05（Fig.3，Qwen1.5-MoE-A2.7B 与 DeepSeek-V2-Lite-Chat 在 MMLU 上），对输出贡献极小却照常进入加载与计算；这构成 STEP"空间感知分配"的直接动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def topk_gating(x, W_gate, b_gate, k, N):
    # x: [seq, hidden]；N = 该层 routed 专家总数
    s = x @ W_gate + b_gate          # (seq, N) 路由分数（Eq.4）
    topk_idx = argsort(s, desc)[:, :k]            # 每 token 选 top-k 索引
    topk_w   = softmax(gather(s, topk_idx), dim=-1)  # (seq, k) 权重（Eq.5）
    # STEP 的扩展：gating 仍对全部 N 个专家算分（含已当选的临时 shared），
    # 保证专家统计一致、支持下一窗口选举；但只有未当选的 k−c 个走动态选择
    return topk_idx, topk_w
```
Annotations：k=每 token 激活 routed 专家数（Mixtral top-2、Qwen1.5-MoE top-4、DeepSeek-V2-Lite top-6），N=routed 专家总数（8/60/64），s=路由分数向量。固定 top-k 的问题：简单 token 浪费计算（高权重集中在 1 个专家时其余 k−1 个几乎白算），困难 token 可能计算不足；STEP 用层内阈值 θ 与窗口投票把"固定 k"变成"层内动态 k_l + 窗口内 k−c"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HuggingFace Transformers 的 MoE layer forward 中 gate 为 nn.Linear(hidden, N)（可带 bias，如 Mixtral 的 noisy top-k gating），后接 softmax+topk；辅助负载均衡损失（switch loss/aux loss）防专家坍缩。STEP 的离线阶段用校准数据集前向收集每层 top-k score 分布，据此设定归一化权重阈值 θ（默认 Mixtral 0.25、Qwen 0.13、DeepSeek 0.07）确定层内 k_l；在线阶段每 decode step 记录 top-2k 专家投票（不只 top-k，扩大候选视野），窗口末按票数选举临时 shared。STEP 与一般 MoE 的区别：gating 计算不缩水（仍算全部专家分数），缩水的是"参与加载与计算的专家集合"。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## Shared Experts 与 Routed Experts（共享专家与路由专家）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE 层的两类专家：Routed Experts 由 gating 每 token 动态选择、输出按 softmax 路由权重加权聚合；Shared Experts 对所有 token 恒激活、不参与路由，其输出以等权平均或加权和加入（STEP Eq.3：y_shared = Σ w_i^s·E_i^s(x) 或 ΣE_i^s(x)/j）。代表配置（STEP 表 I）：Mixtral-8x7B 无 shared（32 层、8 routed、top-2、routed expert (4096,14336)、激活 13B/总量 46.7B）；DeepSeek-V2-Lite-Chat 2 shared + 64 routed（top-6、routed expert (2048,1408)、激活 2.7B/总量 14.3B）；Qwen1.5-MoE-A2.7B 4 shared + 60 routed（top-4、shared (2048,5632)、routed (2048,1408)、激活 2.4B/总量 16B）。共享专家因恒激活可整体预加载、不进入 T_load 公式（STEP Eq.1 明确把 shared 排除）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一个含 shared 的 MoE 层前向（STEP 视角）
y_shared = mean/weighted_sum(FFN_shared_i(h) for i in 1..j)   # 全部 token 恒算
logits = gate(h)                                    # (N,)
topk_idx, w = topk(softmax(logits), k)              # routed 动态选择
y_routed = Σ_i w_i * FFN_routed_{idx_i}(h)          # 只算被选 routed
y = y_shared + y_routed                             # 合并（Eq.2）
```
Annotations：j=shared 数、k=routed 激活数、N=routed 总数。STEP 的关键算法改动：把"选中的高频 routed 专家"在一个 token 窗口内临时升格为 shared（结构从 j shared + k routed 变 j+c shared + k−c routed），使其整窗口常驻 GPU 且每步少动态加载 c 个专家——这是"临时共享专家"的语义（见该条目），不改模型权重、不改 gating 语义（仍算全部专家分数）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DeepSeek-MoE 论文首次系统提出 shared experts；HuggingFace Transformers 中 shared experts 以独立参数组存在（如 Qwen config 的 shared_expert_intermediate_size），MoE forward 先算 shared 再算 routed 最后合并。使用场景：shared 提供通用基础表示（防 routing 失误丢失能力）、routed 提供专长；对 offloading/预取系统，shared 恒激活 = 天然可预载、是"驻留集"的第一优先级（STEP 中 shared 与临时 shared 一起在计算开始前预取常驻），而 routed 的 88%（A100 上 Qwen3-30B-A3B INT8 profiling）执行时间花在专家取数上，正是预取要隐藏的对象。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## 临时共享专家（Temporary Shared Experts，投票选举机制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STEP 提出的在线机制：利用专家选择的时间连续性，把最近解码窗口内高频被选中的 routed 专家临时"升格"为 shared 专家（temporary shared experts），使其在下个窗口内常驻 GPU、每步不再走动态加载，从而把每步动态加载的专家数从 k 降到 k−c（c=当选专家数）。选举方式：把输出序列切成 token 窗口，窗口内每个 decode step 记录 top-2k 专家（不只 top-k），每次出现记一票（反映频率与选择强度）；窗口结束按票数选 top-c 专家为下个窗口的临时 shared。当选后：有效 MoE 结构从 j shared + k routed 变为 j+c shared + k−c routed；临时 shared 总是被预取常驻 GPU，但仅被 gating 选中时才执行计算（与 routed 一致）；gating 仍对全部专家算分数以保证统计一致与后续选举。关键约束：临时 shared 不增加显存——STEP 在固定 cache 预算下用当选专家替换低使用率专家。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 窗口 W 结束时的选举（per layer）
votes = Counter()                       # 全零初始化
for step in W:
    top2k = topk(gate(h_step), 2*k)     # 跟踪 top-2k（>top-k 的候选视野）
    for e in top2k: votes[e] += 1       # 每次出现记一票
elected = votes.top(c)                  # 票数最高的 c 个专家
# 下一窗口：结构 j+c shared + k−c routed；elected 提前预取常驻 GPU
```
Annotations：W=窗口长度（token 数）、k=routed 激活数、c=当选数（表 I：Mixtral c=1、Qwen c=1、DeepSeek c=2）。为什么用 top-2k 投票而非直接 top-k：top-k 只反映"最终被算的专家"，top-2k 能捕捉"接近被选"的高频候选，提前把它们驻留可提高命中率（Table II-IV：CNN/DM 命中率 85.5–98.8%）。时间连续性的依据（Fig.4/5）：长序列生成（LongBench）中一小撮专家被连续 step 反复选中，且不同任务（Summary vs Translation）的连续选中长度分布差异大——这同时引出 token 感知自适应窗口（见该条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为运行时层嵌入 HuggingFace Transformers 推理路径，窗口级选举 + 独立 CUDA stream 异步预取（见 kernel 调度层"专家预取"条目）；与 expert parallelism 正交——每个 EP group 独立维护热专家本地缓存并运行自己的选举/预取。使用效果：decode 阶段受益最大（时间连续性仅存在于 decode），DeepSeek 因时间连续性比 Qwen 更强而预取收益更大（Table IV 高命中率 78.6–95.3%）。退化保护：当选专家整窗口固定，即使实际使用率下降也不频繁换出（避免 transient routing 波动引发抖动）；当窗口长度缩到 1 时停用实际预取、仅保留投票统计。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## Spatial-aware Expert Allocation（空间感知专家分配 / 层内低贡献专家剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STEP 的离线优化：利用专家贡献的空间不均（层内不同专家对输出的贡献差异大），用校准数据集收集每层 top-k 路由权重分布，按归一化权重阈值 θ 识别并剪除持续低贡献的专家，使每层动态激活的 routed 数 k_l 下降。例（论文 IV-B）：某层 top-4 权重 0.62/0.21/0.13/0.04 → 分配 3 个专家（剪 1 个）；另一层 0.72/0.18/0.08/0.02 → 分配 2 个（剪 2 个）；剩余专家权重在计算时重新归一化以保持输出一致。θ=0.2 的示例阈值下，0.03–0.05 区间使平均每层 routed 数降 1–2。默认 θ：Mixtral 0.25、Qwen 0.13、DeepSeek 0.07（表 I）。它对应 T_load 公式（Eq.1）中减小 k_l 的杠杆，同时降低计算量与 PCIe 传输量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线校准（per layer l）
scores = collect_topk_scores(calib_set, layer=l)   # 校准数据集前向收集
w_norm = scores / sum(scores)                        # 归一化（跨层可比）
k_l = count(w_norm > θ)                              # 剪掉 ≤θ 的低贡献专家
# 推理时该层：
topk_idx = topk(gate(x), k_l)                        # 用减小的 k_l
w_i = softmax(gather(gate(x), topk_idx))             # 剩余权重重新归一化
y_routed = Σ w_i * E_{idx_i}(x)                      # 输出一致性保持
```
Annotations：θ=归一化权重阈值、k_l=层 l 的有效 routed 数、Avg. #Experts=层间平均（Table II-IV 扫 2→1.75/1.5、4→3/2.5/2、6→5/4/3）。为什么逐层而非全局固定：不同层权重集中度不同（有的层集中在少数专家、有的均匀分布），统一 top-k 对"集中层"浪费最大；消融（Fig.15/16）显示 STEP 自适应分配在低预算（平均 2-3 专家）下远优于固定专家数分配，因为它把算力集中到对精度关键的层、减少不敏感层的专家数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线阶段在部署前对校准集做一次前向收集每层 top-k 分数分布，按 θ 生成每层 k_l 配置；在线阶段各层按配置的 k_l 运行（θ 扫描可得到不同 Avg. #Experts 工作点）。效果：Mixtral 平均专家 2→1.75 时 MMLU 77.3→77.0、Arc-e 75.8→75.4（几乎无损），Qwen 4→3 时 70.6→70.2；单独启用该组件即贡献 1.46× 加速（消融 Fig.13），且在 prefill 阶段收益最大（低 CER 25% 下减少冗余计算是关键）。与压缩类方法（MoE-I2 剪枝+低秩分解）的区别：STEP 是推理期按贡献选择性激活，不改变模型权重；两者正交可叠加（MoE-I2+STEP decode 24.1 tok/s vs 单用 17.4/18.5）。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## Token-aware Adaptive Candidate Window（Token 感知自适应候选窗口）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STEP 的在线自适应机制：为每个 MoE 层维护奖励分 r_i 与候选窗口大小 d_i（d_i 同时控制"用多少步收集投票"与"当选专家保留多久"），按实际预取准确率动态调整窗口大小，解决固定窗口的调参困境——短窗口上下文不足、预测不可靠，长窗口易误判、过度预取浪费带宽。规则：窗口结束评估该窗口预取准确率——准确率 > th_s(75%) 则 r_i+1，r_i 累计达奖励阈值 τ（窗口 1/2 时 τ=4、窗口 4 时 τ=3、窗口 ≥8 时 τ=3，表 I）则窗口翻倍并重置 r_i；准确率 < th_f(40%) 则窗口减半并重置；介于 th_f 与 th_s 之间则窗口不变并重置。窗口缩到 1 时停用实际预取（收益有限、防带宽浪费），但继续统计投票，准确率回升后重新激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# per layer l，每个解码窗口结束
acc = prefetch_accuracy(window)          # 该窗口预取命中率
if acc > th_s(0.75):
    r_i += 1
    if r_i >= τ: d_i *= 2; r_i = 0       # 连续高准确 → 翻倍窗口（Fig.8a：4→8）
elif acc < th_f(0.40):
    d_i //= 2; r_i = 0                   # 低准确 → 减半（Fig.8b：8→4）
else:
    r_i = 0                              # 平均准确 → 保持（Fig.8c）
if d_i == 1: disable_prefetch()          # 仅统计投票，不实际预取
```
Annotations：r_i=奖励分、d_i=窗口大小、th_s=75%（Good Candidate Accuracy）、th_f=40%（Poor Candidate Accuracy）、τ=奖励阈值。设计动机（Fig.4/5/14）：专家选择的时间连续性在早期解码 step 弱、后期才稳定（Fig.14b：token 0–20、100–120 少预取，200–220、300–320 积极预取）；不同层时间模式不同（Fig.14b 按层独立调窗）；Fig.14a 显示自适应窗口的预取准确率与生成质量始终优于任何固定窗口（4/6/8/16）。Fig.14c 定量：预取准确率 >75% 时增加预取专家数显著降延迟，<40% 时过度预取反而增加延迟——这是 th_s/th_f 取值的依据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每层独立维护 (r_i, d_i)，在窗口边界依据命中率反馈更新；报告 Window Size 为运行时所有层 d_i 的平均（Table II-IV 中窗口 6/8 等）。使用场景：与"临时共享专家选举"同一框架内协作——窗口长度决定投票跨度与当选专家保留时长，两者共享同一套投票统计。效果：消融（Fig.13）中在 spatial allocation + prefetch 基础上再加自适应窗口把加速从 1.52× 提到 2.22×；实验对比固定窗口（Fig.14a）证明自适应在 LongBench 上命中率与 Rouge-L 双优。边界条件：窗口=1 时预取禁用（保留统计、可重新激活），当选专家整窗口固定避免频繁换出。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## 激活重计算（Activation Recompute / Activation Checkpointing，激活重算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Activation Recompute（激活重计算，又名 gradient checkpointing/激活重物化）是训练期的显存优化技术：前向传播时丢弃中间激活（activations），反向传播需要梯度时重新计算这些激活，用额外计算换取显存。STAGE（ISCA'26）将其建模为 workload 生成选项（对应 Korthikanti 等 Reducing Activation Recomputation 与 Grattafiori 等 LLaMA 系列论文的机制），可对同一模型+并行策略生成有/无重计算两种 workload。vault 证据：paper_secs 本论文 A.-Impact-of-Parallelism-Strategies.md（Observation 4，score 47.6）与 DeepSeek-AI.md（HC 超连接论文中提及需要 gradient checkpointing，75.9）；知识库_编译框架.md 另有 "Activation Rematerialization（激活重物化）与 ILP 图调度（训练内存优化）" 条目（同义概念）。
- 从算法pipeline角度拆解：在训练 pipeline 中，前向每一层保留激活供反向使用，显存随层数线性增长；激活重计算把"保留"改为"丢弃+重算"——选择 checkpoint 点（典型每 N 层或按内存预算选层），checkpoint 点处保留激活，其余层激活丢弃；反向时从最近的 checkpoint 重放前向到目标层获得激活。STAGE 的案例（LLaMA-7B, batch=1, TP=8, w/ SP）：峰值显存从 7042.5 MB 降至 6107.0 MB（降低 13.3%），执行时间增加 20.3%（图 11）。显存降低可支撑更大的 DP 度，从而可能整体更快。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- STAGE 通过命令行开关 --activation_recompute 生成对应 workload；框架层实现（PyTorch）用 torch.utils.checkpoint.checkpoint 包裹模块，设置 checkpoint 策略（selective/full）；Megatron-LM/DeepSpeed 提供 activation checkpointing 与 recompute 层选择。STAGE 的建模意义：在部署前模拟"显存-时间"权衡曲线，帮助选择是否启用重计算以及是否因显存释放而采用更高 DP 度。

涉及论文标题：
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs

## DCSR（Doubly Compressed Sparse Row，双重压缩稀疏行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DCSR（Doubly Compressed Sparse Row）是稀疏矩阵的一种两级压缩存储格式，在标准 CSR（Compressed Sparse Row）之上再压缩一层"空行"：CSR 用 row_ptr[ ] 记录每行的起始位置、col_idx[ ] 记录非零列号、val[ ] 记录非零值，但空行仍占用 row_ptr 中的一个槽位；DCSR 增加一个行索引表，只保留非空行，从而跳过整行全零的存储。对高度稀疏（大量整行无非零）的矩阵，DCSR 把"定位下一非空行"从遍历 O(M) 个槽位降到 O(1)（直接查表），显著减少元数据开销与无效访存。SegFold 论文用它作为矩阵 B 的片上存储格式：B 按行粒度处理（row-wise），DCSR 的第二级压缩在调度时以 O(1) 跳过 active window 中无数值交集的空 B 行——这对高度稀疏矩阵（window 内许多 k 行不贡献任何 A-B 交集）至关重要，否则这些空行会被显式枚举浪费周期。论文还在此基础上为每个 active 行增加一个额外 start pointer，跟踪该行尚未消费的非零元素（支持 partial B 行的交错处理）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DCSR 的构造与 SpGEMM 中 B 行读取流程：
```
# CSR 表示（M 行，nnz 非零）
row_ptr[0..M]   # row_ptr[i] 是第 i 行首个非零在 col_idx/val 中的下标
col_idx[0..nnz)
val[0..nnz)
# DCSR：去掉空行，加一行索引
nonempty_row[i]      # 第 i 个非空行的原行号（升序）
row_ptr_dcsr[0..R+1] # 只对 R 个非空行建指针
# 读取第 k 行非零（SegFold 中 B 以行粒度处理）：
r = 查找 nonempty_row 中 <= k 的位置      # O(log R)，或哈希/指针 O(1)
for j in row_ptr_dcsr[r] .. row_ptr_dcsr[r+1]:
    process B_val[j]                       # 该 B 行的第 col_idx[j] 列非零
# SegFold 的 partial-row 扩展：每个 active 行额外存 start_ptr[r]，
# 记录该行已被消费到第几个非零，支持跨周期交错处理多个 B 行
```
空行占比越高，DCSR 相对 CSR 的节省越大：跳过空行查找 O(1)，避免对 window 内大量无交集 k 的显式枚举。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：DCSR 常见于稀疏线性代数库与稀疏加速器（GPU SpGEMM 等用于跳过空行降低调度开销）；SegFold 中 B 以 DCSR + per-active-row start pointer 存储于片上，A 则因 SELECTA 需要按列扫描而采用 column-major（列主序）格式，两者都只存非零元素，内存控制器含 coalescing unit 合并细粒度请求后再发往 cache/DRAM。FuseFlow（稀疏深度学习编译框架）也采用 DCSR/COO 等格式表达稀疏张量，说明该格式在编译框架与加速器两条路径上都是处理"空行密集型稀疏"的标准手段。证据说明：论文未明确说明 DCSR 的具体位宽/实现细节，以上为基于论文描述（DCSR [1] 引用）与通用知识的推断。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow

## 量化感知训练（QAT，Quantization-Aware Training）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量化感知训练（QAT）是在训练/微调过程中模拟量化效果、让模型参数适应低比特量化的训练方法：前向传播插入伪量化（fake quant）算子（量化到低比特再反量化），保持权重为浮点同时让损失看到量化噪声；反向传播用直通估计器（STE）把梯度穿过不可微的量化算子（vault 笔记 knowledge_notes/算法知识笔记/Quantization-Aware Training (QAT).md 与 Straight-Through Estimator (STE).md 有详细定义：量化范围内梯度=1、范围外=0 并被 clamp 截断）。与 PTQ（训练后量化，仅校准不重训）相比 QAT 精度更高但成本高。论文（ISCA 2026）用 QAT [81]（Nagel et al., "Overcoming oscillations in QAT", ICML 2022）把 ResNet50/MobileNetV2 量化到 3/4-bit（及 8-bit 对照）用于 SiPh 加速器精度评估，QAT 同时学习激活与权重的动态范围（量化器在推理时使用）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 论文的 QAT 流程（Sec-III-C/III-D）：①QAT 训练（每数据集分别训练），前向含激活/权重量化器（伪量化），学最优动态范围；②推理时按所学范围做低比特量化；③在每层注入 SiPh 非理想因素（pre-hook 加调制器非线性、post-hook 加 ISI 分布与 AWGN）评估精度。伪代码：
  ```
  # QAT 训练（每批）
  for x, y in train_loader:
      x_q = Q_act(x)          # 伪量化激活到 4-bit（scale=动态范围/15，QAT 学范围）
      w_q = Q_wt(w)           # 伪量化权重
      y_hat = layer(x_q, w_q) # 前向（含 STE 反传量化器）
      loss = CE(y_hat, y); loss.backward()   # 梯度经 STE 穿过量化器
      w -= lr * grad_w        # 权重保持浮点更新
  # 推理（SiPh 精度评估）
  for layer in model:
      y = layer(Q(x), Q(w))          # 低比特 MAC
      y += N(0, σ²_ISI) + Σ N(0, σ²_opt)   # ISI + 光噪声注入（post-hook）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：PyTorch 中用 hook 或量化感知算子实现；常见框架 QAT 工具（torch.ao.quantization、Brevitas、Intel Neural Compressor 等）。论文用法：图像模型走 QAT（3/4-bit 精度损失小），LLM（Qwen2.5-7B）因算力限制改用 AWQ + 激活量化（post-training）而不用 QAT。使用注意：QAT 学到的最优动态范围直接决定量化器 scale；评估 SiPh 加速器时量化后的激活值在电平均匀分布（论文瞬态仿真假设的依据）。论文关键数据：3/4-bit 低比特模型对噪声更不鲁棒（MobileNetV2 3/4/8-bit 需 SNR>20/>12/>2；ResNet50 >12.5/>8.33/>2.5），说明 QAT 恢复精度后仍需信号完整性补偿。

涉及论文标题：
- Shining Light on Silicon Photonic DNN Accelerators

## 激活量化粒度（Per-Tensor / Per-Feature / Per-Block Activation Quantization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 激活量化粒度指对激活张量施加 affine 量化（scale+zero-point，[85] AffineQuant）时共享量化参数的维度范围：per-tensor（整个张量一个 scale/zero-point）、per-feature（激活张量每个 hidden dimension 一个）、per-block（一个块共享，论文取 14 batch × 74 hidden dim，为 Qwen2.5-7B 上困惑度最优的块大小）。粒度越细越能适配通道间动态范围差异（尤其 LLM 的 outlier 激活），但元数据开销越大。论文用它评估 Qwen2.5-7B-instruct-AWQ 部署到 SiPh 加速器时的激活量化损失。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 流程（论文 Sec-III-C/IV-E）：①基线 Qwen2.5-7B-instruct-AWQ：int4 权重（AWQ）+ fp16 激活，Wikitext-2 困惑度 6.79；②激活进一步量化到 int8~int4，按三种粒度：每 hidden dim（per-feature，如 hidden=3584 维每维一个 scale）、或每 (14,74) 块（per-block）。伪代码：
  ```
  # 激活量化（per-block 例，块 (B=14, H=74)）
  for b, h in block_indexes:
      s = (max(act[b,h]) - min(act[b,h])) / (2^bits - 1)   # 每块 scale
      z = round(-min(act[b,h]) / s)                        # zero-point
      act_q[b,h] = clamp(round(act[b,h]/s) + z, 0, 2^bits-1)
  # 困惑度评估（Table-II）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：per-tensor 最简单但 outlier 破坏动态范围；per-feature 对每 hidden dim 用独立 scale；per-block 在 batch×hidden 子块共享 scale（论文块尺寸经搜索确定，更大块困惑度变差）。结果（Table-II）：int8 时 per-block 6.82 ≈ 基线 6.79，per-tensor 17.71；int5 时 per-block 182（好于 per-tensor 83150 与 per-feature 182441）但仍远差于基线；int4 全线崩溃（per-tensor 120 万）。结论：LLM outlier 激活需要细粒度量化与高精度累加（数字加速器用 FP8/24-bit 累加），SiPh 加速器无法用 ADC 量化位补回丢失动态范围，因此低比特 LLM 部署 SiPh 需进一步算法/器件改进。

涉及论文标题：
- Shining Light on Silicon Photonic DNN Accelerators

## 奇异值分解（SVD）与低秩分解（Singular Value Decomposition / Low-Rank Approximation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 奇异值分解（SVD）把任意实矩阵 $\mathbf{W}\in\mathbb{R}^{m\times n}$ 分解为 $\mathbf{W}=\mathbf{U}\mathbf{S}\mathbf{V}^T$：$\mathbf{U}$（$m\times r$，左奇异向量、列空间正交基）、$\mathbf{S}$（对角矩阵，奇异值 $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r\ge0$）、$\mathbf{V}^T$（$r\times n$，右奇异向量、行空间正交基）。低秩近似即保留前 $k$ 大奇异值对应的分量、截断尾部，$k$ 值越接近满秩误差越小；Eckart–Young 定理保证截断 SVD 在 Frobenius 范数下是最优低秩逼近。关键性质：LLM 权重矩阵的奇异值近似指数衰减（SingularBit 论文 Fig.4 实测 Llama2-7B 逐层），即少数主导 rank 分量承载大部分信号能量、长尾贡献极小——这使"按 rank 重要性分配资源"成为可行的压缩信号。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SVD 在 LLM 压缩 pipeline 中的两种用法：(1) 低秩截断压缩（SVD-LLM [40] 用截断感知数据白化对齐奇异值与压缩损失、ASVD [39] 分解前做激活感知变换抑制 outlier、SliceGPT 正交变换后删行删列）：$W\approx\hat{U}\cdot S_{[:k]}\cdot\hat{V}^T$，直接减少参数与 FLOPs，但压缩率不如量化且 rank 截断即"0-bit 分配"会整体丢失信息；(2) 作为精度分配信号（SingularBit 用法，见下一个术语）。SingularBit 的分解式（论文 Eq.1）：$\mathbf{W}^T=\sum_i \sigma_i \mathbf{u}_i \mathbf{v}_i^T$，累计尾部占比 $C_i=\frac{\sum_{k=i}^{r-1}\sigma_k}{\sum_{k=0}^{r-1}\sigma_k}$ 量化每 rank 的信息密度。伪代码：
  ```
  U, S, VT = svd(W.T)              # 一次离线分解，得 U∈R^{ich×r}, S=diag(σ), VT∈R^{r×och}
  # 后续：按 σ 分精度区域 / 或截断保留前 k 个分量
  W_approx = U[:, :k] @ diag(S[:k]) @ VT[:k, :]     # 低秩截断（SVD-LLM/ASVD 式）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：数值上用 LAPACK/CPU-GPU 的 gesvd/gesdd（或 PyTorch torch.linalg.svd、JAX jnp.linalg.svd）求稠密矩阵 SVD；LLM 压缩实践中对每个线性层权重独立分解（O(L) 次小规模 SVD，可离线并行），calibration 数据仅用于量化阶段的 Hessian 而非分解。使用注意：SVD 在 Frobenius 范数意义下最优但不对齐实际激活分布，故 ASVD/SVD-LLM 引入激活感知/白化预处理；SingularBit 不做截断而是把奇异值作为 rank 级混合精度分配依据（保留全 rank、只降位宽），并用 GPTQ 误差反馈框架量化分解后的 U/V^T 分量。论文数据：LLM 权重奇异值近似指数衰减，SingularBit-W 在 2-bit 平均精度下 LLaMA-7B Wiki 困惑度 7.56（RTN/GPTQ/AWQ 为 1.9e3/44.01/2.6e5），证明"全保留+按重要性降位"优于"截断"或"均匀量化"。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

## Rank-aware 混合精度量化（Rank-Aware Mixed-Precision Quantization，奇异值重要性驱动的精度分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Rank-aware 混合精度量化是 SingularBit 的核心算法：对 SVD 分解后的权重矩阵，按奇异值大小把 rank 分量划分成 K 个精度区域，奇异值大的区域（主导信息）分配高比特（4-bit）、奇异值小的长尾区域分配低比特（1–2-bit），实现"全 rank 保留 + 按重要性差异化位宽"的压缩，而不是均匀量化或截断。理论基础是 LLM 权重奇异值近似指数衰减（Fig.4），因此累计尾部占比 $C_i$ 可解析刻画每 rank 信息密度。SingularBit 用单一参数 p（rank ratio）闭式确定 K=4 区域的边界：$C_{r_k}=(1-p)^{K-k}$（Eq.3），并由目标平均精度 $B_{avg}=\frac{1}{R}\sum_k b_k(r_k-r_{k-1})$（Eq.4）反解出 p——全程解析、无需对每模型/每层做启发式超参搜索，泛化到不同架构。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（论文 Algorithm 1 精简）：①对 $W^T$ 做 SVD 得 U、S、V^T；②按累计尾部占比定边界 $\{r_1,r_2,r_3\}$（K=4，位宽 $\{4,3,2,1\}$）；③用 calibration 激活算 Hessian $H_U=x^Tx$，逆序逐块量化 U（每块按所在区域位宽做分层二进制量化），量化误差 $E_u=U-\hat{U}$ 乘 $H_U^{-1}$ 反馈回未量化参数（GPTQ 错误反馈）；④由已量化 $\hat{U}$ 推导有效 Hessian $H_{V^T}=S\hat{U}^T H_U \hat{U}S$（来自 $z=x\hat{U}S$ 的 $H=z^Tz$），正序逐块量化 V^T 并同样做误差反馈；⑤输出 $\hat{U},\hat{V}^T$。张量计算例子（LLaMA-7B 一个 FFN 线性层，W∈R^{11008×4096}）：SVD 后 r≈4096 个 rank 被分为 4 区（如 r1≈前 400 个 rank 用 4-bit、之后 3-bit、2-bit、1-bit），目标 B_avg=2 时 p 由 Eq.3/4 解出；推理时 U 计算走空间混合精度（不同 rank 分给不同 core 并行）、V^T 走时间混合精度（同 core 沿归约维顺序累加）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：离线一次性执行（权重静态，无推理开销）；精度区域表示采用 ARB-LLM 式分层二进制量化（每区域 $W=\sum_{i=1}^{b_k}\alpha_{r,i}\alpha_{c,i}B_i$），位宽 b_k 直接换算成硬件位串行延迟/能耗。设计要点：最大精度限 4-bit（更高精度需把更多 rank 压到低比特区以维持 B_avg，收益被抵消）；不分配 0-bit（不剪枝）——把 (n+1) 位降到 n 位只损失表征能力，而置 0 是完全丢失信息，这解释了 rank 截断方法（SVD-LLM/ASVD）为何在同等压缩率下掉点更多。论文结果：2-bit 下优于 OmniQuant、MagR+OPTQ 等专门低比特方法（LLaMA-7B Wiki 7.56 vs 9.72/9.89），并扩展到 KV 压缩（SingularBit-KV 的 rank 维策略直接复用本边界）。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

## GPTQ 误差反馈量化（Hessian 引导的逐块量化 + 误差补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GPTQ（Frantar et al., ICLR 2023）是一种基于近似二阶信息（Hessian）的 LLM 训练后量化方法：用少量 calibration 激活估计权重对输出影响的 Hessian $H=2X X^T$（X 为校准激活），逐列/逐块贪心量化权重，并把当前块的量化误差通过 $H^{-1}$ 补偿（"错误反馈"，类似最优脑手术 OBS 的层内误差传播）到尚未量化的权重上，使整体量化误差最小。SingularBit 不直接量化原权重，而是把 GPTQ 错误反馈框架应用到 SVD 分解后的 U 与 V^T 分量上：先按 rank 边界逐块量化 U（Hessian $H_U=x^Tx$），再推导计入已量化 $\hat{U}$ 的有效 Hessian $H_{V^T}=S\hat{U}^T H_U \hat{U}S$ 量化 V^T，让后量化的 V^T 能补偿前序 U 的累积误差。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 流程（论文 Algorithm 1）：量化 U 时逆序（idx: ich→0 步长 blocksize）逐块处理，每块 QUANTIZE 输出行/列缩放因子与二进制基 $\{\alpha_{r,i},\alpha_{c,i},B_i\}$，重建 $\hat{U}=\sum\alpha_{r,i}\alpha_{c,i}B_i$，误差 $E_u=U-\hat{U}$ 经 $H_U^{-1}$ 作用后加到剩余未量化参数（$U\leftarrow U-E_uH_U^{-1}$）；V^T 正序（idx: 0→r）同法，用有效 Hessian $H_{V^T}=S\hat{U}^TH_U\hat{U}S$。伪代码：
  ```
  H_U = x^T @ x                              # 校准激活的二阶信息
  for idx in range(ich, 0, -blocksize):      # 逆序量化 U（先量化"更重要"的列？实际按块序）
      alpha_r, alpha_c, B = QUANTIZE(U, boundaries, idx)   # rank 边界决定该块位宽
      U_hat = sum_i alpha_r[i] * alpha_c[i] * B[i]
      E_u = U[:, idx:idx+b] - U_hat
      U[:, :idx] -= E_u @ inv(H_U)           # 误差反馈到已处理(未量化)部分
  H_VT = S @ U_hat.T @ H_U @ U_hat @ S       # 有效 Hessian（计入已量化 U）
  for idx in range(0, r, blocksize):         # 正序量化 V^T
      ...  # 同 U：重建 → 误差 → H_VT^{-1} 反馈
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：标准 GPTQ 有开源实现（https://github.com/IST-DASLab/gptq，AutoGPTQ 等），按列/块分组（group size 128）量化 + $H^{-1}$ 误差补偿；SingularBit 将其改造为"对 SVD 分量"执行并以 rank 边界替代均匀位宽。与标准 GPTQ 的差异：①量化对象是 U/V^T 而非原 W；②位宽逐 rank 区域变化（4/3/2/1-bit）而非整层均匀；③V^T 的 Hessian 需经 $S\hat{U}^T$ 变换以把 U 的量化误差纳入考量。论文数据：标准 GPTQ 在 2-bit 均匀量化下 LLaMA-7B Wiki 困惑度 44.01（严重退化），而 SingularBit-W（GPTQ 框架 + rank-aware 混合精度）达 7.56，说明错误反馈机制+重要性感知位宽缺一不可。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

## 分层二进制量化（Hierarchical Binary Quantization，ARB-LLM 式加法二进制基表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 分层二进制量化（ARB-LLM [47]，Alternating Refined Binarization）把每个精度区域（bitwidth $b_k$）的权重张量表示为一组二进制基矩阵的加权和：$\mathbf{W}=\sum_{i=1}^{b_k}\alpha_{r,i}\,\alpha_{c,i}\,\mathbf{B}_i$，其中 $\mathbf{B}_i\in\{-1,+1\}^{och\times ich}$ 是第 i 个二进制基（逐元素 ±1），$\alpha_{r,i}$ 为行缩放因子、$\alpha_{c,i}$ 为列缩放因子。b_k 个二进制基逐层叠加逼近原权重（类似 OneBit [38] 的 1-bit 符号矩阵+轻量向量，但可扩展到多 bit 层级）。该表示的核心价值：二进制基的乘法只有符号翻转（±1×x = ±x），且与位串行硬件执行天然兼容——分配的 bitwidth 直接换算成硬件延迟与能耗。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（论文 Eq.6 与 Algorithm 1 的 QUANTIZE 步骤）：对每个量化块，按区域位宽 b_k 迭代：①交替精化求当前残差的符号基 $B_i=\mathrm{sign}(W-\sum_{j<i}\alpha_{r,j}\alpha_{c,j}B_j)$（ARB 的交替优化）；②按行/列求最优缩放因子 $\alpha_{r,i},\alpha_{c,i}$（最小二乘意义）；③累加进重建 $\hat{W}$。推理时张量计算例子（一个 4-bit 区域的线性层）：$Y=X\hat{W}=X\sum_{i=1}^4\alpha_{r,i}\alpha_{c,i}B_i=\sum_{i=1}^4\alpha_{r,i}((XB_i)\odot\alpha_{c,i})$——先做 4 次二进制乘加（$XB_i$ 只需取反/加），再按行/列缩放叠加；每减少 1 bit 就少一次基乘加，位宽与计算量线性挂钩。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：ARB-LLM 已有开源实现（https://github.com/ilacher/ARB-LLM，论文未引用具体链接）；训练后离线对每层逐区域精化（可并行）。在 SingularBit 中的使用：作为 rank 边界内所有区域（4/3/2/1-bit）的统一表示，输出 $\hat{U},\hat{V}^T$ 均以 $\sum_i\alpha_{r,i}\alpha_{c,i}B_i$ 存储；硬件侧 activation loader 只为 4 输入通道预计算 8 个 LUT 条目（0 通道编码符号、1–3 通道编码幅度组合），乘法被多路选择器+取反替代（见硬件架构层"LUT 位串行混合精度 Tensor Core"条目）。注意区分：本术语是"权重表示为多二进制基叠加"的表示法，与 LUT 查表量化（码本式非解析量化）不同，后者是聚类码本。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

## KV Cache 量化压缩（Token 维 × Rank 维二维混合精度，SingularBit-KV）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- KV Cache 量化压缩指在自回归解码过程中把不断增长的 key/value 缓存以低比特表示存储，减少 KV 的 DRAM 容量与每步注意力访存带宽（KV 流量随序列长度 O(N²) 增长，长上下文/推理场景成为主瓶颈）。先前方法分两类：(1) 纯量化——KIVI [11] key 逐 channel、value 逐 token 的非对称 2-bit（key 有通道级 outlier、value 有 token 级模式）、KVQuant [16] pre-RoPE 量化+非均匀量化、ZipCache [33] 按注意力显著性分配 token 级位宽、GEAR [30] 量化+低秩校正+稀疏离群补偿；(2) 纯低秩——PALU [31] 低秩投影缓存中间态、MatryoshkaKV、ReCalKV [32]。SingularBit-KV 的创新：在 token 维按注意力重要性分 5 级位宽（b~b+4），同时在同一 token 内按 K/V 投影权重的奇异值边界做 rank 维混合精度——二维同时压缩，且缓存中间表示 K'=xU_{W_K}、V'=xU_{W_V}（只乘 U 矩阵），attention 前用 V^T 重构。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 每 decode 步的三阶段 pipeline（论文 Fig.6）：①注意力图更新：当前注意力图 $A_t\in\mathbb{R}^{H\times N_t}$ 逐 head max pooling 后归一化得 $\tilde{a}_t$，追加进 recent-k 窗口 $M_t\in\mathbb{R}^{k\times N_t}$（k=128）并逐出最旧；②重要性分数：$\mathcal{I}_i=\max_{j\in[t-k+1,t]}\tilde{a}_j[i]$（取近 k 步 query 方向的最大，因在线量化不可逆、需保守保留峰值注意力）；③精度分配+压缩：token 精度策略按线性递增容量调度 $l_i\cdot 2^{b+i}=m\cdot i+c$（$\sum l_i=1$ 解出区间边界 $s_0..s_5$，重尾分布下高重要度区间获得更多量化容量），把 $\mathcal{I}_i$ 映射到 5 级位宽；rank 精度策略在该 token 最大位宽内按 SingularBit-W 边界逐级降精度。伪代码：
  ```
  # 每 decode 步 t
  a_t = normalize(max_pool(A_t, dim=head))       # 逐 head 最大池化
  M_t = push_evict(M_t, a_t, k=128)
  I_i  = max(M_t[:, i])                          # 每 token 重要性（近 k 步最大）
  b_tok = map_token_precision(I_i, l_i*2^(b+i)=m*i+c)   # 5 级 token 位宽
  Kp = x @ U_WK;  Vp = x @ U_WV                  # 只乘 U，缓存中间表示（无在线 SVD）
  qKV = quantize_rankwise(Kp, Vp, b_tok, rank_boundaries)  # FP16→INTx 逐 rank 降位
  packed = bitpack_no_padding(qKV, b_tok)        # 紧凑打包 + 按位宽路由物理地址
  # attention 时: K=Kp@V_WK^T, V=Vp@V_WV^T 重构（tensor core 上，+5%@ctx64/+2%@ctx2048 延迟）
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：算法在 GPU 上用 PyTorch 可实现（注意 K'/V' 缓存 + 每 token 位宽表 + 反量化重构 kernel；KIVI 有开源实现 https://github.com/Zefan-Cai/KIVI），SingularBit 则用专用硬件（SingularBit Compression Engine：precision allocator + FP16-to-INTx 量化器 + 位打包器 + 路由）在线执行，避免 GPU 的格式转换/数据移动开销。使用注意：在线量化的不可逆性决定了重要性打分必须保守（用 max 而非 mean）；base precision b 是压缩-精度权衡旋钮（论文选精度下降 <1% 的工作点）。论文结果：KV2 下 Llama-3-8B-Instruct CoQA 61.5%（FP16 63.5%，仅 -2.0%）/TruthfulQA 59.5%，GSM8K 0.81~0.85（KVQuant/PALU/ZipCache 等 0–30% 崩坏），压缩率 84–86%；LongBench 上 SingularBit-KV 44.4% vs ReCalKV 29.6%。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

## 注意力重要性分数与 recent-k 窗口（Attention-based Token Importance & Recent-k Window，H2O 式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 注意力重要性分数量化每个已缓存 token 对后续生成的贡献，依据是注意力模式具有 locality 与稀疏性（H2O [10] 提出 heavy-hitter token 概念：少数 token 获得主导注意力）。SingularBit-KV 用它决定每个 KV token 的量化位宽：先对当前注意力图逐 attention head 做 max pooling（head-wise max pooling，因为不同 head 关注互补信息，保留任一 head 需要的 token 高精度）并归一化，再维护一个 recent-k 窗口（k=128）记录近 k 步的注意力分布以平滑瞬态模式，最终每 token 重要性取窗口内近 k 步 query 方向的最大值 $\mathcal{I}_i=\max_j\tilde{a}_j[i]$——用 max 而非 mean 是因为在线量化不可逆（一旦低比特存储，恢复高精度需重算 KV，开销大），必须保守保留 token 的峰值需求。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（论文 Sec-IV-B）：①更新：$\tilde{a}_t=\mathrm{normalize}(\max_{h}\mathrm{softmax}(Q_tK^T)[h,:])$，$\tilde{a}_t\in\mathbb{R}^{N_t}$，窗口 $M_t\leftarrow[M_t;\tilde{a}_t]$ 淘汰最旧行；②打分：$\mathcal{I}_i=\max_{j\in[t-k+1,t]}M_t[j,i]$；③分配：把 $[0,1]$ 归一化重要性按线性容量递增的边界 $\{s_0..s_5\}$（$l_i2^{b+i}=m\cdot i+c$）映射到 5 级位宽 b~b+4。伪代码：
  ```
  # 头维最大池化（保留任何 head 需要的 token）
  def agg_attn(A_t):                 # A_t: (H, N_t)
      a = A_t.max(dim=0)             # 逐 head 取最大 → (N_t,)
      return (a - a.min()) / (a.max() - a.min())   # 归一化到 [0,1]
  M_t = deque(maxlen=k)              # recent-k = 128 窗口
  M_t.append(agg_attn(A_t))
  I_i = max(row[i] for row in M_t)   # 近 k 步 query 最大值（保守）
  b_i = threshold_map(I_i, s_0..s_5) # 容量线性递增阈值 → 5 级位宽
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：算法层是轻量统计（逐层维护 k×N 窗口，复杂度 O(k·N·H) 远小于 attention 本体）；软件上可用 PyTorch 实现或结合 H2O（https://github.com/weizhehuang/H2O）的 heavy-hitter 打分；硬件上 SingularBit 用压缩引擎的 max-tracking 逻辑（maximum-tracking logic）在线实现 head-wise max pooling、归一化、recent-k 窗口与阈值映射，无需 CPU 干预。与 H2O/ZipCache 的差异：H2O 二值化逐出 token（剪枝），ZipCache 按显著性分位宽但只 token 一维；SingularBit-KV 把重要性细化为 5 级（捕捉中间重要度）并叠加 rank 维。论文数据：recent-k=128 窗口 + max 聚合使 KV2 下 CoQA 仅掉 2.0%，而二值化/单维方法在 2-bit 普遍掉 20%+。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

## MLLM 推理四阶段 pipeline（Preprocessor / Vision Encoder / Merger / LLM）与 encoder-decoder 资源不对称（RESONATOR 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLLM（Multimodal Large Language Model，如 Qwen2-VL、Kimi-VL、GPT-4o）采用模块化设计：模态专用编码器（图像 ViT / 音频 encoder）产出 token 序列，再由文本中心 decoder 生成回复。RESONATOR 把 MLLM 推理 pipeline 拆为四阶段（Figure 2）：①Preprocessor（CPU 密集，resize/归一化/把图像按动态分辨率切成均匀 tile 或 patch，tile 数随分辨率变化）；②Vision Encoder（ViT，把图像 tile 转成高维 raw visual tokens n_raw，序列长随分辨率增长）；③Merger/Projector（轻量模块，把长 n_raw 压成短 n_final，n_final≪n_raw，降低 LLM prefill 的二次复杂度，但不减轻 encoder 处理全部 n_raw 的负担）；④LLM（处理文本+压缩后视觉 token 的拼接序列，先 prefill 算 KV cache，再自回归 decode）。关键系统含义是 encoder 与 decoder 的资源不对称：encoder 显存占用小（Qwen2-VL 的 ViT-675M 权重 FP16 仅 1.3GB、Kimi-VL 的 MoonViT-400M 仅 0.8GB）但计算随输入分辨率近似二次增长；decoder 则 memory 与 compute 双密集。此不对称是 RESONATOR 调度设计的动机来源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RESONATOR 的四阶段计算流程（以 Qwen2-VL-7B 处理一张 1024×1024 图为例）：
```
# Stage 1 Preprocessor (CPU): 动态分辨率分 tile
tiles = dynamic_tile(resize(image, 1024x1024))        # 可变 tile 数
# Stage 2 Vision Encoder (GPU, ViT): 每 tile 按 patch 化
L_seq = ceil(H(r)*W(r) / P^2)                          # P=patch size(14)，唯一复杂度参数
raw_tokens = ViT_encoder(tiles)                        # n_raw 个 visual tokens，自注意力 ~O(L_seq^2)
# Stage 3 Merger/Projector (轻量): 压缩视觉 token
final_tokens = merger(raw_tokens)                      # n_final << n_raw
# Stage 4 LLM: prefill + decode
kv = prefill(concat(text_tokens, final_tokens))        # 算 KV cache
output = decode_loop(kv)                               # 自回归逐 token 生成
```
Annotations：L_seq 由图像高宽 H(r)/W(r) 与 ViT patch size P 决定（RESONATOR 的 encoder 性能模型只以 L_seq 为复杂度参数，因此可泛化到任意分辨率/宽高比）；encoder 阶段对高分辨率图是 prefill 关键路径的 dominant 瓶颈（Figure 3 端到端 breakdown 显示 encoder 延迟占比随分辨率上升）；Merger 虽缩短 LLM 输入，但 encoder 仍须处理全部 n_raw token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
四阶段是主流开源 MLLM 的标准推理结构，serving 框架（SGLang/vLLM）原生把 encoder 作为预处理步骤挂在 LLM 之前；实现细节由模型仓库给出（Qwen2-VL 的 dynamic resolution + ViT-675M，Kimi-VL 的 MoonViT）。RESONATOR 的关键改法：不再把 encoder 当作固定预处理步骤，而是视为一等公民动态负载——在 Serving 层（SGLang-0.4.7）解耦 encoder 与 decoder，再用 Intra-GPU 共享引擎 + Inter-GPU 动态并行把两者重新耦合。评估模型：Qwen2-VL-7B/72B（ViT-675M）、Kimi-VL-16B（MoonViT-400M）；数据集 MMMUPro、TextVQA。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs

## TDS（Topology Dependency Subgraph，拓扑依赖子图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TDS 是 TAGT（ISCA 2026）为图 Transformer 提出的拓扑感知稀疏子图：把全局 O(N²) 注意力近似为在 TDS 上的注意力，将每个目标顶点 attend 的边数从 O(N) 降到平均 O(m·log_m N)（m=2 时为 O(log N)），总边数 O(m·N·log_m N)=O(N log N)。TDS 由三类边构成：(1) original edges——保留输入图原始局部邻域；(2) fusion edges——自底向上的分层聚合边：沿原生 1D 输入顺序每次递归合并 m 个内存连续顶点为 fusion 顶点（约 log_m N 层直到单根），fusion 顶点持有全部子顶点的聚合特征，作为"高阶代理顶点"保留远程/全局上下文；(3) association edges——指向目标顶点的递归边：每层从目标顶点左右各取 m 个关联顶点（起始下标 p_{l+1}=parent(p_l+m)，若 p_l+m-1 为奇数则再纳入下一顶点且 p_{l+1}=parent(p_l+m+1) 保证集合互斥），提供多粒度远程上下文。TDS 总顶点数约 N + N/(m-1)（m≥2），构造保证任意两个原始顶点通过 fusion+association 边最多 2-hop 可达，因此每个目标顶点的 1-hop 注意力邻域同时含局部邻居、多粒度上下文与全局根——一次稀疏注意力即等效多跳 message passing 的全局效果。
- 论文理论分析（式2-4）：设 A 与 Â 为全/稀疏注意力矩阵，‖Δh_i‖₂ ≤ L‖V‖₂·Σ_{j∉T_i(m)} α_ij + ε_fus(m)，其中 ε_fus(m) 为融合粗粒度误差（随 m 单调不减）；在注意力量重尾衰减 α_{i,(k)} ≤ c·k^{-β}（β>1）假设下尾质量 ≤ O((m·log_m N)^{1-β})。m 控制保真度-效率权衡：m=N 时退化为精确 O(N²) 全局注意力，m=2 时准确率最优且复杂度 O(N log N)。
- 与既有稀疏 GT 方法的区别（论文 Related Work）：AnchorGT/ANS-GT 等用采样/锚点启发式近似长程依赖，拓扑敏感、需按数据集重调；TDS 是确定性、无参数、硬件友好的稀疏化——通过拓扑感知合并而非概率选择保留全局上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TDS 构造与注意力的算法pipeline（m=2）：
  ```
  # 输入：顶点特征 x[0..N-1]（1D 内存顺序），合并基数 m，原始边集 E_orig
  # 阶段1：自底向上分层聚合——生成 fusion 顶点特征
  cur = list(range(N))
  while len(cur) > 1:
      nxt = []
      for i in range(0, len(cur), m):
          fus = 新建 fusion 顶点
          feature(fus) = aggregate(feature(c) for c in cur[i:i+m])   # 如 mean/sum
          for c in cur[i:i+m]: 添加 fusion 边 c -> fus               # 底向上有向边
          nxt.append(fus)
      cur = nxt                                    # 约 log_m N 层到根
  # 阶段2：目标顶点 association 边（左右各取 m 个，递归到上层）
  for 目标顶点 v_k (1D 下标 k):
      p = k + 1                                    # 右侧起始（左侧用 k-1 递减对称）
      for l in range(log_m N):
          在层 l 取下标 p..p+m-1 的 m 个顶点，添加 association 边 v_k -> 它们
          if (p + m - 1) 为奇数:  p = parent(p + m + 1)              # 集合互斥
          else:                   p = parent(p + m)
  # 阶段3：目标顶点注意力（式1）——只在其 TDS 1-hop 邻域上做
  H^v = concat({h_u^l | u ∈ N_TDS(v)})              # K = O(m·log_m N) 个顶点
  h̄_v^{l+1} = softmax(h_v^l·W_Q·(H^v·W_K)^T / √d_K) · (H^v·W_V)
  h_v^{l+1} = FFN(h̄_v^{l+1}) + h̄_v^{l+1}            # FFN + 残差
  ```
- 张量计算示例（N=4, m=2）：基层 v_0..v_3 → fusion u_0=agg(v_0,v_1)、u_1=agg(v_2,v_3) → 根 w_0=agg(u_0,u_1)。目标 v_1 的 1-hop 邻域 = 局部邻居 v_0（original）+ 祖先 u_0、w_0（fusion）+ 远程 u_1（association，覆盖全图上下文），仅对 ~4 个顶点算注意力分数而非全图 4 个顶点（更一般地 O(log N) vs O(N)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件侧：TAGT-S（修改 DGL v2.4.0 的软件实现，跑 A100）验证算法收益——比 TorchGT 快 1.8×–2.5×（TorchGT 依赖 Hamiltonian path、现实图不满足时回退 O(N²)），带宽利用 >60%，但软件 runtime overhead 占 69.8%–86.1% 执行时间，需硬件消除（TAGT 加速器，见 硬件架构 层 TDL/TCU 条目）。
- 硬件侧：TAGT 用 TDS-CSR Table 存 TDS 稀疏图结构，TDL/Topology Data Loader 取数、TCU（FUU+MOU）实时构造 TDS（去重共享 fusion 祖先），FAU 在 TDS 邻域上做流式注意力、SCU 做块级异步 softmax。
- 实验效果：准确率相对 DGL-CPU 全注意力参考下降 <1pp（GT 0.11–0.91pp、Graphormer 0.03–0.55pp、UGformer 0.22–0.84pp、EGformer 0.08–0.88pp），且高于 TorchGT；m=2 为准确率最优；对顶点排序鲁棒（random→METIS 排序准确率 65.08%–65.48% 稳定，local-only 断全局边则掉到 56.12%）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

## Graph Transformer（图 Transformer，GT）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Graph Transformer 是把 Transformer 架构（Vaswani et al., 2017）应用到图结构数据的模型范式：把顶点视为输入 token，用全局（all-to-all）自注意力捕获长程依赖与复杂结构交互，突破传统 GNN message-passing 的局部感受野限制（后者受 over-smoothing、over-squashing、表达力有限与扩展性差困扰）。与标准 Transformer 的区别：(1) 输入是无序顶点集（非固定顺序 token 序列），顶点处理顺序可灵活重排；(2) 顶点特征维度高（D_feat=100–1000）；(3) 常把图结构编码（structural encoding）融入输入特征与注意力矩阵。
- 论文给出的通用更新规则（式1）：对顶点 v，H^v = concat({h_u^l | u ∈ N(v)})（N(v) 为 v 的注意力邻居集，全连接注意力下即全图顶点）；h̄_v^{l+1} = softmax(h_v^l W_Q (H^v W_K)^T / √d_K) · H^v W_V；h_v^{l+1} = FFN(h̄_v^{l+1}) + h̄_v^{l+1}。以 Graphormer 为例：初始嵌入 h_v^{(0)} = x_v + z^-_{deg^-(v)} + z^+_{deg^+(v)}（入/出度可学习嵌入），注意力系数加最短路径距离偏置 bias_{φ(v,u)}。全局注意力使 GT 表达力优于 GNN，但代价是 O(N²) 计算与中间数据。
- 论文评估的 GT 模型（Table III）：Graph Transformer [Dwivedi & Bresson 2020]（4 层、hidden 128、12 head）、Graphormer [Ying et al. 2021]（4 层、hidden 768、8 head）、UGformer [Nguyen et al. 2022]（4 层、hidden 384、4 head）、Edge Transformer / EGformer [Bergen et al. 2021]（8 层、hidden 200、4 head）。profiling（Fig.2）显示 attention 占 67.08%、FFN 占 24.53% 执行时间（合计 91.61%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一个图 Transformer 层的推理 pipeline：① 输入顶点特征 x_v + 结构编码（如度嵌入/最短路径偏置）→ ② Q/K/V 投影（h_v W_Q、H^v W_K、H^v W_V）→ ③ 全对注意力分数 QK^T/√d_K（+ 结构偏置）→ ④ softmax 归一化 → ⑤ value 加权聚合（H^v W_V）→ ⑥ FFN + 残差 → 下一层。全连接注意力下第③步是 N×N 稠密矩阵，是 O(N²) 复杂度与 O(N²) 中间注意力矩阵的来源。
- 与 GNN 的关键差异（Fig.1）：(1) 局部 message-passing vs 全局注意力；(2) 隐式拓扑传播 vs 显式结构编码。因此现有 GNN 优化/加速器（SpMM、CSR 遍历）不能直接支持 GT。
- KV caching 不适用于 GT（论文 Q2 分析）：GT 处理无序顶点集的动态全对交互，无固定顺序的"过去 token"依赖，无法套用 LLM 的 KV Cache；但 TDS 的确定性稀疏依赖结构可做缓存/复用（TDS-CSR）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：通用框架如 DGL（保留 O(N²) 全局注意力）、TorchGT（dual-interleaved 稀疏注意力，依赖 Hamiltonian path）、PyG；TAGT-S 用 TDS 稀疏化修改 DGL。评估数据集（Table II）：Yelp（716,847 顶点/13.9M 边/300 维/100 类）、Reddit（232,965/114.6M/602 维/41 类）、Ogbn-Arxiv（169,343/1.17M/128 维/40 类）、Ogbn-Products（2,449,029/61.9M/100 维/47 类）、Ogbn-Papers100M（111M/1.6B/128 维/172 类），任务均为节点分类。
- 硬件实现：TAGT 加速器（Alveo U280 FPGA）以 TDS 为原生执行表示，TDL/TCU 实时构造 TDS、FAU 流式注意力、SCU 块级异步 softmax（见 硬件架构 层条目）。
- 参考实现：Graphormer 开源（https://github.com/microsoft/Graphormer）；本论文未开源。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

## 全局注意力（Global Attention / All-to-all Attention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 全局注意力指图 Transformer 采用的全对（all-to-all）注意力范式：每个顶点与图中所有其他顶点（含自环）计算注意力分数，邻居集 N(v) = 全图顶点。相比局部 message-passing，它能捕获长程依赖，但产生 O(N²) 计算与 O(N²) 中间注意力矩阵。论文量化（Fig.2/3）：attention+FFN 占 91.61% 执行时间；off-chip 访问占 60.5% 执行时间、SM 利用率 <25%；TorchGT 上 60.3% 取回数据不必要、cache line 利用率仅 18.27%；N=256K 时中间矩阵无法片上缓存、强制 off-chip spilling。大特征矩阵 O(N·D_feat + Nd) 与中间数据双重挤压内存带宽。
- 现有缓解的局限：FlashAttention 类 IO-aware 注意力 kernel 通过 tiling+online softmax 降低稠密注意力的显存流量，但**不消除 O(N²) 顶点对交互次数**；且 GT 的图结构编码使注意力矩阵不规则（非文本 token 的规则模式），块矩阵优化难以直接套用。因此全局注意力仍是 GT 扩展性的根本瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 全对注意力 pipeline（式1，全连接图 G 上）：对每个目标顶点 v，H^v = concat({h_u | u ∈ 全图})（N 个顶点），计算 h_v W_Q · (H^v W_K)^T 得 1×N 分数向量，softmax 后加权 H^v W_V。总体即 N×N 的 QK^T + softmax + PV。复杂度和中间数据均为 O(N²)。
- 对比示例（N=16K 序列）：DGL-CPU 保留 O(N²) 全对注意力作精度参考（Table VI 准确率 65.98%/98.02% 等）；TorchGT 的 dual-interleaved 注意力在缺少 Hamiltonian path 时回退 O(N²)；TAGT/TDS 把每个顶点 attend 的顶点数从 N=16K 降到 O(m·log_m N)=O(log 16K)≈28（m=2），精度下降 <1pp。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：DGL/TorchGT 直接实现；FlashAttention 类 kernel 用 tiling+online softmax 减 IO 但不减交互数。论文通过把全对注意力替换为 TDS 稀疏注意力（original/fusion/association 三类边）消除大部分不必要的注意力计算与 off-chip 移动，同时以"任意两点 ≤2-hop 可达"的拓扑保证保留全局建模能力（多粒度上下文+根顶点直接进入 1-hop 邻域）。
- 硬件：TAGT 的 FAU 把注意力分数流式送 SCU 做块级异步 softmax，全程不物化 N×N 注意力矩阵；GNN 加速器（FlowGNN/MEGA/BingoGCN）因专为稀疏 message-passing 设计，被改造执行 O(N²) 全对注意力时性能远逊于 TAGT（TAGT 平均快 8.2×/6.9×/4.7×）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

## 结构编码（Structural/Positional Encoding，SE/PE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 结构编码是图 Transformer 为捕获顶点拓扑属性而加入的显式编码，可融入输入特征（如度嵌入）或注意力矩阵（如最短路径距离偏置）。以 Graphormer 为例：初始嵌入 h_v^{(0)} = x_v + z^-_{deg^-(v)} + z^+_{deg^+(v)}（入度/出度的可学习嵌入向量），注意力系数加 bias_{φ(v,u)}（φ(v,u) 为顶点间最短路径距离的可学习共享标量）。这类编码使 GT 的注意力计算比文本 Transformer 更不规则（引入拓扑相关偏置），削弱块矩阵优化的有效性。
- TAGT 的关键设计：TAGT 对 SE/PE 语义无关（encoding-agnostic）——只要 SE/PE 在进入 TCU 前被物化为 per-vertex 稠密向量即可。FUU（Feature Update Unit）把原始特征与编码向量做同步取数+轻量拼接+线性投影生成统一基层嵌入；支持不同 SE/PE 方案只需输入维度适配与配置更新，无需微架构改动；甚至兼容需要在线图算子的编码方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SE/PE 在 TAGT 中的 pipeline 位置：TDL 的 Fetch_Coding 6 级流水线从 HBM/片上 buffer 取结构编码向量 → FUU 与 Fetch_Features 取到的原始特征同步拼接 → 线性投影 → 基层（leaf）嵌入写入 TDS-CSR Table 并流式送 MOU 参与 fusion 顶点聚合。整个过程中编码只作为"辅助向量载荷"，不触发编码专用图算子。
- 计算示例（Graphormer）：h_v^{(0)} = x_v + z_{deg^-(v)}^{-} + z_{deg^+(v)}^{+}；注意力分数 s_{v,u} = h_v W_Q (h_u W_K)^T / √d_K + bias_{φ(v,u)}。TAGT 只需把这些 per-vertex 向量/偏置作为输入即可。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现：度嵌入（in/out-degree learnable embeddings）、最短路径距离（SPD）偏置、拉普拉斯特征等；本论文未给出新编码，直接复用 Graphormer 类方案并验证 TAGT 的编码无关性。
- 使用：作为 TAGT 前端输入（每个顶点一个稠密编码向量），无需改硬件；支持向量化 SE/PE 的不同维度仅需配置更新。评估中 4 个 GT 模型（GT/Graphormer/UGformer/EGformer）均以此方式接入。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

## TorchGT（拓扑诱导稀疏注意力图 Transformer 训练框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TorchGT 是 SC 2024 提出的面向大规模图 Transformer 训练的整体系统（Zhang et al., "TORCHGT: A Holistic System for Large-scale Graph Transformer Training"），核心用拓扑诱导的稀疏注意力（topology-induced sparse attention）与 cluster-aware 图并行降低 O(N²) 训练成本。TAGT 论文把 TorchGT 作为 SOTA GPU 软件 baseline（跑在 NVIDIA Tesla A100，6,912 cores、80GB HBM），并指出其三个局限：(1) 优化依赖严格拓扑前提——Hamiltonian path（NP-complete 验证、现实图常不满足）；(2) 前提失败被迫回退 O(N²) 全局注意力；(3) 选择性注意力导致明显准确率损失。TAGT-S（TDS 稀疏化的 DGL 软件实现）在 A100 上比 TorchGT 快 1.8×–2.5×，且准确率高于 TorchGT（Table VI：TorchGT 在 Reddit 上 GT 准确率 93.98% vs TAGT 97.11% 等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TorchGT 的稀疏注意力依赖图结构满足 Hamiltonian path（一条经过所有顶点一次的路径）以定义规则的稀疏注意力模式；对不满足的图回退全对注意力。执行管线与通用 GT 相同（Q/K/V 投影 → 稀疏/全对 QK^T → softmax → PV → FFN），区别只在注意力模式与并行策略（cluster-aware graph partitioning）。
- TAGT 的对照价值：TAGT-S 用 TDS（确定性、无 Hamiltonian 前提的稀疏结构）取代 TorchGT 的稀疏模式，在真实图数据集上持续优于 TorchGT 且精度接近全注意力参考。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：TorchGT 开源（https://github.com/hengruizhang98/torchgt）。TAGT 论文以其在 A100 上的执行时间/带宽利用/准确率为对比基准（Fig.3 profiling：off-chip 访问 60.5%、SM 利用 <25%、60.3% 数据冗余、cache line 利用 18.27%）。
- 使用：作为 GT 训练/推理的 GPU baseline；与 DGL-CPU（全注意力）、TAGT-S（TDS 稀疏化）、TAGT（FPGA 硬件）对比评估。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

## LumiBench（硬件光线追踪基准套件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LumiBench 是 UBC Aamodt 组（Liu et al., IISWC 2023）发布的面向硬件光线追踪的基准套件，配套 Vulkan-Sim 2.0 与 RayTracingInVulkan 应用（Peter Shirley《Ray Tracing In One Weekend》的 Vulkan + NVIDIA RTX 扩展实现）使用。提供 16 个几何复杂度递增的 3D 场景（wknd/ship/bunny/spnza/chsnt/bath/ref/crnvl/fox/party/sprng/lands/frst/park/car/robot），BVH 树大小从 0.2MB（wknd）到 1721.3MB（robot），深度 7-18；每个场景的 BVH 由开源 Intel Embree 库构建（每叶节点 1 primitive）。支持三种 ray tracing 负载：path tracing（closest-hit 主光线+次级光线，最重，每像素 4-16 条弹跳光线）、Ambient Occlusion（最近命中点后向随机方向 4 条光线）、Shadow（向光源 2 条 any-hit 光线）。默认 128x128 分辨率、1 sample/像素，可调分辨率与采样数。TTP 论文用它做主要评估集：15/16 场景在 128x128 完成（park 72h 超时改 64x64），并测 256x256/64x64/32x32 分辨率；chsnt 不支持 AO/SH。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- LumiBench 的 ray tracing pipeline 流程（以 path tracing 为例）：①raygen shader 每像素生成主光线（origin/direction）→ ②trace ray 指令交给 RT unit 做 BVH 遍历（Embree 构建的树），closest-hit 语义找最近命中 → ③命中点处生成次级光线（反射/折射，4-16 条/像素）→ ④每条次级光线再次 trace ray 遍历 BVH → ⑤递归或迭代累加颜色直至收敛。AO/SH 负载：先 trace 一条主光线找最近命中点，再分别向 4 个随机方向（AO，评估环境光遮蔽）或 2 个光源方向（SH，any-hit 提前终止，评估阴影）trace 次级光线。每场景 × 负载 × 分辨率 × 采样数构成实验矩阵；TTP 论文用 path tracing 为主、AO/SH 为辅评估硬件预取器。运行时命令：./RayTracer --scene 20 --width 32 --height 32 --samples 1 > ship_pt.log（--scene 选场景索引）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RayTracingInVulkan（github.com/ubc-aamodt-group/RayTracingInVulkan）用 Vulkan ray tracing pipeline（VK_KHR_ray_tracing）实现各 shader，BVH 用 Embree（embree.org）构建后上传为加速结构；LumiBench 场景文件随套件分发（Zenodo，配套 Vulkan-Sim v2.0）。使用：作为硬件光线追踪架构研究的标准负载集——TTP 论文在 Vulkan-sim 上运行全部 16 场景（park 因 72h 超时降分辨率），对比 DFS/BFS、TTP/Treelet/无预取；关键统计包括每射线访问节点数（表 I）、RT read miss 构成、MPKI、DRAM 带宽等。用途：量化 BVH 遍历的内存瓶颈（表 I 显示 DFS 平均每射线 49.0 节点 vs BFS 70.0），支撑预取/缓存/近存等架构优化的动机与评估。

涉及论文标题：
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing

## State Vector Simulation (SVS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- State Vector Simulation（态矢量模拟）是量子线路模拟（Quantum Circuit Simulation, QCS）最主流的形式：把 n 个 qubit 的量子态表示为一个 2^n 维复向量 |ψ⟩（内存 O(2^n)），把每个量子门表示为一个幺正矩阵 U，门的应用就是一次矩阵向量乘 |ψ'⟩ = U|ψ⟩。从 |0...0⟩ 初态出发按线路顺序乘完所有门得到最终态矢量，再按 Born rule 采样得到经典输出分布。noiseless QCS 只需做一遍完整矩阵向量乘。
- 本论文中 SVS 是 noisy 模拟的基本单元：由于噪声通道的随机性，每个 shot 可能产生不同的"固定噪声门电路"，S 个 shot 就需要 S 次独立 SVS（S-fold compute overhead）；TUSQ 的 ECM+DFTT 正是围绕"减少这 S 次 SVS 的冗余"设计。TUSQ 用 NVIDIA cuStateVec v1.12.0 作为 SVS 的 GPU kernel 后端。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SVS 计算过程伪代码（一个 noiseless 电路）：
  ```
  # 输入：n 比特线路（门序列 g_1..g_m），初态 |0>⊗n
  ψ = zero_state(n)                    # 2^n 维复向量，仅 ψ[0]=1
  for g in gates:
      ψ = apply_unitary(g.matrix(), ψ) # 门矩阵（2^k×2^k，k=1/2）作用于对应 qubit 的振幅
  counts = sample(ψ, shots)            # 按 |ψ[i]|^2 概率采样 shots 次
  ```
- 张量计算：|ψ'⟩ = U|ψ⟩，U 为 2^k×2^k 幺正矩阵，|ψ⟩ 为 2^n 维复向量；单比特门一次乘 2^n 个元素、双比特门一次乘 4·2^n 个元素（本论文操作计数按 1/4 计）。
- noisy 场景：对每个 shot 把噪声通道采样成固定 Pauli 门（如 DEP 的 I/X/Y/Z），得到 S 个不同电路实例，各自从头做 SVS 再平均输出——即本论文要消除的 S-fold 开销来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：主流量子 SDK 都内置 SVS 后端——Qiskit-Aer StatevectorSimulator（qiskit-aer，GPU 版）、NVIDIA CUDA-Q（cudaq 的 statevector 后端，底层 cuStateVec）、cuStateVec 库本身提供 cuStateVecApplyMatrix 等 kernel。使用：Qiskit 中 `StatevectorSimulator().run(circuit, shots=S)`；CUDA-Q 中 `cudaq.sample(kernel, shots_count=S)`。TUSQ 论文把 Qiskit 2.1.0 与 CUDA-Q 0.11.0 的 SVS 作为 baseline，在 NERSC Perlmutter 单 A100 上与 TUSQ 对比，报告平均 59.06×/13.38× 加速。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Density Matrix Simulation (DMS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Density Matrix Simulation（密度矩阵模拟）是把量子态表示为 2^n×2^n 正半定 Hermitian 矩阵 ρ（迹为 1，内存 O(2^{2n})）的模拟范式：无噪声操作是 ρ' = UρU†，一般噪声通道是 ρ' = Σ_i K_i ρ K_i†（Kraus 算子形式），例如去极化通道 ρ'=(1-p)ρ+(p/3)XρX+(p/3)YρY+(p/3)ZρZ。DMS 一次电路执行就能完整刻画噪声统计，不需要对多个电路实例平均。
- 本论文把 DMS 作为 noisy QCS 的"理想但不可扩展"参照：内存 O(2^{2n}) 比 SVS 的 O(2^n) 平方级更大，导致 El Capitan 级超算也只能模拟约 25 qubit，而 30 qubit SVS 在 16GB 笔记本即可运行（论文引 [41] 的估计）。因此 DMS 的内存开销使其在大规模下不可行，SVS 多实例平均成为唯一现实策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DMS 计算过程（一个含噪声的电路）：
  ```
  ρ = |0><0|⊗n                       # 2^n×2^n 密度矩阵，O(2^{2n}) 内存
  for g in gates:
      if g 无噪声: ρ = U ρ U†
      else: ρ = Σ_i K_i ρ K_i†        # Kraus 表示，如 DEP: (√(1-p)I, √(p/3)X, √(p/3)Y, √(p/3)Z)
  p_dist = diag(ρ)                    # 测量：取对角元素
  ```
- 与 SVS 的关系：DEP 通道的展开式把噪声态解释为"加权经典混合"，即 ρ 可看作多个 SVS 电路（固定 I/X/Y/Z 噪声门）输出的加权平均——这正是"noisy 模拟 = S 个 SVS 平均"的理论依据，也是 TUSQ 重要性采样（Pruning）保持集体贡献的数学基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Qiskit-Aer 的 DensityMatrixSimulator、CUDA-Q 支持密度矩阵后端（cudaq.DensityMatrixSimulator）、cuStateVec 也有密度矩阵 kernel（cuStateVec 支持 state matrix 运算）。使用：Qiskit 中 `DensityMatrixSimulator().run(circuit, shots=S)` 一次运行得到含噪统计。论文观点：DMS 精度最优但内存平方增长，仅在少 qubit 场景可用；大规模场景必须回到 SVS 多实例 + 冗余消除（TUSQ）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Error Realization (ER)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Error Realization（错误实现，TUSQ 提出的轻量中间表示 IR）是"从噪声通道采样得到的一组固定噪声门"：对每个 shot，把电路中每个噪声通道（去极化 DEP、测量噪声等）采样成一个确定的 Pauli 门（DEP 采 I/X/Y/Z，测量噪声采 I/X），整条电路就变成一个"固定噪声门电路"；一个 SVS 实例对应一个 ER。含 m 个噪声通道的电路的 ER 就是这 m 个采样的 n 元组，例如 (I₀, X₁, Y₂, ..., I_{m-1})。
- ER 的价值：它是"电路是否产生相同输出"的轻量判据——ER 相同的电路最终态矢量相同，无需实际计算即可合并；不同但等价的 ER（经 Pauli 门穿通后相同）也产生相同输出。TUSQ 用它做冗余检测（ER Tallying、ER Commutation）与重要性加权（Pruning）。低 Hamming weight（更多 I 门）的 ER 出现频率指数级更高（p=1% 时 Hamming weight >2 的概率为零），这是 Pruning 有效性的依据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- ER 生成与使用流程：
  ```
  # 预采样阶段（CPU，一次完成）
  for shot in 1..S:
      er = tuple(sample(channel) for channel in circuit.channels)  # 每个通道采 I/X/Y/Z 或 I/X
      tally[er] += 1                          # 记录唯一 ER 及频次
  # 使用阶段
  for (er_i, s_i) in tally.items():           # ER Tallying
      c_i = 把 er_i 的固定 Pauli 门并入无噪声电路
      |ψ_i⟩ = SVS(c_i)                        # 同一 ER 只算一次
      输出分布 += 从 |ψ_i⟩ 采样 s_i 次
  ```
- ER Commutation 例子（图4B）：两个 shot 的 ER 分别为 (X, II) 与 (I, XX)，不同但等价——X 门穿过 CNOT 的控制比特会在目标比特上额外产生 X 门，因此两者产生相同输出，可合并 shot 计数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TUSQ 在 CPU 预处理阶段预采样全部噪声通道、统计唯一 ER 频次（论文报告 CPU 预处理平均 3.97s、最大 18.52s），随后用 ER 驱动 ECM 三步（Tallying/Commutation/Pruning）确定待模拟电路集合。使用前提：噪声通道可采样成 Pauli 门（测量与去极化噪声天然满足；amplitude/phase damping 经 Pauli twirling 近似）。开源：论文声明开源实现位于 https://github.com/tinaoberoi/TUSQ，但截至 2026-08 仓库仅占位 README、无源码。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Pruning（noisy 量子模拟中的重要性采样，α/β 参数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pruning（剪枝/重要性采样）是 TUSQ Error Characterization Module 的第三阶段：利用"ER 频率随 Hamming weight 指数衰减"这一分布偏斜特性，把 ER 电路分成显著电路 C_S（频率 p_i ≥ α·p_max）与不显著电路 C_I（p_i < α·p_max，α 为阈值，论文取 0.01）。显著电路正常模拟；不显著电路的个体贡献小但集体占比可很大（10-qubit QAOA 1M shots 中 insignificant 合计占 42%），因此不直接丢弃，而是从 C_I 随机采 β 个代表电路（β=100），每个代表按 (p_insig/Σp_t)·p_t 加权采样，保持整体输出分布贡献。
- 这是 TUSQ 唯一引入 fidelity 损失的步骤：relative fidelity difference δ = |f_A-f_B|/(f_A+f_B) 平均 1.66%、最大 7.15%（α=0.01、β=100），对 VQE/Adder/BV 等算法正确性影响可忽略（Adder 320 例中 289 例、BV 380 例中 368 例推断输出比特串不变）。α/β 是用户可调旋钮：要更低 fidelity 偏差就降低 α、提高 β，代价是更多模拟时间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pruning 伪代码：
  ```
  p_max = max(freq of ER circuits)
  C_S = {c_i | p_i >= α·p_max}                 # 显著电路，α=0.01
  C_I = {c_i | p_i <  α·p_max}                 # 不显著电路
  p_insig = Σ_{c_i∈C_I} p_i                    # 集体占比
  K = 从 C_I 均匀随机采样 min(β, |C_I|) 个代表   # β=100
  for c_t in K:                                # 每个代表
      计算 |ψ_t⟩ = SVS(c_t)
      采样次数 = (p_insig / Σ_{c_t∈K} p_t) · p_t   # 按频率加权保持集体贡献
  S_final = |C_S| + min(β, |C_I|)              # 待模拟电路总数 ≪ 原始 S
  ```
- 与 baseline 对比：TQSim 的 fidelity 损失来自内存饱和时缓存不全（统计方法取舍），TUSQ 的 Pruning 把"丢多少"显式化为 α/β 参数；naive SVS（Qiskit/CUDA-Q）完全不剪枝、无损失但计算量随 S 线性增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TUSQ CPU 预处理阶段按上式执行，Pruning 与 ER Tallying/ER Commutation 串接；在 QAOA p=2→6 深度增加时 ER 分布趋于均匀、剪枝有效性下降（深度越大越难区分信号与噪声，p=10 时几乎失效）。使用：用户按误差容忍度设置 α、β（论文默认 α=0.01、β=100）；希望无损时可用 α=0（保留全部电路）。评估指标：speedup γ 与 relative fidelity difference δ 联合权衡。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Depth First Tree Traversal (DFTT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Depth First Tree Traversal（深度优先树遍历，DFTT）是 TUSQ 的计算复用模块：把 ECM 输出的待模拟电路集合（含各自频率）按"共享前缀门"组织成一棵树——节点是电路某时刻的中间态矢量，边是门；根到叶子的路径对应一条电路。用深度优先遍历计算所有叶子输出：沿边正向乘 U（compute）计算，反向乘 U†（uncompute，回滚到公共祖先）后再走另一分支，从而共享前缀计算。例如电路 U₁U_c 与 U₂U_c 共享前缀 U_c：算完第一个输出后回滚到公共节点，再乘 U₂ 得到第二个输出，公共部分只算一次。
- 渐近优势：设树边数 |E|、树高 h、叶子数 N_l，DFTT 每条边最多遍历两次，T_dftt = 2|E| = O(|E|)；naive 每条叶子都从根重走，T_naive = N_l·h = O(|E|log_b|E|)（b 为噪声通道分支数，DEP b=4、测量 b=2）。DFTT 把操作数从 O(|E|log_b|E|) 降到 O(|E|)，且不占用额外内存（不像 TQSim 缓存中间态），速度不依赖可用内存。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DFTT 树遍历伪代码（图5B 的 S1/S6 例子）：
  ```
  # 树：根 a，公共前缀边 a→...→d，S1 分支 d→...→f1，S6 分支 d→...→f6
  stack = [root_state]                 # 当前态矢量
  def dfs(node):
      for child in node.children:
          stack.push(apply_gate(stack.top, edge(child)))   # compute：乘 U
          if child.is_leaf: sample_and_accumulate(stack.top, freq[child])
          else: dfs(child)
          stack.push(apply_gate_inv(stack.top, edge(child)))  # uncompute：乘 U†
  dfs(root)                            # 每条边正反各走一次 = 2|E| 次矩阵向量乘
  ```
- 复杂度推导：b+b²+...+b^h = |E| ⟹ h = log_b((b-1)|E|+b)-1，N_l = b^h = (1-1/b)|E|+1，故 T_naive = O(|E|log_b|E|)。
- 额外并行：内存富余时（如只用 25% 内存），可把根态拷贝到子树并行 DFTT（n 倍并行 n 倍内存），进一步提速。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TUSQ 中 DFTT 调度 cuStateVec kernel 在 GPU 上执行矩阵向量乘，逐边正向/反向遍历；前提是所有边对应幺正门（有逆）。对非幺正通道（mid-circuit measurement、erasure），用 DFTT+Caching：把 non-invertible 边之前的态缓存进 LIFO（容量 K，受内存约束），回滚跨非幺正边时取缓存而非求逆；同一层 MCM 合并为一条边以降低缓存需求；K=3 即可恢复 60%-100% 的 DFTT 性能（surface code 电路，d=3/5/7、p=10^-2/10^-3/10^-4）。DFTT 是无损优化，平均贡献 50.79%（最大 83.58%）的速度提升（消去 log|E| 因子）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Tensor Network Simulation (TNS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Tensor Network Simulation（张量网络模拟）是另一种量子线路模拟范式：与 SVS 类似做矩阵向量乘，但引入 bond dimension D——线路中任意矩阵允许的最大秩；若某矩阵秩超过 D，则把其 D 之外的特征值置 0 做低秩近似（截断）。D 越小近似越强但内存/计算越省，使 TNS 能模拟比 SVS 更宽更深的电路（论文模拟到 40 qubit），代价是输出有近似误差。TNS 对低纠缠电路（自然低 D）尤其高效。
- 本论文把 TNS 作为 TUSQ 的第二个验证后端：TUSQ 的所有组件（ECM+DFTT）都只依赖"矩阵向量乘 + 从向量采样"，因此可直接叠在张量网络模拟器上。TNS+TUSQ vs 未优化 TNS（CUDA-Q tensornet-mps）对 40-qubit QFT/Adder/QAOA(p=2)（bond dimension=16、100k shots、α=0.01、β=100）平均加速 248.39×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TNS 张量计算（低秩近似）：M ≈ M_D = Σ_{i=1}^D σ_i u_i v_i†（截断奇异值分解，σ_i 为前 D 大奇异值），从而把 2^n 维态矢量压缩为 O(n·D²) 规模的张量网络（MPS/MPS 形式）；门的应用变成张量收缩。
- TNS+TUSQ 流水线：① ECM 在 CPU 预采样 ER、合并/剪枝电路实例（与 SVS 场景相同）；② 对每个剩余电路用 TNS 计算输出向量（cuTensorNet v2.9.1 后端）；③ DFTT 用张量网络态在树上的计算/uncompute 复用共享前缀；④ 输出按频率加权采样并平均。
- 实测对比：Unopt TNS vs TNS+TUSQ 时间（秒）：QFT40 1119642→3444、Adder40 628889→2625、QAOA40(p=2) 158407→805；未优化 TNS 在 40 小时超时内未完成（按 100/1k/10k shots 外推）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：NVIDIA cuTensorNet（cuQuantum SDK 组件，v2.9.1）提供 GPU 张量网络模拟 kernel；CUDA-Q 通过 `--target tensornet-mps` 或 `tensornet-mps` flag 调用；本论文的 baseline 用 CUDA-Q 0.11.0 + tensornet-mps 做未优化 TNS。使用：TNS 对 SVS 内存 O(2^n) 不可行的大电路（>30 qubit）是替代方案，但需接受 bond dimension 截断带来的近似；TUSQ 证明冗余消除优化与其正交可叠加。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Depolarizing Channel 与 Pauli Twirling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Depolarizing Channel（去极化通道）是量子噪声建模标准模型之一：以概率 p 把量子态"极化"成完全混合态，等价形式 ρ' = (1-p)ρ + (p/3)XρX + (p/3)YρY + (p/3)ZρZ——以 1-p 概率保持原态、各以 p/3 概率被 X/Y/Z 门作用。本论文把它作为主要噪声模型（默认 p=1%，部分实验 p=0.1%），并指出它是 TUSQ 的 ER 采样与 DFTT 的前提：DEP 采样的 ER 是固定 Pauli 门（I/X/Y/Z，b=4），是幺正的、有逆的。
- Pauli Twirling（Pauli 绕化）是把一般噪声通道（如 amplitude/phase damping、thermal relaxation）近似为 Pauli 通道的技术：用随机共轭 Pauli 门包裹噪声并平均，把退相干通道 ρ→(1-p_X-p_Y-p_Z)ρ + p_X XρX + p_Y YρY + p_Z ZρZ 化到 Pauli 形式，其中 p_X=p_Y=(1-e^{-t/T1})/4、p_Z=(1-e^{-t/T2})/2-(1-e^{-t/T1})/4（T1/T2 为弛豫时间）。Pauli twirling 保证噪声门是幺正 Pauli 门，这是 DFTT 树遍历（要求可逆/uncompute）的硬件前提。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DEP 通道采样（ER 生成的一部分）：
  ```
  def sample_dep(p):
      r = random()
      if r < 1-p: return I
      elif r < 1-p+p/3: return X
      elif r < 1-p+2p/3: return Y
      else: return Z
  # 电路含 m 个 DEP 通道时，ER = (sample_dep(p) for _ in range(m))，b=4
  ```
- Pauli twirling 公式（论文式6）：ρ → (1-p_X-p_Y-p_Z)ρ + p_X XρX + p_Y YρY + p_Z ZρZ；p_X=p_Y=(1-e^{-t/T1})/4，p_Z=(1-e^{-t/T2})/2-(1-e^{-t/T1})/4。X/Y 项只含 T1（amplitude damping）误差，Z 项含 T1+T2。
- 兼容性结论：测量噪声与 DEP 天然满足 DFTT 的幺正性要求；decoherence 经 Pauli twirling 也可纳入；一般非幺正通道（如 erasure）则需 DFTT+Caching（缓存非幺正边前状态）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Qiskit 与 CUDA-Q 的内置 noise model 均支持 depolarizing error（如 Qiskit `depolarizing_error(p, num_qubits)`）与 amplitude/phase damping；CUDA-Q noisy simulation 例子见 https://nvidia.github.io/cuda-quantum/latest/examples/python/noisy_simulations.html。TUSQ 场景：噪声模型的选择决定 ER 的 b 值与幺正性，进而决定用 DFTT 还是 DFTT+Caching；论文 surface code 实验用 Stim 内建电路 + DEP（after_clifford_depolarization）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## TQSim

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TQSim（Wang, Tannu & Nair，ISCA 2025，"Accelerating Simulation of Quantum Circuits under Noise via Computational Reuse"）是加速 noisy 量子线路模拟的近期工作，TUSQ 的主要同类 baseline。核心方法：BFS + memoization + sampling——把 n qubit、深度 d 的电路 C(n,d) 切成 k 个不相交子电路 {sc_i(n,d_i)}，树根为电路起点、深度 i 节点对应子电路 i 结束时的态矢量集合（所有 ER 的 ensemble）；在树的不同深度采样代表性节点并 memoize 其态矢量直到内存饱和，复用保存的态避免从头重算。
- 与 TUSQ 的关键差异：TQSim 的速度依赖可用内存（能缓存多少中间态），内存饱和时必然损失 fidelity（用统计方法在保真度与速度间取舍）；TUSQ 用深度优先遍历（DFTT）实现纯算法加速、不需 memoization（除非非幺正通道），并用 ER 中间表示区分"可无损消除的冗余"（Tallying/Commutation/DFTT）与"可小损消除的不显著计算"（Pruning，α/β 可调）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TQSim 流程对比（同一 noisy 电路、S shots、40GB GPU）：
  ```
  # TQSim（BFS + memoization）
  chop C(n,d) 为子电路 {sc_0..sc_{k-1}}，Σd_i = d
  tree = BFS 逐层生成所有 ER 路径
  在每层采样代表节点 → memoize 其态矢量，直到内存饱和（如 30-qubit 态 8GB，40GB 只能存 5 个）
  后续计算复用缓存态；缓存不足时该部分丢保真度
  # TUSQ（DFTT）
  ECM 先消除冗余（Tallying/Commutation/Pruning）→ 唯一电路建成树
  DFS 正向 compute、反向 uncompute 复用共享前缀，零额外内存 → 纯算法加速
  ```
- 实测：时间与内存密集区（p=1%、1M shots、40GB）同 fidelity 误差下 TUSQ 平均/最大 39.32×/3134.31× 快于 TQSim（6 个 benchmark：adder/bitcode/bv/phasecode/qaoa/qft，5-25 qubit）；低计算非内存密集区（32k shots）TQSim 反而快（BV 3.26×、QFT 2.25×），因为 TUSQ 的 CPU 预处理收益不足以覆盖开销——确认 TUSQ 面向 time+memory critical 场景。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TQSim 论文与代码由作者提供（TUSQ 论文致谢 Meng Wang 分享 TQSim 的 GPU 兼容代码用于基准测试）；TQSim 在 CPU（192GB）与 GPU（40GB）上模拟到 20 qubit（8MB 态矢量，可缓存 24576/5120 个中间态）。使用：作为 noisy 模拟加速 baseline，其最优区间是"计算密集但内存富余"；TUSQ 论文用它证明"缓存方案在 time+memory critical 区失效、需算法级冗余消除"的动机。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Mid-Circuit Measurement (MCM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Mid-Circuit Measurement（线路中途测量）是在量子线路执行过程中、尚未到最终输出前对某些 qubit 进行测量（并把结果用于后续条件操作）的操作。MCM 是非幺正（不可逆）操作：测量塌缩量子态，无法用逆门回滚。在模拟中，MCM 破坏 SVS/DFTT 的"全幺正、可逆"假设——DFTT 的反向 uncompute（乘 U†）对测量边不成立。
- 本论文中 MCM 是 DFTT+Caching 的核心动机：含 MCM 的电路若直接"关掉 DFTT"（每条叶子独立 root-to-leaf 遍历），速度退化为 naive；DFTT+Caching 在每条 non-invertible 边（一层 MCM 合并为一条边）之前的态矢量入 LIFO 缓存（容量 K），回滚跨该边时取缓存态而非求逆，恢复性能。应用场景：FTQC 逻辑级模拟（surface code 轮测量、Magic State Cultivation 验证）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- DFTT+Caching 处理 MCM 的流程（图6，两层 MCM、K=2）：
  ```
  遍历树，遇 pre-MCM 节点 → push 入 LIFO cache（容量 K）
  正向跨 MCM 边（前向 1 操作）正常计算
  反向跨 MCM 边时：不从孩子 uncompute，而是从 cache pop 该 pre-MCM 态（回滚 0 操作）
  该节点的所有孩子子树都回访完后 pop 出（不再需要）
  # K < MCM 层数时：缓存每分支"离叶子最近的 K 个 pre-MCM 节点"，对以最浅缓存节点为根的子树分别 DFTT+Caching
  ```
- 性能恢复 α(K) = (N₁ - N_{DFTT+Caching,K})/(N₁ - N₂)，N₁=DFTT 关闭的操作数、N₂=理想 DFTT 下界；surface code memory 电路（d=3/5/7、p=10^-2/10^-3/10^-4、26/64/118 物理比特，d 轮测量）：容量 3 缓存即可恢复 60%-100%；d 越大（测量轮越多）需要的缓存越大。同一层 MCM 合并成一条树边可大幅降低缓存需求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Qiskit/CUDA-Q 均支持 mid-circuit measurement（如 Qiskit `measure` 加条件 `c_if`/`if_test`，CUDA-Q 的 mid-circuit measurement API）；Stim 内建 surface code 电路含轮测量（rounds=R）。TUSQ 用它验证 FTQC 场景：MSC（Magic State Cultivation）d=3 的 18-qubit 电路含 MCM，原代码 1166.69s（p=10^-4）→ TUSQ 2.24s（520×）；erasure、leakage 等其他非幺正通道同样可用 DFTT+Caching 处理。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Supermarq Benchmark Suite

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Supermarq（Tomesh et al., HPCA 2022，"SupermarQ: A Scalable Quantum Benchmark Suite"，github.com/PrincetonQuantum/Supermarq）是可扩展量子线路基准套件，用于评估量子软件栈（编译器、模拟器、噪声处理等）。TUSQ 用它作为 noisy statevector 模拟的基准集：选择 QAOA、Adder、Bitcode、Phasecode、GHZ、QFT、BV 七类电路，覆盖多种结构（线性：GHZ/Bitcode/Phasecode；并行：QAOA）、qubit 数（13-28）、深度（4-770）、门数（4-1250）与输出分布形态（单峰 Adder/Bitcode/Phasecode、双峰 GHZ、尖峰 QAOA、均匀 QFT）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TUSQ 的 benchmark 配置（表 I）：QAOA 13-25 qubit、depth 82-770、130-1250 门；Adder 4-28、69-289、97-417；Bitcode 5-25、4-144、4-144；Phasecode 5-25、8-48、20-470；GHZ 14-28、14-28、14-28；QFT 14-24、27-47、105-300；BV 4-24、6-26、14-74。评估流程：每个电路加噪声模型（DEP/measurement/Pauli-twirled damping，p=1%）→ 设定 shots（32k/100k/1M/10M）→ 分别跑 TUSQ 与 Qiskit 2.1.0/CUDA-Q 0.11.0/TQSim → 算 speedup γ 与 relative fidelity difference δ。
- 趋势：speedup 随 qubit 数增加（模拟时间指数增长放大任何加速）；固定 qubit 时 speedup 随 QAOA 层数 p 增加而下降（深度增加使 ER 更多样、Tallying/Pruning 有效性降低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：pip install supermarq 或从 GitHub 拉取，生成标准电路（如 `from supermarq.benchmarks.qaoa import QAOA`）并施加噪声再交给模拟器。TUSQ 用它评估 198 个 benchmark（含不同 qubit/深度/shots 组合）并报告平均/最大加速 59.06×/7878.03×（vs Qiskit）与 13.38×/439.38×（vs CUDA-Q）。VQE 正确性验证用 Ising/Heisenberg Hamiltonian（10/15 qubit）而非 Supermarq。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## Magic State Cultivation (MSC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Magic State Cultivation（魔态培育，Gidney, Shutty & Jones, arXiv:2409.17595）是低开销制备高保真 T 魔态（magic state）的技术——T 门是非 Clifford 门，FTQC 中需通过魔态蒸馏/培育获得；MSC 声称"培育 T 态像 CNOT 门一样便宜"。论文 [16] 用 statevector 模拟验证 d=3 的 MSC 电路正确性，更大码距只能靠启发式猜测。
- 本论文把 MSC 作为 TUSQ 的 FTQC 验证用例：MSC 电路含 mid-circuit measurement，TUSQ 以 DFTT+Caching 支持；用 18-qubit、d=3 的 MSC 电路（p=10^-4）对比 [16] 原代码库（数据在 Zenodo 10.5281/zenodo.13777072）：原代码 1166.69s → TUSQ 2.24s，520× 加速。这展示了 TUSQ 对"可扩展模拟器使更大码距 MSC 验证成为可能"的贡献。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MSC 验证流水线：① 用 Stim 等工具在物理层得到逻辑错误率 → ② 对含非 Clifford 门的逻辑级电路做 noisy statevector 模拟（此时 circuit 含 MCM/培育流程）→ ③ 验证培育出的 T 态保真度。TUSQ 的角色在②：逻辑级模拟是深电路、多逻辑比特、time+memory critical，且含非幺正 MCM 边——正好落入 TUSQ（ECM+DFTT+Caching）的最优区间。
- 在 TUSQ 内的执行：ECM 消除冗余电路实例 → DFTT+Caching 沿树遍历（MCM 边取缓存）→ 输出分布与 [16] 原实现对比，同输入下 2.24s vs 1166.69s。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：MSC 电路与数据来自 Gidney 等（Zenodo 10.5281/zenodo.13777072）；Stim 可用于物理层模拟。TUSQ 论文用它论证"TUSQ 可加速 FTQC 逻辑级子程序验证"（类似用途还有 FTQC 逻辑级噪声模拟：物理层 Clifford 用 Stim、逻辑层非 Clifford 深电路用 TUSQ）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## PIMDT（PIM Data Type，PIM 友好列式存储格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIMDT 是论文为 BLIMP（bank 级 PIM）OLAP 数据库提出的列式存储数据类型：把指定的数据库列以 "PIM 友好" 布局常驻存储——整字（如 64-bit）落在单个 DRAM bank 内、BLIMP 核可直接经本地 row buffer 访问，从而避免查询时的软件 relayout。其他 "host 列" 保持原布局。通过 SQL 列约束声明，如 `Bar bigint NOT NULL PIMDT(BLIMP)`。设计约束：(1) 只能用于定长类型（可变长字符串、blob 与 PIMDT 不兼容，因其必须能按元素边界快速 chunk 化）；(2) 列上的算子必须 PIM-amenable（数据并行、可向量化、无跨数据依赖）；(3) 更新/插入需按字节重排（每个插入字节一次写）；(4) 查询时按 configurable chunk 大小把 PIMDT 数据均分到各 bank，满足 32MB bank 容量约束（除列数据外还要容纳算子指令、输入与输出数据）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文流程（SSB SF100）：LINEORDER 的外键列（lo_orderdate、lo_partkey、lo_suppkey、lo_custkey）与过滤列（lo_quantity、lo_discount）声明为 PIMDT 常驻；查询时存储管理器只把"预铺好"的 PIMDT 列分区 relayout 载入各 bank（无需整列查询时重排）→ 算子内核就地执行 → 结果位图/部分物化留在 bank 内链给下一算子 → 最后 host 取回。伪代码级（chunk 化与加载）：
```
# 查询执行前（存储管理器）
for col in query.PIMDT_columns:
    chunks = chunk_by_element_boundary(col, bank_capacity=32MB - reserved)
    # 每 chunk 的元素在单 bank 内连续，host 只做一次 relayout 载入
# 执行时：每 bank 载入 kernel + PIMDT chunk + 辅助数据 → BLIMP 核计算
```
选择 PIMDT 列的判据类似传统索引创建决策：列被查询频率（存储权衡）与 PIM 适性（运行时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文在存储管理层实现 PIMDT（storage manager 判断 offload 所需数据、减少 relayout 依赖、复用已 relayout 数据），查询解析器需解析 PIMDT 列约束语义；无法在 PIM 执行的算子回退 host（此时 PIMDT 列需 relayout 回 host 布局）。端到端效果：PIM 感知规划（含 PIMDT + 晚物化 + 低选择性优先 join 序）比隔离算子外推快 3.2×；隔离计划平均 22% 查询时间在 relayout。论文未声明 PIMDT 实现开源（论文未明确说明）；同组 BLIMP 框架 dovedevic/blimp（https://github.com/dovedevic/blimp）含 /relayout 例程。

涉及论文标题：
- Taking Analytic Databases to the Bank

## 行缓冲对齐桶式哈希表（Row-Buffer-Aligned Bucket Hash Table，BLIMP 哈希表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
面向 BLIMP（bank 级 PIM）设计的哈希表数据结构，用于 join 的 build/probe 与高基数分组聚合。针对 BLIMP 核弱（200MHz RISC-V）且以 1KB row buffer 粒度访存的特点，设计目标：①索引容易——hash 后能直接定位到要读写的一行；②row buffer 对齐——减少 row buffer 切换；③冲突局部性——hash 冲突的值应共处同一行以利用空间局部性。实现：row-buffer 对齐的 hash-indexed bucket 集合；初始桶数为 2 的幂（按期望 load factor 定），桶大小恰好契合 row buffer；桶含 metadata 与一串 slot（每 slot 一个列值，可带 payload——join 的 payload 或聚合器）；冲突时 slot 追加到桶尾，桶满则在桶链末尾建新桶并在原桶记录 next 指针。哈希函数用轻量乘法哈希保证强抗冲突：`hindex = (3634946921 * value + 2096170329) & (initial_buckets - 1)`；因桶大小固定，hindex 可直接换算桶所在 row buffer 地址，实现 hash→地址 O(1) 映射。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BLIMP-S 与 BLIMP-V 的桶内 slot 数差异显著（32-bit key + 8-bit payload 时 BLIMP-S 5 slots、BLIMP-V 24 slots/桶），因 BLIMP-V 的 SIMD 能力使扫描桶的时间逼近取新桶（row buffer 切换）的时间。探测伪代码（Algorithm 2 核心）：
```
for each 元素 v1[i]（从 row buffer 读入）:
    idx ← hash(v1[i]) & (initial_buckets - 1)      # BLIMP-V 可向量化批量 hash
    repeat:
        v3 ← FetchMem(h + BucketRow(idx))          # 打开目标 row buffer 桶
        hit ← (v1[i] ∈ v3.slots)                   # 串行检查 slot 列表
        idx ← v3.next_bucket                        # 桶链下一桶（若桶满溢出）
    until IsNull(idx) or hit
    v1[i] ← hit                                    # 位图置位
```
性能：build 由 host 完成（relayout 广播到各 bank），build 时间与 CPU 侧 Swiss Table 相当（论文引 abseil）；probe 侧随机 row buffer 访问是主要开销——低选择性时整个哈希表可入 host cache（host 占优），高选择性时 host 因 L2/L3 miss 劣化更快而 BLIMP 只受 row buffer 切换惩罚。semijoin 1.4×/2.1×、join 2.1×/3.0×（BLIMP-S/-V）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用场景：hash join 的探测侧、高基数 group-by 聚合（分组值 hash 到 bucket/slot，payload 为聚合器）、set 操作。实现要点：初始桶数按期望 load factor 定且为 2 的幂；桶大小选为"扫描时间 ≈ 取新 row buffer 时间"（BLIMP-S/-V 不同）；链式桶溢出处理；哈希表必须能在 32MB bank 容量内（大 build 侧按分区多轮 build-probe）。论文未声明该数据结构开源（论文未明确说明）。

涉及论文标题：
- Taking Analytic Databases to the Bank

## 物化策略与 PIM 感知查询规划（Materialization Strategies & PIM-Aware Query Planning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DBMS 查询执行中的中间结果物化策略与查询规划启发式。传统（CPU 中心）列存 DBMS：select 输出 index array（随选择性线性增长）或 bitvector/bitmask（定长）；用位图可简化算子间流水（AND 合并两个过滤），但其他算子不接受位图输入时被迫物化（materialization）。CPU 侧启发式：①高选择性谓词先做（减少后续算子处理记录数）；②最小化哈希表大小（塞进 cache）；③左深 join 树。论文发现这些启发式在 BLIMP（bank 级 PIM）上失效或反转：(1) 哈希表 >几 KB 后大小不再影响 PIM 性能（probe 是随机 row buffer 访问、命中率低），只有选择性驱动 join 排序——CPU 推荐的 SDC 序在 BLIMP 上是 6 种序中第 4 差（比最优 CSD 慢 23%）；(2) 物化时机决定 compute domain 转换与 relayout 次数——Early Mat.（全部在 BLIMP 物化）、Hybrid Mat.（只在 domain 转换前物化）、Late Mat.（host 物化），bitvector 定长 vs value array 随选择性线性增长使成本随选择性带变化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
示例查询（SSB Q1.1 SF100）`WHERE Fizz < 25 AND Buzz BETWEEN 1 AND 3` 的五种计划：①Host——两 PIMDT 列先整列 relayout 回 host 再评估（最贵）；②Isolated——两个 BLIMP select 各自隔离执行、结果分别 relayout 回 host 由 host 做 AND（多数 PIM 研究做法；relayout 浪费严重）；③Early Mat.——两个 select 输出数组都留在 bank 内，再派发一个 BLIMP 逻辑 AND 算子合并，最后才回 host（relayout 最少但第一个过滤未让第二个过滤少处理记录）；④Hybrid Mat.——第一个过滤输出 bitvector 留在 PIM，第二个过滤并行处理时隐式 AND，只在 domain 转换前物化（最优区域）；⑤Late Mat.——同 ④ 但全部由 host 物化（适合高选择性，value array 小）。伪代码级：
```
# PIM 感知规划（论文方法）
plan = []
for op in query.operators (PIMDT 列上):
    if op 支持在 BLIMP 执行:
        plan.append(dispatch_blimp(op))          # 预处理→relayout→执行→部分物化/原位保留
    else: plan.append(dispatch_host(op))          # 回退 host
join_order = sort(joins, key=selectivity_asc)     # 低选择性优先（PIM 主驱动）
materialize = "late" if 整体高选择性 else "hybrid" # bitvector 定长 vs value array 线性
```
结果：PIM-optimal 比 Isolated 快 3.2×（平均 22% 时间省在 relayout）、比 CPU-optimal 快平均 28%（最大 40% Q3.3）；bushy join 树（Q4.3）需多次重建哈希表、重复 relayout+build+broadcast，不适合 PIM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文用手工按 PIM 启发式构造查询计划（自动规划器留作 future work），查询执行器支持"PIMDT 列上的算子→BLIMP 工作流、否则回退 host"；解析与分析沿用常规 DBMS（仅额外解析 PIMDT 列约束）。CPU 侧基线用 DuckDB 生成计划（含 bushy join 的 Q4.3）与手调 C++ 单块 kernel。该策略源于列存数据库物化研究（Abadi et al. 2007 物化策略、Vetica late materialization），在 PIM 上下文中重新评估其价值。

涉及论文标题：
- Taking Analytic Databases to the Bank

## 稀疏张量收缩（Sparse Tensor Contraction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
张量收缩是矩阵乘法到多维张量的推广：对两个张量共有的模式求和，产生降阶输出张量。形式化（TensorPrism 式 1）：$C_{f_1,f_2}=\sum_c A_{\{f_1\},\{c\}}B_{\{c\},\{f_2\}}$，其中 $\{f_1\}$、$\{f_2\}$ 为不参与收缩的自由模式（free modes），$\{c\}$ 为收缩模式（contraction modes）；沿每个收缩模式的 fiber（单模式向量切片）长度必须匹配；输入含 m、n 个模式时输出含 $m+n-2|\{c\}|$ 个模式。稀疏版指参与张量含大量零元素、只存/算非零（NNZ），稀疏度跨模式变化且存在跨模式稀疏依赖，是高阶稀疏计算的核心原语。例：3D 张量收缩 $C_{i,j,l}=\sum_k A_{i,j,k}B_{k,l}$；单收缩模式+单自由模式退化为 SpMM $C_{M,L}=A_{M,K}B_{K,L}$。应用：LLM 多头注意力 4D 张量（batch/head/seq/channel）、3D 卷积（多收缩模式 f=2）、科学计算、推荐系统用户-物品-上下文交互、量子态模拟（20-400 阶）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
收缩模式决定稠密输入 B 的取行位置（B[K,:]），自由模式决定输出 C 的位置（C[I,J,:]）——这一"模式角色划分"是数据流设计（push/pull）的依据。TensorPrism 执行伪代码：
```
for each partition P_i (CoGTP 划分, N=PE 数):
    for each clique (I,J,K) in P_i:      # clique=非零元素
        # contraction 顶点 K PUSH 稠密行: 标量-向量乘+向量累加
        partial_C[I,J,:] += B[K,:] * A[I,J,K]
    # 自由顶点 PULL 累加输出行 C[I,J,:]
```
评估覆盖 k∈{64,128}（特征长度）与 f∈{1,2}（收缩阶数）。三类执行路线：mode unfolding（矩阵化）、einsum lowering、张量原生循环变换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：COO/CSR 稀疏格式 + einsum 记号描述收缩 + 展开成 SpMM 或张量原生循环。TensorPrism 的做法：NNZ 坐标→共现图（顶点=索引、边权=共现次数）→ CoGTP 按式 6 划分（决定各模式 tiling 因子）→ 图数据流 push/pull 执行。应用场景：FROSTT 8 数据集（Uber/Nips/Nell-1/Nell-2/Flickr/LBNL-Networks/Chicago-Crime/Amazon-Reviews，3-5 阶、密度 1e-14~1e-2、NNZ 百万到十亿级）+ LLaMA 注意力张量（稀疏化 1%/10%/20%）。性能：相对 SPADE/HotTiles/GSpTC/TCP/HyperSB 几何平均 2.22×/2.40×/1.71×/1.76×/1.49×。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## 张量展开/矩阵化（Tensor Unfolding / Matricization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
张量展开（矩阵化，matricization / mode unfolding）把高阶张量重排成矩阵：选定若干模式合并为矩阵的行/列索引。例如 $A_{i,j,k}$ 把 (i,j) 合并为行得 (IJ×K) 矩阵，或把 (i,k) 合并得 (IK×J) 矩阵；这样张量收缩 $C_{f_1,f_2}=\sum_c A_{f_1,c}B_{c,f_2}$ 变成标准 SpMM $C_{M,L}=A_{M,K}B_{K,L}$（M=自由模式合并、K=收缩模式、L=另一自由模式）。这是 einsum 落地的常见 lowering，用以复用成熟的 SpMM 优化（inner/outer/Gustavson 数据流、自适应 tiling、矩阵重排如列置换与图 islandization）。TensorPrism 指出展开的代价：(1) 元数据膨胀——张量原生格式 O(I+J+K)，unfold 成 CSR/CSC 后 O(IJ+K)；(2) 复用距离膨胀——max reuse distance 从 I+J 变 I×J；(3) 相邻非零邻居减少——2D 中每非零最多 4 个结构相邻邻居，3 阶张量有 6 个；(4) 不同展开方式产生不同稀疏模式，优化 mode-dependent；(5) 把中间结果映射回原张量域时部分计算不可恢复。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
展开→执行→还原三步 pipeline（SPADE/HotTiles baseline 路线）：
```
# unfold: (i,j,k) -> (m=i*J+j, k)
A_mk = reshape_coo(A, order=(i,j,k), row_modes=(i,j))
C_ml = SpMM_rowwise(A_mk, B)    # for m: for k in nnz(A_mk[m]): C[m,:]+=A_mk[m,k]*B[k,:]
C = reshape_back(C_ml, out_modes=(i,j,l))   # 映射回张量域(此处信息部分不可恢复)
```
后果量化（论文 Fig.3）：unfold 后循环变换丢失 50-60% 潜在数据复用，量子模拟（高阶）达 90%；uber 上 SPADE 91% 开销来自稠密行重复取数（复用距离膨胀超过片上容量）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：COO 坐标按目标 mode 顺序重排 → CSR/CSC 压缩（行指针+列索引）。使用场景：所有"借用 SpMM 优化"的稀疏张量加速路线（ExTensor/SEXTANS/DRT/SPADE/HotTiles/Trapezoid/Misam）。TensorPrism 用它作为 baseline 执行模型（SPADE/HotTiles 按各自论文算法贡献 + matricization 扩展），并证明其局限后以共现图替代（划分前统一分析所有维度索引交叠）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## 共现图（Co-occurrence Graph）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
共现图把稀疏高阶张量变换为加权无向图 CoG(V,E)：每个顶点是一个张量索引（r 阶张量 $A_{I,J,K}$ 有 |V|=I+J+K 个顶点），任意两个索引若在同一非零元素（超边）中共现则连边，边权等于共现次数 $W((i,j))=nnz(A[..;i;..;j;..])$（式 2）。完整子图（clique）对应一个非零张量元素；每个非零指示一次类似 SpMM 的向量乘操作。它是超图的 pairwise 投影（数学等价但显式暴露索引重叠），用 CSR 存储足迹 $M_{CSR}=(\sum|r|+1)\times4+E\times4$ 字节（FP32，E 为不重叠边数，$E \ll N\binom{r}{2}$）。基于它可量化张量收缩的复用（式 3）：contraction 模式共享顶点→输入复用（B[K,:] 被多个输出目标复用），free 模式共享顶点→输出复用（同一 C[I,J,:] 聚合多个输入），把 tiling 变成可求解的图划分问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
构建与使用 pipeline：
```
# 构造
for (idx, val) in nonzero_coords:       # 每个非零
    for (u,v) in pairs(idx):            # 全索引对
        W[u,v] += 1                     # 共现计数(去重)
G = CSR(V, E, W)
# 使用: 复用量化 + 划分 + 数据流
Reuse = sum(D(v))/|f1| + sum(D(v))/|c| - ...   # 式3, D=加权度
P = CoGTP_partition(G, N)               # 式6 PCST 式划分
# 执行: 沿图遍历 push/pull
```
例子：$2\times2\times2$ 张量生成 6 顶点图，$W(I_0,K_1)=2$ 表示两索引在 2 个非零中共同出现；三元组 (I0,J0,K0) 构成 clique ⟺ A[I0,J0,K0] 非零。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现（TensorPrism CoG Scheduler 的权重计算单元）：Coordinate Parser 提取索引对、Dimension Pair Selector 选维度对、Hash-based Engine 去重、Index Pair Buffer（FIFO，可配深度如 256）缓冲、图生成器构造 CSR 共现图、成本分析器评估式 6 划分质量。使用场景：任何高阶稀疏张量收缩（LLM 注意力、科学计算、推荐、量子模拟），密度跨 14 个数量级。局限性：边权只统计共现次数不区分输入/输出模式；图构建有预处理开销（较 SPADE/HotTiles/GSpTC 增 8.0%/6.7%/4.2%，远低于 TCP 25.4%）；存储较超图平均增 3.0%（非瓶颈）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## 超图（Hypergraph，张量表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
超图是普通图的推广：一条超边可连接任意数量顶点。在稀疏张量场景中，每个非零元素用一条超边编码其全索引集（如 $A_{i,j,k}\ne0$ ⟺ 超边 {i,j,k}），顶点集=全部模式索引，从而完整保留张量的高阶稀疏结构（哪些索引组合非零）。划分工具如 KaHyPar（k-way hypergraph partitioning）在其上做最小边割+顶点数均衡划分。TensorPrism 指出超图作为张量表示的缺陷：(1) 坐标重叠（索引重叠=数据复用指示）不直接暴露——超边列出非零但无法直接回答"索引 i 与 j 共现几次"；(2) 传统超图划分只均衡顶点数与边割，而边数（=非零元素=关联计算量）代表实际工作量，工作量失衡；(3) 复杂度随张量阶数快速上升（量子模拟 90% 复用机会丢失）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
超图表示→划分→执行的 pipeline（HyperSB baseline 路线）：
```
edges = [set(idx) for (idx, val) in nonzero_coords]   # 每非零一条超边
part = KaHyPar_partition(vertices=all_indices, hyperedges=edges,
                         k=16, objective=min_edgecut)
# 各分区独立执行收缩; 缺陷: 工作量(边数)未均衡, 复用不在目标函数
```
超图与共现图的数学等价性：任意顶点对若共现则连边、边权=共现超边数；TensorPrism 用共现图保留超边语义（可恢复原非零元素）的同时暴露重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：COO 坐标→超边集→超图划分库（KaHyPar，https://github.com/kahypar/kahypar）。使用场景：张量原生稀疏收缩划分（Gündüz et al.、HyperSB）。TensorPrism 以其为 baseline（HyperSB 平均 1.49× 更慢），并说明其仍优于 GSpTC/TCP 的原因：直接表示高维顺序、消除展开导致的中间数据膨胀。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## Prize-Collecting Steiner Tree（PCST，带奖收集斯坦纳树）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PCST 是斯坦纳树问题的带奖变体：给定无向图 G=(V,E)，每条边带非负代价 $c_e$、每个顶点带非负奖 $\pi_v$，目标是找连通子图（树）$T=(V_T,E_T)$ 使 $\min_T(\sum_{e\in E_T}c_e+\sum_{v\notin V_T}\pi_v)$——最小化"连进来的边代价 + 未收集顶点的奖"。与必须连接给定终端的经典斯坦纳树不同，PCST 把"哪些顶点必须连"放宽为"每个顶点可选且给奖"，在连边代价与放弃顶点奖之间取平衡。它是大规模图系统组合优化的经典问题，广泛用于网络设计、设施选址等领域。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TensorPrism 把张量 tiling 重述为"修改版 PCST"（CoGTP）：连通子图=一个分区，边权 W(e)（共现次数）对应"奖"（收集=复用收益），跨分区边（cut）对应"代价"，λ_b 负载均衡为额外约束。目标（式 6）：$\max\{\alpha\sum_i\sum_{e\in P_i}W(e)-\lambda_{cut}\sum_{e_{cross}}W(e)-\lambda_b\sqrt{\sum_i(\sum_{v\in P_i}(D(v)-W/N)^2)}\}$，α=2.0/λ_cut=1.0/λ_b=1.0。求解：BFS 多种子初始化（先收集高连通区）→ Kernighan-Lin 式边界单顶点迁移迭代（每步算 ΔF、保留正增益、ΔF<ε 收敛），每轮复杂度 $O(\sqrt{|V|}d)$（d=平均度 1-10），近线性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（Algorithm 1）：输入共现图 G + 分区数 N，输出 N 个分区。三步：构造共现图（顶点=索引、边权=共现次数）→ BFS 初始化 → 迭代细化。参数选择依据：α=2.0 因高度数顶点提供 O(k²) 复用、O(k) 存储（2:1 收益比）；λ_cut=1.0/λ_b=1.0 匹配单位通信成本与 power-law 图负载分布系数。敏感性（Flickr）：α≤3 稳定、α>3 崩塌；λ_cut>1.5 后性能骤降；λ_b 在 1.0-1.5 达峰；(α=2,λ_cut=1,λ_b=1) 达 93.8% 峰值性能为保守最优。应用：FROSTT 8 数据集 + LLaMA 注意力张量的分区，N=16（PE 数）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## CoGTP（共现图张量划分算法，Co-occurrence Graph Tensor Partitioning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CoGTP 是把稀疏高阶张量收缩的 tiling 形式化为共现图上修改版 PCST 划分的算法，三项目标并行优化：(1) 图内复用——同分区高权值边=高共现=高时间局部性（第一项 α·ΣW(e)）；(2) 跨分区通信——cut 边惩罚（λ_cut·ΣW(e_cross)）；(3) 负载均衡——加权度 D(v)=ΣW(u,v) 估计工作量、二次偏差惩罚（λ_b·RMS 项）。产出 N 个分区（N=PE 数），决定张量 A/B/C 的 tiling 因子。流程：构造共现图→BFS 多种子初始化→Kernighan-Lin 式迭代细化（只迁移边界顶点、算 ΔF、保留正增益、ΔF<ε 收敛），每轮 $O(\sqrt{|V|}d)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 1 概要
G = build_cograph(T)                      # 边权 w(u,v)=nnz(T[..;u;..;v;..])
D(v) = sum_u w(u,v)                       # 加权度=工作量估计
P = BFS_seed_clustering(G, N)             # 初始化
F = objective(P)                          # 式6
while True:
    best = None; max_gain = 0
    for v in boundary_vertices(P):        # 只在分区边界找候选
        for j in candidate_partitions(neighbors(v)):
            gain = delta_objective(v, i->j)
            if gain > max_gain: best=(v,i,j); max_gain=gain
    if best is None: break
    apply(best)                           # 单顶点迁移
    if objective(P) - F < eps: break
    F = objective(P)
```
例：尝试迁移 K1 被丢弃（ΔF 负）、迁移 K0 被保留（ΔF 正）（Fig.6）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：TensorPrism 共现图调度器的成本分析器（占调度器面积 62.9%）按式 6 评估分区质量，加法器汇总、分区索引缓冲存 PE 映射；λ_b 还作为运行时 DMUX 重分配（把局部 buffer 条目分给欠利用 MAC）的算法级前导（双级负载均衡：静态 CoGTP + 运行时 DMUX）。使用场景：FROSTT 8 数据集（密度 1e-14~1e-2）+ LLaMA 注意力张量。效果：2.22×/2.40×/1.71×/1.76×/1.49× 加速（vs SPADE/HotTiles/GSpTC/TCP/HyperSB），DRAM 访问降为 1/2.18/2.11/1.27/1.53，复用效率 67.86%（高 23.7%~57.4%）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## 复用距离（Reuse Distance）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
复用距离是缓存/局部性分析的经典度量：一个数据项两次访问之间被访问的不同数据项数量。距离小=高时间局部性；距离超过片上容量则该复用必然 miss。TensorPrism 把它适配到稀疏张量收缩（论文 §III-B，受 cache 建模 reuse distance 分析启发）：以"稠密行 B[K,:]"为被复用对象，复用距离=两次取同一稠密行之间访问的不同稀疏行数。对 3 阶张量 $A_{i,j,k}$，max reuse distance 上界为 I+J（自由模式大小之和）；unfold (i,j) 成单维后膨胀到 I×J。此外 unfold 还减少相邻邻居（2D 中每非零最多 4 个结构相邻、3 阶张量 6 个），立即复用机会更少。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
trace 驱动统计：
```
stack = {}
for row in access_trace(B):          # 稠密行访问序列
    if row in stack:
        dist = len(rows between prev and now); record(dist)
    stack.push(row)
# 结论: unfold 使 max dist 从 I+J -> I*J
```
后果：复用距离膨胀超出 GLB/片上容量→稠密行被迫重复从 DRAM 取（uber 上 SPADE 91% 开销、2.09× 超额执行时间）；量化出 unfold 后循环变换丢失 50-60%（量子模拟 90%）复用。对策：CoGTP 把高共现顶点（短复用距离的稠密行复用）聚到同分区=直接缩短有效复用距离；式 5 在 GLB 容量 $M_{cap}$ 约束下选 tiling 因子 $M_t$ 使复用距离 ≤ 片上驻留能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：记录稠密行访问序列，用栈/树统计两次访问间的不同行数，得复用距离分布。论文用它做动机分析（Fig.3 数据复用分析）而非运行时开销。在加速器设计中对应 GLB 容量约束（式 5）与 CoGTP 分区目标；48KB/PE 局部存储决定哪些距离可命中。场景：任何稀疏张量收缩的局部性评估。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## SIKE（超奇异同源密钥封装）与 Montgomery ladder 侧信道

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SIKE（Supersingular Isogeny Key Encapsulation）是基于超奇异椭圆曲线同源的候选后量子 KEM，属 NIST PQC 竞赛 4 轮候选（已被 Castryck-Decru 密码学攻击破解，但仍是侧信道研究标杆目标）。Cloudflare 的 CIRCL（Interoperable Reusable Cryptographic Library）库提供 SIKE 实现。SIKE-751 的私钥是 378-bit 二进制整数 m，解密过程用 Montgomery ladder 逐位处理：若第 i 位与第 i−1 位不同（m^i ≠ m^(i-1)），ladder 的 (i+1) 步产生零值、导致解密停滞（stall）、功耗下降；若相同则正常计算、功耗较高。这种"数据相关的功耗差异"正是 Hertzbleed 与本文 TimeGaps 攻击的利用点。
- 在本文中：TimeGaps 采集器在 CIRCL 以 300 个并发 goroutine、10 个随机 378-bit 密钥运行时，逐位猜测过程中收集 TimeGaps；m^i≠m^(i-1) 时（零值 stall、处理器运行在更波动频率环境）无 TimeGap 出现概率 95.48%，m^i=m^(i-1) 时（稳定运行）无 TimeGap 概率升至 96.14%——该区分足以恢复密钥。最终只需确定首位是 0 还是 1，把密钥搜索空间降为 2。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SIKE 解密 pipeline（Montgomery ladder，一位处理）：
```
m = 378-bit 私钥（明文形式）
for i in 1..378:
    if m^i != m^(i-1):        # 相邻位不同
        P_zero ← ladder 步骤产生零值 → 算力 stall → 功耗下降 → 频率波动 → 更少 TimeGaps
    else:                      # 相邻位相同
        正常同源计算 → 功耗稳定 → 更多 TimeGaps（无 TimeGap 概率 96.14% vs 95.48%）
    攻击者统计每个位猜测窗口内"无 TimeGap"概率 → 反推相邻位是否相同 → 重构 m
```
- 该 pipeline 属于"数据相关功耗 → 频率/挂起时间 → 观测分类"的侧信道算法链，与一般 ML 推理 pipeline 不同：输入是密码学运算，输出是密钥位。TimeGaps 作为测量原语插入到攻击者观测层，不修改受害者算法本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：SIKE 实现来自 Cloudflare CIRCL（https://github.com/cloudflare/circl），Go 语言库，攻击时 spawn 300 goroutine 并发跑 10 个密钥制造系统负载；攻击者程序与其并发运行，在 i5-8259U 上经 SSH 访问（减少 iGPU 噪声）。使用场景：评估新侧信道（Hertzbleed、TimeGaps）的密钥提取能力，作为与 SIKE 密码学攻击对比的标准侧信道 benchmark。指标：每 bit 猜测的 TimeGap 分布、无 TimeGap 概率差、最终密钥恢复是否只需 2 次尝试。

- **TIDE 版（macOS/Apple Silicon）**：与 TimeGaps（x86 频率/挂起时间）不同，本文用 TIDE 中断计时原语采集 SIKE 解密时的频率相关计数器变化。实验设置：MacBook Air 2023（M3），CIRCL v1.1 以 300 个并发 goroutine 跑 10 个随机 378-bit 密钥；每次 bit 猜测收集 50,000 个 TIDE 计数器值，只保留超过 3000 万的样本（低于该值的更可能被噪声中断污染）取平均。结果：m^i=m^(i-1)（无 stall）时计数器平均 32,119,284；m^i≠m^(i-1)（Montgomery ladder 零值 stall、处理器低频运行）时平均 31,362,263——可区分即恢复密钥，搜索空间降到首位 0/1 两个候选。前置实验：空闲系统下约 40% 的计数器值落在 40,470,000–40,540,000，对应 M3 P 核 4.050 GHz 下 100 Hz 的定时器中断间隔（macOS 固定间隔定时器中断）。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon


## 空循环计数（loop counting）与 LSTM 网站指纹分类 pipeline

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空循环计数（loop counting）是浏览器环境（无 rdtscp、无高分辨率定时器）下的执行速度测量原语：在固定时间间隔内数"空循环能完成多少次迭代"，迭代数越低说明期间发生过 TimeGap（CPU 挂起导致吞吐下降）。源自 bigger-fish（Cook et al., ISCA 2022，本文引用 [8]）的执行速度侧信道。本文验证：TimeGaps 总时长与 loop counter 值的 Pearson 相关为 −0.70±0.06（强负相关），比 Fish and Chips 报告的 loop counter 与中断处理时间相关 −0.49±0.11 更强，说明 TimeGaps 是 loop counter 凹陷的主要贡献者。
- LSTM 网站指纹分类 pipeline：以 500μs 间隔记录三种通道（native TimeGap 总时长 / 浏览器 loop counter / CPU 频率 scaling_cur_freq），把时间序列输入 32 单元 LSTM（与 prior work [8,44,57] 相同架构与超参），10 折交叉验证（81% 训练/9% 验证/10% 测试）输出 top-1 网站分类。数据集：Alexa Top 150 中前 100 个活跃非成人网站（sites/closed_world.csv），Chrome 每站 100 条 ×15s trace、Tor 每站 30s。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 检测与分类 pipeline：
```
# 采集阶段（每 500μs 一个时间步）
for each 500μs window:
    native:   gaps[i]   = 窗口内 TimeGap 总时长（rdtscp + SegScope）
    browser:  counter[i] = 窗口内空循环完成迭代数（JS）
    freq:     freq[i]   = 读 scaling_cur_freq
# 分类阶段
X = [gaps / counter / freq] 序列（T 时间步）→ LSTM(32 units) 逐时间步 h_t ← LSTM(x_t, h_{t-1})
→ softmax(100 类) → 10 折 CV 平均 top-1 准确率
```
- 例子（固定频率 Chrome）：native TimeGaps 序列 → 92.2±0.7% top-1；同一数据源在默认 DVFS 下 98.0±0.9%（DVFS 增加频率波动信息）；loop counter 浏览器通道固定频率 92.3±0.7%；频率通道固定频率下仅 ~1%（固定频率后频率不再变化，只有 TimeGaps 仍在泄漏）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：native 采集器 C（gaps_collector 系）+ SegScope；浏览器用 JS 空循环；分类用 Python（scikit-learn Random Forest 简化版 / 论文用 LSTM）。artifact 开源 Zenodo https://doi.org/10.5281/zenodo.19450827（MIT），attacker.py 自动化浏览器控制（Selenium+ChromeDriver）、数据采集与准确率评估；参数 --sites_list alexaN / --trace_length / --num_runs / --attack all / --core N，fixed_freq.sh/default_freq.sh 切换频率模式。使用场景：网站指纹攻击（浏览器场景攻击者只需让受害者访问恶意页面）、也可用于评估指纹防御（随机化防御下 TimeGaps 仍 83.1±1.3% vs 中断 61.2%）。

- **TIDE 版（本文，macOS/Apple Silicon）**：不依赖定时器，把 TIDE 中断计时 trace 直接喂给相同架构的 LSTM(32 units) 分类器（10 折 CV，训练 81%/验证 9%）。数据集：closed-world = Alexa top 100 网站×每站 100 条 trace；open-world 另加 Alexa top 1M 中 2000 个网站各 1 条（other-class）；视频 = YouTube 美国 top 20×每视频 200 条 trace，默认在 MacBook Air M3（4E+4P）采集。结果（表 III）：closed-world top-1 93.8%/top-5 98.7%，open-world 91.5%/98.5%，视频 78.1%/97.9%；用固定 24 MHz cntvct_el0 替换 TIDE 计数器（去频率缩放影响）后 93.1%/91.1%/87.2%。对比 Cook 定时器版（94.8%/74.5%）：TIDE 稍低但**在随机定时器防御下仍有效**（定时器版被打回 ~1%/5%）。跨硬件：10 网站 closed-world top-1 96.9%（M3 Air）/93.3%（M1 Pro 2021）/80.1%（M3 Max 2023）；双 TIDE 线程并行采集收益有限（95.7%/69.3%/74.2%），因为 Apple 均匀投递使每核信号变弱；播放 60s 视频做背景噪声时仍 60.6%。
- **loop-counting 增强（基于反推结果）**：因为 Apple 按 active core 数决定 SPI 投递，计算密集线程（不产生中断）会扭曲 trace（图 8）：训练/测试都无噪声 94.8%；训练无噪声+测试有噪声 39.3%；以"网站×active core 数"为联合标签（只改分类器最终层）后，有/无噪声分别 92.8%/93.6%（表 IV）。

涉及论文标题：
- TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon


## 格点手术（Lattice Surgery）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
格点手术（lattice surgery，Horsman et al. 2012）是 surface code 架构上实现多逻辑量子比特操作（尤其是联合逻辑 Pauli 测量）的主流容错方案：在相邻代码 patch 的边界临时修改稳定子集合——merge 阶段沿边界加入联合稳定子把两个 patch 耦合（其乘积等于联合逻辑可观测量，如 ∏S_k^(XX)=X_L^(L)·X_L^(R)），split 阶段恢复原 patch；奇偶测量结果作为经典边信息用于更新 Pauli frame，无需物理纠错。由联合测量可构造 CNOT/S/H 等 Clifford 门；非 Clifford 门（T）则经 gate teleportation 消费 magic state 实现。时间成本与码距 d 线性相关（约 d 轮 code cycle 或 "code beats"）。Web：arXiv:1808.02892（Litinski, Quantum 2019）把表面码操作抽象为 tile-based game——patch 占 tile，虚线边=X 算子、实线边=Z 算子，操作含单 patch 测量（0 代价）、multi-patch 测量（1 代价）、patch deformation（enlarge 1/shrink 0），时间步=surface-code cycle（d 轮测量）。Triage 论文用 multi-patch measurement / patch rotation / idle 作为指令集（[38] 编译器）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
格点手术是 Triage 空间并行性的来源：lattice surgery 临时合并相邻 patch 并在边界测量联合稳定子，使错误在合并区域内空间相关——单解码器若把合并后的大 volume 整体解码会承受超线性复杂度惩罚且无法并行。Triage 的做法是把合并体积按 d×d patch × d 轮切成 slice 图，边界相邻的 slice 之间建立互斥边（空间邻居，同层最多 4 个）：
```
# 一个多 patch parity measurement 的 slice 化（Triage 视角）
for t in 1..T_rounds:                    # 每个 syndrome 测量周期
    for p in merged_patches:             # 合并区内的每个逻辑 patch
        slice S(t,p) → 顶点 V
        对同 patch 的 S(t-1,p) / S(t+1,p) 加时间互斥边（时间邻居）
        对相邻 patch p' 的 S(t,p') 加空间互斥边（lattice surgery 边界）
图 G=(V,E) 二染色 → 偶/奇两个独立集 → 各自并行解码
```
每个 slice 的窗口缓冲（要保留的边界 syndrome 量）由未解析邻居数（degree）决定，直接决定其解码延迟 t_dec=A·volume^α（α=1.17）。这样多量子比特操作从"一个巨大的不可分割解码任务"变成"一组可并行的小任务"，突破时间并行无法拆分多量子比特操作的瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：量子硬件上沿 patch 边界测量联合稳定子（merge/split），编译器（Litinski 风格）把逻辑电路编译成 lattice surgery 操作序列（LLI：multi-patch measurement、patch rotation、idle）并按 tile 布局放置 patch；调度器（如 Triage）决定哪些 slice 同时解码。Web：Watkins et al. 的高性能 surface-code 编译器（Quantum 2024）、LeBlond et al.（ACM TQC 2023）、Hirano & Fujii 的 locality-aware PBC（arXiv:2504.12091）都基于此抽象。Triage 用它做 benchmark 空间维度的并行性来源，并指出空间并行解码 [27] 是它的直接前身之一。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation

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

## 并行窗口解码（Parallel Window Decoding）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
并行窗口解码（Skoric, Browne, Barnes, Gillespie, Campbell, Nature Communications 2023, arXiv:2209.08552）是解决 FTQC 实时解码吞吐瓶颈的协议：把 syndrome 时间流切成固定大小窗口，时间上不相邻的窗口因果独立可并行解码（checkerboard 模式：所有偶窗口并行、再所有奇窗口并行），空间上不同逻辑比特的操作也可分区并行。串行 sliding window 需满足 τ_dec<τ_gen（否则指数级 syndrome 积压，Terhal 论证），且 τ_dec∝N 使任何解码器都存在 code distance 上界；并行窗口通过扩展窗口缓冲（包含邻居边界 syndrome 的 look-ahead 区域）让每个窗口自包含，以更多并行解码器换取吞吐——即使单个解码器慢（τ_dec≥τ_gen）也能维持系统吞吐，把指数退化降为多项式。Web：可与 union-find 或 MWPM 内层解码器结合，数值验证 surface code 无逻辑保真度损失。Triage 论文把其时间维度（time-parallel [24]）与空间维度（[27]）统一进 slice 约束图框架。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
并行窗口解码的输入到输出流程（Triage 视角）：
```
# 输入：连续 syndrome 流（每 d 轮一层）+ M 个解码器
for 时间块 k (checkerboard):                 # 时间维并行
    for 逻辑 patch p:                         # 空间维并行
        w = window(p, k)                       # 窗口 = 时间块内的 slice + 边界缓冲
        buffer(w) = 邻接窗口的边界 syndrome   # 缓冲大小 = 解码 volume，决定延迟
    decode(偶窗口集) in parallel on 解码器池   # 互斥约束内并行
    decode(奇窗口集) in parallel
    窗口边界人工 syndrome 由先解码的窗口产生
# 输出：每窗口的错误链 → 更新 Pauli frame；同步点前需因果锥全部解码
```
Triage 的改进：baseline time-parallel 只在时间维并行、不拆分多量子比特操作（lattice surgery 合并区仍是整块 → 高饱和层 floor）；Triage 以 slice 为原子单元同时并行时间与空间，把合并区切成可并行小窗口，并加资源感知调度（M-for-N 池 + 优先级）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：解码器（pymatching/UF）以窗口为单位跑 MWPM/UF，窗口缓冲由并行度参数与邻居重叠决定；调度器决定窗口→解码器的分配。Triage 用它做三件 baseline：serial sliding window（一次处理一个 lattice surgery 块）、time-parallel window（时间维并行，不拆多量子比特操作）、SWIPER（投机窗口解码，SOTA）。评估参数：解码器数 M 与相对速度 τ_dec/τ_gen 扫描，指标为插入 idle 层数与 LER；Triage 在慢解码器区（τ_dec>τ_gen）仍有效——论文声称"通过调度并行窗口可克服单解码器延迟限制"。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation


## MLA（Multi-Head Latent Attention，多头潜在注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLA 是 DeepSeek 系列（DeepSeek-V2/V3/R1）提出的注意力机制：把每个 token 每层的 K/V 投影压缩进一个低秩 latent 向量（down-projection 得到 c_KV ∈ R^{d_c}，d_c ≪ n_heads×d_head），推理时缓存低维 latent 而非完整 K/V 张量，配合解耦 RoPE（decoupled rotary position encoding，K_R 单独加旋转位置）与 weight absorption 技巧在计算时把 up-projection 吸收进 Q/O 权重。效果：KV cache 尺寸从 2×n_heads×d_head 降到 ≈2×d_c+d_R，与注意力头数解耦——本论文给出的核心论点是"MLA 将 KV cache 大小与 head 数量解耦"，使 R1 能在极长 reasoning 上下文下维持远低于同规模密集模型的 KV footprint（对比 GQA 的 8 KV head 仍随层数线性增长）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MLA 前向（每层，token t）
c_KV = W_DKV @ h_t              # 压缩: h ∈ R^d → latent c_KV ∈ R^{d_c}（缓存 c_KV，而非 K/V 全张量）
k_R  = W_KR @ h_t               # 解耦 RoPE key（d_R 维，带位置编码，单独缓存）
q    = W_Q @ h_t; q_R = W_QR @ h_t
# attention 时用权重吸收后的 W_UQ 展开 latent（KV 不再显式存储/读取全头张量）
score = attention(q, q_R, c_KV, k_R)   # c_KV 经吸收后的上投影参与 QK^T 与 SV
```
Annotations：d_c=低秩 latent 维（DeepSeek 通常 ≈512，远小于 GQA 的 8×128），d_R=解耦 RoPE 维；缓存量 ≈(d_c+d_R) 而非 n_heads×d_head×2；MLA 通过把 KV 压缩进 latent 使 KV 足迹与 head 数无关，是 DeepSeek-R1-671B 长 CoT 推理容量可行的算法根因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DeepSeek 开源实现（DeepSeek-V2/V3 技术报告、FlashMLA kernel 库）；vLLM/SGLang 等 serving 引擎原生支持 MLA 模型。本论文的用法与发现：(1) MLA 是 DeepSeek-R1-671B 维持长上下文推理的架构前提——论文实测 R1 参数为 70B 模型的 10×，但 KV cache 消耗速率反而"适度"（对比密集 70B 的激进容量消耗与提前请求限流），直接支撑"架构级 KV 压缩与硬件容量同等重要"的结论；(2) MLA 与 Pipeline Parallelism 协同：压缩后的 KV 使每 PP stage 可容纳更高 micro-batch 深度、填满 pipeline bubble，这是 R1 在 PP=4+TP=2 下优于纯 TP=8（1663s vs 2047s）的机制之一；(3) 对比 GQA（head 共享、仍线性于层数）与 MHA（无压缩），MLA 是把 KV 与 head 数解耦的算法级容量解药。

涉及论文标题：
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles

## FPMA（Floating-Point Multiplication Approximation，浮点乘法近似）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FPMA 是一种近似浮点乘法技术：基于 Mitchell 对数近似 log2(1+M)≈M（1962），把 IEEE 浮点数的对数值近似为指数与尾数的直接拼接/加法，从而把浮点乘法变成整数加法。具体推导：对 x=(−1)^Sx·2^(Ex−B)·(1+Mx)，有 log2(|x|)=Ex−B+log2(1+Mx)≈Ex−B+Mx；于是乘积 r=x·y 的对数近似为 (Ex+Mx)+(Ey+My)−2B，等价于在拼接的 exponent-mantissa 域上做整数加法 R≈X+Y−B（X=Ex+Mx、Y=Ey+My、R=Er+Mr）。由于消除了乘法器（部分积 O(n²)），FPMA 的硬件成本随位宽近似线性 O(n)，这是它在 UNICORE（S-FPMA）、AxCore（mpFPMA）等加速器中被用作可扩展计算原语的根本原因。局限：对数近似在低比特/含 subnormal 输入时误差大（log2(1+M)≈M 对无前导 1 的 subnormal 不成立），需要 subnormal 归一化与误差补偿配合。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - UNICORE 中 FPMA 乘法的计算过程（W4 例）：操作数先被归一化为内部 E3M2 正常数（无 subnormal）→ 取 X=EW+MW、Y=EA+MA（拼接的 exponent-mantissa 域）→ 整数加法 R=X+Y−B（一次加法即一次"乘法"）→ 双路径补偿：FG 细粒度补偿 C_fg(M_A,M_W)（LUT 预存残差 bit-pattern）拼接到 R 的 LSB 侧恢复低位、CG 粗粒度 1-bit 进位注入修正高位 → 得到近似乘积，转 sign-magnitude 后与部分和累加。示例（FP4 E1M2）：FPMA 对精确积 36 输出 32，CG 补偿无法在 2-bit 尾数粒度下表达修正（结果仍 32），FG 补偿拼接 "01" 后恢复为 36，与全精度乘法一致；FP8（E4M3）例：精确积 66，FPMA 输出 60，CG 补偿调高位到 64、FG 拼接 "01" 恢复 66。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：RTL 中用整数加法器网络替代浮点乘法器（UNICORE 的 S-FPMA 把 FPMA 分解为统一 4-bit 加法 slice、进位链级联成更宽精度；AxCore 用 mpFPMA 支持 W4A16 定宽混合精度）；补偿用小型 LUT（按 (M_A,M_W) 索引）+ 进位注入。使用：量化 LLM GEMM 加速器（AxCore、April、UNICORE）的计算核心，把乘法主导的 GEMM 变成加法主导、位宽可线性扩展的计算；低比特模式必须配合 subnormal 归一化 + FG/CG 补偿（无 FG 时 UNICORE FP4 PPL 崩坏到 1.1E+4–4.9E+6，加补偿后 11.15 与原始 FP4 相同）。开源参考：UNICORE https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## DynFP（Dynamic Floating-Point，分布自适应动态浮点格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DynFP 是 UNICORE 提出的逐 group 可配置的低比特（4-bit/3-bit）浮点格式，定义 DynFP(W_E, W_M, Z, I)：W_E/W_M 为指数/尾数位宽（自适应 E/M 分配，如 E3M0、E2M1、E1M2，优先动态范围或精度），Z 为把冗余负零码重映射为 E3M2 正常域内有价值的值（更细分辨率中间值或更大 outlier 扩展动态范围；若 Z 超 group 格式最大值则符号吸收进 scale），I 为可选的 gap-insertion 空位插入标志（把指数码 E 在位置 ℓ 处拆成 E_hi/E_lo，映射 Φ(1,ℓ,E)=E_hi·2^(ℓ+1)+E_lo，在指数阶梯中插入受控空隙以匹配非均匀分布）。动机：LLM 张量级分布平滑，但 32 元素 group 级分布重尾、非对称、紧聚（图 11），单一静态 FP4 无法表示，静态格式导致精度损失；DynFP 让每 group 选自己的浮点布局。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 数值语义（DynFP 解码伪代码）：
    ```
    if E==0 and M==0 and S==1:  v = Z                       # 负零重映射为有用值
    elif E==0 and M!=0:         v = (-1)^S * 2^(1-B) * M     # subnormal
    else:                       v = (-1)^S * 2^(Phi(I,l,E)-B) * (1+M)  # normal，I-flag 时指数经 Phi 插入空位
    ```
    例：E1M2 布局（2 尾数位）动态范围受限，I-flag 开启时指数码经 Φ 插入 0 bit 扩展覆盖范围；Z 把负零码映射为 0.5 以上 E3M2 正常域值（避免 reintroduce subnormal，Z≥0.5）。权重侧用离线贪心搜索（96 候选 → k=16 palette）选每 group 格式，K/V 侧用 crest factor κ 在线选格式；元数据 = 4-bit（权重）/1-bit（K/V）每 group 格式索引 + 8-bit scale，有效位宽 4.375 bits（比 MXFP4 高 2.9%）。效果：UNICORE-Q 在 4/4/16 各模型 PPL 最低（OPT-6.7B 10.93 vs INT 11.18），zero-shot 平均准确率多数配置最优。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：硬件侧 Unified Format Converter 用 LUT（(格式索引, 数值) 为索引）把 DynFP 编码映射到等效 E3M2 表示，负零经 mux 选 Z，1 cycle 解码；软件侧 artifact Software/Accuracy/（PyTorch，unicore_kernel/quant_utils/ae_scripts）实现格式搜索与量化，运行各表 shell 脚本自动下载模型/数据集。使用：对每个权重/K/V group 选择最优 E/M 布局 + Z + I-flag，使低比特浮点表示匹配 LLM 重尾/非均匀分布；离线权重量化无需激活校准（Llama-2-7B 单 RTX 6000 Ada 约 2 分钟），在线 K/V 量化用 κ 阈值映射（<0.2% QKᵀ FLOPs）。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## 双路径误差补偿（Fine-Grained + Coarse-Grained FPMA Compensation，FG/CG 补偿）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 双路径误差补偿是 UNICORE 恢复 FPMA 近似乘法精度的机制，包含两条互补路径：(1) 粗粒度补偿（CG，Coarse-Grained）——沿用 April 等早期 FPMA 设计的思想，把下采样/近邻尾数组合的误差以 1-bit 形式注入 FPMA 结果的 mantissa 域（作为最低加法 slice 的 carry-in）；(2) 细粒度补偿（FG，Fine-Grained）——由于乘法与 FPMA 结果对给定尾数对 (M_A,M_W) 都是确定的，可在离线扩展精度域预计算残差，把残差的低位部分编码为短 bit-pattern 存入小型 LUT，运行时把 C_fg(M_A,M_W) 拼接到 FPMA 结果的 LSB 侧，等效扩展有效尾数宽度。关键发现：CG 单独在 2-bit 尾数（FP4 E1M2）粒度下无法表达修正（误差量级太小无法触发最后一位），必须由 FG 拼接恢复低位。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 计算例子（图 5）：FP8（E4M3）——精确积 66，FPMA 低估为 60；CG 1-bit 补偿把高位调到 64，FG 拼接低位 "01" 恢复 66（与全精度一致）。FP4（E1M2）——精确积 36，FPMA 输出 32；CG 无法改变结果（仍 32，误差 < 2-bit 尾数粒度），FG 拼接 "01" 后得 36。消融（Table III）：FP16 仅 CG 即近无损（11.02→UNICORE 10.98 vs FP16 10.88）；FP8 CG 不足（11.02 vs FP8 10.98），加 FG 后 10.98 与 FP8 一致；FP4 无 FG 时 PPL 崩坏到 1.1E+4–4.9E+6，FG+CG 后 11.15 与原始 FP4 完全相同。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：FG 用小型 LUT（按 (M_A,M_W) 索引的短 bit-pattern），CG 用 1-bit 进位注入（W8 融合模式利用 B 的空闲位参与粗粒度尾数加法）；FP16/FP8 等高位宽只需 CG 或 CG+FG，FP4/FP3 低比特必须 FG+CG。使用：随精度自适应——补偿级别由格式数值特征决定，使 UNICORE 在所有支持位宽（FP4/FP8/FP16）保持与对应全精度乘法一致或近一致的模型精度。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## Subnormal 归一化与位宽扩展（Subnormal Removal via Bit-Width Expansion）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Subnormal（非规格化数）是 IEEE-754 中指数码为 0、尾数无隐藏前导 1 的浮点数：v=(−1)^S·2^(1−B)·(0+M)，用于向零平滑下溢。问题：subnormal 不满足 FPMA 的对数近似 log2(1+M)≈M（因为无前导 1），且低比特格式（FP4/FP3）中 25–50% 的可表示值是 subnormal，导致 FPMA 产生系统性大误差。UNICORE 的解法是位宽扩展（Bit-Width Expansion）：把低比特浮点操作数无损转换到更宽的内部格式（E3M2），使每个可表示值都成为正常数——对 subnormal 尾数左移直到落入 [1,2)，指数同步减小；要求目标格式尾数位 M'≥M 且指数范围 B'≥B+M（足以吸收 M 次归一化移位）。与 AxCore 的近似重映射（把 subnormal 映射到附近正常值、引入额外噪声）不同，这是精确变换。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 归一化流程（图 4，FP4 → E3M2）：任意 FP4 变体（E3M0/E2M1/E1M2）的 subnormal 形如 (0+M)，按尾数左移位数 s 归一化为 1.M'（M' 为 M 左移 s 位），指数 E'=E−s；E3M2 提供 2 个额外尾数位与更大指数偏置，保证归一化永不再下溢、全部有效位保留——FP4 的整个 subnormal 区域（最小到 E1M2 的 0.5）都落在 E3M2 正常域内。伪代码：
    ```
    if E==0 and M!=0:      # subnormal
        s = leading_zeros_normalize(M)   # 尾数左移 s 位至 [1,2)
        E' = 1 - B + s;  M' = M << s      # 在 E3M2 中成为正常数
    else:                  # normal 直接映射
        E' = E; M' = M
    ```
  - 关键设计：所有权重/激活仍以原始低比特格式存储传输，E3M2 扩展是仅存在于计算数据通路内的临时精确重编码（保留低比特存储带宽收益，计算位宽略有扩展但 FPMA 加法特性使其开销极小）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Unified Format Converter 内嵌归一化逻辑（尾数左移 + 指数补偿），与 DynFP 解码共用 LUT 通路；正常数直接映射、subnormal 左移归一化。使用：在 FPMA 计算前保证所有操作数为正常数，使 Mitchell 对数近似假设成立，消除低比特模式下的系统性误差；是 UNICORE 精度保持管线的第一级，之后接 FG/CG 双路径补偿。通用背景：IEEE-754 的 subnormal 处理也见于 FP8/FP4 格式研究（如 E2M1 变体的 subnormal 区间），vault 笔记 knowledge_notes/硬件知识笔记/Supernormal Support (SR_SP for Low-Precision Formats).md 与 E2M1 (FP4 Format).md 有低比特 subnormal 编码的相关讨论。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## Crest Factor（峰值因数，在线 K/V 格式选择代理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Crest Factor（峰值因数）κ 是信号处理中峰值与 RMS 之比（κ=max|x|/RMS(x)），衡量波形/分布尖峰程度。UNICORE 用它作为在线 K/V 激活量化时选择 DynFP 格式的轻量代理：不同 DynFP 格式（不同 E/M 布局）的量化信噪比（QSNR）随 κ 呈不同行为，可预计算一组阈值把 κ 映射到最合适的 E/M 布局。动机：K/V 激活在线量化（权重是离线搜索），逐 group 穷举全部 DynFP 候选代价过高，需要免穷举的轻量选择方法。κ 计算只需单趟 max-abs 归约 + RMS 归约（每元素 4 次标量运算），对 L≥2K 序列占 QKᵀ FLOPs 不足 0.2%，且完全 memory-bound，可无缝融合进量化 kernel。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在线 K/V 量化流程：
    ```
    # 对每个 K/V group 激活张量 g
    max_abs = reduction_max(|g|)      # 一次 max-abs 归约
    rms     = reduction_rms(g)        # 一次 RMS 归约
    kappa   = max_abs / rms           # 峰值因数
    layout  = threshold_map(kappa)    # 预计算阈值查表 → 最优 E/M 布局
    q_g     = dynfp_quantize(g, layout)  # 按选定布局量化
    ```
    例：κ 高（分布尖峰/长尾）→ 选更大指数位布局（如 E2M1）保动态范围；κ 低（分布平坦/紧聚）→ 选更多尾数位布局（如 E1M2）保精度。对比离线权重侧用贪心搜索选格式（可穷举 96 候选），在线侧用 κ 阈值映射避免逐 group 评估候选。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：量化 kernel 内融合（max-abs + RMS 归约，每元素 4 次标量运算：reduction、sqrt、除法），阈值表离线预计算；UNICORE 的 CF 计算使量化 kernel arithmetic intensity 从 0.63 升到 0.87，仍 memory-bound 无可见开销（Llama-2-7B 激活量化占 prefill 时延 7.1%–20.7%、decode 0.3%–1.6%，且与 GEMM 大部分重叠）。使用：K/V 与 softmax 输出 P 用与激活相同 group size/位宽的量化，仅 K、V 做在线格式选择；是 DynFP 在 K/V cache 上的免校准在线落地方式。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## 贪心调色板格式搜索（Greedy Palette Format Search，自动化权重量化格式搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 贪心调色板格式搜索是 UNICORE 的离线权重量化方法：为每个权重张量构造一个大小为 k（如 16）的紧凑 DynFP 格式调色板（palette），再让每个 group 从调色板中选择局部误差最小的格式。动机：单张量各 group 可能需要多种 DynFP 配置，搜索空间巨大（多个 E/M 布局 × gap-insertion 变体 × 众多 Z 候选），穷举或人工探索不可行；全局每 group 自由选 96 个候选会导致元数据与搜索时间过大。该方法数据驱动、无需激活校准、不引入分布偏置，作为一次性离线步骤（Llama-2-7B 单张 RTX 6000 Ada 约 2 分钟），推理时只用存储的格式索引与 scale。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 算法（DynFP-4 例，三阶段）：
    ```
    # 阶段1 候选池生成：枚举所有可行 DynFP-4 配置 → 96 个候选
    #   = 4 个基础 E/M 布局 (E3M0, E2M1, E1M2, E1M2I) × 24 个 Z 赋值
    #   Z 限制在内部 E3M2 正常域内 (Z >= 0.5) 避免 reintroduce subnormal
    # 阶段2 迭代贪心构造 palette P（容量 k）:
    P = {}
    for t in 1..k:
        if t == 1: f_t = argmin_f global_MSE(f)          # 初始化选全局 MSE 最小的格式
        else:      f_t = argmax_f marginal_MSE_reduction(P ∪ {f_t})  # 每轮选使全局 MSE 边际下降最大的候选
        P = P ∪ {f_t}
    # 阶段3 最终分配：每个 group 从 P 中选局部量化误差最小的格式
    for group g: idx_g = argmin_{f in P} local_MSE(g, f)
        # 存 4-bit 格式索引（16-entry palette）+ 8-bit scale；Z 加载进 Unified Format Converter
    ```
  - 行为类似在表示空间对权重聚类；量化 Llama-2-7B 约 2 分钟/checkpoint，推理期无运行时格式搜索。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：软件侧（PyTorch，artifact Software/Accuracy/ 的 quant_utils）实现候选池枚举、全局 MSE 贪心迭代、逐 group 分配；Z 值在计算新权重张量时加载进硬件 Unified Format Converter；元数据随权重存主存。使用：作为 UNICORE-Q（启用分布自适应 DynFP 量化）的权重量化路径，与在线 crest factor K/V 选择互补；评估显示 UNICORE-Q 在 4/4/16 各模型 PPL 最低、zero-shot 平均准确率多数配置最优（DynFP 增益明显）。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## qLDPC 稳定子测量电路（Quantum LDPC Stabilizer Measurement Circuit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- qLDPC（quantum low-density parity-check）码是量子纠错码（QEC）的一类，其校验矩阵稀疏，编码率远高于表面码（surface code），正从理论研究走向实验 FTQC 核心（编码效率优势使其成为超导平台上表面码之后的候选）。qLDPC 稳定子测量电路（stabilizer measurement circuit）是把 qLDPC 码的稳定子（stabilizer）测量实现为门电路的子程序：每个稳定子测量通常由若干 CNOT（把数据 qubit 与 ancilla qubit 纠缠）+ 测量 + 重置组成，其纠缠门作用在码的 Tanner 图上。由于 qLDPC 码常有长程交互（long-range interactions），在固定局域连接的超导处理器上执行时会产生显著路由开销，这正是 CANOPUS 研究的 FTQC 场景。
- 论文使用的码类型：generalized bicycle（GB）与 bivariate bicycle（BB）码（取自 [53][67]），在 2D heavy-hex 与 square 拓扑上编译稳定子测量电路；评估用标准 memory experiment 模拟。CX-iSWAP 组合 ISA 与此场景契合：稳定子测量中有大量 CNOT，把 SWAP 插入 piggyback 到 CX 上（复合等价 iSWAP）可零额外 2Q 门数完成路由。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 端到端评估 pipeline（CANOPUS 论文 V-B）：① 输入 qLDPC 码（GB/BB 码的校验矩阵）；② 生成稳定子测量逻辑电路（CX 或 CX-iSWAP ISA 表达的 CNOT 网络）；③ 编译/路由：SABRE 或 CANOPUS 把逻辑电路映射到 2D heavy-hex/square 耦合图，输出含 SWAP 的物理电路；④ 用 stim（https://github.com/quantumlib/Stim，Google 的高速稳定子电路模拟器）按文献[6]的电路级噪声模型模拟标准 memory experiment（大量重复的编码/纠错周期，每次测量生成 syndrome）；⑤ 所有 syndrome 用 BP-OSD decoder（belief propagation + ordered statistics decoding，文献[28][53]）解码，得到逻辑错误率（logical error rate）。
- 伪代码（评估逻辑）：
  ```
  for code in {GB, BB}:
      for isa in {CX, CX-iSWAP}:
          circ_logical = build_stabilizer_circuit(code, isa)   # 稳定子测量 CNOT 网络
          for compiler in {SABRE, CANOPUS}:
              circ_phys = compiler.route(circ_logical, coupling_map)  # 插入 SWAP
              pL = stim_memory_experiment(circ_phys, noise_model)     # stim 模拟
              pL = bp_osd_decode(syndromes)                            # BP-OSD 解码
  ```
- 结果（Fig.10）：CANOPUS vs SABRE 的逻辑错误率抑制——CX ISA 下 square 49.4%、heavy-hex 11.4%；CX-iSWAP 下 square 52.6%、heavy-hex 29.3%。差异来源：CANOPUS 编译出的电路 CX/iSWAP 门数与深度更少，尤其 CX-iSWAP 下大量 SWAP 被 CX 吸收。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：stim（Python/C++，Google Quantum AI 开源）是稳定子电路与噪声仿真的行业标准工具（含 Clifford tableau 模拟、syndrome 采样、`detector`/`observable` 定义）；BP-OSD 解码器（Roffe 等，开源实现如 ldpc 库）用于从 syndrome 估计逻辑错误。CANOPUS 实验套件在 ./experiments/eval_qldpc/（Makefile 中 make 准备、make canopus/baselines 运行）。
- 场景意义：qLDPC 稳定子测量电路是"ISA-aware 路由直接提升容错性能"的典型例子——路由开销（SWAP 数）转化为更多噪声门 → 更高逻辑错误率；CANOPUS 把 ISA 合成能力用于路由，直接压低 FTQC 逻辑错误率，是"NISQ 与 FTQC 双场景"的桥接验证。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation

## QFT Kernel（量子傅里叶变换核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量子傅里叶变换（Quantum Fourier Transform, QFT）是 Shor 算法、量子相位估计（QPE）等核心算法的基本子程序：对 n-qubit 态施加 QFT 需要 O(n²) 个受控相位门（CPhase，每个作用于一个 qubit 对），其结构是"蝴蝶式"（butterfly）的相位阶梯。QFT 的 2Q 门全连接模式（任意 qubit 对都可能出现 CPhase）使它在受限拓扑上路由开销显著，是 qubit routing 的经典 benchmark 与"深度最优性"研究对象（Maslov 手工最优方案、TOQM 的 A* 深度最优等）。
- CANOPUS 论文把它作为第一个 case study：证明 n-qubit QFT 在 1D chain 上的最小 SWAP 插入数为 n(n−1)/2 − 2（比 CPhase 数少 2），形成完美的对称蝴蝶结构（Fig.8(b)）；CANOPUS 在 1D chain 上对所有规模达到该理论最优，超越 TOQM（声称实现 Maslov 方案却失败）与 Maslov 手工方案（多 2 个 SWAP）——该结论与目标 ISA 无关。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- QFT 核结构（n=6 示例，Table I）：qft_6 路由前 #Can=15（CPhase 数）、Depth2Q 与最优路由后同为 15/9；qft_12 为 66/21（CANOPUS 在 1D chain 上达到最优，TOQM 为 67/22）。QFT 编译 pipeline：① 生成 QFT 逻辑电路（CPhase 序列，每层一个控制 qubit）；② 逻辑级优化（TKET）+ rebase 为 {Can,U3}；③ 路由（CANOPUS/TOQM 等）→ 物理电路（#Can、Depth2Q 指标）；④ 真机执行（可选）。
- 真机验证例子（V-A）：在 IBM ibm_marrakesh（Heron-R2 QPU，native 门 {CZ, √X, Z(θ), ZZ(θ)}，heavy-hex 拓扑但含足够 1D chain）上编译并执行 n∈{6,8,10,12} 的 QFT，用 Hellinger fidelity 测量（实验 vs 理想输出分布；shots = MAX{4096, 2^n×10}；每个电路追加一层 Hadamard 使理想终态为 |0>^⊗n）。CANOPUS vs QISKIT 默认编译：CZ 门数降 52.9%、2Q 深度降 66.4%、CZ/CX 门集错误降 26.89%、ZZ(θ) 门集错误降 34.98%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：QFT 在 Qiskit 中可用 `qiskit.circuit.library.QFT` 生成；CANOPUS 仓库 `python route_qft.py <n>` 直接对比 SABRE 与 CANOPUS 的 n-qubit QFT 路由结果（#Can 与 2Q 深度）。作为 benchmark，QFT 出现在 QASMBench/MQTBench（论文 Table III 的 qft 18 qubit：#Can 153、Depth2Q 33、Ccount 306）。
- 场景意义：QFT 是"程序模式-ISA-拓扑协同"的样板——子程序展开式（subroutine-unrolling）构造的算法天然适配 chain 拓扑（heavy-hex 反而更高开销）；其 CPhase 全连接模式为 ISA-aware SWAP absorption 与交换性优化提供丰富机会，也是跨编译器（CANOPUS/TOQM/QISKIT）公平比较的黄金 benchmark。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation

## Bootstrapping Key（BSK，自举密钥）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSK 是 TFHE 可编程自举（PBS/盲旋转）的密钥材料：把秘密钥 s 的每个分量（n 个）加密成 GGSW（TRGSW）密文组成的大密钥集合，用于在密文域执行"解密-再加密"式旋转。每个 BSK_i 是 GGSW 密文 = L×(k+1)×(k+1) 多项式矩阵（每个元素为 N 阶多项式；L 为 gadget 分解层数、k 为 GLWE 维数、N 为多项式度）。BSK 总量约 O(n·N·L) 多项式系数：TFHE 为 10s–100s MB 量级（CASCADE 参数集 III：112 MB、参数集 IV：90 MB），远小于 CKKS 的 GB 级密钥。关键复用性质：BSK 是 BSP 的参数、可在同参数下跨多次 BSP 执行复用，但不能在同一 BSP 内跨 HMUX_i 复用（每个 HMUX 需要唯一的 BSK_i）——这一"每 HMUX 唯一 + 跨 BSP 复用"的组合决定了它的存储/带宽策略（可常驻内存，但并发访问量大）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- BSK 在自举 pipeline 中的使用（CASCADE Algorithm 1，参数集 I：n=500、N=1024、L=2、k=1）：
```
for i in 1..n:                       # 盲旋转 = n 次 HMUX 迭代
    BSK_i = (X^(-a_i) - 1) * BSK_i   # 对 BSK_i 做旋转/多项式减法
    ACC_i = BSK_i ⊡ ACC_{i-1}        # 外积：L×(k+1)×(k+1) GGSW 矩阵 × (k+1) RLWE 向量
```
- Annotations：每个 HMUX_i 从 BSK 集合中取唯一 BSK_i（n 个 BSK 各用一次）；外积经 FFT 变逐系数乘后由 VMA 单元执行；BSK 以 GGSW 多项式矩阵驻留（CASCADE 中 126 MB 分布式 SRAM）；流水线并行时 n 个 HMUX 需并发访问 n 个不同 BSK——这就是"并发 BSK 访问带宽"问题（集中式 HBM 无法支撑，CASCADE 用 BSK-distributed 把访问局限在各 chiplet 本地）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 生成：密钥生成时把秘密钥位用 gadget 加密为 GGSW 密文，做一次同态"密钥位查询"（bit extraction）；软件库（TFHE-rs/Concrete）在 keygen 阶段生成并常驻内存。硬件使用：片上/片外驻留策略决定带宽——Morphling 用 HBM+片上 buffer（BSK 复用依赖 batching）；MNEMOS 在 GPU 上把 BSK 分块（TBSK）到共享内存跨 batch 复用（见"LWE/GLWE/GGSW 密文与外部乘积"条目）；CASCADE 把全部 BSK 驻留分布式 SRAM（每 HC 10.5 MB BSK buffer、共 126 MB），消除 BSK 片外搬运，并把并发访问分散/限制在各 chiplet 内（BSK-stationary 数据流）。设计要点：BSK 容量（决定可驻留的最大安全参数）与并发访问带宽（决定可支撑的流水线并行度）是两个相互独立的架构约束。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator

## 事件驱动 SNN 计算管线（Event-Driven SNN Processing Pipeline：Axon.in→Synapse→Dendrite→Soma→Axon.out）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
事件驱动 SNN 计算管线是脉冲神经网络（SNN）在大规模神经形态平台上的基本计算模型：与 ANN 的逐层稠密前向不同，SNN 在离散时间步（1ms 生物学仿真步）内处理稀疏 spike 事件，每个时间步执行一条连续的事件驱动流水线（WaferBRAIN 论文 Fig.2a）：**Axon.in** 汇集输入事件 → **Synapse** 做稀疏权重查找与衰减（memory-bound：每个 spike 触发对其目标突触集合的访问）→ **Dendrite** 通过分段归约（segmented-reduce）累加突触后电流 → **Soma** 积分膜电位状态并判断阈值发放（compute-bound）→ **Axon.out** 通过本地广播与边界触发散射发出 spike，同时 NoC 路由包（communication-bound）。全脑规模（10^11 神经元、10^14 突触）下每步执行 trillions 级事件操作，且 compute/memory/communication 三端异质瓶颈并存——这决定了神经形态硬件必须在存储层次、互连与通信范式上协同设计（WaferBRAIN 的 NAHP + 3D-WSI + switchless dragonfly 正是为此）。生物学实时约束：人脑平均发放率 15-30Hz、1ms 时间步对应 ~3% 每步发放活动，系统须在 1ms step 预算内完成全部事件处理（WaferBRAIN 以"1ms 红线"评估可持续 firing rate）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
事件驱动管线与稀疏 spike 传播的计算过程（一个时间步，WaferBRAIN 建模；spike 传播本身可视为稀疏矩阵-稀疏向量乘 SpMSpV——见 kernel 层 SpMSpV 条目）：
```
# 每个时间步，对每个发放神经元 n_i（发放率 λ，fanout F=256）：
# 1) Axon.in：汇集本节点收到的 spike 事件（AER 包）
# 2) Synapse（memory-bound）：按 spike 的目标神经元/轴突 ID 查稀疏突触表
#    → 取 (DstNeuron, Weight) 邻接表（本地 LNid 索引 或 全局 GAid 索引）
# 3) Dendrite：segmented-reduce 按 DstNeuron 分组累加突触电流 I_syn[ds] += w
# 4) Soma（compute-bound）：膜电位 V[ds] = V[ds]·decay + I_syn[ds]
#    若 V[ds] > V_th → 发放 spike，置 refractory
# 5) Axon.out（communication-bound）：打包 spike →
#    本地目标：mode=0 广播 LNid（区域内路径复用）
#    全局目标：边界 owner 发 mode=1 单播 <POD,Wafer,Node>+GAid
# 6) 路由/传播时延计入 1ms 步预算，最拥塞路由器服务时间 T_max/Θ_router + 最长路径 δ_max 决定步时延
```
具体例子（100B 全脑模型）：95% 突触为区域内稠密连接（广播）、5% 为跨区域长程投射（单播）；每神经元每步 256 个目标（fanout），其中 ~243 本地（广播 1 次覆盖）、~13 跨区域（单播到 5 个随机区域）。评价指标：每节点路由负载 R（packet/s）、流量 T（bit/s）、每步通信时延（对比 1ms）、可持续 firing rate。对照实验：14,012 GPU 的神经仿真慢 65-118.8×、82,944 CPU 节点慢 578×——算法管线本身（LIF 等）不构成瓶颈，事件吞吐由 memory/communication 端决定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：算法侧 = SNN 模型（LIF/IF 等神经元模型、突触权重与延迟、区域化连通矩阵）+ 每步事件驱动调度（只有发放神经元触发计算，无发放则跳过）；硬件侧 = BPU 四模块流水（axon-in 查表 / dendrite DMA+分段归约 / soma 膜电位 / axon-out 打包）+ SRAM 热状态 + 3D-DRAM 突触邻接表 + NoC 广播/单播（见硬件架构库 BPU/NAHP 条目）。使用方式：生物实时交互与闭环研究（dynamic clamp、电刺激）要求 1ms 步预算内的端到端事件处理；系统设计目标 = 在固定实时步预算下最大化可仿真神经规模（容量）。WaferBRAIN 的评估：1B（单 wafer）/16B 皮层（4×4 mesh 或 14×1 dragonfly）/100B 全脑（10×10 mesh 或 14×7 dragonfly）三模型、每节点 1.30-1.49M 神经元、区域 8×8 节点网格（83-95M 神经元/区域）、firing rate 0.1%-4.9% 扫描；NAHP 在 100B dragonfly 下 1ms 界内可持续 3.8% firing rate，超过人脑 15-30Hz 对应的 ~3% 阈值。

涉及论文标题：
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration

## 碰撞测试（Collision Test，生日悖论检测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
碰撞测试是基于生日悖论（birthday paradox）的 RNG 弱点发现测试，用于检测 RNG 输出分布的非均匀性（熵下降）。直觉：一个输出 n 位均匀随机数的 RNG，理论上需要 2^n+1 个样本才保证出现重复（鸽巢原理），但由于生日悖论，出现至少一次碰撞（两个 n-bit 序列相同）的概率在远小于此的样本数时就超过 0.5。对 n=16：k≈√(2·2^n·ln(1/(1-p)))——p=0.5 时约 302 个样本、p>0.99 时约 777 个样本（远低于直觉的 65,537）。当 RNG 输出分布非均匀（熵下降）时，实测碰撞数会偏离均匀随机下的理论期望，因此碰撞偏差是 RNG 安全性退化的灵敏指示器。本论文用其检测环境压力（温度/电压）是否造成碰撞数与期望值的偏差及偏差模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
碰撞测试计算流程：
```
理论期望碰撞数:  E = C(k,2) · (1/2^n)     # C(k,2)=k(k-1)/2 为样本对总数
# k 个 n-bit 样本中任意一对相同的期望对数
```
µRNG 应用流程：① 对每个环境 corner（如 -68°C~+85°C × 电压 1.8V~3.3V）收集 k=2^16 个 16-bit RNG 输出样本（本论文用每次电源循环的首输出高 16 bit 拼接）；② 统计实测碰撞数；③ 计算与期望碰撞数 E 的百分比偏差；④ 跨 corner 比较偏差模式。论文实测（SAM L10/L11）：温度升高碰撞数增加、电压成反比，且温度-碰撞偏差呈指数关系——+85°C 3.3V 时最高 4.2% 偏差，-68°C 1.8V 时最低 -1% 偏差；此行为与环形振荡器熵源特性吻合。伪代码：
```
samples = collect_nbit_outputs(corner, k=2^16)   # 每 corner 采集
observed = count_collisions(samples)             # 哈希表统计重复
expected = comb(k, 2) / 2^n                      # 期望碰撞数
deviation = (observed - expected) / expected     # 偏差百分比
report(corner, deviation)                        # 跨 corner 比较
```
Annotations：n=16（论文分析 16 个最高有效位以匹配 UlSWaP NVM 存储容量 128KB，且最高有效位对 RO 类熵源更敏感）；k=2^16 个样本；collisions 用空间换时间（哈希表 O(k)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：对收集的位流按 n-bit 块分组，用哈希表/排序统计重复块对数，与理论期望对比；可复用 NIST/TestU01 的某些测试思路，但本论文把它作为独立弱点发现测试并跨环境 corner 运行。使用场景：在 NIST 全过的基础上进一步检测"统计上合格但存在可被强力攻击者利用的弱相关"的设备（论文明确：此阶段质量不再是二元 pass/fail，而是寻找环境-输出相关性）。结果解读：SAM L10/L11 温度-碰撞指数关系使其被归为 Class 2（名义间歇可用，极端环境不安全）；MSPM0 L-Series 全 corner 无偏差归为 Class 5。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices

## Moran's I（空间自相关，Spatial Autocorrelation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Moran's I 是度量空间数据自相关的统计量，量化空间数据的聚类/分散程度，取值 [-1,1]：-1 完全分散、0 空间随机、1 完全聚类。定义：
$$I = \frac{N}{W} \cdot \frac{\sum_{i=1}^{N}\sum_{j=1}^{N} w_{ij}(x_i-\bar{x})(x_j-\bar{x})}{\sum_{i=1}^{N}(x_i-\bar{x})^2}$$
其中 N 为元素数、w_ij 为 i/j 的空间权重（论文用共享边界与 k 近邻计算）、W=ΣΣw_ij、x̄ 为均值。本论文创新性地把 Moran's I 应用到 RNG 评估：把 RNG 输出组织成位图（bitmap），用 Moran's I 检测位图上数据的聚类/条带化——真随机位图的期望 Moran's I=0（无空间相关），非零值暴露输出中的空间结构（如 SRAM 阵列布局不对称导致的偏置条带）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
µRNG 应用流程（SRAM 熵源评估）：① 把 4KB SRAM 上电态组织为位图（每个 cell 一个像素：0/1/不稳定）；② 对每个环境 corner（温度 × 压摆率）计算 Moran's I；③ 比较聚类程度。关键实测：名义条件 Moran's I=0.032（基本空间随机，SRAM 可行熵源）；+85°C 慢电压爬坡时 Moran's I=0.127（明显条带化——布局不对称偏置在慢爬坡下暴露）；低温慢爬坡 Moran's I=0.015（数据保持竞争）。伪代码：
```
for corner in corners:                       # 温度 x 压摆率
    bitmap = sram_powerup_state(4KB, corner) # 上电态位图
    w = spatial_weights(bitmap)              # 共享边界 + k 近邻
    I = morans_I(bitmap, w)                  # 公式(5)
    report(corner, I, per_bit_entropy)
```
Annotations：x_i 为第 i 个 cell 的位值；w_ij 刻画 cell 邻接关系；慢压摆率下单元偏向布局决定态形成条带 → Moran's I 显著非零；低温下数据保持使熵崩溃（每 bit 熵 0.004 @-68°C）而条带不明显。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：计算邻接权重矩阵（网格图共享边 + k 近邻）、按公式算 I；成熟实现见于空间统计库（R spdep::moran、Julia SpatialDependence.jl、Python libpysal 等，可用于置换检验/蒙特卡洛 p 值）。本论文把它与碰撞测试、熵估计一起作为 NIST 之外的弱点发现测试。使用场景：检测 RNG/熵源输出中统计测试发现不了的"空间结构"（布局偏置、聚类）；尤其适合 SRAM/位图类熵源。结论：SRAM 熵源受慢压摆率结构印记与低温数据保持两个攻击者可控的不安全源影响，归为 Class 2。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices
