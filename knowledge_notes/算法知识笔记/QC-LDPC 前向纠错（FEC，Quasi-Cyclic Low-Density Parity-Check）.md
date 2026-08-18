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
