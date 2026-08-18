## Codeword Merge（码字合并）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Codeword Merge（码字合并）是 HBM-CASO 提出的片上 ECC 资源压缩技术：利用 RS 码的线性性质，把两个覆盖同一数据区域的小码字（local RS(18,16)）的校验符号线性合并成一个覆盖更大区域的大码字（regional RS(34,32)）的校验符号，从而在不重新编码原始数据的前提下把奇偶校验空间减半。数学上：p_regional_0 = p_local_0 + C * p_local_1，其中 C 是 H 矩阵导出的 GF(2^8) 常数；等价地，两个 local H 矩阵线性组合出 regional H 矩阵的对应行（Vandermonde 结构保证 H_Ri = [H_Li, α^(18i)·H_Li]）。逻辑链：现代 HBM 把 8B ECC parity 空间（每 64B 数据块）全部分配给 ODECC → 想让处理器用更强的 SysECC 需要空间 → 两个 local 码字覆盖 32B 区域却要 4B parity（RS(18,16) 各 2B），而一个 regional RS(34,32) 覆盖同样 32B 只需 2B parity → 合并后省出 2B 给 SysECC。同样原理推广到 16-bit 符号（p_regional_0 = p_local_0L + α^8·p_local_0H 折叠 16-bit parity 为 8-bit）与其它线性码（Hamming SEC(71,64)→SEC(136,128) 用 p_c,i = p_a,i ⊕ p_b,i + 额外整体奇偶位；residue 码 p_c = (p_a + C·p_b) mod m，C = 2^64 mod m）。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Merging Unit 是 HBM-CASO 在 HBM 片内新增的轻量逻辑单元（Figure 3(b)），与 ODECC 编码器串联。运转流程（每 64B 写）：数据分两个 16B 块 → 既有 ODECC 分别生成两个 RS(18,16) local parity（各 2B，共 4B）→ Merging Unit 对这两个 parity 做 GF(2^8) 常数乘加得到 2B regional parity（RS(34,32) 级别）→ 2B 存入 parity 空间、另 2B 空间腾给 global SysECC parity（经原 CRC 通道从控制器传来）→ 存储时同时存 regional ODECC parity 与 global SysECC parity。硬件约束：(1) 合并不跨 pseudo-channel（p_local_0/p_local_1 来自同一 32B 区域），避免跨 PC 物理对齐问题；(2) 最重要的 plocal 生成由既有 ODECC 完成，Merging Unit 只做常数乘加，故仅 +61 cells/+17μm²（较 baseline ODECC 4027 cells/1113μm²，逻辑开销 1.51%）；(3) 读路径复用同一单元做 error detection（重新生成 regional parity 与存储 parity 比较），即"编码逻辑复用为检测"，无需完整 RS 解码器。16-bit symbol 兼容：每个 16-bit parity 拆高低字节经 α^8 加权折叠成 8-bit，形成 regional RS(34,32)。可扩展：未来 ODECC 升级到 regional 级别时，同一合并策略可把 regional 合成 global（4B RS(68,64) + 额外 4B CRC）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GF(2^8) 常数乘法器 + 加法器（bitwise XOR），配置为小有限域运算（比整数乘法简单得多）；编码侧在 ODECC 输出端、检测侧在 ODECC 输入端复用同一套逻辑。使用要点：(1) 必须依赖码的线性性质（RS/Hamming/residue 均满足），非线性码无法合并；(2) 合并后的 regional 码单独看比原 local 码弱（2B 对 4B parity），但覆盖范围相同且配合 SysECC 后总保护增强——论文评估显示 R-mode/G-mode 在全部测试场景无可观察 SDC；(3) CRC parity 可选存于 metadata 空间：存时写路径 CRC 单元反向工作（生成 parity 填 metadata）、读路径做检查。论文未明确说明 Merging Unit 的 RTL 结构细节。

涉及论文标题：
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
