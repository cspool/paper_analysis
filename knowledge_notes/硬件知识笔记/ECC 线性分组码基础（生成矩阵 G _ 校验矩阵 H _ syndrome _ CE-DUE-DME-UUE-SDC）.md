## ECC 线性分组码基础（生成矩阵 G / 校验矩阵 H / syndrome / CE-DUE-DME-UUE-SDC）

术语解释
内存可靠性编码的数学基础：线性分组码用矩阵描述编解码，syndrome 只依赖错误模式，译码结果按可纠/可检/误纠/漏检四类划分，其中误纠与漏检构成 SDC。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ECC 保证数据在物理故障存在下可靠存储与传输。内存系统普遍采用线性分组码（延迟低、硬件简单）：一个 (n, k) 码把 k 位消息 m 映射为 n 位码字 c，附加 r = n − k 位冗余。码由生成矩阵 G（c = mG）与校验矩阵 H（Hc^T = 0）成对描述。译码逻辑链：接收 y = c + e（e 为错误向量）→ 计算 syndrome s = Hy^T = He^T（s 只依赖错误模式、与码字无关）→ 由 s 估计错误 ê → 恢复 ĉ = y − ê。译码结果四分类（本文 Background 定义）：DCE（可检可纠）、DUE（可检不可纠，s 在纠错半径外）、DME（译码器宣告成功却输出错误码字）、UUE（s=0 却已损坏、不可检不可纠）——后两类即 SDC（Silent Data Corruption，静默数据损坏），是最严重的可靠性失败。本文用 GF(2^16) 符号化 H 矩阵：把 16 列一组映射为 GF(2^16) 元素，使符号级条件（SSC/DEC）可在有限域上验证。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 ECC 硬件中，G/H 矩阵直接对应门级结构：编码器 = G 的 XOR 网络（本文编码器为 8 级 XOR tree）；解码器 = syndrome 生成（H 的 XOR tree）+ 错误定位与纠正。运转流程（Cerberus 读路径）：288b 码字进控制器 Decoder3 → 与 H^S-ECC 相乘得 32-bit syndrome → s=0 则单周期通过（error-free 快速路径，不增加 tCL）→ s≠0 时 SSC corrector 与 DEC corrector 并行在单周期内定位错误并翻转错误位。H 矩阵的构造条件即硬件纠错能力：SEC 需列唯一非零、DED 需任意两列和 ≠ 其他列、bounded-fault 需区域内列和 ≠ 区域外列、CRC8 需任意 8 连续列线性无关、SSC 需符号对齐列和唯一、DEC 需任意两列和唯一——本文 Cerberus 的 H2 与 H^S-ECC 分别按这些条件构造。四分类在硬件上的含义：DCE 正常纠正；DUE 触发重试/上报；DME/UUE 是必须用码设计（bounded-fault、检测概率）压制的 SDC 来源。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：二进制线性分组码常用奇数权重列构造（Hsiao 类）保证 SEC-DED；符号码（RS 等）在 GF(2^m) 上实现，符号对齐物理故障域（芯片、DQ pin 或本文的 16-bit 区域）。本文的构造流程可作为范本：先构造满足 SEC-DED + bounded-fault + CRC8 的 H2（16×288）——奇数权重列 + 每个 16 列有界区域的后 8 列为前 8 列 XOR 组合；再把 H2 各区域符号化（映射 GF(2^16)）直接嵌入 H^S-ECC 上半部；下半部用贪心搜索（随机赋 GF(2^16) 元素、二值化检查 syndrome 重叠、重建重叠最多的符号直至满足 SSC+DEC）。使用场景：主存 S-ECC、片上 O-ECC、链路 L-ECC 的编解码器设计；四分类指标（CE/DUE/SDC 比率）是所有 ECC 方案可靠性评估的统一语言。
HBM-CASO 补充视角（ISCA'26，HBM 故障分类与 syndrome 空间）：HBM 故障按物理来源分四类——Single-Bit Fault（SBF，最常见）、Byte-size Burst Fault（BBF，8 连续位，TSV 或 mat 失效，每 8-bit TSV burst 对齐半个 16-bit sub-wordline 故 TSV 故障常表现为 BBF）、Word-sized Burst Fault（WBF，16 连续位，sub-wordline 失效）、Subarray-level Fault（SAF，整 subarray/bank）。ECC 的目标是把 residual fault 压到可接受水平，错误按译码结果分 DCE（可检可纠）、DUE（可检不可纠）、SDC（静默损坏，含 UUE 未检出与 DME 误纠）。syndrome 空间视角的"大码字优势"：RS(72,64) 有 8 个 8-bit check symbol、syndrome 空间 2^64，其中仅 0.02% 对应可纠错误模式（纠错置信度极高）；而 RS(18,16) 只有 2^16 syndrome 空间、约 7% 是可纠模式（更可能未检出/误纠）。同冗余下大码字 Hamming 距离更大：一个 RS(72,64) 任意 4 个符号错都可纠，而四个 RS(18,16) 只纠"每个 16B 块恰好一个错"的均匀分布——真实错误很少如此均匀，故 HBM-CASO 主张用大码字（advanced ECC）。评估方法：每错误模式 10^9 次 Monte Carlo 注入（burst 内每位 50% 概率翻转、至少 1 位翻转），分类 DCE/DUE/SDC；传输保护只要求检测（检出即重传），故单独报 UE%（未检出率）——R/G-mode 用 regional/global 码字重建做读传输检测、用累计 parity 验证做写传输检测。

涉及论文标题：
- Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
