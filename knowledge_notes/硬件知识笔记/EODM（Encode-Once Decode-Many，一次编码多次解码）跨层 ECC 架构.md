## EODM（Encode-Once Decode-Many，一次编码多次解码）跨层 ECC 架构

术语解释
Cerberus 的核心架构：控制器只做一次编码生成共享冗余，该冗余被链路层（写路径检测/重传）、设备层（片上纠错）、系统层（端到端 SSC+DEC）三层按各自角色重复使用，取代传统各层独立编码。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EODM 解决"三层 ECC 独立演进"的三大问题：冗余重复（各层各配一份校验位）、覆盖缺口（层间盲区，如 O-ECC 之外的 out-of-bank 区域）、破坏性干扰（上层 miscorrection 放大）。架构组成（本文 Fig.4）：1 个共享编码器 + 3 个分层解码器。编码器用复合生成矩阵 GS-ECC = G1·G2：G1 把 256b 数据 D 映射为 272b 中间码字 (D+R1)，G2 再映射为 288b 最终码字 ((D+R1)+R2)，R1/R2 各 16b；硬件上用单步 XOR 网络实现，概念上两层映射只是构造方便。三层解码：LL（链路层）用 R2 + H2 做写路径检测、错则重传；DL（设备层）读路径复用同一 H2 做片上 SEC（bounded-fault）；SL（系统层）用完整 R1+R2 做 SSC+DEC 端到端纠错与检测。跨层成立的唯一条件：row(H2) ⊆ row(H^S-ECC)（等价于 GS-ECC 可分解为 G1·G2），满足即可让任意 vendor 的 S-ECC 接入框架。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
数据路径运转（Cerberus HBM 配置，256b 数据 + 32b 冗余）：写——控制器 8 级 XOR tree 用 GS-ECC 一次算出 (D+R1)+R2 → 288b 直接传 DRAM（无独立 L-ECC 编码级）→ DRAM Decoder1 用 H2 校验，失败 ALERT 重传、成功连同冗余写入阵列。读——bank group Decoder2 用同一 H2 生成 syndrome、做 bounded-fault SEC 纠错（约 +4 级逻辑深度）→ DRAM 跳过读侧 L-ECC 生成、转发带冗余的 288b 码字 → 控制器 Decoder3 用 H^S-ECC 生成 32-bit syndrome → SSC（Chien search + 修正 BM）与 DEC（block-pair solver）并行单周期纠错 → 检测单周期、DUE 重试一次。EODM 的硬件收益：消除写路径连续编码级（性能 +0.7% IPC）、存储开销 HBM4 的 18.8% → 12.5%（−33.3%）、DRAM 内传输位宽缩减（能耗 −1.84%）；代价是 Decoder3 的 SSC+DEC 面积（约 124k NAND2，但对 GPU 晶体管数可忽略）。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点（本文 Code Construction）：① H2 构造——奇数权重列保证 SEC-DED，16 列有界区域后 8 列为前 8 列 XOR 保证 BF 与 CRC8；② 跨层嵌入——把 H2 逐区域符号化（GF(2^16)）放进 H^S-ECC 上半部，天然满足 row 包含条件；③ H^S-ECC 下半部贪心搜索——随机赋 GF(2^16) 元素、二值化后检查 SSC/DEC syndrome 无重叠，重叠则重建该符号。可扩展性：冗余从 32b 增至 40b 时同一框架直接复用（更强的检测，Cerberus 40b 的 SDC 率更低）；适用于一切 256-bit 粒度、满足跨层条件的 DRAM。使用场景：面向下一代 HBM（含定制 HBM）与 LPDDR 的单设备通道内存；作为跨厂商协作框架，DRAM 厂商管 H2 侧片上实现、处理器厂商管 H^S-ECC 侧系统实现，通过共享矩阵互操作。
涉及论文标题：
- Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection
