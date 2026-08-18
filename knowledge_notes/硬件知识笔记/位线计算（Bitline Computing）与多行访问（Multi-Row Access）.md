## 位线计算（Bitline Computing）与多行访问（Multi-Row Access）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
位线计算是 in-SRAM 计算（存内 SRAM 计算，见本库 PIM/SRAM-PIM 条目）的核心原位计算原语：SRAM 阵列由二维 bit cell 组成，横为 wordline（字线）、纵为 bitline（位线）；普通读操作只激活一条 wordline，数据经位线流入 sense amplifier（敏感放大器）。位线计算的关键操作是**多行访问（multi-row accessing）**——同时激活两条 wordline（如 WL1 与 WLn），两条 wordline 上的数据同时流入共享位线，sense amplifier 在每条位线末端读出两个 bit 的 AND（在 BL 上）与 NOR（在 BLB 上），即 2-input 的 AND/NOR 逻辑直接在存储阵列内完成。论文给出的 AND/NOR 真值表（Fig.1(b)）：BL 读出 AND、BLB 读出 NOR；组合二者可得到 xor 等更丰富逻辑。多行访问会带来数据破坏风险（多行同时访问可能扰动单元状态），论文指出可通过**降低 wordline 电压使其偏置低于写电压**来避免（代价是 SRAM 频率略降）。为执行加法等复杂算术，阵列末端需要外围电路：EVE/MagiCache 采用的 1-bit 计算外围电路含四层——logic 层（由 AND/NOR 生成 (n)and、(n)or、x(n)or）、add 层（跨位线进位链实现加法）、shift 层（移位单元）、writeback 层（选结果写回 SRAM）。在 FSM/sequencer 控制下，外围电路组合基本逻辑结果即可在阵列内完成复杂算术运算，数据无需搬出阵列到寄存器文件。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PipeIMC 把位线计算作为计算阶段（calculation phase）的原子步骤：sequencer 先取得 phase 对应的源/目的 wordline 索引，再按预写微程序执行 SRAM 操作。基于 8-bit-hybrid 布局（见本库 Hybrid 数据布局条目），加法/逻辑操作在 8 个周期内完成——**奇数周期执行一次多行访问操作（BLC：BitLine Computing，同时激活两条 wordline 在 SA 中做位线计算），偶数周期执行一次写操作**；乘法通过 32 次"移位+加法"迭代完成（105–634 cycles），除法 145–1174 cycles。多行访问与普通读写不同：位线计算在 sense amplifier 中就地产生 AND/NOR 结果，而普通读只读出单行数据。因为操作数在计算 phase 期间被反复读出（多行访问多次引用源 wordline），其他操作不能修改源 wordline——这是 PipeIMC 提出的"read-first read-write 端口冲突"的根源，需要寄存器重命名解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上，位线计算是 Compute Cache [1]（首个 bit-parallel 计算 SRAM 原型）、Neural Cache [6]（bit-serial 数据布局 + FSM 状态机执行模型）、EVE [2]（bit-hybrid 布局 + in-order vector engine）、Duality Cache [11]（in-order SIMT + bit-serial）、MagiCache [9]（cacheline 级细粒度管理）等 in-SRAM 计算架构共同的基础技术；PipeIMC 在其上增加流水化与乱序执行。使用方式：把 SRAM 阵列当向量计算单元——数据以混合/位串行布局驻留 wordline，通过微码序列（多行访问 + 写 + 移位/进位外围电路）执行整数算术与逻辑；SRAM 可集成进 CPU cache（compute ways 与 cache ways 动态分区），kernel 执行时作为 CPU 协处理器。Web 证据：Compute Cache（Aga et al., HPCA 2017）与 EVE（Al-Hawaj et al., HPCA 2023，https://www.csl.cornell.edu/~cbatten/pdfs/alhawaj-eve-hpca2023.pdf）是最早的 bitline-computing 计算缓存代表。Vault 笔记（omnisearch 无命中，text 检索命中均为 DRAM-PIM/LLM serving 方向）无本术语专门笔记证据。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
