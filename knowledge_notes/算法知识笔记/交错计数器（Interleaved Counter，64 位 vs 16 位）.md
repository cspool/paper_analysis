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
