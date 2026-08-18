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
