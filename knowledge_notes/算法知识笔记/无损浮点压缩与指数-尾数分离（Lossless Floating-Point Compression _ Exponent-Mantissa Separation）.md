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
