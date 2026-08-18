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
