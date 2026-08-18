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
