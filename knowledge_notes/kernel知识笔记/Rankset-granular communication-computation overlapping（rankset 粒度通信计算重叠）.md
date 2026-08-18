## Rankset-granular communication-computation overlapping（rankset 粒度通信计算重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-sys 隐藏 GPU↔CHIME-PIM 之间 PCIe 通信开销的机制。问题：GPU-PIM 数据通信与 PIM attention 计算共享内存总线，粗粒度地"全 rank 要么通信要么计算"会互相阻塞，而 PCIe 带宽比 CHIME-PIM 低数个数量级，通信不可忽略。关键观察：一个通道内同一时刻只能访问一个 rank（总线共享），其余 rank 空闲。据此定义 rankset = 从每个通道取一个 rank 组成的最小独立通信/计算单元（同时用满全部通道的最小集合）：对一个 rankset 做通信时，其余 rankset 可并行做计算——DGX-A100 上 4 ranksets 时通信期间保留 3/4 计算能力（每通道 3 ranks 的例子则保留 2/3）。负载均衡：利用各层 KV cache 大小相同的特性，把每个请求的 KV cache 按 layer 粒度 interleaved 存放到各 rank，保证各 rankset 传输量相等。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
ranksets = [ {rank_i from each channel} for i in 0..R-1 ]   # R = 每通道 rank 数
# 异步流水：通信 rankset i 与计算 rankset (i+1) mod R 并行
for t in timeline:
    PCIe.write(QKV_next, rankset = t % R)      # 通信：占该 rankset 的总线
    PIM.compute_attention(rankset = (t-1) % R) # 计算：其余 rankset 不受阻塞
```
数据流：prefill K/V（∝输入长度）、decode QKV（每请求每步 1 token）、attention 输出（1 token/请求）三类数据按 rankset 轮转传输，与上一轮 attention 计算重叠。效果：PCIe 开销最多降 75.08%（不同 batch 配置，4 ranksets 的 DGX-A100）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器按 rankset 分配请求的 KV 存储与传输任务，硬件侧每个 rankset 的通信与计算独立可切换。使用方式：任何"共享总线的多单元异构系统"的通信隐藏——类同 GPU 上 streams/双缓冲，但粒度由内存通道组织（channel/rank）决定；需要配合负载均衡（layer 粒度 interleaved）避免最慢 rankset 拖慢整体。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
