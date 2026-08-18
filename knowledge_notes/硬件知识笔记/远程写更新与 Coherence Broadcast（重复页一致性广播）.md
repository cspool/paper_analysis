## 远程写更新与 Coherence Broadcast（重复页一致性广播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
多 GPU 页复制的一致性机制：GPU 写本地重复页时，必须把新值广播（remote update）给所有持有该页副本的 GPU，保证各副本一致。CDFD 经 duplicated TLB/sharer table 得知共享者 PFN，由 GMMU 发远端写。广播粒度 = 写粒度；论文按 cache line 粒度统计 coherence broadcast：某 cache line 在下一次同线广播或页换出之前都没有被任何副本 GPU 访问，则该次广播记为"无用"。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
广播在写路径 off critical path（不阻塞写者），但占 NVLink 带宽与各副本方的访存带宽，并带来副本方 cache/TLB 干扰。CDFD 用细粒度去重剔除"远端更新多、本地访问少"的子页以控制无用广播：实测每 32MB 重复页平均 21,565 次广播、6,844 次无用（约 68% 有用）；无去重的 CoarseDup 广播更多（ST/C2D 等写同步密集 benchmark 无用广播显著）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPS 以订阅式批量（batched）方式更新订阅者；CDFD 按写即时广播 + CDB 计数统计；NVIDIA read-mostly 语义则是写时失效其他副本（更粗的一致性粒度），CDFD 保持副本并广播更新，在读多写少负载下更高效。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
