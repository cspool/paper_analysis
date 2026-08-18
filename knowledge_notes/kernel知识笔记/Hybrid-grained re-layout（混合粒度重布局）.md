## Hybrid-grained re-layout（混合粒度重布局）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-PIM 在 rank PU 上以 re-layout 单元完成的 in-flight 数据布局变换，解决 DIMM 存储布局与 PIM 计算布局不匹配的问题：DIMM 多 chip 交织下单个元素跨多个 ×8 chip（如 FP16 的 16 bit 跨 2 个 chip），而 PIM 计算要求元素完整位于单一 chip，同时计算需要按 head 映射到指定 chips。两级变换：fine-grained（元素级/位级）——把单元素各位连续排进同一 chip 的 burst beat（元素不跨 chip）；coarse-grained（head 级）——把每个 head 的元素映射到 N_hc 个 chip。数据 offload 时 QKV 先缓存在 rank PU 片上 SRAM，重排后再写入 DRAM chips；onload 反向。对比 CPU 辅助 re-layout（读旧布局+写新布局的往返访存），CHIME 把变换融合进传输路径，消除每层每 token 的累积开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fig.8 例子（8×8 chips、E0-E127 为 head 内元素 0-127、C0-C7 为 chips）：
```
# offload QKV: rank PU SRAM → re-layout unit → DRAM chips
for element e in head:                    # fine-grained：元素不跨 chip
    put bits(e) into contiguous burst beats of chip(e % N_hc)
# coarse-grained：按 N_hc 组织 burst beat 内容
if N_hc == 8:  # 每个 burst beat 只含单 head 元素 → 一个 head 分布到 8 个 chips
    beat = elements_of_single_head()
else:          # N_hc == 1：8 个 head 的元素混排一个 burst beat → 每 head 落单 chip
    beat = elements_of_8_heads()
# onload：reverse 过程，从 chips 读回并恢复原布局
```
效果：最多 -17% attention 延迟（随 token 长度增大占比下降，因计算主导，但 re-layout 开销每层每 token 累积，仍需消除）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：re-layout 单元做于 buffer chip（rank PU 内、逻辑工艺），数据路径上动态重排；与 bubble-free pipelining 的传输流水协同。使用方式：所有 DIMM-PIM/多 chip PIM 的必答题——UPMEM 系用 CPU/专用引擎重排、Facil 用灵活地址映射、PIM-MMU 用 MMU 级变换；CHIME 的贡献是把两级粒度（位级元素对齐 + head 级映射）统一到传输中完成。推广到其他数据（激活、权重）的布局变换同样适用。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
