## Contiguity-Preserving VA Slice 与 Copy-Free 弹性重映射（ConServe）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ConServe（ISCA'26）的 conversation 级虚拟内存管理：每个 conversation 独占一段连续 VA slice，内部按 transformer 层分段 back-to-back（layer-major），所有 turn 的 KV 追加进同一 slice，kernel 用纯算术寻址 VA(t,l)=base+seg_off[l]+t×B_layer+δ（B_layer=2×H_shard×d_head×b），无 block table。slice 初始大小 T_target=min(ctx_len, T_prompt+Q·k)（Q=ShareGPT 每轮 token 的 r=80% 分位数、k=8 turns），利用率 η=U/R>θu=0.90 时弹性增长 R_next=max(⌈γR⌉, ⌈T_target^(new)·B_tok/G⌉)（γ=1.5、G=2 MB 组粒度）。物理页按需映射。resize 用 copy-free remap：layer ℓ 完成后对新 slice 该层段 cuMemMap 同一物理页（KV 数据不搬）、与剩余层计算重叠，迭代边界切换 base/seg_off；旧 slice 置 PENDING_UNMAP 隔离、空闲时批量 cuMemUnmap + TLB invalidation。VA 碎片用 dual-index free-list（length-ordered 视图 best-fit O(log M)；address-ordered 视图 eager coalesce，merge flag 标记 PENDING_UNMAP 可合并区间），需求驱动 + 后台批量两档回收。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
一次 resize 事件流程：token 进入 layer 0 检查 η>0.90 → 计算 R_next → length-ordered free-list best-fit 选新区间 → 逐层（layer ℓ 完成后立即）重叠执行 cuMemMap/cuMemSetAccess 预装新 slice 映射（重叠窗口约 (L−1)/L·T_iteration）→ 迭代边界更新 descriptor 的 base 与 seg_off → 旧 slice 隔离、批量延迟回收并与邻居合并。效果：94.7% resize 完全隐藏、99% 暴露 ≤1.5 ms（不与计算重叠时每次尖峰 4–18 ms）。CUDA VMM API 延迟（2 MB 页微基准）：cuMemAddressReserve 2 µs / cuMemCreate 29.2 µs / cuMemMap 1.9 µs / cuMemSetAccess 36.8 µs / cuMemUnmap 34.3 µs / cuMemRelease 24 µs / cuMemAddressFree 1.6 µs。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA Driver API 虚拟内存接口，进程启动时 cuMemAddressReserve 预留一个 48-bit VA arena，各 conversation 在 arena 内取 slice；集成于 vLLM/vAttention 类框架 + FlashInfer kernel。参数敏感性（论文 V-D）：k=8 最优（过大使 VA 跨度过宽、过小 resize 频繁）；r 不敏感（误估被 copy-free remap 即时纠正）；γ=1.2 慢 5%、γ=2.0 慢 7%。整体收益：TTFT −64.1%~−74.4% (vLLM) / −31.4%~−43.4% (vAttention-Turn) / −8.3%~−19.1% (vAttention-Conv)；decode 吞吐 +19.4%~25.6% / +6.1%~9.5% / +4.2%~6.1%；离线端到端吞吐 +17.7%~35.1% / +8.6%~15.6% / +7.2%~12.1%；SLO attainment +11%~19% / +7%~9% / +3.2%~4.9%。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
