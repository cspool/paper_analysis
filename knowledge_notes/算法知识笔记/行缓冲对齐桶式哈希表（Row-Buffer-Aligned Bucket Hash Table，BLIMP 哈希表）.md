## 行缓冲对齐桶式哈希表（Row-Buffer-Aligned Bucket Hash Table，BLIMP 哈希表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
面向 BLIMP（bank 级 PIM）设计的哈希表数据结构，用于 join 的 build/probe 与高基数分组聚合。针对 BLIMP 核弱（200MHz RISC-V）且以 1KB row buffer 粒度访存的特点，设计目标：①索引容易——hash 后能直接定位到要读写的一行；②row buffer 对齐——减少 row buffer 切换；③冲突局部性——hash 冲突的值应共处同一行以利用空间局部性。实现：row-buffer 对齐的 hash-indexed bucket 集合；初始桶数为 2 的幂（按期望 load factor 定），桶大小恰好契合 row buffer；桶含 metadata 与一串 slot（每 slot 一个列值，可带 payload——join 的 payload 或聚合器）；冲突时 slot 追加到桶尾，桶满则在桶链末尾建新桶并在原桶记录 next 指针。哈希函数用轻量乘法哈希保证强抗冲突：`hindex = (3634946921 * value + 2096170329) & (initial_buckets - 1)`；因桶大小固定，hindex 可直接换算桶所在 row buffer 地址，实现 hash→地址 O(1) 映射。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BLIMP-S 与 BLIMP-V 的桶内 slot 数差异显著（32-bit key + 8-bit payload 时 BLIMP-S 5 slots、BLIMP-V 24 slots/桶），因 BLIMP-V 的 SIMD 能力使扫描桶的时间逼近取新桶（row buffer 切换）的时间。探测伪代码（Algorithm 2 核心）：
```
for each 元素 v1[i]（从 row buffer 读入）:
    idx ← hash(v1[i]) & (initial_buckets - 1)      # BLIMP-V 可向量化批量 hash
    repeat:
        v3 ← FetchMem(h + BucketRow(idx))          # 打开目标 row buffer 桶
        hit ← (v1[i] ∈ v3.slots)                   # 串行检查 slot 列表
        idx ← v3.next_bucket                        # 桶链下一桶（若桶满溢出）
    until IsNull(idx) or hit
    v1[i] ← hit                                    # 位图置位
```
性能：build 由 host 完成（relayout 广播到各 bank），build 时间与 CPU 侧 Swiss Table 相当（论文引 abseil）；probe 侧随机 row buffer 访问是主要开销——低选择性时整个哈希表可入 host cache（host 占优），高选择性时 host 因 L2/L3 miss 劣化更快而 BLIMP 只受 row buffer 切换惩罚。semijoin 1.4×/2.1×、join 2.1×/3.0×（BLIMP-S/-V）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用场景：hash join 的探测侧、高基数 group-by 聚合（分组值 hash 到 bucket/slot，payload 为聚合器）、set 操作。实现要点：初始桶数按期望 load factor 定且为 2 的幂；桶大小选为"扫描时间 ≈ 取新 row buffer 时间"（BLIMP-S/-V 不同）；链式桶溢出处理；哈希表必须能在 32MB bank 容量内（大 build 侧按分区多轮 build-probe）。论文未声明该数据结构开源（论文未明确说明）。

涉及论文标题：
- Taking Analytic Databases to the Bank
