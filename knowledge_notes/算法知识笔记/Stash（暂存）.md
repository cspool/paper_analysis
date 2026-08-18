## Stash（暂存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Stash 是 ORAM 客户端侧的小型可信缓冲区：服务器读回的整条路径先解密落入 stash，客户端在此完成目标块计算；驱逐时无法立即回填的块（桶容量已满）暂留 stash 等待后续驱逐。stash 占用率高度依赖访问模式（重复访问占用小、非重复占用大），因此是访问模式泄露的敏感结构。
- TEE 内要求：每次逻辑访问必须对 stash **oblivious 全扫描**（读与驱逐各一次，共 2ZL 次线性槽更新），使每次 enclave 访问触达相同足迹，防微架构侧信道。因 stash 相对整树很小，线性扫描成本可摊薄。MC-ORAM 中 stash 持有共享 112 位掩码 stash.mask，所有槽计数器随每次逻辑访问整体递增（PathORAM 中保持相等），溢出即 Refresh(stash)。
- 泄露示例（V-C）：stash 条目内容不变→AES-XTS 密文不变，攻击者从密文推断槽是否被覆盖/换块，暴露 stash 占用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TreeToStash（Algorithm 3）：stash 槽 j 的 128 位 AES 块 k
if stash.ctr == 2^16-1: Refresh(stash)
dst = wrMask[j] ? (node[i][j] XOR node.mask XOR stash.mask)   # 掩码域转换
               : stash[j][k].data                             # 不写也保留
stash[j][k].bits = (dst || ctr+1)   # 计数器无条件 +1 → 密文必变
```
- 例子：PathORAM stash=90 槽（naive）/10 槽（+，Oblix 优化）；N=2^14、Z=4 时 stash 每 2^16/(2ZL)≈585 次访问刷新一次掩码，刷新开销 <1% 运行时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：客户端内存中固定大小数组，槽含 112 位掩码数据+16 位计数器；每次访问全扫描（读+驱逐各一遍）。stash 大小静态分配（90/10）。+ 变体（Oblix 思路）只把目标块放入 stash 降低占用，但需每 3 次访问额外驱逐防溢出。
- 使用：ORAM 客户端（TEE 内）的路径数据中转与驱逐缓冲；防侧信道的关键是"无论是否命中都扫描全部槽"。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
