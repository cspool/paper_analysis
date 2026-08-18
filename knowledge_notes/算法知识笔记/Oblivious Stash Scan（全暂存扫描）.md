## Oblivious Stash Scan（全暂存扫描）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Oblivious stash scan 是 TEE 内 ORAM 的防侧信道手段：每次逻辑访问对暂存**全部槽**执行相同操作的线性扫描（读路径与驱逐各一次，共 2ZL 次线性槽更新），无论目标块是否已找到、是否被写入。MC-ORAM 中每个被扫描槽的计数器都 +1（即使 wrMask 为假不写入），保证每次访问暂存每条目的密文都变化。先例：Oblix [22]、ZeroTrace [34]、OBLIVIATE 等。论文 VIII-B 用四种访问模式（LS/均匀随机/高斯/重复访问 RA）验证延迟几乎一致，证明性能只依赖配置参数。
- 必要性：stash 占用率高度依赖访问模式（重复访问小、非重复大）；若处理只在找到目标时停止或只更新被写槽，指令足迹/DRAM 流量随模式变化，TEE 内可被观察。全扫描使每次 enclave 访问触达相同足迹。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 2/3 (readPath/TreeToStash) 的 oblivious 结构：
for j in 1..|stash|:
    wrMask[j] = !found && stash[j].isEmpty; found = found || wrMask[j]
for j in 1..|stash|:                       # 无条件遍历全部槽
    for k in 1..|stash[j]|_bits/128:
        stash[j][k].bits = (wrMask[j] ? node[i][j] XOR node.mask XOR stash.mask
                                      : stash[j][k].data) || (ctr+1)   # 无条件递增
```
- 例子：PathORAM 每次访问产生 2Z·L 次线性 stash 更新（读+驱逐各 Z·L 块处理）；stash=90 槽时全扫描是延迟主导项之一；优化变体（+）只把目标块入 stash 减少占用，但仍保持全扫描。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：两层循环（槽×槽内 AES 块）+ wrMask 条件选择 + 计数器无条件递增；驱逐（StashToTree）反向同理并同时更新树节点。stash 需小（相对整树）以摊薄扫描成本。
- 使用：TEE 内 ORAM 客户端（TDX/SNP）每次访问必做；与掩码/计数器/刷新机制正交，MC-ORAM 保留之。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
