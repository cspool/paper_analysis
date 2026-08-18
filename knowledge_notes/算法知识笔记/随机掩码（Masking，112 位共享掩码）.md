## 随机掩码（Masking，112 位共享掩码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Masking 是让密文非确定性的手段：数据与随机掩码（one-time pad）异或后再加密，使同地址同明文两次写入的加密前值不同。纯 masking 每次访问 128 位块都要重新生成随机掩码，需要持续高熵随机流与昂贵 AES 运算，代价过高（论文实测纯 masking baseline N=2^14/B=256B 平均 38.4ms/访问，比 64 位计数器方案慢 13.5×）。
- MC-ORAM 变体：**同一树节点/暂存内所有 112 位 AES 块共享同一个 112 位随机掩码**（node.mask/stash.mask），数据以 D[i]⊕mask 存储；掩码不随每次访问变化，只在计数器溢出刷新时更新。共享掩码足够的原因：非确定性只需覆盖"同物理位置重复写"场景，不同物理位置由 AES-XTS 的地址 tweak 天然区分。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 初始化（Algorithm 1）与掩码写（Algorithm 3）的字节级流程：
node.mask = Rand(); stash.mask = Rand()        # 每节点/暂存一个 112 位掩码
node[r][i].data = D[i] XOR node.mask           # 落树前掩码化
dst = wrMask[j] ? (node[i][j] XOR node.mask XOR stash.mask)  # 树→暂存掩码域转换
                : stash[j][k].data
stash[j][k].bits = (dst || ctr+1)              # 128 位块 = 112 位掩码数据 || 16 位计数器
```
- 安全论证（VII-A）：同掩码周期内两次访问计数器必不同（概率 1）；跨周期掩码独立均匀 → 加密前值不同概率 1−2^−112 → AES-XTS 密文不同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：每节点/暂存维护一个掩码寄存器；写块时 XOR 掩码；掩码随 Refresh 更新（换新随机掩码并重 XOR 全部块）。掩码只存在于 TEE 内部（客户端可信侧），服务器只见密文。
- 使用：与 16 位计数器组合构成 MC-ORAM 的核心；带宽仅 baseline 的 1.125×（对比 64 位计数器 2×），存储减少 43.75%。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
