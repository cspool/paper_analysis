## 掩码刷新算法（Mask Refresh Algorithm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MC-ORAM 的掩码刷新（Algorithm 6）：当某节点/暂存内任一 16 位计数器达到 2^16−1 即将溢出时，对该节点/暂存的全部 128 位 AES 块重新生成一个随机掩码并清零全部计数器：
```
Refresh(node):
  new_mask = Rand()
  for i in 1..|node|_bits/128:
      node[i].bits = (node[i].data XOR node.mask XOR new_mask) || 0
  node.mask = new_mask
```
- 关键性质：刷新频率只由公开 ORAM 参数（Z、L）与访问次数决定、与输入访问模式无关（VII-B）——节点在 level ℓ 被触达概率 1/2^(L−ℓ)，节点刷新期望 Σ 1/2^(16+L−ℓ) ≤ 2/2^16 ≈ 3.05×10^−5 次/访问；暂存每 2^16/(2ZL) 次逻辑访问刷新一次（N=2^14 时 585 次、N=2^20 时 409 次）。因此刷新不引入新泄漏（access oblivious）且摊销开销 <1%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 触发点（Algorithm 3/5 入口）：
if stash.ctr == 2^16-1: Refresh(stash)
if node[i].ctr == 2^16-1: Refresh(node)
# 刷新期间对全部块重 XOR（旧掩码→新掩码），计数器置 0；
# PathORAM 节点计数器同步递增可同刻溢出；RingORAM 任一溢出即刷整节点。
```
- 例子：N=2^14、Z=4、L=14 时 stash 刷新周期 2^13/14≈585 次访问；每次刷新只重写一个节点/暂存（约一个 AES 块组），成本 <1% 总访问时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TEE 内软件流程，触发于 TreeToStash/StashToTree 入口；生成新掩码（Rand()）、遍历节点全部 128 位块重 XOR、清零计数器。只操作 TEE 内部表示，写回时经 TME AES-XTS 加密。
- 使用：作为"16 位计数器+共享掩码"方案的溢出处理机制，使掩码从"每次访问"降为"每掩码周期"粒度，实现 1.125× 带宽下的密文非确定性。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
