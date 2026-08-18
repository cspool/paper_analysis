## PathORAM（路径 ORAM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PathORAM 是最广泛使用的树形 ORAM 构造（Stefanov et al., CCS'13/J.ACM）：把 N 个数据块（块大小 B）组织成服务器侧高度 L=log(N) 的二叉树，每节点是一个桶，最多存 Z 个 ORAM 块；为防止桶占用率泄露，每桶用 dummy 块补齐到恒为 Z 块。客户端维护位置图（块→叶子映射）与 stash。每次访问：查位置图→读整条根到叶路径入 stash→处理目标块→分配新随机叶子更新位置图→按驱逐规则贪婪回填写回。带宽放大 log(N) 倍。协议简单、客户端状态小，是 ORAM 事实标准。
- 本论文以 PathORAM 为 baseline 与承载协议之一：baseline 采用 64 位交错计数器（每 64 位数据配 64 位计数器，Obelix 风格）实现非确定性，MC-ORAM 在其上加 112+16 位掩码计数器布局。泄露分析（V-B）：确定性加密下第二次重复读可能观察不到树密文变化，攻击者可区分 ⟨Read 0, Read 0⟩ 与 ⟨Read 0, Read 3⟩，区分优势 1/4。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PathORAM 一次访问（含 MC-ORAM 掩码/计数器）：
P = PosM[d]
for node in 路径P:                 # 读路径：每节点 Z 块
    for i in 1..Z:
        wrMask = 计算条件写掩码(暂存空槽)
        TreeToStash(stash, node[i], wrMask)   # 掩码写 + 全暂存计数器+1
process(d); PosM[d] = Rand()
evict(P): for node in P:
    for i in 1..Z:
        StashToTree(stash, node[i], wrMask, found)  # 反向 + 树/暂存计数器+1
# 计数器溢出(2^16-1) → Refresh(node/stash)：新掩码+清零
```
- 例子：N=2^14、Z=4、stash=90，PathORAM 访问延迟 1.48–39.92ms（B=cacheline~2048B）；MC-ORAM 版 0.87–22.56ms，最高 1.82× 加速；暂存优化版（PathORAM+，Oblix 式：只把目标块入 stash、每 3 次访问额外驱逐、stash=10）0.19–8.66ms vs MC-ORAM+ 0.11–5.00ms。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：服务器侧二叉树（每桶 Z 块含 dummy）+ 客户端位置图/stash/驱逐；开源参考 PathORAMSimulator（https://github.com/renling/PathORAMSimulator）。本论文在 Intel TDX VM 内实现（Ubuntu 22.04.5、双路 Xeon 6548Y+、512GB DDR5），并用 Intel PIN 采集 SPEC CPU2017 轨迹确定 ORAM 高度。
- 使用：作为 TEE 内安全嵌入表/安全存储的原语；参数 Z=4、stash naive 90/优化 10；评估 N=2^14/2^23、B=512b(cacheline)/256B/2048B(embedding)。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
