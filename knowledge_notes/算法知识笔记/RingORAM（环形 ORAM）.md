## RingORAM（环形 ORAM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RingORAM 是与 PathORAM 同树的 ORAM 构造（Ren et al., USENIX Security'15）：每桶除 Z 个真实块外加 S 个永久 dummy 块（桶共 Z+S 块，空间更大），从而每次逻辑访问**每桶只读 1 块**（目标块或随机 dummy），而非 PathORAM 的整桶 Z 块，显著降低带宽。驱逐不是每次访问都执行，而是每 A 次访问一次（论文 A=4），驱逐路径按 reverse-lexicographic（逆字典序）固定调度顺序选择、与访问模式无关；驱逐时桶内 (Z+S) 块按 rotation schedule 洗牌。因每桶只读一块，块以 1/(Z+S) 概率保留原内容，确定性加密下仍可泄露（论文 V-B）。
- 本论文把 MC-ORAM 集成到 RingORAM：掩码/计数器/刷新机制不变，差异为单块访问上应用掩码+计数器、节点含 (Z+S) 块、每 A 次访问驱逐、驱逐时洗牌在掩码域内完成（无需额外密码机制）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# RingORAM 单次访问（MC-ORAM 集成版）：
P = PosM[d]
for node in 路径P:
    i = Rand() % (Z+S)            # 每节点只读 1 块（目标或 dummy）
    读 node[i]，TreeToStash 合并入 stash（全暂存扫描+计数器+1）
process(d); PosM[d] = Rand()
每 A 次访问触发 eviction（reverse-lexicographic 路径）：
    整条驱逐路径逐节点读入 stash（每节点全 (Z+S) 块）→ 掩码写回 + 桶内洗牌
# 任一节点计数器溢出 → Refresh 整节点
```
- 例子：S=3、A=4、Z=4、stash=90，RingORAM 访问延迟 0.78–33.08ms；MC-ORAM 版 0.42–19.05ms（最高 1.85×）；RingORAM+（stash=10）0.16–5.83ms vs MC-ORAM+ 0.10–3.44ms（最高 1.60×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：树形桶 (Z+S) 块 + 周期驱逐 + reverse-lexicographic 路径调度 + 桶内洗牌；开源参考 oram_simulator（https://github.com/wangxiao1254/oram_simulator）。论文 RingORAM 计数器在节点内可能不同（单块访问），任一溢出即刷整节点。
- 使用：带宽敏感场景（每桶单块读取降低路径流量）；本论文在 TDX 内评估 N=2^14/2^23 与多块大小，并用于 DLRM/Qwen-8B 安全嵌入（RingORAM+ 作为 ML 端到端 baseline：Qwen-8B 36.2ms→MC-ORAM+ 25.8ms 1.41×、DLRM 6→3.61ms 1.66×）。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
