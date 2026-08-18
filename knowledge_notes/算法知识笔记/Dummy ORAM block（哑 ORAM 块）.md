## Dummy ORAM block（哑 ORAM 块）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Dummy 块是 ORAM 桶填充机制：为防桶占用率泄露，PathORAM 每桶用 dummy 块补齐到恒为 Z 块（RingORAM 为 Z 真实+S 永久 dummy 块）。服务器/攻击者无法区分真实块与 dummy 块，故桶大小恒定、路径流量形状恒定。dummy 块内容可保持为掩码（MC-ORAM 初始化：所有 112 位数据初始化为 node.mask）。
- 泄漏风险：若 dummy（及未变真实块）在确定性加密下密文不变，攻击者可检测"哪个位置没变"从而定位 dummy 与真实块。MC-ORAM 让所有块（含 dummy）计数器随访问递增、溢出刷新掩码，消除该定位信号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PathORAM 桶结构（N=4, L=3, Z=4 示例）：
node = [block0, block1, dummy, dummy]   # 真实块+补齐 dummy，恒 Z 块
# MC-ORAM 初始化（Algorithm 1）：所有 112 位数据= node.mask（dummy 与真实同格式）
# RingORAM：桶 = Z 真实 + S 永久 dummy（Z+S 块），驱逐时洗牌
```
- 例子：RingORAM Z=4、S=3，每桶 7 块；访问时每桶只读 1 块（目标或随机 dummy），dummy 与真实块在掩码/计数器/刷新上同等对待。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：初始化时把桶填满 dummy（内容可为掩码值），驱逐/洗牌时与真实块同规则移动；MC-ORAM 要求 dummy 块与真实块在掩码、计数器递增、节点刷新上完全一致处理，防占位差异泄露。
- 使用：PathORAM/RingORAM 的桶填充与 RingORAM 单块读取（读 dummy 掩盖目标位置）依赖 dummy 块。

涉及论文标题：
- MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs
