## Elastic buffer（弹性缓冲：half-full vs nominal-empty）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
弹性缓冲是 SerDes/PCS 内跨异步时钟域（收发时钟不同源）的同步 FIFO，用于吸收时钟频率漂移。传统 PCIe 设计用 half-full 策略：填充至半满才读出，即使数据就绪也固定驻留半个深度，引入 idle cycle 累积延迟。本论文改 nominal-empty：把"空"视为合法状态，读侧实时对齐时钟、来数即走，消除不必要的中间驻留（staging），平均延迟 -15~20ns，占控制器级延迟改善的主要部分。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
接收路径流程对比：
- half-full：串行输入 → 恢复时钟采样 → 弹性缓冲填至半满 → 并行读出（固定驻留延迟，即使下游空闲）
- nominal-empty：串行输入 → 恢复时钟采样 → 弹性缓冲（空即合法、实时对齐）→ 并行读出（无固定驻留）
web 佐证：PCIe/CXL 接收端传统需累积整帧后 FEC 解码 + CRC 校验；Intel 专利（US20210119730A1 / US20250225024）提出 128B half-flit + CRC 直通低延迟路径、校验失败才回退 256B FEC 路径，与本论文"提前校验/旁路"思路同源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
以异步 FIFO 实现于 PCS，深度按时钟漂移预算设计，关键是正确标记空/满状态机与上溢/下溢处理。本论文将其作为统一控制器物理层三大延迟优化之一（另两项：FEC 旁路、flit 提前校验），配合统一时钟域消除层间握手。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
