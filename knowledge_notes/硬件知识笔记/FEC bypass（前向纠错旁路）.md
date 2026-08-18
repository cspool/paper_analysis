## FEC bypass（前向纠错旁路）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FEC（Forward Error Correction）是 PCIe 6.0/CXL 3.x PHY 为 PAM4 高误码率（原始 BER ~1e-6）引入的纠错机制：3-way 交织码 + CRC，把误码率纠正到 <1e-12（FIT ~5×10⁻¹⁰）。网络类 PAM4 的 FEC 延迟可达 100ns，PCIe 6.0 把 TX+RX FEC 延迟目标压到 ~10ns 内。FEC bypass 指链路质量足够好（高信噪比）时跳过或缩短 FEC 解码路径，换取更低延迟同时维持链路完整性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文在统一控制器的 PCS 内做选择性 FEC 旁路：高 SNR 环境启用 bypass 最小化解码延迟；一般环境保留完整 FEC 纠错能力。逻辑链：PAM4 误码高 → FEC 保证可靠性但增加解码延迟 → 当实测/预估误码远低于 FEC 保护阈值（高 SNR）时，旁路解码可省掉该段延迟而无可靠性损失。FEC 每多纠一个符号，延迟与复杂度近似指数增长，故旁路是省延迟最直接的手段。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PCIe 6.0 FLIT 模式 256B flit = 236B TLP + 6B DLP + 8B CRC + 6B FEC；常规接收流程为先 FEC 解码再 CRC 校验。旁路类实现（web：Intel 专利 US20250225024）：CRC 直通先行校验，通过则跳过 FEC 解码；失败回退完整 FEC 路径。使用场景：板内/机架内短距高 SNR 链路（OSFP 直连、chip-to-chip），本论文 200Gbps OSFP 直连存储节点即此类链路。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
