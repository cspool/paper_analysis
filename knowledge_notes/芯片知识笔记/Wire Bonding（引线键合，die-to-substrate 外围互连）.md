## Wire Bonding（引线键合，die-to-substrate 外围互连）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wire bonding 是最成熟的芯片封装互连：用金属丝（Au/Cu/Al，直径 20-50µm）把 die 上的 pad 与基板/引线框架连接，连接只分布在 die 周边（20-600µm pitch，[9]），成本极低但互连密度低。在 3D 封装谱系（Fig.1）中它是最早、最便宜的一档，与 microbump+TSV、hybrid bonding（2-3µm pitch Cu-Cu 直接键合）相比密度与带宽劣势明显。HybridSpec 用它连接 HB 栈的 logic die 与 substrate（XPU-HB 传输链路），因 SD 的模型级映射把数据移动限制在 draft-verification 边界，该链路带宽需求低。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
HybridSpec 的封装选择逻辑：HBM 体系在 logic tier 用 TSV 接 substrate（高带宽高成本），但 HybridSpec 的 XPU↔HB 主传输只有 draft 模型 KV cache（1-3B draft 几 MB）与 token 列表（几百 B），hidden-state SD 也仅几十 KB，聚合带宽几十 GB/s 即够——用 wire bonding 足够（实验 VI-G 扫描 2-8GB/s/mm 带宽密度与 1-16µs 链路延迟，数据移动 <1% 执行时间，链路延迟是主导开销）。执行流：draft decode 迭代在 HB 栈完成后，draft token 列表经 wire bonding 链路（小包、几十 GB/s）传回 XPU 验证，验证结果回传后清除误推测 KV cache。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：球焊/楔焊工艺把金属丝键合到 die pad 与基板焊盘（成本最低、良率成熟、无需 TSV/interposer）。使用方式：当芯片间数据移动量小或带宽需求低时替代高成本互连——HybridSpec 用它做逻辑 die-to-substrate（省掉 HBM 式 TSV 的成本）；文献中 3D 堆叠历史沿革（wire bonding → microbump+TSV → HB）中它是最低档选项，带宽密度 Gbps/mm 逐级提升。局限：pitch 大（20-600µm）、只到 die 周边、带宽/密度上限低，不适合逐层/逐算子高带宽搬运。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
