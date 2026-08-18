## DRAM Retention Failure（保持性失效）与 True-/Anti-Cell（真/反单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 保持性失效（retention failure）：DRAM 电容电荷随时间泄漏，若在刷新到来前电压降到 BLSA 无法正确判读的水平，即发生 retention bitflip。表征方法：写数据 → 暂停刷新（refresh pause）一段时间 → 读回比对 bitflip 数；DDR4 标准刷新窗口 64ms，实验常用 95 °C + 4s 停刷加速大量失效。True-cell = 充电态表示逻辑 1 的单元（只发生 1→0 翻转），anti-cell = 充电态表示逻辑 0（只发生 0→1）；用 0xFF/0x00 模式的失效分布可反推 true/anti-cell 布局。Web 来源：An Experimental Study of Data Retention Behavior in Modern DRAM Devices（ISCA 2013）、DRAM Retention Behavior with Accelerated Aging（Appl. Sci. 2022）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- DejaVu 的发现：保持性同样受写历史影响——victim 行以 OverWrite（先 0x00 后 0xFF）初始化比 SameWrite（0xFF 写两次）平均多 10.4%（最高 36.7%）的 retention bitflip，0xAA/0x55 下最高 123.2%，方向在三大厂商全部一致；机理与读干扰共享（欠恢复 + 陷阱占据态改变都缩短有效保持时间）。芯片设计含义：retention 测试标准方法（刷新间隔标定、true/anti-cell 逆向）必须固定初始化协议（论文建议统一 SameWrite 或统一 OverWrite），否则测得的最坏保持时间混入写历史偏差；true/anti-cell 布局逆向也是 DejaVu 实验中调整数据模式、正确解读 0xFF/0x00 差异差异的前提（§3.3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：DRAM Bender 停发 auto-refresh 命令 + 控温（heater 95 °C）→ 写模式 → 定时读回比对。使用：确定每行最坏刷新间隔、逆向 true/anti-cell 布局（对齐物理 1/0 调整数据模式）、评估写历史对保持裕量的影响。DejaVu 测试与 ACmin 实验相同的行（128 行/模块 × 50 次重复），使读干扰与保持性失效可对照。

涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
