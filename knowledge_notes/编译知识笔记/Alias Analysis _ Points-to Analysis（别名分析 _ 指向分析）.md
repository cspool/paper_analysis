## Alias Analysis / Points-to Analysis（别名分析 / 指向分析）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 别名分析回答"两个指针/内存访问是否可能指向同一地址"：若可能（may-alias）则访问间存在依赖、不能任意重排或共享资源；若必然（must-alias）或不可能（no-alias）则可为编译器优化提供精确信息。MANATEE 采用 Andersen 的 flow-insensitive points-to analysis [7]，用于页级 liveness 分析：只有精确知道每条 load/store 访问哪个页，才能准确计算页的 live range。
- MANATEE 的别名分析精度 >99%（论文所有 benchmark）：原因有二——约 97% 的指针指向全局数组（其地址从 linker .map 可见，多为 must-alias 或 no-alias）；多数访存指令用简单寻址模式、仅少数用符号模式（目标地址不明确）。对 may-alias 指针，MANATEE 保守分配新颜色，若颜色不足则分配最远可用颜色。
从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：CFG 上指令 `store r, [ptr]`，ptr 经 points-to 分析指向全局数组 arr 的某个元素 → 结合 .map 地址算出目标 NVM 页号 → 该 store 归属该页的访问集合 → 参与该页 live range 计算与着色；若 ptr 是 may-alias（指向 arr 或 brr 之一），则同时加入两个页的访问集合，保守处理。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：编译器静态 points-to 分析（Andersen [7]）+ linker 地址信息辅助；未知迭代次数循环不做指针分析、直接保留专用颜色（Sec. III-C）。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
