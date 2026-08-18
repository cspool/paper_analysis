## Mapi-Pro（基于 ILP 的 profiling 内存映射方法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Mapi-Pro [9] 是 MANATEE 的主要 baseline（state-of-the-art profiling 方法）：用 profiling 收集内存访问 trace，以整数线性规划（ILP）识别 hot/cold 页，把 hot 页映射到 SPM、cold 页留在 NVM；断电时把整个 SPM checkpoint 到 NVM（整 SPM checkpoint）。ILP 保证静态映射在 profiling 数据下最优，但映射一次确定后不再移动。
- 两个核心缺陷（MANATEE 的对比点）：①静态映射无法适应动态变化的局部性——对局部性随时间/代码空间变化的负载，非 hot 页每次访问都走慢速 NVM；②整 SPM JIT checkpoint 需为 failure-atomic 加密/恢复预留大量能量，放大 NVM 访问并牺牲大量收割能量。论文实测 Mapi-Pro 相对 Unsecure 开销达 4.9×，MANATEE 快 2–3×、STM32 4MB 数据集场景快 5.7×。
从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程例子：离线 profiling 程序 → 统计每页访问频率 → 建 ILP 模型（目标：最小化 NVM 访问×延迟等）→ 求解得 hot 页集 → 链接期把 hot 页放 SPM、cold 页留 NVM → 运行时 hot 页直达 SPM、cold 页每次访问都从 NVM 读（加解密）→ 断电时整 SPM 加密写 NVM、恢复时整 SPM 解密读回。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：profiling 工具 + ILP 求解器离线生成映射；运行时整 SPM checkpoint。它是"profiling-based approach"的代表（对比编译器直接方法 memory coloring [63] 与 MANATEE）。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM
