## Dartagnan（内存一致性模型求解器 / 弱内存有界模型检查器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dartagnan（MPI-SWS）是基于有界模型检查（BMC）的弱内存一致性求解器：接受 PPC/x86/AArch64 汇编与 C11 子集、兼容 herd 的 .litmus 格式，输入 CAT 内存模型，把"程序 + 模型"编码为 SMT 公式（Z3 求解），通过 relation analysis 剪枝关系图中无关边（示例中编码边从 221 降到 58），比旧版与 herd 快两个数量级；评测含 4,751 个 Linux litmus 测试与互斥算法。它回答的核心问题是"给定程序执行 trace，目标内存模型下是否存在合法执行"。Web 来源：CAV 2019 论文（https://link.springer.com/content/pdf/10.1007/978-3-030-25540-4_19 ）、仓库 https://github.com/MPI-SWS/dartagnan 。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 HARTBREAKER 中 Dartagnan 是验证链的最后一环（图 6）：RTL 仿真收集的 load 返回值被组装成等价 litmus 测试的 exists 断言 → Dartagnan 检查该结果在 RVWMO（CAT 模型）下是否可达 → 不可达即内存一致性 bug。论文还报告了验证容量的演进：自研 solver 只能处理 ~10 个内存操作（同步锚间距受限、求解极慢）；改用 Dartagnan 后扩展到 ~100 个内存操作——超过多数片上数据结构（store buffer、队列）规模，继续扩大不再带来新 bug（只复现已有 bug）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：命令行输入 .litmus 测试 + CAT 模型文件，输出 allowed/forbidden 判定与可视化（论文利用其可视化定位错误指令）；与 herd7 的差异在于 herd 对大规模执行集合扩展性差、Dartagnan 用 SMT 编码更高效。HARTBREAKER 中 herd7 存在但未接入 Docker 复现流水线（README 注明）。

涉及论文标题：
- HartBreaker: Deterministic Fuzzing of Multi-Hart RISC-V CPUs with Non-Deterministic Programs
