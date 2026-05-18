## DRAM Page Policy (Open/Close/Adaptive Page)

术语是什么？

DRAM Page Policy 决定在完成一次 row access 后，是否立即 precharge（关闭行）还是保持行 open 等待可能的后续访问。三种主要策略：(1) Open-page policy：保持行 open，后续对同一行的访问可直接发 RD/WR 无需 ACT，降低延迟但占用 row-buffer；(2) Close-page policy：每次 access 后立即 PRE，简化控制但每次访问都需 ACT；(3) Adaptive page policy：动态监测访问模式（如 row-buffer hit rate），在 open 和 close 之间切换。传统 MC 的 page policy 需与 bank interleaving 和 refresh 机制协调。

从硬件架构角度拆解术语：

在传统 HBM4 MC 中，page policy 的工作流程：发 ACT 打开 row → 发 RD/WR 完成数据访问 → 检查是否有 pending request 到同一 row（row-buffer hit）→ 如果有 hit，保持 open；如果没有 hit 或 open 时间过长（可能阻塞其他 bank），发 PRE 关闭 → 在 open-page 和 close-page 之间动态决策。MC 需 per-bank 跟踪当前 open row address，并在做调度决策时检查 row-buffer hit/miss。Adaptive page policy 的实现需要额外的监测逻辑和阈值参数。RoMe 由于使用 row granularity access（每次访问整行），天然保证 row-buffer locality——一次 RD_row 读完行内所有数据后立即自动 PRE，从 MC 视角完全消除了 page policy 的需求和对应硬件逻辑。

术语一般如何实现？如何使用？

Page policy 在 MC 硬件中实现，通常与 command scheduler 紧密耦合。Open-page policy 适用于高空间局部性 workload（如 LLM 的顺序大块访问），close-page 适用于随机访问。GPU MC（如 NVIDIA H100 的 HBM controller）多使用 open-page 或 adaptive page policy 以最大化 row-buffer hit rate。RoMe 的 baseline MC 使用 open-page policy；RoMe MC 则因 row granularity 本身消除了 page policy 的需要。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

