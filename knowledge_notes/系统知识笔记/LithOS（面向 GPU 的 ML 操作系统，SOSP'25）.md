## LithOS（面向 GPU 的 ML 操作系统，SOSP'25）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LithOS 是 CMU（Coppock、Kypriotis、Solomon、Skarlatos 等，与 Meta 合作）在 SOSP'25 提出的 GPU 操作系统，面向"高效 ML 的多租户 GPU 执行"。核心能力：TPC 级空间调度（TPC stealing 在工作负载间动态搬移资源）、透明 kernel atomization（把 kernel 原子化以减少队头阻塞、支持动态资源重分配）、硬件 right-sizing（动态确定每 atom 的最小 TPC 资源）、透明电源管理（DVFS，基于 in-flight 工作的频率缩放行为）。论文报告：相对 NVIDIA MPS 尾延迟低 13×、吞吐高 1.6×、约 25% 容量与能量节省。它是 PowerWeave（ISCA'26）的构建基础：PowerWeave 的 Interposer（~5500 行 Rust）与 Governor（~250 行 Python）直接扩展 LithOS 实现。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PowerWeave 中的角色：LithOS 提供 TPC 粒度空间多租户（把 GPU 按 TPC 分给多个应用），但 DVFS 仍按整卡单一频率域运行——它根据所有 collocated 模型的需求选一个 device-wide 频率，这正是 PowerWeave 要破解的 baseline（论文以 LithOS 为 SOTA baseline，其平均节能仅 13% vs PowerWeave 28%）。PowerWeave 在其上叠加：透明 CUDA 调用拦截、每域 kernel 画像、每域频率控制、用户态 Governor。MPS 部署下 PowerWeave 沿用 LithOS 的 TPC assignment。
- 运转流程例子：vLLM serving 多模型 → LithOS TPC scheduler 把 74 个 TPC 分给 3 个租户（18/19/37）→ LithOS 全卡选一个频率（按最苛刻租户的 SLO 需求）→ decode 租户被迫继承 prefill 租户需要的高频率（浪费）→ PowerWeave 替换为每租户域独立频率（各域按自己 SLO slack 调频，平均节能 28% vs 10%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Rust 实现的操作系统层，位于 GPU driver 之上、应用/框架之下，基于 NVIDIA MPS 机制做 TPC 级空间调度（slides: http://www.cs.cmu.edu/~dskarlat/slides/lithos_sosp25_slides.pdf；论文 DOI 10.1145/3731569.3764818；开源仓库截至 2026-08 未确认公开）。使用方式：作为多租户 GPU 共享与电源管理基础设施，应用/框架无需修改即可获得空间隔离与 DVFS；PowerWeave 等后续系统在其上扩展更细粒度的空间 DVFS 控制平面。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
