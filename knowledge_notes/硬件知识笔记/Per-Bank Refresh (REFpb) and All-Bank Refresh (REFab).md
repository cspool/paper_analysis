## Per-Bank Refresh (REFpb) and All-Bank Refresh (REFab)

术语是什么：

DRAM cell 需要通过周期性刷新来保持数据，因为存储电荷会随时间泄漏。两种刷新命令：(1) REFab (Refresh All Banks)：一条命令对所有 bank 同时执行刷新，期间整个 rank/channel 无法访问；(2) REFpb (Per-Bank Refresh)：只刷新单个 bank，其他 bank 可继续服务请求。REFpb 减少了对并发访问的阻塞，但要求 MC 以更细粒度跟踪每个 bank 的刷新状态，增加了调度复杂度。

从硬件架构角度拆解术语：

在传统 HBM4 MC 中，refresh scheduler 按 tREFI 间隔发刷新命令。使用 REFpb 时：MC 维护 per-bank refresh counter，按 tREFIpb 间隔（= tREFI / #banks）依次刷新各 bank；每次 REFpb 阻塞单个 bank 约 tRFCpb（~280ns），其他 bank 可继续服务。这比 REFab（阻塞所有 bank ~tRFCab）提供更好的 bandwidth availability。RoMe 的优化：由于每个 VBA 包含两个 physical bank，若分别发 REFpb 会导致 VBA 被阻塞 2×tRFCpb。RoMe MC 改为每 2×tREFIpb 发一次 per-bank refresh，由 command generator 对 VBA 内两个 bank 间隔 tRREFD 各发一个 REFpb，将 VBA stall 从 2×tRFCpb 降到 tRFCpb+tRREFD。

术语一般如何实现？如何使用？

Refresh 机制由 JEDEC DRAM 标准定义，在 DRAM 芯片内部实现 refresh counter 和 timing。MC 的 refresh scheduler 按 tREFI（通常 ~3.9µs for HBM）产生 refresh 请求，与正常 RD/WR 请求竞争调度。现代 MC 通常采用 "opportunistic refresh" —— 在 bank idle 时提前发 refresh，避免与高优先级 RD/WR 冲突。RoMe 论文的 baseline 和 RoMe 均使用 REFpb 以提供更好的 bandwidth availability。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

