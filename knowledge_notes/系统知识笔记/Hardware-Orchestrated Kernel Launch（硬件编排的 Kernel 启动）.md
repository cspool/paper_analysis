## Hardware-Orchestrated Kernel Launch（硬件编排的 Kernel 启动）

术语是什么？
Hardware-Orchestrated (HO) Kernel Launch 是将 kernel 调度序列从 host CPU 软件 offload 到加速器硬件（如 AGCU）的机制。传统 Software-Orchestrated (SO) 模式下，host CPU 对每个 kernel 依次发出 Program Load → Argument Load → Kernel Execute 命令，host 往返延迟在 kernel 执行时间短的场景（如 autoregressive decoding）中占比显著。HO 模式下，编译器生成静态 kernel schedule 并预加载到 AGCUs，AGCUs 内置的硬件调度器按 schedule 自动发出 kernel 启动命令，消除 host 参与和 PCIe 往返延迟。在 SN40L 上的解码 benchmark 中，HO 相比 SO 提供 1.4×-8× speedup（短 kernel 场景），而 prefill/training（长 kernel）仅 1.1× 提升（因 kernel launch overhead 被长执行时间摊薄）。

从系统架构角度拆解：
```
SO模式: Host→PCIe→AGCUs: Program Load → Host→PCIe→AGCUs: Arg Load → Host→PCIe→AGCUs: Kernel Execute
  → [等待kernel完成] → Host→PCIe→AGCUs: 下一个kernel...  ← host往返每步都参与
HO模式: Host→PCIe→AGCUs: 一次性加载静态schedule
  → AGCUs内部: Kernel1 Execute → [done信号] → Kernel2 Execute → [done] → Kernel3 Execute...
  ← host仅在全部完成后被通知
```
关键差异：SO 模式的 host 往返延迟在 decode 阶段（单 kernel 执行时间可能仅微秒级）占比高；HO 将该部分消除，对 memory-bound 的 decode 场景效果显著。

术语一般如何实现？如何使用？
实现需要：(1) 硬件侧提供 kernel 调度状态机和命令 FIFO（如 SN40L AGCU 的 kernel launch 机制）；(2) 编译器生成包含 kernel 依赖关系和执行顺序的静态 schedule；(3) 硬件支持 kernel 完成信号（如 PCU counter done event）触发下一个 kernel。适用于大量短 kernel 串行执行的场景（如单 batch 自回归解码），不适合 kernel 间有动态依赖（需 host 运行时决策）的场景。

涉及论文标题：
- SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts
