## Triton MPP (Multi-Pass Profiler)

术语是什么？
Triton MPP (Multi-Pass Profiler)是Meta开发的编译器中心的统一GPU profiling框架，解决现代AI accelerator profiling的核心挑战：性能信号分散在多个abstraction layers（Triton DSL → MLIR IR → PTX/SASS → runtime APIs → hardware counters）之间，每个tool有incompatible interfaces和vendor-specific assumptions，profiling结果面向human interpretation（textual reports/dashboards）而非structured automation。MPP将instrumentation、compiler transforms、profiling passes和trace synthesis组合为composable job graph tasks，提供unified structured profiling abstraction。

从编译框架角度拆解术语：
MPP的核心创新——minimally-invasive profiling for async GPU behavior：
```
传统Profiling Pipeline:
  Triton IR tracer → textual output
  NCU profiler      → vendor-specific dashboard
  NVBit instrument  → raw binary traces
  ↑ 每个stage需要手动orchestration、brittle text parsing和人工correlation

MPP Compiler-Centric Job Graph:
  [MLIR instrumentation injection] → [profiling passes] → [trace synthesis] → [structured output]
  
Minimally-Invasive Profiling:
  Direct wait insertion → disrupts async overlap → cascading timing changes → inaccurate
  MPP替代方案:
    1. Capture unmodified base traces (no perturbation)
    2. Apply targeted probe passes (single instruction isolation in specific iterations)
    3. Guard instrumentation (prevent interference)
    4. Fuse results for attribution
  → 在TTGIR level profiling warp-group operations, async copies, TMA transfers
     with negligible perturbation
```

对于现代Triton kernels（使用TMA operations, warp-specialized pipelines, overlapped data movement），传统profilers粗粒度暴露async行为、无法揭示instruction-level memory-computation overlap，MPP解决了这一gap。

术语一般如何实现？如何使用？
MPP集成到KernelEvolve evaluation framework中，通过evaluation code generator自动generate MPP-instrumented scripts。为context memory sub-agent提供structured instruction-level performance data（无需vendor-specific parser），使agent能够进行precise bottleneck diagnosis——例如识别register spilling vs bank conflicts vs TMA async overlap insufficiency——并据此生成targeted optimization recommendations。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
