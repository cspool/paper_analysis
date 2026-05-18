## Accelerator Unit Variations (AUV)

术语是什么？

AUV (Accelerator Unit Variations) 是在AU-enabled CPU共享环境下，AU表现出的三维度可变行为，阻碍直观的资源共享管理。由AUM论文系统化地定义为三个抽象层：(1) **Variation-1: Variable Usage Pattern** (软件层)——不同application和operator的AU使用率高度动态，如LLM prefill AMX cycle ratio 14.4% vs decode 1.5%，big-matrix GEMM AMX有效而small-matrix GEMV用AVX更高效；(2) **Variation-2: Compulsory Frequency Interference** (系统层)——AU高功耗触发TDP-based频率降频，prefill→2.5GHz, decode→3.1GHz，不同AU使用率的core混合共享时频率干扰不稳定；(3) **Variation-3: Dissimilar Resource Bound** (微架构层)——AU的微架构资源需求与传统功能单元不同，AU frontend bound极低(1%)但backend bound高(prefill 92%)，不同AU使用率下LLC/memory bandwidth亲和性变化大。

从系统架构角度拆解术语：

AUV使AUV-oblivious resource managers失效的流程：
1. **SMT sharing fails**：AU application性能因memory contention degradation >200%，co-running OLAP slowdown >40%（Figure 9a），compute-intensive co-runner自身40% degradation被频率拖累（Figure 9b）——因为SMT无法感知AU usage变化和频率interference
2. **RP sharing fails**：单独隔离L2/LLC/BW仅轻微减轻AU slowdown但无法达到最优（Figure 10），因为AU的critical backend bound随资源类型变化——单一维度分区不足以应对三维AUV
3. **Exclusive AU waste**：独占式AU使用避免AUV但留下40-50% CPU idle cores

术语一般如何实现？如何使用？

AUM通过三维度对应三阶段解决AUV：
- Usage-aware: 用ARI (Arithmetic Intensity) 指标选择AMX vs AVX
- Frequency-aware: 将CPU core分为High/Low/None-AU三frequency region
- Bound-aware: 用CAT/MBA profiling minimal resource demand R_AU，再runtime adaptively tune
- AUV最终汇总为离散AUV Model table（含U_AU/C_AU/F_AU/R_AU/P_a/P_t），供online lookup

涉及论文标题：
- AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving
