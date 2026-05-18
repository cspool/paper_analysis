## dGraph, dTask, and dEngine: Data-Property-Aware Diffusion Pipeline Decomposition and Scheduling（数据属性感知的扩散pipeline分解与调度）

术语是什么？通过联网搜索让回答具体和精准。

dGraph、dTask和dEngine是Difflow(ChituDiffusion)提出的三层数据属性感知扩散模型serving抽象：(1) **dGraph**（Data-Property Graph）：扩散pipeline按数据属性局部性分解的子图——consecutive operators with identical output property expressions被归为同一dGraph，每个dGraph内所有操作共享相同优化条件；(2) **dTask**：运行时将用户请求按dGraph分解得到的细粒度任务——一个request的pipeline被分解为多个dTask（每个dGraph一个），dTask携带输入数据属性信息；(3) **dEngine**（Data-Property-Specialized Engine）：编译时为每个dGraph的不同输入数据属性配置（如redundant vs non-redundant prompts, uniform vs ragged shapes）编译的专用执行引擎——每个dEngine针对uniform data properties做了最大化优化。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Difflow中dGraph→dTask→dEngine的运转流程：

```
// === 编译时 (Compile-Time) ===
1. App pipeline DFG → Symbolic property propagation
   // 每个operator的输入/输出维度redundancy表达为boolean symbolic variables
2. Operators with identical output property expressions → grouped into dGraphs
   // 例如：SDXL U-Net分解为3个dGraphs (input-dependent / input-independent / post)
3. for each dGraph:
4.     enumerate input property conditions (e.g., prompt redundant? T/F)
5.     prune unsatisfiable + inessential (speedup <5%) conditions
6.     for each surviving condition:
7.         compile specialized dEngine (with redundancy elimination + ragged regularization)
8.     store dEngines in library keyed by property requirements

// === 运行时 (Runtime) ===
9. Requests arrive within scheduling window → decomposed into dTask pool
10. for each dGraph (in execution order):
11.     unique_dTasks = merge dTasks with identical inputs
12.     // DP scheduling (Algorithm 1):
13.     for each dEngine ∈ dEngines[dGraph]:
14.         B = GetLargestBatch(remaining dTasks matching dEngine's property requirements)
15.         t = Search(g, remaining - B) + EstTime(g, dEngine)
16.         // EstTime uses OLS regression performance model (R²=0.998)
17.     select optimal plan with minimum estimated execution time
18.     dispatch batched dTasks to matched dEngine executor
19.     // Asynchronous property inference overlaps scheduling with execution
```

以edit应用(SDXL Turbo + ControlNet + 16 LoRA styles)为例：
- 编译时：SDXL U-Net分解为3个dGraphs
- 运行时：dGraph1仅有2 unique dTasks（2 unique input images），dGraph2有3 unique dTasks（不同conditioning combinations）→ DP scheduler选择最优dEngine组合→ data-aware batching dispatch

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Difflow开源实现：https://github.com/thu-pacman/chitu/tree/Diffusion。dGraph decomposition通过symbolic boolean variable propagation rules覆盖elementwise/linear/convolution算子实现。Selective dEngine generation通过satisfiability checking (SAT)剪枝conflicting conditions + 5% speedup threshold剪枝inessential conditions。dTask scheduling通过DP with memorization（O(|R|·|L|) per dGraph，R=unique dTasks, L=dEngines）。SDXL U-Net编译：monolithic=16384 engines/11天→dGraph=7 engines/7分钟。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

---
