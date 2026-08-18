# Algorithm 2: Semantic Conflict Detection

adapt scheduling policy(u);

```
Input: Instruction I with semantic annotations, in-flight
       semantic table Fu
foreach r ∈ Fu do
    if not same scope(I, r) then
        continue ;
    if semantic compatibility(I.OpType, r.OpType) then
        // Semantically compatible operations can overlap
        continue ;
    if memory range overlap(I, r) and true dependency(I,
     r) then
        if cannot reorder safely(I, r) then
            // True semantic conflict detected
            return true ;
// Safe to execute in parallel
return false ;
```

TABLE IV: Scheduler scaling with window size.

| W   | Latency  | Gates | Area (mm2<br>) | Power (mW) |
|-----|----------|-------|----------------|------------|
| 8   | 7 cycles | 1.5M  | 0.25           | 100        |
| 16  | 7 cycles | 2.0M  | 0.33           | 120        |
| 32  | 8 cycles | 2.8M  | 0.46           | 150        |
| 64  | 8 cycles | 3.9M  | 0.65           | 180        |
| 128 | 9 cycles | 5.2M  | 0.87           | 240        |
| 256 | 9 cycles | 6.8M  | 1.13           | 300        |

requires 1.5M gates (0.25 mm<sup>2</sup> , 1.5% per-core area, 100 mW). Scaling to W = 256 increases area sub-quadratically to 6.8M gates (4.5× for 32× entries) with bounded 9-cycle dispatch latency, due to logarithmic CAM structures and pipelined arbitration for W ≥ 32. Power remains <0.3% core power at W = 256 as dispatch is sparse (∼5% slots/cycle). The 8-entry baseline suffices for most operators; larger windows benefit only memory-bound kernels with latency, gates, area and power grow in Table IV.

The resulting execution pattern, as shown in Figure 2(c,e), demonstrates more compact cross-operator and cross-iteration concurrency than static approaches can achieve. Static pipelines rely on conservative synchronization assumptions– explicit barriers and fixed latencies (Figure 5)–to ensure correctness, fundamentally limiting overlap. While such pipelines achieve some concurrency, TISA's semantic awareness eliminates these rigid constraints: the scheduler dynamically resolves dependencies and reorders tiles without programmerinserted synchronization, yielding tighter overlap while keeping correctness.

# Algorithm 2: Semantic Conflict Detection

adapt scheduling policy(u);

```
Input: Instruction I with semantic annotations, in-flight
       semantic table Fu
foreach r ∈ Fu do
    if not same scope(I, r) then
        continue ;
    if semantic compatibility(I.OpType, r.OpType) then
        // Semantically compatible operations can overlap
        continue ;
    if memory range overlap(I, r) and true dependency(I,
     r) then
        if cannot reorder safely(I, r) then
            // True semantic conflict detected
            return true ;
// Safe to execute in parallel
return false ;
```

TABLE IV: Scheduler scaling with window size.

| W   | Latency  | Gates | Area (mm2<br>) | Power (mW) |
|-----|----------|-------|----------------|------------|
| 8   | 7 cycles | 1.5M  | 0.25           | 100        |
| 16  | 7 cycles | 2.0M  | 0.33           | 120        |
| 32  | 8 cycles | 2.8M  | 0.46           | 150        |
| 64  | 8 cycles | 3.9M  | 0.65           | 180        |
| 128 | 9 cycles | 5.2M  | 0.87           | 240        |
| 256 | 9 cycles | 6.8M  | 1.13           | 300        |

requires 1.5M gates (0.25 mm<sup>2</sup> , 1.5% per-core area, 100 mW). Scaling to W = 256 increases area sub-quadratically to 6.8M gates (4.5× for 32× entries) with bounded 9-cycle dispatch latency, due to logarithmic CAM structures and pipelined arbitration for W ≥ 32. Power remains <0.3% core power at W = 256 as dispatch is sparse (∼5% slots/cycle). The 8-entry baseline suffices for most operators; larger windows benefit only memory-bound kernels with latency, gates, area and power grow in Table IV.

The resulting execution pattern, as shown in Figure 2(c,e), demonstrates more compact cross-operator and cross-iteration concurrency than static approaches can achieve. Static pipelines rely on conservative synchronization assumptions– explicit barriers and fixed latencies (Figure 5)–to ensure correctness, fundamentally limiting overlap. While such pipelines achieve some concurrency, TISA's semantic awareness eliminates these rigid constraints: the scheduler dynamically resolves dependencies and reorders tiles without programmerinserted synchronization, yielding tighter overlap while keeping correctness.

