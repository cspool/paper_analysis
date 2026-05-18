## Multi-Version Micro Kernel Generation

术语是什么？通过联网搜索让回答具体和精准。
Multi-Version Micro Kernel Generation是Infera compiler为每个micro operator生成多种kernel candidates的策略：通过配置不同的register使用上限（64/96/128）、shared memory上限（48/80/112/144 KiB）、pipeline stage数（2/3/4）、以及spatial/reduction axis tile size组合，为同一micro operator生成多个ILP/TLP/arithmetic intensity trade-off不同的kernel版本。推理时scheduler根据GPU并发状态、kernel hazard和data dependency选择最优kernel版本。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Multi-version kernel generation的配置空间和trade-off：
```
Config Parameter        Values              Controls
─────────────────────────────────────────────────────
Register/thread        64/96/128            TLP vs ILP
                                            (more regs→higher ILP, lower TLP)
Shared memory/block    48/80/112/144 KiB    Tile size & Intensity
Pipeline stages        2/3/4               Global↔shared overlap
Spatial tile size      derived from above   Data processed per block
Reduction tile size    kernel argument      Data processed per reduction axis
Grid size              fixed 64            Global-level parallelism
```
Trade-off formalization: 执行时间 ∝ #inst/IPC。ILP增加→#inst减少但可能降低occupancy(TLP)；Intensity增加→#inst减少但需要更多register/shared memory存储tile数据，压缩TLP。三者竞争register和shared memory资源→需multi-version覆盖Pareto frontier。Infera通过Eq.(4)形式化trade-off: min #inst s.t. ILP×TLP×intensity constraints。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera中，multi-version kernels以固定4 mainloop warps + 4 copy warps结构生成，配置参数在编译期确定。每个kernel版本附带metadata（#inst、register usage、shared memory usage、grid/block dims），供runtime SelectKernels使用。与Ansor/MetaSchedule的搜索式tuning不同，Infera的multi-version generation是zero-tuning的：直接从配置参数推导tile size和resource allocation，无需GPU profiling。候选kernel数量由配置组合数决定（register 3 × shared mem 4 × pipeline 3 = 36种基础配置，部分组合受资源约束不可行），编译完全并行化。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
