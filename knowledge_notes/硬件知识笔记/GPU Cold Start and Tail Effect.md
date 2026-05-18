## GPU Cold Start and Tail Effect

术语是什么？通过联网搜索让回答具体和精准。
GPU Cold Start和Tail Effect是DNN inference中连续kernel执行时GPU pipeline的两种idle现象。Cold Start发生在后续kernel的开始阶段，源于kernel preamble：thread block dispatching（GigaThread Engine分配CTA到SM）、resource allocation（register file/shared memory分配）和prologue pipeline bubbles。Tail Effect发生在前序kernel的结束阶段，源于insufficient threads（部分CTA先完成退出，剩余CTA不足以占满SM）和epilogue pipeline bubble。Infera在A100上的测量显示这两类GPU idle intervals在1-3μs范围。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
GPU kernel执行timeline中的Cold Start和Tail Effect：
```
Time →
Kernel_1: [Preamble][Compute................][Epilogue]
                                                ← Tail Effect (remaining CTAs phase out)
          [GPU Idle 1-3µs]
Kernel_2:          [Preamble][Compute................][Epilogue]
                    ← Cold Start (thread blocks dispatch + resource alloc)
```

**Tail Effect具体机制**：
- Kernel_1的CTA（Cooperative Thread Arrays）陆续完成
- GigaThread Engine逐个释放完成的CTA的SM资源
- 最后几个CTA运行时，大部分SM已空闲
- Epilogue阶段的pipeline bubble（等待最后memory transactions完成）使剩余SM也可能partial idle

**Cold Start具体机制**：
- Kernel_2的grid launch → GigaThread Engine开始dispatch CTA到SM
- 每个SM收到CTA后进行资源分配（register file partitioning, shared memory allocation）
- Prologue阶段：warp scheduler初始化，pipeline filling（指令cache miss、constant memory load等）
- 在足够多CTA被dispatch并进入steady compute前，GPU pipeline units利用率低

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera通过tile-based cooperative scheduling解决cold start/tail effect：(1) warp-level kernel fusion将多个primitive kernel融合为一个fused kernel，消除kernel边界和launch间隙，从而消除inter-kernel cold start/tail effect；(2) daemon kernel device-side fire-and-forget launch（<10μs）比传统host-side launch（含stream tracking overhead）更快，减少launch间隙。测量工具：NVIDIA GPU profiling tools可捕获stall cycles和active cycles比例，论文Figure 1展示了ResNet-50 + TVM在A100上的低GPU utilization（active cycles仅部分时段），cold start/tail effect是贡献因素。随着GPU core数增加（如A100 108 SM → H100 132 SM），这些overhead更加显著[2]。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

