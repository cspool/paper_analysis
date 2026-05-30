## Event-Based Intermediate Representation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Event-Based IR是Cypress编译器的核心中间表示（Figure 7），用于编码异步操作之间的依赖关系。每个异步操作（copy、task invocation/call）产生一个event表示该操作的完成；每个异步操作声明一组precondition events——必须完成的先行操作集合。IR采用SSA（单静态赋值）形式，确保任何有效的操作排序都满足所有event依赖。

Event类型的独特设计：(1) `()` — unit event，表示单次操作完成；(2) `[(N, p), ...]` — event数组，每个维度标注大小N和processor kind p（如WARP、THREAD），由并行循环（pfor）产生——每个数组元素对应一个并行迭代的完成event。Event数组支持索引：(a) 整数索引提取特定迭代的event；(b) broadcast索引`[:]`表示所有维度元素都完成——即等待所有并行迭代完成。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Event-Based IR在Cypress编译器中运转示例（来自论文Figure 8-9）：

```
Dependence Analysis (生成Event IR):
  clear_inner task (warp-level, 4 warps, 32 threads/warp):
    e1: [(4, WARP)] = pfor i in [0, 4), {} do
      CW = tensor([M/4, N], NONE)
      CWp = partition(CW)
      e2: [(32, THREAD)] = pfor j in [0, 32], {} do
        CR = tensor([...], RMEM)
        e3: () = call(clear_thread, CR), {}
        e4: () = copy(CR, CWp[j]), {e3}     // e3 precondition
        yield e4                              // e4 = completion event
      e5: () = copy(CW, C1p[i]), {e2[:]}    // e2[:] = all threads done
      yield e5
    e6: () = copy(C1, C), {e1[:]}            // e1[:] = all warps done

Vectorization (flatten pfor loops):
  i = warp_id(), j = thread_id()
  e3: [(4, WARP), (32, THREAD)] = call(clear_thread, CR), {}
  e4: [(4, WARP), (32, THREAD)] = copy(CR, CWp[j]), {e3[i, j]}
  e5: [(4, WARP)] = copy(CW, C1p[i]), {e4[i, :]}
  e6: () = copy(C1, C), {e5[:]}

CUDA C++ Generation (event→sync lowering):
  e3[i,j]: point-wise → 移除 (SSA保证)
  e4[i,:]: THREAD broadcast → __syncwarp
  e5[:]: WARP broadcast → named barrier
  TMA events → shared memory barrier (mbarrier)
  Tensor Core events → warpgroup sync assembly
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Event-Based IR的关键实现考量：
- 仅存在于编译时（compile-time construct），不存在于生成的代码中——无运行时动态依赖跟踪开销
- Event数组在vectorization后被promote维度、在code generation后被消除或lowering到硬件同步原语
- SSA形式保证依赖顺序的语义正确性，后续pass（vectorization, copy elimination, warp specialization, pipelining）只能增加依赖、不能移除已有的正确依赖
- 事件依赖类似于传统编译器中的def-use/use-def chains，但针对异步操作显式建模并行迭代的完成和同步

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
