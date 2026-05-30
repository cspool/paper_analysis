## Memory Liveness Analysis for GPU Multi-Stream Programs

术语是什么？
Memory Liveness Analysis（内存活跃度分析）是针对GPU多stream程序的内存对象生命周期分析技术。在HuntKTm的memory manager中，通过数据流分析确定每个GPU memory object的live range（从首次被kernel访问到末次被kernel访问的区间），然后将allocation/deallocation指令调度到live range边界处，缩短lifetime至live range。非重叠live range的memory object可复用同一块GPU内存，从而降低peak memory usage。

从kernel调度角度拆解术语：
HuntKTm memory liveness analysis流程（Algorithm 2）：

```
Step 1 - 数据流分析:
  for each kernel call:
    for each GPU pointer parameter:
      通过use-def chain追踪pointer指向的allocation指令
      → 同一allocation分配的所有内存视为一个memory object
      → 所有使用该object的kernel标记为依赖

Step 2 - Live range识别:
  invokeList = 所有依赖memObj的kernel (按原始顺序排序)
  liveRange = [invokeList[0], invokeList[last]]

Step 3 - 延迟Allocation (PostponeMalloc):
  instrList = memObj的allocation + 内容修改指令(cudaMemcpy等)
  insertPoint = invokeList[0]  // live range起点
  将instrList移到insertPoint之前, 转为异步版本
  分配到与insertPoint相同的stream
  添加跨kernel同步确保memObj在使用前已分配

Step 4 - 提前Free (对称算法):
  将cudaFree移到live range终点kernel之后

Step 5 - 冗余同步剪除
```

论文结果：M2从17.6GB降至11.2GB（-36.4%），DL从7.06GB降至4.70GB（-33.3%），平均memory reduction 22.3%。

术语一般如何实现？如何使用？
LLVM pass编译期实现。限制：当前仅处理每memory object最多一次host-device数据传输；多stream可能缩短执行路径减少memory reuse机会（如B&S的10个kernel分配到10个stream后同stream仅1个kernel，无reuse可能）。Peak memory runtime预测使用O(N)近似算法，取各stream最大累积delta memory之和作为上界。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
