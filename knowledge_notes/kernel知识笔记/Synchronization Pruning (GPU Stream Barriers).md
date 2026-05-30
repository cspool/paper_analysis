## Synchronization Pruning (GPU Stream Barriers)

术语是什么？
Synchronization Pruning（同步剪除）是HuntKTm的synchronization generator中的优化技术。当kernel被分配到多个CUDA stream后，必须在有数据依赖的跨stream kernel之间插入CUDA event同步（barrier）。但并非所有依赖都需要显式同步——许多barrier因为依赖传递性（transitivity）和同stream内隐式串行执行而冗余。Synchronization pruning通过三步算法识别并移除这些冗余barrier，只保留最小必要的同步集。

从kernel调度角度拆解术语：
HuntKTm同步剪除算法的三步（以kernel K为目标）：

```
Step 1 (创建初始barrier):
  for each predecessor P of K:
    if P.stream != K.stream:
      createBarrier(P → K)

Step 2 (每stream保留最后一个前驱):
  for each stream S:
    predecessors_in_S = K在S中的所有前驱
    if predecessors_in_S非空:
      仅保留最后执行的前驱的barrier, 移除其余
      // 理由: 同stream FIFO保证更早的前驱一定先于最后前驱完成

Step 3 (隐式同步传递性剪除):
  for each kernel T in K.stream (T在K之前):
    for each predecessor P of K:
      if P在T的stream中 AND P在T之前执行:
        removeBarrier(P → K)
        // K等待T(同stream) AND T等待P(同stream) → K隐式等待P
```

论文Figure 6(b)展示了剪除效果：实线为保留barrier，虚线为被剪除barrier。相比naive为每对依赖创建barrier，剪除后barrier数量显著减少。

术语一般如何实现？如何使用？
编译期LLVM pass实现，具有全局DFG视图——这是相比GrSched（运行时无全局视图）的关键优势。剪除后的CUDA event同步通过`cudaEventRecord`和`cudaStreamWaitEvent` API插入。正确性保证：仅移除那些执行顺序已由其他同步路径隐式保证的barrier。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
