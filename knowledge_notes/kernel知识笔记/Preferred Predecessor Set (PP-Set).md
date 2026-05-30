## Preferred Predecessor Set (PP-Set)

术语是什么？
Preferred Predecessor Set (PP-Set) 是HuntKTm的kernel distributor中使用的启发式概念。对于一个尚未被调度的kernel，其PP-Set定义为：该kernel的所有前驱（predecessor）中，当前位于各自所在stream末尾（即stream中的最后一个已调度kernel）的那些前驱组成的子集。PP-Set的大小直接影响该kernel的调度优先级——PP-Set越小的kernel越先被调度，以最小化跨stream同步的数量。

从kernel调度角度拆解术语：
PP-Set在kernel分配算法中的作用——以论文Figure 6的DFG为例：

```
初始: Stream1=[], Stream2=[], Stream3=[]
Level 1 (kernel A, B, C无前驱):
  Rule ∂ (round-robin): A→S1, B→S2, C→S3
  所有kernel PP-Set=∅

Level 2 (D依赖B; E依赖A,C; F依赖C):
  第1轮排序:
    F: PP-Set={C}, size=1
    D: PP-Set={B}, size=1
    E: PP-Set={A, C}, size=2
    → 先调度F (Rule ∑, 单前驱C → 同Stream3)
  
  PP-Set更新: C不再是Stream3末尾(F现在是末尾)
    E: PP-Set={A}, size=1
    D: PP-Set={B}, size=1
    → 调度D (Rule ∑ → D放入Stream2, 同B)
  
  最后调度E: PP-Set={A} → Rule ∑ → E放入Stream1 (同A)

最终: S1=[A,E], S2=[B,D], S3=[C,F]
跨stream同步: 仅需E→C的barrier (D和F无额外同步)
```

术语一般如何实现？如何使用？
PP-Set是HuntKTm kernel distributor内部的启发式数据结构，在每次kernel调度后动态更新。核心洞察：将kernel放在其PP-Set中某个前驱所在的stream中，可以避免为该前驱创建跨stream同步（因为同stream内的串行执行已隐式保证顺序）。优先调度PP-Set小的kernel，给予它们更多灵活的stream选择空间。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
