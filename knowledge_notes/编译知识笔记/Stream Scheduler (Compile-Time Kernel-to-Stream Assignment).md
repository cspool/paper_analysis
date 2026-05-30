## Stream Scheduler (Compile-Time Kernel-to-Stream Assignment)

术语是什么？
Stream Scheduler 是 HuntKTm 中的编译期 LLVM pass，自动将串行 CUDA 源码转换为多 stream 并发程序。它接收用户编写的串行 kernel 调用序列，自动分析数据依赖、构建 DFG、分配到多个 CUDA stream、并生成最小同步指令集。开发者只需在每个 kernel 参数列表前添加一个常量标注前 N_out 个参数为 writable（约每 kernel 一行 LoC），stream scheduler 自动完成剩余转换。

从编译框架角度拆解术语：
HuntKTm stream scheduler 在编译框架中的工作流程：

```
输入：串行 CUDA 源码（kernel 按顺序 launch）

Step 1: DFG Constructor (LLVM pass)
  - 在 host IR 中定位 __cudaPushCallConfiguration 调用模式 → 识别 kernel launch
  - 通过 writable 参数标注区分 read-only 和 writable 参数
  - 逆序遍历 kernel 调用序列，BFS 识别每个 kernel 的直接前驱
  - 判断 RAW/WAR/WAW 依赖：若两个 kernel 访问同一数据对象且至少一个是 write → 建边
  - Pointer aliasing 处理：同基址派生的指针视为访问同一数据
  - 输出 DFG

Step 2: Kernel Distributor (LLVM pass)
  - DFG 分层（levelization）：同一层内 kernel 无数据依赖
  - 定义 PP-Set：kernel 所有前驱中位于 stream 末尾的子集
  - Level-by-level 分配：
    规则∂: 无前驱 kernel 按 round-robin 均匀分配
    规则∑: 单前驱 kernel 放在同一 stream（避免跨 stream 同步）
    规则∏: 多前驱 kernel 放在 PP-Set 中未被调度后继最少的前驱所在 stream
  - 优先调度 PP-Set 小的 kernel

Step 3: Synchronization Generator (LLVM pass)
  - 三步剪除冗余 barrier：
    为每个不在同 stream 的前驱创建 event 同步
    每个 stream 仅保留最后前驱发出的同步
    利用同 stream 隐式执行顺序 + 依赖传递性消除更多冗余
  - 输出 stream graph（多 stream 执行计划 + 最小同步集）
```

编译框架视角的核心洞察：编译期拥有完整的全局 DFG 视图，使同步剪除能比运行时方法（如 GrSched 无法获取全局图）更激进地消除冗余 barrier。

术语一般如何实现？如何使用？
基于 LLVM 14.0.6 实现为自定义 LLVM pass。所有 pass 操作于 host IR。DFG constructor 的 BFS 算法在 kernel 数量上为 O(N²)（worst-case），实际应用通常低于此。同步生成器基于三步遍历，复杂度 O(N×S)（S=stream 数，实际 ≤10）。编译时间从平均 1.41s 增至 2.40s（7 benchmark 均值）。用户仅需约 15 LoC / 49 token 的代码修改（每 kernel 一行标注），无需引入新编程框架。设计可移植到 HIP、SYCL 等框架。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
