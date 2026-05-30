## Thread-Block-Level Programming Model (SIMB / Single-Instruction-Multiple-Block)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Thread-Block-Level Programming Model是Tilus提出的GPU编程抽象，将程序操作定义在thread block粒度而非传统CUDA的thread粒度，称为SIMB（Single-Instruction-Multiple-Block）。所有变量（scalar、pointer、tensor）都在thread-block级别操作，block内所有线程协作分配和维护状态。Tilus VM的指令集全部在thread-block级别定义，包括：Tensor Allocation（AllocateGlobal/AllocateShared/AllocateRegister）、Tensor Transfer（LoadGlobal/LoadShared/StoreGlobal/StoreShared/CopyAsync）、Register Tensor Computation（Add/Sub/Mul/Div/Mod、Cast、View、Dot）、Control/Sync（Synchronize/Exit/BlockIndices）。与PTX/CUDA thread级编程的关键区别：开发者用View(dtype, layout)改变整个register tensor的类型和layout，编译器自动处理各线程的相应操作，无需手动管理per-thread寄存器和同步。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。
编译流程：1) Python DSL → VM IR：Python Tilus程序被翻译为VM IR指令序列；2) Code Emitting：编译器逐条将VM指令展开为thread级Hidet IR代码（如Dot指令展开为各warp的mma.m16n8k16 PTX指令）；3) 自动向量化：根据数据访问pattern自动选择向量化宽度（cp.async.v4, lds128, ldg128）；4) 同步管理：开发者只需写Synchronize()，编译器在必要时插入__syncthreads()。CopyAsync/CopyAsyncCommitGroup/CopyAsyncWaitGroup指令组合实现global→shared→register三级流水线的声明式控制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tilus的SIMB模型通过约35K行Python和C++代码实现。开发者用Python编写Tilus程序（含grid shape、tensor views和VM指令序列），运行时系统管理kernel的动态加载和执行。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

---
