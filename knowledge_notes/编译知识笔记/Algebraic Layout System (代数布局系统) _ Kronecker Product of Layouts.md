## Algebraic Layout System (代数布局系统) / Kronecker Product of Layouts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
代数布局系统（Algebraic Layout System）是Tilus提出的一种形式化描述GPU register tensor元素跨线程分布的方法。其核心思想是用两种primitive layout的Kronecker product（Kronecker积）构建任意复杂的tensor layout。

两种primitive layout：
- **local(n1, n2)**：所有 n1×n2 个元素存储在单个线程中，映射函数 f(t,i) = (i/n2, i%n2)
- **spatial(n1, n2)**：所有元素分布在 n1×n2 个线程中，每线程一个元素

Kronecker product定义：给定两个layout f, g，其Kronecker积 h = f⊗g 为 h(t,i) = f(t/Tg, i/Ng) \* Sg + g(t%Tg, i%Ng)，其中 Tg, Ng, Sg 分别是g的线程数、每线程元素数和shape。该操作满足结合律但不满足交换律。

实际例子：Tensor Core mma.m16n8k16指令中operand A的layout可表示为 local(2,1)⊗spatial(8,4)⊗local(1,2)，将16×8=128个元素分布在32个线程中，每线程4个元素。Tilus还引入了unified layout representation：每个layout有四个属性——shape、mode_shape（维度拆分后的子维度列表）、spatial_modes（分配给线程的子维度）和local_modes（分配给线程内局部存储的子维度），通过split→distribute→merge三步完成逻辑索引→(thread_index, local_index)映射。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。
在Tilus编译流程中：1) 编程阶段：开发者用Kronecker product表达式指定register tensor的layout；2) 指令选择阶段：编译器根据layout兼容性选择最高效的PTX指令——若目标register tensor的layout与spatial(8,4).repeat(1,4)兼容则使用ldmatrix指令，否则fallback到lds指令；3) 类型降低阶段：layout系统实现零开销的register tensor reinterpretation——当两个tensor的线程数和每线程bit数相同时，View指令可在无数据移动的情况下同时改变类型和layout，这是Tilus消除Triton shared memory layout conversion瓶颈的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
layout系统在Tilus中通过C++/Python实现：Python DSL层提供local()、spatial()、column_spatial()等API；编译器在IR中维护每个register tensor的layout属性；Kronecker product的代数性质（结合律）被用于layout简化和优化。该layout系统closed under Kronecker product，即两个layout的Kronecker积结果可用同一表示形式表达，使编译器可以推导layout等价性和兼容性。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

---
