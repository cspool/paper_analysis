## Hidet IR (Task-Mapping Intermediate Representation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hidet是Yaoyao Ding等人在ASPLOS 2023提出的深度学习编译器，核心是task-mapping programming paradigm。Hidet IR是一种类CUDA C的中间表示，将调度过程直接嵌入tensor程序中——开发者通过task mapping指定每个算子的计算分配和顺序。与传统declarative scheduling（如TVM的compute/schedule分离）不同，Hidet允许program-statement-level的细粒度优化。在Tilus中，Hidet IR被用作代码生成的后端。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。
在Tilus中Hidet IR充当VM IR和CUDA C之间的桥梁：VM IR → Code Emitting（每条VM指令展开为Hidet IR语句序列）→ Low-Precision Lowering（在Hidet IR上应用bitwise操作展开、PRMT/LOP3选择）→ Hidet codegen生成CUDA C → nvcc编译为.cubin binary。选择Hidet作为后端的原因是它提供比直接生成PTX更高级的抽象（便于实现低精度操作的类型降低pass），同时保留足够的低级控制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hidet通过torch.compile(backend='hidet')或独立API使用，开源地址：https://github.com/hidet-org/hidet，Apache 2.0 license。Tilus在此基础上增加了约35K行代码实现DSL、VM IR、优化passes和低精度类型支持。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

---
