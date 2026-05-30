## Memory Layout Composition (内存布局组合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Memory Layout Composition 是 TileLang 基于 IterVar 代数系统的 composable Layout 抽象，用于描述多维数组索引到物理内存地址的转换过程。一个 Layout 在物理地址层面表示为线性地址表达式：∑_i y_i × s_i，其中 y_i 是第 i 维的索引，s_i 是 stride。TileLang 的 Layout 定义包含 iter_vars（可选 range 信息）和 forward_index 表达式，共同构成代数函数 f: K^n → K^m，编码从高层索引到内存地址的转换。Layout 的关键特性是 composable（可组合）——多个 Layout transformation 可以堆叠（stack），形成链式的地址转换。例如：原始多维索引 → swizzle layout → padding layout → 线性内存地址。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Layout Composition 在 TileLang 编译框架的三个层面发挥作用：
```
1. Shared Memory 层面（避免 bank conflict）:
   T.gemm 对 A_shared, B_shared 默认应用 MakeSwizzleLayout
   Layout(padding(dim=0, factor=8)) ∘ Layout(swizzle(bits=3))
   → 产生 bank-conflict-free 的 shared memory 访问 pattern

2. Register File 层面（Fragment Layout 推断）:
   多维 buffer 索引 → Fragment Layout → (thread_id, register_id)
   例如: C_local[m_idx, n_idx] → forward_index: 
     thread_id = (m_idx % 4) * 8 + (n_idx % 8)  (伪示例)
     reg_id    = (m_idx // 4) * 2 + (n_idx // 8)

3. Global Memory 层面（L2 cache locality）:
   T.use_swizzle(10) → swizzle thread block order
   → 相邻 thread block 访问的 global memory 区域在 L2 cache 中有更好的 spatial locality
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Layout 抽象源自 TVM 的 IterVar 系统，TileLang 进行了扩展：(1) 支持 non-bijective transformation（如 padding——一个虚拟地址映射到多个物理地址）；(2) 支持 Fragment Layout（输出二维）；(3) 支持 layout swizzling 作为 built-in strategy。TileLang 的 Arithmetic Analyzer 使用 forward_index 表达式和 iter_var ranges 推断 transformed buffer shape 和访问边界。用户通过 T.annotate_layout 或 T.use_swizzle 使用。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
