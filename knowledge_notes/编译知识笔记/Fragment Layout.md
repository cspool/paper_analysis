## Fragment Layout

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fragment Layout 是 TileLang 对标准 Layout 抽象（f: K^n → K^m）的扩展，专门用于描述 GPU register files 在 thread block 内的分布。与标准 Layout 输出单维线性内存地址不同，Fragment Layout 总是产生二维输出 f: K^n → K²，其中第一维表示 thread 在 block 内的位置（thread index），第二维表示该 thread 的 local register file 中的索引（register index）。例如，kernel 中通过 T.alloc_fragment(block_M, block_N) 分配了 block 级的 register file C_local[128,128]，Fragment Layout 精确描述了这个 128×128 矩阵如何划分到 128 个 threads：每个 thread 持有哪些行/列的元素。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Fragment Layout 的四种 primitive 组合操作（图 6）：
```
base_layout: 单个 warp 消费 m16k16 矩阵的 MMA 指令级别 layout

1. repeat: 
   warp_layout = base_layout.repeat(dim=0, factor=2)  
   → 单 warp 消费 m32k16 矩阵 (沿 m 维扩展 2×)

2. repeat_on_thread:
   block_level = warp_layout.repeat_on_thread(dim=0, factor=2)
   → 将特定维度的数据分配到更多 threads

3. replicate:
   block_layout = warp_layout.replicate(dim=1, factor=4)
   → 4 个 warp 各自消费 m32k16 → block 消费 m128k16

组合结果：4 个 warp × m32k16/warp = m128k16 block 级消费
每个 warp 内的 thread distribution 由 warp_layout 描述
warp 间关系由 replicate 描述
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fragment Layout 的实现基于 IterVar 代数系统（类似 TVM 的 IterVar），Layout function 编码为 forward_index 表达式：f(iter_vars) → (thread_index, register_index)。编译时 Arithmetic Analyzer 使用 forward_index 表达式和已知的 buffer shape 推断 buffer bounds、分配 register file 大小、生成 thread-level index 计算代码。用户通常不需要手动编写 Fragment Layout——Layout Inference Pass 自动推导。但用户可以通过实现 InferLayout 接口注册自定义 Tile Operator 的 layout 策略。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
