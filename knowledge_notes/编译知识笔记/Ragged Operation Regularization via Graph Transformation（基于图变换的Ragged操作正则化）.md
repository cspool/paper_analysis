## Ragged Operation Regularization via Graph Transformation（基于图变换的Ragged操作正则化）

术语是什么？通过联网搜索让回答具体和精准。

Ragged Operation Regularization是Difflow编译器的核心优化技术：将扩散模型batching中不同shape请求产生的ragged tensor操作（输入维度各异），通过graph transformation rules（transpose + reshape / transpose + im2col等）转化为regular tensor操作，从而直接使用cuBLAS/cuDNN等成熟kernel库执行，无需为每种ragged operator手写custom kernel。该方法区分两类操作：(1) data-sharing operations（有共享权重，如convolution/linear）需transform；(2) data-independent operations（无共享数据，如transpose/reduce）可直接embarrassingly parallel执行。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

Ragged operation regularization的graph transformation流程：

```
// === 操作分类 ===
1. if op shares weights/data across requests (conv, linear, attention):
2.     → data-sharing: need regularization transformation
3. elif op does NOT share data (transpose, reduce, elementwise without shared weights):
4.     → data-independent: embarrassingly parallel per-request execution

// === Data-sharing Ragged Operation Regularization ===
// 例1: Ragged Matmul
// Input: [b, m̂, k] (m̂: ragged dim, 各请求不同) × Weight: [k, n]
// Transformation:
5.     [b, m̂, k] → transpose → [b, k, m̂] → reshape → [b·m̂, k] (regular)
6.     Regular Matmul: [b·m̂, k] × [k, n] → [b·m̂, n]
7.     → transpose → [b, n, m̂] → reshape → [b, m̂, n] (restore ragged layout)

// 例2: Ragged Elementwise (e.g., Add with ragged input + regular weight)
// Input: [n, c, ĥ, ŵ] (ĥ, ŵ ragged) + Weight: [c, 1, 1]
// Transformation:
8.     [n, c, ĥ, ŵ] → T/R (transpose+reshape) → [n·ĥ·ŵ, c] (regular)
9.     Regular Add: [n·ĥ·ŵ, c] + broadcast([1, c]) → [n·ĥ·ŵ, c]
10.    → T/R → [n, c, ĥ, ŵ] (restore)

// 例3: Ragged Convolution
// Input: [n, c, ĥ, ŵ] (ragged) + Filter: [f, c, r, s]
// Transformation:
11.    [n, c, ĥ, ŵ] → im2col → [n·ĥ·ŵ, c·r·s] (regular, image-to-column)
12.    Regular Matmul: [n·ĥ·ŵ, c·r·s] × [c·r·s, f] → [n·ĥ·ŵ, f]
```

关键设计：(a) 通过transpose+reshape（T/R）将ragged dimension与batch dimension合并→得到regular dimension→直接用cuBLAS等成熟kernel；(b) im2col将ragged spatial convolution转为regular GEMM；(c) 不要求手写ragged kernel（auto kernel generators也难以处理），只需少量图变换规则即可覆盖所有data-sharing ops。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Difflow实现了一套图变换规则（transpose/reshape/im2col）作为compiler pass应用于dEngine编译时。规则针对operator pattern匹配：ragged matmul→fuse batch+ragged dim→regular matmul；ragged elementwise→T/R flatten→regular elementwise；ragged convolution→im2col→regular matmul。data-independent operations保持按请求tiling并行（round-robin tile-to-thread-block mapping），无需变换。该方法将ragged batching支持从"手写所有ragged kernel"降低为"添加~4个图变换规则"，且可flexibly扩展至new model architectures。

涉及论文标题：
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models
