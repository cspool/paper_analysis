# SmartMem: Layout Transformation Elimination and Adaptation for Efficient DNN Execution on Mobile

注：不过分追求细节，懂得大意即可。

**想法：**

从layout对性能影响、layout custom分类每个算子；fuse/eliminate layout transformation operators；optimize the output layout；

# 1、背景

**transformer引入local attetion降低计算量，需要频繁的explicit layout变换**（reshape、transpose、gather）来划分window；

**conv+attention的模型中有很多implicit layout变换**；

**layout转换消耗时间。**

Q：GPU平台的layout优化，和异构平台的layout优化，哪种更优？CPU+GPU平台的端到端延时？

> **[图片提取文字 (image.png)]:**
> performing inference on mobile devices. Focusing on emerging transformers (specifically the ones with computationally efficient Swin-like architectures) and large models (e.g., Stable Diffusion and LLMs) based on transformers, we observe that layout transformations between the computational operators cause a significant slowdown in these applications. This
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image.png)

> **[图片提取文字 (image.png)]:**
> Table 1. Latency and transformation breakdown across various models. 'Lat.' shorts for Latency. 'Exp.' refers to the latency associated with explicitly transforming the tensor's layout, such as Transpose and Reshape. 'Imp.' denotes the latency incurred by implicit layout transformations. 'Comp.' indicates the latency attributed to the remaining operators. 'SD' represents Stable Diffusion model. These results are collected using MNN [32] on a Snapdragon 8 Gen 2 platform. MACs means the number of multiply-accumulate operations, and GMACS represents giga MACs per second.
> 
> | Model                   | #MACs<br>(G) | #Layout<br>transform | Lat. (ms) | Lat. b<br>Imp. |      | own (%)<br>Comp. | Speed (GMACS) |
> |-------------------------|--------------|----------------------|-----------|----------------|------|------------------|---------------|
> | ResNet50 [27]           | 4.1          | 3                    | 14        | 4.8            | 0.2  | 95               | 293           |
> | FST [34]<br>RegNet [58] | 162<br>3.2   | 32                   | 1,506     | 70.7           | 1.8  | 27.5<br>83.3     | 108<br>56     |
> | CrossFormer [69]        | 5.0          | 208                  | 336       | 15.3           | 55.2 | 29.5             | 15            |
> | Swin [46]               | 4.6          | 242                  | 342       | 14.7           | 54.1 | 31.2             | 15.2          |
> | AutoFormer [8, 9]       | 4.7          | 233                  | 335       | 13.3           | 54.2 | 32.5             | 14            |
> | CSwin [19]              | 6.9          | 769                  | 703       | 14.3           | 50.2 | 35.5             | 10            |
> | SD-TextEncoder [59]     | 6.7          | 183                  | 133       | 15.1           | 36.3 | 48.6             | 44            |
> | SD-UNet [59]            | 90           | 533                  | 2172      | 19.4           | 42.1 | 38.5             | 42            |
> | Pythia-1B [6]           | 119          | 385                  | 3034      | 11.7           | 31.7 | 56.6             | 39            |
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%201.png)

**spatial类型算子使用local attention/window**

> **[图片提取文字 (image.png)]:**
> foundational (vanilla) Transformer model by introducing local-attention Transformer models [8, 9, 18, 43, 46] that reduce computational complexity.
> 
> the input sequence. DL practitioners have built upon the
> 
> Local attention focuses on a subset of input tokens (typically within a window) at a time and requires less computation. To achieve computational reduction, frequent explicit shape/data reorganization (Reshaping, Transposing, Gathering) is used to split and transpose segments within the input
> 
> data into these smaller windows. <u>Another trend involves</u> combining traditional ConvNets with Transformers and designing new model structures [7, 12, 19, 25, 68, 74, 77] to benefit from both structures. These new structures introduce
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%202.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1. Examples of layout transformation in DNNs.
> 
> implicit data reorganization due to different layout preferences for various operators. Figure 1 shows the two types of data reorganization, which we will discuss next.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%203.png)

**消除reshape、transpose，执行相邻的flatten和spatial类型算子，有如下问题；**

Q1：邻接算子何时使用相同的layout，即何时变换输出layout；

Q2：什么layout最高效。

Q3：基于layout实现算子，优化access pattern和简化index computation。

Q4：将layout映射到内存层次（2.5D mem）。

> **[图片提取文字 (image.png)]:**
> - erations that follow the transpose. Eliminating reshape and transpose operations, and performing the two operations, i.e., those before and after the transpose, on the same layout should be able to improve performance. However, eliminating such layout transformation involves many important (inter-dependent) questions.
> - Q1: In a large and complex computational graph capturing a large DNN, deciding when two consecutive operations should use the same layout of data.
> - **Q2:** For two operations where we decide to use the same layout, choosing the layout that leads to efficient execution of both operands.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%204.png)

> **[图片提取文字 (image.png)]:**
> - Q3: Implementing an operation efficiently for a chosen layout (distinct from the original layout), including deciding access pattern and simplifying index computations.
> - Q4: Mapping the chosen layout to memory hierarchy, especially the 2.5D memory (further discussed in Section 2.3).
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%205.png)

**2.5D Texture vs. 1D Buffer**

texture Buffer 长于stencil、spatial类型计算，但短于reshape。

> **[图片提取文字 (image.png)]:**
> The GPU texture memory specializes in improving twodimensional spatial locality. Since each cache element can be a vector with a fixed length of four, and the cache itself has two dimensions (referred to as width and height), we refer it to as 2.5D memory. Originally designed to facilitate graphical rendering processes, this design also offers significant advantages for stencil and similar computations. For
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%206.png)

> **[图片提取文字 (image.png)]:**
> Table 2. Memory comparison on mobile GPUs.
> 
> |             | Characteristics                                                                   | 1D Buffer                          | 2.5D Texture                                         |
> |-------------|-----------------------------------------------------------------------------------|------------------------------------|------------------------------------------------------|
> | Computation | Acceleration engine<br>Automatic bounds checking<br>Hardware interpolation        | N<br>N<br>N                        | Y<br>Y<br>Y                                          |
> | Data access | Organization Addressing Dedicated cache Data locality Direct memory access on CPU | Contiguous Pointer-based No 1D Yes | Multidimensional<br>Coordinates<br>Yes<br>2.5D<br>No |
> 
> ![](_page_0_Picture_2.jpeg)
> 
> Figure 2. Layout transformation and 2.5D memory.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%207.png)

# 2、方法

> **[图片提取文字 (image.png)]:**
> paper presents SmartMem, a comprehensive framework for eliminating most layout transformations, with the idea that multiple operators can use the same tensor layout through careful choice of layout and implementation of operations. Our approach is based on classifying the operators into four groups, and considering combinations of producer-consumer edges between the operators. We develop a set of methods for searching such layouts. Another component of our work is developing efficient memory layouts for 2.5 dimensional memory commonly seen in mobile devices. Our experimen-
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%208.png)

**算子分类：**

**1、计算性能和输入layout有关。**

**2、输出layout是否可变（是否由计算模式确定）。**

**不同layout组合的多操作数算子可能有不同分类。**

> **[图片提取文字 (image.png)]:**
> The first dimension is whether the performance of the computation depends upon the input layout or is independent The second dimension is whether the output layout is customizable (perhaps in view of the computation pattern chosen for operator's implementation). These two dimensions result in four quadrants and each operator can be mapped to one quadrant. In some cases, one operator may be placed in different quadrants depending on whether the layout of its different operands is the same or different. Table 3 shows
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%209.png)

**张量加法的计算性能和输入layout无关（Input Layout Independent），输出layout可定制（Variable）；**

**张量乘法的性能和输入layout相关（Input Layout Dependent），输出layout可定制；**

> **[图片提取文字 (image.png)]:**
> $$C_{n-d}^l \leftarrow A_{n-d}^{l_1} + B_{n-d}^{l_1}$$
> 
> In the above operation, tensors A and B have the same layout denoted as  $l_1$ . Having an identical layout, and with an addition operation needing to touch each element once, the computational performance is not sensitive to the layout  $l_1$ , i.e., addition can be performed efficiently with traversal of the two matrices matching their (identical) physical layout. At the same time, the layout of the output tensor C can be customized, for example, based on the needs of the downstream operations.
> 
> Moving clockwise, we consider matrix multiplication example  $C_{2-d}^l \leftarrow A_{2-d}^{l_1} \cdot B_{2-d}^{l_2}$ . As a multiplication operation involves temporal reuse of data, the performance of the operation is clearly sensitive to the input layouts (even if the layouts  $l_1$  and  $l_2$  are identical). At the same time, the output layout can be customized for this application. An operator like Transpose, on the other hand, has a well-defined output layout (dependent upon the input layout). Given the memory transformations involved, the performance can be sensitive to the layout of the input operand. Finally, consider an operator like Slice. Because of simple selection involved, the performance is insensitive to the layout of the input. Moreover, by definition, the output layout has to match the input layout and is therefore not customizable
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2010.png)

> **[图片提取文字 (image.png)]:**
> Table 3. Operator classification based on input layout dependency and output layout flexibility. 'Comp.' stands for Computation.
> 
> | Output layout Comp.             | Variable (Computation<br>order dependent)                                                                                                                                                                                                                                                                                                      | Fixed                                                                                                                                                                                                                                            |
> |---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
> | Input layout<br>dependent (ILD) | $\begin{aligned} & \text{Conv:} C_{4-d}^{l} \leftarrow A_{4-d}^{l_1} * B_{4-d}^{l_2} \\ & \text{MatMul:} C_{2-d}^{l} \leftarrow A_{2-d}^{l_1} \cdot B_{2-d}^{l_2} \\ & \text{LayerNorm:} C_{n-d}^{l} \leftarrow A_{n-d}^{l_1} \odot B_{2-d}^{l_2} \\ & \text{Softmax:} C_{n-d}^{l} \leftarrow A_{n-d}^{l_1} \odot B_{2-d}^{l_2} \end{aligned}$ | $\begin{aligned} \text{Reshape:} B_{n-d}^L \leftarrow A_{m-d}^{l_1} \\ \text{Transpose:} B_{n-d}^L \leftarrow A_{n-d}^{l_1} \\ \text{DtoS:} B_{n-d}^L \leftarrow A_{n-d}^{l_1} \\ \text{StoD:} B_{n-d}^L \leftarrow A_{n-d}^{l_1} \end{aligned}$ |
> | Input layout independent (ILI)  | $\begin{aligned} & \text{Unary:} B_{n-d}^l \leftarrow A_{n-d}^{l_1} \\ \text{Add:} C_{n-d}^l \leftarrow A_{n-d}^{l_1} + B_{n-d}^{l_1} \end{aligned}$                                                                                                                                                                                           | $\begin{aligned} \text{Gather:} B_{n-d}^L \leftarrow A_{m-d}^{l_1} \\ \text{Slice:} B_{n-d}^L \leftarrow A_{n-d}^{l_1} \end{aligned}$                                                                                                            |
> 
> Unary refers to an operator applying a function to each element of a single input. DtoS and StoD mean DepthToSpace and SpaceToDepth, respectively.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2011.png)

**输入A、输出B、算子O在不同layout的形式化表达**

> **[图片提取文字 (image.png)]:**
> Take Input Layout Dependent & Variable (ILD-Variable) as an example. Let A and B represent the input tensor and the output tensor, respectively, and  $\mathcal{O}^{ILD-Variable}$  denotes the operator. The permutations  $\pi$  and  $\sigma$  represent the flexibility in processing and generating data, respectively. More specifically, The permutation  $\pi$  represents a rearrangement of these dimensions that affects the input layout, and  $\sigma$  represents a rearrangement of the computed result's dimensions affecting the output layout. Denoting mathematically,
> 
> $$B_{\sigma(i),\sigma(j),\dots,\sigma(n)} = \mathcal{O}_{\pi(i),\pi(j),\dots,\pi(m)}^{ILD-Variable} (A_{i,j,\dots,m})$$
> 
> The input tensor is accessed as  $A'_{\pi(i),\pi(j),\dots,\pi(m)} = A_{i,j,\dots,m}$  during the computation, where A' is the tensor A with its layout altered according to  $\pi$ . This rearrangement can lead to different memory access patterns, which might impact cache performance and processing speed. The computation is then performed, and the result is organized in the output tensor B as  $B'_{i,j,\dots,n} = \mathcal{O}(A'_{\pi(i),\pi(j),\dots,\pi(m)})$ . Finally, the output tensor B is produced with a layout determined by  $\sigma$ , i.e.  $B_{\sigma(i),\sigma(j),\dots,\sigma(n)} = B'_{i,j,\dots,n}$ .
> 
> Each of these categories reflects a different combination of data layout considerations and computational patterns, which are crucial for optimizing the performance of DNNs. The permutation functions  $\pi$  and  $\sigma$  accommodate the flexibility in data layout and output generation, which can be exploited for performance enhancements such as minimizing memory traffic, improving cache usage, or optimizing parallel execution strategies.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2012.png)

**表5是不同算子组合的4种优化策略；**

**表6是优化后算子的layout，优化后输出layout被有更高优化复杂度的算子决定、或被保留算子决定。**

直观上：layout对性能影响越大，优化复杂度越大；Variable比Fixed优化复杂度大。 

算子融合基于DNN fusion。

> **[图片提取文字 (image.png)]:**
> of an operator. As stated above, the four combinations are: input layout dependent and variable output (ILD & Variable), input layout independent and variable output (ILI & Variable), input layout dependent and fixed output (ILD & Fixed), and input layout independent and fixed output (ILI & Fixed). From a performance optimization perspective, we observe that their "optimization complexity" gradually decreases. For example, ILD & Variable requires us to be aware of both the input and output layouts while ILI & Fixed has no requirement about either of the layouts.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2013.png)

> **[图片提取文字 (image.png)]:**
> **Table 5. Operator combination - action.** ILD & Var is short for ILD & Variable.
> 
> | First       | Second | ILD & Variable | ILI & Variable | ILD & Fixed    | ILI & Fixed    |
> |-------------|--------|----------------|----------------|----------------|----------------|
> | ILD & Var   | Action | Keep both      | Try fuse       | Eliminate 2nd  | Eliminate 2nd  |
> | ILI & Var   | Action | Try fuse       | Try fuse       | Eliminate 2nd  | Eliminate 2nd  |
> | ILD & Fixed | Action | Eliminate 1st  | Eliminate 1st  | Eliminate both | Eliminate both |
> | ILI & Fixed | Action | Eliminate 1st  | Eliminate 1st  | Eliminate both | Eliminate both |
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2014.png)

> **[图片提取文字 (image.png)]:**
> Table 6. Operator combination and their corresponding design decisions.
> 
> | First | Second | ILD & Var       | ILI & Var      | ILD & Fixed    | ILI & Fixed     |
> |-------|--------|-----------------|----------------|----------------|-----------------|
> | ILD & | Output | ILD & Variable* | ILD & Variable | ILD & Variable | II D & Variable |
> |       | -      |                 |                |                |                 |
> | Var   | Layout | Search both     | Search fused   | Search 1st     | Search 1st      |
> | ILI & | Output | ILD & Variable  | ILI & Variable | ILI & Variable | ILI & Variable  |
> | Var   | Layout | Search fused    | No search      | No search      | No search       |
> | ILD & | Output | ILD & Variable  | ILI & Variable | N/A            | N/A             |
> | Fixed | Layout | Search 2nd      | No search      | No search      | No search       |
> | ILI & | Output | ILD & Variable  | ILI & Variable | N/A            | N/A             |
> | Fixed | Layout | Search 2nd      | No search      | No search      | No search       |
> 
> <sup>\*</sup> Both operators are ILD & Variable.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2015.png)

> **[图片提取文字 (image.png)]:**
> four levels of computation optimizations (marked with different colors): keeping both operators, trying to fuse them, eliminating one operator (either the first or the second), and eliminating both operators, which represent the optimization opportunities from low to high. Correspondingly, Table 6 summarizes the resulting output type and the input/output layout search policies after the computation optimizations explained above. The resulting output type is decided by the operator with a higher optimization complexity or the preserved operator, for example, after the computation optimization of a pair of operators with ILD & Variable and ILI & Variable types, respectively, the resulting (fused) operator is ILD & Variable, and after eliminating the second operator in a pair of ILD & Variable and ILD & Fixed operators, the resulting operator is in ILD & Variable, too.
> 
> with different colors): searching input and output layouts for both operators, searching input and output layouts for the fused operator, searching for either the first or the second, and no need to perform any search operation, representing the varied levels of optimization processing difficulties from high to low. It is worth noting that the layout search only happens for the operator pairs involving ILD & Variable. To explain the ideas, consider a pair of operators, i.e., Conv+Reshape (ILD & Variable + ILD & Fixed) as an example. Table 5 implies that Reshape can be eliminated 1. According to Table 6, the preserved operator (Conv) is still in ILD & Variable and SmartMem needs to search for its input layout.
> 
> The input/output layout search also has four levels (marked
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2016.png)

> **[图片提取文字 (image.png)]:**
> With respect to the first question, SmartMem relies on the techniques based on the DNNFusion project [51] to decide if an operator fusion is legal. Based on DNNFusion, the subsequent operator elimination and layout selection designed in SmartMem bring more opportunities for beneficial fusions, enhancing execution performance (as we show through our
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2017.png)

**Fixed类型算子可以采用elimination，即reduct到split、merge、identity表达的张量变形。**

fuse是将算子计算的形式表达融合，减少中间结果的搬运；

elimination将算子的内存访问简化，通过Index Comprehension。

> **[图片提取文字 (image.png)]:**
> Elimination can be leveraged for cases involving any operator with a Fixed output type (i.e., the cases marked by either yellow or green) to further improve the performance of the memory access in the fused operator. More specifically, after fusing a sequence of layout transformation operators, these operators can be replaced by index computations for the operator it is fused with.
> 
> Strength reduction on index computation. To further reduce the overhead for index computation during data loading and
> 
> storage, we propose the following optimizations for index computation. As a background, Reshape transforms an input with shape  $[d_1, \ldots, d_m]$  into an output with shape  $d'_1, \ldots, d'_n$ where the product of the new shape must equal to the old shape. Transpose permutes the dimensions according to a given order – formally:  $Out_{i,...,i_n} \leftarrow In_{\pi(i),...,\pi(i_m)}$ . However, it turns out that using the linear representation for all indexes directly leads to redundant computations, especially when multiple Reshape and Transpose operations are stacked together. Instead, our strength reduction strategy analyzes the index dependencies between consecutive layers. As shown in Figure 3, we define the index dependencies as identity, split, and merge based on static shape information in the computational graph. This allows us to transform indexes according to their dependency types using operands such as "//, %" (used in *split*) and "\*, +" (used in *merge*), "="
> 
> (used in *identity*). Since modular and divide operations are
> 
> expensive on GPUs, we also simplify these terms by apply-
> 
> ing mathematical strength reduction rules. For example, if
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2018.png)

> **[图片提取文字 (image.png)]:**
> An additional point to be mentioned is that before index computation simplification, the fused operator is combining the logic of multiple layout transformations, rendering memory accesses fragmented and difficult to optimize. After the simplification process described earlier, the memory access pattern is more straightforward and exposes aggressive optimization opportunities.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> tion from the input, right: index computation (no opt.).
> 
> [2, 256, 4]
> 
> i' = i \* 8 + j // (4 \* 8)
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2020.png)

**算子fuse、eliminate到只剩下ILD & Var，是贪心策略。**

**只需要探索ILD & Var之间的layout，让生产者的输出满足消费者的输入layout。（确定何时变换layout）**

**reduction dimension是输入张量经reduce类型算子后被规约的维度，比如矩阵乘的乘累加对应的维度k。保证reduction维度内存连续。**

> **[图片提取文字 (image.png)]:**
> Our heuristic: reduction dimension. Reduction dimension(s) for an operand of an operator is the (set of) dimension along which data elements are involved in an aggregation. Take MatMul with  $A_{i,k}$  and  $B_{k,j}$  as inputs as an example. Its reduction dimension is k for both matrices A and B.
> 
> To examine how the notion of reduction dimensions is used, we revisit Tables 5 and 6. After multiple rounds of operator fusion and operator elimination, all preserved operators are ILD & Variable – this is because all operators in other types including ILI & Variable are fused into ILD & Variable eventually. The next step is to find the layout on each edge/operator pair (with ILD & Variable type). Specifically, this step forces the first operator of each edge (i.e., producer) to generate the data layout preferred by the second operator (consumer). In our approach, this preferred data layout is decided by the reduction dimension of the second operator. For example, for the above MatMul example, we should keep the elements in both  $A_{i,k}$  and  $B_{k,j}$  along the reduction dimension (k) continuously stored in the memory. The insight here is that forcing the producer to generate a layout based on the reduction dimension of the consumer incurs relatively low added overhead as compared to other options. Specifically, sub-optimally writing results turns out to be better
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2021.png)

**算子间一对多时，producer算子输出layout的reduction维度允许有k个，当实际reduction超过k时，需要额外拷贝。k由memory的连续访问特性决定，Texture Memory中k是2；**

一对多时，前序算子的输出张量有多个reduction维度，可以综合出统一的输出layout。

L0的输出有D1和D3两个reduction维度，在图5中分别在0.5D和1D上连续存储。

> **[图片提取文字 (image.png)]:**
> Global: layout selections based on reduction dimension. Searching for optimal layouts when one producer operator may have multiple consumer operators becomes challenging. We optimize the layout for the producer based on the collective needs from its consumers. Specifically, we optimize corresponding to the first k reduction dimensions required by the consumers, where k is the number of dimensions along which we can perform continuous memory access without any linearization and index computation. For example, k = 2 for 2.5D memory. Section 3.3 will elaborate more details for 2.5D memory. If consumers require more than k optimized layouts, SmartMem needs to maintain several copies of data with different layouts, and each layout is in this optimized combined format.
> 
> Example. Figure 4 illustrates our reduction dimension-based layout selection approach on a simplified computation graph. Figure 4(a) shows the original computation graph with a series of operators and the dimensions of their intermediate results. Transpose and Reshape (which splits the dimension  $D_0$  into  $D_2$  and  $D_3$  for all successor operators) are inserted here for aligning the dimension for different kinds of operators (MatMul and Conv). Figure 4 (b) demonstrates an optimized computation graph with our reduction dimension-based layout selection. Transpose and Reshape are both ILD-Fixed and hence can be eliminated (as shown in Table 5). Then the output tensor of MatMul is consumed directly by Reduce, Reduce', and Conv (FusedConv); however, these operators suggest two reduction dimensions in which,
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2022.png)

> **[图片提取文字 (image.png)]:**
> Reduce' has a reduction dimension of D1 while Reduce and Conv(FusedConv) have a common reduction dimension of D3. Assuming a mobile GPU with 2.5D memory (and cache), SmartMem can combine these two reduction dimension requirements (i.e., optimized layout requirements) for consumers of MatMul into an uniform data layout ( $L^0$  as shown in the figure) and preferably map D1 and D3 to the two memory dimensions that allow continuous and direct indexed memory access. The layout generations of Reduce  $(L^1)$  and Reduce'  $(L^2)$  are more straightforward because they have no reduction dimension requirement. We will explain more details in Section 3.3.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2023.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2024.png)

**使用texture memory并探索layout**

**GPU中texure memory是2.5D memory**（x，y，4），x和y分别是1维，4是0.5维；

texure memory可存储2D、3D张量，无需index computation；

**reduction维度尽量连续存储**，可提高locality，因此L0（D1，D2，D3）的D1对应0.5D；

L1（D1，D2，1）、L2（1，D2，D3）没有reduction维度；但其和L0作为fusedconv的输入张量，因此L1、L2和L0的映射方式相似，将L1的D1、L2的D3映射到0.5D。

> **[图片提取文字 (image.png)]:**
> ## 3.3 Mapping Tensor to Texture Memory and Other Optimizations
> 
> 2.5D memory (and corresponding dedicated read-only cache) on mobile GPUs allows more flexible index computation and eliminates index linearization if a tensor's dimension is less than 3. It also facilitates better exploring data reuse opportunities for 2D and 3D tensors. Thus, SmartMem also leverages 2.5D memory to further improve our optimized tensor layout design and memory access as mentioned earlier.
> 
> Optimized tensor layout on 2.5D memory. Figure 5 shows 3 sample tensor layouts when mapping a 3D tensor with varied shapes to 2.5D memory, corresponding to  $L_0$ ,  $L_1$ , and  $L_2$  in Figure 4, respectively ( $L_3$  depends on its consumers). D1, D2, and D3 denote the dimensions of these tensors – both  $L^1$  and  $L^2$  have a dimension of size 1. Specifically, the red circles denote that these dimensions are specified as reduction dimensions by the consumer operators of this tensor, for example,  $L^0$  has two reduction dimensions, D1 and D3 decided by Reduce' and FusedConv (in Figure 4), respectively, while  $L^1$  and  $L^2$  have no reduction dimensions.
> 
> It is beneficial to store the data along a reduction dimension continuously to improve data locality and allow better SIMD load and reduction operations. Thus, to map a tensor with two reduction dimensions (e.g.,  $L^0$  with D1 and D3) to 2.5D memory, SmartMem partitions one reduction dimension (D1). Each such partition has k elements (k = 4 in this example to match the size of the 0.5D in 2.5D) and stores  $k \times 1$  another reduction dimension elements as a chunk along the dimensions of 2.5D memory that can be accessed in both directions. This layout results in efficient memory access and
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2025.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> **Figure 5. Sample layouts on 2.5D memory.** Red circles denote reduction dimensions.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2026.png)

> **[图片提取文字 (image.png)]:**
> $L^1$  and  $L^2$  do not have any reduction dimension because they are consumed by an element-wise addition operator (Add), so theoretically we are allowed to map either D1/D3 or D2 to that 0.5 dimension of 2.5D memory. However, because
> 
> SIMD operations for consumer operators using either D1 or
> 
> D3 as their reduction dimensions.
> 
>  $L^1$  and  $L^2$  are used together with  $L^0$  in the FusedConv and their element-wise addition operations have been fused with the Conv, SmartMem maps them to 2.5D memory in a similar manner to  $L^0$  (i.e., mapping D1 and D3 to the 0.5D of 2.5D memory, respectively), avoiding extra index computations.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2027.png)

**layout优化后的效果**

> **[图片提取文字 (image.png)]:**
> Optimized memory access based on tensor layout. Figure 6 shows the high-level idea of our optimized data access on 2.5D memory by comparing the data access orders before and after this optimization. Figure 6 (a) shows the original computational graph, Figure 6 (b) shows the computational graph and data access order after the fusion and operator elimination offered by SmartMem, and Figure 6 (c) shows the computational graph and data access order after our optimized tensor layout mapping and memory access optimization. Before the optimized tensor layout mapping, although the Reshape and Transpose operations are fused (and eliminated), the memory access pattern is complex and fragmented, resulting in poor data locality (and similarly low SIMD efficiency). The optimized tensor layout mapping offers us an opportunity to access this tensor along its reduction dimension with a stride of 1, thus improving the data locality on 2.5D memory and cache, and parallel and SIMD efficiency on mobile GPUs.
> 
> Other optimizations. In addition to the previously mentioned optimizations, our framework incorporates an auto-tuning mechanism that utilizes Genetic Algorithms [51] for generating GPU execution configurations. These configurations include block dimensions, unrolling factors, and tiling shapes.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2028.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6. Data access patterns comparison before and after layout and access optimizations. In left (a), two consecutive Shape operations reshape the output from LayerNorm across different dimensions.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2029.png)

a→b：fuse、eliminate算子，但未优化算子输出的layout，a和b的layout由LN、reshape等算子原生决定，reduction维度可能不连续存储；

b→c：优化算子输出的layout，让reduction维度连续存储，提高locality；

# 3、实验设计、效果

**exp、baseline**

> **[图片提取文字 (image.png)]:**
> tal results show that SmartMem outperforms 5 state-of-theart DNN execution frameworks on mobile devices across 18 varied neural networks, including CNNs, Transformers with both local and global attention, as well as LLMs. In particular, compared to DNNFusion, SmartMem achieves an
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2030.png)

**实验展示目标：**

> **[图片提取文字 (image.png)]:**
> objectives of this evaluation are as follows: 1) To exhibit the notable improvements SmartMem offers against existing, cutting-edge DNN frameworks on a mobile GPU, 2) To explore how various optimizations contribute to these performance enhancements, 3) To demonstrate the portability of proposed optimizations in SmartMem by evaluating the execution times on two other mobile platforms, and 4) To illustrate that our proposed optimizations enable performance improvement on a desktop-level GPU, which is less resourceconstrained and has a traditional (one-dimensional) memory. Particularly, SmartMem is compared against MNN [32], NCNN [50], TFLite [1], TVM [10], and DNNFusion [51] (refers as DNNF), on the mobile GPU.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2031.png)

**实验环境**

> **[图片提取文字 (image.png)]:**
> **DNN Workloads:** Our evaluation is conducted on 18 stateof-the-art DNN models with three different structures, including Transformer, ConvNet, and Hybrid (i.e., ones having both Transformer and ConvNet structures), as well as Stable Diffusion and LLMs. Table 7 characterizes them with a comparison of their type, attention mechanism, the number of parameters, the number of operators prior to optimizations, as well as the number of multiply-accumulate operations (MACs). We have 1) six Transformer models (Auto-Former [8, 9], CrossFormer [69], Swin [46], ViT [20], Stable Diffusion - TextEncoder [59], Pythia [6]), 2) four Convolution models (ConvNext [47], RegNet [58], ResNext [72], Yolo-V8 [33]), and 3) eight Hybrid models with both Transformer and Convolution structures (BiFormer [77], CSwin [19], EfficientVit [7], FlattenFormer [25], SMTFormer [43]), Conformer [24], StableDiffusion - UNet and VAEDecoder [59].
> 
> The Transformer structure can serve as a backbone for different CV and NLP tasks. Due to the space limitations, we report the results on the object detection task (for Yolo-V8) and image classification task (for all other models) in this evaluation. Since the training dataset has a negligible impact on the final inference latency, this section reports results from one training dataset for each model. Yolo-V8 is trained on MS COCO dataset [42]; Conformer, Stable Diffusion Models (SD-TextEncoder, SD-UNet, SD-VAEDecoder), and Pythia are from the pretrained checkpoints; other models are trained on ImageNet dataset [15]. Since the model accuracy is the same across all frameworks, our evaluation focuses only on execution latency.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2032.png)

> **[图片提取文字 (image.png)]:**
> vious work (DNNFusion [51]) that supports extensive operator fusion. We conduct our major experiments on a Oneplus cell phone using the high-end Qualcomm Snapdragon 8 Gen 2 platform [56], which includes a Qualcomm Kryo octa-core CPU and a Qualcomm Adreno 740 GPU with 16 GB of unified memory (shared by both CPU and GPU). In addition, to demonstrate our portability, we test SmartMem on an earlier generation of the Qualcomm platform - Snapdragon
> 
> Evaluation environment. SmartMem is built upon our pre-
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2033.png)

> **[图片提取文字 (image.png)]:**
> 835 [55], which has more limited resources, and on a MediaTek platform with Dimensity 700 SOCs. The Snapdragon 835 consists of an ARM Octa-core CPU, an Adreno 540 GPU, and 6 GB of unified memory. The MediaTek Dimensity 700 is equipped with an ARM Octa-core CPU, a Mali-G57 GPU, and 4 GB of unified memory. Furthermore, we evaluate our optimizations on an NVIDIA GPU to illustrate our generality. For this comparison, we implement our optimizations (excluding the mobile device-specific optimization for the 2.5D memory layout) in TorchInductor and compare it against the base version of TorchInductor. For all evaluated models and frameworks on mobile devices, GPU execution uses 16-bit floating-point representation. On desktop-level GPU, we use 32-bit floating-point representation for evaluation purposes. Noting that, we use 16-bit floating-point for mobile GPU because it is a common data type that all the frameworks support. Although other data types may have different accuracies, our optimizations are based on operator semantics thus not limited to specific data types. The batch size is set to 1 for all models unless otherwise specified. With the results we reported in this section, we have utilized the auto-tuning capabilities available in MNN, TVM, and TorchInductor to achieve the best possible performance. Each experiment is executed 50 times and only the average numbers are reported - as the variance was negligible, it is omitted for readability.
![image.png](SmartMem%20Layout%20Transformation%20Elimination%20and%20Ada%20241e-26fc/image%2034.png)

**实验效果**