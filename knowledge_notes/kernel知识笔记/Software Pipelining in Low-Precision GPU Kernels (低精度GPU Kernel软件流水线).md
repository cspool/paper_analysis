## Software Pipelining in Low-Precision GPU Kernels (低精度GPU Kernel软件流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Software pipelining（软件流水线）在低精度GPU kernel中指将全局内存→共享内存的数据拷贝与Tensor Core计算重叠执行的优化技术。Tilus通过CopyAsync、CopyAsyncCommitGroup()、CopyAsyncWaitGroup(n)三个VM指令实现对异步拷贝流水线的声明式控制。CopyAsync发起一次async copy任务（触发cp.async硬件指令），CommitGroup标记一组任务的边界，WaitGroup(n)阻塞直到in-flight group数≤n。配合shared memory的多级缓冲，可实现global→shared copy与Tensor Core computation的完整重叠——当Compute正在处理tile_i时，DMA已在加载tile_{i+1}到shared memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
双缓冲软件流水线伪代码（decode stage matmul, K维循环）：
```
# 初始化: prefetch第一K tile
CopyAsync(shared_buf[0], global_weight_tile[0])
CopyAsyncCommitGroup()

for k in range(1, num_k_tiles):
    CopyAsync(shared_buf[k%2], global_weight_tile[k])        # 异步加载下一tile
    CopyAsyncCommitGroup()
    CopyAsyncWaitGroup(2)                                      # 最多2组in-flight

    a_tile = LoadGlobal(A_global, layout, offset=[:, k-1:])   # 从global加载activation
    b_tile = LoadShared(shared_buf[(k-1)%2], layout, offset)  # 从shared加载权重
    b_tile = View(b_tile, target_dtype, target_layout)         # 零开销reinterpret
    b_tile = Cast(b_tile, f16)                                  # 向量化casting
    C_accum = Dot(a_tile, b_tile, C_accum)                     # Tensor Core计算
    Synchronize()                                               # 等待当前iteration完成

# 处理最后一个tile
CopyAsyncWaitGroup(0)
# ... (最后一个tile的计算)
```

与Triton（Figure 1a）对比：Triton同样支持cp.async但受限于shared memory layout conversion瓶颈；与Ladder（Figure 1b）对比：Ladder的primitive-style scheduling根本不支持software pipelining（weight loading与computation完全串行）。Tilus结合了pipelining和零开销layout reinterpretation两个优势。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tilus的CopyAsync在编译时映射为PTX cp.async指令（或cp.async.v4向量化版本），CommitGroup/WaitGroup映射为cp.async.commit_group和cp.async.wait_group。编译器自动选择向量化宽度并计算shared memory buffer大小。软件流水线在batch>1的decode场景中尤为关键——此时compute load增大，pipelining的overlap收益显著。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation
