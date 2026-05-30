## Inter-Kernel Dependency Detection via Read/Write Segments

术语是什么？
Inter-Kernel Dependency Detection via Read/Write Segments（通过读写段进行核间依赖检测）是 ACS 框架用于在运行时自动发现 GPU kernel 间数据依赖关系的机制。通过为每个 kernel 标注其读写的内存地址范围（read_segments 和 write_segments），ACS 在 kernel 插入调度窗口时比较这些地址范围的重叠：若 kernel A 的 write_segments 与 kernel B 的 read_segments 或 write_segments 有地址重叠（RAW / WAW 冲突），则 A 是 B 的 upstream kernel。这种基于地址范围的检测方法比追踪具体内存访问模式更轻量，适合运行时使用。

从kernel调度角度拆解术语：
```
// 依赖检测算法 (Algorithm 1 from ACS paper)
Input: kernel_in (新kernel的write_segments)
       window_kernels (调度窗口中已有kernel的read+write segments)
Output: upstream_list (新kernel依赖的kernel ID列表)

upstream_list = []
rwslist_new = kernel_in.read_segments ∪ kernel_in.write_segments

for each existing_kernel in window_kernels:
    is_dependent = false
    for each seg_1 in rwslist_new:
        for each ws_2 in existing_kernel.write_segments:
            start_1 = seg_1.start_addr
            end_1   = seg_1.start_addr + seg_1.size
            start_2 = ws_2.start_addr
            end_2   = ws_2.start_addr + ws_2.size
            
            if start_1 < end_2 AND end_1 > start_2:  // 地址范围重叠
                is_dependent = true
    if is_dependent:
        upstream_list.add(existing_kernel.id)
```

RW-segments 的定义通过 `ACS_wrapper` 实现：程序员或库开发者实现 `get_addresses()` 函数（在 kernel launch 前调用），将 kernel 的指针参数解析为起始虚拟地址和大小。对于矩阵乘法等常见 kernel，segments 从函数原型直接可得（如 `matmul(input1[M×N], input2[N×K], output[M×K])` → 3 个 segment）。对于间接内存访问的 kernel，保守地标记为访问全部 GPU memory。segments 也可通过 GPUOcelot 等二进制分析工具自动提取。

术语一般如何实现？如何使用？
程序员通过 `ACS_wrapper` 结构体标注 kernel：`__read_segments__` 和 `__write_segments__` 列表（起始地址+大小的 pair），以及 `get_addresses(dim3 blocks, dim3 threads, ...)` 函数。解析后的虚拟地址范围存储在 48-bit 的 segment 描述符中（起始地址+大小）。依赖检查延迟取决于 segments 数量和窗口大小：N=16, 6 segments: 410ns; N=32, 10 segments: 1640ns。对于无法确定访问范围的 kernel（如间接内存访问），ACS 将该 kernel 的 write_segments 设为整个 GPU memory 范围，在依赖检查中保守对待。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
