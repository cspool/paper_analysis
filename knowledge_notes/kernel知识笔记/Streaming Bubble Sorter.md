## Streaming Bubble Sorter

术语是什么？通过联网搜索让回答具体和精准。
Streaming Bubble Sorter是Focus SEC中用于top-k selection的轻量级硬件排序器。它将importance analyzer中的a个并行max units级联为a-way bubble sorter，以streaming方式逐步refine top-a tokens，最终以O(M·a·k) cycles完成top-k selection（M为候选数，k为保留数，a为并行度）。与传统full sorting (O(M log M))或global sorting不同，streaming bubble sorter利用bubble sort的局部比较特性实现pipelined、low-area的streaming top-k。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Streaming Bubble Sorter的硬件执行流程：
```
# a-way streaming bubble sort for top-k from M candidates
# a: number of parallel max units (e.g., a=4)
# k: number of top elements to retain (k < M)

top_a = [0] * a  # maintain top-a elements in sorted order (descending)
result = []  # final top-k output
remaining = M

while remaining > 0:
    # Stream in a new elements per cycle
    batch = stream_next_a_elements()  # a elements from importance vector
    for new_elem in batch:
        # Bubble insert: compare and swap to maintain descending order
        pos = a - 1
        while pos > 0 and top_a[pos-1] < new_elem:
            top_a[pos] = top_a[pos-1]
            pos -= 1
        top_a[pos] = new_elem
        # top_a now contains the a largest elements seen so far
    remaining -= a

# After all M elements processed, top_a contains the a largest values
# To get top-k (k may be larger than a): iterate with reduced scope
# Actually: chain multiple a-way sorters for k > a
# Complexity: M * a * k / a = M * k cycles (a cancels out partially)
# More precisely: M/a * k cycles for k <= a
```
论文证明SEC sorting与image attention GEMM完全重叠：sorting需要 M·a·k cycles，attention GEMM需要 M·(M+T)·h·n/(a·b) cycles，ratio = (M+T)·h·n/(k·b)。当h·n=3584, b=32, k<M+T时ratio ≫ 1，sorting远在attention GEMM完成前结束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Streaming Bubble Sorter在Focus SEC中用SystemVerilog实现。它复用importance analyzer的并行max units（a个），通过级联和feedback path构成bubble chain。每个cycle接受a个新scores，与当前top-a比较并更新。Area开销计入SEC的1.9%总面积（含analyzer + sorter + encoder）。此设计可参数化a和k以适应不同accuracy/complexity trade-off。相比full sorting logic，bubble sorter的area和latency均远小。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

