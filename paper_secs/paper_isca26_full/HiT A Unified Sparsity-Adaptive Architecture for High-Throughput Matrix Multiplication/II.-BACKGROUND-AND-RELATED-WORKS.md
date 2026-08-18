# II. BACKGROUND AND RELATED WORKS

Matrix multiplication involves taking two matrices, A (size *M*×*K*) and B (size *K*×*N*), and producing a new matrix C (size *M*×*N*). The multiplication can be computed using one of the three basic dataflows, each defining a specific execution flow, as shown in Fig. 3.

Inner-product (IP) dataflow computes the dot product by calculating *one* element of C at a time, multiplying a row of A with a column of B. This process repeats until all elements of a row in C are computed before moving to the next row. Gustavson (Gust), also known as row-stationary dataflow, is similar to IP but computes an entire *row* of partial outputs at a time with each element of A. Outer-product (OP) multiplies each element of a column of A with the corresponding row of B, updating a row of C. The process repeats for all elements in the same column of A before moving to its next column.

In accelerator design, dataflow selection is critical as it directly influences the design of sparsity-handling modules and the rate of both intersections and reductions. The following subsections review prior accelerators that implement these dataflows.

