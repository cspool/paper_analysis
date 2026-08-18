# D. Performance Under Other PIM Workloads

Many kernels used in machine learning and data analytics fall into read-intensive categories and rely on the small PIM buffer and registers to partially reduce or aggregate results that are later read out by the host, as opposed to results being written to memory during PIM execution. Beyond the GEMV case analyzed in detail, similar access patterns arise in operations such as column-value filtering in databases, distance calculations used in machine learning and vector database search. Figure 16(a) reports performance for such read-intensive kernels from PIMBench [71] as a speedup of

the all-bank PIM over a rank-PIM. Since these kernels do not perform PIM-side writes, non-reliable bank-PIM and reliable bank-PIM achieve nearly identical performance.

Linear Regression achieves near-ideal speedup  $(7.9\times)$  because it streams long input vectors while returning only four reduced sums whose output size is independent of vector length. KNN achieves the lowest speedup  $(1.6\times)$  among these kernels because it has less data reuse than the other workloads, requiring frequent readouts from the host.

Figure 16(b) compares the performance of non-reliable and reliable bank-PIM against rank-PIM on write-heavy kernels, with each kernel annotated by its read/execute-to-write ratio. Because reliable bank-PIM updates rank-level ECC codewords on every write, writes behave the same as CPU writes, making them approximately  $8\times$  slower than writes in non-reliable bank-PIM. The write-performance degradation is not larger because even non-reliable bank-PIM provides lower acceleration for writes than reads: reads achieve up to  $8\times$  speedup over rank-PIM, whereas writes are limited to  $2\times$  due to tCCD L being  $4\times$  longer for writes than for reads.

As shown in Figure 16(b), write overhead of reliable bank-PIM increases as the read-and-execute portion to write decreases. Nevertheless, reliable bank-PIM outperforms rank-PIM in K-means  $(1.6\times)$  and Image Downscaling  $(1.3\times)$  when the read-and-execute portion is high enough. Extremely write-intensive kernels (vector add and AXPY) perform better with a rank PIM given the high write overhead of the reliable bank PIM  $(0.7\times)$ . The K-means kernel has a small amount of host readouts that reduce its speedup compared to the rank-PIM.

The rank-level correction overhead from VRT errors is independent from the write overheads and is kept negligible by Codeword Flip and hardware-based rank-level correction, less than 2.1% overhead across write-heavy and host-read kernels. The error correction overhead for Filter by Key is larger because this kernel has a larger fraction of read PIM operations compared to the other host-read kernels.

