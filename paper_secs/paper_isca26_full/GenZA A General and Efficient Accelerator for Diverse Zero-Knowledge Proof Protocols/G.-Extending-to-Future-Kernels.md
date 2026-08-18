# *G. Extending to Future Kernels*

ZKP protocols are still under fast development. Nevertheless, we believe extending GenZA to support future ZKP kernels is highly possible. Polynomials on algebraic rings/fields remain as a central data structure in modern cryptography. The multi-bitwidth multipliers in GenZA can support various bitwidths, and the 2D array of lane-based PEs is efficient for polynomial and vector processing. Therefore, it is likely that the underlying hardware architecture of GenZA can remain effective for newly developed kernels. What we need to do is to propose optimized mapping schemes that best match the computation and data access patterns of these kernels.

