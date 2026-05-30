# B PSEUDO-CODE FOR BLOCK-SPARSE KERNELS

Figures 11 and 12 show CUDA pseudo-code for our SDD and DSD kernels, respectively. The DDS operation follows DSD closely, but with the two inputs swapped. Both figures show pseudo-code for the case where neither input matrix is transposed. Our approach for handling transposition of the sparse matrix input in DSD and DDS is described in

§5.1.4. Relative to Figure 12, this technique adds a layer of indirection to the tile loading from matrix *a* inside the main loop. Concretely, we load the offset of the next nonzero block in the threadblock's row from the *transpose indices* shown in Figure 6 prior to loading the block for computation.

