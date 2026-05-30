# Sparse mixed scores and values
 $y \leftarrow \text{sparse}(AV, i_b)$ 

return  $y$ 

spatial distribution our sparse indices, based on the assigned patterns and the exact input. After that, we conduct the sparse attention computations with our optimized GPU kernels. The implementation details of our kernels can be found in Appendix C.4. Noted that the sparse mask is static for *A-shape* heads, so there is no overhead in building the dynamic masks, and only sparse calculation is required.

- (i) Vertical-Slash head. As shown in Algorithm 2, due to the continuity of vertical and slash lines, we matmul the last query vector  $\mathbf{Q}_{[-\text{last}_q:]}$  and key vector  $\mathbf{K}$  to produce the estimated attention matrix  $\widehat{\mathbf{A}}$ , which, in turn, is used to determine the indices for the vertical  $i_v$  and slash  $i_s$  lines. After obtaining the sparse indices for the vertical and slash lines, we convert them into a sparse format  $i_{vs}$ . Using these sparse indices, we perform block-sparse calculations of the attention weights and attention output.
- (ii) Block-Sparse head. Per Algorithm 3, mean pooling is applied on Q and K to obtain  $\widehat{Q}$  and  $\widehat{K}$ , respectively. The two matrices are multiplied to get the estimated block-level attention weights  $\widehat{A}$ . Since the mean pooling and matrix multiplication operations are commutative, the resulting attention weights are approximately equivalent to the actual attention weights after mean pooling. This allows us to approximate the actual attention weights' block-sparse pattern with minimal overhead. Similarly, we build a sparse index  $i_b$  and use it to compute the sparse attention weights and attention output.

