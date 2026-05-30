# B. Traditional Einsums

An Einsum expression defines a computation on a set of tensor operands using an iteration space that specifies the set of points where the computations are performed [35], [39]. For example, we describe matrix-matrix multiplication (GEMM) with the following Einsum:

$$Z_{m,n} = A_{k,m} \times B_{k,n} \tag{1}$$

where A and B are input 2-tensors of shape  $K \times M$  and  $K \times N$ , respectively. Z is an output 2-tensor with shape  $M \times N$ . Throughout this paper, we use the same symbol for both the shape and *name* of a rank (e.g., rank K in A has a shape of K).

The *iteration space* of this Einsum is  $[0, K) \times [0, M) \times [0, N)$ . An evaluation of this Einsum must: (1) walk every (k, m, n) point in the iteration space; and, at each point (2) project into the *data space* of all input tensors, (3) multiply the corresponding data values, and (4) place the result at the corresponding data point in Z. If a value already exists at an (m, n) point in Z (due to computation at the same (m, n) point for a different k in the iteration space), reduce the two values together using addition. Note that the Einsum specifies *what* to compute; it does not indicate the order in which one walks the iteration space. These aspects are left to the *mapping* [9], [35], [41].

We also note that we can view the iteration space itself as a tensor. In the example above, this tensor has shape  $K \times M \times N$ . Therefore, we define a special fibertree—called the iteration space fibertree or *is-fibertree*—that is the fibertree representation of this iteration space tensor.

## C. Extended Einsums

Traditional Einsums sufficiently express standard tensor algebra, including those supported in Basic Linear Algebra Subprograms (BLAS) [19], [30] and tensor network contractions [1]. However, they cannot handle more complex computations. The recently proposed Extended General Einsums notation (EDGE) [39], extends Einsums to handle graph algorithm computations. We find this abstraction useful for also expressing complex tensor algebra computations and use its notation throughout the paper. We now briefly summarize the portions of EDGE that we leverage.

1) User-Defined Computations: EDGE separates computations into three "actions": map (\(\lambda\)), reduce (\(\lambda\)), and populate (=) [39]. Map specifies the pair-wise computation between the shared ranks of two tensors, reduce specifies the computation for the reduction step of an Einsum, and default populate (=) places a computed value from the right-hand side (RHS) of the Einsum to its location on the left-hand side (LHS).

Each map and reduce action contains two operations: merge and compute. Compute defines the operation to apply between two data values, and can be *any* user-defined function. Merge specifies which regions of the iteration space to touch; execution will not need to access the data space corresponding to culled points. Together, merge and compute precisely define the computations in an Einsum. Common merge operations include intersection  $(\cap)$ , which touches points with non-zero values in *both* operands; and union  $(\cup)$ , which touches points where at least one of the operands is non-zero.

The full EDGE specification for GEMM is then:

$$Z_{m,n} = A_{k,m} \cdot B_{k,n} :: \bigwedge_{k} \times (\cap) \bigvee_{k} + (\cup), \tag{2}$$

where  $\bigwedge_k$  specifies a map action between A and B on the k rank and the intersection merge operator  $(\cap)$  culls k points where at least one operand is zero. The compute operator  $(\times)$  multiplies the data values of coordinates surviving intersection. The reduce action  $(\bigvee_k)$  on the k rank gathers all non-empty points in the k rank and reduces them using addition (+).

In this work, we use three user-defined computations:

- 1) Maximum  $(\max(\cup))$  takes the maximum of two values. Suppose we have the following expression:  $Z_m = A_m \cdot B_m :: \bigwedge_m \max(\cup)$ . The union merge operator  $(\cup)$  filters out any m coordinates where both operands contain 0 (and places 0 in the output). The max compute operator then returns the maximum of the two operands.
- 2) Divide  $(\div(\leftarrow))$  divides two data values. Given the following expression,  $Z_m = A_m \cdot B_m :: \bigwedge_m \div(\leftarrow)$ , the merge operator  $(\leftarrow)$  only touches m points where there is a non-zero value in the B operand (see [39,

- Appendix]), and the compute operator divides the data value in A with the data value in B.
- 3) Subtraction and Exponentiation: To apply the exponential to an expression that subtracts two tensors, we use the following expression:  $Z_m = A_m \cdot B_m :: \bigwedge_m \text{sub-then-exp}(\mathbb{1})$ . The user-defined operator (sub-then-exp) performs  $A_m$  minus  $B_m$  then applies the exponential to the result. The merge operator,  $\mathbb{1}$ , is EDGE's "pass-through" operator, which touches all m points in the iteration space.

In addition to map and reduce, EDGE enables the expression of user-defined *unary* operations on tensors. For example, we can express the application of the non-linear, sigmoid function  $(\sigma)$  on each element of a tensor A as  $Z_m = \sigma(A_m)$ .

- 2) Shorthand Notation: Throughout this paper, we take advantage of EDGE's shorthand notation [39] in the following ways:
  - We drop all reduce actions that consist of add and union in the compute and merge operator, respectively  $(\bigvee +(\cup))$ . Thus,  $Z_m=A_{k,m}::\bigvee_k+(\cup)$  becomes  $Z_m=A_{k,m}$ .
  - We express all map actions using infix notation; that is,  $A_{k,m} \cdot B_{k,n} :: \bigwedge_k \times (\cap)$  becomes  $A_{k,m} \times B_{k,n}$ .
  - When max is part of a map action  $(A_m \cdot B_m :: \bigwedge_m \max(\cup))$ , we replace it with the following shorthand:  $\max(A_m, B_m)$ .
  - When  $\div$  is part of a map action  $(A_m \cdot B_m :: \bigwedge_m \div (\leftarrow))$ , we replace it with the following:  $A_m/B_m$ .
  - When sub-then-exp is part of the map action  $(A_m \cdot B_m :: \bigwedge_m \text{sub-then-exp}(\mathbb{1}))$ , we replace it with the shorthand  $e^{A_m B_m}$ .
  - We can express rank variable expressions with only one valid coordinate (e.g.,  $S_{i:i=2}$ ) using just the coordinate (in this case,  $S_2$ ).
- 3) Filtering Rank Expressions: EDGE also enables expressing Einsums that touch only a subset of the data space of their constituent tensors. For example, we may express the prefix sum of a tensor  $A_k$  with the following Einsum:

$$S_{i+1} = A_{k:k \le i}$$

For each coordinate i,  $S_{i+1}$  is built by reducing together the subset of A whose coordinates are  $\leq i$ . Note that this definition of prefix sum computes the entire sum for a given i without iteratively reusing the previous sum.

4) Expressing Iterative Computations: EDGE expresses recursion and iteration through generative/iterative ranks. We use the term *standard* ranks to differentiate non-iterative ranks from iterative ranks. We can express the iterative prefix sum as follows:

$$S_{i+1} = S_i + A_i \tag{3}$$

$$\diamond: i > K \tag{4}$$

Here, S is a tensor with the iterative rank, I, ranging from 0 to K (inclusive). Statement 4 indicates the stopping condition for the iterative expression (when i is greater than or equal to K).

5) Cascades of Einsums: TeAAL [35] introduces the concept of cascades of Einsums, which expresses directed acyclic graphs (DAGs) of Einsum expressions as a sequence of sub-Einsums. One can view the unrolled iterative expression in Einsum 3 as a cascade:

$$S_1 = S_0 + A_0$$
  
 $S_2 = S_1 + A_1$   
...  
 $S_K = S_{K-1} + A_K$ 

Finally, we use the EDGE *Initialization* label to specify computations that initialize tensors, which occur once. We use the EDGE *Extended Einsum(s)* label to specify the computation that occurs on each iteration of a cascade of Einsums [39]. For example, see (Einsum) Cascade 5.

## D. Mapping and Binding

While the cascade of Einsums specifies what computation is required, the *mapping* and *binding* describe how it should occur [9], [35], [41], [51]. We use the concept of *logical tasks* to define these terms. A logical task is a grouping of points in the iteration spaces of all Einsums. Tasks are defined such that each point in the iteration spaces is assigned to exactly one task. Logical tasks can be as small as a single point or as large as an entire iteration space. In the final schedule, each logical task must be assigned to exactly one compute unit that finishes the given task before moving onto the next task.

The mapping, therefore, describes a *task graph*, a directed, acyclic graph whose nodes are logical tasks and edges are dependencies between the tasks. Mapping specifications typically include aspects such as loop order, partitioning, and work scheduling (sequential vs. parallel operations) [35]. Thus, the dependencies in the task graph can be true dependencies (enforced by the cascade) or additional ordering constraints imposed by the mapping specification.

The binding describes how the tasks are bound to the actual hardware, including which compute unit each task is associated with, when that task will be executed, and where the inputs and outputs are stored in the memory hierarchy. This binding must obey the dependencies present in the task graph and the physical limitations of the architecture but is otherwise unconstrained.

