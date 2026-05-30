# <span id="page-5-1"></span>5.1 Representation of Execution Scheme

Recording State Information Using the Scheduling Table. As analyzed above, the execution scheme of a block can be represented through combinations of execution states and their associated workload values  $\{s_i\}$ . To facilitate visibility and manipulation of this execution structure, we introduce a table-based representation, called the *Scheduling Table (ScT)*. This provides a structured and cumulative view of how sub-batch workloads are distributed across execution states and sub-blocks, offering a more intuitive and analyzable form than listing state variables directly.

**Definition 4 (Scheduling Table (ScT)).** For a block B with subblock sequence  $C_B = \{B_1, B_2, \dots, B_N\}$  and pipeline-derived state set  $S_B = \{1, \dots, 2N-1\}$ , the *scheduling table*  $\mathbf{ScT} \in \mathbb{R}^{(2N-1)\times N}$  records the cumulative number of sub-batches processed by each sub-block across states.

- (i) ScT has 2N 1 rows and N columns. Row i corresponds to State-i, and column j corresponds to sub-block  $B_j \in C_B$ .
- (ii) The entry  $ScT_{i,j}$  denotes the total number of sub-batches processed by sub-block  $B_j$  from State-1 through State-i, inclusive.
- (iii) Let  $\mathcal{A}_{B_j} \subseteq \mathcal{S}_B$  denote the involved state set of sub-block  $B_j$ . The cumulative processed sub-batches for  $B_j$  up to State-i is

$$ScT_{i,j} = \sum_{k \in \mathcal{A}_{B_i} \cap \{1,\dots,i\}} s_k.$$

Constraint-Form Equivalence of Definition 5. As illustrated above,  $ScT_{i,j}$  are derived according to Definition 5. To enable integration with our MILP formulation, we now express an equivalent set of constraints - Eqs. 1 through 6 - for computing ScT in a constraint-based format. Eq. 1 ensures that all entries in ScT are non-negative integers. Eqs. 2 and 3 define boundary conditions for execution: Eq. 2 corresponds to the stage range before any subblock  $B_j$  becomes active (i.e.,  $\mathcal{A}_{B_j} \cap \{1, ..., i\} = \emptyset$ ), while Eq. 3 corresponds to the point after  $B_j$  has completed execution (i.e.,  $\mathcal{A}_{B_i} \cap \{1, \ldots, i\} = \mathcal{A}_{B_i}$ ). Eq. 4 enforces the monotonicity of accumulated sub-batch processing for each sub-block. Eq. 5 encodes data dependencies using the binary variable  $d_{m,j} \in \mathcal{D}$ , where  $d_{m,j} = 1$  if sub-block  $B_i$  depends on the output of sub-block  $B_m$ , and  $d_{m,i} = 0$ otherwise. This dependency set  $\mathcal{D}$  is fixed once the model architecture is given, and its inclusion ensures that dependent sub-blocks are not assigned to overlapping states in a parallel pattern. Finally, Eq. 6 describes the cumulative accumulation of sub-batch workloads across states, consistent with the semantics of ScT.

<span id="page-5-5"></span><span id="page-5-3"></span>
$$ScT_{i,j} \in \mathbb{N} \cup \{0\} \tag{1}$$

$$ScT_{i,j} = 0, \quad i \in [1, ..., j-1]$$
 (2)

<span id="page-5-6"></span>
$$ScT_{i,j} = BS/BS_{sub}, \quad i \in [N+j-1,...,2N-1]$$
 (3)

<span id="page-5-7"></span>
$$ScT_{i+1,j} \ge ScT_{i,j} \tag{4}$$

<span id="page-5-8"></span>
$$ScT_{i,m} \ge ScT_{i,j} + d_{m,j}, \quad i \in [j, \dots, N + j - 2]$$
(5)

<span id="page-5-4"></span>
$$ScT_{i,j} = s_i + ScT_{i-1,j}, \quad i \in [j, ..., N + j - 2]$$
 (6)

