# A. Optimal Partitioning

Let us call  $th_i$  and  $tc_i$  the estimated times that a hot and a cold worker, respectively, take to execute tile i. Similarly,  $bh_i$  and  $bc_i$  are the estimated number of bytes read/written

from memory if a hot and a cold worker, respectively, execute tile i. Then, the total execution time of all the hot workers processing the hot tiles in parallel  $(th_{total})$  and of all the cold workers processing the cold tiles in parallel  $(tc_{total})$  are:

$$th_{total} = \Sigma_{i \in hot} \frac{th_i}{N_{hw}}$$
  $tc_{total} = \Sigma_{i \in cold} \frac{tc_i}{N_{cw}}$  (2)

and the total number of bytes read/written from memory by all the workers  $(b_{total})$ , all the hot workers  $(bh_{total})$ , and all the cold workers  $(bc_{total})$  are:

$$b_{total} = bh_{total} + bc_{total} = \sum_{i \in hot} bh_i + \sum_{i \in cold} bc_i$$
 (3)

Assuming that hot and cold workers are operating in parallel, and that BW is the total memory bandwidth of the heterogeneous architecture, the optimal partitioning is the solution to the following optimization problem:

$$minimize\{max\{max\{th_{total}, tc_{total}\}, \frac{b_{total}}{BW}\}\} \qquad (4)$$

This optimization problem is not trivial. For example, it cannot be trivially solved by assigning each tile to the worker type that is estimated to be faster. This is because of two factors: (1) the workers operate in parallel and (2) the bandwidth saturation impacts the total runtime.

In addition, in some architectures, there is no mechanism to avoid data races when heterogeneous workers are writing to the same output memory locations. In this case, each worker type must update a private output buffer and the buffers are merged at the end of the execution. This introduces an additional  $t_{merge}$  cost term in the final execution time. This term can be estimated by considering the memory footprint of the output buffers and the system memory bandwidth. In this case, the optimization problem can be expressed as:

$$minimize\{max\{max\{th_{total}, tc_{total}\}, \frac{b_{total}}{BW}\} + t_{merge}\}$$
(5)

We assume a buffer accumulation design such that  $t_{merge}$  does not depend on the data that has been written in each buffer. Hence,  $t_{merge}$  has the same value for all the possible partitionings. With this assumption, the optimal solution for equations 4 and 5 is the same. In addition, for some sparse matrices, the  $t_{merge}$  cost might be too high to justify parallel operation of the heterogeneous workers. Instead, in such cases it might be faster that the workers execute serially, using the

![](_page_7_Figure_0.jpeg)

Fig. 8: Optimization subproblems.

same output buffer. In this case, the optimization problem involves minimizing the *minimum* of

$$max\{max\{th_{total}, tc_{total}\}, \frac{b_{total}}{BW}\} + t_{merge}$$
 (6)

and the predicted runtime for the serial operation:

$$max\{th_{total}, \frac{bh_{total}}{BW}\} + max\{tc_{total}, \frac{bc_{total}}{BW}\}$$
 (7)

Thus, the full optimization problem becomes:

$$minimize\{min\{max\{max\{th_{total}, tc_{total}\}, \frac{b_{total}}{BW}\} + t_{merge}, max\{th_{total}, \frac{bh_{total}}{BW}\} + max\{tc_{total}, \frac{bc_{total}}{BW}\}\}\}$$
(8)

