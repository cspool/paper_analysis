# <span id="page-16-1"></span>A Momery-balanced Micro-batch Chunking in Sequence Blaster

We illustrate the memory-balanced micro-batch chunking algorithm in Sequence Blaster based on dynamic programming. Specifically, given a batch of sequences  $\mathcal{B} = \{S_k\}$  that has already been sorted according to takeaway #2, we split them into consecutive M micro-batches, and micro-batch  $\mathcal{M}_i$  contains all sequences k satisfying  $j_{i-1} \leq k < j_i$ , where  $j_i$  is the ending indices for  $\mathcal{M}_i$  ( $j_0 = 0$ ). To balance the token amount of each micro-batch, we aim to minimize the maximum total token number of each micro-batch as follows:

$$\underset{\{j_i\}}{\text{arg min}} \max_{i \in [1,M]} \{ \sum_{k \in [j_{i-1},j_i)} s_k \}. \tag{23}$$

Again, we solve the problem via dynamic programming. Denote DP[k][i] as the optimal value when blasting the first k sequences into i micro-batches. Starting with DP[0][0] = 0, we can solve the problem via the following state transition formula:

$$DP[k][i] = \min_{j \in [i-1,k-1]} \{ \max\{DP[j][i-1], \sum_{l \in [j+1,k]} s_l \} \}, (24)$$

where  $\sum_{l \in [j+1,k]} s_l$  represents the total token number of the  $i^{th}$  micro-batch when splitting micro-batch at the  $S_j$ . DP[K][M] denotes the optimal solution, and the optimized values of  $j_i$  splits the global batch data into M micro-batches with balanced memory consumption.

