# A. Physical Storage Modeling

Layoutloop models physical storage as (num\_line × line\_size) 2D array with "conflict\_depth" specifying number of lines in each bank with the following reasoning.

**Bank Organizations:** Current storage uses diverse organizations, including 2D/3D with various groupings. Managing these disparate physical organizations can be complex. However, as storage is usually accessed line by line (or block), we can abstract different organizations into a logical *num\_line* × *line\_size* 2D array. This abstraction allows layout modeling to handle these 2D abstract arrays directly, retaining generality without dealing with specific physical organizations.

**Bank Port Constraints:** Storage comes with an inherent limitation of the total number of ports in each bank. Concurrent

 $<sup>^2</sup>$ A flattening of 4 iActs dimensions (N = 1, C = 3, H = 224, W = 224) into two nested loop (Fig. 3) introduces 8! = 40320 order possibilities and (1,2,16,16) factorization possibility. The product leads to  $10^8$  layout choices.

read/write operations exceeding available read/write ports lead to bank conflicts. Thus, *conflict\_depth* is utilized to denote the total number of lines within a single bank.

## B. Bank Conflicts Assessment

Layoutloop models slowdown by judging whether bank conflicts occur when analyzing data access to the on-chip buffer with a specific layout. A  $\max(N_P/N_L, 1)$  slowdown is introduced if  $N_L$  lines are accessed from a bank with  $N_P$  ports. Finally, we also modify Timeloop's mapper to consider data layout during dataflow search.

#### VI. EVALUATION

