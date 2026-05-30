# <span id="page-3-1"></span>*C. Custom Tile Order*

The third requirement for efficient synchronization is minimizing waiting time of consumer kernels. However, the CUDA runtime can schedule thread blocks on SMs in any arbitrary order, which can lead to unpredictable wait times. Ideally, thread blocks of the consumer kernel should be scheduled in the order the producer kernel generate tiles.

cuSync enables execution of both producer and consumer kernel's thread blocks in a custom scheduling order independent of how the CUDA runtime schedules thread blocks. In our example, each thread block calls stage.tile() (line [4\)](#page-4-7) to obtain the tile it needs to compute. The parameter RowMajor (lines [18](#page-4-1)[–20\)](#page-4-2) ensures that both kernels produce tiles in a row major order, i.e., first all thread blocks in x, then in y, and finally in z. Figure [4b](#page-4-0) defines the RowMajor order as a function (line [29\)](#page-4-8). A tile order function takes a tile index in the 3-D grid and returns a distinct 1-D index for the tile. Internally, cuSync maintains an array that maps a linear tile index to a 3-D index. For each thread block, cuSync increments an atomic global counter and returns the 3-D index in the array for the previous counter value. In summary, cuSync allows easy experimentation with diverse scheduling orders to obtain the best performance.

## <span id="page-3-2"></span>*D. Synchronizing Dependent Tiles*

The final requirement for fine-grained synchronization is to ensure that the dependence between tiles of producerand consumer-kernels is maintained using a synchronization mechanism. cuSync provides two functions, wait and post to enforce this dependency. For instance, in our example, the wait function is called twice (line [6](#page-4-9) and [8\)](#page-4-10) before loading the tiles of A and B, and the post function is called once (line [12\)](#page-4-11) for the producer kernel after computing the tile. However, the consumer kernel only needs to wait on the output of the producer kernel, i.e., input A of the consumer kernel.

```
1 //CUDA Kernel to compute C = A * B
                                                      1 class CuStage < Policy >
2 global void gemm(f16* A, f16* B, f16* C,
                                                      2 void init() {
                       int K, CuStage stage) {
3
                                                      3
                                                           sems = /*Init semaphores using Policy*/}
4
    stage.start(); row, col = stage.tile();
                                                         void post(dim2 tile, dim2 grid) {
    for (tk = 0; tk < K; tk += TileK) {
5
                                                            __syncthreads();
                                                      5
       stage.wait (A, row, tk);
6
                                                            if(threadIdx == \{0,0,0\})
                                                      6
       LoadTileToShMem(Ash, A, row, tk);
                                                           __threadfence_system();
 7
                                                      7
       stage.wait(B, col, tk);
 8
                                                      8
                                                              sem = &sems[Policy.sem(tile, grid)];
9
       LoadTileToShMem(Bsh, B, col, tk);
                                                      9
                                                              atomicAdd(sem, 1);}
10
      MultiplyAccumulate(C, Ash, Bsh,
                                                     10
                                                         void wait(dim2 tile, dim2 grid) {
                           row, col, tk);}
11
                                                            sem = &sems[Policy.sem(tile, grid)];
                                                     11
12
    stage.post(row, col);
                                                            if(threadIdx == \{0,0,0\})
                                                     12
                                                              while (*sem != Policy.value(tile,grid));
13 }
14 void MLP (int BS, int H, f16* X, f16* W<sub>1</sub>,
                                                     14
                                                            __syncthreads(); }
            f16* XW_1, f16* W_2, f16* XW_{12}) {
                                                     15
15
    dim2 grid1 = \{4*H/8, B\}/tile1;
16
                                                     16 class TileSync
    dim2 grid2 = {H, B}/tile2;
                                                          int sem(dim2 tile, dim2 grid) {
17
                                                     17
    CuStage<RowMajor, RowSync>
                                                     18
                                                            //Distinct semaphore for each tile
18
      prod(grid1,tile1);
                                                            return tile.x*grid.y + tile.y;}
19
                                                     19
                                                          int value(dim2 tile, dim2 grid) {return 1;}
20
    CuStage<RowMajor, RowSync>
                                                     20
21
      cons(grid2,tile2);
                                                     21
    // declare prod to cons[XW1] dependency
                                                     22 class RowSync
23
    CuSync::dependency(prod, cons, XW1);
                                                     23
                                                        int sem(dim2 tile, dim2 grid) {
24
    // invoke the producer gemm
                                                     24
                                                            //Tiles of same row share semaphore
                                                            return tile.y;}
    gemm<<<grid1, tb1, prod.stream()>>>
                                                     25
25
26
       (X, W_1, XW_1, H, prod);
                                                     26
                                                          int value(dim2 tile, dim2 grid) {
27
    // invoke waitKernel and then consumer
                                                     27
                                                            return grid.x;}
28
    cons.waitKernel();
                                                     28
                                                     29 int RowMajor(dim2 tile, dim2 grid) {
    gemm<<<grid2, tb2, cons.stream()>>>
29
       (XW_1, W_2, XW_{12}, 4*H/8, cons); 
                                                            return tile.y*grid.x + tile.x;}
```

<span id="page-4-6"></span><span id="page-4-5"></span><span id="page-4-4"></span><span id="page-4-3"></span><span id="page-4-2"></span><span id="page-4-1"></span>(a) The kernels are invoked on different streams. The wait kernel ensures the order of kernel invocation. The post and wait methods ensure tile dependency. Changes to the GeMM kernel are <u>underlined</u>.

<span id="page-4-24"></span><span id="page-4-23"></span><span id="page-4-22"></span><span id="page-4-21"></span><span id="page-4-20"></span><span id="page-4-19"></span><span id="page-4-18"></span><span id="page-4-17"></span><span id="page-4-16"></span><span id="page-4-15"></span><span id="page-4-14"></span><span id="page-4-13"></span><span id="page-4-8"></span>(b) TileSync creates a semaphore for each tile. RowSync trades concurrency for synchronization by creating a single semaphore per row.

Fig. 4: Fine-grained synchronization of two GeMMs of MLP using cuSync's TileSync and RowSync policies.

This dependency is specified in line 23. Therefore, the wait before loading a tile of A waits for the corresponding post of the producer kernel, and the wait before loading a tile of B becomes a no-op. Since the producer kernel have no dependency, both waits are no-ops for the producer kernel.

cuSync provides a mechanism for synchronizing producer and consumer tiles based on an arbitrary *synchronization policy* (or policy in short). cuSync uses an array of global memory semaphores for synchronization, where each producer tile is associated with only one semaphore and a semaphore's value represents the status of its producer tiles. Thus, a policy is a mapping of one or more producer tiles to one semaphore. For example, the finest grained synchronization policy, we call *TileSync*, waits for each producer tile and is defined as a one-to-one map of a producer tile to a semaphore. A policy requires implementation of two methods: (i) sem, which returns the semaphore for the given tile, and (ii) value, which returns the expected value of semaphore when the tile is ready. We below describe details of three methods of CuStage required for our synchronization mechanism (lines 2–9 in Figure 4b).

init: The init method allocates and initializes the array of semaphores in the global memory based on the given policy.

post: The post method calls \_\_syncthreads and a

memory fence to ensure that all threads of the thread block has computed the tile and all global memory writes are visible to other kernels (line 5–7). Finally, the method obtains the semaphore for the tile using the policy and increments the semaphore (line 9).

wait: The wait method obtains the semaphore for the given tile using the policy and then wait on the value of semaphore in a while loop using only the first thread of the thread block (line 13). While the first thread is waiting, all other threads of the thread block are blocked on the \_\_syncthreads (line 14). When the semaphore changes to the expected value, all threads of the thread-block proceeds from the \_\_syncthreads.

#### E. Synchronization Policies

cuSync allows implementation of diverse synchronization policies easily. As described earlier, each policy requires implementing sem and value methods. Below we discuss two general policies that are applicable to all kernels in our workloads.

**TileSync** is the finest-grained policy that synchronizes on each producer tile (lines 16–20 in Figure 4b). To minimize the wait time of the consumer-kernel, both kernels compute their

tiles in a row major order. The sem method returns distinct semaphore for each tile (line 17) and the value method returns 1 to signify that the tile is computed (line 20). For example, in Figure 4a to compute a tile  $E^{xy}$ , the TileSync policy requires waiting first on  $C^{x0}$  and then on  $C^{x1}$ .

RowSync synchronizes on each row of the producer kernel requiring less synchronizations than TileSync (lines 22–27 in Figure 4b). For example, for two GeMMs of Figure 4a, TileSync requires 12 synchronizations in total, while RowSync requires 6 synchronizations by sharing the same semaphore for all tiles computing the same row of C. Thus, the sem method returns the row of the given tile and the value method returns the value when the row is ready, i.e., the number of tiles in a row (line 23–26). To minimize the wait time, both kernels schedule their tiles in a row major order. RowSync can also be used for synchronizing Conv2D kernels. Section V shows that for large GeMMs and Conv2Ds the high number of synchronizations is a bottleneck.

#### <span id="page-5-0"></span>IV. AUTO-TUNING OF POLICIES AND TILE ORDERS

The process of obtaining the best performance involves experimenting with several synchronization policies and tile processing orders. The best policy and tile order depends on computations, data sizes, and the GPU architecture. However, doing this process manually is both tedious and error-prone.

Therefore, cuSyncGen is a tool that takes dependencies specified by the user and generates the optimal tile processing order and multiple synchronization policies as CUDA code for cuSync. cuSync currently requires the user to manually modify the GPU kernels to instantiate CuStage with generated policies and tile processing order similar to the MLP example (Figure 4a). The modularity of cuSync allows the user to easily plug diverse synchronization policies and tile processing orders.

#### A. Workflow

The workflow of cuSyncGen is as follows:

- 1) The user describes a chain of dependencies between kernel tiles and the grid values for all kernels.
- cuSyncGen checks bounds of producer and consumer tiles based on grid values.
- cuSyncGen generates a tile processing order as CUDA code that minimizes the wait time.
- 4) cuSyncGen generates CUDA code for multiple policies.
- 5) The user modifies the workload to support cuSync and plugs the generated CUDA code to cuSync.

The rest of the section describes each of these steps.

**Describe Dependencies** The user describes dependencies between tiles of kernels using a DSL embedded in C++. Figure 5a shows the dependency between both GeMMs of MLP described in the DSL. First, the DSL code must define each kernel's grid dimensions with their maximum value. The example defines x and y dimensions for both grids (line 1–4). Specifying the exact values for a grid enables generating efficient code and doing bounds checking for correctness.

```
1 Dim x, y;
2 //Max value of all dimensions of both GeMMs
3 Grid g1(x, y, \frac{H}{2*TileN}, \frac{B*S}{TileN});
4 Grid g2(x, y, \frac{-H^{-2}}{TileN}, \frac{B*S}{TileN});
5 //Tile is produced by each thread block
6 Tile prod(x, y), cons(x, y);
7 //All col tiles for a row from 0 to \frac{\text{H}}{2*\text{TileN}}
8 ForAll prodCols(prod, x, Range(g1.x));
9 //Tile of 2nd GeMM depends on all
10 //col tiles of 1st GeMM
11 Dep dep({g2, cons}, {g1, prodCols});
                         (a) GPT-3's MLP
1 Dim x, y;
2 //First GeMM Grid
3 \text{ Grid gl}(x, y,
                       8*TileN /
Grid
4 //P, R, and T
                       \underline{\mathtt{B*(S}} + \mathtt{S'})
                                   \frac{B*(S+S')}{TileM});
5 \text{ Grid gP}(x, y,
                       TileN

B*(S+S')

TileN

B*(S+S')
6 \text{ Grid } gR(x, y,
7 \text{ Grid gT}(x, y,
                                   \frac{1}{8*\text{TileM}});
8 //Second GeMM Grid
9 Grid g2(x, y, \frac{H}{8*TileN}, \frac{B}{TileM});
10 //P to 1st GeMM
11 //Strided Tile Dependencies: stride=\frac{H}{8*TileN}
12 Dep dep1P(\{gP, Tile(x,y)\},
    {g1, Tile(x,y), Tile(x+\frac{n}{8*TileN},y)});
14 Dep depPR({gR, Tile(x,y)},
      {gP, ForAll(Tile(x,y), y, Range(gP.y))});
16 Dep depTR1(\{gT, Tile(x,y)\},
      {gR, Tile(x, y)}, {g1, Tile(x + \frac{2*H}{8*TileN}, y)});
18 //2nd GeMM to T
19 dep23(\{g3, Tile(x,y)\}, \{gT, Tile(\frac{x}{TileM},y)\});
                            (b) Attention
1 Dim x, y;
2 //First GeMM Grid
3 Grid g1(x, y, \frac{c}{TileM}, 4 //Second GeMM Grid
5 Grid g2(x, y, \frac{c}{TileM}, \frac{B*P*Q}{TileM}); 6 //2nd Conv2D to 1st Conv2D
7 Dep dep(\{g2, Tile(x,y)\}, \{g1, Tile(\frac{x}{R*S},y)\});
                         (c) Two Conv2Ds
```

<span id="page-5-8"></span><span id="page-5-7"></span>Fig. 5: Dependencies in the cuSyncGen DSL. TileM and TileN are tile size of GeMMs in row and column respectively.

Then, the DSL code constructs producer and consumer tiles by specifying an affine function over each dimension of the grid. The example creates a producer and consumer tile for each thread block in the grid and creates a range of column tiles using ForAll (line 6–8). Finally, the code specifies the dependence between one consumer tile and one or more producer tiles (line 11).

Generate Tile Processing Order cuSyncGen generates a tile processing order for each kernel to minimize the waiting time. To discuss the process, consider a dependency where a consumer tile, C(x,y), depends on N producer tiles,  $\{P(x,a_0y+b_0), P(x,a_1y+b_1), \ldots, P(x,a_{N-1}y+b_{N-1})\}$ .

We achieve minimum wait time when the consumer kernel consumes tiles in the same order as they are produced by the producer kernel. Thus, we schedule all N producer tiles consecutively for each consumer tile using the following code:

```
1 int prodOrder(dim2 tile, dim2 grid) {
2   int linear = bid.y*gDim.x + bid.x, y = 0;
3   if (tile.y%a<sub>0</sub> <= b<sub>0</sub>) y = 0;
4   //Similarly for tiles till N-2
5   else if (tile.y%a<sub>N-1</sub> <= b<sub>N-1</sub>) y = N-1;
6   return linear/N+y;}
```

This code obtains the 1-D linear index of a tile, finds the tile index within the group of N tiles, and returns the new linear index. We also set the consumer kernel to follow the row major order of tiles. Our MLP example uses the row major order, i.e., all groups of  $\frac{\rm H}{\rm TileN}$  consecutive producer tiles are scheduled consecutively. It is straightforward to extend this method to a chain of dependent kernels by extending the dependence from the last consumer kernel to the first producer kernel and then generating code for each kernel.

Generating Policies cuSyncGen generates multiple synchronization policies for each dependence. For the following discussion, consider a dependence where a consumer tile, C(x,y), depends on N producer tiles,  $\{P(x,a_0y+b_0),P(x,a_1y+b_1),\ldots,P(x,a_{N-1}y+b_{N-1})\}$ . cuSyncGen generates two policies for the dependence in each dimension: (i) map each tile to a distinct semaphore, or (ii) map all N tiles to the same semaphore. The code generated for the considered dependence and the value of  $M \in \{1,N\}$  is:

```
1 int sem(dim2 tile, dim2 grid) {
2    int y = 0;
3    if (tile.y*a_0 <= b_0)
4     y = (tile.y-b_0)/a_0;
5    //Similarly for tiles till M-2
6    else if (tile.y*a_{M-1} <= b_{M-1})
7     y = (tile.y-b_{M-1})/a_{M-1};
8    else y = tile.y;
9    return y*grid.x + tile.x;}
10 int value(dim2 tile, dim2 grid) {return M;}
```

After considering both cases for the innermost dimension, the phase moves to the outer dimension, and follows the same method. In our MLP example, cuSyncGen generates two policies: (i) TileSync that maps each tile to a distinct semaphore, and (ii) RowSync that maps all column tiles of the same row to the same semaphore.

Running the Generated Code We require the user to modify the workload to support running cuSync by adding wait calls before every tile load and post call after computing a tile. For example, in the case of MLP, we require the user to do the changes of Figure 4a. The modularity of cuSync enables plugging multiple policies and tile processing order. So, the user can execute all generated policies and obtain the policy with least execution time.

#### B. Computation Dependencies in ML Models

We now show how to specify dependencies of Attention and Conv2D cases in cuSyncGen.

**Attention** contains two dependencies between its three kernels

(Figure 5b). In the first dependency, an element of the dot product depends on three elements in the same row with a stride of  $\frac{H}{8}$  of the first GeMM output (line 13). In addition to TileSync and RowSync, for this dependence cuSyncGen also generates a policy, we call StridedSync, that maps all three producer tiles of the first GeMM to the same semaphore. Thus, StridedSync waits until all three tiles of the first GeMM are computed before continuing with the dot product of tiles. Moreover, cuSyncGen generates the tile order that schedules these three tiles consecutively. For other dependencies, cuSyncGen generates both TileSync and RowSync, while processing tiles in a RowMajor order.

Conv2D using the implict GeMM algorithm converts a convolution of B input images of size [P,Q,C] with a kernel matrix of size [R,S] into a GeMM of matrices [B\*P\*Q,C\*R\*S] with [C\*R\*S,C]. Figure 5c shows the dependency between two Conv2Ds using the implicit GeMM algorithm. Thus, the dependency describes that each tile of the second implicit GeMM depends on all column tiles of the first implicit GeMM output (line 7). cuSyncGen generates two policies for this dependency: (i) RowSync to synchronize each row, and (ii) Conv2DTileSync policy to synchronize each tile. Moreover, cuSyncGen generates a row major order for both Conv2Ds.

#### <span id="page-6-0"></span>C. Optimizations

cuSyncGen automatically perform several optimizations to improve the performance of a cuSync synchronized workload. These optimizations depend on the architecture details of the GPU, occupancy of CUDA kernels, and grid sizes. The optimizations are as follows:

**Avoid Wait Kernel** The wait-kernel mechanism ensures that all thread blocks of the producer kernel are scheduled on the GPU before the consumer kernel. However, if both producer and consumer kernels can be executed in less than two waves, we do not need the wait-kernel mechanism.

**Avoid Custom Tile Processing Order** We can also avoid a custom tile processing order when all tiles of producer and consumer-kernels can be executed in two waves.

Reorder Tile Loads and Synchronization The general workflow of tile based CUDA kernels is to load a tile of all inputs into shared memory or registers and then perform operations on all tiles. We can re-order the waiting of tile of one input with the loading of other tile, to overlap the waiting of one tile with the loading of the other input's tile. For example, in Figure 4a the second GeMM kernel loads a tile of both inputs (A and B) and compute the tile of output matrix (C) (line 6–9). We can reorder the loading of B tile with the waiting on A tile, i.e., swap lines 6–7 with lines 8–9. Since there is no waiting for tile of B, loading a B tile can overlap with waiting of A tile, leading to improved performance. cusyncgen automatically performs the reordering if the user annotate tile loading in kernels with #pragma tile.

<span id="page-7-1"></span>TABLE III: FRACTION OF LINES OF CODE CHANGED IN GEMM, FUSED SOFTMAX-DROPOUT, AND CONV2D KER-NELS TO SUPPORT USING C USY N C.

| Kernel          | Implementation | Lines Changed |          |  |
|-----------------|----------------|---------------|----------|--|
|                 |                | Number        | Fraction |  |
| GeMM            | CUTLASS        | 25            | 0.5%     |  |
| Softmax-Dropout | Ours           | 5             | 1%       |  |
| Conv2D          | CUTLASS        | 22            | 0.6%     |  |

