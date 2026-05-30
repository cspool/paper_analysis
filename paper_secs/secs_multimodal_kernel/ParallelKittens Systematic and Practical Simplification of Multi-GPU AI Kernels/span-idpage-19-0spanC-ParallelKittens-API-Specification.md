# <span id="page-19-0"></span>C ParallelKittens API Specification

We provide the full specification of PK primitives, including each function's name, signature, parameters, and description.

```
template <int axis, cache_policy policy, kittens::ducks::st::all ST,
```

#### Template Parameters:

- axis: Tensor axis for the operation (0-3).
- policy: Cache policy (NORMAL or cache hint).
- ST: Shared tile type.
- PGL: Parallel global layout type.
- COORD: Coordinate type for indexing.

#### Parameters:

- dst: Destination parallel global layout.
- src: Source shared memory tile.
- idx: Coordinate specifying the destination position.

Description: Asynchronously stores a shared memory tile to multicast memory using the Tensor Memory Accelerator (TMA). Launched by a single thread.

```
template <int axis, cache_policy policy, kittens::ducks::st::all ST,
```

Template Parameters:

- axis: Tensor axis for the operation (0-3).
- policy: Cache policy (NORMAL or cache hint).
- ST: Shared tile type.
- PGL: Parallel global layout type.
- COORD: Coordinate type for indexing.

### Parameters:

- dst: Destination parallel global layout.
- src: Source shared memory tile.
- idx: Coordinate specifying the destination position.

Description: Asynchronously performs an atomic add reduction from a shared memory tile to multicast memory via TMA. The operation atomically adds the source tile values to the existing values at the destination. Launched by a single thread.

```
template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP,
          kittens::ducks::pgl::all PGL, kittens::ducks::gl::all GL>
__device__ void reduce(GL &dst, const coord &dst_idx, PGL &src, const coord &src_idx)
```

### Template Parameters:

- TILE ROWS: Number of rows in the tile.
- TILE COLS: Number of columns in the tile.
- OP: Reduction operation to apply (sum, max, or min).
- PGL: Parallel global layout type.
- GL: Global layout type.

### Parameters:

- dst: Reference to the destination global layout.
- dst idx: Coordinate specifying the destination tile's position.
- src: Reference to the source parallel global layout.
- src idx: Coordinate specifying the source tile's position.

Description: Performs a reduction from multicast memory to device-local global memory. The function loads data from the source parallel global layout using in-network reduction operations and stores the result to the destination global layout. Collectively launched by one or more warps. Each warp processes multiple rows of the tile, performing the specified reduction operation during the multicast load and then writing the reduced values to the destination global memory.

```
template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP,
          kittens::ducks::pgl::all PGL>
__device__ void all_reduce(PGL &dst_and_src, const coord &idx)
```

### Template Parameters:

- TILE ROWS: Number of rows in the tile.
- TILE COLS: Number of columns in the tile.
- OP: Reduction operation to apply (sum, max, or min).
- PGL: Parallel global layout type.

### Parameters:

- dst and src: Reference to the parallel global layout object.
- idx: Coordinate specifying the tile's position with batch (b), depth (d), row (r), and column (c) indices.

Description: Performs an all-reduce collective operation on a tile of data on multicast memory. The function reduces data across all participating GPUs for the specified tile. Collectively launched by one or more warps. Each warp processes multiple rows, loading data from multicast memory with the specified reduction operation, then writing the result back to the same multicast location. The operation leverages in-network acceleration hardware to efficiently perform the reduction without explicit peer-to-peer copies.

```
__device__ void signal(const barrier_t &barrier, const coord &idx,
                       const int dst_dev_idx, const int val)
```

### Parameters:

- barrier: Reference to the barrier object (parallel global layout of integers).
- idx: Element-wise coordinate specifying the barrier location.
- dst dev idx: Target device index to signal.
- val: Value to add to the barrier counter.

Description: Signals a specific device's barrier by atomically adding a value to its counter. This primitive is used to coordinate synchronization between thread blocks and GPUs.

```
__device__ void signal_all(const barrier_t &barrier, const coord &idx, const int val)
```

### Parameters:

- barrier: Reference to the barrier object.
- idx: Element-wise coordinate specifying the barrier location.
- val: Value to add to all devices' barrier counters.

Description: Signals all devices simultaneously by performing a multicast atomic add operation. Uses in-network multicast hardware to efficiently update barrier counters across all participating devices with a single operation.

```
__device__ void wait(const barrier_t &barrier, const coord &idx,
                     const int dev_idx, const int expected)
```

### Parameters:

- barrier: Reference to the barrier object.
- idx: Element-wise coordinate specifying the barrier location.
- dev idx: Device index to wait on.
- expected: Expected barrier value to wait for.

Description: Waits until a device's barrier counter reaches the expected value. Continuously polls the barrier location using relaxed memory ordering loads until the expected value is observed. This provides a spinning wait mechanism for inter-SM and inter-GPU synchronization.

```
__device__ void barrier(const barrier_t &barrier, const coord &idx, const int dev_idx)
```

### Parameters:

- barrier: Reference to the barrier object.
- idx: Element-wise coordinate specifying the barrier location.
- dev idx: Current device index.

Description: Implements a complete barrier synchronization across all devices. This ensures all participating GPUs reach the same synchronization point before proceeding.

### <span id="page-22-0"></span>D ParallelKittens Program Template and Example Kernels

Load-Compute-Store-Communicate (LCSC) Template. The LCSC template provides a structured approach for implementing multi-GPU kernels with specialized worker components. The template enables flexible warp/SM specialization and overlapping strategies for compute, memory, and communication operations.

High-level Template Structure:

```
struct lcsc_template {
    static void loader(globals, comp_sem, comp_smem, comp_regs);
    static void storer(globals, comp_sem, comp_smem, comp_regs);
    static void consumer(globals, comp_sem, comp_smem, comp_regs);
    static void communicator(globals, comm_sem, comm_smem, comm_regs);
};
```

### Required Components:

- comp sem: struct of semaphores for synchronization within compute SMs.
- comm sem: struct of semaphores for synchronization within communication SMs.
- comp smem: struct of shared memory layouts for compute SMs.
- comm smem: struct of shared memory layouts for communication SMs.
- comp regs: struct of register state for compute workers.
- comm regs: struct of register state for communication workers.

### Workers:

- loader: Performs memory loads from local or peer HBM using TMA.
- storer: Performs memory stores to local or peer HBM.
- consumer: Performs tensor/CUDA core operations on loaded data.
- communicator: Performs dedicated inter-GPU communication. Executes on separate communication SMs.

Execution Model: The template automatically distributes SMs between computation and communication roles based on num comm sms, passed in to the host entry function. Compute SMs execute loader, storer, and consumer functions with producer-consumer synchronization through semaphores. Communication SMs execute the communicator function independently. The framework handles warpgroup specialization, register allocation, and task distribution across workers. Programmers can utilize this template by defining the above struct, and passing it to the launch interface:

lcsc::launch\_kernel<config, globals, lcsc\_template>(G, stream);

### Where the parameters are:

- config: Compile-time configuration struct defining SM and thread counts.
- globals: Runtime globals struct containing device memory pointers and parameters.
- lcsc template: User-defined LCSC template implementation.

- G: Instance of globals struct.
- stream: CUDA stream for kernel execution.

We present a fused GEMM + all-reduce (AR) kernel implemented using the LCSC template in Figure 18. We highlight that the kernel contains *both* a fully optimized GEMM and fused all-reduce logic, with the communication-relevant code comprising only about 10 lines of device code. We also open-source all remaining kernels evaluated in this paper through our GitHub repository.

```
inline void loader(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) {
     _device_
        int2 idx = interpret task(regs.task id):
        for (int red_idx = 0; red_idx < regs.num_iters; red_idx++) {
3
 4
            wait(sem.inputs_finished[regs.stage], get_phasebit<1>(regs.phasebits, regs.stage));
5
            update_phasebit<1>(regs.phasebits, regs.stage);
6
            tma::expect_bytes(sem.inputs_arrived[regs.stage], sizeof(A_tile) * 2 + sizeof(B_tile));
            if (red_idx == PIPELINE_STAGES - 1) {
8
                \verb|wait(sem.outputs_finished|, get_phasebit<1>(regs.phasebits|, PIPELINE_STAGES));|
9
                update_phasebit<1>(regs.phasebits, PIPELINE_STAGES);
10
11
            for (int i = 0; i < 2; i++)
                tma::load_async(smem.inputs[regs.stage].A[i], G.A, {idx.x * 2 + i, red_idx}, sem.
                      inputs_arrived[regs.stage]);
13
            tma::load_async(smem.inputs[regs.stage].B, G.B, {red_idx, idx.y}, sem.inputs_arrived[regs.stage])
            regs.stage = (regs.stage + 1) % PIPELINE_STAGES;
14
15
        }
    }
16
17
18
               inline void storer(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) {
19
        int2 idx = interpret_task(regs.task_id);
20
        wait(sem.outputs_arrived, get_phasebit<0>(regs.phasebits, 0));
21
        update_phasebit <0>(regs.phasebits, 0);
22
        for (int i = 0; i < 2; i++)
23
24
            tma::store_async(G.C[G.dev_idx], regs.C[i], {idx.x * 2 + i, idx.y});
        tma::store_async_read_wait();
25
        arrive(sem.outputs_finished);
26
        int signal_dev_idx = regs.task_id % NUM_DEVICES;
27
        device < NUM_DEVICES >:: signal(G.barrier, {idx.x, idx.y}, signal_dev_idx, 1);
28
29
30
               inline void consumer(const globals &G, comp_sem &sem, comp_smem &smem, comp_regs &regs) {
        rt_fl<ROW_BLOCK / 8, COL_BLOCK > C_accum;
31
32
        warp::zero(C_accum);
33
        for (int red_idx = 0; red_idx < regs.num_iters; red_idx++) {</pre>
34
            wait(sem.inputs_arrived[regs.stage], get_phasebit<0>(regs.phasebits, regs.stage));
35
            update_phasebit<0>(regs.phasebits, regs.stage);
36
            warpgroup::mma_AB(C_accum, smem.inputs[regs.stage].A[regs.warpgroup_id], smem.inputs[regs.stage].
                  → B);
37
            warpgroup::mma_async_wait();
            warp::arrive(sem.inputs_finished[regs.stage]);
39
            regs.stage = (regs.stage + 1) % PIPELINE_STAGES;
40
41
        group <8>::sync(3);
        warpgroup::store(regs.C[regs.warpgroup_id], C_accum);
43
        warpgroup::sync(regs.warpgroup_id + 1);
44
        warpgroup::arrive(sem.outputs_arrived);
45
46
47
     _device__ inline void communicator(const globals &G, comm_sem &sem, comm_smem &smem, comm_regs &regs) {
48
        int2 idx = interpret_task(regs.task_id);
49
        if (threadIdx.x == 0)
            device < NUM_DEVICES >:: wait (G.barrier, {idx.x, idx.y}, G.dev_idx, NUM_DEVICES);
51
          syncthreads():
52
        group < NUM_WARPS >:: all_reduce < ROW_BLOCK, COL_BLOCK, reduce_op:: ADD > (G.C, {idx.x, idx.y});
```

<span id="page-23-1"></span>Figure 18: Fused GEMM + AR kernel implemented with the LCSC template

